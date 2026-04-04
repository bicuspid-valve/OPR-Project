"""Main PPO training loop orchestration."""
from __future__ import annotations

import csv
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.multiprocessing as _mp
import torch.nn as nn

from models import ResolvedUnit

from ml_training.config import TrainingConfig, _force_tensor_device, _resolve_device
from ml_training.entropy import EntropyTargetTuner
from ml_training.checkpoint import (
    CheckpointPool, _make_model, load_model_state_dict,
    get_heuristic_fraction,
)
from ml_training.collection import (
    _WORKER_COUNT, _MAX_SHARED_OPPONENTS,
    _init_shared_worker, _collect_episodes_shared_worker,
)
from ml_training.gae import compute_gae
from ml_training.loss import replay_tactical_log_probs_flat, compute_loss_flat
from ml_training.metrics import (
    TrainingMetrics, _load_hof_armies, _load_hof_ml_armies, _generate_army_pair,
)


def run_training(
    config: TrainingConfig | None = None,
    army_pairs: list[tuple[list[ResolvedUnit], list[ResolvedUnit]]] | None = None,
    verbose: bool = True,
    restart: bool = False,
) -> tuple[nn.Module, TrainingMetrics]:
    """Run the full PPO training loop.

    Parameters
    ----------
    config : training hyperparameters (uses defaults if None)
    army_pairs : optional fixed set of (army_a, army_b) tuples.
                 If None, generates random armies each game (requires evolution module).
    verbose : print progress to stdout

    Returns
    -------
    (trained_model, metrics)
    """
    if config is None:
        config = TrainingConfig()

    is_tactical = config.model_type == "tactical"
    device = _resolve_device(config.device)

    # Toggle C extension in main process
    import fast_core
    fast_core.USE_C_EXT = config.use_c_ext and fast_core.is_available()
    c_ext_label = "ON" if fast_core.USE_C_EXT else "OFF"

    # Load hall-of-fame armies for mixed training
    hof_armies = _load_hof_armies()
    hof_ml_armies = _load_hof_ml_armies()
    if verbose:
        print(f"Loaded {len(hof_armies)} HoF armies, {len(hof_ml_armies)} HoF-ML armies")

    if verbose:
        device_label = str(device)
        if device.type == "cuda":
            device_label += f" ({torch.cuda.get_device_name(device)})"
        print(f"Model type: {config.model_type} | C extension: {c_ext_label} | Device: {device_label}")

    model = _make_model(config.model_type)
    start_batch = 0
    if not restart:
        final_path = Path(config.checkpoint_dir) / "final_model.pt"
        if final_path.exists():
            sd = load_model_state_dict(final_path)
            model.load_state_dict(sd, strict=False)
            # Extract batch_num from checkpoint if available
            _ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
            if isinstance(_ckpt, dict) and "batch_num" in _ckpt:
                start_batch = _ckpt["batch_num"]
            if verbose:
                print(f"Resumed from {final_path} (batch {start_batch})")
        else:
            # No final_model.pt — try the newest checkpoint by creation time
            ckpt_dir = Path(config.checkpoint_dir)
            if ckpt_dir.exists():
                checkpoints = sorted(
                    ckpt_dir.glob("checkpoint_batch_*.pt"),
                    key=lambda p: p.stat().st_mtime,
                )
            else:
                checkpoints = []
            if checkpoints:
                newest = checkpoints[-1]
                sd = load_model_state_dict(newest)
                model.load_state_dict(sd, strict=False)
                _ckpt = torch.load(newest, map_location="cpu", weights_only=False)
                if isinstance(_ckpt, dict) and "batch_num" in _ckpt:
                    start_batch = _ckpt["batch_num"]
                if verbose:
                    print(f"No final_model.pt — resumed from newest checkpoint {newest.name} (batch {start_batch})")
            elif verbose:
                print("No final_model.pt or checkpoints found — starting from scratch")
    elif verbose:
        print("Restart requested — training from scratch")

    if restart:
        # Remove all previous checkpoint files
        ckpt_dir = Path(config.checkpoint_dir)
        if ckpt_dir.exists():
            existing = list(ckpt_dir.glob("checkpoint_batch_*.pt"))
            final = ckpt_dir / "final_model.pt"
            to_remove = len(existing) + (1 if final.exists() else 0)
            if to_remove > 0:
                answer = input(f"restart=True will DELETE {to_remove} checkpoint file(s) in {ckpt_dir}/. Continue? [y/N] ")
                if answer.strip().lower() != "y":
                    print("Aborted.")
                    raise SystemExit(1)
                for f in existing:
                    f.unlink()
                if final.exists():
                    final.unlink()
                if verbose:
                    print(f"Removed {to_remove} old checkpoint file(s)")

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Per-head entropy target tuner (tactical model only)
    entropy_tuner: EntropyTargetTuner | None = None
    alpha_optimizer: torch.optim.Adam | None = None
    if is_tactical and config.use_entropy_targets:
        entropy_tuner = EntropyTargetTuner(config).to(device)
        # Load tuner state if resuming
        if not restart:
            tuner_path = Path(config.checkpoint_dir) / "entropy_tuner.pt"
            if tuner_path.exists():
                entropy_tuner.load_state_dict(
                    torch.load(tuner_path, map_location="cpu", weights_only=True))
                if verbose:
                    print(f"Loaded entropy tuner from {tuner_path}")
        alpha_optimizer = torch.optim.Adam(
            entropy_tuner.parameters(), lr=config.entropy_alpha_lr)
        if verbose:
            print(f"Entropy targets: fraction={config.entropy_target_fraction}, "
                  f"move={config.entropy_target_move:.3f}, "
                  f"dest={config.entropy_target_dest:.3f}")
    if verbose and config.ppo_minibatch_games > 0:
        print(f"PPO minibatching: {config.ppo_minibatch_games} games per minibatch, "
              f"{config.ppo_epochs} epochs")
    if verbose and config.planning_rate > 0:
        _pr_end = f" → {config.planning_rate_end}" if config.planning_rate_end is not None else ""
        print(f"Planning-augmented training: rate={config.planning_rate}{_pr_end}, "
              f"K={config.training_planning_K} C={config.training_planning_C} "
              f"M={config.training_planning_M} N={config.training_planning_N}, "
              f"distill_max_weight={config.planning_distill_max_weight}")

    checkpoint_pool = CheckpointPool(
        max_size=config.max_checkpoints,
        save_dir=config.checkpoint_dir,
        model_type=config.model_type,
        seed_existing=0 if restart else config.max_checkpoints,
    )
    if not restart and checkpoint_pool.entries and verbose:
        print(f"Seeded checkpoint pool with {len(checkpoint_pool.entries)} existing checkpoint(s)")
    metrics = TrainingMetrics()

    # Open training log CSV (append mode so it survives restarts)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_{config.model_type}.csv"
    _log_is_new = restart or not log_path.exists() or log_path.stat().st_size == 0
    _log_file = open(log_path, "w" if restart else "a", newline="")
    _log_writer = csv.writer(_log_file)
    if _log_is_new:
        _log_writer.writerow([
            "timestamp", "batch", "loss", "mean_entropy", "entropy_coeff",
            "mean_reward", "h_hof_wr", "h_ml_wr", "sp_hof_wr", "sp_ml_wr",
            "sp_rnd_wr", "h_frac", "batch_time", "aux_loss",
            "weighted_aux", "non_aux_loss", "clip_frac",
            "ent_unit", "ent_move", "ent_dest",
            "ent_charge", "ent_shoot",
            "alpha_unit", "alpha_move", "alpha_dest",
            "alpha_charge", "alpha_shoot",
            "val_heuristic", "val_sp_mirror", "val_sp_hof",
            "val_sp_ml", "val_sp_random",
            "plan_activations", "plan_improve_rate",
            "plan_mean_vdelta", "plan_distill_loss",
            "plan_argmax_rate",
            "plan_dl_unit", "plan_dl_move",
            "plan_dl_charge", "plan_dl_shoot",
            "dest_obj_prox",
        ])
    _log_writer.writerow([datetime.now().isoformat(), "---",
                          f"Training started (start_batch={start_batch})",
                          "", "", "", "", "", "", "", "", "", ""])
    _log_file.flush()

    # Save initial checkpoint
    checkpoint_pool.save(model, 0)

    start_time = time.time()
    batch_times: list[float] = []

    # --- Shared-memory pool setup ---
    worker_count = config.worker_count if config.worker_count is not None else _WORKER_COUNT

    shared_model = _make_model(config.model_type)
    shared_model.share_memory()
    shared_model.eval()

    shared_opponents: list[nn.Module] = []
    for _ in range(_MAX_SHARED_OPPONENTS):
        m = _make_model(config.model_type)
        m.share_memory()
        m.eval()
        shared_opponents.append(m)

    ctx = _mp.get_context('spawn')
    pool = ctx.Pool(
        processes=worker_count,
        initializer=_init_shared_worker,
        initargs=(shared_model, shared_opponents, config.model_type,
                  config.use_c_ext),
    )

    for batch_num in range(start_batch + 1, start_batch + config.num_batches + 1):
        batch_start = time.time()
        heuristic_fraction = get_heuristic_fraction(metrics.heuristic_win_rate)

        # --- Phase 1: build game specs and deduplicate opponent weights ---
        # Copy current training weights to shared memory (map to CPU for workers)
        cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()} if device.type != "cpu" else model.state_dict()
        shared_model.load_state_dict(cpu_sd)

        game_specs = []           # per-game: (res_a, res_b, sa_data, sb_data, opp_type, opp_sd_index)
        opponent_state_dicts = [] # deduplicated list of opponent state dicts
        _opp_path_cache: dict[str, int] = {}  # checkpoint path -> index into opponent_state_dicts

        for game_idx in range(config.batch_size):
            # Select opponent
            if random.random() < heuristic_fraction:
                opp_type = "heuristic"
                opp_sd_idx = -1
            elif random.random() < 0.5:
                # Mirror self-play: current model plays both sides, learn from both
                opp_type = "selfplay_mirror"
                opp_sd_idx = -1
            else:
                opp_path = checkpoint_pool.sample_opponent_path()
                if opp_path is not None:
                    opp_type = "selfplay"
                    path_key = str(opp_path)
                    if path_key not in _opp_path_cache:
                        _opp_path_cache[path_key] = len(opponent_state_dicts)
                        opponent_state_dicts.append(
                            checkpoint_pool.load_state_dict(opp_path))
                    opp_sd_idx = _opp_path_cache[path_key]
                else:
                    # No checkpoints available yet, fall back to mirror
                    opp_type = "selfplay_mirror"
                    opp_sd_idx = -1

            # Generate or sample armies
            if army_pairs is not None:
                res_a, res_b = random.choice(army_pairs)
                states_a_data = [("killer", "ranged", -1)] * len(res_a)
                states_b_data = [("killer", "ranged", -1)] * len(res_b)
                army_type = "random"
            else:
                res_a, res_b, states_a, states_b, army_type = _generate_army_pair(
                    opp_type=opp_type, hof_armies=hof_armies,
                    hof_ml_armies=hof_ml_armies)
                states_a_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a]
                states_b_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b]

            game_specs.append((res_a, res_b, states_a_data, states_b_data, opp_type, opp_sd_idx, army_type))

        # --- Phase 2: load opponent weights into shared memory, dispatch ---
        # Copy unique opponent state dicts into shared opponent model slots
        opp_slot_map: dict[int, int] = {}
        for i, sd in enumerate(opponent_state_dicts):
            if i < _MAX_SHARED_OPPONENTS:
                shared_opponents[i].load_state_dict(sd, strict=False)
                opp_slot_map[i] = i

        # Compute reward shaping scale (anneals linearly to 0)
        # When resuming a prior run, disable shaping entirely — the model
        # has already graduated past the shaping phase.
        if not restart and start_batch > 0:
            shaping_scale = 0.0
        elif config.shaping_anneal_end > 0:
            shaping_progress = (batch_num - start_batch) / config.num_batches
            if config.time_limit is not None:
                elapsed_min = (time.time() - start_time) / 60.0
                shaping_progress = max(shaping_progress, elapsed_min / config.time_limit)
            shaping_progress = min(shaping_progress, 1.0)
            shaping_scale = max(0.0, 1.0 - shaping_progress / config.shaping_anneal_end)
        else:
            shaping_scale = 0.0

        # Build planning config for workers (if planning is enabled)
        planning_config_for_workers = None
        if config.planning_rate > 0:
            # Compute effective planning rate (with optional annealing)
            _plan_progress = (batch_num - start_batch) / config.num_batches
            if config.time_limit is not None:
                _plan_progress = max(_plan_progress,
                                     (time.time() - start_time) / 60.0 / config.time_limit)
            _plan_progress = min(_plan_progress, 1.0)
            if config.planning_rate_end is not None:
                effective_planning_rate = (
                    config.planning_rate
                    + _plan_progress * (config.planning_rate_end - config.planning_rate)
                )
            else:
                effective_planning_rate = config.planning_rate
            planning_config_for_workers = {
                "planning_rate": effective_planning_rate,
                "planning_params": {
                    "K_UNITS": config.training_planning_K,
                    "C_SAMPLES_PER_UNIT": config.training_planning_C,
                    "M_ROLLOUTS": config.training_planning_M,
                    "N_LOOKAHEAD": config.training_planning_N,
                },
            }

        n_chunks = worker_count
        chunk_size = max(1, len(game_specs) // n_chunks)
        chunks = []
        for i in range(0, len(game_specs), chunk_size):
            chunk = game_specs[i : i + chunk_size]
            chunks.append((opp_slot_map, chunk, shaping_scale,
                           planning_config_for_workers))

        chunk_results = list(pool.map(_collect_episodes_shared_worker, chunks))
        trajectories = [ep for chunk in chunk_results for ep in chunk]

        # --- Phase 3: compute GAE advantages (fixed across PPO epochs) ---
        model.train()
        all_trajs = [traj_rounds for traj_rounds, _, _, _ in trajectories]
        all_advantages, all_returns = compute_gae(
            all_trajs, gamma=1.0, gae_lambda=config.gae_lambda,
            unit_local_blend=config.unit_local_advantage_blend,
        )

        # Record game outcomes for metrics tracking
        # Skip mirror_b entries — they share the same game as the A-side entry
        opp_types = []
        for _, result, opp_type, army_type in trajectories:
            if opp_type != "mirror_b":
                metrics.record_game(result, opp_type, army_type)
            opp_types.append(opp_type)

        # --- Phase 4: PPO multi-epoch update ---
        # Anneal entropy coefficient: linear from start to end
        progress2 = (batch_num - start_batch) / config.num_batches
        if config.time_limit is not None:
            elapsed_minutes = (time.time() - start_time) / 60.0
            progress1 = elapsed_minutes / config.time_limit
            progress = max(progress1, progress2)
        else:
            progress = progress2
        progress = min(progress, 1.0)
        entropy_coeff = config.entropy_coeff_start + progress * (config.entropy_coeff_end - config.entropy_coeff_start)

        # Pre-flatten advantages/returns/old_log_probs for vectorized tactical path
        if is_tactical:
            flat_old_lps = torch.tensor(
                [s.old_log_prob for traj in all_trajs for s in traj],
                dtype=torch.float32, device=device,
            )
            flat_advantages_t = torch.tensor(
                [a for adv in all_advantages for a in adv],
                dtype=torch.float32, device=device,
            )
            flat_returns_t = torch.tensor(
                [r for ret in all_returns for r in ret],
                dtype=torch.float32, device=device,
            )
            # Precompute per-game step counts and cumulative offsets for minibatching
            game_step_counts = [len(traj) for traj in all_trajs]
            game_step_offsets = [0] * len(all_trajs)
            for gi in range(1, len(all_trajs)):
                game_step_offsets[gi] = game_step_offsets[gi - 1] + game_step_counts[gi - 1]

        # Snapshot model weights before PPO epochs so we can rollback on NaN
        pre_ppo_state = {k: v.clone() for k, v in model.state_dict().items()}

        minibatch_games = config.ppo_minibatch_games
        nan_detected = False
        for _ppo_epoch in range(config.ppo_epochs):
            if is_tactical and minibatch_games > 0 and len(all_trajs) > minibatch_games:
                # --- Minibatched PPO for tactical model ---
                game_indices = list(range(len(all_trajs)))
                random.shuffle(game_indices)
                epoch_metrics: dict[str, float] = {}
                epoch_count = 0

                for mb_start in range(0, len(game_indices), minibatch_games):
                    mb_game_idx = game_indices[mb_start:mb_start + minibatch_games]

                    # Gather trajectories and flat tensor slices for this minibatch
                    mb_trajs = [all_trajs[i] for i in mb_game_idx]
                    mb_flat_idx: list[int] = []
                    for gi in mb_game_idx:
                        off = game_step_offsets[gi]
                        mb_flat_idx.extend(range(off, off + game_step_counts[gi]))
                    if not mb_flat_idx:
                        continue
                    idx_t = torch.tensor(mb_flat_idx, dtype=torch.long, device=device)
                    mb_old_lps = flat_old_lps[idx_t]
                    mb_advantages = flat_advantages_t[idx_t]
                    mb_returns = flat_returns_t[idx_t]

                    # Forward pass + loss on minibatch (device-aware)
                    with _force_tensor_device(device):
                        mb_flat_result = replay_tactical_log_probs_flat(model, mb_trajs)
                        mb_flat_steps = [s for traj in mb_trajs for s in traj]
                        loss, loss_metrics = compute_loss_flat(
                            mb_flat_result, mb_old_lps, mb_advantages, mb_returns,
                            config.clip_epsilon, config.value_coeff, entropy_coeff,
                            aux_coeff=config.aux_coeff, aux_ratio=config.aux_ratio,
                            flat_steps=mb_flat_steps,
                            entropy_tuner=entropy_tuner,
                            planning_distill_max_weight=(
                                config.planning_distill_max_weight
                                if config.planning_rate > 0 else 0.0),
                            dest_obj_proximity_coeff=config.dest_obj_proximity_coeff,
                        )

                    optimizer.zero_grad()
                    if alpha_optimizer is not None:
                        alpha_optimizer.zero_grad()
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  WARNING: NaN/Inf loss at batch {batch_num}, rolling back weights")
                        model.load_state_dict(pre_ppo_state)
                        nan_detected = True
                        break
                    loss.backward()

                    # Backprop alpha loss separately
                    alpha_loss_tensor = loss_metrics.pop("_alpha_loss_tensor", None)
                    if alpha_loss_tensor is not None and alpha_optimizer is not None:
                        alpha_loss_tensor.backward()
                        alpha_optimizer.step()

                    # Check for NaN in gradients before stepping
                    grad_nan = any(
                        p.grad is not None and torch.isnan(p.grad).any()
                        for p in model.parameters()
                    )
                    if grad_nan:
                        print(f"  WARNING: NaN gradients at batch {batch_num}, rolling back weights")
                        model.load_state_dict(pre_ppo_state)
                        nan_detected = True
                        break
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Accumulate metrics for logging (weighted by step count)
                    n_mb = len(mb_flat_idx)
                    for k, v in loss_metrics.items():
                        if isinstance(v, (int, float)):
                            epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v * n_mb
                    # Accumulate per-head entropy separately
                    phe_mb = loss_metrics.get("per_head_entropy", {})
                    for hk, hv in phe_mb.items():
                        epoch_metrics[f"_phe_{hk}"] = epoch_metrics.get(f"_phe_{hk}", 0.0) + hv * n_mb
                    # Accumulate per-opponent-type value estimates
                    opp_val_mb = loss_metrics.get("per_opp_type_mean_values", {})
                    for ok, ov in opp_val_mb.items():
                        epoch_metrics[f"_opv_{ok}"] = epoch_metrics.get(f"_opv_{ok}", 0.0) + ov * n_mb
                    # Accumulate per-head planning distillation sub-losses
                    pds_mb = loss_metrics.get("planning_distill_sub", {})
                    for pk, pv in pds_mb.items():
                        epoch_metrics[f"_pds_{pk}"] = epoch_metrics.get(f"_pds_{pk}", 0.0) + pv * n_mb
                    epoch_count += n_mb

                if nan_detected:
                    break
                # Average metrics across minibatches
                if epoch_count > 0:
                    phe_agg = {}
                    opv_agg = {}
                    pds_agg = {}
                    non_phe = {}
                    for k, v in epoch_metrics.items():
                        if k.startswith("_phe_"):
                            phe_agg[k[5:]] = v / epoch_count
                        elif k.startswith("_opv_"):
                            opv_agg[k[5:]] = v / epoch_count
                        elif k.startswith("_pds_"):
                            pds_agg[k[5:]] = v / epoch_count
                        else:
                            non_phe[k] = v / epoch_count
                    loss_metrics = non_phe
                    loss_metrics["per_head_entropy"] = phe_agg
                    loss_metrics["per_opp_type_mean_values"] = opv_agg
                    loss_metrics["planning_distill_sub"] = pds_agg

            elif is_tactical:
                # Full-batch tactical path (minibatch disabled or batch too small)
                with _force_tensor_device(device):
                    flat_result = replay_tactical_log_probs_flat(model, all_trajs)
                    _flat_steps = [s for traj in all_trajs for s in traj]
                    loss, loss_metrics = compute_loss_flat(
                        flat_result, flat_old_lps, flat_advantages_t, flat_returns_t,
                        config.clip_epsilon, config.value_coeff, entropy_coeff,
                        aux_coeff=config.aux_coeff, aux_ratio=config.aux_ratio,
                        flat_steps=_flat_steps,
                        entropy_tuner=entropy_tuner,
                        planning_distill_max_weight=(
                            config.planning_distill_max_weight
                            if config.planning_rate > 0 else 0.0),
                        dest_obj_proximity_coeff=config.dest_obj_proximity_coeff,
                    )

                optimizer.zero_grad()
                if alpha_optimizer is not None:
                    alpha_optimizer.zero_grad()
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  WARNING: NaN/Inf loss at batch {batch_num}, rolling back weights")
                    model.load_state_dict(pre_ppo_state)
                    nan_detected = True
                    break
                loss.backward()

                alpha_loss_tensor = loss_metrics.pop("_alpha_loss_tensor", None)
                if alpha_loss_tensor is not None and alpha_optimizer is not None:
                    alpha_loss_tensor.backward()
                    alpha_optimizer.step()

                grad_nan = any(
                    p.grad is not None and torch.isnan(p.grad).any()
                    for p in model.parameters()
                )
                if grad_nan:
                    print(f"  WARNING: NaN gradients at batch {batch_num}, rolling back weights")
                    model.load_state_dict(pre_ppo_state)
                    nan_detected = True
                    break
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # Log
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        batch_log = metrics.log_batch(batch_num, loss_metrics, heuristic_fraction)
        if verbose:
            recent_avg = sum(batch_times[-10:]) / len(batch_times[-10:])
            batches_remaining = (start_batch + config.num_batches) - batch_num
            est_remaining = recent_avg * batches_remaining
            if config.time_limit is not None:
                time_limit_remaining = config.time_limit * 60 - (time.time() - start_time)
                est_remaining = min(est_remaining, max(0, time_limit_remaining))
            eft = datetime.now() + timedelta(seconds=est_remaining)
            eft_str = eft.strftime("%H:%M")

            w_aux = loss_metrics.get('weighted_aux', 0.0)
            non_aux = loss_metrics.get('non_aux_loss', loss_metrics['loss'])
            aux_str = f" | Loss(policy={non_aux:.4f} aux={w_aux:.4f})" if w_aux != 0.0 else ""
            plan_acts = loss_metrics.get('planning_activations', 0)
            plan_str = ""
            if plan_acts > 0:
                plan_ir = loss_metrics.get('planning_improvement_rate', 0.0)
                plan_ar = loss_metrics.get('planning_argmax_rate', 0.0)
                plan_dl = loss_metrics.get('planning_distill_loss', 0.0)
                plan_str = f" | Plan({int(round(plan_acts))} ir={plan_ir:.2f} ar={plan_ar:.2f} dl={plan_dl:.4f})"
            phe = loss_metrics.get('per_head_entropy', {})
            if entropy_tuner is not None:
                alphas = entropy_tuner.alpha_summary()
                ent_str = (f"Entropy: {loss_metrics['mean_entropy']:.4f} "
                           f"[u={phe.get('unit', 0):.3f} m={phe.get('move', 0):.3f} "
                           f"dst={phe.get('dest', 0):.3f} "
                           f"c={phe.get('charge', 0):.3f} s={phe.get('shoot', 0):.3f}] "
                           f"(α u={alphas['unit']:.3f} m={alphas['move']:.3f} "
                           f"dst={alphas['dest']:.3f} "
                           f"c={alphas['charge']:.3f} s={alphas['shoot']:.3f})")
            else:
                ent_str = (f"Entropy: {loss_metrics['mean_entropy']:.4f} "
                           f"[u={phe.get('unit', 0):.3f} m={phe.get('move', 0):.3f} "
                           f"dst={phe.get('dest', 0):.3f} "
                           f"c={phe.get('charge', 0):.3f} s={phe.get('shoot', 0):.3f}] "
                           f"(c={entropy_coeff:.4f})")
            print(
                f"Batch {batch_num:04d} | "
                f"Loss: {loss_metrics['loss']:.4f} | "
                f"{ent_str} | "
                f"Reward: {loss_metrics['mean_reward']:.3f}{aux_str}{plan_str} | "
                f"H-HoF: {metrics.heuristic_hof_win_rate:.3f} | "
                f"H-ML: {metrics.heuristic_hof_ml_win_rate:.3f} | "
                f"SP-HoF: {metrics.selfplay_hof_win_rate:.3f} | "
                f"SP-ML: {metrics.selfplay_hof_ml_win_rate:.3f} | "
                f"SP-Rnd: {metrics.selfplay_random_win_rate:.3f} | "
                f"H-Frac: {heuristic_fraction:.2f} | "
                f"Clip: {loss_metrics.get('clip_frac', 0.0):.3f} | "
                f"{batch_time:.1f}s | EFT {eft_str}",
                flush=True,
            )

        # Log to CSV
        if entropy_tuner is not None:
            alphas = entropy_tuner.alpha_summary()
            alpha_cols = [f"{alphas[k]:.4f}" for k in EntropyTargetTuner.HEAD_NAMES]
        else:
            alpha_cols = [""] * len(EntropyTargetTuner.HEAD_NAMES)
        _opp_val_dict = loss_metrics.get("per_opp_type_mean_values", {})
        _opp_val_cols = [
            f"{_opp_val_dict.get(k, '')}"
            for k in ("mean_value_heuristic", "mean_value_sp_mirror",
                       "mean_value_sp_hof", "mean_value_sp_ml", "mean_value_sp_random")
        ]
        _log_writer.writerow([
            datetime.now().isoformat(), batch_num,
            f"{loss_metrics['loss']:.4f}",
            f"{loss_metrics['mean_entropy']:.4f}",
            f"{entropy_coeff:.4f}",
            f"{loss_metrics['mean_reward']:.3f}",
            f"{metrics.heuristic_hof_win_rate:.3f}",
            f"{metrics.heuristic_hof_ml_win_rate:.3f}",
            f"{metrics.selfplay_hof_win_rate:.3f}",
            f"{metrics.selfplay_hof_ml_win_rate:.3f}",
            f"{metrics.selfplay_random_win_rate:.3f}",
            f"{heuristic_fraction:.2f}",
            f"{batch_time:.1f}",
            f"{loss_metrics.get('aux_loss', 0.0):.4f}",
            f"{loss_metrics.get('weighted_aux', 0.0):.4f}",
            f"{loss_metrics.get('non_aux_loss', 0.0):.4f}",
            f"{loss_metrics.get('clip_frac', 0.0):.4f}",
            *[f"{loss_metrics.get('per_head_entropy', {}).get(k, 0.0):.4f}"
              for k in ("unit", "move", "dest", "charge", "shoot")],
            *alpha_cols,
            *_opp_val_cols,
            f"{loss_metrics.get('planning_activations', 0)}",
            f"{loss_metrics.get('planning_improvement_rate', 0.0):.4f}",
            f"{loss_metrics.get('planning_mean_value_delta', 0.0):.4f}",
            f"{loss_metrics.get('planning_distill_loss', 0.0):.6f}",
            f"{loss_metrics.get('planning_argmax_rate', 0.0):.4f}",
            *[f"{loss_metrics.get('planning_distill_sub', {}).get(k, 0.0):.6f}"
              for k in ("unit", "move", "charge", "shoot")],
            f"{loss_metrics.get('dest_obj_proximity', 0.0):.4f}",
        ])
        _log_file.flush()

        # Checkpoint
        if batch_num % config.checkpoint_interval == 0:
            checkpoint_pool.save(model, batch_num)

        # Time limit check
        if config.time_limit is not None:
            elapsed = time.time() - start_time
            if elapsed >= config.time_limit * 60:
                if verbose:
                    print(f"\nTIME LIMIT reached ({config.time_limit} min) after batch {batch_num}.")
                break

    pool.close()
    pool.join()

    # Move model back to CPU for saving and downstream use
    model.to("cpu")
    if entropy_tuner is not None:
        entropy_tuner.to("cpu")

    # Close training log
    _log_writer.writerow([datetime.now().isoformat(), "---",
                          f"Training finished (batch {batch_num})",
                          "", "", "", "", "", "", "", "", "", ""])
    _log_file.close()

    # Save final model
    final_path = Path(config.checkpoint_dir) / "final_model.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "batch_num": batch_num}, final_path)

    # Save entropy tuner state (separate file for easy loading)
    if entropy_tuner is not None:
        tuner_path = Path(config.checkpoint_dir) / "entropy_tuner.pt"
        torch.save(entropy_tuner.state_dict(), tuner_path)

    return model, metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TrainingConfig(
        num_batches=10,
        batch_size=8,
        checkpoint_dir="ml_checkpoints_test",
    )
    model, metrics = run_training(config=cfg, verbose=True)
    print(f"\nTraining complete. Final heuristic win rate: {metrics.heuristic_win_rate:.3f}")
