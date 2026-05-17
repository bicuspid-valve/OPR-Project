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
from ml_training.loss import (
    replay_tactical_log_probs_flat, compute_loss_flat,
    prepare_replay_data, replay_from_prepared,
)
from ml_training.metrics import (
    TrainingMetrics, _load_hof_armies, _load_hof_ml_armies, _generate_army_pair,
)


def _format_per_phase_value(loss_metrics: dict) -> list[str]:
    """Format per-phase value-head diagnostics for CSV output.

    Returns 9 cells: total_loss + 4 per-phase losses + 4 per-phase means.
    Reads from the scalar-mirror keys (per_phase_value_{mean,loss}_{pre,sel,mt,dest})
    rather than the list-valued keys — the PPO minibatch accumulator drops
    non-scalar values, so the lists never survive minibatch aggregation.
    Writes empty strings when the scalar keys are absent (flag was off).
    """
    # Presence of *any* per-phase scalar key signals the flag was on for at
    # least one minibatch of this batch — enough to emit numeric cells.
    phases = ("pre", "sel", "mt", "dest")
    if not any(f"per_phase_value_mean_{p}" in loss_metrics for p in phases):
        return [""] * 9
    total = loss_metrics.get("per_phase_value_loss", 0.0)
    return [
        f"{total:.6f}",
        *(f"{loss_metrics.get(f'per_phase_value_loss_{p}', 0.0):.6f}" for p in phases),
        *(f"{loss_metrics.get(f'per_phase_value_mean_{p}', 0.0):.6f}" for p in phases),
    ]


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

    # Load the map once at training startup, before the multiprocessing
    # workers are spawned (the MapData dataclass is pickleable, so workers
    # receive a copy in each dispatch chunk). When map_path is None we keep
    # the legacy empty-board layout.
    map_data = None
    if config.map_path is not None:
        from map_loader import load_map
        map_data = load_map(config.map_path)
        # Install the map in the main process too so any in-process code
        # paths (e.g. validation games) see the right OBJECTIVES.
        from board import Board as _Board
        from map_loader import apply_map as _apply_map
        _apply_map(_Board(), map_data, build_vis_cover=False)
        if verbose:
            print(f"Loaded map: {config.map_path} "
                  f"({len(map_data.terrain)} terrain pieces, "
                  f"{len(map_data.objectives)} objectives, "
                  f"DZ-A={len(map_data.deployment_a)} cells, "
                  f"DZ-B={len(map_data.deployment_b)} cells)")
            print(f"Train deployment: {config.train_deployment}")

    # Toggle C extension in main process
    import fast_core
    fast_core.USE_C_EXT = config.use_c_ext and fast_core.is_available()
    c_ext_label = "ON" if fast_core.USE_C_EXT else "OFF"

    # Toggle phase-reencode path in main process (propagated to workers via initargs).
    from ml_integration_tactical import set_phase_reencode_enabled, is_phase_reencode_enabled
    set_phase_reencode_enabled(config.phase_reencode_enabled)
    if verbose:
        print(f"Phase re-encode: config={config.phase_reencode_enabled} "
              f"flag={is_phase_reencode_enabled()}")

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
            if isinstance(_ckpt, dict) and "n_iters" in _ckpt:
                model.n_iters = _ckpt["n_iters"]
            if verbose:
                print(f"Resumed from {final_path} (batch {start_batch})")
        else:
            # No final_model.pt — try the checkpoint with the highest batch number
            import re
            ckpt_dir = Path(config.checkpoint_dir)
            if ckpt_dir.exists():
                checkpoints = sorted(
                    ckpt_dir.glob("checkpoint_batch_*.pt"),
                    key=lambda p: int(m.group(1)) if (m := re.search(r"checkpoint_batch_(\d+)", p.stem)) else -1,
                )
            else:
                checkpoints = []
            if checkpoints:
                newest = checkpoints[-1]
                sd = load_model_state_dict(newest)
                model.load_state_dict(sd, strict=False)
                m = re.search(r"checkpoint_batch_(\d+)", newest.stem)
                if m:
                    start_batch = int(m.group(1))
                if verbose:
                    print(f"No final_model.pt — resumed from newest checkpoint {newest.name} (batch {start_batch})")
            elif verbose:
                print("No final_model.pt or checkpoints found — starting from scratch")
    elif verbose:
        print("Restart requested — training from scratch")

    # V_old shaping: capture the pre-restart final_model.pt state_dict NOW —
    # the deletion block below will unlink the file. We instantiate the
    # shared model lower down once we know the deletion was confirmed.
    v_old_state_dict = None
    if config.shaping_old_value:
        if not restart:
            raise ValueError(
                "shaping_old_value=True requires restart=True (V_old is the "
                "frozen pre-restart final_model; without restart the live "
                "model is the same checkpoint)."
            )
        final_path = Path(config.checkpoint_dir) / "final_model.pt"
        if not final_path.exists():
            raise FileNotFoundError(
                f"shaping_old_value=True but {final_path} not found. "
                "Run a baseline training first to produce final_model.pt."
            )
        v_old_state_dict = load_model_state_dict(final_path)
        if verbose:
            print(f"V_old shaping enabled — captured value head from {final_path}")

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

    # Phase 1 MPO scaffolding: a parallel π_old snapshot used by the
    # Phase 2 KL trust-region term. Materialised only when
    # config.kl_trust_region_beta > 0 so legacy runs pay nothing — no
    # extra parameters, no extra forward pass, no extra memory. The
    # snapshot is refreshed from `model.state_dict()` once per outer
    # batch iteration (see snapshot site below), matching the
    # pre_ppo_state cadence.
    model_old: nn.Module | None = None
    if config.kl_trust_region_beta > 0.0:
        model_old = _make_model(config.model_type)
        model_old.load_state_dict(model.state_dict(), strict=True)
        model_old.to(device)
        model_old.eval()
        for _p in model_old.parameters():
            _p.requires_grad_(False)
        if verbose:
            print(f"KL trust region snapshot active (β={config.kl_trust_region_beta})")

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
                  f"dest_frac={config.entropy_target_dest_fraction:.3f}")
    if verbose and config.ppo_minibatch_games > 0:
        print(f"PPO minibatching: {config.ppo_minibatch_games} games per minibatch, "
              f"{config.ppo_epochs} epochs")
    if verbose and config.planning_rate > 0:
        _pr_end = f" → {config.planning_rate_end}" if config.planning_rate_end is not None else ""
        print(f"Planning-augmented training: rate={config.planning_rate}{_pr_end}, "
              f"K={config.training_planning_K} C={config.training_planning_C} "
              f"M={config.training_planning_M} N={config.training_planning_N}, "
              f"distill_max_weight={config.planning_distill_max_weight} "
              f"distill_mode={config.planning_distill_mode}")

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
            "plan_distill_ramp",
            "plan_argmax_rate",
            "plan_dl_unit", "plan_dl_move",
            "plan_dl_charge", "plan_dl_shoot",
            "plan_dl_dest",
            "plan_tgt_peak_unit", "plan_pol_peak_unit",
            "plan_tgt_peak_move", "plan_pol_peak_move",
            "plan_tgt_peak_charge", "plan_pol_peak_charge",
            "plan_tgt_peak_shoot", "plan_pol_peak_shoot",
            "plan_tgt_peak_dest", "plan_pol_peak_dest",
            "plan_argmax_agree_unit", "plan_argmax_agree_move",
            "plan_argmax_agree_charge", "plan_argmax_agree_shoot",
            "plan_argmax_agree_dest",
            "wr_side_a", "wr_side_b", "val_side_a", "val_side_b",
            "shoot_eff_reward",
            "charge_eff_reward",
            "ml_h_shoot_eff",
            "ml_h_charge_eff",
            "h_shoot_eff_reward",
            "h_charge_eff_reward",
            # Main value loss — comparison baseline for per-phase V head loss.
            "value_loss",
            # Per-phase value head diagnostics (phase_reencode flag only).
            # Empty when flag is off. Phase order: pre/sel/mt/dest matches
            # PHASE_PRE_SELECT/POST_SELECT/POST_MOVETYPE/POST_DEST.
            "pp_v_loss", "pp_v_loss_pre", "pp_v_loss_sel",
            "pp_v_loss_mt", "pp_v_loss_dest",
            "pp_v_mean_pre", "pp_v_mean_sel",
            "pp_v_mean_mt", "pp_v_mean_dest",
            # MPO trust region — reported for every batch but only nonzero
            # when β > 0 and a model_old snapshot is active. Useful for
            # verifying the switch fired post-mpo_switch_batch and for
            # tracking whether π is staying close to the per-batch snapshot.
            "kl_trust_region_beta", "kl_trust_region_loss",
            "kl_trust_unit", "kl_trust_move", "kl_trust_dest",
            "kl_trust_charge", "kl_trust_shoot",
        ])
    _log_writer.writerow([datetime.now().isoformat(), "---",
                          f"Training started (start_batch={start_batch})",
                          "", "", "", "", "", "", "", "", "", ""])
    _log_file.flush()

    # Save initial checkpoint (use start_batch so resumed runs don't
    # overwrite the batch-number signal with 0)
    checkpoint_pool.save(model, start_batch)

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

    shared_v_old: nn.Module | None = None
    if v_old_state_dict is not None:
        shared_v_old = _make_model(config.model_type)
        shared_v_old.load_state_dict(v_old_state_dict, strict=False)
        shared_v_old.eval()
        for _p in shared_v_old.parameters():
            _p.requires_grad_(False)
        shared_v_old.share_memory()

    ctx = _mp.get_context('spawn')
    pool = ctx.Pool(
        processes=worker_count,
        initializer=_init_shared_worker,
        initargs=(shared_model, shared_opponents, config.model_type,
                  config.use_c_ext, config.phase_reencode_enabled,
                  shared_v_old),
    )

    # --- Helper: build game specs, load opponent weights, dispatch workers ---
    def _build_and_dispatch(batch_num_for_specs, async_mode=False):
        """Build game specs and dispatch episode collection.

        Returns (async_result, shaping_scale) if async_mode, else
        (trajectories, shaping_scale).
        """
        hf = get_heuristic_fraction(metrics.heuristic_win_rate)

        game_specs = []
        opponent_state_dicts = []
        _opp_path_cache: dict[str, int] = {}

        for _ in range(config.batch_size):
            if random.random() < hf:
                opp_type = "heuristic"
                opp_sd_idx = -1
            elif random.random() < 0.5:
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
                    opp_type = "selfplay_mirror"
                    opp_sd_idx = -1

            if army_pairs is not None:
                res_a, res_b = random.choice(army_pairs)
                states_a_data = [("killer", "ranged", -1)] * len(res_a)
                states_b_data = [("killer", "ranged", -1)] * len(res_b)
                # No hero attachments in army_pairs path — entries map 1:1
                # to UnitStates already (assumed to be hero-merged upstream).
                attach_a: list[tuple[int, bool]] = [(-1, False)] * len(res_a)
                attach_b: list[tuple[int, bool]] = [(-1, False)] * len(res_b)
                army_type = "random"
            else:
                # Request the per-entry hero-attach data so the generator
                # can redo the merge on its fresh UnitState list. Without
                # this, HoF armies with len(entries) > MAX_UNITS_PER_SIDE
                # (heroes inflate the entry count past 10) crash deploy_armies
                # because the unmerged heroes overflow the model's slot grid.
                (res_a, res_b, states_a, states_b, army_type,
                 attach_a, attach_b) = _generate_army_pair(
                    opp_type=opp_type, hof_armies=hof_armies,
                    hof_ml_armies=hof_ml_armies,
                    return_attach_data=True)
                states_a_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a]
                states_b_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b]

            game_specs.append((res_a, res_b, states_a_data, states_b_data,
                               opp_type, opp_sd_idx, army_type, attach_a, attach_b))

        opp_slot_map: dict[int, int] = {}
        for i, sd in enumerate(opponent_state_dicts):
            if i < _MAX_SHARED_OPPONENTS:
                shared_opponents[i].load_state_dict(sd, strict=False)
                opp_slot_map[i] = i

        # Shaping scale
        if not restart and start_batch > 0:
            ss = 0.0
        elif config.shaping_anneal_end > 0:
            sp = (batch_num_for_specs - start_batch) / config.num_batches
            if config.time_limit is not None:
                sp = max(sp, (time.time() - start_time) / 60.0 / config.time_limit)
            sp = min(sp, 1.0)
            ss = max(0.0, 1.0 - sp / config.shaping_anneal_end)
        else:
            ss = 0.0

        # Planning config
        pcw = None
        if config.planning_rate > 0:
            pp = (batch_num_for_specs - start_batch) / config.num_batches
            if config.time_limit is not None:
                pp = max(pp, (time.time() - start_time) / 60.0 / config.time_limit)
            pp = min(pp, 1.0)
            base_rate = (config.planning_rate + pp * (config.planning_rate_end - config.planning_rate)
                         if config.planning_rate_end is not None else config.planning_rate)
            # Warmup/ramp: 0 during warmup, then linear 0->1 over ramp window.
            b0 = batch_num_for_specs - start_batch
            warmup = config.planning_warmup_batches
            ramp = config.planning_distill_ramp_batches
            if b0 <= warmup:
                ramp_frac = 0.0
            elif ramp > 0:
                ramp_frac = min(1.0, (b0 - warmup) / ramp)
            else:
                ramp_frac = 1.0
            epr = base_rate * ramp_frac
            if epr > 0:
                pcw = {
                    "planning_rate": epr,
                    "planning_params": {
                        "K_UNITS": config.training_planning_K,
                        "C_SAMPLES_PER_UNIT": config.training_planning_C,
                        "M_ROLLOUTS": config.training_planning_M,
                        "N_LOOKAHEAD": config.training_planning_N,
                        "SEQUENTIAL_HALVING": config.training_planning_sequential_halving,
                        "SH_SCHEDULE": tuple(config.training_planning_sh_schedule),
                    },
                }

        n_chunks = worker_count
        chunk_sz = max(1, len(game_specs) // n_chunks)
        chunks = []
        v_old_active = shared_v_old is not None
        for ci in range(0, len(game_specs), chunk_sz):
            chunks.append((opp_slot_map, game_specs[ci:ci + chunk_sz], ss,
                            pcw, v_old_active, map_data, config.train_deployment))

        if async_mode:
            return pool.map_async(_collect_episodes_shared_worker, chunks), ss
        else:
            results = list(pool.map(_collect_episodes_shared_worker, chunks))
            return [ep for chunk in results for ep in chunk], ss

    # --- Dispatch first batch synchronously ---
    cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()} if device.type != "cpu" else model.state_dict()
    shared_model.load_state_dict(cpu_sd)
    pending_result, _ = _build_and_dispatch(start_batch + 1, async_mode=True)

    for batch_num in range(start_batch + 1, start_batch + config.num_batches + 1):
        batch_start = time.time()

        # --- Wait for current batch's episode collection ---
        chunk_results = pending_result.get()
        trajectories = [ep for chunk in chunk_results for ep in chunk]
        heuristic_fraction = get_heuristic_fraction(metrics.heuristic_win_rate)

        # --- Dispatch NEXT batch's collection (overlaps with PPO below) ---
        is_last = batch_num >= start_batch + config.num_batches
        if not is_last:
            # Time limit: skip dispatch if we'd exceed it
            _skip_next = False
            if config.time_limit is not None:
                if (time.time() - start_time) >= config.time_limit * 60:
                    _skip_next = True
            if not _skip_next:
                cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()} if device.type != "cpu" else model.state_dict()
                shared_model.load_state_dict(cpu_sd)
                pending_result, _ = _build_and_dispatch(batch_num + 1, async_mode=True)
            else:
                is_last = True  # will not enter another iteration

        # --- Phase 3: compute GAE advantages (fixed across PPO epochs) ---
        model.train()
        # Episodes are 7-tuples once deploy training is wired; 6-tuples on
        # the legacy path. Index 0 is the trajectory in both cases.
        all_trajs = [ep[0] for ep in trajectories]
        all_advantages, all_returns = compute_gae(
            all_trajs, gamma=1.0, gae_lambda=config.gae_lambda,
            unit_local_blend=config.unit_local_advantage_blend,
        )

        # Record game outcomes for metrics tracking. record_game routes each
        # game to the right deques by opp_type: "heuristic" → heuristic WR,
        # "selfplay" → self-play WR (checkpoints only), "selfplay_mirror" /
        # "mirror_b" → per-physical-side WR.
        opp_types = []
        _h_shoot_eff_sum = 0.0
        _h_charge_eff_sum = 0.0
        _h_shoot_n = 0
        _h_charge_n = 0
        _ml_h_shoot_eff_sum = 0.0
        _ml_h_charge_eff_sum = 0.0
        _ml_h_shoot_n = 0
        _ml_h_charge_n = 0
        deploy_records_all: list = []
        deploy_returns_all: list[float] = []
        for ep_tuple in trajectories:
            # Backwards-compat: episodes are 6-tuples in legacy collection
            # paths and 7-tuples once deploy training is wired (the 7th
            # element is the per-game DeploymentRecord list).
            if len(ep_tuple) == 7:
                traj, result, opp_type, army_type, phys_side, h_eff, _deploy_recs = ep_tuple
                if (config.train_deployment
                        and config.deploy_loss_coeff > 0.0
                        and _deploy_recs):
                    # Terminal-outcome return from this side's perspective.
                    # result is already main-perspective ("main"/"opp"/"draw")
                    # so the sign maps cleanly: main=+1, opp=-1, draw=0.
                    terminal = 1.0 if result == "main" else (-1.0 if result == "opp" else 0.0)
                    # Post-deploy value-head bonus: V(s_first_tactical) under
                    # the collection policy approximates the model's belief
                    # at the deploy→turn-1 boundary, smoothing the long-
                    # horizon credit assignment over ~10-15 deploy steps.
                    bonus_v = 0.0
                    if traj and config.deploy_post_value_bonus != 0.0:
                        bonus_v = float(getattr(traj[0], "old_value", 0.0))
                    per_record_return = terminal + config.deploy_post_value_bonus * bonus_v
                    for r in _deploy_recs:
                        deploy_records_all.append(r)
                        deploy_returns_all.append(per_record_return)
            else:
                traj, result, opp_type, army_type, phys_side, h_eff = ep_tuple
            metrics.record_game(result, opp_type, army_type,
                                physical_side=phys_side)
            opp_types.append(opp_type)
            if h_eff is not None:
                _h_shoot_eff_sum += h_eff['shoot']
                _h_charge_eff_sum += h_eff['charge']
                _h_shoot_n += h_eff['shoot_n']
                _h_charge_n += h_eff['charge_n']
                for s in traj:
                    if s.shooting_efficiency_reward != 0.0:
                        _ml_h_shoot_eff_sum += s.shooting_efficiency_reward
                        _ml_h_shoot_n += 1
                    if s.charge_efficiency_reward != 0.0:
                        _ml_h_charge_eff_sum += s.charge_efficiency_reward
                        _ml_h_charge_n += 1

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

        # Planning distill ramp (same schedule as the rollout rate ramp).
        _b0 = batch_num - start_batch
        if _b0 <= config.planning_warmup_batches:
            distill_ramp = 0.0
        elif config.planning_distill_ramp_batches > 0:
            distill_ramp = min(
                1.0,
                (_b0 - config.planning_warmup_batches) / config.planning_distill_ramp_batches,
            )
        else:
            distill_ramp = 1.0

        # MPO switch + KL trust-region β resolution.
        #
        # mpo_switch_batch (absolute batch number) flips the loss from
        # `planning_distill_mode` (legacy PPO + ce_chosen / soft_kl) to
        # "mpo_marginal" and activates the KL trust region. When the
        # switch is unset, behaviour is unchanged: configured mode +
        # β = config.kl_trust_region_beta (with optional schedule from
        # start_batch).
        #
        # When the switch is set:
        #   batch_num <= mpo_switch_batch → legacy mode, β = 0.
        #   batch_num >  mpo_switch_batch → "mpo_marginal", β starts at
        #     config.kl_trust_region_beta and (if mpo_kl_beta_end and
        #     mpo_kl_beta_ramp_batches are set) linearly anneals toward
        #     mpo_kl_beta_end over the next mpo_kl_beta_ramp_batches
        #     batches counted from the switch.
        if config.mpo_switch_batch is not None:
            mpo_active = batch_num > config.mpo_switch_batch
            effective_distill_mode = (
                "mpo_marginal" if mpo_active else config.planning_distill_mode
            )
            if not mpo_active:
                kl_beta_now = 0.0
            elif (config.kl_trust_region_beta > 0
                    and config.mpo_kl_beta_end is not None
                    and config.mpo_kl_beta_ramp_batches > 0):
                _post = batch_num - config.mpo_switch_batch - 1  # 0-indexed
                _kb_progress = min(1.0, max(0.0, _post / config.mpo_kl_beta_ramp_batches))
                kl_beta_now = (
                    config.kl_trust_region_beta
                    + _kb_progress * (config.mpo_kl_beta_end - config.kl_trust_region_beta)
                )
            else:
                kl_beta_now = config.kl_trust_region_beta
        else:
            mpo_active = False
            effective_distill_mode = config.planning_distill_mode
            if (config.kl_trust_region_beta > 0
                    and config.mpo_kl_beta_end is not None
                    and config.mpo_kl_beta_ramp_batches > 0):
                _kb_progress = min(1.0, max(0.0, _b0 / config.mpo_kl_beta_ramp_batches))
                kl_beta_now = (
                    config.kl_trust_region_beta
                    + _kb_progress * (config.mpo_kl_beta_end - config.kl_trust_region_beta)
                )
            else:
                kl_beta_now = config.kl_trust_region_beta

        # One-time banner when the switch flips this batch.
        if (verbose and config.mpo_switch_batch is not None
                and batch_num == config.mpo_switch_batch + 1):
            print(f"  [MPO] switching loss to 'mpo_marginal' at batch "
                  f"{batch_num} (was {config.planning_distill_mode!r}); "
                  f"β = {kl_beta_now}")

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

            # Pre-build all replay tensors ONCE (the key optimisation)
            prepared = prepare_replay_data(all_trajs, device=device)
            all_flat_steps = [s for traj in all_trajs for s in traj]

        # Snapshot model weights before PPO epochs so we can rollback on NaN
        pre_ppo_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Refresh π_old to the policy at the start of this batch's PPO
        # epochs. The KL trust region (Phase 2) is computed against this
        # snapshot, not against per-rollout behaviour. Same cadence as
        # pre_ppo_state so the snapshot is always consistent with the
        # policy that produced this batch's gradients.
        if model_old is not None:
            model_old.load_state_dict(pre_ppo_state, strict=True)

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

                    # Gather flat tensor slices for this minibatch
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

                    # Forward pass using pre-built tensors (no _force_tensor_device needed)
                    mb_flat_result = replay_from_prepared(
                        model, prepared, idx_t,
                        n_episodes=len(mb_game_idx),
                    )
                    # π_old forward — only when KL trust region is active.
                    # No grad through the snapshot; its params are already
                    # frozen but no_grad still saves activation memory.
                    mb_flat_result_old = None
                    if model_old is not None and kl_beta_now > 0:
                        with torch.no_grad():
                            mb_flat_result_old = replay_from_prepared(
                                model_old, prepared, idx_t,
                                n_episodes=len(mb_game_idx),
                            )
                    mb_flat_steps = [all_flat_steps[i] for i in mb_flat_idx]
                    with _force_tensor_device(device):
                        loss, loss_metrics = compute_loss_flat(
                            mb_flat_result, mb_old_lps, mb_advantages, mb_returns,
                            config.clip_epsilon, config.value_coeff, entropy_coeff,
                            aux_coeff=config.aux_coeff, aux_ratio=config.aux_ratio,
                            flat_steps=mb_flat_steps,
                            entropy_tuner=entropy_tuner,
                            planning_distill_max_weight=(
                                config.planning_distill_max_weight
                                if config.planning_rate > 0 else 0.0),
                            planning_distill_ramp=distill_ramp,
                            planning_distill_mode=effective_distill_mode,
                            mpo_eta=config.mpo_eta,
                            flat_result_old=mb_flat_result_old,
                            kl_trust_region_beta=kl_beta_now,
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
                    phe_mb = loss_metrics.get("per_head_entropy", {})
                    phn_mb = loss_metrics.get("per_head_n", {})
                    for hk, hv in phe_mb.items():
                        # Weight by per-head gated sample count so the aggregate is
                        # a true mean over active samples, not diluted by minibatches
                        # that happened to contain few samples where the head is active.
                        w = phn_mb.get(hk, n_mb)
                        epoch_metrics[f"_phe_{hk}"] = epoch_metrics.get(f"_phe_{hk}", 0.0) + hv * w
                        epoch_metrics[f"_phn_{hk}"] = epoch_metrics.get(f"_phn_{hk}", 0.0) + w
                    opp_val_mb = loss_metrics.get("per_opp_type_mean_values", {})
                    for ok, ov in opp_val_mb.items():
                        epoch_metrics[f"_opv_{ok}"] = epoch_metrics.get(f"_opv_{ok}", 0.0) + ov * n_mb
                    pds_mb = loss_metrics.get("planning_distill_sub", {})
                    for pk, pv in pds_mb.items():
                        epoch_metrics[f"_pds_{pk}"] = epoch_metrics.get(f"_pds_{pk}", 0.0) + pv * n_mb
                    pdp_mb = loss_metrics.get("planning_distill_peaks", {})
                    for pk, pv in pdp_mb.items():
                        epoch_metrics[f"_pdp_{pk}"] = epoch_metrics.get(f"_pdp_{pk}", 0.0) + pv * n_mb
                    klt_mb = loss_metrics.get("kl_trust_per_head", {})
                    for kk, kv in klt_mb.items():
                        epoch_metrics[f"_klt_{kk}"] = epoch_metrics.get(f"_klt_{kk}", 0.0) + kv * n_mb
                    epoch_count += n_mb

                if nan_detected:
                    break
                # Average metrics across minibatches
                if epoch_count > 0:
                    phe_agg = {}
                    opv_agg = {}
                    pds_agg = {}
                    pdp_agg = {}
                    klt_agg = {}
                    non_phe = {}
                    # First pass: pull out per-head weight sums so we can divide
                    # per-head entropies by their true gated-sample total.
                    phn_sums = {
                        k[5:]: v for k, v in epoch_metrics.items() if k.startswith("_phn_")
                    }
                    for k, v in epoch_metrics.items():
                        if k.startswith("_phe_"):
                            head = k[5:]
                            w = phn_sums.get(head, 0.0)
                            phe_agg[head] = (v / w) if w > 0 else 0.0
                        elif k.startswith("_phn_"):
                            continue
                        elif k.startswith("_opv_"):
                            opv_agg[k[5:]] = v / epoch_count
                        elif k.startswith("_pds_"):
                            pds_agg[k[5:]] = v / epoch_count
                        elif k.startswith("_pdp_"):
                            pdp_agg[k[5:]] = v / epoch_count
                        elif k.startswith("_klt_"):
                            klt_agg[k[5:]] = v / epoch_count
                        else:
                            non_phe[k] = v / epoch_count
                    loss_metrics = non_phe
                    loss_metrics["per_head_entropy"] = phe_agg
                    loss_metrics["per_opp_type_mean_values"] = opv_agg
                    loss_metrics["planning_distill_sub"] = pds_agg
                    loss_metrics["planning_distill_peaks"] = pdp_agg
                    loss_metrics["kl_trust_per_head"] = klt_agg

            elif is_tactical:
                # Full-batch tactical path (minibatch disabled or batch too small)
                all_idx_t = torch.arange(prepared.n_steps, dtype=torch.long, device=device)
                flat_result = replay_from_prepared(
                    model, prepared, all_idx_t,
                    n_episodes=len(all_trajs),
                )
                flat_result_old_full = None
                if model_old is not None and kl_beta_now > 0:
                    with torch.no_grad():
                        flat_result_old_full = replay_from_prepared(
                            model_old, prepared, all_idx_t,
                            n_episodes=len(all_trajs),
                        )
                with _force_tensor_device(device):
                    loss, loss_metrics = compute_loss_flat(
                        flat_result, flat_old_lps, flat_advantages_t, flat_returns_t,
                        config.clip_epsilon, config.value_coeff, entropy_coeff,
                        aux_coeff=config.aux_coeff, aux_ratio=config.aux_ratio,
                        flat_steps=all_flat_steps,
                        entropy_tuner=entropy_tuner,
                        planning_distill_max_weight=(
                            config.planning_distill_max_weight
                            if config.planning_rate > 0 else 0.0),
                        planning_distill_ramp=distill_ramp,
                        planning_distill_mode=effective_distill_mode,
                        mpo_eta=config.mpo_eta,
                        flat_result_old=flat_result_old_full,
                        kl_trust_region_beta=kl_beta_now,
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

            # Deployment-head PPO update — one optimizer step per epoch,
            # in addition to the tactical step above. Same model parameters,
            # so the gradients accumulate into the deploy heads + the trunk.
            # Kept as a separate step (rather than summed into the tactical
            # loss) because the deploy minibatch is small and unminibatched
            # — folding it into per-minibatch tactical loss would overweight
            # it n_minibatches-fold without making the math cleaner.
            if (config.train_deployment
                    and config.deploy_loss_coeff > 0.0
                    and deploy_records_all
                    and not nan_detected):
                from ml_training.deploy_loss import compute_deploy_loss
                deploy_returns_t = torch.tensor(
                    deploy_returns_all, dtype=torch.float32, device=device,
                )
                with _force_tensor_device(device):
                    dep_out = compute_deploy_loss(
                        model, deploy_records_all, deploy_returns_t,
                        clip_eps=config.clip_epsilon,
                        value_coef=config.value_coeff,
                        entropy_coef=entropy_coeff,
                        device=device,
                    )
                dep_loss = config.deploy_loss_coeff * dep_out["total_loss"]
                optimizer.zero_grad()
                if torch.isnan(dep_loss) or torch.isinf(dep_loss):
                    print(f"  WARNING: NaN/Inf deploy loss at batch {batch_num}, rolling back weights")
                    model.load_state_dict(pre_ppo_state)
                    nan_detected = True
                else:
                    dep_loss.backward()
                    grad_nan = any(
                        p.grad is not None and torch.isnan(p.grad).any()
                        for p in model.parameters()
                    )
                    if grad_nan:
                        print(f"  WARNING: NaN deploy gradients at batch {batch_num}, rolling back weights")
                        model.load_state_dict(pre_ppo_state)
                        nan_detected = True
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        # Fold deploy metrics into loss_metrics so the logger
                        # picks them up alongside the tactical metrics.
                        loss_metrics["deploy_policy_loss"] = dep_out["policy_loss"]
                        loss_metrics["deploy_value_loss"] = dep_out["value_loss"]
                        loss_metrics["deploy_entropy"] = dep_out["entropy"]
                        loss_metrics["deploy_approx_kl"] = dep_out["approx_kl"]
                        loss_metrics["deploy_clip_frac"] = dep_out["clip_frac"]
                        loss_metrics["deploy_n_records"] = len(deploy_records_all)

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
            f"{loss_metrics.get('planning_distill_ramp', 1.0):.4f}",
            f"{loss_metrics.get('planning_argmax_rate', 0.0):.4f}",
            *[f"{loss_metrics.get('planning_distill_sub', {}).get(k, 0.0):.6f}"
              for k in ("unit", "move", "charge", "shoot", "dest")],
            *[f"{loss_metrics.get('planning_distill_peaks', {}).get(k, 0.0):.6f}"
              for k in ("tgt_unit", "pol_unit",
                        "tgt_move", "pol_move",
                        "tgt_charge", "pol_charge",
                        "tgt_shoot", "pol_shoot",
                        "tgt_dest", "pol_dest")],
            *[f"{loss_metrics.get('planning_distill_peaks', {}).get(k, 0.0):.4f}"
              for k in ("agree_unit", "agree_move",
                        "agree_charge", "agree_shoot",
                        "agree_dest")],
            f"{metrics.a_side_win_rate:.3f}",
            f"{metrics.b_side_win_rate:.3f}",
            f"{_opp_val_dict.get('mean_value_side_a', '')}",
            f"{_opp_val_dict.get('mean_value_side_b', '')}",
            f"{loss_metrics.get('mean_shoot_eff_reward', 0.0):.6f}",
            f"{loss_metrics.get('mean_charge_eff_reward', 0.0):.6f}",
            f"{_ml_h_shoot_eff_sum / max(_ml_h_shoot_n, 1):.6f}",
            f"{_ml_h_charge_eff_sum / max(_ml_h_charge_n, 1):.6f}",
            f"{_h_shoot_eff_sum / max(_h_shoot_n, 1):.6f}",
            f"{_h_charge_eff_sum / max(_h_charge_n, 1):.6f}",
            f"{loss_metrics.get('value_loss', 0.0):.6f}",
            # Per-phase V head: aggregate loss, then per-phase loss + mean output.
            # Empty strings when flag is off so the CSV round-trips cleanly.
            *_format_per_phase_value(loss_metrics),
            # MPO trust region. β=0 and KL=0 when the trust region is
            # inactive (pre-switch or in pure-PPO/legacy modes), so these
            # cells are populated for every row even on legacy runs.
            f"{loss_metrics.get('kl_trust_region_beta', 0.0):.4f}",
            f"{loss_metrics.get('kl_trust_region_loss', 0.0):.6f}",
            *[f"{loss_metrics.get('kl_trust_per_head', {}).get(k, 0.0):.6f}"
              for k in ("unit", "move", "dest", "charge", "shoot")],
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
    torch.save({
        "model_state_dict": model.state_dict(),
        "batch_num": batch_num,
        "n_iters": model.n_iters,
    }, final_path)

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
