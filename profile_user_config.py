"""Profile ml_train() with the user's exact arguments.

Replicates the real loop body inline so we can time each phase. Runs N_BATCHES
real batches, skipping the cgroup re-exec (it's a harness concern, not a perf
one). Matches user's call:

    ml_train(num_batches=300000, batch_size=256, time_limit=360,
             model_type="tactical", use_c_ext=True, restart_training=False,
             memory_max="14G", memory_swap_max="40G", worker_count=6,
             planning_rate=0.2, minibatch_size=64, blend_ratio=0.25)

planning_warmup_batches / planning_distill_ramp_batches default to 0 under
restart=False, so planning is active from batch 1.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.multiprocessing as _mp

from ml_training import (
    TrainingConfig, CheckpointPool, TrainingMetrics,
    get_heuristic_fraction, _generate_army_pair,
    _collect_episodes_shared_worker, _init_shared_worker,
    compute_loss_flat, compute_gae,
    prepare_replay_data, replay_from_prepared,
    _MAX_SHARED_OPPONENTS,
    _load_hof_armies, _load_hof_ml_armies,
    _make_model, _resolve_device, _force_tensor_device,
    EntropyTargetTuner, load_model_state_dict,
)

# ── Exact user config ──
N_BATCHES        = 3            # profile this many real batches
BATCH_SIZE       = 256
WORKER_COUNT     = 6
MINIBATCH_GAMES  = 64
BLEND_RATIO      = 0.25
PLANNING_RATE    = float(os.environ.get("PROFILE_PLANNING_RATE", "0.2"))
MODEL_TYPE       = "tactical"
USE_C_EXT        = True
DEVICE           = "auto"
PHASE_REENCODE   = True


def _phase_timings_print(label, phases, total):
    print(f"\n  {label}")
    for k, v in phases.items():
        pct = 100 * v / total if total > 0 else 0
        print(f"    {k:<28} {v:>8.3f}s  {pct:>5.1f}%")


def main():
    import fast_core
    fast_core.USE_C_EXT = USE_C_EXT and fast_core.is_available()
    from ml_integration_tactical import set_phase_reencode_enabled
    set_phase_reencode_enabled(PHASE_REENCODE)

    device = _resolve_device(DEVICE)

    print("=" * 74)
    print(f"Profiling ml_train with user's exact config — {N_BATCHES} batches")
    print(f"  batch_size={BATCH_SIZE}  workers={WORKER_COUNT}  "
          f"minibatch_games={MINIBATCH_GAMES}")
    print(f"  planning_rate={PLANNING_RATE}  blend_ratio={BLEND_RATIO}  "
          f"c_ext={fast_core.USE_C_EXT}  device={device}")
    print("=" * 74)

    config = TrainingConfig(
        num_batches=300000,
        batch_size=BATCH_SIZE,
        time_limit=360,
        model_type=MODEL_TYPE,
        use_c_ext=USE_C_EXT,
        worker_count=WORKER_COUNT,
        device=DEVICE,
        planning_rate=PLANNING_RATE,
        planning_warmup_batches=0,
        planning_distill_ramp_batches=0,
        ppo_minibatch_games=MINIBATCH_GAMES,
        unit_local_advantage_blend=BLEND_RATIO,
        phase_reencode_enabled=PHASE_REENCODE,
    )

    # ── Load HoF armies ──
    t0 = time.perf_counter()
    hof_armies = _load_hof_armies()
    hof_ml_armies = _load_hof_ml_armies()
    t_hof = time.perf_counter() - t0
    print(f"\nHoF load: {t_hof:.3f}s  "
          f"({len(hof_armies)} HoF, {len(hof_ml_armies)} HoF-ML)")

    # ── Model setup (resume like the real path does) ──
    t0 = time.perf_counter()
    model = _make_model(MODEL_TYPE)
    final_path = Path("ml_checkpoints") / "final_model.pt"
    start_batch = 0
    if final_path.exists():
        ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            start_batch = ckpt.get("batch_num", 0)
            if "n_iters" in ckpt:
                model.n_iters = ckpt["n_iters"]
        print(f"  Resumed from {final_path} (batch {start_batch})")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    entropy_tuner = EntropyTargetTuner(config).to(device)
    tuner_path = Path("ml_checkpoints") / "entropy_tuner.pt"
    if tuner_path.exists():
        entropy_tuner.load_state_dict(
            torch.load(tuner_path, map_location="cpu", weights_only=True))
    alpha_optimizer = torch.optim.Adam(
        entropy_tuner.parameters(), lr=config.entropy_alpha_lr)
    t_model = time.perf_counter() - t0
    print(f"Model/optim setup: {t_model:.3f}s")

    # ── Checkpoint pool ──
    t0 = time.perf_counter()
    checkpoint_pool = CheckpointPool(
        max_size=config.max_checkpoints,
        save_dir="ml_checkpoints",
        model_type=MODEL_TYPE,
        seed_existing=config.max_checkpoints,
    )
    t_ckpt = time.perf_counter() - t0
    print(f"Checkpoint pool: {t_ckpt:.3f}s ({len(checkpoint_pool.entries)} entries)")
    metrics = TrainingMetrics()

    # ── Shared-memory pool ──
    t0 = time.perf_counter()
    shared_model = _make_model(MODEL_TYPE)
    shared_model.share_memory()
    shared_model.eval()
    shared_opponents = []
    for _ in range(_MAX_SHARED_OPPONENTS):
        m = _make_model(MODEL_TYPE)
        m.share_memory()
        m.eval()
        shared_opponents.append(m)
    ctx = _mp.get_context("spawn")
    pool = ctx.Pool(
        processes=WORKER_COUNT,
        initializer=_init_shared_worker,
        initargs=(shared_model, shared_opponents, MODEL_TYPE,
                  USE_C_EXT, PHASE_REENCODE),
    )
    t_pool = time.perf_counter() - t0
    print(f"Worker pool setup (spawn×{WORKER_COUNT}): {t_pool:.3f}s")

    # ── Planning config (matches loop.py build_and_dispatch) ──
    def _planning_config_for(batch_num_for_specs):
        # b0 starts at 1 → ramp_frac = 1.0 (warmup=0, ramp=0)
        b0 = batch_num_for_specs - start_batch
        if b0 <= 0:
            return None
        return {
            "planning_rate": PLANNING_RATE,
            "planning_params": {
                "K_UNITS": config.training_planning_K,
                "C_SAMPLES_PER_UNIT": config.training_planning_C,
                "M_ROLLOUTS": config.training_planning_M,
                "N_LOOKAHEAD": config.training_planning_N,
                "SEQUENTIAL_HALVING": config.training_planning_sequential_halving,
                "SH_SCHEDULE": tuple(config.training_planning_sh_schedule),
            },
        }

    def _build_and_dispatch(batch_num_for_specs, async_mode=False):
        hf = get_heuristic_fraction(metrics.heuristic_win_rate)
        game_specs = []
        opponent_state_dicts = []
        _opp_path_cache = {}
        for _ in range(BATCH_SIZE):
            if random.random() < hf:
                opp_type, opp_sd_idx = "heuristic", -1
            elif random.random() < 0.5:
                opp_type, opp_sd_idx = "selfplay_mirror", -1
            else:
                opp_path = checkpoint_pool.sample_opponent_path()
                if opp_path is not None:
                    opp_type = "selfplay"
                    key = str(opp_path)
                    if key not in _opp_path_cache:
                        _opp_path_cache[key] = len(opponent_state_dicts)
                        opponent_state_dicts.append(
                            checkpoint_pool.load_state_dict(opp_path))
                    opp_sd_idx = _opp_path_cache[key]
                else:
                    opp_type, opp_sd_idx = "selfplay_mirror", -1
            res_a, res_b, states_a, states_b, army_type = _generate_army_pair(
                opp_type=opp_type, hof_armies=hof_armies,
                hof_ml_armies=hof_ml_armies)
            sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective)
                       for u in states_a]
            sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective)
                       for u in states_b]
            game_specs.append((res_a, res_b, sa_data, sb_data,
                               opp_type, opp_sd_idx, army_type))
        opp_slot_map = {}
        for i, sd in enumerate(opponent_state_dicts):
            if i < _MAX_SHARED_OPPONENTS:
                shared_opponents[i].load_state_dict(sd, strict=False)
                opp_slot_map[i] = i

        ss = 0.0  # shaping off (start_batch > 0, restart=False)
        pcw = _planning_config_for(batch_num_for_specs)
        n_chunks = WORKER_COUNT
        chunk_sz = max(1, len(game_specs) // n_chunks)
        chunks = []
        for ci in range(0, len(game_specs), chunk_sz):
            chunks.append((opp_slot_map, game_specs[ci:ci + chunk_sz], ss, pcw))
        if async_mode:
            return pool.map_async(_collect_episodes_shared_worker, chunks)
        return list(pool.map(_collect_episodes_shared_worker, chunks))

    # ── Warmup: dispatch first batch ──
    t0 = time.perf_counter()
    cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    shared_model.load_state_dict(cpu_sd)
    pending_result = _build_and_dispatch(start_batch + 1, async_mode=True)
    t_first_dispatch = time.perf_counter() - t0
    print(f"First dispatch (spec build + weight sync): {t_first_dispatch:.3f}s")

    phase_totals = {}
    ppo_mb_totals = {"replay": 0.0, "loss": 0.0, "backward": 0.0,
                      "optim": 0.0, "index": 0.0, "other": 0.0}
    n_mb_total = 0
    n_planned_total = 0

    print(f"\n{'─'*74}")
    for batch_num in range(start_batch + 1, start_batch + 1 + N_BATCHES):
        batch_start = time.perf_counter()
        phases = {}

        # ── Wait for current batch's episodes ──
        t0 = time.perf_counter()
        chunk_results = pending_result.get()
        trajectories = [ep for chunk in chunk_results for ep in chunk]
        phases["episode_wait"] = time.perf_counter() - t0

        # Count planning activations in this batch
        plan_count = 0
        for traj_tuple in trajectories:
            traj = traj_tuple[0]
            for s in traj:
                if getattr(s, "was_planned", False) or \
                   getattr(s, "planning_improvement", None) is not None:
                    plan_count += 1
        n_planned_total += plan_count

        # ── Dispatch next batch (overlap with PPO) ──
        t0 = time.perf_counter()
        is_last = (batch_num == start_batch + N_BATCHES)
        if not is_last:
            cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            shared_model.load_state_dict(cpu_sd)
            pending_result = _build_and_dispatch(batch_num + 1, async_mode=True)
        phases["dispatch_next"] = time.perf_counter() - t0

        # ── GAE ──
        t0 = time.perf_counter()
        all_trajs = [tt[0] for tt in trajectories]
        all_advantages, all_returns = compute_gae(
            all_trajs, gamma=1.0, gae_lambda=config.gae_lambda,
            unit_local_blend=config.unit_local_advantage_blend,
        )
        for tt in trajectories:
            traj, result, opp_type, army_type, phys_side, _ = tt
            metrics.record_game(result, opp_type, army_type,
                                physical_side=phys_side)
        phases["gae"] = time.perf_counter() - t0

        # ── Flatten + prepare replay tensors ──
        t0 = time.perf_counter()
        flat_old_lps = torch.tensor(
            [s.old_log_prob for tr in all_trajs for s in tr],
            dtype=torch.float32, device=device)
        flat_advantages_t = torch.tensor(
            [a for adv in all_advantages for a in adv],
            dtype=torch.float32, device=device)
        flat_returns_t = torch.tensor(
            [r for ret in all_returns for r in ret],
            dtype=torch.float32, device=device)
        game_step_counts = [len(tr) for tr in all_trajs]
        game_step_offsets = [0] * len(all_trajs)
        for gi in range(1, len(all_trajs)):
            game_step_offsets[gi] = (game_step_offsets[gi - 1]
                                      + game_step_counts[gi - 1])
        prepared = prepare_replay_data(all_trajs, device=device)
        all_flat_steps = [s for tr in all_trajs for s in tr]
        phases["prepare"] = time.perf_counter() - t0

        total_steps_batch = sum(game_step_counts)

        # ── PPO update ──
        t0_ppo = time.perf_counter()
        t_replay = t_loss = t_back = t_opt = t_idx = 0.0
        n_mb = 0
        pre_ppo_state = {k: v.clone() for k, v in model.state_dict().items()}
        for _epoch in range(config.ppo_epochs):
            if MINIBATCH_GAMES > 0 and len(all_trajs) > MINIBATCH_GAMES:
                game_indices = list(range(len(all_trajs)))
                random.shuffle(game_indices)
                for mb_start in range(0, len(game_indices), MINIBATCH_GAMES):
                    mb_game_idx = game_indices[mb_start:mb_start + MINIBATCH_GAMES]

                    # index gather
                    _t = time.perf_counter()
                    mb_flat_idx = []
                    for gi in mb_game_idx:
                        off = game_step_offsets[gi]
                        mb_flat_idx.extend(range(off, off + game_step_counts[gi]))
                    if not mb_flat_idx:
                        continue
                    idx_t = torch.tensor(mb_flat_idx, dtype=torch.long, device=device)
                    mb_old_lps = flat_old_lps[idx_t]
                    mb_adv = flat_advantages_t[idx_t]
                    mb_ret = flat_returns_t[idx_t]
                    t_idx += time.perf_counter() - _t

                    # replay forward
                    _t = time.perf_counter()
                    mb_flat_result = replay_from_prepared(
                        model, prepared, idx_t, n_episodes=len(mb_game_idx))
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_replay += time.perf_counter() - _t

                    # loss
                    _t = time.perf_counter()
                    mb_flat_steps = [all_flat_steps[i] for i in mb_flat_idx]
                    with _force_tensor_device(device):
                        loss, loss_metrics = compute_loss_flat(
                            mb_flat_result, mb_old_lps, mb_adv, mb_ret,
                            config.clip_epsilon, config.value_coeff, 0.01,
                            aux_coeff=config.aux_coeff, aux_ratio=config.aux_ratio,
                            flat_steps=mb_flat_steps,
                            entropy_tuner=entropy_tuner,
                            planning_distill_max_weight=(
                                config.planning_distill_max_weight
                                if PLANNING_RATE > 0 else 0.0),
                            planning_distill_ramp=1.0,
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_loss += time.perf_counter() - _t

                    # backward
                    _t = time.perf_counter()
                    optimizer.zero_grad()
                    alpha_optimizer.zero_grad()
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        loss.backward()
                        alpha_loss_tensor = loss_metrics.pop("_alpha_loss_tensor", None)
                        if alpha_loss_tensor is not None:
                            alpha_loss_tensor.backward()
                            alpha_optimizer.step()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=1.0)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_back += time.perf_counter() - _t

                    # optimizer step
                    _t = time.perf_counter()
                    optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_opt += time.perf_counter() - _t

                    n_mb += 1
        phases["ppo_total"] = time.perf_counter() - t0_ppo
        phases["  ppo_replay_fwd"] = t_replay
        phases["  ppo_loss"] = t_loss
        phases["  ppo_backward"] = t_back
        phases["  ppo_optim_step"] = t_opt
        phases["  ppo_index_gather"] = t_idx
        phases["  ppo_other"] = (phases["ppo_total"] - t_replay - t_loss
                                  - t_back - t_opt - t_idx)

        total = time.perf_counter() - batch_start
        phases["TOTAL"] = total

        for k, v in phases.items():
            phase_totals[k] = phase_totals.get(k, 0.0) + v
        ppo_mb_totals["replay"] += t_replay
        ppo_mb_totals["loss"] += t_loss
        ppo_mb_totals["backward"] += t_back
        ppo_mb_totals["optim"] += t_opt
        ppo_mb_totals["index"] += t_idx
        ppo_mb_totals["other"] += phases["  ppo_other"]
        n_mb_total += n_mb

        _phase_timings_print(
            f"Batch {batch_num}: total {total:.2f}s, "
            f"steps={total_steps_batch}, minibatches={n_mb}, planned_acts={plan_count}",
            phases, total,
        )

    pool.close()
    pool.join()

    # ── Summary ──
    print(f"\n{'='*74}")
    print(f"AVERAGE OVER {N_BATCHES} BATCHES")
    print(f"{'='*74}")
    avg_total = phase_totals["TOTAL"] / N_BATCHES
    for k, v in phase_totals.items():
        avg = v / N_BATCHES
        pct = 100 * avg / avg_total if k != "TOTAL" else 100.0
        print(f"  {k:<28} {avg:>8.3f}s  {pct:>5.1f}%")
    print(f"\n  Games/sec (wall): {BATCH_SIZE / avg_total:.2f}")
    print(f"  Minibatches/batch avg: {n_mb_total / N_BATCHES:.1f}")
    print(f"  Planned activations/batch avg: {n_planned_total / N_BATCHES:.0f}")

    # ── Per-minibatch PPO breakdown ──
    print(f"\n{'─'*74}")
    print(f"PPO PER-MINIBATCH (avg over {n_mb_total} minibatches):")
    for k, total_t in ppo_mb_totals.items():
        per = (total_t / n_mb_total * 1000) if n_mb_total else 0
        print(f"  ppo_{k:<14} {per:>8.2f} ms/mb")

    # ── Extrapolation ──
    print(f"\n{'─'*74}")
    print(f"EXTRAPOLATION (360 min time limit):")
    tl_s = 360 * 60
    batches_in_limit = tl_s / avg_total
    games_in_limit = batches_in_limit * BATCH_SIZE
    print(f"  at {avg_total:.1f}s/batch → ~{batches_in_limit:.0f} batches, "
          f"{games_in_limit:.0f} games in 6h")
    print(f"  (num_batches=300000 would require "
          f"{avg_total * 300000 / 3600:.0f} h at this rate)")


if __name__ == "__main__":
    main()
