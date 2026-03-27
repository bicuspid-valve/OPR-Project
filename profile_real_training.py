"""Profile the REAL training loop with fine-grained phase timing.

Runs 5 batches with production parameters and reports where the time goes,
so we can compare against the profiling script results.

Usage:
    .venv/bin/python profile_real_training.py
"""
from __future__ import annotations

import os
import random
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.multiprocessing as _mp
import torch.nn as nn

from ml_training import (
    TrainingConfig, CheckpointPool, TrainingMetrics,
    get_heuristic_fraction, _generate_army_pair,
    _collect_episodes_shared_worker, _init_shared_worker,
    replay_tactical_log_probs_flat, compute_loss_flat, compute_gae,
    _WORKER_COUNT, _MAX_SHARED_OPPONENTS,
    _load_hof_armies, _load_hof_ml_armies,
    _make_model, _resolve_device, _force_tensor_device,
    EntropyTargetTuner,
)
from ml_model_tactical import TacticalModel

NUM_BATCHES = 5
BATCH_SIZE = 512
WORKER_COUNT = 6
MINIBATCH_GAMES = 64
DEVICE = "cuda"


def main():
    device = _resolve_device(DEVICE)
    print("=" * 70)
    print(f"Real Training Loop Profiler — {NUM_BATCHES} batches")
    print(f"  batch_size={BATCH_SIZE}, workers={WORKER_COUNT}, "
          f"minibatch={MINIBATCH_GAMES}, device={device}")
    print("=" * 70)

    config = TrainingConfig(
        batch_size=BATCH_SIZE,
        worker_count=WORKER_COUNT,
        ppo_minibatch_games=MINIBATCH_GAMES,
        device=DEVICE,
    )

    # ── Load HoF armies (real training does this) ──
    t0 = time.perf_counter()
    hof_armies = _load_hof_armies()
    hof_ml_armies = _load_hof_ml_armies()
    t_hof = time.perf_counter() - t0
    print(f"\nHoF load: {t_hof:.3f}s ({len(hof_armies)} HoF, {len(hof_ml_armies)} HoF-ML)")

    # ── Model setup ──
    t0 = time.perf_counter()
    model = _make_model("tactical")
    # Load checkpoint like real training
    from pathlib import Path
    final_path = Path("ml_checkpoints") / "final_model.pt"
    start_batch = 0
    if final_path.exists():
        ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            start_batch = ckpt.get("batch_num", 0)
        print(f"  Loaded checkpoint (batch {start_batch})")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    entropy_tuner = EntropyTargetTuner(config).to(device)
    alpha_optimizer = torch.optim.Adam(entropy_tuner.parameters(), lr=config.entropy_alpha_lr)
    t_model = time.perf_counter() - t0
    print(f"Model setup: {t_model:.3f}s")

    # ── Checkpoint pool ──
    t0 = time.perf_counter()
    checkpoint_pool = CheckpointPool(
        max_size=20, save_dir="ml_checkpoints", model_type="tactical",
        seed_existing=5)
    t_ckpt = time.perf_counter() - t0
    print(f"Checkpoint pool: {t_ckpt:.3f}s ({len(checkpoint_pool.entries)} entries)")

    metrics = TrainingMetrics()

    # ── Shared-memory pool ──
    t0 = time.perf_counter()
    shared_model = _make_model("tactical")
    shared_model.share_memory()
    shared_model.eval()
    shared_opponents = []
    for _ in range(_MAX_SHARED_OPPONENTS):
        m = _make_model("tactical")
        m.share_memory()
        m.eval()
        shared_opponents.append(m)

    ctx = _mp.get_context("spawn")
    pool = ctx.Pool(
        processes=WORKER_COUNT,
        initializer=_init_shared_worker,
        initargs=(shared_model, shared_opponents, "tactical", True),
    )
    t_pool = time.perf_counter() - t0
    print(f"Pool setup: {t_pool:.3f}s")

    # ── Run batches with phase timing ──
    print(f"\n{'─'*70}")
    print(f"{'Phase':<25} {'Time(s)':>8}  {'%':>5}")
    print(f"{'─'*70}")

    phase_totals = {}

    for batch_num in range(1, NUM_BATCHES + 1):
        batch_start = time.perf_counter()
        phases = {}
        heuristic_fraction = get_heuristic_fraction(metrics.heuristic_win_rate)

        # -- Weight sync --
        t0 = time.perf_counter()
        cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        shared_model.load_state_dict(cpu_sd)
        phases["weight_sync"] = time.perf_counter() - t0

        # -- Game spec generation (army gen + opponent scheduling) --
        t0 = time.perf_counter()
        game_specs = []
        opponent_state_dicts = []
        _opp_path_cache = {}

        for _ in range(BATCH_SIZE):
            if random.random() < heuristic_fraction:
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

            res_a, res_b, states_a, states_b, army_type = _generate_army_pair(
                opp_type=opp_type, hof_armies=hof_armies,
                hof_ml_armies=hof_ml_armies)
            states_a_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a]
            states_b_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b]
            game_specs.append((res_a, res_b, states_a_data, states_b_data, opp_type, opp_sd_idx, army_type))
        phases["game_specs"] = time.perf_counter() - t0

        # -- Opponent weight loading --
        t0 = time.perf_counter()
        opp_slot_map = {}
        for i, sd in enumerate(opponent_state_dicts):
            if i < _MAX_SHARED_OPPONENTS:
                shared_opponents[i].load_state_dict(sd, strict=False)
                opp_slot_map[i] = i
        phases["opp_weight_load"] = time.perf_counter() - t0

        # -- Shaping scale --
        shaping_scale = 0.0  # matches resumed training

        # -- Episode collection (workers) --
        t0 = time.perf_counter()
        n_chunks = WORKER_COUNT
        chunk_size = max(1, len(game_specs) // n_chunks)
        chunks = []
        for i in range(0, len(game_specs), chunk_size):
            chunk = game_specs[i : i + chunk_size]
            chunks.append((opp_slot_map, chunk, shaping_scale))
        chunk_results = list(pool.map(_collect_episodes_shared_worker, chunks))
        trajectories = [ep for chunk in chunk_results for ep in chunk]
        phases["episode_sim"] = time.perf_counter() - t0

        # -- GAE --
        t0 = time.perf_counter()
        all_trajs = [t[0] for t in trajectories]
        all_advantages, all_returns = compute_gae(all_trajs, gamma=1.0, gae_lambda=0.95)
        for traj_tuple in trajectories:
            if traj_tuple[2] != "mirror_b":
                metrics.record_game(traj_tuple[1], traj_tuple[2])
        phases["gae"] = time.perf_counter() - t0

        # -- Flatten tensors --
        t0 = time.perf_counter()
        flat_old_lps = torch.tensor(
            [s.old_log_prob for traj in all_trajs for s in traj],
            dtype=torch.float32, device=device)
        flat_advantages_t = torch.tensor(
            [a for adv in all_advantages for a in adv],
            dtype=torch.float32, device=device)
        flat_returns_t = torch.tensor(
            [r for ret in all_returns for r in ret],
            dtype=torch.float32, device=device)
        game_step_counts = [len(traj) for traj in all_trajs]
        game_step_offsets = [0] * len(all_trajs)
        for gi in range(1, len(all_trajs)):
            game_step_offsets[gi] = game_step_offsets[gi - 1] + game_step_counts[gi - 1]
        phases["flatten"] = time.perf_counter() - t0

        # -- PPO update --
        t0 = time.perf_counter()
        pre_ppo_state = {k: v.clone() for k, v in model.state_dict().items()}
        for _ppo_epoch in range(config.ppo_epochs):
            game_indices = list(range(len(all_trajs)))
            random.shuffle(game_indices)
            for mb_start in range(0, len(game_indices), MINIBATCH_GAMES):
                mb_game_idx = game_indices[mb_start:mb_start + MINIBATCH_GAMES]
                mb_trajs = [all_trajs[i] for i in mb_game_idx]
                mb_flat_idx = []
                for gi in mb_game_idx:
                    off = game_step_offsets[gi]
                    mb_flat_idx.extend(range(off, off + game_step_counts[gi]))
                if not mb_flat_idx:
                    continue
                idx_t = torch.tensor(mb_flat_idx, dtype=torch.long, device=device)
                mb_old_lps = flat_old_lps[idx_t]
                mb_advantages = flat_advantages_t[idx_t]
                mb_returns = flat_returns_t[idx_t]

                with _force_tensor_device(device):
                    mb_flat_result = replay_tactical_log_probs_flat(model, mb_trajs)
                    mb_flat_steps = [s for traj in mb_trajs for s in traj]
                    loss, loss_metrics = compute_loss_flat(
                        mb_flat_result, mb_old_lps, mb_advantages, mb_returns,
                        config.clip_epsilon, config.value_coeff, 0.01,
                        aux_coeff=config.aux_coeff, aux_ratio=config.aux_ratio,
                        flat_steps=mb_flat_steps,
                        entropy_tuner=entropy_tuner,
                    )

                optimizer.zero_grad()
                alpha_optimizer.zero_grad()
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    alpha_loss_tensor = loss_metrics.pop("_alpha_loss_tensor", None)
                    if alpha_loss_tensor is not None:
                        alpha_loss_tensor.backward()
                        alpha_optimizer.step()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        phases["ppo_update"] = time.perf_counter() - t0

        total = time.perf_counter() - batch_start
        phases["TOTAL"] = total

        # Print this batch
        print(f"\n  Batch {batch_num}:")
        for phase, t in phases.items():
            pct = 100 * t / total if phase != "TOTAL" else 100.0
            phase_totals[phase] = phase_totals.get(phase, 0.0) + t
            label = f"  {phase}" if phase != "TOTAL" else f"  {'TOTAL':─<25}"
            print(f"    {phase:<22} {t:>8.3f}s  {pct:>5.1f}%")

    pool.close()
    pool.join()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"AVERAGE OVER {NUM_BATCHES} BATCHES")
    print(f"{'='*70}")
    avg_total = phase_totals["TOTAL"] / NUM_BATCHES
    for phase, total_t in phase_totals.items():
        avg = total_t / NUM_BATCHES
        pct = 100 * avg / avg_total if phase != "TOTAL" else 100.0
        print(f"  {phase:<22} {avg:>8.3f}s  {pct:>5.1f}%")
    print(f"\n  Games/sec: {BATCH_SIZE / avg_total:.1f}")

    # ── Comparison with profiling script ──
    print(f"\n{'─'*70}")
    print("Compare with profiling script (cuda, MB=64, workers=6):")
    print("  Profiling: 11.0s/batch, sim=5.8s, replay=3.6s, loss=0.5s")
    print(f"  Real:      {avg_total:.1f}s/batch, "
          f"sim={phase_totals['episode_sim']/NUM_BATCHES:.1f}s, "
          f"ppo={phase_totals['ppo_update']/NUM_BATCHES:.1f}s")
    gap = avg_total - 11.0
    print(f"  Gap: {gap:+.1f}s — see breakdown above for where it goes")


if __name__ == "__main__":
    main()
