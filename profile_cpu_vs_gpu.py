"""Profile training speed: CPU vs GPU × minibatch size × worker count.

Auto-finishes after 40 minutes total.  Each configuration gets an equal
time budget and runs as many training batches as it can within that budget
(using the same time-limit technique as the main training loop).

Usage:
    cd TP_OPR_Project
    .venv/bin/python profile_cpu_vs_gpu.py
"""
from __future__ import annotations

import csv
import gc
import math
import os
import random
import statistics
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.multiprocessing as _mp
import torch.nn as nn
import torch.nn.functional as F

from ml_training import (
    TrainingConfig, CheckpointPool, TrainingMetrics,
    get_heuristic_fraction, _generate_army_pair,
    _collect_episodes_shared_worker, _init_shared_worker,
    replay_tactical_log_probs_flat, compute_loss_flat, compute_gae,
    _WORKER_COUNT, _MAX_SHARED_OPPONENTS, EntropyTargetTuner,
    TacticalActivationRecord, FlatReplayResult,
)
from ml_model_tactical import TacticalModel

# ── Configuration ────────────────────────────────────────────────────────
TOTAL_TIME_LIMIT_MIN = 40          # total wall-clock budget in minutes
BATCH_SIZE = 512                   # games per training batch (from line 731)
WARMUP_BATCHES = 1                 # discard first batch from timing

# Test grid
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

MINIBATCH_SIZES = [16, 32, 64]     # ppo_minibatch_games
WORKER_COUNTS = [2, 4, 6]          # multiprocessing pool workers

# ── Device-aware tensor creation ─────────────────────────────────────────

@contextmanager
def force_tensor_device(device):
    """Monkey-patch torch tensor-creation functions to target *device*.

    This lets us call the existing replay_tactical_log_probs_flat (which
    hard-codes CPU tensors) with a model that lives on GPU — every
    intermediate tensor will be created on the same device.
    """
    if str(device) == "cpu":
        yield
        return

    _orig_from_numpy = torch.from_numpy
    _orig_tensor = torch.tensor
    _orig_full = torch.full
    _orig_zeros = torch.zeros
    _orig_ones = torch.ones

    def _from_numpy(a):
        return _orig_from_numpy(a).to(device)

    def _tensor(*args, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = device
        return _orig_tensor(*args, **kwargs)

    def _full(size, fill_value, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = device
        return _orig_full(size, fill_value, **kwargs)

    def _zeros(*size, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = device
        return _orig_zeros(*size, **kwargs)

    def _ones(*size, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = device
        return _orig_ones(*size, **kwargs)

    torch.from_numpy = _from_numpy
    torch.tensor = _tensor
    torch.full = _full
    torch.zeros = _zeros
    torch.ones = _ones
    try:
        yield
    finally:
        torch.from_numpy = _orig_from_numpy
        torch.tensor = _orig_tensor
        torch.full = _orig_full
        torch.zeros = _orig_zeros
        torch.ones = _orig_ones


# ── Single-configuration profiler ────────────────────────────────────────

def profile_config(
    device_name: str,
    minibatch_games: int,
    worker_count: int,
    time_budget_sec: float,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Run training batches for one configuration until *time_budget_sec*
    is exhausted.  Returns a dict of timing statistics."""

    device = torch.device(device_name)
    print(f"\n{'─'*70}")
    print(f"Config: device={device_name}  minibatch={minibatch_games}  "
          f"workers={worker_count}  budget={time_budget_sec:.0f}s")
    print(f"{'─'*70}")

    # ── Build model + shared-memory models ──
    model = TacticalModel()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    shared_model = TacticalModel()
    shared_model.share_memory()
    shared_model.eval()

    shared_opponents: list[nn.Module] = []
    for _ in range(_MAX_SHARED_OPPONENTS):
        m = TacticalModel()
        m.share_memory()
        m.eval()
        shared_opponents.append(m)

    # Use a temporary checkpoint dir so we don't pollute real checkpoints
    ckpt_dir = f"ml_checkpoints_profile_{device_name}"
    checkpoint_pool = CheckpointPool(
        max_size=5, save_dir=ckpt_dir, model_type="tactical", seed_existing=0)
    checkpoint_pool.save(model, 0)

    metrics = TrainingMetrics()

    # ── Multiprocessing pool ──
    ctx = _mp.get_context("spawn")
    pool = ctx.Pool(
        processes=worker_count,
        initializer=_init_shared_worker,
        initargs=(shared_model, shared_opponents, "tactical", True),
    )

    # ── Timing accumulators ──
    phase_times = {
        "army_gen": [],
        "episode_collect": [],
        "gae": [],
        "replay_forward": [],
        "loss_backward": [],
        "total_batch": [],
    }
    games_completed = 0

    config_start = time.time()
    batch_num = 0

    try:
        while True:
            # Time-limit check (same technique as run_training)
            elapsed = time.time() - config_start
            if elapsed >= time_budget_sec:
                break

            batch_num += 1
            batch_start = time.perf_counter()
            is_warmup = batch_num <= WARMUP_BATCHES

            heuristic_fraction = get_heuristic_fraction(
                metrics.heuristic_win_rate)

            # Copy training weights → shared model (always on CPU)
            shared_model.load_state_dict(model.state_dict())

            # ── Phase 1: build game specs ──
            t0 = time.perf_counter()
            game_specs = []
            for _ in range(batch_size):
                opp_type = "heuristic"
                opp_sd_idx = -1
                if random.random() >= heuristic_fraction:
                    opp_path = checkpoint_pool.sample_opponent_path()
                    if opp_path is not None:
                        opp_type = "selfplay"
                        opp_sd_idx = 0  # only one checkpoint for profiling
                    else:
                        opp_type = "selfplay_mirror"

                res_a, res_b, states_a, states_b, army_type = _generate_army_pair(
                    opp_type=opp_type)
                states_a_data = [(u.ai_role, u.combat_preference,
                                  u.assigned_objective) for u in states_a]
                states_b_data = [(u.ai_role, u.combat_preference,
                                  u.assigned_objective) for u in states_b]
                game_specs.append((res_a, res_b, states_a_data, states_b_data,
                                   opp_type, opp_sd_idx, army_type))
            army_gen_time = time.perf_counter() - t0

            # Copy opponent weights to shared slots
            opp_slot_map: dict[int, int] = {}

            # ── Phase 2: parallel episode collection (always CPU workers) ──
            t0 = time.perf_counter()
            chunk_size = max(1, len(game_specs) // worker_count)
            chunks = []
            for i in range(0, len(game_specs), chunk_size):
                chunk = game_specs[i : i + chunk_size]
                chunks.append((opp_slot_map, chunk, 0.0))  # shaping_scale=0

            chunk_results = list(pool.map(
                _collect_episodes_shared_worker, chunks))
            trajectories = [ep for chunk in chunk_results for ep in chunk]
            episode_time = time.perf_counter() - t0

            # ── Phase 3: compute GAE advantages ──
            t0 = time.perf_counter()
            all_trajs = [traj_rounds for traj_rounds, _, _, _ in trajectories]
            all_advantages, all_returns = compute_gae(all_trajs, gamma=1.0,
                                                       gae_lambda=0.95)
            for traj_tuple in trajectories:
                result_str = traj_tuple[1]
                opp_t = traj_tuple[2]
                if opp_t != "mirror_b":
                    metrics.record_game(result_str, opp_t)
            gae_time = time.perf_counter() - t0

            # ── Phase 4: PPO update on target device ──
            # Move model to target device for forward + backward
            model.to(device)
            optimizer_device = torch.optim.Adam(model.parameters(), lr=1e-4)

            # Pre-flatten tensors
            flat_old_lps = torch.tensor(
                [s.old_log_prob for traj in all_trajs for s in traj],
                dtype=torch.float32, device=device)
            flat_advantages_t = torch.tensor(
                [a for adv in all_advantages for a in adv],
                dtype=torch.float32, device=device)
            flat_returns_t = torch.tensor(
                [r for ret in all_returns for r in ret],
                dtype=torch.float32, device=device)

            # Precompute per-game step counts and offsets for minibatching
            game_step_counts = [len(traj) for traj in all_trajs]
            game_step_offsets = [0] * len(all_trajs)
            for gi in range(1, len(all_trajs)):
                game_step_offsets[gi] = (game_step_offsets[gi - 1]
                                         + game_step_counts[gi - 1])

            # PPO epochs with minibatching
            t0_replay = time.perf_counter()
            ppo_epochs = 3
            replay_time_accum = 0.0
            loss_time_accum = 0.0

            for _ppo_epoch in range(ppo_epochs):
                game_indices = list(range(len(all_trajs)))
                random.shuffle(game_indices)

                for mb_start in range(0, len(game_indices), minibatch_games):
                    mb_game_idx = game_indices[
                        mb_start : mb_start + minibatch_games]

                    mb_trajs = [all_trajs[i] for i in mb_game_idx]
                    mb_flat_idx: list[int] = []
                    for gi in mb_game_idx:
                        off = game_step_offsets[gi]
                        mb_flat_idx.extend(
                            range(off, off + game_step_counts[gi]))
                    if not mb_flat_idx:
                        continue
                    idx_t = torch.tensor(mb_flat_idx, dtype=torch.long,
                                         device=device)
                    mb_old_lps = flat_old_lps[idx_t]
                    mb_advantages = flat_advantages_t[idx_t]
                    mb_returns = flat_returns_t[idx_t]

                    # Forward pass (replay) on target device
                    t_fwd = time.perf_counter()
                    with force_tensor_device(device):
                        mb_flat_result = replay_tactical_log_probs_flat(
                            model, mb_trajs)
                    if device_name == "cuda":
                        torch.cuda.synchronize()
                    replay_time_accum += time.perf_counter() - t_fwd

                    # Loss + backward on target device
                    t_loss = time.perf_counter()
                    mb_flat_steps = [s for traj in mb_trajs for s in traj]
                    loss, _ = compute_loss_flat(
                        mb_flat_result, mb_old_lps, mb_advantages,
                        mb_returns, clip_epsilon=0.2, value_coeff=0.5,
                        entropy_coeff=0.01, flat_steps=mb_flat_steps)

                    optimizer_device.zero_grad()
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0)
                        optimizer_device.step()
                    if device_name == "cuda":
                        torch.cuda.synchronize()
                    loss_time_accum += time.perf_counter() - t_loss

            # Move model back to CPU (for shared_model sync next batch)
            model.to("cpu")

            total_batch = time.perf_counter() - batch_start

            if not is_warmup:
                phase_times["army_gen"].append(army_gen_time)
                phase_times["episode_collect"].append(episode_time)
                phase_times["gae"].append(gae_time)
                phase_times["replay_forward"].append(replay_time_accum)
                phase_times["loss_backward"].append(loss_time_accum)
                phase_times["total_batch"].append(total_batch)
                games_completed += batch_size

            status = "WARMUP" if is_warmup else f"batch {batch_num}"
            print(f"  [{status}] {total_batch:.2f}s total | "
                  f"sim={episode_time:.2f}s | "
                  f"replay={replay_time_accum:.3f}s | "
                  f"loss+bwd={loss_time_accum:.3f}s | "
                  f"army={army_gen_time:.2f}s | "
                  f"gae={gae_time:.3f}s")

    finally:
        pool.close()
        pool.join()

    # ── Compute summary stats ──
    result = {
        "device": device_name,
        "minibatch_games": minibatch_games,
        "worker_count": worker_count,
        "batches_completed": max(0, batch_num - WARMUP_BATCHES),
        "games_completed": games_completed,
    }

    if phase_times["total_batch"]:
        for phase, times in phase_times.items():
            result[f"{phase}_mean"] = statistics.mean(times)
            if len(times) > 1:
                result[f"{phase}_std"] = statistics.stdev(times)
            else:
                result[f"{phase}_std"] = 0.0
        result["games_per_sec"] = (
            batch_size / result["total_batch_mean"])
        result["ppo_pct"] = 100 * (
            result["replay_forward_mean"] + result["loss_backward_mean"]
        ) / result["total_batch_mean"]
        result["sim_pct"] = (
            100 * result["episode_collect_mean"]
            / result["total_batch_mean"])
    else:
        # Ensure all keys exist even when no batches completed, so the
        # CSV DictWriter doesn't drop columns.
        for phase in phase_times:
            result[f"{phase}_mean"] = 0.0
            result[f"{phase}_std"] = 0.0
        result["games_per_sec"] = 0.0
        result["ppo_pct"] = 0.0
        result["sim_pct"] = 0.0

    # Clean up temp checkpoints
    ckpt_path = Path(ckpt_dir)
    if ckpt_path.exists():
        for f in ckpt_path.glob("*.pt"):
            f.unlink()
        ckpt_path.rmdir()

    return result


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CPU vs GPU Training Profiler")
    print(f"  Batch size: {BATCH_SIZE} games/batch")
    print(f"  Devices: {DEVICES}")
    print(f"  Minibatch sizes: {MINIBATCH_SIZES}")
    print(f"  Worker counts: {WORKER_COUNTS}")
    print(f"  Total time budget: {TOTAL_TIME_LIMIT_MIN} minutes")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  GPU: not available (CPU-only comparison)")
    print("=" * 70)

    configs = []
    for device in DEVICES:
        for mb in MINIBATCH_SIZES:
            for wc in WORKER_COUNTS:
                configs.append((device, mb, wc))

    n_configs = len(configs)
    time_per_config = (TOTAL_TIME_LIMIT_MIN * 60) / n_configs
    print(f"\n{n_configs} configurations, {time_per_config:.0f}s each")

    global_start = time.time()
    all_results: list[dict] = []

    for i, (device, mb, wc) in enumerate(configs):
        # Recompute remaining time and distribute evenly among remaining
        elapsed = time.time() - global_start
        remaining = TOTAL_TIME_LIMIT_MIN * 60 - elapsed
        configs_left = n_configs - i
        budget = remaining / configs_left

        if budget < 30:
            print(f"\nSkipping remaining configs — only {budget:.0f}s left")
            break

        result = profile_config(device, mb, wc, budget, BATCH_SIZE)
        all_results.append(result)

        # Force cleanup between configs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ────────────────────────────────────────────────────
    total_elapsed = time.time() - global_start
    print("\n" + "=" * 70)
    print(f"PROFILING COMPLETE — {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min)")
    print("=" * 70)

    # Header
    print(f"\n{'Device':>6} {'MB':>4} {'Wk':>3} {'Batches':>7} "
          f"{'Games/s':>8} {'Batch(s)':>8} {'Sim%':>5} {'PPO%':>5} "
          f"{'Sim(s)':>7} {'Replay(s)':>9} {'Loss(s)':>8}")
    print("─" * 90)

    for r in all_results:
        if r["batches_completed"] == 0:
            print(f"{r['device']:>6} {r['minibatch_games']:>4} "
                  f"{r['worker_count']:>3}   (no data — budget too short)")
            continue
        print(f"{r['device']:>6} {r['minibatch_games']:>4} "
              f"{r['worker_count']:>3} {r['batches_completed']:>7} "
              f"{r['games_per_sec']:>8.1f} "
              f"{r['total_batch_mean']:>8.2f} "
              f"{r['sim_pct']:>5.1f} "
              f"{r['ppo_pct']:>5.1f} "
              f"{r['episode_collect_mean']:>7.2f} "
              f"{r['replay_forward_mean']:>9.3f} "
              f"{r['loss_backward_mean']:>8.3f}")

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = Path("profile_cpu_vs_gpu_results.csv")
    with open(csv_path, "w", newline="") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")

    # ── Key comparisons ──────────────────────────────────────────────────
    if len(all_results) >= 2:
        cpu_results = [r for r in all_results
                       if r["device"] == "cpu" and r["batches_completed"] > 0]
        gpu_results = [r for r in all_results
                       if r["device"] == "cuda" and r["batches_completed"] > 0]

        if cpu_results and gpu_results:
            print("\n── CPU vs GPU comparison (matched configs) ──")
            for cpu_r in cpu_results:
                for gpu_r in gpu_results:
                    if (cpu_r["minibatch_games"] == gpu_r["minibatch_games"]
                            and cpu_r["worker_count"] == gpu_r["worker_count"]):
                        mb = cpu_r["minibatch_games"]
                        wc = cpu_r["worker_count"]
                        speedup = (cpu_r["total_batch_mean"]
                                   / gpu_r["total_batch_mean"])
                        ppo_speedup_num = (
                            cpu_r.get("replay_forward_mean", 0)
                            + cpu_r.get("loss_backward_mean", 0))
                        ppo_speedup_den = (
                            gpu_r.get("replay_forward_mean", 0)
                            + gpu_r.get("loss_backward_mean", 0))
                        ppo_speedup = (ppo_speedup_num / ppo_speedup_den
                                       if ppo_speedup_den > 0 else 0)
                        print(f"  MB={mb:>3} Workers={wc}: "
                              f"total {speedup:.2f}x | "
                              f"PPO phase {ppo_speedup:.2f}x")


if __name__ == "__main__":
    main()
