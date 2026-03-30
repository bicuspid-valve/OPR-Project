"""Profile ML training to identify time sinks."""
from __future__ import annotations

import os
import time
import random
import statistics

import torch
import torch.multiprocessing as _mp

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ml_training import (
    TrainingConfig, CheckpointPool, EMABaseline,
    TrainingMetrics, get_heuristic_fraction,
    _generate_army_pair, _collect_episodes_shared_worker, _init_shared_worker,
    replay_tactical_log_probs_flat, compute_loss_flat, compute_gae,
    _WORKER_COUNT, _MAX_SHARED_OPPONENTS,
)
from ml_model_tactical import TacticalModel
from ml_features import encode_state_tactical, precompute_damage


def profile_training(num_batches=5, batch_size=16):
    print(f"=== ML Training Profiler ===")
    print(f"Batches: {num_batches}, Batch size: {batch_size}, Workers: {_WORKER_COUNT}")
    print()

    model = TacticalModel()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    checkpoint_pool = CheckpointPool(max_size=5, save_dir="ml_checkpoints_profile")
    baseline_h = EMABaseline()
    baseline_s = EMABaseline()
    metrics = TrainingMetrics()
    checkpoint_pool.save(model, 0)

    # Shared-memory pool
    shared_model = TacticalModel()
    shared_model.share_memory()
    shared_model.eval()
    shared_opponents = []
    for _ in range(_MAX_SHARED_OPPONENTS):
        m = TacticalModel()
        m.share_memory()
        m.eval()
        shared_opponents.append(m)

    ctx = _mp.get_context("spawn")
    pool = ctx.Pool(
        processes=_WORKER_COUNT,
        initializer=_init_shared_worker,
        initargs=(shared_model, shared_opponents),
    )

    phase_times = {
        "army_gen": [],
        "work_build": [],
        "episode_collect": [],
        "replay_logprobs": [],
        "loss_backward": [],
        "total_batch": [],
    }

    # Also profile individual components
    single_army_gen_times = []

    # Profile single army generation
    print("--- Profiling army generation (20 samples) ---")
    for _ in range(20):
        t0 = time.perf_counter()
        _generate_army_pair()
        single_army_gen_times.append(time.perf_counter() - t0)
    print(f"  Army pair generation: {statistics.mean(single_army_gen_times)*1000:.1f}ms avg "
          f"(min {min(single_army_gen_times)*1000:.1f}, max {max(single_army_gen_times)*1000:.1f})")

    # Profile feature encoding
    print("\n--- Profiling feature encoding (50 samples) ---")
    res_a, res_b, states_a, states_b, *_ = _generate_army_pair()
    from board import Board
    from game import deploy_armies
    board = Board()
    deploy_armies(states_a, states_b, board)

    encode_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        encode_state_tactical(states_a, states_b, 1, board, "A")
        encode_times.append(time.perf_counter() - t0)
    print(f"  encode_state_tactical: {statistics.mean(encode_times)*1000:.2f}ms avg")

    # Profile precompute_damage
    damage_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        precompute_damage([u.unit for u in states_a], [u.unit for u in states_b])
        damage_times.append(time.perf_counter() - t0)
    print(f"  precompute_damage: {statistics.mean(damage_times)*1000:.2f}ms avg")

    # Profile model forward pass
    print("\n--- Profiling model forward pass ---")
    state_vec = encode_state_tactical(states_a, states_b, 1, board, "A")
    fwd_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(state_vec)
        fwd_times.append(time.perf_counter() - t0)
    print(f"  Single forward pass (no grad): {statistics.mean(fwd_times)*1000:.3f}ms avg")

    # Batched forward
    batch_vecs = torch.stack([state_vec] * batch_size)
    fwd_batch_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        model(batch_vecs)
        fwd_batch_times.append(time.perf_counter() - t0)
    print(f"  Batched forward ({batch_size}): {statistics.mean(fwd_batch_times)*1000:.3f}ms avg")

    # Full training loop profiling
    print(f"\n--- Profiling full training batches ({num_batches} batches) ---")
    for batch_num in range(1, num_batches + 1):
        batch_start = time.perf_counter()
        heuristic_fraction = get_heuristic_fraction(metrics.heuristic_win_rate)

        # Copy weights to shared model
        shared_model.load_state_dict(model.state_dict())

        # Phase 1: Build work items
        t0 = time.perf_counter()
        game_specs = []
        opponent_state_dicts = []
        _opp_sd_cache: dict[int, int] = {}
        army_gen_time = 0
        for _ in range(batch_size):
            opp_type = "heuristic"
            opp_sd_idx = -1
            if random.random() >= heuristic_fraction:
                opp_sd = checkpoint_pool.sample_opponent_state_dict()
                if opp_sd is not None:
                    opp_type = "selfplay"
                    sd_id = id(opp_sd)
                    if sd_id not in _opp_sd_cache:
                        _opp_sd_cache[sd_id] = len(opponent_state_dicts)
                        opponent_state_dicts.append(opp_sd)
                    opp_sd_idx = _opp_sd_cache[sd_id]

            t_ag = time.perf_counter()
            res_a, res_b, states_a, states_b, *_ = _generate_army_pair()
            army_gen_time += time.perf_counter() - t_ag
            states_a_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a]
            states_b_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b]
            game_specs.append((res_a, res_b, states_a_data, states_b_data, opp_type, opp_sd_idx, "random"))
        work_build_time = time.perf_counter() - t0
        phase_times["army_gen"].append(army_gen_time)
        phase_times["work_build"].append(work_build_time)

        # Copy opponent weights to shared slots
        opp_slot_map: dict[int, int] = {}
        for i, sd in enumerate(opponent_state_dicts):
            if i < _MAX_SHARED_OPPONENTS:
                shared_opponents[i].load_state_dict(sd)
                opp_slot_map[i] = i

        # Phase 2: Parallel episode collection
        t0 = time.perf_counter()
        n_chunks = _WORKER_COUNT
        chunk_size_val = max(1, len(game_specs) // n_chunks)
        chunks = [
            (opp_slot_map, game_specs[i : i + chunk_size_val], 1.0)
            for i in range(0, len(game_specs), chunk_size_val)
        ]
        trajectories_raw = list(pool.map(_collect_episodes_shared_worker, chunks))
        trajectories = [ep for chunk in trajectories_raw for ep in chunk]
        episode_time = time.perf_counter() - t0
        phase_times["episode_collect"].append(episode_time)

        # Phase 3: Replay log-probs
        t0 = time.perf_counter()
        model.train()
        all_trajs = [t[0] for t in trajectories]
        flat_result = replay_tactical_log_probs_flat(model, all_trajs)
        replay_time = time.perf_counter() - t0
        phase_times["replay_logprobs"].append(replay_time)

        # Phase 4: Loss + backward
        t0 = time.perf_counter()
        for traj_tuple in trajectories:
            result = traj_tuple[1]
            opp_type = traj_tuple[2]
            metrics.record_game(result, opp_type)
        all_advantages, all_returns = compute_gae(all_trajs)
        # Flatten advantages/returns
        flat_advantages = torch.tensor([a for ep in all_advantages for a in ep], dtype=torch.float32)
        flat_returns = torch.tensor([r for ep in all_returns for r in ep], dtype=torch.float32)
        flat_old_lp = torch.tensor(
            [step.old_log_prob for traj in all_trajs for step in traj], dtype=torch.float32)
        loss, loss_metrics = compute_loss_flat(
            flat_result, flat_old_lp, flat_advantages, flat_returns,
            clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_time = time.perf_counter() - t0
        phase_times["loss_backward"].append(loss_time)

        total = time.perf_counter() - batch_start
        phase_times["total_batch"].append(total)

        print(f"  Batch {batch_num}: {total:.2f}s total | "
              f"army_gen={army_gen_time:.2f}s | "
              f"work_build={work_build_time:.2f}s | "
              f"episodes={episode_time:.2f}s | "
              f"replay={replay_time:.3f}s | "
              f"loss+bwd={loss_time:.3f}s")

    pool.close()
    pool.join()

    # Summary
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)
    for phase, times in phase_times.items():
        avg = statistics.mean(times)
        total_pct = 100 * avg / statistics.mean(phase_times["total_batch"]) if phase != "total_batch" else 100
        print(f"  {phase:20s}: {avg:.3f}s avg ({total_pct:5.1f}% of batch)")

    print(f"\n  Games/second: {batch_size / statistics.mean(phase_times['total_batch']):.1f}")
    print(f"  Batch size: {batch_size}, Workers: {_WORKER_COUNT}")

    # Breakdown of where within episodes the time goes
    print(f"\n--- Time breakdown ---")
    ep_avg = statistics.mean(phase_times["episode_collect"])
    replay_avg = statistics.mean(phase_times["replay_logprobs"])
    army_avg = statistics.mean(phase_times["army_gen"])
    loss_avg = statistics.mean(phase_times["loss_backward"])
    batch_avg = statistics.mean(phase_times["total_batch"])
    other = batch_avg - ep_avg - replay_avg - army_avg - loss_avg

    print(f"  Army generation:    {army_avg:.3f}s ({100*army_avg/batch_avg:.1f}%)")
    print(f"  Episode simulation: {ep_avg:.3f}s ({100*ep_avg/batch_avg:.1f}%)")
    print(f"  Replay log-probs:   {replay_avg:.3f}s ({100*replay_avg/batch_avg:.1f}%)")
    print(f"  Loss + backward:    {loss_avg:.3f}s ({100*loss_avg/batch_avg:.1f}%)")
    print(f"  Overhead:           {other:.3f}s ({100*other/batch_avg:.1f}%)")


if __name__ == "__main__":
    profile_training(num_batches=2, batch_size=512)
