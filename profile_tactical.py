"""Profile tactical model training to identify bottlenecks."""
from __future__ import annotations

import os
import time
import random
import statistics

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.multiprocessing as _mp

from ml_training import (
    TrainingConfig, TacticalModel, CheckpointPool, _make_model,
    TrainingMetrics, get_heuristic_fraction, compute_gae,
    _generate_army_pair, _collect_episodes_shared_worker, _init_shared_worker,
    replay_tactical_log_probs_batch, replay_tactical_log_probs_flat,
    compute_loss, compute_loss_flat,
    sample_tactical_actions_no_grad, _WORKER_COUNT, _MAX_SHARED_OPPONENTS,
    TacticalActivationRecord,
)
from ml_features import encode_state_tactical, precompute_damage, TACTICAL_TOTAL_FEATURES
from board import Board
from game import deploy_armies
from models import UnitState


def profile_components():
    """Profile individual components of the tactical training pipeline."""
    print("=" * 70)
    print("TACTICAL MODEL TRAINING — COMPONENT PROFILER")
    print(f"Workers: {_WORKER_COUNT}")
    print("=" * 70)

    # --- 1. Army generation ---
    print("\n--- 1. Army generation (20 samples) ---")
    army_gen_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _generate_army_pair()
        army_gen_times.append(time.perf_counter() - t0)
    print(f"  Mean: {statistics.mean(army_gen_times)*1000:.1f}ms | "
          f"Min: {min(army_gen_times)*1000:.1f}ms | Max: {max(army_gen_times)*1000:.1f}ms")

    # --- 2. State encoding (tactical vs strategic) ---
    print("\n--- 2. State encoding ---")
    res_a, res_b, states_a, states_b, *_ = _generate_army_pair()
    board = Board()
    deploy_armies(states_a, states_b, board)

    # Precompute damage
    dmg_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        precompute_damage([u.unit for u in states_a], [u.unit for u in states_b])
        dmg_times.append(time.perf_counter() - t0)
    print(f"  precompute_damage: {statistics.mean(dmg_times)*1000:.2f}ms avg")

    fr_a, fm_a = precompute_damage([u.unit for u in states_a], [u.unit for u in states_b])
    fr_b, fm_b = precompute_damage([u.unit for u in states_b], [u.unit for u in states_a])
    pts_a = sum(u.unit.points for u in states_a)
    pts_b = sum(u.unit.points for u in states_b)

    encode_times = []
    for _ in range(200):
        t0 = time.perf_counter()
        encode_state_tactical(
            states_a, states_b, 1, board, "A",
            friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
            enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
            total_friendly_points=pts_a, total_enemy_points=pts_b,
        )
        encode_times.append(time.perf_counter() - t0)
    print(f"  encode_state_tactical: {statistics.mean(encode_times)*1000:.2f}ms avg")

    # How many encode calls per game? ~10 activations per side × 4 rounds = ~40 for Player A
    n_alive_a = sum(1 for u in states_a if u.models_alive > 0)
    print(f"  Alive units (Player A): {n_alive_a}")
    print(f"  Est. encode calls per game: ~{n_alive_a * 4} (Player A side only)")
    est_encode_per_game = n_alive_a * 4 * statistics.mean(encode_times)
    print(f"  Est. encode time per game: {est_encode_per_game*1000:.1f}ms")

    # --- 3. Model forward pass ---
    print("\n--- 3. Model forward pass ---")
    model = TacticalModel()
    model.eval()

    state_vec = encode_state_tactical(
        states_a, states_b, 1, board, "A",
        friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
        enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
        total_friendly_points=pts_a, total_enemy_points=pts_b,
    )
    alive_mask = torch.tensor([u.models_alive > 0 for u in states_a] +
                              [False] * (10 - len(states_a)), dtype=torch.bool)[:10]

    # Single forward
    fwd_times = []
    for _ in range(200):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(state_vec, alive_mask)
        fwd_times.append(time.perf_counter() - t0)
    print(f"  Single forward (no grad): {statistics.mean(fwd_times)*1000:.3f}ms avg")
    print(f"  Est. forward passes per game: ~{n_alive_a * 4}")
    est_fwd_per_game = n_alive_a * 4 * statistics.mean(fwd_times)
    print(f"  Est. forward pass time per game: {est_fwd_per_game*1000:.1f}ms")

    # Batched forward (simulating replay)
    for batch_n in [64, 256, 1024, 4096]:
        batch_vecs = state_vec.unsqueeze(0).expand(batch_n, -1).contiguous()
        batch_masks = alive_mask.unsqueeze(0).expand(batch_n, -1).contiguous()
        fwd_batch_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            model(batch_vecs, batch_masks)
            fwd_batch_times.append(time.perf_counter() - t0)
        print(f"  Batched forward ({batch_n:4d} steps): {statistics.mean(fwd_batch_times)*1000:.2f}ms avg "
              f"({statistics.mean(fwd_batch_times)*1000/batch_n:.3f}ms/step)")

    # --- 4. Full episode simulation (single game, no multiprocessing) ---
    print("\n--- 4. Single episode simulation ---")
    from ml_training import _run_single_episode_tactical
    from board import OBJECTIVES as BOARD_OBJECTIVES

    episode_times = []
    episode_step_counts = []
    for _ in range(10):
        res_a2, res_b2, states_a2, states_b2, *_ = _generate_army_pair()
        sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a2]
        sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b2]

        t0 = time.perf_counter()
        traj, result, opp_type, _traj_b = _run_single_episode_tactical(
            model, None, res_a2, res_b2, sa_data, sb_data, "heuristic", BOARD_OBJECTIVES,
        )
        episode_times.append(time.perf_counter() - t0)
        episode_step_counts.append(len(traj))

    print(f"  Mean episode time: {statistics.mean(episode_times)*1000:.1f}ms")
    print(f"  Mean steps per episode: {statistics.mean(episode_step_counts):.1f}")
    print(f"  Time per step: {statistics.mean(episode_times)*1000/statistics.mean(episode_step_counts):.2f}ms")

    # Breakdown: model forward vs game simulation
    print(f"  Est. model time per episode: {statistics.mean(episode_step_counts) * statistics.mean(fwd_times)*1000:.1f}ms")
    est_model_frac = (statistics.mean(episode_step_counts) * statistics.mean(fwd_times)) / statistics.mean(episode_times)
    print(f"  Est. model fraction: {est_model_frac*100:.1f}%")
    print(f"  Est. game-sim fraction: {(1-est_model_frac)*100:.1f}%")

    # --- 5. Replay log-probs (batched) ---
    print("\n--- 5. Replay log-probs (batched) ---")
    model.train()
    # Collect some trajectories for replay test
    all_trajs = []
    for _ in range(5):
        res_a2, res_b2, states_a2, states_b2, *_ = _generate_army_pair()
        sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a2]
        sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b2]
        model.eval()
        traj, _, _, _ = _run_single_episode_tactical(
            model, None, res_a2, res_b2, sa_data, sb_data, "heuristic", BOARD_OBJECTIVES,
        )
        all_trajs.append(traj)
    model.train()

    total_steps = sum(len(t) for t in all_trajs)
    print(f"  Total steps across {len(all_trajs)} episodes: {total_steps}")

    replay_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        replay_tactical_log_probs_batch(model, all_trajs)
        replay_times.append(time.perf_counter() - t0)
    print(f"  Replay time ({total_steps} steps): {statistics.mean(replay_times)*1000:.2f}ms avg")
    print(f"  Per-step replay: {statistics.mean(replay_times)*1000/total_steps:.3f}ms")

    # Scale estimate for full batch
    for batch_size in [64, 128, 256]:
        est_steps = batch_size * statistics.mean(episode_step_counts)
        est_replay = est_steps * (statistics.mean(replay_times) / total_steps)
        print(f"  Est. replay for batch_size={batch_size}: {est_replay*1000:.0f}ms ({est_steps:.0f} steps)")


def profile_full_batch(num_batches=3, batch_size=64):
    """Profile full training batches with phase breakdown."""
    print("\n" + "=" * 70)
    print(f"FULL BATCH PROFILING — {num_batches} batches × {batch_size} games")
    print("=" * 70)

    config = TrainingConfig(
        num_batches=num_batches,
        batch_size=batch_size,
        model_type="tactical",
        checkpoint_dir="ml_checkpoints_profile_tactical",
    )

    model = _make_model("tactical")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    checkpoint_pool = CheckpointPool(max_size=5, save_dir=config.checkpoint_dir, model_type="tactical")
    metrics = TrainingMetrics()
    checkpoint_pool.save(model, 0)

    # Shared-memory setup
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
        processes=_WORKER_COUNT,
        initializer=_init_shared_worker,
        initargs=(shared_model, shared_opponents, "tactical"),
    )

    phase_times = {
        "army_gen": [],
        "work_build": [],
        "episode_collect": [],
        "gae_compute": [],
        "replay_logprobs": [],
        "loss_backward": [],
        "total_batch": [],
    }
    step_counts = []

    for batch_num in range(1, num_batches + 1):
        batch_start = time.perf_counter()
        heuristic_fraction = get_heuristic_fraction(metrics.heuristic_win_rate)

        shared_model.load_state_dict(model.state_dict())

        # Phase 1: Build work items
        t0 = time.perf_counter()
        game_specs = []
        opponent_state_dicts = []
        _opp_path_cache = {}
        army_gen_time = 0

        for _ in range(batch_size):
            opp_type = "heuristic"
            opp_sd_idx = -1
            if random.random() >= heuristic_fraction:
                opp_path = checkpoint_pool.sample_opponent_path()
                if opp_path is not None:
                    opp_type = "selfplay"
                    path_key = str(opp_path)
                    if path_key not in _opp_path_cache:
                        _opp_path_cache[path_key] = len(opponent_state_dicts)
                        opponent_state_dicts.append(checkpoint_pool.load_state_dict(opp_path))
                    opp_sd_idx = _opp_path_cache[path_key]

            t_ag = time.perf_counter()
            res_a, res_b, states_a, states_b, army_type = _generate_army_pair()
            army_gen_time += time.perf_counter() - t_ag

            sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a]
            sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b]
            game_specs.append((res_a, res_b, sa_data, sb_data, opp_type, opp_sd_idx, army_type))

        work_build_time = time.perf_counter() - t0
        phase_times["army_gen"].append(army_gen_time)
        phase_times["work_build"].append(work_build_time)

        # Copy opponent weights
        opp_slot_map = {}
        for i, sd in enumerate(opponent_state_dicts):
            if i < _MAX_SHARED_OPPONENTS:
                shared_opponents[i].load_state_dict(sd)
                opp_slot_map[i] = i

        # Phase 2: Episode collection
        t0 = time.perf_counter()
        n_chunks = _WORKER_COUNT
        chunk_size_val = max(1, len(game_specs) // n_chunks)
        chunks = [
            (opp_slot_map, game_specs[i : i + chunk_size_val])
            for i in range(0, len(game_specs), chunk_size_val)
        ]
        chunk_results = list(pool.map(_collect_episodes_shared_worker, chunks))
        trajectories = [ep for chunk in chunk_results for ep in chunk]
        episode_time = time.perf_counter() - t0
        phase_times["episode_collect"].append(episode_time)

        all_trajs = [t[0] for t in trajectories]
        total_steps = sum(len(t) for t in all_trajs)
        step_counts.append(total_steps)

        # Phase 3: GAE
        t0 = time.perf_counter()
        all_advantages, all_returns = compute_gae(all_trajs, gamma=1.0, gae_lambda=config.gae_lambda)
        gae_time = time.perf_counter() - t0
        phase_times["gae_compute"].append(gae_time)

        opp_types = [ot for _, _, ot, _ in trajectories]
        for _, result, opp_type, army_type in trajectories:
            metrics.record_game(result, opp_type, army_type)

        # Phase 4: PPO (3 epochs) — vectorized flat path
        replay_total = 0
        loss_total = 0
        model.train()

        flat_old_lps = torch.tensor(
            [s.old_log_prob for traj in all_trajs for s in traj],
            dtype=torch.float32,
        )
        flat_advantages_t = torch.tensor(
            [a for adv in all_advantages for a in adv],
            dtype=torch.float32,
        )
        flat_returns_t = torch.tensor(
            [r for ret in all_returns for r in ret],
            dtype=torch.float32,
        )

        for _ppo_epoch in range(config.ppo_epochs):
            t0 = time.perf_counter()
            flat_result = replay_tactical_log_probs_flat(model, all_trajs)
            replay_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            loss, loss_metrics = compute_loss_flat(
                flat_result, flat_old_lps, flat_advantages_t, flat_returns_t,
                config.clip_epsilon, config.value_coeff, config.entropy_coeff_start,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += time.perf_counter() - t0

        phase_times["replay_logprobs"].append(replay_total)
        phase_times["loss_backward"].append(loss_total)

        total = time.perf_counter() - batch_start
        phase_times["total_batch"].append(total)

        print(f"  Batch {batch_num}: {total:.2f}s | "
              f"army={army_gen_time:.2f}s | ep={episode_time:.2f}s | "
              f"gae={gae_time:.3f}s | replay={replay_total:.3f}s | "
              f"loss={loss_total:.3f}s | steps={total_steps}")

    pool.close()
    pool.join()

    # Summary
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)
    batch_avg = statistics.mean(phase_times["total_batch"])
    for phase, times in phase_times.items():
        avg = statistics.mean(times)
        pct = 100 * avg / batch_avg if phase != "total_batch" else 100
        print(f"  {phase:20s}: {avg:.3f}s avg ({pct:5.1f}% of batch)")

    avg_steps = statistics.mean(step_counts)
    print(f"\n  Avg steps per batch: {avg_steps:.0f} ({avg_steps/batch_size:.1f} per game)")
    print(f"  Games/second: {batch_size / batch_avg:.1f}")
    print(f"  Steps/second: {avg_steps / batch_avg:.0f}")

    ep_avg = statistics.mean(phase_times["episode_collect"])
    print(f"\n  Episode collection: {ep_avg:.2f}s ({ep_avg/batch_size*1000:.1f}ms/game)")
    print(f"  (this includes {_WORKER_COUNT} parallel workers)")
    print(f"  Effective single-thread game time: {ep_avg * _WORKER_COUNT / batch_size * 1000:.1f}ms/game")

    # Cleanup
    import shutil
    profile_dir = "ml_checkpoints_profile_tactical"
    if os.path.exists(profile_dir):
        shutil.rmtree(profile_dir)


if __name__ == "__main__":
    profile_components()
    profile_full_batch(num_batches=3, batch_size=64)
