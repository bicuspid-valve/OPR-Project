"""Phase-level profiler: single-process inline run for clean sub-phase timing."""
from __future__ import annotations
import os, time

os.environ["_ML_TRAIN_CGROUP"] = "1"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BATCH_SIZE = 8
N_BATCHES = 3

if __name__ == "__main__":
    import torch
    import random
    import numpy as np

    torch.set_num_threads(1)

    from ml_training import TrainingConfig
    from ml_training.collection import (
        _run_games_batched_tactical,
        _init_shared_worker, _get_opponent_type_idx,
    )
    from ml_training.metrics import _generate_army_pair
    from ml_training.loop import (
        compute_gae, replay_tactical_log_probs_flat, compute_loss_flat,
        _make_model, CheckpointPool,
    )
    from ml_features import (
        encode_state_tactical as _orig_encode,
        precompute_damage as _orig_precompute,
    )
    from ml_integration_tactical import (
        compute_destination_candidates as _orig_dest_cands,
        compute_destination_features as _orig_dest_feats,
    )
    from ml_training.sampling import (
        _batched_sample_tactical_no_grad as _orig_batched_infer,
    )
    import ml_features
    import ml_integration_tactical
    import ml_training.collection as coll
    import ml_training.sampling as samp
    import fast_core
    fast_core.USE_C_EXT = fast_core.is_available()

    # ---- Accumulators ----
    t_encode = [0.0]
    t_dcands = [0.0]
    t_dfeats = [0.0]
    t_infer = [0.0]
    t_precomp = [0.0]
    t_army_gen = [0.0]
    n_act = [0]
    n_encode = [0]

    # ---- Patch functions ----
    def _w_encode(*a, **kw):
        t0 = time.perf_counter(); r = _orig_encode(*a, **kw)
        t_encode[0] += time.perf_counter() - t0; n_encode[0] += 1; return r

    def _w_precompute(*a, **kw):
        t0 = time.perf_counter(); r = _orig_precompute(*a, **kw)
        t_precomp[0] += time.perf_counter() - t0; return r

    def _w_dest_cands(*a, **kw):
        t0 = time.perf_counter(); r = _orig_dest_cands(*a, **kw)
        t_dcands[0] += time.perf_counter() - t0; return r

    def _w_dest_feats(*a, **kw):
        t0 = time.perf_counter(); r = _orig_dest_feats(*a, **kw)
        t_dfeats[0] += time.perf_counter() - t0; return r

    def _w_infer(*a, **kw):
        t0 = time.perf_counter(); r = _orig_batched_infer(*a, **kw)
        t_infer[0] += time.perf_counter() - t0
        n_act[0] += len(a[1]) if len(a) > 1 else 0
        return r

    ml_features.encode_state_tactical = _w_encode
    coll.encode_state_tactical = _w_encode
    coll.precompute_damage = _w_precompute
    ml_integration_tactical.compute_destination_candidates = _w_dest_cands
    ml_integration_tactical.compute_destination_features = _w_dest_feats
    coll.compute_destination_candidates = _w_dest_cands
    coll.compute_destination_features = _w_dest_feats
    samp._batched_sample_tactical_no_grad = _w_infer
    coll._batched_sample_tactical_no_grad = _w_infer

    # ---- Setup model ----
    model = _make_model("tactical")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.eval()  # for inference during collection

    print("=" * 70)
    print(f"SINGLE-PROCESS PROFILER — batch={BATCH_SIZE}, {N_BATCHES} batches")
    print(f"C extension: {'ON' if fast_core.USE_C_EXT else 'OFF'}")
    print("=" * 70)

    all_batch_times = []
    t_gae_total = 0.0
    t_replay_total = 0.0
    t_loss_total = 0.0
    t_backward_total = 0.0
    t_optim_total = 0.0
    t_collect_total = 0.0
    total_games = 0

    for batch_num in range(1, N_BATCHES + 1):
        batch_start = time.perf_counter()

        # Build game specs
        t0 = time.perf_counter()
        game_specs = []
        for _ in range(BATCH_SIZE):
            res_a, res_b, states_a, states_b, army_type = _generate_army_pair()
            states_a_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a]
            states_b_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b]
            game_specs.append((res_a, res_b, states_a_data, states_b_data,
                              "heuristic", -1, army_type))
        t_army_gen[0] += time.perf_counter() - t0

        # Episode collection (inline, single process)
        t0 = time.perf_counter()
        trajectories = _run_games_batched_tactical(model, game_specs, {},
                                                    shaping_scale=0.0,
                                                    planning_rate=0.01,
                                                    planning_params={"K_UNITS": 3, "C_SAMPLES_PER_UNIT": 3,
                                                                     "M_ROLLOUTS": 4, "N_LOOKAHEAD": 3})
        t_collect = time.perf_counter() - t0
        t_collect_total += t_collect
        total_games += len(trajectories)

        # GAE
        all_trajs = [t[0] for t in trajectories]
        t0 = time.perf_counter()
        all_advantages, all_returns = compute_gae(all_trajs, gamma=1.0, gae_lambda=0.95,
                                                   unit_local_blend=0.25)
        t_gae = time.perf_counter() - t0
        t_gae_total += t_gae

        # PPO update
        model.train()
        flat_old_lps = torch.tensor([s.old_log_prob for traj in all_trajs for s in traj],
                                     dtype=torch.float32)
        flat_adv = torch.tensor([a for adv in all_advantages for a in adv], dtype=torch.float32)
        flat_ret = torch.tensor([r for ret in all_returns for r in ret], dtype=torch.float32)

        t0 = time.perf_counter()
        flat_result = replay_tactical_log_probs_flat(model, all_trajs)
        t_replay = time.perf_counter() - t0
        t_replay_total += t_replay

        t0 = time.perf_counter()
        flat_steps = [s for traj in all_trajs for s in traj]
        loss, loss_metrics = compute_loss_flat(flat_result, flat_old_lps, flat_adv, flat_ret,
                                               0.2, 0.5, 0.01, flat_steps=flat_steps)
        t_loss = time.perf_counter() - t0
        t_loss_total += t_loss

        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        t_bwd = time.perf_counter() - t0
        t_backward_total += t_bwd

        t0 = time.perf_counter()
        optimizer.step()
        t_opt = time.perf_counter() - t0
        t_optim_total += t_opt

        model.eval()
        batch_time = time.perf_counter() - batch_start
        all_batch_times.append(batch_time)

        print(f"  Batch {batch_num}: {batch_time:.1f}s | collect={t_collect:.1f}s "
              f"replay={t_replay:.2f}s loss={t_loss:.3f}s bwd={t_bwd:.3f}s")

    wall_elapsed = sum(all_batch_times)

    # ---- Report ----
    print("\n" + "=" * 70)
    print(f"PROFILING RESULTS  ({N_BATCHES} batches, {total_games} games, {wall_elapsed:.1f}s)")
    print("=" * 70)

    print(f"\n  === TOP-LEVEL PHASES ===")
    print(f"\n  {'Phase':<35s} {'Total':>8s} {'Per-batch':>10s} {'% total':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8}")
    other = wall_elapsed - t_collect_total - t_gae_total - t_replay_total - t_loss_total - t_backward_total - t_optim_total
    for label, t in [
        ("Episode collection", t_collect_total),
        ("GAE computation", t_gae_total),
        ("Replay log-probs (fwd)", t_replay_total),
        ("compute_loss_flat", t_loss_total),
        ("loss.backward() + clip", t_backward_total),
        ("optimizer.step()", t_optim_total),
        ("Army gen + other", other),
    ]:
        pct = 100 * t / wall_elapsed
        print(f"  {label:<35s} {t:>7.1f}s {t/N_BATCHES:>9.2f}s {pct:>7.1f}%")

    # Sub-phase breakdown within episode collection
    measured = t_encode[0] + t_dcands[0] + t_dfeats[0] + t_infer[0] + t_precomp[0]
    unmeasured = max(0, t_collect_total - measured)

    print(f"\n  === WITHIN EPISODE COLLECTION ({t_collect_total:.1f}s) ===")
    print(f"  Inference calls: {n_act[0]} | encode calls: {n_encode[0]}")
    print(f"\n  {'Sub-phase':<35s} {'Total':>8s} {'Per-call':>10s} {'% collect':>10s}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10}")
    for label, t, per_n in [
        ("encode_state_tactical", t_encode[0], n_encode[0]),
        ("dest candidates (Dijkstra)", t_dcands[0], n_encode[0]),
        ("dest features", t_dfeats[0], n_encode[0]),
        ("batched model inference", t_infer[0], n_act[0]),
        ("precompute_damage", t_precomp[0], total_games),
        ("game mechanics + other", unmeasured, total_games),
    ]:
        pct = 100 * t / t_collect_total if t_collect_total > 0 else 0
        per = t / per_n * 1000 if per_n > 0 else 0
        print(f"  {label:<35s} {t:>7.1f}s {per:>8.1f}ms {pct:>9.1f}%")

    print(f"\n  --- Summary ---")
    sim_pct = 100 * t_collect_total / wall_elapsed
    ml_t = t_gae_total + t_replay_total + t_loss_total + t_backward_total + t_optim_total
    ml_pct = 100 * ml_t / wall_elapsed
    print(f"  Simulation:  {t_collect_total:.1f}s ({sim_pct:.1f}%)")
    print(f"  ML total:    {ml_t:.1f}s ({ml_pct:.1f}%)")
    print(f"  Games/sec:   {total_games / wall_elapsed:.2f}")
    if total_games > 0:
        print(f"  Avg activations/game: {n_act[0] / total_games:.1f}")
