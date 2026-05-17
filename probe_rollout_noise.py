"""How noisy is an 8-rollout Q estimate vs a 100-rollout reference?

Per state: sample K stratified candidates, run M_TOTAL=108 independent
rollouts each, then split disjointly into 100 "truth" and 8 "noisy". Compare
rankings via Spearman, Kendall, top-K overlap, and a top-quartile Spearman
(where rank order matters most for argmax/topK use).

Parallelism: state-level. Each of N_STATES workers collects its own
decision state, runs all rollouts, returns metrics. Each worker pins
torch.set_num_threads(1) to avoid BLAS over-subscription on a 6-core box.
"""
from __future__ import annotations

import multiprocessing as mp
import random
import sys
import time

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr


CANDIDATES_PER_STATE = 500
M_TOTAL = 108
M_TRUTH = 100
M_NOISY = 8
SEED = 42
GAMES_PER_COLLECTION_BATCH = 4

# Rounds to sample states from, with how many states per round. The previous
# unfiltered run incidentally drew all states from round 3-4; this lets us
# look at early-game noise where more units are alive.
TARGET_ROUNDS = {1: 2, 2: 2}


def _compute_Q_samples(model, cand, snap, ctx, current_is_a, round_num,
                       m_rollouts):
    """Like probe.compute_Q but returns the per-rollout V samples instead of
    the mean — caller can split/subset them."""
    # Imported lazily inside worker so spawn'd children don't double-import.
    from ml_features import MAX_UNITS_PER_SIDE
    from ml_integration_tactical import execute_decoded_decision
    from ml_planning import _execute_activation, restore_game_state, simulate_forward

    units_a, units_b, board = ctx.units_a, ctx.units_b, ctx.board
    samples = np.empty(m_rollouts, dtype=np.float32)

    for i in range(m_rollouts):
        restore_game_state(snap, units_a, units_b, board)
        my_units = units_a if current_is_a else units_b
        opp_units = units_b if current_is_a else units_a
        unit = my_units[cand.unit_idx]
        dest = (cand.dest_col, cand.dest_row) if cand.dest_col is not None else None
        target_ranking = [cand.shoot_target_idx] + [
            j for j in range(MAX_UNITS_PER_SIDE) if j != cand.shoot_target_idx
        ]
        action, goal, charge_target, reason = execute_decoded_decision(
            unit, opp_units, cand.move_type, dest,
            cand.charge_target_idx, cand.shoot_target_idx,
            is_advance_reachable=cand.advance_reachable,
        )
        opp_wiped = _execute_activation(
            unit, action, goal, charge_target, reason, target_ranking,
            my_units, opp_units, board, ctx.mode,
        )
        if opp_wiped:
            v_a = 1.0 if current_is_a else -1.0
        elif not any(u.models_alive > 0 for u in my_units):
            v_a = -1.0 if current_is_a else 1.0
        else:
            v_a = simulate_forward(
                units_a, units_b, board, model,
                n_activations=1, current_is_a=not current_is_a,
                round_num=round_num, mode=ctx.mode,
                fr_a=ctx.fr_a, fm_a=ctx.fm_a,
                fr_b=ctx.fr_b, fm_b=ctx.fm_b,
                pts_a=ctx.pts_a, pts_b=ctx.pts_b,
            )
            if v_a is None:
                v_a = 0.0
        v_persp = v_a if current_is_a else -v_a
        samples[i] = v_persp

    restore_game_state(snap, units_a, units_b, board)
    return samples


def _topk_overlap(truth, noisy, k):
    k = min(k, len(truth))
    truth_top = set(np.argsort(-truth)[:k].tolist())
    noisy_top = set(np.argsort(-noisy)[:k].tolist())
    return len(truth_top & noisy_top) / k


def _state_worker(args):
    """One state's worth of work: collect a state, sample candidates, label,
    return per-state metrics dict."""
    worker_id, seed, target_round = args

    # CRITICAL: prevent BLAS over-subscription. Each rollout is a single
    # batch=1 forward pass; multi-threaded MKL/OpenBLAS makes it slower
    # *and* hammers all cores. The dataset workers do this for the same reason.
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    rng = random.Random(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    import probe_identifier_premise as probe
    from ml_planning import restore_game_state
    from ml_training.identifier_dataset import stratified_sample_candidates

    model = probe.load_frozen_model()

    # Find one good decision state (one that has at least 1 candidate)
    chosen = None
    games_so_far = 0
    while chosen is None:
        states, contexts = probe.collect_decision_states(
            model, n_games=GAMES_PER_COLLECTION_BATCH,
        )
        rng.shuffle(states)
        for st in states:
            if st.round_num != target_round:
                continue
            ctx = contexts[st.game_idx]
            restore_game_state(st.snap, ctx.units_a, ctx.units_b, ctx.board)
            my_units = ctx.units_a if st.current_is_a else ctx.units_b
            opp_units = ctx.units_b if st.current_is_a else ctx.units_a
            player = "A" if st.current_is_a else "B"
            test_cands = stratified_sample_candidates(
                my_units, opp_units, ctx.board, player,
                K=CANDIDATES_PER_STATE, rng=random.Random(seed),
            )
            if test_cands:
                chosen = (st, ctx, test_cands)
                break
        games_so_far += GAMES_PER_COLLECTION_BATCH
        if games_so_far > 100:
            return dict(worker_id=worker_id, error="no candidates after 100 games")

    st, ctx, cands = chosen
    print(f"[w{worker_id}/r{target_round}] state ready: {len(cands)} candidates "
          f"(round={st.round_num}, side={'A' if st.current_is_a else 'B'})",
          flush=True)

    t0 = time.time()
    all_samples = np.empty((len(cands), M_TOTAL), dtype=np.float32)
    for c_idx, cand in enumerate(cands):
        all_samples[c_idx] = _compute_Q_samples(
            model, cand, st.snap, ctx,
            current_is_a=st.current_is_a,
            round_num=st.round_num,
            m_rollouts=M_TOTAL,
        )
        if (c_idx + 1) % 100 == 0:
            rate = (c_idx + 1) / (time.time() - t0)
            eta = (len(cands) - c_idx - 1) / max(rate, 1e-6)
            print(f"[w{worker_id}/r{target_round}]   cand {c_idx+1}/{len(cands)} "
                  f"({rate:.2f} cand/s, eta {eta:.0f}s)", flush=True)

    truth_q = all_samples[:, :M_TRUTH].mean(axis=1)
    noisy_q = all_samples[:, M_TRUTH:M_TRUTH + M_NOISY].mean(axis=1)

    rho, _ = spearmanr(truth_q, noisy_q)
    tau, _ = kendalltau(truth_q, noisy_q)
    top10 = _topk_overlap(truth_q, noisy_q, 10)
    top50 = _topk_overlap(truth_q, noisy_q, 50)
    q1_idx = np.argsort(-truth_q)[: max(2, len(cands) // 4)]
    rho_q1, _ = spearmanr(truth_q[q1_idx], noisy_q[q1_idx])

    elapsed = time.time() - t0
    print(f"[w{worker_id}/r{target_round}] DONE in {elapsed:.0f}s: "
          f"rho={rho:.3f} tau={tau:.3f} "
          f"top10={top10:.2f} top50={top50:.2f} rho_top25%={rho_q1:.3f} "
          f"truth_std={truth_q.std():.3f}", flush=True)

    return dict(
        worker_id=worker_id, target_round=target_round, n_cands=len(cands),
        rho=float(rho), tau=float(tau),
        top10=float(top10), top50=float(top50), rho_top25=float(rho_q1),
        truth_std=float(truth_q.std()), truth_mean=float(truth_q.mean()),
        elapsed=elapsed,
    )


def main():
    args = []
    for round_num, n_states in TARGET_ROUNDS.items():
        for _ in range(n_states):
            wid = len(args)
            args.append((wid, SEED + wid * 10_000 + 1, round_num))
    print(f"[noise] launching {len(args)} state-workers "
          f"(1 BLAS thread each), targets={TARGET_ROUNDS}")
    ctx = mp.get_context("spawn")
    overall_t0 = time.time()
    with ctx.Pool(processes=len(args)) as pool:
        results = pool.map(_state_worker, args)

    print("\n" + "=" * 70)
    print(f"Summary across {len(results)} states "
          f"(wall {time.time() - overall_t0:.0f}s)")
    print("=" * 70)
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid results.")
        return
    for k in ["rho", "tau", "top10", "top50", "rho_top25"]:
        vals = np.array([r[k] for r in valid])
        print(f"  {k:12s} mean={vals.mean():+.3f} "
              f"min={vals.min():+.3f} max={vals.max():+.3f}")
    print("\n  per-state truth-Q distribution:")
    for r in valid:
        print(f"    w{r['worker_id']}/r{r['target_round']}: "
              f"mean={r['truth_mean']:+.3f} "
              f"std={r['truth_std']:.3f} (low std => "
              f"candidates indistinguishable, ranking is ~noise)")


if __name__ == "__main__":
    main()
