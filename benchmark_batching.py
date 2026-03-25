"""Benchmark: compare batching strategies for per-activation tactical ML in evolution.

Tests:
  1. Forward pass time vs batch size (pure model overhead)
  2. Game logic throughput vs worker count (no ML)
  3. Full per-activation ML games: 1 worker (serial) vs N workers (current approach)
  4. Hybrid: N workers, each batching a chunk of games via generator+coordinator
"""
from __future__ import annotations

import copy
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import torch
from pathlib import Path
from ml_training import load_model_state_dict

MODEL_PATH = str(Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt")

# ---------------------------------------------------------------------------
# Helpers: generate fresh test data each time
# ---------------------------------------------------------------------------

def _make_pairs(n_armies=100, seed=42):
    """Generate armies and pair them into matchups. Returns fresh data each call."""
    from evolution import generate_random_army, resolve_army
    random.seed(seed)
    armies = [generate_random_army(mode="objectives") for _ in range(n_armies)]
    resolved = [resolve_army(a) for a in armies]
    pairs = []
    indices = list(range(n_armies))
    random.shuffle(indices)
    for k in range(0, n_armies - 1, 2):
        i, j = indices[k], indices[k + 1]
        pairs.append((armies[i], armies[j], resolved[i], resolved[j]))
    return pairs


def _chunk_pairs(pairs, nw):
    """Split pairs across nw workers."""
    chunks = [[] for _ in range(nw)]
    for idx, p in enumerate(pairs):
        chunks[idx % nw].append(p)
    return [(2, chunk) for chunk in chunks if chunk]  # 2 games per matchup


def _noop_warmup(_):
    """Dummy function to force worker process spawn."""
    pass


# ---------------------------------------------------------------------------
# Test 1: Forward pass time vs batch size
# ---------------------------------------------------------------------------

def bench_forward_pass():
    """Measure model forward pass time for different batch sizes."""
    from ml_model_tactical import TacticalModel
    from ml_features import TACTICAL_TOTAL_FEATURES, MAX_UNITS_PER_SIDE

    model = TacticalModel()
    model.load_state_dict(
        load_model_state_dict(MODEL_PATH),
        strict=False)
    model.eval()
    torch.set_num_threads(1)

    batch_sizes = [1, 2, 5, 10, 25, 50, 100]
    warmup = 50
    trials = 500

    print("=== Test 1: Forward pass time vs batch size ===")
    print(f"{'Batch':>6}  {'Total (ms)':>10}  {'Per-sample (ms)':>15}  {'Speedup vs 1':>13}")

    baseline_per_sample = None

    for bs in batch_sizes:
        x = torch.randn(bs, TACTICAL_TOTAL_FEATURES)
        mask = torch.ones(bs, MAX_UNITS_PER_SIDE, dtype=torch.bool)

        with torch.no_grad():
            for _ in range(warmup):
                model(x, mask)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(trials):
                model(x, mask)
        elapsed = time.perf_counter() - t0

        total_ms = 1000 * elapsed / trials
        per_sample_ms = total_ms / bs

        if baseline_per_sample is None:
            baseline_per_sample = per_sample_ms
            speedup = 1.0
        else:
            speedup = baseline_per_sample / per_sample_ms

        print(f"{bs:>6}  {total_ms:>10.3f}  {per_sample_ms:>15.4f}  {speedup:>13.1f}x")

    print()


# ---------------------------------------------------------------------------
# Test 2: Game logic throughput vs worker count (no ML)
# ---------------------------------------------------------------------------

def _run_games_no_ml(args):
    """Worker: play N games without ML."""
    n_games, population_pairs = args
    from game import simulate_game
    from evolution import _make_unit_states

    for army_i, army_j, res_i, res_j in population_pairs:
        for _ in range(n_games):
            sa = _make_unit_states(army_i, res_i, "A")
            sb = _make_unit_states(army_j, res_j, "B")
            simulate_game(res_i, res_j, mode="objectives", states_a=sa, states_b=sb)


def bench_game_logic_scaling():
    """Measure game logic wall time at different worker counts."""
    print("=== Test 2: Game logic wall time vs worker count (no ML) ===")

    worker_counts = [1, 2, 4, 8]
    print(f"Total games: 100 (50 matchups x 2)")
    print(f"{'Workers':>8}  {'Wall time (s)':>13}  {'Speedup':>8}")

    baseline_time = None

    for nw in worker_counts:
        pairs = _make_pairs(seed=nw * 1000)  # Different seed per run
        work = _chunk_pairs(pairs, nw)

        if nw == 1:
            t0 = time.perf_counter()
            _run_games_no_ml(work[0])
            wall = time.perf_counter() - t0
        else:
            with ProcessPoolExecutor(max_workers=nw) as pool:
                # Warmup: spawn workers + import
                list(pool.map(_noop_warmup, range(nw)))
                t0 = time.perf_counter()
                list(pool.map(_run_games_no_ml, work))
                wall = time.perf_counter() - t0

        if baseline_time is None:
            baseline_time = wall
        speedup = baseline_time / wall
        print(f"{nw:>8}  {wall:>13.3f}  {speedup:>8.2f}x")

    print()


# ---------------------------------------------------------------------------
# Test 3: Full per-activation ML games at different worker counts
# ---------------------------------------------------------------------------

_g_bench_model = None

def _init_bench_worker(model_path):
    global _g_bench_model
    from ml_model_tactical import TacticalModel
    _g_bench_model = TacticalModel()
    _g_bench_model.load_state_dict(
        load_model_state_dict(model_path),
        strict=False)
    _g_bench_model.eval()
    torch.set_num_threads(1)


def _run_games_ml(args):
    """Worker: play N matchups with per-activation ML."""
    n_games, population_pairs = args
    from game import simulate_game
    from evolution import _make_unit_states

    for army_i, army_j, res_i, res_j in population_pairs:
        for _ in range(n_games):
            sa = _make_unit_states(army_i, res_i, "A")
            sb = _make_unit_states(army_j, res_j, "B")
            simulate_game(res_i, res_j, mode="objectives", states_a=sa, states_b=sb,
                          ml_model_a=_g_bench_model, ml_model_b=_g_bench_model,
                          ml_batch_tactical=False)


def bench_full_ml_scaling():
    """Measure full per-activation ML game wall time at different worker counts."""
    print("=== Test 3: Full per-activation ML wall time vs worker count ===")

    worker_counts = [1, 2, 4, 8]
    print(f"Total games: 100 (50 matchups x 2)")
    print(f"{'Workers':>8}  {'Wall time (s)':>13}  {'Speedup':>8}")

    baseline_time = None

    for nw in worker_counts:
        pairs = _make_pairs(seed=nw * 1000)
        work = _chunk_pairs(pairs, nw)

        if nw == 1:
            _init_bench_worker(MODEL_PATH)
            t0 = time.perf_counter()
            _run_games_ml(work[0])
            wall = time.perf_counter() - t0
        else:
            with ProcessPoolExecutor(max_workers=nw,
                                     initializer=_init_bench_worker,
                                     initargs=(MODEL_PATH,)) as pool:
                # Warmup: spawn workers + load model
                list(pool.map(_noop_warmup, range(nw)))
                t0 = time.perf_counter()
                list(pool.map(_run_games_ml, work))
                wall = time.perf_counter() - t0

        if baseline_time is None:
            baseline_time = wall
        speedup = baseline_time / wall
        print(f"{nw:>8}  {wall:>13.3f}  {speedup:>8.2f}x")

    print()


# ---------------------------------------------------------------------------
# Test 4: Hybrid — generator + coordinator batching within worker
# ---------------------------------------------------------------------------

def _run_games_batched(args):
    """Worker: run games as generators, batch ML forward passes via coordinator."""
    n_games, population_pairs = args
    from game import simulate_game
    from evolution import _make_unit_states
    from ml_integration_tactical import batched_argmax_tactical

    # Create all game generators
    generators = []
    for army_i, army_j, res_i, res_j in population_pairs:
        for _ in range(n_games):
            sa = _make_unit_states(army_i, res_i, "A")
            sb = _make_unit_states(army_j, res_j, "B")
            gen = simulate_game(res_i, res_j, mode="objectives",
                                states_a=sa, states_b=sb,
                                ml_model_a=_g_bench_model,
                                ml_model_b=_g_bench_model,
                                ml_batch_tactical=False,
                                ml_coroutine_mode=True)
            generators.append(gen)

    # Prime all generators
    pending = []
    results = [None] * len(generators)
    for i, gen in enumerate(generators):
        try:
            req = next(gen)
            pending.append((i, req))
        except StopIteration as e:
            results[i] = e.value

    # Coordinator loop
    with torch.no_grad():
        while pending:
            reqs = [req for _, req in pending]
            batch_results = batched_argmax_tactical(_g_bench_model, reqs)

            next_pending = []
            for k, (i, _req) in enumerate(pending):
                try:
                    next_req = generators[i].send(batch_results[k])
                    next_pending.append((i, next_req))
                except StopIteration as e:
                    results[i] = e.value

            pending = next_pending


def bench_hybrid_batched():
    """Measure hybrid batched approach at different worker counts."""
    print("=== Test 4: Hybrid batched (generator+coordinator per worker) ===")

    worker_counts = [1, 2, 4, 8]
    print(f"Total games: 100 (50 matchups x 2)")
    print(f"{'Workers':>8}  {'Wall time (s)':>13}")

    for nw in worker_counts:
        pairs = _make_pairs(seed=nw * 1000)
        work = _chunk_pairs(pairs, nw)

        if nw == 1:
            _init_bench_worker(MODEL_PATH)
            t0 = time.perf_counter()
            _run_games_batched(work[0])
            wall = time.perf_counter() - t0
        else:
            with ProcessPoolExecutor(max_workers=nw,
                                     initializer=_init_bench_worker,
                                     initargs=(MODEL_PATH,)) as pool:
                # Warmup: spawn workers + load model
                list(pool.map(_noop_warmup, range(nw)))
                t0 = time.perf_counter()
                list(pool.map(_run_games_batched, work))
                wall = time.perf_counter() - t0

        print(f"{nw:>8}  {wall:>13.3f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"CPU count: {os.cpu_count()}")
    print(f"Torch threads: 1 (pinned per worker)\n")

    bench_forward_pass()
    bench_game_logic_scaling()
    bench_full_ml_scaling()
    bench_hybrid_batched()
