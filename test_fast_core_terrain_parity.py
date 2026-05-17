"""TERRAIN_SPEC.md §6 — C/Python pathfinding parity on terrain-bearing boards.

Asserts that the C-accelerated Dijkstra (c_pathfind_move +
c_dijkstra_reachable_set) produces identical reachable cell sets and chosen
destinations as the pure-Python fallback across a bank of randomized layouts.
Run: python3 test_fast_core_terrain_parity.py
"""
from __future__ import annotations

import os
import random
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from board import Board, TerrainPiece, CoverType, MovementType, COLS, ROWS
import fast_core


def _random_layout(rng: random.Random, n_pieces: int) -> list[TerrainPiece]:
    pieces: list[TerrainPiece] = []
    used: set[tuple[int, int]] = set()
    for _ in range(n_pieces):
        for _try in range(80):
            x_lo = rng.randint(0, COLS - 6)
            y_lo = rng.randint(0, ROWS - 6)
            x_hi = x_lo + rng.randint(1, 5)
            y_hi = y_lo + rng.randint(1, 5)
            sqs = [(c, r) for c in range(x_lo, x_hi + 1)
                   for r in range(y_lo, y_hi + 1)]
            if any(s in used for s in sqs):
                continue
            mt = rng.choice([MovementType.OPEN, MovementType.DIFFICULT,
                             MovementType.IMPASSIBLE])
            if mt == MovementType.IMPASSIBLE:
                ct = rng.choice([CoverType.OBSCURING, CoverType.BLOCKING])
            else:
                ct = rng.choice([CoverType.SHELTERING, CoverType.OBSCURING])
            pieces.append(TerrainPiece(x_lo, x_hi, y_lo, y_hi, ct, mt))
            used.update(sqs)
            break
    return pieces


def _reachable_set(board: Board, start: tuple[int, int], budget: float,
                   flying: bool, strider: bool) -> set[tuple[int, int]]:
    imp = board.impassible_grid if board.terrain else None
    diff = board.difficult_grid if board.terrain else None
    arr = fast_core.fast_dijkstra_reachable_set(
        start, budget, board.occupancy, set(),
        flying=flying, strider=strider,
        impassible_grid=imp, difficult_grid=diff)
    return set(map(tuple, arr.tolist()))


def _pathfind(board: Board, start: tuple[int, int], goal: tuple[int, int],
              budget: float, flying: bool, strider: bool) -> tuple[int, int]:
    imp = board.impassible_grid if board.terrain else None
    diff = board.difficult_grid if board.terrain else None
    return fast_core.fast_pathfind_move(
        start, goal, budget, board.occupancy, set(),
        flying=flying, strider=strider,
        impassible_grid=imp, difficult_grid=diff)


def test_reachable_set_parity(n_trials: int = 30) -> int:
    fails = 0
    rng = random.Random(42)
    for trial in range(n_trials):
        pieces = _random_layout(rng, rng.randint(1, 6))
        b = Board()
        b.set_terrain(pieces, build_vis_cover=False)

        # Pick a non-impassible start
        for _ in range(50):
            start = (rng.randrange(COLS), rng.randrange(ROWS))
            if not b.impassible_grid[start[1] * COLS + start[0]]:
                break

        budget = rng.choice([6.0, 8.0, 12.0])
        flying = rng.choice([False, True])
        strider = rng.choice([False, True])

        fast_core.USE_C_EXT = True
        c_set = _reachable_set(b, start, budget, flying, strider)
        fast_core.USE_C_EXT = False
        py_set = _reachable_set(b, start, budget, flying, strider)
        fast_core.USE_C_EXT = True

        if c_set != py_set:
            extra_c = c_set - py_set
            extra_py = py_set - c_set
            print(f"  [FAIL] trial={trial} start={start} budget={budget} "
                  f"fly={flying} str={strider} | "
                  f"|C-Py|={len(extra_c)} |Py-C|={len(extra_py)}")
            fails += 1
        else:
            print(f"  [OK]   trial={trial} ({len(c_set)} cells reachable)")
    return fails


def test_destination_parity(n_trials: int = 20) -> int:
    fails = 0
    rng = random.Random(7)
    for trial in range(n_trials):
        pieces = _random_layout(rng, rng.randint(1, 5))
        b = Board()
        b.set_terrain(pieces, build_vis_cover=False)

        for _ in range(50):
            start = (rng.randrange(COLS), rng.randrange(ROWS))
            if not b.impassible_grid[start[1] * COLS + start[0]]:
                break
        for _ in range(50):
            goal = (rng.randrange(COLS), rng.randrange(ROWS))
            if not b.impassible_grid[goal[1] * COLS + goal[0]]:
                break

        budget = rng.choice([6.0, 8.0, 12.0])
        flying = rng.choice([False, True])
        strider = rng.choice([False, True])

        fast_core.USE_C_EXT = True
        c_dst = _pathfind(b, start, goal, budget, flying, strider)
        fast_core.USE_C_EXT = False
        py_dst = _pathfind(b, start, goal, budget, flying, strider)
        fast_core.USE_C_EXT = True

        if c_dst != py_dst:
            print(f"  [FAIL] trial={trial} start={start} goal={goal} "
                  f"budget={budget} fly={flying} str={strider} | "
                  f"C={c_dst} Py={py_dst}")
            fails += 1
        else:
            print(f"  [OK]   trial={trial} dst={c_dst}")
    return fails


if __name__ == "__main__":
    total = 0
    print("\n--- test_reachable_set_parity ---")
    total += test_reachable_set_parity()
    print("\n--- test_destination_parity ---")
    total += test_destination_parity()
    print(f"\n=== {total} failure(s) ===")
    sys.exit(1 if total else 0)
