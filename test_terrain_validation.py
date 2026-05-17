"""TERRAIN_SPEC.md §6 — terrain validation tests.

Covers §2.2: bounds, overlap detection, BLOCKING-must-be-IMPASSIBLE.
Run: python3 test_terrain_validation.py
"""
from __future__ import annotations

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from board import (
    Board, TerrainPiece, CoverType, MovementType, COLS, ROWS,
)


def expect_raises(fn, name: str) -> int:
    try:
        fn()
    except ValueError:
        print(f"  [OK]   {name} raised ValueError as expected")
        return 0
    print(f"  [FAIL] {name} did NOT raise ValueError")
    return 1


def test_bounds() -> int:
    fails = 0
    fails += expect_raises(
        lambda: Board().set_terrain([
            TerrainPiece(-1, 5, 0, 5, CoverType.SHELTERING, MovementType.OPEN)]),
        "negative x_lo")
    fails += expect_raises(
        lambda: Board().set_terrain([
            TerrainPiece(0, COLS, 0, 5, CoverType.SHELTERING, MovementType.OPEN)]),
        "x_hi == COLS (out of range)")
    fails += expect_raises(
        lambda: Board().set_terrain([
            TerrainPiece(0, 5, 0, ROWS, CoverType.SHELTERING, MovementType.OPEN)]),
        "y_hi == ROWS (out of range)")
    fails += expect_raises(
        lambda: Board().set_terrain([
            TerrainPiece(5, 4, 0, 5, CoverType.SHELTERING, MovementType.OPEN)]),
        "x_lo > x_hi")
    return fails


def test_overlap() -> int:
    fails = 0
    a = TerrainPiece(0, 5, 0, 5, CoverType.SHELTERING, MovementType.OPEN)
    b = TerrainPiece(3, 7, 3, 7, CoverType.OBSCURING, MovementType.OPEN)
    fails += expect_raises(
        lambda: Board().set_terrain([a, b]),
        "two overlapping pieces")
    # Edge-touching is allowed (not overlapping)
    c = TerrainPiece(0, 5, 0, 5, CoverType.SHELTERING, MovementType.OPEN)
    d = TerrainPiece(6, 9, 0, 5, CoverType.OBSCURING, MovementType.OPEN)
    try:
        Board().set_terrain([c, d])
        print("  [OK]   edge-adjacent pieces accepted")
    except ValueError as e:
        print(f"  [FAIL] edge-adjacent pieces rejected: {e}")
        fails += 1
    return fails


def test_blocking_must_be_impassible() -> int:
    fails = 0
    fails += expect_raises(
        lambda: Board().set_terrain([
            TerrainPiece(10, 12, 10, 12, CoverType.BLOCKING, MovementType.OPEN)]),
        "BLOCKING with OPEN movement")
    fails += expect_raises(
        lambda: Board().set_terrain([
            TerrainPiece(10, 12, 10, 12, CoverType.BLOCKING, MovementType.DIFFICULT)]),
        "BLOCKING with DIFFICULT movement")
    # Valid combination
    try:
        b = Board()
        b.set_terrain([TerrainPiece(10, 12, 10, 12, CoverType.BLOCKING,
                                     MovementType.IMPASSIBLE)])
        print("  [OK]   BLOCKING + IMPASSIBLE accepted")
    except ValueError as e:
        print(f"  [FAIL] BLOCKING + IMPASSIBLE rejected: {e}")
        fails += 1
    return fails


def test_derived_grids() -> int:
    fails = 0
    b = Board()
    b.set_terrain([
        TerrainPiece(10, 12, 10, 12, CoverType.OBSCURING, MovementType.DIFFICULT),
        TerrainPiece(20, 22, 20, 22, CoverType.BLOCKING, MovementType.IMPASSIBLE),
    ])
    # difficult_grid populated for piece 1
    if b.difficult_grid[10 * COLS + 10] == 1:
        print("  [OK]   difficult_grid set at piece-1 cells")
    else:
        print("  [FAIL] difficult_grid NOT set at piece-1 cells")
        fails += 1
    # impassible_grid populated for piece 2
    if b.impassible_grid[20 * COLS + 20] == 1:
        print("  [OK]   impassible_grid set at piece-2 cells")
    else:
        print("  [FAIL] impassible_grid NOT set at piece-2 cells")
        fails += 1
    # Open cell is in neither
    if (b.difficult_grid[5 * COLS + 5] == 0 and
            b.impassible_grid[5 * COLS + 5] == 0):
        print("  [OK]   open cell in neither grid")
    else:
        print("  [FAIL] open cell incorrectly flagged")
        fails += 1
    return fails


if __name__ == "__main__":
    total = 0
    print("\n--- test_bounds ---"); total += test_bounds()
    print("\n--- test_overlap ---"); total += test_overlap()
    print("\n--- test_blocking_must_be_impassible ---")
    total += test_blocking_must_be_impassible()
    print("\n--- test_derived_grids ---"); total += test_derived_grids()
    print(f"\n=== {total} failure(s) ===")
    sys.exit(1 if total else 0)
