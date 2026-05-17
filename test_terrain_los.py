"""TERRAIN_SPEC.md §6 — line-of-sight predicate tests.

Covers §4.2/§4.3: per-piece obscure boolean, combined-blocking visibility,
sheltering doesn't pass cover through, BLOCKING obscure → cover when target
still visible.
Run: python3 test_terrain_los.py
"""
from __future__ import annotations

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from board import TerrainPiece, CoverType, MovementType
from terrain_los import (
    line_crosses_terrain, obscured_by, is_visible, is_in_cover,
    shooter_cover_state,
)


def _check(name: str, ok: bool, detail: str = "") -> int:
    if ok:
        print(f"  [OK]   {name}")
        return 0
    print(f"  [FAIL] {name} {detail}")
    return 1


def test_line_crosses_basic() -> int:
    fails = 0
    X = TerrainPiece(10, 13, 10, 13, CoverType.BLOCKING, MovementType.IMPASSIBLE)
    fails += _check("line through interior crosses",
                    line_crosses_terrain((5.0, 11.5), (20.0, 11.5), X))
    fails += _check("line on top edge does NOT cross",
                    not line_crosses_terrain((5.0, 10.0), (20.0, 10.0), X))
    fails += _check("line missing piece does NOT cross",
                    not line_crosses_terrain((5.0, 5.0), (20.0, 5.0), X))
    return fails


def test_per_piece_obscure() -> int:
    X = TerrainPiece(10, 13, 10, 13, CoverType.BLOCKING, MovementType.IMPASSIBLE)
    obs = obscured_by((5, 11), (20, 11), X)
    return _check("target obscured by intervening piece", obs)


def test_combined_visibility_single_blocking() -> int:
    X = TerrainPiece(10, 13, 10, 13, CoverType.BLOCKING, MovementType.IMPASSIBLE)
    return _check("single full-block piece breaks visibility",
                  not is_visible((5, 11), (20, 11), [X]))


def test_combined_visibility_gap() -> int:
    # Two BLOCKING pieces with a clear gap in between — visibility holds.
    A = TerrainPiece(10, 13, 8, 9, CoverType.BLOCKING, MovementType.IMPASSIBLE)
    B = TerrainPiece(10, 13, 13, 14, CoverType.BLOCKING, MovementType.IMPASSIBLE)
    # Shooter (5, 11) to target (20, 11) — center-line passes through y=11
    # which is between the two pieces (y=10..12 is open).
    return _check("two BLOCKING pieces with a gap remain visible",
                  is_visible((5, 11), (20, 11), [A, B]))


def test_sheltering_does_not_pass_cover_through() -> int:
    # Shooter outside, target outside, sheltering piece in between → no cover.
    X = TerrainPiece(15, 18, 10, 13, CoverType.SHELTERING, MovementType.OPEN)
    return _check("sheltering doesn't grant cover via passing-through",
                  not is_in_cover((5, 11), (25, 11), [X]))


def test_sheltering_grants_cover_to_models_inside() -> int:
    X = TerrainPiece(15, 18, 10, 13, CoverType.SHELTERING, MovementType.OPEN)
    return _check("sheltering grants cover when target inside",
                  is_in_cover((5, 11), (16, 11), [X]))


def test_blocking_obscure_implies_cover() -> int:
    # When a BLOCKING piece per-piece-obscures the target, is_in_cover should
    # return True. (Visibility may or may not hold — that gate is applied at
    # the unit level in resolve_shooting; the predicate itself doesn't care.)
    A = TerrainPiece(15, 17, 10, 13, CoverType.BLOCKING, MovementType.IMPASSIBLE)
    return _check("BLOCKING obscure ⇒ in cover (per §4.3 table)",
                  is_in_cover((5, 12), (25, 12), [A]))


def test_obscuring_interior_imposes_cover_outward() -> int:
    # Shooter inside an OBSCURING piece (interior, not on edge) shooting out:
    # outbound segments cross the piece's interior → target in cover.
    X = TerrainPiece(10, 14, 10, 14, CoverType.OBSCURING, MovementType.OPEN)
    return _check("shooter in interior of obscuring imposes cover",
                  is_in_cover((12, 12), (25, 12), [X]))


def test_shooter_cover_state_majority() -> int:
    fails = 0
    X = TerrainPiece(15, 18, 10, 13, CoverType.SHELTERING, MovementType.OPEN)
    targets_inside = [(16, 11), (17, 12), (16, 13), (15, 11)]
    n_bad, vis = shooter_cover_state((5, 11), targets_inside, [X])
    fails += _check("all 4 targets inside sheltering → 4 in cover",
                    n_bad == 4, f"n_bad={n_bad}")
    fails += _check("all 4 targets visible (sheltering doesn't break LOS)",
                    all(vis), f"vis={vis}")
    return fails


if __name__ == "__main__":
    total = 0
    print("\n--- test_line_crosses_basic ---"); total += test_line_crosses_basic()
    print("\n--- test_per_piece_obscure ---"); total += test_per_piece_obscure()
    print("\n--- test_combined_visibility_single_blocking ---")
    total += test_combined_visibility_single_blocking()
    print("\n--- test_combined_visibility_gap ---")
    total += test_combined_visibility_gap()
    print("\n--- test_sheltering_does_not_pass_cover_through ---")
    total += test_sheltering_does_not_pass_cover_through()
    print("\n--- test_sheltering_grants_cover_to_models_inside ---")
    total += test_sheltering_grants_cover_to_models_inside()
    print("\n--- test_blocking_obscure_implies_cover ---")
    total += test_blocking_obscure_implies_cover()
    print("\n--- test_obscuring_interior_imposes_cover_outward ---")
    total += test_obscuring_interior_imposes_cover_outward()
    print("\n--- test_shooter_cover_state_majority ---")
    total += test_shooter_cover_state_majority()
    print(f"\n=== {total} failure(s) ===")
    sys.exit(1 if total else 0)
