"""TERRAIN_SPEC.md §6 — ml_features terrain encoding tests.

Validates terrain plane channel layout and DEST per-hex terrain features.
Run: python3 test_ml_features_terrain.py
"""
from __future__ import annotations

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from board import Board, TerrainPiece, CoverType, MovementType, COLS, ROWS


def _check(name: str, ok: bool, detail: str = "") -> int:
    if ok:
        print(f"  [OK]   {name}")
        return 0
    print(f"  [FAIL] {name} {detail}")
    return 1


def test_terrain_planes_shape_and_onehot() -> int:
    fails = 0
    try:
        from ml_features import encode_terrain_planes, TERRAIN_CHANNELS
    except ImportError as e:
        return _check("ml_features importable", False, str(e))
    b = Board()
    b.set_terrain([
        TerrainPiece(10, 12, 10, 12, CoverType.SHELTERING, MovementType.OPEN),
        TerrainPiece(20, 22, 20, 22, CoverType.BLOCKING,
                     MovementType.IMPASSIBLE),
        TerrainPiece(30, 33, 30, 33, CoverType.OBSCURING,
                     MovementType.DIFFICULT),
    ], build_vis_cover=False)
    planes = encode_terrain_planes(b)
    fails += _check("planes shape (K, ROWS, COLS)",
                    planes.shape == (TERRAIN_CHANNELS, ROWS, COLS),
                    f"got {planes.shape}")
    # Each cell hot in exactly one channel
    sums = planes.sum(axis=0)
    fails += _check("each cell one-hot (sum across channels == 1)",
                    bool((sums == 1).all()),
                    f"min/max sum = {sums.min()}/{sums.max()}")
    # Total mass equals number of cells
    fails += _check("total mass equals COLS*ROWS",
                    int(planes.sum()) == COLS * ROWS,
                    f"got {planes.sum()}")
    # No-terrain board → all in channel 0
    b0 = Board()
    p0 = encode_terrain_planes(b0)
    fails += _check("empty board → all open (channel 0)",
                    bool((p0[0] == 1).all()) and bool((p0[1:] == 0).all()))
    return fails


def test_dest_feature_dim_includes_terrain() -> int:
    try:
        from ml_features import DEST_FEATURE_DIM
    except ImportError as e:
        return _check("ml_features importable", False, str(e))
    return _check("DEST_FEATURE_DIM = 82 (76 base + 6 terrain)",
                  DEST_FEATURE_DIM == 82, f"got {DEST_FEATURE_DIM}")


def test_dest_features_populated_with_board() -> int:
    fails = 0
    try:
        from ml_integration_tactical import compute_destination_features
        from ml_features import DEST_FEATURE_DIM, MAX_UNITS_PER_SIDE
    except ImportError as e:
        return _check("ml_integration_tactical importable", False, str(e))
    # Build a minimal candidate set: 3 candidates, two on terrain, one open.
    candidates = np.zeros((10, 2), dtype=np.int32)
    candidates[0] = (5, 5)     # open
    candidates[1] = (11, 11)   # SHELTERING/OPEN
    candidates[2] = (21, 21)   # BLOCKING/IMPASSIBLE (model can't end here; still
                                # asks for the feature)
    mask = np.zeros(10, dtype=bool)
    mask[:3] = True
    fr_match = np.zeros((1, MAX_UNITS_PER_SIDE, 7), dtype=np.float32)
    er_match = np.zeros((1, MAX_UNITS_PER_SIDE, 7), dtype=np.float32)
    me_match = np.zeros((1, MAX_UNITS_PER_SIDE), dtype=np.float32)

    b = Board()
    b.set_terrain([
        TerrainPiece(10, 12, 10, 12, CoverType.SHELTERING, MovementType.OPEN),
        TerrainPiece(20, 22, 20, 22, CoverType.BLOCKING,
                     MovementType.IMPASSIBLE),
    ], build_vis_cover=False)

    feats = compute_destination_features(
        candidates, mask, unit=None, unit_slot=0, player="A",
        enemy_units=[],
        enemy_alive_mask=np.zeros(MAX_UNITS_PER_SIDE, dtype=bool),
        friendly_ranged_matchups=fr_match,
        enemy_ranged_matchups=er_match,
        melee_matchups=me_match,
        move_budget=12.0,
        unit_centre=(36.0, 24.0),
        unit_alive_frac=1.0,
        advance_reachable=mask,
        board=b,
    )

    fails += _check("feats shape includes terrain block",
                    feats.shape[1] == DEST_FEATURE_DIM,
                    f"got width={feats.shape[1]}")
    # Cell 0 (open): movement OPEN one-hot at [79], no cover bits.
    fails += _check("open cell → movement OPEN bit set",
                    feats[0, 79] == 1.0)
    fails += _check("open cell → no cover bits",
                    feats[0, 76] == 0 and feats[0, 77] == 0 and feats[0, 78] == 0)
    # Cell 1 (SHELTERING/OPEN)
    fails += _check("sheltering cell → cover bit [76] set",
                    feats[1, 76] == 1.0)
    fails += _check("sheltering cell → movement OPEN bit set",
                    feats[1, 79] == 1.0)
    # Cell 2 (BLOCKING/IMPASSIBLE)
    fails += _check("blocking cell → cover bit [78] set",
                    feats[2, 78] == 1.0)
    fails += _check("blocking cell → movement IMPASSIBLE bit set",
                    feats[2, 81] == 1.0)
    return fails


if __name__ == "__main__":
    total = 0
    print("\n--- test_terrain_planes_shape_and_onehot ---")
    total += test_terrain_planes_shape_and_onehot()
    print("\n--- test_dest_feature_dim_includes_terrain ---")
    total += test_dest_feature_dim_includes_terrain()
    print("\n--- test_dest_features_populated_with_board ---")
    total += test_dest_features_populated_with_board()
    print(f"\n=== {total} failure(s) ===")
    sys.exit(1 if total else 0)
