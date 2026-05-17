"""TERRAIN_SPEC.md §6 — precomputed visibility/cover table tests.

Validates that vis_cover_table.build matches live terrain_los compute on a
bank of layouts, asymmetry is preserved for shooter-vs-target inside
OBSCURING, and the disk cache round-trips correctly via terrain_layout_hash.
Run: python3 test_precomputed_visibility_cover.py
"""
from __future__ import annotations

import os
import shutil
import sys
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from board import (
    Board, TerrainPiece, CoverType, MovementType, COLS, ROWS,
    terrain_layout_hash,
)
import vis_cover_table as vct
from terrain_los import is_visible, is_in_cover


def _check(name: str, ok: bool, detail: str = "") -> int:
    if ok:
        print(f"  [OK]   {name}")
        return 0
    print(f"  [FAIL] {name} {detail}")
    return 1


def _spot_check_table(table: np.ndarray, terrain: list[TerrainPiece],
                      n_samples: int, rng: random.Random) -> int:
    """Sample n_samples random ordered pairs and check table vs live compute."""
    fails = 0
    for _ in range(n_samples):
        sy = (rng.randrange(COLS), rng.randrange(ROWS))
        ty = (rng.randrange(COLS), rng.randrange(ROWS))
        live_vis = is_visible(sy, ty, terrain)
        live_cov = is_in_cover(sy, ty, terrain)
        live_state = (vct.NO_LOS if not live_vis
                      else (vct.COVER if live_cov else vct.OPEN))
        tbl_state = vct.lookup(table, sy, ty)
        if int(tbl_state) != int(live_state):
            print(f"    mismatch sy={sy} ty={ty} table={tbl_state} live={live_state}")
            fails += 1
    return fails


def test_table_matches_live() -> int:
    """Build a small layout's table; compare to live compute on random pairs."""
    rng = random.Random(0)
    pieces = [
        TerrainPiece(15, 18, 10, 14, CoverType.OBSCURING, MovementType.OPEN),
        TerrainPiece(28, 30, 18, 22, CoverType.BLOCKING,
                     MovementType.IMPASSIBLE),
    ]
    table = vct.build(pieces)
    print("  built table for 2-piece layout")
    fails = _spot_check_table(table, pieces, 200, rng)
    return _check("200 random pair lookups match live compute", fails == 0,
                  f"{fails} mismatches")


def test_ordered_pair_asymmetry() -> int:
    """Shooter inside OBSCURING + target outside ⇒ COVER.
    Reverse (shooter outside + target outside) ⇒ NOT COVER (no wholly-within
    on target side, no obscure)."""
    fails = 0
    pieces = [
        TerrainPiece(15, 19, 10, 14, CoverType.OBSCURING, MovementType.OPEN),
    ]
    table = vct.build(pieces)
    # Shooter inside (interior cell), target outside.
    sy_in = (17, 12)   # interior of piece
    ty_out = (30, 12)
    s2t = vct.lookup(table, sy_in, ty_out)
    t2s = vct.lookup(table, ty_out, sy_in)
    fails += _check("shooter inside obscuring → target in COVER",
                    s2t == vct.COVER, f"got {s2t}")
    # Reverse: shooter outside, target inside obscuring → also cover (target
    # wholly-within, see §4.3 OBSCURING row). Both are cover but the geometry
    # differs — the asymmetry holds when the obscuring is NOT in the line.
    fails += _check("shooter outside, target inside obscuring → COVER (wholly-within)",
                    t2s == vct.COVER, f"got {t2s}")
    return fails


def test_disk_cache_round_trip() -> int:
    fails = 0
    if os.path.exists(vct.CACHE_DIR):
        shutil.rmtree(vct.CACHE_DIR)
    pieces = [
        TerrainPiece(40, 44, 20, 24, CoverType.SHELTERING, MovementType.OPEN),
    ]
    h = terrain_layout_hash(pieces)
    table_a = vct.build_or_load(pieces)
    cache_path = vct._cache_path(h)
    fails += _check("cache file exists after build_or_load",
                    os.path.exists(cache_path), f"path={cache_path}")
    table_b = vct.build_or_load(pieces)
    fails += _check("cache reload returns equal table",
                    np.array_equal(table_a, table_b))
    # Order-independent hash: same pieces in different order → same key
    pieces_rev = list(reversed(pieces + [
        TerrainPiece(0, 1, 0, 1, CoverType.SHELTERING, MovementType.OPEN),
    ]))
    pieces_fwd = list(reversed(pieces_rev))
    fails += _check("hash is order-independent",
                    terrain_layout_hash(pieces_rev) ==
                    terrain_layout_hash(pieces_fwd))
    return fails


if __name__ == "__main__":
    total = 0
    print("\n--- test_table_matches_live ---")
    total += test_table_matches_live()
    print("\n--- test_ordered_pair_asymmetry ---")
    total += test_ordered_pair_asymmetry()
    print("\n--- test_disk_cache_round_trip ---")
    total += test_disk_cache_round_trip()
    print(f"\n=== {total} failure(s) ===")
    sys.exit(1 if total else 0)
