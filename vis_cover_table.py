"""Precomputed visibility/cover table — TERRAIN_SPEC.md §5.6.

For an immutable terrain layout, the per-(shooter_sq, target_sq) result of
visibility and cover predicates is deterministic. We precompute the whole
COLS x ROWS x COLS x ROWS table once at deployment time and cache it on disk
keyed on a terrain layout hash.

Each entry is one byte:
  OPEN   = 0  — target visible, no cover bonus
  COVER  = 1  — target visible, +1 defense from cover
  NO_LOS = 2  — target not visible

Indexed by *ordered* square pairs: a shooter inside an OBSCURING piece imposes
cover on its target via §4.3's wholly-within rule, but the reverse arrangement
does not. The table exposes ordered pairs even though the underlying line
geometry is symmetric.
"""
from __future__ import annotations

import os

import numpy as np

from board import COLS, ROWS, TerrainPiece, terrain_layout_hash
from terrain_los import is_visible, is_in_cover

OPEN = 0
COVER = 1
NO_LOS = 2

_TOTAL = COLS * ROWS  # 3456 on 72x48


def _flat(sq: tuple[int, int]) -> int:
    return sq[1] * COLS + sq[0]


def lookup(table: np.ndarray, shooter_sq: tuple[int, int],
           target_sq: tuple[int, int]) -> int:
    """O(1) table lookup. ``table`` is the (TOTAL, TOTAL) uint8 array
    returned by :func:`build`."""
    return int(table[_flat(shooter_sq), _flat(target_sq)])


def build(terrain: list[TerrainPiece]) -> np.ndarray:
    """Build the full (TOTAL, TOTAL) uint8 table from a terrain layout.

    Cost: ~COLS²·ROWS² ordered-pair evaluations, each O(|terrain|) AABB +
    line-vs-rectangle tests. The Python reference is correct but slow on
    non-trivial layouts; a C-accelerated equivalent in ``_fast_core`` is
    listed as a follow-up.
    """
    table = np.empty((_TOTAL, _TOTAL), dtype=np.uint8)
    if not terrain:
        table.fill(OPEN)
        return table
    # Live geometric compute. Iterate in row-major over both squares so cache
    # behavior is predictable; relies on the fact that is_visible/is_in_cover
    # are pure functions of squares + terrain.
    for sy_idx in range(_TOTAL):
        sy = (sy_idx % COLS, sy_idx // COLS)
        for ty_idx in range(_TOTAL):
            ty = (ty_idx % COLS, ty_idx // COLS)
            if not is_visible(sy, ty, terrain):
                table[sy_idx, ty_idx] = NO_LOS
            elif is_in_cover(sy, ty, terrain):
                table[sy_idx, ty_idx] = COVER
            else:
                table[sy_idx, ty_idx] = OPEN
    return table


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(__file__), "terrain_cache")


def _cache_path(layout_hash: str) -> str:
    return os.path.join(CACHE_DIR, f"vis_cover_{layout_hash}.npy")


def build_or_load(terrain: list[TerrainPiece],
                  use_disk_cache: bool = True) -> np.ndarray:
    """Return the table for ``terrain``; cached on disk between runs.

    The cache key is a stable hash of the terrain layout (sorted-tuple
    SHA-256). Repeated games on the same layout pay zero rebuild cost.
    """
    if not terrain or not use_disk_cache:
        return build(terrain)
    layout_hash = terrain_layout_hash(terrain)
    path = _cache_path(layout_hash)
    if os.path.exists(path):
        try:
            arr = np.load(path)
            if arr.shape == (_TOTAL, _TOTAL) and arr.dtype == np.uint8:
                return arr
        except (OSError, ValueError):
            pass  # corrupted cache — rebuild
    table = build(terrain)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, table)
    except OSError:
        pass  # cache write failure is non-fatal
    return table
