"""Wrapper around _fast_core C extension with automatic fallback to pure Python.

Usage from other modules:
    from fast_core import USE_C_EXT, fast_greedy_move, fast_find_kite_point, ...

The global USE_C_EXT can be toggled at runtime:
    import fast_core
    fast_core.USE_C_EXT = False   # force pure-Python path
"""
from __future__ import annotations

import array
import struct

try:
    import _fast_core
    _HAS_C_EXT = True
except ImportError:
    _HAS_C_EXT = False

# Runtime toggle — defaults to True if the C extension is available.
USE_C_EXT: bool = _HAS_C_EXT


def is_available() -> bool:
    """Return True if the compiled C extension is importable."""
    return _HAS_C_EXT


# ---------------------------------------------------------------------------
# Marshalling helpers
# ---------------------------------------------------------------------------

def _positions_to_int_bytes(positions: set[tuple[int, int]] | list[tuple[int, int]]) -> bytes:
    """Pack positions into a flat int32 buffer: [c0, r0, c1, r1, ...]."""
    if not positions:
        return b""
    arr = array.array("i")
    for c, r in positions:
        arr.append(c)
        arr.append(r)
    return arr.tobytes()


def _doubles_to_bytes(pairs: list[tuple[float, float]]) -> bytes:
    """Pack (x, y) pairs into a flat double buffer."""
    if not pairs:
        return b""
    arr = array.array("d")
    for x, y in pairs:
        arr.append(x)
        arr.append(y)
    return arr.tobytes()


# ---------------------------------------------------------------------------
# fast_greedy_move
# ---------------------------------------------------------------------------

def fast_greedy_move(
    start: tuple[int, int],
    goal: tuple[int, int],
    budget: float,
    occupancy: bytearray,
    enemy_positions: set[tuple[int, int]],
    is_charge: bool = False,
    flying: bool = False,
    exclusion_grid: bytearray | None = None,
    cols: int = 72,
    rows: int = 48,
) -> tuple[int, int]:
    """Drop-in replacement for movement._greedy_move using the C extension.

    Falls back to the pure-Python implementation when USE_C_EXT is False or
    the extension is not available.
    """
    if not USE_C_EXT or not _HAS_C_EXT:
        # Deferred import to avoid circular deps at module load time
        from movement import _greedy_move
        return _greedy_move(start, goal, budget,
                            type("B", (), {"occupancy": occupancy,
                                           "is_free": lambda s, c, r: 0 <= c < cols and 0 <= r < rows and not occupancy[r * cols + c],
                                           "is_occupied": lambda s, c, r: bool(occupancy[r * cols + c])})(),
                            enemy_positions, is_charge=is_charge, flying=flying,
                            exclusion_grid=exclusion_grid)

    col, row = start
    gc, gr = goal

    # Already adjacent to enemies?
    if exclusion_grid is not None:
        already_adjacent = bool(exclusion_grid[row * cols + col])
    else:
        already_adjacent = False
        for dc in range(-1, 2):
            for dr in range(-1, 2):
                if dc == 0 and dr == 0:
                    continue
                if (col + dc, row + dr) in enemy_positions:
                    already_adjacent = True
                    break
            if already_adjacent:
                break

    # Build exclusion grid if needed
    if exclusion_grid is None:
        from movement import build_exclusion_grid
        exclusion_grid = build_exclusion_grid(enemy_positions)

    enemy_bytes = _positions_to_int_bytes(enemy_positions)
    n_enemies = len(enemy_positions)

    return _fast_core.c_greedy_move(
        col, row, gc, gr, budget,
        bytes(occupancy), bytes(exclusion_grid), enemy_bytes,
        n_enemies, cols, rows,
        int(is_charge), int(flying), int(already_adjacent),
    )


# ---------------------------------------------------------------------------
# fast_pathfind_move (Dijkstra — replaces greedy)
# ---------------------------------------------------------------------------

def fast_pathfind_move(
    start: tuple[int, int],
    goal: tuple[int, int],
    budget: float,
    occupancy: bytearray,
    enemy_positions: set[tuple[int, int]],
    is_charge: bool = False,
    flying: bool = False,
    exclusion_grid: bytearray | None = None,
    cols: int = 72,
    rows: int = 48,
) -> tuple[int, int]:
    """Dijkstra-based pathfinding — drop-in replacement for fast_greedy_move.

    Explores all reachable cells within the movement budget, then returns
    the one closest to *goal* that is not occupied or enemy-held.
    """
    if not USE_C_EXT or not _HAS_C_EXT:
        from movement import _greedy_move
        return _greedy_move(start, goal, budget,
                            type("B", (), {"occupancy": occupancy,
                                           "is_free": lambda s, c, r: 0 <= c < cols and 0 <= r < rows and not occupancy[r * cols + c],
                                           "is_occupied": lambda s, c, r: bool(occupancy[r * cols + c])})(),
                            enemy_positions, is_charge=is_charge, flying=flying,
                            exclusion_grid=exclusion_grid)

    col, row = start
    gc, gr = goal

    # Already adjacent to enemies?
    if exclusion_grid is not None:
        already_adjacent = bool(exclusion_grid[row * cols + col])
    else:
        already_adjacent = False
        for dc in range(-1, 2):
            for dr in range(-1, 2):
                if dc == 0 and dr == 0:
                    continue
                if (col + dc, row + dr) in enemy_positions:
                    already_adjacent = True
                    break
            if already_adjacent:
                break

    # Build exclusion grid if needed
    if exclusion_grid is None:
        from movement import build_exclusion_grid
        exclusion_grid = build_exclusion_grid(enemy_positions)

    enemy_bytes = _positions_to_int_bytes(enemy_positions)
    n_enemies = len(enemy_positions)

    return _fast_core.c_pathfind_move(
        col, row, gc, gr, budget,
        bytes(occupancy), bytes(exclusion_grid), enemy_bytes,
        n_enemies, cols, rows,
        int(is_charge), int(flying), int(already_adjacent),
    )


# ---------------------------------------------------------------------------
# fast_find_kite_point
# ---------------------------------------------------------------------------

def fast_find_kite_point(
    cx: float, cy: float,
    tcx: float, tcy: float,
    enemy_centres: list[tuple[float, float]],
    move_budget: float,
    weapon_range: float | None,
) -> tuple[int, int]:
    """Drop-in replacement for ml_integration_tactical._find_kite_point's core loop."""
    if not USE_C_EXT or not _HAS_C_EXT or not enemy_centres:
        # Fallback: return rounded centre (caller should use original _find_kite_point)
        return (int(round(cx)), int(round(cy)))

    enemy_bytes = _doubles_to_bytes(enemy_centres)
    has_wr = 1 if weapon_range is not None else 0
    wr = weapon_range if weapon_range is not None else 0.0

    return _fast_core.c_find_kite_point(
        cx, cy, tcx, tcy,
        enemy_bytes, len(enemy_centres),
        move_budget, wr, has_wr,
    )


# ---------------------------------------------------------------------------
# fast_min_dists_sq
# ---------------------------------------------------------------------------

def fast_min_dists_sq(
    a_positions: list[tuple[int, int]],
    t_positions: list[tuple[int, int]],
) -> list[int]:
    """Drop-in replacement for combat._precompute_min_dists_sq."""
    if not USE_C_EXT or not _HAS_C_EXT:
        from combat import _precompute_min_dists_sq
        return _precompute_min_dists_sq(a_positions, t_positions)

    a_bytes = _positions_to_int_bytes(a_positions)
    t_bytes = _positions_to_int_bytes(t_positions)
    return _fast_core.c_min_dists_sq(a_bytes, len(a_positions), t_bytes, len(t_positions))


# ---------------------------------------------------------------------------
# fast_encode_distances
# ---------------------------------------------------------------------------

def fast_encode_distances(
    cx: float, cy: float,
    targets: list[tuple[float, float]],
    inv_diag: float,
) -> list[float]:
    """Drop-in replacement for the distance-computation loop in _encode_unit_into."""
    if not USE_C_EXT or not _HAS_C_EXT:
        import math
        return [math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2) * inv_diag
                for tx, ty in targets]

    target_bytes = _doubles_to_bytes(targets)
    return _fast_core.c_encode_distances(cx, cy, target_bytes, len(targets), inv_diag)
