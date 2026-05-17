"""Wrapper around _fast_core C extension with automatic fallback to pure Python.

Usage from other modules:
    from fast_core import USE_C_EXT, fast_greedy_move, fast_find_kite_point, ...

The global USE_C_EXT can be toggled at runtime:
    import fast_core
    fast_core.USE_C_EXT = False   # force pure-Python path
"""
from __future__ import annotations

import array
import heapq
import struct

import numpy as np

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

_EMPTY_TERRAIN_GRID: bytes | None = None


def _empty_terrain_grid(cols: int, rows: int) -> bytes:
    """Return a cached zero-filled terrain grid for the standard board."""
    global _EMPTY_TERRAIN_GRID
    if _EMPTY_TERRAIN_GRID is None or len(_EMPTY_TERRAIN_GRID) != cols * rows:
        _EMPTY_TERRAIN_GRID = bytes(cols * rows)
    return _EMPTY_TERRAIN_GRID


def fast_pathfind_move(
    start: tuple[int, int],
    goal: tuple[int, int],
    budget: float,
    occupancy: bytearray,
    enemy_positions: set[tuple[int, int]],
    is_charge: bool = False,
    flying: bool = False,
    strider: bool = False,
    exclusion_grid: bytearray | None = None,
    cols: int = 72,
    rows: int = 48,
    enemy_bytes: bytes | None = None,
    impassible_grid: bytearray | bytes | None = None,
    difficult_grid: bytearray | bytes | None = None,
) -> tuple[int, int]:
    """Dijkstra-based pathfinding — drop-in replacement for fast_greedy_move.

    Explores all reachable cells within the movement budget, then returns
    the one closest to *goal* that is not occupied, enemy-held, or in
    impassible terrain. Difficult terrain caps total path length at 6.0"
    (unless the unit is *flying* or *strider* — see TERRAIN_SPEC.md §3).

    ``enemy_bytes`` is an optional pre-packed int32 buffer of enemy positions
    (as produced by ``_positions_to_int_bytes``). Passing it in avoids
    re-packing on every model call when the caller walks many models of
    the same unit.

    ``impassible_grid`` and ``difficult_grid`` are optional COLS*ROWS byte
    grids (typically ``board.impassible_grid``/``board.difficult_grid``).
    When omitted, an all-zero grid is used (no terrain).
    """
    if not USE_C_EXT or not _HAS_C_EXT:
        from movement import _greedy_move
        impassible = bytearray(impassible_grid) if impassible_grid is not None else bytearray(cols * rows)
        difficult = bytearray(difficult_grid) if difficult_grid is not None else bytearray(cols * rows)
        board_shim = type("B", (), {
            "occupancy": occupancy,
            "impassible_grid": impassible,
            "difficult_grid": difficult,
            "terrain": [None] if (impassible_grid is not None or difficult_grid is not None) else [],
            "is_free": lambda s, c, r: 0 <= c < cols and 0 <= r < rows and not occupancy[r * cols + c],
            "is_occupied": lambda s, c, r: bool(occupancy[r * cols + c]),
        })()
        return _greedy_move(start, goal, budget, board_shim, enemy_positions,
                            is_charge=is_charge, flying=flying, strider=strider,
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

    if enemy_bytes is None:
        enemy_bytes = _positions_to_int_bytes(enemy_positions)
    n_enemies = len(enemy_positions)

    imp_bytes = bytes(impassible_grid) if impassible_grid is not None else _empty_terrain_grid(cols, rows)
    diff_bytes = bytes(difficult_grid) if difficult_grid is not None else _empty_terrain_grid(cols, rows)

    return _fast_core.c_pathfind_move(
        col, row, gc, gr, budget,
        bytes(occupancy), bytes(exclusion_grid), enemy_bytes,
        imp_bytes, diff_bytes,
        n_enemies, cols, rows,
        int(is_charge), int(flying), int(already_adjacent), int(strider),
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


# ---------------------------------------------------------------------------
# fast_dijkstra_reachable_set
# ---------------------------------------------------------------------------

def fast_dijkstra_reachable_set(
    start: tuple[int, int],
    budget: float,
    occupancy: bytearray,
    enemy_positions: set[tuple[int, int]],
    is_charge: bool = False,
    flying: bool = False,
    strider: bool = False,
    exclusion_grid: bytearray | None = None,
    cols: int = 72,
    rows: int = 48,
    impassible_grid: bytearray | None = None,
    difficult_grid: bytearray | None = None,
) -> np.ndarray:
    """Return all reachable (col, row) cells within movement budget.

    Returns (N, 2) int32 numpy array.  Falls back to pure-Python Dijkstra
    when the C extension is unavailable. When terrain (impassible/difficult)
    or strider is active, the search runs in Python regardless — see
    TERRAIN_SPEC.md §3.3 for the (square, has_entered_difficult) state space.
    """
    col, row = start

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

    if USE_C_EXT and _HAS_C_EXT:
        enemy_bytes = _positions_to_int_bytes(enemy_positions)
        n_enemies = len(enemy_positions)
        imp_bytes = bytes(impassible_grid) if impassible_grid is not None else _empty_terrain_grid(cols, rows)
        diff_bytes = bytes(difficult_grid) if difficult_grid is not None else _empty_terrain_grid(cols, rows)

        raw = _fast_core.c_dijkstra_reachable_set(
            col, row, budget,
            bytes(occupancy), bytes(exclusion_grid), enemy_bytes,
            imp_bytes, diff_bytes,
            n_enemies, cols, rows,
            int(is_charge), int(flying), int(already_adjacent), int(strider),
        )
        if len(raw) == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.frombuffer(raw, dtype=np.int32).reshape(-1, 2).copy()

    has_terrain = (impassible_grid is not None and any(impassible_grid)) or (
        difficult_grid is not None and any(difficult_grid))

    # --- Pure-Python fallback: Dijkstra over (cell, has_entered_difficult) ---
    check_exclusion = not is_charge and not already_adjacent
    no_enemies = not enemy_positions
    total = cols * rows
    INF_COST = 999999999

    apply_difficult_cap = has_terrain and difficult_grid is not None and not flying and not strider
    block_impassible = has_terrain and impassible_grid is not None and not flying

    if apply_difficult_cap:
        cost_arr = [INF_COST] * (2 * total)
    else:
        cost_arr = [INF_COST] * total
    start_idx = row * cols + col
    cost_arr[start_idx] = 0

    budget_milli = int(budget * 1000 + 0.5)
    cap_milli = 6000  # 6.0" cap per §3.2
    pq: list[tuple[int, int, int, int]] = [(0, col, row, 0)]

    _DIRS = [(1,0,1000), (-1,0,1000), (0,1,1000), (0,-1,1000),
             (1,1,1414), (1,-1,1414), (-1,1,1414), (-1,-1,1414)]

    while pq:
        c, cc, cr, layer = heapq.heappop(pq)
        cidx = cr * cols + cc
        slot = (layer * total + cidx) if apply_difficult_cap else cidx
        if c > cost_arr[slot]:
            continue
        for ddc, ddr, step_cost in _DIRS:
            nc = cc + ddc
            nr = cr + ddr
            if nc < 0 or nc >= cols or nr < 0 or nr >= rows:
                continue
            new_cost = c + step_cost
            if new_cost > budget_milli + 10:
                continue
            if not flying and not no_enemies and (nc, nr) in enemy_positions:
                continue
            if check_exclusion and exclusion_grid[nr * cols + nc]:
                continue
            n_grid_idx = nr * cols + nc
            if block_impassible and impassible_grid[n_grid_idx]:
                continue
            new_layer = layer
            if apply_difficult_cap and difficult_grid[n_grid_idx]:
                new_layer = 1
            if apply_difficult_cap and new_layer == 1 and new_cost > cap_milli + 10:
                continue
            nslot = (new_layer * total + n_grid_idx) if apply_difficult_cap else n_grid_idx
            if new_cost < cost_arr[nslot]:
                cost_arr[nslot] = new_cost
                heapq.heappush(pq, (new_cost, nc, nr, new_layer))

    # Collect results — destination cannot be impassible (even for flying).
    result = []
    for idx in range(total):
        if apply_difficult_cap:
            best = cost_arr[idx]
            if cost_arr[total + idx] < best:
                best = cost_arr[total + idx]
        else:
            best = cost_arr[idx]
        if best >= INF_COST:
            continue
        if occupancy[idx]:
            continue
        if has_terrain and impassible_grid is not None and impassible_grid[idx]:
            continue
        cc = idx % cols
        cr = idx // cols
        if not no_enemies and (cc, cr) in enemy_positions:
            continue
        result.append((cc, cr))

    if not result:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(result, dtype=np.int32)


# ---------------------------------------------------------------------------
# fast_build_exclusion_grid
# ---------------------------------------------------------------------------

def fast_build_exclusion_grid(
    enemy_positions: set[tuple[int, int]] | list[tuple[int, int]],
    cols: int = 72,
    rows: int = 48,
    enemy_bytes: bytes | None = None,
) -> bytearray:
    """Drop-in replacement for movement.build_exclusion_grid using the C extension."""
    if not USE_C_EXT or not _HAS_C_EXT:
        from movement import build_exclusion_grid
        return build_exclusion_grid(enemy_positions)
    if enemy_bytes is None:
        enemy_bytes = _positions_to_int_bytes(enemy_positions)
    n_enemies = len(enemy_positions)
    return _fast_core.c_build_exclusion_grid(enemy_bytes, n_enemies, cols, rows)


# ---------------------------------------------------------------------------
# fast_compute_post_move_rel
# ---------------------------------------------------------------------------

def fast_compute_post_move_rel(
    post_x: float,
    post_y: float,
    enemy_positions: list[tuple[float, float]],
    inv_diag: float,
) -> "np.ndarray":
    """Compute (sin θ, cos θ, normalised_dist) × 10 as (30,) float32 numpy array.
    Caller wraps with torch.from_numpy (zero-copy) as needed.
    """
    if not USE_C_EXT or not _HAS_C_EXT:
        import math
        n = len(enemy_positions)
        out = np.zeros(n * 3, dtype=np.float32)
        for i, (ex, ey) in enumerate(enemy_positions):
            dx = ex - post_x
            dy = ey - post_y
            d = math.sqrt(dx * dx + dy * dy)
            base = i * 3
            if d >= 1e-6:
                out[base] = dy / d
                out[base + 1] = dx / d
            out[base + 2] = d * inv_diag
        return out

    enemy_bytes = _doubles_to_bytes(enemy_positions)
    raw = _fast_core.c_compute_post_move_rel(
        float(post_x), float(post_y), enemy_bytes, len(enemy_positions),
        float(inv_diag))
    return np.frombuffer(raw, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# fast_encode_unit_tactical
# ---------------------------------------------------------------------------

def fast_encode_unit_tactical(
    scalars: np.ndarray,                      # (15,) float64
    objectives: np.ndarray,                   # (5, 2) float64, C-contiguous
    opp_positions: np.ndarray,                # (10, 2) float64
    opp_advance_distances: np.ndarray,        # (10,) float64
    same_positions: np.ndarray,               # (10, 2) float64
    ranged_matchups: np.ndarray,              # (10, 7) float32
    melee_matchups: np.ndarray,               # (10,) float32
    buf: np.ndarray,                          # (TACTICAL_TOTAL_FEATURES,) float32, writable
    offset: int,
    inv_diag: float,
    max_tough: float,
    max_models: float,
    max_speed: float,
    obj_seize_range: float,
    dead_sentinel: tuple[float, float],
    cols: int,
    rows: int,
) -> None:
    """Write one unit's 200-float block into buf at offset via the C extension.

    Caller is responsible for skipping dead units (models_alive <= 0) and
    ensuring all input arrays are contiguous with the declared dtype. The
    function matches ml_features._encode_unit_tactical_into bit-equivalently
    up to float32 rounding on sin/cos.
    """
    if not USE_C_EXT or not _HAS_C_EXT:
        raise RuntimeError(
            "fast_encode_unit_tactical requires the C extension. "
            "Use ml_features._encode_unit_tactical_into directly as a fallback.")

    # Use the buffer protocol directly — no tobytes() copies — but ensure
    # contiguity for any array that might be a slice of a larger matrix.
    rng = (ranged_matchups if ranged_matchups.flags.c_contiguous
           else np.ascontiguousarray(ranged_matchups))
    mel = (melee_matchups if melee_matchups.flags.c_contiguous
           else np.ascontiguousarray(melee_matchups))
    _fast_core.c_encode_unit_tactical(
        scalars, objectives, opp_positions, opp_advance_distances,
        same_positions, rng, mel,
        buf, int(offset),
        float(inv_diag),
        float(max_tough), float(max_models), float(max_speed),
        float(obj_seize_range),
        float(dead_sentinel[0]), float(dead_sentinel[1]),
        int(cols), int(rows),
    )
