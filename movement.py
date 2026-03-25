"""Movement execution: Dijkstra pathfinding, coherency validation."""
from __future__ import annotations

import heapq
import math

from board import Board, COLS, ROWS, dist, dist_sq
import fast_core as _fc

SQRT2 = math.sqrt(2)

# Coherency thresholds (centre-to-centre)
COHERENCY_NEAR_SQ = 4    # 2" c2c → within 1" edge-to-edge
COHERENCY_FAR_SQ = 100   # 10" c2c → within 9" edge-to-edge

# 8 directions: (dc, dr, cost)
_DIRS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
]


def is_in_exclusion_zone(col: int, row: int,
                         enemy_positions: set[tuple[int, int]]) -> bool:
    """Check if (col, row) is within 1\" (adjacent incl. diagonal) of any enemy model."""
    for dc in range(-1, 2):
        for dr in range(-1, 2):
            if dc == 0 and dr == 0:
                continue
            if (col + dc, row + dr) in enemy_positions:
                return True
    return False


def build_exclusion_grid(enemy_positions: set[tuple[int, int]]) -> bytearray:
    """Precompute a flat grid marking squares within 1\" of any enemy model.
    Uses O(1) index lookup instead of set membership checks."""
    grid = bytearray(COLS * ROWS)
    for c, r in enemy_positions:
        for dc in range(-1, 2):
            for dr in range(-1, 2):
                if dc == 0 and dr == 0:
                    continue
                nc, nr = c + dc, r + dr
                if 0 <= nc < COLS and 0 <= nr < ROWS:
                    grid[nr * COLS + nc] = 1
    return grid


def check_coherency(positions: list[tuple[int, int]]) -> bool:
    """Check unit coherency: each model within 2" c2c of at least one other,
    and all models within 10" c2c of each other."""
    n = len(positions)
    if n <= 1:
        return True

    # Near-coherency: each model within 2" of at least one other
    for i in range(n):
        has_near = False
        for j in range(n):
            if i != j and dist_sq(positions[i], positions[j]) <= COHERENCY_NEAR_SQ:
                has_near = True
                break
        if not has_near:
            return False

    # Far-coherency: all pairs within 10"
    for i in range(n):
        for j in range(i + 1, n):
            if dist_sq(positions[i], positions[j]) > COHERENCY_FAR_SQ:
                return False

    return True


def _greedy_move(start: tuple[int, int], goal: tuple[int, int],
                 budget: float, board: Board,
                 enemy_positions: set[tuple[int, int]],
                 is_charge: bool = False,
                 flying: bool = False,
                 exclusion_grid: bytearray | None = None) -> tuple[int, int]:
    """Move one model toward goal using Dijkstra pathfinding within budget.

    Explores all reachable cells within the movement budget, then picks the
    reachable cell closest to *goal* that is not occupied or enemy-held.
    Unlike the old greedy approach, this can route around friendly models
    and other obstacles that require a temporary detour away from the goal.

    Cannot move through enemy-occupied squares (unless flying).
    Can move through friendly-occupied squares but not end there.
    Non-charge movement rejects squares in the 1\" exclusion zone
    (unless the model started adjacent to enemies)."""

    # --- C-accelerated fast path ---
    if _fc.USE_C_EXT:
        if exclusion_grid is None:
            exclusion_grid = build_exclusion_grid(enemy_positions)
        result = _fc.fast_pathfind_move(
            start, goal, budget, board.occupancy, enemy_positions,
            is_charge=is_charge, flying=flying,
            exclusion_grid=exclusion_grid, cols=COLS, rows=ROWS,
        )
        # C path may return a position that is occupied if it couldn't find
        # an alternative — in that case fall back to start (matching Python behavior).
        if result != start and board.is_occupied(result[0], result[1]):
            return start
        return result

    # --- Pure Python Dijkstra ---
    col, row = start
    gc, gr = goal

    dc0 = col - gc
    dr0 = row - gr
    start_dist_sq = dc0 * dc0 + dr0 * dr0
    if start_dist_sq == 0:
        return start

    max_col = COLS
    max_row = ROWS
    occupancy = board.occupancy

    # If already adjacent to enemies, exclusion zone doesn't apply
    if exclusion_grid is not None:
        already_adjacent = bool(exclusion_grid[row * max_col + col])
    else:
        already_adjacent = is_in_exclusion_zone(col, row, enemy_positions)
    check_exclusion = not is_charge and not already_adjacent

    if check_exclusion and exclusion_grid is None:
        exclusion_grid = build_exclusion_grid(enemy_positions)

    no_enemies = not enemy_positions
    has_excl = check_exclusion and exclusion_grid is not None

    # Dijkstra: explore all reachable cells within budget
    # Use a flat cost array for speed (grid is only 72*48 = 3456 cells)
    total = max_col * max_row
    INF_COST = 999999999
    # cost_arr[cell_index] = best cost in milli-inches to reach cell
    cost_arr = [INF_COST] * total
    start_idx = row * max_col + col
    cost_arr[start_idx] = 0

    budget_milli = int(budget * 1000 + 0.5)
    # Priority queue: (cost_milli, col, row)
    pq = [(0, col, row)]

    while pq:
        c, cc, cr = heapq.heappop(pq)
        cidx = cr * max_col + cc
        if c > cost_arr[cidx]:
            continue
        for ddc, ddr, step_cost in _DIRS:
            nc = cc + ddc
            nr = cr + ddr
            if nc < 0 or nc >= max_col or nr < 0 or nr >= max_row:
                continue
            if not flying and not no_enemies and (nc, nr) in enemy_positions:
                continue
            if has_excl and exclusion_grid[nr * max_col + nc]:
                continue
            new_cost = c + int(step_cost * 1000 + 0.5)
            if new_cost > budget_milli + 10:  # small tolerance
                continue
            nidx = nr * max_col + nc
            if new_cost < cost_arr[nidx]:
                cost_arr[nidx] = new_cost
                heapq.heappush(pq, (new_cost, nc, nr))

    # Find best reachable, non-occupied cell closest to goal
    best_pos = start
    best_goal_dist = start_dist_sq
    for idx in range(total):
        if cost_arr[idx] >= INF_COST:
            continue
        cc = idx % max_col
        cr = idx // max_col
        if occupancy[idx]:
            continue
        if not no_enemies and (cc, cr) in enemy_positions:
            continue
        dgc = cc - gc
        dgr = cr - gr
        d = dgc * dgc + dgr * dgr
        if d < best_goal_dist:
            best_goal_dist = d
            best_pos = (cc, cr)

    return best_pos


def execute_movement(unit_state, goal: tuple[int, int], budget: float,
                     board: Board, enemy_positions: set[tuple[int, int]],
                     is_charge: bool = False, flying: bool = False,
                     range_target: tuple[int, int] | None = None,
                     weapon_range: float = 0):
    """Move all models in a unit toward goal, maintaining coherency.
    Models closest to goal move first.

    If *range_target* and *weapon_range* are provided (kite moves), a post-pass
    nudges any models that ended up outside weapon range toward *range_target*
    so that as many models as possible can fire.
    """
    positions = unit_state.alive_positions()
    n = len(positions)
    if n == 0 or budget <= 0:
        return

    # Precompute exclusion grid once for all models
    exclusion_grid = None if is_charge else build_exclusion_grid(enemy_positions)

    # Sort models by distance to goal (closest first)
    model_order = sorted(range(n), key=lambda i: dist_sq(positions[i], goal))

    # Remove all models from occupancy first
    for i in range(n):
        c, r = unit_state.positions[i]
        board.remove(c, r)

    # Track new positions — finalized models stay on the board so _greedy_move
    # sees them as occupied and won't end on them (O(n) vs old O(n²) approach).
    new_positions: list[tuple[int, int]] = [None] * n  # type: ignore
    placed_positions: list[tuple[int, int]] = []  # already-finalized positions

    for idx in model_order:
        old_pos = unit_state.positions[idx]

        # Leashed goal: the first model (leader) targets the raw goal.
        # Subsequent models target a point clamped to within coherency
        # distance of the nearest already-placed teammate so the unit
        # stays together when navigating around obstacles.
        model_goal = goal
        if placed_positions:
            # Find nearest already-placed teammate
            nearest_pp = min(placed_positions, key=lambda p: dist_sq(p, goal))
            # If the raw goal is far from all placed models, clamp it to
            # a point along the goal direction within coherency range of
            # the nearest-to-goal placed model.
            gd_sq = dist_sq(nearest_pp, goal)
            if gd_sq > COHERENCY_NEAR_SQ:
                # Leash: point along nearest_pp→goal at coherency distance
                gd = math.sqrt(gd_sq)
                leash = math.sqrt(COHERENCY_NEAR_SQ)  # 2.0
                t = leash / gd
                lx = nearest_pp[0] + t * (goal[0] - nearest_pp[0])
                ly = nearest_pp[1] + t * (goal[1] - nearest_pp[1])
                model_goal = (int(round(lx)), int(round(ly)))

        new_pos = _greedy_move(old_pos, model_goal, budget, board, enemy_positions,
                              is_charge=is_charge, flying=flying,
                              exclusion_grid=exclusion_grid)

        # _greedy_move avoids occupied squares at the end, but if the model
        # couldn't move (returns start) and start is now occupied by a
        # finalized teammate, find an adjacent free square.
        # Collision resolution: prefer squares near teammates over near goal.
        if board.is_occupied(new_pos[0], new_pos[1]):
            best_alt = None
            best_alt_dist = 999999
            for dc, dr, _ in _DIRS:
                nc, nr = new_pos[0] + dc, new_pos[1] + dr
                if board.is_free(nc, nr) and (nc, nr) not in enemy_positions:
                    if placed_positions:
                        # Prefer proximity to nearest teammate
                        d = min(dist_sq((nc, nr), pp) for pp in placed_positions)
                    else:
                        dg_c = nc - goal[0]
                        dg_r = nr - goal[1]
                        d = dg_c * dg_c + dg_r * dg_r
                    if d < best_alt_dist:
                        best_alt = (nc, nr)
                        best_alt_dist = d
            if best_alt:
                new_pos = best_alt
            else:
                new_pos = old_pos  # Stay put

        new_positions[idx] = new_pos
        placed_positions.append(new_pos)
        board.place(new_pos[0], new_pos[1])

    # Update unit positions (models are already on the board)
    for i in range(n):
        unit_state.positions[i] = new_positions[i]

    # --- Range-ensure post-pass (kite moves) ---
    # Nudge models that landed outside weapon range toward the target.
    # Only apply if the unit theoretically had enough budget to reach weapon
    # range (pre-move centre distance − budget ≤ weapon_range).  This prevents
    # the nudge from granting effective extra movement to units that were
    # genuinely too far away.
    if range_target is not None and weapon_range > 0:
        # positions[] is the pre-move snapshot captured at the top of this function
        cx = sum(p[0] for p in positions) / n
        cy = sum(p[1] for p in positions) / n
        pre_dist = math.sqrt((cx - range_target[0]) ** 2 + (cy - range_target[1]) ** 2)
        if pre_dist - budget <= weapon_range:
            _ensure_weapon_range(unit_state, board, enemy_positions,
                                 range_target, weapon_range,
                                 exclusion_grid=exclusion_grid)

    # Coherency repair: if violated, pull stragglers toward the group centre
    if n > 1 and not check_coherency(unit_state.alive_positions()):
        _repair_coherency(unit_state, board, enemy_positions,
                          exclusion_grid=exclusion_grid,
                          is_charge=is_charge)


def _ensure_weapon_range(unit_state, board: Board,
                         enemy_positions: set[tuple[int, int]],
                         range_target: tuple[int, int],
                         weapon_range: float,
                         exclusion_grid: bytearray | None = None):
    """Post-pass: nudge models that are outside *weapon_range* of *range_target*
    toward the target so they can contribute to shooting.

    Each out-of-range model gets a small greedy push (up to 3\") toward the
    target.  The budget is capped so models don't overshoot into melee."""
    n = unit_state.models_alive
    if n == 0:
        return

    range_sq = weapon_range * weapon_range
    tx, ty = range_target

    if exclusion_grid is None:
        exclusion_grid = build_exclusion_grid(enemy_positions)

    # Sort: nudge the farthest-from-target models first so they get first pick
    # of free squares closer in.
    order = sorted(range(n), key=lambda i: -dist_sq(unit_state.positions[i], range_target))

    for idx in order:
        pos = unit_state.positions[idx]
        if dist_sq(pos, range_target) <= range_sq:
            continue  # already in range

        # Nudge toward target with a small budget (up to 3")
        board.remove(pos[0], pos[1])
        nudge_goal = range_target
        new_pos = _greedy_move(pos, nudge_goal, 3.0, board, enemy_positions,
                               exclusion_grid=exclusion_grid)

        # Accept nudge only if it got us closer to being in range
        if dist_sq(new_pos, range_target) < dist_sq(pos, range_target):
            if board.is_free(new_pos[0], new_pos[1]):
                unit_state.positions[idx] = new_pos
                board.place(new_pos[0], new_pos[1])
            else:
                board.place(pos[0], pos[1])
        else:
            board.place(pos[0], pos[1])


def _repair_coherency(unit_state, board: Board,
                      enemy_positions: set[tuple[int, int]],
                      exclusion_grid: bytearray | None = None,
                      is_charge: bool = False):
    """Pull straggling models toward unit centre to restore coherency.

    When *is_charge* is True the exclusion zone is intentionally ignored
    (matching charge movement rules) so the repair doesn't get blocked
    by enemy proximity that the charge was allowed to ignore."""
    positions = unit_state.alive_positions()
    n = len(positions)
    if n <= 1:
        return

    # For charges, keep exclusion_grid as None so _greedy_move ignores
    # the exclusion zone (is_charge=True path).  For normal moves, build
    # the grid if the caller didn't provide one.
    if not is_charge and exclusion_grid is None:
        exclusion_grid = build_exclusion_grid(enemy_positions)

    # Compute centre
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n
    centre = (int(round(cx)), int(round(cy)))

    # Find models that violate near-coherency
    for _ in range(5):  # max repair iterations
        if check_coherency(unit_state.alive_positions()):
            return

        for i in range(n):
            pos = unit_state.positions[i]
            # Check if this model has near-coherency
            has_near = False
            for j in range(n):
                if i != j and dist_sq(unit_state.positions[i], unit_state.positions[j]) <= COHERENCY_NEAR_SQ:
                    has_near = True
                    break
            if has_near:
                continue

            # Pull toward centre
            board.remove(pos[0], pos[1])
            new_pos = _greedy_move(pos, centre, 4.0, board, enemy_positions,
                                   is_charge=is_charge,
                                   exclusion_grid=exclusion_grid)
            if new_pos != pos and board.is_free(new_pos[0], new_pos[1]):
                unit_state.positions[i] = new_pos
                board.place(new_pos[0], new_pos[1])
            else:
                board.place(pos[0], pos[1])


# ===================================================================
# CHARGE MOVEMENT
# ===================================================================

def execute_charge_movement(charger, target, board: Board,
                            enemy_positions: set[tuple[int, int]]):
    """Move charger toward target using rush budget, ignoring exclusion zone."""
    tc = target.centre()
    goal = (int(round(tc[0])), int(round(tc[1])))
    budget = charger.unit.rush_distance
    execute_movement(charger, goal, budget, board, enemy_positions,
                     is_charge=True, flying=charger.unit.flying)


def execute_counter_charge(defender, charger, board: Board):
    """Defender models not in melee range move up to 3\" toward charger."""
    charger_positions = set(charger.alive_positions())
    if not charger_positions:
        return

    cc = charger.centre()
    goal = (int(round(cc[0])), int(round(cc[1])))

    n = defender.models_alive
    if n == 0:
        return

    for i in range(n):
        pos = defender.positions[i]
        # Check if already within 2 squares c2c of any charger model
        in_range = False
        for cp in charger_positions:
            if dist_sq(pos, cp) <= 4:
                in_range = True
                break
        if in_range:
            continue

        # Move this model up to 3" toward charger centroid
        board.remove(pos[0], pos[1])
        new_pos = _greedy_move(pos, goal, 3.0, board, set(),
                               is_charge=True)
        if board.is_free(new_pos[0], new_pos[1]):
            defender.positions[i] = new_pos
            board.place(new_pos[0], new_pos[1])
        else:
            board.place(pos[0], pos[1])


def post_melee_separation(charger, defender, board: Board,
                          enemy_positions: set[tuple[int, int]]):
    """Move charger models the shortest distance to be >1\" from all enemies (up to 3\")."""
    n = charger.models_alive
    if n == 0:
        return

    exclusion_grid = build_exclusion_grid(enemy_positions)

    # Remove all charger models from occupancy first
    for i in range(n):
        board.remove(charger.positions[i][0], charger.positions[i][1])

    placed: set[tuple[int, int]] = set()

    for i in range(n):
        pos = charger.positions[i]

        # Temporarily place already-moved models so BFS avoids them
        for p in placed:
            board.place(p[0], p[1])

        if not exclusion_grid[pos[1] * COLS + pos[0]]:
            # Doesn't need to move, but still mark as placed
            charger.positions[i] = pos
        else:
            best = _find_nearest_outside_exclusion(
                pos, board, enemy_positions, budget=3.0,
                exclusion_grid=exclusion_grid)
            if best is not None:
                charger.positions[i] = best
            # else stays at pos

        placed.add(charger.positions[i])

        # Remove temporary occupancy
        for p in placed:
            board.remove(p[0], p[1])

    # Place all models at final positions
    for i in range(n):
        board.place(charger.positions[i][0], charger.positions[i][1])

    # Repair coherency if needed
    if n > 1 and not check_coherency(charger.alive_positions()):
        _repair_coherency(charger, board, enemy_positions,
                          exclusion_grid=exclusion_grid)


def _find_nearest_outside_exclusion(
    start: tuple[int, int], board: Board,
    enemy_positions: set[tuple[int, int]],
    budget: float,
    exclusion_grid: bytearray | None = None,
) -> tuple[int, int] | None:
    """BFS from start to find the nearest free square that is outside the
    exclusion zone of all enemies and within movement budget."""
    import heapq

    if exclusion_grid is None:
        exclusion_grid = build_exclusion_grid(enemy_positions)

    # Dijkstra by movement cost (diagonal = sqrt2, cardinal = 1)
    visited: set[tuple[int, int]] = set()
    # (cost, col, row)
    heap: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]

    while heap:
        cost, col, row = heapq.heappop(heap)
        if (col, row) in visited:
            continue
        visited.add((col, row))

        # Check if this square is valid: free, outside exclusion zone
        if (col, row) != start:
            if not board.is_free(col, row):
                continue
            if (col, row) in enemy_positions:
                continue
            if not exclusion_grid[row * COLS + col]:
                return (col, row)

        # Expand neighbours
        for dc, dr, step_cost in _DIRS:
            nc, nr = col + dc, row + dr
            new_cost = cost + step_cost
            if new_cost > budget + 0.01:
                continue
            if not board.in_bounds(nc, nr):
                continue
            if (nc, nr) in enemy_positions:
                continue
            if (nc, nr) not in visited:
                heapq.heappush(heap, (new_cost, nc, nr))

    return None


def consolidation_move(survivor, board: Board,
                       enemies: list, objectives: list,
                       mode: str = "objectives"):
    """Surviving unit moves up to 3\" toward nearest objective or enemy."""
    if survivor.models_alive <= 0:
        return

    # Pick goal
    sc = survivor.centre()
    best_goal = None
    best_dist = 999999.0

    if mode == "objectives" and objectives:
        for obj in objectives:
            d = dist_sq((int(sc[0]), int(sc[1])), obj)
            if d < best_dist:
                best_dist = d
                best_goal = obj
    else:
        for e in enemies:
            if e.models_alive <= 0:
                continue
            ec = e.centre()
            d = (sc[0] - ec[0]) ** 2 + (sc[1] - ec[1]) ** 2
            if d < best_dist:
                best_dist = d
                best_goal = (int(round(ec[0])), int(round(ec[1])))

    if best_goal is None:
        return

    enemy_positions: set[tuple[int, int]] = set()
    for e in enemies:
        for p in e.alive_positions():
            enemy_positions.add(p)

    execute_movement(survivor, best_goal, 3.0, board, enemy_positions)
