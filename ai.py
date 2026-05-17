"""AI role logic: target selection, action choice, activation ordering, role reassignment."""
from __future__ import annotations

import math

from board import Board, COLS, ROWS, OBJECTIVES, OBJ_SEIZE_RANGE, HOME_OBJ_A, HOME_OBJ_B, dist, dist_sq
import fast_core as _fc


def _obj_goal(obj: tuple) -> tuple[int, int]:
    """Convert an objective position (possibly float) to an integer movement goal."""
    return (int(round(obj[0])), int(round(obj[1])))
from combat import (
    expected_damage_score, expected_melee_damage_score,
    models_in_range, can_shoot_any,
    is_full_volley, evaluate_target,
)
from models import UnitState

OBJ_NAMES = ["Centre", "A-side", "B-side", "Home-A", "Home-B"]


# ===================================================================
# TARGET SELECTION
# ===================================================================

def _base_target_score(attacker: UnitState, target: UnitState) -> float:
    """Expected wounds / target points value — same as original pick_target."""
    hit_prob = (7 - attacker.unit.quality) / 6.0
    expected_wounds = 0.0
    for w in attacker.unit.weapons:
        eff = models_in_range(attacker, target, w.range_inches)
        if eff == 0:
            continue
        p = 5 / 6 if w.reliable else hit_prob
        hits = w.attacks * p * eff
        if w.blast:
            hits *= min(w.blast, target.models_alive)
        eff_def = min(target.unit.defense + w.ap, 7)
        block_prob = max((7 - eff_def) / 6.0, 1 / 6)
        wounds = hits * (1 - block_prob)
        if w.deadly:
            wounds *= w.deadly
        expected_wounds += wounds
    return expected_wounds / max(target.unit.points, 1)


def _is_near_objective(unit: UnitState) -> bool:
    """Check if any alive model is within 3" of any objective."""
    threshold_sq = OBJ_SEIZE_RANGE * OBJ_SEIZE_RANGE
    for pos in unit.alive_positions():
        for obj in OBJECTIVES:
            if dist_sq(pos, obj) <= threshold_sq:
                return True
    return False


def pick_target_killer(attacker: UnitState, enemies: list[UnitState],
                       target_multipliers: list[float] | None = None,
                       board: Board | None = None) -> UnitState | None:
    """Killer targeting: max wounds/cost, prefer full-volley targets."""
    best = None
    best_score = -1.0
    best_full = False

    for i, enemy in enumerate(enemies):
        if enemy.models_alive <= 0:
            continue
        can_shoot, score, full = evaluate_target(attacker, enemy, board)
        if not can_shoot:
            continue
        if target_multipliers is not None and i < len(target_multipliers):
            score *= target_multipliers[i]
        # Prefer full volley targets
        if full and not best_full:
            best = enemy
            best_score = score
            best_full = True
        elif full == best_full and score > best_score:
            best = enemy
            best_score = score
            best_full = full

    return best


def pick_target_clearer(attacker: UnitState, enemies: list[UnitState],
                        target_multipliers: list[float] | None = None,
                        board: Board | None = None) -> UnitState | None:
    """Objective-Clearer targeting: 3x multiplier for enemies near objectives."""
    best = None
    best_score = -1.0
    best_full = False

    for i, enemy in enumerate(enemies):
        if enemy.models_alive <= 0:
            continue
        can_shoot, score, full = evaluate_target(attacker, enemy, board)
        if not can_shoot:
            continue
        if _is_near_objective(enemy):
            score *= 3.0
        if target_multipliers is not None and i < len(target_multipliers):
            score *= target_multipliers[i]
        if full and not best_full:
            best = enemy
            best_score = score
            best_full = True
        elif full == best_full and score > best_score:
            best = enemy
            best_score = score
            best_full = full

    return best


def pick_target_holder(attacker: UnitState, enemies: list[UnitState],
                       target_multipliers: list[float] | None = None,
                       board: Board | None = None) -> UnitState | None:
    """Objective-Holder targeting: only targets of opportunity (in range now)."""
    best = None
    best_score = -1.0

    for i, enemy in enumerate(enemies):
        if enemy.models_alive <= 0:
            continue
        can_shoot, score, _full = evaluate_target(attacker, enemy, board)
        if not can_shoot:
            continue
        if target_multipliers is not None and i < len(target_multipliers):
            score *= target_multipliers[i]
        if score > best_score:
            best = enemy
            best_score = score

    return best


def pick_target(attacker: UnitState, enemies: list[UnitState],
                target_multipliers: list[float] | None = None,
                board: Board | None = None) -> UnitState | None:
    """Dispatch to role-specific target selection."""
    role = attacker.ai_role
    if role == "objective_clearer":
        return pick_target_clearer(attacker, enemies, target_multipliers, board)
    elif role == "objective_holder":
        return pick_target_holder(attacker, enemies, target_multipliers, board)
    elif role == "home_objective_holder":
        # Standard target priority (like killer) while sitting on home objective
        return pick_target_killer(attacker, enemies, target_multipliers, board)
    else:
        return pick_target_killer(attacker, enemies, target_multipliers, board)


# ===================================================================
# ACTION CHOICE
# ===================================================================

def _unit_on_objective(unit: UnitState, obj_idx: int) -> bool:
    """Check if any alive model is within 3" of the assigned objective."""
    if obj_idx < 0 or obj_idx >= len(OBJECTIVES):
        return False
    obj = OBJECTIVES[obj_idx]
    threshold_sq = OBJ_SEIZE_RANGE * OBJ_SEIZE_RANGE
    for pos in unit.alive_positions():
        if dist_sq(pos, obj) <= threshold_sq:
            return True
    return False


def _dist_to_objective(unit: UnitState, obj_idx: int) -> float:
    """Distance from unit centre to assigned objective."""
    if obj_idx < 0 or obj_idx >= len(OBJECTIVES):
        return 999.0
    obj = OBJECTIVES[obj_idx]
    cx, cy = unit.centre()
    return math.sqrt((cx - obj[0]) ** 2 + (cy - obj[1]) ** 2)


def choose_action_and_goal(unit: UnitState, enemies: list[UnitState],
                           board: Board, mode: str = "objectives",
                           target_multipliers: list[float] | None = None,
                           ) -> tuple[str, tuple[int, int] | None, UnitState | None, str]:
    """Determine action (hold/advance/rush/charge) and movement goal for a unit.
    Returns (action, goal_position, charge_target, reason).
    charge_target is only set when action == "charge".
    """
    if unit.unit.artillery:
        return "hold", None, None, "artillery holds position"

    combat_pref = getattr(unit, 'combat_preference', 'ranged')

    is_holder_role = unit.ai_role in ("objective_holder", "home_objective_holder")

    if combat_pref == "melee":
        if mode == "kill_points" or not is_holder_role:
            return _choose_melee_killer_action(unit, enemies, board)
        else:
            return _choose_melee_holder_action(unit, enemies, board)

    if mode == "kill_points" or not is_holder_role:
        action, goal, reason = _choose_killer_action(unit, enemies, target_multipliers, board)
        return action, goal, None, reason
    else:
        action, goal, reason = _choose_holder_action(unit, enemies)
        return action, goal, None, reason


def _choose_killer_action(unit: UnitState,
                          enemies: list[UnitState],
                          target_multipliers: list[float] | None = None,
                          board: Board | None = None,
                          ) -> tuple[str, tuple[int, int] | None, str]:
    """Killer/Clearer action: prioritize shooting, advance to get in range.
    Respects movement_stance for ranged-preference killers:
      - kite: hold at max weapon range, advance (never rush) toward range edge
      - normal: existing behaviour
      - aggressive: always close distance, rush when out of range
    """
    stance = getattr(unit, 'movement_stance', 'normal')
    combat_pref = getattr(unit, 'combat_preference', 'ranged')
    # Stance only applies to ranged-preference killers (not holders, not melee)
    use_stance = (combat_pref == "ranged"
                  and unit.ai_role not in ("objective_holder", "home_objective_holder"))

    best_target = pick_target(unit, enemies, target_multipliers, board)

    if best_target is not None:
        if is_full_volley(unit, best_target, board):
            # All weapons in range
            if use_stance and stance == "aggressive":
                # Aggressive: keep closing even when in range
                target_centre = best_target.centre()
                goal = (int(round(target_centre[0])), int(round(target_centre[1])))
                return "advance", goal, "aggressive: closing distance despite full volley"
            if use_stance and stance == "kite":
                # Kite: hold position (already in range)
                return "hold", None, "kite: holding at range"
            # Normal: hold
            return "hold", None, "all weapons in range"

        # Not all weapons in range — need to move closer
        target_centre = best_target.centre()

        if use_stance and stance == "kite":
            # Kite: advance toward max weapon range position, never rush
            goal = _kite_goal(unit, best_target)
            return "advance", goal, "kite: advancing to max weapon range"

        if use_stance and stance == "aggressive":
            # Aggressive: advance if full volley is reachable, else rush
            goal = (int(round(target_centre[0])), int(round(target_centre[1])))
            if _can_full_volley_after_advance(unit, best_target):
                return "advance", goal, "aggressive: advancing to full volley range"
            return "rush", goal, "aggressive: rushing toward target"

        # Normal: advance toward target
        goal = (int(round(target_centre[0])), int(round(target_centre[1])))
        return "advance", goal, "advancing to get weapons in range"

    # No target in range — rush toward favoured target
    favoured = _pick_favoured_target(unit, enemies)
    if favoured:
        target_centre = favoured.centre()

        if use_stance and stance == "kite":
            # Kite: advance (never rush) toward max weapon range position
            goal = _kite_goal(unit, favoured)
            return "advance", goal, "kite: no target in range, advancing toward max range"

        goal = (int(round(target_centre[0])), int(round(target_centre[1])))
        return "rush", goal, "no targets in range, rushing toward favoured target"

    return "hold", None, "no enemies found"


def _can_full_volley_after_advance(unit: UnitState, target: UnitState) -> bool:
    """Check if advancing would bring all ranged weapons into range for a full volley."""
    # Find the minimum ranged weapon range across all alive models
    min_range = float('inf')
    for mi in range(unit.models_alive):
        for w in unit.weapons_per_model[mi]:
            if not w.melee:
                min_range = min(min_range, w.range_inches)
    if min_range == float('inf'):
        return False  # no ranged weapons
    # Conservative estimate: centre-to-centre distance minus advance budget
    acx, acy = unit.centre()
    tcx, tcy = target.centre()
    dx = acx - tcx
    dy = acy - tcy
    distance = math.sqrt(dx * dx + dy * dy)
    return distance - unit.unit.advance_distance <= min_range


def _kite_goal(unit: UnitState, target: UnitState) -> tuple[int, int]:
    """Compute a goal position at max weapon range from the target along the
    attacker-to-target line.  Uses a 2\" inward buffer so that model spread
    and greedy-pathfinding undershoot still leave models within weapon range."""
    max_range = unit.unit.max_weapon_range
    # Buffer: 2" inward so models that fan out around the goal stay in range
    effective_range = max(max_range - 2, max_range * 0.5)
    acx, acy = unit.centre()
    tcx, tcy = target.centre()
    dx = acx - tcx
    dy = acy - tcy
    d = math.sqrt(dx * dx + dy * dy)
    if d > 0:
        goal_x = tcx + (dx / d) * effective_range
        goal_y = tcy + (dy / d) * effective_range
    else:
        # On top of target — move away along y axis
        goal_x = tcx
        goal_y = tcy + effective_range
    return (int(round(goal_x)), int(round(goal_y)))


def _choose_holder_action(unit: UnitState,
                          enemies: list[UnitState]) -> tuple[str, tuple[int, int] | None, str]:
    """Objective-Holder action: reach and hold objectives."""
    obj_idx = unit.assigned_objective
    if obj_idx < 0:
        obj_idx = 0  # fallback to centre

    obj = OBJECTIVES[obj_idx]
    obj_name = OBJ_NAMES[obj_idx] if obj_idx < len(OBJ_NAMES) else f"obj {obj_idx}"

    if _unit_on_objective(unit, obj_idx):
        return "hold", None, f"holding {obj_name} objective"

    d = _dist_to_objective(unit, obj_idx)
    advance_dist = unit.unit.advance_distance

    if d <= advance_dist + OBJ_SEIZE_RANGE:
        return "advance", _obj_goal(obj), f"{obj_name} objective in advance range ({d:.0f}\" away)"

    return "rush", _obj_goal(obj), f"{obj_name} objective not in advance range ({d:.0f}\" away)"


def _nearest_enemy(unit: UnitState, enemies: list[UnitState]) -> UnitState | None:
    """Find nearest alive enemy by centre-to-centre distance."""
    cx, cy = unit.centre()
    best = None
    best_d = 999999.0
    for e in enemies:
        if e.models_alive <= 0:
            continue
        ex, ey = e.centre()
        d = (cx - ex) ** 2 + (cy - ey) ** 2
        if d < best_d:
            best_d = d
            best = e
    return best


def _ideal_ranged_target_score(attacker: UnitState, target: UnitState) -> float:
    """Expected wounds/points assuming all models are in range of all ranged weapons."""
    hit_prob = (7 - attacker.unit.quality) / 6.0
    expected_wounds = 0.0
    for w in attacker.unit.weapons:
        if w.range_inches <= 0:
            continue
        p = 5 / 6 if w.reliable else hit_prob
        hits = w.attacks * p * attacker.models_alive
        if w.blast:
            hits *= min(w.blast, target.models_alive)
        eff_def = min(target.unit.defense + w.ap, 7)
        block_prob = max((7 - eff_def) / 6.0, 1 / 6)
        wounds = hits * (1 - block_prob)
        if w.deadly:
            wounds *= w.deadly
        expected_wounds += wounds
    return expected_wounds / max(target.unit.points, 1)


def _pick_favoured_target(unit: UnitState, enemies: list[UnitState]) -> UnitState | None:
    """Pick the best target to rush toward, ignoring range constraints.
    Uses melee scoring for melee-preference units, ideal ranged scoring otherwise.
    Falls back to nearest enemy if no weapon-based score is positive."""
    combat_pref = getattr(unit, 'combat_preference', 'ranged')
    best = None
    best_score = -1.0

    for enemy in enemies:
        if enemy.models_alive <= 0:
            continue
        if combat_pref == "melee":
            score = _melee_target_score(unit, enemy)
        else:
            score = _ideal_ranged_target_score(unit, enemy)
        if score > best_score:
            best_score = score
            best = enemy

    if best is None:
        best = _nearest_enemy(unit, enemies)
    return best


# ===================================================================
# CHARGE TARGET SELECTION
# ===================================================================

_MELEE_RANGE_SQ = 4  # 2 squares c2c — mirrors combat.MELEE_RANGE_SQ


def _can_charge(unit: UnitState, target: UnitState,
                board: Board | None = None,
                enemy_positions: set[tuple[int, int]] | None = None) -> bool:
    """Can the charger physically reach within 2\" of any target model?

    Stage 1 (cheap): centre-to-centre Euclidean filter against charge_distance.
    Stage 2 (terrain-aware): from the charger model nearest the target's
    centre, run a charge-aware pathfind. The charge is legal iff the
    returned cell lands within 2\" c2c of any defender model. Stage 2 is
    skipped when there is no terrain on the board (the Euclidean filter is
    then sufficient) or when *board* is None (legacy callers).
    """
    # Use charge_distance so Versatile Reach's +2" charge bonus is honoured.
    charge_budget = unit.unit.charge_distance
    cx, cy = unit.centre()
    tx, ty = target.centre()
    # Stage 1: quick centre-to-centre check (squared to avoid sqrt)
    threshold = charge_budget + 2
    d_sq = (cx - tx) ** 2 + (cy - ty) ** 2
    if d_sq >= threshold * threshold:
        return False

    # Stage 2: terrain-aware reach check
    if board is None or not board.terrain:
        return True

    charger_positions = unit.alive_positions()
    target_positions = target.alive_positions()
    if not charger_positions or not target_positions:
        return False

    goal = (int(round(tx)), int(round(ty)))
    nearest = min(charger_positions,
                  key=lambda p: (p[0] - goal[0]) ** 2 + (p[1] - goal[1]) ** 2)

    if enemy_positions is None:
        enemy_positions = set(target_positions)

    result = _fc.fast_pathfind_move(
        nearest, goal, charge_budget, board.occupancy, enemy_positions,
        is_charge=True,
        flying=unit.unit.flying,
        strider=unit.unit.strider,
        cols=COLS, rows=ROWS,
        impassible_grid=board.impassible_grid,
        difficult_grid=board.difficult_grid,
    )

    for tp in target_positions:
        dc = result[0] - tp[0]
        dr = result[1] - tp[1]
        if dc * dc + dr * dr <= _MELEE_RANGE_SQ:
            return True
    return False


def _melee_target_score(attacker: UnitState, target: UnitState) -> float:
    """Expected melee wounds / target points."""
    hit_prob = (7 - attacker.unit.quality) / 6.0
    expected_wounds = 0.0
    for w in attacker.unit.weapons:
        if not w.melee:
            continue
        p = 5 / 6 if w.reliable else hit_prob
        hits = w.attacks * p * attacker.models_alive
        if w.blast:
            hits *= min(w.blast, target.models_alive)
        eff_def = min(target.unit.defense + w.ap, 7)
        block_prob = max((7 - eff_def) / 6.0, 1 / 6)
        wounds = hits * (1 - block_prob)
        if w.deadly:
            wounds *= w.deadly
        expected_wounds += wounds
    return expected_wounds / max(target.unit.points, 1)


def pick_charge_target(attacker: UnitState,
                       enemies: list[UnitState],
                       board: Board | None = None) -> UnitState | None:
    """Pick the best charge target from enemies within charge range."""
    best = None
    best_score = -1.0

    enemy_positions = _collect_enemy_positions(enemies) if board is not None else None

    for enemy in enemies:
        if enemy.models_alive <= 0:
            continue
        if not _can_charge(attacker, enemy, board, enemy_positions):
            continue
        score = _melee_target_score(attacker, enemy)
        if attacker.ai_role == "objective_clearer" and _is_near_objective(enemy):
            score *= 3.0
        if score > best_score:
            best_score = score
            best = enemy

    return best


def _collect_enemy_positions(enemies: list[UnitState]) -> set[tuple[int, int]]:
    """Union of alive model positions across all enemy units."""
    positions: set[tuple[int, int]] = set()
    for e in enemies:
        if e.models_alive > 0:
            positions.update(e.alive_positions())
    return positions


# ===================================================================
# MELEE AI DECISION LOGIC
# ===================================================================

def _choose_melee_killer_action(
        unit: UnitState,
        enemies: list[UnitState],
        board: Board | None = None,
) -> tuple[str, tuple[int, int] | None, UnitState | None, str]:
    """Melee killer/clearer: charge if possible, else advance+shoot, else rush."""
    target = pick_charge_target(unit, enemies, board)
    if target is not None:
        tc = target.centre()
        goal = (int(round(tc[0])), int(round(tc[1])))
        return "charge", goal, target, "charge target in range"

    # Fall back to ranged behaviour
    action, goal, reason = _choose_killer_action(unit, enemies, board=board)
    if action == "hold" and goal is None:
        # No ranged target — rush toward favoured target
        favoured = _pick_favoured_target(unit, enemies)
        if favoured:
            fc = favoured.centre()
            goal = (int(round(fc[0])), int(round(fc[1])))
            return "rush", goal, None, "no charge/ranged targets, rushing toward favoured target"
    return action, goal, None, reason


def _choose_melee_holder_action(
        unit: UnitState,
        enemies: list[UnitState],
        board: Board | None = None,
) -> tuple[str, tuple[int, int] | None, UnitState | None, str]:
    """Melee holder: move to objective, charge only if target is near objective."""
    obj_idx = unit.assigned_objective
    if obj_idx < 0:
        obj_idx = 0
    obj = OBJECTIVES[obj_idx]
    obj_name = OBJ_NAMES[obj_idx] if obj_idx < len(OBJ_NAMES) else f"obj {obj_idx}"

    if not _unit_on_objective(unit, obj_idx):
        # Not on objective — check if we can charge an enemy near/between us and objective
        charge_target = _pick_holder_charge_target(unit, enemies, obj, board)
        if charge_target is not None:
            tc = charge_target.centre()
            goal = (int(round(tc[0])), int(round(tc[1])))
            return "charge", goal, charge_target, f"charging enemy near {obj_name} objective"

        # No charge — rush/advance toward objective
        d = _dist_to_objective(unit, obj_idx)
        if d <= unit.unit.advance_distance + OBJ_SEIZE_RANGE:
            return "advance", _obj_goal(obj), None, f"{obj_name} objective in advance range ({d:.0f}\" away)"
        return "rush", _obj_goal(obj), None, f"{obj_name} objective not in advance range ({d:.0f}\" away)"

    # On objective — charge only if target is within 4\" of objective
    charge_target = _pick_holder_charge_target(unit, enemies, obj, board)
    if charge_target is not None:
        tc = charge_target.centre()
        goal = (int(round(tc[0])), int(round(tc[1])))
        return "charge", goal, charge_target, f"enemy near {obj_name} objective, charging"

    # Hold and shoot
    return "hold", None, None, f"holding {obj_name} objective"


def _pick_holder_charge_target(unit: UnitState, enemies: list[UnitState],
                               obj: tuple[int, int],
                               board: Board | None = None) -> UnitState | None:
    """Pick the best charge target for a holder: must be chargeable and near the objective."""
    best = None
    best_score = -1.0
    enemy_positions = _collect_enemy_positions(enemies) if board is not None else None
    for enemy in enemies:
        if enemy.models_alive <= 0:
            continue
        if not _can_charge(unit, enemy, board, enemy_positions):
            continue
        ec = enemy.centre()
        d_to_obj = math.sqrt((ec[0] - obj[0]) ** 2 + (ec[1] - obj[1]) ** 2)
        if d_to_obj > OBJ_SEIZE_RANGE + 4.0:
            continue
        score = _melee_target_score(unit, enemy)
        # Prefer targets closer to the objective
        score /= max(d_to_obj, 1.0)
        if score > best_score:
            best_score = score
            best = enemy
    return best


# ===================================================================
# ACTIVATION PRIORITY
# ===================================================================

def activation_order(units: list[UnitState], enemies: list[UnitState] | None = None,
                     mode: str = "objectives",
                     board: Board | None = None) -> list[UnitState]:
    """Sort units for activation priority.
    In objectives mode (6-tier):
      1. Melee killers that can charge (desc expected melee damage)
      2. Melee clearers that can charge
      3. Ranged killers (desc expected ranged damage)
      4. Ranged clearers (desc expected ranged damage)
      5. Objective holders (desc distance to objective)
      6. Melee killers/clearers that cannot charge
    In kill_points mode: chargers first, then others.
    """
    active = [u for u in units if u.models_alive > 0 and not u.activated]
    if not active:
        return []

    enemies = enemies or []
    enemy_positions = _collect_enemy_positions(enemies) if board is not None else None

    def _can_charge_any(u: UnitState) -> bool:
        pref = getattr(u, 'combat_preference', 'ranged')
        if pref != 'melee':
            return False
        for e in enemies:
            if e.models_alive > 0 and _can_charge(u, e, board, enemy_positions):
                return True
        return False

    if mode == "kill_points":
        chargers = [u for u in active if _can_charge_any(u)]
        others = [u for u in active if u not in chargers]
        chargers.sort(key=lambda u: expected_melee_damage_score(u.unit), reverse=True)
        others.sort(key=lambda u: expected_damage_score(u.unit), reverse=True)
        return chargers + others

    # Objectives mode — 6 tiers
    melee_killers_charge = []
    melee_clearers_charge = []
    ranged_killers = []
    ranged_clearers = []
    holders = []
    melee_no_charge = []

    for u in active:
        pref = getattr(u, 'combat_preference', 'ranged')
        if u.ai_role in ("objective_holder", "home_objective_holder"):
            holders.append(u)
        elif pref == "melee":
            can_c = _can_charge_any(u)
            if can_c and u.ai_role == "killer":
                melee_killers_charge.append(u)
            elif can_c and u.ai_role == "objective_clearer":
                melee_clearers_charge.append(u)
            else:
                melee_no_charge.append(u)
        else:
            if u.ai_role == "killer":
                ranged_killers.append(u)
            else:
                ranged_clearers.append(u)

    melee_killers_charge.sort(key=lambda u: expected_melee_damage_score(u.unit), reverse=True)
    melee_clearers_charge.sort(key=lambda u: expected_melee_damage_score(u.unit), reverse=True)
    ranged_killers.sort(key=lambda u: expected_damage_score(u.unit), reverse=True)
    ranged_clearers.sort(key=lambda u: expected_damage_score(u.unit), reverse=True)
    holders.sort(key=lambda u: _dist_to_objective(u, u.assigned_objective), reverse=True)
    melee_no_charge.sort(key=lambda u: expected_damage_score(u.unit), reverse=True)

    return (melee_killers_charge + melee_clearers_charge +
            ranged_killers + ranged_clearers + holders + melee_no_charge)


# ===================================================================
# OBJECTIVE ASSIGNMENT & ROLE REASSIGNMENT
# ===================================================================

def _home_obj_for_player(player: str) -> int:
    """Return the home objective index for a player."""
    return HOME_OBJ_A if player == "A" else HOME_OBJ_B


def assign_objectives(units: list[UnitState]):
    """Assign objectives to objective-holder and home-objective-holder units at game start."""
    if not units:
        return
    player = units[0].owner
    own_home_obj = _home_obj_for_player(player)

    # First: assign home_objective_holders to their home objective
    for u in units:
        if u.ai_role == "home_objective_holder" and u.models_alive > 0:
            u.assigned_objective = own_home_obj

    # Then: assign regular objective_holders to non-home objectives
    holders = [u for u in units if u.ai_role == "objective_holder" and u.models_alive > 0]
    if not holders:
        return

    assigned_objs: set[int] = set()

    # Assign each holder to nearest unassigned objective, skipping own home objective
    for holder in holders:
        cx, cy = holder.centre()
        best_idx = -1
        best_d = 999999.0
        for oi, obj in enumerate(OBJECTIVES):
            if oi in assigned_objs:
                continue
            if oi == own_home_obj:
                continue  # regular holders ignore own home objective
            d = (cx - obj[0]) ** 2 + (cy - obj[1]) ** 2
            if d < best_d:
                best_d = d
                best_idx = oi
        if best_idx >= 0:
            holder.assigned_objective = best_idx
            assigned_objs.add(best_idx)
        else:
            # More holders than non-home objectives → assign to centre (index 0)
            holder.assigned_objective = 0


def reassign_roles(units: list[UnitState]):
    """Dynamic role reassignment at start of activation phase (§6.3).
    If an objective has no holder that can reach it this turn, reassign
    the nearest non-holder. Skips home objectives (handled by home_objective_holders)."""
    if not units:
        return
    player = units[0].owner
    own_home_obj = _home_obj_for_player(player)

    for oi, obj in enumerate(OBJECTIVES):
        if oi == own_home_obj:
            continue  # home objectives are not dynamically reassigned
        # Check if any holder can reach within 3" with a rush
        has_reachable_holder = False
        for u in units:
            if u.models_alive <= 0:
                continue
            if u.ai_role != "objective_holder":
                continue
            if u.assigned_objective != oi:
                continue
            cx, cy = u.centre()
            d = math.sqrt((cx - obj[0]) ** 2 + (cy - obj[1]) ** 2)
            if d <= u.unit.rush_distance + OBJ_SEIZE_RANGE:
                has_reachable_holder = True
                break

        if has_reachable_holder:
            continue

        # Find nearest non-holder to reassign
        best_unit = None
        best_d = 999999.0
        for u in units:
            if u.models_alive <= 0:
                continue
            if u.ai_role in ("objective_holder", "home_objective_holder"):
                continue
            cx, cy = u.centre()
            d = (cx - obj[0]) ** 2 + (cy - obj[1]) ** 2
            if d < best_d:
                best_d = d
                best_unit = u

        if best_unit:
            best_unit.ai_role = "objective_holder"
            best_unit.assigned_objective = oi
