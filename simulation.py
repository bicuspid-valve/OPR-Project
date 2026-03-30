"""Reusable game simulation helpers.

Extracts the core activation execution, AI decision, and round management
logic so it can be shared between:
- PlayViewer._run_ai_activation (live game)
- AI suggestion callback (replay viewer)
- "Next suggestion" forward simulation
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from board import Board, OBJECTIVES
from models import UnitState
from combat import (
    resolve_shooting, check_morale,
    resolve_melee, resolve_impact, check_melee_morale,
)
from movement import (
    execute_movement, execute_charge_movement, execute_counter_charge,
    post_melee_separation, consolidation_move,
)
from game import _sync_dead_models, _collect_enemy_positions, _kite_range_params
from ai import assign_objectives, reassign_roles

OBJ_NAMES = ["Centre", "A-side", "B-side", "Home-A", "Home-B"]


# ===================================================================
# DATA TYPES
# ===================================================================

@dataclass
class ActivationResult:
    """Result of executing a single unit activation."""
    description: str
    combat_stats: dict | None = None


# ===================================================================
# HELPERS
# ===================================================================

def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _label(unit: UnitState, labels: list[str],
           unit_to_idx: dict[int, int]) -> str:
    idx = unit_to_idx.get(id(unit))
    if idx is not None and idx < len(labels):
        return labels[idx]
    return "Unknown"


# ===================================================================
# ACTIVATION EXECUTION
# ===================================================================

def execute_activation(
    active: UnitState,
    action: str,
    goal: tuple[int, int] | None,
    charge_target: UnitState | None,
    reason: str,
    my_units: list[UnitState],
    opp_units: list[UnitState],
    board: Board,
    labels: list[str],
    unit_to_idx: dict[int, int],
    mode: str,
    resolve_shoot_target=None,
) -> ActivationResult:
    """Execute one unit's activation (charge/advance/rush/hold).

    Marks the unit as activated, executes movement and combat,
    returns a description string and optional combat stats dict.

    resolve_shoot_target: callable(active, opp_units) -> UnitState | None
        Called after movement to pick a shooting target.  Pass None to
        skip shooting entirely (e.g. rush, or no resolver available).
    """
    active.activated = True
    al = _label(active, labels, unit_to_idx)

    desc_parts: list[str] = []
    combat_stats: dict | None = None
    verbs = {"hold": "Holds", "advance": "Advances", "rush": "Rushes"}

    # ── CHARGE ──────────────────────────────────────────────────
    if action == "charge" and charge_target is not None:
        cl = _label(charge_target, labels, unit_to_idx)

        pre = active.centre()
        epos = _collect_enemy_positions(opp_units)
        execute_charge_movement(active, charge_target, board, epos)
        post = active.centre()
        md = _dist(pre, post)
        execute_counter_charge(charge_target, active, board)

        desc_parts.append(f"{al} charges {cl} {md:.0f}\" ({reason})")

        impact_info = ""
        if active.unit.impact > 0:
            imp = resolve_impact(active, charge_target)
            _sync_dead_models(charge_target, board)
            if imp['impact_hits'] > 0:
                impact_info = (f"Impact: {imp['impact_hits']} hits, "
                               f"{imp['impact_wounds']} wounds")

        charger_w = 0
        if charge_target.models_alive > 0:
            combat_stats = resolve_melee(
                active, charge_target, is_charge=True, recorded=True)
            charger_w = combat_stats['wounds_dealt'] if combat_stats else 0
            _sync_dead_models(charge_target, board)

        defender_w = 0
        if active.models_alive > 0 and charge_target.models_alive > 0:
            ds = resolve_melee(
                charge_target, active, is_strike_back=True, recorded=True)
            defender_w = ds['wounds_dealt'] if ds else 0
            _sync_dead_models(active, board)

        melee_parts = []
        if impact_info:
            melee_parts.append(impact_info)
        melee_parts.append(
            f"Melee: {charger_w} wounds dealt, {defender_w} received")

        if active.models_alive > 0 and charge_target.models_alive > 0:
            check_melee_morale(active, charger_w, defender_w)
            check_melee_morale(charge_target, defender_w, charger_w)
            _sync_dead_models(active, board)
            _sync_dead_models(charge_target, board)

        if charge_target.models_alive <= 0:
            melee_parts.append(f"{cl} destroyed!")
        if active.models_alive <= 0:
            melee_parts.append(f"{al} destroyed!")

        active.fatigued = True
        if charge_target.models_alive > 0:
            charge_target.fatigued = True

        if active.models_alive > 0 and charge_target.models_alive > 0:
            epos = _collect_enemy_positions(opp_units)
            post_melee_separation(active, charge_target, board, epos)
        elif active.models_alive > 0:
            consolidation_move(active, board, opp_units, OBJECTIVES, mode)
        elif charge_target.models_alive > 0:
            consolidation_move(charge_target, board, my_units,
                               OBJECTIVES, mode)

        desc_parts.append("-- " + ", ".join(melee_parts))
        if combat_stats is None:
            combat_stats = {}
        combat_stats['combat_type'] = 'melee'
        combat_stats['charger_wounds'] = charger_w
        combat_stats['defender_wounds'] = defender_w
        if impact_info:
            combat_stats['impact_info'] = impact_info

    # ── ADVANCE / RUSH ──────────────────────────────────────────
    elif action in ("advance", "rush") and goal is not None:
        budget = (active.unit.advance_distance if action == "advance"
                  else active.unit.rush_distance)
        pre = active.centre()
        epos = _collect_enemy_positions(opp_units)
        rt, wr = _kite_range_params(active, opp_units, reason)
        execute_movement(active, goal, budget, board, epos,
                         flying=active.unit.flying,
                         range_target=rt, weapon_range=wr)
        post = active.centre()
        md = _dist(pre, post)
        desc_parts.append(
            f"{al} {verbs.get(action, action)} {md:.0f}\" ({reason})")

        if action != "rush":
            combat_stats, shot_desc = _resolve_shooting_phase(
                active, opp_units, board, labels, unit_to_idx,
                resolve_shoot_target)
            if shot_desc:
                desc_parts.append(shot_desc)

    # ── HOLD ────────────────────────────────────────────────────
    elif action == "hold":
        desc_parts.append(f"{al} Holds ({reason})")
        combat_stats, shot_desc = _resolve_shooting_phase(
            active, opp_units, board, labels, unit_to_idx,
            resolve_shoot_target)
        if shot_desc:
            desc_parts.append(shot_desc)

    return ActivationResult(
        description=" ".join(desc_parts),
        combat_stats=combat_stats,
    )


def _resolve_shooting_phase(
    active: UnitState,
    opp_units: list[UnitState],
    board: Board,
    labels: list[str],
    unit_to_idx: dict[int, int],
    resolve_shoot_target,
) -> tuple[dict | None, str]:
    """Handle the shooting sub-phase after hold/advance.

    Returns (combat_stats, description_fragment).
    """
    if active.shaken:
        active.shaken = False
        return None, "(was Shaken, recovers)"

    if resolve_shoot_target is None:
        return None, "(no targets in range)"

    tgt = resolve_shoot_target(active, opp_units)
    if tgt is None:
        return None, "(no targets in range)"

    tl = _label(tgt, labels, unit_to_idx)
    before_alive = tgt.models_alive
    combat_stats = resolve_shooting(active, tgt, recorded=True)
    check_morale(tgt)
    _sync_dead_models(tgt, board)
    killed = before_alive - tgt.models_alive

    if killed > 0:
        if tgt.models_alive <= 0:
            desc = f"and shoots {tl}, destroying the unit!"
        else:
            desc = (f"and shoots {tl}, killing {killed} "
                    f"model{'s' if killed != 1 else ''}")
    else:
        desc = f"and shoots {tl}, no casualties"
    return combat_stats, desc


# ===================================================================
# AI DECISION
# ===================================================================

def get_ai_decision(
    model,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    *,
    apply_tactical_fn,
    plan_fn=None,
    use_planning: bool = False,
    planning_params: dict | None = None,
    fr_friendly=None, fm_friendly=None,
    fr_enemy=None, fm_enemy=None,
    pts_friendly: int = 0, pts_enemy: int = 0,
    # Planning needs both sides' matchup data
    units_a=None, units_b=None,
    fr_a=None, fm_a=None,
    fr_b=None, fm_b=None,
    pts_a: int = 0, pts_b: int = 0,
    mode: str = "objectives",
):
    """Get the AI's decision for one activation.

    Returns (active, target_ranking, action, goal, charge_target,
             reason, ml_assessment).
    """
    if use_planning and plan_fn is not None:
        (active, target_ranking, action, goal,
         charge_target, reason, planning_cands) = plan_fn(
            model, friendly_units, enemy_units, round_num,
            board, player,
            units_a if units_a is not None else friendly_units,
            units_b if units_b is not None else enemy_units,
            player == "A", mode,
            friendly_ranged_matchups=fr_friendly,
            friendly_melee_matchups=fm_friendly,
            enemy_ranged_matchups=fr_enemy,
            enemy_melee_matchups=fm_enemy,
            total_friendly_points=pts_friendly,
            total_enemy_points=pts_enemy,
            fr_a=fr_a, fm_a=fm_a, fr_b=fr_b, fm_b=fm_b,
            pts_a=pts_a, pts_b=pts_b,
            planning_params=planning_params,
        )
        ml_assessment = ({'planning_candidates': planning_cands}
                         if planning_cands else None)
    else:
        (active, target_ranking, action, goal,
         charge_target, reason, assess) = apply_tactical_fn(
            model, friendly_units, enemy_units, round_num,
            board, player,
            friendly_ranged_matchups=fr_friendly,
            friendly_melee_matchups=fm_friendly,
            enemy_ranged_matchups=fr_enemy,
            enemy_melee_matchups=fm_enemy,
            total_friendly_points=pts_friendly,
            total_enemy_points=pts_enemy,
        )
        ml_assessment = assess

    return (active, target_ranking, action, goal,
            charge_target, reason, ml_assessment)


# ===================================================================
# ROUND MANAGEMENT
# ===================================================================

def check_game_over(units_a: list[UnitState],
                    units_b: list[UnitState]) -> bool:
    """True if either side has been completely wiped out."""
    a_alive = any(u.models_alive > 0 for u in units_a)
    b_alive = any(u.models_alive > 0 for u in units_b)
    return not a_alive or not b_alive


def check_round_over(
    units_a: list[UnitState],
    units_b: list[UnitState],
) -> tuple[bool, bool, bool | None, bool]:
    """Check whether both sides have exhausted their activations.

    Returns (a_done, b_done, a_finished_first_or_None, round_over).
    *a_finished_first* is None when both finish simultaneously.
    """
    a_can = any(u.models_alive > 0 and not u.activated for u in units_a)
    b_can = any(u.models_alive > 0 and not u.activated for u in units_b)
    a_done = not a_can
    b_done = not b_can
    if a_done and not b_done:
        aff: bool | None = True
    elif b_done and not a_done:
        aff = False
    else:
        aff = None
    return a_done, b_done, aff, a_done and b_done


def start_round(
    units_a: list[UnitState],
    units_b: list[UnitState],
    round_num: int,
    mode: str,
    a_first: bool = True,
    a_finished_first: bool = True,
    ml_sides: frozenset[str] = frozenset(),
) -> bool:
    """Reset activations and assign objectives for a new round.

    *ml_sides* is a frozenset of ``"A"``/``"B"`` strings indicating which
    sides are ML-controlled.  Role reassignment is skipped for ML sides
    (the model handles its own objective targeting).

    Returns *current_is_a* — True if Player A acts first this round.
    """
    for u in units_a:
        u.activated = False
        u.fatigued = False
    for u in units_b:
        u.activated = False
        u.fatigued = False

    if mode != "kill_points":
        if "A" not in ml_sides:
            reassign_roles(units_a)
        if "B" not in ml_sides:
            reassign_roles(units_b)

    return a_first if round_num == 0 else a_finished_first


def end_round(
    board: Board,
    units_a: list[UnitState],
    units_b: list[UnitState],
    round_num: int,
    mode: str,
) -> str:
    """Update objectives and return end-of-round description."""
    if mode != "kill_points":
        board.update_objectives(units_a, units_b)

    if mode == "kill_points":
        a_kp = sum(u.unit.points for u in units_b if u.models_alive <= 0)
        b_kp = sum(u.unit.points for u in units_a if u.models_alive <= 0)
        return (f"End of Round {round_num + 1} -- "
                f"Kill Points: A: {a_kp}pts, B: {b_kp}pts")
    else:
        obj_parts = []
        for oi, ctrl in enumerate(board.objective_control):
            if ctrl:
                obj_parts.append(f"{OBJ_NAMES[oi]}: Player {ctrl}")
            else:
                obj_parts.append(f"{OBJ_NAMES[oi]}: Neutral")
        return (f"End of Round {round_num + 1} -- "
                f"{', '.join(obj_parts)}")


def score_game(
    board: Board,
    units_a: list[UnitState],
    units_b: list[UnitState],
    mode: str,
) -> str:
    """Determine the winner. Returns ``'A'``, ``'B'``, or ``'draw'``."""
    if mode == "kill_points":
        a_kp = sum(u.unit.points for u in units_b if u.models_alive <= 0)
        b_kp = sum(u.unit.points for u in units_a if u.models_alive <= 0)
        if a_kp > b_kp:
            return "A"
        if b_kp > a_kp:
            return "B"
        return "draw"
    a_objs = board.count_objectives("A")
    b_objs = board.count_objectives("B")
    if a_objs > b_objs:
        return "A"
    if b_objs > a_objs:
        return "B"
    return "draw"
