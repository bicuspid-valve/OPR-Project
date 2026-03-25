"""Full game loop: deployment, 4-round structure, alternating activations, objective scoring."""
from __future__ import annotations

import math
import random

from board import (
    Board, COLS, ROWS, OBJECTIVES,
    DEPLOY_A_FRONT_ROW, DEPLOY_B_FRONT_ROW,
    DEPLOY_A_MIN_ROW, DEPLOY_A_MAX_ROW,
    DEPLOY_B_MIN_ROW, DEPLOY_B_MAX_ROW,
    SCOUT_A_ROW, SCOUT_B_ROW,
    HOME_OBJ_A, HOME_OBJ_B,
)
from models import ResolvedUnit, UnitState
from combat import (
    resolve_shooting, check_morale,
    resolve_melee, resolve_impact, check_melee_morale,
)
from movement import (
    execute_movement, is_in_exclusion_zone,
    execute_charge_movement, execute_counter_charge,
    post_melee_separation, consolidation_move,
)
from ai import (
    pick_target, choose_action_and_goal, activation_order,
    assign_objectives, reassign_roles,
)


def _is_tactical_model(model) -> bool:
    """Return True if model is a TacticalModel (per-activation), False for StrategicModel."""
    return hasattr(model, 'unit_selection_head')


# ===================================================================
# DEPLOYMENT
# ===================================================================

def _place_unit_at(unit: UnitState, col: int, row: int, board: Board,
                   enemy_positions: set[tuple[int, int]] | None = None):
    """Place a multi-model unit starting at (col, row), spreading laterally then backward.
    Respects 1\" exclusion zone from enemy_positions if provided."""
    n = unit.unit.models
    unit.positions = []
    is_player_a = (unit.owner == "A")
    row_dir = -1 if is_player_a else 1  # backwards into deployment zone
    ep = enemy_positions or set()

    placed = 0
    # Try placing models in expanding rings from the start position
    for offset in range(max(COLS, ROWS)):
        if placed >= n:
            break
        for dc in range(-offset, offset + 1):
            if placed >= n:
                break
            for dr_mult in ([0] if offset == 0 else range(-offset, offset + 1)):
                if placed >= n:
                    break
                if offset > 0 and abs(dc) != offset and abs(dr_mult) != offset:
                    continue
                nc = col + dc
                # For backward expansion, prefer going backward
                nr = row + dr_mult * (-row_dir if dr_mult != 0 else 0)
                if offset == 0:
                    nr = row
                else:
                    nr = row + dr_mult
                if board.is_free(nc, nr):
                    # Check deployment zone bounds
                    if is_player_a and not (DEPLOY_A_MIN_ROW <= nr <= max(DEPLOY_A_MAX_ROW, SCOUT_A_ROW)):
                        continue
                    if not is_player_a and not (min(DEPLOY_B_MIN_ROW, SCOUT_B_ROW) <= nr <= DEPLOY_B_MAX_ROW):
                        continue
                    # Check exclusion zone
                    if ep and is_in_exclusion_zone(nc, nr, ep):
                        continue
                    unit.positions.append((nc, nr))
                    board.place(nc, nr)
                    placed += 1

    # Fallback: if we couldn't place all models, search more broadly
    if placed < n:
        if is_player_a:
            row_range = range(DEPLOY_A_MIN_ROW, DEPLOY_A_MAX_ROW + 1)
        else:
            row_range = range(DEPLOY_B_MIN_ROW, DEPLOY_B_MAX_ROW + 1)
        for r in row_range:
            for c in range(COLS):
                if placed >= n:
                    break
                if board.is_free(c, r) and not (ep and is_in_exclusion_zone(c, r, ep)):
                    unit.positions.append((c, r))
                    board.place(c, r)
                    placed += 1
            if placed >= n:
                break

    unit.models_alive = placed
    unit.weapons_per_model = unit.weapons_per_model[:placed]
    if unit.wounds_per_model:
        unit.wounds_per_model = unit.wounds_per_model[:placed]


def _compute_deploy_columns(count: int, player: str = "A") -> list[int]:
    """Compute evenly spaced column positions across the 72-column width.

    Player B gets 180°-mirrored columns (COLS-1-col) so that both sides
    occupy rotationally symmetric positions, eliminating the side-info
    leak in the ML feature encoding.
    """
    if count == 0:
        return []
    if count == 1:
        col = 36
        return [COLS - 1 - col if player == "B" else col]
    spacing = COLS / (count + 1)
    cols = [int(round(spacing * (i + 1))) for i in range(count)]
    if player == "B":
        cols = [COLS - 1 - c for c in cols]
    return cols


def _deploy_non_scouts(units: list[UnitState], player: str, board: Board):
    """Deploy non-scout units for a player."""
    is_a = (player == "A")
    front_row = DEPLOY_A_FRONT_ROW if is_a else DEPLOY_B_FRONT_ROW
    home_obj = OBJECTIVES[HOME_OBJ_A if is_a else HOME_OBJ_B]

    non_scouts = [u for u in units if not u.unit.scout]
    home_holders = [u for u in non_scouts if u.ai_role == "home_objective_holder"]
    holders = [u for u in non_scouts if u.ai_role == "objective_holder"]
    others = [u for u in non_scouts
              if u.ai_role not in ("objective_holder", "home_objective_holder")]

    # Deploy home_objective_holders on the home objective
    for u in home_holders:
        _place_unit_at(u, home_obj[0], home_obj[1], board)

    holder_cols = _compute_deploy_columns(len(holders), player)
    other_cols = _compute_deploy_columns(len(others), player)

    for i, u in enumerate(holders):
        col = holder_cols[i] if i < len(holder_cols) else (36 if is_a else COLS - 1 - 36)
        _place_unit_at(u, col, front_row, board)

    for i, u in enumerate(others):
        col = other_cols[i] if i < len(other_cols) else (36 if is_a else COLS - 1 - 36)
        _place_unit_at(u, col, front_row, board)


def _deploy_scouts(units: list[UnitState], player: str, board: Board,
                   enemy_positions: set[tuple[int, int]]):
    """Deploy scout units forward, respecting 1\" exclusion from enemies."""
    is_a = (player == "A")
    scout_row = SCOUT_A_ROW if is_a else SCOUT_B_ROW

    scouts = [u for u in units if u.unit.scout]
    scout_cols = _compute_deploy_columns(len(scouts), player)
    for i, u in enumerate(scouts):
        col = scout_cols[i] if i < len(scout_cols) else (36 if is_a else COLS - 1 - 36)
        _place_unit_at(u, col, scout_row, board, enemy_positions=enemy_positions)


def deploy_armies(units_a: list[UnitState], units_b: list[UnitState],
                  board: Board):
    """Deploy both armies with proper scout ordering per §3.3.
    Non-scouts deploy first (order randomised), then scouts deploy in
    random order — each batch excludes the other side's already-placed units."""
    _deploy_non_scouts(units_a, "A", board)
    _deploy_non_scouts(units_b, "B", board)

    # Randomise which side deploys scouts first so neither has an advantage
    if random.random() < 0.5:
        first_units, first_player = units_a, "A"
        second_units, second_player = units_b, "B"
    else:
        first_units, first_player = units_b, "B"
        second_units, second_player = units_a, "A"

    enemy_pos = _collect_enemy_positions(second_units)
    _deploy_scouts(first_units, first_player, board, enemy_pos)

    enemy_pos = _collect_enemy_positions(first_units)
    _deploy_scouts(second_units, second_player, board, enemy_pos)


# ===================================================================
# GAME LOOP
# ===================================================================

def _collect_enemy_positions(units: list[UnitState]) -> set[tuple[int, int]]:
    """Collect all occupied positions of a player's units."""
    positions: set[tuple[int, int]] = set()
    for u in units:
        for pos in u.alive_positions():
            positions.add(pos)
    return positions


def _kite_range_params(active: UnitState, enemies: list[UnitState],
                       reason: str) -> tuple[tuple[int, int] | None, float]:
    """Return (range_target, weapon_range) for kite moves, else (None, 0).

    Finds the nearest alive enemy centre so that execute_movement can nudge
    models that ended up outside weapon range back toward the target."""
    if "kite" not in reason:
        return None, 0
    wr = active.unit.max_weapon_range
    if wr <= 0:
        return None, 0
    ac = active.centre()
    best_target = None
    best_dsq = float('inf')
    for e in enemies:
        if e.models_alive <= 0:
            continue
        ec = e.centre()
        dsq = (ac[0] - ec[0]) ** 2 + (ac[1] - ec[1]) ** 2
        if dsq < best_dsq:
            best_dsq = dsq
            best_target = ec
    if best_target is None:
        return None, 0
    return (int(round(best_target[0])), int(round(best_target[1]))), float(wr)


def simulate_game(army_a: list[ResolvedUnit],
                  army_b: list[ResolvedUnit],
                  mode: str = "objectives",
                  states_a: list[UnitState] | None = None,
                  states_b: list[UnitState] | None = None,
                  ml_model_a=None,
                  ml_model_b=None,
                  ml_sampling=False,
                  ml_batch_tactical=True,
                  ml_coroutine_mode=False,
                  ml_planning=False,
                  planning_params: dict | None = None):
    """Play one grid-based tactical game.

    Returns 'A', 'B', or 'draw' when ml_coroutine_mode=False (default).
    Returns a generator when ml_coroutine_mode=True — the generator yields
    InferenceRequest and receives InferenceResult via .send(); final game
    result is delivered via StopIteration.value.

    ml_planning: True (both sides), "A"/"B" (one side only), or False.
    planning_params: optional dict of planning parameters (K, C, M, N).
    """
    if ml_coroutine_mode:
        return _simulate_game_coroutine(
            army_a, army_b, mode=mode, states_a=states_a, states_b=states_b,
            ml_model_a=ml_model_a, ml_model_b=ml_model_b)
    return _simulate_game_impl(
        army_a, army_b, mode=mode, states_a=states_a, states_b=states_b,
        ml_model_a=ml_model_a, ml_model_b=ml_model_b,
        ml_sampling=ml_sampling,
        ml_planning=ml_planning, planning_params=planning_params)


def _simulate_game_impl(army_a, army_b, mode="objectives",
                        states_a=None, states_b=None,
                        ml_model_a=None, ml_model_b=None,
                        ml_sampling=False,
                        ml_planning=False, planning_params=None,
                        _tactical_inference_fn=None) -> str:
    """Internal: standard (non-generator) game loop.

    _tactical_inference_fn: optional callable(my_units, opp_units, round_num, board,
        player, my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts)
        -> (active, target_ranking, action, goal, charge_target, reason).
        When provided, replaces the apply_tactical_model call for per-activation tactical
        decisions.  Used by _simulate_game_coroutine to inject yield-based inference.
    """
    board = Board()
    is_kill_points = (mode == "kill_points")

    # Create unit states (or use pre-built ones with hero merging)
    if states_a is not None:
        units_a = states_a
    else:
        units_a = [UnitState(u) for u in army_a]
        for u in units_a:
            u.owner = "A"

    if states_b is not None:
        units_b = states_b
    else:
        units_b = [UnitState(u) for u in army_b]
        for u in units_b:
            u.owner = "B"

    # Deployment
    deploy_armies(units_a, units_b, board)

    # ML setup: lazy imports and precomputed damage bases
    use_ml = ml_model_a is not None or ml_model_b is not None
    _tactical_a = ml_model_a is not None and _is_tactical_model(ml_model_a)
    _tactical_b = ml_model_b is not None and _is_tactical_model(ml_model_b)
    if use_ml:
        if ml_sampling:
            from ml_integration import ml_activation_order
            from ml_integration import apply_model_outputs_sampling as apply_model_outputs
            from ml_features import precompute_damage
            if _tactical_a or _tactical_b:
                from ml_integration_tactical import (
                    apply_tactical_model_sampling as apply_tactical_model,
                    pick_target_from_ranking,
                )
        else:
            from ml_integration import apply_model_outputs, ml_activation_order
            from ml_features import precompute_damage
            if _tactical_a or _tactical_b:
                from ml_integration_tactical import (
                    apply_tactical_model,
                    pick_target_from_ranking,
                )
        if ml_planning and (_tactical_a or _tactical_b):
            from ml_planning import plan_activation as _plan_activation
        _fr_a, _fm_a = precompute_damage([u.unit for u in units_a],
                                         [u.unit for u in units_b])
        _fr_b, _fm_b = precompute_damage([u.unit for u in units_b],
                                         [u.unit for u in units_a])
        _pts_a = sum(u.unit.points for u in units_a)
        _pts_b = sum(u.unit.points for u in units_b)

    # Assign objectives to holders (objectives mode only)
    # Skip for ML-controlled sides — the model sets objectives at round 1
    if not is_kill_points:
        if ml_model_a is None:
            assign_objectives(units_a)
        if ml_model_b is None:
            assign_objectives(units_b)

    # Determine first player (coin flip)
    a_first = random.random() < 0.5
    a_finished_first = a_first  # Track who finishes first each round

    # 4 rounds
    for round_num in range(4):
        # Reset activations and fatigue
        for u in units_a:
            u.activated = False
            u.fatigued = False
        for u in units_b:
            u.activated = False
            u.fatigued = False

        # First player for this round
        if round_num == 0:
            current_is_a = a_first
        else:
            current_is_a = a_finished_first

        # Role reassignment (objectives mode only, skip for ML-controlled sides)
        if not is_kill_points:
            if ml_model_a is None:
                reassign_roles(units_a)
            if ml_model_b is None:
                reassign_roles(units_b)

        # ML forward pass at round start
        target_mults_a = None
        target_mults_b = None
        if ml_model_a is not None and not _tactical_a:
            target_mults_a, _ = apply_model_outputs(
                ml_model_a, units_a, units_b, round_num + 1, board, "A",
                friendly_ranged_matchups=_fr_a, friendly_melee_matchups=_fm_a,
                enemy_ranged_matchups=_fr_b, enemy_melee_matchups=_fm_b,
                total_friendly_points=_pts_a, total_enemy_points=_pts_b,
            )
        if ml_model_b is not None and not _tactical_b:
            target_mults_b, _ = apply_model_outputs(
                ml_model_b, units_b, units_a, round_num + 1, board, "B",
                friendly_ranged_matchups=_fr_b, friendly_melee_matchups=_fm_b,
                enemy_ranged_matchups=_fr_a, enemy_melee_matchups=_fm_a,
                total_friendly_points=_pts_b, total_enemy_points=_pts_a,
            )

        # Track who finishes first
        a_done = False
        b_done = False
        a_finished_first = True  # default

        # Alternating activations
        while True:
            if current_is_a:
                my_units, opp_units = units_a, units_b
                my_ml = ml_model_a
                my_mults = target_mults_a
                my_tactical = _tactical_a
                _my_fr, _my_fm = (_fr_a, _fm_a) if use_ml else (None, None)
                _opp_fr, _opp_fm = (_fr_b, _fm_b) if use_ml else (None, None)
                my_pts, opp_pts = (_pts_a, _pts_b) if use_ml else (0, 0)
                my_player = "A"
            else:
                my_units, opp_units = units_b, units_a
                my_ml = ml_model_b
                my_mults = target_mults_b
                my_tactical = _tactical_b
                _my_fr, _my_fm = (_fr_b, _fm_b) if use_ml else (None, None)
                _opp_fr, _opp_fm = (_fr_a, _fm_a) if use_ml else (None, None)
                my_pts, opp_pts = (_pts_b, _pts_a) if use_ml else (0, 0)
                my_player = "B"

            # ML tactical decision state (set when ML tactical path is used)
            _ml_tac_decision = False
            _ml_target_ranking: list[int] = []
            _ml_action = "hold"
            _ml_goal = None
            _ml_charge_target = None
            _ml_reason = ""

            # Tactical model: per-activation
            if my_ml is not None and my_tactical:
                if _tactical_inference_fn is not None:
                    # Injected inference (used by coroutine batching)
                    active, _ml_target_ranking, _ml_action, _ml_goal, _ml_charge_target, _ml_reason = (
                        _tactical_inference_fn(
                            my_units, opp_units, round_num + 1, board, my_player,
                            _my_fr, _my_fm, _opp_fr, _opp_fm, my_pts, opp_pts))
                    _ml_tac_decision = active is not None
                elif ml_planning and (ml_planning is True or ml_planning == my_player):
                    # Monte Carlo planning (eval only)
                    active, _ml_target_ranking, _ml_action, _ml_goal, _ml_charge_target, _ml_reason, _ = (
                        _plan_activation(
                            my_ml, my_units, opp_units, round_num + 1, board, my_player,
                            units_a, units_b, current_is_a, mode,
                            friendly_ranged_matchups=_my_fr, friendly_melee_matchups=_my_fm,
                            enemy_ranged_matchups=_opp_fr, enemy_melee_matchups=_opp_fm,
                            total_friendly_points=my_pts, total_enemy_points=opp_pts,
                            fr_a=_fr_a, fm_a=_fm_a, fr_b=_fr_b, fm_b=_fm_b,
                            pts_a=_pts_a, pts_b=_pts_b,
                            planning_params=planning_params,
                        ))
                    _ml_tac_decision = active is not None
                else:
                    # Per-activation path (eval argmax / training sampling)
                    active, _ml_target_ranking, _ml_action, _ml_goal, _ml_charge_target, _ml_reason, _ = (
                        apply_tactical_model(
                            my_ml, my_units, opp_units, round_num + 1, board, my_player,
                            friendly_ranged_matchups=_my_fr, friendly_melee_matchups=_my_fm,
                            enemy_ranged_matchups=_opp_fr, enemy_melee_matchups=_opp_fm,
                            total_friendly_points=my_pts, total_enemy_points=opp_pts,
                        ))
                    _ml_tac_decision = active is not None
            elif my_ml is not None:
                # Strategic model: use pre-computed activation order
                ordered = ml_activation_order(my_units)
                active = ordered[0] if ordered else None
            else:
                ordered = activation_order(my_units, enemies=opp_units, mode=mode)
                active = ordered[0] if ordered else None

            if active is None:
                # Current player has no more units to activate
                if current_is_a:
                    a_done = True
                    if not b_done:
                        a_finished_first = True
                else:
                    b_done = True
                    if not a_done:
                        a_finished_first = False

                if a_done and b_done:
                    break
                current_is_a = not current_is_a
                continue

            active.activated = True

            # Determine action and goal
            if _ml_tac_decision:
                action, goal, charge_target, _reason = _ml_action, _ml_goal, _ml_charge_target, _ml_reason
            else:
                action, goal, charge_target, _reason = choose_action_and_goal(
                    active, opp_units, board, mode=mode,
                    target_multipliers=my_mults)

            if action == "charge" and charge_target is not None:
                # Full charge sequence
                enemy_positions = _collect_enemy_positions(opp_units)
                execute_charge_movement(active, charge_target, board, enemy_positions)
                execute_counter_charge(charge_target, active, board)

                # Impact
                if active.unit.impact > 0:
                    resolve_impact(active, charge_target)
                    _sync_dead_models(charge_target, board)

                # Charger swings
                charger_wounds = 0
                if charge_target.models_alive > 0:
                    charger_wounds = resolve_melee(active, charge_target, is_charge=True) or 0
                    _sync_dead_models(charge_target, board)

                # Defender strikes back
                defender_wounds = 0
                if active.models_alive > 0 and charge_target.models_alive > 0:
                    defender_wounds = resolve_melee(charge_target, active, is_strike_back=True) or 0
                    _sync_dead_models(active, board)

                # Melee morale
                if active.models_alive > 0 and charge_target.models_alive > 0:
                    check_melee_morale(active, charger_wounds, defender_wounds)
                    check_melee_morale(charge_target, defender_wounds, charger_wounds)
                    _sync_dead_models(active, board)
                    _sync_dead_models(charge_target, board)

                # Fatigue
                active.fatigued = True
                if charge_target.models_alive > 0:
                    charge_target.fatigued = True

                # Post-melee separation or consolidation
                if active.models_alive > 0 and charge_target.models_alive > 0:
                    enemy_positions = _collect_enemy_positions(opp_units)
                    post_melee_separation(active, charge_target, board, enemy_positions)
                elif active.models_alive > 0:
                    from board import OBJECTIVES
                    consolidation_move(active, board, opp_units, OBJECTIVES, mode)
                elif charge_target.models_alive > 0:
                    from board import OBJECTIVES
                    consolidation_move(charge_target, board, my_units, OBJECTIVES, mode)

            elif action in ("advance", "rush") and goal is not None:
                # Execute movement
                budget = active.unit.advance_distance if action == "advance" else active.unit.rush_distance
                enemy_positions = _collect_enemy_positions(opp_units)
                rt, wr = _kite_range_params(active, opp_units, _reason)
                execute_movement(active, goal, budget, board, enemy_positions,
                                 flying=active.unit.flying,
                                 range_target=rt, weapon_range=wr)

                # Execute shooting (not if rushing or shaken)
                if action != "rush":
                    if active.shaken:
                        active.shaken = False
                    else:
                        if _ml_tac_decision:
                            target = pick_target_from_ranking(active, opp_units, _ml_target_ranking)
                        else:
                            target = pick_target(active, opp_units,
                                                 target_multipliers=my_mults)
                        if target is not None:
                            resolve_shooting(active, target)
                            check_morale(target)
                            _sync_dead_models(target, board)

            elif action == "hold":
                # Execute shooting
                if active.shaken:
                    active.shaken = False
                else:
                    if _ml_tac_decision:
                        target = pick_target_from_ranking(active, opp_units, _ml_target_ranking)
                    else:
                        target = pick_target(active, opp_units,
                                             target_multipliers=my_mults)
                    if target is not None:
                        resolve_shooting(active, target)
                        check_morale(target)
                        _sync_dead_models(target, board)

            # Check if opponent army destroyed
            opp_alive = any(u.models_alive > 0 for u in opp_units)
            if not opp_alive:
                break

            current_is_a = not current_is_a

        # End of round: check objective control (objectives mode only)
        if not is_kill_points:
            board.update_objectives(units_a, units_b)

    # Game end scoring
    if is_kill_points:
        # Kill points: sum points of fully destroyed enemy units
        a_kill_pts = sum(u.unit.points for u in units_b if u.models_alive <= 0)
        b_kill_pts = sum(u.unit.points for u in units_a if u.models_alive <= 0)
        if a_kill_pts > b_kill_pts:
            return "A"
        elif b_kill_pts > a_kill_pts:
            return "B"
        return "draw"
    else:
        a_objs = board.count_objectives("A")
        b_objs = board.count_objectives("B")
        if a_objs > b_objs:
            return "A"
        elif b_objs > a_objs:
            return "B"
        return "draw"


def _simulate_game_coroutine(army_a, army_b, mode="objectives",
                              states_a=None, states_b=None,
                              ml_model_a=None, ml_model_b=None):
    """Generator variant of simulate_game for cross-game batched inference.

    Yields InferenceRequest at each per-activation tactical decision point.
    Receives InferenceResult via .send(). Final result ('A'/'B'/'draw') is
    delivered via StopIteration.value.
    """
    import torch as _torch
    from ml_features import encode_state_tactical as _encode_tac
    from ml_integration_tactical import (
        InferenceRequest, decode_tactical_result, pick_target_from_ranking,
        MAX_UNITS_PER_SIDE as _MAX_UNITS,
        _get_model_space_positions as _ms_pos,
        _get_movement_budgets as _mv_budgets,
        _get_max_weapon_ranges as _mwr,
    )

    board = Board()
    is_kill_points = (mode == "kill_points")

    if states_a is not None:
        units_a = states_a
    else:
        units_a = [UnitState(u) for u in army_a]
        for u in units_a:
            u.owner = "A"

    if states_b is not None:
        units_b = states_b
    else:
        units_b = [UnitState(u) for u in army_b]
        for u in units_b:
            u.owner = "B"

    deploy_armies(units_a, units_b, board)

    _tactical_a = ml_model_a is not None and _is_tactical_model(ml_model_a)
    _tactical_b = ml_model_b is not None and _is_tactical_model(ml_model_b)

    from ml_integration import apply_model_outputs, ml_activation_order
    from ml_features import precompute_damage

    _fr_a, _fm_a = precompute_damage([u.unit for u in units_a],
                                     [u.unit for u in units_b])
    _fr_b, _fm_b = precompute_damage([u.unit for u in units_b],
                                     [u.unit for u in units_a])
    _pts_a = sum(u.unit.points for u in units_a)
    _pts_b = sum(u.unit.points for u in units_b)

    if not is_kill_points:
        if ml_model_a is None:
            assign_objectives(units_a)
        if ml_model_b is None:
            assign_objectives(units_b)

    a_first = random.random() < 0.5
    a_finished_first = a_first

    for round_num in range(4):
        for u in units_a:
            u.activated = False
            u.fatigued = False
        for u in units_b:
            u.activated = False
            u.fatigued = False

        current_is_a = a_first if round_num == 0 else a_finished_first

        if not is_kill_points:
            if ml_model_a is None:
                reassign_roles(units_a)
            if ml_model_b is None:
                reassign_roles(units_b)

        # Strategic model round-start pass (for non-tactical sides)
        target_mults_a = None
        target_mults_b = None
        if ml_model_a is not None and not _tactical_a:
            target_mults_a, _ = apply_model_outputs(
                ml_model_a, units_a, units_b, round_num + 1, board, "A",
                friendly_ranged_matchups=_fr_a, friendly_melee_matchups=_fm_a,
                enemy_ranged_matchups=_fr_b, enemy_melee_matchups=_fm_b,
                total_friendly_points=_pts_a, total_enemy_points=_pts_b,
            )
        if ml_model_b is not None and not _tactical_b:
            target_mults_b, _ = apply_model_outputs(
                ml_model_b, units_b, units_a, round_num + 1, board, "B",
                friendly_ranged_matchups=_fr_b, friendly_melee_matchups=_fm_b,
                enemy_ranged_matchups=_fr_a, enemy_melee_matchups=_fm_a,
                total_friendly_points=_pts_b, total_enemy_points=_pts_a,
            )

        a_done = False
        b_done = False
        a_finished_first = True

        while True:
            if current_is_a:
                my_units, opp_units = units_a, units_b
                my_ml = ml_model_a
                my_mults = target_mults_a
                my_tactical = _tactical_a
                _my_fr, _my_fm = _fr_a, _fm_a
                _opp_fr, _opp_fm = _fr_b, _fm_b
                my_pts, opp_pts = _pts_a, _pts_b
                my_player = "A"
            else:
                my_units, opp_units = units_b, units_a
                my_ml = ml_model_b
                my_mults = target_mults_b
                my_tactical = _tactical_b
                _my_fr, _my_fm = _fr_b, _fm_b
                _opp_fr, _opp_fm = _fr_a, _fm_a
                my_pts, opp_pts = _pts_b, _pts_a
                my_player = "B"

            # ML tactical decision state
            _ml_tac_decision = False
            _ml_target_ranking: list[int] = []

            # Tactical per-activation: encode, yield, decode
            if my_ml is not None and my_tactical:
                _alive = [
                    (i < len(my_units)
                     and my_units[i].models_alive > 0
                     and not my_units[i].activated)
                    for i in range(_MAX_UNITS)
                ]
                if any(_alive):
                    _mask = _torch.tensor(_alive, dtype=_torch.bool)
                    _enemy_mask = _torch.tensor(
                        [(i < len(opp_units) and opp_units[i].models_alive > 0)
                         for i in range(_MAX_UNITS)],
                        dtype=_torch.bool,
                    )
                    _vec = _encode_tac(
                        my_units, opp_units, round_num + 1, board, my_player,
                        friendly_ranged_matchups=_my_fr,
                        friendly_melee_matchups=_my_fm,
                        enemy_ranged_matchups=_opp_fr,
                        enemy_melee_matchups=_opp_fm,
                        total_friendly_points=my_pts,
                        total_enemy_points=opp_pts,
                    )
                    _f_pos = _ms_pos(my_units, my_player)
                    _e_pos = _ms_pos(opp_units, my_player)
                    _adv_d, _rush_d = _mv_budgets(my_units)
                    _mwr_d = _mwr(my_units)
                    _ir = yield InferenceRequest(
                        _vec, _mask, _enemy_mask, my_player,
                        _f_pos, _e_pos, _adv_d, _rush_d, _mwr_d,
                    )
                    active, _ml_target_ranking, _ml_action, _ml_goal, _ml_charge_target, _ml_reason = (
                        decode_tactical_result(_ir, my_units, opp_units, board, my_player))
                    _ml_tac_decision = active is not None
                else:
                    active = None
            elif my_ml is not None:
                ordered = ml_activation_order(my_units)
                active = ordered[0] if ordered else None
            else:
                ordered = activation_order(my_units, enemies=opp_units, mode=mode)
                active = ordered[0] if ordered else None

            if active is None:
                if current_is_a:
                    a_done = True
                    if not b_done:
                        a_finished_first = True
                else:
                    b_done = True
                    if not a_done:
                        a_finished_first = False

                if a_done and b_done:
                    break
                current_is_a = not current_is_a
                continue

            active.activated = True

            if _ml_tac_decision:
                action, goal, charge_target, _reason = _ml_action, _ml_goal, _ml_charge_target, _ml_reason
            else:
                action, goal, charge_target, _reason = choose_action_and_goal(
                    active, opp_units, board, mode=mode,
                    target_multipliers=my_mults)

            if action == "charge" and charge_target is not None:
                enemy_positions = _collect_enemy_positions(opp_units)
                execute_charge_movement(active, charge_target, board, enemy_positions)
                execute_counter_charge(charge_target, active, board)
                if active.unit.impact > 0:
                    resolve_impact(active, charge_target)
                    _sync_dead_models(charge_target, board)
                charger_wounds = 0
                if charge_target.models_alive > 0:
                    charger_wounds = resolve_melee(active, charge_target, is_charge=True) or 0
                    _sync_dead_models(charge_target, board)
                defender_wounds = 0
                if active.models_alive > 0 and charge_target.models_alive > 0:
                    defender_wounds = resolve_melee(charge_target, active, is_strike_back=True) or 0
                    _sync_dead_models(active, board)
                if active.models_alive > 0 and charge_target.models_alive > 0:
                    check_melee_morale(active, charger_wounds, defender_wounds)
                    check_melee_morale(charge_target, defender_wounds, charger_wounds)
                    _sync_dead_models(active, board)
                    _sync_dead_models(charge_target, board)
                active.fatigued = True
                if charge_target.models_alive > 0:
                    charge_target.fatigued = True
                if active.models_alive > 0 and charge_target.models_alive > 0:
                    enemy_positions = _collect_enemy_positions(opp_units)
                    post_melee_separation(active, charge_target, board, enemy_positions)
                elif active.models_alive > 0:
                    from board import OBJECTIVES
                    consolidation_move(active, board, opp_units, OBJECTIVES, mode)
                elif charge_target.models_alive > 0:
                    from board import OBJECTIVES
                    consolidation_move(charge_target, board, my_units, OBJECTIVES, mode)
            elif action in ("advance", "rush") and goal is not None:
                budget = active.unit.advance_distance if action == "advance" else active.unit.rush_distance
                enemy_positions = _collect_enemy_positions(opp_units)
                rt, wr = _kite_range_params(active, opp_units, _reason)
                execute_movement(active, goal, budget, board, enemy_positions,
                                 flying=active.unit.flying,
                                 range_target=rt, weapon_range=wr)
                if action != "rush":
                    if active.shaken:
                        active.shaken = False
                    else:
                        if _ml_tac_decision:
                            target = pick_target_from_ranking(active, opp_units, _ml_target_ranking)
                        else:
                            target = pick_target(active, opp_units,
                                                 target_multipliers=my_mults)
                        if target is not None:
                            resolve_shooting(active, target)
                            check_morale(target)
                            _sync_dead_models(target, board)
            elif action == "hold":
                if active.shaken:
                    active.shaken = False
                else:
                    if _ml_tac_decision:
                        target = pick_target_from_ranking(active, opp_units, _ml_target_ranking)
                    else:
                        target = pick_target(active, opp_units,
                                             target_multipliers=my_mults)
                    if target is not None:
                        resolve_shooting(active, target)
                        check_morale(target)
                        _sync_dead_models(target, board)

            opp_alive = any(u.models_alive > 0 for u in opp_units)
            if not opp_alive:
                break

            current_is_a = not current_is_a

        if not is_kill_points:
            board.update_objectives(units_a, units_b)

    if is_kill_points:
        a_kill_pts = sum(u.unit.points for u in units_b if u.models_alive <= 0)
        b_kill_pts = sum(u.unit.points for u in units_a if u.models_alive <= 0)
        if a_kill_pts > b_kill_pts:
            return "A"
        elif b_kill_pts > a_kill_pts:
            return "B"
        return "draw"
    else:
        a_objs = board.count_objectives("A")
        b_objs = board.count_objectives("B")
        if a_objs > b_objs:
            return "A"
        elif b_objs > a_objs:
            return "B"
        return "draw"


def _sync_dead_models(unit: UnitState, board: Board):
    """Remove dead model positions from the board by draining the removed-positions buffer."""
    for col, row in unit._removed_positions:
        board.remove(col, row)
    unit._removed_positions.clear()


# ===================================================================
# RECORDED GAME (for viewer)
# ===================================================================

def _base_name(unit: UnitState) -> str:
    """Get the base unit name without upgrade details."""
    from templates import get_templates_dict
    tpl = get_templates_dict().get(unit.unit.template_id)
    return tpl.name if tpl else unit.unit.name


def _make_unit_labels(units_a: list[UnitState],
                      units_b: list[UnitState]) -> list[str]:
    """Create display labels for all units, numbering duplicates."""
    labels = []
    for player, units in [("A", units_a), ("B", units_b)]:
        name_counts: dict[str, int] = {}
        for u in units:
            n = _base_name(u)
            name_counts[n] = name_counts.get(n, 0) + 1
        name_seen: dict[str, int] = {}
        for u in units:
            n = _base_name(u)
            if name_counts[n] > 1:
                name_seen[n] = name_seen.get(n, 0) + 1
                labels.append(f"Player {player}'s {n} {name_seen[n]}")
            else:
                labels.append(f"Player {player}'s {n}")
    return labels


def _snapshot(all_units: list[UnitState],
              obj_control: list[str]) -> dict:
    """Capture current positions and alive counts for all units."""
    return {
        'positions': [list(u.alive_positions()) for u in all_units],
        'alive': [u.models_alive for u in all_units],
        'wounds': [list(u.wounds_per_model[:u.models_alive]) for u in all_units],
        'activated': [u.activated for u in all_units],
        'shaken': [u.shaken for u in all_units],
        'fatigued': [u.fatigued for u in all_units],
        'objectives': list(obj_control),
    }


ACTION_VERBS = {"hold": "Holds", "advance": "Advances", "rush": "Rushes"}


def simulate_game_recorded(army_a: list[ResolvedUnit],
                           army_b: list[ResolvedUnit],
                           mode: str = "objectives",
                           states_a: list[UnitState] | None = None,
                           states_b: list[UnitState] | None = None,
                           ml_model_a=None,
                           ml_model_b=None,
                           ml_sampling=False,
                           ml_planning=False,
                           planning_params: dict | None = None) -> tuple[str, list[dict], list[str], list[str], list[int], list[dict]]:
    """Play a recorded game. Returns (result, frames, unit_labels, unit_owners, unit_points).
    Each frame = {'positions', 'alive', 'objectives', 'description', 'round'}.
    mode: "objectives" (default) or "kill_points".
    ml_model_a/ml_model_b: if provided, use ML model for that player instead of heuristic AI.
    ml_sampling: if True, sample from model distributions instead of argmax.
    Assessment info in frames is always from Player A's perspective.
    """
    board = Board()
    is_kill_points = (mode == "kill_points")

    if states_a is not None:
        units_a = states_a
    else:
        units_a = [UnitState(u) for u in army_a]
        for u in units_a:
            u.owner = "A"

    if states_b is not None:
        units_b = states_b
    else:
        units_b = [UnitState(u) for u in army_b]
        for u in units_b:
            u.owner = "B"

    all_units = units_a + units_b
    labels = _make_unit_labels(units_a, units_b)
    owners = [u.owner for u in all_units]

    # Build unit index lookup
    unit_to_idx: dict[int, int] = {id(u): i for i, u in enumerate(all_units)}

    frames: list[dict] = []

    # ML setup
    use_ml = ml_model_a is not None or ml_model_b is not None
    _tactical_a = ml_model_a is not None and _is_tactical_model(ml_model_a)
    _tactical_b = ml_model_b is not None and _is_tactical_model(ml_model_b)
    if use_ml:
        if ml_sampling:
            from ml_integration import ml_activation_order
            from ml_integration import apply_model_outputs_sampling as apply_model_outputs
            from ml_features import precompute_damage
            if _tactical_a or _tactical_b:
                from ml_integration_tactical import (
                    apply_tactical_model_sampling as apply_tactical_model,
                    pick_target_from_ranking,
                )
        else:
            from ml_integration import apply_model_outputs, ml_activation_order
            from ml_features import precompute_damage
            if _tactical_a or _tactical_b:
                from ml_integration_tactical import (
                    apply_tactical_model,
                    pick_target_from_ranking,
                )
        _fr_a, _fm_a = precompute_damage([u.unit for u in units_a],
                                         [u.unit for u in units_b])
        _fr_b, _fm_b = precompute_damage([u.unit for u in units_b],
                                         [u.unit for u in units_a])
        _pts_a = sum(u.unit.points for u in units_a)
        _pts_b = sum(u.unit.points for u in units_b)
        if ml_planning and (_tactical_a or _tactical_b):
            from ml_planning import plan_activation as _plan_activation

    # Deployment
    deploy_armies(units_a, units_b, board)
    if not is_kill_points:
        if ml_model_a is None:
            assign_objectives(units_a)
        if ml_model_b is None:
            assign_objectives(units_b)

    snap = _snapshot(all_units, board.objective_control)
    snap['description'] = "Deployment complete"
    snap['round'] = 0
    frames.append(snap)

    a_first = random.random() < 0.5
    a_finished_first = a_first

    # Current ML assessment (updated each round when ML is active)
    ml_assessment: dict | None = None

    for round_num in range(4):
        for u in units_a:
            u.activated = False
            u.fatigued = False
        for u in units_b:
            u.activated = False
            u.fatigued = False

        current_is_a = a_first if round_num == 0 else a_finished_first

        if not is_kill_points:
            if ml_model_a is None:
                reassign_roles(units_a)
            if ml_model_b is None:
                reassign_roles(units_b)

        # ML forward pass at round start (strategic only — tactical runs per activation)
        target_mults_a = None
        target_mults_b = None
        if ml_model_a is not None and not _tactical_a:
            target_mults_a, ml_assessment = apply_model_outputs(
                ml_model_a, units_a, units_b, round_num + 1, board, "A",
                friendly_ranged_matchups=_fr_a, friendly_melee_matchups=_fm_a,
                enemy_ranged_matchups=_fr_b, enemy_melee_matchups=_fm_b,
                total_friendly_points=_pts_a, total_enemy_points=_pts_b,
            )
        if ml_model_b is not None and not _tactical_b:
            target_mults_b, _ = apply_model_outputs(
                ml_model_b, units_b, units_a, round_num + 1, board, "B",
                friendly_ranged_matchups=_fr_b, friendly_melee_matchups=_fm_b,
                enemy_ranged_matchups=_fr_a, enemy_melee_matchups=_fm_a,
                total_friendly_points=_pts_b, total_enemy_points=_pts_a,
            )

        a_done = False
        b_done = False
        a_finished_first = True

        while True:
            if current_is_a:
                my_units, opp_units = units_a, units_b
                my_ml = ml_model_a
                my_mults = target_mults_a
                my_tactical = _tactical_a
                _my_fr, _my_fm = (_fr_a, _fm_a) if use_ml else (None, None)
                _opp_fr, _opp_fm = (_fr_b, _fm_b) if use_ml else (None, None)
                my_pts, opp_pts = (_pts_a, _pts_b) if use_ml else (0, 0)
                my_player = "A"
            else:
                my_units, opp_units = units_b, units_a
                my_ml = ml_model_b
                my_mults = target_mults_b
                my_tactical = _tactical_b
                _my_fr, _my_fm = (_fr_b, _fm_b) if use_ml else (None, None)
                _opp_fr, _opp_fm = (_fr_a, _fm_a) if use_ml else (None, None)
                my_pts, opp_pts = (_pts_b, _pts_a) if use_ml else (0, 0)
                my_player = "B"

            # ML tactical decision state
            _ml_tac_decision = False
            _ml_target_ranking: list[int] = []
            _ml_action = "hold"
            _ml_goal = None
            _ml_charge_target = None
            _ml_reason = ""

            # Tactical model: run per-activation
            if my_ml is not None and my_tactical and ml_planning and (ml_planning is True or ml_planning == my_player):
                # Monte Carlo planning (eval only)
                active, _ml_target_ranking, _ml_action, _ml_goal, _ml_charge_target, _ml_reason, _planning_cands = (
                    _plan_activation(
                        my_ml, my_units, opp_units, round_num + 1, board, my_player,
                        units_a, units_b, current_is_a, mode,
                        friendly_ranged_matchups=_my_fr, friendly_melee_matchups=_my_fm,
                        enemy_ranged_matchups=_opp_fr, enemy_melee_matchups=_opp_fm,
                        total_friendly_points=my_pts, total_enemy_points=opp_pts,
                        fr_a=_fr_a, fm_a=_fm_a, fr_b=_fr_b, fm_b=_fm_b,
                        pts_a=_pts_a, pts_b=_pts_b,
                        planning_params=planning_params,
                    ))
                _ml_tac_decision = active is not None
                ml_assessment = {'planning_candidates': _planning_cands} if _planning_cands else None
            elif my_ml is not None and my_tactical:
                active, _ml_target_ranking, _ml_action, _ml_goal, _ml_charge_target, _ml_reason, _assess = (
                    apply_tactical_model(
                        my_ml, my_units, opp_units, round_num + 1, board, my_player,
                        friendly_ranged_matchups=_my_fr, friendly_melee_matchups=_my_fm,
                        enemy_ranged_matchups=_opp_fr, enemy_melee_matchups=_opp_fm,
                        total_friendly_points=my_pts, total_enemy_points=opp_pts,
                    ))
                _ml_tac_decision = active is not None
                # Only store assessment from Player A's perspective
                if current_is_a:
                    ml_assessment = _assess
            elif my_ml is not None:
                ordered = ml_activation_order(my_units)
                active = ordered[0] if ordered else None
            else:
                ordered = activation_order(my_units, enemies=opp_units, mode=mode)
                active = ordered[0] if ordered else None

            if active is None:
                if current_is_a:
                    a_done = True
                    if not b_done:
                        a_finished_first = True
                else:
                    b_done = True
                    if not a_done:
                        a_finished_first = False
                if a_done and b_done:
                    break
                current_is_a = not current_is_a
                continue

            active.activated = True
            active_idx = unit_to_idx[id(active)]
            active_label = labels[active_idx]

            if _ml_tac_decision:
                action, goal, charge_target, ai_reason = _ml_action, _ml_goal, _ml_charge_target, _ml_reason
            else:
                action, goal, charge_target, ai_reason = choose_action_and_goal(
                    active, opp_units, board, mode=mode,
                    target_multipliers=my_mults)

            # Build description
            pre_centre = active.centre()
            desc_parts = [f"{active_label} {ACTION_VERBS.get(action, action)} ({ai_reason})"]
            combat_stats = None

            if action == "charge" and charge_target is not None:
                target_idx = unit_to_idx[id(charge_target)]
                target_label = labels[target_idx]

                enemy_positions = _collect_enemy_positions(opp_units)
                execute_charge_movement(active, charge_target, board, enemy_positions)
                post_centre = active.centre()
                move_dist = math.sqrt((post_centre[0] - pre_centre[0]) ** 2 + (post_centre[1] - pre_centre[1]) ** 2)
                desc_parts = [f"{active_label} charges {target_label} {move_dist:.0f}\" ({ai_reason})"]
                execute_counter_charge(charge_target, active, board)

                # Impact
                impact_info = ""
                if active.unit.impact > 0:
                    imp = resolve_impact(active, charge_target)
                    _sync_dead_models(charge_target, board)
                    if imp['impact_hits'] > 0:
                        impact_info = f"Impact: {imp['impact_hits']} hits, {imp['impact_wounds']} wounds"

                # Charger swings
                charger_wounds = 0
                before_target = charge_target.models_alive
                if charge_target.models_alive > 0:
                    combat_stats = resolve_melee(active, charge_target, is_charge=True, recorded=True)
                    charger_wounds = combat_stats['wounds_dealt'] if combat_stats else 0
                    _sync_dead_models(charge_target, board)

                # Defender strikes back
                defender_wounds = 0
                before_active = active.models_alive
                if active.models_alive > 0 and charge_target.models_alive > 0:
                    def_stats = resolve_melee(charge_target, active, is_strike_back=True, recorded=True)
                    defender_wounds = def_stats['wounds_dealt'] if def_stats else 0
                    _sync_dead_models(active, board)

                # Build melee description
                melee_parts = []
                if impact_info:
                    melee_parts.append(impact_info)
                target_killed = before_target - charge_target.models_alive
                active_killed = before_active - active.models_alive
                melee_parts.append(f"Melee: {charger_wounds} wounds dealt, {defender_wounds} received")

                # Melee morale
                if active.models_alive > 0 and charge_target.models_alive > 0:
                    check_melee_morale(active, charger_wounds, defender_wounds)
                    check_melee_morale(charge_target, defender_wounds, charger_wounds)
                    _sync_dead_models(active, board)
                    _sync_dead_models(charge_target, board)

                if charge_target.models_alive <= 0:
                    melee_parts.append(f"{target_label} destroyed!")
                if active.models_alive <= 0:
                    melee_parts.append(f"{active_label} destroyed!")

                # Fatigue
                active.fatigued = True
                if charge_target.models_alive > 0:
                    charge_target.fatigued = True

                # Post-melee separation or consolidation
                if active.models_alive > 0 and charge_target.models_alive > 0:
                    enemy_positions = _collect_enemy_positions(opp_units)
                    post_melee_separation(active, charge_target, board, enemy_positions)
                elif active.models_alive > 0:
                    from board import OBJECTIVES as _OBJS
                    consolidation_move(active, board, opp_units, _OBJS, mode)
                elif charge_target.models_alive > 0:
                    from board import OBJECTIVES as _OBJS
                    consolidation_move(charge_target, board, my_units, _OBJS, mode)

                desc_parts.append("-- " + ", ".join(melee_parts))

                # Build compound melee stats for viewer
                if combat_stats is None:
                    combat_stats = {}
                combat_stats['combat_type'] = 'melee'
                combat_stats['charger_wounds'] = charger_wounds
                combat_stats['defender_wounds'] = defender_wounds
                if impact_info:
                    combat_stats['impact_info'] = impact_info

            elif action in ("advance", "rush") and goal is not None:
                budget = active.unit.advance_distance if action == "advance" else active.unit.rush_distance
                enemy_positions = _collect_enemy_positions(opp_units)
                rt, wr = _kite_range_params(active, opp_units, ai_reason)
                execute_movement(active, goal, budget, board, enemy_positions,
                                 flying=active.unit.flying,
                                 range_target=rt, weapon_range=wr)
                post_centre = active.centre()
                move_dist = math.sqrt((post_centre[0] - pre_centre[0]) ** 2 + (post_centre[1] - pre_centre[1]) ** 2)
                desc_parts = [f"{active_label} {ACTION_VERBS.get(action, action)} {move_dist:.0f}\" ({ai_reason})"]

                if action != "rush":
                    if active.shaken:
                        active.shaken = False
                        desc_parts.append("(was Shaken, recovers)")
                    else:
                        if _ml_tac_decision:
                            target = pick_target_from_ranking(active, opp_units, _ml_target_ranking)
                        else:
                            target = pick_target(active, opp_units,
                                                 target_multipliers=my_mults)
                        if target is not None:
                            target_idx = unit_to_idx[id(target)]
                            target_label = labels[target_idx]
                            before = target.models_alive
                            combat_stats = resolve_shooting(active, target, recorded=True)
                            check_morale(target)
                            _sync_dead_models(target, board)
                            killed = before - target.models_alive
                            if killed > 0:
                                if target.models_alive <= 0:
                                    desc_parts.append(f"and shoots {target_label}, destroying the unit!")
                                else:
                                    desc_parts.append(f"and shoots {target_label}, killing {killed} model{'s' if killed != 1 else ''}")
                            else:
                                desc_parts.append(f"and shoots {target_label}, no casualties")
                        else:
                            desc_parts.append("(no targets in range)")

            elif action == "hold":
                if active.shaken:
                    active.shaken = False
                    desc_parts.append("(was Shaken, recovers)")
                else:
                    if _ml_tac_decision:
                        target = pick_target_from_ranking(active, opp_units, _ml_target_ranking)
                    else:
                        target = pick_target(active, opp_units,
                                             target_multipliers=my_mults)
                    if target is not None:
                        target_idx = unit_to_idx[id(target)]
                        target_label = labels[target_idx]
                        before = target.models_alive
                        combat_stats = resolve_shooting(active, target, recorded=True)
                        check_morale(target)
                        _sync_dead_models(target, board)
                        killed = before - target.models_alive
                        if killed > 0:
                            if target.models_alive <= 0:
                                desc_parts.append(f"and shoots {target_label}, destroying the unit!")
                            else:
                                desc_parts.append(f"and shoots {target_label}, killing {killed} model{'s' if killed != 1 else ''}")
                        else:
                            desc_parts.append(f"and shoots {target_label}, no casualties")
                    else:
                        desc_parts.append("(no targets in range)")

            opp_alive = any(u.models_alive > 0 for u in opp_units)

            snap = _snapshot(all_units, board.objective_control)
            snap['description'] = " ".join(desc_parts)
            snap['round'] = round_num + 1
            snap['combat_stats'] = combat_stats
            if ml_assessment is not None:
                snap['ml_assessment'] = ml_assessment
            frames.append(snap)

            if not opp_alive:
                break

            current_is_a = not current_is_a

        if is_kill_points:
            # Record end-of-round kill points
            a_kp = sum(u.unit.points for u in units_b if u.models_alive <= 0)
            b_kp = sum(u.unit.points for u in units_a if u.models_alive <= 0)
            snap = _snapshot(all_units, board.objective_control)
            snap['description'] = f"End of Round {round_num + 1} -- Kill Points: Player A: {a_kp}pts, Player B: {b_kp}pts"
            snap['round'] = round_num + 1
            frames.append(snap)
        else:
            board.update_objectives(units_a, units_b)
            # Record end-of-round objective state
            obj_names = ["Centre", "A-side", "B-side", "Home-A", "Home-B"]
            obj_parts = []
            for oi, ctrl in enumerate(board.objective_control):
                if ctrl:
                    obj_parts.append(f"{obj_names[oi]}: Player {ctrl}")
                else:
                    obj_parts.append(f"{obj_names[oi]}: Neutral")
            snap = _snapshot(all_units, board.objective_control)
            snap['description'] = f"End of Round {round_num + 1} -- Objectives: {', '.join(obj_parts)}"
            snap['round'] = round_num + 1
            frames.append(snap)

    if is_kill_points:
        a_kp = sum(u.unit.points for u in units_b if u.models_alive <= 0)
        b_kp = sum(u.unit.points for u in units_a if u.models_alive <= 0)
        if a_kp > b_kp:
            result = "A"
        elif b_kp > a_kp:
            result = "B"
        else:
            result = "draw"
        snap = _snapshot(all_units, board.objective_control)
        snap['description'] = f"Game Over -- Kill Points: Player A: {a_kp}pts, Player B: {b_kp}pts -- {'Draw' if result == 'draw' else 'Player ' + result + ' wins!'}"
        snap['round'] = 5
        snap['kill_points'] = {'A': a_kp, 'B': b_kp}
        frames.append(snap)
    else:
        a_objs = board.count_objectives("A")
        b_objs = board.count_objectives("B")
        if a_objs > b_objs:
            result = "A"
        elif b_objs > a_objs:
            result = "B"
        else:
            result = "draw"
        snap = _snapshot(all_units, board.objective_control)
        snap['description'] = f"Game Over -- Player A: {a_objs} objectives, Player B: {b_objs} objectives -- {'Draw' if result == 'draw' else 'Player ' + result + ' wins!'}"
        snap['round'] = 5
        frames.append(snap)

    unit_points = [u.unit.points for u in all_units]

    # Build unit_info dicts for the viewer
    unit_info: list[dict] = []
    for u in all_units:
        ru = u.unit
        special = []
        if ru.scout: special.append("Scout")
        if ru.stealth: special.append("Stealth")
        if ru.fast: special.append("Fast")
        if ru.relentless: special.append("Relentless")
        if ru.highborn: special.append("Highborn")
        if ru.artillery: special.append("Artillery")
        if ru.fearless: special.append("Fearless")
        if ru.regeneration: special.append("Regeneration")
        if ru.shielded: special.append("Shielded")
        if ru.furious: special.append("Furious")
        if ru.fortified: special.append("Fortified")
        if ru.impact: special.append(f"Impact({ru.impact})")
        if ru.tough: special.append(f"Tough({ru.tough})")
        if ru.piercing_spotter: special.append("Piercing Spotter")

        weapons_summary: list[str] = []
        seen: dict[str, int] = {}
        for w in ru.weapons:
            key = w.name
            seen[key] = seen.get(key, 0) + 1
        for wname, count in seen.items():
            w_obj = next(w for w in ru.weapons if w.name == wname)
            abilities = []
            if w_obj.ap: abilities.append(f"AP({w_obj.ap})")
            if w_obj.blast: abilities.append(f"Blast({w_obj.blast})")
            if w_obj.deadly: abilities.append(f"Deadly({w_obj.deadly})")
            if w_obj.crack: abilities.append("Crack")
            if w_obj.rending: abilities.append("Rending")
            if w_obj.reliable: abilities.append("Reliable")
            if w_obj.takedown: abilities.append("Takedown")
            if w_obj.unstoppable: abilities.append("Unstoppable")
            if w_obj.bane: abilities.append("Bane")
            range_str = f'{w_obj.range_inches}"' if w_obj.range_inches > 0 else "melee"
            prefix = f"{count}x " if count > 1 else ""
            ab_str = f"  [{', '.join(abilities)}]" if abilities else ""
            weapons_summary.append(f"{prefix}{wname} ({range_str}, A{w_obj.attacks}{ab_str})")

        # ML input features (static, at-start values the tactical model sees)
        ml_feats: dict[str, float | list[float]] | None = None
        if use_ml:
            idx = all_units.index(u)
            owner = owners[idx]
            if owner == "A":
                side_idx = units_a.index(u)
                rb_matchups = _fr_a[side_idx] if side_idx < len(_fr_a) else []
                mb_list = _fm_a[side_idx] if side_idx < len(_fm_a) else []
                total_pts = _pts_a
                enemy_units = units_b
            else:
                side_idx = units_b.index(u)
                rb_matchups = _fr_b[side_idx] if side_idx < len(_fr_b) else []
                mb_list = _fm_b[side_idx] if side_idx < len(_fm_b) else []
                total_pts = _pts_b
                enemy_units = units_a
            # rb_matchups is [num_enemies][NUM_RANGE_THRESHOLDS]
            rb_flat = [v for row in rb_matchups for v in row]
            rb = sum(rb_flat) if rb_flat else 0.0
            mb = sum(mb_list) if len(mb_list) > 0 else 0.0
            enemy_names = [_base_name(e) for e in enemy_units]
            speed = 0.0 if ru.artillery else float(ru.rush_distance)
            # AP flags
            ap_flags = [0.0] * 5
            deadly_flags = [0.0] * 3
            for w in ru.weapons:
                if w.ap == 0: ap_flags[0] = 1.0
                if w.ap == 1: ap_flags[1] = 1.0
                if w.ap == 2: ap_flags[2] = 1.0
                if w.ap == 3: ap_flags[3] = 1.0
                if w.ap >= 4: ap_flags[4] = 1.0
                if w.deadly == 0: deadly_flags[0] = 1.0
                if w.deadly == 3: deadly_flags[1] = 1.0
                if w.deadly >= 6: deadly_flags[2] = 1.0
            wound_count = ru.tough if ru.tough > 0 else 1
            ml_feats = {
                'toughness': round(wound_count / 24, 4),
                'model_count': round(ru.models / 10, 4),
                'defense': round((7 - ru.defense) / 5.0, 4),
                'ranged_dmg': round(rb / max(len(rb_flat), 1), 4),
                'ranged_dmg_list': [round(v, 4) for v in rb_flat],
                'ranged_matchups': [[round(v, 4) for v in row] for row in rb_matchups],
                'enemy_names': enemy_names,
                'melee_dmg': round(mb / max(len(mb_list), 1), 4),
                'melee_dmg_list': [round(v, 4) for v in mb_list],
                'speed': round(speed / 24.0, 4),
                'points_frac': round(ru.points / max(total_pts, 1), 4),
                'ap_flags': ap_flags,
                'deadly_flags': deadly_flags,
                'flying': 1.0 if ru.flying else 0.0,
                'artillery': 1.0 if ru.artillery else 0.0,
                'stealth': 1.0 if ru.stealth else 0.0,
                'fearless': 1.0 if ru.fearless else 0.0,
                'fear': 1.0 if ru.fear > 0 else 0.0,
            }

        unit_info.append({
            'template_id': ru.template_id,
            'name': ru.name,
            'models': ru.models,
            'tough': ru.tough,
            'quality': ru.quality,
            'defense': ru.defense,
            'points': ru.points,
            'special': special,
            'weapons': weapons_summary,
            'ai_role': u.ai_role,
            'combat_preference': u.combat_preference,
            'ml_features': ml_feats,
        })

    return result, frames, labels, owners, unit_points, unit_info
