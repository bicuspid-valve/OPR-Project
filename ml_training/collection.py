"""Episode collection: single episodes, generators, workers, and batched inference."""
from __future__ import annotations

import random

import torch
import torch.nn as nn

from board import Board
from models import ResolvedUnit, UnitState
from ml_features import MAX_UNITS_PER_SIDE, TACTICAL_UNIT_FEATURES, encode_state_tactical, precompute_damage, extract_can_charge_mask
from ml_model_tactical import (
    TacticalModel, NUM_MOVE_TYPES, MOVE_HOLD, MOVE_ADVANCE, MOVE_RUSH, MOVE_CHARGE,
)
from ml_integration_tactical import (
    MOVE_TYPE_NAMES, execute_decoded_decision, pick_target_from_ranking,
    compute_post_move_rel, compute_post_move_position,
    _get_model_space_positions, _get_movement_budgets, _get_max_weapon_ranges,
)
from ml_features import _flip_x, _flip_y

from ml_training.config import (
    TacticalActivationRecord, _TacticalInferenceRequest, _TacticalSamplingResult,
    _get_opponent_type_idx,
)
from ml_training.rewards import (
    compute_round_reward, terminal_reward, _make_round_snapshot,
    compute_objective_capture_reward, _any_friendly_on_objective,
)
from board import OBJECTIVES as _OBJECTIVES_FOR_REWARD, OBJ_SEIZE_RANGE as _OBJ_SEIZE_RANGE
from ml_training.sampling import sample_tactical_actions_no_grad, _batched_sample_tactical_no_grad
from ml_training.checkpoint import _make_model

# ---------------------------------------------------------------------------
# Parallel episode collection (worker + replay)
# ---------------------------------------------------------------------------

_WORKER_COUNT = 6

# Maximum number of shared-memory opponent model slots
_MAX_SHARED_OPPONENTS = 5

# ---------------------------------------------------------------------------
# Shared-memory worker globals (set by _init_shared_worker in child processes)
# ---------------------------------------------------------------------------

_g_shared_model: nn.Module | None = None
_g_shared_opponents: list[nn.Module] = []
_g_worker_model_type: str = "tactical"


def _init_shared_worker(shared_model, shared_opponents, model_type="tactical",
                         use_c_ext=True):
    """Initialize worker process with references to shared-memory models."""
    global _g_shared_model, _g_shared_opponents, _g_worker_model_type
    _g_shared_model = shared_model
    _g_shared_opponents = shared_opponents
    _g_worker_model_type = model_type
    # Each worker runs small single-sample inferences — using multiple torch
    # threads per worker causes massive oversubscription (8 workers × 8 threads
    # = 64 threads on 16 logical cores).  Pin to 1 thread per worker.
    torch.set_num_threads(1)
    # Toggle C extension in worker processes
    import fast_core
    fast_core.USE_C_EXT = use_c_ext and fast_core.is_available()


def _collect_episodes_shared_worker(args) -> list[tuple[list[TacticalActivationRecord], str, str, str]]:
    """Run training episodes using shared-memory models.

    Like _collect_episodes_chunked_worker but reads model weights directly from
    shared memory instead of deserializing state dicts.  Only lightweight
    game specs and an opponent slot map are sent via IPC.

    Args is (opp_slot_map, game_specs, shaping_scale[, planning_config]) where
    opp_slot_map maps opp_sd_index -> index into _g_shared_opponents (or absent
    for heuristic). shaping_scale controls the per-round reward shaping
    magnitude (1.0 = full, 0.0 = off).

    Returns list of (trajectory_rounds, result, opponent_type, army_type).
    """
    if len(args) >= 4:
        opp_slot_map, game_specs, shaping_scale, planning_config = args
    else:
        opp_slot_map, game_specs, shaping_scale = args
        planning_config = None

    from board import OBJECTIVES as BOARD_OBJECTIVES

    model = _g_shared_model

    # Map opponent indices to shared opponent models
    opp_models: dict[int, nn.Module] = {}
    for spec in game_specs:
        opp_sd_idx = spec[5]
        if opp_sd_idx >= 0 and opp_sd_idx not in opp_models:
            slot = opp_slot_map.get(opp_sd_idx)
            if slot is not None and slot < len(_g_shared_opponents):
                opp_models[opp_sd_idx] = _g_shared_opponents[slot]

    planning_rate = 0.0
    planning_params = None
    if planning_config is not None:
        planning_rate = planning_config.get("planning_rate", 0.0)
        planning_params = planning_config.get("planning_params")

    return _run_games_batched_tactical(model, game_specs, opp_models,
                                       shaping_scale=shaping_scale,
                                       planning_rate=planning_rate,
                                       planning_params=planning_params)


def _collect_episodes_chunked_worker(args) -> list:
    """Run multiple training episodes in one worker, rebuilding models only once.

    Accepts (model_state_dict, opponent_state_dicts, game_specs, model_type) where
    game_specs is a list of (res_a, res_b, states_a_data, states_b_data, opponent_type, opp_sd_index, army_type).
    opp_sd_index is an index into opponent_state_dicts (or -1 for no opponent model).

    Returns a list of (trajectory_steps, game_result, opponent_type, army_type) per game.
    """
    model_state_dict, opponent_state_dicts, game_specs, model_type = args

    # Prevent torch thread oversubscription across workers
    torch.set_num_threads(1)

    from game import deploy_armies, _collect_enemy_positions, _sync_dead_models
    from ai import (
        pick_target, choose_action_and_goal, activation_order,
        assign_objectives, reassign_roles,
    )
    from combat import resolve_shooting, check_morale, resolve_melee, resolve_impact, check_melee_morale
    from movement import (
        execute_movement, execute_charge_movement, execute_counter_charge,
        post_melee_separation, consolidation_move,
    )
    from board import OBJECTIVES as BOARD_OBJECTIVES

    # Build the training model once for the whole chunk
    model = _make_model(model_type)
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

    # Build opponent models once (deduplicated by index)
    opp_models: dict[int, nn.Module] = {}
    for spec in game_specs:
        opp_sd_idx = spec[5]
        if opp_sd_idx >= 0 and opp_sd_idx not in opp_models:
            opp_model = _make_model(model_type)
            opp_model.load_state_dict(opponent_state_dicts[opp_sd_idx], strict=False)
            opp_model.eval()
            opp_models[opp_sd_idx] = opp_model

    results = []
    for res_a, res_b, states_a_data, states_b_data, opponent_type, opp_sd_idx, army_type in game_specs:
        opponent_model = opp_models.get(opp_sd_idx)
        traj_a, result, opp_t, traj_b = _run_single_episode_tactical(
            model, opponent_model, res_a, res_b, states_a_data, states_b_data,
            opponent_type, BOARD_OBJECTIVES, army_type=army_type,
        )
        results.append((traj_a, result, opp_t, army_type))
        if traj_b is not None:
            results.append((traj_b, result, "mirror_b", army_type))
    return results

def _run_single_episode_tactical(model, opponent_model, res_a, res_b,
                                  states_a_data, states_b_data, opponent_type,
                                  BOARD_OBJECTIVES, shaping_scale=1.0,
                                  army_type="random",
                                  planning_rate=0.0, planning_params=None):
    """Run one training episode with the tactical (per-activation) model.

    Player A uses the new sequential-conditioned sampling path (§4.1) with
    execute_decoded_decision + pick_target_from_ranking for action resolution.
    """
    from game import deploy_armies, _collect_enemy_positions, _sync_dead_models
    from ai import (
        pick_target, choose_action_and_goal, activation_order,
        assign_objectives, reassign_roles,
    )
    from combat import resolve_shooting, check_morale, resolve_melee, resolve_impact, check_melee_morale
    from movement import (
        execute_movement, execute_charge_movement, execute_counter_charge,
        post_melee_separation, consolidation_move,
    )
    from ml_integration_tactical import apply_tactical_model_sampling
    is_mirror = (opponent_type == "selfplay_mirror")
    opponent_type_idx = _get_opponent_type_idx(opponent_type, army_type)

    # Rebuild UnitState objects
    units_a = [UnitState(ru) for ru in res_a]
    for u in units_a:
        u.owner = "A"
    units_b = [UnitState(ru) for ru in res_b]
    for u in units_b:
        u.owner = "B"

    for u, (ai_role, combat_pref, assigned_obj) in zip(units_a, states_a_data):
        u.ai_role = ai_role
        u.combat_preference = combat_pref
        u.assigned_objective = assigned_obj
    for u, (ai_role, combat_pref, assigned_obj) in zip(units_b, states_b_data):
        u.ai_role = ai_role
        u.combat_preference = combat_pref
        u.assigned_objective = assigned_obj

    board = Board()
    deploy_armies(units_a, units_b, board)

    fr_a, fm_a = precompute_damage([u.unit for u in units_a], [u.unit for u in units_b])
    fr_b, fm_b = precompute_damage([u.unit for u in units_b], [u.unit for u in units_a])
    pts_a = sum(u.unit.points for u in units_a)
    pts_b = sum(u.unit.points for u in units_b)

    if opponent_type == "heuristic":
        assign_objectives(units_b)

    a_first = random.random() < 0.5
    a_finished_first = a_first

    trajectory: list[TacticalActivationRecord] = []
    trajectory_b: list[TacticalActivationRecord] | None = [] if is_mirror else None
    prev_a_kill_pts = 0.0
    prev_b_kill_pts = 0.0
    prev_b_fkp = 0.0
    prev_b_ekp = 0.0

    # Activation counters for countdown targets
    _a_act_total = 0
    _b_act_total = 0
    _traj_a_counts: list[tuple[int, int]] = []   # (a_so_far, b_so_far) per A trajectory step
    _traj_b_counts: list[tuple[int, int]] = []   # (b_so_far, a_so_far) per B trajectory step (mirror)

    for round_num in range(1, 5):
        for u in units_a + units_b:
            u.activated = False
            u.fatigued = False

        current_is_a = a_first if round_num == 1 else a_finished_first

        # Player B decisions at round start (heuristic only; tactical opponents
        # and mirror decide per-activation, not per-round).
        target_mults_b = None
        if opponent_type == "heuristic":
            reassign_roles(units_b)

        # Track steps in this round for reward assignment
        round_step_indices: list[int] = []
        round_step_indices_b: list[int] = []

        # --- Alternating activations ---
        a_done = False
        b_done = False
        a_finished_first = True

        # Per-activation state for ML-driven sides (set in decision block,
        # consumed in execution block)
        _a_tac_action: str = "hold"
        _a_tac_goal: tuple[int, int] | None = None
        _a_tac_charge_target: UnitState | None = None
        _a_tac_target_ranking: list[int] = []

        while True:
            if current_is_a:
                my_units, opp_units = units_a, units_b
                my_mults = None  # tactical model provides per-activation
            else:
                my_units, opp_units = units_b, units_a
                my_mults = target_mults_b

            _opp_tac_decision = False

            if current_is_a:
                # --- Player A: tactical model decides (new conditioned path) ---
                # Build alive+unactivated mask
                alive_mask_list = []
                for i in range(MAX_UNITS_PER_SIDE):
                    if i < len(units_a):
                        us = units_a[i]
                        alive_mask_list.append(us.models_alive > 0 and not us.activated)
                    else:
                        alive_mask_list.append(False)

                if not any(alive_mask_list):
                    a_done = True
                    if not b_done:
                        a_finished_first = True
                    if a_done and b_done:
                        break
                    current_is_a = not current_is_a
                    continue

                alive_mask = torch.tensor(alive_mask_list, dtype=torch.bool)

                # Build enemy_alive_mask (§1.11)
                enemy_alive_mask_list = [
                    (i < len(units_b) and units_b[i].models_alive > 0)
                    for i in range(MAX_UNITS_PER_SIDE)
                ]
                enemy_alive_mask = torch.tensor(enemy_alive_mask_list, dtype=torch.bool)

                # Encode state
                state_vec = encode_state_tactical(
                    units_a, units_b, round_num, board, "A",
                    friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
                    enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
                    total_friendly_points=pts_a, total_enemy_points=pts_b,
                )
                state_vec_np = state_vec.numpy()

                # Compute model-space positions for sampling
                a_friendly_pos = _get_model_space_positions(units_a, "A")
                a_enemy_pos = _get_model_space_positions(units_b, "A")
                a_adv_dists, a_rush_dists = _get_movement_budgets(units_a)
                a_max_wr = _get_max_weapon_ranges(units_a)

                # Decide: planning or policy sampling
                _was_planned = False
                _planning_improved = False
                _planning_value_delta = 0.0
                _planning_unit_values = None
                _planning_unit_indices = None
                _planning_move_values = None
                _planning_move_indices = None
                _planning_charge_values = None
                _planning_charge_indices = None
                _planning_shoot_values = None
                _planning_shoot_indices = None

                use_planning = (
                    planning_rate > 0
                    and random.random() < planning_rate
                )

                if use_planning:
                    from ml_planning import plan_training_activation
                    (sel_idx, move_type_a, sampled_angle_a, sampled_frac_a,
                     charge_tgt_a, shoot_tgt_a, _a_tac_target_ranking,
                     pmr_a, old_lp, value_est, shoot_mask_a,
                     _was_planned, _planning_improved, _planning_value_delta,
                     _planning_unit_values, _planning_unit_indices,
                     _planning_move_values, _planning_move_indices,
                     _planning_charge_values, _planning_charge_indices,
                     _planning_shoot_values, _planning_shoot_indices,
                    ) = plan_training_activation(
                        model, state_vec, alive_mask, enemy_alive_mask,
                        units_a, units_b, round_num, board, "A",
                        current_is_a=current_is_a,
                        mode="objectives",
                        friendly_positions=a_friendly_pos,
                        enemy_positions=a_enemy_pos,
                        advance_distances=a_adv_dists,
                        rush_distances=a_rush_dists,
                        max_weapon_ranges=a_max_wr,
                        fr_a=fr_a, fm_a=fm_a, fr_b=fr_b, fm_b=fm_b,
                        pts_a=pts_a, pts_b=pts_b,
                        planning_params=planning_params,
                        opponent_type=opponent_type_idx,
                    )
                else:
                    (sel_idx, move_type_a, sampled_angle_a, sampled_frac_a,
                     charge_tgt_a, shoot_tgt_a, _a_tac_target_ranking,
                     pmr_a, old_lp, value_est, shoot_mask_a) = sample_tactical_actions_no_grad(
                        model, state_vec, alive_mask, enemy_alive_mask,
                        a_friendly_pos, a_enemy_pos, a_adv_dists, a_rush_dists,
                        a_max_wr, opponent_type_idx=opponent_type_idx,
                    )

                # Record for PPO replay
                step = TacticalActivationRecord(
                    state_vec=state_vec_np,
                    alive_mask=alive_mask_list,
                    enemy_alive_mask=enemy_alive_mask_list,
                    unit_idx=sel_idx,
                    move_type=move_type_a,
                    sampled_angle=sampled_angle_a,
                    sampled_distance_frac=sampled_frac_a,
                    charge_target_idx=charge_tgt_a,
                    shoot_target_idx=shoot_tgt_a,
                    shoot_mask=shoot_mask_a,
                    post_move_rel=pmr_a,
                    old_log_prob=old_lp,
                    old_value=value_est,
                    opponent_type_idx=opponent_type_idx,
                    was_planned=_was_planned,
                    planning_improved=_planning_improved,
                    planning_value_delta=_planning_value_delta,
                    planning_unit_values=_planning_unit_values,
                    planning_unit_indices=_planning_unit_indices,
                    planning_move_values=_planning_move_values,
                    planning_move_indices=_planning_move_indices,
                    planning_charge_values=_planning_charge_values,
                    planning_charge_indices=_planning_charge_indices,
                    planning_shoot_values=_planning_shoot_values,
                    planning_shoot_indices=_planning_shoot_indices,
                )
                round_step_indices.append(len(trajectory))
                trajectory.append(step)
                _a_act_total += 1
                _traj_a_counts.append((_a_act_total, _b_act_total))

                active = units_a[sel_idx]
                active.activated = True

                # Compute destination in game-space
                _a_dest = None
                if move_type_a in (MOVE_ADVANCE, MOVE_RUSH):
                    ucx, ucy = a_friendly_pos[sel_idx]
                    budget = a_adv_dists[sel_idx] if move_type_a == MOVE_ADVANCE else a_rush_dists[sel_idx]
                    px, py = compute_post_move_position(ucx, ucy, sampled_angle_a, sampled_frac_a * budget)
                    # Convert model-space → game-space
                    _a_dest = (px, py)  # Player A: model-space == game-space

                _a_tac_action, _a_tac_goal, _a_tac_charge_target, _a_reason = execute_decoded_decision(
                    active, units_b, move_type_a, _a_dest, charge_tgt_a, shoot_tgt_a,
                )
            else:
                # --- Player B: mirror, heuristic, strategic, or tactical model ---
                _b_target_ranking: list[int] = []
                _b_action = "hold"
                _b_goal = None
                _b_charge_target = None
                if is_mirror:
                    # Mirror self-play: B uses same tactical model as A
                    b_alive_list = []
                    for i in range(MAX_UNITS_PER_SIDE):
                        if i < len(units_b):
                            us = units_b[i]
                            b_alive_list.append(us.models_alive > 0 and not us.activated)
                        else:
                            b_alive_list.append(False)

                    if not any(b_alive_list):
                        active = None
                    else:
                        b_alive_mask = torch.tensor(b_alive_list, dtype=torch.bool)
                        b_enemy_alive_list = [
                            (i < len(units_a) and units_a[i].models_alive > 0)
                            for i in range(MAX_UNITS_PER_SIDE)
                        ]
                        b_enemy_alive_mask = torch.tensor(b_enemy_alive_list, dtype=torch.bool)

                        b_state_vec = encode_state_tactical(
                            units_b, units_a, round_num, board, "B",
                            friendly_ranged_matchups=fr_b, friendly_melee_matchups=fm_b,
                            enemy_ranged_matchups=fr_a, enemy_melee_matchups=fm_a,
                            total_friendly_points=pts_b, total_enemy_points=pts_a,
                        )
                        b_state_vec_np = b_state_vec.numpy()

                        b_friendly_pos = _get_model_space_positions(units_b, "B")
                        b_enemy_pos = _get_model_space_positions(units_a, "B")
                        b_adv_dists, b_rush_dists = _get_movement_budgets(units_b)
                        b_max_wr = _get_max_weapon_ranges(units_b)

                        (sel_b, mt_b, sa_b, sf_b, ct_b, st_b,
                         _b_target_ranking, pmr_b, olp_b, val_b, sm_b) = sample_tactical_actions_no_grad(
                            model, b_state_vec, b_alive_mask, b_enemy_alive_mask,
                            b_friendly_pos, b_enemy_pos, b_adv_dists, b_rush_dists,
                            b_max_wr, opponent_type_idx=opponent_type_idx,
                        )

                        step_b = TacticalActivationRecord(
                            state_vec=b_state_vec_np,
                            alive_mask=b_alive_list,
                            enemy_alive_mask=b_enemy_alive_list,
                            unit_idx=sel_b,
                            move_type=mt_b,
                            sampled_angle=sa_b,
                            sampled_distance_frac=sf_b,
                            charge_target_idx=ct_b,
                            shoot_target_idx=st_b,
                            shoot_mask=sm_b,
                            post_move_rel=pmr_b,
                            old_log_prob=olp_b,
                            old_value=val_b,
                            opponent_type_idx=opponent_type_idx,
                        )
                        round_step_indices_b.append(len(trajectory_b))
                        trajectory_b.append(step_b)
                        _b_act_total += 1
                        _traj_b_counts.append((_b_act_total, _a_act_total))

                        active = units_b[sel_b]
                        active.activated = True

                        # Compute B destination in game-space
                        _b_dest = None
                        if mt_b in (MOVE_ADVANCE, MOVE_RUSH):
                            bcx, bcy = b_friendly_pos[sel_b]
                            bgt = b_adv_dists[sel_b] if mt_b == MOVE_ADVANCE else b_rush_dists[sel_b]
                            bpx, bpy = compute_post_move_position(bcx, bcy, sa_b, sf_b * bgt)
                            _b_dest = (_flip_x(bpx), _flip_y(bpy))  # model-space → game-space for B

                        _b_action, _b_goal, _b_charge_target, _ = execute_decoded_decision(
                            active, units_a, mt_b, _b_dest, ct_b, st_b,
                        )

                    _opp_tac_decision = active is not None
                elif opponent_model is not None:
                    (selected, _b_target_ranking, _b_action, _b_goal,
                     _b_charge_target, _b_reason, _) = apply_tactical_model_sampling(
                        opponent_model, my_units, opp_units, round_num, board, "B",
                        friendly_ranged_matchups=fr_b, friendly_melee_matchups=fm_b,
                        enemy_ranged_matchups=fr_a, enemy_melee_matchups=fm_a,
                        total_friendly_points=pts_b, total_enemy_points=pts_a,
                    )
                    active = selected
                    _opp_tac_decision = active is not None
                else:
                    ordered = activation_order(my_units, enemies=opp_units, mode="objectives")
                    active = ordered[0] if ordered else None

                if active is None:
                    b_done = True
                    if not a_done:
                        a_finished_first = False
                    if a_done and b_done:
                        break
                    current_is_a = not current_is_a
                    continue

                if not is_mirror or not _opp_tac_decision:
                    active.activated = True
                # Count B activations (non-mirror; mirror already counted above)
                if not is_mirror:
                    _b_act_total += 1

            # --- Execute the activation ---
            # Determine action / goal / charge_target and target ranking for shooting
            if current_is_a:
                # Player A: already resolved via execute_decoded_decision above
                action = _a_tac_action
                goal = _a_tac_goal
                charge_target = _a_tac_charge_target
                _active_target_ranking = _a_tac_target_ranking
            elif _opp_tac_decision:
                action, goal, charge_target = _b_action, _b_goal, _b_charge_target
                _active_target_ranking = _b_target_ranking
            else:
                action, goal, charge_target, _reason = choose_action_and_goal(
                    active, opp_units, board, mode="objectives",
                    target_multipliers=my_mults,
                )
                _active_target_ranking = []  # not used — falls through to pick_target

            # Pre-move snapshot: which objectives already have a friendly unit on them?
            _pre_move_friendly_on_objs: list[bool] | None = None
            if current_is_a or (is_mirror and _opp_tac_decision):
                _obj_friendly = units_a if current_is_a else units_b
                _pre_move_friendly_on_objs = [
                    _any_friendly_on_objective(obj_pos, _obj_friendly)
                    for obj_pos in _OBJECTIVES_FOR_REWARD
                ]

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
                    consolidation_move(active, board, opp_units, BOARD_OBJECTIVES, "objectives")
                elif charge_target.models_alive > 0:
                    consolidation_move(charge_target, board, my_units, BOARD_OBJECTIVES, "objectives")

            elif action in ("advance", "rush") and goal is not None:
                budget = active.unit.advance_distance if action == "advance" else active.unit.rush_distance
                enemy_positions = _collect_enemy_positions(opp_units)
                execute_movement(active, goal, budget, board, enemy_positions,
                                 flying=active.unit.flying)

                if action != "rush":
                    if active.shaken:
                        active.shaken = False
                    else:
                        if current_is_a or _opp_tac_decision:
                            target = pick_target_from_ranking(active, opp_units, _active_target_ranking)
                        else:
                            target = pick_target(active, opp_units, target_multipliers=my_mults)
                        if target is not None:
                            resolve_shooting(active, target)
                            check_morale(target)
                            _sync_dead_models(target, board)

            elif action == "hold":
                if active.shaken:
                    active.shaken = False
                else:
                    if current_is_a or _opp_tac_decision:
                        target = pick_target_from_ranking(active, opp_units, _active_target_ranking)
                    else:
                        target = pick_target(active, opp_units, target_multipliers=my_mults)
                    if target is not None:
                        resolve_shooting(active, target)
                        check_morale(target)
                        _sync_dead_models(target, board)

            # Per-activation objective capture reward
            if _pre_move_friendly_on_objs is not None and active is not None and active.models_alive > 0:
                if current_is_a:
                    _cap_reward = compute_objective_capture_reward(
                        active, units_a, board, "A", round_num, shaping_scale,
                        _pre_move_friendly_on_objs,
                    )
                    if _cap_reward != 0.0 and round_step_indices:
                        trajectory[round_step_indices[-1]].reward += _cap_reward
                elif is_mirror and round_step_indices_b:
                    _cap_reward = compute_objective_capture_reward(
                        active, units_b, board, "B", round_num, shaping_scale,
                        _pre_move_friendly_on_objs,
                    )
                    if _cap_reward != 0.0:
                        trajectory_b[round_step_indices_b[-1]].reward += _cap_reward

            current_is_a = not current_is_a

        # End of round: update objectives
        board.update_objectives(units_a, units_b)

        # Assign round kill reward to the last activation step of this round
        reward, prev_a_kill_pts, prev_b_kill_pts = compute_round_reward(
            units_a, units_b, board, "A", pts_a,
            prev_a_kill_pts, prev_b_kill_pts,
            shaping_scale=shaping_scale,
            round_num=round_num,
        )
        if round_step_indices:
            trajectory[round_step_indices[-1]].reward += reward

        if is_mirror and round_step_indices_b:
            reward_b, prev_b_fkp, prev_b_ekp = compute_round_reward(
                units_b, units_a, board, "B", pts_b,
                prev_b_fkp, prev_b_ekp,
                shaping_scale=shaping_scale,
                round_num=round_num,
            )
            trajectory_b[round_step_indices_b[-1]].reward = reward_b

        # Aux targets: short-horizon (end-of-current-round) survival + obj control
        snap_a = _make_round_snapshot(units_a, units_b, board, "A")
        for si in round_step_indices:
            trajectory[si].friendly_survival_target_short = snap_a.friendly_survival
            trajectory[si].enemy_survival_target_short = snap_a.enemy_survival
            trajectory[si].obj_control_target_short = snap_a.obj_control
        if is_mirror:
            snap_b = _make_round_snapshot(units_b, units_a, board, "B")
            for si in round_step_indices_b:
                trajectory_b[si].friendly_survival_target_short = snap_b.friendly_survival
                trajectory_b[si].enemy_survival_target_short = snap_b.enemy_survival
                trajectory_b[si].obj_control_target_short = snap_b.obj_control

    # Determine winner
    a_objs = board.count_objectives("A")
    b_objs = board.count_objectives("B")
    if a_objs > b_objs:
        result = "A"
    elif b_objs > a_objs:
        result = "B"
    else:
        result = "draw"

    if trajectory:
        trajectory[-1].reward += terminal_reward(result, "A", a_objs, b_objs)

    if is_mirror and trajectory_b:
        trajectory_b[-1].reward += terminal_reward(result, "B", a_objs, b_objs)

    # Aux targets: long-horizon (end-of-game) survival + obj control (backfill to all steps)
    final_snap_a = _make_round_snapshot(units_a, units_b, board, "A")
    for step in trajectory:
        step.friendly_survival_target = final_snap_a.friendly_survival
        step.enemy_survival_target = final_snap_a.enemy_survival
        step.obj_control_target = final_snap_a.obj_control
    if is_mirror and trajectory_b:
        final_snap_b = _make_round_snapshot(units_b, units_a, board, "B")
        for step in trajectory_b:
            step.friendly_survival_target = final_snap_b.friendly_survival
            step.enemy_survival_target = final_snap_b.enemy_survival
            step.obj_control_target = final_snap_b.obj_control

    # Activation countdown targets (backfill to all steps)
    for i, step in enumerate(trajectory):
        a_so_far, b_so_far = _traj_a_counts[i]
        step.friendly_activations_remaining = float(_a_act_total - a_so_far)
        step.enemy_activations_remaining = float(_b_act_total - b_so_far)
    if is_mirror and trajectory_b:
        for i, step in enumerate(trajectory_b):
            b_so_far, a_so_far = _traj_b_counts[i]
            step.friendly_activations_remaining = float(_b_act_total - b_so_far)
            step.enemy_activations_remaining = float(_a_act_total - a_so_far)

    return trajectory, result, opponent_type, trajectory_b


# ---------------------------------------------------------------------------
# Coroutine-batched tactical episode collection
# ---------------------------------------------------------------------------

def _episode_tactical_generator(opponent_model,
                                res_a, res_b,
                                states_a_data, states_b_data, opponent_type,
                                BOARD_OBJECTIVES, shaping_scale=1.0,
                                army_type="random",
                                planning_enabled=False):
    """Generator version of _run_single_episode_tactical for batched inference.

    Yields _TacticalInferenceRequest at each ML decision point.
    Receives _TacticalSamplingResult via generator.send().
    Returns (trajectory, game_result, opponent_type, trajectory_b) via StopIteration.value.

    Player A inference is always yielded (model_key="main").
    Player B inference is yielded for tactical opponents (model_key="opponent")
    or with model_key="main" for mirror self-play.
    """
    from game import deploy_armies, _collect_enemy_positions, _sync_dead_models
    from ai import (
        pick_target, choose_action_and_goal, activation_order,
        assign_objectives, reassign_roles,
    )
    from combat import resolve_shooting, check_morale, resolve_melee, resolve_impact, check_melee_morale
    from movement import (
        execute_movement, execute_charge_movement, execute_counter_charge,
        post_melee_separation, consolidation_move,
    )
    is_mirror = (opponent_type == "selfplay_mirror")
    opponent_type_idx = _get_opponent_type_idx(opponent_type, army_type)

    # Rebuild UnitState objects
    units_a = [UnitState(ru) for ru in res_a]
    for u in units_a:
        u.owner = "A"
    units_b = [UnitState(ru) for ru in res_b]
    for u in units_b:
        u.owner = "B"

    for u, (ai_role, combat_pref, assigned_obj) in zip(units_a, states_a_data):
        u.ai_role = ai_role
        u.combat_preference = combat_pref
        u.assigned_objective = assigned_obj
    for u, (ai_role, combat_pref, assigned_obj) in zip(units_b, states_b_data):
        u.ai_role = ai_role
        u.combat_preference = combat_pref
        u.assigned_objective = assigned_obj

    board = Board()
    deploy_armies(units_a, units_b, board)

    fr_a, fm_a = precompute_damage([u.unit for u in units_a], [u.unit for u in units_b])
    fr_b, fm_b = precompute_damage([u.unit for u in units_b], [u.unit for u in units_a])
    pts_a = sum(u.unit.points for u in units_a)
    pts_b = sum(u.unit.points for u in units_b)

    if opponent_type == "heuristic":
        assign_objectives(units_b)

    a_first = random.random() < 0.5
    a_finished_first = a_first

    trajectory: list[TacticalActivationRecord] = []
    trajectory_b: list[TacticalActivationRecord] | None = [] if is_mirror else None
    prev_a_kill_pts = 0.0
    prev_b_kill_pts = 0.0
    prev_b_fkp = 0.0
    prev_b_ekp = 0.0

    # Activation counters for countdown targets
    _a_act_total = 0
    _b_act_total = 0
    _traj_a_counts: list[tuple[int, int]] = []
    _traj_b_counts: list[tuple[int, int]] = []

    for round_num in range(1, 5):
        for u in units_a + units_b:
            u.activated = False
            u.fatigued = False

        current_is_a = a_first if round_num == 1 else a_finished_first

        # Player B round-start decisions (heuristic only; tactical opponents
        # and mirror decide per-activation, not per-round).
        target_mults_b = None
        if opponent_type == "heuristic":
            reassign_roles(units_b)

        round_step_indices: list[int] = []
        round_step_indices_b: list[int] = []

        a_done = False
        b_done = False
        a_finished_first = True

        _a_tac_action: str = "hold"
        _a_tac_goal = None
        _a_tac_charge_target = None
        _a_tac_target_ranking: list[int] = []

        while True:
            if current_is_a:
                my_units, opp_units = units_a, units_b
                my_mults = None
            else:
                my_units, opp_units = units_b, units_a
                my_mults = target_mults_b

            _opp_tac_decision = False

            if current_is_a:
                # --- Player A: yield for main model inference ---
                alive_mask_list = []
                for i in range(MAX_UNITS_PER_SIDE):
                    if i < len(units_a):
                        us = units_a[i]
                        alive_mask_list.append(us.models_alive > 0 and not us.activated)
                    else:
                        alive_mask_list.append(False)

                if not any(alive_mask_list):
                    a_done = True
                    if not b_done:
                        a_finished_first = True
                    if a_done and b_done:
                        break
                    current_is_a = not current_is_a
                    continue

                alive_mask = torch.tensor(alive_mask_list, dtype=torch.bool)

                enemy_alive_mask_list = [
                    (i < len(units_b) and units_b[i].models_alive > 0)
                    for i in range(MAX_UNITS_PER_SIDE)
                ]
                enemy_alive_mask = torch.tensor(enemy_alive_mask_list, dtype=torch.bool)

                state_vec = encode_state_tactical(
                    units_a, units_b, round_num, board, "A",
                    friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
                    enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
                    total_friendly_points=pts_a, total_enemy_points=pts_b,
                )
                state_vec_np = state_vec.numpy()

                # Compute model-space positions for inference
                a_friendly_pos = _get_model_space_positions(units_a, "A")
                a_enemy_pos = _get_model_space_positions(units_b, "A")
                a_adv_dists, a_rush_dists = _get_movement_budgets(units_a)
                a_max_wr = _get_max_weapon_ranges(units_a)

                # >>> YIELD for batched inference (or planning, decided by coordinator) <<<
                _req = _TacticalInferenceRequest(
                    state_vec, alive_mask, enemy_alive_mask, "main",
                    a_friendly_pos, a_enemy_pos, a_adv_dists, a_rush_dists,
                    a_max_wr, opponent_type_idx=opponent_type_idx,
                )
                if planning_enabled:
                    _req.planning_units_a = units_a
                    _req.planning_units_b = units_b
                    _req.planning_board = board
                    _req.planning_round_num = round_num
                    _req.planning_current_is_a = current_is_a
                    _req.planning_fr_a = fr_a
                    _req.planning_fm_a = fm_a
                    _req.planning_fr_b = fr_b
                    _req.planning_fm_b = fm_b
                    _req.planning_pts_a = pts_a
                    _req.planning_pts_b = pts_b
                    _req.planning_opponent_type_idx = opponent_type_idx

                _inf_result = yield _req

                sel_idx = _inf_result.unit_idx
                move_type_a = _inf_result.move_type
                sampled_angle_a = _inf_result.sampled_angle
                sampled_frac_a = _inf_result.sampled_distance_frac
                charge_tgt_a = _inf_result.charge_target_idx
                shoot_tgt_a = _inf_result.shoot_target_idx
                _a_tac_target_ranking = _inf_result.target_ranking
                pmr_a = _inf_result.post_move_rel
                old_lp = _inf_result.old_log_prob
                value_est = _inf_result.value
                shoot_mask_a = _inf_result.shoot_mask

                step = TacticalActivationRecord(
                    state_vec=state_vec_np,
                    alive_mask=alive_mask_list,
                    enemy_alive_mask=enemy_alive_mask_list,
                    unit_idx=sel_idx,
                    move_type=move_type_a,
                    sampled_angle=sampled_angle_a,
                    sampled_distance_frac=sampled_frac_a,
                    charge_target_idx=charge_tgt_a,
                    shoot_target_idx=shoot_tgt_a,
                    shoot_mask=shoot_mask_a,
                    post_move_rel=pmr_a,
                    old_log_prob=old_lp,
                    old_value=value_est,
                    opponent_type_idx=opponent_type_idx,
                    was_planned=_inf_result.was_planned,
                    planning_improved=_inf_result.planning_improved,
                    planning_value_delta=_inf_result.planning_value_delta,
                    planning_unit_values=_inf_result.planning_unit_values,
                    planning_unit_indices=_inf_result.planning_unit_indices,
                    planning_move_values=_inf_result.planning_move_values,
                    planning_move_indices=_inf_result.planning_move_indices,
                    planning_charge_values=_inf_result.planning_charge_values,
                    planning_charge_indices=_inf_result.planning_charge_indices,
                    planning_shoot_values=_inf_result.planning_shoot_values,
                    planning_shoot_indices=_inf_result.planning_shoot_indices,
                )
                round_step_indices.append(len(trajectory))
                trajectory.append(step)
                _a_act_total += 1
                _traj_a_counts.append((_a_act_total, _b_act_total))

                active = units_a[sel_idx]
                active.activated = True

                _a_dest = None
                if move_type_a in (MOVE_ADVANCE, MOVE_RUSH):
                    ucx, ucy = a_friendly_pos[sel_idx]
                    budget = a_adv_dists[sel_idx] if move_type_a == MOVE_ADVANCE else a_rush_dists[sel_idx]
                    px, py = compute_post_move_position(ucx, ucy, sampled_angle_a, sampled_frac_a * budget)
                    _a_dest = (px, py)

                _a_tac_action, _a_tac_goal, _a_tac_charge_target, _a_reason = execute_decoded_decision(
                    active, units_b, move_type_a, _a_dest, charge_tgt_a, shoot_tgt_a,
                )
            else:
                # --- Player B ---
                _b_target_ranking: list[int] = []
                _b_action = "hold"
                _b_goal = None
                _b_charge_target = None

                if is_mirror:
                    # Mirror self-play: yield B inference via main model
                    b_alive_list = []
                    for i in range(MAX_UNITS_PER_SIDE):
                        if i < len(units_b):
                            us = units_b[i]
                            b_alive_list.append(us.models_alive > 0 and not us.activated)
                        else:
                            b_alive_list.append(False)

                    if not any(b_alive_list):
                        active = None
                    else:
                        b_alive_mask = torch.tensor(b_alive_list, dtype=torch.bool)
                        b_enemy_alive_list = [
                            (i < len(units_a) and units_a[i].models_alive > 0)
                            for i in range(MAX_UNITS_PER_SIDE)
                        ]
                        b_enemy_alive_mask = torch.tensor(b_enemy_alive_list, dtype=torch.bool)
                        b_state_vec = encode_state_tactical(
                            units_b, units_a, round_num, board, "B",
                            friendly_ranged_matchups=fr_b, friendly_melee_matchups=fm_b,
                            enemy_ranged_matchups=fr_a, enemy_melee_matchups=fm_a,
                            total_friendly_points=pts_b, total_enemy_points=pts_a,
                        )
                        b_state_vec_np = b_state_vec.numpy()

                        b_friendly_pos = _get_model_space_positions(units_b, "B")
                        b_enemy_pos = _get_model_space_positions(units_a, "B")
                        b_adv_dists, b_rush_dists = _get_movement_budgets(units_b)
                        b_max_wr = _get_max_weapon_ranges(units_b)

                        # >>> YIELD for batched main-model inference (mirror B) <<<
                        _b_inf = yield _TacticalInferenceRequest(
                            b_state_vec, b_alive_mask, b_enemy_alive_mask, "main",
                            b_friendly_pos, b_enemy_pos, b_adv_dists, b_rush_dists,
                            b_max_wr, opponent_type_idx=opponent_type_idx,
                        )

                        sel_b = _b_inf.unit_idx
                        if (sel_b < len(units_b)
                                and units_b[sel_b].models_alive > 0
                                and not units_b[sel_b].activated):
                            active = units_b[sel_b]
                            _b_target_ranking = _b_inf.target_ranking

                            step_b = TacticalActivationRecord(
                                state_vec=b_state_vec_np,
                                alive_mask=b_alive_list,
                                enemy_alive_mask=b_enemy_alive_list,
                                unit_idx=sel_b,
                                move_type=_b_inf.move_type,
                                sampled_angle=_b_inf.sampled_angle,
                                sampled_distance_frac=_b_inf.sampled_distance_frac,
                                charge_target_idx=_b_inf.charge_target_idx,
                                shoot_target_idx=_b_inf.shoot_target_idx,
                                shoot_mask=_b_inf.shoot_mask,
                                post_move_rel=_b_inf.post_move_rel,
                                old_log_prob=_b_inf.old_log_prob,
                                old_value=_b_inf.value,
                                opponent_type_idx=opponent_type_idx,
                            )
                            round_step_indices_b.append(len(trajectory_b))
                            trajectory_b.append(step_b)
                            _b_act_total += 1
                            _traj_b_counts.append((_b_act_total, _a_act_total))

                            _b_dest = None
                            if _b_inf.move_type in (MOVE_ADVANCE, MOVE_RUSH):
                                bcx, bcy = b_friendly_pos[sel_b]
                                bgt = b_adv_dists[sel_b] if _b_inf.move_type == MOVE_ADVANCE else b_rush_dists[sel_b]
                                bpx, bpy = compute_post_move_position(bcx, bcy, _b_inf.sampled_angle, _b_inf.sampled_distance_frac * bgt)
                                _b_dest = (_flip_x(bpx), _flip_y(bpy))

                            _b_action, _b_goal, _b_charge_target, _ = execute_decoded_decision(
                                active, units_a, _b_inf.move_type, _b_dest,
                                _b_inf.charge_target_idx, _b_inf.shoot_target_idx,
                            )
                            active.activated = True
                        else:
                            active = None

                    _opp_tac_decision = active is not None

                elif opponent_model is not None:
                    # Build B's masks and encode state, then yield
                    b_alive_list = []
                    for i in range(MAX_UNITS_PER_SIDE):
                        if i < len(units_b):
                            us = units_b[i]
                            b_alive_list.append(us.models_alive > 0 and not us.activated)
                        else:
                            b_alive_list.append(False)

                    if not any(b_alive_list):
                        active = None
                    else:
                        b_alive_mask = torch.tensor(b_alive_list, dtype=torch.bool)
                        b_enemy_alive_mask = torch.tensor(
                            [(i < len(units_a) and units_a[i].models_alive > 0)
                             for i in range(MAX_UNITS_PER_SIDE)],
                            dtype=torch.bool,
                        )
                        b_state_vec = encode_state_tactical(
                            units_b, units_a, round_num, board, "B",
                            friendly_ranged_matchups=fr_b, friendly_melee_matchups=fm_b,
                            enemy_ranged_matchups=fr_a, enemy_melee_matchups=fm_a,
                            total_friendly_points=pts_b, total_enemy_points=pts_a,
                        )

                        b_friendly_pos_opp = _get_model_space_positions(units_b, "B")
                        b_enemy_pos_opp = _get_model_space_positions(units_a, "B")
                        b_adv_dists_opp, b_rush_dists_opp = _get_movement_budgets(units_b)
                        b_max_wr_opp = _get_max_weapon_ranges(units_b)

                        # >>> YIELD for batched opponent-model inference <<<
                        _b_inf = yield _TacticalInferenceRequest(
                            b_state_vec, b_alive_mask, b_enemy_alive_mask, "opponent",
                            b_friendly_pos_opp, b_enemy_pos_opp, b_adv_dists_opp, b_rush_dists_opp,
                            b_max_wr_opp,
                        )

                        sel_b = _b_inf.unit_idx
                        if (sel_b < len(units_b)
                                and units_b[sel_b].models_alive > 0
                                and not units_b[sel_b].activated):
                            active = units_b[sel_b]
                            _b_target_ranking = _b_inf.target_ranking
                            _b_dest_opp = None
                            if _b_inf.move_type in (MOVE_ADVANCE, MOVE_RUSH):
                                bcx, bcy = b_friendly_pos_opp[sel_b]
                                bgt = b_adv_dists_opp[sel_b] if _b_inf.move_type == MOVE_ADVANCE else b_rush_dists_opp[sel_b]
                                bpx, bpy = compute_post_move_position(bcx, bcy, _b_inf.sampled_angle, _b_inf.sampled_distance_frac * bgt)
                                _b_dest_opp = (_flip_x(bpx), _flip_y(bpy))
                            _b_action, _b_goal, _b_charge_target, _ = execute_decoded_decision(
                                active, units_a, _b_inf.move_type, _b_dest_opp,
                                _b_inf.charge_target_idx, _b_inf.shoot_target_idx,
                            )
                        else:
                            active = None

                    _opp_tac_decision = active is not None
                else:
                    ordered = activation_order(my_units, enemies=opp_units, mode="objectives")
                    active = ordered[0] if ordered else None

                if active is None:
                    b_done = True
                    if not a_done:
                        a_finished_first = False
                    if a_done and b_done:
                        break
                    current_is_a = not current_is_a
                    continue

                if not (is_mirror and _opp_tac_decision):
                    active.activated = True
                # Count B activations (non-mirror; mirror already counted above)
                if not is_mirror:
                    _b_act_total += 1

            # --- Execute the activation (identical to _run_single_episode_tactical) ---
            if current_is_a:
                action = _a_tac_action
                goal = _a_tac_goal
                charge_target = _a_tac_charge_target
                _active_target_ranking = _a_tac_target_ranking
            elif _opp_tac_decision:
                action, goal, charge_target = _b_action, _b_goal, _b_charge_target
                _active_target_ranking = _b_target_ranking
            else:
                action, goal, charge_target, _reason = choose_action_and_goal(
                    active, opp_units, board, mode="objectives",
                    target_multipliers=my_mults,
                )
                _active_target_ranking = []

            # Pre-move snapshot: which objectives already have a friendly unit on them?
            _pre_move_friendly_on_objs_g: list[bool] | None = None
            if current_is_a or (is_mirror and _opp_tac_decision):
                _obj_friendly_g = units_a if current_is_a else units_b
                _pre_move_friendly_on_objs_g = [
                    _any_friendly_on_objective(obj_pos, _obj_friendly_g)
                    for obj_pos in _OBJECTIVES_FOR_REWARD
                ]

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
                    consolidation_move(active, board, opp_units, BOARD_OBJECTIVES, "objectives")
                elif charge_target.models_alive > 0:
                    consolidation_move(charge_target, board, my_units, BOARD_OBJECTIVES, "objectives")

            elif action in ("advance", "rush") and goal is not None:
                budget = active.unit.advance_distance if action == "advance" else active.unit.rush_distance
                enemy_positions = _collect_enemy_positions(opp_units)
                execute_movement(active, goal, budget, board, enemy_positions,
                                 flying=active.unit.flying)

                if action != "rush":
                    if active.shaken:
                        active.shaken = False
                    else:
                        if current_is_a or _opp_tac_decision:
                            target = pick_target_from_ranking(active, opp_units, _active_target_ranking)
                        else:
                            target = pick_target(active, opp_units, target_multipliers=my_mults)
                        if target is not None:
                            resolve_shooting(active, target)
                            check_morale(target)
                            _sync_dead_models(target, board)

            elif action == "hold":
                if active.shaken:
                    active.shaken = False
                else:
                    if current_is_a or _opp_tac_decision:
                        target = pick_target_from_ranking(active, opp_units, _active_target_ranking)
                    else:
                        target = pick_target(active, opp_units, target_multipliers=my_mults)
                    if target is not None:
                        resolve_shooting(active, target)
                        check_morale(target)
                        _sync_dead_models(target, board)

            # Per-activation objective capture reward
            if _pre_move_friendly_on_objs_g is not None and active is not None and active.models_alive > 0:
                if current_is_a:
                    _cap_reward = compute_objective_capture_reward(
                        active, units_a, board, "A", round_num, shaping_scale,
                        _pre_move_friendly_on_objs_g,
                    )
                    if _cap_reward != 0.0 and round_step_indices:
                        trajectory[round_step_indices[-1]].reward += _cap_reward
                elif is_mirror and round_step_indices_b:
                    _cap_reward = compute_objective_capture_reward(
                        active, units_b, board, "B", round_num, shaping_scale,
                        _pre_move_friendly_on_objs_g,
                    )
                    if _cap_reward != 0.0:
                        trajectory_b[round_step_indices_b[-1]].reward += _cap_reward

            current_is_a = not current_is_a

        # End of round
        board.update_objectives(units_a, units_b)

        reward, prev_a_kill_pts, prev_b_kill_pts = compute_round_reward(
            units_a, units_b, board, "A", pts_a,
            prev_a_kill_pts, prev_b_kill_pts,
            shaping_scale=shaping_scale,
            round_num=round_num,
        )
        if round_step_indices:
            trajectory[round_step_indices[-1]].reward += reward

        if is_mirror and round_step_indices_b:
            reward_b, prev_b_fkp, prev_b_ekp = compute_round_reward(
                units_b, units_a, board, "B", pts_b,
                prev_b_fkp, prev_b_ekp,
                shaping_scale=shaping_scale,
                round_num=round_num,
            )
            trajectory_b[round_step_indices_b[-1]].reward += reward_b

        # Aux targets: short-horizon (end-of-current-round) survival + obj control
        snap_a = _make_round_snapshot(units_a, units_b, board, "A")
        for si in round_step_indices:
            trajectory[si].friendly_survival_target_short = snap_a.friendly_survival
            trajectory[si].enemy_survival_target_short = snap_a.enemy_survival
            trajectory[si].obj_control_target_short = snap_a.obj_control
        if is_mirror:
            snap_b = _make_round_snapshot(units_b, units_a, board, "B")
            for si in round_step_indices_b:
                trajectory_b[si].friendly_survival_target_short = snap_b.friendly_survival
                trajectory_b[si].enemy_survival_target_short = snap_b.enemy_survival
                trajectory_b[si].obj_control_target_short = snap_b.obj_control

    # Determine winner
    a_objs = board.count_objectives("A")
    b_objs = board.count_objectives("B")
    if a_objs > b_objs:
        result = "A"
    elif b_objs > a_objs:
        result = "B"
    else:
        result = "draw"

    if trajectory:
        trajectory[-1].reward += terminal_reward(result, "A", a_objs, b_objs)

    if is_mirror and trajectory_b:
        trajectory_b[-1].reward += terminal_reward(result, "B", a_objs, b_objs)

    # Aux targets: long-horizon (end-of-game) survival + obj control (backfill to all steps)
    final_snap_a = _make_round_snapshot(units_a, units_b, board, "A")
    for step in trajectory:
        step.friendly_survival_target = final_snap_a.friendly_survival
        step.enemy_survival_target = final_snap_a.enemy_survival
        step.obj_control_target = final_snap_a.obj_control
    if is_mirror and trajectory_b:
        final_snap_b = _make_round_snapshot(units_b, units_a, board, "B")
        for step in trajectory_b:
            step.friendly_survival_target = final_snap_b.friendly_survival
            step.enemy_survival_target = final_snap_b.enemy_survival
            step.obj_control_target = final_snap_b.obj_control

    # Activation countdown targets (backfill to all steps)
    for i, step in enumerate(trajectory):
        a_so_far, b_so_far = _traj_a_counts[i]
        step.friendly_activations_remaining = float(_a_act_total - a_so_far)
        step.enemy_activations_remaining = float(_b_act_total - b_so_far)
    if is_mirror and trajectory_b:
        for i, step in enumerate(trajectory_b):
            b_so_far, a_so_far = _traj_b_counts[i]
            step.friendly_activations_remaining = float(_b_act_total - b_so_far)
            step.enemy_activations_remaining = float(_a_act_total - a_so_far)

    return trajectory, result, opponent_type, trajectory_b


def _run_games_batched_tactical(
    main_model: TacticalModel,
    game_specs: list,
    opp_models: dict,
    shaping_scale: float = 1.0,
    planning_rate: float = 0.0,
    planning_params: dict | None = None,
) -> list[tuple]:
    """Run multiple tactical training games with batched inference.

    Creates generator coroutines for each game and advances them in lockstep,
    batching main-model and opponent-model forward passes separately.

    Returns list of (trajectory, result, opponent_type, army_type) per game.
    """
    from board import OBJECTIVES as BOARD_OBJECTIVES

    # Create generators and track opponent models for tactical opponents
    generators: list = []
    game_army_types: list[str] = []
    game_opp_tactical_models: dict[int, nn.Module] = {}

    for i, (res_a, res_b, sa_data, sb_data, opp_type, opp_sd_idx, army_type) in enumerate(game_specs):
        opp_model = opp_models.get(opp_sd_idx)

        if opp_model is not None:
            game_opp_tactical_models[i] = opp_model

        gen = _episode_tactical_generator(
            None,
            res_a, res_b, sa_data, sb_data, opp_type, BOARD_OBJECTIVES,
            shaping_scale=shaping_scale,
            army_type=army_type,
            planning_enabled=(planning_rate > 0),
        )
        generators.append(gen)
        game_army_types.append(army_type)

    # Initialize all generators (advance to first yield or completion)
    active: dict[int, tuple] = {}
    finished: dict[int, tuple] = {}

    for i, gen in enumerate(generators):
        try:
            req = next(gen)
            active[i] = (gen, req)
        except StopIteration as e:
            finished[i] = e.value

    # Main batching loop
    while active:
        # Group requests by model
        main_gids: list[int] = []
        main_reqs: list[_TacticalInferenceRequest] = []
        opp_by_model: dict[int, tuple] = {}

        for gid, (gen, req) in active.items():
            if req.model_key == "main":
                main_gids.append(gid)
                main_reqs.append(req)
            else:
                opp_model = game_opp_tactical_models[gid]
                mid = id(opp_model)
                if mid not in opp_by_model:
                    opp_by_model[mid] = (opp_model, [])
                opp_by_model[mid][1].append((gid, req))

        all_results: dict[int, _TacticalSamplingResult] = {}

        # Synchronized planning decision: roll once for all Player A requests
        use_planning_this_round = (
            planning_rate > 0
            and main_reqs
            and random.random() < planning_rate
        )

        if use_planning_this_round:
            # Planning round: run plan_training_activation for each A request
            from ml_planning import plan_training_activation
            for gid, req in zip(main_gids, main_reqs):
                if req.planning_units_a is not None:
                    (uid, mt, ang, frac, ct, st, ranking,
                     pmr, olp, val, sm,
                     wp, pi, pvd, puv, pui,
                     pmv, pmi, pcv, pci, psv, psi,
                    ) = plan_training_activation(
                        main_model, req.state_vec, req.alive_mask,
                        req.enemy_alive_mask,
                        req.planning_units_a, req.planning_units_b,
                        req.planning_round_num, req.planning_board, "A",
                        current_is_a=req.planning_current_is_a,
                        mode="objectives",
                        friendly_positions=req.friendly_positions,
                        enemy_positions=req.enemy_positions,
                        advance_distances=req.advance_distances,
                        rush_distances=req.rush_distances,
                        max_weapon_ranges=req.max_weapon_ranges,
                        fr_a=req.planning_fr_a, fm_a=req.planning_fm_a,
                        fr_b=req.planning_fr_b, fm_b=req.planning_fm_b,
                        pts_a=req.planning_pts_a, pts_b=req.planning_pts_b,
                        planning_params=planning_params,
                        opponent_type=req.planning_opponent_type_idx,
                    )
                    all_results[gid] = _TacticalSamplingResult(
                        unit_idx=uid, move_type=mt,
                        sampled_angle=ang, sampled_distance_frac=frac,
                        charge_target_idx=ct, shoot_target_idx=st,
                        target_ranking=ranking, post_move_rel=pmr,
                        old_log_prob=olp, value=val, shoot_mask=sm,
                        was_planned=wp, planning_improved=pi,
                        planning_value_delta=pvd,
                        planning_unit_values=puv,
                        planning_unit_indices=pui,
                        planning_move_values=pmv,
                        planning_move_indices=pmi,
                        planning_charge_values=pcv,
                        planning_charge_indices=pci,
                        planning_shoot_values=psv,
                        planning_shoot_indices=psi,
                    )
                else:
                    # B-side mirror request with model_key="main" — no planning
                    pass

            # Any main requests not handled (e.g. mirror B) get normal inference
            unhandled_gids = [g for g in main_gids if g not in all_results]
            if unhandled_gids:
                unhandled_reqs = [main_reqs[main_gids.index(g)]
                                  for g in unhandled_gids]
                batch_results = _batched_sample_tactical_no_grad(
                    main_model, unhandled_reqs)
                for gid, res in zip(unhandled_gids, batch_results):
                    all_results[gid] = res
        else:
            # Normal round: batched inference for all main requests
            if main_reqs:
                batch_results = _batched_sample_tactical_no_grad(
                    main_model, main_reqs)
                for gid, res in zip(main_gids, batch_results):
                    all_results[gid] = res

        # Batch each opponent model separately (never planned)
        for mid, (opp_m, gid_reqs) in opp_by_model.items():
            reqs = [r for _, r in gid_reqs]
            batch_results = _batched_sample_tactical_no_grad(opp_m, reqs)
            for (gid, _), res in zip(gid_reqs, batch_results):
                all_results[gid] = res

        # Advance generators with results
        new_active: dict[int, tuple] = {}
        for gid, res in all_results.items():
            gen = active[gid][0]
            try:
                next_req = gen.send(res)
                new_active[gid] = (gen, next_req)
            except StopIteration as e:
                finished[gid] = e.value

        active = new_active

    # Return results in original order, adding army_type
    results = []
    for i in range(len(generators)):
        traj, result, opp_type, traj_b = finished[i]
        results.append((traj, result, opp_type, game_army_types[i]))
        if traj_b is not None:
            results.append((traj_b, result, "mirror_b", game_army_types[i]))
    return results
