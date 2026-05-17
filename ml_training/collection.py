"""Episode collection: single episodes, generators, workers, and batched inference."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import Board
from models import ResolvedUnit, UnitState
from ml_features import (
    MAX_UNITS_PER_SIDE, TACTICAL_UNIT_FEATURES, encode_state_tactical,
    precompute_damage, extract_can_charge_mask,
    MAX_DEST_CANDIDATES, DEST_FEATURE_DIM,
    _flip_x, _flip_y,
)
from ml_model_tactical import (
    TacticalModel, NUM_MOVE_TYPES, MOVE_MOVE, MOVE_CHARGE,
)
from ml_integration_tactical import (
    MOVE_TYPE_NAMES, execute_decoded_decision, pick_target_from_ranking,
    compute_post_move_rel,
    compute_destination_candidates, compute_destination_features,
    compute_unit_visibility_arrays, compute_unit_expected_damage_arrays,
    build_dest_enemy_cache,
    _get_model_space_positions, _get_movement_budgets, _get_max_weapon_ranges,
    project_post_move_unit_state, is_phase_reencode_enabled,
)

from ml_training.config import (
    TacticalActivationRecord, _TacticalInferenceRequest, _TacticalSamplingResult,
    _get_opponent_type_idx,
)
from ml_training.rewards import (
    compute_round_reward, terminal_reward, _make_round_snapshot,
    compute_objective_capture_reward, _any_friendly_on_objective,
    compute_shooting_efficiency_reward,
    compute_charge_efficiency_reward,
)
from board import OBJECTIVES as _OBJECTIVES_FOR_REWARD, OBJ_SEIZE_RANGE as _OBJ_SEIZE_RANGE
from ml_training.sampling import sample_tactical_actions_no_grad, _batched_sample_tactical_no_grad
from ml_training.checkpoint import _make_model

# ---------------------------------------------------------------------------
# Parallel episode collection (worker + replay)
# ---------------------------------------------------------------------------

_WORKER_COUNT = 6

# Maximum number of shared-memory opponent model slots
_MAX_SHARED_OPPONENTS = 20

# ---------------------------------------------------------------------------
# Shared-memory worker globals (set by _init_shared_worker in child processes)
# ---------------------------------------------------------------------------

_g_shared_model: nn.Module | None = None
_g_shared_opponents: list[nn.Module] = []
_g_worker_model_type: str = "tactical"
_g_shared_v_old: nn.Module | None = None


def _maybe_build_post_move_state_vec(
    *,
    units_friendly: list,
    units_enemy: list,
    round_num: int,
    board,
    model_side: str,
    sel_idx: int,
    move_type: int,
    dest_candidates,
    dest_selected_idx: int,
    dest_advance_reachable,
    fr_matchups, fm_matchups,
    er_matchups, em_matchups,
    pts_friendly, pts_enemy,
) -> np.ndarray | None:
    """Build the post-move state_vec for the POST_DEST trunk re-encode.

    Returns None when the phase-reencode flag is off, on charge/shaken
    activations, or when the destination is invalid — replay reuses state_vec
    in those cases (same h for all phases given state_vec_post == state_vec).

    Only called once per activation during collection; the ~16 KB/activation
    storage cost is the dominant buffer-size impact of the refactor.
    """
    if not is_phase_reencode_enabled():
        return None
    if move_type != MOVE_MOVE:
        return None
    selected_unit = units_friendly[sel_idx]
    if selected_unit.shaken:
        return None
    if dest_selected_idx < 0 or len(dest_candidates) == 0:
        return None

    dest_hex = dest_candidates[dest_selected_idx]
    dest_col = int(dest_hex[0])
    dest_row = int(dest_hex[1])
    is_rush = True
    if dest_advance_reachable is not None and dest_selected_idx < len(dest_advance_reachable):
        is_rush = not bool(dest_advance_reachable[dest_selected_idx])

    post_unit = project_post_move_unit_state(
        selected_unit, (dest_col, dest_row), is_rush=is_rush,
    )
    friendly_post = list(units_friendly)
    friendly_post[sel_idx] = post_unit

    state_vec_post = encode_state_tactical(
        friendly_post, units_enemy, round_num, board, model_side,
        friendly_ranged_matchups=fr_matchups,
        friendly_melee_matchups=fm_matchups,
        enemy_ranged_matchups=er_matchups,
        enemy_melee_matchups=em_matchups,
        total_friendly_points=pts_friendly,
        total_enemy_points=pts_enemy,
    )
    return state_vec_post.numpy()


def _init_shared_worker(shared_model, shared_opponents, model_type="tactical",
                         use_c_ext=True, phase_reencode_enabled=True,
                         shared_v_old=None):
    """Initialize worker process with references to shared-memory models."""
    global _g_shared_model, _g_shared_opponents, _g_worker_model_type
    global _g_shared_v_old
    _g_shared_model = shared_model
    _g_shared_opponents = shared_opponents
    _g_worker_model_type = model_type
    _g_shared_v_old = shared_v_old
    # Each worker runs small single-sample inferences — using multiple torch
    # threads per worker causes massive oversubscription (8 workers × 8 threads
    # = 64 threads on 16 logical cores).  Pin to 1 thread per worker.
    torch.set_num_threads(1)
    # Toggle C extension in worker processes
    import fast_core
    fast_core.USE_C_EXT = use_c_ext and fast_core.is_available()
    # Worker processes have their own module-level state — flag must be set
    # here or they'll collect under the legacy path while the main process
    # uses the phased path.
    from ml_integration_tactical import set_phase_reencode_enabled
    set_phase_reencode_enabled(phase_reencode_enabled)


def _collect_episodes_shared_worker(args) -> list[tuple[list[TacticalActivationRecord], str, str, str]]:
    """Run training episodes using shared-memory models.

    Like _collect_episodes_chunked_worker but reads model weights directly from
    shared memory instead of deserializing state dicts.  Only lightweight
    game specs and an opponent slot map are sent via IPC.

    Args is (opp_slot_map, game_specs, shaping_scale[, planning_config[, v_old_shaping]])
    where opp_slot_map maps opp_sd_index -> index into _g_shared_opponents (or
    absent for heuristic). shaping_scale controls the per-round reward shaping
    magnitude (1.0 = full, 0.0 = off). v_old_shaping flips heuristic shaping
    off and applies V_old potential-based shaping using _g_shared_v_old.

    Returns list of (trajectory_rounds, result, opponent_type, army_type).
    """
    v_old_shaping = False
    map_data = None
    train_deployment = False
    if len(args) >= 7:
        (opp_slot_map, game_specs, shaping_scale, planning_config,
         v_old_shaping, map_data, train_deployment) = args
    elif len(args) >= 5:
        opp_slot_map, game_specs, shaping_scale, planning_config, v_old_shaping = args
    elif len(args) >= 4:
        opp_slot_map, game_specs, shaping_scale, planning_config = args
    else:
        opp_slot_map, game_specs, shaping_scale = args
        planning_config = None

    from board import OBJECTIVES as BOARD_OBJECTIVES

    # Map install: workers run in their own process — they need apply_map to
    # mutate the worker's module-level OBJECTIVES (the engine's downstream
    # readers, e.g. ml_features.encode_state_tactical, import OBJECTIVES at
    # module load and use it for objective-cell features). One call per worker
    # is enough: the mutation is idempotent for the same map.
    if map_data is not None:
        from board import Board as _Board
        from map_loader import apply_map as _apply_map
        _apply_map(_Board(), map_data, build_vis_cover=False)

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

    v_old_model = _g_shared_v_old if v_old_shaping else None
    return _run_games_batched_tactical(model, game_specs, opp_models,
                                       shaping_scale=shaping_scale,
                                       planning_rate=planning_rate,
                                       planning_params=planning_params,
                                       v_old_model=v_old_model,
                                       map_data=map_data,
                                       train_deployment=train_deployment)




# ---------------------------------------------------------------------------
# Coroutine-batched tactical episode collection
# ---------------------------------------------------------------------------

def _episode_tactical_generator(opponent_model,
                                res_a, res_b,
                                states_a_data, states_b_data, opponent_type,
                                BOARD_OBJECTIVES, shaping_scale=1.0,
                                army_type="random",
                                planning_enabled=False,
                                has_tactical_opponent=False,
                                model_side: str = "A",
                                map_data=None,
                                main_model_for_deploy=None,
                                attach_a: list[tuple[int, bool]] | None = None,
                                attach_b: list[tuple[int, bool]] | None = None):
    """Run one training episode with the tactical per-activation model.

    Yields _TacticalInferenceRequest at each ML decision point.
    Receives _TacticalSamplingResult via generator.send().
    Returns (trajectory, game_result, opponent_type, trajectory_b) via StopIteration.value.

    The main trajectory holds the learning model's activations. Its physical
    side is chosen by ``model_side`` ("A" or "B"); for mirror self-play the
    main trajectory always holds side A and trajectory_b holds side B, so
    ``model_side`` is effectively ignored.

    Caller convention: ``res_a``/``states_a_data`` describe the army deployed
    on physical side A (rows 0-11); ``res_b`` describes the army on side B.
    ``model_side`` tells this generator which physical side the main learning
    model occupies. The caller is responsible for swapping the positional
    args when randomizing sides.
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

    # For mirror self-play, both sides are the main model — we fix
    # model_side to "A" so the A branch / mirror-B split in the activation
    # loop keeps working unchanged. For non-mirror games the caller can
    # randomise which physical side the main model plays.
    if is_mirror:
        model_side = "A"
    _opp_side = "B" if model_side == "A" else "A"

    # Internal convention: ``units_a`` is ALWAYS the main model's army and
    # ``units_b`` is ALWAYS the opponent's, regardless of which physical
    # side they occupy. Owner tags match the physical side so that the game
    # engine (which counts objectives by owner) still works.

    # Rebuild UnitState objects — units_a is ALWAYS the main model's army.
    # Owner tags match physical deployment: main on physical model_side,
    # opponent on physical _opp_side.
    #
    # res_a/res_b are PER-ENTRY (one ResolvedUnit per ArmyList entry,
    # including hero entries that are meant to be merged into a host). To
    # reach the post-merge shape that states_a_data describes, we have to
    # redo the hero merge here using attach_a/attach_b. Without this step,
    # hero entries on HoF armies (where forceorg permits entries=11 with
    # attached=1) survive as standalone UnitStates and the resulting
    # 11-unit army overflows MAX_UNITS_PER_SIDE during deployment.
    def _build_side(res, attach, side_label):
        all_units = [UnitState(copy.copy(ru)) for ru in res]
        for u in all_units:
            u.owner = side_label
        merged: set[int] = set()
        if attach is not None:
            for i, (attached_to, is_hero) in enumerate(attach):
                if i >= len(all_units):
                    break
                if (attached_to is None or attached_to < 0 or not is_hero
                        or attached_to >= len(all_units) or attached_to == i):
                    continue
                merge_hero_into_unit(res[i], all_units[attached_to])
                merged.add(i)
        return [u for i, u in enumerate(all_units) if i not in merged]

    import copy
    from models import merge_hero_into_unit
    units_a = _build_side(res_a, attach_a, model_side)
    units_b = _build_side(res_b, attach_b, _opp_side)

    # ai_role/combat_preference/assigned_objective come from the POST-merge
    # state list — its length matches units_a/units_b now that the merge
    # mirrors _make_unit_states.
    for u, (ai_role, combat_pref, assigned_obj) in zip(units_a, states_a_data):
        u.ai_role = ai_role
        u.combat_preference = combat_pref
        u.assigned_objective = assigned_obj
    for u, (ai_role, combat_pref, assigned_obj) in zip(units_b, states_b_data):
        u.ai_role = ai_role
        u.combat_preference = combat_pref
        u.assigned_objective = assigned_obj

    board = Board()
    if map_data is not None:
        from map_loader import apply_map as _apply_map
        _apply_map(board, map_data)

    # Build per-side deployment decision functions. When main_model_for_deploy
    # is provided, the main model drives placement for its side (and, in
    # mirror self-play, the opponent side too — both are the same model).
    # Records are appended into deploy_records_* so the PPO step can replay
    # the deploy heads. Without a model, deploy_armies defaults to the
    # legacy role-anchored heuristic.
    deploy_records_main: list = []
    deploy_records_opp: list = []
    fn_main = fn_opp = None
    if main_model_for_deploy is not None:
        from ml_training.deploy_collection import make_model_deploy_decision_fn
        # Main learning model controls "model_side"; the opponent side runs
        # the model too in mirror self-play, and otherwise either runs an
        # opponent checkpoint (only the main side's records are used for the
        # PPO update) or the legacy heuristic.
        fn_main = make_model_deploy_decision_fn(
            main_model_for_deploy, player=model_side,
            opponent_type_idx=opponent_type_idx, side_idx=(0 if model_side == "A" else 1),
            record_into=deploy_records_main,
        )
        _opponent_deploy_model = None
        if is_mirror:
            _opponent_deploy_model = main_model_for_deploy
        elif has_tactical_opponent and opponent_model is not None:
            _opponent_deploy_model = opponent_model
        if _opponent_deploy_model is not None:
            fn_opp = make_model_deploy_decision_fn(
                _opponent_deploy_model, player=_opp_side,
                opponent_type_idx=opponent_type_idx,
                side_idx=(0 if _opp_side == "A" else 1),
                # Only record opponent-side records for the mirror case (both
                # sides train); for tactical opponents the records are not
                # used in the main model's loss.
                record_into=deploy_records_opp if is_mirror else None,
            )

    # deploy_armies puts its first positional arg on physical side A and
    # second on physical side B — so if the main model is on side B we
    # pass (units_b, units_a) to land the opponent on A and main on B.
    if model_side == "A":
        _df_a, _df_b = fn_main, fn_opp
        deploy_armies(units_a, units_b, board,
                      decision_fn_a=_df_a, decision_fn_b=_df_b)
    else:
        _df_a, _df_b = fn_opp, fn_main
        deploy_armies(units_b, units_a, board,
                      decision_fn_a=_df_a, decision_fn_b=_df_b)

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

    # Heuristic opponent efficiency tracking (for baseline comparison)
    _h_shoot_eff_total = 0.0
    _h_charge_eff_total = 0.0
    _h_shoot_count = 0
    _h_charge_count = 0

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
        # and mirror decide per-activation, not per-round). units_b is the
        # opponent by convention, so roles are always reassigned on it.
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
                    units_a, units_b, round_num, board, model_side,
                    friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
                    enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
                    total_friendly_points=pts_a, total_enemy_points=pts_b,
                )
                state_vec_np = state_vec.numpy()

                # Compute model-space positions for inference
                a_friendly_pos = _get_model_space_positions(units_a, model_side)
                a_enemy_pos = _get_model_space_positions(units_b, model_side)
                a_adv_dists, a_rush_dists = _get_movement_budgets(units_a)
                a_max_wr = _get_max_weapon_ranges(units_a)

                # Pre-compute dest candidates (rush budget) for all alive A units
                # Features are computed lazily in sampling after unit selection.
                _ga_enemy_occ = _collect_enemy_positions(units_b)
                _ga_enemy_alive_np = np.array(enemy_alive_mask_list, dtype=np.bool_)
                _ga_cands_dict = {}
                _ga_mask_dict = {}
                _ga_ar_dict = {}
                _ga_dest_vis_dict = {}
                _ga_static_vis_dict = {}
                _ga_dest_dmg_dict = {}
                _ga_static_dmg_dict = {}
                _ga_e_dmg_table = getattr(board, 'expected_damage_table', None)
                for _gui in range(min(len(units_a), MAX_UNITS_PER_SIDE)):
                    if alive_mask_list[_gui]:
                        _gc, _gm, _gar = compute_destination_candidates(
                            units_a[_gui], board, _ga_enemy_occ, model_side,
                        )
                        _ga_cands_dict[_gui] = _gc
                        _ga_mask_dict[_gui] = _gm
                        _ga_ar_dict[_gui] = _gar
                        _gdv, _gsv = compute_unit_visibility_arrays(
                            units_a[_gui], _gc, _gm, units_b, board,
                        )
                        _ga_dest_vis_dict[_gui] = _gdv
                        _ga_static_vis_dict[_gui] = _gsv
                        _gdd, _gsd = compute_unit_expected_damage_arrays(
                            units_a[_gui], _gc, _gm, units_b, board, _ga_e_dmg_table,
                        )
                        _ga_dest_dmg_dict[_gui] = _gdd
                        _ga_static_dmg_dict[_gui] = _gsd
                _ga_enemy_cache = build_dest_enemy_cache(units_b, _ga_enemy_alive_np, model_side)

                # >>> YIELD for batched inference (or planning, decided by coordinator) <<<
                _req = _TacticalInferenceRequest(
                    state_vec, alive_mask, enemy_alive_mask, "main",
                    a_friendly_pos, a_enemy_pos, a_adv_dists, a_rush_dists,
                    a_max_wr, opponent_type_idx=opponent_type_idx,
                    player=model_side,
                )
                _req.dest_candidates_per_unit = _ga_cands_dict
                _req.dest_mask_per_unit = _ga_mask_dict
                _req.dest_advance_reachable_per_unit = _ga_ar_dict
                _req.dest_features_per_unit = None  # computed lazily
                _req.dest_visibility_per_unit = _ga_dest_vis_dict
                _req.static_visibility_per_unit = _ga_static_vis_dict
                _req.dest_expected_damage_per_unit = _ga_dest_dmg_dict
                _req.static_expected_damage_per_unit = _ga_static_dmg_dict
                _req.dest_lazy_units = units_a
                _req.dest_lazy_enemy_units = units_b
                _req.dest_lazy_enemy_alive = _ga_enemy_alive_np
                _req.dest_lazy_fr_matchups = fr_a
                _req.dest_lazy_er_matchups = fr_b
                _req.dest_lazy_melee_matchups = fm_b
                _req.dest_lazy_player = model_side
                _req.dest_lazy_enemy_cache = _ga_enemy_cache
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
                    # A-side requests always pass "A" — by collection convention
                    # (lines 241-243) units_a is the main model's army regardless
                    # of physical side, and the planner expects friendly to map
                    # to its internal "A". Setting this to model_side breaks
                    # non-mirror games where model_side="B" because the dispatcher
                    # then swaps friendly/enemy and the planner runs on the wrong
                    # unit list.
                    _req.planning_player = "A"

                _inf_result = yield _req

                sel_idx = _inf_result.unit_idx
                move_type_a = _inf_result.move_type
                charge_tgt_a = _inf_result.charge_target_idx
                shoot_tgt_a = _inf_result.shoot_target_idx
                _a_tac_target_ranking = _inf_result.target_ranking
                pmr_a = _inf_result.post_move_rel
                old_lp = _inf_result.old_log_prob
                value_est = _inf_result.value
                shoot_mask_a = _inf_result.shoot_mask
                cover_aware_dmg_a = getattr(_inf_result, 'cover_aware_dmg', None)
                dest_cands_a = _inf_result.dest_candidates
                dest_sel_a = _inf_result.dest_selected_idx
                dest_ar_a = _inf_result.dest_advance_reachable

                # Build recomputation data for PPO replay (avoids storing ~60KB features)
                _sel_unit = units_a[sel_idx]
                _sel_cx, _sel_cy = _sel_unit.centre()
                # Store model-space centre for replay — flip for side B.
                if model_side == "A":
                    _sel_mcx, _sel_mcy = _sel_cx, _sel_cy
                else:
                    _sel_mcx, _sel_mcy = _flip_x(_sel_cx), _flip_y(_sel_cy)
                _a_recomp = {
                    'player': model_side,
                    'unit_cx': _sel_mcx, 'unit_cy': _sel_mcy,
                    'unit_alive_frac': _sel_unit.models_alive / max(_sel_unit.unit.models, 1),
                    'move_budget': a_rush_dists[sel_idx],
                    'enemy_cache': _ga_enemy_cache,
                    'enemy_alive_mask': _ga_enemy_alive_np,
                    'fr_matchups': fr_a, 'er_matchups': fr_b,
                    'melee_matchups': fm_b,
                }

                _a_state_vec_post_np = _maybe_build_post_move_state_vec(
                    units_friendly=units_a, units_enemy=units_b,
                    round_num=round_num, board=board, model_side=model_side,
                    sel_idx=sel_idx, move_type=move_type_a,
                    dest_candidates=dest_cands_a,
                    dest_selected_idx=dest_sel_a,
                    dest_advance_reachable=dest_ar_a,
                    fr_matchups=fr_a, fm_matchups=fm_a,
                    er_matchups=fr_b, em_matchups=fm_b,
                    pts_friendly=pts_a, pts_enemy=pts_b,
                )

                step = TacticalActivationRecord(
                    state_vec=state_vec_np,
                    alive_mask=alive_mask_list,
                    enemy_alive_mask=enemy_alive_mask_list,
                    unit_idx=sel_idx,
                    move_type=move_type_a,
                    dest_candidates=dest_cands_a,
                    dest_advance_reachable=dest_ar_a,
                    dest_selected_idx=dest_sel_a,
                    dest_recomp=_a_recomp,
                    charge_target_idx=charge_tgt_a,
                    shoot_target_idx=shoot_tgt_a,
                    shoot_mask=shoot_mask_a,
                    cover_aware_dmg=cover_aware_dmg_a,
                    post_move_rel=pmr_a,
                    state_vec_post=_a_state_vec_post_np,
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
                    planning_dest_values=_inf_result.planning_dest_values,
                    planning_dest_indices=_inf_result.planning_dest_indices,
                )
                round_step_indices.append(len(trajectory))
                trajectory.append(step)
                _a_act_total += 1
                _traj_a_counts.append((_a_act_total, _b_act_total))

                active = units_a[sel_idx]
                active.activated = True

                # Compute destination in game-space from selected candidate hex
                _a_dest = None
                _a_is_ar = True
                if move_type_a == MOVE_MOVE and dest_sel_a >= 0 and len(dest_cands_a) > 0:
                    _dest_hex = dest_cands_a[dest_sel_a]
                    _a_dest = (float(_dest_hex[0]), float(_dest_hex[1]))  # game-space
                    if dest_ar_a is not None and dest_sel_a < len(dest_ar_a):
                        _a_is_ar = dest_ar_a[dest_sel_a]

                _a_tac_action, _a_tac_goal, _a_tac_charge_target, _a_reason = execute_decoded_decision(
                    active, units_b, move_type_a, _a_dest, charge_tgt_a, shoot_tgt_a,
                    is_advance_reachable=_a_is_ar,
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
                            units_b, units_a, round_num, board, _opp_side,
                            friendly_ranged_matchups=fr_b, friendly_melee_matchups=fm_b,
                            enemy_ranged_matchups=fr_a, enemy_melee_matchups=fm_a,
                            total_friendly_points=pts_b, total_enemy_points=pts_a,
                        )
                        b_state_vec_np = b_state_vec.numpy()

                        b_friendly_pos = _get_model_space_positions(units_b, _opp_side)
                        b_enemy_pos = _get_model_space_positions(units_a, _opp_side)
                        b_adv_dists, b_rush_dists = _get_movement_budgets(units_b)
                        b_max_wr = _get_max_weapon_ranges(units_b)

                        # Pre-compute dest candidates for all alive B units (features lazy)
                        _gb_enemy_occ = _collect_enemy_positions(units_a)
                        _gb_enemy_alive_np = np.array(b_enemy_alive_list, dtype=np.bool_)
                        _gb_cands_dict = {}
                        _gb_mask_dict = {}
                        _gb_ar_dict = {}
                        _gb_dest_vis_dict = {}
                        _gb_static_vis_dict = {}
                        _gb_dest_dmg_dict = {}
                        _gb_static_dmg_dict = {}
                        _gb_e_dmg_table = getattr(board, 'expected_damage_table', None)
                        for _gbui in range(min(len(units_b), MAX_UNITS_PER_SIDE)):
                            if b_alive_list[_gbui]:
                                _gbc, _gbm, _gbar = compute_destination_candidates(
                                    units_b[_gbui], board, _gb_enemy_occ, _opp_side,
                                )
                                _gb_cands_dict[_gbui] = _gbc
                                _gb_mask_dict[_gbui] = _gbm
                                _gb_ar_dict[_gbui] = _gbar
                                _gbdv, _gbsv = compute_unit_visibility_arrays(
                                    units_b[_gbui], _gbc, _gbm, units_a, board,
                                )
                                _gb_dest_vis_dict[_gbui] = _gbdv
                                _gb_static_vis_dict[_gbui] = _gbsv
                                _gbdd, _gbsd = compute_unit_expected_damage_arrays(
                                    units_b[_gbui], _gbc, _gbm, units_a, board, _gb_e_dmg_table,
                                )
                                _gb_dest_dmg_dict[_gbui] = _gbdd
                                _gb_static_dmg_dict[_gbui] = _gbsd
                        _gb_enemy_cache = build_dest_enemy_cache(units_a, _gb_enemy_alive_np, _opp_side)

                        # >>> YIELD for batched main-model inference (mirror B) <<<
                        _gb_req = _TacticalInferenceRequest(
                            b_state_vec, b_alive_mask, b_enemy_alive_mask, "main",
                            b_friendly_pos, b_enemy_pos, b_adv_dists, b_rush_dists,
                            b_max_wr, opponent_type_idx=opponent_type_idx,
                            player=_opp_side,
                        )
                        _gb_req.dest_candidates_per_unit = _gb_cands_dict
                        _gb_req.dest_mask_per_unit = _gb_mask_dict
                        _gb_req.dest_advance_reachable_per_unit = _gb_ar_dict
                        _gb_req.dest_features_per_unit = None
                        _gb_req.dest_visibility_per_unit = _gb_dest_vis_dict
                        _gb_req.static_visibility_per_unit = _gb_static_vis_dict
                        _gb_req.dest_expected_damage_per_unit = _gb_dest_dmg_dict
                        _gb_req.static_expected_damage_per_unit = _gb_static_dmg_dict
                        _gb_req.dest_lazy_units = units_b
                        _gb_req.dest_lazy_enemy_units = units_a
                        _gb_req.dest_lazy_enemy_alive = _gb_enemy_alive_np
                        _gb_req.dest_lazy_fr_matchups = fr_b
                        _gb_req.dest_lazy_er_matchups = fr_a
                        _gb_req.dest_lazy_melee_matchups = fm_a
                        _gb_req.dest_lazy_player = _opp_side
                        _gb_req.dest_lazy_enemy_cache = _gb_enemy_cache
                        if planning_enabled:
                            _gb_req.planning_units_a = units_a
                            _gb_req.planning_units_b = units_b
                            _gb_req.planning_board = board
                            _gb_req.planning_round_num = round_num
                            _gb_req.planning_current_is_a = current_is_a
                            _gb_req.planning_fr_a = fr_a
                            _gb_req.planning_fm_a = fm_a
                            _gb_req.planning_fr_b = fr_b
                            _gb_req.planning_fm_b = fm_b
                            _gb_req.planning_pts_a = pts_a
                            _gb_req.planning_pts_b = pts_b
                            _gb_req.planning_opponent_type_idx = opponent_type_idx
                            _gb_req.planning_player = _opp_side
                        _b_inf = yield _gb_req

                        sel_b = _b_inf.unit_idx
                        if (sel_b < len(units_b)
                                and units_b[sel_b].models_alive > 0
                                and not units_b[sel_b].activated):
                            active = units_b[sel_b]
                            _b_target_ranking = _b_inf.target_ranking

                            _selb = units_b[sel_b]
                            _selb_cx, _selb_cy = _selb.centre()
                            _selb_mcx, _selb_mcy = _flip_x(_selb_cx), _flip_y(_selb_cy)
                            _b_recomp_m = {
                                'player': _opp_side,
                                'unit_cx': _selb_mcx, 'unit_cy': _selb_mcy,
                                'unit_alive_frac': _selb.models_alive / max(_selb.unit.models, 1),
                                'move_budget': b_rush_dists[sel_b],
                                'enemy_cache': _gb_enemy_cache,
                                'enemy_alive_mask': _gb_enemy_alive_np,
                                'fr_matchups': fr_b, 'er_matchups': fr_a,
                                'melee_matchups': fm_a,
                            }
                            _b_state_vec_post_np = _maybe_build_post_move_state_vec(
                                units_friendly=units_b, units_enemy=units_a,
                                round_num=round_num, board=board, model_side=_opp_side,
                                sel_idx=sel_b, move_type=_b_inf.move_type,
                                dest_candidates=_b_inf.dest_candidates,
                                dest_selected_idx=_b_inf.dest_selected_idx,
                                dest_advance_reachable=_b_inf.dest_advance_reachable,
                                fr_matchups=fr_b, fm_matchups=fm_b,
                                er_matchups=fr_a, em_matchups=fm_a,
                                pts_friendly=pts_b, pts_enemy=pts_a,
                            )
                            step_b = TacticalActivationRecord(
                                state_vec=b_state_vec_np,
                                alive_mask=b_alive_list,
                                enemy_alive_mask=b_enemy_alive_list,
                                unit_idx=sel_b,
                                move_type=_b_inf.move_type,
                                dest_candidates=_b_inf.dest_candidates,
                                dest_advance_reachable=_b_inf.dest_advance_reachable,
                                dest_selected_idx=_b_inf.dest_selected_idx,
                                dest_recomp=_b_recomp_m,
                                charge_target_idx=_b_inf.charge_target_idx,
                                shoot_target_idx=_b_inf.shoot_target_idx,
                                shoot_mask=_b_inf.shoot_mask,
                                cover_aware_dmg=getattr(_b_inf, 'cover_aware_dmg', None),
                                post_move_rel=_b_inf.post_move_rel,
                                state_vec_post=_b_state_vec_post_np,
                                old_log_prob=_b_inf.old_log_prob,
                                old_value=_b_inf.value,
                                opponent_type_idx=opponent_type_idx,
                            )
                            round_step_indices_b.append(len(trajectory_b))
                            trajectory_b.append(step_b)
                            _b_act_total += 1
                            _traj_b_counts.append((_b_act_total, _a_act_total))

                            _b_dest = None
                            _b_is_ar = True
                            if (_b_inf.move_type == MOVE_MOVE
                                    and _b_inf.dest_selected_idx >= 0
                                    and len(_b_inf.dest_candidates) > 0):
                                _bdhex = _b_inf.dest_candidates[_b_inf.dest_selected_idx]
                                # Candidates are in game-space
                                _b_dest = (float(_bdhex[0]), float(_bdhex[1]))
                                if (_b_inf.dest_advance_reachable is not None
                                        and _b_inf.dest_selected_idx < len(_b_inf.dest_advance_reachable)):
                                    _b_is_ar = _b_inf.dest_advance_reachable[_b_inf.dest_selected_idx]

                            _b_action, _b_goal, _b_charge_target, _ = execute_decoded_decision(
                                active, units_a, _b_inf.move_type, _b_dest,
                                _b_inf.charge_target_idx, _b_inf.shoot_target_idx,
                                is_advance_reachable=_b_is_ar,
                            )
                            active.activated = True
                        else:
                            active = None

                    _opp_tac_decision = active is not None

                elif opponent_model is not None or has_tactical_opponent:
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
                            units_b, units_a, round_num, board, _opp_side,
                            friendly_ranged_matchups=fr_b, friendly_melee_matchups=fm_b,
                            enemy_ranged_matchups=fr_a, enemy_melee_matchups=fm_a,
                            total_friendly_points=pts_b, total_enemy_points=pts_a,
                        )

                        b_friendly_pos_opp = _get_model_space_positions(units_b, _opp_side)
                        b_enemy_pos_opp = _get_model_space_positions(units_a, _opp_side)
                        b_adv_dists_opp, b_rush_dists_opp = _get_movement_budgets(units_b)
                        b_max_wr_opp = _get_max_weapon_ranges(units_b)

                        # Pre-compute dest candidates for opponent B units (features lazy)
                        _gopp_enemy_occ = _collect_enemy_positions(units_a)
                        _gopp_enemy_alive_np = np.array(
                            [(i < len(units_a) and units_a[i].models_alive > 0)
                             for i in range(MAX_UNITS_PER_SIDE)],
                            dtype=np.bool_,
                        )
                        _gopp_cands_dict = {}
                        _gopp_mask_dict = {}
                        _gopp_ar_dict = {}
                        _gopp_dest_vis_dict = {}
                        _gopp_static_vis_dict = {}
                        _gopp_dest_dmg_dict = {}
                        _gopp_static_dmg_dict = {}
                        _gopp_e_dmg_table = getattr(board, 'expected_damage_table', None)
                        for _goppi in range(min(len(units_b), MAX_UNITS_PER_SIDE)):
                            if b_alive_list[_goppi]:
                                _goppc, _goppm, _goppar = compute_destination_candidates(
                                    units_b[_goppi], board, _gopp_enemy_occ, _opp_side,
                                )
                                _gopp_cands_dict[_goppi] = _goppc
                                _gopp_mask_dict[_goppi] = _goppm
                                _gopp_ar_dict[_goppi] = _goppar
                                _goppdv, _goppsv = compute_unit_visibility_arrays(
                                    units_b[_goppi], _goppc, _goppm, units_a, board,
                                )
                                _gopp_dest_vis_dict[_goppi] = _goppdv
                                _gopp_static_vis_dict[_goppi] = _goppsv
                                _goppdd, _goppsd = compute_unit_expected_damage_arrays(
                                    units_b[_goppi], _goppc, _goppm, units_a, board, _gopp_e_dmg_table,
                                )
                                _gopp_dest_dmg_dict[_goppi] = _goppdd
                                _gopp_static_dmg_dict[_goppi] = _goppsd
                        _gopp_enemy_cache = build_dest_enemy_cache(units_a, _gopp_enemy_alive_np, _opp_side)

                        # >>> YIELD for batched opponent-model inference <<<
                        _gopp_req = _TacticalInferenceRequest(
                            b_state_vec, b_alive_mask, b_enemy_alive_mask, "opponent",
                            b_friendly_pos_opp, b_enemy_pos_opp, b_adv_dists_opp, b_rush_dists_opp,
                            b_max_wr_opp,
                            player=_opp_side,
                        )
                        _gopp_req.dest_candidates_per_unit = _gopp_cands_dict
                        _gopp_req.dest_mask_per_unit = _gopp_mask_dict
                        _gopp_req.dest_advance_reachable_per_unit = _gopp_ar_dict
                        _gopp_req.dest_features_per_unit = None
                        _gopp_req.dest_visibility_per_unit = _gopp_dest_vis_dict
                        _gopp_req.static_visibility_per_unit = _gopp_static_vis_dict
                        _gopp_req.dest_expected_damage_per_unit = _gopp_dest_dmg_dict
                        _gopp_req.static_expected_damage_per_unit = _gopp_static_dmg_dict
                        _gopp_req.dest_lazy_units = units_b
                        _gopp_req.dest_lazy_enemy_units = units_a
                        _gopp_req.dest_lazy_enemy_alive = _gopp_enemy_alive_np
                        _gopp_req.dest_lazy_fr_matchups = fr_b
                        _gopp_req.dest_lazy_er_matchups = fr_a
                        _gopp_req.dest_lazy_melee_matchups = fm_a
                        _gopp_req.dest_lazy_player = _opp_side
                        _gopp_req.dest_lazy_enemy_cache = _gopp_enemy_cache
                        _b_inf = yield _gopp_req

                        sel_b = _b_inf.unit_idx
                        if (sel_b < len(units_b)
                                and units_b[sel_b].models_alive > 0
                                and not units_b[sel_b].activated):
                            active = units_b[sel_b]
                            _b_target_ranking = _b_inf.target_ranking
                            _b_dest_opp = None
                            _b_is_ar_opp = True
                            if (_b_inf.move_type == MOVE_MOVE
                                    and _b_inf.dest_selected_idx >= 0
                                    and len(_b_inf.dest_candidates) > 0):
                                _bdhex_opp = _b_inf.dest_candidates[_b_inf.dest_selected_idx]
                                # Candidates are in game-space
                                _b_dest_opp = (float(_bdhex_opp[0]), float(_bdhex_opp[1]))
                                if (_b_inf.dest_advance_reachable is not None
                                        and _b_inf.dest_selected_idx < len(_b_inf.dest_advance_reachable)):
                                    _b_is_ar_opp = _b_inf.dest_advance_reachable[_b_inf.dest_selected_idx]
                            _b_action, _b_goal, _b_charge_target, _ = execute_decoded_decision(
                                active, units_a, _b_inf.move_type, _b_dest_opp,
                                _b_inf.charge_target_idx, _b_inf.shoot_target_idx,
                                is_advance_reachable=_b_is_ar_opp,
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

            _shoot_target = None  # track ranged shooting for efficiency reward

            # --- Execute the activation ---
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

            # Shaken units must hold and recover — no shooting, no charging
            if active.shaken:
                active.shaken = False
            elif action == "charge" and charge_target is not None:
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
                    if current_is_a or _opp_tac_decision:
                        target = pick_target_from_ranking(active, opp_units, _active_target_ranking)
                    else:
                        target = pick_target(active, opp_units, target_multipliers=my_mults)
                    if target is not None:
                        _shoot_target = target
                        resolve_shooting(active, target)
                        check_morale(target)
                        _sync_dead_models(target, board)

            elif action == "hold":
                if current_is_a or _opp_tac_decision:
                    target = pick_target_from_ranking(active, opp_units, _active_target_ranking)
                else:
                    target = pick_target(active, opp_units, target_multipliers=my_mults)
                if target is not None:
                    _shoot_target = target
                    resolve_shooting(active, target)
                    check_morale(target)
                    _sync_dead_models(target, board)

            # Per-activation shooting efficiency reward
            if _shoot_target is not None:
                _st_idx = next((i for i, u in enumerate(opp_units) if u is _shoot_target), -1)
                if _st_idx >= 0:
                    if current_is_a and round_step_indices:
                        _sr = compute_shooting_efficiency_reward(
                            active, _shoot_target, sel_idx, _st_idx,
                            fr_a, pts_a, round_num,
                        )
                        if _sr != 0.0:
                            trajectory[round_step_indices[-1]].reward += shaping_scale * _sr
                            trajectory[round_step_indices[-1]].shooting_efficiency_reward += _sr
                    elif is_mirror and round_step_indices_b:
                        _sr = compute_shooting_efficiency_reward(
                            active, _shoot_target, sel_b, _st_idx,
                            fr_b, pts_b, round_num,
                        )
                        if _sr != 0.0:
                            trajectory_b[round_step_indices_b[-1]].reward += shaping_scale * _sr
                            trajectory_b[round_step_indices_b[-1]].shooting_efficiency_reward += _sr
                    elif not current_is_a and not _opp_tac_decision:
                        _h_atk_idx = next((i for i, u in enumerate(units_b) if u is active), -1)
                        if _h_atk_idx >= 0 and _st_idx < MAX_UNITS_PER_SIDE:
                            _h_shoot_eff_total += compute_shooting_efficiency_reward(
                                active, _shoot_target, _h_atk_idx, _st_idx,
                                fr_b, pts_b, round_num,
                            )
                            _h_shoot_count += 1

            # Per-activation charge efficiency reward
            if action == "charge" and charge_target is not None:
                _ct_idx = next((i for i, u in enumerate(opp_units) if u is charge_target), -1)
                if _ct_idx >= 0:
                    if current_is_a and round_step_indices:
                        _cr = compute_charge_efficiency_reward(
                            active, charge_target, sel_idx, _ct_idx,
                            fm_a, pts_a, round_num,
                        )
                        if _cr != 0.0:
                            trajectory[round_step_indices[-1]].reward += shaping_scale * _cr
                            trajectory[round_step_indices[-1]].charge_efficiency_reward += _cr
                    elif is_mirror and round_step_indices_b:
                        _cr = compute_charge_efficiency_reward(
                            active, charge_target, sel_b, _ct_idx,
                            fm_b, pts_b, round_num,
                        )
                        if _cr != 0.0:
                            trajectory_b[round_step_indices_b[-1]].reward += shaping_scale * _cr
                            trajectory_b[round_step_indices_b[-1]].charge_efficiency_reward += _cr
                    elif not current_is_a and not _opp_tac_decision:
                        _h_atk_idx = next((i for i, u in enumerate(units_b) if u is active), -1)
                        if _h_atk_idx >= 0 and _ct_idx < MAX_UNITS_PER_SIDE:
                            _h_charge_eff_total += compute_charge_efficiency_reward(
                                active, charge_target, _h_atk_idx, _ct_idx,
                                fm_b, pts_b, round_num,
                            )
                            _h_charge_count += 1

            # Per-activation objective capture reward
            if _pre_move_friendly_on_objs_g is not None and active is not None and active.models_alive > 0:
                if current_is_a:
                    _cap_reward = compute_objective_capture_reward(
                        active, units_a, board, model_side, round_num, shaping_scale,
                        _pre_move_friendly_on_objs_g,
                    )
                    if _cap_reward != 0.0 and round_step_indices:
                        trajectory[round_step_indices[-1]].reward += _cap_reward
                elif is_mirror and round_step_indices_b:
                    _cap_reward = compute_objective_capture_reward(
                        active, units_b, board, _opp_side, round_num, shaping_scale,
                        _pre_move_friendly_on_objs_g,
                    )
                    if _cap_reward != 0.0:
                        trajectory_b[round_step_indices_b[-1]].reward += _cap_reward

            current_is_a = not current_is_a

        # End of round
        board.update_objectives(units_a, units_b)

        reward, prev_a_kill_pts, prev_b_kill_pts = compute_round_reward(
            units_a, units_b, board, model_side, pts_a,
            prev_a_kill_pts, prev_b_kill_pts,
            shaping_scale=shaping_scale,
            round_num=round_num,
        )
        if round_step_indices:
            trajectory[round_step_indices[-1]].reward += reward

        if is_mirror and round_step_indices_b:
            reward_b, prev_b_fkp, prev_b_ekp = compute_round_reward(
                units_b, units_a, board, _opp_side, pts_b,
                prev_b_fkp, prev_b_ekp,
                shaping_scale=shaping_scale,
                round_num=round_num,
            )
            trajectory_b[round_step_indices_b[-1]].reward += reward_b

        # Aux targets: short-horizon (end-of-current-round) survival + obj control
        snap_a = _make_round_snapshot(units_a, units_b, board, model_side)
        for si in round_step_indices:
            trajectory[si].friendly_survival_target_short = snap_a.friendly_survival
            trajectory[si].enemy_survival_target_short = snap_a.enemy_survival
            trajectory[si].obj_control_target_short = snap_a.obj_control
        if is_mirror:
            snap_b = _make_round_snapshot(units_b, units_a, board, _opp_side)
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
        trajectory[-1].reward += terminal_reward(result, model_side, a_objs, b_objs)

    if is_mirror and trajectory_b:
        trajectory_b[-1].reward += terminal_reward(result, _opp_side, a_objs, b_objs)

    # Aux targets: long-horizon (end-of-game) survival + obj control (backfill to all steps)
    final_snap_a = _make_round_snapshot(units_a, units_b, board, model_side)
    for step in trajectory:
        step.friendly_survival_target = final_snap_a.friendly_survival
        step.enemy_survival_target = final_snap_a.enemy_survival
        step.obj_control_target = final_snap_a.obj_control
    if is_mirror and trajectory_b:
        final_snap_b = _make_round_snapshot(units_b, units_a, board, _opp_side)
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

    heuristic_eff = {
        'shoot': _h_shoot_eff_total,
        'charge': _h_charge_eff_total,
        'shoot_n': _h_shoot_count,
        'charge_n': _h_charge_count,
    } if opponent_type == "heuristic" else None
    # Tagged deploy records: (records_main, records_opp_for_mirror_only).
    # When deployment training is disabled (no main_model_for_deploy was
    # passed), both lists are empty so downstream consumers can no-op.
    deploy_payload = (deploy_records_main, deploy_records_opp)
    return (trajectory, result, opponent_type, trajectory_b,
            heuristic_eff, deploy_payload)


def _apply_v_old_shaping(
    finished: dict,
    game_model_sides: list[str],
    v_old_model: TacticalModel,
    shaping_scale: float,
) -> None:
    """Add Φ-delta shaping to every collected trajectory in place.

    Stacks all step state_vecs across trajectories into a single batched
    forward through V_old, then writes per-step shaping rewards back to
    each TacticalActivationRecord. Terminal Φ is taken as 0.
    """
    rows: list[np.ndarray] = []
    opp_indices: list[int] = []
    side_indices: list[int] = []
    bounds: list[tuple[int, int, list]] = []  # (start, end, traj_records)

    for gid, val in finished.items():
        traj, _result, _opp_type, traj_b, _h_eff = val
        m_side = game_model_sides[gid]
        if traj:
            start = len(rows)
            for r in traj:
                rows.append(r.state_vec)
                opp_indices.append(int(r.opponent_type_idx))
                side_indices.append(0 if m_side == "A" else 1)
            bounds.append((start, len(rows), traj))
        if traj_b:
            opp_m_side = "B" if m_side == "A" else "A"
            start = len(rows)
            for r in traj_b:
                rows.append(r.state_vec)
                opp_indices.append(int(r.opponent_type_idx))
                side_indices.append(0 if opp_m_side == "A" else 1)
            bounds.append((start, len(rows), traj_b))

    if not rows:
        return

    state_vecs = torch.from_numpy(np.stack(rows)).float()
    opp_t = torch.tensor(opp_indices, dtype=torch.long)
    side_t = torch.tensor(side_indices, dtype=torch.long)

    with torch.no_grad():
        v_olds = v_old_model.value_only(
            state_vecs, opponent_type=opp_t, side=side_t
        ).cpu().numpy()

    for start, end, traj in bounds:
        T = end - start
        for i in range(T):
            v_curr = float(v_olds[start + i])
            v_next = float(v_olds[start + i + 1]) if i + 1 < T else 0.0
            traj[i].reward += shaping_scale * (v_next - v_curr)


def _run_games_batched_tactical(
    main_model: TacticalModel,
    game_specs: list,
    opp_models: dict,
    shaping_scale: float = 1.0,
    planning_rate: float = 0.0,
    planning_params: dict | None = None,
    randomize_sides: bool = True,
    v_old_model: TacticalModel | None = None,
    map_data=None,
    train_deployment: bool = False,
) -> list[tuple]:
    """Run multiple tactical training games with batched inference.

    Creates generator coroutines for each game and advances them in lockstep,
    batching main-model and opponent-model forward passes separately.

    If ``randomize_sides`` is True, each non-mirror game coin-flips whether
    the main learning model plays physical side A or side B. Mirror
    self-play games are not randomized (both sides are the main model
    already). Credit assignment is handled inside the generator via the
    ``model_side`` argument: the main trajectory always carries the main
    model's activations and its terminal reward is computed from the main
    model's perspective regardless of physical side.

    Returns list of (trajectory, result, opponent_type, army_type) per game.
    """
    from board import OBJECTIVES as BOARD_OBJECTIVES

    # When V_old shaping is active, the heuristic shapers must not fire — the
    # generator gets shaping_scale=0 and we apply Φ-deltas after collection.
    use_v_old_shaping = v_old_model is not None
    gen_shaping_scale = 0.0 if use_v_old_shaping else shaping_scale

    # Create generators and track opponent models for tactical opponents
    generators: list = []
    game_army_types: list[str] = []
    game_model_sides: list[str] = []
    game_opp_tactical_models: dict[int, nn.Module] = {}

    for i, spec in enumerate(game_specs):
        # Specs from loop.py are 9-tuples (the last two carry per-entry hero
        # attach data so the generator can redo the merge). Legacy callers
        # (probe/profile scripts) still pass 7-tuples — synthesise empty
        # attach data so heroes simply pass through as standalone units.
        if len(spec) == 9:
            (res_a, res_b, sa_data, sb_data, opp_type, opp_sd_idx, army_type,
             attach_a, attach_b) = spec
        else:
            (res_a, res_b, sa_data, sb_data, opp_type, opp_sd_idx, army_type) = spec
            attach_a = [(-1, False)] * len(res_a)
            attach_b = [(-1, False)] * len(res_b)
        opp_model = opp_models.get(opp_sd_idx)

        if opp_model is not None:
            game_opp_tactical_models[i] = opp_model

        is_mirror_game = (opp_type == "selfplay_mirror")
        if randomize_sides and not is_mirror_game and random.random() < 0.5:
            _model_side = "B"
        else:
            _model_side = "A"

        gen = _episode_tactical_generator(
            opp_model,
            res_a, res_b, sa_data, sb_data, opp_type, BOARD_OBJECTIVES,
            shaping_scale=gen_shaping_scale,
            army_type=army_type,
            # Planning is restricted to current-vs-current mirror matches so
            # both sides see (and learn from) planned trajectories under
            # symmetric conditions. Non-mirror games (vs heuristic, HoF,
            # checkpoints, random) train without planning so the model gets
            # broader state-distribution coverage from fair-skill play.
            planning_enabled=(planning_rate > 0 and is_mirror_game),
            has_tactical_opponent=(opp_model is not None),
            model_side=_model_side,
            map_data=map_data,
            main_model_for_deploy=(main_model if train_deployment else None),
            attach_a=attach_a,
            attach_b=attach_b,
        )
        generators.append(gen)
        game_army_types.append(army_type)
        game_model_sides.append(_model_side)

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

        # Synchronized planning decision: roll once for all main-model requests
        # (covers A-side and mirror B-side; opponent-checkpoint requests use
        # model_key="opponent" and never enter this loop).
        use_planning_this_round = (
            planning_rate > 0
            and main_reqs
            and random.random() < planning_rate
        )

        if use_planning_this_round:
            # Planning round: run plan_training_activation for each main request
            from ml_planning import plan_training_activation
            for gid, req in zip(main_gids, main_reqs):
                if req.planning_units_a is not None:
                    if req.planning_player == "A":
                        _plan_friendly = req.planning_units_a
                        _plan_enemy = req.planning_units_b
                    else:
                        _plan_friendly = req.planning_units_b
                        _plan_enemy = req.planning_units_a
                    _plan_out = plan_training_activation(
                        main_model, req.state_vec, req.alive_mask,
                        req.enemy_alive_mask,
                        _plan_friendly, _plan_enemy,
                        req.planning_round_num, req.planning_board,
                        req.planning_player,
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
                    if _plan_out is None:
                        # Planning bailed (no valid live units after safety
                        # clamp); fall through to the unhandled path below so
                        # this gid is picked up by the normal inference batch.
                        continue
                    (uid, mt, plan_dcol, plan_drow, plan_dcidx,
                     ct, st, ranking,
                     pmr, olp, val, sm,
                     wp, pi, pvd, puv, pui,
                     pmv, pmi, pcv, pci, psv, psi,
                     pdv, pdi, plan_dcands, plan_dar,
                    ) = _plan_out
                    # Planning returns dest candidates for the chosen unit
                    # (MOVE_MOVE only); pass them into the replay record so the
                    # dest head gets replay log-probs and distillation targets.
                    _dcands = plan_dcands if plan_dcands is not None else []
                    _dar = plan_dar if plan_dar is not None else None
                    _dmask = [True] * len(_dcands)
                    all_results[gid] = _TacticalSamplingResult(
                        unit_idx=uid, move_type=mt,
                        dest_candidates=_dcands, dest_mask=_dmask,
                        dest_features=[],
                        dest_selected_idx=plan_dcidx,
                        dest_advance_reachable=_dar,
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
                        planning_dest_values=pdv,
                        planning_dest_indices=pdi,
                    )
                else:
                    # Main request without planning fields (e.g. planning was
                    # disabled when the generator was constructed) — fall through.
                    pass

            # Any main requests not handled get normal inference
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

    # Apply V_old potential-based shaping to all collected trajectories.
    # Simpler form r_t += scale · (V_old(s_{t+1}) − V_old(s_t)) with terminal
    # Φ ≡ 0; telescopes to a constant per-trajectory shift, redistributing
    # value information to per-step gradients. Heuristic shapers are already
    # disabled (gen_shaping_scale=0) when this branch fires.
    if use_v_old_shaping and shaping_scale > 0.0:
        _apply_v_old_shaping(finished, game_model_sides, v_old_model, shaping_scale)

    # Return results in original order, adding army_type.
    # Transform physical winner ("A"/"B") to main-perspective ("main"/"opp")
    # so record_game sees the same semantic regardless of which physical
    # side the main model was assigned to this game.
    def _to_main(result: str, main_side: str) -> str:
        if result == "draw":
            return "draw"
        return "main" if result == main_side else "opp"

    results = []
    for i in range(len(generators)):
        traj, result, opp_type, traj_b, h_eff, deploy_payload = finished[i]
        deploy_records_main, deploy_records_opp = deploy_payload
        m_side = game_model_sides[i]
        results.append((traj, _to_main(result, m_side), opp_type,
                        game_army_types[i], m_side, h_eff, deploy_records_main))
        if traj_b is not None:
            opp_m_side = "B" if m_side == "A" else "A"
            results.append((traj_b, _to_main(result, opp_m_side), "mirror_b",
                            game_army_types[i], opp_m_side, None, deploy_records_opp))
    return results
