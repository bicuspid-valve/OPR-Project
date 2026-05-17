"""Monte Carlo planning for eval-time tactical search (§3 of tactical_planning_spec).

At evaluation time, instead of taking the model's argmax decisions directly,
this module runs a Monte Carlo search: sample candidate action tuples, simulate
each forward through the game, and pick the candidate with the best expected
outcome.  Applied per-activation — every time the ML agent needs to decide what
to do with a unit, it searches before committing.

Planning is NOT used during training.
"""
from __future__ import annotations

import atexit
from collections import Counter
import math
import os
import pickle
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torch.multiprocessing as _mp

from board import Board, COLS, ROWS, OBJECTIVES
from models import UnitState
from combat import (
    resolve_shooting, check_morale,
    resolve_melee, resolve_impact, check_melee_morale,
)
from movement import (
    execute_movement,
    execute_charge_movement, execute_counter_charge,
    post_melee_separation, consolidation_move,
)
from combat import evaluate_target
from ml_features import (
    MAX_UNITS_PER_SIDE,
    TACTICAL_UNIT_FEATURES,
    MAX_DEST_CANDIDATES, DEST_FEATURE_DIM,
    encode_state_tactical,
    precompute_damage,
    extract_can_charge_mask,
    _flip_x, _flip_y,
)
from ml_integration_tactical import (
    execute_decoded_decision, pick_target_from_ranking,
    compute_post_move_rel, compute_in_range_mask, compute_in_range_mask_batched,
    compute_destination_candidates, compute_destination_features,
    _get_model_space_positions, _get_movement_budgets,
    _get_max_weapon_ranges, MOVE_TYPE_NAMES,
    project_post_move_unit_state,
)
from ml_model_tactical import (
    TacticalModel, TacticalModelOutput,
    NUM_MOVE_TYPES, MOVE_MOVE, MOVE_CHARGE,
    PHASE_PRE_SELECT, PHASE_POST_SELECT, PHASE_POST_MOVETYPE, PHASE_POST_DEST,
)
from simulation import start_round, end_round, score_game

_ML_BOTH_SIDES: frozenset[str] = frozenset({"A", "B"})

# ---------------------------------------------------------------------------
# Default planning parameters (§3.4)
# ---------------------------------------------------------------------------

DEFAULT_K_UNITS = 6               # Candidate units to evaluate
DEFAULT_C_SAMPLES_PER_UNIT = 4    # Action tuples sampled per candidate unit
DEFAULT_M_ROLLOUTS = 4            # Rollouts per candidate (dice averaging)
DEFAULT_N_LOOKAHEAD = 4           # Activations simulated forward before value eval
DEFAULT_NUM_WORKERS = 6           # 0 = auto (os.cpu_count()), 1 = sequential


# ---------------------------------------------------------------------------
# Snapshot / Restore (§3.2)
# ---------------------------------------------------------------------------

@dataclass
class _UnitSnapshot:
    """Lightweight copy of all mutable UnitState fields."""
    models_alive: int
    wounds_per_model: list[int]
    shaken: bool
    morale_checked: bool
    activated: bool
    fatigued: bool
    positions: list[tuple[int, int]]
    weapons_per_model: list[list]
    removed_positions: list[tuple[int, int]]
    ai_role: str
    combat_preference: str
    assigned_objective: int
    movement_stance: str
    owner: str
    hero_model_index: int


@dataclass
class _BoardSnapshot:
    """Lightweight copy of mutable Board fields."""
    occupancy: bytearray
    objective_control: list[str]


@dataclass
class GameSnapshot:
    """Full game state snapshot for rollback."""
    units_a: list[_UnitSnapshot]
    units_b: list[_UnitSnapshot]
    board: _BoardSnapshot


def _snap_unit(us: UnitState) -> _UnitSnapshot:
    return _UnitSnapshot(
        models_alive=us.models_alive,
        wounds_per_model=list(us.wounds_per_model),
        shaken=us.shaken,
        morale_checked=us.morale_checked,
        activated=us.activated,
        fatigued=us.fatigued,
        positions=list(us.positions),
        weapons_per_model=[list(wl) for wl in us.weapons_per_model],
        removed_positions=list(us._removed_positions),
        ai_role=us.ai_role,
        combat_preference=us.combat_preference,
        assigned_objective=us.assigned_objective,
        movement_stance=us.movement_stance,
        owner=us.owner,
        hero_model_index=us.hero_model_index,
    )


def _restore_unit(snap: _UnitSnapshot, us: UnitState) -> None:
    """Restore mutable fields of *us* from *snap* in-place."""
    us.models_alive = snap.models_alive
    us.wounds_per_model = list(snap.wounds_per_model)
    us.shaken = snap.shaken
    us.morale_checked = snap.morale_checked
    us.activated = snap.activated
    us.fatigued = snap.fatigued
    us.positions = list(snap.positions)
    us.weapons_per_model = [list(wl) for wl in snap.weapons_per_model]
    us._removed_positions = list(snap.removed_positions)
    us.ai_role = snap.ai_role
    us.combat_preference = snap.combat_preference
    us.assigned_objective = snap.assigned_objective
    us.movement_stance = snap.movement_stance
    us.owner = snap.owner
    us.hero_model_index = snap.hero_model_index


def _restore_board(snap: _BoardSnapshot, board: Board) -> None:
    """Restore mutable fields of *board* from *snap* in-place."""
    board.occupancy = bytearray(snap.occupancy)
    board.objective_control = list(snap.objective_control)


def snapshot_game_state(
    units_a: list[UnitState],
    units_b: list[UnitState],
    board: Board,
) -> GameSnapshot:
    """Create a lightweight snapshot of the full mutable game state."""
    return GameSnapshot(
        units_a=[_snap_unit(u) for u in units_a],
        units_b=[_snap_unit(u) for u in units_b],
        board=_BoardSnapshot(
            occupancy=bytearray(board.occupancy),
            objective_control=list(board.objective_control),
        ),
    )


def restore_game_state(
    snapshot: GameSnapshot,
    units_a: list[UnitState],
    units_b: list[UnitState],
    board: Board,
) -> None:
    """Restore game state from *snapshot* in-place.

    The UnitState and Board objects are mutated rather than replaced, so that
    all existing references remain valid.
    """
    for snap, us in zip(snapshot.units_a, units_a):
        _restore_unit(snap, us)
    for snap, us in zip(snapshot.units_b, units_b):
        _restore_unit(snap, us)
    _restore_board(snapshot.board, board)


# ---------------------------------------------------------------------------
# Helpers (mirroring game.py private utilities)
# ---------------------------------------------------------------------------

def _sync_dead_models(unit: UnitState, board: Board) -> None:
    for col, row in unit._removed_positions:
        board.remove(col, row)
    unit._removed_positions.clear()


def _collect_enemy_positions(units: list[UnitState]) -> set[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    for u in units:
        for pos in u.alive_positions():
            positions.add(pos)
    return positions


def _kite_range_params(
    active: UnitState, enemies: list[UnitState], reason: str,
) -> tuple[tuple[int, int] | None, float]:
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


# ---------------------------------------------------------------------------
# Execute one activation (shared by simulate_forward and plan_activation)
# ---------------------------------------------------------------------------

def _execute_activation(
    active: UnitState,
    action: str,
    goal: tuple[int, int] | None,
    charge_target: UnitState | None,
    reason: str,
    target_ranking: list[int],
    my_units: list[UnitState],
    opp_units: list[UnitState],
    board: Board,
    mode: str,
) -> bool:
    """Execute a single activation's action.  Returns True if opponent is wiped."""
    active.activated = True

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
            consolidation_move(active, board, opp_units, OBJECTIVES, mode)
        elif charge_target.models_alive > 0:
            consolidation_move(charge_target, board, my_units, OBJECTIVES, mode)

    elif action in ("advance", "rush") and goal is not None:
        budget = active.unit.advance_distance if action == "advance" else active.unit.rush_distance
        enemy_positions = _collect_enemy_positions(opp_units)
        rt, wr = _kite_range_params(active, opp_units, reason)
        execute_movement(active, goal, budget, board, enemy_positions,
                         flying=active.unit.flying,
                         strider=active.unit.strider,
                         range_target=rt, weapon_range=wr)

        if action != "rush":
            if active.shaken:
                active.shaken = False
            else:
                target = pick_target_from_ranking(active, opp_units, target_ranking, board)
                if target is not None:
                    resolve_shooting(active, target, board=board)
                    check_morale(target)
                    _sync_dead_models(target, board)

    elif action == "hold":
        if active.shaken:
            active.shaken = False
        else:
            target = pick_target_from_ranking(active, opp_units, target_ranking, board)
            if target is not None:
                resolve_shooting(active, target, board=board)
                check_morale(target)
                _sync_dead_models(target, board)

    # Check if opponent wiped
    return not any(u.models_alive > 0 for u in opp_units)


# ---------------------------------------------------------------------------
# Forward simulation (§3.5)
# ---------------------------------------------------------------------------

def _build_masks(friendly: list[UnitState], enemy: list[UnitState]):
    """Build alive_mask and enemy_alive_mask tensors."""
    alive_mask = torch.tensor(
        [(i < len(friendly) and friendly[i].models_alive > 0 and not friendly[i].activated)
         for i in range(MAX_UNITS_PER_SIDE)],
        dtype=torch.bool,
    )
    enemy_alive_mask = torch.tensor(
        [(i < len(enemy) and enemy[i].models_alive > 0)
         for i in range(MAX_UNITS_PER_SIDE)],
        dtype=torch.bool,
    )
    return alive_mask, enemy_alive_mask


@torch.no_grad()
def simulate_forward(
    units_a: list[UnitState],
    units_b: list[UnitState],
    board: Board,
    model: TacticalModel,
    n_activations: int,
    current_is_a: bool,
    round_num: int,
    mode: str,
    fr_a: list[list[list[float]]], fm_a: list[list[float]],
    fr_b: list[list[list[float]]], fm_b: list[list[float]],
    pts_a: int, pts_b: int,
) -> float | None:
    """Simulate *n_activations* forward using argmax policy.

    Returns the model's value estimate of the resulting state (from Player A's
    perspective), or the actual game outcome (+1/-1/0) if the game ends early.

    NOTE: This function is hardcoded to Player A's perspective and is only used
    by the profiling script.  The active planning path uses _rollout_generator
    which returns values from the planning player's perspective.
    """
    a_finished_first: bool | None = None

    for _ in range(n_activations):
        if current_is_a:
            my_units, opp_units = units_a, units_b
            player = "A"
            my_fr, my_fm = fr_a, fm_a
            opp_fr, opp_fm = fr_b, fm_b
            my_pts, opp_pts_ = pts_a, pts_b
        else:
            my_units, opp_units = units_b, units_a
            player = "B"
            my_fr, my_fm = fr_b, fm_b
            opp_fr, opp_fm = fr_a, fm_a
            my_pts, opp_pts_ = pts_b, pts_a

        alive_mask, enemy_alive_mask = _build_masks(my_units, opp_units)

        if not alive_mask.any():
            # This side has no more activatable units; switch
            if a_finished_first is None:
                a_finished_first = current_is_a

            current_is_a = not current_is_a
            # Check if the other side also done
            if current_is_a:
                other_mask, _ = _build_masks(units_a, units_b)
            else:
                other_mask, _ = _build_masks(units_b, units_a)
            if not other_mask.any():
                # Both sides exhausted — round boundary
                # round_num is 1-indexed; pass 0-indexed to engine functions
                end_round(board, units_a, units_b, round_num - 1, mode)
                round_num += 1
                if round_num > 4:
                    # Game over after round 4 — return actual result
                    winner = score_game(board, units_a, units_b, mode)
                    if winner == "A":
                        return 1.0
                    if winner == "B":
                        return -1.0
                    return 0.0
                _aff = a_finished_first if a_finished_first is not None else True
                current_is_a = start_round(
                    units_a, units_b, round_num - 1, mode,
                    a_finished_first=_aff, ml_sides=_ML_BOTH_SIDES,
                )
                a_finished_first = None
            continue

        state_vec = encode_state_tactical(
            my_units, opp_units, round_num, board, player,
            friendly_ranged_matchups=my_fr, friendly_melee_matchups=my_fm,
            enemy_ranged_matchups=opp_fr, enemy_melee_matchups=opp_fm,
            total_friendly_points=my_pts, total_enemy_points=opp_pts_,
        )

        import numpy as np

        # Pass 1: get unit selection, move type, destination
        # Build destination candidates if advance/rush so we can pass them
        # to the model's forward pass.
        out = model(state_vec, alive_mask, enemy_alive_mask)

        selected_idx = int(out.unit_logits.argmax().item())
        selected_unit = my_units[selected_idx]
        move_type = int(out.move_logits.argmax().item())

        friendly_positions = _get_model_space_positions(my_units, player)
        enemy_positions_ms = _get_model_space_positions(opp_units, player)
        unit_cx, unit_cy = friendly_positions[selected_idx]

        dest = None
        if move_type == MOVE_MOVE:
            enemy_pos_set = _collect_enemy_positions(opp_units)
            candidates, cand_mask, adv_reachable = compute_destination_candidates(
                selected_unit, board, enemy_pos_set, player)
            eam_np = np.array(
                [(i < len(opp_units) and opp_units[i].models_alive > 0)
                 for i in range(MAX_UNITS_PER_SIDE)], dtype=np.bool_)
            budget = float(selected_unit.unit.rush_distance)
            _n_f = len(my_units)
            _n_e = len(opp_units)
            _fr_matchup = np.zeros((_n_f, MAX_UNITS_PER_SIDE, 7), dtype=np.float32)
            _er_matchup = np.zeros((_n_e, MAX_UNITS_PER_SIDE, 7), dtype=np.float32)
            _mm_matchup = np.zeros((_n_e, MAX_UNITS_PER_SIDE), dtype=np.float32)
            dest_feats_np = compute_destination_features(
                candidates, cand_mask, selected_unit, selected_idx, player,
                opp_units, eam_np, _fr_matchup, _er_matchup, _mm_matchup, budget,
                advance_reachable=adv_reachable)
            dest_features_t = torch.from_numpy(dest_feats_np).unsqueeze(0)
            dest_mask_t = torch.from_numpy(cand_mask.astype(np.bool_)).unsqueeze(0)

            # Re-run forward with dest_features to get dest_logits
            out = model(state_vec, alive_mask, enemy_alive_mask,
                        forced_unit_idx=selected_idx,
                        dest_features=dest_features_t, dest_mask=dest_mask_t)
            best_cand = int(out.dest_logits.squeeze(0).argmax().item())
            dest_col, dest_row = int(candidates[best_cand, 0]), int(candidates[best_cand, 1])
            dest = (dest_col, dest_row)

            # Post-move position in model-space for shoot head
            post_x, post_y = float(dest_col), float(dest_row)
            if player == "B":
                post_x = _flip_x(post_x)
                post_y = _flip_y(post_y)
        else:
            post_x, post_y = unit_cx, unit_cy

        # Compute post-move relative features for shoot head
        post_move_rel = compute_post_move_rel(post_x, post_y, enemy_positions_ms)

        # Pass 2: re-run with post_move_rel for conditioned shoot/charge targets
        out2 = model(state_vec, alive_mask, enemy_alive_mask,
                      forced_unit_idx=selected_idx, post_move_rel=post_move_rel)

        charge_target_idx = int(out2.charge_target_logits.argmax().item()) if enemy_alive_mask.any() else 0
        max_wr = max(
            (w.range_inches for w in selected_unit.unit.weapons if not w.melee),
            default=0.0,
        )
        shoot_range_mask = compute_in_range_mask(
            post_move_rel, float(max_wr), enemy_alive_mask)
        masked_shoot_logits = out2.shoot_target_logits.masked_fill(~shoot_range_mask, float('-inf'))
        shoot_target_idx = int(masked_shoot_logits.argmax().item()) if shoot_range_mask.any() else 0
        target_ranking = torch.argsort(masked_shoot_logits, descending=True).tolist()

        action, goal, charge_target, reason = execute_decoded_decision(
            selected_unit, opp_units, move_type, dest,
            charge_target_idx, shoot_target_idx,
        )

        opp_wiped = _execute_activation(
            selected_unit, action, goal, charge_target, reason,
            target_ranking, my_units, opp_units, board, mode,
        )

        if opp_wiped:
            # Game over — side that just moved wins
            return 1.0 if current_is_a else -1.0

        # Check if the moving side's army is also destroyed (e.g. mutual melee)
        my_alive = any(u.models_alive > 0 for u in my_units)
        if not my_alive:
            return -1.0 if current_is_a else 1.0

        current_is_a = not current_is_a

    # Evaluate resulting state with value head (from Player A's perspective)
    # Always encode from A's perspective for consistency
    alive_mask_a, enemy_alive_mask_a = _build_masks(units_a, units_b)
    state_vec = encode_state_tactical(
        units_a, units_b, round_num, board, "A",
        friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
        enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
        total_friendly_points=pts_a, total_enemy_points=pts_b,
    )
    out = model(state_vec, alive_mask_a, enemy_alive_mask_a)
    return out.value.item()


# ---------------------------------------------------------------------------
# Batched planning infrastructure (shared-memory workers + per-worker batching)
# ---------------------------------------------------------------------------

@dataclass
class _PlanningInferenceRequest:
    """Yielded by rollout generator when it needs a batched ML forward pass."""
    state_vec: torch.Tensor         # (2811,)
    alive_mask: torch.Tensor        # (10,)
    enemy_alive_mask: torch.Tensor  # (10,)
    friendly_positions: list        # 10 × (float, float) model-space
    enemy_positions: list           # 10 × (float, float) model-space
    advance_distances: list         # per friendly slot (float)
    rush_distances: list            # per friendly slot (float)
    max_weapon_ranges: list         # per friendly slot (float)


@dataclass
class _PlanningInferenceResult:
    """Sent back to rollout generator with batched argmax outputs."""
    unit_idx: int
    move_type: int
    dest_col: int
    dest_row: int
    charge_target_idx: int
    shoot_target_idx: int
    target_ranking: list
    value: float


# -- Shared-memory pool globals --
_planning_shared_model: TacticalModel | None = None
_planning_pool: _mp.pool.Pool | None = None
_pool_model_id: int | None = None

# Per-worker global (set by _init_planning_worker_shared in child processes)
_g_planning_model: TacticalModel | None = None


def _init_planning_worker_shared(shared_model: TacticalModel) -> None:
    """Initialize worker process with reference to shared-memory model."""
    global _g_planning_model
    _g_planning_model = shared_model
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    import fast_core
    fast_core.USE_C_EXT = fast_core.is_available()


def _ensure_planning_pool(model: TacticalModel, num_workers: int = 0):
    """Return (lazily created) shared-memory worker pool."""
    global _planning_shared_model, _planning_pool, _pool_model_id

    if num_workers <= 0:
        num_workers = max(1, os.cpu_count() or 4)

    if _planning_pool is not None:
        if _pool_model_id == id(model):
            return _planning_pool
        # Model object changed — update shared weights in-place
        _planning_shared_model.load_state_dict(model.state_dict())
        _pool_model_id = id(model)
        return _planning_pool

    # First call — create shared model + pool
    _planning_shared_model = TacticalModel()
    _planning_shared_model.load_state_dict(model.state_dict())
    _planning_shared_model.share_memory()
    _planning_shared_model.eval()

    ctx = _mp.get_context("spawn")
    _planning_pool = ctx.Pool(
        processes=num_workers,
        initializer=_init_planning_worker_shared,
        initargs=(_planning_shared_model,),
    )
    _pool_model_id = id(model)
    return _planning_pool


def _cleanup_planning_pool():
    global _planning_pool
    if _planning_pool is not None:
        _planning_pool.terminate()
        _planning_pool.join()
        _planning_pool = None


atexit.register(_cleanup_planning_pool)


# ---------------------------------------------------------------------------
# Batched argmax forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _batched_argmax_forward(
    model: TacticalModel,
    requests: list[_PlanningInferenceRequest],
) -> list[_PlanningInferenceResult]:
    """Run batched argmax forward for multiple rollout states.

    Replaces 2×B sequential model() calls with 1 batched trunk + manual heads.
    """
    n = len(requests)
    if n == 0:
        return []

    n_units = MAX_UNITS_PER_SIDE

    # Stack inputs
    state_batch = torch.stack([r.state_vec for r in requests])        # (B, 2811)
    alive_batch = torch.stack([r.alive_mask for r in requests])       # (B, 10)
    enemy_batch = torch.stack([r.enemy_alive_mask for r in requests]) # (B, 10)

    # Trunk (single pass)
    h, units, _round_oh = model.trunk(state_batch)                        # (B, 512), (B, 20, 200), (B, 4)

    # Unit selection — argmax
    unit_logits = model.unit_selection_head(h)                         # (B, 10)
    unit_logits = unit_logits.masked_fill(~alive_batch, float('-inf'))
    unit_indices = unit_logits.argmax(dim=-1)                          # (B,)

    # Extract per-sample unit features from unit embeddings
    unit_features = units[:, :n_units, :].gather(
        1, unit_indices.unsqueeze(1).unsqueeze(2).expand(n, 1, TACTICAL_UNIT_FEATURES),
    ).squeeze(1).detach()                                              # (B, 200)

    # Extract can_charge mask for each sample's selected unit
    can_charge_batch = extract_can_charge_mask(state_batch, unit_indices)  # (B, 10)

    # Move type — argmax
    h_uf = torch.cat([h, unit_features], dim=-1)                       # (B, 268)
    move_logits = model.move_type_head(h_uf)                           # (B, 4)
    # Mask charge when no enemy is in charge range
    no_chargeable = ~can_charge_batch.any(dim=-1)                      # (B,)
    move_logits = move_logits.clone()
    move_logits[:, MOVE_CHARGE] = move_logits[:, MOVE_CHARGE].masked_fill(no_chargeable, float('-inf'))
    move_indices = move_logits.argmax(dim=-1)                          # (B,)
    move_onehot = F.one_hot(move_indices, NUM_MOVE_TYPES).float()      # (B, 4)

    # Destination pointer: no Board available in batched rollouts, so use
    # centroid fallback (unit stays at current position for post-move calcs).
    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)       # (B, 272)

    # Charge target — argmax, mask by alive AND chargeable
    charge_logits = model.compute_charge_logits(
        h, units, unit_indices, enemy_batch, can_charge_batch,
    )                                                                  # (B, 10)
    no_enemies = ~enemy_batch.any(dim=-1)                              # (B,)
    charge_indices = charge_logits.argmax(dim=-1)                      # (B,)

    # Per-sample: use unit centroid as post-move position (centroid fallback)
    unit_list = unit_indices.tolist()
    move_list = move_indices.tolist()

    # Store centroid game-space coords for dest_col/dest_row
    dest_cols: list[int] = []
    dest_rows: list[int] = []
    pmr_tensors: list[torch.Tensor] = []
    for i in range(n):
        uid = unit_list[i]
        # Use unit's current centroid position (model-space) as destination
        px, py = requests[i].friendly_positions[uid]
        pmr = compute_post_move_rel(px, py, requests[i].enemy_positions)
        pmr_tensors.append(pmr)
        # Store centroid as game-space coords (just round the model-space pos;
        # for player B the flip is applied when converting to game dest later)
        dest_cols.append(int(round(px)))
        dest_rows.append(int(round(py)))

    # Batched shoot pointer head
    pmr_batch = torch.stack(pmr_tensors)                               # (B, 30)
    max_wr_list = [requests[i].max_weapon_ranges[unit_list[i]] for i in range(n)]
    max_wr_t = torch.tensor(max_wr_list, dtype=torch.float32)
    shoot_mask_batch = compute_in_range_mask_batched(pmr_batch, max_wr_t, enemy_batch)
    shoot_logits = model.compute_shoot_logits(
        h, units, unit_indices, pmr_batch, enemy_batch,
        shoot_range_mask=shoot_mask_batch,
    )                                                                  # (B, 10)
    no_shootable = ~shoot_mask_batch.any(dim=-1)                       # (B,)
    shoot_logits = shoot_logits.masked_fill(no_shootable.unsqueeze(-1), 0.0)
    shoot_indices = shoot_logits.argmax(dim=-1)                        # (B,)

    # Value head
    values = model.value_head(h, _round_oh).reshape(-1)                 # (B,)

    # Build target rankings
    charge_list = charge_indices.tolist()
    shoot_list = shoot_indices.tolist()
    val_list = values.tolist()
    if not isinstance(val_list, list):
        val_list = [val_list]

    results = []
    for i in range(n):
        ranking = (list(range(n_units)) if no_enemies[i] else
                   torch.argsort(shoot_logits[i], descending=True).tolist())
        results.append(_PlanningInferenceResult(
            unit_idx=unit_list[i],
            move_type=move_list[i],
            dest_col=dest_cols[i],
            dest_row=dest_rows[i],
            charge_target_idx=charge_list[i],
            shoot_target_idx=shoot_list[i],
            target_ranking=ranking,
            value=val_list[i],
        ))
    return results


# ---------------------------------------------------------------------------
# Rollout generator coroutine + chunk-batched worker
# ---------------------------------------------------------------------------

def _rollout_generator(
    units_a, units_b, board,
    unit_idx, action, goal, charge_target_idx, reason, target_ranking,
    friendly_is_a, current_is_a, round_num, mode, player,
    N, fr_a, fm_a, fr_b, fm_b, pts_a, pts_b,
):
    """Generator coroutine for one rollout.

    Yields _PlanningInferenceRequest at each lookahead step.
    Receives _PlanningInferenceResult via .send().
    Returns the final value (float) from the *planning player's* perspective
    (positive = good for the planning player).
    """
    friendly = units_a if friendly_is_a else units_b
    enemy = units_b if friendly_is_a else units_a
    active_unit = friendly[unit_idx]
    charge_target = enemy[charge_target_idx] if charge_target_idx >= 0 else None

    # Execute the candidate's initial activation
    opp_wiped = _execute_activation(
        active_unit, action, goal, charge_target, reason,
        target_ranking, friendly, enemy, board, mode,
    )

    if opp_wiped:
        return 1.0   # planning player wiped the opponent
    our_alive = any(u.models_alive > 0 for u in friendly)
    if not our_alive:
        return -1.0   # planning player's units destroyed

    # N-step lookahead (replaces simulate_forward)
    la_is_a = not current_is_a  # after our activation, opponent moves next
    la_a_finished_first: bool | None = None  # track who finishes first this round

    for _ in range(N):
        if la_is_a:
            my_units, opp_units = units_a, units_b
            la_player = "A"
            my_fr, my_fm = fr_a, fm_a
            opp_fr, opp_fm = fr_b, fm_b
            my_pts, opp_pts = pts_a, pts_b
        else:
            my_units, opp_units = units_b, units_a
            la_player = "B"
            my_fr, my_fm = fr_b, fm_b
            opp_fr, opp_fm = fr_a, fm_a
            my_pts, opp_pts = pts_b, pts_a

        alive_mask, enemy_alive_mask = _build_masks(my_units, opp_units)

        if not alive_mask.any():
            # This side has no activatable units left
            if la_a_finished_first is None:
                la_a_finished_first = la_is_a

            la_is_a = not la_is_a
            if la_is_a:
                other_mask, _ = _build_masks(units_a, units_b)
            else:
                other_mask, _ = _build_masks(units_b, units_a)
            if not other_mask.any():
                # Both sides exhausted — round boundary
                # round_num is 1-indexed; pass 0-indexed to engine functions
                end_round(board, units_a, units_b, round_num - 1, mode)
                round_num += 1
                if round_num > 4:
                    # Game over after round 4 — return actual result
                    winner = score_game(board, units_a, units_b, mode)
                    if winner == "draw":
                        return 0.0
                    return 1.0 if (winner == "A") == friendly_is_a else -1.0
                # Start new round (both sides ML-controlled in rollouts)
                _aff = la_a_finished_first if la_a_finished_first is not None else True
                la_is_a = start_round(
                    units_a, units_b, round_num - 1, mode,
                    a_finished_first=_aff, ml_sides=_ML_BOTH_SIDES,
                )
                la_a_finished_first = None
            continue

        state_vec = encode_state_tactical(
            my_units, opp_units, round_num, board, la_player,
            friendly_ranged_matchups=my_fr, friendly_melee_matchups=my_fm,
            enemy_ranged_matchups=opp_fr, enemy_melee_matchups=opp_fm,
            total_friendly_points=my_pts, total_enemy_points=opp_pts,
        )
        f_pos = _get_model_space_positions(my_units, la_player)
        e_pos = _get_model_space_positions(opp_units, la_player)
        adv_dists, rush_dists = _get_movement_budgets(my_units)
        mwr = _get_max_weapon_ranges(my_units)

        # >>> YIELD for batched inference <<<
        result = yield _PlanningInferenceRequest(
            state_vec, alive_mask, enemy_alive_mask,
            f_pos, e_pos, adv_dists, rush_dists, mwr,
        )

        # Decode result into game action
        selected_unit = my_units[result.unit_idx]
        mt = result.move_type

        # Compute game-space destination from pointer result.
        # In batched rollouts (centroid fallback), dest_col/dest_row are
        # model-space centroid coords; flip to game-space for player B.
        dest = None
        if mt == MOVE_MOVE:
            gx, gy = float(result.dest_col), float(result.dest_row)
            if la_player == "B":
                gx = _flip_x(gx)
                gy = _flip_y(gy)
            dest = (gx, gy)

        la_action, la_goal, la_charge, la_reason = execute_decoded_decision(
            selected_unit, opp_units, mt, dest,
            result.charge_target_idx, result.shoot_target_idx,
            is_advance_reachable=getattr(result, 'is_advance_reachable', True),
        )

        opp_wiped = _execute_activation(
            selected_unit, la_action, la_goal, la_charge, la_reason,
            result.target_ranking, my_units, opp_units, board, mode,
        )

        if opp_wiped:
            # la_is_a just wiped B; good for planning player iff planning player is A
            return 1.0 if la_is_a == friendly_is_a else -1.0
        my_still = any(u.models_alive > 0 for u in my_units)
        if not my_still:
            # la_is_a's side destroyed; bad for planning player iff planning player is A
            return -1.0 if la_is_a == friendly_is_a else 1.0

        la_is_a = not la_is_a

    # Final value estimation — encode from the planning player's perspective
    # so that the value head output directly represents "how good for me"
    if friendly_is_a:
        val_friendly, val_enemy = units_a, units_b
        val_fr, val_fm = fr_a, fm_a
        val_er, val_em = fr_b, fm_b
        val_fpts, val_epts = pts_a, pts_b
    else:
        val_friendly, val_enemy = units_b, units_a
        val_fr, val_fm = fr_b, fm_b
        val_er, val_em = fr_a, fm_a
        val_fpts, val_epts = pts_b, pts_a

    val_alive, val_enemy_alive = _build_masks(val_friendly, val_enemy)
    state_vec = encode_state_tactical(
        val_friendly, val_enemy, round_num, board, player,
        friendly_ranged_matchups=val_fr, friendly_melee_matchups=val_fm,
        enemy_ranged_matchups=val_er, enemy_melee_matchups=val_em,
        total_friendly_points=val_fpts, total_enemy_points=val_epts,
    )
    f_pos = _get_model_space_positions(val_friendly, player)
    e_pos = _get_model_space_positions(val_enemy, player)
    adv_dists, rush_dists = _get_movement_budgets(val_friendly)
    mwr = _get_max_weapon_ranges(val_friendly)

    final_result = yield _PlanningInferenceRequest(
        state_vec, val_alive, val_enemy_alive,
        f_pos, e_pos, adv_dists, rush_dists, mwr,
    )
    return final_result.value


def _clone_unit_state(us: UnitState) -> UnitState:
    """Cheap deep clone of a UnitState for per-rollout isolation.

    Uses object.__new__ to skip __post_init__/reset() — we copy every mutable
    field explicitly below, and share immutable references (unit, hero_unit).
    Roughly 5× faster than pickle.loads round-trip for typical units.
    """
    new = object.__new__(UnitState)
    new.unit = us.unit                              # immutable ResolvedUnit
    new.models_alive = us.models_alive
    new.wounds_per_model = us.wounds_per_model[:]
    new.shaken = us.shaken
    new.morale_checked = us.morale_checked
    new.activated = us.activated
    new.fatigued = us.fatigued
    new.ai_role = us.ai_role
    new.combat_preference = us.combat_preference
    new.assigned_objective = us.assigned_objective
    new.positions = us.positions[:]
    new.weapons_per_model = [mw[:] for mw in us.weapons_per_model]
    new._removed_positions = us._removed_positions[:]
    new.owner = us.owner
    new.movement_stance = us.movement_stance
    new.hero_model_index = us.hero_model_index
    new.hero_unit = us.hero_unit                    # immutable ResolvedUnit or None
    return new


def _clone_board(board: Board) -> Board:
    """Cheap clone of a Board (occupancy grid + objective control list)."""
    new = object.__new__(Board)
    new.occupancy = bytearray(board.occupancy)
    new.objective_control = list(board.objective_control)
    return new


def _run_chunk_batched_raw(args, *, model_override=None, live_state=None):
    """Per-worker function: evaluate a chunk of candidates with batched inference.

    Returns raw per-rollout values (list[list[float]] of length len(candidate_chunk),
    with each inner list holding M values). Call _run_chunk_batched to get
    per-candidate averages (preserves the original signature for eval-path
    callers going through pool.map).

    Fast path: when ``live_state`` is a tuple
    ``(master_ua, master_ub, master_bd, fr_a, fm_a, fr_b, fm_b, pts_a, pts_b)``
    of in-process references, per-rollout state is cloned directly via
    ``_clone_unit_state`` / ``_clone_board`` instead of deserialised from
    ``state_bytes`` — saves ~60% of the pickle round-trip cost. ``state_bytes``
    is ignored in this path but still passed for signature compatibility.
    """
    (state_bytes, candidate_chunk, M, N, round_num, mode, player,
     friendly_is_a, current_is_a) = args

    model = model_override if model_override is not None else _g_planning_model

    # Create M rollout generators per candidate, each with independent game state
    generators: dict[int, tuple] = {}  # gen_id → (gen, candidate_local_idx)
    finished_values: dict[int, list] = {}  # candidate_local_idx → list of values
    gen_to_cand: dict[int, int] = {}  # gen_id → candidate_local_idx

    if live_state is not None:
        master_ua, master_ub, master_bd, _fr_a, _fm_a, _fr_b, _fm_b, _pts_a, _pts_b = live_state

    gen_id = 0
    for ci, ca in enumerate(candidate_chunk):
        uid_ca = ca[0]
        ranking_ca = ca[4]
        action_ca = ca[7]
        goal_ca = ca[8]
        ct_idx_ca = ca[9]
        reason_ca = ca[10]
        finished_values[ci] = []

        for _ in range(M):
            # Each rollout needs independent game state.
            if live_state is not None:
                ua = [_clone_unit_state(u) for u in master_ua]
                ub = [_clone_unit_state(u) for u in master_ub]
                bd = _clone_board(master_bd)
            else:
                ua, ub, bd, _fr_a, _fm_a, _fr_b, _fm_b, _pts_a, _pts_b = pickle.loads(state_bytes)

            gen = _rollout_generator(
                ua, ub, bd,
                uid_ca, action_ca, goal_ca, ct_idx_ca, reason_ca, ranking_ca,
                friendly_is_a, current_is_a, round_num, mode, player,
                N, _fr_a, _fm_a, _fr_b, _fm_b, _pts_a, _pts_b,
            )
            gen_to_cand[gen_id] = ci
            generators[gen_id] = gen
            gen_id += 1

    # Initialize all generators — advance to first yield or completion
    active: dict[int, tuple] = {}  # gen_id → (gen, request)
    for gid, gen in generators.items():
        try:
            req = next(gen)
            active[gid] = (gen, req)
        except StopIteration as e:
            ci = gen_to_cand[gid]
            finished_values[ci].append(e.value)

    # Coordinator loop: batch inference, advance generators
    while active:
        gids = list(active.keys())
        reqs = [active[gid][1] for gid in gids]

        # Batched forward pass
        results = _batched_argmax_forward(model, reqs)

        # Send results back to generators
        new_active: dict[int, tuple] = {}
        for gid, res in zip(gids, results):
            gen = active[gid][0]
            try:
                next_req = gen.send(res)
                new_active[gid] = (gen, next_req)
            except StopIteration as e:
                ci = gen_to_cand[gid]
                finished_values[ci].append(e.value)

        active = new_active

    return [list(finished_values[ci]) for ci in range(len(candidate_chunk))]


def _run_chunk_batched(args, *, model_override=None):
    """Average-returning wrapper around _run_chunk_batched_raw for eval paths."""
    raw = _run_chunk_batched_raw(args, model_override=model_override)
    return [sum(vs) / len(vs) if vs else 0.0 for vs in raw]


# ---------------------------------------------------------------------------
# Plan activation (§3.3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _apply_with_dest_override_pp_v(
    model, friendly_units, enemy_units, round_num, board, player,
    num_dests: int,
    friendly_ranged_matchups=None, friendly_melee_matchups=None,
    enemy_ranged_matchups=None, enemy_melee_matchups=None,
    total_friendly_points=None, total_enemy_points=None,
):
    """Standard inference, but for MOVE_MOVE the destination is overridden:
    sample num_dests uniform-random legal dests, pick the one that maximises
    pp_v_dest. Charge & shoot heads are re-computed from the new post-move
    state so the chosen action is coherent.

    Returns the same 7-tuple as apply_tactical_model.
    """
    import torch
    import numpy as np
    from ml_integration_tactical import (
        _apply_tactical_model_phased, _flip_x as _fx, _flip_y as _fy,
    )
    from ml_features import extract_is_shaken

    # 1. Run normal inference to get the policy's argmax decision (and to
    #    let us know whether move_type is MOVE_MOVE without re-deriving
    #    the entire chain twice).
    decision = _apply_tactical_model_phased(
        model, friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=friendly_ranged_matchups,
        friendly_melee_matchups=friendly_melee_matchups,
        enemy_ranged_matchups=enemy_ranged_matchups,
        enemy_melee_matchups=enemy_melee_matchups,
        total_friendly_points=total_friendly_points,
        total_enemy_points=total_enemy_points,
    )
    selected_unit, target_ranking, action, goal, charge_target, reason, assessment = decision

    move_type_name = assessment.get('move_type', '')
    if move_type_name != MOVE_TYPE_NAMES[MOVE_MOVE]:
        # Not a MOVE_MOVE — no override; return standard decision unchanged.
        return decision

    selected_idx = assessment.get('selected_slot', None)
    if selected_idx is None:
        return decision

    # 2. Re-derive the phase chain up to h_mt for the overridden-dest pp_v_dest evaluation.
    state_vec = encode_state_tactical(
        friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=friendly_ranged_matchups,
        friendly_melee_matchups=friendly_melee_matchups,
        enemy_ranged_matchups=enemy_ranged_matchups,
        enemy_melee_matchups=enemy_melee_matchups,
        total_friendly_points=total_friendly_points,
        total_enemy_points=total_enemy_points,
    )
    if bool(extract_is_shaken(state_vec, selected_idx).item()):
        # Shaken units skip dest selection; can't override.
        return decision

    state_batched = state_vec.unsqueeze(0)
    h_pre, _, _ = model.encode(state_batched, phase=PHASE_PRE_SELECT,
                               acting_unit_idx=None, h_prev=None)
    h_sel, _, _ = model.encode(state_batched, phase=PHASE_POST_SELECT,
                               acting_unit_idx=selected_idx, h_prev=h_pre)
    h_mt, _, _ = model.encode(state_batched, phase=PHASE_POST_MOVETYPE,
                              acting_unit_idx=selected_idx, h_prev=h_sel)

    # 3. Sample num_dests random legal destinations & evaluate pp_v_dest.
    enemy_pos_set = _collect_enemy_positions(enemy_units)
    cands, cmask, adv_reach = compute_destination_candidates(
        selected_unit, board, enemy_pos_set, player)
    valid = [i for i in range(len(cands)) if cmask[i]]
    if not valid:
        return decision
    sampled = [random.choice(valid) for _ in range(num_dests)]

    # Build batched post-move states for all sampled dests (one trunk encode each).
    best_v = -float('inf')
    best_idx = sampled[0]
    for d_idx in sampled:
        dc, dr = int(cands[d_idx, 0]), int(cands[d_idx, 1])
        is_rush_d = not bool(adv_reach[d_idx])
        post_unit = project_post_move_unit_state(selected_unit, (dc, dr), is_rush=is_rush_d)
        friendly_post = list(friendly_units)
        friendly_post[selected_idx] = post_unit
        sv_post = encode_state_tactical(
            friendly_post, enemy_units, round_num, board, player,
            friendly_ranged_matchups=friendly_ranged_matchups,
            friendly_melee_matchups=friendly_melee_matchups,
            enemy_ranged_matchups=enemy_ranged_matchups,
            enemy_melee_matchups=enemy_melee_matchups,
            total_friendly_points=total_friendly_points,
            total_enemy_points=total_enemy_points,
        )
        h_dest, _, _ = model.encode(sv_post.unsqueeze(0), phase=PHASE_POST_DEST,
                                    acting_unit_idx=selected_idx, h_prev=h_mt)
        v = float(model.per_phase_value(h_dest, PHASE_POST_DEST).item())
        if v > best_v:
            best_v = v
            best_idx = d_idx

    # 4. Commit to best dest & re-derive charge/shoot from the new post-move state.
    dc, dr = int(cands[best_idx, 0]), int(cands[best_idx, 1])
    is_rush = not bool(adv_reach[best_idx])
    post_unit = project_post_move_unit_state(selected_unit, (dc, dr), is_rush=is_rush)
    friendly_post = list(friendly_units)
    friendly_post[selected_idx] = post_unit
    sv_post = encode_state_tactical(
        friendly_post, enemy_units, round_num, board, player,
        friendly_ranged_matchups=friendly_ranged_matchups,
        friendly_melee_matchups=friendly_melee_matchups,
        enemy_ranged_matchups=enemy_ranged_matchups,
        enemy_melee_matchups=enemy_melee_matchups,
        total_friendly_points=total_friendly_points,
        total_enemy_points=total_enemy_points,
    )
    h_dest, units_dest, _ = model.encode(
        sv_post.unsqueeze(0), phase=PHASE_POST_DEST,
        acting_unit_idx=selected_idx, h_prev=h_mt,
    )

    enemy_alive_mask = torch.tensor(
        [(i < len(enemy_units) and enemy_units[i].models_alive > 0)
         for i in range(MAX_UNITS_PER_SIDE)],
        dtype=torch.bool,
    )
    can_charge_mask = extract_can_charge_mask(state_vec, selected_idx)
    enemy_alive_batched = enemy_alive_mask.unsqueeze(0)

    charge_logits = model.compute_charge_logits(
        h_dest, units_dest, selected_idx,
        enemy_alive_batched, can_charge_mask.unsqueeze(0),
    ).squeeze(0)

    enemy_positions = _get_model_space_positions(enemy_units, player)
    px, py = float(dc), float(dr)
    if player == "B":
        px = _fx(px); py = _fy(py)
    post_move_rel = compute_post_move_rel(px, py, enemy_positions)
    max_wr = max((w.range_inches for w in selected_unit.unit.weapons if not w.melee),
                 default=0.0)
    shoot_range_mask = compute_in_range_mask(post_move_rel, float(max_wr), enemy_alive_mask)

    shoot_logits = model.compute_shoot_logits(
        h_dest, units_dest, selected_idx,
        post_move_rel.unsqueeze(0), enemy_alive_batched,
        shoot_range_mask=None,
    ).squeeze(0)
    masked_shoot = shoot_logits.masked_fill(~shoot_range_mask, float('-inf'))

    charge_target_idx = int(charge_logits.argmax().item()) if enemy_alive_mask.any() else 0
    shoot_target_idx = int(masked_shoot.argmax().item()) if shoot_range_mask.any() else 0
    new_target_ranking = torch.argsort(masked_shoot, descending=True).tolist()

    # 5. Build the new action via execute_decoded_decision.
    new_action, new_goal, new_charge_target, new_reason = execute_decoded_decision(
        selected_unit, enemy_units, MOVE_MOVE, (dc, dr),
        charge_target_idx, shoot_target_idx,
        is_advance_reachable=not is_rush,
    )

    # Patch assessment for downstream consumers (viewer etc.) — not strictly
    # required for game progression but keeps it informative.
    assessment = dict(assessment)
    assessment['dest_selected'] = (dc, dr)
    assessment['_dest_overridden_via'] = 'pp_v_dest'
    assessment['_dest_override_n'] = num_dests
    assessment['shoot_target_idx'] = shoot_target_idx
    assessment['charge_target_idx'] = charge_target_idx

    return (selected_unit, new_target_ranking, new_action, new_goal,
            new_charge_target, new_reason, assessment)


def _run_dest_value_calibration(
    model, state_vec, friendly_units, enemy_units, round_num, board, player,
    units_a, units_b, current_is_a, mode,
    fr_a, fm_a, fr_b, fm_b, pts_a, pts_b,
    friendly_ranged_matchups, friendly_melee_matchups,
    enemy_ranged_matchups, enemy_melee_matchups,
    total_friendly_points, total_enemy_points,
    num_dests: int, M: int, N: int,
    log_path: str, activation_id: int,
) -> None:
    """Calibration probe: for the policy-argmax unit, sample num_dests
    random destinations and compute (pp_v_dest, rollout_v) for each.
    Appends rows to log_path. Skips if move_type != MOVE_MOVE.
    """
    import numpy as np
    import torch
    state_batched = state_vec.unsqueeze(0)

    # Phase chain — replicates the inference path so we can pluck h_mt.
    h_pre, _units_pre, _ = model.encode(
        state_batched, phase=PHASE_PRE_SELECT,
        acting_unit_idx=None, h_prev=None,
    )

    # Argmax unit (with alive mask)
    alive_mask = torch.tensor(
        [(i < len(friendly_units) and friendly_units[i].models_alive > 0
          and not friendly_units[i].activated)
         for i in range(MAX_UNITS_PER_SIDE)],
        dtype=torch.bool,
    )
    if not alive_mask.any():
        return
    enemy_alive_mask = torch.tensor(
        [(i < len(enemy_units) and enemy_units[i].models_alive > 0)
         for i in range(MAX_UNITS_PER_SIDE)],
        dtype=torch.bool,
    )
    unit_logits = model.unit_selection_head(h_pre).squeeze(0)
    unit_logits = unit_logits.masked_fill(~alive_mask, float('-inf'))
    selected_idx = int(unit_logits.argmax().item())
    selected_unit = friendly_units[selected_idx]

    # Phase POST_SELECT
    h_sel, units_sel, _ = model.encode(
        state_batched, phase=PHASE_POST_SELECT,
        acting_unit_idx=selected_idx, h_prev=h_pre,
    )

    # Argmax move_type
    unit_features = model._extract_unit_features(units_sel.squeeze(0), selected_idx).detach()
    h_uf = torch.cat([h_sel, unit_features.unsqueeze(0)], dim=-1)
    move_logits = model.move_type_head(h_uf).squeeze(0)
    can_charge_mask = extract_can_charge_mask(state_vec, selected_idx)
    if not can_charge_mask.any():
        move_logits = move_logits.clone()
        move_logits[MOVE_CHARGE] = float('-inf')
    move_type = int(move_logits.argmax().item())
    if move_type != MOVE_MOVE:
        return

    # Phase POST_MOVETYPE — used as h_prev for the POST_DEST encode below
    h_mt, _units_mt, _ = model.encode(
        state_batched, phase=PHASE_POST_MOVETYPE,
        acting_unit_idx=selected_idx, h_prev=h_sel,
    )

    # Destination candidates
    enemy_pos_set = _collect_enemy_positions(enemy_units)
    cands, cmask, adv_reach = compute_destination_candidates(
        selected_unit, board, enemy_pos_set, player)
    valid = [i for i in range(len(cands)) if cmask[i]]
    if not valid:
        return

    sampled = [random.choice(valid) for _ in range(num_dests)]

    # Per-candidate: build post-move state, encode at POST_DEST, get pp_v_dest
    pp_v_dests: list[float] = []
    candidate_actions: list[tuple] = []
    enemy_positions = _get_model_space_positions(enemy_units, player)
    max_wr = max((w.range_inches for w in selected_unit.unit.weapons if not w.melee), default=0.0)

    for d_idx in sampled:
        dest_col, dest_row = int(cands[d_idx, 0]), int(cands[d_idx, 1])
        is_rush = not bool(adv_reach[d_idx])
        post_unit = project_post_move_unit_state(selected_unit, (dest_col, dest_row), is_rush=is_rush)
        friendly_post = list(friendly_units)
        friendly_post[selected_idx] = post_unit
        state_vec_post = encode_state_tactical(
            friendly_post, enemy_units, round_num, board, player,
            friendly_ranged_matchups=friendly_ranged_matchups,
            friendly_melee_matchups=friendly_melee_matchups,
            enemy_ranged_matchups=enemy_ranged_matchups,
            enemy_melee_matchups=enemy_melee_matchups,
            total_friendly_points=total_friendly_points,
            total_enemy_points=total_enemy_points,
        )
        h_dest, _, _ = model.encode(
            state_vec_post.unsqueeze(0), phase=PHASE_POST_DEST,
            acting_unit_idx=selected_idx, h_prev=h_mt,
        )
        v_dest = float(model.per_phase_value(h_dest, PHASE_POST_DEST).item())
        pp_v_dests.append(v_dest)

        # Build candidate_action for rollout. Shoot target: uniform over
        # legal at the new position; charge target: -1 (not charging).
        post_x, post_y = float(dest_col), float(dest_row)
        if player == "B":
            post_x = _flip_x(post_x); post_y = _flip_y(post_y)
        post_move_rel = compute_post_move_rel(post_x, post_y, enemy_positions)
        shoot_range_mask = compute_in_range_mask(post_move_rel, float(max_wr), enemy_alive_mask)
        no_shootable = (not enemy_alive_mask.any()) or (not shoot_range_mask.any())
        if no_shootable:
            shoot_target_idx = 0
        else:
            _legal = (enemy_alive_mask & shoot_range_mask).float()
            _u = _legal / _legal.sum().clamp(min=1)
            shoot_target_idx = int(torch.multinomial(_u, 1).item())

        action_str, goal, _charge_target_unit, reason = execute_decoded_decision(
            selected_unit, enemy_units, MOVE_MOVE, (dest_col, dest_row),
            -1, shoot_target_idx, is_advance_reachable=not is_rush,
        )
        will_not_shoot = no_shootable
        target_ranking = list(range(MAX_UNITS_PER_SIDE))
        candidate_actions.append((
            selected_idx, MOVE_MOVE, dest_col, dest_row, target_ranking,
            -1, shoot_target_idx, action_str, goal, -1, reason, will_not_shoot,
        ))

    # Rollouts
    state_bytes = pickle.dumps((
        list(units_a), list(units_b), board,
        fr_a, fm_a, fr_b, fm_b, pts_a, pts_b,
    ))
    friendly_is_a = (player == "A")
    raw_values = _run_chunk_batched_raw((
        state_bytes, candidate_actions, M, N, round_num, mode, player,
        friendly_is_a, current_is_a,
    ), model_override=model, live_state=None)
    rollout_means = [(sum(vs) / len(vs)) if vs else 0.0 for vs in raw_values]

    # Append to CSV
    import os
    write_header = not os.path.exists(log_path)
    with open(log_path, "a") as f:
        if write_header:
            f.write("activation_id,player,unit_idx,dest_idx,dest_col,dest_row,"
                    "pp_v_dest,rollout_v\n")
        for d_idx, pp_v, rv, ca in zip(sampled, pp_v_dests, rollout_means,
                                       candidate_actions):
            f.write(f"{activation_id},{player},{selected_idx},{d_idx},"
                    f"{ca[2]},{ca[3]},{pp_v:.6f},{rv:.6f}\n")


def plan_activation(
    model: TacticalModel,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    units_a: list[UnitState],
    units_b: list[UnitState],
    current_is_a: bool,
    mode: str,
    *,
    friendly_ranged_matchups: list[list[list[float]]] | None = None,
    friendly_melee_matchups: list[list[float]] | None = None,
    enemy_ranged_matchups: list[list[list[float]]] | None = None,
    enemy_melee_matchups: list[list[float]] | None = None,
    total_friendly_points: int | None = None,
    total_enemy_points: int | None = None,
    fr_a: list[list[list[float]]] | None = None,
    fm_a: list[list[float]] | None = None,
    fr_b: list[list[list[float]]] | None = None,
    fm_b: list[list[float]] | None = None,
    pts_a: int = 0,
    pts_b: int = 0,
    planning_params: dict | None = None,
) -> tuple[UnitState | None, list[int], str, tuple[int, int] | None, UnitState | None, str, list[dict]]:
    """Monte Carlo search for the best activation decision.

    Returns (selected_unit, target_ranking, action, goal, charge_target, reason,
             planning_candidates) where planning_candidates is a list of dicts
             describing every candidate action evaluated and its rollout score.
    """
    import time as _time
    _t0 = _time.perf_counter()

    params = planning_params or {}
    # Per-side params: when comparing two configurations head-to-head we
    # pass {"_per_side_params": {"A": <dict>, "B": <dict>}} so that the
    # planner picks the right config based on `player`. Falls through to
    # normal behaviour when the wrapper key isn't present.
    if "_per_side_params" in params:
        params = params["_per_side_params"].get(player, params)

    # Dest-override mode: standard inference but for MOVE_MOVE the
    # destination is replaced by the pp_v_dest top-1 of N random legal
    # samples. Charge & shoot are re-derived from the new post-move state.
    _dest_override_n = int(params.get("DEST_OVERRIDE_PP_V_N", 0))
    if _dest_override_n > 0:
        return _apply_with_dest_override_pp_v(
            model, friendly_units, enemy_units, round_num, board, player,
            num_dests=_dest_override_n,
            friendly_ranged_matchups=friendly_ranged_matchups,
            friendly_melee_matchups=friendly_melee_matchups,
            enemy_ranged_matchups=enemy_ranged_matchups,
            enemy_melee_matchups=enemy_melee_matchups,
            total_friendly_points=total_friendly_points,
            total_enemy_points=total_enemy_points,
        )

    # Calibration mode: run the dest-value calibration probe as a
    # side-effect, then delegate the actual decision to the policy via
    # apply_tactical_model so the game proceeds with the policy's argmax.
    _cal_n = int(params.get("CALIBRATION_DEST_PROBE", 0))
    if _cal_n > 0:
        # Encode state once for the probe (re-encoded inside, but we need it).
        _state_vec = encode_state_tactical(
            friendly_units, enemy_units, round_num, board, player,
            friendly_ranged_matchups=friendly_ranged_matchups,
            friendly_melee_matchups=friendly_melee_matchups,
            enemy_ranged_matchups=enemy_ranged_matchups,
            enemy_melee_matchups=enemy_melee_matchups,
            total_friendly_points=total_friendly_points,
            total_enemy_points=total_enemy_points,
        )
        _fr_a_use = fr_a if fr_a is not None else (friendly_ranged_matchups if player == "A" else enemy_ranged_matchups)
        _fm_a_use = fm_a if fm_a is not None else (friendly_melee_matchups if player == "A" else enemy_melee_matchups)
        _fr_b_use = fr_b if fr_b is not None else (enemy_ranged_matchups if player == "A" else friendly_ranged_matchups)
        _fm_b_use = fm_b if fm_b is not None else (enemy_melee_matchups if player == "A" else friendly_melee_matchups)
        _pts_a_use = pts_a if pts_a else (total_friendly_points if player == "A" else total_enemy_points) or 0
        _pts_b_use = pts_b if pts_b else (total_enemy_points if player == "A" else total_friendly_points) or 0

        # Per-call activation id is taken from a counter file so callers
        # don't need to thread it; cheap and self-contained.
        _log = params.get("CALIBRATION_LOG_PATH", "/tmp/dest_calibration.csv")
        _ctr_path = _log + ".ctr"
        _aid = 0
        try:
            with open(_ctr_path, "r") as _f:
                _aid = int(_f.read().strip() or "0")
        except FileNotFoundError:
            _aid = 0
        with open(_ctr_path, "w") as _f:
            _f.write(str(_aid + 1))

        _run_dest_value_calibration(
            model, _state_vec, friendly_units, enemy_units, round_num, board, player,
            units_a, units_b, current_is_a, mode,
            _fr_a_use, _fm_a_use, _fr_b_use, _fm_b_use, _pts_a_use, _pts_b_use,
            friendly_ranged_matchups, friendly_melee_matchups,
            enemy_ranged_matchups, enemy_melee_matchups,
            total_friendly_points, total_enemy_points,
            num_dests=_cal_n,
            M=int(params.get("M_ROLLOUTS", DEFAULT_M_ROLLOUTS)),
            N=int(params.get("N_LOOKAHEAD", DEFAULT_N_LOOKAHEAD)),
            log_path=_log, activation_id=_aid,
        )

        # Now return the policy's argmax decision so the game proceeds normally.
        from ml_integration_tactical import apply_tactical_model as _atm
        _decision = _atm(
            model, friendly_units, enemy_units, round_num, board, player,
            friendly_ranged_matchups=friendly_ranged_matchups,
            friendly_melee_matchups=friendly_melee_matchups,
            enemy_ranged_matchups=enemy_ranged_matchups,
            enemy_melee_matchups=enemy_melee_matchups,
            total_friendly_points=total_friendly_points,
            total_enemy_points=total_enemy_points,
        )
        # apply_tactical_model returns 7-tuple matching plan_activation's shape.
        _sel, _rank, _act, _goal, _ct, _reason, _ = _decision
        return _sel, _rank, _act, _goal, _ct, _reason, []
    K = params.get("K_UNITS", DEFAULT_K_UNITS)
    C = params.get("C_SAMPLES_PER_UNIT", DEFAULT_C_SAMPLES_PER_UNIT)
    M = params.get("M_ROLLOUTS", DEFAULT_M_ROLLOUTS)
    N = params.get("N_LOOKAHEAD", DEFAULT_N_LOOKAHEAD)
    num_workers = params.get("NUM_WORKERS", DEFAULT_NUM_WORKERS)
    verbose = params.get("VERBOSE", False)
    parallel = (num_workers != 1)
    # Diagnostic flags for the policy-vs-planner-with-random-candidates
    # probe. K_INDEPENDENT_UNIFORM=K bypasses the K_UNITS×C structure
    # entirely: generates exactly K candidates, each picking a unit
    # uniformly with replacement from alive units. UNIFORM_ALT_SAMPLING
    # makes all sub-action sampling (move/dest/charge/shoot) uniform-
    # over-legal instead of policy's softmax. Defaults preserve normal
    # eval-time behaviour. See probe_planner_vs_policy.py.
    k_indep = int(params.get("K_INDEPENDENT_UNIFORM", 0))
    uniform_alt = bool(params.get("UNIFORM_ALT_SAMPLING", False))
    # When set with k_indep, the first candidate is forced to be the
    # policy's joint argmax (argmax unit + argmax sub-actions); the
    # remaining K-1 are uniform-random as before. K=1 then becomes a
    # pure-policy mirror baseline.
    argmax_first = bool(params.get("INCLUDE_ARGMAX_FIRST", False))
    # When >= 0, splits the non-argmax slots: the first N use policy
    # multinomial sampling, the remaining use uniform-over-legal. -1
    # (default) keeps the existing single-mode behaviour driven by
    # uniform_alt. Use this to construct mixed candidate pools, e.g.
    # 1 argmax + 9 policy + 30 uniform.
    n_policy_samples = int(params.get("N_POLICY_SAMPLES", -1))
    if k_indep > 0 and n_policy_samples < 0:
        uniform_alt = True  # k_indep without explicit split → all uniform

    # Build masks
    alive_mask = torch.tensor(
        [(i < len(friendly_units) and friendly_units[i].models_alive > 0
          and not friendly_units[i].activated)
         for i in range(MAX_UNITS_PER_SIDE)],
        dtype=torch.bool,
    )
    if not alive_mask.any():
        return None, [], "hold", None, None, "no units available", []

    enemy_alive_mask = torch.tensor(
        [(i < len(enemy_units) and enemy_units[i].models_alive > 0)
         for i in range(MAX_UNITS_PER_SIDE)],
        dtype=torch.bool,
    )

    # Encode state
    state_vec = encode_state_tactical(
        friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=friendly_ranged_matchups,
        friendly_melee_matchups=friendly_melee_matchups,
        enemy_ranged_matchups=enemy_ranged_matchups,
        enemy_melee_matchups=enemy_melee_matchups,
        total_friendly_points=total_friendly_points,
        total_enemy_points=total_enemy_points,
    )

    # 1. One trunk pass
    x = state_vec.unsqueeze(0)  # (1, TACTICAL_TOTAL_FEATURES)
    h, units, _ = model.trunk(x)
    h = h.squeeze(0)              # (512,)
    units = units.squeeze(0)      # (20, 200)

    unit_logits = model.unit_selection_head(h.unsqueeze(0)).squeeze(0)  # (10,)
    unit_logits = unit_logits.masked_fill(~alive_mask, float('-inf'))

    # 2. Select candidate units. Default: top-K by policy probability.
    # K_INDEPENDENT_UNIFORM mode: K candidates, each unit picked
    # uniformly with replacement from alive units.
    unit_probs = torch.softmax(unit_logits, dim=-1)
    num_alive = int(alive_mask.sum().item())
    if k_indep > 0:
        import random as _random
        _alive_ids = [i for i in range(alive_mask.shape[-1]) if bool(alive_mask[i])]
        if argmax_first:
            argmax_unit_id = int(unit_probs.argmax().item())
            # First slot = argmax_unit; remaining K-1 slots = uniform random
            # (with replacement). K=1 collapses to the argmax candidate alone.
            candidate_units = [argmax_unit_id] + [
                _random.choice(_alive_ids) for _ in range(k_indep - 1)
            ]
        else:
            candidate_units = [_random.choice(_alive_ids) for _ in range(k_indep)]
    else:
        k = min(K, num_alive)
        _, top_indices = torch.topk(unit_probs, k)
        candidate_units = top_indices.tolist()

    # 3. Precompute per-unit features (trunk pass already done)
    # We'll run conditioned heads manually per sample to respect the chain.

    # Precompute damage for rollouts if not provided
    _fr_a = fr_a if fr_a is not None else (friendly_ranged_matchups if player == "A" else enemy_ranged_matchups)
    _fm_a = fm_a if fm_a is not None else (friendly_melee_matchups if player == "A" else enemy_melee_matchups)
    _fr_b = fr_b if fr_b is not None else (enemy_ranged_matchups if player == "A" else friendly_ranged_matchups)
    _fm_b = fm_b if fm_b is not None else (enemy_melee_matchups if player == "A" else friendly_melee_matchups)
    _pts_a = pts_a if pts_a else (total_friendly_points if player == "A" else total_enemy_points) or 0
    _pts_b = pts_b if pts_b else (total_enemy_points if player == "A" else total_friendly_points) or 0

    # --- Generate candidate actions ---
    # Each entry: (uid, move_type, dest_col, dest_row, target_ranking,
    #              charge_target_idx, shoot_target_idx, action, goal, ct_idx, reason,
    #              no_shoot)
    # Sampling respects the conditional head chain:
    #   move_type   ~ P(mt  | h, unit_feat)
    #   destination ~ Pointer(candidates | h, unit_feat, mt)
    #   charge_tgt  ~ P(ct  | h, unit_feat, mt)
    #   shoot_tgt   ~ P(st  | h, unit_feat, mt, post_move_rel)
    import numpy as np
    candidate_actions: list[tuple] = []
    _seen_keys: set[tuple] = set()  # dedup by resolved action

    h_b = h.unsqueeze(0)  # (1, H) for head inputs

    # Get model-space positions for post-move computation
    friendly_positions = _get_model_space_positions(friendly_units, player)
    enemy_positions = _get_model_space_positions(enemy_units, player)

    no_enemies = not enemy_alive_mask.any()
    enemy_pos_set = _collect_enemy_positions(enemy_units)
    eam_np = np.array(
        [(i < len(enemy_units) and enemy_units[i].models_alive > 0)
         for i in range(MAX_UNITS_PER_SIDE)], dtype=np.bool_)

    for _cand_i, uid in enumerate(candidate_units):
        # Tracks whether THIS slot must use the policy's joint argmax
        # (only the first slot when k_indep+argmax_first).
        is_argmax_slot = (k_indep > 0 and argmax_first and _cand_i == 0)
        # Per-slot routing for mixed pools: when N_POLICY_SAMPLES is set,
        # the first N non-argmax slots use policy sampling, the rest
        # uniform. Without it we fall back to the global uniform_alt.
        if k_indep > 0 and n_policy_samples >= 0 and not is_argmax_slot:
            _non_argmax_idx = _cand_i - (1 if argmax_first else 0)
            slot_uniform = (_non_argmax_idx >= n_policy_samples)
        else:
            slot_uniform = uniform_alt
        unit = friendly_units[uid]
        max_wr = max(
            (w.range_inches for w in unit.unit.weapons if not w.melee),
            default=0.0,
        )
        unit_features = model._extract_unit_features(units, uid).detach()  # (200,)
        uf_b = unit_features.unsqueeze(0)  # (1, 200)

        # Extract can_charge mask for this unit
        can_charge_mask = extract_can_charge_mask(state_vec, uid)  # (10,) bool

        # Move type logits depend on h + unit_features — compute once per unit
        h_uf = torch.cat([h_b, uf_b], dim=-1)           # (1, 268)
        move_logits = model.move_type_head(h_uf).squeeze(0)  # (4,)
        # Mask charge when no enemy is in charge range
        if not can_charge_mask.any():
            move_logits = move_logits.clone()
            move_logits[MOVE_CHARGE] = float('-inf')
        move_probs = torch.softmax(move_logits, dim=-1)

        # Precompute destination candidates (unified rush budget)
        _dest_cache_move: tuple | None = None
        if move_probs[MOVE_MOVE].item() > 0:
            cands, cmask, adv_reach = compute_destination_candidates(
                unit, board, enemy_pos_set, player)
            budget = float(unit.unit.rush_distance)
            _n_f = len(friendly_units)
            _n_e = len(enemy_units)
            _fr_m = (np.array(friendly_ranged_matchups, dtype=np.float32)
                     if friendly_ranged_matchups is not None
                     else np.zeros((_n_f, MAX_UNITS_PER_SIDE, 7), dtype=np.float32))
            _er_m = (np.array(enemy_ranged_matchups, dtype=np.float32)
                     if enemy_ranged_matchups is not None
                     else np.zeros((_n_e, MAX_UNITS_PER_SIDE, 7), dtype=np.float32))
            _mm_m = (np.array(friendly_melee_matchups, dtype=np.float32).reshape(_n_f, -1)[:, :MAX_UNITS_PER_SIDE]
                     if friendly_melee_matchups is not None
                     else np.zeros((_n_e, MAX_UNITS_PER_SIDE), dtype=np.float32))
            dest_feats_np = compute_destination_features(
                cands, cmask, unit, uid, player,
                enemy_units, eam_np, _fr_m, _er_m, _mm_m, budget,
                advance_reachable=adv_reach)
            dest_features_t = torch.from_numpy(dest_feats_np).unsqueeze(0)
            dest_mask_t = torch.from_numpy(cmask.astype(np.bool_)).unsqueeze(0)

            mt_onehot = F.one_hot(
                torch.tensor(MOVE_MOVE), NUM_MOVE_TYPES
            ).float().unsqueeze(0)
            h_uf_m_cand = torch.cat([h_b, uf_b, mt_onehot], dim=-1)
            dest_logits = model.compute_dest_logits(
                h_uf_m_cand, dest_features_t, dest_mask_t).squeeze(0)
            _dest_cache_move = (cands, cmask, adv_reach, dest_feats_np, dest_logits)

        # In k_indep mode each candidate_units entry is a fresh independent
        # candidate, so we draw exactly one sub-action sample per entry.
        n_samples = 1 if k_indep > 0 else C
        for sample_i in range(n_samples):
            # 1. Sample move type
            if is_argmax_slot:
                move_type = int(move_probs.argmax().item())
            elif slot_uniform:
                _legal = torch.isfinite(move_logits).float()
                _u = _legal / _legal.sum().clamp(min=1)
                move_type = int(torch.multinomial(_u, 1).item())
            else:
                move_type = int(torch.multinomial(move_probs, 1).item())
            move_onehot = F.one_hot(
                torch.tensor(move_type), NUM_MOVE_TYPES
            ).float().unsqueeze(0)

            h_uf_m = torch.cat([h_b, uf_b, move_onehot], dim=-1)

            # 2. Destination pointer: sample or cycle through top-K candidates
            unit_cx, unit_cy = friendly_positions[uid]
            dest_col, dest_row = int(round(unit_cx)), int(round(unit_cy))  # default: centroid
            _pick_ar = True  # advance-reachable for selected dest

            if move_type == MOVE_MOVE and _dest_cache_move is not None:
                cands, cmask, adv_reach, _, dest_logits = _dest_cache_move
                n_valid = int(cmask.sum())
                if n_valid > 0:
                    if is_argmax_slot:
                        pick_idx = int(dest_logits.argmax().item())
                    elif slot_uniform:
                        _legal = torch.isfinite(dest_logits).float()
                        _u = _legal / _legal.sum().clamp(min=1)
                        pick_idx = int(torch.multinomial(_u, 1).item())
                    else:
                        dest_probs = torch.softmax(dest_logits, dim=-1)
                        _, top_dest = torch.topk(dest_probs, min(C, n_valid))
                        pick_idx = top_dest[sample_i % len(top_dest)].item()
                    dest_col = int(cands[pick_idx, 0])
                    dest_row = int(cands[pick_idx, 1])
                    _pick_ar = bool(adv_reach[pick_idx])

                post_x, post_y = float(dest_col), float(dest_row)
                if player == "B":
                    post_x = _flip_x(post_x)
                    post_y = _flip_y(post_y)
            else:
                post_x, post_y = unit_cx, unit_cy

            # 3. Sample charge target (pointer head)
            charge_logits = model.compute_charge_logits(
                h_b.squeeze(0), units.squeeze(0), uid,
                enemy_alive_mask, can_charge_mask,
            )
            no_chargeable = no_enemies or not (enemy_alive_mask & can_charge_mask).any()
            if no_chargeable:
                charge_target_idx = 0
            elif is_argmax_slot:
                charge_target_idx = int(charge_logits.argmax().item())
            elif slot_uniform:
                _legal = (enemy_alive_mask & can_charge_mask).float()
                _u = _legal / _legal.sum().clamp(min=1)
                charge_target_idx = int(torch.multinomial(_u, 1).item())
            else:
                charge_probs = torch.softmax(charge_logits, dim=-1)
                charge_target_idx = int(torch.multinomial(charge_probs, 1).item())

            # 4. Compute post_move_rel and sample shoot target
            post_move_rel = compute_post_move_rel(post_x, post_y, enemy_positions)
            shoot_range_mask = compute_in_range_mask(
                post_move_rel, float(max_wr), enemy_alive_mask)
            shoot_logits = model.compute_shoot_logits(
                h_b.squeeze(0), units.squeeze(0), uid,
                post_move_rel, enemy_alive_mask,
                shoot_range_mask=shoot_range_mask,
            )
            no_shootable = no_enemies or not shoot_range_mask.any()
            if no_shootable:
                shoot_target_idx = 0
            elif is_argmax_slot:
                shoot_target_idx = int(shoot_logits.argmax().item())
            elif slot_uniform:
                _legal = (enemy_alive_mask & shoot_range_mask).float()
                _u = _legal / _legal.sum().clamp(min=1)
                shoot_target_idx = int(torch.multinomial(_u, 1).item())
            else:
                shoot_probs = torch.softmax(shoot_logits, dim=-1)
                shoot_target_idx = int(torch.multinomial(shoot_probs, 1).item())

            target_ranking = torch.argsort(shoot_logits, descending=True).tolist()

            # Convert to game-space destination
            dest = None
            if move_type == MOVE_MOVE:
                dest = (dest_col, dest_row)

            # Resolve the candidate action
            action, goal, charge_target_unit, reason = execute_decoded_decision(
                unit, enemy_units, move_type, dest,
                charge_target_idx, shoot_target_idx,
                is_advance_reachable=_pick_ar,
            )

            # Convert charge_target reference to index for serialization
            ct_idx = -1
            if charge_target_unit is not None:
                for j, eu in enumerate(enemy_units):
                    if eu is charge_target_unit:
                        ct_idx = j
                        break

            # Deduplicate: skip if we already have an identical resolved action
            dedup_key = (uid, move_type, action, goal, ct_idx)
            if dedup_key in _seen_keys:
                continue
            _seen_keys.add(dedup_key)

            will_not_shoot = (
                no_shootable
                or (move_type == MOVE_MOVE and not _pick_ar)
                or move_type == MOVE_CHARGE
            )
            candidate_actions.append((
                uid, move_type, dest_col, dest_row, target_ranking,
                charge_target_idx, shoot_target_idx,
                action, goal, ct_idx, reason, will_not_shoot,
            ))

    # --- Evaluate candidates (chunked + batched) ---
    friendly_is_a = (player == "A")

    # Serialize game state once for all workers / rollouts
    state_bytes = pickle.dumps((
        list(units_a), list(units_b), board,
        _fr_a, _fm_a, _fr_b, _fm_b, _pts_a, _pts_b,
    ))

    if parallel and len(candidate_actions) > 1:
        pool = _ensure_planning_pool(model, num_workers)
        effective_workers = num_workers if num_workers > 0 else (os.cpu_count() or 4)

        # Chunk candidates across workers
        n_cands = len(candidate_actions)
        chunk_size = max(1, n_cands // effective_workers)
        chunks = []
        for i in range(0, n_cands, chunk_size):
            chunk = candidate_actions[i : i + chunk_size]
            chunks.append((
                state_bytes, chunk, M, N, round_num, mode, player,
                friendly_is_a, current_is_a,
            ))

        chunk_results = list(pool.map(_run_chunk_batched, chunks))
        avg_values = [v for chunk_vals in chunk_results for v in chunk_vals]
    else:
        # Sequential but still batched within the single "chunk"
        avg_values = _run_chunk_batched((
            state_bytes, candidate_actions, M, N, round_num, mode, player,
            friendly_is_a, current_is_a,
        ), model_override=model)

    if verbose:
        _t1 = _time.perf_counter()
        _mode_label = f"parallel({num_workers or os.cpu_count()})" if parallel else "sequential"
        print(f"  [planning] {len(candidate_actions)} candidates × M={M}: "
              f"{_t1 - _t0:.2f}s ({_mode_label})")

    if not candidate_actions:
        return None, [], "hold", None, None, "no candidates", []

    # 5. Pick the best candidate
    best_idx = max(range(len(avg_values)), key=lambda i: avg_values[i])
    best_ca = candidate_actions[best_idx]
    best_val = avg_values[best_idx]

    selected_unit = friendly_units[best_ca[0]]
    best_ranking = best_ca[4]
    action = best_ca[7]
    goal = best_ca[8]
    ct_idx = best_ca[9]
    charge_target = enemy_units[ct_idx] if ct_idx >= 0 else None
    reason = best_ca[10]

    # Build planning_candidates list for viewer diagnostics
    # Disambiguate duplicate unit names (e.g. two "Support Artillery" → A, B)
    def _build_labels(units):
        counts = Counter(u.unit.name for u in units)
        seq: dict[str, int] = {}
        labels: dict[int, str] = {}
        for i, u in enumerate(units):
            base = u.unit.name
            if counts[base] > 1:
                idx = seq.get(base, 0)
                labels[i] = f"{base} {chr(65 + idx)}"
                seq[base] = idx + 1
            else:
                labels[i] = base
        return labels

    _uid_label = _build_labels(friendly_units)
    _eid_label = _build_labels(enemy_units)

    planning_candidates = []
    for i, (ca, v) in enumerate(zip(candidate_actions, avg_values)):
        uid_c = ca[0]
        move_type_c = ca[1]
        ranking_c = ca[4]
        no_shoot_c = ca[11]
        # Find top-ranked alive enemy for this candidate
        if no_shoot_c:
            top_target_name = "N/A"
        else:
            top_target_name = None
            for tidx in ranking_c:
                if tidx < len(enemy_units) and enemy_units[tidx].models_alive > 0:
                    top_target_name = _eid_label[tidx]
                    break
        planning_candidates.append({
            'unit_idx': uid_c,
            'unit_name': _uid_label[uid_c],
            'move_type': MOVE_TYPE_NAMES[move_type_c],
            'dest_col': ca[2],
            'dest_row': ca[3],
            'action': ca[7],
            'goal': ca[8],
            'reason': ca[10],
            'value': v,
            'top_target': top_target_name,
            'selected': (i == best_idx),
        })

    return selected_unit, best_ranking, action, goal, charge_target, reason, planning_candidates


# ---------------------------------------------------------------------------
# Training-time planning (§3.10 of training_augmentation_spec)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plan_training_activation(
    model: TacticalModel,
    state_vec: torch.Tensor,
    alive_mask: torch.Tensor,
    enemy_alive_mask: torch.Tensor,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    *,
    current_is_a: bool,
    mode: str,
    friendly_positions: list[tuple[float, float]],
    enemy_positions: list[tuple[float, float]],
    advance_distances: list[float],
    rush_distances: list[float],
    max_weapon_ranges: list[float] | None = None,
    fr_a=None, fm_a=None, fr_b=None, fm_b=None,
    pts_a: int = 0, pts_b: int = 0,
    planning_params: dict | None = None,
    opponent_type: int | None = None,
) -> tuple:
    """Training-time planning with policy-argmax baseline.

    Runs single-threaded (no worker pool). Returns the same action tuple
    format as sample_tactical_actions_no_grad, plus planning metadata.

    Returns:
        (unit_idx, move_type, dest_col, dest_row, dest_cand_idx,
         charge_target_idx, shoot_target_idx, target_ranking,
         post_move_rel, old_log_prob, value, shoot_mask,
         was_planned, planning_improved, planning_value_delta,
         planning_unit_values, planning_unit_indices)
    """
    import numpy as np

    params = planning_params or {}
    K = params.get("K_UNITS", 3)
    C = params.get("C_SAMPLES_PER_UNIT", 3)
    M = params.get("M_ROLLOUTS", 4)
    N = params.get("N_LOOKAHEAD", 3)
    sh_enabled = params.get("SEQUENTIAL_HALVING", False)
    sh_schedule = tuple(params.get("SH_SCHEDULE", ()))
    # Diagnostic flag for the uniform-baseline probe. When True, the K-1
    # non-argmax candidate units are picked uniformly from alive units
    # (rather than top-K by policy), and all sub-action sampling
    # (move type / dest / charge / shoot) for non-argmax candidates uses
    # uniform-over-legal-mask in place of the policy's softmax. The
    # argmax candidate (ui=0, si=0) is unchanged. Default False — no
    # behavioural change for normal training/eval.
    uniform_alt = params.get("UNIFORM_ALT_SAMPLING", False)

    # MPO/distillation extension. Per-slot temperature warming and an
    # explicit (argmax, policy, temp, uniform) candidate mix. Defaults
    # preserve legacy behaviour bit-exactly:
    #   N_POLICY_SAMPLES = -1 → all non-argmax slots are routed by the
    #     legacy uniform_alt flag (uniform if True, else policy/topk).
    #   N_POLICY_SAMPLES >= 0 → opt into the explicit mix. Slots are
    #     filled in order: argmax(1) → policy_τ=1 (N_POLICY_SAMPLES) →
    #     policy_τ=TAU (N_TEMP_SAMPLES) → remainder uniform-over-legal.
    # When the explicit mix is active, dest sampling switches from the
    # legacy topk-cycle to true multinomial (consistent with move/
    # charge/shoot). TAU only affects "temp" slots; "policy" slots are
    # always τ=1 regardless of TAU.
    tau = float(params.get("TAU", 1.0))
    n_policy_samples = int(params.get("N_POLICY_SAMPLES", -1))
    n_temp_samples = int(params.get("N_TEMP_SAMPLES", 0))

    eps = 1e-8
    no_enemies = not enemy_alive_mask.any()

    # Derive per-side matchups for destination features
    if player == "A":
        friendly_ranged_matchups = fr_a
        friendly_melee_matchups = fm_a
        enemy_ranged_matchups = fr_b
    else:
        friendly_ranged_matchups = fr_b
        friendly_melee_matchups = fm_b
        enemy_ranged_matchups = fr_a

    # Guard: alive_mask is MAX_UNITS_PER_SIDE wide; `friendly_units` may be
    # shorter (unit list mutated between request build and planning call).
    # Clamp any "alive" slots beyond the actual list to False so topk/argmax
    # cannot select a non-existent unit.
    n_friendly = len(friendly_units)
    if n_friendly < alive_mask.shape[-1]:
        alive_mask = alive_mask.clone()
        alive_mask[n_friendly:] = False

    # --- Single trunk pass ---
    x = state_vec.unsqueeze(0)
    am = alive_mask.unsqueeze(0)
    h, units, round_onehot = model.trunk(x)

    unit_logits = model.unit_selection_head(h)
    unit_logits = unit_logits.masked_fill(~am, float('-inf'))
    unit_probs = torch.softmax(unit_logits, dim=-1).squeeze(0)

    # --- Argmax unit (candidate 0) ---
    argmax_unit = int(unit_probs.argmax().item())

    # Select K candidate units. Default: top-K by policy probability.
    # Uniform-baseline probe: argmax_unit is forced first, the remaining
    # K-1 are drawn uniformly from the other alive units.
    num_alive = int(alive_mask.sum().item())
    if num_alive == 0:
        # Nothing to plan for — signal "no planning" and let the caller fall
        # back to the normal inference path on the next inference request.
        return None
    k = min(K, num_alive)
    if uniform_alt:
        import random as _random
        _alive_ids = [i for i in range(alive_mask.shape[-1])
                      if bool(alive_mask[i]) and i != argmax_unit]
        _random.shuffle(_alive_ids)
        candidate_units = [argmax_unit] + _alive_ids[:k - 1]
    else:
        _, top_indices = torch.topk(unit_probs, k)
        candidate_units = top_indices.tolist()

    # Defense-in-depth: the alive_mask clamp above should guarantee every uid
    # is < len(friendly_units), but if the unit list was mutated further
    # between the clamp and here, drop any stragglers. If nothing survives,
    # bail to the non-planned inference path.
    n_friendly_now = len(friendly_units)
    candidate_units = [uid for uid in candidate_units if uid < n_friendly_now]
    if not candidate_units:
        return None
    if argmax_unit >= n_friendly_now:
        argmax_unit = candidate_units[0]

    # Ensure argmax_unit is first; remaining K-1 units are sampled from top-K
    if argmax_unit in candidate_units:
        candidate_units.remove(argmax_unit)
    candidate_units = [argmax_unit] + candidate_units[:K - 1]

    h_b = h  # (1, H)

    # --- Generate candidate actions ---
    # Same format as plan_activation: (uid, move_type, dest_col, dest_row, ranking,
    #   charge_tgt_idx, shoot_tgt_idx, action, goal, ct_idx, reason)
    candidate_actions: list[tuple] = []
    candidate_to_unit: list[int] = []  # maps candidate idx -> unit slot
    candidate_shoot_masks: list[torch.Tensor] = []  # per-candidate shoot range masks
    _seen_keys: set[tuple] = set()

    enemy_pos_set = _collect_enemy_positions(enemy_units)
    eam_np = np.array(
        [(i < len(enemy_units) and enemy_units[i].models_alive > 0)
         for i in range(MAX_UNITS_PER_SIDE)], dtype=np.bool_)

    for ui, uid in enumerate(candidate_units):
        unit = friendly_units[uid]
        max_wr = max(
            (w.range_inches for w in unit.unit.weapons if not w.melee),
            default=0.0,
        )
        unit_features = model._extract_unit_features(
            units.squeeze(0), uid).detach()
        uf_b = unit_features.unsqueeze(0)

        # Extract can_charge mask for this unit
        can_charge_mask = extract_can_charge_mask(state_vec, uid)  # (10,) bool

        h_uf = torch.cat([h_b, uf_b], dim=-1)
        move_logits = model.move_type_head(h_uf).squeeze(0)
        # Mask charge when no enemy is in charge range
        if not can_charge_mask.any():
            move_logits = move_logits.clone()
            move_logits[MOVE_CHARGE] = float('-inf')
        move_probs = torch.softmax(move_logits, dim=-1)

        # Precompute destination candidates (unified rush budget)
        _dest_cache_move: tuple | None = None
        if move_probs[MOVE_MOVE].item() > 0:
            cands, cmask, adv_reach = compute_destination_candidates(
                unit, board, enemy_pos_set, player)
            budget = float(unit.unit.rush_distance)
            _n_f = len(friendly_units)
            _n_e = len(enemy_units)
            _fr_m = (np.array(friendly_ranged_matchups, dtype=np.float32)
                     if friendly_ranged_matchups is not None
                     else np.zeros((_n_f, MAX_UNITS_PER_SIDE, 7), dtype=np.float32))
            _er_m = (np.array(enemy_ranged_matchups, dtype=np.float32)
                     if enemy_ranged_matchups is not None
                     else np.zeros((_n_e, MAX_UNITS_PER_SIDE, 7), dtype=np.float32))
            _mm_m = (np.array(friendly_melee_matchups, dtype=np.float32).reshape(_n_f, -1)[:, :MAX_UNITS_PER_SIDE]
                     if friendly_melee_matchups is not None
                     else np.zeros((_n_e, MAX_UNITS_PER_SIDE), dtype=np.float32))
            dest_feats_np = compute_destination_features(
                cands, cmask, unit, uid, player,
                enemy_units, eam_np, _fr_m, _er_m, _mm_m, budget,
                advance_reachable=adv_reach)
            dest_features_t = torch.from_numpy(dest_feats_np).unsqueeze(0)
            dest_mask_t = torch.from_numpy(cmask.astype(np.bool_)).unsqueeze(0)

            mt_onehot = F.one_hot(
                torch.tensor(MOVE_MOVE), NUM_MOVE_TYPES
            ).float().unsqueeze(0)
            h_uf_m_cand = torch.cat([h_b, uf_b, mt_onehot], dim=-1)
            dest_logits = model.compute_dest_logits(
                h_uf_m_cand, dest_features_t, dest_mask_t).squeeze(0)
            _dest_cache_move = (cands, cmask, adv_reach, dest_logits)

        # Argmax unit: 1 argmax action + C-1 sampled actions; others: C sampled
        n_samples = C

        for si in range(n_samples):
            # First sample of argmax unit is deterministic; all others are sampled
            is_argmax_action = (ui == 0 and si == 0)

            # Resolve per-slot sampling mode and temperature.
            #   "argmax"  — joint argmax (ui=0, si=0).
            #   "policy"  — multinomial over softmax(logits)        (τ=1).
            #   "temp"    — multinomial over softmax(logits / tau)  (τ=tau).
            #   "uniform" — multinomial over uniform-over-legal.
            # Legacy path: n_policy_samples < 0 keeps every non-argmax slot
            # driven by the uniform_alt flag, identical to pre-MPO behaviour.
            # Mix path: n_policy_samples >= 0 routes slots in order
            #   argmax(1) → policy(N_POLICY) → temp(N_TEMP) → uniform(rest).
            if is_argmax_action:
                sample_mode = "argmax"
                tau_slot = 1.0
            elif n_policy_samples < 0:
                sample_mode = "uniform" if uniform_alt else "policy"
                tau_slot = 1.0
            else:
                _na_idx = ui * C + si - 1  # 0-indexed slot among non-argmax
                if _na_idx < n_policy_samples:
                    sample_mode = "policy"
                    tau_slot = 1.0
                elif _na_idx < n_policy_samples + n_temp_samples:
                    sample_mode = "temp"
                    tau_slot = tau
                else:
                    sample_mode = "uniform"
                    tau_slot = 1.0

            if sample_mode == "argmax":
                move_type = int(move_probs.argmax().item())
            elif sample_mode == "uniform":
                _legal = torch.isfinite(move_logits).float()
                _u = _legal / _legal.sum().clamp(min=1)
                move_type = int(torch.multinomial(_u, 1).item())
            else:  # "policy" (τ=1) or "temp" (τ=tau)
                move_probs_slot = (
                    move_probs if tau_slot == 1.0
                    else torch.softmax(move_logits / tau_slot, dim=-1))
                move_type = int(torch.multinomial(move_probs_slot, 1).item())

            move_onehot = F.one_hot(
                torch.tensor(move_type), NUM_MOVE_TYPES
            ).float().unsqueeze(0)
            h_uf_m = torch.cat([h_b, uf_b, move_onehot], dim=-1)

            # Destination pointer
            unit_cx, unit_cy = friendly_positions[uid]
            dest_col, dest_row = int(round(unit_cx)), int(round(unit_cy))
            _pick_ar = True  # advance-reachable for selected dest
            pick_idx = -1    # dest candidate index (MOVE_MOVE only)

            if move_type == MOVE_MOVE and _dest_cache_move is not None:
                cands, cmask, adv_reach, dest_logits = _dest_cache_move
                n_valid = int(cmask.sum())
                if n_valid > 0:
                    if sample_mode == "argmax":
                        pick_idx = int(dest_logits.argmax().item())
                    elif sample_mode == "uniform":
                        _legal = torch.isfinite(dest_logits).float()
                        _u = _legal / _legal.sum().clamp(min=1)
                        pick_idx = int(torch.multinomial(_u, 1).item())
                    elif n_policy_samples < 0:
                        # Legacy policy branch: deterministic topk cycle.
                        # Preserved bit-exactly when not in mix mode.
                        dest_probs = torch.softmax(dest_logits, dim=-1)
                        _, top_dest = torch.topk(dest_probs, min(C, n_valid))
                        pick_idx = top_dest[si % len(top_dest)].item()
                    else:
                        # Mix mode: true multinomial with per-slot τ.
                        dest_probs_slot = torch.softmax(
                            dest_logits / tau_slot, dim=-1)
                        pick_idx = int(torch.multinomial(dest_probs_slot, 1).item())
                    dest_col = int(cands[pick_idx, 0])
                    dest_row = int(cands[pick_idx, 1])
                    _pick_ar = bool(adv_reach[pick_idx])

                post_x, post_y = float(dest_col), float(dest_row)
                if player == "B":
                    post_x = _flip_x(post_x)
                    post_y = _flip_y(post_y)
            else:
                post_x, post_y = unit_cx, unit_cy

            # Charge target (pointer head)
            charge_logits = model.compute_charge_logits(
                h_b.squeeze(0), units.squeeze(0), uid,
                enemy_alive_mask, can_charge_mask,
            )
            no_chargeable = no_enemies or not (enemy_alive_mask & can_charge_mask).any()
            if no_chargeable:
                charge_target_idx = 0
            elif sample_mode == "argmax":
                charge_target_idx = int(charge_logits.argmax().item())
            elif sample_mode == "uniform":
                _legal = (enemy_alive_mask & can_charge_mask).float()
                _u = _legal / _legal.sum().clamp(min=1)
                charge_target_idx = int(torch.multinomial(_u, 1).item())
            else:  # "policy" (τ=1) or "temp" (τ=tau)
                charge_probs_slot = torch.softmax(
                    charge_logits / tau_slot, dim=-1)
                charge_target_idx = int(
                    torch.multinomial(charge_probs_slot, 1).item())

            # Shoot target (pointer head)
            post_move_rel = compute_post_move_rel(
                post_x, post_y, enemy_positions)
            shoot_range_mask = compute_in_range_mask(
                post_move_rel, float(max_wr), enemy_alive_mask)
            shoot_logits = model.compute_shoot_logits(
                h_b.squeeze(0), units.squeeze(0), uid,
                post_move_rel, enemy_alive_mask,
                shoot_range_mask=shoot_range_mask,
            )
            no_shootable = no_enemies or not shoot_range_mask.any()
            if no_shootable:
                shoot_target_idx = 0
            elif sample_mode == "argmax":
                shoot_target_idx = int(shoot_logits.argmax().item())
            elif sample_mode == "uniform":
                _legal = (enemy_alive_mask & shoot_range_mask).float()
                _u = _legal / _legal.sum().clamp(min=1)
                shoot_target_idx = int(torch.multinomial(_u, 1).item())
            else:  # "policy" (τ=1) or "temp" (τ=tau)
                shoot_probs_slot = torch.softmax(
                    shoot_logits / tau_slot, dim=-1)
                shoot_target_idx = int(
                    torch.multinomial(shoot_probs_slot, 1).item())

            target_ranking = torch.argsort(
                shoot_logits, descending=True).tolist()

            # Resolve to game-space action
            dest = None
            if move_type == MOVE_MOVE:
                dest = (dest_col, dest_row)

            action_str, goal, charge_target_unit, reason = \
                execute_decoded_decision(
                    unit, enemy_units, move_type, dest,
                    charge_target_idx, shoot_target_idx,
                    is_advance_reachable=_pick_ar,
                )

            ct_idx = -1
            if charge_target_unit is not None:
                for j, eu in enumerate(enemy_units):
                    if eu is charge_target_unit:
                        ct_idx = j
                        break

            dedup_key = (uid, move_type, action_str, goal, ct_idx)
            if dedup_key in _seen_keys:
                continue
            _seen_keys.add(dedup_key)

            will_not_shoot = (
                no_shootable
                or (move_type == MOVE_MOVE and not _pick_ar)
                or move_type == MOVE_CHARGE
            )
            candidate_actions.append((
                uid, move_type, dest_col, dest_row, target_ranking,
                charge_target_idx, shoot_target_idx,
                action_str, goal, ct_idx, reason, will_not_shoot,
                pick_idx,
            ))
            candidate_to_unit.append(uid)
            candidate_shoot_masks.append(shoot_range_mask)

    if not candidate_actions:
        # Fallback: impossible edge case — signal "no planning" so the caller
        # falls back to the normal inference path on the next request.
        return None

    # --- Evaluate candidates via rollouts (single-threaded) ---
    friendly_is_a = (player == "A")

    # Determine units_a/units_b ordering
    if friendly_is_a:
        units_a_list = list(friendly_units)
        units_b_list = list(enemy_units)
    else:
        units_a_list = list(enemy_units)
        units_b_list = list(friendly_units)

    # Training-time planning is always in-process (shared-memory worker pool),
    # so we keep live references and clone per rollout instead of pickling the
    # entire state tree. state_bytes remains empty — the live_state kwarg gates
    # the fast path inside _run_chunk_batched_raw.
    live_state = (
        units_a_list, units_b_list, board,
        fr_a, fm_a, fr_b, fm_b, pts_a, pts_b,
    )
    state_bytes = b""

    use_sh = (sh_enabled
              and len(sh_schedule) >= 1
              and sum(sh_schedule) == M
              and len(candidate_actions) > 1)

    if use_sh:
        # Sequential halving with dummy-candidate padding to a fixed pool size.
        # Dummies have -inf value and sort to the bottom, absorbing early cuts
        # when the real candidate count is small. Argmax (candidate 0) is
        # protected from elimination so the planning_improved comparison always
        # has a full-M estimate on both sides.
        PAD = 9
        n_real = len(candidate_actions)
        is_real = [ci < n_real for ci in range(PAD)]
        per_cand_values: list[list[float]] = [[] for _ in range(PAD)]
        alive: list[int] = list(range(PAD))
        argmax_ci = 0

        def _mean_for(ci: int) -> float:
            if not is_real[ci]:
                return float('-inf')
            vs = per_cand_values[ci]
            return (sum(vs) / len(vs)) if vs else float('-inf')

        for phase_i, phase_m in enumerate(sh_schedule):
            real_alive = [ci for ci in alive if is_real[ci]]
            if real_alive and phase_m > 0:
                real_actions = [candidate_actions[ci] for ci in real_alive]
                raw = _run_chunk_batched_raw((
                    state_bytes, real_actions, phase_m, N, round_num, mode, player,
                    friendly_is_a, current_is_a,
                ), model_override=model, live_state=live_state)
                for local_i, ci in enumerate(real_alive):
                    per_cand_values[ci].extend(raw[local_i])

            is_last_phase = (phase_i == len(sh_schedule) - 1)
            if not is_last_phase:
                n_alive = len(alive)
                n_drop = (n_alive + 1) // 2  # ceil(n_alive / 2)
                ranked = sorted(alive, key=_mean_for, reverse=True)
                survivors = ranked[:n_alive - n_drop]
                # Argmax immunity: swap it in for the worst survivor if excluded
                if argmax_ci in alive and argmax_ci not in survivors:
                    survivors = survivors[:-1] + [argmax_ci]
                alive = survivors

        avg_values = [
            (sum(per_cand_values[ci]) / len(per_cand_values[ci]))
            if per_cand_values[ci] else 0.0
            for ci in range(n_real)
        ]
    else:
        raw_values = _run_chunk_batched_raw((
            state_bytes, candidate_actions, M, N, round_num, mode, player,
            friendly_is_a, current_is_a,
        ), model_override=model, live_state=live_state)
        avg_values = [
            (sum(vs) / len(vs)) if vs else 0.0 for vs in raw_values
        ]

    # --- Identify argmax candidate(s) and compute per-unit average values ---
    # The first candidate(s) belong to the argmax unit (could be just 1)
    argmax_cand_idx = 0  # candidate 0 is always the argmax action
    argmax_value = avg_values[0]

    # Per-unit average values for distillation target
    unit_value_sums: dict[int, float] = {}
    unit_value_counts: dict[int, int] = {}
    for ci, (uid_ci, val_ci) in enumerate(
            zip(candidate_to_unit, avg_values)):
        unit_value_sums[uid_ci] = unit_value_sums.get(uid_ci, 0.0) + val_ci
        unit_value_counts[uid_ci] = unit_value_counts.get(uid_ci, 0) + 1

    planning_unit_indices = list(unit_value_sums.keys())
    planning_unit_values = [
        unit_value_sums[u] / unit_value_counts[u]
        for u in planning_unit_indices
    ]

    # Per-(unit, move_type) and per-(unit, target) values for sub-head distillation
    move_value_sums: dict[tuple[int, int], float] = {}
    move_value_counts: dict[tuple[int, int], int] = {}
    charge_value_sums: dict[tuple[int, int], float] = {}
    charge_value_counts: dict[tuple[int, int], int] = {}
    shoot_value_sums: dict[tuple[int, int], float] = {}
    shoot_value_counts: dict[tuple[int, int], int] = {}
    dest_value_sums: dict[tuple[int, int], float] = {}
    dest_value_counts: dict[tuple[int, int], int] = {}
    for ci, (uid_ci, val_ci) in enumerate(
            zip(candidate_to_unit, avg_values)):
        ca = candidate_actions[ci]
        mt = ca[1]  # move_type
        key_mt = (uid_ci, mt)
        move_value_sums[key_mt] = move_value_sums.get(key_mt, 0.0) + val_ci
        move_value_counts[key_mt] = move_value_counts.get(key_mt, 0) + 1
        if mt == MOVE_CHARGE:
            ct = ca[5]  # charge_target_idx
            if ct >= 0:
                key_ct = (uid_ci, ct)
                charge_value_sums[key_ct] = charge_value_sums.get(key_ct, 0.0) + val_ci
                charge_value_counts[key_ct] = charge_value_counts.get(key_ct, 0) + 1
        if mt == MOVE_MOVE and not ca[11]:  # will_not_shoot == False means can shoot
            st = ca[6]  # shoot_target_idx
            if st >= 0:
                key_st = (uid_ci, st)
                shoot_value_sums[key_st] = shoot_value_sums.get(key_st, 0.0) + val_ci
                shoot_value_counts[key_st] = shoot_value_counts.get(key_st, 0) + 1
        if mt == MOVE_MOVE:
            dci = ca[12]  # dest candidate idx
            if dci >= 0:
                key_dc = (uid_ci, dci)
                dest_value_sums[key_dc] = dest_value_sums.get(key_dc, 0.0) + val_ci
                dest_value_counts[key_dc] = dest_value_counts.get(key_dc, 0) + 1

    # --- Selection: compare argmax vs best non-argmax ---
    best_search_idx = argmax_cand_idx
    best_search_value = argmax_value
    for ci in range(len(avg_values)):
        if ci != argmax_cand_idx and avg_values[ci] > best_search_value:
            best_search_idx = ci
            best_search_value = avg_values[ci]

    planning_improved = (best_search_idx != argmax_cand_idx
                         and best_search_value > argmax_value)
    planning_value_delta = max(best_search_value - argmax_value, 0.0)

    if planning_improved:
        chosen_idx = best_search_idx
    else:
        chosen_idx = argmax_cand_idx

    chosen = candidate_actions[chosen_idx]
    chosen_uid = chosen[0]
    chosen_move_type = chosen[1]
    chosen_dest_col = chosen[2]
    chosen_dest_row = chosen[3]
    chosen_ranking = chosen[4]
    chosen_charge_tgt = chosen[5]
    chosen_shoot_tgt = chosen[6]

    # Extract sub-head distillation targets for chosen unit
    planning_move_indices = []
    planning_move_values = []
    for mt in range(NUM_MOVE_TYPES):
        key = (chosen_uid, mt)
        if key in move_value_counts:
            planning_move_indices.append(mt)
            planning_move_values.append(
                move_value_sums[key] / move_value_counts[key])

    planning_charge_indices = []
    planning_charge_values = []
    for key, cnt in charge_value_counts.items():
        if key[0] == chosen_uid:
            planning_charge_indices.append(key[1])
            planning_charge_values.append(charge_value_sums[key] / cnt)

    # Shoot distillation: only include targets reachable from ALL candidate
    # positions for the chosen unit (pairwise validity).  This avoids
    # distilling toward targets that aren't reachable from some positions.
    planning_shoot_indices = []
    planning_shoot_values = []
    # Collect shoot masks for all MOVE_MOVE candidates that can shoot
    _chosen_shoot_masks = []
    for ci, uid_ci in enumerate(candidate_to_unit):
        if uid_ci == chosen_uid:
            ca = candidate_actions[ci]
            if ca[1] == MOVE_MOVE and not ca[11]:  # can shoot (will_not_shoot == False)
                _chosen_shoot_masks.append(candidate_shoot_masks[ci])
    if _chosen_shoot_masks:
        # Intersection: target must be in range from every candidate position
        common_mask = _chosen_shoot_masks[0]
        for m in _chosen_shoot_masks[1:]:
            common_mask = common_mask & m
        for key, cnt in shoot_value_counts.items():
            if key[0] == chosen_uid:
                tgt_idx = key[1]
                if common_mask[tgt_idx]:
                    planning_shoot_indices.append(tgt_idx)
                    planning_shoot_values.append(
                        shoot_value_sums[key] / cnt)

    # Dest distillation: per-candidate average values for the chosen unit.
    # Only meaningful when the chosen move type is MOVE_MOVE (the dest head
    # is the destination pointer conditioned on move_type=move).
    planning_dest_indices: list[int] = []
    planning_dest_values: list[float] = []
    if chosen_move_type == MOVE_MOVE:
        for key, cnt in dest_value_counts.items():
            if key[0] == chosen_uid:
                planning_dest_indices.append(key[1])
                planning_dest_values.append(
                    dest_value_sums[key] / cnt)

    # --- Compute old_log_prob: π_policy(a_chosen | s) ---
    # Re-use trunk output already computed; run conditioned heads for the
    # chosen action to get the policy's sampling log-prob.
    chosen_unit_lp = torch.log(unit_probs[chosen_uid] + eps).item()

    unit_features = model._extract_unit_features(
        units.squeeze(0), chosen_uid).detach()
    uf_b = unit_features.unsqueeze(0)

    # Extract can_charge mask for the chosen unit
    can_charge_mask = extract_can_charge_mask(state_vec, chosen_uid)  # (10,) bool

    h_uf = torch.cat([h_b, uf_b], dim=-1)
    move_logits_ch = model.move_type_head(h_uf).squeeze(0)
    # Mask charge when no enemy is in charge range
    if not can_charge_mask.any():
        move_logits_ch = move_logits_ch.clone()
        move_logits_ch[MOVE_CHARGE] = float('-inf')
    move_probs_ch = torch.softmax(move_logits_ch, dim=-1)
    move_lp = torch.log(move_probs_ch[chosen_move_type] + eps).item()

    move_onehot_ch = F.one_hot(
        torch.tensor(chosen_move_type), NUM_MOVE_TYPES
    ).float().unsqueeze(0)
    h_uf_m_ch = torch.cat([h_b, uf_b, move_onehot_ch], dim=-1)

    # Destination pointer log-prob (categorical over candidate hexes)
    dest_lp = 0.0
    _chosen_ar = True  # advance-reachable for chosen dest
    cands_ch = None
    adv_reach_ch = None
    chosen_cand_idx = -1
    if chosen_move_type == MOVE_MOVE:
        # Recompute candidates for the chosen unit
        chosen_unit = friendly_units[chosen_uid]
        cands_ch, cmask_ch, adv_reach_ch = compute_destination_candidates(
            chosen_unit, board, enemy_pos_set, player)
        budget_ch = float(chosen_unit.unit.rush_distance)
        _n_f = len(friendly_units)
        _n_e = len(enemy_units)
        _fr_m = (np.array(friendly_ranged_matchups, dtype=np.float32)
                 if friendly_ranged_matchups is not None
                 else np.zeros((_n_f, MAX_UNITS_PER_SIDE, 7), dtype=np.float32))
        _er_m = (np.array(enemy_ranged_matchups, dtype=np.float32)
                 if enemy_ranged_matchups is not None
                 else np.zeros((_n_e, MAX_UNITS_PER_SIDE, 7), dtype=np.float32))
        _mm_m = (np.array(friendly_melee_matchups, dtype=np.float32).reshape(_n_f, -1)[:, :MAX_UNITS_PER_SIDE]
                 if friendly_melee_matchups is not None
                 else np.zeros((_n_e, MAX_UNITS_PER_SIDE), dtype=np.float32))
        dest_feats_ch = compute_destination_features(
            cands_ch, cmask_ch, chosen_unit, chosen_uid, player,
            enemy_units, eam_np, _fr_m, _er_m, _mm_m, budget_ch,
            advance_reachable=adv_reach_ch)
        dest_features_ch_t = torch.from_numpy(dest_feats_ch).unsqueeze(0)
        dest_mask_ch_t = torch.from_numpy(cmask_ch.astype(np.bool_)).unsqueeze(0)
        dest_logits_ch = model.compute_dest_logits(
            h_uf_m_ch, dest_features_ch_t, dest_mask_ch_t).squeeze(0)
        dest_log_probs = torch.log_softmax(dest_logits_ch, dim=-1)
        # Find which candidate index matches chosen_dest_col, chosen_dest_row
        chosen_cand_idx = 0  # default to centroid
        for ci_ch in range(int(cmask_ch.sum())):
            if (int(cands_ch[ci_ch, 0]) == chosen_dest_col
                    and int(cands_ch[ci_ch, 1]) == chosen_dest_row):
                chosen_cand_idx = ci_ch
                break
        _chosen_ar = bool(adv_reach_ch[chosen_cand_idx])
        dest_lp = float(dest_log_probs[chosen_cand_idx].item())
        dest_lp = max(-20.0, min(20.0, dest_lp))

    # Charge log-prob (pointer head)
    charge_logits_ch = model.compute_charge_logits(
        h_b.squeeze(0), units.squeeze(0), chosen_uid,
        enemy_alive_mask, can_charge_mask,
    )
    no_chargeable = no_enemies or not (enemy_alive_mask & can_charge_mask).any()
    if no_chargeable:
        charge_lp = 0.0
    else:
        charge_probs_ch = torch.softmax(charge_logits_ch, dim=-1)
        charge_lp = torch.log(
            charge_probs_ch[chosen_charge_tgt] + eps).item()

    # Shoot log-prob (need post-move position for chosen action)
    unit_cx, unit_cy = friendly_positions[chosen_uid]
    if chosen_move_type == MOVE_MOVE:
        ch_px, ch_py = float(chosen_dest_col), float(chosen_dest_row)
        if player == "B":
            ch_px = _flip_x(ch_px)
            ch_py = _flip_y(ch_py)
    else:
        ch_px, ch_py = unit_cx, unit_cy

    ch_pmr = compute_post_move_rel(ch_px, ch_py, enemy_positions)
    ch_max_wr = max(
        (w.range_inches for w in friendly_units[chosen_uid].unit.weapons
         if not w.melee), default=0.0)
    ch_shoot_mask = compute_in_range_mask(
        ch_pmr, float(ch_max_wr), enemy_alive_mask)
    ch_shoot_logits = model.compute_shoot_logits(
        h_b.squeeze(0), units.squeeze(0), chosen_uid,
        ch_pmr, enemy_alive_mask, shoot_range_mask=ch_shoot_mask,
    )
    ch_no_shootable = no_enemies or not ch_shoot_mask.any()
    if ch_no_shootable:
        shoot_lp = 0.0
    else:
        shoot_probs_ch = torch.softmax(ch_shoot_logits, dim=-1)
        shoot_lp = torch.log(
            shoot_probs_ch[chosen_shoot_tgt] + eps).item()

    # Combine log-probs (same rules as sample_tactical_actions_no_grad)
    old_log_prob = chosen_unit_lp + move_lp
    if chosen_move_type == MOVE_MOVE:
        old_log_prob += dest_lp
    if chosen_move_type == MOVE_MOVE and _chosen_ar:
        old_log_prob += shoot_lp
    if chosen_move_type == MOVE_CHARGE:
        old_log_prob += charge_lp

    # Value estimate (from policy, not from planning rollouts)
    value_est = model.value_head(
        h, round_onehot,
        model._get_opp_embed(h, opponent_type),
    ).squeeze(0).item()

    shoot_mask_list = ch_shoot_mask.tolist()
    pmr_list = ch_pmr.tolist()

    # Dest candidate data for replay (only populated when MOVE_MOVE).
    # Unpadded: compute_destination_candidates returns tight (n_valid, *) arrays.
    if chosen_move_type == MOVE_MOVE and cands_ch is not None:
        dest_candidates_out = cands_ch
        dest_advance_reachable_out = adv_reach_ch.tolist()
        dest_selected_idx_out = chosen_cand_idx
    else:
        dest_candidates_out = None
        dest_advance_reachable_out = None
        dest_selected_idx_out = -1

    return (
        chosen_uid, chosen_move_type, chosen_dest_col, chosen_dest_row,
        dest_selected_idx_out,  # dest_cand_idx (real index into candidates)
        chosen_charge_tgt, chosen_shoot_tgt, chosen_ranking,
        pmr_list, old_log_prob, value_est, shoot_mask_list,
        True,  # was_planned
        planning_improved,
        planning_value_delta,
        planning_unit_values,
        planning_unit_indices,
        planning_move_values or None,
        planning_move_indices or None,
        planning_charge_values or None,
        planning_charge_indices or None,
        planning_shoot_values or None,
        planning_shoot_indices or None,
        planning_dest_values or None,
        planning_dest_indices or None,
        dest_candidates_out,
        dest_advance_reachable_out,
    )
