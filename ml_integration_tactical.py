"""ML tactical integration v2: wire TacticalModel outputs to game state per activation.

Free-movement model: the model picks a movement type (hold/advance/rush/charge),
then a direction+distance (for advance/rush) or charge target (for charge),
then a shooting target (for hold/advance).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from board import Board, COLS, ROWS, OBJECTIVES, OBJ_SEIZE_RANGE, dist, dist_sq
from combat import evaluate_target
from models import UnitState
from ml_features import (
    MAX_UNITS_PER_SIDE,
    TACTICAL_UNIT_FEATURES,
    BOARD_DIAG,
    encode_state_tactical,
    precompute_damage,
    _flip_x,
    _flip_y,
)

from ml_model_tactical import (
    TacticalModel, TacticalModelOutput,
    NUM_MOVE_TYPES, MOVE_HOLD, MOVE_ADVANCE, MOVE_RUSH, MOVE_CHARGE,
    POST_MOVE_REL_FEATURES,
)

from ml_features import BOARD_DIAG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOVE_TYPE_NAMES = ["hold", "advance", "rush", "charge"]

_INV_BOARD_DIAG = 1.0 / BOARD_DIAG

# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------

_timing_encode_s: float = 0.0
_timing_forward_s: float = 0.0
_timing_calls: int = 0


def reset_timing() -> None:
    global _timing_encode_s, _timing_forward_s, _timing_calls
    _timing_encode_s = 0.0
    _timing_forward_s = 0.0
    _timing_calls = 0


def get_timing() -> dict:
    return {
        'encode_s': _timing_encode_s,
        'forward_s': _timing_forward_s,
        'calls': _timing_calls,
    }


# ---------------------------------------------------------------------------
# Coroutine-mode dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InferenceRequest:
    """Yielded by simulate_game when it needs an ML decision."""
    state_vec: torch.Tensor        # (2811,) pre-encoded
    alive_mask: torch.Tensor       # (10,) bool
    enemy_alive_mask: torch.Tensor # (10,) bool
    player: str                    # "A" or "B"
    friendly_positions: list[tuple[float, float]]  # model-space, 10 slots
    enemy_positions: list[tuple[float, float]]     # model-space, 10 slots
    advance_distances: list[float]                 # per friendly slot
    rush_distances: list[float]                    # per friendly slot
    max_weapon_ranges: list[float]                 # max ranged weapon range per friendly slot


@dataclass
class InferenceResult:
    """Fully-decoded decision sent back to simulate_game by the coordinator."""
    unit_idx: int                       # selected friendly slot
    move_type: int                      # 0=hold, 1=advance, 2=rush, 3=charge
    direction_angle: float              # radians (argmax from direction head)
    distance_frac: float                # 0-1 (mean of Beta from distance head)
    charge_target_idx: int              # enemy slot
    shoot_target_idx: int               # enemy slot
    target_ranking: list[int]           # shoot target ranking (descending)
    value: float


# ---------------------------------------------------------------------------
# Post-move relative features
# ---------------------------------------------------------------------------

def compute_post_move_rel(
    post_x: float,
    post_y: float,
    enemy_positions: list[tuple[float, float]],
) -> torch.Tensor:
    """Compute (sin θ, cos θ, dist) from post-move position to each of 10 enemy slots.

    Returns (30,) tensor.
    """
    feats = torch.zeros(POST_MOVE_REL_FEATURES)
    for i, (ex, ey) in enumerate(enemy_positions):
        dx = ex - post_x
        dy = ey - post_y
        d = math.sqrt(dx * dx + dy * dy)
        base = i * 3
        if d < 1e-6:
            feats[base] = 0.0
            feats[base + 1] = 0.0
        else:
            feats[base] = dy / d       # sin θ
            feats[base + 1] = dx / d   # cos θ
        feats[base + 2] = d * _INV_BOARD_DIAG
    return feats


# ---------------------------------------------------------------------------
# Weapon range helpers (for in-range masking)
# ---------------------------------------------------------------------------

def _get_max_weapon_ranges(units: list[UnitState]) -> list[float]:
    """Get max ranged weapon range for each of up to 10 friendly unit slots.

    Dead/missing slots get 0.0.
    """
    ranges: list[float] = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(units) and units[i].models_alive > 0:
            max_r = max(
                (w.range_inches for w in units[i].unit.weapons if not w.melee),
                default=0.0,
            )
            ranges.append(float(max_r))
        else:
            ranges.append(0.0)
    return ranges


def compute_in_range_mask(
    post_move_rel: torch.Tensor,
    max_weapon_range: float,
    enemy_alive_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute a bool mask of enemies within weapon range from post-move position.

    Uses centre-to-centre distance from post_move_rel features.

    Parameters
    ----------
    post_move_rel : (30,) — (sin θ, cos θ, normalised_dist) × 10 enemies
    max_weapon_range : max ranged weapon range of the selected unit (in cells)
    enemy_alive_mask : (10,) bool

    Returns
    -------
    (10,) bool — True for enemies that are alive AND in range.
    """
    # Extract normalised distances (every 3rd element starting at index 2)
    distances = post_move_rel[2::3] * BOARD_DIAG  # (10,) actual distance in cells
    in_range = distances <= max_weapon_range
    return in_range & enemy_alive_mask


def compute_in_range_mask_batched(
    post_move_rel_batch: torch.Tensor,
    max_weapon_ranges: torch.Tensor,
    enemy_alive_batch: torch.Tensor,
) -> torch.Tensor:
    """Batched version of compute_in_range_mask.

    Parameters
    ----------
    post_move_rel_batch : (N, 30)
    max_weapon_ranges : (N,) — max weapon range per sample
    enemy_alive_batch : (N, 10) bool

    Returns
    -------
    (N, 10) bool
    """
    distances = post_move_rel_batch[:, 2::3] * BOARD_DIAG  # (N, 10)
    in_range = distances <= max_weapon_ranges.unsqueeze(-1)
    return in_range & enemy_alive_batch


# ---------------------------------------------------------------------------
# Direction / distance decoding helpers
# ---------------------------------------------------------------------------

def decode_direction_params(direction_params: torch.Tensor) -> tuple[float, float]:
    """Extract (mean_sin, mean_cos) → angle from direction head output.

    Returns (angle_radians, concentration).
    """
    raw_sin = direction_params[0].item()
    raw_cos = direction_params[1].item()
    log_conc = direction_params[2].item()

    # Normalise to unit circle
    norm = math.sqrt(raw_sin * raw_sin + raw_cos * raw_cos)
    if norm < 1e-6:
        angle = 0.0
    else:
        angle = math.atan2(raw_sin / norm, raw_cos / norm)

    concentration = min(torch.nn.functional.softplus(torch.tensor(log_conc)).item() + 0.1, 80.0)
    return angle, concentration


def decode_distance_params(distance_params: torch.Tensor) -> tuple[float, float, float]:
    """Extract Beta distribution parameters and mean from distance head output.

    Returns (alpha, beta, mean_frac).
    """
    alpha = min(F.softplus(distance_params[0]).item() + 1.01, 100.0)
    beta = min(F.softplus(distance_params[1]).item() + 1.01, 100.0)
    mean_frac = alpha / (alpha + beta)
    return alpha, beta, mean_frac


def compute_post_move_position(
    cx: float, cy: float,
    angle: float, distance: float,
) -> tuple[float, float]:
    """Compute destination from current position, angle, and distance.

    Clamps to board boundaries.
    """
    dx = math.cos(angle) * distance
    dy = math.sin(angle) * distance
    nx = max(0.0, min(float(COLS - 1), cx + dx))
    ny = max(0.0, min(float(ROWS - 1), cy + dy))
    return nx, ny


# ---------------------------------------------------------------------------
# Model-space position helpers
# ---------------------------------------------------------------------------

def _get_model_space_positions(
    units: list[UnitState], player: str,
) -> list[tuple[float, float]]:
    """Get model-space centre positions for up to 10 unit slots.

    Dead/missing slots get board centre sentinel.
    """
    sentinel = ((COLS - 1) / 2.0, (ROWS - 1) / 2.0)
    positions: list[tuple[float, float]] = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(units) and units[i].models_alive > 0:
            cx, cy = units[i].centre()
            if player == "B":
                cx = _flip_x(cx)
                cy = _flip_y(cy)
            positions.append((cx, cy))
        else:
            positions.append(sentinel)
    return positions


def _get_movement_budgets(
    units: list[UnitState],
) -> tuple[list[float], list[float]]:
    """Get advance and rush distances for up to 10 unit slots."""
    advance = []
    rush = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(units) and units[i].models_alive > 0:
            advance.append(float(units[i].unit.advance_distance))
            rush.append(float(units[i].unit.rush_distance))
        else:
            advance.append(0.0)
            rush.append(0.0)
    return advance, rush


# ---------------------------------------------------------------------------
# Decode + Execute (replaces old execute_ml_decision)
# ---------------------------------------------------------------------------

def _target_position(target: UnitState) -> tuple[int, int]:
    """Integer centre position of a target unit."""
    cx, cy = target.centre()
    return (int(round(cx)), int(round(cy)))


def execute_decoded_decision(
    unit: UnitState,
    enemies: list[UnitState],
    move_type: int,
    dest: tuple[float, float] | None,
    charge_target_idx: int,
    shoot_target_idx: int,
) -> tuple[str, tuple[int, int] | None, UnitState | None, str]:
    """Translate decoded model decision into (action, goal_position, charge_target, reason).

    Parameters
    ----------
    unit : the activated unit
    enemies : enemy unit list
    move_type : 0=hold, 1=advance, 2=rush, 3=charge
    dest : (x, y) destination for advance/rush; None for hold/charge
    charge_target_idx : enemy slot for charge
    shoot_target_idx : enemy slot for shooting (hold/advance)
    """
    # Artillery: must hold
    if unit.unit.artillery:
        return ("hold", None, None, "artillery holds position")

    alive_enemies = [e for e in enemies if e.models_alive > 0]
    if not alive_enemies:
        return ("hold", None, None, "no enemies alive")

    if move_type == MOVE_HOLD:
        return ("hold", None, None, "model chose hold")

    if move_type == MOVE_CHARGE:
        if charge_target_idx < len(enemies) and enemies[charge_target_idx].models_alive > 0:
            target = enemies[charge_target_idx]
            tpos = _target_position(target)
            # Validate charge range (centre-to-centre)
            cx, cy = unit.centre()
            tx, ty = target.centre()
            threshold = unit.unit.rush_distance + 2
            if (cx - tx) ** 2 + (cy - ty) ** 2 < threshold * threshold:
                return ("charge", tpos, target, "model chose charge")
            # Out of charge range — rush toward target instead
            return ("rush", tpos, None, "charge target out of range, rushing toward it")
        # Fallback: charge failed (target dead), rush toward nearest enemy
        if alive_enemies:
            tpos = _target_position(alive_enemies[0])
            return ("rush", tpos, None, "charge target dead, rushing nearest")
        return ("hold", None, None, "charge target dead, no enemies")

    if move_type == MOVE_ADVANCE:
        if dest is not None:
            goal = (int(round(dest[0])), int(round(dest[1])))
            return ("advance", goal, None, "model chose advance")
        return ("hold", None, None, "advance with no destination")

    if move_type == MOVE_RUSH:
        if dest is not None:
            goal = (int(round(dest[0])), int(round(dest[1])))
            return ("rush", goal, None, "model chose rush")
        return ("hold", None, None, "rush with no destination")

    return ("hold", None, None, f"unknown move_type: {move_type}")


# ---------------------------------------------------------------------------
# Coroutine-mode decode (used by _simulate_game_coroutine)
# ---------------------------------------------------------------------------

def decode_tactical_result(
    ir: InferenceResult,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    board: Board,
    player: str,
) -> tuple[UnitState | None, list[int], str, tuple[int, int] | None, UnitState | None, str]:
    """Unpack a fully-decoded InferenceResult into game-level decision.

    The coordinator has already done the full two-pass model decode (via
    batched_argmax_tactical). This function translates the decoded indices
    into game objects and computes the destination in game-space.

    Returns (selected_unit, target_ranking, action, goal, charge_target, reason).
    """
    selected_idx = ir.unit_idx
    if selected_idx >= len(friendly_units) or friendly_units[selected_idx].models_alive <= 0:
        return None, [], "hold", None, None, "selected unit unavailable"
    selected_unit = friendly_units[selected_idx]

    move_type = ir.move_type
    angle = ir.direction_angle
    mean_frac = ir.distance_frac

    # Compute post-move position in model-space → game-space destination
    friendly_positions = _get_model_space_positions(friendly_units, player)
    unit_cx, unit_cy = friendly_positions[selected_idx]
    if move_type == MOVE_ADVANCE:
        budget = float(selected_unit.unit.advance_distance)
        post_x, post_y = compute_post_move_position(unit_cx, unit_cy, angle, mean_frac * budget)
    elif move_type == MOVE_RUSH:
        budget = float(selected_unit.unit.rush_distance)
        post_x, post_y = compute_post_move_position(unit_cx, unit_cy, angle, mean_frac * budget)
    else:
        post_x, post_y = unit_cx, unit_cy

    dest = None
    if move_type in (MOVE_ADVANCE, MOVE_RUSH):
        gx, gy = post_x, post_y
        if player == "B":
            gx = _flip_x(gx)
            gy = _flip_y(gy)
        dest = (gx, gy)

    action, goal, charge_target, reason = execute_decoded_decision(
        selected_unit, enemy_units, move_type, dest, ir.charge_target_idx, ir.shoot_target_idx,
    )

    return selected_unit, ir.target_ranking, action, goal, charge_target, reason


# ---------------------------------------------------------------------------
# Batched argmax tactical inference (for evolution / evaluation)
# ---------------------------------------------------------------------------

def batched_argmax_tactical(
    model: TacticalModel,
    requests: list[InferenceRequest],
) -> list[InferenceResult]:
    """Run batched two-pass argmax inference for multiple concurrent games.

    Mirrors the logic of apply_tactical_model but batched:
    - Pass 1: trunk → unit selection → move type → direction/distance → charge target
    - Per-sample: compute post-move position and post_move_rel features
    - Pass 2 (batched): re-run shoot target head with post-move features

    Returns one InferenceResult per request with fully-decoded decisions.
    """
    n = len(requests)
    if n == 0:
        return []

    n_units = MAX_UNITS_PER_SIDE

    # Stack inputs
    state_batch = torch.stack([r.state_vec for r in requests])              # (N, feat)
    alive_batch = torch.stack([r.alive_mask for r in requests])             # (N, 10)
    enemy_alive_batch = torch.stack([r.enemy_alive_mask for r in requests]) # (N, 10)

    # Trunk
    h = model.trunk(state_batch)                                            # (N, 128)

    # Unit selection (argmax)
    unit_logits = model.unit_selection_head(h)                              # (N, 10)
    unit_logits = unit_logits.masked_fill(~alive_batch, float('-inf'))
    unit_indices = unit_logits.argmax(dim=-1)                               # (N,)

    # Extract per-sample unit features
    friendly_block = state_batch[:, :n_units * TACTICAL_UNIT_FEATURES].reshape(
        n, n_units, TACTICAL_UNIT_FEATURES,
    )
    unit_features = friendly_block.gather(
        1, unit_indices.unsqueeze(1).unsqueeze(2).expand(n, 1, TACTICAL_UNIT_FEATURES),
    ).squeeze(1).detach()                                                   # (N, UF)

    # Move type (argmax)
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf)                                # (N, 4)
    move_indices = move_logits.argmax(dim=-1)                               # (N,)
    move_onehot = F.one_hot(move_indices, NUM_MOVE_TYPES).float()           # (N, 4)

    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)            # (N, 272)

    # Direction + Distance (decode to argmax values)
    direction_raw = model.direction_head(h_uf_m)                            # (N, 3)
    distance_raw = model.distance_head(h_uf_m)                              # (N, 2)

    # Charge target (argmax)
    charge_logits = model.charge_target_head(h_uf_m)                        # (N, 10)
    charge_logits = charge_logits.masked_fill(~enemy_alive_batch, float('-inf'))
    no_enemies = ~enemy_alive_batch.any(dim=-1)                             # (N,)
    charge_logits = charge_logits.masked_fill(no_enemies.unsqueeze(-1), 0.0)
    charge_indices = charge_logits.argmax(dim=-1)                           # (N,)

    # Value
    values = model.value_head(h).squeeze(-1)                                # (N,)

    # Per-sample: decode direction/distance, compute post-move position + post_move_rel
    unit_list = unit_indices.tolist()
    move_list = move_indices.tolist()

    pmr_tensors: list[torch.Tensor] = []
    angles: list[float] = []
    mean_fracs: list[float] = []
    for i in range(n):
        req = requests[i]
        uid = unit_list[i]
        mt = move_list[i]

        # Decode direction
        angle, _conc = decode_direction_params(direction_raw[i])
        angles.append(angle)

        # Decode distance
        _alpha, _beta, mf = decode_distance_params(distance_raw[i])
        mean_fracs.append(mf)

        # Compute post-move position in model-space
        ucx, ucy = req.friendly_positions[uid]
        if mt == MOVE_ADVANCE:
            budget = req.advance_distances[uid]
            px, py = compute_post_move_position(ucx, ucy, angle, mf * budget)
        elif mt == MOVE_RUSH:
            budget = req.rush_distances[uid]
            px, py = compute_post_move_position(ucx, ucy, angle, mf * budget)
        else:
            px, py = ucx, ucy

        pmr = compute_post_move_rel(px, py, req.enemy_positions)
        pmr_tensors.append(pmr)

    # Batched shoot target head with post-move features + in-range mask (pass 2)
    pmr_batch = torch.stack(pmr_tensors)                                    # (N, 30)
    shoot_input = torch.cat([h, unit_features, move_onehot, pmr_batch], dim=-1)
    shoot_logits = model.shoot_target_head(shoot_input)                     # (N, 10)
    # Mask by alive AND in-range
    max_wr_list = [requests[i].max_weapon_ranges[unit_list[i]] for i in range(n)]
    max_wr_t = torch.tensor(max_wr_list, dtype=torch.float32)
    shoot_mask_batch = compute_in_range_mask_batched(pmr_batch, max_wr_t, enemy_alive_batch)
    shoot_logits = shoot_logits.masked_fill(~shoot_mask_batch, float('-inf'))
    no_shootable = ~shoot_mask_batch.any(dim=-1)
    shoot_logits = shoot_logits.masked_fill(no_shootable.unsqueeze(-1), 0.0)
    shoot_indices = shoot_logits.argmax(dim=-1)                             # (N,)

    # Build results
    charge_list = charge_indices.tolist()
    shoot_list = shoot_indices.tolist()
    val_list = values.tolist()

    results: list[InferenceResult] = []
    for i in range(n):
        ranking = (
            list(range(n_units)) if no_shootable[i] else
            torch.argsort(shoot_logits[i], descending=True).tolist()
        )
        results.append(InferenceResult(
            unit_idx=unit_list[i],
            move_type=move_list[i],
            direction_angle=angles[i],
            distance_frac=mean_fracs[i],
            charge_target_idx=charge_list[i],
            shoot_target_idx=shoot_list[i],
            target_ranking=ranking,
            value=val_list[i],
        ))

    return results


# ---------------------------------------------------------------------------
# Main integration entry point (argmax)
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_tactical_model(
    model: TacticalModel,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    *,
    friendly_ranged_matchups=None,
    friendly_melee_matchups=None,
    enemy_ranged_matchups=None,
    enemy_melee_matchups=None,
    total_friendly_points: int | None = None,
    total_enemy_points: int | None = None,
) -> tuple[UnitState | None, list[int], str, tuple[int, int] | None, UnitState | None, str, dict]:
    """Encode state, run tactical model, decode and execute one activation.

    Returns (selected_unit, target_ranking, action, goal, charge_target, reason, assessment).
    """
    global _timing_encode_s, _timing_forward_s, _timing_calls

    # Build alive+unactivated mask
    alive_mask_list = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(friendly_units):
            us = friendly_units[i]
            alive_mask_list.append(us.models_alive > 0 and not us.activated)
        else:
            alive_mask_list.append(False)

    if not any(alive_mask_list):
        return None, [], "hold", None, None, "no units available", {}

    alive_mask = torch.tensor(alive_mask_list, dtype=torch.bool)

    enemy_alive_mask = torch.tensor(
        [(i < len(enemy_units) and enemy_units[i].models_alive > 0)
         for i in range(MAX_UNITS_PER_SIDE)],
        dtype=torch.bool,
    )

    # Encode state
    _t0 = time.perf_counter()
    state_vec = encode_state_tactical(
        friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=friendly_ranged_matchups,
        friendly_melee_matchups=friendly_melee_matchups,
        enemy_ranged_matchups=enemy_ranged_matchups,
        enemy_melee_matchups=enemy_melee_matchups,
        total_friendly_points=total_friendly_points,
        total_enemy_points=total_enemy_points,
    )
    _t1 = time.perf_counter()
    _timing_encode_s += _t1 - _t0

    # Get model-space positions for post-move computation
    friendly_positions = _get_model_space_positions(friendly_units, player)
    enemy_positions = _get_model_space_positions(enemy_units, player)

    # --- Forward pass 1: get unit + move_type + direction/distance ---
    _t2 = time.perf_counter()
    out = model(state_vec, alive_mask, enemy_alive_mask)
    _t3 = time.perf_counter()
    _timing_forward_s += _t3 - _t2
    _timing_calls += 1

    # Decode unit selection
    selected_idx = int(out.unit_logits.argmax().item())
    selected_unit = friendly_units[selected_idx]

    # Decode move type
    move_type = int(out.move_logits.argmax().item())

    # Decode direction + distance (for advance/rush)
    angle, concentration = decode_direction_params(out.direction_params)
    alpha, beta, mean_frac = decode_distance_params(out.distance_params)

    # Compute post-move position
    unit_cx, unit_cy = friendly_positions[selected_idx]
    if move_type == MOVE_ADVANCE:
        budget = float(selected_unit.unit.advance_distance)
        dist_move = mean_frac * budget
        post_x, post_y = compute_post_move_position(unit_cx, unit_cy, angle, dist_move)
    elif move_type == MOVE_RUSH:
        budget = float(selected_unit.unit.rush_distance)
        dist_move = mean_frac * budget
        post_x, post_y = compute_post_move_position(unit_cx, unit_cy, angle, dist_move)
    else:
        post_x, post_y = unit_cx, unit_cy

    # Compute post-move relative features and re-run shoot head with them
    post_move_rel = compute_post_move_rel(post_x, post_y, enemy_positions)

    # --- Forward pass 2: re-run with post-move features for shooting head ---
    out2 = model(state_vec, alive_mask, enemy_alive_mask,
                 forced_unit_idx=selected_idx, post_move_rel=post_move_rel)

    # Apply in-range mask to shoot logits
    max_wr = max(
        (w.range_inches for w in selected_unit.unit.weapons if not w.melee),
        default=0.0,
    )
    shoot_range_mask = compute_in_range_mask(post_move_rel, float(max_wr), enemy_alive_mask)
    masked_shoot_logits = out2.shoot_target_logits.masked_fill(~shoot_range_mask, float('-inf'))

    # Decode targets
    charge_target_idx = int(out2.charge_target_logits.argmax().item()) if enemy_alive_mask.any() else 0
    shoot_target_idx = int(masked_shoot_logits.argmax().item()) if shoot_range_mask.any() else 0

    # Build target ranking from shoot logits (for pick_target_from_ranking compat)
    target_ranking = torch.argsort(masked_shoot_logits, descending=True).tolist()

    # Compute destination in game-space
    dest = None
    if move_type in (MOVE_ADVANCE, MOVE_RUSH):
        # Convert model-space back to game-space
        gx, gy = post_x, post_y
        if player == "B":
            gx = _flip_x(gx)
            gy = _flip_y(gy)
        dest = (gx, gy)

    # Execute decision
    action, goal, charge_target, reason = execute_decoded_decision(
        selected_unit, enemy_units, move_type, dest, charge_target_idx, shoot_target_idx,
    )

    # Assessment for viewer / diagnostics
    move_conf = torch.softmax(out.move_logits, dim=-1)
    assessment = {
        'value': out.value.item(),
        'selected_slot': selected_idx,
        'selected_name': selected_unit.unit.name,
        'unit_selection_logits': out.unit_logits.tolist(),
        'move_type': MOVE_TYPE_NAMES[move_type],
        'move_type_confidence': move_conf[move_type].item(),
        'move_type_probs': move_conf.tolist(),
        'direction_angle': angle,
        'direction_concentration': concentration,
        'distance_frac': mean_frac,
        'distance_alpha': alpha,
        'distance_beta': beta,
        'charge_target_idx': charge_target_idx,
        'charge_target_logits': out2.charge_target_logits.tolist(),
        'shoot_target_idx': shoot_target_idx,
        'target_ranking': target_ranking,
        'target_scores': torch.softmax(masked_shoot_logits, dim=-1).tolist(),
        'action': action,
        'reason': reason,
        'friendly_names': [
            fu.unit.name if i < len(friendly_units) and friendly_units[i].models_alive > 0
            else None
            for i, fu in enumerate(friendly_units)
        ] + [None] * (MAX_UNITS_PER_SIDE - len(friendly_units)),
        'enemy_names': [
            eu.unit.name if i < len(enemy_units) and enemy_units[i].models_alive > 0
            else None
            for i, eu in enumerate(enemy_units)
        ] + [None] * (MAX_UNITS_PER_SIDE - len(enemy_units)),
    }

    # Auxiliary prediction heads (survival + objective control)
    if hasattr(model, 'aux_friendly_survival_head'):
        h = model.trunk(state_vec.unsqueeze(0))  # (1, H)
        fs_raw = model.aux_friendly_survival_head(h).view(MAX_UNITS_PER_SIDE, 2)
        fs_alpha = F.softplus(fs_raw[:, 0]) + 0.01
        fs_beta = F.softplus(fs_raw[:, 1]) + 0.01
        fs_mean = (fs_alpha / (fs_alpha + fs_beta)).tolist()
        assessment['friendly_survival'] = fs_mean

        es_raw = model.aux_enemy_survival_head(h).view(MAX_UNITS_PER_SIDE, 2)
        es_alpha = F.softplus(es_raw[:, 0]) + 0.01
        es_beta = F.softplus(es_raw[:, 1]) + 0.01
        es_mean = (es_alpha / (es_alpha + es_beta)).tolist()
        assessment['enemy_survival'] = es_mean

        obj_logits = model.aux_obj_control_head(h).view(5, 3)
        obj_probs = torch.softmax(obj_logits, dim=-1).tolist()
        assessment['obj_control_probs'] = obj_probs

    return selected_unit, target_ranking, action, goal, charge_target, reason, assessment


# ---------------------------------------------------------------------------
# Sampling variant (for self-play opponents during training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_tactical_model_sampling(
    model: TacticalModel,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    *,
    friendly_ranged_matchups=None,
    friendly_melee_matchups=None,
    enemy_ranged_matchups=None,
    enemy_melee_matchups=None,
    total_friendly_points: int | None = None,
    total_enemy_points: int | None = None,
) -> tuple[UnitState | None, list[int], str, tuple[int, int] | None, UnitState | None, str, dict]:
    """Like apply_tactical_model but samples from distributions instead of argmax."""
    # Build masks
    alive_mask_list = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(friendly_units):
            us = friendly_units[i]
            alive_mask_list.append(us.models_alive > 0 and not us.activated)
        else:
            alive_mask_list.append(False)

    if not any(alive_mask_list):
        return None, [], "hold", None, None, "no units available", {}

    alive_mask = torch.tensor(alive_mask_list, dtype=torch.bool)
    enemy_alive_mask = torch.tensor(
        [(i < len(enemy_units) and enemy_units[i].models_alive > 0)
         for i in range(MAX_UNITS_PER_SIDE)],
        dtype=torch.bool,
    )

    state_vec = encode_state_tactical(
        friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=friendly_ranged_matchups,
        friendly_melee_matchups=friendly_melee_matchups,
        enemy_ranged_matchups=enemy_ranged_matchups,
        enemy_melee_matchups=enemy_melee_matchups,
        total_friendly_points=total_friendly_points,
        total_enemy_points=total_enemy_points,
    )

    friendly_positions = _get_model_space_positions(friendly_units, player)
    enemy_positions = _get_model_space_positions(enemy_units, player)

    x = state_vec.unsqueeze(0)
    am = alive_mask.unsqueeze(0)

    # --- Trunk ---
    h = model.trunk(x)

    # --- Unit selection (sample) ---
    unit_logits = model.unit_selection_head(h)
    unit_logits = unit_logits.masked_fill(~am, float('-inf'))
    unit_probs = torch.softmax(unit_logits, dim=-1).squeeze(0)
    selected_idx = int(torch.multinomial(unit_probs, 1).item())
    selected_unit = friendly_units[selected_idx]

    unit_features = model._extract_unit_features(x, selected_idx).detach()

    # --- Move type head (sample) ---
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf).squeeze(0)
    move_probs = torch.softmax(move_logits, dim=-1)
    move_type = int(torch.multinomial(move_probs, 1).item())

    move_onehot = F.one_hot(
        torch.tensor(move_type), NUM_MOVE_TYPES
    ).float().unsqueeze(0)

    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)

    # --- Direction + Distance (sample from continuous distributions) ---
    direction_raw = model.direction_head(h_uf_m).squeeze(0)
    distance_raw = model.distance_head(h_uf_m).squeeze(0)

    angle, concentration = decode_direction_params(direction_raw)
    alpha, beta_val, _ = decode_distance_params(distance_raw)

    # Sample direction from von Mises
    von_mises = torch.distributions.VonMises(
        torch.tensor(angle), torch.tensor(concentration)
    )
    sampled_angle = von_mises.sample().item()

    # Sample distance from Beta
    beta_dist = torch.distributions.Beta(
        torch.tensor(alpha), torch.tensor(beta_val)
    )
    sampled_frac = beta_dist.sample().item()

    # Compute post-move position (model-space)
    unit_cx, unit_cy = friendly_positions[selected_idx]
    if move_type == MOVE_ADVANCE:
        budget = float(selected_unit.unit.advance_distance)
        dist_move = sampled_frac * budget
        post_x, post_y = compute_post_move_position(unit_cx, unit_cy, sampled_angle, dist_move)
    elif move_type == MOVE_RUSH:
        budget = float(selected_unit.unit.rush_distance)
        dist_move = sampled_frac * budget
        post_x, post_y = compute_post_move_position(unit_cx, unit_cy, sampled_angle, dist_move)
    else:
        post_x, post_y = unit_cx, unit_cy

    # Compute post-move relative features
    post_move_rel = compute_post_move_rel(post_x, post_y, enemy_positions).unsqueeze(0)

    # --- Charge target (sample) ---
    charge_logits = model.charge_target_head(h_uf_m).squeeze(0)
    charge_logits = charge_logits.masked_fill(~enemy_alive_mask, float('-inf'))
    no_enemies = not enemy_alive_mask.any()
    if no_enemies:
        charge_target_idx = 0
    else:
        charge_probs = torch.softmax(charge_logits, dim=-1)
        charge_target_idx = int(torch.multinomial(charge_probs, 1).item())

    # --- Shoot target (sample, with post-move features + in-range mask) ---
    shoot_input = torch.cat([h, unit_features, move_onehot, post_move_rel], dim=-1)
    shoot_logits = model.shoot_target_head(shoot_input).squeeze(0)
    # Mask by alive AND in-range
    max_wr = max(
        (w.range_inches for w in selected_unit.unit.weapons if not w.melee),
        default=0.0,
    )
    shoot_range_mask = compute_in_range_mask(
        post_move_rel.squeeze(0), float(max_wr), enemy_alive_mask)
    shoot_logits = shoot_logits.masked_fill(~shoot_range_mask, float('-inf'))
    no_shootable = not shoot_range_mask.any()
    if no_enemies or no_shootable:
        shoot_target_idx = 0
    else:
        shoot_probs = torch.softmax(shoot_logits, dim=-1)
        shoot_target_idx = int(torch.multinomial(shoot_probs, 1).item())

    target_ranking = torch.argsort(shoot_logits, descending=True).tolist()

    # Convert to game-space destination
    dest = None
    if move_type in (MOVE_ADVANCE, MOVE_RUSH):
        gx, gy = post_x, post_y
        if player == "B":
            gx = _flip_x(gx)
            gy = _flip_y(gy)
        dest = (gx, gy)

    action, goal, charge_target, reason = execute_decoded_decision(
        selected_unit, enemy_units, move_type, dest, charge_target_idx, shoot_target_idx,
    )

    return selected_unit, target_ranking, action, goal, charge_target, reason, {}


# ---------------------------------------------------------------------------
# Target picking from ranking (used by combat resolution)
# ---------------------------------------------------------------------------

def pick_target_from_ranking(
    attacker: UnitState,
    enemies: list[UnitState],
    target_ranking: list[int],
) -> UnitState | None:
    """Walk the ML target ranking and return the highest-ranked enemy in weapon range."""
    for slot_idx in target_ranking:
        if slot_idx >= len(enemies):
            continue
        enemy = enemies[slot_idx]
        if enemy.models_alive <= 0:
            continue
        can_shoot, _, _ = evaluate_target(attacker, enemy)
        if can_shoot:
            return enemy
    return None
