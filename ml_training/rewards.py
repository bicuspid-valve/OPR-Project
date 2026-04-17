"""Reward computation and auxiliary prediction targets."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from board import Board, OBJECTIVES, OBJ_SEIZE_RANGE
from models import UnitState
from ml_features import MAX_UNITS_PER_SIDE, RANGE_THRESHOLDS


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_round_reward(
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    board: Board,
    player: str,
    total_army_points: int,
    prev_friendly_kill_pts: float,
    prev_enemy_kill_pts: float,
    shaping_scale: float = 1.0,
    round_num: int = 1,
) -> tuple[float, float, float]:
    """Compute shaped kill reward for one round.

    Returns (reward, new_friendly_kill_pts, new_enemy_kill_pts).

    friendly_kill_pts = points of enemy units we've destroyed (cumulative).
    enemy_kill_pts = points of our units the enemy has destroyed (cumulative).

    shaping_scale: multiplier for the per-round shaping reward (1.0 = full,
    0.0 = off).  Annealed over training so the policy gradually shifts to
    learning from the terminal margin reward.

    round_num: 1-4, used for phase-dependent reward weighting.  Kill weight
    decreases over the game as the focus shifts to territory control (handled
    by per-activation objective capture rewards separately).
    """
    # Kill points (cumulative)
    friendly_kill_pts = sum(u.unit.points for u in enemy_units if u.models_alive <= 0)
    enemy_kill_pts = sum(u.unit.points for u in friendly_units if u.models_alive <= 0)

    # Points killed this round
    friendly_killed_this_round = friendly_kill_pts - prev_friendly_kill_pts
    enemy_killed_this_round = enemy_kill_pts - prev_enemy_kill_pts

    # Phase-dependent weighting: kills matter more early
    t = (round_num - 1) / 3.0           # 0.0 in round 1 → 1.0 in round 4
    kill_weight = 0.15 - 0.10 * t        # 0.15 → 0.05

    pts = max(total_army_points, 1)
    reward = shaping_scale * (
        kill_weight * (friendly_killed_this_round - enemy_killed_this_round) / pts
    )

    return reward, friendly_kill_pts, enemy_kill_pts


def _any_friendly_on_objective(
    obj_pos: tuple[int, int],
    friendly_units: list[UnitState],
    exclude_unit: UnitState | None = None,
) -> bool:
    """Return True if any non-shaken friendly unit has a model within OBJ_SEIZE_RANGE."""
    threshold_sq = OBJ_SEIZE_RANGE * OBJ_SEIZE_RANGE
    for u in friendly_units:
        if u is exclude_unit:
            continue
        if u.models_alive <= 0 or u.shaken:
            continue
        for pos in u.alive_positions():
            if _dist_sq(pos, obj_pos) <= threshold_sq:
                return True
    return False


def _dist_sq(a: tuple, b: tuple) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def compute_objective_capture_reward(
    active_unit: UnitState,
    friendly_units: list[UnitState],
    board: Board,
    player: str,
    round_num: int,
    shaping_scale: float,
    pre_move_friendly_on_objs: list[bool],
) -> float:
    """Per-activation reward for newly occupying a neutral/enemy objective.

    Gives a small reward when the active unit ends its activation within
    OBJ_SEIZE_RANGE of a neutral or enemy-controlled objective that had no
    non-shaken friendly unit on it at the start of the activation.

    The reward is tripled in round 4 to emphasise late-game objective play.

    Parameters
    ----------
    active_unit : the unit that just activated
    friendly_units : all friendly units (including active_unit)
    board : current board state
    player : "A" or "B"
    round_num : 1-4
    shaping_scale : anneal multiplier (1.0 → 0.0 over training)
    pre_move_friendly_on_objs : length-5 list of bools — whether a friendly
        unit was on each objective at the START of this activation (before
        the unit moved).  Computed by caller before movement.

    Returns
    -------
    Scalar reward to add to this activation's reward.
    """
    if shaping_scale <= 0.0 or active_unit.models_alive <= 0:
        return 0.0

    _BASE_OBJ_CAPTURE_REWARD = 0.02
    round_mult = 3.0 if round_num == 4 else 1.0

    threshold_sq = OBJ_SEIZE_RANGE * OBJ_SEIZE_RANGE
    friend_tag = player
    reward = 0.0

    for oi, (oc, orow) in enumerate(OBJECTIVES):
        # Only reward capturing neutral or enemy objectives
        if board.objective_control[oi] == friend_tag:
            continue
        # Must not have had a friendly unit on it at activation start
        if pre_move_friendly_on_objs[oi]:
            continue
        # Check if the active unit is now on this objective
        for pos in active_unit.alive_positions():
            if _dist_sq(pos, (oc, orow)) <= threshold_sq:
                reward += _BASE_OBJ_CAPTURE_REWARD * round_mult
                break

    return shaping_scale * reward


def compute_shooting_efficiency_reward(
    attacker: UnitState,
    target: UnitState,
    attacker_idx: int,
    target_idx: int,
    ranged_matchups: np.ndarray,
    total_army_points: int,
    round_num: int,
) -> float:
    """Per-activation reward for shooting efficiency (pre-anneal).

    Returns the reward value **before** shaping_scale is applied.  The caller
    is responsible for multiplying by shaping_scale when adding to the
    trajectory reward, so that the raw (pre-anneal) value can be logged
    separately for diagnostics.

    The expected damage is looked up from the precomputed ranged matchup
    table (kill fraction at the appropriate range bucket), scaled by the
    attacker's surviving model fraction, and converted to points.

    Parameters
    ----------
    attacker : the shooting unit (post-move)
    target : the target unit (pre-resolve)
    attacker_idx : index of attacker in its side's unit list
    target_idx : index of target in the opposing unit list
    ranged_matchups : precomputed array, shape (n_friendly, MAX_UNITS, NUM_RANGE)
    total_army_points : total points of the attacker's army (for normalisation)
    round_num : 1-4
    """
    # Distance between attacker and target centres
    ax, ay = attacker.centre()
    tx, ty = target.centre()
    dist = math.sqrt((ax - tx) ** 2 + (ay - ty) ** 2)

    # Find the largest range threshold <= dist (conservative: may slightly
    # overestimate by including weapons that can't quite reach).
    # If dist < smallest threshold, use bucket 0 (all ranged weapons).
    bucket = 0
    for k, threshold in enumerate(RANGE_THRESHOLDS):
        if threshold <= dist:
            bucket = k
        else:
            break

    # Kill fraction (precomputed assuming all models alive)
    kill_frac = float(ranged_matchups[attacker_idx, target_idx, bucket])

    # Scale by attacker's surviving model fraction (fewer models = fewer weapons)
    alive_frac = attacker.models_alive / max(attacker.unit.models, 1)

    # Expected points of damage
    expected_pts = kill_frac * alive_frac * target.unit.points

    # Normalise by total army points (same scale as kill reward)
    pts = max(total_army_points, 1)

    # Phase-dependent weight: stronger early when shooting decisions
    # matter most and kill shaping is also active.
    t = (round_num - 1) / 3.0           # 0.0 in round 1 → 1.0 in round 4
    weight = 0.20 - 0.10 * t            # 0.20 → 0.10

    return weight * expected_pts / pts


def compute_charge_efficiency_reward(
    attacker: UnitState,
    target: UnitState,
    attacker_idx: int,
    target_idx: int,
    melee_matchups: np.ndarray,
    total_army_points: int,
    round_num: int,
) -> float:
    """Per-activation reward for charge target efficiency (pre-anneal).

    Returns the reward value **before** shaping_scale is applied.  The caller
    is responsible for multiplying by shaping_scale when adding to the
    trajectory reward, so that the raw (pre-anneal) value can be logged
    separately for diagnostics.

    Uses the precomputed melee matchup table (expected kill fraction) scaled
    by the attacker's surviving model fraction, converted to points.

    Parameters
    ----------
    attacker : the charging unit (pre-combat)
    target : the charge target (pre-combat)
    attacker_idx : index of attacker in its side's unit list
    target_idx : index of target in the opposing unit list
    melee_matchups : precomputed array, shape (n_friendly, MAX_UNITS)
    total_army_points : total points of the attacker's army (for normalisation)
    round_num : 1-4
    """
    # Kill fraction (precomputed assuming all models alive)
    kill_frac = float(melee_matchups[attacker_idx, target_idx])

    # Scale by attacker's surviving model fraction (fewer models = fewer attacks)
    alive_frac = attacker.models_alive / max(attacker.unit.models, 1)

    # Expected points of damage
    expected_pts = kill_frac * alive_frac * target.unit.points

    # Normalise by total army points (same scale as kill reward)
    pts = max(total_army_points, 1)

    # Phase-dependent weight: same schedule as shooting efficiency
    t = (round_num - 1) / 3.0           # 0.0 in round 1 → 1.0 in round 4
    weight = 0.20 - 0.10 * t            # 0.20 → 0.10

    return weight * expected_pts / pts


def terminal_reward(result: str, player: str, a_objs: int = 0, b_objs: int = 0) -> float:
    """Margin-based terminal reward.

    Bulk of the reward (+/-0.5) is for winning or losing.  The remaining
    +/-0.5 scales linearly with the objective margin so dominant victories
    are rewarded more than narrow ones.

    With 5 objectives the maximum margin is 5, giving a reward range of
    [-1.0, +1.0].  Draws remain 0.0.
    """
    max_objs = 5
    if player == "A":
        margin = (a_objs - b_objs) / max_objs          # in [-1, +1]
    else:
        margin = (b_objs - a_objs) / max_objs

    if result == player:
        return 0.5 + 0.5 * margin                       # [+0.5, +1.0]
    elif result == "draw":
        return 0.0
    return -0.5 + 0.5 * margin                          # [-1.0, -0.5]


# ---------------------------------------------------------------------------
# Auxiliary prediction targets
# ---------------------------------------------------------------------------

def _compute_survival_fracs(
    units: list[UnitState],
    n_slots: int = MAX_UNITS_PER_SIDE,
) -> list[float]:
    """Return survival fraction (models_alive / models) for each unit slot.

    Dead or missing slots get 0.0.
    """
    fracs = []
    for i in range(n_slots):
        if i < len(units) and units[i].unit.models > 0:
            fracs.append(units[i].models_alive / units[i].unit.models)
        else:
            fracs.append(0.0)
    return fracs


def _compute_obj_control_target(
    obj_control: list[str],
    player: str,
) -> list[int]:
    """Encode objective control as class indices from *player*'s perspective.

    0 = controlled by player (friendly), 1 = controlled by opponent (enemy),
    2 = neutral / uncontrolled.

    Indices are in model-space order (matching _objective_control_mapped /
    _get_model_objectives in ml_features), so index 1 is always the player's
    own side and index 2 the enemy side regardless of physical deployment.
    """
    enemy = "B" if player == "A" else "A"
    order = [0, 1, 2, 3, 4] if player == "A" else [0, 2, 1, 4, 3]

    def _cls(idx: int) -> int:
        ctrl = obj_control[idx]
        if ctrl == player:
            return 0
        if ctrl == enemy:
            return 1
        return 2

    return [_cls(i) for i in order]


@dataclass
class RoundSnapshot:
    """Snapshot of game state at a round boundary for multi-horizon aux targets."""
    friendly_survival: list[float]  # length 10, per-unit survival fraction
    enemy_survival: list[float]     # length 10
    obj_control: list[int]          # length 5, 0=friendly 1=enemy 2=neutral


def _make_round_snapshot(
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    board,
    player: str,
) -> RoundSnapshot:
    """Snapshot survival fracs and objective control at round boundary."""
    fs = _compute_survival_fracs(friendly_units)
    es = _compute_survival_fracs(enemy_units)
    obj = _compute_obj_control_target(board.objective_control, player)
    return RoundSnapshot(fs, es, obj)
