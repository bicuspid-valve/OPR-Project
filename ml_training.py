"""ML training loop: PPO with GAE, reward shaping, opponent scheduling, checkpoints."""
from __future__ import annotations

import copy
import csv
import math
import os
import random
import time
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path

from contextlib import contextmanager

import numpy as np
import torch
import torch.multiprocessing as _mp
import torch.nn as nn
import torch.nn.functional as F

from board import Board
from models import ResolvedUnit, UnitState
from ml_features import MAX_UNITS_PER_SIDE, TACTICAL_UNIT_FEATURES, encode_state_tactical, precompute_damage
from ml_model_tactical import (
    TacticalModel, NUM_MOVE_TYPES, MOVE_HOLD, MOVE_ADVANCE, MOVE_RUSH, MOVE_CHARGE,
    POST_MOVE_REL_FEATURES,
)
from ml_integration_tactical import (
    MOVE_TYPE_NAMES, execute_decoded_decision, pick_target_from_ranking,
    compute_post_move_rel, compute_post_move_position,
    decode_direction_params, decode_distance_params,
    _get_model_space_positions, _get_movement_budgets, _get_max_weapon_ranges,
    compute_in_range_mask, compute_in_range_mask_batched,
)
from ml_features import BOARD_DIAG, _flip_x, _flip_y

# Try to import army generation utilities (optional — only needed for real training)
try:
    from evolution import generate_random_army, resolve_army, _make_unit_states
    _HAS_EVOLUTION = True
except Exception:
    _HAS_EVOLUTION = False


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    """Resolve 'auto'/'cuda'/'cpu' to a torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@contextmanager
def _force_tensor_device(device: torch.device):
    """Monkey-patch torch tensor-creation to target *device*.

    Allows replay_tactical_log_probs_flat (which hard-codes CPU tensors via
    torch.from_numpy / torch.tensor) to run with a model on GPU — every
    intermediate tensor is created on the correct device.
    """
    if device.type == "cpu":
        yield
        return

    _orig_from_numpy = torch.from_numpy
    _orig_tensor = torch.tensor
    _orig_full = torch.full
    _orig_zeros = torch.zeros
    _orig_ones = torch.ones

    def _from_numpy(a):
        return _orig_from_numpy(a).to(device)

    def _tensor(*args, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = device
        return _orig_tensor(*args, **kwargs)

    def _full(size, fill_value, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = device
        return _orig_full(size, fill_value, **kwargs)

    def _zeros(*size, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = device
        return _orig_zeros(*size, **kwargs)

    def _ones(*size, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = device
        return _orig_ones(*size, **kwargs)

    torch.from_numpy = _from_numpy
    torch.tensor = _tensor
    torch.full = _full
    torch.zeros = _zeros
    torch.ones = _ones
    try:
        yield
    finally:
        torch.from_numpy = _orig_from_numpy
        torch.tensor = _orig_tensor
        torch.full = _orig_full
        torch.zeros = _orig_zeros
        torch.ones = _orig_ones


# ---------------------------------------------------------------------------
# Constants / Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyperparameters for the training loop."""
    lr: float = 1e-4
    batch_size: int = 64
    entropy_coeff_start: float = 0.01
    entropy_coeff_end: float = 0.01
    baseline_alpha: float = 0.01
    checkpoint_interval: int = 50      # save checkpoint every N batches
    max_checkpoints: int = 20          # checkpoint pool size
    heuristic_window: int = 200        # rolling window for heuristic win rate
    num_batches: int = 1000
    time_limit: float | None = None  # wall-clock limit in minutes (None = no limit)
    checkpoint_dir: str = "ml_checkpoints"
    log_dir: str = "ml_logs"
    # PPO hyperparameters
    clip_epsilon: float = 0.2          # PPO clipping range
    value_coeff: float = 0.5           # value loss weight
    gae_lambda: float = 0.95           # GAE lambda for advantage estimation
    ppo_epochs: int = 3                # gradient steps per batch of episodes
    ppo_minibatch_games: int = 64       # games per PPO minibatch (0 = full batch)
    model_type: str = "tactical"      # per-activation tactical model
    use_c_ext: bool = True             # use C extension for hot loops (if compiled)
    worker_count: int | None = 6       # number of pool workers (None = use module default)
    device: str = "auto"               # "auto" (GPU if available), "cuda", or "cpu"
    shaping_anneal_end: float = 0.5    # fraction of training at which per-round reward shaping reaches 0
    aux_coeff: float = 0.1            # max weight for auxiliary prediction losses (survival + obj control)
    aux_ratio: float = 0.2             # aux contributes at most this fraction of policy loss magnitude
    # Per-head entropy targets (adaptive entropy tuning)
    use_entropy_targets: bool = True   # if True, use per-head adaptive entropy; else use entropy_coeff
    entropy_target_fraction: float = 0.25  # fraction of max entropy for masked categoricals
    entropy_target_move: float = 0.25 * math.log(4)      # ~0.347
    entropy_target_dir: float = 0.75                       # von Mises direction
    entropy_target_dist: float = -0.25                     # Beta distance
    entropy_alpha_lr: float = 1e-4                         # learning rate for entropy alpha params


# ---------------------------------------------------------------------------
# Per-head entropy target tuner (SAC-style adaptive entropy)
# ---------------------------------------------------------------------------

class EntropyTargetTuner(nn.Module):
    """Maintains learnable log-alpha per policy head for adaptive entropy.

    Each head has an independent coefficient alpha_i = exp(log_alpha_i).
    The alpha loss drives entropy toward per-head targets:
        alpha_loss_i = -alpha_i * (entropy_i - target_i)

    For masked categorical heads (unit, charge, shoot), the target is
    computed dynamically as fraction * ln(num_valid_actions).
    """

    # Head names for logging/serialization
    HEAD_NAMES = ("unit", "move", "dir", "dist", "charge", "shoot")

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        # One log-alpha per head (initialized to ~0.01 effective alpha)
        init_val = math.log(0.01)
        self.log_alphas = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(init_val))
            for name in self.HEAD_NAMES
        })
        # Fixed targets for non-masked heads
        self.target_move = config.entropy_target_move
        self.target_dir = config.entropy_target_dir
        self.target_dist = config.entropy_target_dist
        # Fraction for dynamic masked-categorical targets
        self.target_fraction = config.entropy_target_fraction

    # Alpha bounds: prevent runaway entropy bonus
    LOG_ALPHA_MIN = math.log(0.001)  # alpha >= 0.001
    LOG_ALPHA_MAX = math.log(0.1)    # alpha <= 0.1

    def get_alpha(self, head: str) -> torch.Tensor:
        """Return the positive alpha coefficient for a head."""
        clamped = self.log_alphas[head].clamp(self.LOG_ALPHA_MIN, self.LOG_ALPHA_MAX)
        return clamped.exp()

    def compute_entropy_bonus(
        self,
        unit_ent: torch.Tensor,      # (N,)
        move_ent: torch.Tensor,      # (N,)
        dir_ent: torch.Tensor,       # (N,)
        dist_ent: torch.Tensor,      # (N,)
        charge_ent: torch.Tensor,    # (N,)
        shoot_ent: torch.Tensor,     # (N,)
        is_adv_rush: torch.Tensor,   # (N,) bool — direction/distance active
        is_hold_adv: torch.Tensor,   # (N,) bool — shoot active
        is_charge: torch.Tensor,     # (N,) bool — charge active
    ) -> torch.Tensor:
        """Compute the weighted entropy bonus for the policy loss.

        Returns a scalar: sum of alpha_i * mean_entropy_i across active heads.
        """
        # Detach alphas so the policy loss gradient doesn't flow through them —
        # only the separate alpha_loss should update the alpha parameters.
        bonus = self.get_alpha("unit").detach() * unit_ent.mean()
        bonus = bonus + self.get_alpha("move").detach() * move_ent.mean()

        # Direction + distance: only active for advance/rush
        n_adv_rush = is_adv_rush.sum().clamp(min=1)
        bonus = bonus + self.get_alpha("dir").detach() * (dir_ent * is_adv_rush).sum() / n_adv_rush
        bonus = bonus + self.get_alpha("dist").detach() * (dist_ent * is_adv_rush).sum() / n_adv_rush

        # Charge: only active for charge moves
        n_charge = is_charge.sum().clamp(min=1)
        bonus = bonus + self.get_alpha("charge").detach() * (charge_ent * is_charge).sum() / n_charge

        # Shoot: only active for hold/advance
        n_hold_adv = is_hold_adv.sum().clamp(min=1)
        bonus = bonus + self.get_alpha("shoot").detach() * (shoot_ent * is_hold_adv).sum() / n_hold_adv

        return bonus

    def compute_alpha_loss(
        self,
        unit_ent: torch.Tensor,       # (N,)
        move_ent: torch.Tensor,       # (N,)
        dir_ent: torch.Tensor,        # (N,)
        dist_ent: torch.Tensor,       # (N,)
        charge_ent: torch.Tensor,     # (N,)
        shoot_ent: torch.Tensor,      # (N,)
        is_adv_rush: torch.Tensor,    # (N,) bool
        is_hold_adv: torch.Tensor,    # (N,) bool
        is_charge: torch.Tensor,      # (N,) bool
        alive_mask: torch.Tensor,     # (N, 10) — for unit target
        enemy_alive_mask: torch.Tensor,  # (N, 10) — for charge target
        shoot_mask: torch.Tensor,     # (N, 10) — for shoot target
    ) -> torch.Tensor:
        """Compute the dual alpha loss that drives entropy toward targets.

        All entropy values are detached so the alpha loss doesn't affect the policy.
        """
        loss = torch.tensor(0.0)

        # Unit selection: target = fraction * ln(num_alive)
        n_alive = alive_mask.sum(dim=-1).clamp(min=1).float()
        unit_target = self.target_fraction * torch.log(n_alive)
        loss = loss + self.get_alpha("unit") * (unit_ent.detach() - unit_target).mean()

        # Move type: fixed target
        loss = loss + self.get_alpha("move") * (move_ent.detach() - self.target_move).mean()

        # Direction: fixed target, only for advance/rush steps
        if is_adv_rush.any():
            n_ar = is_adv_rush.sum().clamp(min=1)
            mean_dir_ent = (dir_ent.detach() * is_adv_rush).sum() / n_ar
            loss = loss + self.get_alpha("dir") * (mean_dir_ent - self.target_dir)

        # Distance: fixed target, only for advance/rush steps
        if is_adv_rush.any():
            n_ar = is_adv_rush.sum().clamp(min=1)
            mean_dist_ent = (dist_ent.detach() * is_adv_rush).sum() / n_ar
            loss = loss + self.get_alpha("dist") * (mean_dist_ent - self.target_dist)

        # Charge target: target = fraction * ln(num_enemy_alive), only for charge steps
        if is_charge.any():
            n_enemy_alive = enemy_alive_mask.sum(dim=-1).clamp(min=1).float()
            charge_target = self.target_fraction * torch.log(n_enemy_alive)
            n_ch = is_charge.sum().clamp(min=1)
            mean_charge_ent = (charge_ent.detach() * is_charge).sum() / n_ch
            mean_charge_target = (charge_target * is_charge).sum() / n_ch
            loss = loss + self.get_alpha("charge") * (mean_charge_ent - mean_charge_target)

        # Shoot target: target = fraction * ln(num_in_range), only for hold/advance steps
        if is_hold_adv.any():
            n_shootable = shoot_mask.sum(dim=-1).clamp(min=1).float()
            shoot_target = self.target_fraction * torch.log(n_shootable)
            n_ha = is_hold_adv.sum().clamp(min=1)
            mean_shoot_ent = (shoot_ent.detach() * is_hold_adv).sum() / n_ha
            mean_shoot_target = (shoot_target * is_hold_adv).sum() / n_ha
            loss = loss + self.get_alpha("shoot") * (mean_shoot_ent - mean_shoot_target)

        return loss

    def alpha_summary(self) -> dict[str, float]:
        """Return current alpha values for logging."""
        return {name: self.log_alphas[name].exp().item() for name in self.HEAD_NAMES}


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
    """Compute shaped reward for one round with phase-dependent weighting.

    Returns (reward, new_friendly_kill_pts, new_enemy_kill_pts).

    friendly_kill_pts = points of enemy units we've destroyed (cumulative).
    enemy_kill_pts = points of our units the enemy has destroyed (cumulative).

    shaping_scale: multiplier for the per-round shaping reward (1.0 = full,
    0.0 = off).  Annealed over training so the policy gradually shifts to
    learning from the terminal margin reward.

    round_num: 1-4, used for phase-dependent reward weighting.  Early rounds
    emphasise kills (force-projection), later rounds emphasise objectives
    (territory control).
    """
    # Objective control
    friend_tag = player
    enemy_tag = "B" if player == "A" else "A"
    friendly_objs = sum(1 for c in board.objective_control if c == friend_tag)
    enemy_objs = sum(1 for c in board.objective_control if c == enemy_tag)

    # Kill points (cumulative)
    friendly_kill_pts = sum(u.unit.points for u in enemy_units if u.models_alive <= 0)
    enemy_kill_pts = sum(u.unit.points for u in friendly_units if u.models_alive <= 0)

    # Points killed this round
    friendly_killed_this_round = friendly_kill_pts - prev_friendly_kill_pts
    enemy_killed_this_round = enemy_kill_pts - prev_enemy_kill_pts

    # Phase-dependent weighting: kills matter early, objectives matter late
    t = (round_num - 1) / 3.0           # 0.0 in round 1 → 1.0 in round 4
    kill_weight = 0.15 - 0.10 * t        # 0.15 → 0.05
    obj_weight  = 0.005 + 0.015 * t      # 0.005 → 0.02

    pts = max(total_army_points, 1)
    reward = shaping_scale * (
        obj_weight * (friendly_objs - enemy_objs)
        + kill_weight * (friendly_killed_this_round - enemy_killed_this_round) / pts
    )

    return reward, friendly_kill_pts, enemy_kill_pts


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
    """
    enemy = "B" if player == "A" else "A"
    target = []
    for ctrl in obj_control:
        if ctrl == player:
            target.append(0)
        elif ctrl == enemy:
            target.append(1)
        else:
            target.append(2)
    return target


# ---------------------------------------------------------------------------
# Action sampling & log-prob computation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tactical model: per-activation data structures & sampling
# ---------------------------------------------------------------------------

@dataclass
class TacticalActivationRecord:
    """Serializable trajectory data for one activation (tactical v2 model).

    Stores the sequential decisions: unit, move_type, direction+distance
    (continuous), charge_target, shoot_target, plus masks and value.
    """
    state_vec: list[float]                # flattened encoded state (2811 floats)
    alive_mask: list[bool]                # which friendly slots were alive+unactivated
    enemy_alive_mask: list[bool]          # which enemy slots were alive
    unit_idx: int                         # which unit was selected
    move_type: int                        # 0=hold, 1=advance, 2=rush, 3=charge
    sampled_angle: float                  # direction in radians (for advance/rush)
    sampled_distance_frac: float          # 0-1 fraction of movement budget
    charge_target_idx: int                # enemy slot for charge
    shoot_target_idx: int                 # enemy slot for shooting (hold/advance)
    shoot_mask: list[bool]                # enemy alive AND in weapon range (10 bools)
    post_move_rel: list[float]            # 30 post-move relative features
    reward: float = 0.0
    old_log_prob: float = 0.0             # sum of log-probs under collection policy
    old_value: float = 0.0                # value estimate under collection policy
    # Auxiliary prediction targets (filled after round/game completion)
    friendly_survival_target: list[float] | None = None  # 10 fracs, end-of-round
    enemy_survival_target: list[float] | None = None     # 10 fracs, end-of-round
    obj_control_target: list[int] | None = None          # 5 ints, end-of-game (0=friendly, 1=enemy, 2=neutral)


@dataclass
class _TacticalInferenceRequest:
    """Yielded by episode generator when it needs a batched ML forward pass."""
    state_vec: torch.Tensor        # (TACTICAL_TOTAL_FEATURES,) encoded state
    alive_mask: torch.Tensor       # (MAX_UNITS_PER_SIDE,) bool
    enemy_alive_mask: torch.Tensor # (MAX_UNITS_PER_SIDE,) bool
    model_key: str                 # "main" or "opponent"
    friendly_positions: list[tuple[float, float]]  # model-space, 10 slots
    enemy_positions: list[tuple[float, float]]     # model-space, 10 slots
    advance_distances: list[float]                 # per friendly slot
    rush_distances: list[float]                    # per friendly slot
    max_weapon_ranges: list[float]                 # max ranged weapon range per friendly slot


@dataclass
class _TacticalSamplingResult:
    """Sent back to episode generator with batched sampling outputs."""
    unit_idx: int
    move_type: int              # 0-3
    sampled_angle: float        # radians
    sampled_distance_frac: float  # 0-1
    charge_target_idx: int      # enemy slot
    shoot_target_idx: int       # enemy slot
    target_ranking: list        # shoot target ranking for compat
    post_move_rel: list[float]  # 30 floats
    old_log_prob: float
    value: float
    shoot_mask: list[bool]      # enemy alive AND in weapon range (10 bools)


@torch.no_grad()
def sample_tactical_actions_no_grad(
    model: TacticalModel,
    state_vec: torch.Tensor,           # (2811,)
    alive_mask: torch.Tensor,          # (10,) bool — friendly alive+unactivated
    enemy_alive_mask: torch.Tensor,    # (10,) bool — enemy alive
    friendly_positions: list[tuple[float, float]],  # model-space, 10 slots
    enemy_positions: list[tuple[float, float]],     # model-space, 10 slots
    advance_distances: list[float],                 # per friendly slot
    rush_distances: list[float],                    # per friendly slot
    max_weapon_ranges: list[float] | None = None,   # max ranged weapon range per friendly slot
) -> tuple[int, int, float, float, int, int, list[int], list[float], float, float, list[bool]]:
    """Sample tactical v2 actions with sequential conditioning (no gradient tracking).

    Returns (unit_idx, move_type, sampled_angle, sampled_distance_frac,
             charge_target_idx, shoot_target_idx,
             target_ranking, post_move_rel, old_log_prob, value, shoot_mask).
    """
    eps = 1e-8

    x = state_vec.unsqueeze(0)
    am = alive_mask.unsqueeze(0)

    # --- Trunk ---
    h, u_attended, _attn_w, round_onehot = model.trunk(x)

    # --- Unit selection (sample) ---
    unit_logits = model.unit_selection_head(h)
    unit_logits = unit_logits.masked_fill(~am, float('-inf'))
    unit_probs = torch.softmax(unit_logits, dim=-1).squeeze(0)
    unit_idx = int(torch.multinomial(unit_probs, 1).item())
    unit_lp = torch.log(unit_probs[unit_idx] + eps).item()

    unit_features = model._extract_unit_features(u_attended, unit_idx).detach()

    # --- Move type head (sample) ---
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf).squeeze(0)
    move_probs = torch.softmax(move_logits, dim=-1)
    move_type = int(torch.multinomial(move_probs, 1).item())
    move_lp = torch.log(move_probs[move_type] + eps).item()

    move_onehot = F.one_hot(
        torch.tensor(move_type), NUM_MOVE_TYPES,
    ).float().unsqueeze(0)
    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)

    # --- Direction + Distance (sample from continuous) ---
    direction_raw = model.direction_head(h_uf_m).squeeze(0)
    distance_raw = model.distance_head(h_uf_m).squeeze(0)

    angle, concentration = decode_direction_params(direction_raw)
    alpha, beta_val, _ = decode_distance_params(distance_raw)

    # Use numpy for single-sample — 14x faster than torch VonMises
    sampled_angle = float(np.random.vonmises(angle, concentration))
    # VonMises log-prob: κ·cos(x - μ) - log(2π·I₀(κ))  [i0e-based for numerical stability]
    _conc_t = torch.tensor(concentration)
    _i0e_c = torch.special.i0e(_conc_t)
    _log_i0_c = concentration + math.log(max(float(_i0e_c.item()), 1e-20))
    dir_lp = concentration * math.cos(sampled_angle - angle) - (math.log(2.0 * math.pi) + _log_i0_c)

    sampled_frac = float(np.random.beta(alpha, beta_val))
    clamped_frac = max(1e-4, min(1.0 - 1e-4, sampled_frac))
    # Beta log-prob via torch (single scalar, fast enough)
    dist_lp = max(-20.0, min(20.0, torch.distributions.Beta(
        torch.tensor(alpha), torch.tensor(beta_val),
    ).log_prob(torch.tensor(clamped_frac)).item()))

    # Compute post-move position (model-space)
    unit_cx, unit_cy = friendly_positions[unit_idx]
    if move_type == MOVE_ADVANCE:
        budget = advance_distances[unit_idx]
        post_x, post_y = compute_post_move_position(
            unit_cx, unit_cy, sampled_angle, sampled_frac * budget)
    elif move_type == MOVE_RUSH:
        budget = rush_distances[unit_idx]
        post_x, post_y = compute_post_move_position(
            unit_cx, unit_cy, sampled_angle, sampled_frac * budget)
    else:
        post_x, post_y = unit_cx, unit_cy

    post_move_rel = compute_post_move_rel(post_x, post_y, enemy_positions)
    post_move_rel_unsq = post_move_rel.unsqueeze(0)

    # --- Charge target (sample) ---
    charge_logits = model.charge_target_head(h_uf_m).squeeze(0)
    charge_logits = charge_logits.masked_fill(~enemy_alive_mask, float('-inf'))
    no_enemies = not enemy_alive_mask.any()
    if no_enemies:
        charge_target_idx = 0
        charge_lp = 0.0
    else:
        charge_probs = torch.softmax(charge_logits, dim=-1)
        charge_target_idx = int(torch.multinomial(charge_probs, 1).item())
        charge_lp = torch.log(charge_probs[charge_target_idx] + eps).item()

    # --- Shoot target (sample, with post-move features + in-range mask) ---
    shoot_input = torch.cat([h, unit_features, move_onehot, post_move_rel_unsq], dim=-1)
    shoot_logits = model.shoot_target_head(shoot_input).squeeze(0)
    # Mask by alive AND in-range
    if max_weapon_ranges is not None:
        shoot_mask_t = compute_in_range_mask(
            post_move_rel, max_weapon_ranges[unit_idx], enemy_alive_mask)
    else:
        shoot_mask_t = enemy_alive_mask
    shoot_logits = shoot_logits.masked_fill(~shoot_mask_t, float('-inf'))
    shoot_mask_list = shoot_mask_t.tolist()
    no_shootable = not shoot_mask_t.any()
    if no_enemies or no_shootable:
        shoot_target_idx = 0
        shoot_lp = 0.0
    else:
        shoot_probs = torch.softmax(shoot_logits, dim=-1)
        shoot_target_idx = int(torch.multinomial(shoot_probs, 1).item())
        shoot_lp = torch.log(shoot_probs[shoot_target_idx] + eps).item()

    target_ranking = torch.argsort(shoot_logits, descending=True).tolist()

    # --- Value (round-conditioned) ---
    value = model.value_head(h, round_onehot).squeeze(0).item()

    # Log-prob: sum across active heads based on move_type
    # Always: unit + move_type
    # Advance/rush: + direction + distance + shoot_target
    # Hold: + shoot_target
    # Charge: + charge_target
    old_log_prob = unit_lp + move_lp
    if move_type in (MOVE_ADVANCE, MOVE_RUSH):
        old_log_prob += dir_lp + dist_lp
    if move_type in (MOVE_HOLD, MOVE_ADVANCE):
        old_log_prob += shoot_lp
    if move_type == MOVE_CHARGE:
        old_log_prob += charge_lp

    return (unit_idx, move_type, sampled_angle, sampled_frac,
            charge_target_idx, shoot_target_idx, target_ranking,
            post_move_rel.tolist(), old_log_prob, value, shoot_mask_list)


# ---------------------------------------------------------------------------
# Batched sampling for coroutine-based episode collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _batched_sample_tactical_no_grad(
    model: TacticalModel,
    requests: list[_TacticalInferenceRequest],
) -> list[_TacticalSamplingResult]:
    """Run batched forward pass with sampling for multiple concurrent games.

    The batched version handles discrete heads (unit, move_type, charge_target,
    shoot_target) in parallel.  Continuous heads (direction, distance) are sampled
    per-sample because VonMises/Beta don't batch well with per-sample parameters.
    """
    n = len(requests)
    if n == 0:
        return []

    n_units = MAX_UNITS_PER_SIDE
    eps = 1e-8

    # Stack inputs
    state_batch = torch.stack([r.state_vec for r in requests])              # (N, feat)
    alive_batch = torch.stack([r.alive_mask for r in requests])             # (N, 10)
    enemy_alive_batch = torch.stack([r.enemy_alive_mask for r in requests]) # (N, 10)

    # Trunk
    h, u_attended, _attn_w, round_onehot = model.trunk(state_batch)           # (N, 512), (N, 20, 180), ..., (N, 4)
    if torch.isnan(h).any() or torch.isinf(h).any():
        print("  WARNING: NaN/Inf in trunk output during data collection — clamping")
        h = torch.nan_to_num(h, nan=0.0, posinf=50.0, neginf=-50.0)

    # Unit selection
    unit_logits = model.unit_selection_head(h)                              # (N, 10)
    unit_logits = unit_logits.masked_fill(~alive_batch, float('-inf'))
    # Guard against all-dead rows (all -inf → NaN softmax)
    all_dead = ~alive_batch.any(dim=1, keepdim=True)
    unit_logits = unit_logits.masked_fill(all_dead, 0.0)
    unit_logits = torch.nan_to_num(unit_logits, nan=0.0, posinf=50.0, neginf=-50.0)
    unit_probs = torch.softmax(unit_logits, dim=-1)
    unit_indices = torch.multinomial(unit_probs, 1).squeeze(-1)             # (N,)
    unit_log_probs = torch.log_softmax(unit_logits, dim=-1)
    unit_lp = unit_log_probs.gather(1, unit_indices.unsqueeze(1)).squeeze(1)

    # Extract per-sample unit features from attended embeddings
    unit_features = u_attended[:, :n_units, :].gather(
        1, unit_indices.unsqueeze(1).unsqueeze(2).expand(n, 1, TACTICAL_UNIT_FEATURES),
    ).squeeze(1).detach()                                                   # (N, UF)

    # Move type
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf)                                # (N, 4)
    move_logits = torch.nan_to_num(move_logits, nan=0.0, posinf=50.0, neginf=-50.0)
    move_probs = torch.softmax(move_logits, dim=-1)
    move_indices = torch.multinomial(move_probs, 1).squeeze(-1)             # (N,)
    move_log_probs = torch.log_softmax(move_logits, dim=-1)
    move_lp = move_log_probs.gather(1, move_indices.unsqueeze(1)).squeeze(1)
    move_onehot = F.one_hot(move_indices, NUM_MOVE_TYPES).float()           # (N, 4)

    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)            # (N, 272)

    # Direction + Distance (raw outputs, batched)
    direction_raw = model.direction_head(h_uf_m)                            # (N, 3)
    direction_raw = torch.nan_to_num(direction_raw, nan=0.0, posinf=50.0, neginf=-50.0)
    distance_raw = model.distance_head(h_uf_m)                              # (N, 2)
    distance_raw = torch.nan_to_num(distance_raw, nan=0.0, posinf=50.0, neginf=-50.0)

    # Charge target
    charge_logits = model.charge_target_head(h_uf_m)                        # (N, 10)
    charge_logits = charge_logits.masked_fill(~enemy_alive_batch, float('-inf'))
    no_enemies = ~enemy_alive_batch.any(dim=-1)                             # (N,)
    charge_logits = charge_logits.masked_fill(no_enemies.unsqueeze(-1), 0.0)
    charge_logits = torch.nan_to_num(charge_logits, nan=0.0, posinf=50.0, neginf=-50.0)
    charge_probs = torch.softmax(charge_logits, dim=-1)
    if no_enemies.any():
        uniform_c = torch.full_like(charge_probs, 1.0 / n_units)
        charge_probs = torch.where(no_enemies.unsqueeze(-1), uniform_c, charge_probs)
    charge_indices = torch.multinomial(charge_probs, 1).squeeze(-1)
    charge_log_probs = torch.log_softmax(charge_logits, dim=-1)
    # For no-enemy rows, log_softmax gives log(1/N) which is fine as a fallback
    charge_lp = charge_log_probs.gather(1, charge_indices.unsqueeze(1)).squeeze(1)

    # Batched continuous sampling + per-sample post-move + batched shoot head
    unit_list = unit_indices.tolist()
    move_list = move_indices.tolist()
    charge_list = charge_indices.tolist()
    values = model.value_head(h, round_onehot)

    # --- Batched VonMises sampling ---
    raw_sin = direction_raw[:, 0]
    raw_cos = direction_raw[:, 1]
    log_conc = direction_raw[:, 2]
    dir_norm = torch.sqrt(raw_sin * raw_sin + raw_cos * raw_cos).clamp(min=1e-6)
    mean_angles = torch.atan2(raw_sin / dir_norm, raw_cos / dir_norm)
    concentrations = (F.softplus(log_conc) + 0.1).clamp(max=80.0)
    vm_batch = torch.distributions.VonMises(mean_angles, concentrations)
    sampled_angles_t = vm_batch.sample()
    # Stable VonMises log-prob using exponentially-scaled Bessel i0e
    i0e_c = torch.special.i0e(concentrations)
    log_i0_c = concentrations + torch.log(i0e_c.clamp(min=1e-20))
    dir_lps_t = concentrations * torch.cos(sampled_angles_t - mean_angles) - (math.log(2.0 * math.pi) + log_i0_c)

    # --- Batched Beta sampling ---
    alphas = (F.softplus(distance_raw[:, 0]) + 1.01).clamp(max=100.0)
    beta_vals = (F.softplus(distance_raw[:, 1]) + 1.01).clamp(max=100.0)
    beta_batch = torch.distributions.Beta(alphas, beta_vals)
    sampled_fracs_t = beta_batch.sample()
    dist_lps_t = beta_batch.log_prob(sampled_fracs_t.clamp(1e-4, 1.0 - 1e-4)).clamp(-20.0, 20.0)

    sampled_angles = sampled_angles_t.tolist()
    sampled_fracs = sampled_fracs_t.tolist()
    dir_lps = dir_lps_t.tolist()
    dist_lps = dist_lps_t.tolist()

    # --- Per-sample post-move positions + build post_move_rel tensors ---
    post_move_rels: list[list[float]] = []
    pmr_tensors: list[torch.Tensor] = []
    for i in range(n):
        req = requests[i]
        uid = unit_list[i]
        mt = move_list[i]
        sa = sampled_angles[i]
        sf = sampled_fracs[i]

        ucx, ucy = req.friendly_positions[uid]
        if mt == MOVE_ADVANCE:
            budget = req.advance_distances[uid]
            px, py = compute_post_move_position(ucx, ucy, sa, sf * budget)
        elif mt == MOVE_RUSH:
            budget = req.rush_distances[uid]
            px, py = compute_post_move_position(ucx, ucy, sa, sf * budget)
        else:
            px, py = ucx, ucy

        pmr = compute_post_move_rel(px, py, req.enemy_positions)
        post_move_rels.append(pmr.tolist())
        pmr_tensors.append(pmr)

    # --- Batched shoot target head (with in-range masking) ---
    pmr_batch = torch.stack(pmr_tensors)  # (N, 30)
    shoot_input_batch = torch.cat([h, unit_features, move_onehot, pmr_batch], dim=-1)
    shoot_logits_batch = model.shoot_target_head(shoot_input_batch)  # (N, 10)

    # Build per-sample max weapon range for the selected unit
    max_wr_list = [requests[i].max_weapon_ranges[unit_list[i]] for i in range(n)]
    max_wr_t = torch.tensor(max_wr_list, dtype=torch.float32)
    shoot_mask_batch = compute_in_range_mask_batched(pmr_batch, max_wr_t, enemy_alive_batch)

    shoot_logits_batch = shoot_logits_batch.masked_fill(~shoot_mask_batch, float('-inf'))
    no_shootable = ~shoot_mask_batch.any(dim=-1)  # (N,) — no enemies in range
    shoot_logits_batch = shoot_logits_batch.masked_fill(no_shootable.unsqueeze(-1), 0.0)
    shoot_logits_batch = torch.nan_to_num(shoot_logits_batch, nan=0.0, posinf=50.0, neginf=-50.0)

    # Handle no-shootable case
    shoot_probs_batch = torch.softmax(shoot_logits_batch, dim=-1)
    if no_shootable.any():
        uniform_s = torch.full_like(shoot_probs_batch, 1.0 / n_units)
        shoot_probs_batch = torch.where(no_shootable.unsqueeze(-1), uniform_s, shoot_probs_batch)
    shoot_indices_batch = torch.multinomial(shoot_probs_batch, 1).squeeze(-1)
    shoot_log_probs_batch = torch.log_softmax(shoot_logits_batch, dim=-1)
    shoot_lps_t = shoot_log_probs_batch.gather(1, shoot_indices_batch.unsqueeze(1)).squeeze(1)
    # Zero out log-prob for no-shootable samples
    shoot_lps_t = shoot_lps_t.masked_fill(no_shootable, 0.0)

    shoot_indices_list = shoot_indices_batch.tolist()
    shoot_lps = shoot_lps_t.tolist()
    shoot_mask_lists = shoot_mask_batch.tolist()
    rankings_list = [
        (list(range(n_units)) if no_shootable[i] else
         torch.argsort(shoot_logits_batch[i], descending=True).tolist())
        for i in range(n)
    ]

    # Compute total log-probs per sample
    lp_list = unit_lp.tolist()
    val_list = values.tolist()

    results = []
    for i in range(n):
        mt = move_list[i]
        total_lp = lp_list[i] + move_lp[i].item()
        if mt in (MOVE_ADVANCE, MOVE_RUSH):
            total_lp += dir_lps[i] + dist_lps[i]
        if mt in (MOVE_HOLD, MOVE_ADVANCE):
            total_lp += shoot_lps[i]
        if mt == MOVE_CHARGE:
            total_lp += charge_lp[i].item()

        results.append(_TacticalSamplingResult(
            unit_idx=unit_list[i],
            move_type=mt,
            sampled_angle=sampled_angles[i],
            sampled_distance_frac=sampled_fracs[i],
            charge_target_idx=charge_list[i],
            shoot_target_idx=shoot_indices_list[i],
            target_ranking=rankings_list[i],
            post_move_rel=post_move_rels[i],
            old_log_prob=total_lp,
            value=val_list[i],
            shoot_mask=shoot_mask_lists[i],
        ))
    return results


# ---------------------------------------------------------------------------
# Baseline (exponential moving average)
# ---------------------------------------------------------------------------

class EMABaseline:
    """Exponential moving average baseline for advantage computation."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.value = 0.0
        self._initialized = False

    def update(self, game_reward: float) -> None:
        if not self._initialized:
            self.value = game_reward
            self._initialized = True
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * game_reward

    def get(self) -> float:
        return self.value


# ---------------------------------------------------------------------------
# Opponent scheduling
# ---------------------------------------------------------------------------

def get_heuristic_fraction(win_rate: float) -> float:
    """Determine heuristic opponent fraction from rolling win rate (§5.6)."""
    if win_rate < 0.55:
        return 0.50
    elif win_rate <= 0.65:
        return 0.30
    else:
        return 0.20


# ---------------------------------------------------------------------------
# Checkpoint pool
# ---------------------------------------------------------------------------

def _make_model(model_type: str = "tactical") -> nn.Module:
    """Create a fresh model instance."""
    return TacticalModel()


def load_model_state_dict(path) -> dict:
    """Load a model state dict from a checkpoint file.

    Handles both the legacy format (raw state_dict) and the new format
    (dict with 'model_state_dict' and 'batch_num' keys).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


class CheckpointPool:
    """Manages a pool of past model checkpoints for self-play (§5.6)."""

    def __init__(self, max_size: int = 20, save_dir: str = "ml_checkpoints",
                 model_type: str = "tactical",
                 seed_existing: int = 0):
        self.max_size = max_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[Path] = []
        self.model_type = model_type

        # Optionally seed the pool with the N newest existing checkpoints
        if seed_existing > 0:
            existing = sorted(
                self.save_dir.glob("checkpoint_batch_*.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            for p in existing[-seed_existing:]:
                self.entries.append(p)

    def save(self, model: nn.Module, batch_num: int) -> None:
        """Save a checkpoint and add to pool, evicting oldest if full."""
        path = self.save_dir / f"checkpoint_batch_{batch_num:06d}.pt"
        torch.save(model.state_dict(), path)
        self.entries.append(path)
        # Evict oldest if over capacity
        while len(self.entries) > self.max_size:
            old = self.entries.pop(0)
            if old.exists():
                old.unlink()

    def sample_opponent(self) -> nn.Module | None:
        """Load a random checkpoint as an opponent. Returns None if pool is empty."""
        if not self.entries:
            return None
        path = random.choice(self.entries)
        if not path.exists():
            self.entries.remove(path)
            return None
        opponent = _make_model(self.model_type)
        opponent.load_state_dict(torch.load(path, weights_only=True), strict=False)
        opponent.eval()
        return opponent

    def sample_opponent_state_dict(self) -> dict | None:
        """Load a random checkpoint's state_dict (for passing to worker processes)."""
        if not self.entries:
            return None
        path = random.choice(self.entries)
        if not path.exists():
            self.entries.remove(path)
            return None
        return torch.load(path, weights_only=True)

    def sample_opponent_path(self) -> Path | None:
        """Return a random checkpoint path (without loading). Returns None if empty."""
        if not self.entries:
            return None
        path = random.choice(self.entries)
        if not path.exists():
            self.entries.remove(path)
            return None
        return path

    def load_state_dict(self, path: Path) -> dict:
        """Load a checkpoint's state_dict from the given path."""
        return torch.load(path, weights_only=True)

    def __len__(self) -> int:
        return len(self.entries)


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

    Args is (opp_slot_map, game_specs, shaping_scale) where opp_slot_map maps
    opp_sd_index -> index into _g_shared_opponents (or absent for heuristic).
    shaping_scale controls the per-round reward shaping magnitude (1.0 = full,
    0.0 = off).

    Returns list of (trajectory_rounds, result, opponent_type, army_type).
    """
    opp_slot_map, game_specs, shaping_scale = args

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

    return _run_games_batched_tactical(model, game_specs, opp_models,
                                       shaping_scale=shaping_scale)


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
            opponent_type, BOARD_OBJECTIVES,
        )
        results.append((traj_a, result, opp_t, army_type))
        if traj_b is not None:
            results.append((traj_b, result, "mirror_b", army_type))
    return results
def _run_single_episode_tactical(model, opponent_model, res_a, res_b,
                                  states_a_data, states_b_data, opponent_type,
                                  BOARD_OBJECTIVES, shaping_scale=1.0):
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
                state_vec_list = state_vec.tolist()

                # Compute model-space positions for sampling
                a_friendly_pos = _get_model_space_positions(units_a, "A")
                a_enemy_pos = _get_model_space_positions(units_b, "A")
                a_adv_dists, a_rush_dists = _get_movement_budgets(units_a)
                a_max_wr = _get_max_weapon_ranges(units_a)

                # Sample actions with sequential conditioning
                (sel_idx, move_type_a, sampled_angle_a, sampled_frac_a,
                 charge_tgt_a, shoot_tgt_a, _a_tac_target_ranking,
                 pmr_a, old_lp, value_est, shoot_mask_a) = sample_tactical_actions_no_grad(
                    model, state_vec, alive_mask, enemy_alive_mask,
                    a_friendly_pos, a_enemy_pos, a_adv_dists, a_rush_dists,
                    a_max_wr,
                )

                # Record for PPO replay
                step = TacticalActivationRecord(
                    state_vec=state_vec_list,
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
                )
                round_step_indices.append(len(trajectory))
                trajectory.append(step)

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
                        b_state_vec_list = b_state_vec.tolist()

                        b_friendly_pos = _get_model_space_positions(units_b, "B")
                        b_enemy_pos = _get_model_space_positions(units_a, "B")
                        b_adv_dists, b_rush_dists = _get_movement_budgets(units_b)
                        b_max_wr = _get_max_weapon_ranges(units_b)

                        (sel_b, mt_b, sa_b, sf_b, ct_b, st_b,
                         _b_target_ranking, pmr_b, olp_b, val_b, sm_b) = sample_tactical_actions_no_grad(
                            model, b_state_vec, b_alive_mask, b_enemy_alive_mask,
                            b_friendly_pos, b_enemy_pos, b_adv_dists, b_rush_dists,
                            b_max_wr,
                        )

                        step_b = TacticalActivationRecord(
                            state_vec=b_state_vec_list,
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
                        )
                        round_step_indices_b.append(len(trajectory_b))
                        trajectory_b.append(step_b)

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

            current_is_a = not current_is_a

        # End of round: update objectives
        board.update_objectives(units_a, units_b)

        # Assign round reward to the last activation step of this round
        reward, prev_a_kill_pts, prev_b_kill_pts = compute_round_reward(
            units_a, units_b, board, "A", pts_a,
            prev_a_kill_pts, prev_b_kill_pts,
            shaping_scale=shaping_scale,
            round_num=round_num,
        )
        if round_step_indices:
            trajectory[round_step_indices[-1]].reward = reward

        if is_mirror and round_step_indices_b:
            reward_b, prev_b_fkp, prev_b_ekp = compute_round_reward(
                units_b, units_a, board, "B", pts_b,
                prev_b_fkp, prev_b_ekp,
                shaping_scale=shaping_scale,
                round_num=round_num,
            )
            trajectory_b[round_step_indices_b[-1]].reward = reward_b

        # Aux targets: end-of-round survival fractions
        a_surv = _compute_survival_fracs(units_a)
        b_surv = _compute_survival_fracs(units_b)
        for si in round_step_indices:
            trajectory[si].friendly_survival_target = a_surv
            trajectory[si].enemy_survival_target = b_surv
        if is_mirror:
            for si in round_step_indices_b:
                trajectory_b[si].friendly_survival_target = b_surv
                trajectory_b[si].enemy_survival_target = a_surv

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

    # Aux targets: end-of-game objective control (backfill to all steps)
    obj_target_a = _compute_obj_control_target(board.objective_control, "A")
    for step in trajectory:
        step.obj_control_target = obj_target_a
    if is_mirror and trajectory_b:
        obj_target_b = _compute_obj_control_target(board.objective_control, "B")
        for step in trajectory_b:
            step.obj_control_target = obj_target_b

    return trajectory, result, opponent_type, trajectory_b


# ---------------------------------------------------------------------------
# Coroutine-batched tactical episode collection
# ---------------------------------------------------------------------------

def _episode_tactical_generator(opponent_model,
                                res_a, res_b,
                                states_a_data, states_b_data, opponent_type,
                                BOARD_OBJECTIVES, shaping_scale=1.0):
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
                state_vec_list = state_vec.tolist()

                # Compute model-space positions for inference
                a_friendly_pos = _get_model_space_positions(units_a, "A")
                a_enemy_pos = _get_model_space_positions(units_b, "A")
                a_adv_dists, a_rush_dists = _get_movement_budgets(units_a)
                a_max_wr = _get_max_weapon_ranges(units_a)

                # >>> YIELD for batched main-model inference <<<
                _inf_result = yield _TacticalInferenceRequest(
                    state_vec, alive_mask, enemy_alive_mask, "main",
                    a_friendly_pos, a_enemy_pos, a_adv_dists, a_rush_dists,
                    a_max_wr,
                )

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

                step = TacticalActivationRecord(
                    state_vec=state_vec_list,
                    alive_mask=alive_mask_list,
                    enemy_alive_mask=enemy_alive_mask_list,
                    unit_idx=sel_idx,
                    move_type=move_type_a,
                    sampled_angle=sampled_angle_a,
                    sampled_distance_frac=sampled_frac_a,
                    charge_target_idx=charge_tgt_a,
                    shoot_target_idx=shoot_tgt_a,
                    shoot_mask=_inf_result.shoot_mask,
                    post_move_rel=pmr_a,
                    old_log_prob=old_lp,
                    old_value=value_est,
                )
                round_step_indices.append(len(trajectory))
                trajectory.append(step)

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
                        b_state_vec_list = b_state_vec.tolist()

                        b_friendly_pos = _get_model_space_positions(units_b, "B")
                        b_enemy_pos = _get_model_space_positions(units_a, "B")
                        b_adv_dists, b_rush_dists = _get_movement_budgets(units_b)
                        b_max_wr = _get_max_weapon_ranges(units_b)

                        # >>> YIELD for batched main-model inference (mirror B) <<<
                        _b_inf = yield _TacticalInferenceRequest(
                            b_state_vec, b_alive_mask, b_enemy_alive_mask, "main",
                            b_friendly_pos, b_enemy_pos, b_adv_dists, b_rush_dists,
                            b_max_wr,
                        )

                        sel_b = _b_inf.unit_idx
                        if (sel_b < len(units_b)
                                and units_b[sel_b].models_alive > 0
                                and not units_b[sel_b].activated):
                            active = units_b[sel_b]
                            _b_target_ranking = _b_inf.target_ranking

                            step_b = TacticalActivationRecord(
                                state_vec=b_state_vec_list,
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
                            )
                            round_step_indices_b.append(len(trajectory_b))
                            trajectory_b.append(step_b)

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
            trajectory[round_step_indices[-1]].reward = reward

        if is_mirror and round_step_indices_b:
            reward_b, prev_b_fkp, prev_b_ekp = compute_round_reward(
                units_b, units_a, board, "B", pts_b,
                prev_b_fkp, prev_b_ekp,
                shaping_scale=shaping_scale,
                round_num=round_num,
            )
            trajectory_b[round_step_indices_b[-1]].reward = reward_b

        # Aux targets: end-of-round survival fractions
        a_surv = _compute_survival_fracs(units_a)
        b_surv = _compute_survival_fracs(units_b)
        for si in round_step_indices:
            trajectory[si].friendly_survival_target = a_surv
            trajectory[si].enemy_survival_target = b_surv
        if is_mirror:
            for si in round_step_indices_b:
                trajectory_b[si].friendly_survival_target = b_surv
                trajectory_b[si].enemy_survival_target = a_surv

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

    # Aux targets: end-of-game objective control (backfill to all steps)
    obj_target_a = _compute_obj_control_target(board.objective_control, "A")
    for step in trajectory:
        step.obj_control_target = obj_target_a
    if is_mirror and trajectory_b:
        obj_target_b = _compute_obj_control_target(board.objective_control, "B")
        for step in trajectory_b:
            step.obj_control_target = obj_target_b

    return trajectory, result, opponent_type, trajectory_b


def _run_games_batched_tactical(
    main_model: TacticalModel,
    game_specs: list,
    opp_models: dict,
    shaping_scale: float = 1.0,
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

        # Batch main model forward pass
        if main_reqs:
            batch_results = _batched_sample_tactical_no_grad(main_model, main_reqs)
            for gid, res in zip(main_gids, batch_results):
                all_results[gid] = res

        # Batch each opponent model separately
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

# ---------------------------------------------------------------------------
# Flat (vectorized) replay + loss for tactical model — avoids per-step objects
# ---------------------------------------------------------------------------

@dataclass
class FlatReplayResult:
    """Flat tensor output from batched tactical replay.

    Keeps everything as contiguous tensors so compute_loss_flat can operate
    without any Python-level per-step iteration.
    """
    log_probs: torch.Tensor    # (N,) — sum of log-probs across 5 heads
    entropies: torch.Tensor    # (N,) — mean entropy across active heads (for logging)
    values: torch.Tensor       # (N,) — value estimates
    n_episodes: int
    total_reward: float        # pre-computed sum of all rewards
    # Per-head entropies for entropy target tuning
    unit_entropies: torch.Tensor | None = None     # (N,)
    move_entropies: torch.Tensor | None = None     # (N,)
    dir_entropies: torch.Tensor | None = None      # (N,)
    dist_entropies: torch.Tensor | None = None     # (N,)
    charge_entropies: torch.Tensor | None = None   # (N,)
    shoot_entropies: torch.Tensor | None = None    # (N,)
    # Per-step masks for conditional heads
    is_adv_rush: torch.Tensor | None = None        # (N,) bool
    is_hold_adv: torch.Tensor | None = None        # (N,) bool
    is_charge: torch.Tensor | None = None          # (N,) bool
    alive_mask: torch.Tensor | None = None         # (N, 10)
    enemy_alive_mask: torch.Tensor | None = None   # (N, 10)
    shoot_mask: torch.Tensor | None = None         # (N, 10)
    # Auxiliary head outputs (None when aux heads not present on model)
    aux_friendly_surv_alpha: torch.Tensor | None = None   # (N, 10)
    aux_friendly_surv_beta: torch.Tensor | None = None    # (N, 10)
    aux_enemy_surv_alpha: torch.Tensor | None = None      # (N, 10)
    aux_enemy_surv_beta: torch.Tensor | None = None       # (N, 10)
    aux_obj_control_logits: torch.Tensor | None = None    # (N, 5, 3)


def replay_tactical_log_probs_flat(
    model: TacticalModel,
    all_trajectories: list[list[TacticalActivationRecord]],
) -> FlatReplayResult:
    """Replay trajectories through the v2 tactical model and compute log-probs.

    Returns flat (N,) tensors for direct use by compute_loss_flat.
    Handles mixed log-prob computation: discrete heads (unit, move_type,
    charge_target, shoot_target) + continuous heads (direction, distance).
    """
    flat_steps: list[TacticalActivationRecord] = []
    for traj in all_trajectories:
        flat_steps.extend(traj)

    n_steps = len(flat_steps)
    if n_steps == 0:
        return FlatReplayResult(
            log_probs=torch.zeros(0),
            entropies=torch.zeros(0),
            values=torch.zeros(0),
            n_episodes=len(all_trajectories),
            total_reward=0.0,
        )

    n_units = MAX_UNITS_PER_SIDE

    # Stack state vectors → (N, 2811) — use numpy intermediary for speed
    state_batch = torch.from_numpy(
        np.array([s.state_vec for s in flat_steps], dtype=np.float32))

    # Build alive masks → (N, 10) — vectorized via numpy
    alive_np = np.zeros((n_steps, n_units), dtype=np.bool_)
    enemy_alive_np = np.zeros((n_steps, n_units), dtype=np.bool_)
    for i, s in enumerate(flat_steps):
        n_a = min(n_units, len(s.alive_mask))
        alive_np[i, :n_a] = s.alive_mask[:n_a]
        n_e = min(n_units, len(s.enemy_alive_mask))
        enemy_alive_np[i, :n_e] = s.enemy_alive_mask[:n_e]
    alive_batch = torch.from_numpy(alive_np)
    enemy_alive_batch = torch.from_numpy(enemy_alive_np)

    # === Trunk ===
    h, u_attended, _attn_w, round_onehot = model.trunk(state_batch)  # (N, 512), (N, 20, 180), ..., (N, 4)
    if torch.isnan(h).any() or torch.isinf(h).any():
        print("  WARNING: NaN/Inf in trunk output during replay — clamping")
        h = torch.nan_to_num(h, nan=0.0, posinf=50.0, neginf=-50.0)

    # === Unit selection head ===
    unit_logits = model.unit_selection_head(h)                    # (N, 10)
    unit_logits = unit_logits.masked_fill(~alive_batch, float('-inf'))

    # === Extract unit features from attended embeddings ===
    unit_indices = torch.tensor([s.unit_idx for s in flat_steps], dtype=torch.long)
    unit_features = u_attended[:, :n_units, :].gather(
        1, unit_indices.unsqueeze(1).unsqueeze(2).expand(n_steps, 1, TACTICAL_UNIT_FEATURES),
    ).squeeze(1).detach()

    # === Move type head ===
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf)                      # (N, 4)

    # Conditioning: stored move_type → one-hot
    move_indices = torch.from_numpy(np.array([s.move_type for s in flat_steps], dtype=np.int64))
    move_onehot = F.one_hot(move_indices, NUM_MOVE_TYPES).float()

    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)  # (N, 272)

    # === Direction + Distance heads (raw outputs) ===
    direction_raw = model.direction_head(h_uf_m)                  # (N, 3)
    direction_raw = torch.nan_to_num(direction_raw, nan=0.0, posinf=50.0, neginf=-50.0)
    distance_raw = model.distance_head(h_uf_m)                    # (N, 2)
    distance_raw = torch.nan_to_num(distance_raw, nan=0.0, posinf=50.0, neginf=-50.0)

    # === Charge target head ===
    charge_logits = model.charge_target_head(h_uf_m)              # (N, 10)
    charge_logits = charge_logits.masked_fill(~enemy_alive_batch, float('-inf'))

    # === Shoot target head (with stored post-move features + shoot mask) ===
    post_move_rel_batch = torch.from_numpy(
        np.array([s.post_move_rel for s in flat_steps], dtype=np.float32))  # (N, 30)
    shoot_input = torch.cat([h, unit_features, move_onehot, post_move_rel_batch], dim=-1)
    shoot_logits = model.shoot_target_head(shoot_input)           # (N, 10)
    # Use stored shoot_mask (alive AND in-range) if available, else fall back to enemy_alive
    if hasattr(flat_steps[0], 'shoot_mask') and flat_steps[0].shoot_mask is not None:
        shoot_mask_batch = torch.tensor(
            [s.shoot_mask for s in flat_steps], dtype=torch.bool)
    else:
        shoot_mask_batch = enemy_alive_batch
    shoot_logits = shoot_logits.masked_fill(~shoot_mask_batch, float('-inf'))

    # === Value (round-conditioned) ===
    values = model.value_head(h, round_onehot)

    # === Log-probs & entropies ===
    eps = 1e-8

    # Unit selection — guard against all-dead rows (all -inf logits → NaN softmax)
    # and against NaN logits from diverged model weights
    all_dead = ~alive_batch.any(dim=1, keepdim=True)  # (N, 1)
    safe_unit_logits = unit_logits.masked_fill(all_dead, 0.0)   # uniform fallback
    safe_unit_logits = torch.nan_to_num(safe_unit_logits, nan=0.0, posinf=50.0, neginf=-50.0)
    unit_log_probs = torch.log_softmax(safe_unit_logits, dim=-1)
    unit_lp = unit_log_probs.gather(1, unit_indices.unsqueeze(1)).squeeze(1)
    unit_ent = torch.distributions.Categorical(logits=safe_unit_logits).entropy()

    # Move type
    move_logits = torch.nan_to_num(move_logits, nan=0.0, posinf=50.0, neginf=-50.0)
    move_dist = torch.distributions.Categorical(logits=move_logits)
    move_lp = move_dist.log_prob(move_indices)
    move_ent = move_dist.entropy()

    # Direction (von Mises log-prob) — vectorized
    stored_angles = torch.from_numpy(np.array([s.sampled_angle for s in flat_steps], dtype=np.float32))
    raw_sin = direction_raw[:, 0]
    raw_cos = direction_raw[:, 1]
    log_conc = direction_raw[:, 2]
    norm = torch.sqrt(raw_sin * raw_sin + raw_cos * raw_cos).clamp(min=1e-6)
    mean_angle = torch.atan2(raw_sin / norm, raw_cos / norm)
    conc = (F.softplus(log_conc) + 0.1).clamp(max=80.0)
    # VonMises log-prob: κ·cos(x - μ) - log(2π·I₀(κ))
    # Use exponentially-scaled Bessel functions for numerical stability:
    #   log I₀(κ) = κ + log(i0e(κ))  where i0e(κ) = I₀(κ)·e^(-κ)
    i0e_conc = torch.special.i0e(conc)
    log_i0 = conc + torch.log(i0e_conc.clamp(min=1e-20))
    log_norm = math.log(2.0 * math.pi) + log_i0
    dir_lp = conc * torch.cos(stored_angles - mean_angle) - log_norm
    # VonMises entropy: log(2π·I₀(κ)) - κ·I₁(κ)/I₀(κ)
    # I₁/I₀ = i1e(κ)/i0e(κ)  (exponential factors cancel)
    i1e_conc = torch.special.i1e(conc)
    ratio_i1_i0 = i1e_conc / i0e_conc.clamp(min=1e-10)
    dir_ent = log_norm - conc * ratio_i1_i0

    # Distance (Beta log-prob) — vectorized
    stored_fracs = torch.from_numpy(np.array([s.sampled_distance_frac for s in flat_steps], dtype=np.float32))
    alpha = (F.softplus(distance_raw[:, 0]) + 1.01).clamp(max=100.0)
    beta_val = (F.softplus(distance_raw[:, 1]) + 1.01).clamp(max=100.0)
    clamped_fracs = stored_fracs.clamp(1e-4, 1.0 - 1e-4)
    beta_dist = torch.distributions.Beta(alpha, beta_val)
    dist_frac_lp = beta_dist.log_prob(clamped_fracs).clamp(-20.0, 20.0)
    dist_frac_ent = beta_dist.entropy()

    # Charge target — guard against all-dead rows before softmax
    charge_indices = torch.from_numpy(np.array([s.charge_target_idx for s in flat_steps], dtype=np.int64))
    enemy_all_dead = ~enemy_alive_batch.any(dim=-1, keepdim=True)
    safe_charge_logits = charge_logits.masked_fill(enemy_all_dead, 0.0)
    safe_charge_logits = torch.nan_to_num(safe_charge_logits, nan=0.0, posinf=50.0, neginf=-50.0)
    charge_log_probs = torch.log_softmax(safe_charge_logits, dim=-1)
    charge_lp = charge_log_probs.gather(1, charge_indices.unsqueeze(1)).squeeze(1)
    charge_ent = torch.distributions.Categorical(logits=safe_charge_logits).entropy()

    # Shoot target — guard against no-shootable-target rows before softmax
    shoot_indices = torch.from_numpy(np.array([s.shoot_target_idx for s in flat_steps], dtype=np.int64))
    no_shootable = ~shoot_mask_batch.any(dim=-1, keepdim=True)
    safe_shoot_logits = shoot_logits.masked_fill(no_shootable, 0.0)
    safe_shoot_logits = torch.nan_to_num(safe_shoot_logits, nan=0.0, posinf=50.0, neginf=-50.0)
    shoot_log_probs = torch.log_softmax(safe_shoot_logits, dim=-1)
    shoot_lp = shoot_log_probs.gather(1, shoot_indices.unsqueeze(1)).squeeze(1)
    # Zero out log-prob for no-shootable rows to match collection path
    shoot_lp = shoot_lp.masked_fill(no_shootable.squeeze(-1), 0.0)
    shoot_ent = torch.distributions.Categorical(logits=safe_shoot_logits).entropy()

    # === Combine log-probs based on move type ===
    # Always: unit + move_type
    total_lp = unit_lp + move_lp
    total_ent = unit_ent + move_ent
    n_heads = torch.full((n_steps,), 2.0)  # count active heads for entropy averaging

    # Advance/rush: + direction + distance
    is_adv_rush = (move_indices == MOVE_ADVANCE) | (move_indices == MOVE_RUSH)
    total_lp = total_lp + torch.where(is_adv_rush, dir_lp + dist_frac_lp, torch.zeros_like(dir_lp))
    total_ent = total_ent + torch.where(is_adv_rush, dir_ent + dist_frac_ent, torch.zeros_like(dir_ent))
    n_heads = n_heads + torch.where(is_adv_rush, torch.tensor(2.0), torch.tensor(0.0))

    # Hold/advance: + shoot_target
    is_hold_adv = (move_indices == MOVE_HOLD) | (move_indices == MOVE_ADVANCE)
    total_lp = total_lp + torch.where(is_hold_adv, shoot_lp, torch.zeros_like(shoot_lp))
    total_ent = total_ent + torch.where(is_hold_adv, shoot_ent, torch.zeros_like(shoot_ent))
    n_heads = n_heads + torch.where(is_hold_adv, torch.tensor(1.0), torch.tensor(0.0))

    # Charge: + charge_target
    is_charge = move_indices == MOVE_CHARGE
    total_lp = total_lp + torch.where(is_charge, charge_lp, torch.zeros_like(charge_lp))
    total_ent = total_ent + torch.where(is_charge, charge_ent, torch.zeros_like(charge_ent))
    n_heads = n_heads + torch.where(is_charge, torch.tensor(1.0), torch.tensor(0.0))

    mean_ent = total_ent / n_heads.clamp(min=1.0)

    total_reward = sum(s.reward for s in flat_steps)

    # === Auxiliary prediction heads ===
    aux_fs_alpha = aux_fs_beta = aux_es_alpha = aux_es_beta = None
    aux_obj_logits = None
    if hasattr(model, 'aux_friendly_survival_head'):
        # Friendly survival: Beta(α, β) per unit — no +1 floor (allow bimodal)
        fs_raw = model.aux_friendly_survival_head(h).view(n_steps, n_units, 2)
        aux_fs_alpha = F.softplus(fs_raw[..., 0]) + 0.01   # (N, 10), > 0
        aux_fs_beta = F.softplus(fs_raw[..., 1]) + 0.01    # (N, 10), > 0

        # Enemy survival
        es_raw = model.aux_enemy_survival_head(h).view(n_steps, n_units, 2)
        aux_es_alpha = F.softplus(es_raw[..., 0]) + 0.01
        aux_es_beta = F.softplus(es_raw[..., 1]) + 0.01

        # Objective control: 5 objectives × 3 classes
        aux_obj_logits = model.aux_obj_control_head(h).view(n_steps, 5, 3)

    return FlatReplayResult(
        log_probs=total_lp,
        entropies=mean_ent,
        values=values,
        n_episodes=len(all_trajectories),
        total_reward=total_reward,
        # Per-head entropies
        unit_entropies=unit_ent,
        move_entropies=move_ent,
        dir_entropies=dir_ent,
        dist_entropies=dist_frac_ent,
        charge_entropies=charge_ent,
        shoot_entropies=shoot_ent,
        # Conditional head masks
        is_adv_rush=is_adv_rush,
        is_hold_adv=is_hold_adv,
        is_charge=is_charge,
        alive_mask=alive_batch,
        enemy_alive_mask=enemy_alive_batch,
        shoot_mask=shoot_mask_batch,
        # Auxiliary heads
        aux_friendly_surv_alpha=aux_fs_alpha,
        aux_friendly_surv_beta=aux_fs_beta,
        aux_enemy_surv_alpha=aux_es_alpha,
        aux_enemy_surv_beta=aux_es_beta,
        aux_obj_control_logits=aux_obj_logits,
    )


def compute_loss_flat(
    flat_result: FlatReplayResult,
    flat_old_log_probs: torch.Tensor,
    flat_advantages: torch.Tensor,
    flat_returns: torch.Tensor,
    clip_epsilon: float,
    value_coeff: float,
    entropy_coeff: float,
    aux_coeff: float = 0.0,
    aux_ratio: float = 0.2,
    flat_steps: list[TacticalActivationRecord] | None = None,
    entropy_tuner: EntropyTargetTuner | None = None,
) -> tuple[torch.Tensor, dict]:
    """Vectorized PPO loss — no Python loops over individual steps.

    All inputs are flat (N,) tensors aligned by step index.
    When aux_coeff > 0 and flat_steps is provided, computes auxiliary
    prediction losses (survival Beta NLL + objective control CE).

    If entropy_tuner is provided, uses per-head adaptive entropy targets
    instead of the flat entropy_coeff.  The alpha loss is computed and
    returned in metrics["alpha_loss"] for the caller to backprop separately.
    """
    n = flat_result.log_probs.shape[0]
    if n == 0:
        zero = torch.tensor(0.0)
        return zero, {
            "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
            "mean_entropy": 0.0, "mean_reward": 0.0, "aux_loss": 0.0,
            "alpha_loss": 0.0,
        }

    # Normalize advantages (zero-mean, unit-variance) for stable gradients
    adv_std, adv_mean = torch.std_mean(flat_advantages)
    if adv_std > 1e-8:
        flat_advantages = (flat_advantages - adv_mean) / adv_std

    # PPO clipped surrogate — fully vectorized
    # Clamp log-ratio to prevent exp() overflow (±5 → ratio in [~0.007, ~148])
    log_ratio = flat_result.log_probs - flat_old_log_probs
    # Diagnostic: print ratio stats on first call to help debug clip fraction issues
    if not hasattr(compute_loss_flat, '_diag_done'):
        compute_loss_flat._diag_done = True
        _lr = log_ratio.detach()
        _abs_lr = _lr.abs()
        _clipped = (_abs_lr > clip_epsilon).float().mean().item()
        print(f"  [DIAG] First-call log-ratio stats: "
              f"mean={_lr.mean().item():.4f} std={_lr.std().item():.4f} "
              f"min={_lr.min().item():.4f} max={_lr.max().item():.4f} "
              f"median={_lr.median().item():.4f} "
              f"|>0.5|={(_abs_lr > 0.5).float().mean().item():.3f} "
              f"|>1.0|={(_abs_lr > 1.0).float().mean().item():.3f} "
              f"|>2.0|={(_abs_lr > 2.0).float().mean().item():.3f} "
              f"clip_frac={_clipped:.3f}")
        # Show per-head breakdown
        _n = flat_result.log_probs.shape[0]
        _new_lps = flat_result.log_probs.detach()
        _old_lps = flat_old_log_probs.detach()
        print(f"  [DIAG] new_lp: mean={_new_lps.mean().item():.4f} std={_new_lps.std().item():.4f} "
              f"min={_new_lps.min().item():.4f} max={_new_lps.max().item():.4f}")
        print(f"  [DIAG] old_lp: mean={_old_lps.mean().item():.4f} std={_old_lps.std().item():.4f} "
              f"min={_old_lps.min().item():.4f} max={_old_lps.max().item():.4f}")
    log_ratio = log_ratio.clamp(-5.0, 5.0)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * flat_advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * flat_advantages
    mean_policy_loss = (-torch.min(surr1, surr2)).mean()

    # Clip fraction: how often the ratio was clipped
    clip_frac = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()

    # Value loss
    mean_value_loss = ((flat_result.values - flat_returns) ** 2).mean()

    # Entropy (aggregate + per-head)
    mean_entropy = flat_result.entropies.mean()
    per_head_entropy = {}
    if flat_result.unit_entropies is not None:
        for name, ent_t in [
            ("unit", flat_result.unit_entropies),
            ("move", flat_result.move_entropies),
            ("dir", flat_result.dir_entropies),
            ("dist", flat_result.dist_entropies),
            ("charge", flat_result.charge_entropies),
            ("shoot", flat_result.shoot_entropies),
        ]:
            per_head_entropy[name] = ent_t.mean().item() if ent_t is not None else 0.0
    alpha_loss_val = 0.0

    if entropy_tuner is not None and flat_result.unit_entropies is not None:
        # Per-head adaptive entropy bonus
        entropy_bonus = entropy_tuner.compute_entropy_bonus(
            flat_result.unit_entropies,
            flat_result.move_entropies,
            flat_result.dir_entropies,
            flat_result.dist_entropies,
            flat_result.charge_entropies,
            flat_result.shoot_entropies,
            flat_result.is_adv_rush,
            flat_result.is_hold_adv,
            flat_result.is_charge,
        )
        loss = mean_policy_loss + value_coeff * mean_value_loss - entropy_bonus

        # Alpha loss (caller backprops this separately through the alpha optimizer)
        alpha_loss = entropy_tuner.compute_alpha_loss(
            flat_result.unit_entropies,
            flat_result.move_entropies,
            flat_result.dir_entropies,
            flat_result.dist_entropies,
            flat_result.charge_entropies,
            flat_result.shoot_entropies,
            flat_result.is_adv_rush,
            flat_result.is_hold_adv,
            flat_result.is_charge,
            flat_result.alive_mask,
            flat_result.enemy_alive_mask,
            flat_result.shoot_mask,
        )
        alpha_loss_val = alpha_loss.item()
    else:
        # Legacy single-coefficient path
        loss = mean_policy_loss + value_coeff * mean_value_loss - entropy_coeff * mean_entropy
        alpha_loss = None

    # --- Auxiliary prediction losses (adaptive coefficient) ---
    aux_loss_val = 0.0
    effective_aux_coeff = 0.0
    if (aux_coeff > 0
            and flat_steps is not None
            and flat_result.aux_friendly_surv_alpha is not None):
        _aux_loss = _compute_aux_loss(flat_result, flat_steps)
        if _aux_loss is not None:
            aux_loss_val = _aux_loss.item()
            policy_mag = abs(loss.item())
            raw_aux_mag = abs(aux_loss_val)
            # Scale so aux contributes at most aux_ratio of policy magnitude
            effective_aux_coeff = aux_ratio * policy_mag / max(raw_aux_mag, 1e-6)
            effective_aux_coeff = min(effective_aux_coeff, aux_coeff)
            loss = loss + effective_aux_coeff * _aux_loss

    weighted_aux = effective_aux_coeff * aux_loss_val
    non_aux_loss = loss.item() - weighted_aux
    metrics = {
        "loss": loss.item(),
        "policy_loss": mean_policy_loss.item(),
        "value_loss": mean_value_loss.item(),
        "mean_entropy": mean_entropy.item(),
        "mean_reward": flat_result.total_reward / max(flat_result.n_episodes, 1),
        "aux_loss": aux_loss_val,
        "weighted_aux": weighted_aux,
        "non_aux_loss": non_aux_loss,
        "clip_frac": clip_frac,
        "per_head_entropy": per_head_entropy,
        "alpha_loss": alpha_loss_val,
        "_alpha_loss_tensor": alpha_loss,  # for backprop (not serialized)
    }
    return loss, metrics


def _compute_aux_loss(
    flat_result: FlatReplayResult,
    flat_steps: list[TacticalActivationRecord],
) -> torch.Tensor | None:
    """Compute combined auxiliary prediction loss from survival + objective heads.

    Returns a scalar tensor, or None if no valid targets are available.
    """
    n = len(flat_steps)
    n_units = MAX_UNITS_PER_SIDE
    losses: list[torch.Tensor] = []

    # --- Survival losses (Beta NLL) ---
    # Build target tensors; skip steps without targets (shouldn't happen, but be safe)
    fs_targets = []
    es_targets = []
    valid_surv = []
    for i, s in enumerate(flat_steps):
        if s.friendly_survival_target is not None and s.enemy_survival_target is not None:
            fs_targets.append(s.friendly_survival_target)
            es_targets.append(s.enemy_survival_target)
            valid_surv.append(i)

    if valid_surv:
        idx = torch.tensor(valid_surv, dtype=torch.long)
        fs_t = torch.tensor(fs_targets, dtype=torch.float32)  # (M, 10)
        es_t = torch.tensor(es_targets, dtype=torch.float32)  # (M, 10)

        # Clamp targets to (eps, 1-eps) for Beta log-prob
        eps = 1e-3
        fs_t = fs_t.clamp(eps, 1.0 - eps)
        es_t = es_t.clamp(eps, 1.0 - eps)

        # Friendly survival Beta NLL
        fs_alpha = flat_result.aux_friendly_surv_alpha[idx]  # (M, 10)
        fs_beta = flat_result.aux_friendly_surv_beta[idx]
        fs_dist = torch.distributions.Beta(fs_alpha.clamp(max=100.0), fs_beta.clamp(max=100.0))
        fs_nll = -fs_dist.log_prob(fs_t).mean()
        losses.append(fs_nll)

        # Enemy survival Beta NLL
        es_alpha = flat_result.aux_enemy_surv_alpha[idx]
        es_beta = flat_result.aux_enemy_surv_beta[idx]
        es_dist = torch.distributions.Beta(es_alpha.clamp(max=100.0), es_beta.clamp(max=100.0))
        es_nll = -es_dist.log_prob(es_t).mean()
        losses.append(es_nll)

    # --- Objective control loss (cross-entropy) ---
    obj_targets = []
    valid_obj = []
    for i, s in enumerate(flat_steps):
        if s.obj_control_target is not None:
            obj_targets.append(s.obj_control_target)
            valid_obj.append(i)

    if valid_obj and flat_result.aux_obj_control_logits is not None:
        idx = torch.tensor(valid_obj, dtype=torch.long)
        obj_t = torch.tensor(obj_targets, dtype=torch.long)  # (M, 5)
        obj_logits = flat_result.aux_obj_control_logits[idx]  # (M, 5, 3)
        # Reshape for cross_entropy: (M*5, 3) vs (M*5,)
        obj_ce = F.cross_entropy(obj_logits.reshape(-1, 3), obj_t.reshape(-1))
        losses.append(obj_ce)

    if not losses:
        return None
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# GAE advantage estimation
# ---------------------------------------------------------------------------

def compute_gae(
    all_trajectories: list[list[TacticalActivationRecord]],
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[list[list[float]], list[list[float]]]:
    """Compute GAE advantages and returns for all episodes.

    Uses old_value from collection time (stored in TacticalActivationRecord).
    Returns (all_advantages, all_returns).
    """
    all_advantages: list[list[float]] = []
    all_returns: list[list[float]] = []

    for trajectory in all_trajectories:
        T = len(trajectory)
        advantages = [0.0] * T
        returns = [0.0] * T
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = trajectory[t + 1].old_value if t < T - 1 else 0.0
            delta = trajectory[t].reward + gamma * next_value - trajectory[t].old_value
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae
            returns[t] = last_gae + trajectory[t].old_value
        all_advantages.append(advantages)
        all_returns.append(returns)

    return all_advantages, all_returns


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Rolling metrics for monitoring training progress."""
    heuristic_results: deque = field(default_factory=lambda: deque(maxlen=200))
    selfplay_results: deque = field(default_factory=lambda: deque(maxlen=200))
    heuristic_hof_results: deque = field(default_factory=lambda: deque(maxlen=200))
    heuristic_hof_ml_results: deque = field(default_factory=lambda: deque(maxlen=200))
    heuristic_random_results: deque = field(default_factory=lambda: deque(maxlen=200))
    selfplay_hof_results: deque = field(default_factory=lambda: deque(maxlen=200))
    selfplay_hof_ml_results: deque = field(default_factory=lambda: deque(maxlen=200))
    selfplay_random_results: deque = field(default_factory=lambda: deque(maxlen=200))
    batch_logs: list[dict] = field(default_factory=list)

    def record_game(self, result: str, opponent_type: str, army_type: str = "random") -> None:
        win = 1.0 if result == "A" else (0.5 if result == "draw" else 0.0)
        if opponent_type == "heuristic":
            self.heuristic_results.append(win)
            if army_type == "hof":
                self.heuristic_hof_results.append(win)
            elif army_type == "hof_ml":
                self.heuristic_hof_ml_results.append(win)
            else:
                self.heuristic_random_results.append(win)
        else:
            self.selfplay_results.append(win)
            if army_type == "hof":
                self.selfplay_hof_results.append(win)
            elif army_type == "hof_ml":
                self.selfplay_hof_ml_results.append(win)
            else:
                self.selfplay_random_results.append(win)

    @property
    def heuristic_win_rate(self) -> float:
        if not self.heuristic_results:
            return 0.5
        return sum(self.heuristic_results) / len(self.heuristic_results)

    @property
    def selfplay_win_rate(self) -> float:
        if not self.selfplay_results:
            return 0.5
        return sum(self.selfplay_results) / len(self.selfplay_results)

    def _wr(self, dq: deque) -> float:
        if not dq:
            return 0.5
        return sum(dq) / len(dq)

    @property
    def heuristic_hof_win_rate(self) -> float:
        return self._wr(self.heuristic_hof_results)

    @property
    def heuristic_hof_ml_win_rate(self) -> float:
        return self._wr(self.heuristic_hof_ml_results)

    @property
    def heuristic_random_win_rate(self) -> float:
        return self._wr(self.heuristic_random_results)

    @property
    def selfplay_hof_win_rate(self) -> float:
        return self._wr(self.selfplay_hof_results)

    @property
    def selfplay_hof_ml_win_rate(self) -> float:
        return self._wr(self.selfplay_hof_ml_results)

    @property
    def selfplay_random_win_rate(self) -> float:
        return self._wr(self.selfplay_random_results)

    def log_batch(self, batch_num: int, loss_metrics: dict,
                  heuristic_fraction: float) -> dict:
        entry = {
            "batch": batch_num,
            "heuristic_win_rate": round(self.heuristic_win_rate, 4),
            "selfplay_win_rate": round(self.selfplay_win_rate, 4),
            "heuristic_hof_wr": round(self.heuristic_hof_win_rate, 4),
            "heuristic_hof_ml_wr": round(self.heuristic_hof_ml_win_rate, 4),
            "heuristic_random_wr": round(self.heuristic_random_win_rate, 4),
            "selfplay_hof_wr": round(self.selfplay_hof_win_rate, 4),
            "selfplay_hof_ml_wr": round(self.selfplay_hof_ml_win_rate, 4),
            "selfplay_random_wr": round(self.selfplay_random_win_rate, 4),
            "heuristic_fraction": round(heuristic_fraction, 2),
            **{k: round(v, 6) for k, v in loss_metrics.items() if isinstance(v, (int, float))},
        }
        self.batch_logs.append(entry)
        return entry


# ---------------------------------------------------------------------------
# Army list helpers for training
# ---------------------------------------------------------------------------

def _load_hof_armies_from_file(filename: str) -> list:
    """Load army lists from results/<filename>.

    Returns a list of ArmyList objects, or an empty list if the file
    is missing or the evolution module is unavailable.
    """
    if not _HAS_EVOLUTION:
        return []
    try:
        from evolution import make_entry
        from models import ArmyList
        hof_path = Path(__file__).resolve().parent / "results" / filename
        if not hof_path.exists():
            return []
        import json
        with open(hof_path) as f:
            hof_data = json.load(f)
        armies = []
        for entry_data in hof_data:
            army = ArmyList()
            for e in entry_data["entries"]:
                entry = make_entry(
                    e["template_id"],
                    upgrades=e.get("upgrades", {}),
                    ai_role=e.get("ai_role", "killer"),
                )
                entry.combat_preference = e.get("combat_preference", "ranged")
                army.entries.append(entry)
            armies.append(army)
        return armies
    except Exception:
        return []


def _load_hof_armies() -> list:
    """Load army lists from results/hall_of_fame.json."""
    return _load_hof_armies_from_file("hall_of_fame.json")


def _load_hof_ml_armies() -> list:
    """Load army lists from results/hall_of_fame_ml.json."""
    return _load_hof_armies_from_file("hall_of_fame_ml.json")


def _generate_army_pair(
    opp_type: str = "heuristic",
    hof_armies: list | None = None,
    hof_ml_armies: list | None = None,
) -> tuple[list[ResolvedUnit], list[ResolvedUnit],
           list[UnitState], list[UnitState], str]:
    """Generate a pair of armies for a training game.

    Army selection depends on opponent type:

    **vs heuristic** (player B is heuristic):
    - Player B (heuristic) always gets a hall_of_fame.json list.
    - Player A (ML) gets a hall_of_fame.json list 50% / hall_of_fame_ml.json 50%.
    - Falls back to random if the required HoF files are unavailable.

    **vs selfplay** (both players are ML):
    - Both players get the same list *type*: random 50%, hall_of_fame.json 25%,
      hall_of_fame_ml.json 25%.
    - Falls back to random when a selected HoF source is unavailable.

    Returns (resolved_a, resolved_b, states_a, states_b, army_type).
    army_type is "hof", "hof_ml", or "random".
    """
    if not _HAS_EVOLUTION:
        raise RuntimeError(
            "evolution module not available — cannot generate random armies. "
            "Use run_training_batch() with pre-built armies instead."
        )

    if opp_type == "heuristic":
        # Player B (heuristic) always from hall_of_fame.json
        if hof_armies:
            army_b = random.choice(hof_armies)
        else:
            army_b = generate_random_army(mode="objectives")

        # Player A (ML): 50% hall_of_fame.json, 50% hall_of_fame_ml.json
        if random.random() < 0.5:
            if hof_armies:
                army_a = random.choice(hof_armies)
                army_type = "hof"
            else:
                army_a = generate_random_army(mode="objectives")
                army_type = "random"
        else:
            if hof_ml_armies:
                army_a = random.choice(hof_ml_armies)
                army_type = "hof_ml"
            else:
                army_a = generate_random_army(mode="objectives")
                army_type = "random"
    else:
        # Self-play: both get same type — random 50%, hof 25%, hof_ml 25%
        roll = random.random()
        if roll < 0.5:
            army_a = generate_random_army(mode="objectives")
            army_b = generate_random_army(mode="objectives")
            army_type = "random"
        elif roll < 0.75:
            if hof_armies:
                army_a = random.choice(hof_armies)
                army_b = random.choice(hof_armies)
                army_type = "hof"
            else:
                army_a = generate_random_army(mode="objectives")
                army_b = generate_random_army(mode="objectives")
                army_type = "random"
        else:
            if hof_ml_armies:
                army_a = random.choice(hof_ml_armies)
                army_b = random.choice(hof_ml_armies)
                army_type = "hof_ml"
            else:
                army_a = generate_random_army(mode="objectives")
                army_b = generate_random_army(mode="objectives")
                army_type = "random"

    res_a = resolve_army(army_a)
    res_b = resolve_army(army_b)
    states_a = _make_unit_states(army_a, res_a, "A")
    states_b = _make_unit_states(army_b, res_b, "B")
    return res_a, res_b, states_a, states_b, army_type


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(
    config: TrainingConfig | None = None,
    army_pairs: list[tuple[list[ResolvedUnit], list[ResolvedUnit]]] | None = None,
    verbose: bool = True,
    restart: bool = False,
) -> tuple[nn.Module, TrainingMetrics]:
    """Run the full PPO training loop.

    Parameters
    ----------
    config : training hyperparameters (uses defaults if None)
    army_pairs : optional fixed set of (army_a, army_b) tuples.
                 If None, generates random armies each game (requires evolution module).
    verbose : print progress to stdout

    Returns
    -------
    (trained_model, metrics)
    """
    if config is None:
        config = TrainingConfig()

    is_tactical = config.model_type == "tactical"
    device = _resolve_device(config.device)

    # Toggle C extension in main process
    import fast_core
    fast_core.USE_C_EXT = config.use_c_ext and fast_core.is_available()
    c_ext_label = "ON" if fast_core.USE_C_EXT else "OFF"

    # Load hall-of-fame armies for mixed training
    hof_armies = _load_hof_armies()
    hof_ml_armies = _load_hof_ml_armies()
    if verbose:
        print(f"Loaded {len(hof_armies)} HoF armies, {len(hof_ml_armies)} HoF-ML armies")

    if verbose:
        device_label = str(device)
        if device.type == "cuda":
            device_label += f" ({torch.cuda.get_device_name(device)})"
        print(f"Model type: {config.model_type} | C extension: {c_ext_label} | Device: {device_label}")

    model = _make_model(config.model_type)
    start_batch = 0
    if not restart:
        final_path = Path(config.checkpoint_dir) / "final_model.pt"
        if final_path.exists():
            ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
                start_batch = ckpt.get("batch_num", 0)
            else:
                # Legacy format: raw state_dict
                model.load_state_dict(ckpt, strict=False)
            if verbose:
                print(f"Resumed from {final_path} (batch {start_batch})")
        else:
            # No final_model.pt — try the newest checkpoint by creation time
            ckpt_dir = Path(config.checkpoint_dir)
            if ckpt_dir.exists():
                checkpoints = sorted(
                    ckpt_dir.glob("checkpoint_batch_*.pt"),
                    key=lambda p: p.stat().st_mtime,
                )
            else:
                checkpoints = []
            if checkpoints:
                newest = checkpoints[-1]
                ckpt = torch.load(newest, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                    start_batch = ckpt.get("batch_num", 0)
                else:
                    model.load_state_dict(ckpt, strict=False)
                if verbose:
                    print(f"No final_model.pt — resumed from newest checkpoint {newest.name} (batch {start_batch})")
            elif verbose:
                print("No final_model.pt or checkpoints found — starting from scratch")
    elif verbose:
        print("Restart requested — training from scratch")

    if restart:
        # Remove all previous checkpoint files
        ckpt_dir = Path(config.checkpoint_dir)
        if ckpt_dir.exists():
            existing = list(ckpt_dir.glob("checkpoint_batch_*.pt"))
            final = ckpt_dir / "final_model.pt"
            to_remove = len(existing) + (1 if final.exists() else 0)
            if to_remove > 0:
                answer = input(f"restart=True will DELETE {to_remove} checkpoint file(s) in {ckpt_dir}/. Continue? [y/N] ")
                if answer.strip().lower() != "y":
                    print("Aborted.")
                    raise SystemExit(1)
                for f in existing:
                    f.unlink()
                if final.exists():
                    final.unlink()
                if verbose:
                    print(f"Removed {to_remove} old checkpoint file(s)")

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Per-head entropy target tuner (tactical model only)
    entropy_tuner: EntropyTargetTuner | None = None
    alpha_optimizer: torch.optim.Adam | None = None
    if is_tactical and config.use_entropy_targets:
        entropy_tuner = EntropyTargetTuner(config).to(device)
        # Load tuner state if resuming
        if not restart:
            tuner_path = Path(config.checkpoint_dir) / "entropy_tuner.pt"
            if tuner_path.exists():
                entropy_tuner.load_state_dict(
                    torch.load(tuner_path, map_location="cpu", weights_only=True))
                if verbose:
                    print(f"Loaded entropy tuner from {tuner_path}")
        alpha_optimizer = torch.optim.Adam(
            entropy_tuner.parameters(), lr=config.entropy_alpha_lr)
        if verbose:
            print(f"Entropy targets: fraction={config.entropy_target_fraction}, "
                  f"move={config.entropy_target_move:.3f}, "
                  f"dir={config.entropy_target_dir:.3f}, "
                  f"dist={config.entropy_target_dist:.3f}")
    if verbose and config.ppo_minibatch_games > 0:
        print(f"PPO minibatching: {config.ppo_minibatch_games} games per minibatch, "
              f"{config.ppo_epochs} epochs")

    checkpoint_pool = CheckpointPool(
        max_size=config.max_checkpoints,
        save_dir=config.checkpoint_dir,
        model_type=config.model_type,
        seed_existing=0 if restart else 5,
    )
    if not restart and checkpoint_pool.entries and verbose:
        print(f"Seeded checkpoint pool with {len(checkpoint_pool.entries)} existing checkpoint(s)")
    metrics = TrainingMetrics()

    # Open training log CSV (append mode so it survives restarts)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_{config.model_type}.csv"
    _log_is_new = restart or not log_path.exists() or log_path.stat().st_size == 0
    _log_file = open(log_path, "w" if restart else "a", newline="")
    _log_writer = csv.writer(_log_file)
    if _log_is_new:
        _log_writer.writerow([
            "timestamp", "batch", "loss", "mean_entropy", "entropy_coeff",
            "mean_reward", "h_hof_wr", "h_ml_wr", "sp_hof_wr", "sp_ml_wr",
            "sp_rnd_wr", "h_frac", "batch_time", "aux_loss",
            "weighted_aux", "non_aux_loss", "clip_frac",
            "ent_unit", "ent_move", "ent_dir", "ent_dist",
            "ent_charge", "ent_shoot",
            "alpha_unit", "alpha_move", "alpha_dir", "alpha_dist",
            "alpha_charge", "alpha_shoot",
        ])
    _log_writer.writerow([datetime.now().isoformat(), "---",
                          f"Training started (start_batch={start_batch})",
                          "", "", "", "", "", "", "", "", "", ""])
    _log_file.flush()

    # Save initial checkpoint
    checkpoint_pool.save(model, 0)

    start_time = time.time()
    batch_times: list[float] = []

    # --- Shared-memory pool setup ---
    worker_count = config.worker_count if config.worker_count is not None else _WORKER_COUNT

    shared_model = _make_model(config.model_type)
    shared_model.share_memory()
    shared_model.eval()

    shared_opponents: list[nn.Module] = []
    for _ in range(_MAX_SHARED_OPPONENTS):
        m = _make_model(config.model_type)
        m.share_memory()
        m.eval()
        shared_opponents.append(m)

    ctx = _mp.get_context('spawn')
    pool = ctx.Pool(
        processes=worker_count,
        initializer=_init_shared_worker,
        initargs=(shared_model, shared_opponents, config.model_type,
                  config.use_c_ext),
    )

    for batch_num in range(start_batch + 1, start_batch + config.num_batches + 1):
        batch_start = time.time()
        heuristic_fraction = get_heuristic_fraction(metrics.heuristic_win_rate)

        # --- Phase 1: build game specs and deduplicate opponent weights ---
        # Copy current training weights to shared memory (map to CPU for workers)
        cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()} if device.type != "cpu" else model.state_dict()
        shared_model.load_state_dict(cpu_sd)

        game_specs = []           # per-game: (res_a, res_b, sa_data, sb_data, opp_type, opp_sd_index)
        opponent_state_dicts = [] # deduplicated list of opponent state dicts
        _opp_path_cache: dict[str, int] = {}  # checkpoint path -> index into opponent_state_dicts

        for game_idx in range(config.batch_size):
            # Select opponent
            if random.random() < heuristic_fraction:
                opp_type = "heuristic"
                opp_sd_idx = -1
            elif random.random() < 0.5:
                # Mirror self-play: current model plays both sides, learn from both
                opp_type = "selfplay_mirror"
                opp_sd_idx = -1
            else:
                opp_path = checkpoint_pool.sample_opponent_path()
                if opp_path is not None:
                    opp_type = "selfplay"
                    path_key = str(opp_path)
                    if path_key not in _opp_path_cache:
                        _opp_path_cache[path_key] = len(opponent_state_dicts)
                        opponent_state_dicts.append(
                            checkpoint_pool.load_state_dict(opp_path))
                    opp_sd_idx = _opp_path_cache[path_key]
                else:
                    # No checkpoints available yet, fall back to mirror
                    opp_type = "selfplay_mirror"
                    opp_sd_idx = -1

            # Generate or sample armies
            if army_pairs is not None:
                res_a, res_b = random.choice(army_pairs)
                states_a_data = [("killer", "ranged", -1)] * len(res_a)
                states_b_data = [("killer", "ranged", -1)] * len(res_b)
                army_type = "random"
            else:
                res_a, res_b, states_a, states_b, army_type = _generate_army_pair(
                    opp_type=opp_type, hof_armies=hof_armies,
                    hof_ml_armies=hof_ml_armies)
                states_a_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a]
                states_b_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b]

            game_specs.append((res_a, res_b, states_a_data, states_b_data, opp_type, opp_sd_idx, army_type))

        # --- Phase 2: load opponent weights into shared memory, dispatch ---
        # Copy unique opponent state dicts into shared opponent model slots
        opp_slot_map: dict[int, int] = {}
        for i, sd in enumerate(opponent_state_dicts):
            if i < _MAX_SHARED_OPPONENTS:
                shared_opponents[i].load_state_dict(sd, strict=False)
                opp_slot_map[i] = i

        # Compute reward shaping scale (anneals linearly to 0)
        # When resuming a prior run, disable shaping entirely — the model
        # has already graduated past the shaping phase.
        if not restart and start_batch > 0:
            shaping_scale = 0.0
        elif config.shaping_anneal_end > 0:
            shaping_progress = (batch_num - start_batch) / config.num_batches
            if config.time_limit is not None:
                elapsed_min = (time.time() - start_time) / 60.0
                shaping_progress = max(shaping_progress, elapsed_min / config.time_limit)
            shaping_progress = min(shaping_progress, 1.0)
            shaping_scale = max(0.0, 1.0 - shaping_progress / config.shaping_anneal_end)
        else:
            shaping_scale = 0.0

        n_chunks = worker_count
        chunk_size = max(1, len(game_specs) // n_chunks)
        chunks = []
        for i in range(0, len(game_specs), chunk_size):
            chunk = game_specs[i : i + chunk_size]
            chunks.append((opp_slot_map, chunk, shaping_scale))

        chunk_results = list(pool.map(_collect_episodes_shared_worker, chunks))
        trajectories = [ep for chunk in chunk_results for ep in chunk]

        # --- Phase 3: compute GAE advantages (fixed across PPO epochs) ---
        model.train()
        all_trajs = [traj_rounds for traj_rounds, _, _, _ in trajectories]
        all_advantages, all_returns = compute_gae(
            all_trajs, gamma=1.0, gae_lambda=config.gae_lambda,
        )

        # Record game outcomes for metrics tracking
        # Skip mirror_b entries — they share the same game as the A-side entry
        opp_types = []
        for _, result, opp_type, army_type in trajectories:
            if opp_type != "mirror_b":
                metrics.record_game(result, opp_type, army_type)
            opp_types.append(opp_type)

        # --- Phase 4: PPO multi-epoch update ---
        # Anneal entropy coefficient: linear from start to end
        progress2 = (batch_num - start_batch) / config.num_batches
        if config.time_limit is not None:
            elapsed_minutes = (time.time() - start_time) / 60.0
            progress1 = elapsed_minutes / config.time_limit
            progress = max(progress1, progress2)
        else:
            progress = progress2
        progress = min(progress, 1.0)
        entropy_coeff = config.entropy_coeff_start + progress * (config.entropy_coeff_end - config.entropy_coeff_start)

        # Pre-flatten advantages/returns/old_log_probs for vectorized tactical path
        if is_tactical:
            flat_old_lps = torch.tensor(
                [s.old_log_prob for traj in all_trajs for s in traj],
                dtype=torch.float32, device=device,
            )
            flat_advantages_t = torch.tensor(
                [a for adv in all_advantages for a in adv],
                dtype=torch.float32, device=device,
            )
            flat_returns_t = torch.tensor(
                [r for ret in all_returns for r in ret],
                dtype=torch.float32, device=device,
            )
            # Precompute per-game step counts and cumulative offsets for minibatching
            game_step_counts = [len(traj) for traj in all_trajs]
            game_step_offsets = [0] * len(all_trajs)
            for gi in range(1, len(all_trajs)):
                game_step_offsets[gi] = game_step_offsets[gi - 1] + game_step_counts[gi - 1]

        # Snapshot model weights before PPO epochs so we can rollback on NaN
        pre_ppo_state = {k: v.clone() for k, v in model.state_dict().items()}

        minibatch_games = config.ppo_minibatch_games
        nan_detected = False
        for _ppo_epoch in range(config.ppo_epochs):
            if is_tactical and minibatch_games > 0 and len(all_trajs) > minibatch_games:
                # --- Minibatched PPO for tactical model ---
                game_indices = list(range(len(all_trajs)))
                random.shuffle(game_indices)
                epoch_metrics: dict[str, float] = {}
                epoch_count = 0

                for mb_start in range(0, len(game_indices), minibatch_games):
                    mb_game_idx = game_indices[mb_start:mb_start + minibatch_games]

                    # Gather trajectories and flat tensor slices for this minibatch
                    mb_trajs = [all_trajs[i] for i in mb_game_idx]
                    mb_flat_idx: list[int] = []
                    for gi in mb_game_idx:
                        off = game_step_offsets[gi]
                        mb_flat_idx.extend(range(off, off + game_step_counts[gi]))
                    if not mb_flat_idx:
                        continue
                    idx_t = torch.tensor(mb_flat_idx, dtype=torch.long, device=device)
                    mb_old_lps = flat_old_lps[idx_t]
                    mb_advantages = flat_advantages_t[idx_t]
                    mb_returns = flat_returns_t[idx_t]

                    # Forward pass + loss on minibatch (device-aware)
                    with _force_tensor_device(device):
                        mb_flat_result = replay_tactical_log_probs_flat(model, mb_trajs)
                        mb_flat_steps = [s for traj in mb_trajs for s in traj]
                        loss, loss_metrics = compute_loss_flat(
                            mb_flat_result, mb_old_lps, mb_advantages, mb_returns,
                            config.clip_epsilon, config.value_coeff, entropy_coeff,
                            aux_coeff=config.aux_coeff, aux_ratio=config.aux_ratio,
                            flat_steps=mb_flat_steps,
                            entropy_tuner=entropy_tuner,
                        )

                    optimizer.zero_grad()
                    if alpha_optimizer is not None:
                        alpha_optimizer.zero_grad()
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  WARNING: NaN/Inf loss at batch {batch_num}, rolling back weights")
                        model.load_state_dict(pre_ppo_state)
                        nan_detected = True
                        break
                    loss.backward()

                    # Backprop alpha loss separately
                    alpha_loss_tensor = loss_metrics.pop("_alpha_loss_tensor", None)
                    if alpha_loss_tensor is not None and alpha_optimizer is not None:
                        alpha_loss_tensor.backward()
                        alpha_optimizer.step()

                    # Check for NaN in gradients before stepping
                    grad_nan = any(
                        p.grad is not None and torch.isnan(p.grad).any()
                        for p in model.parameters()
                    )
                    if grad_nan:
                        print(f"  WARNING: NaN gradients at batch {batch_num}, rolling back weights")
                        model.load_state_dict(pre_ppo_state)
                        nan_detected = True
                        break
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Accumulate metrics for logging (weighted by step count)
                    n_mb = len(mb_flat_idx)
                    for k, v in loss_metrics.items():
                        if isinstance(v, (int, float)):
                            epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v * n_mb
                    # Accumulate per-head entropy separately
                    phe_mb = loss_metrics.get("per_head_entropy", {})
                    for hk, hv in phe_mb.items():
                        epoch_metrics[f"_phe_{hk}"] = epoch_metrics.get(f"_phe_{hk}", 0.0) + hv * n_mb
                    epoch_count += n_mb

                if nan_detected:
                    break
                # Average metrics across minibatches
                if epoch_count > 0:
                    phe_agg = {}
                    non_phe = {}
                    for k, v in epoch_metrics.items():
                        if k.startswith("_phe_"):
                            phe_agg[k[5:]] = v / epoch_count
                        else:
                            non_phe[k] = v / epoch_count
                    loss_metrics = non_phe
                    loss_metrics["per_head_entropy"] = phe_agg

            elif is_tactical:
                # Full-batch tactical path (minibatch disabled or batch too small)
                with _force_tensor_device(device):
                    flat_result = replay_tactical_log_probs_flat(model, all_trajs)
                    _flat_steps = [s for traj in all_trajs for s in traj]
                    loss, loss_metrics = compute_loss_flat(
                        flat_result, flat_old_lps, flat_advantages_t, flat_returns_t,
                        config.clip_epsilon, config.value_coeff, entropy_coeff,
                        aux_coeff=config.aux_coeff, aux_ratio=config.aux_ratio,
                        flat_steps=_flat_steps,
                        entropy_tuner=entropy_tuner,
                    )

                optimizer.zero_grad()
                if alpha_optimizer is not None:
                    alpha_optimizer.zero_grad()
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  WARNING: NaN/Inf loss at batch {batch_num}, rolling back weights")
                    model.load_state_dict(pre_ppo_state)
                    nan_detected = True
                    break
                loss.backward()

                alpha_loss_tensor = loss_metrics.pop("_alpha_loss_tensor", None)
                if alpha_loss_tensor is not None and alpha_optimizer is not None:
                    alpha_loss_tensor.backward()
                    alpha_optimizer.step()

                grad_nan = any(
                    p.grad is not None and torch.isnan(p.grad).any()
                    for p in model.parameters()
                )
                if grad_nan:
                    print(f"  WARNING: NaN gradients at batch {batch_num}, rolling back weights")
                    model.load_state_dict(pre_ppo_state)
                    nan_detected = True
                    break
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # Log
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        batch_log = metrics.log_batch(batch_num, loss_metrics, heuristic_fraction)
        if verbose:
            recent_avg = sum(batch_times[-10:]) / len(batch_times[-10:])
            batches_remaining = (start_batch + config.num_batches) - batch_num
            est_remaining = recent_avg * batches_remaining
            if config.time_limit is not None:
                time_limit_remaining = config.time_limit * 60 - (time.time() - start_time)
                est_remaining = min(est_remaining, max(0, time_limit_remaining))
            eft = datetime.now() + timedelta(seconds=est_remaining)
            eft_str = eft.strftime("%H:%M")

            w_aux = loss_metrics.get('weighted_aux', 0.0)
            non_aux = loss_metrics.get('non_aux_loss', loss_metrics['loss'])
            aux_str = f" | Loss(policy={non_aux:.4f} aux={w_aux:.4f})" if w_aux != 0.0 else ""
            phe = loss_metrics.get('per_head_entropy', {})
            if entropy_tuner is not None:
                alphas = entropy_tuner.alpha_summary()
                ent_str = (f"Entropy: {loss_metrics['mean_entropy']:.4f} "
                           f"[u={phe.get('unit', 0):.3f} m={phe.get('move', 0):.3f} "
                           f"d={phe.get('dir', 0):.3f} D={phe.get('dist', 0):.3f} "
                           f"c={phe.get('charge', 0):.3f} s={phe.get('shoot', 0):.3f}] "
                           f"(α u={alphas['unit']:.3f} m={alphas['move']:.3f} "
                           f"d={alphas['dir']:.3f} D={alphas['dist']:.3f} "
                           f"c={alphas['charge']:.3f} s={alphas['shoot']:.3f})")
            else:
                ent_str = (f"Entropy: {loss_metrics['mean_entropy']:.4f} "
                           f"[u={phe.get('unit', 0):.3f} m={phe.get('move', 0):.3f} "
                           f"d={phe.get('dir', 0):.3f} D={phe.get('dist', 0):.3f} "
                           f"c={phe.get('charge', 0):.3f} s={phe.get('shoot', 0):.3f}] "
                           f"(c={entropy_coeff:.4f})")
            print(
                f"Batch {batch_num:04d} | "
                f"Loss: {loss_metrics['loss']:.4f} | "
                f"{ent_str} | "
                f"Reward: {loss_metrics['mean_reward']:.3f}{aux_str} | "
                f"H-HoF: {metrics.heuristic_hof_win_rate:.3f} | "
                f"H-ML: {metrics.heuristic_hof_ml_win_rate:.3f} | "
                f"SP-HoF: {metrics.selfplay_hof_win_rate:.3f} | "
                f"SP-ML: {metrics.selfplay_hof_ml_win_rate:.3f} | "
                f"SP-Rnd: {metrics.selfplay_random_win_rate:.3f} | "
                f"H-Frac: {heuristic_fraction:.2f} | "
                f"Clip: {loss_metrics.get('clip_frac', 0.0):.3f} | "
                f"{batch_time:.1f}s | EFT {eft_str}",
                flush=True,
            )

        # Log to CSV
        if entropy_tuner is not None:
            alphas = entropy_tuner.alpha_summary()
            alpha_cols = [f"{alphas[k]:.4f}" for k in EntropyTargetTuner.HEAD_NAMES]
        else:
            alpha_cols = [""] * 6
        _log_writer.writerow([
            datetime.now().isoformat(), batch_num,
            f"{loss_metrics['loss']:.4f}",
            f"{loss_metrics['mean_entropy']:.4f}",
            f"{entropy_coeff:.4f}",
            f"{loss_metrics['mean_reward']:.3f}",
            f"{metrics.heuristic_hof_win_rate:.3f}",
            f"{metrics.heuristic_hof_ml_win_rate:.3f}",
            f"{metrics.selfplay_hof_win_rate:.3f}",
            f"{metrics.selfplay_hof_ml_win_rate:.3f}",
            f"{metrics.selfplay_random_win_rate:.3f}",
            f"{heuristic_fraction:.2f}",
            f"{batch_time:.1f}",
            f"{loss_metrics.get('aux_loss', 0.0):.4f}",
            f"{loss_metrics.get('weighted_aux', 0.0):.4f}",
            f"{loss_metrics.get('non_aux_loss', 0.0):.4f}",
            f"{loss_metrics.get('clip_frac', 0.0):.4f}",
            *[f"{loss_metrics.get('per_head_entropy', {}).get(k, 0.0):.4f}"
              for k in ("unit", "move", "dir", "dist", "charge", "shoot")],
            *alpha_cols,
        ])
        _log_file.flush()

        # Checkpoint
        if batch_num % config.checkpoint_interval == 0:
            checkpoint_pool.save(model, batch_num)

        # Time limit check
        if config.time_limit is not None:
            elapsed = time.time() - start_time
            if elapsed >= config.time_limit * 60:
                if verbose:
                    print(f"\nTIME LIMIT reached ({config.time_limit} min) after batch {batch_num}.")
                break

    pool.close()
    pool.join()

    # Move model back to CPU for saving and downstream use
    model.to("cpu")
    if entropy_tuner is not None:
        entropy_tuner.to("cpu")

    # Close training log
    _log_writer.writerow([datetime.now().isoformat(), "---",
                          f"Training finished (batch {batch_num})",
                          "", "", "", "", "", "", "", "", "", ""])
    _log_file.close()

    # Save final model
    final_path = Path(config.checkpoint_dir) / "final_model.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "batch_num": batch_num}, final_path)

    # Save entropy tuner state (separate file for easy loading)
    if entropy_tuner is not None:
        tuner_path = Path(config.checkpoint_dir) / "entropy_tuner.pt"
        torch.save(entropy_tuner.state_dict(), tuner_path)

    return model, metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TrainingConfig(
        num_batches=10,
        batch_size=8,
        checkpoint_dir="ml_checkpoints_test",
    )
    model, metrics = run_training(config=cfg, verbose=True)
    print(f"\nTraining complete. Final heuristic win rate: {metrics.heuristic_win_rate:.3f}")
