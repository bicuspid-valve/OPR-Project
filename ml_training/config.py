"""Configuration, data structures, and device helpers for ML training."""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


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
    entropy_target_dest: float = 1.5                        # combined destination mixture: H(weights) + Σ w_k·H(component_k)
    entropy_alpha_lr: float = 1e-4                         # learning rate for entropy alpha params
    # Planning-augmented training (Expert Iteration)
    planning_rate: float = 0.0                # probability of planning per activation (0 = disabled)
    planning_rate_end: float | None = None    # if set, anneal planning_rate linearly to this value
    planning_distill_max_weight: float = 0.1  # max weight for distillation KL loss
    training_planning_K: int = 3              # candidate units (reduced from eval default 6)
    training_planning_C: int = 3              # action samples per unit (reduced from 4)
    training_planning_M: int = 4              # rollouts per candidate (same as eval)
    training_planning_N: int = 3              # lookahead activations (reduced from 4)
    # Unit-local advantage blending
    unit_local_advantage_blend: float = 0.0   # 0 = pure global GAE, 0.2-0.3 = blend in unit-local GAE
    # 4th destination component: objective proximity auxiliary loss
    dest_obj_proximity_coeff: float = 0.05    # weight for "move toward nearest objective" loss on 4th dest component


# ---------------------------------------------------------------------------
# Tactical model: per-activation data structures
# ---------------------------------------------------------------------------

@dataclass
class TacticalActivationRecord:
    """Serializable trajectory data for one activation (tactical v2 model).

    Stores the sequential decisions: unit, move_type, direction+distance
    (continuous), charge_target, shoot_target, plus masks and value.
    """
    state_vec: np.ndarray                 # flattened encoded state (4016 float32)
    alive_mask: list[bool]                # which friendly slots were alive+unactivated
    enemy_alive_mask: list[bool]          # which enemy slots were alive
    unit_idx: int                         # which unit was selected
    move_type: int                        # 0=hold, 1=advance, 2=rush, 3=charge
    sampled_angle: float                  # direction in radians (for advance/rush)
    sampled_distance_frac: float          # 0-1 fraction of movement budget (clamped, may exceed 1 before clamp)
    dest_component_idx: int               # which mixture component was selected
    charge_target_idx: int                # enemy slot for charge
    shoot_target_idx: int                 # enemy slot for shooting (hold/advance)
    shoot_mask: list[bool]                # enemy alive AND in weapon range (10 bools)
    post_move_rel: np.ndarray              # (30,) post-move relative features
    unit_cx: float = 0.0                  # model-space x of selected unit
    unit_cy: float = 0.0                  # model-space y of selected unit
    move_budget: float = 0.0              # advance/rush distance for selected unit (0 for hold/charge)
    reward: float = 0.0
    old_log_prob: float = 0.0             # sum of log-probs under collection policy
    old_value: float = 0.0                # value estimate under collection policy
    # Opponent conditioning (CTDE — value head only)
    opponent_type_idx: int = 0            # index into NUM_OPPONENT_TYPES
    # Auxiliary prediction targets — long-horizon (end-of-game)
    friendly_survival_target: list[float] | None = None  # 10 fracs, end-of-game
    enemy_survival_target: list[float] | None = None     # 10 fracs, end-of-game
    obj_control_target: list[int] | None = None          # 5 ints, end-of-game (0=friendly, 1=enemy, 2=neutral)
    # Auxiliary prediction targets — short-horizon (end-of-current-round)
    friendly_survival_target_short: list[float] | None = None  # 10 fracs, end-of-current-round
    enemy_survival_target_short: list[float] | None = None     # 10 fracs, end-of-current-round
    obj_control_target_short: list[int] | None = None          # 5 ints, end-of-current-round
    # Activation countdown targets (backfilled from trajectory)
    friendly_activations_remaining: float | None = None  # how many more friendly activations until game ends
    enemy_activations_remaining: float | None = None     # how many more enemy activations until game ends
    # Planning augmentation data (None for non-planned activations)
    was_planned: bool = False
    planning_improved: bool = False
    planning_value_delta: float = 0.0
    planning_unit_values: list[float] | None = None   # per-candidate-unit avg rollout values
    planning_unit_indices: list[int] | None = None     # which unit slots were evaluated
    # Sub-head distillation targets (per chosen unit's candidates)
    planning_move_values: list[float] | None = None    # per-move-type avg rollout values
    planning_move_indices: list[int] | None = None     # which move types were evaluated
    planning_charge_values: list[float] | None = None  # per-charge-target avg rollout values
    planning_charge_indices: list[int] | None = None   # which charge target slots
    planning_shoot_values: list[float] | None = None   # per-shoot-target avg rollout values
    planning_shoot_indices: list[int] | None = None    # which shoot target slots


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
    opponent_type_idx: int = 0                     # index into NUM_OPPONENT_TYPES (for value head conditioning)
    # Planning support — populated for Player A requests when planning is enabled.
    # These are references (same process), not copies.
    planning_units_a: list | None = None       # list[UnitState]
    planning_units_b: list | None = None       # list[UnitState]
    planning_board: object | None = None       # Board
    planning_round_num: int = 0
    planning_current_is_a: bool = True
    planning_fr_a: list | None = None
    planning_fm_a: list | None = None
    planning_fr_b: list | None = None
    planning_fm_b: list | None = None
    planning_pts_a: int = 0
    planning_pts_b: int = 0
    planning_opponent_type_idx: int = 0


@dataclass
class _TacticalSamplingResult:
    """Sent back to episode generator with batched sampling outputs."""
    unit_idx: int
    move_type: int              # 0-3
    sampled_angle: float        # radians
    sampled_distance_frac: float  # 0-1 (clamped from mixture)
    dest_component_idx: int     # which mixture component was selected
    charge_target_idx: int      # enemy slot
    shoot_target_idx: int       # enemy slot
    target_ranking: list        # shoot target ranking for compat
    post_move_rel: np.ndarray   # (30,) floats
    old_log_prob: float
    value: float
    shoot_mask: list[bool]      # enemy alive AND in weapon range (10 bools)
    # Planning metadata (populated by coordinator when planning was used)
    was_planned: bool = False
    planning_improved: bool = False
    planning_value_delta: float = 0.0
    planning_unit_values: list[float] | None = None
    planning_unit_indices: list[int] | None = None
    # Sub-head distillation targets
    planning_move_values: list[float] | None = None
    planning_move_indices: list[int] | None = None
    planning_charge_values: list[float] | None = None
    planning_charge_indices: list[int] | None = None
    planning_shoot_values: list[float] | None = None
    planning_shoot_indices: list[int] | None = None


# ---------------------------------------------------------------------------
# Opponent type mapping (for value head conditioning)
# ---------------------------------------------------------------------------

_OPP_TYPE_MAP = {
    "heuristic": 0,
    "selfplay_mirror": 1,
    "selfplay_hof": 2,
    "selfplay_ml": 3,
    "selfplay_random": 4,
}


def _get_opponent_type_idx(opp_type: str, army_type: str) -> int:
    """Map opponent type string + army type to an integer index.

    Parameters
    ----------
    opp_type : "heuristic", "selfplay_mirror", or "selfplay"
    army_type : "hof", "hof_ml", or "random" (distinguishes selfplay variants)
    """
    if opp_type == "heuristic":
        return 0
    if opp_type == "selfplay_mirror":
        return 1
    # For checkpoint-based self-play, use army_type to distinguish
    if army_type == "hof":
        return 2
    if army_type == "hof_ml":
        return 3
    return 4  # random
