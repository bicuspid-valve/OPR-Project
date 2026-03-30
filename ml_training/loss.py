"""Flat (vectorized) replay, PPO loss, auxiliary losses, and planning distillation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_features import MAX_UNITS_PER_SIDE, TACTICAL_UNIT_FEATURES, extract_can_charge_mask
from ml_model_tactical import (
    TacticalModel, NUM_MOVE_TYPES, MOVE_HOLD, MOVE_ADVANCE, MOVE_RUSH, MOVE_CHARGE,
    NUM_OPPONENT_TYPES,
)

from ml_training.config import TacticalActivationRecord
from ml_training.entropy import EntropyTargetTuner


# ---------------------------------------------------------------------------
# Flat replay result
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
    # Auxiliary head outputs — long-horizon (None when aux heads not present on model)
    aux_friendly_surv_alpha: torch.Tensor | None = None   # (N, 10)
    aux_friendly_surv_beta: torch.Tensor | None = None    # (N, 10)
    aux_enemy_surv_alpha: torch.Tensor | None = None      # (N, 10)
    aux_enemy_surv_beta: torch.Tensor | None = None       # (N, 10)
    aux_obj_control_logits: torch.Tensor | None = None    # (N, 5, 3)
    # Auxiliary head outputs — short-horizon (end-of-current-round)
    aux_friendly_surv_alpha_short: torch.Tensor | None = None   # (N, 10)
    aux_friendly_surv_beta_short: torch.Tensor | None = None    # (N, 10)
    aux_enemy_surv_alpha_short: torch.Tensor | None = None      # (N, 10)
    aux_enemy_surv_beta_short: torch.Tensor | None = None       # (N, 10)
    aux_obj_control_logits_short: torch.Tensor | None = None    # (N, 5, 3)
    # Activation countdown head outputs
    aux_friendly_act_remaining: torch.Tensor | None = None   # (N,) predicted friendly activations remaining
    aux_enemy_act_remaining: torch.Tensor | None = None       # (N,) predicted enemy activations remaining
    # Per-opponent-type mean value estimates (diagnostic)
    per_opp_type_mean_values: dict[str, float] | None = None
    # Logits for planning distillation loss
    unit_logits: torch.Tensor | None = None    # (N, 10) — raw logits after alive masking
    move_logits: torch.Tensor | None = None    # (N, 4) — move type logits conditioned on chosen unit
    charge_logits: torch.Tensor | None = None  # (N, 10) — charge target logits (masked)
    shoot_logits: torch.Tensor | None = None   # (N, 10) — shoot target logits (masked)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

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

    # Stack state vectors → (N, feat) — np.stack is fast when elements are arrays
    state_batch = torch.from_numpy(
        np.stack([s.state_vec for s in flat_steps]))

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
    h, units, round_onehot = model.trunk(state_batch)              # (N, 512), (N, 20, 200), (N, 4)
    if torch.isnan(h).any() or torch.isinf(h).any():
        print("  WARNING: NaN/Inf in trunk output during replay — clamping")
        h = torch.nan_to_num(h, nan=0.0, posinf=50.0, neginf=-50.0)

    # === Unit selection head ===
    unit_logits = model.unit_selection_head(h)                    # (N, 10)
    unit_logits = unit_logits.masked_fill(~alive_batch, float('-inf'))

    # === Extract unit features from unit embeddings ===
    unit_indices = torch.tensor([s.unit_idx for s in flat_steps], dtype=torch.long)
    unit_features = units[:, :n_units, :].gather(
        1, unit_indices.unsqueeze(1).unsqueeze(2).expand(n_steps, 1, TACTICAL_UNIT_FEATURES),
    ).squeeze(1).detach()

    # === Move type head ===
    # Extract can_charge mask for each sample's stored unit
    can_charge_batch = extract_can_charge_mask(state_batch, unit_indices)  # (N, 10)

    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf)                      # (N, 4)
    # Mask charge when no enemy is in charge range
    no_chargeable = ~can_charge_batch.any(dim=-1)                 # (N,)
    move_logits = move_logits.clone()
    move_logits[:, MOVE_CHARGE] = move_logits[:, MOVE_CHARGE].masked_fill(no_chargeable, float('-inf'))

    # Conditioning: stored move_type → one-hot
    move_indices = torch.from_numpy(np.array([s.move_type for s in flat_steps], dtype=np.int64))
    move_onehot = F.one_hot(move_indices, NUM_MOVE_TYPES).float()

    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)  # (N, 272)

    # === Direction + Distance heads (raw outputs) ===
    direction_raw = model.direction_head(h_uf_m)                  # (N, 3)
    direction_raw = torch.nan_to_num(direction_raw, nan=0.0, posinf=50.0, neginf=-50.0)
    distance_raw = model.distance_head(h_uf_m)                    # (N, 2)
    distance_raw = torch.nan_to_num(distance_raw, nan=0.0, posinf=50.0, neginf=-50.0)

    # === Charge target head — mask by alive AND chargeable ===
    charge_logits = model.charge_target_head(h_uf_m)              # (N, 10)
    charge_logits = charge_logits.masked_fill(~enemy_alive_batch, float('-inf'))
    charge_logits = charge_logits.masked_fill(~can_charge_batch, float('-inf'))

    # === Shoot target head (with stored post-move features + shoot mask) ===
    post_move_rel_batch = torch.from_numpy(
        np.stack([s.post_move_rel for s in flat_steps])).float()  # (N, 30)
    shoot_input = torch.cat([h, unit_features, move_onehot, post_move_rel_batch], dim=-1)
    shoot_logits = model.shoot_target_head(shoot_input)           # (N, 10)
    # Use stored shoot_mask (alive AND in-range) if available, else fall back to enemy_alive
    if hasattr(flat_steps[0], 'shoot_mask') and flat_steps[0].shoot_mask is not None:
        shoot_mask_batch = torch.tensor(
            [s.shoot_mask for s in flat_steps], dtype=torch.bool)
    else:
        shoot_mask_batch = enemy_alive_batch
    shoot_logits = shoot_logits.masked_fill(~shoot_mask_batch, float('-inf'))

    # === Value (round + opponent conditioned) ===
    # Build per-step opponent type embeddings for value head
    opp_type_indices = torch.tensor(
        [s.opponent_type_idx for s in flat_steps], dtype=torch.long)
    opp_embed_batch = model.opponent_embedding(opp_type_indices)  # (N, OPP_EMBED_DIM)
    values = model.value_head(h, round_onehot, opp_embed_batch)

    # Per-opponent-type mean value estimates (diagnostic)
    _opp_type_names = ["heuristic", "sp_mirror", "sp_hof", "sp_ml", "sp_random"]
    per_opp_type_mean_values: dict[str, float] = {}
    with torch.no_grad():
        for ot_idx, ot_name in enumerate(_opp_type_names):
            mask = opp_type_indices == ot_idx
            if mask.any():
                per_opp_type_mean_values[f"mean_value_{ot_name}"] = values[mask].mean().item()

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
    aux_fs_alpha_short = aux_fs_beta_short = aux_es_alpha_short = aux_es_beta_short = None
    aux_obj_logits_short = None
    if hasattr(model, 'aux_friendly_survival_head'):
        # Long-horizon (end-of-game)
        fs_raw = model.aux_friendly_survival_head(h).view(n_steps, n_units, 2)
        aux_fs_alpha = F.softplus(fs_raw[..., 0]) + 0.01   # (N, 10), > 0
        aux_fs_beta = F.softplus(fs_raw[..., 1]) + 0.01    # (N, 10), > 0

        es_raw = model.aux_enemy_survival_head(h).view(n_steps, n_units, 2)
        aux_es_alpha = F.softplus(es_raw[..., 0]) + 0.01
        aux_es_beta = F.softplus(es_raw[..., 1]) + 0.01

        aux_obj_logits = model.aux_obj_control_head(h).view(n_steps, 5, 3)

        # Short-horizon (end-of-current-round)
        if hasattr(model, 'aux_friendly_survival_head_short'):
            fs_raw_s = model.aux_friendly_survival_head_short(h).view(n_steps, n_units, 2)
            aux_fs_alpha_short = F.softplus(fs_raw_s[..., 0]) + 0.01
            aux_fs_beta_short = F.softplus(fs_raw_s[..., 1]) + 0.01

            es_raw_s = model.aux_enemy_survival_head_short(h).view(n_steps, n_units, 2)
            aux_es_alpha_short = F.softplus(es_raw_s[..., 0]) + 0.01
            aux_es_beta_short = F.softplus(es_raw_s[..., 1]) + 0.01

            aux_obj_logits_short = model.aux_obj_control_head_short(h).view(n_steps, 5, 3)

    # Activation countdown heads
    aux_f_act_rem = aux_e_act_rem = None
    if hasattr(model, 'aux_friendly_activations_head'):
        aux_f_act_rem = F.softplus(model.aux_friendly_activations_head(h).squeeze(-1))  # (N,), ≥ 0
        aux_e_act_rem = F.softplus(model.aux_enemy_activations_head(h).squeeze(-1))     # (N,), ≥ 0

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
        aux_friendly_surv_alpha_short=aux_fs_alpha_short,
        aux_friendly_surv_beta_short=aux_fs_beta_short,
        aux_enemy_surv_alpha_short=aux_es_alpha_short,
        aux_enemy_surv_beta_short=aux_es_beta_short,
        aux_obj_control_logits_short=aux_obj_logits_short,
        aux_friendly_act_remaining=aux_f_act_rem,
        aux_enemy_act_remaining=aux_e_act_rem,
        per_opp_type_mean_values=per_opp_type_mean_values,
        unit_logits=unit_logits,
        move_logits=move_logits,
        charge_logits=charge_logits,
        shoot_logits=shoot_logits,
    )


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------

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
    planning_distill_max_weight: float = 0.0,
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

    # --- Planning distillation loss ---
    distill_loss_val = 0.0
    distill_sub_losses: dict[str, float] = {}
    if planning_distill_max_weight > 0 and flat_steps is not None:
        _distill = _compute_planning_distill_loss(
            flat_result, flat_steps, planning_distill_max_weight,
        )
        if _distill is not None:
            _distill_tensor, distill_sub_losses = _distill
            distill_loss_val = _distill_tensor.item()
            loss = loss + _distill_tensor

    # --- Planning activation metrics ---
    planning_activations = 0
    planning_improvements = 0
    planning_argmax_best = 0
    planning_value_deltas: list[float] = []
    if flat_steps is not None:
        for s in flat_steps:
            if s.was_planned:
                planning_activations += 1
                if s.planning_improved:
                    planning_improvements += 1
                    planning_value_deltas.append(s.planning_value_delta)
                else:
                    planning_argmax_best += 1

    weighted_aux = effective_aux_coeff * aux_loss_val
    non_aux_loss = loss.item() - weighted_aux - distill_loss_val
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
        "per_opp_type_mean_values": flat_result.per_opp_type_mean_values or {},
        "planning_distill_loss": distill_loss_val,
        "planning_distill_sub": distill_sub_losses,
        "planning_activations": planning_activations,
        "planning_improvement_rate": (
            planning_improvements / planning_activations
            if planning_activations > 0 else 0.0
        ),
        "planning_mean_value_delta": (
            sum(planning_value_deltas) / len(planning_value_deltas)
            if planning_value_deltas else 0.0
        ),
        "planning_argmax_rate": (
            planning_argmax_best / planning_activations
            if planning_activations > 0 else 0.0
        ),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Auxiliary losses
# ---------------------------------------------------------------------------

def _compute_aux_loss_horizon(
    flat_result: FlatReplayResult,
    flat_steps: list[TacticalActivationRecord],
    use_short: bool,
) -> torch.Tensor | None:
    """Compute auxiliary loss for one horizon (short = end-of-round, long = end-of-game).

    Each horizon uses its own dedicated prediction heads. Returns a scalar
    tensor, or None if no valid targets or head outputs are available.
    """
    # Select the correct head outputs for this horizon
    if use_short:
        fs_alpha_all = flat_result.aux_friendly_surv_alpha_short
        fs_beta_all = flat_result.aux_friendly_surv_beta_short
        es_alpha_all = flat_result.aux_enemy_surv_alpha_short
        es_beta_all = flat_result.aux_enemy_surv_beta_short
        obj_logits_all = flat_result.aux_obj_control_logits_short
    else:
        fs_alpha_all = flat_result.aux_friendly_surv_alpha
        fs_beta_all = flat_result.aux_friendly_surv_beta
        es_alpha_all = flat_result.aux_enemy_surv_alpha
        es_beta_all = flat_result.aux_enemy_surv_beta
        obj_logits_all = flat_result.aux_obj_control_logits

    if fs_alpha_all is None:
        return None

    losses: list[torch.Tensor] = []

    # --- Survival losses (Beta NLL) ---
    fs_targets = []
    es_targets = []
    valid_surv = []
    for i, s in enumerate(flat_steps):
        fs = s.friendly_survival_target_short if use_short else s.friendly_survival_target
        es = s.enemy_survival_target_short if use_short else s.enemy_survival_target
        if fs is not None and es is not None:
            fs_targets.append(fs)
            es_targets.append(es)
            valid_surv.append(i)

    if valid_surv:
        idx = torch.tensor(valid_surv, dtype=torch.long)
        fs_t = torch.tensor(fs_targets, dtype=torch.float32)  # (M, 10)
        es_t = torch.tensor(es_targets, dtype=torch.float32)  # (M, 10)

        eps = 1e-3
        fs_t = fs_t.clamp(eps, 1.0 - eps)
        es_t = es_t.clamp(eps, 1.0 - eps)

        fs_alpha = fs_alpha_all[idx]  # (M, 10)
        fs_beta = fs_beta_all[idx]
        fs_dist = torch.distributions.Beta(fs_alpha.clamp(max=100.0), fs_beta.clamp(max=100.0))
        fs_nll = -fs_dist.log_prob(fs_t).mean()
        losses.append(fs_nll)

        es_alpha = es_alpha_all[idx]
        es_beta = es_beta_all[idx]
        es_dist = torch.distributions.Beta(es_alpha.clamp(max=100.0), es_beta.clamp(max=100.0))
        es_nll = -es_dist.log_prob(es_t).mean()
        losses.append(es_nll)

    # --- Objective control loss (cross-entropy) ---
    obj_targets = []
    valid_obj = []
    for i, s in enumerate(flat_steps):
        obj = s.obj_control_target_short if use_short else s.obj_control_target
        if obj is not None:
            obj_targets.append(obj)
            valid_obj.append(i)

    if valid_obj and obj_logits_all is not None:
        idx = torch.tensor(valid_obj, dtype=torch.long)
        obj_t = torch.tensor(obj_targets, dtype=torch.long)  # (M, 5)
        obj_logits = obj_logits_all[idx]  # (M, 5, 3)
        obj_ce = F.cross_entropy(obj_logits.reshape(-1, 3), obj_t.reshape(-1))
        losses.append(obj_ce)

    if not losses:
        return None
    return torch.stack(losses).mean()


def _compute_countdown_loss(
    flat_result: FlatReplayResult,
    flat_steps: list[TacticalActivationRecord],
) -> torch.Tensor | None:
    """MSE loss for activation countdown prediction heads.

    Targets are remaining friendly/enemy activations until game end,
    backfilled from the trajectory during collection.
    """
    if flat_result.aux_friendly_act_remaining is None:
        return None

    f_targets = []
    e_targets = []
    valid = []
    for i, s in enumerate(flat_steps):
        if s.friendly_activations_remaining is not None and s.enemy_activations_remaining is not None:
            f_targets.append(s.friendly_activations_remaining)
            e_targets.append(s.enemy_activations_remaining)
            valid.append(i)

    if not valid:
        return None

    idx = torch.tensor(valid, dtype=torch.long)
    f_t = torch.tensor(f_targets, dtype=torch.float32)
    e_t = torch.tensor(e_targets, dtype=torch.float32)

    f_pred = flat_result.aux_friendly_act_remaining[idx]
    e_pred = flat_result.aux_enemy_act_remaining[idx]

    f_mse = F.mse_loss(f_pred, f_t)
    e_mse = F.mse_loss(e_pred, e_t)
    return (f_mse + e_mse) * 0.5


def _compute_aux_loss(
    flat_result: FlatReplayResult,
    flat_steps: list[TacticalActivationRecord],
) -> torch.Tensor | None:
    """Compute combined auxiliary loss from all auxiliary heads.

    Includes short-horizon, long-horizon, and activation countdown losses.
    Each head's gradient is independent.

    Returns a scalar tensor, or None if no valid targets are available.
    """
    short_loss = _compute_aux_loss_horizon(flat_result, flat_steps, use_short=True)
    long_loss = _compute_aux_loss_horizon(flat_result, flat_steps, use_short=False)
    countdown_loss = _compute_countdown_loss(flat_result, flat_steps)

    losses = [l for l in (short_loss, long_loss, countdown_loss) if l is not None]
    if not losses:
        return None
    return torch.stack(losses).sum()


# ---------------------------------------------------------------------------
# Planning distillation loss
# ---------------------------------------------------------------------------

def _kl_from_soft_target(
    target: torch.Tensor, log_probs: torch.Tensor,
) -> torch.Tensor:
    """KL(target || policy) with NaN/Inf-safe handling.

    Args:
        target: (P, K) soft target probabilities (0 for unevaluated slots)
        log_probs: (P, K) policy log-probabilities
    Returns:
        (P,) per-example KL divergence
    """
    raw_kl = target * (target.clamp(min=1e-8).log() - log_probs)
    return torch.nan_to_num(raw_kl, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=-1)


def _compute_planning_distill_loss(
    flat_result: FlatReplayResult,
    flat_steps: list[TacticalActivationRecord],
    max_weight: float,
) -> tuple[torch.Tensor, dict] | None:
    """Gated distillation loss from planned activations across all heads.

    Only applies to activations where planning found a better action than
    the policy argmax. Weight scales with the value gap.

    Returns (total_loss, sub_loss_dict) or None if no planned-improved steps.
    """
    if flat_result.unit_logits is None:
        return None

    # --- Collect per-step data for each head ---
    # Unit head
    unit_indices, unit_targets, unit_weights = [], [], []
    # Move type head
    move_indices, move_targets, move_weights = [], [], []
    # Charge target head
    charge_indices, charge_targets, charge_weights = [], [], []
    # Shoot target head
    shoot_indices, shoot_targets, shoot_weights = [], [], []

    for i, s in enumerate(flat_steps):
        if not s.was_planned or not s.planning_improved:
            continue
        w = min(s.planning_value_delta, max_weight)

        # --- Unit head (existing logic) ---
        if s.planning_unit_values is not None and s.planning_unit_indices is not None:
            target = torch.full((MAX_UNITS_PER_SIDE,), float('-inf'))
            for idx, val in zip(s.planning_unit_indices, s.planning_unit_values):
                target[idx] = val
            target = torch.softmax(target, dim=0)
            unit_indices.append(i)
            unit_targets.append(target)
            unit_weights.append(w)

        # --- Move type head: ≥2 distinct move types for chosen unit ---
        if (s.planning_move_values is not None
                and s.planning_move_indices is not None
                and len(s.planning_move_indices) >= 2):
            target = torch.full((NUM_MOVE_TYPES,), float('-inf'))
            for idx, val in zip(s.planning_move_indices, s.planning_move_values):
                target[idx] = val
            target = torch.softmax(target, dim=0)
            move_indices.append(i)
            move_targets.append(target)
            move_weights.append(w)

        # --- Charge target head: chosen move is charge AND ≥2 distinct targets ---
        if (s.move_type == MOVE_CHARGE
                and s.planning_charge_values is not None
                and s.planning_charge_indices is not None
                and len(s.planning_charge_indices) >= 2):
            target = torch.full((MAX_UNITS_PER_SIDE,), float('-inf'))
            for idx, val in zip(s.planning_charge_indices, s.planning_charge_values):
                target[idx] = val
            target = torch.softmax(target, dim=0)
            charge_indices.append(i)
            charge_targets.append(target)
            charge_weights.append(w)

        # --- Shoot target head: chosen move is hold/advance AND ≥2 distinct targets ---
        if (s.move_type in (MOVE_HOLD, MOVE_ADVANCE)
                and s.planning_shoot_values is not None
                and s.planning_shoot_indices is not None
                and len(s.planning_shoot_indices) >= 2):
            target = torch.full((MAX_UNITS_PER_SIDE,), float('-inf'))
            for idx, val in zip(s.planning_shoot_indices, s.planning_shoot_values):
                target[idx] = val
            target = torch.softmax(target, dim=0)
            shoot_indices.append(i)
            shoot_targets.append(target)
            shoot_weights.append(w)

    if not unit_indices and not move_indices and not charge_indices and not shoot_indices:
        return None

    device = flat_result.unit_logits.device
    total_loss = torch.tensor(0.0, device=device)
    sub_losses: dict[str, float] = {}

    # --- Unit selection KL ---
    if unit_indices:
        idx_t = torch.tensor(unit_indices, dtype=torch.long, device=device)
        tgt = torch.stack(unit_targets).to(device)
        w_t = torch.tensor(unit_weights, dtype=torch.float32, device=device)
        log_p = F.log_softmax(flat_result.unit_logits[idx_t], dim=-1)
        kl = _kl_from_soft_target(tgt, log_p)
        unit_loss = (w_t * kl).mean()
        total_loss = total_loss + unit_loss
        sub_losses["unit"] = unit_loss.item()

    # --- Move type KL ---
    if move_indices and flat_result.move_logits is not None:
        idx_t = torch.tensor(move_indices, dtype=torch.long, device=device)
        tgt = torch.stack(move_targets).to(device)
        w_t = torch.tensor(move_weights, dtype=torch.float32, device=device)
        log_p = F.log_softmax(flat_result.move_logits[idx_t], dim=-1)
        kl = _kl_from_soft_target(tgt, log_p)
        move_loss = (w_t * kl).mean()
        total_loss = total_loss + move_loss
        sub_losses["move"] = move_loss.item()

    # --- Charge target KL ---
    if charge_indices and flat_result.charge_logits is not None:
        idx_t = torch.tensor(charge_indices, dtype=torch.long, device=device)
        tgt = torch.stack(charge_targets).to(device)
        w_t = torch.tensor(charge_weights, dtype=torch.float32, device=device)
        log_p = F.log_softmax(flat_result.charge_logits[idx_t], dim=-1)
        kl = _kl_from_soft_target(tgt, log_p)
        charge_loss = (w_t * kl).mean()
        total_loss = total_loss + charge_loss
        sub_losses["charge"] = charge_loss.item()

    # --- Shoot target KL ---
    if shoot_indices and flat_result.shoot_logits is not None:
        idx_t = torch.tensor(shoot_indices, dtype=torch.long, device=device)
        tgt = torch.stack(shoot_targets).to(device)
        w_t = torch.tensor(shoot_weights, dtype=torch.float32, device=device)
        log_p = F.log_softmax(flat_result.shoot_logits[idx_t], dim=-1)
        kl = _kl_from_soft_target(tgt, log_p)
        shoot_loss = (w_t * kl).mean()
        total_loss = total_loss + shoot_loss
        sub_losses["shoot"] = shoot_loss.item()

    return total_loss, sub_losses
