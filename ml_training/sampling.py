"""Action sampling (no gradient) for tactical model — single and batched."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ml_features import MAX_UNITS_PER_SIDE, TACTICAL_UNIT_FEATURES, extract_can_charge_mask
from ml_model_tactical import (
    TacticalModel, NUM_MOVE_TYPES, MOVE_HOLD, MOVE_ADVANCE, MOVE_RUSH, MOVE_CHARGE,
)
from ml_integration_tactical import (
    compute_post_move_rel, compute_post_move_position,
    decode_direction_params, decode_distance_params,
    compute_in_range_mask, compute_in_range_mask_batched,
)

from ml_training.config import _TacticalInferenceRequest, _TacticalSamplingResult


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
    opponent_type_idx: int | None = None,          # index into NUM_OPPONENT_TYPES (for value head conditioning)
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
    h, units, round_onehot = model.trunk(x)

    # --- Unit selection (sample) ---
    unit_logits = model.unit_selection_head(h)
    unit_logits = unit_logits.masked_fill(~am, float('-inf'))
    unit_probs = torch.softmax(unit_logits, dim=-1).squeeze(0)
    unit_idx = int(torch.multinomial(unit_probs, 1).item())
    unit_lp = torch.log(unit_probs[unit_idx] + eps).item()

    unit_features = model._extract_unit_features(units, unit_idx).detach()

    # Extract can_charge mask for the selected unit
    can_charge_mask = extract_can_charge_mask(state_vec, unit_idx)           # (10,) bool

    # --- Move type head (sample) ---
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf).squeeze(0)
    # Mask charge when no enemy is in charge range
    if not can_charge_mask.any():
        move_logits = move_logits.clone()
        move_logits[MOVE_CHARGE] = float('-inf')
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

    # --- Charge target (sample) — mask by alive AND chargeable ---
    charge_logits = model.charge_target_head(h_uf_m).squeeze(0)
    charge_logits = charge_logits.masked_fill(~enemy_alive_mask, float('-inf'))
    charge_logits = charge_logits.masked_fill(~can_charge_mask, float('-inf'))
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

    # --- Value (round + opponent conditioned) ---
    opp_embed = model._get_opp_embed(h, opponent_type_idx)
    value = model.value_head(h, round_onehot, opp_embed).squeeze(0).item()

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
            post_move_rel.numpy(), old_log_prob, value, shoot_mask_list)


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
    h, units, round_onehot = model.trunk(state_batch)                          # (N, 512), (N, 20, 200), (N, 4)
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

    # Extract per-sample unit features from unit embeddings
    unit_features = units[:, :n_units, :].gather(
        1, unit_indices.unsqueeze(1).unsqueeze(2).expand(n, 1, TACTICAL_UNIT_FEATURES),
    ).squeeze(1).detach()                                                   # (N, UF)

    # Extract can_charge mask for each sample's selected unit
    can_charge_batch = extract_can_charge_mask(state_batch, unit_indices)    # (N, 10)

    # Move type
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf)                                # (N, 4)
    # Mask charge when no enemy is in charge range
    no_chargeable = ~can_charge_batch.any(dim=-1)                           # (N,)
    move_logits = move_logits.clone()
    move_logits[:, MOVE_CHARGE] = move_logits[:, MOVE_CHARGE].masked_fill(no_chargeable, float('-inf'))
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

    # Charge target — mask by alive AND chargeable
    charge_logits = model.charge_target_head(h_uf_m)                        # (N, 10)
    charge_logits = charge_logits.masked_fill(~enemy_alive_batch, float('-inf'))
    charge_logits = charge_logits.masked_fill(~can_charge_batch, float('-inf'))
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
    opp_type_indices = torch.tensor(
        [r.opponent_type_idx for r in requests], dtype=torch.long)
    opp_embed_batch = model.opponent_embedding(opp_type_indices)  # (N, OPP_EMBED_DIM)
    values = model.value_head(h, round_onehot, opp_embed_batch)

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
        post_move_rels.append(pmr.numpy())
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
