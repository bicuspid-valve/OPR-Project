"""Action sampling (no gradient) for tactical model — single and batched."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ml_features import MAX_UNITS_PER_SIDE, TACTICAL_UNIT_FEATURES, extract_can_charge_mask
from ml_model_tactical import (
    TacticalModel, NUM_MOVE_TYPES, MOVE_HOLD, MOVE_ADVANCE, MOVE_RUSH, MOVE_CHARGE,
    NUM_DEST_COMPONENTS, DEST_PARAMS_PER_COMPONENT,
)
from ml_integration_tactical import (
    compute_post_move_rel, compute_post_move_position,
    decode_destination_params, decode_destination_argmax,
    compute_in_range_mask, compute_in_range_mask_batched,
    DEST_DIST_MAX, DEST_DIST_MIN,
)

from ml_training.config import _TacticalInferenceRequest, _TacticalSamplingResult


def _sample_from_dest_mixture_single(
    destination_raw: torch.Tensor,
) -> tuple[float, float, int, float]:
    """Sample one (angle, distance_frac, component_idx, log_prob) from the destination mixture.

    Uses numpy for speed (single-sample path).
    """
    K = NUM_DEST_COMPONENTS
    P = DEST_PARAMS_PER_COMPONENT
    eps = 1e-8

    angles, concentrations, dist_means, dist_sigmas, mixture_logits, probs = (
        decode_destination_params(destination_raw))

    # Sample component (normalize probs for numerical safety)
    probs_arr = np.array(probs, dtype=np.float64)
    probs_arr /= probs_arr.sum()
    comp_idx = int(np.random.choice(K, p=probs_arr))

    # Sample angle from VonMises for this component
    sampled_angle = float(np.random.vonmises(angles[comp_idx], concentrations[comp_idx]))

    # Sample distance: Gaussian in sigmoid-space, then map to [DEST_DIST_MIN, DEST_DIST_MAX]
    mu_raw = dist_means[comp_idx]
    sigma = dist_sigmas[comp_idx]
    z = float(np.random.normal(mu_raw, sigma))
    raw_frac = 1.0 / (1.0 + math.exp(-z))  # sigmoid
    sampled_frac_unclamped = DEST_DIST_MIN + (DEST_DIST_MAX - DEST_DIST_MIN) * raw_frac
    sampled_frac = max(0.0, min(1.0, sampled_frac_unclamped))

    # Log-prob: log(sum_k w_k * VonMises(angle|k) * N_sigmoid(dist|k))
    # We compute log-sum-exp over components
    log_components = []
    for k in range(K):
        log_w = math.log(max(probs[k], eps))
        # VonMises log-prob for this component
        conc_k = concentrations[k]
        _conc_t = torch.tensor(conc_k)
        _i0e = torch.special.i0e(_conc_t)
        _log_i0 = conc_k + math.log(max(float(_i0e.item()), 1e-20))
        vm_lp = conc_k * math.cos(sampled_angle - angles[k]) - (math.log(2.0 * math.pi) + _log_i0)

        # Gaussian log-prob on the pre-sigmoid value z, with Jacobian for sigmoid transform
        # p(frac) = N(z | mu, sigma) / |d(frac)/dz| where d(frac)/dz = (MAX-MIN)*sigmoid'(z)
        # But we stored the actual angle+frac, so we need to work backwards:
        # z = logit((frac_unclamped - MIN) / (MAX - MIN))
        # For clamped values at 0 or 1, use CDF
        if sampled_frac_unclamped <= 0.0:
            # P(frac <= 0) = P(z <= logit(-MIN/(MAX-MIN)))
            z_boundary = math.log(max(-DEST_DIST_MIN / (DEST_DIST_MAX - DEST_DIST_MIN), eps)
                                  / max(1.0 + DEST_DIST_MIN / (DEST_DIST_MAX - DEST_DIST_MIN), eps))
            # CDF via erfc: Phi(x) = 0.5 * erfc(-x / sqrt(2))
            _cdf_arg = (z_boundary - dist_means[k]) / (dist_sigmas[k] * math.sqrt(2.0))
            _cdf_val = 0.5 * math.erfc(-_cdf_arg)
            dist_lp = math.log(max(_cdf_val, eps))
        elif sampled_frac_unclamped >= 1.0:
            # P(frac >= 1) = P(z >= logit((1-MIN)/(MAX-MIN)))
            t = (1.0 - DEST_DIST_MIN) / (DEST_DIST_MAX - DEST_DIST_MIN)
            z_boundary = math.log(max(t, eps) / max(1.0 - t, eps))
            # Survival function via erfc: 1 - Phi(x) = 0.5 * erfc(x / sqrt(2))
            _sf_arg = (z_boundary - dist_means[k]) / (dist_sigmas[k] * math.sqrt(2.0))
            _sf_val = 0.5 * math.erfc(_sf_arg)
            dist_lp = math.log(max(_sf_val, eps))
        else:
            # Normal case: Gaussian density on z with Jacobian correction
            mu_k = dist_means[k]
            sig_k = dist_sigmas[k]
            # z for this component's evaluation
            t = (sampled_frac_unclamped - DEST_DIST_MIN) / (DEST_DIST_MAX - DEST_DIST_MIN)
            t = max(eps, min(1.0 - eps, t))
            z_val = math.log(t / (1.0 - t))
            gauss_lp = -0.5 * ((z_val - mu_k) / sig_k) ** 2 - math.log(sig_k) - 0.5 * math.log(2 * math.pi)
            # Jacobian: dz/dfrac = 1/((MAX-MIN)*t*(1-t))
            jacobian_lp = -math.log((DEST_DIST_MAX - DEST_DIST_MIN) * t * (1.0 - t))
            dist_lp = gauss_lp + jacobian_lp

        log_components.append(log_w + vm_lp + dist_lp)

    # Log-sum-exp
    max_lc = max(log_components)
    log_prob = max_lc + math.log(sum(math.exp(lc - max_lc) for lc in log_components))

    return sampled_angle, sampled_frac, comp_idx, log_prob


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
) -> tuple[int, int, float, float, int, int, int, list[int], list[float], float, float, list[bool]]:
    """Sample tactical v2 actions with sequential conditioning (no gradient tracking).

    Returns (unit_idx, move_type, sampled_angle, sampled_distance_frac,
             dest_component_idx, charge_target_idx, shoot_target_idx,
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

    # --- Destination mixture (sample) ---
    dest_raw = model.destination_head(h_uf_m).squeeze(0)
    sampled_angle, sampled_frac, dest_comp_idx, dest_lp = _sample_from_dest_mixture_single(dest_raw)

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
    # Advance/rush: + destination + shoot_target
    # Hold: + shoot_target
    # Charge: + charge_target
    old_log_prob = unit_lp + move_lp
    if move_type in (MOVE_ADVANCE, MOVE_RUSH):
        old_log_prob += dest_lp
    if move_type in (MOVE_HOLD, MOVE_ADVANCE):
        old_log_prob += shoot_lp
    if move_type == MOVE_CHARGE:
        old_log_prob += charge_lp

    return (unit_idx, move_type, sampled_angle, sampled_frac,
            dest_comp_idx, charge_target_idx, shoot_target_idx, target_ranking,
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
    shoot_target) in parallel.  The destination mixture is sampled per-sample.
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

    # Destination mixture (batched raw output, per-sample sampling)
    dest_raw = model.destination_head(h_uf_m)                               # (N, 18)
    dest_raw = torch.nan_to_num(dest_raw, nan=0.0, posinf=50.0, neginf=-50.0)

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
    charge_lp = charge_log_probs.gather(1, charge_indices.unsqueeze(1)).squeeze(1)

    # Value
    unit_list = unit_indices.tolist()
    move_list = move_indices.tolist()
    charge_list = charge_indices.tolist()
    opp_type_indices = torch.tensor(
        [r.opponent_type_idx for r in requests], dtype=torch.long)
    opp_embed_batch = model.opponent_embedding(opp_type_indices)
    values = model.value_head(h, round_onehot, opp_embed_batch)

    # --- Per-sample destination sampling + post-move ---
    sampled_angles: list[float] = []
    sampled_fracs: list[float] = []
    dest_comp_indices: list[int] = []
    dest_lps: list[float] = []
    post_move_rels: list[list[float]] = []
    pmr_tensors: list[torch.Tensor] = []

    for i in range(n):
        req = requests[i]
        uid = unit_list[i]
        mt = move_list[i]

        sa, sf, comp_idx, dlp = _sample_from_dest_mixture_single(dest_raw[i])
        sampled_angles.append(sa)
        sampled_fracs.append(sf)
        dest_comp_indices.append(comp_idx)
        dest_lps.append(dlp)

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
    no_shootable = ~shoot_mask_batch.any(dim=-1)
    shoot_logits_batch = shoot_logits_batch.masked_fill(no_shootable.unsqueeze(-1), 0.0)
    shoot_logits_batch = torch.nan_to_num(shoot_logits_batch, nan=0.0, posinf=50.0, neginf=-50.0)

    shoot_probs_batch = torch.softmax(shoot_logits_batch, dim=-1)
    if no_shootable.any():
        uniform_s = torch.full_like(shoot_probs_batch, 1.0 / n_units)
        shoot_probs_batch = torch.where(no_shootable.unsqueeze(-1), uniform_s, shoot_probs_batch)
    shoot_indices_batch = torch.multinomial(shoot_probs_batch, 1).squeeze(-1)
    shoot_log_probs_batch = torch.log_softmax(shoot_logits_batch, dim=-1)
    shoot_lps_t = shoot_log_probs_batch.gather(1, shoot_indices_batch.unsqueeze(1)).squeeze(1)
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
            total_lp += dest_lps[i]
        if mt in (MOVE_HOLD, MOVE_ADVANCE):
            total_lp += shoot_lps[i]
        if mt == MOVE_CHARGE:
            total_lp += charge_lp[i].item()

        results.append(_TacticalSamplingResult(
            unit_idx=unit_list[i],
            move_type=mt,
            sampled_angle=sampled_angles[i],
            sampled_distance_frac=sampled_fracs[i],
            dest_component_idx=dest_comp_indices[i],
            charge_target_idx=charge_list[i],
            shoot_target_idx=shoot_indices_list[i],
            target_ranking=rankings_list[i],
            post_move_rel=post_move_rels[i],
            old_log_prob=total_lp,
            value=val_list[i],
            shoot_mask=shoot_mask_lists[i],
        ))
    return results
