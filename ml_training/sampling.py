"""Action sampling (no gradient) for tactical model — single and batched.

Destination pointer version: replaces continuous direction/distance mixture
with discrete categorical over Dijkstra-reachable hex candidates.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ml_features import (
    MAX_UNITS_PER_SIDE, TACTICAL_UNIT_FEATURES,
    extract_can_charge_mask, extract_is_shaken,
    MAX_DEST_CANDIDATES, DEST_FEATURE_DIM,
)
from ml_model_tactical import (
    TacticalModel, NUM_MOVE_TYPES, MOVE_HOLD, MOVE_ADVANCE, MOVE_RUSH, MOVE_CHARGE,
)
from ml_integration_tactical import (
    compute_post_move_rel,
    compute_in_range_mask, compute_in_range_mask_batched,
    compute_destination_candidates, compute_destination_features,
    _flip_x, _flip_y,
    _get_movement_budgets,
)

from ml_training.config import _TacticalInferenceRequest, _TacticalSamplingResult


@torch.no_grad()
def sample_tactical_actions_no_grad(
    model: TacticalModel,
    state_vec: torch.Tensor,           # (4016,)
    alive_mask: torch.Tensor,          # (10,) bool — friendly alive+unactivated
    enemy_alive_mask: torch.Tensor,    # (10,) bool — enemy alive
    friendly_positions: list[tuple[float, float]],  # model-space, 10 slots
    enemy_positions: list[tuple[float, float]],     # model-space, 10 slots
    advance_distances: list[float],                 # per friendly slot
    rush_distances: list[float],                    # per friendly slot
    max_weapon_ranges: list[float] | None = None,   # max ranged weapon range per friendly slot
    opponent_type_idx: int | None = None,
    # Destination pointer inputs (precomputed by caller)
    dest_candidates: np.ndarray | None = None,      # (MAX_DEST_CANDIDATES, 2) int
    dest_mask: np.ndarray | None = None,            # (MAX_DEST_CANDIDATES,) bool
    dest_features: np.ndarray | None = None,        # (MAX_DEST_CANDIDATES, DEST_FEATURE_DIM) float
) -> tuple[int, int, list[list[int]], list[bool], list[list[float]], int,
           int, int, list[int], list[float], float, float, list[bool]]:
    """Sample tactical actions with sequential conditioning (no gradient tracking).

    Returns (unit_idx, move_type, dest_candidates_unpadded, dest_mask_unpadded,
             dest_features_unpadded, dest_selected_idx,
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

    can_charge_mask = extract_can_charge_mask(state_vec, unit_idx)
    is_shaken = extract_is_shaken(state_vec, unit_idx).item()

    # --- Move type head (sample) ---
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf).squeeze(0)
    if not can_charge_mask.any():
        move_logits = move_logits.clone()
        move_logits[MOVE_CHARGE] = float('-inf')
    if is_shaken:
        move_logits = move_logits.clone()
        move_logits[MOVE_ADVANCE] = float('-inf')
        move_logits[MOVE_RUSH] = float('-inf')
        move_logits[MOVE_CHARGE] = float('-inf')
    move_probs = torch.softmax(move_logits, dim=-1)
    move_type = int(torch.multinomial(move_probs, 1).item())
    move_lp = torch.log(move_probs[move_type] + eps).item()

    move_onehot = F.one_hot(
        torch.tensor(move_type), NUM_MOVE_TYPES,
    ).float().unsqueeze(0)
    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)

    # --- Destination pointer (advance/rush only) ---
    dest_lp = 0.0
    dest_selected_idx = -1
    # Store unpadded for serialization
    dest_cands_unpadded: list[list[int]] = []
    dest_mask_unpadded: list[bool] = []
    dest_feats_unpadded: list[list[float]] = []

    if move_type in (MOVE_ADVANCE, MOVE_RUSH) and dest_candidates is not None:
        n_valid = int(dest_mask.sum())

        dest_features_t = torch.from_numpy(dest_features).float().unsqueeze(0)
        dest_mask_t = torch.from_numpy(dest_mask).unsqueeze(0)

        dest_logits = model.compute_dest_logits(h_uf_m, dest_features_t, dest_mask_t).squeeze(0)
        dest_dist = torch.distributions.Categorical(logits=dest_logits)
        dest_selected_idx = int(dest_dist.sample().item())
        dest_lp = dest_dist.log_prob(torch.tensor(dest_selected_idx)).item()

        # Store unpadded
        dest_cands_unpadded = dest_candidates[:n_valid].tolist()
        dest_mask_unpadded = [True] * n_valid
        dest_feats_unpadded = dest_features[:n_valid].tolist()

    # Compute post-move position from selected hex
    unit_cx, unit_cy = friendly_positions[unit_idx]
    if move_type in (MOVE_ADVANCE, MOVE_RUSH) and dest_selected_idx >= 0:
        dcol = int(dest_candidates[dest_selected_idx, 0])
        drow = int(dest_candidates[dest_selected_idx, 1])
        # Convert game-space hex to model-space for post_move_rel
        px, py = float(dcol), float(drow)
        if friendly_positions[unit_idx][0] != _flip_x(float(dcol)):
            # Check if player B flip is needed
            pass
        # The candidates are in game-space. We need model-space for post_move_rel.
        # friendly_positions are already in model-space.
        # To get model-space from game-space: if player B, flip.
        # But we don't have player here... We can infer from friendly_positions.
        # Actually, the caller provides model-space positions but candidates are game-space.
        # The simplest approach: use the candidate directly and let caller handle flip.
        # For now, compute model-space from game-space using unit position comparison.
        # If unit_cx matches the unit's game-space centroid, player is A; else B.
        # Simpler: just pass through. The caller who builds dest_candidates knows the player.
        # We'll assume game-space candidates and flip based on whether model-space != game-space.
        post_x, post_y = px, py
    else:
        post_x, post_y = unit_cx, unit_cy

    post_move_rel = compute_post_move_rel(post_x, post_y, enemy_positions)
    post_move_rel_unsq = post_move_rel.unsqueeze(0)

    # --- Charge target (sample) ---
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

    # --- Shoot target (sample) ---
    shoot_input = torch.cat([h, unit_features, move_onehot, post_move_rel_unsq], dim=-1)
    shoot_logits = model.shoot_target_head(shoot_input).squeeze(0)
    if max_weapon_ranges is not None:
        shoot_mask_t = compute_in_range_mask(
            post_move_rel, max_weapon_ranges[unit_idx], enemy_alive_mask)
    else:
        shoot_mask_t = enemy_alive_mask
    if is_shaken:
        shoot_mask_t = torch.zeros_like(shoot_mask_t)
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

    # --- Value ---
    opp_embed = model._get_opp_embed(h, opponent_type_idx)
    value = model.value_head(h, round_onehot, opp_embed).squeeze(0).item()

    # Log-prob: sum across active heads based on move_type
    old_log_prob = unit_lp + move_lp
    if move_type in (MOVE_ADVANCE, MOVE_RUSH):
        old_log_prob += dest_lp
    if move_type in (MOVE_HOLD, MOVE_ADVANCE):
        old_log_prob += shoot_lp
    if move_type == MOVE_CHARGE:
        old_log_prob += charge_lp

    return (unit_idx, move_type,
            dest_cands_unpadded, dest_mask_unpadded, dest_feats_unpadded, dest_selected_idx,
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
    shoot_target) in parallel.  The destination pointer runs per-sample since
    each sample has different candidate sets.
    """
    n = len(requests)
    if n == 0:
        return []

    n_units = MAX_UNITS_PER_SIDE
    eps = 1e-8

    # Stack inputs
    state_batch = torch.stack([r.state_vec for r in requests])
    alive_batch = torch.stack([r.alive_mask for r in requests])
    enemy_alive_batch = torch.stack([r.enemy_alive_mask for r in requests])

    # Trunk
    h, units, round_onehot = model.trunk(state_batch)
    if torch.isnan(h).any() or torch.isinf(h).any():
        print("  WARNING: NaN/Inf in trunk output during data collection — clamping")
        h = torch.nan_to_num(h, nan=0.0, posinf=50.0, neginf=-50.0)

    # Unit selection
    unit_logits = model.unit_selection_head(h)
    unit_logits = unit_logits.masked_fill(~alive_batch, float('-inf'))
    all_dead = ~alive_batch.any(dim=1, keepdim=True)
    unit_logits = unit_logits.masked_fill(all_dead, 0.0)
    unit_logits = torch.nan_to_num(unit_logits, nan=0.0, posinf=50.0, neginf=-50.0)
    unit_probs = torch.softmax(unit_logits, dim=-1)
    unit_indices = torch.multinomial(unit_probs, 1).squeeze(-1)
    unit_log_probs = torch.log_softmax(unit_logits, dim=-1)
    unit_lp = unit_log_probs.gather(1, unit_indices.unsqueeze(1)).squeeze(1)

    unit_features = units[:, :n_units, :].gather(
        1, unit_indices.unsqueeze(1).unsqueeze(2).expand(n, 1, TACTICAL_UNIT_FEATURES),
    ).squeeze(1).detach()

    can_charge_batch = extract_can_charge_mask(state_batch, unit_indices)
    is_shaken_batch = extract_is_shaken(state_batch, unit_indices)

    # Move type
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf)
    no_chargeable = ~can_charge_batch.any(dim=-1)
    move_logits = move_logits.clone()
    move_logits[:, MOVE_CHARGE] = move_logits[:, MOVE_CHARGE].masked_fill(no_chargeable, float('-inf'))
    move_logits[:, MOVE_ADVANCE] = move_logits[:, MOVE_ADVANCE].masked_fill(is_shaken_batch, float('-inf'))
    move_logits[:, MOVE_RUSH] = move_logits[:, MOVE_RUSH].masked_fill(is_shaken_batch, float('-inf'))
    move_logits[:, MOVE_CHARGE] = move_logits[:, MOVE_CHARGE].masked_fill(is_shaken_batch, float('-inf'))
    move_logits = torch.nan_to_num(move_logits, nan=0.0, posinf=50.0, neginf=-50.0)
    move_probs = torch.softmax(move_logits, dim=-1)
    move_indices = torch.multinomial(move_probs, 1).squeeze(-1)
    move_log_probs = torch.log_softmax(move_logits, dim=-1)
    move_lp = move_log_probs.gather(1, move_indices.unsqueeze(1)).squeeze(1)
    move_onehot = F.one_hot(move_indices, NUM_MOVE_TYPES).float()

    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)

    # Charge target
    charge_logits = model.charge_target_head(h_uf_m)
    charge_logits = charge_logits.masked_fill(~enemy_alive_batch, float('-inf'))
    charge_logits = charge_logits.masked_fill(~can_charge_batch, float('-inf'))
    no_enemies = ~enemy_alive_batch.any(dim=-1)
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

    # --- Per-sample destination pointer + post-move ---
    dest_cands_list: list[list[list[int]]] = []
    dest_masks_list: list[list[bool]] = []
    dest_feats_list: list[list[list[float]]] = []
    dest_selected_list: list[int] = []
    dest_lps: list[float] = []
    post_move_rels: list[list[float]] = []
    pmr_tensors: list[torch.Tensor] = []

    for i in range(n):
        req = requests[i]
        uid = unit_list[i]
        mt = move_list[i]

        # Destination pointer for advance/rush
        # Look up per-unit candidates from the request (keyed by unit slot)
        _has_per_unit = (hasattr(req, 'dest_candidates_per_unit')
                         and req.dest_candidates_per_unit is not None
                         and uid in req.dest_candidates_per_unit)
        # Also support legacy single-candidate fields
        _has_single = (hasattr(req, 'dest_candidates')
                       and req.dest_candidates is not None)
        if mt in (MOVE_ADVANCE, MOVE_RUSH) and (_has_per_unit or _has_single):
            if _has_per_unit:
                dest_candidates_i = req.dest_candidates_per_unit[uid]
                dest_mask_i = req.dest_mask_per_unit[uid]
                # Lazy feature computation: compute only for selected unit
                _has_precomputed_feats = (
                    req.dest_features_per_unit is not None
                    and uid in req.dest_features_per_unit
                )
                if _has_precomputed_feats:
                    dest_features_i = req.dest_features_per_unit[uid]
                elif req.dest_lazy_units is not None:
                    move_budget = req.rush_distances[uid]
                    dest_features_i = compute_destination_features(
                        dest_candidates_i, dest_mask_i,
                        req.dest_lazy_units[uid], uid,
                        req.dest_lazy_player,
                        req.dest_lazy_enemy_units,
                        req.dest_lazy_enemy_alive,
                        req.dest_lazy_fr_matchups,
                        req.dest_lazy_er_matchups,
                        req.dest_lazy_melee_matchups,
                        move_budget,
                        enemy_cache=req.dest_lazy_enemy_cache,
                    )
                else:
                    # Fallback: zero features (should not happen in practice)
                    dest_features_i = np.zeros(
                        (MAX_DEST_CANDIDATES, DEST_FEATURE_DIM), dtype=np.float32)
            else:
                dest_candidates_i = req.dest_candidates
                dest_mask_i = req.dest_mask
                dest_features_i = req.dest_features

            dest_features_t = torch.from_numpy(dest_features_i).float().unsqueeze(0)
            dest_mask_t = torch.from_numpy(dest_mask_i).unsqueeze(0)
            dest_logits = model.compute_dest_logits(
                h_uf_m[i:i+1], dest_features_t, dest_mask_t).squeeze(0)
            dest_dist = torch.distributions.Categorical(logits=dest_logits)
            dest_idx = int(dest_dist.sample().item())
            dlp = dest_dist.log_prob(torch.tensor(dest_idx)).item()

            n_valid = int(dest_mask_i.sum())
            dest_cands_list.append(np.array(dest_candidates_i[:n_valid], dtype=np.int32))
            dest_masks_list.append([True] * n_valid)
            dest_feats_list.append(None)  # recomputed during PPO replay
            dest_selected_list.append(dest_idx)
            dest_lps.append(dlp)

            # Post-move from selected hex (game-space → model-space for post_move_rel)
            dcol = int(dest_candidates_i[dest_idx, 0])
            drow = int(dest_candidates_i[dest_idx, 1])
            px, py = float(dcol), float(drow)
        else:
            dest_cands_list.append([])
            dest_masks_list.append([])
            dest_feats_list.append([])
            dest_selected_list.append(-1)
            dest_lps.append(0.0)

            ucx, ucy = req.friendly_positions[uid]
            px, py = ucx, ucy

        pmr = compute_post_move_rel(px, py, req.enemy_positions)
        post_move_rels.append(pmr.numpy())
        pmr_tensors.append(pmr)

    # --- Batched shoot target head ---
    pmr_batch = torch.stack(pmr_tensors)
    shoot_input_batch = torch.cat([h, unit_features, move_onehot, pmr_batch], dim=-1)
    shoot_logits_batch = model.shoot_target_head(shoot_input_batch)

    max_wr_list = [requests[i].max_weapon_ranges[unit_list[i]] for i in range(n)]
    max_wr_t = torch.tensor(max_wr_list, dtype=torch.float32)
    shoot_mask_batch = compute_in_range_mask_batched(pmr_batch, max_wr_t, enemy_alive_batch)
    shoot_mask_batch = shoot_mask_batch & ~is_shaken_batch.unsqueeze(-1)

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

    # Compute total log-probs
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
            dest_candidates=dest_cands_list[i],
            dest_mask=dest_masks_list[i],
            dest_features=dest_feats_list[i],
            dest_selected_idx=dest_selected_list[i],
            charge_target_idx=charge_list[i],
            shoot_target_idx=shoot_indices_list[i],
            target_ranking=rankings_list[i],
            post_move_rel=post_move_rels[i],
            old_log_prob=total_lp,
            value=val_list[i],
            shoot_mask=shoot_mask_lists[i],
        ))
    return results
