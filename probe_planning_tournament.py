"""Planning tournament: compare 4 candidate-pool strategies head-to-head.

Each player at every activation builds a pool of up to 20 candidate actions,
evaluates each via 16 dice rollouts (player action + opp activation + V), and
plays the action with the highest mean rollout-V. Player 4 skips the rollout
selection and just plays the policy argmax.

Pool composition:
  P1: 1 argmax + 19 policy samples
  P2: 1 argmax +  4 policy samples + 15 stratified-uniform candidates
  P3: 1 argmax +  4 policy samples + top-15 (by identifier head A) of 300 random
  P4: 1 argmax (no rollouts)

Trunk usage (Option B): the original frozen trunk drives all policy decisions
(argmax + sampling) and the rollout V at leaf states; the fine-tuned trunk
provides h for the identifier head only (Player 3's filter step).

Self-plays games where each side picks a random ml_hof army independently.
Round-robin over the 4 players (6 pairings); per-pairing N games with random
side assignment. Output: per-pairing win rates with normal CIs.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import probe_identifier_premise as probe
from board import Board
from evolution import resolve_army, _make_unit_states
from game import deploy_armies
from ml_features import (
    MAX_UNITS_PER_SIDE, encode_state_tactical,
    extract_can_charge_mask, extract_is_shaken, precompute_damage,
)
from ml_integration_tactical import (
    _flip_x, _flip_y, _get_model_space_positions,
    compute_destination_candidates, compute_destination_features,
    compute_in_range_mask, compute_post_move_rel,
    execute_decoded_decision,
)
from ml_model_tactical import (
    IdentifierHead, MOVE_CHARGE, MOVE_MOVE, NUM_MOVE_TYPES,
)
from ml_planning import (
    _build_masks, _collect_enemy_positions, _execute_activation,
    restore_game_state, simulate_forward, snapshot_game_state,
)
from ml_training.checkpoint import _make_model, load_model_state_dict
from ml_training.identifier_dataset import stratified_sample_candidates
from ml_training.metrics import _load_hof_ml_armies
from simulation import end_round, score_game


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TRUNK_PATH = "ml_checkpoints/final_model.pt"
DEFAULT_TRUNK_FT_PATH = "ml_checkpoints/final_model_id_finetuned.pt"
DEFAULT_HEAD_PATH = "ml_checkpoints/identifier_head_finetuned.pt"

POOL_SIZE = 20             # final pool size for rollout selection (P1-P3)
N_ROLLOUTS_PER_CAND = 16   # rollouts per pool candidate
N_RANDOM_FOR_FILTER = 300  # candidates sampled for P3 before identifier filter
N_FILTER_KEEP = 15         # how many P3 keeps from the 300

#PLAYERS = ("P1", "P2", "P3", "P4")
PLAYERS = ("P1","P2")
PAIRINGS = list(combinations(PLAYERS, 2))


# ---------------------------------------------------------------------------
# Decision sampling — argmax & policy multinomial via the *original* trunk
# ---------------------------------------------------------------------------

@torch.no_grad()
def _decide_via_policy(
    model, my_units, opp_units, board, round_num, player,
    my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts,
    *,
    sample: bool,
) -> probe.ArgmaxDecision:
    """One full multi-pass policy decision. `sample=False` is greedy argmax;
    `sample=True` draws via softmax multinomial at each head. Returns the
    same ArgmaxDecision struct probe._apply_decision consumes."""
    eps = 1e-8
    alive_mask, enemy_alive_mask = _build_masks(my_units, opp_units)
    state_vec = encode_state_tactical(
        my_units, opp_units, round_num, board, player,
        friendly_ranged_matchups=my_fr, friendly_melee_matchups=my_fm,
        enemy_ranged_matchups=opp_fr, enemy_melee_matchups=opp_fm,
        total_friendly_points=my_pts, total_enemy_points=opp_pts,
    )

    out = model(state_vec, alive_mask, enemy_alive_mask)

    # --- Unit head ---
    unit_logits = out.unit_logits.masked_fill(~alive_mask, float("-inf"))
    if sample:
        unit_probs = F.softmax(unit_logits, dim=-1)
        if not torch.isfinite(unit_probs).all() or unit_probs.sum() <= 0:
            unit_probs = alive_mask.float()
            unit_probs = unit_probs / unit_probs.sum()
        selected_idx = int(torch.multinomial(unit_probs, 1).item())
    else:
        selected_idx = int(unit_logits.argmax().item())
    selected_unit = my_units[selected_idx]

    # --- Move type head ---
    can_charge_mask = extract_can_charge_mask(state_vec, selected_idx)
    is_shaken = bool(extract_is_shaken(state_vec, selected_idx).item())
    move_logits = out.move_logits.clone()
    if (not can_charge_mask.any()) or is_shaken:
        move_logits[MOVE_CHARGE] = float("-inf")
    if sample:
        move_probs = F.softmax(move_logits, dim=-1)
        if not torch.isfinite(move_probs).all() or move_probs.sum() <= 0:
            move_type = MOVE_MOVE
        else:
            move_type = int(torch.multinomial(move_probs, 1).item())
    else:
        move_type = int(move_logits.argmax().item())

    enemy_positions_ms = _get_model_space_positions(opp_units, player)
    friendly_positions = _get_model_space_positions(my_units, player)
    unit_cx, unit_cy = friendly_positions[selected_idx]

    # --- Destination head (only if MOVE_MOVE) ---
    dest_col = dest_row = None
    if move_type == MOVE_MOVE:
        enemy_pos_set = _collect_enemy_positions(opp_units)
        candidates, cand_mask, adv = compute_destination_candidates(
            selected_unit, board, enemy_pos_set, player)
        eam_np = np.array(
            [(i < len(opp_units) and opp_units[i].models_alive > 0)
             for i in range(MAX_UNITS_PER_SIDE)], dtype=np.bool_)
        n_f = len(my_units)
        n_e = len(opp_units)
        fr_zero = np.zeros((n_f, MAX_UNITS_PER_SIDE, 7), dtype=np.float32)
        er_zero = np.zeros((n_e, MAX_UNITS_PER_SIDE, 7), dtype=np.float32)
        mm_zero = np.zeros((n_e, MAX_UNITS_PER_SIDE), dtype=np.float32)
        budget = float(selected_unit.unit.rush_distance)
        dest_feats_np = compute_destination_features(
            candidates, cand_mask, selected_unit, selected_idx, player,
            opp_units, eam_np, fr_zero, er_zero, mm_zero, budget,
            advance_reachable=adv,
        )
        dest_features_t = torch.from_numpy(dest_feats_np).unsqueeze(0)
        dest_mask_t = torch.from_numpy(cand_mask.astype(np.bool_)).unsqueeze(0)
        out_d = model(state_vec, alive_mask, enemy_alive_mask,
                      forced_unit_idx=selected_idx,
                      dest_features=dest_features_t, dest_mask=dest_mask_t)
        dest_logits = out_d.dest_logits.squeeze(0).masked_fill(
            ~dest_mask_t.squeeze(0), float("-inf"))
        if sample:
            dest_probs = F.softmax(dest_logits, dim=-1)
            if not torch.isfinite(dest_probs).all() or dest_probs.sum() <= 0:
                # All-zero / NaN — fall back to argmax to avoid crash
                best_cand = int(dest_logits.argmax().item())
            else:
                best_cand = int(torch.multinomial(dest_probs, 1).item())
        else:
            best_cand = int(dest_logits.argmax().item())
        dest_col = int(candidates[best_cand, 0])
        dest_row = int(candidates[best_cand, 1])
        post_x = float(dest_col)
        post_y = float(dest_row)
        if player == "B":
            post_x = _flip_x(post_x)
            post_y = _flip_y(post_y)
    else:
        post_x, post_y = unit_cx, unit_cy

    post_move_rel = compute_post_move_rel(post_x, post_y, enemy_positions_ms)

    # --- Charge / shoot heads ---
    out2 = model(state_vec, alive_mask, enemy_alive_mask,
                 forced_unit_idx=selected_idx, post_move_rel=post_move_rel)

    if enemy_alive_mask.any():
        # Mask charge logits to alive enemies — necessary for multinomial,
        # since softmax over a row of −inf produces NaN. Argmax tolerates
        # unmasked logits, sampling does not.
        masked_charge = out2.charge_target_logits.masked_fill(
            ~enemy_alive_mask, float("-inf"))
        if sample:
            charge_probs = F.softmax(masked_charge, dim=-1)
            if not torch.isfinite(charge_probs).all() or charge_probs.sum() <= 0:
                # Fallback: uniform over alive enemies
                charge_probs = enemy_alive_mask.float()
                charge_probs = charge_probs / charge_probs.sum()
            charge_target_idx = int(torch.multinomial(charge_probs, 1).item())
        else:
            charge_target_idx = int(masked_charge.argmax().item())
    else:
        charge_target_idx = 0

    max_wr = max(
        (w.range_inches for w in selected_unit.unit.weapons if not w.melee),
        default=0.0,
    )
    shoot_range_mask = compute_in_range_mask(
        post_move_rel, float(max_wr), enemy_alive_mask)
    masked_shoot = out2.shoot_target_logits.masked_fill(
        ~shoot_range_mask, float("-inf"))
    if shoot_range_mask.any():
        if sample:
            shoot_probs = F.softmax(masked_shoot, dim=-1)
            if not torch.isfinite(shoot_probs).all() or shoot_probs.sum() <= 0:
                shoot_probs = shoot_range_mask.float()
                shoot_probs = shoot_probs / shoot_probs.sum()
            shoot_target_idx = int(torch.multinomial(shoot_probs, 1).item())
        else:
            shoot_target_idx = int(masked_shoot.argmax().item())
    else:
        shoot_target_idx = 0
    target_ranking = torch.argsort(masked_shoot, descending=True).tolist()

    return probe.ArgmaxDecision(
        unit_idx=selected_idx, move_type=move_type,
        dest_col=dest_col, dest_row=dest_row,
        charge_target_idx=charge_target_idx,
        shoot_target_idx=shoot_target_idx,
        target_ranking=target_ranking,
    )


def _candidate_to_decision(c: probe.CandidateAction) -> probe.ArgmaxDecision:
    """Convert a stratified-sampled CandidateAction to the ArgmaxDecision form
    the apply/score helpers consume. shoot_target_idx is preserved as the head
    of target_ranking (rollouts use the same convention as compute_Q)."""
    target_ranking = [c.shoot_target_idx] + [
        i for i in range(MAX_UNITS_PER_SIDE) if i != c.shoot_target_idx
    ]
    return probe.ArgmaxDecision(
        unit_idx=c.unit_idx, move_type=c.move_type,
        dest_col=c.dest_col, dest_row=c.dest_row,
        charge_target_idx=c.charge_target_idx,
        shoot_target_idx=c.shoot_target_idx,
        target_ranking=target_ranking,
    )


# ---------------------------------------------------------------------------
# Rollout-based scoring of a decision
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_decision_via_rollouts(
    model, decision: probe.ArgmaxDecision,
    snap, ctx: probe.GameContext, current_is_a: bool,
    round_num: int, m_rollouts: int,
) -> float:
    """Q(s,a) ≈ mean over m_rollouts of (apply a + 1 opp activation + V).
    Uses the *original* trunk for V at the leaf state."""
    units_a, units_b, board = ctx.units_a, ctx.units_b, ctx.board
    samples = np.empty(m_rollouts, dtype=np.float32)

    for i in range(m_rollouts):
        restore_game_state(snap, units_a, units_b, board)
        my_units = units_a if current_is_a else units_b
        opp_units = units_b if current_is_a else units_a
        unit = my_units[decision.unit_idx]
        dest = (
            (decision.dest_col, decision.dest_row)
            if decision.dest_col is not None else None
        )
        action, goal, charge_target, reason = execute_decoded_decision(
            unit, opp_units, decision.move_type, dest,
            decision.charge_target_idx, decision.shoot_target_idx,
        )
        opp_wiped = _execute_activation(
            unit, action, goal, charge_target, reason,
            decision.target_ranking, my_units, opp_units, board, ctx.mode,
        )
        if opp_wiped:
            v_a = 1.0 if current_is_a else -1.0
        elif not any(u.models_alive > 0 for u in my_units):
            v_a = -1.0 if current_is_a else 1.0
        else:
            v_a = simulate_forward(
                units_a, units_b, board, model,
                n_activations=1, current_is_a=not current_is_a,
                round_num=round_num, mode=ctx.mode,
                fr_a=ctx.fr_a, fm_a=ctx.fm_a,
                fr_b=ctx.fr_b, fm_b=ctx.fm_b,
                pts_a=ctx.pts_a, pts_b=ctx.pts_b,
            )
            if v_a is None:
                v_a = 0.0
        v_persp = v_a if current_is_a else -v_a
        samples[i] = v_persp

    restore_game_state(snap, units_a, units_b, board)
    return float(samples.mean())


# ---------------------------------------------------------------------------
# Identifier head scoring (for P3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_candidates_with_identifier(
    head: IdentifierHead,
    trunk_ft,                              # the *fine-tuned* trunk
    cands: list[probe.CandidateAction],
    my_units, opp_units, board, round_num, player,
    my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts,
) -> np.ndarray:
    """Compute the head's predicted advantage A for each candidate, using the
    fine-tuned trunk's h. Returns an (n_cands,) float array."""
    state_vec = encode_state_tactical(
        my_units, opp_units, round_num, board, player,
        friendly_ranged_matchups=my_fr, friendly_melee_matchups=my_fm,
        enemy_ranged_matchups=opp_fr, enemy_melee_matchups=opp_fm,
        total_friendly_points=my_pts, total_enemy_points=opp_pts,
    )
    h, _, _ = trunk_ft.trunk(state_vec.unsqueeze(0))
    h = h.squeeze(0)  # (TRUNK_WIDTH,)

    # _gather_unit_features helper: extract a unit's 200-dim slice from state_vec.
    # Mirrors the helper in train_identifier._build_batch.
    from ml_features import TACTICAL_UNIT_FEATURES
    def _slice_unit(unit_slot: int) -> torch.Tensor:
        base = unit_slot * TACTICAL_UNIT_FEATURES
        return state_vec[base : base + TACTICAL_UNIT_FEATURES]

    # Build per-cand input tensors and run the head in a single batched call.
    n = len(cands)
    h_batch = h.unsqueeze(0).expand(n, -1)
    unit_idx = torch.tensor([c.unit_idx for c in cands], dtype=torch.long)
    move_type = torch.tensor([c.move_type for c in cands], dtype=torch.long)

    unit_feat = torch.stack([_slice_unit(c.unit_idx) for c in cands])
    charge_feat = torch.stack([
        _slice_unit(MAX_UNITS_PER_SIDE + c.charge_target_idx) for c in cands
    ])
    shoot_feat = torch.stack([
        _slice_unit(MAX_UNITS_PER_SIDE + c.shoot_target_idx) for c in cands
    ])

    # Active flags & dest_feat — mirror _build_batch's masking.
    is_charge = move_type == MOVE_CHARGE
    is_move = ~is_charge

    # dest_active: only for MOVE_MOVE with dest_features available
    dest_feat = torch.zeros(n, 76, dtype=torch.float32)
    dest_active = torch.zeros(n, dtype=torch.float32)
    for i, c in enumerate(cands):
        if c.move_type == MOVE_MOVE and c.dest_features is not None:
            dest_feat[i] = torch.from_numpy(
                c.dest_features[c.dest_idx].astype(np.float32))
            dest_active[i] = 1.0

    charge_active = is_charge.float()
    # shoot_active: for moves, requires post-move-in-range check. For
    # simplicity here, mark active when the candidate had a valid shoot
    # selection and we're not charging — the head was trained the same way
    # via _build_batch's `cand_shoot_active` field.
    shoot_active = torch.tensor(
        [(c.move_type != MOVE_CHARGE
          and c.advance_reachable
          and c.shoot_target_idx is not None) for c in cands],
        dtype=torch.float32,
    )

    charge_feat = charge_feat * charge_active.unsqueeze(-1)
    shoot_feat = shoot_feat * shoot_active.unsqueeze(-1)
    dest_feat = dest_feat * dest_active.unsqueeze(-1)

    active_flags = torch.stack(
        [charge_active, shoot_active, dest_active, move_type.float()], dim=-1)

    A = head(
        h=h_batch, unit_feat=unit_feat,
        charge_feat=charge_feat, shoot_feat=shoot_feat,
        dest_feat=dest_feat, unit_idx=unit_idx,
        move_type=move_type, active_flags=active_flags,
    )
    return A.cpu().numpy()


# ---------------------------------------------------------------------------
# Pool builders — one per player
# ---------------------------------------------------------------------------

def _build_pool(
    player: str, model_orig, trunk_ft, head, rng,
    snap, ctx, current_is_a, round_num,
) -> list[probe.ArgmaxDecision]:
    """Build the candidate pool for `player`. All decisions returned in
    ArgmaxDecision form so the rollout scorer treats them uniformly."""
    my_units = ctx.units_a if current_is_a else ctx.units_b
    opp_units = ctx.units_b if current_is_a else ctx.units_a
    side = "A" if current_is_a else "B"
    my_fr = ctx.fr_a if current_is_a else ctx.fr_b
    my_fm = ctx.fm_a if current_is_a else ctx.fm_b
    opp_fr = ctx.fr_b if current_is_a else ctx.fr_a
    opp_fm = ctx.fm_b if current_is_a else ctx.fm_a
    my_pts = ctx.pts_a if current_is_a else ctx.pts_b
    opp_pts = ctx.pts_b if current_is_a else ctx.pts_a

    # All players start with the policy argmax — restore snap each time
    # because the policy passes mutate trunk dropout state etc. (no, actually
    # they don't, but it's cheap and defensive).
    restore_game_state(snap, ctx.units_a, ctx.units_b, ctx.board)
    argmax = _decide_via_policy(
        model_orig, my_units, opp_units, ctx.board, round_num, side,
        my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts, sample=False,
    )

    if player == "P4":
        return [argmax]

    pool: list[probe.ArgmaxDecision] = [argmax]

    # Policy samples — count varies per player
    n_policy = {"P1": 19, "P2": 4, "P3": 4}[player]
    for _ in range(n_policy):
        restore_game_state(snap, ctx.units_a, ctx.units_b, ctx.board)
        d = _decide_via_policy(
            model_orig, my_units, opp_units, ctx.board, round_num, side,
            my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts, sample=True,
        )
        pool.append(d)

    if player == "P1":
        return pool

    # P2 / P3 add additional candidates from stratified sampling
    restore_game_state(snap, ctx.units_a, ctx.units_b, ctx.board)
    if player == "P2":
        cands = stratified_sample_candidates(
            my_units, opp_units, ctx.board, side, K=15, rng=rng,
        )
        pool.extend(_candidate_to_decision(c) for c in cands[:15])
        return pool

    # P3: sample 300, score all with identifier, take top 15
    cands = stratified_sample_candidates(
        my_units, opp_units, ctx.board, side, K=N_RANDOM_FOR_FILTER, rng=rng,
    )
    if not cands:
        return pool
    scores = _score_candidates_with_identifier(
        head, trunk_ft, cands, my_units, opp_units, ctx.board,
        round_num, side, my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts,
    )
    keep_idx = np.argsort(-scores)[:N_FILTER_KEEP]
    pool.extend(_candidate_to_decision(cands[int(i)]) for i in keep_idx)
    return pool


# ---------------------------------------------------------------------------
# Single-state decision: pool → rollout → best decision
# ---------------------------------------------------------------------------

def _player_decide(
    player: str, model_orig, trunk_ft, head, rng,
    snap, ctx, current_is_a, round_num,
) -> probe.ArgmaxDecision:
    pool = _build_pool(player, model_orig, trunk_ft, head, rng,
                       snap, ctx, current_is_a, round_num)
    if len(pool) == 1:
        return pool[0]
    best_idx = -1
    best_q = -float("inf")
    for i, d in enumerate(pool):
        q = _score_decision_via_rollouts(
            model_orig, d, snap, ctx, current_is_a, round_num,
            m_rollouts=N_ROLLOUTS_PER_CAND,
        )
        if q > best_q:
            best_q = q
            best_idx = i
    return pool[best_idx]


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def _new_game_context_from_armies(
    army_a, army_b, mode: str = "objectives",
) -> probe.GameContext:
    """Like probe._new_game_context but with caller-provided armies."""
    res_a = resolve_army(army_a)
    res_b = resolve_army(army_b)
    units_a = _make_unit_states(army_a, res_a, "A")
    units_b = _make_unit_states(army_b, res_b, "B")
    board = Board()
    deploy_armies(units_a, units_b, board)
    fr_a, fm_a = precompute_damage([u.unit for u in units_a],
                                   [u.unit for u in units_b])
    fr_b, fm_b = precompute_damage([u.unit for u in units_b],
                                   [u.unit for u in units_a])
    pts_a = sum(u.unit.points for u in units_a)
    pts_b = sum(u.unit.points for u in units_b)
    return probe.GameContext(
        units_a=units_a, units_b=units_b, board=board,
        fr_a=fr_a, fm_a=fm_a, fr_b=fr_b, fm_b=fm_b,
        pts_a=pts_a, pts_b=pts_b, mode=mode,
        a_first=random.random() < 0.5,
    )


def play_game(
    player_a: str, player_b: str,
    model_orig, trunk_ft, head, rng,
    army_a, army_b, mode: str = "objectives",
) -> str:
    """Play one game, return 'A', 'B', or 'draw'."""
    ctx = _new_game_context_from_armies(army_a, army_b, mode)
    units_a, units_b, board = ctx.units_a, ctx.units_b, ctx.board
    a_first = ctx.a_first
    a_finished_first = a_first

    for round_num in range(1, 5):
        for u in units_a:
            u.activated = False
            u.fatigued = False
        for u in units_b:
            u.activated = False
            u.fatigued = False
        current_is_a = a_first if round_num == 1 else a_finished_first
        a_done = b_done = False

        while True:
            my_units = units_a if current_is_a else units_b
            alive_mask, _ = _build_masks(my_units,
                                         units_b if current_is_a else units_a)
            if not alive_mask.any():
                if current_is_a:
                    a_done = True
                    if not b_done:
                        a_finished_first = True
                else:
                    b_done = True
                    if not a_done:
                        a_finished_first = False
                if a_done and b_done:
                    break
                current_is_a = not current_is_a
                continue

            snap = snapshot_game_state(units_a, units_b, board)
            current_player = player_a if current_is_a else player_b
            decision = _player_decide(
                current_player, model_orig, trunk_ft, head, rng,
                snap, ctx, current_is_a, round_num,
            )
            opp_wiped = probe._apply_decision(
                decision,
                units_a if current_is_a else units_b,
                units_b if current_is_a else units_a,
                board, mode,
            )
            if opp_wiped or not any(
                u.models_alive > 0
                for u in (units_a if current_is_a else units_b)
            ):
                # Game-ending: scoring will resolve based on alive state.
                return score_game(board, units_a, units_b, mode)

            current_is_a = not current_is_a

        # End of round
        if mode != "kill_points":
            board.update_objectives(units_a, units_b)
        if not any(u.models_alive > 0 for u in units_a):
            return score_game(board, units_a, units_b, mode)
        if not any(u.models_alive > 0 for u in units_b):
            return score_game(board, units_a, units_b, mode)
        end_round(board, units_a, units_b, round_num - 1, mode)

    return score_game(board, units_a, units_b, mode)


# ---------------------------------------------------------------------------
# Worker — plays one game with given pairing
# ---------------------------------------------------------------------------

@dataclass
class _WorkerArgs:
    pairing_idx: int
    game_idx: int
    player_a: str
    player_b: str
    seed: int
    trunk_path: str
    trunk_ft_path: str
    head_path: str


_WORKER_STATE: dict = {}


def _init_worker(trunk_path: str, trunk_ft_path: str, head_path: str):
    """Per-worker init: load both trunks and the head once."""
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    model_orig = _make_model("tactical")
    model_orig.load_state_dict(load_model_state_dict(trunk_path), strict=False)
    model_orig.eval()
    for p in model_orig.parameters():
        p.requires_grad_(False)

    trunk_ft = _make_model("tactical")
    ft_payload = torch.load(trunk_ft_path, map_location="cpu", weights_only=False)
    if isinstance(ft_payload, dict) and "model_state_dict" in ft_payload:
        ft_state = ft_payload["model_state_dict"]
    else:
        ft_state = ft_payload
    trunk_ft.load_state_dict(ft_state, strict=False)
    trunk_ft.eval()
    for p in trunk_ft.parameters():
        p.requires_grad_(False)

    head = IdentifierHead()
    head_payload = torch.load(head_path, map_location="cpu", weights_only=False)
    head.load_state_dict(head_payload["model_state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    hof = _load_hof_ml_armies()
    if not hof:
        raise RuntimeError("hall_of_fame_ml.json is empty — can't run tournament")

    _WORKER_STATE.update(dict(
        model_orig=model_orig, trunk_ft=trunk_ft, head=head, hof=hof,
    ))


def _worker_play_one(args: _WorkerArgs) -> tuple[int, int, str, str, str]:
    """Returns (pairing_idx, game_idx, player_a, player_b, result)."""
    rng = random.Random(args.seed)
    s = _WORKER_STATE
    army_a = rng.choice(s["hof"])
    army_b = rng.choice(s["hof"])
    # Random side assignment per game: with 50% probability swap players.
    if rng.random() < 0.5:
        pa, pb = args.player_a, args.player_b
    else:
        pa, pb = args.player_b, args.player_a
    result = play_game(pa, pb, s["model_orig"], s["trunk_ft"], s["head"],
                       rng, army_a, army_b, mode="objectives")
    # Map back to canonical (player_a, player_b) order.
    if (pa, pb) != (args.player_a, args.player_b):
        if result == "A": result = "B"
        elif result == "B": result = "A"
    return args.pairing_idx, args.game_idx, args.player_a, args.player_b, result


# ---------------------------------------------------------------------------
# Tournament driver
# ---------------------------------------------------------------------------

def run_tournament(
    n_games_per_pairing: int = 100,
    n_workers: int = 6,
    seed: int = 42,
    trunk_path: str = DEFAULT_TRUNK_PATH,
    trunk_ft_path: str = DEFAULT_TRUNK_FT_PATH,
    head_path: str = DEFAULT_HEAD_PATH,
    output_csv: str = "results/planning_tournament.csv",
    pairings: list[tuple[str, str]] | None = None,
) -> dict:
    if pairings is None:
        pairings = PAIRINGS
    print("=" * 70)
    print(f"Planning tournament: {len(pairings)} pairings × "
          f"{n_games_per_pairing} games = {len(pairings)*n_games_per_pairing} total")
    print(f"  policy/V trunk: {trunk_path}")
    print(f"  identifier:     {trunk_ft_path} + head {head_path}")
    print(f"  workers: {n_workers}, seed: {seed}")
    print("=" * 70)

    args_list: list[_WorkerArgs] = []
    base = seed
    for p_idx, (pa, pb) in enumerate(pairings):
        for g in range(n_games_per_pairing):
            args_list.append(_WorkerArgs(
                pairing_idx=p_idx, game_idx=g,
                player_a=pa, player_b=pb,
                seed=base + p_idx * 100_000 + g,
                trunk_path=trunk_path, trunk_ft_path=trunk_ft_path,
                head_path=head_path,
            ))

    ctx = mp.get_context("spawn")
    results: list[tuple[int, int, str, str, str]] = []
    t0 = time.time()
    last_log = t0
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(trunk_path, trunk_ft_path, head_path),
    ) as pool:
        for r in pool.imap_unordered(_worker_play_one, args_list, chunksize=1):
            results.append(r)
            now = time.time()
            if now - last_log > 30:
                done = len(results)
                rate = done / max(1.0, now - t0)
                eta = (len(args_list) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(args_list)} games "
                      f"({rate:.2f}/s, eta {eta/60:.1f} min)", flush=True)
                last_log = now
    elapsed = time.time() - t0
    print(f"\n{len(results)} games in {elapsed:.0f}s "
          f"({len(results)/elapsed:.2f}/s)")

    # Aggregate per-pairing wins/losses/draws.
    summary: dict[tuple[str, str], dict] = {
        (pa, pb): {"wins_a": 0, "wins_b": 0, "draws": 0, "n": 0}
        for pa, pb in pairings
    }
    for _p_idx, _g_idx, pa, pb, res in results:
        s = summary[(pa, pb)]
        s["n"] += 1
        if res == "A": s["wins_a"] += 1
        elif res == "B": s["wins_b"] += 1
        else: s["draws"] += 1

    # Write CSV.
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_a", "player_b", "n", "wins_a", "wins_b",
                    "draws", "win_rate_a", "ci95_half"])
        for (pa, pb), s in summary.items():
            n = s["n"]
            wr = (s["wins_a"] + 0.5 * s["draws"]) / max(1, n)
            ci = 1.96 * np.sqrt(wr * (1 - wr) / max(1, n))
            w.writerow([pa, pb, n, s["wins_a"], s["wins_b"], s["draws"],
                        f"{wr:.3f}", f"{ci:.3f}"])

    # Console summary.
    print("\n" + "=" * 70)
    print(f"{'pairing':<10} {'n':>4} {'A wins':>7} {'B wins':>7} "
          f"{'draws':>6} {'WR(A)':>10}")
    print("-" * 70)
    for (pa, pb), s in summary.items():
        n = s["n"]
        wr = (s["wins_a"] + 0.5 * s["draws"]) / max(1, n)
        ci = 1.96 * np.sqrt(wr * (1 - wr) / max(1, n))
        print(f"{pa} vs {pb:<5} {n:>4} {s['wins_a']:>7} {s['wins_b']:>7} "
              f"{s['draws']:>6} {wr:>+.3f} ±{ci:.3f}")
    print(f"\nResults saved to {out_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-games", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trunk", default=DEFAULT_TRUNK_PATH)
    parser.add_argument("--trunk-ft", default=DEFAULT_TRUNK_FT_PATH)
    parser.add_argument("--head", default=DEFAULT_HEAD_PATH)
    parser.add_argument("--out", default="results/planning_tournament.csv")
    parser.add_argument("--only-pair", nargs=2, metavar=("P_A", "P_B"),
                        help="Run only this single pairing, e.g. "
                             "--only-pair P2 P3")
    args = parser.parse_args()

    pairings = None
    if args.only_pair:
        pa, pb = args.only_pair
        if pa not in PLAYERS or pb not in PLAYERS:
            raise SystemExit(f"--only-pair players must be in {PLAYERS}")
        # Match canonical order (P1, P2, P3, P4) so summary keys are consistent.
        canonical = (pa, pb) if PLAYERS.index(pa) < PLAYERS.index(pb) else (pb, pa)
        pairings = [canonical]

    run_tournament(
        n_games_per_pairing=args.n_games,
        n_workers=args.workers,
        seed=args.seed,
        trunk_path=args.trunk,
        trunk_ft_path=args.trunk_ft,
        head_path=args.head,
        output_csv=args.out,
        pairings=pairings,
    )


if __name__ == "__main__":
    main()
