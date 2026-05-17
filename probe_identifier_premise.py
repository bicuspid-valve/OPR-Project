"""Phase 0 probe for the gap-identifier project.

Tests the premise that "actions with high V(s,a) - beta*log pi(a|s) actually
win in rollouts more often than the policy's argmax action would."

If this probe doesn't show meaningful signal, the rest of the identifier project
(dataset generation, supervised training, planner integration) is a waste.

Pipeline:
    1. Load frozen policy from ml_checkpoints/final_model.pt
    2. Self-play a handful of games, snapshotting decision states
    3. Sample N states; per state, enumerate ~K candidate actions stratified
       across the head structure
    4. For each candidate compute log pi (a|s) via the frozen policy and
       Q(s,a) via M dice rollouts of (apply action -> 1 opponent activation -> V)
    5. Calibrate beta = std(Q) / std(log pi) on the dataset
    6. Bin candidates by gap percentile; compare mean Q across bins, against
       the policy-argmax baseline and an oracle top-Q upper bound
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from board import Board
from evolution import generate_random_army, resolve_army, _make_unit_states
from game import deploy_armies
from ml_features import (
    MAX_UNITS_PER_SIDE,
    encode_state_tactical,
    extract_can_charge_mask,
    extract_is_shaken,
    precompute_damage,
)
from ml_integration_tactical import (
    _flip_x,
    _flip_y,
    _get_model_space_positions,
    compute_destination_candidates,
    compute_destination_features,
    compute_in_range_mask,
    compute_post_move_rel,
    execute_decoded_decision,
)
from ml_model_tactical import MOVE_CHARGE, MOVE_MOVE, NUM_MOVE_TYPES
from ml_planning import (
    _build_masks,
    _collect_enemy_positions,
    _execute_activation,
    restore_game_state,
    simulate_forward,
    snapshot_game_state,
)
from ml_training.checkpoint import _make_model, load_model_state_dict
from simulation import end_round, score_game, start_round


CHECKPOINT_PATH = "ml_checkpoints/final_model.pt"
_ML_BOTH_SIDES = frozenset({"A", "B"})


# ---------------------------------------------------------------------------
# Frozen-model loader (hard convention: always final_model.pt)
# ---------------------------------------------------------------------------

def load_frozen_model(checkpoint_path: str = CHECKPOINT_PATH):
    """Load the canonical frozen policy/value network.

    Errors loudly if the checkpoint is missing — the identifier project is
    defined relative to a fixed pi snapshot and there is no sensible fallback.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Frozen policy checkpoint not found: {checkpoint_path}\n"
            f"The identifier pipeline is defined against a fixed pi snapshot.\n"
            f"Run a training pass to produce final_model.pt first."
        )
    model = _make_model("tactical")
    sd = load_model_state_dict(checkpoint_path)
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# State setup
# ---------------------------------------------------------------------------

@dataclass
class GameContext:
    """Everything a state needs to step forward / encode / score."""
    units_a: list
    units_b: list
    board: Board
    fr_a: list
    fm_a: list
    fr_b: list
    fm_b: list
    pts_a: int
    pts_b: int
    mode: str
    a_first: bool


@dataclass
class DecisionState:
    """Snapshot taken just before the side-to-move acts."""
    game_idx: int
    round_num: int
    activation_idx: int
    snap: object
    current_is_a: bool
    a_finished_first: bool


def _new_game_context(mode: str = "objectives") -> GameContext:
    # enforce_forceorg=True so randomly-generated armies actually respect the
    # max_entries (10) cap. Without it, ~0.5% of armies have >10 units, which
    # the model can't represent (unit head is hard-sized at MAX_UNITS_PER_SIDE)
    # and the candidate sampler would crash on indexing past slot 9.
    army_a = generate_random_army(mode, enforce_forceorg=True)
    army_b = generate_random_army(mode, enforce_forceorg=True)
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
    return GameContext(
        units_a=units_a, units_b=units_b, board=board,
        fr_a=fr_a, fm_a=fm_a, fr_b=fr_b, fm_b=fm_b,
        pts_a=pts_a, pts_b=pts_b, mode=mode,
        a_first=random.random() < 0.5,
    )


# ---------------------------------------------------------------------------
# Single-step argmax activation — replicates simulate_forward's per-step body
# but exposes the decoded action so we can record / compare against it
# ---------------------------------------------------------------------------

@dataclass
class ArgmaxDecision:
    unit_idx: int
    move_type: int
    dest_col: int | None
    dest_row: int | None
    charge_target_idx: int
    shoot_target_idx: int
    target_ranking: list[int]


def _argmax_decision(
    model,
    my_units, opp_units,
    board, round_num, player,
    my_fr, my_fm, opp_fr, opp_fm,
    my_pts, opp_pts,
) -> ArgmaxDecision:
    """Run the multi-pass argmax forward and return the decoded action tuple."""
    alive_mask, enemy_alive_mask = _build_masks(my_units, opp_units)
    state_vec = encode_state_tactical(
        my_units, opp_units, round_num, board, player,
        friendly_ranged_matchups=my_fr, friendly_melee_matchups=my_fm,
        enemy_ranged_matchups=opp_fr, enemy_melee_matchups=opp_fm,
        total_friendly_points=my_pts, total_enemy_points=opp_pts,
    )

    out = model(state_vec, alive_mask, enemy_alive_mask)
    selected_idx = int(out.unit_logits.argmax().item())
    selected_unit = my_units[selected_idx]
    move_type = int(out.move_logits.argmax().item())

    enemy_positions_ms = _get_model_space_positions(opp_units, player)
    friendly_positions = _get_model_space_positions(my_units, player)
    unit_cx, unit_cy = friendly_positions[selected_idx]

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
        out = model(state_vec, alive_mask, enemy_alive_mask,
                    forced_unit_idx=selected_idx,
                    dest_features=dest_features_t, dest_mask=dest_mask_t)
        best_cand = int(out.dest_logits.squeeze(0).argmax().item())
        dest_col = int(candidates[best_cand, 0])
        dest_row = int(candidates[best_cand, 1])
        post_x, post_y = float(dest_col), float(dest_row)
        if player == "B":
            post_x = _flip_x(post_x)
            post_y = _flip_y(post_y)
    else:
        post_x, post_y = unit_cx, unit_cy

    post_move_rel = compute_post_move_rel(post_x, post_y, enemy_positions_ms)
    out2 = model(state_vec, alive_mask, enemy_alive_mask,
                 forced_unit_idx=selected_idx, post_move_rel=post_move_rel)

    charge_target_idx = (
        int(out2.charge_target_logits.argmax().item()) if enemy_alive_mask.any() else 0
    )
    max_wr = max(
        (w.range_inches for w in selected_unit.unit.weapons if not w.melee),
        default=0.0,
    )
    shoot_range_mask = compute_in_range_mask(
        post_move_rel, float(max_wr), enemy_alive_mask)
    masked_shoot = out2.shoot_target_logits.masked_fill(~shoot_range_mask, float('-inf'))
    if shoot_range_mask.any():
        shoot_target_idx = int(masked_shoot.argmax().item())
    else:
        shoot_target_idx = 0
    target_ranking = torch.argsort(masked_shoot, descending=True).tolist()

    return ArgmaxDecision(
        unit_idx=selected_idx, move_type=move_type,
        dest_col=dest_col, dest_row=dest_row,
        charge_target_idx=charge_target_idx,
        shoot_target_idx=shoot_target_idx,
        target_ranking=target_ranking,
    )


def _apply_decision(
    decision: ArgmaxDecision,
    my_units, opp_units, board, mode,
):
    """Decode and execute one activation. Returns True if the opponent is wiped."""
    selected_unit = my_units[decision.unit_idx]
    dest = (
        (decision.dest_col, decision.dest_row)
        if decision.dest_col is not None else None
    )
    action, goal, charge_target, reason = execute_decoded_decision(
        selected_unit, opp_units, decision.move_type, dest,
        decision.charge_target_idx, decision.shoot_target_idx,
    )
    return _execute_activation(
        selected_unit, action, goal, charge_target, reason,
        decision.target_ranking, my_units, opp_units, board, mode,
    )


# ---------------------------------------------------------------------------
# Self-play with snapshot-per-decision
# ---------------------------------------------------------------------------

def collect_decision_states(
    model, n_games: int, mode: str = "objectives",
) -> tuple[list[DecisionState], list[GameContext]]:
    """Self-play n_games and snapshot every (state, side-to-move) pair.

    Snapshots are taken BEFORE the side-to-move's argmax action is applied,
    so each snapshot represents a candidate decision point.

    Returns (states, contexts) — states reference contexts[states[i].game_idx].
    """
    contexts: list[GameContext] = []
    states: list[DecisionState] = []

    for game_idx in range(n_games):
        ctx = _new_game_context(mode)
        contexts.append(ctx)
        units_a, units_b, board = ctx.units_a, ctx.units_b, ctx.board
        a_finished_first = ctx.a_first
        activation_idx = 0

        for round_num in range(1, 5):
            for u in units_a:
                u.activated = False
                u.fatigued = False
            for u in units_b:
                u.activated = False
                u.fatigued = False
            current_is_a = ctx.a_first if round_num == 1 else a_finished_first
            a_done = b_done = False
            this_round_a_first = current_is_a

            while True:
                my_units = units_a if current_is_a else units_b
                opp_units = units_b if current_is_a else units_a
                player = "A" if current_is_a else "B"
                my_fr = ctx.fr_a if current_is_a else ctx.fr_b
                my_fm = ctx.fm_a if current_is_a else ctx.fm_b
                opp_fr = ctx.fr_b if current_is_a else ctx.fr_a
                opp_fm = ctx.fm_b if current_is_a else ctx.fm_a
                my_pts = ctx.pts_a if current_is_a else ctx.pts_b
                opp_pts = ctx.pts_b if current_is_a else ctx.pts_a

                alive_mask, _ = _build_masks(my_units, opp_units)
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

                # Snapshot before deciding
                snap = snapshot_game_state(units_a, units_b, board)
                states.append(DecisionState(
                    game_idx=game_idx, round_num=round_num,
                    activation_idx=activation_idx,
                    snap=snap, current_is_a=current_is_a,
                    a_finished_first=a_finished_first,
                ))
                activation_idx += 1

                decision = _argmax_decision(
                    model, my_units, opp_units, board, round_num, player,
                    my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts,
                )
                opp_wiped = _apply_decision(decision, my_units, opp_units, board, mode)

                if opp_wiped or not any(u.models_alive > 0 for u in my_units):
                    return states, contexts  # game is over; we have plenty of states

                current_is_a = not current_is_a

            # End of round
            board.update_objectives(units_a, units_b) if mode != "kill_points" else None
            if not any(u.models_alive > 0 for u in units_a):
                break
            if not any(u.models_alive > 0 for u in units_b):
                break
            end_round(board, units_a, units_b, round_num - 1, mode)

    return states, contexts


# ---------------------------------------------------------------------------
# Action enumeration (stratified) per state
# ---------------------------------------------------------------------------

@dataclass
class CandidateAction:
    unit_idx: int
    move_type: int
    dest_idx: int  # index into the unit's dest_candidates, or -1
    dest_col: int | None
    dest_row: int | None
    charge_target_idx: int
    shoot_target_idx: int
    is_shaken: bool
    advance_reachable: bool
    # Cached per-unit info needed for log-pi computation
    dest_candidates: object  # ndarray
    dest_features: object    # ndarray
    dest_mask: object        # ndarray


def _per_unit_dest_arrays(unit, opp_units, board, player):
    """Build dest candidate / mask / features arrays for one unit."""
    enemy_pos_set = _collect_enemy_positions(opp_units)
    candidates, cand_mask, adv = compute_destination_candidates(
        unit, board, enemy_pos_set, player)
    eam_np = np.array(
        [(i < len(opp_units) and opp_units[i].models_alive > 0)
         for i in range(MAX_UNITS_PER_SIDE)], dtype=np.bool_)
    fr_zero = np.zeros((1, MAX_UNITS_PER_SIDE, 7), dtype=np.float32)
    er_zero = np.zeros((len(opp_units), MAX_UNITS_PER_SIDE, 7), dtype=np.float32)
    mm_zero = np.zeros((len(opp_units), MAX_UNITS_PER_SIDE), dtype=np.float32)
    budget = float(unit.unit.rush_distance)
    feats = compute_destination_features(
        candidates, cand_mask, unit, 0, player,
        opp_units, eam_np, fr_zero, er_zero, mm_zero, budget,
        advance_reachable=adv,
    )
    return candidates, cand_mask, adv, feats


def enumerate_candidates(
    my_units, opp_units, board, player,
    max_per_state: int,
    rng: random.Random,
) -> list[CandidateAction]:
    """Stratified candidate enumeration for one decision state.

    Strategy: for each alive+unactivated friendly unit, sample some moves
    (with destination + shoot target) and some charges (vs each chargeable
    enemy). Cap total candidates at max_per_state via uniform downsampling.
    """
    candidates: list[CandidateAction] = []
    state_vec_dummy = None  # only need can_charge / shaken from per-unit features

    for unit_idx, unit in enumerate(my_units):
        if unit.models_alive <= 0 or unit.activated:
            continue
        # We need state_vec to extract can_charge / shaken — encode once
        if state_vec_dummy is None:
            state_vec_dummy = encode_state_tactical(
                my_units, opp_units, 1, board, player,  # round_num doesn't affect can_charge/shaken
            )
        can_charge_mask = extract_can_charge_mask(state_vec_dummy, unit_idx)
        is_shaken = bool(extract_is_shaken(state_vec_dummy, unit_idx).item())

        # MOVE actions
        if not is_shaken:
            cands, cand_mask, adv, feats = _per_unit_dest_arrays(
                unit, opp_units, board, player)
            valid_dest = [i for i in range(len(cands)) if cand_mask[i]]
            # Subsample destinations to keep cost down
            n_dest_sample = min(8, len(valid_dest))
            sampled_dest = rng.sample(valid_dest, n_dest_sample) if valid_dest else []
            for d_idx in sampled_dest:
                # No-shoot variant
                candidates.append(CandidateAction(
                    unit_idx=unit_idx, move_type=MOVE_MOVE,
                    dest_idx=d_idx,
                    dest_col=int(cands[d_idx, 0]), dest_row=int(cands[d_idx, 1]),
                    charge_target_idx=0,
                    shoot_target_idx=0,  # will be ignored if no enemies in range
                    is_shaken=False,
                    advance_reachable=bool(adv[d_idx]),
                    dest_candidates=cands, dest_features=feats, dest_mask=cand_mask,
                ))
                # Shoot variants — pick a couple of alive enemies
                alive_enemies = [
                    i for i, e in enumerate(opp_units)
                    if e.models_alive > 0 and i < MAX_UNITS_PER_SIDE
                ]
                for shoot_t in alive_enemies[:3]:  # cap shoot variants per dest
                    candidates.append(CandidateAction(
                        unit_idx=unit_idx, move_type=MOVE_MOVE,
                        dest_idx=d_idx,
                        dest_col=int(cands[d_idx, 0]), dest_row=int(cands[d_idx, 1]),
                        charge_target_idx=0,
                        shoot_target_idx=shoot_t,
                        is_shaken=False,
                        advance_reachable=bool(adv[d_idx]),
                        dest_candidates=cands, dest_features=feats, dest_mask=cand_mask,
                    ))
        else:
            # Shaken: only "hold to recover" (move=move with dest=current pos won't model
            # cleanly here; closest analog is move=move with no real action, skip)
            pass

        # CHARGE actions
        if not is_shaken and can_charge_mask.any():
            for c_idx in range(MAX_UNITS_PER_SIDE):
                if not can_charge_mask[c_idx]:
                    continue
                if c_idx >= len(opp_units) or opp_units[c_idx].models_alive <= 0:
                    continue
                candidates.append(CandidateAction(
                    unit_idx=unit_idx, move_type=MOVE_CHARGE,
                    dest_idx=-1, dest_col=None, dest_row=None,
                    charge_target_idx=c_idx, shoot_target_idx=0,
                    is_shaken=False, advance_reachable=False,
                    dest_candidates=None, dest_features=None, dest_mask=None,
                ))

    # Downsample if oversized
    if len(candidates) > max_per_state:
        candidates = rng.sample(candidates, max_per_state)
    return candidates


# ---------------------------------------------------------------------------
# log pi (a|s) computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_log_pi(
    model, cand: CandidateAction,
    my_units, opp_units, board, player, round_num,
    my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts,
) -> float:
    """Compute log pi(a|s) under the frozen policy by running the multi-pass
    forward and reading off log-probs at the chosen indices."""
    eps = 1e-8
    alive_mask, enemy_alive_mask = _build_masks(my_units, opp_units)
    state_vec = encode_state_tactical(
        my_units, opp_units, round_num, board, player,
        friendly_ranged_matchups=my_fr, friendly_melee_matchups=my_fm,
        enemy_ranged_matchups=opp_fr, enemy_melee_matchups=opp_fm,
        total_friendly_points=my_pts, total_enemy_points=opp_pts,
    )

    can_charge_mask = extract_can_charge_mask(state_vec, cand.unit_idx)
    is_shaken = bool(extract_is_shaken(state_vec, cand.unit_idx).item())

    # Pass 1: trunk + unit head + move head
    out = model(state_vec, alive_mask, enemy_alive_mask)
    unit_logits = out.unit_logits.masked_fill(~alive_mask, float('-inf'))
    unit_lp = F.log_softmax(unit_logits, dim=-1)[cand.unit_idx].item()

    move_logits = out.move_logits.clone()
    if not can_charge_mask.any() or is_shaken:
        move_logits[MOVE_CHARGE] = float('-inf')
    move_lp = F.log_softmax(move_logits, dim=-1)[cand.move_type].item()

    log_pi = unit_lp + move_lp

    # Dest log-prob if move_type == MOVE_MOVE
    if cand.move_type == MOVE_MOVE and not is_shaken and cand.dest_candidates is not None:
        dest_features_t = torch.from_numpy(cand.dest_features).float().unsqueeze(0)
        dest_mask_t = torch.from_numpy(cand.dest_mask.astype(np.bool_)).unsqueeze(0)
        out2 = model(
            state_vec, alive_mask, enemy_alive_mask,
            forced_unit_idx=cand.unit_idx,
            dest_features=dest_features_t, dest_mask=dest_mask_t,
        )
        dl = out2.dest_logits
        if dl is None:
            raise RuntimeError(
                f"dest_logits is None — feats shape {cand.dest_features.shape}, "
                f"mask sum {int(cand.dest_mask.sum())}, dest_idx {cand.dest_idx}"
            )
        if dl.dim() == 2:
            dl = dl.squeeze(0)
        if dl.dim() == 0 or dl.numel() <= cand.dest_idx:
            raise RuntimeError(
                f"dest_logits malformed: shape {tuple(dl.shape)}, "
                f"feats shape {cand.dest_features.shape}, "
                f"mask sum {int(cand.dest_mask.sum())}, dest_idx {cand.dest_idx}"
            )
        dest_lp = F.log_softmax(dl, dim=-1)[cand.dest_idx].item()
        log_pi += dest_lp

    # Charge / shoot log-probs require post_move_rel-conditioned pass
    enemy_positions_ms = _get_model_space_positions(opp_units, player)
    friendly_positions = _get_model_space_positions(my_units, player)
    if cand.move_type == MOVE_MOVE and cand.dest_candidates is not None:
        post_x, post_y = float(cand.dest_col), float(cand.dest_row)
        if player == "B":
            post_x = _flip_x(post_x); post_y = _flip_y(post_y)
    else:
        post_x, post_y = friendly_positions[cand.unit_idx]
    post_move_rel = compute_post_move_rel(post_x, post_y, enemy_positions_ms)

    out3 = model(
        state_vec, alive_mask, enemy_alive_mask,
        forced_unit_idx=cand.unit_idx, post_move_rel=post_move_rel,
    )

    if cand.move_type == MOVE_CHARGE and enemy_alive_mask.any():
        charge_lp = F.log_softmax(out3.charge_target_logits, dim=-1)[cand.charge_target_idx].item()
        log_pi += charge_lp
    elif cand.move_type == MOVE_MOVE and not is_shaken and cand.advance_reachable:
        # Only compute shoot log-prob for advance-reachable moves (rush => no shoot)
        unit = my_units[cand.unit_idx]
        max_wr = max(
            (w.range_inches for w in unit.unit.weapons if not w.melee),
            default=0.0,
        )
        shoot_range_mask = compute_in_range_mask(
            post_move_rel, float(max_wr), enemy_alive_mask)
        if shoot_range_mask[cand.shoot_target_idx]:
            masked = out3.shoot_target_logits.masked_fill(~shoot_range_mask, float('-inf'))
            shoot_lp = F.log_softmax(masked, dim=-1)[cand.shoot_target_idx].item()
            log_pi += shoot_lp
        # If shoot target is not actually in range, leave shoot lp at 0 — the action
        # decoder will fall back to "no shoot" so this candidate is degenerate but
        # we still include it; gap will be uninformative.

    return log_pi


# ---------------------------------------------------------------------------
# Q(s,a) computation via M dice rollouts of (apply -> 1 opponent activation -> V)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_Q(
    model, cand: CandidateAction,
    snap, ctx: GameContext, current_is_a: bool,
    round_num: int, m_rollouts: int,
) -> float:
    """Q(s,a) = mean over m_rollouts of (V after applying a then one opp activation).

    Returned in the side-to-move's perspective.
    """
    units_a = ctx.units_a
    units_b = ctx.units_b
    board = ctx.board
    samples: list[float] = []

    for _ in range(m_rollouts):
        restore_game_state(snap, units_a, units_b, board)

        my_units = units_a if current_is_a else units_b
        opp_units = units_b if current_is_a else units_a
        unit = my_units[cand.unit_idx]
        dest = (cand.dest_col, cand.dest_row) if cand.dest_col is not None else None

        # Build a target_ranking that prefers the chosen shoot target
        target_ranking = [cand.shoot_target_idx] + [
            i for i in range(MAX_UNITS_PER_SIDE) if i != cand.shoot_target_idx
        ]

        action, goal, charge_target, reason = execute_decoded_decision(
            unit, opp_units, cand.move_type, dest,
            cand.charge_target_idx, cand.shoot_target_idx,
            is_advance_reachable=cand.advance_reachable,
        )
        opp_wiped = _execute_activation(
            unit, action, goal, charge_target, reason, target_ranking,
            my_units, opp_units, board, ctx.mode,
        )

        if opp_wiped:
            v_a = 1.0 if current_is_a else -1.0
        elif not any(u.models_alive > 0 for u in my_units):
            v_a = -1.0 if current_is_a else 1.0
        else:
            # One opponent activation under argmax + value of resulting state
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

        # simulate_forward returns from A's perspective; flip for side-to-move
        v_persp = v_a if current_is_a else -v_a
        samples.append(float(v_persp))

    # Restore to clean state for next candidate
    restore_game_state(snap, units_a, units_b, board)
    return statistics.fmean(samples)


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def run_probe(
    n_games: int,
    n_states: int,
    max_candidates_per_state: int,
    m_rollouts: int,
    seed: int,
):
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[probe] loading frozen model from {CHECKPOINT_PATH}")
    model = load_frozen_model()

    print(f"[probe] generating {n_games} self-play games for state collection")
    t0 = time.time()
    states, contexts = collect_decision_states(model, n_games)
    print(f"[probe]   collected {len(states)} decision states in {time.time()-t0:.1f}s")

    if len(states) == 0:
        print("[probe] no states collected — aborting")
        return

    sampled = rng.sample(states, min(n_states, len(states)))
    print(f"[probe] scoring {len(sampled)} states")

    # Per-record: dict with state_id, unit_idx, move_type, ..., log_pi, Q,
    # was_argmax (bool — was this the policy's argmax action for this state?)
    records: list[dict] = []
    argmax_per_state: dict[int, dict] = {}  # state_id -> argmax record

    for state_id, st in enumerate(sampled):
        ctx = contexts[st.game_idx]
        # Restore the snapshot into the live state objects so encoders see the
        # right state
        restore_game_state(st.snap, ctx.units_a, ctx.units_b, ctx.board)

        my_units = ctx.units_a if st.current_is_a else ctx.units_b
        opp_units = ctx.units_b if st.current_is_a else ctx.units_a
        player = "A" if st.current_is_a else "B"
        my_fr = ctx.fr_a if st.current_is_a else ctx.fr_b
        my_fm = ctx.fm_a if st.current_is_a else ctx.fm_b
        opp_fr = ctx.fr_b if st.current_is_a else ctx.fr_a
        opp_fm = ctx.fm_b if st.current_is_a else ctx.fm_a
        my_pts = ctx.pts_a if st.current_is_a else ctx.pts_b
        opp_pts = ctx.pts_b if st.current_is_a else ctx.pts_a

        # Enumerate candidates
        cands = enumerate_candidates(
            my_units, opp_units, ctx.board, player,
            max_per_state=max_candidates_per_state, rng=rng,
        )
        if not cands:
            continue

        # Also score the policy argmax separately so we have a baseline that's
        # guaranteed to be from the policy
        argmax = _argmax_decision(
            model, my_units, opp_units, ctx.board, st.round_num, player,
            my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts,
        )
        # Build a CandidateAction matching the argmax for log-pi/Q parity
        if argmax.move_type == MOVE_MOVE and argmax.dest_col is not None:
            unit = my_units[argmax.unit_idx]
            cands_arr, cand_mask, adv, feats = _per_unit_dest_arrays(
                unit, opp_units, ctx.board, player)
            # Find the dest_idx matching argmax's chosen hex
            match = np.where(
                (cands_arr[:, 0] == argmax.dest_col)
                & (cands_arr[:, 1] == argmax.dest_row)
            )[0]
            if len(match) > 0:
                d_idx = int(match[0])
                argmax_cand = CandidateAction(
                    unit_idx=argmax.unit_idx, move_type=MOVE_MOVE, dest_idx=d_idx,
                    dest_col=argmax.dest_col, dest_row=argmax.dest_row,
                    charge_target_idx=argmax.charge_target_idx,
                    shoot_target_idx=argmax.shoot_target_idx,
                    is_shaken=False, advance_reachable=bool(adv[d_idx]),
                    dest_candidates=cands_arr, dest_features=feats, dest_mask=cand_mask,
                )
            else:
                argmax_cand = None
        else:
            argmax_cand = CandidateAction(
                unit_idx=argmax.unit_idx, move_type=argmax.move_type,
                dest_idx=-1, dest_col=None, dest_row=None,
                charge_target_idx=argmax.charge_target_idx,
                shoot_target_idx=argmax.shoot_target_idx,
                is_shaken=False, advance_reachable=False,
                dest_candidates=None, dest_features=None, dest_mask=None,
            )

        # Score every candidate (and the argmax)
        all_to_score = list(cands)
        if argmax_cand is not None:
            all_to_score.append(argmax_cand)

        for cand in all_to_score:
            # Restore before each scoring (compute_Q mutates)
            restore_game_state(st.snap, ctx.units_a, ctx.units_b, ctx.board)
            log_pi = compute_log_pi(
                model, cand, my_units, opp_units, ctx.board, player, st.round_num,
                my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts,
            )
            Q = compute_Q(
                model, cand, st.snap, ctx,
                current_is_a=st.current_is_a, round_num=st.round_num,
                m_rollouts=m_rollouts,
            )
            rec = dict(
                state_id=state_id, unit_idx=cand.unit_idx, move_type=cand.move_type,
                dest_col=cand.dest_col, dest_row=cand.dest_row,
                charge=cand.charge_target_idx, shoot=cand.shoot_target_idx,
                log_pi=log_pi, Q=Q,
            )
            records.append(rec)
            if cand is argmax_cand:
                argmax_per_state[state_id] = rec

        if (state_id + 1) % 5 == 0 or state_id + 1 == len(sampled):
            print(f"[probe]   scored {state_id+1}/{len(sampled)} states ({len(records)} candidates)")

    if not records:
        print("[probe] no records to analyze")
        return

    # Filter out records where log_pi is non-finite (impossible-under-policy
    # actions — these are likely action tuples that survived enumeration but
    # the model assigns zero probability to via masking).
    n_total = len(records)
    records_finite = [r for r in records if np.isfinite(r["log_pi"])]
    n_dropped = n_total - len(records_finite)
    if n_dropped:
        print(f"[probe]   dropped {n_dropped}/{n_total} candidates with non-finite log pi")
        # Also drop the corresponding argmax entries (in practice argmax should
        # always be finite, but be defensive)
        argmax_per_state = {
            sid: r for sid, r in argmax_per_state.items()
            if np.isfinite(r["log_pi"])
        }
    records = records_finite
    if not records:
        print("[probe] all records had non-finite log pi — aborting")
        return

    Qs = np.array([r["Q"] for r in records], dtype=np.float64)
    logpis = np.array([r["log_pi"] for r in records], dtype=np.float64)

    # Calibrate beta = std(Q) / std(log pi)
    sq, sp = float(np.std(Qs)), float(np.std(logpis))
    beta = sq / max(sp, 1e-6)
    print(f"\n[probe] calibrated beta = {beta:.4f}  (std(Q)={sq:.4f}, std(logpi)={sp:.4f})")

    gaps = Qs - beta * logpis

    # Bin by gap percentile
    qs_idx = np.argsort(gaps)  # ascending
    n = len(gaps)
    bins = {
        "bottom-25%": qs_idx[: n // 4],
        "mid-50%":    qs_idx[n // 4 : 3 * n // 4],
        "top-25%":    qs_idx[3 * n // 4 :],
    }

    print("\n[probe] mean rollout Q by gap-percentile bin:")
    for name, idx in bins.items():
        if len(idx) == 0:
            continue
        print(f"  {name:>12s}  n={len(idx):4d}  mean Q={Qs[idx].mean():+.4f}"
              f"  mean log pi={logpis[idx].mean():+.4f}")

    # Per-state comparison: top-1-by-gap vs argmax vs oracle
    by_state: dict[int, list[dict]] = {}
    for r in records:
        by_state.setdefault(r["state_id"], []).append(r)

    deltas_gap = []  # top-1-by-gap Q  -  argmax Q
    deltas_oracle = []  # top-1-by-Q Q  -  argmax Q
    deltas_bottom = []  # bottom-1-by-gap Q - argmax Q
    for sid, rs in by_state.items():
        if sid not in argmax_per_state:
            continue
        argmax_Q = argmax_per_state[sid]["Q"]
        # Exclude the argmax record itself when ranking
        rs_no_argmax = [r for r in rs if r is not argmax_per_state[sid]]
        if not rs_no_argmax:
            continue
        rs_no_argmax.sort(key=lambda r: r["Q"] - beta * r["log_pi"], reverse=True)
        top_gap_Q = rs_no_argmax[0]["Q"]
        bot_gap_Q = rs_no_argmax[-1]["Q"]
        oracle_Q = max(r["Q"] for r in rs_no_argmax)
        deltas_gap.append(top_gap_Q - argmax_Q)
        deltas_oracle.append(oracle_Q - argmax_Q)
        deltas_bottom.append(bot_gap_Q - argmax_Q)

    if deltas_gap:
        print(f"\n[probe] per-state vs policy-argmax (n={len(deltas_gap)} states):")
        print(f"  top-1-by-gap     - argmax  :  mean Δ = {np.mean(deltas_gap):+.4f}"
              f"   median = {np.median(deltas_gap):+.4f}"
              f"   wins (Δ>0) = {(np.array(deltas_gap)>0).mean():.2%}")
        print(f"  top-1-by-Q       - argmax  :  mean Δ = {np.mean(deltas_oracle):+.4f}"
              f"   median = {np.median(deltas_oracle):+.4f}"
              f"   wins (Δ>0) = {(np.array(deltas_oracle)>0).mean():.2%}")
        print(f"  bottom-1-by-gap  - argmax  :  mean Δ = {np.mean(deltas_bottom):+.4f}"
              f"   median = {np.median(deltas_bottom):+.4f}")

    print("\n[probe] interpretation:")
    print("  - if top-25% mean Q  >  bottom-25% mean Q by a clear margin: gap has ranking signal")
    print("  - if top-1-by-gap Δ  >  0 on average:     gap-ranked alternative beats argmax in rollouts")
    print("  - if top-1-by-gap Δ  ≈ top-1-by-Q Δ:     gap is identifying near-optimal moves under the policy gap")


def main():
    parser = argparse.ArgumentParser(description="Phase 0 probe for gap identifier")
    parser.add_argument("--games", type=int, default=5,
                        help="Self-play games for state collection")
    parser.add_argument("--states", type=int, default=20,
                        help="States sampled for scoring")
    parser.add_argument("--candidates", type=int, default=60,
                        help="Max candidate actions per state")
    parser.add_argument("--rollouts", type=int, default=4,
                        help="Dice rollouts per (state, action)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_probe(
        n_games=args.games,
        n_states=args.states,
        max_candidates_per_state=args.candidates,
        m_rollouts=args.rollouts,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
