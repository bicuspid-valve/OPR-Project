"""Test whether the move/charge/dest/shoot heads depend on unit features
or are driven by board state.

For each friendly activation where the active unit has 2+ targets in range
(or 2+ reachable destinations), re-runs each head with every *other*
friendly unit's features swapped in (same trunk h). Reports:
  - % of swapped-unit outputs that agree with the active unit's argmax
  - Average KL divergence (original || swapped)

Run specifically for Shifters activations to see if swapping in Great
Elemental features changes the movement/charge/shoot decisions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path

from ml_training import load_model_state_dict
from ml_model_tactical import (
    TacticalModel, NUM_MOVE_TYPES, MOVE_MOVE, MOVE_CHARGE,
)
from ml_features import (
    encode_state_tactical, MAX_UNITS_PER_SIDE, extract_can_charge_mask,
)
from ml_integration_tactical import (
    compute_post_move_rel, compute_in_range_mask,
    compute_destination_candidates, compute_destination_features,
    _get_model_space_positions,
)
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList

_DIR = Path(__file__).resolve().parent

NUM_GAMES = 100

# ------------------------------------------------------------------
# Event log
# ------------------------------------------------------------------

@dataclass
class HeadSwap:
    active_name: str
    active_template_id: str
    swap_name: str
    swap_template_id: str
    # Move type (2 classes)
    move_probs_orig: list[float]
    move_probs_swap: list[float]
    # Charge target (10 slots, masked)
    charge_probs_orig: list[float]
    charge_probs_swap: list[float]
    charge_n_valid: int
    # Shoot target (10 slots, masked)
    shoot_probs_orig: list[float]
    shoot_probs_swap: list[float]
    shoot_n_valid: int
    # Destination (variable candidates)
    dest_probs_orig: list[float]
    dest_probs_swap: list[float]
    dest_n_valid: int


_swaps: list[HeadSwap] = []


# ------------------------------------------------------------------
# Filter: which active units to instrument
# ------------------------------------------------------------------

_FILTER_TEMPLATE_ID = None  # e.g. "shifters" or None for all


# ------------------------------------------------------------------
# Monkey-patch apply_tactical_model
# ------------------------------------------------------------------

def _kl(p: list[float], q: list[float]) -> float:
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-10 and qi > 1e-10:
            kl += pi * math.log(pi / qi)
    return kl


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> list[float]:
    masked = logits.masked_fill(~mask, float('-inf'))
    if not mask.any():
        return [0.0] * logits.shape[-1]
    return torch.softmax(masked, dim=-1).tolist()


def _install_hook(filter_tid: str | None):
    import ml_integration_tactical as ml_mod
    _original_apply = ml_mod.apply_tactical_model

    def _patched_apply(model, friendly_units, enemy_units, round_num, board,
                       player, **kw):
        result = _original_apply(model, friendly_units, enemy_units, round_num,
                                 board, player, **kw)
        active = result[0]
        if active is None:
            return result

        # Filter to specific unit type if requested
        if filter_tid is not None and active.unit.template_id != filter_tid:
            return result

        # Need fresh encode
        state_vec = encode_state_tactical(
            friendly_units, enemy_units, round_num, board, player,
            friendly_ranged_matchups=kw.get('friendly_ranged_matchups'),
            friendly_melee_matchups=kw.get('friendly_melee_matchups'),
            enemy_ranged_matchups=kw.get('enemy_ranged_matchups'),
            enemy_melee_matchups=kw.get('enemy_melee_matchups'),
            total_friendly_points=kw.get('total_friendly_points'),
            total_enemy_points=kw.get('total_enemy_points'),
        )
        enemy_alive_mask = torch.tensor(
            [i < len(enemy_units) and enemy_units[i].models_alive > 0
             for i in range(MAX_UNITS_PER_SIDE)], dtype=torch.bool)

        # Find active's slot
        active_idx = None
        for i, u in enumerate(friendly_units):
            if id(u) == id(active):
                active_idx = i
                break
        if active_idx is None:
            return result

        with torch.no_grad():
            h, units, _ = model.trunk(state_vec.unsqueeze(0))
            h = h.squeeze(0)
            units = units.squeeze(0)

            # Compute can_charge_mask and post_move_rel for the ACTIVE unit
            # (we keep these fixed — we're only swapping unit_features).
            active_can_charge = extract_can_charge_mask(state_vec, active_idx)
            positions = _get_model_space_positions(friendly_units, player)
            enemy_positions = _get_model_space_positions(enemy_units, player)
            acx, acy = positions[active_idx]
            # Use pre-movement position for post_move_rel (approximation — the
            # real post_move_rel uses the chosen dest, but here we're doing a
            # counterfactual "what would each unit decide from the SAME state")
            post_move_rel = compute_post_move_rel(acx, acy, enemy_positions)

            move_onehot_move = F.one_hot(torch.tensor(MOVE_MOVE), NUM_MOVE_TYPES).float()

            def _run_heads(unit_feats: torch.Tensor) -> dict:
                """Run all four heads with the given unit features."""
                # move_type head: h + uf
                h_uf = torch.cat([h, unit_feats])
                move_logits = model.move_type_head(h_uf)
                # mask charge option if no enemy in charge range
                no_chargeable = ~active_can_charge.any()
                if no_chargeable:
                    move_logits = move_logits.clone()
                    move_logits[MOVE_CHARGE] = float('-inf')
                move_probs = torch.softmax(move_logits, dim=-1).tolist()

                # Use the ACTIVE unit's argmax move_onehot for conditioning
                chosen_move = int(move_logits.argmax().item())
                move_onehot = F.one_hot(torch.tensor(chosen_move), NUM_MOVE_TYPES).float()

                # Pointer heads read the acting unit slice from `units` at
                # active_idx. Substitute unit_feats so the swap is visible to
                # candidate features (matchup row, survival, tough).
                units_swapped = units.clone()
                units_swapped[active_idx] = unit_feats

                charge_mask = enemy_alive_mask & active_can_charge
                charge_logits = model.compute_charge_logits(
                    h, units_swapped, active_idx, enemy_alive_mask, active_can_charge,
                )
                charge_probs = _masked_softmax(charge_logits, charge_mask)

                # shoot head (pointer)
                max_wr = max(
                    (w.range_inches for w in active.unit.weapons if not w.melee),
                    default=0.0)
                shoot_range_mask = compute_in_range_mask(post_move_rel, float(max_wr),
                                                         enemy_alive_mask)
                shoot_logits = model.compute_shoot_logits(
                    h, units_swapped, active_idx, post_move_rel,
                    enemy_alive_mask, shoot_range_mask=shoot_range_mask,
                )
                shoot_probs = _masked_softmax(shoot_logits, shoot_range_mask)

                return {
                    'move_probs': move_probs,
                    'charge_probs': charge_probs,
                    'charge_n_valid': int(charge_mask.sum().item()),
                    'shoot_probs': shoot_probs,
                    'shoot_n_valid': int(shoot_range_mask.sum().item()),
                }

            # Destination pointer — needs per-unit candidates
            def _run_dest(unit_feats: torch.Tensor,
                          dest_features_t: torch.Tensor,
                          dest_mask_t: torch.Tensor) -> list[float]:
                # Use MOVE_MOVE onehot (standard for dest)
                h_uf_m = torch.cat([h, unit_feats, move_onehot_move])
                dest_logits = model.compute_dest_logits(
                    h_uf_m.unsqueeze(0),
                    dest_features_t.unsqueeze(0),
                    dest_mask_t.unsqueeze(0),
                ).squeeze(0)
                # Mask invalid destinations
                dest_logits = dest_logits.masked_fill(~dest_mask_t, float('-inf'))
                if not dest_mask_t.any():
                    return [0.0] * dest_logits.shape[-1]
                return torch.softmax(dest_logits, dim=-1).tolist()

            # Compute dest candidates for ACTIVE unit (we keep candidates fixed
            # — only swapping unit features)
            enemy_pos_set: set[tuple[int, int]] = set()
            for eu in enemy_units:
                if eu.models_alive > 0:
                    for pos in eu.alive_positions():
                        enemy_pos_set.add(pos)

            candidates, cand_mask, adv_reachable = compute_destination_candidates(
                active, board, enemy_pos_set, player)
            budget = float(active.unit.rush_distance)
            n_dest_valid = int(cand_mask.sum())

            if n_dest_valid >= 2:
                # Get matchups for destination features
                fr = kw.get('friendly_ranged_matchups')
                er = kw.get('enemy_ranged_matchups')
                em = kw.get('enemy_melee_matchups')
                enemy_alive_np = np.array([
                    i < len(enemy_units) and enemy_units[i].models_alive > 0
                    for i in range(MAX_UNITS_PER_SIDE)], dtype=np.bool_)
                dest_feats_np = compute_destination_features(
                    candidates, cand_mask, active, active_idx, player,
                    enemy_units, enemy_alive_np, fr, er, em,
                    budget, advance_reachable=adv_reachable)
                dest_feats_t = torch.from_numpy(dest_feats_np).float()
                dest_mask_t = torch.from_numpy(cand_mask)
            else:
                dest_feats_t = None
                dest_mask_t = None

            # Run heads for the ACTIVE unit
            active_feats = model._extract_unit_features(units, active_idx)
            orig = _run_heads(active_feats)
            if dest_feats_t is not None:
                orig_dest = _run_dest(active_feats, dest_feats_t, dest_mask_t)
            else:
                orig_dest = []

            # Run for each OTHER alive friendly unit (features swapped in)
            for i in range(len(friendly_units)):
                if i == active_idx:
                    continue
                if friendly_units[i].models_alive <= 0:
                    continue
                swap_feats = model._extract_unit_features(units, i)
                swapped = _run_heads(swap_feats)
                if dest_feats_t is not None:
                    swap_dest = _run_dest(swap_feats, dest_feats_t, dest_mask_t)
                else:
                    swap_dest = []

                _swaps.append(HeadSwap(
                    active_name=active.unit.name,
                    active_template_id=active.unit.template_id,
                    swap_name=friendly_units[i].unit.name,
                    swap_template_id=friendly_units[i].unit.template_id,
                    move_probs_orig=orig['move_probs'],
                    move_probs_swap=swapped['move_probs'],
                    charge_probs_orig=orig['charge_probs'],
                    charge_probs_swap=swapped['charge_probs'],
                    charge_n_valid=orig['charge_n_valid'],
                    shoot_probs_orig=orig['shoot_probs'],
                    shoot_probs_swap=swapped['shoot_probs'],
                    shoot_n_valid=orig['shoot_n_valid'],
                    dest_probs_orig=orig_dest,
                    dest_probs_swap=swap_dest,
                    dest_n_valid=n_dest_valid,
                ))

        return result

    ml_mod.apply_tactical_model = _patched_apply


# ------------------------------------------------------------------
# Army loading
# ------------------------------------------------------------------

def load_army_from_hof(hof_entry: dict) -> ArmyList:
    army = ArmyList()
    for e in hof_entry["entries"]:
        entry = make_entry(
            e["template_id"],
            upgrades=e.get("upgrades", {}),
            ai_role=e.get("ai_role", "killer"),
        )
        entry.combat_preference = e.get("combat_preference", "ranged")
        army.entries.append(entry)
    return army


def has_unit(states: list, tid: str | None) -> bool:
    if tid is None:
        return True
    return any(s.unit.template_id == tid for s in states)


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

def _top_match(p: list[float], q: list[float]) -> bool:
    if not p or not q:
        return True
    return p.index(max(p)) == q.index(max(q))


def analyse(swaps: list[HeadSwap], filter_tid: str | None):
    label = f"active = {filter_tid}" if filter_tid else "all active units"
    print(f"\n{'=' * 70}")
    print(f"HEAD INDEPENDENCE — {label}")
    print(f"{'=' * 70}")
    print(f"\nTotal swap comparisons: {len(swaps)}")

    # Aggregate agreement & KL per head, limited to cases with ≥2 valid options
    head_metrics: dict[str, dict] = {
        'move (2 options)':   {'agree': [], 'kl': [], 'filter': lambda s: True},
        'charge (≥2 valid)':  {'agree': [], 'kl': [], 'filter': lambda s: s.charge_n_valid >= 2},
        'shoot  (≥2 valid)':  {'agree': [], 'kl': [], 'filter': lambda s: s.shoot_n_valid >= 2},
        'dest   (≥2 valid)':  {'agree': [], 'kl': [], 'filter': lambda s: s.dest_n_valid >= 2},
    }

    for s in swaps:
        for head_name, m in head_metrics.items():
            if not m['filter'](s):
                continue
            if head_name.startswith('move'):
                p, q = s.move_probs_orig, s.move_probs_swap
            elif head_name.startswith('charge'):
                p, q = s.charge_probs_orig, s.charge_probs_swap
            elif head_name.startswith('shoot'):
                p, q = s.shoot_probs_orig, s.shoot_probs_swap
            else:
                p, q = s.dest_probs_orig, s.dest_probs_swap
            if not p or not q:
                continue
            m['agree'].append(_top_match(p, q))
            m['kl'].append(_kl(p, q))

    print(f"\n--- AGREEMENT WITH ACTIVE UNIT'S ARGMAX (after feature swap) ---")
    print(f"  {'Head':<22s} {'N':>6s} {'Agree%':>7s} {'AvgKL':>7s}")
    for head_name, m in head_metrics.items():
        if not m['agree']:
            print(f"  {head_name:<22s} {0:>6d}  (no data)")
            continue
        n = len(m['agree'])
        pct = sum(m['agree']) / n * 100
        avg_kl = sum(m['kl']) / n
        print(f"  {head_name:<22s} {n:>6d} {pct:>6.1f}% {avg_kl:>7.4f}")

    # Disagreement breakdown by swapped unit type — for shoot + dest
    for head_name in ['shoot  (≥2 valid)', 'dest   (≥2 valid)', 'charge (≥2 valid)']:
        print(f"\n--- {head_name}: AGREEMENT BY SWAPPED UNIT ---")
        by_swap: dict[str, list[bool]] = {}
        kl_by_swap: dict[str, list[float]] = {}
        for s in swaps:
            if not head_metrics[head_name]['filter'](s):
                continue
            if head_name.startswith('shoot'):
                p, q = s.shoot_probs_orig, s.shoot_probs_swap
            elif head_name.startswith('dest'):
                p, q = s.dest_probs_orig, s.dest_probs_swap
            else:
                p, q = s.charge_probs_orig, s.charge_probs_swap
            if not p or not q:
                continue
            by_swap.setdefault(s.swap_name, []).append(_top_match(p, q))
            kl_by_swap.setdefault(s.swap_name, []).append(_kl(p, q))

        items = sorted(by_swap.items(), key=lambda x: -len(x[1]))[:15]
        print(f"  {'Swapped unit':<50s} {'N':>5s} {'Agree%':>7s} {'AvgKL':>7s}")
        for name, agrees in items:
            n = len(agrees)
            pct = sum(agrees) / n * 100
            avg_k = sum(kl_by_swap[name]) / n
            print(f"  {name:<50s} {n:>5d} {pct:>6.1f}% {avg_k:>7.4f}")

    # Focused look: for Shifters activations, swap in Great Elemental — show examples
    if filter_tid == 'shifters':
        ge_swaps = [s for s in swaps if s.swap_template_id == 'great_elemental']
        print(f"\n--- SHIFTERS → GREAT ELEMENTAL SWAPS: {len(ge_swaps)} events ---")
        if ge_swaps:
            # move
            ma = sum(1 for s in ge_swaps if _top_match(s.move_probs_orig, s.move_probs_swap))
            print(f"  move agree: {ma}/{len(ge_swaps)} ({ma/len(ge_swaps)*100:.1f}%)")
            # charge (if valid)
            ch = [s for s in ge_swaps if s.charge_n_valid >= 2]
            if ch:
                ca = sum(1 for s in ch if _top_match(s.charge_probs_orig, s.charge_probs_swap))
                print(f"  charge agree (≥2 valid, n={len(ch)}): {ca}/{len(ch)} "
                      f"({ca/len(ch)*100:.1f}%)")
            # shoot
            sh = [s for s in ge_swaps if s.shoot_n_valid >= 2]
            if sh:
                sa = sum(1 for s in sh if _top_match(s.shoot_probs_orig, s.shoot_probs_swap))
                print(f"  shoot  agree (≥2 valid, n={len(sh)}): {sa}/{len(sh)} "
                      f"({sa/len(sh)*100:.1f}%)")
            # dest
            de = [s for s in ge_swaps if s.dest_n_valid >= 2]
            if de:
                da = sum(1 for s in de if _top_match(s.dest_probs_orig, s.dest_probs_swap))
                avg_k = sum(_kl(s.dest_probs_orig, s.dest_probs_swap) for s in de) / len(de)
                print(f"  dest   agree (≥2 valid, n={len(de)}): {da}/{len(de)} "
                      f"({da/len(de)*100:.1f}%)  avgKL={avg_k:.4f}")

                # Example disagreements
                disagreements = [s for s in de
                                 if not _top_match(s.dest_probs_orig, s.dest_probs_swap)]
                if disagreements:
                    print(f"\n  First 5 dest disagreements:")
                    for s in disagreements[:5]:
                        top_o = s.dest_probs_orig.index(max(s.dest_probs_orig))
                        top_s = s.dest_probs_swap.index(max(s.dest_probs_swap))
                        print(f"    Shifters picks hex {top_o} @ {max(s.dest_probs_orig):.1%}  |  "
                              f"GE picks hex {top_s} @ {max(s.dest_probs_swap):.1%}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    from game import simulate_game

    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-tid", default="shifters",
                        help="Only instrument activations of this template_id (default: shifters)")
    parser.add_argument("--games", type=int, default=NUM_GAMES)
    args = parser.parse_args()

    filter_tid = args.filter_tid if args.filter_tid != "all" else None
    _install_hook(filter_tid)

    hof_path = _DIR / "results" / "hall_of_fame_ml.json"
    with open(hof_path) as f:
        hof_ml_data = json.load(f)
    print(f"Loaded {len(hof_ml_data)} armies from hall_of_fame_ml.json")

    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded model from {checkpoint_path.name}")

    target_label = filter_tid if filter_tid else "any unit"
    print(f"\nRunning {args.games} ML-vs-ML games — instrumenting '{target_label}' activations...\n")

    wins = {"A": 0, "B": 0, "draw": 0}
    games_played = 0
    games_attempted = 0
    t0 = time.time()

    while games_played < args.games:
        army_a = load_army_from_hof(random.choice(hof_ml_data))
        army_b = load_army_from_hof(random.choice(hof_ml_data))
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")

        games_attempted += 1
        # Require at least one of the filter units in the game
        if filter_tid is not None and not (has_unit(sa, filter_tid) or has_unit(sb, filter_tid)):
            continue

        result = simulate_game(
            res_a, res_b, mode="objectives",
            states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
        )
        wins[result] += 1
        games_played += 1

        if games_played % 10 == 0:
            elapsed = time.time() - t0
            per_game = elapsed / games_played
            eta = per_game * (args.games - games_played)
            print(f"  Game {games_played:3d}/{args.games}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)  "
                  f"swaps: {len(_swaps)}  (attempted: {games_attempted})")

    elapsed = time.time() - t0
    print(f"\nCompleted {args.games} games in {elapsed:.1f}s "
          f"({games_attempted} attempted, {games_attempted - args.games} skipped)")
    print(f"Results: A={wins['A']}  B={wins['B']}  Draw={wins['draw']}")
    print(f"Recorded {len(_swaps)} swap comparisons")

    analyse(_swaps, filter_tid)
