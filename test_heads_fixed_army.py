"""Head-independence test with a fixed diverse army.

Uses a mirror-match: both sides run the same 5-unit army containing one
of each: Protectors (5 models), Shifters (5), Great Elemental (1),
Elemental Protectors (3), AG Tank (1).

For every activation, we record the active unit's output from each head,
then sequentially swap in the features of the other 4 unit types (same
trunk h, same post_move_rel, same dest candidates — only the active
unit's features change) and record the swapped outputs.

This directly answers: "when the active unit is Shifters and its
features are replaced by Great Elemental's, do the heads produce
different decisions?"
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
from dataclasses import dataclass
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

# The five unit template IDs in the fixed army
ARMY_TEMPLATES = [
    "protectors",          # [5 models, T0]  — horde
    "shifters",            # [5 models, T0]  — horde
    "great_elemental",     # [1 model, T12]  — tough solo
    "elemental_protectors",# [3 models, T3]  — small elite
    "ag_tank",             # [1 model, T12]  — vehicle
]

NUM_GAMES = 20  # smaller default since each game produces many swaps


# ------------------------------------------------------------------
# Event log
# ------------------------------------------------------------------

@dataclass
class HeadSwap:
    active_tid: str
    swap_tid: str  # "__self__" for the original (un-swapped) run
    # Per-head outputs (probs aligned to each head's output space)
    move_probs: list[float]         # 2
    charge_probs: list[float]       # 10 (masked softmax)
    charge_n_valid: int
    shoot_probs: list[float]        # 10 (masked softmax)
    shoot_n_valid: int
    dest_probs: list[float]         # variable length
    dest_n_valid: int


_swaps: list[HeadSwap] = []


# ------------------------------------------------------------------
# Army builder
# ------------------------------------------------------------------

def build_fixed_army() -> ArmyList:
    army = ArmyList()
    for tid in ARMY_TEMPLATES:
        entry = make_entry(tid, upgrades={}, ai_role="killer")
        entry.combat_preference = "ranged"
        army.entries.append(entry)
    return army


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _kl(p: list[float], q: list[float]) -> float:
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-10 and qi > 1e-10:
            kl += pi * math.log(pi / qi)
    return kl


def _top_match(p: list[float], q: list[float]) -> bool:
    if not p or not q:
        return True
    return p.index(max(p)) == q.index(max(q))


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> list[float]:
    masked = logits.masked_fill(~mask, float('-inf'))
    if not mask.any():
        return [0.0] * logits.shape[-1]
    return torch.softmax(masked, dim=-1).tolist()


# ------------------------------------------------------------------
# Monkey-patch
# ------------------------------------------------------------------

def _install_hook():
    import ml_integration_tactical as ml_mod
    _original_apply = ml_mod.apply_tactical_model

    def _patched_apply(model, friendly_units, enemy_units, round_num, board,
                       player, **kw):
        result = _original_apply(model, friendly_units, enemy_units, round_num,
                                 board, player, **kw)
        active = result[0]
        if active is None:
            return result

        # Only instrument activations of units in ARMY_TEMPLATES
        active_tid = active.unit.template_id
        if active_tid not in ARMY_TEMPLATES:
            return result

        # Find the slot for each of the 5 template types among friendly units
        tid_to_slot: dict[str, int] = {}
        for i, u in enumerate(friendly_units):
            if u.unit.template_id in ARMY_TEMPLATES and u.unit.template_id not in tid_to_slot:
                if u.models_alive > 0:
                    tid_to_slot[u.unit.template_id] = i

        if active_tid not in tid_to_slot:
            return result
        active_idx = tid_to_slot[active_tid]

        # Re-encode state
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

        with torch.no_grad():
            h, units, _ = model.trunk(state_vec.unsqueeze(0))
            h = h.squeeze(0)
            units = units.squeeze(0)

            active_can_charge = extract_can_charge_mask(state_vec, active_idx)
            positions = _get_model_space_positions(friendly_units, player)
            enemy_positions_ms = _get_model_space_positions(enemy_units, player)
            acx, acy = positions[active_idx]
            post_move_rel = compute_post_move_rel(acx, acy, enemy_positions_ms)

            move_onehot_move = F.one_hot(torch.tensor(MOVE_MOVE), NUM_MOVE_TYPES).float()

            # Dest candidates for ACTIVE unit (held fixed — changing them
            # would change the argmax space, making comparison meaningless).
            # But per-hex features (especially offensive-value & advance-
            # reachable) DO depend on the active unit's weapons/range/budget,
            # so we recompute them per swap using the swapped unit's slot.
            enemy_pos_set: set[tuple[int, int]] = set()
            for eu in enemy_units:
                if eu.models_alive > 0:
                    for pos in eu.alive_positions():
                        enemy_pos_set.add(pos)
            candidates, cand_mask, adv_reachable = compute_destination_candidates(
                active, board, enemy_pos_set, player)
            n_dest_valid = int(cand_mask.sum())
            fr = kw.get('friendly_ranged_matchups')
            er = kw.get('enemy_ranged_matchups')
            em = kw.get('enemy_melee_matchups')
            enemy_alive_np = np.array([
                i < len(enemy_units) and enemy_units[i].models_alive > 0
                for i in range(MAX_UNITS_PER_SIDE)], dtype=np.bool_)

            def _dest_feats_for(slot_idx: int, slot_unit: UnitState) -> tuple | None:
                """Compute dest features using slot_idx's matchup row and the
                slot_unit's movement budget and advance-reachable mask.

                Keeps the candidate set fixed (active unit's candidates) so the
                argmax space is comparable across swaps.
                """
                if n_dest_valid < 1:
                    return None
                # Recompute advance_reachable using the SWAPPED unit's advance
                # budget (so "can shoot from here" reflects that unit).
                swap_adv_budget = float(slot_unit.unit.advance_distance)
                swap_rush_budget = float(slot_unit.unit.rush_distance)
                # For hexes beyond swap's advance budget (but within active's
                # rush budget) — rank as rush-only for the swapped unit.
                # Approximate by comparing euclidean distance from the ACTIVE
                # unit's centre. Same candidates, but reclassify.
                acx_g, acy_g = active.centre()
                cols = candidates[:, 0].astype(np.float64)
                rows = candidates[:, 1].astype(np.float64)
                dx = cols - acx_g
                dy = rows - acy_g
                dists = np.sqrt(dx * dx + dy * dy)
                swap_adv_reach = (dists <= swap_adv_budget) & cand_mask
                # Budget (used for normalization in features) = swap rush budget
                dfn = compute_destination_features(
                    candidates, cand_mask, slot_unit, slot_idx, player,
                    enemy_units, enemy_alive_np, fr, er, em,
                    swap_rush_budget, advance_reachable=swap_adv_reach)
                return (torch.from_numpy(dfn).float(),
                        torch.from_numpy(cand_mask))

            # Active unit's dest features (original run)
            dest_orig = _dest_feats_for(active_idx, active)
            dest_feats_t_active = dest_orig[0] if dest_orig is not None else None
            dest_mask_t_active = dest_orig[1] if dest_orig is not None else None

            active_max_wr = max(
                (w.range_inches for w in active.unit.weapons if not w.melee),
                default=0.0)
            active_shoot_range_mask = compute_in_range_mask(
                post_move_rel, float(active_max_wr), enemy_alive_mask)
            charge_mask = enemy_alive_mask & active_can_charge

            def _run(unit_feats: torch.Tensor,
                     dest_feats_t: torch.Tensor | None,
                     dest_mask_t: torch.Tensor | None,
                     shoot_range_mask: torch.Tensor) -> dict:
                # move head
                h_uf = torch.cat([h, unit_feats])
                move_logits = model.move_type_head(h_uf)
                if not active_can_charge.any():
                    move_logits = move_logits.clone()
                    move_logits[MOVE_CHARGE] = float('-inf')
                move_probs = torch.softmax(move_logits, dim=-1).tolist()

                chosen_move = int(move_logits.argmax().item())
                move_onehot = F.one_hot(torch.tensor(chosen_move), NUM_MOVE_TYPES).float()
                h_uf_m = torch.cat([h, unit_feats, move_onehot])

                # Pointer heads read the acting unit's slice from `units` at
                # active_idx. Substitute unit_feats at that slot so the swap
                # affects candidate features (matchup row, survival, tough).
                units_swapped = units.clone()
                units_swapped[active_idx] = unit_feats

                # charge head (pointer)
                charge_logits = model.compute_charge_logits(
                    h, units_swapped, active_idx, enemy_alive_mask, active_can_charge,
                )
                charge_probs = _masked_softmax(charge_logits, charge_mask)

                # shoot head (pointer) — always conditions on MOVE_MOVE (matches game flow)
                shoot_logits = model.compute_shoot_logits(
                    h, units_swapped, active_idx, post_move_rel,
                    enemy_alive_mask, shoot_range_mask=shoot_range_mask,
                )
                shoot_probs = _masked_softmax(shoot_logits, shoot_range_mask)

                # dest head
                if dest_feats_t is not None:
                    h_uf_m_dest = torch.cat([h, unit_feats, move_onehot_move])
                    dest_logits = model.compute_dest_logits(
                        h_uf_m_dest.unsqueeze(0),
                        dest_feats_t.unsqueeze(0),
                        dest_mask_t.unsqueeze(0),
                    ).squeeze(0)
                    dest_logits = dest_logits.masked_fill(~dest_mask_t, float('-inf'))
                    dest_probs = torch.softmax(dest_logits, dim=-1).tolist()
                else:
                    dest_probs = []

                return {
                    'move_probs': move_probs,
                    'charge_probs': charge_probs,
                    'shoot_probs': shoot_probs,
                    'dest_probs': dest_probs,
                }

            # Original (active unit's own features + dest features + range)
            active_feats = model._extract_unit_features(units, active_idx)
            orig = _run(active_feats, dest_feats_t_active, dest_mask_t_active,
                        active_shoot_range_mask)
            _swaps.append(HeadSwap(
                active_tid=active_tid,
                swap_tid="__self__",
                move_probs=orig['move_probs'],
                charge_probs=orig['charge_probs'],
                charge_n_valid=int(charge_mask.sum().item()),
                shoot_probs=orig['shoot_probs'],
                shoot_n_valid=int(active_shoot_range_mask.sum().item()),
                dest_probs=orig['dest_probs'],
                dest_n_valid=n_dest_valid,
            ))

            # Swap in each other unit type's features (sequentially).
            # Recompute per-hex dest features using the SWAPPED unit's slot
            # (matchup row) and movement budget (advance-reachable flag).
            for swap_tid in ARMY_TEMPLATES:
                if swap_tid == active_tid:
                    continue
                if swap_tid not in tid_to_slot:
                    continue  # that unit is dead
                swap_idx = tid_to_slot[swap_tid]
                swap_unit = friendly_units[swap_idx]
                swap_feats = model._extract_unit_features(units, swap_idx)
                swap_dest = _dest_feats_for(swap_idx, swap_unit)
                dest_feats_t_swap = swap_dest[0] if swap_dest is not None else None
                dest_mask_t_swap = swap_dest[1] if swap_dest is not None else None
                # Recompute shoot range mask with SWAPPED unit's weapon range
                swap_max_wr = max(
                    (w.range_inches for w in swap_unit.unit.weapons if not w.melee),
                    default=0.0)
                swap_shoot_range_mask = compute_in_range_mask(
                    post_move_rel, float(swap_max_wr), enemy_alive_mask)
                swapped = _run(swap_feats, dest_feats_t_swap, dest_mask_t_swap,
                               swap_shoot_range_mask)
                _swaps.append(HeadSwap(
                    active_tid=active_tid,
                    swap_tid=swap_tid,
                    move_probs=swapped['move_probs'],
                    charge_probs=swapped['charge_probs'],
                    charge_n_valid=int(charge_mask.sum().item()),
                    shoot_probs=swapped['shoot_probs'],
                    shoot_n_valid=int(swap_shoot_range_mask.sum().item()),
                    dest_probs=swapped['dest_probs'],
                    dest_n_valid=n_dest_valid,
                ))

        return result

    ml_mod.apply_tactical_model = _patched_apply


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

def analyse(swaps: list[HeadSwap]):
    # Group by (active_tid, activation_id). We don't have an explicit
    # activation id, so walk the list: each "__self__" entry starts a new
    # activation and is followed by 0..4 swaps.
    activations: list[dict] = []  # each: {active_tid, orig: HeadSwap, swaps: {tid: HeadSwap}}
    current = None
    for s in swaps:
        if s.swap_tid == "__self__":
            if current is not None:
                activations.append(current)
            current = {'active_tid': s.active_tid, 'orig': s, 'swaps': {}}
        else:
            if current is not None:
                current['swaps'][s.swap_tid] = s
    if current is not None:
        activations.append(current)

    print("\n" + "=" * 70)
    print("HEAD INDEPENDENCE — FIXED 5-UNIT ARMY")
    print("=" * 70)
    print(f"\nArmy: {', '.join(ARMY_TEMPLATES)}")
    print(f"Total activations: {len(activations)}")

    # Activation counts by active unit type
    from collections import Counter
    counts = Counter(a['active_tid'] for a in activations)
    print(f"\n--- ACTIVATIONS BY UNIT TYPE ---")
    for tid in ARMY_TEMPLATES:
        print(f"  {tid:<25s} {counts.get(tid, 0):>4d}")

    # Summary table: agreement % and avg KL for (active, swap) pair, per head
    def _collect(head: str, filter_valid):
        """Returns dict[(active_tid, swap_tid)] = (n, agree_count, kl_sum)."""
        table: dict[tuple[str, str], list[float]] = {}
        for act in activations:
            orig = act['orig']
            if not filter_valid(orig):
                continue
            orig_p = getattr(orig, f'{head}_probs')
            if not orig_p:
                continue
            for swap_tid, swap in act['swaps'].items():
                swap_p = getattr(swap, f'{head}_probs')
                if not swap_p:
                    continue
                key = (act['active_tid'], swap_tid)
                if key not in table:
                    table[key] = [0, 0, 0.0]
                table[key][0] += 1
                if _top_match(orig_p, swap_p):
                    table[key][1] += 1
                table[key][2] += _kl(orig_p, swap_p)
        return table

    def _print_matrix(title: str, table: dict, note: str = ""):
        print(f"\n--- {title} ---")
        if note:
            print(f"  {note}")
        header = f"  {'active ↓ / swap →':<23s}"
        for tid in ARMY_TEMPLATES:
            header += f"{tid[:12]:>13s}"
        print(header)
        for atid in ARMY_TEMPLATES:
            row = f"  {atid:<23s}"
            for stid in ARMY_TEMPLATES:
                if atid == stid:
                    row += f"{'—':>13s}"
                    continue
                if (atid, stid) in table:
                    n, agree, kl_sum = table[(atid, stid)]
                    if n == 0:
                        row += f"{'  (0)':>13s}"
                    else:
                        pct = agree / n * 100
                        avg_kl = kl_sum / n
                        row += f" {pct:>4.0f}% k{avg_kl:>5.3f}"
                else:
                    row += f"{' (no data)':>13s}"
            print(row)

    move_table = _collect('move', lambda o: True)
    charge_table = _collect('charge', lambda o: o.charge_n_valid >= 2)
    shoot_table = _collect('shoot', lambda o: o.shoot_n_valid >= 2)
    dest_table = _collect('dest', lambda o: o.dest_n_valid >= 2)

    _print_matrix("MOVE TYPE: agree% / avgKL",
                  move_table, "(cell format: NN% k0.nnn)")
    _print_matrix("CHARGE TARGET: agree% / avgKL (≥2 valid)", charge_table)
    _print_matrix("SHOOT TARGET: agree% / avgKL (≥2 valid)", shoot_table)
    _print_matrix("DESTINATION: agree% / avgKL (≥2 valid)", dest_table)

    # Aggregate per head
    print(f"\n--- OVERALL PER-HEAD AGREEMENT (across all active×swap pairs) ---")
    for name, table in [('move', move_table), ('charge', charge_table),
                         ('shoot', shoot_table), ('dest', dest_table)]:
        total_n = sum(v[0] for v in table.values())
        total_agree = sum(v[1] for v in table.values())
        total_kl = sum(v[2] for v in table.values())
        if total_n == 0:
            print(f"  {name:<10s} no data")
            continue
        print(f"  {name:<10s} n={total_n:>5d}  "
              f"agree={total_agree/total_n*100:5.1f}%  avgKL={total_kl/total_n:.4f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    from game import simulate_game

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=NUM_GAMES)
    args = parser.parse_args()

    _install_hook()

    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded model from {checkpoint_path.name}")

    # Build the fixed army once and reuse
    print(f"\nBuilding fixed army: {ARMY_TEMPLATES}")
    army_template = build_fixed_army()
    for e in army_template.entries:
        print(f"  {e.template_id:<25s} cost={e.computed_cost}pts")
    total_cost = sum(e.computed_cost for e in army_template.entries)
    print(f"  total: {total_cost}pts")

    print(f"\nRunning {args.games} ML-vs-ML mirror-match games...\n")
    wins = {"A": 0, "B": 0, "draw": 0}
    t0 = time.time()

    for i in range(args.games):
        # Fresh deep copies for each game (fitness etc.)
        import copy
        army_a = copy.deepcopy(army_template)
        army_b = copy.deepcopy(army_template)
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")

        result = simulate_game(
            res_a, res_b, mode="objectives",
            states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
        )
        wins[result] += 1

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            per_game = elapsed / (i + 1)
            eta = per_game * (args.games - i - 1)
            # count activations so far: __self__ entries
            n_acts = sum(1 for s in _swaps if s.swap_tid == "__self__")
            print(f"  Game {i+1:3d}/{args.games}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)  "
                  f"activations: {n_acts}")

    elapsed = time.time() - t0
    print(f"\nCompleted {args.games} games in {elapsed:.1f}s")
    print(f"Results: A={wins['A']}  B={wins['B']}  Draw={wins['draw']}")
    n_acts = sum(1 for s in _swaps if s.swap_tid == "__self__")
    print(f"Recorded {n_acts} activations, {len(_swaps) - n_acts} swaps")

    analyse(_swaps)
