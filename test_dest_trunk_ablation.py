"""Ablate the trunk's influence on the destination head.

For each activation in a fixed-army mirror match, run the destination head
three ways on the same per-hex features:
  (1) real h     (the trunk output as normal)
  (2) zero h     (h replaced by a zero vector)
  (3) random h   (h replaced by a fixed random vector, kept constant
                  across activations to isolate "non-informative trunk")

If the dest head's output barely changes when h is zeroed/randomised,
the head is essentially a function of the per-hex features (offensive
value, advance-reachable, proximity) and not of the global game context.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

from ml_training import load_model_state_dict
from ml_model_tactical import (
    TacticalModel, NUM_MOVE_TYPES, MOVE_MOVE, TRUNK_WIDTH,
)
from ml_features import (
    encode_state_tactical, MAX_UNITS_PER_SIDE,
)
from ml_integration_tactical import (
    compute_destination_candidates, compute_destination_features,
)
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList

_DIR = Path(__file__).resolve().parent

ARMY_TEMPLATES = [
    "protectors",
    "shifters",
    "great_elemental",
    "elemental_protectors",
    "ag_tank",
]

NUM_GAMES = 100


# ------------------------------------------------------------------
# Event log
# ------------------------------------------------------------------

@dataclass
class DestAblation:
    active_tid: str
    n_valid: int
    probs_real: list[float]
    probs_zero: list[float]
    probs_rand: list[float]


_events: list[DestAblation] = []


# Fixed random h — sampled once at module load, reused across all calls
_RANDOM_H: torch.Tensor | None = None


def _get_random_h() -> torch.Tensor:
    global _RANDOM_H
    if _RANDOM_H is None:
        # Seed this specifically so runs are reproducible
        g = torch.Generator()
        g.manual_seed(42)
        # Match stem output distribution: LayerNorm → ReLU, so values are
        # non-negative. Use a half-normal-ish distribution to roughly match.
        _RANDOM_H = torch.randn(TRUNK_WIDTH, generator=g).abs() * 0.5
    return _RANDOM_H


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
# Monkey-patch
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


def _topk_overlap(p: list[float], q: list[float], k: int) -> int:
    """How many of p's top-k are also in q's top-k."""
    n = len(p)
    if n == 0:
        return 0
    top_p = set(sorted(range(n), key=lambda i: -p[i])[:k])
    top_q = set(sorted(range(n), key=lambda i: -q[i])[:k])
    return len(top_p & top_q)


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
        active_tid = active.unit.template_id
        if active_tid not in ARMY_TEMPLATES:
            return result

        # Find active slot
        active_idx = None
        for i, u in enumerate(friendly_units):
            if id(u) == id(active):
                active_idx = i
                break
        if active_idx is None:
            return result

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
            h_real, units, _ = model.trunk(state_vec.unsqueeze(0))
            h_real = h_real.squeeze(0)
            units = units.squeeze(0)

            unit_feats = model._extract_unit_features(units, active_idx)
            move_onehot_move = F.one_hot(torch.tensor(MOVE_MOVE), NUM_MOVE_TYPES).float()

            # Build dest candidates + features (using active unit's correct info)
            enemy_pos_set: set[tuple[int, int]] = set()
            for eu in enemy_units:
                if eu.models_alive > 0:
                    for pos in eu.alive_positions():
                        enemy_pos_set.add(pos)
            candidates, cand_mask, adv_reachable = compute_destination_candidates(
                active, board, enemy_pos_set, player)
            n_valid = int(cand_mask.sum())
            if n_valid < 2:
                return result

            fr = kw.get('friendly_ranged_matchups')
            er = kw.get('enemy_ranged_matchups')
            em = kw.get('enemy_melee_matchups')
            enemy_alive_np = np.array([
                i < len(enemy_units) and enemy_units[i].models_alive > 0
                for i in range(MAX_UNITS_PER_SIDE)], dtype=np.bool_)
            budget = float(active.unit.rush_distance)
            dest_feats_np = compute_destination_features(
                candidates, cand_mask, active, active_idx, player,
                enemy_units, enemy_alive_np, fr, er, em,
                budget, advance_reachable=adv_reachable)
            dest_feats_t = torch.from_numpy(dest_feats_np).float()
            dest_mask_t = torch.from_numpy(cand_mask)

            def _dest_probs(h_vec: torch.Tensor) -> list[float]:
                h_uf_m = torch.cat([h_vec, unit_feats, move_onehot_move])
                dest_logits = model.compute_dest_logits(
                    h_uf_m.unsqueeze(0),
                    dest_feats_t.unsqueeze(0),
                    dest_mask_t.unsqueeze(0),
                ).squeeze(0)
                dest_logits = dest_logits.masked_fill(~dest_mask_t, float('-inf'))
                return torch.softmax(dest_logits, dim=-1).tolist()

            h_zero = torch.zeros_like(h_real)
            h_rand = _get_random_h()

            probs_real = _dest_probs(h_real)
            probs_zero = _dest_probs(h_zero)
            probs_rand = _dest_probs(h_rand)

        _events.append(DestAblation(
            active_tid=active_tid,
            n_valid=n_valid,
            probs_real=probs_real,
            probs_zero=probs_zero,
            probs_rand=probs_rand,
        ))

        return result

    ml_mod.apply_tactical_model = _patched_apply


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

def analyse(events: list[DestAblation]):
    print("\n" + "=" * 70)
    print("DEST HEAD — TRUNK ABLATION")
    print("=" * 70)
    print(f"\nArmy: {', '.join(ARMY_TEMPLATES)}")
    print(f"Total activations with ≥2 valid dest candidates: {len(events)}")
    if not events:
        return

    # Baseline: random h compared to real h
    def _summary(label: str, getter):
        agree = 0
        kl_sum = 0.0
        top3_sum = 0
        max_prob_real = 0.0
        max_prob_alt = 0.0
        for e in events:
            alt = getter(e)
            if _top_match(e.probs_real, alt):
                agree += 1
            kl_sum += _kl(e.probs_real, alt)
            k = min(3, e.n_valid)
            top3_sum += _topk_overlap(e.probs_real, alt, k) / k
            max_prob_real += max(e.probs_real)
            max_prob_alt += max(alt)
        n = len(events)
        print(f"\n--- {label} ---")
        print(f"  argmax agree:           {agree}/{n}  ({agree/n*100:.1f}%)")
        print(f"  avg KL(real || {label.split()[0].lower()}): {kl_sum/n:.4f}")
        print(f"  avg top-3 overlap:      {top3_sum/n*100:.1f}%")
        print(f"  avg max-prob real:      {max_prob_real/n:.4f}")
        print(f"  avg max-prob alt:       {max_prob_alt/n:.4f}")

    _summary("real h vs ZERO h", lambda e: e.probs_zero)
    _summary("real h vs RANDOM h (fixed)", lambda e: e.probs_rand)

    # Per-active-unit breakdown
    print(f"\n--- PER ACTIVE UNIT (zero h) ---")
    print(f"  {'Active':<25s} {'N':>5s} {'Agree%':>7s} {'AvgKL':>7s} {'Top3%':>6s}")
    for tid in ARMY_TEMPLATES:
        sub = [e for e in events if e.active_tid == tid]
        if not sub:
            continue
        agree = sum(1 for e in sub if _top_match(e.probs_real, e.probs_zero))
        kl = sum(_kl(e.probs_real, e.probs_zero) for e in sub) / len(sub)
        top3 = sum(_topk_overlap(e.probs_real, e.probs_zero, min(3, e.n_valid)) / min(3, e.n_valid)
                   for e in sub) / len(sub)
        print(f"  {tid:<25s} {len(sub):>5d} {agree/len(sub)*100:>6.1f}% "
              f"{kl:>7.4f} {top3*100:>5.1f}%")

    print(f"\n--- PER ACTIVE UNIT (random h) ---")
    print(f"  {'Active':<25s} {'N':>5s} {'Agree%':>7s} {'AvgKL':>7s} {'Top3%':>6s}")
    for tid in ARMY_TEMPLATES:
        sub = [e for e in events if e.active_tid == tid]
        if not sub:
            continue
        agree = sum(1 for e in sub if _top_match(e.probs_real, e.probs_rand))
        kl = sum(_kl(e.probs_real, e.probs_rand) for e in sub) / len(sub)
        top3 = sum(_topk_overlap(e.probs_real, e.probs_rand, min(3, e.n_valid)) / min(3, e.n_valid)
                   for e in sub) / len(sub)
        print(f"  {tid:<25s} {len(sub):>5d} {agree/len(sub)*100:>6.1f}% "
              f"{kl:>7.4f} {top3*100:>5.1f}%")

    # Reference: compare zero vs random (to each other) to see if the
    # head's output depends on WHICH non-informative h we use
    print(f"\n--- SANITY: zero h vs random h (should be similar if head ignores h) ---")
    agree = sum(1 for e in events if _top_match(e.probs_zero, e.probs_rand))
    kl = sum(_kl(e.probs_zero, e.probs_rand) for e in events) / len(events)
    print(f"  argmax agree: {agree}/{len(events)}  ({agree/len(events)*100:.1f}%)")
    print(f"  avg KL(zero || random): {kl:.4f}")

    # Entropy comparison: does the head become less confident without h?
    print(f"\n--- CONFIDENCE (avg top-choice prob) ---")
    avg_real = sum(max(e.probs_real) for e in events) / len(events)
    avg_zero = sum(max(e.probs_zero) for e in events) / len(events)
    avg_rand = sum(max(e.probs_rand) for e in events) / len(events)
    print(f"  real h:   {avg_real:.4f}")
    print(f"  zero h:   {avg_zero:.4f}")
    print(f"  random h: {avg_rand:.4f}")


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

    army_template = build_fixed_army()
    total_cost = sum(e.computed_cost for e in army_template.entries)
    print(f"Army: {[e.template_id for e in army_template.entries]}")
    print(f"Total cost: {total_cost}pts")
    print(f"Random-h vector seed: 42  (norm={_get_random_h().norm().item():.3f})")

    print(f"\nRunning {args.games} mirror-match games...\n")
    wins = {"A": 0, "B": 0, "draw": 0}
    t0 = time.time()

    for i in range(args.games):
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

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (args.games - i - 1)
            print(f"  Game {i+1:3d}/{args.games}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)  "
                  f"events: {len(_events)}")

    elapsed = time.time() - t0
    print(f"\nCompleted {args.games} games in {elapsed:.1f}s")
    print(f"Results: A={wins['A']}  B={wins['B']}  Draw={wins['draw']}")
    print(f"Events: {len(_events)}")

    analyse(_events)
