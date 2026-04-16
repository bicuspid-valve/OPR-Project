"""Inspect the Burst Mortar's shoot-target probability distribution.

For each Burst Mortar activation, records:
  - How many targets are in range
  - The softmax probabilities over all in-range targets
  - Which target was chosen and its properties
  - The entropy of the targeting distribution

Runs 100 ML-vs-ML games with at least 1 Burst Mortar per game.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import torch
from dataclasses import dataclass, field
from pathlib import Path

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList

_DIR = Path(__file__).resolve().parent

NUM_GAMES = 100


# ------------------------------------------------------------------
# Event log
# ------------------------------------------------------------------

@dataclass
class TargetCandidate:
    slot_idx: int
    name: str
    models_alive: int
    original_size: int
    tough: int
    prob: float


@dataclass
class ShootDecision:
    n_in_range: int
    entropy: float
    max_prob: float
    chosen_idx: int
    candidates: list[TargetCandidate]
    owner: str
    round_num: int


_decisions: list[ShootDecision] = []


# ------------------------------------------------------------------
# Monkey-patch apply_tactical_model to capture targeting distributions
# ------------------------------------------------------------------

def _classify_burst_mortar(unit_state) -> bool:
    ru = unit_state.unit
    if ru.template_id != "support_artillery":
        return False
    return any(w.name == "Burst Mortar" for w in ru.weapons)


_burst_mortar_ids: set[int] = set()


def _install_hooks():
    import ml_integration_tactical as ml_mod
    _original_apply = ml_mod.apply_tactical_model

    def _patched_apply(model, friendly_units, enemy_units, round_num, board,
                       player, **kw):
        result = _original_apply(model, friendly_units, enemy_units, round_num,
                                 board, player, **kw)
        # result = (active, target_ranking, action, goal, charge_target, reason, assessment)
        active = result[0]
        if active is None:
            return result

        if id(active) not in _burst_mortar_ids:
            return result

        assessment = result[6]
        target_scores = assessment.get('target_scores', [])
        target_ranking = assessment.get('target_ranking', [])
        action = assessment.get('action', '')
        shoot_idx = assessment.get('shoot_target_idx', -1)

        # Only log if the unit is shooting (hold or advance, not rush/charge)
        if action in ('rush',):
            return result

        # Build candidate list from scores
        candidates = []
        n_in_range = 0
        for slot_idx, prob in enumerate(target_scores):
            if prob > 0 and slot_idx < len(enemy_units) and enemy_units[slot_idx].models_alive > 0:
                eu = enemy_units[slot_idx]
                candidates.append(TargetCandidate(
                    slot_idx=slot_idx,
                    name=eu.unit.name,
                    models_alive=eu.models_alive,
                    original_size=eu.unit.models,
                    tough=eu.unit.tough,
                    prob=prob,
                ))
                n_in_range += 1

        if n_in_range == 0:
            return result

        # Sort by probability (descending)
        candidates.sort(key=lambda c: c.prob, reverse=True)

        # Compute entropy
        entropy = 0.0
        max_prob = 0.0
        for c in candidates:
            if c.prob > 0:
                entropy -= c.prob * math.log(c.prob)
                max_prob = max(max_prob, c.prob)

        _decisions.append(ShootDecision(
            n_in_range=n_in_range,
            entropy=entropy,
            max_prob=max_prob,
            chosen_idx=shoot_idx,
            candidates=candidates,
            owner=active.owner,
            round_num=round_num,
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


def has_burst_mortar(states: list) -> bool:
    return any(_classify_burst_mortar(s) for s in states)


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

def analyse(decisions: list[ShootDecision]):
    print("\n" + "=" * 70)
    print("BURST MORTAR — TARGETING PREFERENCES")
    print("=" * 70)
    print(f"\nTotal activations with targets in range: {len(decisions)}")

    # --- Number of targets in range ---
    range_counts: dict[int, int] = {}
    for d in decisions:
        range_counts[d.n_in_range] = range_counts.get(d.n_in_range, 0) + 1
    print(f"\n--- TARGETS IN RANGE ---")
    for k in sorted(range_counts):
        print(f"  {k} targets: {range_counts[k]} activations "
              f"({range_counts[k]/len(decisions)*100:.1f}%)")

    # --- Entropy and confidence ---
    avg_entropy = sum(d.entropy for d in decisions) / len(decisions)
    avg_max_prob = sum(d.max_prob for d in decisions) / len(decisions)
    # Max possible entropy for reference
    avg_n = sum(d.n_in_range for d in decisions) / len(decisions)
    max_ent_ref = math.log(avg_n) if avg_n > 1 else 0
    print(f"\n--- CONFIDENCE ---")
    print(f"  avg targets in range:    {avg_n:.1f}")
    print(f"  avg entropy:             {avg_entropy:.3f}  "
          f"(max for {avg_n:.0f} targets = {max_ent_ref:.3f})")
    print(f"  avg top-choice prob:     {avg_max_prob:.3f}")

    # Entropy distribution
    ent_buckets = {"< 0.5": 0, "0.5-1.0": 0, "1.0-1.5": 0, "> 1.5": 0}
    for d in decisions:
        if d.entropy < 0.5:
            ent_buckets["< 0.5"] += 1
        elif d.entropy < 1.0:
            ent_buckets["0.5-1.0"] += 1
        elif d.entropy < 1.5:
            ent_buckets["1.0-1.5"] += 1
        else:
            ent_buckets["> 1.5"] += 1
    print(f"\n  Entropy distribution:")
    for label, count in ent_buckets.items():
        print(f"    {label:>8s}: {count:>3d} ({count/len(decisions)*100:.1f}%)")

    # Top-choice probability distribution
    conf_buckets = {"> 0.9": 0, "0.7-0.9": 0, "0.5-0.7": 0, "0.3-0.5": 0, "< 0.3": 0}
    for d in decisions:
        if d.max_prob > 0.9:
            conf_buckets["> 0.9"] += 1
        elif d.max_prob > 0.7:
            conf_buckets["0.7-0.9"] += 1
        elif d.max_prob > 0.5:
            conf_buckets["0.5-0.7"] += 1
        elif d.max_prob > 0.3:
            conf_buckets["0.3-0.5"] += 1
        else:
            conf_buckets["< 0.3"] += 1
    print(f"\n  Top-choice probability distribution:")
    for label, count in conf_buckets.items():
        print(f"    {label:>8s}: {count:>3d} ({count/len(decisions)*100:.1f}%)")

    # --- What does it favour? ---
    # Aggregate probability mass by target type
    total_prob_multi = 0.0
    total_prob_single = 0.0
    total_prob_tough = 0.0
    total_prob_nontough = 0.0
    total_prob = 0.0
    for d in decisions:
        for c in d.candidates:
            total_prob += c.prob
            if c.models_alive > 1:
                total_prob_multi += c.prob
            else:
                total_prob_single += c.prob
            if c.tough >= 3:
                total_prob_tough += c.prob
            else:
                total_prob_nontough += c.prob

    print(f"\n--- AGGREGATE PROBABILITY MASS BY TARGET TYPE ---")
    print(f"  (total probability sums to {total_prob:.1f} across {len(decisions)} activations)")
    print(f"  Multi-model (>1 alive): {total_prob_multi:.1f}  "
          f"({total_prob_multi/total_prob*100:.1f}%)")
    print(f"  Single-model (1 alive): {total_prob_single:.1f}  "
          f"({total_prob_single/total_prob*100:.1f}%)")
    print(f"  Tough(3+):              {total_prob_tough:.1f}  "
          f"({total_prob_tough/total_prob*100:.1f}%)")
    print(f"  Non-tough:              {total_prob_nontough:.1f}  "
          f"({total_prob_nontough/total_prob*100:.1f}%)")

    # --- Per-target-name probability mass ---
    name_prob: dict[str, float] = {}
    name_count: dict[str, int] = {}  # how many times this target was a candidate
    name_chosen: dict[str, int] = {}  # how many times this target was chosen
    for d in decisions:
        for c in d.candidates:
            label = f"{c.name} [{c.original_size}mod, T{c.tough}]"
            name_prob[label] = name_prob.get(label, 0.0) + c.prob
            name_count[label] = name_count.get(label, 0) + 1
            if c.slot_idx == d.chosen_idx:
                name_chosen[label] = name_chosen.get(label, 0) + 1

    print(f"\n--- PER-TARGET PREFERENCES (sorted by total prob mass) ---")
    print(f"  {'Target':<55s} {'TotProb':>7s} {'AvgProb':>7s} {'#Avail':>6s} {'#Chosen':>7s}")
    for label, prob in sorted(name_prob.items(), key=lambda x: -x[1])[:25]:
        avg_p = prob / name_count[label]
        chosen = name_chosen.get(label, 0)
        print(f"  {label:<55s} {prob:>7.1f} {avg_p:>7.3f} {name_count[label]:>6d} {chosen:>7d}")

    # --- Example activations (show full distributions) ---
    # Pick a few interesting ones: high entropy, low entropy, many targets
    print(f"\n--- EXAMPLE ACTIVATIONS ---")

    # Highest entropy
    by_ent = sorted(decisions, key=lambda d: d.entropy, reverse=True)
    for label, picker in [
        ("Highest entropy (most uncertain)", lambda: by_ent[0]),
        ("Lowest entropy (most confident)", lambda: by_ent[-1]),
        ("Most targets in range", lambda: max(decisions, key=lambda d: d.n_in_range)),
    ]:
        d = picker()
        print(f"\n  {label}:")
        print(f"    {d.n_in_range} targets in range, entropy={d.entropy:.3f}, "
              f"round={d.round_num}")
        for c in d.candidates:
            marker = " ← CHOSEN" if c.slot_idx == d.chosen_idx else ""
            print(f"      [{c.prob:5.1%}] {c.name} ({c.models_alive}/{c.original_size} models, "
                  f"T{c.tough}){marker}")

    # Show 5 random activations with 3+ targets
    multi_target = [d for d in decisions if d.n_in_range >= 3]
    if len(multi_target) >= 5:
        print(f"\n  5 random activations with 3+ targets:")
        for d in random.sample(multi_target, 5):
            print(f"\n    {d.n_in_range} targets, entropy={d.entropy:.3f}, round={d.round_num}:")
            for c in d.candidates:
                marker = " ← CHOSEN" if c.slot_idx == d.chosen_idx else ""
                print(f"      [{c.prob:5.1%}] {c.name} ({c.models_alive}/{c.original_size} models, "
                      f"T{c.tough}){marker}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    from game import simulate_game

    _install_hooks()

    hof_path = _DIR / "results" / "hall_of_fame_ml.json"
    if not hof_path.exists():
        print(f"Error: {hof_path} not found")
        sys.exit(1)
    with open(hof_path) as f:
        hof_ml_data = json.load(f)
    print(f"Loaded {len(hof_ml_data)} armies from hall_of_fame_ml.json")

    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found")
        sys.exit(1)
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded model from {checkpoint_path.name}")

    print(f"\nRunning {NUM_GAMES} ML-vs-ML games with Burst Mortar...\n")
    wins = {"A": 0, "B": 0, "draw": 0}
    games_played = 0
    games_attempted = 0
    t0 = time.time()

    while games_played < NUM_GAMES:
        army_a = load_army_from_hof(random.choice(hof_ml_data))
        army_b = load_army_from_hof(random.choice(hof_ml_data))
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")

        games_attempted += 1

        if not has_burst_mortar(sa) and not has_burst_mortar(sb):
            continue

        _burst_mortar_ids.clear()
        for s in sa + sb:
            if _classify_burst_mortar(s):
                _burst_mortar_ids.add(id(s))

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
            eta = per_game * (NUM_GAMES - games_played)
            print(f"  Game {games_played:3d}/{NUM_GAMES}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)  "
                  f"decisions: {len(_decisions)}  (attempted: {games_attempted})")

    elapsed = time.time() - t0
    print(f"\nCompleted {NUM_GAMES} games in {elapsed:.1f}s "
          f"({games_attempted} attempted, {games_attempted - NUM_GAMES} skipped)")
    print(f"Results: A={wins['A']}  B={wins['B']}  Draw={wins['draw']}")
    print(f"Recorded {len(_decisions)} targeting decisions")

    analyse(_decisions)
