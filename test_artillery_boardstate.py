"""Test whether the Burst Mortar's targeting is driven by board state or unit features.

For each Burst Mortar activation, re-runs the shoot head with every other
friendly unit's features swapped in (same trunk h, same post_move_rel).
If the targeting distribution barely changes, the shoot head is driven by
the global board state (h), not the unit-specific features.

Runs 100 ML-vs-ML games with at least 1 Burst Mortar per game.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel, NUM_MOVE_TYPES, MOVE_MOVE
from ml_features import (
    encode_state_tactical, MAX_UNITS_PER_SIDE, TACTICAL_UNIT_FEATURES,
)
from ml_integration_tactical import compute_post_move_rel, compute_in_range_mask
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList

_DIR = Path(__file__).resolve().parent

NUM_GAMES = 100


# ------------------------------------------------------------------
# Event log
# ------------------------------------------------------------------

@dataclass
class SwapResult:
    """One Burst Mortar activation with counterfactual targeting from other units."""
    n_in_range: int
    burst_mortar_probs: list[float]       # softmax probs (10,) for the BM
    burst_mortar_chosen: int              # chosen enemy slot
    other_unit_probs: list[list[float]]   # one (10,) list per other friendly unit
    other_unit_names: list[str]
    other_unit_template_ids: list[str]
    enemy_names: list[str]                # name per slot (10,)
    enemy_models: list[int]               # models alive per slot
    enemy_tough: list[int]                # tough per slot
    enemy_alive: list[bool]               # alive mask per slot


_results: list[SwapResult] = []


# ------------------------------------------------------------------
# Burst Mortar detection
# ------------------------------------------------------------------

def _is_burst_mortar(unit_state) -> bool:
    ru = unit_state.unit
    if ru.template_id != "support_artillery":
        return False
    return any(w.name == "Burst Mortar" for w in ru.weapons)


_burst_mortar_ids: set[int] = set()


# ------------------------------------------------------------------
# Monkey-patch
# ------------------------------------------------------------------

def _install_hooks():
    import ml_integration_tactical as ml_mod
    _original_apply = ml_mod.apply_tactical_model

    def _patched_apply(model, friendly_units, enemy_units, round_num, board,
                       player, **kw):
        result = _original_apply(model, friendly_units, enemy_units, round_num,
                                 board, player, **kw)
        active = result[0]
        if active is None or id(active) not in _burst_mortar_ids:
            return result

        assessment = result[6]
        action = assessment.get('action', '')
        if action == 'rush':
            return result

        # Re-encode state and run trunk
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

        # Find BM's slot index
        bm_idx = None
        for i, u in enumerate(friendly_units):
            if id(u) == id(active):
                bm_idx = i
                break
        if bm_idx is None:
            return result

        with torch.no_grad():
            h, units, round_onehot = model.trunk(state_vec.unsqueeze(0))
            h = h.squeeze(0)
            units = units.squeeze(0)  # (20, 200)

            # BM's unit features and post_move_rel
            bm_features = model._extract_unit_features(units, bm_idx)
            # BM doesn't move — use current position for post_move_rel
            from ml_integration_tactical import (
                _get_model_space_positions, _flip_x, _flip_y,
            )
            positions = _get_model_space_positions(friendly_units, player)
            enemy_positions = _get_model_space_positions(enemy_units, player)
            bm_cx, bm_cy = positions[bm_idx]
            post_move_rel = compute_post_move_rel(bm_cx, bm_cy, enemy_positions)

            move_onehot = F.one_hot(torch.tensor(MOVE_MOVE), NUM_MOVE_TYPES).float()

            # Apply range mask (BM has 30" range)
            max_wr = max((w.range_inches for w in active.unit.weapons if not w.melee), default=0)
            range_mask = compute_in_range_mask(post_move_rel, float(max_wr), enemy_alive_mask)

            # Get BM's shoot logits (pointer head)
            bm_logits = model.compute_shoot_logits(
                h, units, bm_idx, post_move_rel, enemy_alive_mask,
                shoot_range_mask=range_mask,
            )
            bm_probs = torch.softmax(bm_logits, dim=-1).tolist()
            bm_chosen = assessment.get('shoot_target_idx', -1)

            # Now run every other alive friendly unit's features through
            # the same shoot head (same h, same post_move_rel, same masks).
            # For the pointer head this means swapping the unit slot into
            # `units` so the candidate features use that unit's matchup row
            # and stats.
            other_probs = []
            other_names = []
            other_tids = []
            for i in range(len(friendly_units)):
                if i == bm_idx:
                    continue
                if friendly_units[i].models_alive <= 0:
                    continue
                units_swapped = units.clone()
                units_swapped[bm_idx] = units[i]
                logits = model.compute_shoot_logits(
                    h, units_swapped, bm_idx, post_move_rel, enemy_alive_mask,
                    shoot_range_mask=range_mask,
                )
                probs = torch.softmax(logits, dim=-1).tolist()
                other_probs.append(probs)
                other_names.append(friendly_units[i].unit.name)
                other_tids.append(friendly_units[i].unit.template_id)

        # Build enemy info
        enemy_names = []
        enemy_models = []
        enemy_tough = []
        enemy_alive = []
        for i in range(MAX_UNITS_PER_SIDE):
            if i < len(enemy_units) and enemy_units[i].models_alive > 0:
                enemy_names.append(enemy_units[i].unit.name)
                enemy_models.append(enemy_units[i].models_alive)
                enemy_tough.append(enemy_units[i].unit.tough)
                enemy_alive.append(True)
            else:
                enemy_names.append("")
                enemy_models.append(0)
                enemy_tough.append(0)
                enemy_alive.append(False)

        _results.append(SwapResult(
            n_in_range=int(range_mask.sum().item()),
            burst_mortar_probs=bm_probs,
            burst_mortar_chosen=bm_chosen,
            other_unit_probs=other_probs,
            other_unit_names=other_names,
            other_unit_template_ids=other_tids,
            enemy_names=enemy_names,
            enemy_models=enemy_models,
            enemy_tough=enemy_tough,
            enemy_alive=enemy_alive,
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
    return any(_is_burst_mortar(s) for s in states)


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

def _kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(P || Q) over valid (non-zero) entries."""
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-10 and qi > 1e-10:
            kl += pi * math.log(pi / qi)
    return kl


def _top_choice_match(p: list[float], q: list[float]) -> bool:
    """Do P and Q agree on the argmax?"""
    return p.index(max(p)) == q.index(max(q))


def analyse(results: list[SwapResult]):
    print("\n" + "=" * 70)
    print("BURST MORTAR — BOARD STATE vs UNIT FEATURES")
    print("=" * 70)
    print(f"\nTotal activations analysed: {len(results)}")

    # Filter to activations with 2+ targets (where there's a choice)
    multi = [r for r in results if r.n_in_range >= 2]
    print(f"Activations with 2+ targets: {len(multi)}")

    if not multi:
        print("Not enough data.")
        return

    # For each activation, compute how often other units agree on the top choice
    agree_counts = []
    kl_divs = []
    for r in multi:
        n_agree = 0
        n_other = len(r.other_unit_probs)
        for other_p in r.other_unit_probs:
            if _top_choice_match(r.burst_mortar_probs, other_p):
                n_agree += 1
            kl_divs.append(_kl_divergence(r.burst_mortar_probs, other_p))
        if n_other > 0:
            agree_counts.append(n_agree / n_other)

    avg_agree = sum(agree_counts) / len(agree_counts)
    avg_kl = sum(kl_divs) / len(kl_divs) if kl_divs else 0

    print(f"\n--- AGREEMENT ON TOP CHOICE ---")
    print(f"  When other units' features are swapped in (same h, same position):")
    print(f"  Average % of other units that agree on BM's top target: {avg_agree:.1%}")
    print(f"  Average KL divergence (BM || other): {avg_kl:.4f}")
    print(f"  (KL ≈ 0 means identical distributions; KL > 1 means very different)")

    # Breakdown: agreement rate by unit type
    agree_by_type: dict[str, list[bool]] = {}
    kl_by_type: dict[str, list[float]] = {}
    for r in multi:
        for i, (other_p, name, tid) in enumerate(
            zip(r.other_unit_probs, r.other_unit_names, r.other_unit_template_ids)
        ):
            agrees = _top_choice_match(r.burst_mortar_probs, other_p)
            kl = _kl_divergence(r.burst_mortar_probs, other_p)
            agree_by_type.setdefault(name, []).append(agrees)
            kl_by_type.setdefault(name, []).append(kl)

    print(f"\n--- AGREEMENT BY SWAPPED UNIT TYPE ---")
    print(f"  {'Unit swapped in':<50s} {'Agree%':>6s} {'AvgKL':>7s} {'N':>5s}")
    for name in sorted(agree_by_type, key=lambda n: -len(agree_by_type[n])):
        agrees = agree_by_type[name]
        kls = kl_by_type[name]
        pct = sum(agrees) / len(agrees) * 100
        avg_k = sum(kls) / len(kls)
        print(f"  {name:<50s} {pct:>5.1f}% {avg_k:>7.4f} {len(agrees):>5d}")

    # Example: show a few activations where there IS disagreement
    disagree = []
    for r in multi:
        for other_p, name in zip(r.other_unit_probs, r.other_unit_names):
            if not _top_choice_match(r.burst_mortar_probs, other_p):
                disagree.append((r, other_p, name))

    print(f"\n--- EXAMPLES WHERE SWAPPED UNIT DISAGREES ---")
    print(f"  ({len(disagree)} total disagreements)")
    for r, other_p, swap_name in disagree[:8]:
        bm_top = r.burst_mortar_probs.index(max(r.burst_mortar_probs))
        other_top = other_p.index(max(other_p))
        bm_target = r.enemy_names[bm_top]
        other_target = r.enemy_names[other_top]
        bm_mods = r.enemy_models[bm_top]
        other_mods = r.enemy_models[other_top]
        bm_t = r.enemy_tough[bm_top]
        other_t = r.enemy_tough[other_top]
        print(f"\n    BM picks: {bm_target} ({bm_mods}mod, T{bm_t}) @ {max(r.burst_mortar_probs):.1%}")
        print(f"    {swap_name} picks: {other_target} ({other_mods}mod, T{other_t}) @ {max(other_p):.1%}")

        # Show full distribution comparison for this case
        in_range = [(i, r.enemy_names[i], r.enemy_models[i], r.enemy_tough[i],
                      r.burst_mortar_probs[i], other_p[i])
                     for i in range(10) if r.enemy_alive[i] and r.burst_mortar_probs[i] > 0.001 or other_p[i] > 0.001]
        for idx, ename, emods, etough, bp, op in sorted(in_range, key=lambda x: -x[4]):
            print(f"      {ename:<35s} {emods}mod T{etough}  BM={bp:5.1%}  {swap_name}={op:5.1%}")


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
            if _is_burst_mortar(s):
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
                  f"swaps: {len(_results)}  (attempted: {games_attempted})")

    elapsed = time.time() - t0
    print(f"\nCompleted {NUM_GAMES} games in {elapsed:.1f}s "
          f"({games_attempted} attempted, {games_attempted - NUM_GAMES} skipped)")
    print(f"Results: A={wins['A']}  B={wins['B']}  Draw={wins['draw']}")
    print(f"Recorded {len(_results)} swap analyses")

    analyse(_results)
