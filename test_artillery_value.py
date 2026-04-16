"""Measure the value head's response to Burst Mortar targeting decisions.

For each Burst Mortar shooting event, encodes the game state before and
after the shot, runs the value head, and records the delta. Compares
average value delta for shots at single-model vs multi-model targets.

Runs 100 ML-vs-ML games with at least 1 Burst Mortar per game.
"""
from __future__ import annotations

import json
import random
import sys
import time
import torch
from dataclasses import dataclass
from pathlib import Path

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from ml_features import encode_state_tactical, MAX_UNITS_PER_SIDE
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList

_DIR = Path(__file__).resolve().parent

NUM_GAMES = 100


# ------------------------------------------------------------------
# Event log
# ------------------------------------------------------------------

@dataclass
class ValueEvent:
    target_name: str
    target_models_alive: int
    target_original_size: int
    target_tough: int
    value_before: float
    value_after: float
    delta: float
    owner: str           # "A" or "B"
    round_num: int


_events: list[ValueEvent] = []


# ------------------------------------------------------------------
# Activation context — set by apply_tactical_model patch
# ------------------------------------------------------------------

_ctx: dict = {}


def _encode_and_value(model, friendly_units, enemy_units, round_num, board,
                      player, fr, fm, er, em, fpts, epts) -> float:
    """Encode current state and return value head scalar."""
    state_vec = encode_state_tactical(
        friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=fr, friendly_melee_matchups=fm,
        enemy_ranged_matchups=er, enemy_melee_matchups=em,
        total_friendly_points=fpts, total_enemy_points=epts,
    )
    alive_mask = torch.tensor(
        [i < len(friendly_units) and friendly_units[i].models_alive > 0
         and not friendly_units[i].activated
         for i in range(MAX_UNITS_PER_SIDE)], dtype=torch.bool)
    enemy_alive_mask = torch.tensor(
        [i < len(enemy_units) and enemy_units[i].models_alive > 0
         for i in range(MAX_UNITS_PER_SIDE)], dtype=torch.bool)
    with torch.no_grad():
        out = model(state_vec, alive_mask, enemy_alive_mask)
    return out.value.item()


# ------------------------------------------------------------------
# Monkey-patches
# ------------------------------------------------------------------

def _classify_burst_mortar(unit_state) -> bool:
    """True if unit is a Support Artillery with Burst Mortar."""
    ru = unit_state.unit
    if ru.template_id != "support_artillery":
        return False
    return any(w.name == "Burst Mortar" for w in ru.weapons)


# Registry of burst mortar UnitState IDs
_burst_mortar_ids: set[int] = set()


def _install_hooks():
    """Patch apply_tactical_model and resolve_shooting."""

    # --- Patch apply_tactical_model to capture context ---
    import ml_integration_tactical as ml_mod
    _original_apply = ml_mod.apply_tactical_model

    def _patched_apply(model, friendly_units, enemy_units, round_num, board,
                       player, **kw):
        _ctx.update({
            'model': model,
            'friendly': friendly_units,
            'enemy': enemy_units,
            'round': round_num,
            'board': board,
            'player': player,
            'fr': kw.get('friendly_ranged_matchups'),
            'fm': kw.get('friendly_melee_matchups'),
            'er': kw.get('enemy_ranged_matchups'),
            'em': kw.get('enemy_melee_matchups'),
            'fpts': kw.get('total_friendly_points'),
            'epts': kw.get('total_enemy_points'),
        })
        return _original_apply(model, friendly_units, enemy_units, round_num,
                               board, player, **kw)

    ml_mod.apply_tactical_model = _patched_apply

    # Also patch the import inside game.py (it imports lazily)
    import game as game_mod
    _original_resolve = game_mod.resolve_shooting

    def _patched_resolve(attacker, defender, recorded=False):
        is_burst = id(attacker) in _burst_mortar_ids
        # Capture target state BEFORE the shot resolves
        pre_models_alive = defender.models_alive
        if is_burst and _ctx:
            value_before = _encode_and_value(
                _ctx['model'], _ctx['friendly'], _ctx['enemy'],
                _ctx['round'], _ctx['board'], _ctx['player'],
                _ctx['fr'], _ctx['fm'], _ctx['er'], _ctx['em'],
                _ctx['fpts'], _ctx['epts'])
        else:
            value_before = None

        result = _original_resolve(attacker, defender, recorded=recorded)

        if is_burst and _ctx and value_before is not None:
            value_after = _encode_and_value(
                _ctx['model'], _ctx['friendly'], _ctx['enemy'],
                _ctx['round'], _ctx['board'], _ctx['player'],
                _ctx['fr'], _ctx['fm'], _ctx['er'], _ctx['em'],
                _ctx['fpts'], _ctx['epts'])
            delta = value_after - value_before
            _events.append(ValueEvent(
                target_name=defender.unit.name,
                target_models_alive=pre_models_alive,
                target_original_size=defender.unit.models,
                target_tough=defender.unit.tough,
                value_before=value_before,
                value_after=value_after,
                delta=delta,
                owner=attacker.owner,
                round_num=_ctx['round'],
            ))

        return result

    game_mod.resolve_shooting = _patched_resolve


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

def _fmt(val: float) -> str:
    return f"{val:+.4f}"


def analyse(events: list[ValueEvent]):
    multi = [e for e in events if e.target_models_alive > 1]
    single = [e for e in events if e.target_models_alive == 1]

    multi_orig = [e for e in events if e.target_original_size > 1]
    single_orig = [e for e in events if e.target_original_size == 1]

    tough = [e for e in events if e.target_tough >= 3]
    non_tough = [e for e in events if e.target_tough == 0]

    print("\n" + "=" * 70)
    print("BURST MORTAR — VALUE HEAD RESPONSE TO TARGETING DECISIONS")
    print("=" * 70)
    print(f"\nTotal events: {len(events)}")
    print(f"  vs multi-model (>1 alive): {len(multi)}")
    print(f"  vs single-model (1 alive): {len(single)}")

    print(f"\n--- VALUE DELTA (after − before shot) ---")
    print(f"  (positive = model thinks position improved)")

    for label, evts in [
        ("All targets", events),
        ("vs multi-model (>1 alive)", multi),
        ("vs single-model (1 alive)", single),
        ("vs multi-model (template)", multi_orig),
        ("vs single-model (template)", single_orig),
        ("vs tough(3+)", tough),
        ("vs non-tough", non_tough),
    ]:
        if not evts:
            print(f"\n  {label}: no events")
            continue
        deltas = [e.delta for e in evts]
        avg_delta = sum(deltas) / len(deltas)
        avg_before = sum(e.value_before for e in evts) / len(evts)
        avg_after = sum(e.value_after for e in evts) / len(evts)
        print(f"\n  {label} ({len(evts)} events):")
        print(f"    avg value before: {avg_before:+.4f}")
        print(f"    avg value after:  {avg_after:+.4f}")
        print(f"    avg delta:        {avg_delta:+.4f}")

    if multi and single:
        avg_multi = sum(e.delta for e in multi) / len(multi)
        avg_single = sum(e.delta for e in single) / len(single)
        print(f"\n--- KEY COMPARISON ---")
        print(f"  avg delta vs multi-model:  {avg_multi:+.4f}")
        print(f"  avg delta vs single-model: {avg_single:+.4f}")
        print(f"  difference:                {avg_multi - avg_single:+.4f}")
        if avg_multi > avg_single:
            print(f"  → Value head sees MORE improvement from shooting multi-model targets")
        else:
            print(f"  → Value head sees MORE improvement from shooting single-model targets")

    # Breakdown by target models alive
    print(f"\n--- DELTA BY TARGET MODELS ALIVE ---")
    by_alive: dict[int, list[float]] = {}
    for e in events:
        by_alive.setdefault(e.target_models_alive, []).append(e.delta)
    for k in sorted(by_alive):
        vals = by_alive[k]
        avg = sum(vals) / len(vals)
        print(f"  {k:>2d} models alive: avg delta={avg:+.4f}  (n={len(vals)})")

    # Breakdown by target name
    print(f"\n--- DELTA BY TARGET UNIT ---")
    by_target: dict[str, list[float]] = {}
    for e in events:
        label = f"{e.target_name} [{e.target_original_size}mod, T{e.target_tough}]"
        by_target.setdefault(label, []).append(e.delta)
    for tgt, vals in sorted(by_target.items(), key=lambda x: -len(x[1])):
        avg = sum(vals) / len(vals)
        print(f"  {tgt:<55s} n={len(vals):>3d}  avg_delta={avg:+.4f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    from game import simulate_game

    _install_hooks()

    # Load HoF armies
    hof_path = _DIR / "results" / "hall_of_fame_ml.json"
    if not hof_path.exists():
        print(f"Error: {hof_path} not found")
        sys.exit(1)
    with open(hof_path) as f:
        hof_ml_data = json.load(f)
    print(f"Loaded {len(hof_ml_data)} armies from hall_of_fame_ml.json")

    # Load ML model
    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found")
        sys.exit(1)
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded model from {checkpoint_path.name}")

    # Run games (skip those without burst mortar)
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

        # Register burst mortar units
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
                  f"events: {len(_events)}  (attempted: {games_attempted})")

    elapsed = time.time() - t0
    print(f"\nCompleted {NUM_GAMES} games in {elapsed:.1f}s "
          f"({games_attempted} attempted, {games_attempted - NUM_GAMES} skipped)")
    print(f"Results: A={wins['A']}  B={wins['B']}  Draw={wins['draw']}")
    print(f"Recorded {len(_events)} burst mortar value events")

    analyse(_events)
