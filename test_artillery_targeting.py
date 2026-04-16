"""Measure how each Support Artillery variant targets single-model vs
multi-model and tough vs non-tough units.

Runs 100 ML-vs-ML games (skipping games without artillery, so exactly 100
games WITH artillery are played). Logs every shooting event from a Support
Artillery unit, split by weapon variant:

  - Burst Mortar  (default, Blast(3))
  - Heavy Mortar  (Deadly(6))
  - AA-Cannon     (6A, AP1, Unstoppable)
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList

_DIR = Path(__file__).resolve().parent

NUM_GAMES = 100


# ------------------------------------------------------------------
# Shooting event log
# ------------------------------------------------------------------

@dataclass
class ArtilleryEvent:
    variant: str                   # "burst_mortar", "heavy_mortar", "aa_cannon"
    owner: str                     # "A" or "B"
    target_name: str
    target_original_size: int      # unit template model count
    target_models_alive: int       # models alive when targeted
    target_tough: int              # tough value (0 = not tough)


_events: list[ArtilleryEvent] = []

# Track which UnitState IDs are artillery and their variant
_artillery_registry: dict[int, str] = {}


def _classify_artillery_variant(unit_state) -> str | None:
    """Determine which artillery variant a UnitState is, or None if not artillery."""
    ru = unit_state.unit
    if ru.template_id != "support_artillery":
        return None
    for w in ru.weapons:
        if w.name == "Burst Mortar":
            return "burst_mortar"
        if w.name == "Heavy Mortar":
            return "heavy_mortar"
        if w.name == "AA-Cannon":
            return "aa_cannon"
    return None


# ------------------------------------------------------------------
# Monkey-patch resolve_shooting to log artillery events
# ------------------------------------------------------------------

def _install_logging_hook():
    import game as game_module
    _original = game_module.resolve_shooting

    def _logging_resolve_shooting(attacker, defender, recorded=False):
        variant = _artillery_registry.get(id(attacker))
        if variant is not None:
            ranged_weapons = [w for w in attacker.unit.weapons if not w.melee]
            if ranged_weapons:
                _events.append(ArtilleryEvent(
                    variant=variant,
                    owner=attacker.owner,
                    target_name=defender.unit.name,
                    target_original_size=defender.unit.models,
                    target_models_alive=defender.models_alive,
                    target_tough=defender.unit.tough,
                ))
        return _original(attacker, defender, recorded=recorded)

    game_module.resolve_shooting = _logging_resolve_shooting


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


def army_has_artillery(states: list) -> bool:
    return any(_classify_artillery_variant(s) is not None for s in states)


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  N/A"
    return f"{n / total * 100:5.1f}%"


def analyse(events: list[ArtilleryEvent]):
    variants = ["burst_mortar", "heavy_mortar", "aa_cannon"]
    labels = {"burst_mortar": "Burst Mortar (Blast 3)",
              "heavy_mortar": "Heavy Mortar (Deadly 6)",
              "aa_cannon":    "AA-Cannon (6A, AP1, Unstoppable)"}

    print("\n" + "=" * 70)
    print("ARTILLERY TARGETING ANALYSIS")
    print("=" * 70)
    print(f"\nTotal artillery shooting events: {len(events)}")

    for v in variants:
        ve = [e for e in events if e.variant == v]
        print(f"  {labels[v]:>40s}: {len(ve):>4d} events")

    for v in variants:
        ve = [e for e in events if e.variant == v]
        if not ve:
            continue

        multi_alive = sum(1 for e in ve if e.target_models_alive > 1)
        multi_orig = sum(1 for e in ve if e.target_original_size > 1)
        tough3 = sum(1 for e in ve if e.target_tough >= 3)
        non_tough = sum(1 for e in ve if e.target_tough == 0)
        n = len(ve)

        print(f"\n--- {labels[v]} ({n} events) ---")
        print(f"  Target has >1 model alive:    {_pct(multi_alive, n):>6}  ({multi_alive}/{n})")
        print(f"  Target template is multi-model:{_pct(multi_orig, n):>6}  ({multi_orig}/{n})")
        print(f"  Target is tough(3+):          {_pct(tough3, n):>6}  ({tough3}/{n})")
        print(f"  Target is non-tough:          {_pct(non_tough, n):>6}  ({non_tough}/{n})")

        # Target breakdown
        target_counts: dict[str, int] = {}
        for e in ve:
            t_label = f"{e.target_name} [{e.target_original_size}mod, T{e.target_tough}]"
            target_counts[t_label] = target_counts.get(t_label, 0) + 1
        print(f"\n  Target breakdown:")
        for tgt, count in sorted(target_counts.items(), key=lambda x: -x[1]):
            pct = count / n * 100
            print(f"    {tgt:<50s} {count:>3d} ({pct:4.1f}%)")

        # Models-alive distribution
        alive_counts: dict[int, int] = {}
        for e in ve:
            alive_counts[e.target_models_alive] = alive_counts.get(e.target_models_alive, 0) + 1
        dist_str = ", ".join(f"{k}mod={v}" for k, v in sorted(alive_counts.items()))
        print(f"\n  Target models-alive distribution: {dist_str}")

    # Comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON: % of shots at multi-model targets (>1 alive)")
    print(f"{'=' * 70}")
    for v in variants:
        ve = [e for e in events if e.variant == v]
        if not ve:
            continue
        multi = sum(1 for e in ve if e.target_models_alive > 1)
        print(f"  {labels[v]:>40s}: {_pct(multi, len(ve)):>6}  ({multi}/{len(ve)})")

    non_arty_note = "(compare with non-artillery baseline from test_targeting_bias.py)"
    print(f"\n  {non_arty_note}")

    print(f"\n{'=' * 70}")
    print("COMPARISON: % of shots at tough(3+) targets")
    print(f"{'=' * 70}")
    for v in variants:
        ve = [e for e in events if e.variant == v]
        if not ve:
            continue
        tough = sum(1 for e in ve if e.target_tough >= 3)
        print(f"  {labels[v]:>40s}: {_pct(tough, len(ve)):>6}  ({tough}/{len(ve)})")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from game import simulate_game

    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic", action="store_true",
                        help="Run heuristic-vs-heuristic with hall_of_fame.json")
    args = parser.parse_args()

    _install_logging_hook()

    if args.heuristic:
        hof_path = _DIR / "results" / "hall_of_fame.json"
        hof_label = "hall_of_fame.json"
        mode_label = "Heuristic-vs-Heuristic"
    else:
        hof_path = _DIR / "results" / "hall_of_fame_ml.json"
        hof_label = "hall_of_fame_ml.json"
        mode_label = "ML-vs-ML"

    if not hof_path.exists():
        print(f"Error: {hof_path} not found")
        sys.exit(1)
    with open(hof_path) as f:
        hof_data = json.load(f)
    print(f"Loaded {len(hof_data)} armies from {hof_label}")

    model = None
    if not args.heuristic:
        checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
        if not checkpoint_path.exists():
            print(f"Error: {checkpoint_path} not found")
            sys.exit(1)
        state_dict = load_model_state_dict(checkpoint_path)
        model = TacticalModel()
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"Loaded model from {checkpoint_path.name}")

    # Run games (skip those without artillery)
    print(f"\nRunning {NUM_GAMES} {mode_label} games with artillery...\n")
    wins = {"A": 0, "B": 0, "draw": 0}
    games_played = 0
    games_attempted = 0
    t0 = time.time()

    while games_played < NUM_GAMES:
        army_a = load_army_from_hof(random.choice(hof_data))
        army_b = load_army_from_hof(random.choice(hof_data))
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")

        games_attempted += 1

        # Skip if neither side has artillery
        if not army_has_artillery(sa) and not army_has_artillery(sb):
            continue

        # Register artillery units
        for s in sa + sb:
            v = _classify_artillery_variant(s)
            if v is not None:
                _artillery_registry[id(s)] = v

        result = simulate_game(
            res_a, res_b, mode="objectives",
            states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
        )
        wins[result] += 1
        games_played += 1

        # Clean up registry (UnitState objects may be reused)
        for s in sa + sb:
            _artillery_registry.pop(id(s), None)

        if games_played % 10 == 0:
            elapsed = time.time() - t0
            per_game = elapsed / games_played
            eta = per_game * (NUM_GAMES - games_played)
            print(f"  Game {games_played:3d}/{NUM_GAMES}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)  "
                  f"events so far: {len(_events)}  "
                  f"(attempted: {games_attempted})")

    elapsed = time.time() - t0
    print(f"\nCompleted {NUM_GAMES} {mode_label} games in {elapsed:.1f}s "
          f"({games_attempted} attempted, {games_attempted - NUM_GAMES} skipped)")
    print(f"Results: A={wins['A']}  B={wins['B']}  Draw={wins['draw']}")
    print(f"Recorded {len(_events)} artillery shooting events")

    analyse(_events)
