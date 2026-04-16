"""Measure ML model targeting biases: blast → multi-model, deadly → tough(3+).

Runs 100 ML-vs-ML games using hall_of_fame_ml.json armies and logs every
ranged shooting event.  For each event we record the attacker's weapon
properties and the target's unit properties, then report:

  * % of shots from blast-carrying units aimed at multi-model targets
  * % of shots from non-blast units aimed at multi-model targets
  * % of shots from deadly-carrying units aimed at tough(3+) targets
  * % of shots from non-deadly units aimed at tough(3+) targets

A well-trained model should show a clear preference for blast → hordes
and deadly → tough targets compared to the baseline (non-blast / non-deadly).
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
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
class ShootingEvent:
    # Attacker
    attacker_name: str
    attacker_has_blast: bool
    attacker_max_blast: int
    attacker_has_deadly: bool
    attacker_max_deadly: int
    # Target (at time of targeting, before wounds applied)
    target_name: str
    target_original_size: int      # unit template size
    target_models_alive: int       # models alive when targeted
    target_tough: int              # tough value (0 = no tough)


_shooting_log: list[ShootingEvent] = []


# ------------------------------------------------------------------
# Monkey-patch resolve_shooting to log events
# ------------------------------------------------------------------

def _install_logging_hook():
    """Patch game.resolve_shooting to record targeting decisions."""
    import game as game_module
    _original = game_module.resolve_shooting

    def _logging_resolve_shooting(attacker, defender, recorded=False):
        # Classify attacker weapons (ranged only)
        ranged_weapons = [w for w in attacker.unit.weapons if not w.melee]
        has_blast = any(w.blast > 0 for w in ranged_weapons)
        max_blast = max((w.blast for w in ranged_weapons), default=0)
        has_deadly = any(w.deadly > 0 for w in ranged_weapons)
        max_deadly = max((w.deadly for w in ranged_weapons), default=0)

        if ranged_weapons:
            _shooting_log.append(ShootingEvent(
                attacker_name=attacker.unit.name,
                attacker_has_blast=has_blast,
                attacker_max_blast=max_blast,
                attacker_has_deadly=has_deadly,
                attacker_max_deadly=max_deadly,
                target_name=defender.unit.name,
                target_original_size=defender.unit.models,
                target_models_alive=defender.models_alive,
                target_tough=defender.unit.tough,
            ))

        return _original(attacker, defender, recorded=recorded)

    game_module.resolve_shooting = _logging_resolve_shooting


# ------------------------------------------------------------------
# Army loading (same pattern as other diagnostic scripts)
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


# ------------------------------------------------------------------
# Analysis helpers
# ------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  N/A"
    return f"{n / total * 100:5.1f}%"


def analyse(events: list[ShootingEvent]):
    # ------ Blast analysis ------
    blast_events = [e for e in events if e.attacker_has_blast]
    non_blast_events = [e for e in events if not e.attacker_has_blast]

    blast_vs_multi = sum(1 for e in blast_events if e.target_models_alive > 1)
    non_blast_vs_multi = sum(1 for e in non_blast_events if e.target_models_alive > 1)

    # Also check against original unit size (structural multi-model)
    blast_vs_multi_orig = sum(1 for e in blast_events if e.target_original_size > 1)
    non_blast_vs_multi_orig = sum(1 for e in non_blast_events if e.target_original_size > 1)

    # ------ Deadly analysis ------
    deadly_events = [e for e in events if e.attacker_has_deadly]
    non_deadly_events = [e for e in events if not e.attacker_has_deadly]

    deadly_vs_tough3 = sum(1 for e in deadly_events if e.target_tough >= 3)
    non_deadly_vs_tough3 = sum(1 for e in non_deadly_events if e.target_tough >= 3)

    # Also check against any tough (>= 1)
    deadly_vs_any_tough = sum(1 for e in deadly_events if e.target_tough >= 1)
    non_deadly_vs_any_tough = sum(1 for e in non_deadly_events if e.target_tough >= 1)

    print("\n" + "=" * 64)
    print("TARGETING BIAS ANALYSIS")
    print("=" * 64)

    print(f"\nTotal shooting events: {len(events)}")
    print(f"  Blast-carrying attackers:     {len(blast_events)}")
    print(f"  Non-blast attackers:          {len(non_blast_events)}")
    print(f"  Deadly-carrying attackers:    {len(deadly_events)}")
    print(f"  Non-deadly attackers:         {len(non_deadly_events)}")

    print(f"\n--- BLAST vs MULTI-MODEL TARGETS ---")
    print(f"  (multi-model = target has >1 model alive at time of shooting)")
    print(f"  Blast  → multi-model target: {_pct(blast_vs_multi, len(blast_events)):>6}"
          f"  ({blast_vs_multi}/{len(blast_events)})")
    print(f"  Other  → multi-model target: {_pct(non_blast_vs_multi, len(non_blast_events)):>6}"
          f"  ({non_blast_vs_multi}/{len(non_blast_events)})")
    if len(blast_events) and len(non_blast_events):
        blast_rate = blast_vs_multi / len(blast_events)
        other_rate = non_blast_vs_multi / len(non_blast_events)
        diff = blast_rate - other_rate
        print(f"  Δ (blast − other):           {diff:+.1%}")

    print(f"\n  (multi-model = target template has size > 1)")
    print(f"  Blast  → multi-model (orig): {_pct(blast_vs_multi_orig, len(blast_events)):>6}"
          f"  ({blast_vs_multi_orig}/{len(blast_events)})")
    print(f"  Other  → multi-model (orig): {_pct(non_blast_vs_multi_orig, len(non_blast_events)):>6}"
          f"  ({non_blast_vs_multi_orig}/{len(non_blast_events)})")

    print(f"\n--- DEADLY vs TOUGH(3+) TARGETS ---")
    print(f"  Deadly → tough(3+) target:   {_pct(deadly_vs_tough3, len(deadly_events)):>6}"
          f"  ({deadly_vs_tough3}/{len(deadly_events)})")
    print(f"  Other  → tough(3+) target:   {_pct(non_deadly_vs_tough3, len(non_deadly_events)):>6}"
          f"  ({non_deadly_vs_tough3}/{len(non_deadly_events)})")
    if len(deadly_events) and len(non_deadly_events):
        deadly_rate = deadly_vs_tough3 / len(deadly_events)
        other_rate = non_deadly_vs_tough3 / len(non_deadly_events)
        diff = deadly_rate - other_rate
        print(f"  Δ (deadly − other):          {diff:+.1%}")

    print(f"\n--- DEADLY vs ANY TOUGH TARGETS ---")
    print(f"  Deadly → tough(1+) target:   {_pct(deadly_vs_any_tough, len(deadly_events)):>6}"
          f"  ({deadly_vs_any_tough}/{len(deadly_events)})")
    print(f"  Other  → tough(1+) target:   {_pct(non_deadly_vs_any_tough, len(non_deadly_events)):>6}"
          f"  ({non_deadly_vs_any_tough}/{len(non_deadly_events)})")

    # ------ Detailed weapon breakdown ------
    print(f"\n--- TOP ATTACKER → TARGET PAIRS (blast weapons) ---")
    blast_pairs: dict[tuple[str, str], int] = {}
    for e in blast_events:
        key = (e.attacker_name, e.target_name)
        blast_pairs[key] = blast_pairs.get(key, 0) + 1
    for (atk, tgt), count in sorted(blast_pairs.items(), key=lambda x: -x[1])[:10]:
        print(f"  {atk:>30s} → {tgt:<25s} x{count}")

    print(f"\n--- TOP ATTACKER → TARGET PAIRS (deadly weapons) ---")
    deadly_pairs: dict[tuple[str, str], int] = {}
    for e in deadly_events:
        key = (e.attacker_name, e.target_name)
        deadly_pairs[key] = deadly_pairs.get(key, 0) + 1
    for (atk, tgt), count in sorted(deadly_pairs.items(), key=lambda x: -x[1])[:10]:
        print(f"  {atk:>30s} → {tgt:<25s} x{count}")

    # ------ Target model count distribution ------
    print(f"\n--- TARGET MODELS-ALIVE DISTRIBUTION ---")
    for label, evts in [("Blast attackers", blast_events),
                         ("Non-blast attackers", non_blast_events)]:
        if not evts:
            continue
        counts = {}
        for e in evts:
            counts[e.target_models_alive] = counts.get(e.target_models_alive, 0) + 1
        dist = sorted(counts.items())
        dist_str = ", ".join(f"{k}mod={v}" for k, v in dist)
        print(f"  {label}: {dist_str}")

    print(f"\n--- TARGET TOUGH DISTRIBUTION ---")
    for label, evts in [("Deadly attackers", deadly_events),
                         ("Non-deadly attackers", non_deadly_events)]:
        if not evts:
            continue
        counts = {}
        for e in evts:
            counts[e.target_tough] = counts.get(e.target_tough, 0) + 1
        dist = sorted(counts.items())
        dist_str = ", ".join(f"T{k}={v}" for k, v in dist)
        print(f"  {label}: {dist_str}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    from game import simulate_game

    # Install the logging hook before running games
    _install_logging_hook()

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

    # Run games
    print(f"\nRunning {NUM_GAMES} ML-vs-ML games...\n")
    wins = {"A": 0, "B": 0, "draw": 0}
    t0 = time.time()

    for i in range(NUM_GAMES):
        army_a = load_army_from_hof(random.choice(hof_ml_data))
        army_b = load_army_from_hof(random.choice(hof_ml_data))
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
            per_game = elapsed / (i + 1)
            eta = per_game * (NUM_GAMES - i - 1)
            print(f"  Game {i+1:3d}/{NUM_GAMES}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)  "
                  f"events so far: {len(_shooting_log)}")

    elapsed = time.time() - t0
    print(f"\nCompleted {NUM_GAMES} games in {elapsed:.1f}s")
    print(f"Results: A={wins['A']}  B={wins['B']}  Draw={wins['draw']}")
    print(f"Recorded {len(_shooting_log)} shooting events")

    # Analyse
    analyse(_shooting_log)
