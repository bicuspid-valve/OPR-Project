"""Test: ML agent (hall_of_fame_ml.json) vs Heuristic agent (hall_of_fame.json) — 50 games.

The final game is shown in the viewer.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from models import ArmyList
from evolution import make_entry, resolve_army, _make_unit_states
from game import simulate_game, simulate_game_recorded
from viewer import show_game


def _load_hof(filename: str) -> list[ArmyList]:
    """Load army lists from a hall-of-fame JSON file."""
    path = Path(__file__).resolve().parent / "results" / filename
    with open(path) as f:
        hof_data = json.load(f)

    armies = []
    for entry_data in hof_data:
        army = ArmyList()
        for e in entry_data["entries"]:
            entry = make_entry(
                e["template_id"],
                upgrades=e.get("upgrades", {}),
                ai_role=e.get("ai_role", "killer"),
            )
            entry.combat_preference = e.get("combat_preference", "ranged")
            army.entries.append(entry)
        army.fitness = entry_data.get("fitness", 0.0)
        armies.append(army)
    return armies


def _load_ml_model():
    """Load the trained ML model (tactical or strategic)."""
    from ml_training import load_model_state_dict
    model_path = Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"ML model not found: {model_path}")

    from ml_model_tactical import TacticalModel
    from ml_model import StrategicModel

    state_dict = load_model_state_dict(model_path)
    try:
        model = TacticalModel()
        model.load_state_dict(state_dict)
        model_type = "tactical"
    except RuntimeError:
        model = StrategicModel()
        model.load_state_dict(state_dict, strict=False)
        model_type = "strategic"

    model.eval()
    return model, model_type


def run_test(num_games: int = 50):
    print("=" * 70)
    print("ML (hall_of_fame_ml) vs Heuristic (hall_of_fame) — "
          f"{num_games} games")
    print("=" * 70)

    # Load armies
    ml_armies = _load_hof("hall_of_fame_ml.json")
    heuristic_armies = _load_hof("hall_of_fame.json")
    print(f"Loaded {len(ml_armies)} ML armies, {len(heuristic_armies)} heuristic armies")

    # Load ML model
    ml_model, model_type = _load_ml_model()
    print(f"ML model type: {model_type}")
    print("-" * 70)

    ml_wins = 0
    heuristic_wins = 0
    draws = 0
    start = time.time()

    for i in range(num_games):
        ml_army = random.choice(ml_armies)
        h_army = random.choice(heuristic_armies)

        ml_resolved = resolve_army(ml_army)
        h_resolved = resolve_army(h_army)

        sa = _make_unit_states(ml_army, ml_resolved, "A")
        sb = _make_unit_states(h_army, h_resolved, "B")

        is_last = (i == num_games - 1)

        if is_last:
            # Record the final game for viewer
            result, frames, labels, owners, unit_points, unit_info = \
                simulate_game_recorded(
                    ml_resolved, h_resolved, mode="objectives",
                    states_a=sa, states_b=sb,
                    ml_model_a=ml_model)
        else:
            result = simulate_game(
                ml_resolved, h_resolved, mode="objectives",
                states_a=sa, states_b=sb,
                ml_model_a=ml_model)

        if result == "A":
            ml_wins += 1
            tag = "ML wins"
        elif result == "B":
            heuristic_wins += 1
            tag = "Heuristic wins"
        else:
            draws += 1
            tag = "Draw"

        elapsed = time.time() - start
        print(f"  Game {i+1:3d}/{num_games} | {tag:16s} | "
              f"ML {ml_wins}-{heuristic_wins}-{draws} Heuristic | "
              f"{elapsed:.1f}s")

    total = time.time() - start
    print("=" * 70)
    print(f"RESULTS ({num_games} games, {total:.1f}s)")
    print(f"  ML wins:        {ml_wins:3d}  ({100*ml_wins/num_games:.1f}%)")
    print(f"  Heuristic wins: {heuristic_wins:3d}  ({100*heuristic_wins/num_games:.1f}%)")
    print(f"  Draws:          {draws:3d}  ({100*draws/num_games:.1f}%)")
    ml_wr = (ml_wins + 0.5 * draws) / num_games
    print(f"  ML win rate:    {ml_wr:.3f}")
    print("=" * 70)

    # Show the final game in the viewer
    print(f"\nLaunching viewer for game {num_games} ({len(frames)} frames)...")
    show_game(frames, labels, owners, mode="objectives",
              unit_points=unit_points, unit_info=unit_info)


if __name__ == "__main__":
    run_test(50)
