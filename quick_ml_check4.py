"""Quick script: ML tactical with planning (A) vs heuristic (B).

ML agent uses a random army from hall_of_fame_ml.json with Monte Carlo planning (K=4, C=10, M=64).
Heuristic agent uses a random army from hall_of_fame.json with default heuristic AI.
"""

import json
import random
from pathlib import Path
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList
from game import simulate_game, simulate_game_recorded
from viewer import show_game

_DIR = Path(__file__).resolve().parent


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


if __name__ == '__main__':
    # Load hall of fame armies
    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof_ml_data = json.load(f)
    with open(_DIR / "results" / "hall_of_fame.json") as f:
        hof_data = json.load(f)

    # Load ML model
    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model_label = "tactical"
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded {model_label} model from {checkpoint_path}")

    PLANNING_PARAMS = {
        "K_UNITS": 4,
        "C_SAMPLES_PER_UNIT": 4,
        "M_ROLLOUTS": 8,
    }

    NUM_GAMES = 50
    wins = {"A": 0, "B": 0, "draw": 0}
    winner_labels = {"A": "ML+Planning", "B": "Heuristic", "draw": "Draw"}

    print(f"Playing {NUM_GAMES} games: ML+Planning (K=4,C=10,M=64) vs Heuristic...")
    print(f"  ML uses random ML HoF armies, Heuristic uses random HoF armies.")
    for i in range(NUM_GAMES):
        army_a = load_army_from_hof(random.choice(hof_ml_data))
        army_b = load_army_from_hof(random.choice(hof_data))
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")
        result = simulate_game(
            res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
            ml_model_a=model,
            ml_planning="A", planning_params=PLANNING_PARAMS,
        )
        wins[result] += 1
        print(f"  Game {i+1}/{NUM_GAMES}: {winner_labels.get(result, result)}", end="\r")

    print()
    print(f"Results over {NUM_GAMES} games:")
    if NUM_GAMES > 0:
        print(f"  ML+Planning: {wins['A']:>3} wins ({wins['A']/NUM_GAMES*100:.1f}%)")
        print(f"  Heuristic:   {wins['B']:>3} wins ({wins['B']/NUM_GAMES*100:.1f}%)")
        print(f"  Draws:       {wins['draw']:>3}      ({wins['draw']/NUM_GAMES*100:.1f}%)")
    else:
        print("  (no games played)")

    # Play one final game with recording for the viewer — random matchup
    idx_a = random.randrange(len(hof_ml_data))
    idx_b = random.randrange(len(hof_data))
    viewer_army_a = load_army_from_hof(hof_ml_data[idx_a])
    viewer_army_b = load_army_from_hof(hof_data[idx_b])
    print(f"\nPlaying final game for viewer: ML HoF #{idx_a+1} (ML+Planning) vs HoF #{idx_b+1} (Heuristic)...")
    res_a = resolve_army(viewer_army_a)
    res_b = resolve_army(viewer_army_b)
    sa = _make_unit_states(viewer_army_a, res_a, "A")
    sb = _make_unit_states(viewer_army_b, res_b, "B")
    result, frames, labels, owners, unit_points, unit_info = simulate_game_recorded(
        res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
        ml_model_a=model,
        ml_planning="A", planning_params=PLANNING_PARAMS,
    )
    viewer_labels = {"A": f"ML+Planning (ML HoF #{idx_a+1})", "B": f"Heuristic (HoF #{idx_b+1})", "draw": "Draw"}
    print(f"Final game: {viewer_labels.get(result, result)} — {len(frames)} frames")
    show_game(frames, labels, owners, mode="objectives", unit_points=unit_points, unit_info=unit_info)
