"""Quick script: heuristic AI vs heuristic AI on map2.

Plays a batch of headless games for win-rate stats, then plays one
additional game with frame recording and pops the viewer. Both sides
pull random armies from ``results/hall_of_fame.json`` so the matchup
reflects the same army distribution training sees on the heuristic
opponent path.

Run:  python quick_heuristic_game.py
"""

import json
import random
from pathlib import Path

from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList
from game import simulate_game, simulate_game_recorded
from map_loader import load_map
from viewer import show_game

_DIR = Path(__file__).resolve().parent
MAP_PATH = "maps/map2.json"
NUM_GAMES = 5


def load_army_from_hof(hof_entry: dict) -> ArmyList:
    army = ArmyList()
    for e in hof_entry["entries"]:
        entry = make_entry(
            e["template_id"],
            upgrades=e.get("upgrades", {}),
            ai_role=e.get("ai_role", "killer"),
        )
        entry.combat_preference = e.get("combat_preference", "ranged")
        entry.attached_to = e.get("attached_to", -1)
        army.entries.append(entry)
    return army


def _play_one(map_data, hof_data, record: bool = False):
    army_a = load_army_from_hof(random.choice(hof_data))
    army_b = load_army_from_hof(random.choice(hof_data))
    res_a = resolve_army(army_a)
    res_b = resolve_army(army_b)
    sa = _make_unit_states(army_a, res_a, "A")
    sb = _make_unit_states(army_b, res_b, "B")
    if record:
        return simulate_game_recorded(
            res_a, res_b, mode="objectives",
            states_a=sa, states_b=sb,
            map_data=map_data,
        )
    return simulate_game(
        res_a, res_b, mode="objectives",
        states_a=sa, states_b=sb,
        map_data=map_data,
    )


if __name__ == "__main__":
    import fast_core
    fast_core.USE_C_EXT = fast_core.is_available()

    hof_path = _DIR / "results" / "hall_of_fame.json"
    with open(hof_path) as f:
        hof_data = json.load(f)
    print(f"Loaded {len(hof_data)} HoF armies from {hof_path.name}")

    map_data = load_map(_DIR / MAP_PATH)
    print(f"Map: {MAP_PATH} "
          f"({len(map_data.terrain)} terrain pieces, "
          f"{len(map_data.objectives)} objectives)")

    wins = {"A": 0, "B": 0, "draw": 0}
    print(f"\nPlaying {NUM_GAMES} heuristic-vs-heuristic games...")
    for i in range(NUM_GAMES):
        result = _play_one(map_data, hof_data, record=False)
        wins[result] += 1
        print(f"  Game {i + 1}/{NUM_GAMES}: {result}", end="\r")
    print()
    total = sum(wins.values())
    print(f"\nResults over {total} games:")
    print(f"  A wins: {wins['A']:>3} ({wins['A'] / total * 100:.1f}%)")
    print(f"  B wins: {wins['B']:>3} ({wins['B'] / total * 100:.1f}%)")
    print(f"  Draws:  {wins['draw']:>3} ({wins['draw'] / total * 100:.1f}%)")

    print("\nPlaying one final game with recording for the viewer...")
    result, frames, labels, owners, unit_points, unit_info = _play_one(
        map_data, hof_data, record=True,
    )
    winner = {"A": "Player A", "B": "Player B", "draw": "Draw"}[result]
    print(f"Final game: {winner} — {len(frames)} frames")
    show_game(frames, labels, owners, mode="objectives",
              unit_points=unit_points, unit_info=unit_info,
              map_data=map_data)
