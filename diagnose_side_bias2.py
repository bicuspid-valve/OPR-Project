"""Quick test: run the exact same baseline as quick_ml_check5.py
multiple times (50 games each) to see if A-bias is reproducible."""

import json
import random
import time
from pathlib import Path

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList
from game import simulate_game

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


if __name__ == "__main__":
    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof_ml_data = json.load(f)

    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    NUM_BATCHES = 6
    GAMES_PER_BATCH = 50

    print(f"Running {NUM_BATCHES} batches of {GAMES_PER_BATCH} games each")
    print(f"(ML vs ML, no planning, hall_of_fame_ml.json for both sides)\n")

    all_a = 0
    all_b = 0
    all_d = 0

    for batch in range(NUM_BATCHES):
        wins = {"A": 0, "B": 0, "draw": 0}
        for i in range(GAMES_PER_BATCH):
            army_a = load_army_from_hof(random.choice(hof_ml_data))
            army_b = load_army_from_hof(random.choice(hof_ml_data))
            res_a = resolve_army(army_a)
            res_b = resolve_army(army_b)
            sa = _make_unit_states(army_a, res_a, "A")
            sb = _make_unit_states(army_b, res_b, "B")
            result = simulate_game(
                res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
                ml_model_a=model, ml_model_b=model,
            )
            wins[result] += 1

        all_a += wins["A"]
        all_b += wins["B"]
        all_d += wins["draw"]
        print(f"  Batch {batch+1}: A={wins['A']:>2}  B={wins['B']:>2}  "
              f"D={wins['draw']:>2}  (A ratio: {wins['A']/(wins['A']+wins['B']) if wins['A']+wins['B'] else 0:.2f})")

    total = all_a + all_b + all_d
    print(f"\n  TOTAL ({total} games): A={all_a}  B={all_b}  D={all_d}")
    print(f"  A win%: {all_a/total*100:.1f}%  B win%: {all_b/total*100:.1f}%  "
          f"Draw%: {all_d/total*100:.1f}%")
    print(f"  A/(A+B) ratio: {all_a/(all_a+all_b):.3f}")