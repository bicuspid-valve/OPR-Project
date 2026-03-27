"""Quick retest: 200 games of K4 C4 M64 N4 (planning) vs ML (no planning)."""

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
NUM_GAMES = 200
PARAMS = {"K_UNITS": 4, "C_SAMPLES_PER_UNIT": 4, "M_ROLLOUTS": 64, "N_LOOKAHEAD": 4}


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

    print(f"K4 C4 M64 N4 vs ML — {NUM_GAMES} games")
    wins = {"A": 0, "B": 0, "draw": 0}
    game_times = []

    for i in range(NUM_GAMES):
        army_a = load_army_from_hof(random.choice(hof_ml_data))
        army_b = load_army_from_hof(random.choice(hof_ml_data))
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")

        t0 = time.time()
        result = simulate_game(
            res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
            ml_planning="A", planning_params=PARAMS,
        )
        dt = time.time() - t0
        game_times.append(dt)
        wins[result] += 1

        if (i + 1) % 10 == 0 or i == 0:
            wr = (wins["A"] + 0.5 * wins["draw"]) / (i + 1) * 100
            print(f"  [{i+1:>3}/{NUM_GAMES}] W={wins['A']} L={wins['B']} D={wins['draw']}  "
                  f"WR={wr:.1f}%  avg={sum(game_times)/len(game_times):.2f}s/game")

    plan_wr = (wins["A"] + 0.5 * wins["draw"]) / NUM_GAMES * 100
    print(f"\nFinal: {wins['A']}W {wins['B']}L {wins['draw']}D  —  Plan WR: {plan_wr:.1f}%")
    print(f"Avg {sum(game_times)/len(game_times):.2f}s/game, total {sum(game_times):.0f}s")