"""Play a single ML-vs-ML game (HoF armies, no planning) and open the viewer."""

import random
from pathlib import Path

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import HallOfFame, resolve_army, _make_unit_states
from game import simulate_game_recorded

_DIR = Path(__file__).resolve().parent

# Load model
model_path = _DIR / "ml_checkpoints" / "final_model.pt"
sd = load_model_state_dict(model_path)
model = TacticalModel()
model.load_state_dict(sd, strict=False)
model.eval()

# Pick two random HoF armies
hof = HallOfFame.load_from_json(_DIR / "results" / "hall_of_fame_ml.json")
entry_a, entry_b = random.sample(hof.entries, 2)
army_a, res_a = entry_a.army, resolve_army(entry_a.army)
army_b, res_b = entry_b.army, resolve_army(entry_b.army)
sa = _make_unit_states(army_a, res_a, "A")
sb = _make_unit_states(army_b, res_b, "B")

# Play
result, frames, labels, owners, unit_points, unit_info = simulate_game_recorded(
    res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
    ml_model_a=model, ml_model_b=model,
)
print(f"Result: {result}  ({len(frames)} frames)")

# Launch viewer
from viewer import show_game
show_game(frames, labels, owners, mode="objectives",
          unit_points=unit_points, unit_info=unit_info)
