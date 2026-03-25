"""Quick script: load final_model.pt, play ML vs heuristic until ML wins, show viewer."""

from pathlib import Path
from ml_model import StrategicModel
from ml_training import load_model_state_dict
from evolution import generate_random_army, resolve_army, _make_unit_states
from game import simulate_game_recorded
from viewer import show_game

_DIR = Path(__file__).resolve().parent

model = StrategicModel()
model.load_state_dict(
    load_model_state_dict(_DIR / "ml_checkpoints" / "final_model.pt"),
    strict=False,
)
model.eval()

attempts = 0
while True:
    attempts += 1
    army_a = generate_random_army(mode="objectives")
    army_b = generate_random_army(mode="objectives")
    res_a = resolve_army(army_a)
    res_b = resolve_army(army_b)
    sa = _make_unit_states(army_a, res_a, "A")
    sb = _make_unit_states(army_b, res_b, "B")

    result, frames, labels, owners, unit_points, unit_info = simulate_game_recorded(
        res_a, res_b, mode="objectives", states_a=sa, states_b=sb, ml_model_a=model,
    )

    if result == "A":
        print(f"ML player wins after {attempts} attempt(s)! ({len(frames)} frames)")
        break
    if attempts >= 200:
        print(f"No win in 200 attempts, showing last game.")
        break

show_game(frames, labels, owners, mode="objectives", unit_points=unit_points, unit_info=unit_info)
