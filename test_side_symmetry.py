"""Test that ML features are symmetric between A and B sides.

Checks objective symmetry, board symmetry, and runs a batch of games
(ML vs ML) to check for win-rate bias.
"""

import json
import random
import time
from pathlib import Path

import numpy as np

from board import OBJECTIVES, COLS, ROWS
from ml_features import _get_model_objectives
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList
from game import simulate_game

_DIR = Path(__file__).resolve().parent


# ── Test 1: Model-space objectives are symmetric under flip + remap ──

def test_objective_symmetry():
    """Verify _get_model_objectives returns identical values for A and B."""
    objs_a = _get_model_objectives("A")
    objs_b = _get_model_objectives("B")

    print("=== Objective Symmetry Test ===")
    print(f"  Player A objectives (model-space): {objs_a}")
    print(f"  Player B objectives (model-space): {objs_b}")

    all_match = True
    for i, (a, b) in enumerate(zip(objs_a, objs_b)):
        match = np.allclose(a, b, atol=1e-6)
        label = ["Centre", "My-side", "Enemy-side", "My-home", "Enemy-home"][i]
        status = "OK" if match else "MISMATCH"
        if not match:
            all_match = False
        print(f"  [{status}] {label}: A={a}, B={b}")

    print(f"  Result: {'PASS' if all_match else 'FAIL'}\n")
    return all_match


# ── Test 2: Physical board objective symmetry ──

def test_board_objective_symmetry():
    """Verify physical objectives are 180°-rotationally symmetric."""
    print("=== Physical Objective Symmetry Test ===")
    # For 180° rotation: (col, row) -> (COLS-1-col, ROWS-1-row)
    # Centre is excluded — no integer centre on an even grid.
    all_ok = True
    # Check paired objectives: A-side↔B-side, Home-A↔Home-B
    check_pairs = [
        ("A-side", 1, "B-side", 2),
        ("Home-A", 3, "Home-B", 4),
    ]
    for name_a, idx_a, name_b, idx_b in check_pairs:
        col_a, row_a = OBJECTIVES[idx_a]
        col_b, row_b = OBJECTIVES[idx_b]
        expected_col = COLS - 1 - col_a
        expected_row = ROWS - 1 - row_a
        ok = (expected_col == col_b and expected_row == row_b)
        if not ok:
            all_ok = False
        status = "OK" if ok else "MISMATCH"
        print(f"  [{status}] {name_a} ({col_a},{row_a}) <-> {name_b} ({col_b},{row_b})  "
              f"expected ({expected_col},{expected_row})")

    # Centre: report its position (no integer self-symmetry on even grid)
    cc, cr = OBJECTIVES[0]
    print(f"  [INFO] Centre at ({cc},{cr}) — no integer self-symmetry on {COLS}x{ROWS} grid")

    print(f"  Result: {'PASS' if all_ok else 'FAIL'}\n")
    return all_ok


# ── Test 3: ML vs ML win-rate bias ──

def test_win_rate_bias(num_games=200):
    """Run ML vs ML games and check for A/B bias."""
    print(f"=== Win-Rate Bias Test ({num_games} games) ===")

    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof = json.load(f)

    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model_label = "tactical"
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Model: {model_label}")

    def load_army(entry):
        army = ArmyList()
        for e in entry["entries"]:
            ent = make_entry(e["template_id"], upgrades=e.get("upgrades", {}),
                             ai_role=e.get("ai_role", "killer"))
            ent.combat_preference = e.get("combat_preference", "ranged")
            army.entries.append(ent)
        return army

    wins = {"A": 0, "B": 0, "draw": 0}
    t0 = time.time()

    for i in range(num_games):
        army_a = load_army(random.choice(hof))
        army_b = load_army(random.choice(hof))
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")
        result = simulate_game(
            res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
        )
        wins[result] += 1
        print(f"  Game {i+1}/{num_games}", end="\r")

    elapsed = time.time() - t0
    n = sum(wins.values())
    a_rate = wins["A"] / n * 100
    b_rate = wins["B"] / n * 100
    d_rate = wins["draw"] / n * 100

    print(f"  Completed in {elapsed:.1f}s ({elapsed/n:.2f}s/game)       ")
    print(f"  A wins: {wins['A']:>4} ({a_rate:.1f}%)")
    print(f"  B wins: {wins['B']:>4} ({b_rate:.1f}%)")
    print(f"  Draws:  {wins['draw']:>4} ({d_rate:.1f}%)")

    # Simple binomial test: under H0 (fair), P(A wins | not draw) = 0.5
    decisive = wins["A"] + wins["B"]
    if decisive > 0:
        a_frac = wins["A"] / decisive
        # 95% CI for proportion
        se = (0.25 / decisive) ** 0.5
        ci_lo = a_frac - 1.96 * se
        ci_hi = a_frac + 1.96 * se
        print(f"  A win fraction (excl. draws): {a_frac:.3f}  "
              f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
        fair = ci_lo <= 0.5 <= ci_hi
        print(f"  50% within CI: {'YES (no significant bias)' if fair else 'NO (significant bias detected)'}")
    print()


if __name__ == "__main__":
    test_board_objective_symmetry()
    test_objective_symmetry()
    test_win_rate_bias()
