"""Test that ML features are perfectly symmetric between A and B sides.

Creates identical game states mirrored for both sides and verifies that
encode_state() produces identical feature vectors regardless of which
side the player is on. Also runs a batch of games (ML vs ML) to check
for win-rate bias.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from board import Board, OBJECTIVES, COLS, ROWS
from ml_features import (
    encode_state, _get_model_objectives, _flip_x, _flip_y,
    TOTAL_FEATURES, UNIT_FEATURES, MAX_UNITS_PER_SIDE,
)
from ml_model import StrategicModel
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList, UnitState
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


# ── Test 2: Feature encoding symmetry ──

def test_feature_symmetry():
    """Deploy the same army on both sides via deploy_armies and verify
    that encode_state produces identical feature vectors for A and B.

    Since both sides deploy at the same columns (not 180°-mirrored columns),
    we manually mirror B's positions to test pure encoding symmetry, then
    also test with real (non-mirrored) deployment to measure the leak.
    """
    print("=== Feature Encoding Symmetry Test ===")

    from board import COLS, ROWS
    from game import deploy_armies

    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof = json.load(f)

    def load_army(entry):
        army = ArmyList()
        for e in entry["entries"]:
            ent = make_entry(e["template_id"], upgrades=e.get("upgrades", {}),
                             ai_role=e.get("ai_role", "killer"))
            ent.combat_preference = e.get("combat_preference", "ranged")
            army.entries.append(ent)
        return army

    random.seed(42)
    hof_entry = random.choice(hof)
    army = load_army(hof_entry)
    resolved = resolve_army(army)

    # --- Part A: manually mirrored positions (pure encoding test) ---
    print("\n  Part A: manually 180°-mirrored positions")
    states_a = _make_unit_states(army, resolved, "A")
    states_b = _make_unit_states(army, resolved, "B")

    board_a = Board()
    random.seed(99)
    deploy_armies(states_a, [], board_a)

    # Mirror A's positions to create B's positions
    for sa, sb in zip(states_a, states_b):
        sb.positions = [(COLS - 1 - c, ROWS - 1 - r) for c, r in sa.positions]
        sb.models_alive = sa.models_alive
        sb.wounds_per_model = list(sa.wounds_per_model)
        sb.weapons_per_model = list(sa.weapons_per_model)

    board_both = Board()
    for us in states_a:
        for c, r in us.positions[:us.models_alive]:
            board_both.place(c, r)
    for us in states_b:
        for c, r in us.positions[:us.models_alive]:
            board_both.place(c, r)

    all_pass = True
    for round_num in [1, 4]:
        feat_a = encode_state(states_a, states_b, round_num, board_both, "A")
        feat_b = encode_state(states_b, states_a, round_num, board_both, "B")

        diff = (feat_a - feat_b).abs()
        max_diff = diff.max().item()
        num_diffs = (diff > 1e-6).sum().item()

        status = "PASS" if max_diff < 1e-5 else "FAIL"
        if max_diff >= 1e-5:
            all_pass = False
        print(f"    Round {round_num}: max_diff={max_diff:.8f}, "
              f"mismatched={num_diffs}/{TOTAL_FEATURES}  [{status}]")

        if max_diff > 1e-5:
            diff_indices = torch.where(diff > 1e-6)[0].tolist()
            for idx in diff_indices[:10]:
                slot = idx // UNIT_FEATURES
                feat_in_slot = idx % UNIT_FEATURES
                side = "friendly" if slot < MAX_UNITS_PER_SIDE else "enemy"
                unit_idx = slot % MAX_UNITS_PER_SIDE
                if idx >= MAX_UNITS_PER_SIDE * 2 * UNIT_FEATURES:
                    label = f"global[{idx - MAX_UNITS_PER_SIDE * 2 * UNIT_FEATURES}]"
                else:
                    label = f"{side}[{unit_idx}].feat[{feat_in_slot}]"
                print(f"      idx {idx}: {label}  A={feat_a[idx]:.6f}  B={feat_b[idx]:.6f}")

    # --- Part B: real deployment (both sides use same columns) ---
    print("\n  Part B: real deployment (same columns, measures deployment leak)")
    states_a2 = _make_unit_states(army, resolved, "A")
    states_b2 = _make_unit_states(army, resolved, "B")
    board2 = Board()
    random.seed(99)
    deploy_armies(states_a2, states_b2, board2)

    feat_a2 = encode_state(states_a2, states_b2, 1, board2, "A")
    feat_b2 = encode_state(states_b2, states_a2, 1, board2, "B")
    diff2 = (feat_a2 - feat_b2).abs()
    max_diff2 = diff2.max().item()
    num_diffs2 = (diff2 > 1e-6).sum().item()

    # Check position features specifically (feat indices 10, 11 per unit)
    pos_diffs = []
    for slot in range(MAX_UNITS_PER_SIDE * 2):
        base = slot * UNIT_FEATURES
        for off in [10, 11]:
            d = diff2[base + off].item()
            if d > 1e-6:
                pos_diffs.append((slot, off, feat_a2[base + off].item(),
                                  feat_b2[base + off].item(), d))

    print(f"    max_diff={max_diff2:.8f}, mismatched={num_diffs2}/{TOTAL_FEATURES}")
    if pos_diffs:
        print(f"    Position feature diffs (deployment column asymmetry): {len(pos_diffs)}")
        for slot, off, va, vb, d in pos_diffs[:6]:
            axis = "x" if off == 10 else "y"
            side = "friendly" if slot < 10 else "enemy"
            print(f"      {side}[{slot%10}].{axis}: A={va:.4f}  B={vb:.4f}  diff={d:.4f}")
    else:
        print(f"    No position diffs — deployment is symmetric!")

    print(f"\n  Encoding symmetry (mirrored positions): {'PASS' if all_pass else 'FAIL'}")
    print()


# ── Test 3: Physical board objective symmetry ──

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


# ── Test 4: ML vs ML win-rate bias ──

def test_win_rate_bias(num_games=200):
    """Run ML vs ML games and check for A/B bias."""
    print(f"=== Win-Rate Bias Test ({num_games} games) ===")

    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof = json.load(f)

    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    state_dict = load_model_state_dict(checkpoint_path)
    is_tactical = any(k.startswith("unit_selection_head") for k in state_dict)
    if is_tactical:
        model = TacticalModel()
        label = "tactical"
    else:
        model = StrategicModel()
        label = "strategic"
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Model: {label}")

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
    test_feature_symmetry()
    test_win_rate_bias()
