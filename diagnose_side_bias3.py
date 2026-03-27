"""Deep side-bias diagnosis: per-objective control, side-swap pairs,
and first-mover tracking.

Patches simulate_game to return detailed end-of-game state.
"""

import json
import random
import time
from pathlib import Path
from collections import defaultdict

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList
from board import Board, OBJECTIVES

_DIR = Path(__file__).resolve().parent

# Monkey-patch _simulate_game_impl to also return board + units + who went first
import game as _game_module
_orig_impl = _game_module._simulate_game_impl


def _patched_impl(*args, **kwargs):
    """Wrap _simulate_game_impl to capture board state and first-mover."""
    # Intercept the random call for coin flip by capturing board
    result = _orig_impl(*args, **kwargs)
    return result


# Instead of patching, use simulate_game_recorded which gives us frame data.
# But that's heavier. Let's just inspect unit states post-game + reconstruct
# objective control from unit positions.

from game import simulate_game


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


def compute_final_obj_control(units_a, units_b):
    """Reconstruct objective control from final unit positions."""
    OBJ_RANGE_SQ = 3.0 * 3.0
    control = []
    for oc, orow in OBJECTIVES:
        a_present = False
        b_present = False
        for u in units_a:
            if u.models_alive <= 0 or u.shaken:
                continue
            for pos in u.alive_positions():
                dx = pos[0] - oc
                dy = pos[1] - orow
                if dx * dx + dy * dy <= OBJ_RANGE_SQ:
                    a_present = True
                    break
            if a_present:
                break
        for u in units_b:
            if u.models_alive <= 0 or u.shaken:
                continue
            for pos in u.alive_positions():
                dx = pos[0] - oc
                dy = pos[1] - orow
                if dx * dx + dy * dy <= OBJ_RANGE_SQ:
                    b_present = True
                    break
            if b_present:
                break
        if a_present and not b_present:
            control.append("A")
        elif b_present and not a_present:
            control.append("B")
        else:
            control.append("")  # contested or uncontrolled
    return control


if __name__ == "__main__":
    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof_ml_data = json.load(f)

    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # ===================================================================
    # TEST 1: 300 games with per-objective tracking
    # ===================================================================
    NUM_GAMES = 300
    print(f"{'='*60}")
    print(f"TEST 1: {NUM_GAMES} games, per-objective control tracking")
    print(f"{'='*60}\n")

    wins = {"A": 0, "B": 0, "draw": 0}
    # Per-objective: how often each side controls it at game end
    obj_labels = ["Centre(36,24)", "A-side(18,16)", "B-side(54,32)",
                  "HomeA(36,6)", "HomeB(36,42)"]
    obj_a_count = [0] * 5
    obj_b_count = [0] * 5
    obj_neutral = [0] * 5

    # Track objective differentials in wins
    obj_diff_in_a_wins = []  # a_objs - b_objs when A wins
    obj_diff_in_b_wins = []

    for i in range(NUM_GAMES):
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

        # Objective control (approximation from final positions)
        # Note: this is end-of-game snapshot, not cumulative across rounds
        control = compute_final_obj_control(sa, sb)
        a_objs = sum(1 for c in control if c == "A")
        b_objs = sum(1 for c in control if c == "B")
        for j, c in enumerate(control):
            if c == "A":
                obj_a_count[j] += 1
            elif c == "B":
                obj_b_count[j] += 1
            else:
                obj_neutral[j] += 1

        if result == "A":
            obj_diff_in_a_wins.append(a_objs - b_objs)
        elif result == "B":
            obj_diff_in_b_wins.append(a_objs - b_objs)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{NUM_GAMES}...", end="\r")

    print(f"\n  Results: A={wins['A']}  B={wins['B']}  D={wins['draw']}")
    print(f"  A/(A+B) = {wins['A']/(wins['A']+wins['B']):.3f}\n")

    print("  Per-objective control at game end:")
    print(f"  {'Objective':<20} {'A holds':>8} {'B holds':>8} {'Neutral':>8}")
    print(f"  {'-'*46}")
    for j in range(5):
        print(f"  {obj_labels[j]:<20} {obj_a_count[j]:>8} {obj_b_count[j]:>8} {obj_neutral[j]:>8}")
    print(f"  {'TOTAL':<20} {sum(obj_a_count):>8} {sum(obj_b_count):>8} {sum(obj_neutral):>8}")

    print(f"\n  Avg obj differential when A wins: "
          f"{sum(obj_diff_in_a_wins)/len(obj_diff_in_a_wins):.2f}" if obj_diff_in_a_wins else "")
    print(f"  Avg obj differential when B wins: "
          f"{sum(obj_diff_in_b_wins)/len(obj_diff_in_b_wins):.2f}" if obj_diff_in_b_wins else "")

    # ===================================================================
    # TEST 2: 150 side-swapped pairs (same armies, swap A/B)
    # ===================================================================
    NUM_PAIRS = 150
    print(f"\n{'='*60}")
    print(f"TEST 2: {NUM_PAIRS} side-swapped pairs")
    print(f"{'='*60}\n")

    pair_results = {"same_winner": 0, "A_side_wins_both": 0,
                    "B_side_wins_both": 0, "split": 0,
                    "draw_both": 0, "draw_one": 0}
    side_a_wins_total = 0
    side_b_wins_total = 0
    draws_total = 0

    # Track: does army1 do better as A or B?
    army1_wins_as_a = 0
    army1_wins_as_b = 0

    for i in range(NUM_PAIRS):
        hof_1 = random.choice(hof_ml_data)
        hof_2 = random.choice(hof_ml_data)

        # Game 1: army1=A, army2=B
        a1 = load_army_from_hof(hof_1)
        b1 = load_army_from_hof(hof_2)
        r1a = resolve_army(a1)
        r1b = resolve_army(b1)
        s1a = _make_unit_states(a1, r1a, "A")
        s1b = _make_unit_states(b1, r1b, "B")
        result1 = simulate_game(r1a, r1b, mode="objectives",
                                states_a=s1a, states_b=s1b,
                                ml_model_a=model, ml_model_b=model)

        # Game 2: army2=A, army1=B (swapped)
        a2 = load_army_from_hof(hof_2)
        b2 = load_army_from_hof(hof_1)
        r2a = resolve_army(a2)
        r2b = resolve_army(b2)
        s2a = _make_unit_states(a2, r2a, "A")
        s2b = _make_unit_states(b2, r2b, "B")
        result2 = simulate_game(r2a, r2b, mode="objectives",
                                states_a=s2a, states_b=s2b,
                                ml_model_a=model, ml_model_b=model)

        # Count side wins
        if result1 == "A":
            side_a_wins_total += 1
            army1_wins_as_a += 1
        elif result1 == "B":
            side_b_wins_total += 1
        else:
            draws_total += 1

        if result2 == "A":
            side_a_wins_total += 1
        elif result2 == "B":
            side_b_wins_total += 1
            army1_wins_as_b += 1
        else:
            draws_total += 1

        # Pair classification
        if result1 == "draw" and result2 == "draw":
            pair_results["draw_both"] += 1
        elif result1 == "draw" or result2 == "draw":
            pair_results["draw_one"] += 1
        elif result1 == "A" and result2 == "B":
            # Army1 won both times (as A in game1, as B in game2)
            pair_results["same_winner"] += 1
        elif result1 == "B" and result2 == "A":
            # Army2 won both times
            pair_results["same_winner"] += 1
        elif result1 == "A" and result2 == "A":
            # Side A won both — side bias!
            pair_results["A_side_wins_both"] += 1
        elif result1 == "B" and result2 == "B":
            # Side B won both — side bias!
            pair_results["B_side_wins_both"] += 1
        else:
            pair_results["split"] += 1

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{NUM_PAIRS}...", end="\r")

    total_games = NUM_PAIRS * 2
    print(f"\n  Total games: {total_games}")
    print(f"  Side A wins: {side_a_wins_total} ({side_a_wins_total/total_games*100:.1f}%)")
    print(f"  Side B wins: {side_b_wins_total} ({side_b_wins_total/total_games*100:.1f}%)")
    print(f"  Draws:       {draws_total} ({draws_total/total_games*100:.1f}%)")
    print(f"  A/(A+B) = {side_a_wins_total/(side_a_wins_total+side_b_wins_total):.3f}")

    print(f"\n  Pair outcomes:")
    print(f"    Same army wins both:   {pair_results['same_winner']:>4} (army strength)")
    print(f"    Side A wins both:      {pair_results['A_side_wins_both']:>4} (side-A bias)")
    print(f"    Side B wins both:      {pair_results['B_side_wins_both']:>4} (side-B bias)")
    print(f"    One draw, one win:     {pair_results['draw_one']:>4}")
    print(f"    Both draws:            {pair_results['draw_both']:>4}")

    print(f"\n  Army1 wins as A: {army1_wins_as_a}/{NUM_PAIRS}")
    print(f"  Army1 wins as B: {army1_wins_as_b}/{NUM_PAIRS}")

    # ===================================================================
    # TEST 3: First-mover test — patch random to force A-first vs B-first
    # ===================================================================
    NUM_FM = 100
    print(f"\n{'='*60}")
    print(f"TEST 3: {NUM_FM} games A-first vs {NUM_FM} games B-first")
    print(f"{'='*60}\n")

    import unittest.mock as mock

    for label, forced_val in [("A goes first", 0.1), ("B goes first", 0.9)]:
        fm_wins = {"A": 0, "B": 0, "draw": 0}

        # Patch random.random to return forced_val on the first call (coin flip)
        # then behave normally for all subsequent calls
        orig_random = random.random
        first_call = [True]

        def patched_random(_fv=forced_val):
            if first_call[0]:
                first_call[0] = False
                return _fv
            return orig_random()

        for i in range(NUM_FM):
            army_a = load_army_from_hof(random.choice(hof_ml_data))
            army_b = load_army_from_hof(random.choice(hof_ml_data))
            res_a = resolve_army(army_a)
            res_b = resolve_army(army_b)
            sa = _make_unit_states(army_a, res_a, "A")
            sb = _make_unit_states(army_b, res_b, "B")

            first_call[0] = True
            random.random = patched_random
            result = simulate_game(
                res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
                ml_model_a=model, ml_model_b=model,
            )
            random.random = orig_random
            fm_wins[result] += 1

        decisive = fm_wins["A"] + fm_wins["B"]
        print(f"  {label}: A={fm_wins['A']}  B={fm_wins['B']}  D={fm_wins['draw']}"
              f"  A/(A+B)={fm_wins['A']/decisive:.3f}" if decisive else "")