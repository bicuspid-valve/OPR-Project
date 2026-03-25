"""Quick test: what causes the SP-WR bias in tactical model?

Test 1: Tactical self-play (A=sample, B=argmax), random model
Test 2: Heuristic vs Heuristic (no ML)
Test 3: Tactical self-play, same armies with sides swapped
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import random
from ml_training import (
    _make_model, _generate_army_pair,
    _run_single_episode_tactical, _run_single_episode,
)
from board import OBJECTIVES


def run_test(label, n_games, run_fn):
    wins = 0.0
    for i in range(n_games):
        result = run_fn()
        if result == "A":
            wins += 1
        elif result == "draw":
            wins += 0.5
        if (i + 1) % 50 == 0:
            print(f"  {label}: {i+1}/{n_games}, A-WR={wins/(i+1):.3f}")
    wr = wins / n_games
    print(f"  {label}: FINAL A-WR={wr:.3f} ({wins}/{n_games})\n")
    return wr


def main():
    N = 200

    # Tactical model
    tac_model = _make_model("tactical")
    tac_model.eval()

    # Strategic model for comparison
    str_model = _make_model("strategic")
    str_model.eval()

    # Test 1: Tactical self-play (sample vs argmax)
    print("=== Test 1: TACTICAL self-play (A=sample, B=argmax), random model ===")
    def game_tactical():
        res_a, res_b, sa, sb, *_ = _generate_army_pair()
        sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sa]
        sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sb]
        _, result, _ = _run_single_episode_tactical(
            tac_model, tac_model, res_a, res_b, sa_data, sb_data, "selfplay", OBJECTIVES)
        return result
    wr1 = run_test("Tactical-selfplay", N, game_tactical)

    # Test 2: Strategic self-play for comparison
    print("=== Test 2: STRATEGIC self-play (A=sample, B=argmax), random model ===")
    def game_strategic():
        res_a, res_b, sa, sb, *_ = _generate_army_pair()
        sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sa]
        sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sb]
        _, result, _ = _run_single_episode(
            str_model, str_model, res_a, res_b, sa_data, sb_data, "selfplay", OBJECTIVES)
        return result
    wr2 = run_test("Strategic-selfplay", N, game_strategic)

    # Test 3: Heuristic vs Heuristic (baseline)
    print("=== Test 3: Heuristic vs Heuristic (no ML) ===")
    from game import simulate_game
    from evolution import generate_random_army, resolve_army, _make_unit_states
    def game_heuristic():
        army_a = generate_random_army(mode="objectives")
        army_b = generate_random_army(mode="objectives")
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")
        return simulate_game(res_a, res_b, mode="objectives", states_a=sa, states_b=sb)
    wr3 = run_test("Heuristic-vs-Heuristic", N, game_heuristic)

    # Test 4: Tactical self-play with armies swapped
    print("=== Test 4: Tactical self-play, same armies swapped ===")
    wins_normal = 0.0
    wins_swapped = 0.0
    for i in range(N):
        res_a, res_b, sa, sb, *_ = _generate_army_pair()
        sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sa]
        sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sb]
        # Normal
        _, r1, _ = _run_single_episode_tactical(
            tac_model, tac_model, res_a, res_b, sa_data, sb_data, "selfplay", OBJECTIVES)
        if r1 == "A": wins_normal += 1
        elif r1 == "draw": wins_normal += 0.5
        # Swapped armies
        _, r2, _ = _run_single_episode_tactical(
            tac_model, tac_model, res_b, res_a, sb_data, sa_data, "selfplay", OBJECTIVES)
        if r2 == "A": wins_swapped += 1
        elif r2 == "draw": wins_swapped += 0.5

    print(f"  Normal A-WR:  {wins_normal/N:.3f}")
    print(f"  Swapped A-WR: {wins_swapped/N:.3f}")
    print(f"  (If both >0.5, bias is in Player A's code path)\n")

    print("=== Summary ===")
    print(f"  Tactical self-play:     {wr1:.3f}")
    print(f"  Strategic self-play:    {wr2:.3f}")
    print(f"  Heuristic vs Heuristic: {wr3:.3f}")
    print(f"  Tactical normal A-WR:   {wins_normal/N:.3f}")
    print(f"  Tactical swapped A-WR:  {wins_swapped/N:.3f}")


if __name__ == "__main__":
    main()
