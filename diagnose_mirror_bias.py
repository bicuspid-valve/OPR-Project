"""Diagnostic: run N mirror games with a frozen random tactical model
and count physical-A vs physical-B wins.

Mirror games pit the main model against a literal copy of itself on the
same weights, so any win-rate asymmetry points at a bug in the game /
collection pipeline (not the ML path). A fresh random-weights model on
a fair game should yield ~50/50 A/B across mirror games.

Usage: python diagnose_mirror_bias.py [n_games]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evolution import generate_random_army, resolve_army, _make_unit_states
from game import simulate_game, deploy_armies
from board import Board
from ml_features import encode_state_tactical, precompute_damage
from ml_model_tactical import TacticalModel
from ml_training.collection import _run_games_batched_tactical


def build_mirror_game_specs(n_games: int) -> list[tuple]:
    specs = []
    for _ in range(n_games):
        army = generate_random_army(mode="objectives")
        resolved = resolve_army(army)
        states_a = _make_unit_states(army, resolved, "A")
        states_b = _make_unit_states(army, resolved, "B")
        states_a_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a]
        states_b_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b]
        specs.append((resolved, resolved, states_a_data, states_b_data,
                      "selfplay_mirror", -1, "random"))
    return specs


def run_training_path_heuristic(n_games: int) -> dict:
    """Run N non-mirror heuristic games through the training collection path
    (_run_games_batched_tactical) with side randomization. Compare the
    recorded win rate against the equivalent simulate_game run — they should
    match (within per-side averaging)."""
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model = TacticalModel()
    model.eval()

    specs = []
    for _ in range(n_games):
        army_main = generate_random_army(mode="objectives")
        army_opp = generate_random_army(mode="objectives")
        res_main = resolve_army(army_main)
        res_opp = resolve_army(army_opp)
        states_main = _make_unit_states(army_main, res_main, "A")
        states_opp = _make_unit_states(army_opp, res_opp, "B")
        sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_main]
        sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_opp]
        specs.append((res_main, res_opp, sa_data, sb_data,
                      "heuristic", -1, "random"))

    results = _run_games_batched_tactical(model, specs, opp_models={})
    counts = {"main": 0, "opp": 0, "draw": 0}
    for _, result, opp_type, _ in results:
        if opp_type == "heuristic":
            counts[result] = counts.get(result, 0) + 1
    return counts


def run_heuristic_mirror(n_games: int) -> tuple[int, int, int]:
    """Run N mirror games through simulate_game (pure heuristic, no ML).
    Returns (a_wins, b_wins, draws)."""
    a_wins = b_wins = draws = 0
    for _ in range(n_games):
        army = generate_random_army(mode="objectives")
        resolved = resolve_army(army)
        states_a = _make_unit_states(army, resolved, "A")
        states_b = _make_unit_states(army, resolved, "B")
        result = simulate_game(resolved, resolved, mode="objectives",
                               states_a=states_a, states_b=states_b)
        if result == "A":
            a_wins += 1
        elif result == "B":
            b_wins += 1
        else:
            draws += 1
    return a_wins, b_wins, draws


def _print_split(label: str, a_wins: int, b_wins: int, draws: int) -> None:
    total = a_wins + b_wins + draws
    print(f"\n{label} (N={total}):")
    print(f"  Physical A wins: {a_wins:4d}  ({100*a_wins/total:.1f}%)")
    print(f"  Physical B wins: {b_wins:4d}  ({100*b_wins/total:.1f}%)")
    print(f"  Draws:           {draws:4d}  ({100*draws/total:.1f}%)")
    decisive = a_wins + b_wins
    if decisive > 0:
        a_rate = a_wins / decisive
        print(f"  Decisive-game split: A={a_rate*100:.1f}% vs B={(1-a_rate)*100:.1f}%")
        if abs(a_rate - 0.5) > 0.08:
            print(f"  *** SIDE BIAS DETECTED ***")
        else:
            print(f"  No significant side bias.")


def compare_mirror_features():
    """Deploy one random army on both sides, encode state from A's and B's
    perspectives, and compare the feature vectors. If the encoding is truly
    mirror-symmetric the two vectors should be element-wise identical (or
    at least numerically very close)."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    army = generate_random_army(mode="objectives")
    resolved = resolve_army(army)
    units_a = _make_unit_states(army, resolved, "A")
    units_b = _make_unit_states(army, resolved, "B")
    for u in units_a:
        u.owner = "A"
    for u in units_b:
        u.owner = "B"

    board = Board()
    deploy_armies(units_a, units_b, board)

    # Print unit centroids to verify mirror deployment
    print(f"\n--- Unit centroids after deployment ---")
    for i, (ua, ub) in enumerate(zip(units_a, units_b)):
        if ua.models_alive > 0 and ub.models_alive > 0:
            a_cx, a_cy = ua.centre()
            b_cx, b_cy = ub.centre()
            # Mirror of A should equal B: flipped_A = (71-a_cx, 47-a_cy)
            mirror_a = (71 - a_cx, 47 - a_cy)
            dx = b_cx - mirror_a[0]
            dy = b_cy - mirror_a[1]
            flag = "" if abs(dx) < 0.01 and abs(dy) < 0.01 else "  <-- not mirror"
            print(f"  unit {i}: A=({a_cx:.1f},{a_cy:.1f})  "
                  f"B=({b_cx:.1f},{b_cy:.1f})  "
                  f"mirror_of_A=({mirror_a[0]:.1f},{mirror_a[1]:.1f}){flag}")

    fr_a, fm_a = precompute_damage([u.unit for u in units_a], [u.unit for u in units_b])
    fr_b, fm_b = precompute_damage([u.unit for u in units_b], [u.unit for u in units_a])
    pts_a = sum(u.unit.points for u in units_a)
    pts_b = sum(u.unit.points for u in units_b)

    # Encode from A's perspective (friendly=A)
    state_a = encode_state_tactical(
        units_a, units_b, round_num=1, board=board, player="A",
        friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
        enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
        total_friendly_points=pts_a, total_enemy_points=pts_b,
    ).numpy()

    # Encode from B's perspective (friendly=B)
    state_b = encode_state_tactical(
        units_b, units_a, round_num=1, board=board, player="B",
        friendly_ranged_matchups=fr_b, friendly_melee_matchups=fm_b,
        enemy_ranged_matchups=fr_a, enemy_melee_matchups=fm_a,
        total_friendly_points=pts_b, total_enemy_points=pts_a,
    ).numpy()

    print(f"\n=== Mirror feature-encoding symmetry test ===")
    print(f"State vec length: {len(state_a)}")
    print(f"A-state: norm={np.linalg.norm(state_a):.4f}  "
          f"mean={state_a.mean():.4f}  nonzero={int((state_a != 0).sum())}")
    print(f"B-state: norm={np.linalg.norm(state_b):.4f}  "
          f"mean={state_b.mean():.4f}  nonzero={int((state_b != 0).sum())}")
    diff = state_a - state_b
    print(f"Elementwise diff: max_abs={np.abs(diff).max():.6f}  "
          f"mean_abs={np.abs(diff).mean():.6f}  "
          f"nonzero_idx={int((diff != 0).sum())}")

    if np.abs(diff).max() > 1e-5:
        # Show the top-10 positions that differ most
        top_idx = np.argsort(-np.abs(diff))[:10]
        print(f"Top 10 differing feature indices:")
        for idx in top_idx:
            print(f"  [{idx:5d}] A={state_a[idx]:.5f}  B={state_b[idx]:.5f}  "
                  f"diff={diff[idx]:+.5f}")
    else:
        print("Features are element-wise mirror-symmetric (within 1e-5).")


def run_ml_mirror_seeded(n_games: int, seed: int) -> tuple[int, int, int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TacticalModel()
    model.eval()
    specs = build_mirror_game_specs(n_games)
    results = _run_games_batched_tactical(model, specs, opp_models={})
    a = sum(1 for _, r, t, _ in results if t == "selfplay_mirror" and r == "main")
    b = sum(1 for _, r, t, _ in results if t == "selfplay_mirror" and r == "opp")
    d = sum(1 for _, r, t, _ in results if t == "selfplay_mirror" and r == "draw")
    return a, b, d


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    if len(sys.argv) > 2 and sys.argv[2] == "heuristic":
        print(f"=== Training path: main (random model) vs heuristic (N={n_games}) ===")
        counts = run_training_path_heuristic(n_games)
        total = sum(counts.values())
        main_wr = (counts["main"] + 0.5 * counts["draw"]) / max(total, 1)
        print(f"  main wins: {counts['main']}  ({100*counts['main']/total:.1f}%)")
        print(f"  opp wins:  {counts['opp']}   ({100*counts['opp']/total:.1f}%)")
        print(f"  draws:     {counts['draw']}   ({100*counts['draw']/total:.1f}%)")
        print(f"  recorded win rate: {main_wr:.3f}")
        print(f"  (random policy vs heuristic should be near 0.05)")
        return

    if len(sys.argv) > 2 and sys.argv[2] == "seedsweep":
        print(f"=== Seed sweep: ML mirror with different random inits (N={n_games}/seed) ===")
        for seed in [42, 123, 7, 2024, 99]:
            a, b, d = run_ml_mirror_seeded(n_games, seed)
            decisive = a + b
            rate = a / decisive if decisive else 0.5
            print(f"  seed={seed:5d}:  A={a:4d}  B={b:4d}  D={d:4d}  "
                  f"A-rate={rate*100:5.1f}%")
        return

    compare_mirror_features()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Test 1: heuristic-vs-heuristic mirror via simulate_game ---
    print(f"=== Heuristic mirror (simulate_game, no ML) ===")
    print(f"Running {n_games} mirror games...")
    h_a, h_b, h_d = run_heuristic_mirror(n_games)
    _print_split("Heuristic mirror outcomes", h_a, h_b, h_d)

    # Re-seed so the ML test is independent of how many RNG draws the
    # heuristic test consumed (makes the two paths directly comparable).
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Test 2: ML mirror via _run_games_batched_tactical ---
    print(f"\n=== ML mirror (_run_games_batched_tactical, frozen random model) ===")

    # Zero-weights sanity check: the model always outputs zeros, which
    # makes every action uniformly random under the masked softmax. This
    # isolates the game engine from any model-output bias.
    zeroed = TacticalModel()
    zeroed.eval()
    with torch.no_grad():
        for p in zeroed.parameters():
            p.zero_()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    specs_z = build_mirror_game_specs(n_games)
    results_z = _run_games_batched_tactical(zeroed, specs_z, opp_models={})
    a_w = sum(1 for _, r, t, _ in results_z if t == "selfplay_mirror" and r == "main")
    b_w = sum(1 for _, r, t, _ in results_z if t == "selfplay_mirror" and r == "opp")
    d_w = sum(1 for _, r, t, _ in results_z if t == "selfplay_mirror" and r == "draw")
    _print_split("Zero-weights mirror (uniform random actions)", a_w, b_w, d_w)

    print(f"\n-- Random-weights mirror --")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model = TacticalModel()
    model.eval()

    print(f"Building {n_games} mirror game specs...")
    specs = build_mirror_game_specs(n_games)

    print(f"Running {n_games} mirror games with frozen random-weights model...")
    results = _run_games_batched_tactical(model, specs, opp_models={})

    # Each mirror game emits two trajectories:
    #   opp_type="selfplay_mirror" (physical-A side of the main model)
    #   opp_type="mirror_b"        (physical-B side)
    # After the _to_main transform in _run_games_batched_tactical:
    #   selfplay_mirror: "main"=A wins, "opp"=B wins, "draw"=draw
    #   mirror_b: "main"=B wins, "opp"=A wins, "draw"=draw  (complementary)
    a_side = {"main": 0, "opp": 0, "draw": 0}
    b_side = {"main": 0, "opp": 0, "draw": 0}
    a_activations: list[int] = []
    b_activations: list[int] = []
    a_move_types: dict[int, int] = {0: 0, 1: 0}
    b_move_types: dict[int, int] = {0: 0, 1: 0}
    a_rush_only = 0
    b_rush_only = 0
    for traj, result, opp_type, _ in results:
        if opp_type == "selfplay_mirror":
            a_side[result] = a_side.get(result, 0) + 1
            a_activations.append(len(traj))
            for step in traj:
                a_move_types[step.move_type] = a_move_types.get(step.move_type, 0) + 1
                if (step.move_type == 0 and step.dest_advance_reachable is not None
                        and 0 <= step.dest_selected_idx < len(step.dest_advance_reachable)
                        and not step.dest_advance_reachable[step.dest_selected_idx]):
                    a_rush_only += 1
        elif opp_type == "mirror_b":
            b_side[result] = b_side.get(result, 0) + 1
            b_activations.append(len(traj))
            for step in traj:
                b_move_types[step.move_type] = b_move_types.get(step.move_type, 0) + 1
                if (step.move_type == 0 and step.dest_advance_reachable is not None
                        and 0 <= step.dest_selected_idx < len(step.dest_advance_reachable)
                        and not step.dest_advance_reachable[step.dest_selected_idx]):
                    b_rush_only += 1

    if a_activations and b_activations:
        a_mean = sum(a_activations) / len(a_activations)
        b_mean = sum(b_activations) / len(b_activations)
        print(f"\nPer-side stats (mirror):")
        print(f"  Activations/game: A={a_mean:.2f}  B={b_mean:.2f}")
        print(f"  MOVE_MOVE count:  A={a_move_types[0]:5d}  B={b_move_types[0]:5d}")
        print(f"  MOVE_CHARGE count:A={a_move_types[1]:5d}  B={b_move_types[1]:5d}")
        print(f"  rush-only moves:  A={a_rush_only:5d}  B={b_rush_only:5d}")

    a_wins = a_side["main"]
    b_wins = a_side["opp"]
    draws = a_side["draw"]
    _print_split("ML mirror outcomes", a_wins, b_wins, draws)

    # Sanity check: mirror_b should report the complementary perspective.
    # a_side['main'] should equal b_side['opp'] (both mean "physical A won").
    print(f"\nComplementarity sanity check:")
    print(f"  a_side['main'] (A wins) = {a_side['main']}  vs  b_side['opp'] = {b_side['opp']}")
    print(f"  a_side['opp']  (B wins) = {a_side['opp']}  vs  b_side['main'] = {b_side['main']}")
    print(f"  a_side['draw']          = {a_side['draw']}  vs  b_side['draw'] = {b_side['draw']}")
    if (a_side['main'] != b_side['opp']
            or a_side['opp'] != b_side['main']
            or a_side['draw'] != b_side['draw']):
        print("  *** MISMATCH — _to_main transform is wrong for mirror_b ***")
    else:
        print("  OK — perspectives are consistent.")


if __name__ == "__main__":
    main()
