"""Diagnose the centre-objective asymmetry.

Physical centre is at (36, 24), which flips to (35, 23) — NOT self-symmetric.
Model centre is at (35.5, 23.5) — self-symmetric.
Scout rows: A=23, B=24.

This script checks:
1. Distance from each side's scout row to the PHYSICAL centre
2. How often each side controls the centre at deployment
3. Value prediction on a manually created symmetric state
4. Whether the projected objective control leaks asymmetry
"""
from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import numpy as np
import torch

from board import OBJECTIVES, COLS, ROWS, SCOUT_A_ROW, SCOUT_B_ROW, OBJ_SEIZE_RANGE
from models import ArmyList
from evolution import HallOfFame, resolve_army, _make_unit_states
from game import Board, UnitState, deploy_armies
from ml_features import (
    encode_state_tactical, precompute_damage,
    _get_model_objectives, _flip_x, _flip_y,
    _projected_objective_control_mapped, _objective_control_mapped,
    TACTICAL_UNIT_FEATURES, MAX_UNITS_PER_SIDE, TACTICAL_TOTAL_FEATURES,
)
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel


def main():
    import fast_core
    fast_core.USE_C_EXT = fast_core.is_available()

    print("=" * 60)
    print("CENTRE OBJECTIVE ASYMMETRY ANALYSIS")
    print("=" * 60)

    # 1. Physical vs model centre
    phys_centre = OBJECTIVES[0]
    model_centre = (35.5, 23.5)
    flipped_phys = (COLS - 1 - phys_centre[0], ROWS - 1 - phys_centre[1])

    print(f"\nPhysical centre objective: {phys_centre}")
    print(f"Physical centre flipped:   {flipped_phys}")
    print(f"Self-symmetric?            {'YES' if phys_centre == flipped_phys else 'NO — ASYMMETRIC'}")
    print(f"Model centre objective:    {model_centre}")
    print(f"Model centre flipped:      ({_flip_x(model_centre[0])}, {_flip_y(model_centre[1])})")
    print()

    # 2. Scout row distances to physical centre
    print("Scout row distances to PHYSICAL centre (36, 24):")
    dist_a = abs(SCOUT_A_ROW - phys_centre[1])
    dist_b = abs(SCOUT_B_ROW - phys_centre[1])
    print(f"  Scout A (row {SCOUT_A_ROW}): {dist_a} rows from centre row {phys_centre[1]}")
    print(f"  Scout B (row {SCOUT_B_ROW}): {dist_b} rows from centre row {phys_centre[1]}")
    print(f"  B is {'closer' if dist_b < dist_a else 'farther' if dist_b > dist_a else 'same distance'}")
    print()

    # 3. Check all physical objectives for symmetry
    print("Physical objective symmetry check:")
    for i, (c, r) in enumerate(OBJECTIVES):
        fc, fr = COLS - 1 - c, ROWS - 1 - r
        names = ["Centre", "A-side", "B-side", "Home-A", "Home-B"]
        sym = "SELF-SYMMETRIC" if (c, r) == (fc, fr) else f"flips to ({fc}, {fr})"
        print(f"  {names[i]}: ({c}, {r})  {sym}")
    print()

    # 4. Deploy mirror match and check centre control
    hof_path = Path(__file__).resolve().parent / "results" / "hall_of_fame_ml.json"
    hof = HallOfFame.load_from_json(hof_path)
    armies = [(e.army, resolve_army(e.army)) for e in hof.entries]
    army, res = armies[0]

    # Run many deployments and check who controls centre
    N = 1000
    a_controls_centre = 0
    b_controls_centre = 0
    neutral_centre = 0

    for _ in range(N):
        sa = _make_unit_states(army, res, "A")
        sb = _make_unit_states(army, res, "B")
        board = Board()
        deploy_armies(sa, sb, board)
        board.update_objectives(sa, sb)

        ctrl = board.objective_control[0]  # centre
        if ctrl == "A":
            a_controls_centre += 1
        elif ctrl == "B":
            b_controls_centre += 1
        else:
            neutral_centre += 1

    print(f"Centre control after deployment ({N} mirror games):")
    print(f"  A controls: {a_controls_centre} ({100*a_controls_centre/N:.1f}%)")
    print(f"  B controls: {b_controls_centre} ({100*b_controls_centre/N:.1f}%)")
    print(f"  Neutral:    {neutral_centre} ({100*neutral_centre/N:.1f}%)")
    print()

    # 5. Check projected objective control at deployment
    sa = _make_unit_states(army, res, "A")
    sb = _make_unit_states(army, res, "B")
    board = Board()
    deploy_armies(sa, sb, board)

    proj_a = _projected_objective_control_mapped(board, sa, sb, "A")
    proj_b = _projected_objective_control_mapped(board, sb, sa, "B")

    names = ["Centre(my-pov)", "My-side", "Enemy-side", "My-home", "Enemy-home"]
    print("Projected objective control at deployment:")
    print(f"  {'Objective':<20} {'A view':>8} {'B view':>8} {'Sum':>8}")
    for i, name in enumerate(names):
        print(f"  {name:<20} {proj_a[i]:>+8.2f} {proj_b[i]:>+8.2f} {proj_a[i]+proj_b[i]:>+8.2f}")
    print()

    # 6. Load model and check V_A + V_B on manually symmetric state
    model_path = Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt"
    sd = load_model_state_dict(model_path)
    model = TacticalModel()
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Use a fresh deployment
    fr_a, fm_a = precompute_damage([u.unit for u in sa], [u.unit for u in sb])
    fr_b, fm_b = precompute_damage([u.unit for u in sb], [u.unit for u in sa])
    pts_a = sum(u.unit.points for u in sa)
    pts_b = sum(u.unit.points for u in sb)

    vec_a = encode_state_tactical(
        sa, sb, 1, board, "A",
        friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
        enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
        total_friendly_points=pts_a, total_enemy_points=pts_b,
    )
    vec_b = encode_state_tactical(
        sb, sa, 1, board, "B",
        friendly_ranged_matchups=fr_b, friendly_melee_matchups=fm_b,
        enemy_ranged_matchups=fr_a, enemy_melee_matchups=fm_a,
        total_friendly_points=pts_b, total_enemy_points=pts_a,
    )

    with torch.no_grad():
        h_a, _, round_a = model.trunk(vec_a.unsqueeze(0))
        opp_a = model._get_opp_embed(h_a, None)
        v_a = model.value_head(h_a, round_a, opp_a).item()

        h_b, _, round_b = model.trunk(vec_b.unsqueeze(0))
        opp_b = model._get_opp_embed(h_b, None)
        v_b = model.value_head(h_b, round_b, opp_b).item()

    print(f"Value at deployment (mirror match, same army both sides):")
    print(f"  V_A = {v_a:+.4f}")
    print(f"  V_B = {v_b:+.4f}")
    print(f"  Sum = {v_a + v_b:+.4f}  (ideal = 0)")
    print()

    # 7. Check distance from each unit to PHYSICAL vs MODEL centre
    print("Unit distances to PHYSICAL centre (36,24) vs MODEL centre (35.5, 23.5):")
    print(f"  {'Unit':<35} {'Phys dist':>10} {'Model dist':>10} {'Delta':>8}")
    for i, us in enumerate(sa):
        if us.models_alive <= 0:
            continue
        cx, cy = us.centre()
        pd = math.sqrt((cx - 36)**2 + (cy - 24)**2)
        md = math.sqrt((cx - 35.5)**2 + (cy - 23.5)**2)
        print(f"  A{i} {us.unit.name:<30} {pd:>10.2f} {md:>10.2f} {pd-md:>+8.2f}")
    for i, us in enumerate(sb):
        if us.models_alive <= 0:
            continue
        cx, cy = us.centre()
        pd = math.sqrt((cx - 36)**2 + (cy - 24)**2)
        md = math.sqrt((cx - 35.5)**2 + (cy - 23.5)**2)
        print(f"  B{i} {us.unit.name:<30} {pd:>10.2f} {md:>10.2f} {pd-md:>+8.2f}")


def check_centre_control_during_games():
    """Run some games and check who controls the centre at end of each round."""
    import fast_core
    fast_core.USE_C_EXT = fast_core.is_available()

    model_path = Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt"
    sd = load_model_state_dict(model_path)
    model = TacticalModel()
    model.load_state_dict(sd, strict=False)
    model.eval()

    hof_path = Path(__file__).resolve().parent / "results" / "hall_of_fame_ml.json"
    hof = HallOfFame.load_from_json(hof_path)
    armies = [(e.army, resolve_army(e.army)) for e in hof.entries]

    from game import simulate_game_recorded

    N = 500
    # Track centre control by round
    centre_by_round = {r: {"A": 0, "B": 0, "": 0} for r in range(1, 5)}
    wins = {"A": 0, "B": 0, "draw": 0}

    for i in range(N):
        (army_a, res_a), (army_b, res_b) = random.sample(armies, 2)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")
        result, frames, labels, owners, pts, _ = simulate_game_recorded(
            res_a, res_b, states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
        )
        wins[result] = wins.get(result, 0) + 1

        # Find end-of-round frames
        for f in frames:
            rn = f.get('round', 0)
            desc = f.get('description', '')
            if 'End of Round' in desc or ('Objectives:' in desc and rn > 0):
                ctrl = f['objectives'][0]  # centre
                centre_by_round[rn][ctrl] += 1

    print(f"\n{'='*60}")
    print(f"CENTRE CONTROL DURING GAMEPLAY ({N} games)")
    print(f"{'='*60}")
    print(f"Win rates: A={wins['A']}/{N} ({100*wins['A']/N:.0f}%)  "
          f"B={wins['B']}/{N} ({100*wins['B']/N:.0f}%)  "
          f"draw={wins.get('draw',0)}/{N}")
    print()
    for rn in range(1, 5):
        total = sum(centre_by_round[rn].values())
        if total == 0:
            continue
        a = centre_by_round[rn]["A"]
        b = centre_by_round[rn]["B"]
        n = centre_by_round[rn][""]
        print(f"  Round {rn}: A controls {a} ({100*a/total:.0f}%)  "
              f"B controls {b} ({100*b/total:.0f}%)  "
              f"neutral {n} ({100*n/total:.0f}%)")


if __name__ == "__main__":
    main()
    check_centre_control_during_games()
