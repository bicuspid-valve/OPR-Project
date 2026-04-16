"""Diagnose Round 1 asymmetry: what happens during the first round
that makes A and B board states distinguishable?

Uses simulate_game_recorded for both ML and heuristic so we can
extract exact post-Round-1 positions from frames.
"""
from __future__ import annotations

import random
from pathlib import Path
from collections import defaultdict

import numpy as np

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from board import ROWS, COLS
from evolution import HallOfFame, resolve_army, _make_unit_states
from game import simulate_game_recorded
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel

MIDLINE_Y = (ROWS - 1) / 2.0  # 23.5


def analyse_games(label, N, armies, model=None):
    a_actions = defaultdict(int)
    b_actions = defaultdict(int)
    a_dist_to_mid_r1 = []
    b_dist_to_mid_r1 = []
    a_dist_to_mid_deploy = []
    b_dist_to_mid_deploy = []
    a_kills_r1 = []
    b_kills_r1 = []
    centre_r1 = {"A": 0, "B": 0, "": 0}
    wins = {"A": 0, "B": 0, "draw": 0}

    for gi in range(N):
        (aa, ra), (ab, rb) = random.sample(armies, 2)
        sa = _make_unit_states(aa, ra, "A")
        sb = _make_unit_states(ab, rb, "B")

        result, frames, labels_f, owners_f, pts_f, _ = simulate_game_recorded(
            ra, rb, states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model)
        wins[result] += 1

        # --- Deployment frame (round 0) ---
        for f in frames:
            if f.get("round") == 0:
                for ui in range(len(owners_f)):
                    positions = f["positions"][ui]
                    if not positions:
                        continue
                    cy = sum(p[1] for p in positions) / len(positions)
                    if owners_f[ui] == "A":
                        a_dist_to_mid_deploy.append(MIDLINE_Y - cy)
                    else:
                        b_dist_to_mid_deploy.append(cy - MIDLINE_Y)
                break

        # --- Round 1 activations (action types) ---
        for f in frames:
            if f.get("round") != 1:
                continue
            desc = f.get("description", "")
            if "Objectives:" in desc:
                continue

            # Determine owner from description
            owner = None
            for li, lbl in enumerate(labels_f):
                if desc.startswith(lbl):
                    owner = owners_f[li]
                    break
            if owner is None:
                continue

            action = "hold"
            dl = desc.lower()
            if "advance" in dl:
                action = "advance"
            elif "rush" in dl:
                action = "rush"
            elif "charge" in dl:
                action = "charge"
            elif "hold" in dl:
                action = "hold"

            if owner == "A":
                a_actions[action] += 1
            else:
                b_actions[action] += 1

        # --- End of Round 1 frame ---
        for f in frames:
            if f.get("round") == 1 and "Objectives:" in f.get("description", ""):
                # Positions
                for ui in range(len(owners_f)):
                    positions = f["positions"][ui]
                    if not positions:
                        continue
                    cy = sum(p[1] for p in positions) / len(positions)
                    if owners_f[ui] == "A":
                        a_dist_to_mid_r1.append(MIDLINE_Y - cy)
                    else:
                        b_dist_to_mid_r1.append(cy - MIDLINE_Y)

                # Centre control
                centre_r1[f["objectives"][0]] += 1

                # Kill points
                alive = f["alive"]
                ak = sum(pts_f[j] for j in range(len(owners_f))
                         if owners_f[j] == "B" and alive[j] <= 0)
                bk = sum(pts_f[j] for j in range(len(owners_f))
                         if owners_f[j] == "A" and alive[j] <= 0)
                a_kills_r1.append(ak)
                b_kills_r1.append(bk)
                break

        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{N}")

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"  {label} ({N} games)")
    print(f"  Wins: A={wins['A']} ({100*wins['A']/N:.1f}%)  "
          f"B={wins['B']} ({100*wins['B']/N:.1f}%)  "
          f"draw={wins['draw']} ({100*wins['draw']/N:.1f}%)")
    print(f"{'='*60}")

    print(f"\n  Action distribution in Round 1:")
    total_a = sum(a_actions.values()) or 1
    total_b = sum(b_actions.values()) or 1
    print(f"  {'Action':<12} {'A count':>8} {'A %':>7} {'B count':>8} {'B %':>7} {'Delta':>7}")
    for act in ["advance", "rush", "hold", "charge"]:
        ac = a_actions.get(act, 0)
        bc = b_actions.get(act, 0)
        print(f"  {act:<12} {ac:>8} {100*ac/total_a:>6.1f}% {bc:>8} {100*bc/total_b:>6.1f}% "
              f"{100*bc/total_b - 100*ac/total_a:>+6.1f}%")
    print(f"  {'TOTAL':<12} {total_a:>8}         {total_b:>8}")

    if a_dist_to_mid_deploy:
        ad = np.array(a_dist_to_mid_deploy)
        bd = np.array(b_dist_to_mid_deploy)
        print(f"\n  Distance to midline at DEPLOYMENT (positive = own half):")
        print(f"    A units: {ad.mean():+.2f} ± {ad.std():.2f}  (min {ad.min():+.1f}, max {ad.max():+.1f})")
        print(f"    B units: {bd.mean():+.2f} ± {bd.std():.2f}  (min {bd.min():+.1f}, max {bd.max():+.1f})")

    if a_dist_to_mid_r1:
        am = np.array(a_dist_to_mid_r1)
        bm = np.array(b_dist_to_mid_r1)
        print(f"\n  Distance to midline AFTER ROUND 1 (positive = own half):")
        print(f"    A units: {am.mean():+.2f} ± {am.std():.2f}")
        print(f"    B units: {bm.mean():+.2f} ± {bm.std():.2f}")
        print(f"    A past midline: {(am < 0).mean():.1%}")
        print(f"    B past midline: {(bm < 0).mean():.1%}")
        print(f"    A-B gap: {am.mean() - bm.mean():+.2f}")

    t = sum(centre_r1.values()) or 1
    print(f"\n  Centre control after R1:")
    print(f"    A: {centre_r1['A']} ({100*centre_r1['A']/t:.1f}%)  "
          f"B: {centre_r1['B']} ({100*centre_r1['B']/t:.1f}%)  "
          f"Neutral: {centre_r1['']} ({100*centre_r1['']/t:.1f}%)")

    if a_kills_r1:
        print(f"\n  Kill points in R1:")
        print(f"    A kills: {np.mean(a_kills_r1):.0f} ± {np.std(a_kills_r1):.0f}")
        print(f"    B kills: {np.mean(b_kills_r1):.0f} ± {np.std(b_kills_r1):.0f}")
        print(f"    Delta: {np.mean(a_kills_r1) - np.mean(b_kills_r1):+.0f}")


def main():
    model = TacticalModel()
    model.load_state_dict(load_model_state_dict(
        Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt"), strict=False)
    model.eval()

    hof = HallOfFame.load_from_json(
        Path(__file__).resolve().parent / "results" / "hall_of_fame_ml.json")
    armies = [(e.army, resolve_army(e.army)) for e in hof.entries]

    N = 300
    print("=== HEURISTIC AI ===")
    analyse_games("Heuristic AI", N, armies, model=None)
    print("\n\n=== ML AI ===")
    analyse_games("ML AI", N, armies, model=model)


if __name__ == "__main__":
    main()
