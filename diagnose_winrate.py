"""Deep dive into WHY B wins 63% vs A's 25%.

Track per-round stats across 500 games:
- Objective control for all 5 objectives at end of each round
- Kill points (cumulative) for each side
- Final objective counts and margins
- Win rate by first-player (who goes first in round 1)
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from evolution import HallOfFame, resolve_army, _make_unit_states
from game import simulate_game_recorded
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from board import OBJECTIVES


def main():
    model_path = Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt"
    sd = load_model_state_dict(model_path)
    model = TacticalModel()
    model.load_state_dict(sd, strict=False)
    model.eval()

    hof = HallOfFame.load_from_json(Path(__file__).resolve().parent / "results" / "hall_of_fame_ml.json")
    armies = [(e.army, resolve_army(e.army)) for e in hof.entries]

    N = 500
    OBJ_NAMES = ["Centre", "A-side", "B-side", "Home-A", "Home-B"]

    # Per-round objective control: round -> obj_index -> {"A": n, "B": n, "": n}
    obj_ctrl = {r: {oi: {"A": 0, "B": 0, "": 0} for oi in range(5)} for r in range(1, 5)}

    # Per-round kill points
    kill_pts_a_by_round = {r: [] for r in range(1, 5)}  # pts of B killed by A
    kill_pts_b_by_round = {r: [] for r in range(1, 5)}  # pts of A killed by B

    # Final stats
    final_a_objs = []
    final_b_objs = []
    final_a_kills = []
    final_b_kills = []
    wins = {"A": 0, "B": 0, "draw": 0}

    # Activations per side
    a_activations = []
    b_activations = []

    # Track who goes first
    wins_when_a_first = {"A": 0, "B": 0, "draw": 0}
    wins_when_b_first = {"A": 0, "B": 0, "draw": 0}

    t0 = time.time()
    for game_i in range(N):
        (army_a, res_a), (army_b, res_b) = random.sample(armies, 2)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")

        pts_a_total = sum(u.unit.points for u in sa)
        pts_b_total = sum(u.unit.points for u in sb)

        result, frames, labels, owners, unit_pts, _ = simulate_game_recorded(
            res_a, res_b, states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
        )
        wins[result] += 1

        # Determine who went first from frame descriptions
        a_first = None
        for f in frames:
            desc = f.get("description", "")
            if "Player A" in desc and f.get("round", 0) == 1:
                a_first = True
                break
            elif "Player B" in desc and f.get("round", 0) == 1:
                a_first = False
                break
        if a_first is True:
            wins_when_a_first[result] += 1
        elif a_first is False:
            wins_when_b_first[result] += 1

        # Count activations per side
        n_a_act = sum(1 for f in frames if "Player A" in f.get("description", "") and f.get("round", 0) > 0)
        n_b_act = sum(1 for f in frames if "Player B" in f.get("description", "") and f.get("round", 0) > 0)
        a_activations.append(n_a_act)
        b_activations.append(n_b_act)

        # Find end-of-round frames for objectives
        for f in frames:
            rn = f.get("round", 0)
            desc = f.get("description", "")
            if rn > 0 and "Objectives:" in desc:
                for oi in range(5):
                    ctrl = f["objectives"][oi]
                    obj_ctrl[rn][oi][ctrl] += 1

        # Kill points per round from frame descriptions
        for f in frames:
            rn = f.get("round", 0)
            desc = f.get("description", "")
            if rn > 0 and "Objectives:" in desc:
                # Compute kill points from alive counts
                alive = f["alive"]
                a_killed = sum(unit_pts[j] for j in range(len(owners))
                               if owners[j] == "B" and alive[j] <= 0)
                b_killed = sum(unit_pts[j] for j in range(len(owners))
                               if owners[j] == "A" and alive[j] <= 0)
                kill_pts_a_by_round[rn].append(a_killed)
                kill_pts_b_by_round[rn].append(b_killed)

        # Final stats
        last_obj_frame = None
        for f in reversed(frames):
            if "Objectives:" in f.get("description", ""):
                last_obj_frame = f
                break

        if last_obj_frame:
            a_obj = sum(1 for c in last_obj_frame["objectives"] if c == "A")
            b_obj = sum(1 for c in last_obj_frame["objectives"] if c == "B")
            final_a_objs.append(a_obj)
            final_b_objs.append(b_obj)

        alive = frames[-1]["alive"] if frames else []
        if alive:
            a_kill = sum(unit_pts[j] for j in range(len(owners))
                         if owners[j] == "B" and alive[j] <= 0)
            b_kill = sum(unit_pts[j] for j in range(len(owners))
                         if owners[j] == "A" and alive[j] <= 0)
            final_a_kills.append(a_kill)
            final_b_kills.append(b_kill)

        if (game_i + 1) % 100 == 0:
            print(f"  {game_i+1}/{N} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"GAME OUTCOME ANALYSIS ({N} ML-vs-ML games, {elapsed:.0f}s)")
    print(f"{'='*70}")

    print(f"\nWin rates:")
    print(f"  A: {wins['A']}/{N} ({100*wins['A']/N:.1f}%)")
    print(f"  B: {wins['B']}/{N} ({100*wins['B']/N:.1f}%)")
    print(f"  draw: {wins['draw']}/{N} ({100*wins['draw']/N:.1f}%)")

    n_af = sum(wins_when_a_first.values())
    n_bf = sum(wins_when_b_first.values())
    if n_af > 0:
        print(f"\n  When A goes first ({n_af} games):")
        print(f"    A wins: {wins_when_a_first['A']} ({100*wins_when_a_first['A']/n_af:.1f}%)")
        print(f"    B wins: {wins_when_a_first['B']} ({100*wins_when_a_first['B']/n_af:.1f}%)")
    if n_bf > 0:
        print(f"  When B goes first ({n_bf} games):")
        print(f"    A wins: {wins_when_b_first['A']} ({100*wins_when_b_first['A']/n_bf:.1f}%)")
        print(f"    B wins: {wins_when_b_first['B']} ({100*wins_when_b_first['B']/n_bf:.1f}%)")

    print(f"\nActivations per game:")
    print(f"  A: {np.mean(a_activations):.1f} ± {np.std(a_activations):.1f}")
    print(f"  B: {np.mean(b_activations):.1f} ± {np.std(b_activations):.1f}")

    # Objective control per round
    print(f"\n{'='*70}")
    print(f"OBJECTIVE CONTROL BY ROUND")
    print(f"{'='*70}")
    for rn in range(1, 5):
        print(f"\n  Round {rn}:")
        print(f"  {'Objective':<12} {'A ctrl':>8} {'B ctrl':>8} {'Neutral':>8} {'B adv':>8}")
        print(f"  {'-'*48}")
        total = None
        for oi in range(5):
            a = obj_ctrl[rn][oi]["A"]
            b = obj_ctrl[rn][oi]["B"]
            n = obj_ctrl[rn][oi][""]
            t = a + b + n
            if t == 0:
                continue
            if total is None:
                total = t
            b_adv = (b - a) / t
            print(f"  {OBJ_NAMES[oi]:<12} {100*a/t:>7.1f}% {100*b/t:>7.1f}% {100*n/t:>7.1f}% {b_adv:>+7.1%}")

    # Kill points
    print(f"\n{'='*70}")
    print(f"CUMULATIVE KILL POINTS BY ROUND")
    print(f"{'='*70}")
    for rn in range(1, 5):
        ak = np.array(kill_pts_a_by_round[rn]) if kill_pts_a_by_round[rn] else np.array([0])
        bk = np.array(kill_pts_b_by_round[rn]) if kill_pts_b_by_round[rn] else np.array([0])
        print(f"  Round {rn}: A kills {ak.mean():.0f}±{ak.std():.0f} pts  |  "
              f"B kills {bk.mean():.0f}±{bk.std():.0f} pts  |  "
              f"delta {(ak-bk).mean():+.0f}")

    # Final objective counts
    fa = np.array(final_a_objs)
    fb = np.array(final_b_objs)
    print(f"\n{'='*70}")
    print(f"FINAL GAME STATE")
    print(f"{'='*70}")
    print(f"  Final objectives:  A={fa.mean():.2f}±{fa.std():.2f}  B={fb.mean():.2f}±{fb.std():.2f}")
    print(f"  Final kill pts:    A kills {np.mean(final_a_kills):.0f}±{np.std(final_a_kills):.0f}  "
          f"B kills {np.mean(final_b_kills):.0f}±{np.std(final_b_kills):.0f}")

    # Distribution of final objective counts
    print(f"\n  Final A objectives distribution:")
    for c in range(6):
        n = np.sum(fa == c)
        print(f"    {c} objectives: {n} games ({100*n/len(fa):.1f}%)")
    print(f"  Final B objectives distribution:")
    for c in range(6):
        n = np.sum(fb == c)
        print(f"    {c} objectives: {n} games ({100*n/len(fb):.1f}%)")


if __name__ == "__main__":
    main()
