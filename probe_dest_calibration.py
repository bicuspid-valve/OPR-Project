"""Probe: per-activation dest value-head calibration test.

For each MOVE_MOVE activation in one game (planner side), sample 100
random destinations from the policy-argmax unit's legal dest candidates
and record:
  - pp_v_dest: per-phase POST_DEST value head's evaluation (cheap, 1 fwd)
  - rollout_v: M=16 rollouts × N=2 lookahead, mean V (expensive)

The game proceeds with the policy's argmax decision regardless — the
calibration is a side-effect.

Output: ml_logs/dest_calibration.csv with one row per (activation, dest).
After the run, computes rank correlation per activation between pp_v_dest
and rollout_v, plus overall pooled correlation.
"""
import json
import os
import random
from pathlib import Path

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList
from game import simulate_game

_DIR = Path(__file__).resolve().parent
LOG_PATH = str(_DIR / "ml_logs" / "dest_calibration.csv")
NUM_DESTS = 100
M_ROLLOUTS = 16
N_LOOKAHEAD = 2


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


def main():
    # Reset log + counter
    os.makedirs(_DIR / "ml_logs", exist_ok=True)
    for p in (LOG_PATH, LOG_PATH + ".ctr"):
        if os.path.exists(p):
            os.remove(p)

    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof = json.load(f)
    ckpt = _DIR / "ml_checkpoints" / "final_model.pt"

    model = TacticalModel()
    model.load_state_dict(load_model_state_dict(ckpt), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    random.seed(42)
    aa = load_army_from_hof(random.choice(hof))
    ab = load_army_from_hof(random.choice(hof))
    ra = resolve_army(aa); rb = resolve_army(ab)
    sa = _make_unit_states(aa, ra, "A"); sb = _make_unit_states(ab, rb, "B")

    pp = {
        "CALIBRATION_DEST_PROBE": NUM_DESTS,
        "CALIBRATION_LOG_PATH": LOG_PATH,
        "M_ROLLOUTS": M_ROLLOUTS,
        "N_LOOKAHEAD": N_LOOKAHEAD,
        "NUM_WORKERS": 1,
    }
    print(f"checkpoint: {ckpt}")
    print(f"NUM_DESTS={NUM_DESTS}, M={M_ROLLOUTS}, N={N_LOOKAHEAD}")
    print(f"log path: {LOG_PATH}")
    print()
    print("playing one game with calibration on side A...")

    result = simulate_game(
        ra, rb, mode="objectives",
        states_a=sa, states_b=sb,
        ml_model_a=model, ml_model_b=model,
        ml_planning="A",
        planning_params=pp,
    )
    print(f"game result: {result}")

    # Analyze
    if not os.path.exists(LOG_PATH):
        print("No calibration data was logged (no MOVE_MOVE activations on A side?)")
        return

    import csv
    rows = []
    with open(LOG_PATH) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            r["pp_v_dest"] = float(r["pp_v_dest"])
            r["rollout_v"] = float(r["rollout_v"])
            r["activation_id"] = int(r["activation_id"])
            rows.append(r)
    print(f"\nlogged {len(rows)} (activation, dest) rows across "
          f"{len(set(r['activation_id'] for r in rows))} activations")

    # Per-activation rank correlation (Spearman via numpy)
    import numpy as np
    from collections import defaultdict
    per_act = defaultdict(list)
    for r in rows:
        per_act[r["activation_id"]].append((r["pp_v_dest"], r["rollout_v"]))

    def spearman(xs, ys):
        rx = np.argsort(np.argsort(xs)).astype(float)
        ry = np.argsort(np.argsort(ys)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    def pearson(xs, ys):
        return float(np.corrcoef(xs, ys)[0, 1])

    rhos = []
    pears = []
    print(f"\nPer-activation rank correlation (Spearman):")
    print(f"{'act_id':>6s}  {'n':>4s}  {'spearman':>10s}  {'pearson':>10s}  "
          f"{'top1_match':>10s}  {'top5_overlap':>12s}")
    for aid in sorted(per_act):
        pairs = per_act[aid]
        xs = np.array([p[0] for p in pairs])
        ys = np.array([p[1] for p in pairs])
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            continue
        rho = spearman(xs, ys)
        pr = pearson(xs, ys)
        rhos.append(rho); pears.append(pr)
        # top-1 agreement
        top1_match = (np.argmax(xs) == np.argmax(ys))
        # top-5 overlap (Jaccard-like)
        top5_x = set(np.argsort(-xs)[:5])
        top5_y = set(np.argsort(-ys)[:5])
        overlap = len(top5_x & top5_y) / 5.0
        print(f"{aid:>6d}  {len(pairs):>4d}  {rho:>10.4f}  {pr:>10.4f}  "
              f"{str(top1_match):>10s}  {overlap:>12.2f}")

    if rhos:
        print(f"\nMean Spearman rho across activations: {np.mean(rhos):.4f}  "
              f"(median {np.median(rhos):.4f})")
        print(f"Mean Pearson r across activations:    {np.mean(pears):.4f}  "
              f"(median {np.median(pears):.4f})")

    # Pooled correlation across all rows
    if rows:
        all_x = np.array([r["pp_v_dest"] for r in rows])
        all_y = np.array([r["rollout_v"] for r in rows])
        print(f"\nPooled across all rows (n={len(rows)}):")
        print(f"  Spearman: {spearman(all_x, all_y):.4f}")
        print(f"  Pearson:  {pearson(all_x, all_y):.4f}")


if __name__ == "__main__":
    main()
