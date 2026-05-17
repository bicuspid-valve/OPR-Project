#!/usr/bin/env python3
"""Probe: how often does the policy's argmax beat 8 uniformly-random
legal alternatives in the V-head's ranking?

This separates "policy is finding good actions" from the order-statistic
floor of best-of-K-policy-samples (which capped IR at ~76% in
probe_planning_M.py). Here the alternatives aren't drawn from the
policy — they're uniformly sampled from legal actions for uniformly-
chosen alive units. So:

  argmax_rate ≈ 1/9 ≈ 11%   ⟹  policy isn't meaningfully better than
                                random; V-head can't tell argmax apart
  argmax_rate climbs over training
                            ⟹  policy + V-head are jointly identifying
                                good actions

Sweeps a list of checkpoints and reports the trajectory.

Single-threaded by design. Pin with `taskset -c <core>` if the host has
an active training job.

Usage:
    python probe_uniform_baseline.py --n-games 15
    python probe_uniform_baseline.py --checkpoints \
        ml_checkpoints/checkpoint_batch_000000.pt \
        ml_checkpoints/checkpoint_batch_000050.pt \
        ml_checkpoints/checkpoint_batch_000200.pt
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from ml_model_tactical import TacticalModel  # noqa: E402
from ml_training import load_model_state_dict  # noqa: E402
from ml_training.collection import _run_games_batched_tactical  # noqa: E402
from ml_training.metrics import (  # noqa: E402
    _generate_army_pair, _load_hof_armies, _load_hof_ml_armies,
)


PLANNING_PARAMS = {
    "K_UNITS": 3,
    "C_SAMPLES_PER_UNIT": 3,
    "M_ROLLOUTS": 16,
    "N_LOOKAHEAD": 2,
    "SEQUENTIAL_HALVING": False,
    "UNIFORM_ALT_SAMPLING": True,  # the whole point
}


def collect_planned_records(model: TacticalModel, n_games: int, seed: int) -> list:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    hof_armies = _load_hof_armies()
    hof_ml_armies = _load_hof_ml_armies()

    game_specs = []
    for _ in range(n_games):
        res_a, res_b, states_a, states_b, army_type = _generate_army_pair(
            opp_type="selfplay_mirror",
            hof_armies=hof_armies, hof_ml_armies=hof_ml_armies,
        )
        sa = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a]
        sb = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b]
        game_specs.append((res_a, res_b, sa, sb, "selfplay_mirror", -1, army_type))

    results = _run_games_batched_tactical(
        main_model=model,
        game_specs=game_specs,
        opp_models={},
        shaping_scale=1.0,
        planning_rate=1.0,
        planning_params=PLANNING_PARAMS,
        randomize_sides=False,
    )
    planned = []
    for entry in results:
        traj = entry[0]
        for rec in traj:
            if getattr(rec, "was_planned", False):
                planned.append(rec)
    return planned


def summarize(records: list) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "argmax_rate": float("nan"),
                "se": float("nan"), "mean_vdelta": float("nan")}
    n_argmax_best = sum(1 for r in records if not r.planning_improved)
    ar = n_argmax_best / n
    se = math.sqrt(ar * (1 - ar) / n) if n > 1 else float("nan")
    vds = [r.planning_value_delta for r in records if r.planning_improved]
    mean_vd = sum(vds) / len(vds) if vds else 0.0
    return {"n": n, "argmax_rate": ar, "se": se, "mean_vdelta": mean_vd}


def parse_batch(name: str) -> int:
    """checkpoint_batch_000123.pt → 123."""
    stem = Path(name).stem
    digits = "".join(c for c in stem.split("_")[-1] if c.isdigit())
    return int(digits) if digits else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", type=str, nargs="*", default=None,
                    help="Specific checkpoint paths. Default: subsample of "
                         "ml_checkpoints/.")
    ap.add_argument("--n-games", type=int, default=15,
                    help="Games per checkpoint. ~10 planning calls/game per side.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-csv", type=str, default="ml_logs/uniform_baseline.csv",
                    help="Where to save the trajectory.")
    args = ap.parse_args()

    if args.checkpoints:
        ckpt_paths = [Path(p) for p in args.checkpoints]
    else:
        ckdir = Path(__file__).parent / "ml_checkpoints"
        all_ck = sorted(ckdir.glob("checkpoint_batch_*.pt"),
                        key=lambda p: parse_batch(p.name))
        # Subsample: 0, 50, 100, 200, 300, 400, 500, 600, 700
        targets = {0, 50, 100, 200, 300, 400, 500, 600, 700}
        by_batch = {parse_batch(p.name): p for p in all_ck}
        ckpt_paths = [by_batch[b] for b in sorted(targets) if b in by_batch]

    if not ckpt_paths:
        print("No checkpoints found.", file=sys.stderr); sys.exit(1)

    print(f"checkpoints: {len(ckpt_paths)}, n_games each: {args.n_games}, seed: {args.seed}")
    print(f"planning_params: {PLANNING_PARAMS}")
    print()

    rows = []
    for path in ckpt_paths:
        batch = parse_batch(path.name)
        t0 = time.time()
        model = TacticalModel()
        model.load_state_dict(load_model_state_dict(str(path)), strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        records = collect_planned_records(model, args.n_games, seed=args.seed)
        s = summarize(records)
        s["batch"] = batch
        s["elapsed"] = time.time() - t0
        rows.append(s)
        print(f"  batch {batch:>5d}: n={s['n']:>4d}  "
              f"argmax_best={s['argmax_rate']:.3f}±{s['se']:.3f}  "
              f"mean_vdelta={s['mean_vdelta']:.4f}  ({s['elapsed']:.0f}s)")

    print()
    print(f"{'batch':>6s}  {'n':>5s}  {'argmax_rate':>14s}  {'mean_vΔ':>9s}")
    print("-" * 50)
    for r in rows:
        ar_str = f"{r['argmax_rate']:.3f}±{r['se']:.3f}"
        print(f"{r['batch']:>6d}  {r['n']:>5d}  {ar_str:>14s}  {r['mean_vdelta']:>9.4f}")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write("batch,n,argmax_rate,se,mean_vdelta,elapsed_s\n")
            for r in rows:
                f.write(f"{r['batch']},{r['n']},{r['argmax_rate']:.6f},"
                        f"{r['se']:.6f},{r['mean_vdelta']:.6f},"
                        f"{r['elapsed']:.1f}\n")
        print(f"\nSaved trajectory to {out}")

    if len(rows) >= 2:
        first = rows[0]; last = rows[-1]
        d = last["argmax_rate"] - first["argmax_rate"]
        z = d / math.sqrt(first["se"]**2 + last["se"]**2) if first["se"] > 0 else 0
        print(f"\nargmax_rate(batch {last['batch']}) - argmax_rate(batch {first['batch']}) = "
              f"{d:+.3f}  (z = {z:+.2f})")
        if abs(z) < 1.96:
            print("  → no significant change. Either policy isn't learning to "
                  "identify good actions vs random,\n    or the V-head can't "
                  "rank them. Check loss/win-rate trajectories.")
        elif z > 0:
            print("  → argmax is winning more vs random alternatives over training.\n"
                  "    Consistent with the policy genuinely learning to pick good actions.")
        else:
            print("  → argmax is winning LESS over training. Unexpected; investigate.")


if __name__ == "__main__":
    main()
