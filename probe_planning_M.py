#!/usr/bin/env python3
"""Probe how plan_improve_rate (IR) and plan_mean_vdelta depend on M_ROLLOUTS.

If the IR plateau (~76%) is driven by V-head noise, IR should drop as M
rises (better V estimates → fewer false flips of "best != argmax").
If the plateau is structural (best-of-K order statistic over similar-
quality candidates), IR will be roughly flat across M.

Runs N_GAMES games for each M value in --m-values, with planning forced
on at every activation, then aggregates IR / vdelta / argmax_rate per M.

Single-threaded by design: safe to run alongside an active training job
provided the OS schedules it to a different core (use `taskset -c <N>`
to pin if needed).

Usage:
    taskset -c 7 python probe_planning_M.py --checkpoint ml_checkpoints/checkpoint_batch_000650.pt
    python probe_planning_M.py --quick    # 5 games per M, faster
"""
from __future__ import annotations

# Single-thread enforcement — must run before numpy/torch import.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import math
import random
import statistics
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


def collect_planning_stats(
    model: TacticalModel,
    n_games: int,
    M: int,
    seed: int,
) -> list:
    """Run n_games heuristic-opponent games with planning forced on at
    every activation, using the provided M_ROLLOUTS. Returns the list of
    every TacticalActivationRecord whose was_planned flag is set.

    Sequential halving is disabled for this probe so every candidate gets
    exactly M rollouts — matters for the M-effect comparison to be clean.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    hof_armies = _load_hof_armies()
    hof_ml_armies = _load_hof_ml_armies()

    # Use selfplay_mirror — collection.py:1132 gates planning_enabled on
    # is_mirror_game, so heuristic-opp games don't trigger the planner.
    game_specs = []
    for _ in range(n_games):
        res_a, res_b, states_a, states_b, army_type = _generate_army_pair(
            opp_type="selfplay_mirror",
            hof_armies=hof_armies,
            hof_ml_armies=hof_ml_armies,
        )
        states_a_data = [(u.ai_role, u.combat_preference, u.assigned_objective)
                         for u in states_a]
        states_b_data = [(u.ai_role, u.combat_preference, u.assigned_objective)
                         for u in states_b]
        game_specs.append((res_a, res_b, states_a_data, states_b_data,
                           "selfplay_mirror", -1, army_type))

    planning_params = {
        "K_UNITS": 3,
        "C_SAMPLES_PER_UNIT": 3,
        "M_ROLLOUTS": M,
        "N_LOOKAHEAD": 2,
        "SEQUENTIAL_HALVING": False,  # keep comparison clean: every candidate gets M
    }

    results = _run_games_batched_tactical(
        main_model=model,
        game_specs=game_specs,
        opp_models={},  # heuristic-only games need no opponent checkpoints
        shaping_scale=1.0,
        planning_rate=1.0,
        planning_params=planning_params,
        randomize_sides=False,
    )

    planned: list = []
    for entry in results:
        # _run_games_batched_tactical returns
        # (traj, result, opp_type, army_type, m_side, h_eff) per game,
        # plus a second tuple for the mirror-B side when applicable.
        traj = entry[0]
        for rec in traj:
            if getattr(rec, "was_planned", False):
                planned.append(rec)
    return planned


def summarize(records: list) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "ir": float("nan"), "ar": float("nan"),
                "se_ir": float("nan"), "mean_vdelta": float("nan"),
                "vdelta_std": float("nan")}
    n_improved = sum(1 for r in records if r.planning_improved)
    ir = n_improved / n
    ar = 1.0 - ir
    vdeltas = [r.planning_value_delta for r in records if r.planning_improved]
    mean_vdelta = sum(vdeltas) / len(vdeltas) if vdeltas else 0.0
    se_ir = math.sqrt(ir * (1 - ir) / n) if n > 1 else float("nan")
    return {
        "n": n,
        "ir": ir,
        "ar": ar,
        "se_ir": se_ir,
        "mean_vdelta": mean_vdelta,
        "vdelta_std": statistics.pstdev(vdeltas) if len(vdeltas) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to .pt checkpoint. Default: latest in ml_checkpoints/.")
    ap.add_argument("--n-games", type=int, default=20,
                    help="Games per M value. Default 20 (~200 activations).")
    ap.add_argument("--m-values", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--seed", type=int, default=42,
                    help="Same seed across M values gives matched-game comparison.")
    ap.add_argument("--quick", action="store_true",
                    help="Override n-games to 5 for a fast smoke test.")
    args = ap.parse_args()

    if args.quick:
        args.n_games = 5

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckdir = Path(__file__).parent / "ml_checkpoints"
        cks = sorted(ckdir.glob("checkpoint_batch_*.pt"))
        if not cks:
            print(f"No checkpoints in {ckdir}", file=sys.stderr)
            sys.exit(1)
        ckpt_path = str(cks[-1])
    print(f"checkpoint: {ckpt_path}")
    print(f"n_games per M: {args.n_games}    M values: {args.m_values}    seed: {args.seed}")
    print(f"single-threaded: torch={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    print()

    model = TacticalModel()
    model.load_state_dict(load_model_state_dict(ckpt_path), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    for M in args.m_values:
        t0 = time.time()
        records = collect_planning_stats(model, args.n_games, M, seed=args.seed)
        elapsed = time.time() - t0
        s = summarize(records)
        s["M"] = M
        s["elapsed_s"] = elapsed
        rows.append(s)
        print(f"  M={M:3d}: n_planned={s['n']:4d}  IR={s['ir']:.3f}±{s['se_ir']:.3f}  "
              f"AR={s['ar']:.3f}  mean_vdelta={s['mean_vdelta']:.4f}  "
              f"({elapsed:.0f}s)")

    print()
    print("Summary table:")
    print(f"{'M':>4s}  {'n':>5s}  {'IR':>9s}  {'AR':>6s}  {'mean_vΔ':>9s}  {'vΔ_std':>8s}  {'elapsed':>9s}")
    for r in rows:
        ir_str = f"{r['ir']:.3f}±{r['se_ir']:.3f}"
        print(f"{r['M']:>4d}  {r['n']:>5d}  {ir_str:>9s}  "
              f"{r['ar']:>6.3f}  {r['mean_vdelta']:>9.4f}  "
              f"{r['vdelta_std']:>8.4f}  {r['elapsed_s']:>8.1f}s")

    print()
    if len(rows) >= 2:
        ir_lo, ir_hi = rows[0]["ir"], rows[-1]["ir"]
        diff = ir_hi - ir_lo
        pooled_se = math.sqrt(rows[0]["se_ir"]**2 + rows[-1]["se_ir"]**2)
        z = diff / pooled_se if pooled_se > 0 else 0.0
        print(f"IR(M={rows[-1]['M']}) - IR(M={rows[0]['M']}) = {diff:+.3f}  (z = {z:+.2f})")
        if abs(z) < 1.96:
            print("  → no statistically significant trend across M.")
            print("    Consistent with structural floor (best-of-K geometry).")
        elif z < 0:
            print("  → IR drops as M rises. Consistent with noise-bound floor;")
            print("    higher M reduces V-head variance and lets candidate 0 win more often.")
        else:
            print("  → IR rises with M (unexpected). Worth investigating.")


if __name__ == "__main__":
    main()
