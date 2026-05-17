#!/usr/bin/env python3
"""
Plot training metrics from training_tactical.csv.

Usage:
    python result_plotting.py [csv_path] [block_size]

Defaults:
    csv_path   = ml_logs/training_tactical.csv
    block_size = 10
"""

import csv
import sys
import pathlib
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def load_csv(path: str) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        columns = {h: [] for h in headers}
        timestamps: list = []
        session_start: list = []
        next_is_session_start = True  # first data row starts a session
        for row in reader:
            # Skip non-data rows (e.g. "Training started"/"Training finished").
            # Any non-data row marks a session boundary — the next real batch
            # is the first of a new session and its wall-time delta to the
            # previous batch spans a shutdown/restart gap.
            try:
                float(row["batch"])
            except (ValueError, TypeError):
                next_is_session_start = True
                continue
            for h in headers:
                try:
                    columns[h].append(float(row[h]))
                except (ValueError, TypeError):
                    columns[h].append(float("nan"))
            try:
                ts = datetime.fromisoformat(row["timestamp"])
            except (ValueError, KeyError, TypeError):
                ts = None
            timestamps.append(ts)
            session_start.append(next_is_session_start)
            next_is_session_start = False
    # Convert to arrays and sort by batch
    for h in headers:
        columns[h] = np.array(columns[h])
    order = np.argsort(columns["batch"])
    for h in headers:
        columns[h] = columns[h][order]
    ts_sorted = [timestamps[i] for i in order]
    ss_sorted = [session_start[i] for i in order]

    # Derive per-batch wall time (seconds) from consecutive timestamps.
    # NaN for the first batch overall and for the first batch of each
    # resumed session, since that delta spans the inter-session gap.
    n = len(ts_sorted)
    bt = np.full(n, np.nan)
    for i in range(1, n):
        if ss_sorted[i]:
            continue
        t_prev, t_cur = ts_sorted[i - 1], ts_sorted[i]
        if t_prev is None or t_cur is None:
            continue
        dt = (t_cur - t_prev).total_seconds()
        if dt >= 0:
            bt[i] = dt
    columns["batch_time_derived"] = bt
    return columns


def block_average(data: dict, col: str, block_size: int):
    vals = data[col]
    batches = data["batch"]
    bx, by = [], []
    for i in range(0, len(vals), block_size):
        chunk_b = batches[i : i + block_size]
        chunk_v = vals[i : i + block_size]
        mask = ~np.isnan(chunk_v)
        if mask.any():
            bx.append(np.median(chunk_b[mask]))
            by.append(np.mean(chunk_v[mask]))
    return np.array(bx), np.array(by)


def main():
    user_size = int(input("Please enter block size: "))
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "ml_logs/training_tactical.csv"
    block_size = int(sys.argv[2]) if len(sys.argv) > 2 else user_size

    path = pathlib.Path(__file__).parent / csv_path
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    data = load_csv(path)
    n = len(data["batch"])
    print(f"Loaded {n} batches from {path.name}")

    fig = plt.figure(figsize=(16, 50))
    fig.suptitle(
        f"Tactical Training Dashboard  (block avg = {block_size} batches)",
        fontsize=14,
        fontweight="bold",
    )
    gs = GridSpec(12, 2, figure=fig, hspace=0.45, wspace=0.28)

    # --- Panel 1: Win rates vs heuristic ---
    ax1 = fig.add_subplot(gs[0, 0])
    for col, label, color in [
        ("h_hof_wr", "HoF armies", "#534AB7"),
        ("h_ml_wr", "HoF-ML armies", "#1D9E75"),
    ]:
        bx, by = block_average(data, col, block_size)
        ax1.plot(bx, by, marker="o", markersize=3, linewidth=1.5, color=color, label=label)
    ax1.set_ylabel("Win Rate")
    ax1.set_title("Win Rate vs Heuristic (by army source)")
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)

    # --- Panel 2: Win rates self-play ---
    ax2 = fig.add_subplot(gs[0, 1])
    for col, label, color in [
        ("sp_hof_wr", "HoF armies", "#D85A30"),
        ("sp_ml_wr", "HoF-ML armies", "#3266AD"),
        ("sp_rnd_wr", "Random armies", "#888888"),
    ]:
        bx, by = block_average(data, col, block_size)
        ax2.plot(bx, by, marker="o", markersize=3, linewidth=1.5, color=color, label=label)
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax2.set_ylabel("Win Rate")
    ax2.set_title("Win Rate vs Self-Play (by army source)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.3, 0.7)
    ax2.grid(alpha=0.3)

    # --- Panel 3: Loss ---
    ax3 = fig.add_subplot(gs[1, 0])
    bx, by = block_average(data, "loss", block_size)
    ax3.plot(bx, by, marker="o", markersize=3, linewidth=1.5, color="#C0392B")
    if len(by) > 2 and by[0] > by[1] * 3:
        ax3.set_ylim(top=by[1] * 2.5)
    ax3.set_ylabel("Loss")
    ax3.set_title("Policy + Value Loss")
    ax3.grid(alpha=0.3)

    # --- Panel 4: Mean reward ---
    ax4 = fig.add_subplot(gs[1, 1])
    bx, by = block_average(data, "mean_reward", block_size)
    ax4.plot(bx, by, marker="o", markersize=3, linewidth=1.5, color="#2E86C1")
    ax4.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax4.set_ylabel("Mean Reward")
    ax4.set_title("Mean Episode Reward")
    ax4.grid(alpha=0.3)

    # --- Panel 5: PPO Clip Fraction ---
    ax5 = fig.add_subplot(gs[2, 0])
    if "clip_frac" in data:
        bx, by = block_average(data, "clip_frac", block_size)
        ax5.plot(bx, by, marker="o", markersize=3, linewidth=1.5, color="#E67E22")
        ax5.axhline(0.2, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="Warning (0.2)")
        ax5.legend(fontsize=8)
    else:
        ax5.text(0.5, 0.5, "clip_frac not in CSV\n(older log format)",
                 transform=ax5.transAxes, ha="center", va="center", fontsize=10, color="gray")
    ax5.set_ylabel("Clip Fraction")
    ax5.set_title("PPO Clip Fraction")
    ax5.set_ylim(bottom=0)
    ax5.grid(alpha=0.3)

    # --- Panel 6: Overall Policy Entropy ---
    ax6 = fig.add_subplot(gs[2, 1])
    bx, by = block_average(data, "mean_entropy", block_size)
    ax6.plot(bx, by, marker="o", markersize=3, linewidth=2.0, color="#8E44AD")
    ax6.set_ylabel("Entropy")
    ax6.set_title("Overall Policy Entropy")
    ax6.set_ylim(bottom=0)
    ax6.grid(alpha=0.3)

    # --- Panel 6b: Per-head Entropy with target reference lines ---
    # Targets mirror config.py / entropy.py:
    #   entropy_target_fraction = 0.25      (masked categoricals: unit, charge, shoot)
    #   entropy_target_move     = 0.25·ln 2 (fixed, ~0.173)
    #   entropy_target_dest_fraction = 0.25 of ln(N_valid) — raw target varies.
    # For unit/charge/shoot the per-sample target = 0.25·ln(N_legal); the dashed
    # line is the upper bound at N_legal = 10 (all units alive / all enemies
    # alive / all targets in range), so actual targets are usually lower.
    _frac = 0.25
    _ln10 = float(np.log(10))
    _ln2 = float(np.log(2))
    _per_head = [
        ("unit",   _frac * _ln10, "#E6194B", "target (N=10 upper bound)"),
        ("move",   _frac * _ln2,  "#3CB44B", "target (0.25·ln 2)"),
        ("dest",   None,          "#0082C8", None),
        ("charge", _frac * _ln10, "#F58231", "target (N=10 upper bound)"),
        ("shoot",  _frac * _ln10, "#911EB4", "target (N=10 upper bound)"),
    ]
    sub6b = gs[3:5, :].subgridspec(3, 2, hspace=0.5, wspace=0.2)
    for _i, (_head, _tgt, _color, _tgt_label) in enumerate(_per_head):
        _row, _col_i = divmod(_i, 2)
        _ax = fig.add_subplot(sub6b[_row, _col_i])
        _col = f"ent_{_head}"
        if _col in data:
            _bx, _by = block_average(data, _col, block_size)
            _ax.plot(_bx, _by, linewidth=1.5, color=_color, label=_head)
        if _tgt is not None:
            _ax.axhline(_tgt, color="black", linestyle="--",
                        linewidth=0.9, alpha=0.6, label=_tgt_label)
        if _head == "dest":
            _ax.text(0.5, 0.97,
                     "target = 0.25·ln(N_valid)\n(N_valid not logged)",
                     transform=_ax.transAxes, ha="center", va="top",
                     fontsize=7, color="gray")
        _ax.set_title(f"ent_{_head}", fontsize=10)
        _ax.set_xlabel("Batch", fontsize=8)
        if _col_i == 0:
            _ax.set_ylabel("Entropy")
        _ax.set_ylim(bottom=0)
        _ax.grid(alpha=0.3)
        _ax.legend(fontsize=7, loc="best")

    # --- Panel 7: Per-opponent-type value estimates ---
    ax7 = fig.add_subplot(gs[5, 0])
    _val_series = [
        ("val_heuristic",  "vs Heuristic",          "#534AB7"),
        ("val_sp_mirror",  "vs Self (mirror)",       "#1D9E75"),
        ("val_sp_hof",     "vs Checkpoint (HoF)",    "#D85A30"),
        ("val_sp_ml",      "vs Checkpoint (HoF-ML)", "#3266AD"),
        ("val_sp_random",  "vs Checkpoint (random)",  "#888888"),
    ]
    _has_val = any(c in data for c, _, _ in _val_series)
    if _has_val:
        for col, label, color in _val_series:
            if col in data:
                bx, by = block_average(data, col, block_size)
                ax7.plot(bx, by, marker="o", markersize=2, linewidth=1.3,
                         color=color, label=label)
        ax7.legend(fontsize=7, ncol=2)
    else:
        ax7.text(0.5, 0.5, "val_* columns not in CSV\n(older log format)",
                 transform=ax7.transAxes, ha="center", va="center",
                 fontsize=10, color="gray")
    ax7.set_ylabel("Mean Value")
    ax7.set_title("Opponent-Conditioned Value Estimates")
    ax7.grid(alpha=0.3)

    # --- Panel 8: Planning distill losses ---
    ax8 = fig.add_subplot(gs[5, 1])
    _has_plan = "plan_distill_loss" in data
    if _has_plan:
        bx, by = block_average(data, "plan_distill_loss", block_size)
        ax8.plot(bx, by, marker="s", markersize=2, linewidth=1.3,
                 color="#C0392B", alpha=0.8, label="Distill loss (total)")
        # Sub-head breakdown (if available)
        _dl_colors = {"unit": "#E74C3C", "move": "#F39C12",
                      "charge": "#8E44AD", "shoot": "#27AE60",
                      "dest": "#1ABC9C"}
        for _dlk, _dlc in _dl_colors.items():
            _col = f"plan_dl_{_dlk}"
            if _col in data:
                _bx, _by = block_average(data, _col, block_size)
                ax8.plot(_bx, _by, linewidth=1.0, alpha=0.6,
                         color=_dlc, label=f"DL {_dlk}")
        ax8.set_ylabel("Distill Loss")
        ax8.legend(fontsize=7, ncol=2)
    else:
        ax8.text(0.5, 0.5, "plan_* columns not in CSV\n(older log format)",
                 transform=ax8.transAxes, ha="center", va="center",
                 fontsize=10, color="gray")
    ax8.set_title("Planning Distill Losses")
    ax8.grid(alpha=0.3)

    # --- Panel 8b: Plan improvement rate & mean V-delta ---
    # Twin-axis because the two series live on different scales:
    # improve_rate is a fraction in [0, 1]; mean_vdelta is a value-scale
    # quantity typically ~0.05–0.10. Plotted together because they jointly
    # characterize planner quality: rate = how often planning helps,
    # v-delta = by how much when it does.
    ax8c = fig.add_subplot(gs[6, :])
    _has_improve = "plan_improve_rate" in data
    if _has_improve:
        bx, by = block_average(data, "plan_improve_rate", block_size)
        ax8c.plot(bx, by, marker="o", markersize=2, linewidth=1.5,
                  color="#2E86C1", label="Improvement rate")
        ax8c.set_ylabel("Improvement Rate", color="#2E86C1")
        ax8c.set_ylim(-0.05, 1.05)
        ax8c.tick_params(axis="y", labelcolor="#2E86C1")
        ax8c.legend(fontsize=8, loc="upper left")
        if "plan_mean_vdelta" in data:
            ax8d = ax8c.twinx()
            bx2, by2 = block_average(data, "plan_mean_vdelta", block_size)
            ax8d.plot(bx2, by2, marker="s", markersize=2, linewidth=1.3,
                      color="#E67E22", alpha=0.85, label="Mean V-delta")
            ax8d.set_ylabel("Mean V-delta", color="#E67E22")
            ax8d.tick_params(axis="y", labelcolor="#E67E22")
            ax8d.legend(fontsize=8, loc="upper right")
    else:
        ax8c.text(0.5, 0.5,
                  "plan_improve_rate not in CSV\n(older log format)",
                  transform=ax8c.transAxes, ha="center", va="center",
                  fontsize=10, color="gray")
    ax8c.set_title("Plan Improvement Rate & Mean V-Delta")
    ax8c.set_xlabel("Batch")
    ax8c.grid(alpha=0.3)

    # --- Panel 9: A/B Side Symmetry ---
    ax9a = fig.add_subplot(gs[7, 0])
    _has_side_wr = "wr_side_a" in data and "wr_side_b" in data
    if _has_side_wr:
        bx, by = block_average(data, "wr_side_a", block_size)
        ax9a.plot(bx, by, marker="o", markersize=2, linewidth=1.5,
                  color="#2E86C1", label="A-side WR")
        bx, by = block_average(data, "wr_side_b", block_size)
        ax9a.plot(bx, by, marker="o", markersize=2, linewidth=1.5,
                  color="#E74C3C", label="B-side WR")
        ax9a.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax9a.legend(fontsize=8)
    else:
        ax9a.text(0.5, 0.5, "wr_side_* not in CSV\n(older log format)",
                  transform=ax9a.transAxes, ha="center", va="center",
                  fontsize=10, color="gray")
    ax9a.set_ylabel("Win Rate")
    ax9a.set_title("Win Rate by Physical Side (mirror self-play)")
    ax9a.set_ylim(0.2, 0.8)
    ax9a.grid(alpha=0.3)

    ax9b = fig.add_subplot(gs[7, 1])
    _has_side_val = "val_side_a" in data and "val_side_b" in data
    if _has_side_val:
        bx_a, by_a = block_average(data, "val_side_a", block_size)
        bx_b, by_b = block_average(data, "val_side_b", block_size)
        ax9b.plot(bx_a, by_a, marker="o", markersize=2, linewidth=1.5,
                  color="#2E86C1", label="V (A-side)")
        ax9b.plot(bx_b, by_b, marker="o", markersize=2, linewidth=1.5,
                  color="#E74C3C", label="V (B-side)")
        # Plot the gap (V_A + V_B, should be ~0)
        min_len = min(len(by_a), len(by_b))
        if min_len > 0:
            gap_x = bx_a[:min_len]
            gap_y = by_a[:min_len] + by_b[:min_len]
            ax9b.plot(gap_x, gap_y, linewidth=1.5, linestyle="--",
                      color="#27AE60", label="V_A + V_B (gap)")
        ax9b.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax9b.legend(fontsize=7)
    else:
        ax9b.text(0.5, 0.5, "val_side_* not in CSV\n(older log format)",
                  transform=ax9b.transAxes, ha="center", va="center",
                  fontsize=10, color="gray")
    ax9b.set_ylabel("Mean Value")
    ax9b.set_title("Value by Physical Side (mirror self-play)")
    ax9b.grid(alpha=0.3)

    # --- Panel 10: Shooting & Charge efficiency rewards ---
    # Primary comparison: ML vs heuristic from the SAME heuristic-opponent games
    # (ml_h_* columns). All-games ML metric (shoot/charge_eff_reward) shown
    # as faint dotted lines for reference.
    ax10_dest = fig.add_subplot(gs[8, :])
    _has_eff = False
    if "ml_h_shoot_eff" in data:
        bx, by = block_average(data, "ml_h_shoot_eff", block_size)
        ax10_dest.plot(bx, by, marker="o", markersize=2, linewidth=1.5, color="#E67E22", label="Shooting (ML)")
        _has_eff = True
    elif "shoot_eff_reward" in data:
        bx, by = block_average(data, "shoot_eff_reward", block_size)
        ax10_dest.plot(bx, by, marker="o", markersize=2, linewidth=1.5, color="#E67E22", label="Shooting (ML, all games)")
        _has_eff = True
    if "ml_h_charge_eff" in data:
        bx, by = block_average(data, "ml_h_charge_eff", block_size)
        ax10_dest.plot(bx, by, marker="o", markersize=2, linewidth=1.5, color="#8E44AD", label="Charge (ML)")
        _has_eff = True
    elif "charge_eff_reward" in data:
        bx, by = block_average(data, "charge_eff_reward", block_size)
        ax10_dest.plot(bx, by, marker="o", markersize=2, linewidth=1.5, color="#8E44AD", label="Charge (ML, all games)")
        _has_eff = True
    if "h_shoot_eff_reward" in data:
        bx, by = block_average(data, "h_shoot_eff_reward", block_size)
        ax10_dest.plot(bx, by, linewidth=1.5, linestyle="--", color="#E67E22", alpha=0.6, label="Shooting (heuristic)")
        _has_eff = True
    if "h_charge_eff_reward" in data:
        bx, by = block_average(data, "h_charge_eff_reward", block_size)
        ax10_dest.plot(bx, by, linewidth=1.5, linestyle="--", color="#8E44AD", alpha=0.6, label="Charge (heuristic)")
        _has_eff = True
    if _has_eff:
        ax10_dest.legend(fontsize=8)
    else:
        ax10_dest.text(0.5, 0.5, "shoot/charge_eff_reward not in CSV\n(older log format)",
                       transform=ax10_dest.transAxes, ha="center", va="center",
                       fontsize=10, color="gray")
    ax10_dest.set_ylabel("Mean Efficiency Reward")
    ax10_dest.set_title("Target Efficiency Rewards (expected pts of damage per episode, vs heuristic)")
    ax10_dest.set_ylim(bottom=0)
    ax10_dest.grid(alpha=0.3)

    # --- Panel 11: Per-head alpha coefficients (full width) ---
    ax10 = fig.add_subplot(gs[9, :])
    alpha_cols = [c for c in data if c.startswith("alpha_")]
    colors10 = plt.cm.Set2(np.linspace(0, 1, len(alpha_cols)))
    for col, color in zip(alpha_cols, colors10):
        head_name = col.replace("alpha_", "")
        bx, by = block_average(data, col, block_size)
        ax10.plot(bx, by, linewidth=1.5, color=color, label=head_name)
    ax10.set_ylabel("Alpha")
    ax10.set_xlabel("Batch")
    ax10.set_title("Adaptive Entropy Coefficients (per head)")
    ax10.legend(fontsize=7, ncol=2)
    ax10.grid(alpha=0.3)

    # --- Panel 12: Per-phase value head diagnostics (phase_reencode only) ---
    # Left: per-phase MSE against GAE returns; right: per-phase mean V output.
    # "Are the per-phase V heads learning?" — a yes looks like:
    #   • losses decreasing over time (heads finding a useful signal at all)
    #   • losses converging toward the main V loss (calibration)
    #   • mean outputs moving away from 0 (heads not stuck at init)
    #   • phases spreading (phase embedding differentiating decision stages)
    # A no looks like: loss stays flat at ~var(returns), mean output sticks
    # near 0 throughout, or all four phase curves overlap exactly.
    pp_phase_cols = [
        ("pp_v_loss_pre",  "pp_v_mean_pre",  "PRE_SELECT",    "#3266AD"),
        ("pp_v_loss_sel",  "pp_v_mean_sel",  "POST_SELECT",   "#1D9E75"),
        ("pp_v_loss_mt",   "pp_v_mean_mt",   "POST_MOVETYPE", "#E67E22"),
        ("pp_v_loss_dest", "pp_v_mean_dest", "POST_DEST",     "#C0392B"),
    ]
    _has_pp = any(lcol in data for lcol, _, _, _ in pp_phase_cols)

    ax_pp_loss = fig.add_subplot(gs[10, 0])
    if _has_pp:
        # Main value loss as a reference — per-phase V heads are trained
        # against the same target and *should* converge toward this curve.
        if "value_loss" in data:
            bx, by = block_average(data, "value_loss", block_size)
            ax_pp_loss.plot(bx, by, linewidth=1.8, color="black",
                             linestyle="--", alpha=0.5, label="Main V loss")
        for lcol, _, label, color in pp_phase_cols:
            if lcol in data:
                bx, by = block_average(data, lcol, block_size)
                ax_pp_loss.plot(bx, by, marker="o", markersize=2,
                                 linewidth=1.4, color=color, label=label)
        ax_pp_loss.legend(fontsize=7, ncol=2)
    else:
        ax_pp_loss.text(
            0.5, 0.5,
            "pp_v_* columns not in CSV\n"
            "(phase_reencode flag was off or older log format)",
            transform=ax_pp_loss.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
    ax_pp_loss.set_ylabel("MSE vs returns")
    ax_pp_loss.set_title("Per-Phase Value Head Loss")
    ax_pp_loss.set_xlabel("Batch")
    ax_pp_loss.set_ylim(bottom=0)
    ax_pp_loss.grid(alpha=0.3)

    ax_pp_mean = fig.add_subplot(gs[10, 1])
    if _has_pp:
        # mean_reward as a reference — the target the heads should approach.
        if "mean_reward" in data:
            bx, by = block_average(data, "mean_reward", block_size)
            ax_pp_mean.plot(bx, by, linewidth=1.8, color="black",
                             linestyle="--", alpha=0.5, label="Mean reward")
        ax_pp_mean.axhline(0, color="black", linewidth=0.8, alpha=0.3)
        for _, mcol, label, color in pp_phase_cols:
            if mcol in data:
                bx, by = block_average(data, mcol, block_size)
                ax_pp_mean.plot(bx, by, marker="o", markersize=2,
                                 linewidth=1.4, color=color, label=label)
        ax_pp_mean.legend(fontsize=7, ncol=2)
    else:
        ax_pp_mean.text(
            0.5, 0.5,
            "pp_v_* columns not in CSV\n"
            "(phase_reencode flag was off or older log format)",
            transform=ax_pp_mean.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
    ax_pp_mean.set_ylabel("Mean V output")
    ax_pp_mean.set_title("Per-Phase Value Head Mean Output")
    ax_pp_mean.set_xlabel("Batch")
    ax_pp_mean.grid(alpha=0.3)

    # --- Panel 13: Batch wall time (full width) ---
    # Derived from consecutive row timestamps; first batch of each session
    # is NaN so inter-session shutdown/restart gaps don't show up as spikes.
    ax_bt = fig.add_subplot(gs[11, :])
    bt = data.get("batch_time_derived")
    if bt is not None and np.isfinite(bt).any():
        bx, by = block_average(data, "batch_time_derived", block_size)
        ax_bt.plot(bx, by, marker="o", markersize=2, linewidth=1.5,
                   color="#16A085", label=f"Block mean ({block_size} batches)")
        finite = bt[np.isfinite(bt)]
        if finite.size:
            med = float(np.median(finite))
            ax_bt.axhline(med, color="black", linestyle="--", linewidth=0.8,
                          alpha=0.5, label=f"Overall median ({med:.1f}s)")
        ax_bt.legend(fontsize=8)
    else:
        ax_bt.text(0.5, 0.5, "no parseable timestamps",
                   transform=ax_bt.transAxes, ha="center", va="center",
                   fontsize=10, color="gray")
    ax_bt.set_ylabel("Seconds / batch")
    ax_bt.set_xlabel("Batch")
    ax_bt.set_title("Batch Wall Time (derived from timestamps, session gaps masked)")
    ax_bt.set_ylim(bottom=0)
    ax_bt.grid(alpha=0.3)

    # Enable minor gridlines on all panels for readability
    for ax in fig.get_axes():
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.12, linewidth=0.5)

    out_path = path.parent.parent / "training_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    # -- Scrollable display window --
    root = tk.Tk()
    root.title("Tactical Training Dashboard")
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{min(sw, 1700)}x{sh - 100}")

    outer = tk.Frame(root)
    outer.pack(fill=tk.BOTH, expand=True)

    tk_canvas = tk.Canvas(outer)
    vbar = tk.Scrollbar(outer, orient=tk.VERTICAL, command=tk_canvas.yview)
    tk_canvas.configure(yscrollcommand=vbar.set)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    tk_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    inner = tk.Frame(tk_canvas)
    tk_canvas.create_window((0, 0), window=inner, anchor=tk.NW)

    fig_canvas = FigureCanvasTkAgg(fig, master=inner)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack()

    inner.update_idletasks()
    tk_canvas.configure(scrollregion=tk_canvas.bbox("all"))

    def _scroll(event):
        if event.num == 4:
            tk_canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            tk_canvas.yview_scroll(3, "units")
    tk_canvas.bind_all("<Button-4>", _scroll)
    tk_canvas.bind_all("<Button-5>", _scroll)

    root.mainloop()


if __name__ == "__main__":
    main()
