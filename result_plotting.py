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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_csv(path: str) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        columns = {h: [] for h in headers}
        for row in reader:
            # Skip non-data rows (e.g. "Training started" marker)
            try:
                float(row["batch"])
            except (ValueError, TypeError):
                continue
            for h in headers:
                try:
                    columns[h].append(float(row[h]))
                except (ValueError, TypeError):
                    columns[h].append(float("nan"))
    # Convert to arrays and sort by batch
    for h in headers:
        columns[h] = np.array(columns[h])
    order = np.argsort(columns["batch"])
    for h in headers:
        columns[h] = columns[h][order]
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

    fig = plt.figure(figsize=(16, 22))
    fig.suptitle(
        f"Tactical Training Dashboard  (block avg = {block_size} batches)",
        fontsize=14,
        fontweight="bold",
    )
    gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.28)

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

    # --- Panel 6: Entropy (aggregate + per-head) ---
    ax6 = fig.add_subplot(gs[2, 1])
    bx, by = block_average(data, "mean_entropy", block_size)
    ax6.plot(bx, by, marker="o", markersize=3, linewidth=2.0, color="#8E44AD", label="Total")
    ent_cols = [c for c in data if c.startswith("ent_")]
    if ent_cols:
        ent_colors = plt.cm.Set2(np.linspace(0, 1, len(ent_cols)))
        for col, color in zip(ent_cols, ent_colors):
            head_name = col.replace("ent_", "")
            bx, by = block_average(data, col, block_size)
            ax6.plot(bx, by, linewidth=1.2, color=color, alpha=0.7, label=head_name)
        ax6.legend(fontsize=7, ncol=3)
    ax6.set_ylabel("Entropy")
    ax6.set_title("Policy Entropy (per head)")
    ax6.grid(alpha=0.3)

    # --- Panel 7: Per-opponent-type value estimates ---
    ax7 = fig.add_subplot(gs[3, 0])
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

    # --- Panel 8: Planning metrics ---
    ax8 = fig.add_subplot(gs[3, 1])
    _has_plan = "plan_improve_rate" in data
    if _has_plan:
        bx, by = block_average(data, "plan_improve_rate", block_size)
        ax8.plot(bx, by, marker="o", markersize=2, linewidth=1.5,
                 color="#2E86C1", label="Improvement rate")
        ax8.set_ylabel("Improvement Rate", color="#2E86C1")
        ax8.set_ylim(-0.05, 1.05)
        ax8.legend(fontsize=8, loc="upper left")

        if "plan_distill_loss" in data:
            ax8b = ax8.twinx()
            bx2, by2 = block_average(data, "plan_distill_loss", block_size)
            ax8b.plot(bx2, by2, marker="s", markersize=2, linewidth=1.3,
                      color="#C0392B", alpha=0.8, label="Distill loss (total)")
            # Sub-head breakdown (if available)
            _dl_colors = {"unit": "#E74C3C", "move": "#F39C12",
                          "charge": "#8E44AD", "shoot": "#27AE60"}
            for _dlk, _dlc in _dl_colors.items():
                _col = f"plan_dl_{_dlk}"
                if _col in data:
                    _bx, _by = block_average(data, _col, block_size)
                    ax8b.plot(_bx, _by, linewidth=1.0, alpha=0.6,
                              color=_dlc, label=f"DL {_dlk}")
            ax8b.set_ylabel("Distill Loss", color="#C0392B")
            ax8b.legend(fontsize=7, loc="upper right")
    else:
        ax8.text(0.5, 0.5, "plan_* columns not in CSV\n(older log format)",
                 transform=ax8.transAxes, ha="center", va="center",
                 fontsize=10, color="gray")
    ax8.set_title("Planning Metrics")
    ax8.grid(alpha=0.3)

    # --- Panel 9: Per-head alpha coefficients (full width) ---
    ax9 = fig.add_subplot(gs[4, :])
    alpha_cols = [c for c in data if c.startswith("alpha_")]
    colors9 = plt.cm.Set2(np.linspace(0, 1, len(alpha_cols)))
    for col, color in zip(alpha_cols, colors9):
        head_name = col.replace("alpha_", "")
        bx, by = block_average(data, col, block_size)
        ax9.plot(bx, by, linewidth=1.5, color=color, label=head_name)
    ax9.set_ylabel("Alpha")
    ax9.set_xlabel("Batch")
    ax9.set_title("Adaptive Entropy Coefficients (per head)")
    ax9.legend(fontsize=7, ncol=2)
    ax9.grid(alpha=0.3)

    out_path = path.parent.parent / "training_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
