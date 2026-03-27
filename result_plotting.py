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

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(
        f"Tactical Training Dashboard  (block avg = {block_size} batches)",
        fontsize=14,
        fontweight="bold",
    )
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.28)

    # --- Panel 1: Win rates vs heuristic ---
    ax1 = fig.add_subplot(gs[0, 0])
    for col, label, color in [
        ("h_hof_wr", "vs Heuristic HoF", "#534AB7"),
        ("h_ml_wr", "vs Heuristic ML", "#1D9E75"),
    ]:
        bx, by = block_average(data, col, block_size)
        ax1.plot(bx, by, marker="o", markersize=3, linewidth=1.5, color=color, label=label)
    ax1.set_ylabel("Win Rate")
    ax1.set_title("Win Rate vs Heuristic Opponents")
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)

    # --- Panel 2: Win rates self-play ---
    ax2 = fig.add_subplot(gs[0, 1])
    for col, label, color in [
        ("sp_hof_wr", "vs SP HoF", "#D85A30"),
        ("sp_ml_wr", "vs SP ML", "#3266AD"),
        ("sp_rnd_wr", "vs SP Random", "#888888"),
    ]:
        bx, by = block_average(data, col, block_size)
        ax2.plot(bx, by, marker="o", markersize=3, linewidth=1.5, color=color, label=label)
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax2.set_ylabel("Win Rate")
    ax2.set_title("Win Rate vs Self-Play Opponents")
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

    # --- Panel 7: Per-head alpha coefficients (full width) ---
    ax7 = fig.add_subplot(gs[3, :])
    alpha_cols = [c for c in data if c.startswith("alpha_")]
    colors7 = plt.cm.Set2(np.linspace(0, 1, len(alpha_cols)))
    for col, color in zip(alpha_cols, colors7):
        head_name = col.replace("alpha_", "")
        bx, by = block_average(data, col, block_size)
        ax7.plot(bx, by, linewidth=1.5, color=color, label=head_name)
    ax7.set_ylabel("Alpha")
    ax7.set_xlabel("Batch")
    ax7.set_title("Adaptive Entropy Coefficients (per head)")
    ax7.legend(fontsize=7, ncol=2)
    ax7.grid(alpha=0.3)

    out_path = path.parent.parent / "training_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
