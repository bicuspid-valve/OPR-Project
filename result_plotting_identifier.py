#!/usr/bin/env python3
"""
Plot identifier-head training results from the saved .pt payload.

Usage:
    python result_plotting_identifier.py [payload_path] [--no-val-preds]

Defaults:
    payload_path = ml_checkpoints/identifier_head.pt

The history panels (train/val MSE, Pearson r, decile spread, epoch time) are
always plotted. The diagnostic panels (predicted-vs-true scatter, residuals,
per-state Spearman, top-K agreement, Q vs log-pi gap) require re-running the
trained head on the held-out val set — pass --no-val-preds to skip if you
just want the history view (e.g. mid-training).
"""

import sys
import pathlib
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from scipy.stats import spearmanr

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Val recompute: load trunk + head, run forward over val candidates
# ---------------------------------------------------------------------------

def recompute_val_predictions(payload, batch_size: int = 512):
    """Re-run the trained head on the held-out val candidates.

    Returns dict with: preds, targets, log_pi, state_idx (per-cand), val_cand_idx.
    Uses CPU by default — val sets are small.
    """
    from ml_training.train_identifier import (
        IdentifierHead, load_chunks, precompute_h, _build_batch,
    )
    from ml_training.checkpoint import _make_model, load_model_state_dict

    print(f"[plot] loading chunks from {payload['data_dir']}")
    chunks, _ = load_chunks(payload["data_dir"])
    print(f"[plot]   {chunks.state_vec.shape[0]} states, "
          f"{chunks.cand_Q.shape[0]} candidates")

    # Recover val candidate indices from val_state_idx
    val_state_idx = payload["val_state_idx"].long()
    state_is_val = torch.zeros(chunks.state_vec.shape[0], dtype=torch.bool)
    state_is_val[val_state_idx] = True
    cand_is_val = state_is_val[chunks.cand_state_idx.long()]
    val_cand_idx = torch.where(cand_is_val)[0]
    print(f"[plot]   val: {len(val_state_idx)} states, "
          f"{len(val_cand_idx)} candidates")

    device = torch.device("cpu")

    # Trunk (frozen)
    print(f"[plot] loading trunk from {payload['checkpoint_used']}")
    trunk = _make_model("tactical")
    sd = load_model_state_dict(payload["checkpoint_used"])
    trunk.load_state_dict(sd, strict=False)
    trunk.eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    trunk.to(device)

    # Head
    head = IdentifierHead()
    head.load_state_dict(payload["model_state_dict"])
    head.eval()
    head.to(device)

    # Per-state h cache so the trunk only runs once per state
    print("[plot] precomputing h cache for val states")
    h_cache = precompute_h(trunk, chunks.state_vec, device).to(device)

    print(f"[plot] running head on {len(val_cand_idx)} val candidates")
    preds, targets = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(val_cand_idx), batch_size):
            batch_cand = val_cand_idx[i : i + batch_size]
            batch = _build_batch(chunks, h_cache, batch_cand, device)
            Q_pred = head(
                h=batch["h"], unit_feat=batch["unit_feat"],
                charge_feat=batch["charge_feat"], shoot_feat=batch["shoot_feat"],
                dest_feat=batch["dest_feat"], unit_idx=batch["unit_idx"],
                move_type=batch["move_type"], active_flags=batch["active_flags"],
            )
            preds.append(Q_pred.cpu())
            targets.append(batch["Q_target"].cpu())
    preds_np = torch.cat(preds, dim=0).numpy()
    targets_np = torch.cat(targets, dim=0).numpy()
    log_pi_np = chunks.cand_log_pi[val_cand_idx].numpy()
    state_idx_np = chunks.cand_state_idx[val_cand_idx].numpy()
    print(f"[plot]   done in {time.time() - t0:.1f}s")

    return dict(
        preds=preds_np, targets=targets_np,
        log_pi=log_pi_np, state_idx=state_idx_np,
        val_cand_idx=val_cand_idx.numpy(),
    )


# ---------------------------------------------------------------------------
# Per-state ranking metrics
# ---------------------------------------------------------------------------

def per_state_ranking(preds, targets, state_idx, k_top=10):
    """For each val state, compute Spearman ρ and top-k overlap between
    predicted Q ranking and true Q ranking. States with constant target
    (degenerate) are skipped."""
    rhos: list[float] = []
    top_k: list[float] = []
    top_q: list[float] = []  # top-quartile overlap
    target_stds: list[float] = []

    unique_states = np.unique(state_idx)
    for s in unique_states:
        mask = state_idx == s
        if mask.sum() < 4:
            continue
        t = targets[mask]
        p = preds[mask]
        if t.std() < 1e-6 or p.std() < 1e-6:
            continue
        rho, _ = spearmanr(t, p)
        if not np.isfinite(rho):
            continue
        rhos.append(float(rho))
        target_stds.append(float(t.std()))

        n = len(t)
        kt = min(k_top, n)
        true_top = set(np.argsort(-t)[:kt].tolist())
        pred_top = set(np.argsort(-p)[:kt].tolist())
        top_k.append(len(true_top & pred_top) / kt)

        kq = max(2, n // 4)
        true_q = set(np.argsort(-t)[:kq].tolist())
        pred_q = set(np.argsort(-p)[:kq].tolist())
        top_q.append(len(true_q & pred_q) / kq)

    return (np.array(rhos), np.array(top_k),
            np.array(top_q), np.array(target_stds))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    skip_preds = "--no-val-preds" in args
    args = [a for a in args if a != "--no-val-preds"]
    payload_path = args[0] if args else "ml_checkpoints/identifier_head.pt"

    path = Path(payload_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        print(f"Payload not found: {path}")
        sys.exit(1)

    print(f"[plot] loading payload from {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    history = payload.get("history", [])
    config = payload.get("config", {})
    n_states = payload.get("n_states", "?")
    n_cands = payload.get("n_candidates", "?")
    n_games = payload.get("n_games", "?")
    beta = payload.get("beta_calibrated", float("nan"))

    print(f"[plot]   {len(history)} epochs, "
          f"{n_states} states / {n_cands} cands / {n_games} games, "
          f"beta={beta:.4f}")

    val_preds = None
    if not skip_preds and "val_state_idx" in payload:
        try:
            val_preds = recompute_val_predictions(payload)
        except Exception as e:
            print(f"[plot] val-prediction step failed: {e}")
            print("[plot]   continuing without diagnostic panels")

    # ----- Build figure -----
    fig = plt.figure(figsize=(16, 38))
    fig.suptitle(
        "Identifier Head Training Dashboard"
        + (f"  (head: {Path(payload_path).name})" if payload_path else ""),
        fontsize=14, fontweight="bold",
    )
    gs = GridSpec(9, 2, figure=fig, hspace=0.45, wspace=0.28)

    # History as arrays
    if history:
        epochs = np.array([h["epoch"] for h in history])
        train_mse = np.array([h["train_mse"] for h in history])
        val_mse = np.array([h["val_mse"] for h in history])
        val_pear = np.array([h.get("val_pearson", float("nan")) for h in history])
        top_dec = np.array([h.get("val_top_decile_label", float("nan")) for h in history])
        bot_dec = np.array([h.get("val_bot_decile_label", float("nan")) for h in history])
        ep_t = np.array([h.get("seconds", float("nan")) for h in history])
        ws_rho_mean = np.array([h.get("ws_rho_mean", float("nan")) for h in history])
        ws_rho_med = np.array([h.get("ws_rho_median", float("nan")) for h in history])
        ws_top10 = np.array([h.get("ws_top10", float("nan")) for h in history])
        ws_top25 = np.array([h.get("ws_top25pct", float("nan")) for h in history])
        ws_n = np.array([h.get("ws_n_states", 0) for h in history])
        ws_top10_pred_q = np.array([h.get("ws_top10_pred_q", float("nan")) for h in history])
        ws_top10_oracle_q = np.array([h.get("ws_top10_oracle_q", float("nan")) for h in history])
        ws_bot10_pred_q = np.array([h.get("ws_bot10_pred_q", float("nan")) for h in history])
    else:
        epochs = train_mse = val_mse = val_pear = top_dec = bot_dec = ep_t = np.array([])
        ws_rho_mean = ws_rho_med = ws_top10 = ws_top25 = ws_n = np.array([])
        ws_top10_pred_q = ws_top10_oracle_q = ws_bot10_pred_q = np.array([])

    # --- Panel 1: Train vs Val MSE ---
    ax = fig.add_subplot(gs[0, 0])
    if epochs.size:
        ax.plot(epochs, train_mse, marker="o", markersize=3, linewidth=1.5,
                color="#534AB7", label="Train MSE")
        ax.plot(epochs, val_mse, marker="o", markersize=3, linewidth=1.5,
                color="#D85A30", label="Val MSE")
        ax.legend(fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Train / Val MSE per Epoch")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    # --- Panel 2: Val Pearson r ---
    ax = fig.add_subplot(gs[0, 1])
    if epochs.size:
        ax.plot(epochs, val_pear, marker="o", markersize=3, linewidth=1.5,
                color="#1D9E75", label="Val r")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.legend(fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pearson r")
    ax.set_title("Val Pearson Correlation (predicted Q vs label Q)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # --- Panel 3: Val top/bot decile label Q ---
    # Gap between these is the "ranking quality" signal: if the head ranks
    # candidates correctly, its top-10% predictions should land on candidates
    # with much higher actual Q than its bottom-10% predictions.
    ax = fig.add_subplot(gs[1, 0])
    if epochs.size:
        ax.plot(epochs, top_dec, marker="o", markersize=3, linewidth=1.5,
                color="#27AE60", label="Top-decile mean Q")
        ax.plot(epochs, bot_dec, marker="o", markersize=3, linewidth=1.5,
                color="#C0392B", label="Bot-decile mean Q")
        ax.fill_between(epochs, bot_dec, top_dec, color="gray", alpha=0.12)
        ax.legend(fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean label Q")
    ax.set_title("Val Decile Spread (head's top-10% vs bot-10% predictions)")
    ax.grid(alpha=0.3)

    # --- Panel 4: Per-epoch wall time ---
    ax = fig.add_subplot(gs[1, 1])
    if epochs.size and np.isfinite(ep_t).any():
        ax.plot(epochs, ep_t, marker="o", markersize=3, linewidth=1.5,
                color="#16A085")
        med = float(np.nanmedian(ep_t))
        ax.axhline(med, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.5, label=f"Median ({med:.1f}s)")
        ax.legend(fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.set_title("Per-Epoch Wall Time")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    # --- Panel 4a: Within-state Spearman ρ trajectory ---
    # The honest ranking metric — pooled Pearson r is dominated by
    # between-state mean-Q variance, but ws_ρ measures whether the head can
    # rank candidates *within a single state*, which is what a planner cares
    # about. Mean and median are both shown because the distribution is
    # often skewed by a few decided / low-spread states pulling ρ down.
    ax = fig.add_subplot(gs[2, 0])
    if epochs.size and np.isfinite(ws_rho_mean).any():
        ax.plot(epochs, ws_rho_mean, marker="o", markersize=3, linewidth=1.5,
                color="#1D9E75", label="Mean ws_ρ")
        ax.plot(epochs, ws_rho_med, marker="s", markersize=3, linewidth=1.3,
                color="#E67E22", label="Median ws_ρ", alpha=0.85)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.axhline(0.95, color="gray", linestyle=":", linewidth=0.8,
                   alpha=0.5, label="8-vs-100 rollout ceiling (~0.95)")
        ax.legend(fontsize=7)
        if len(ws_n) and ws_n[0] > 0:
            ax.text(0.02, 0.04, f"~{int(ws_n[-1])} usable val states",
                    transform=ax.transAxes, fontsize=7, va="bottom",
                    color="gray")
    else:
        ax.text(0.5, 0.5, "ws_ρ not in history\n(retrain to populate)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Within-state Spearman ρ")
    ax.set_title("Within-State Ranking ρ per Epoch")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # --- Panel 4b: Within-state top-K agreement trajectory ---
    ax = fig.add_subplot(gs[2, 1])
    if epochs.size and np.isfinite(ws_top10).any():
        ax.plot(epochs, ws_top10, marker="o", markersize=3, linewidth=1.5,
                color="#3266AD", label="Mean top-10 overlap")
        ax.plot(epochs, ws_top25, marker="s", markersize=3, linewidth=1.3,
                color="#8E44AD", label="Mean top-25% overlap", alpha=0.85)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "ws top-K not in history\n(retrain to populate)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Overlap fraction")
    ax.set_title("Within-State Top-K Overlap per Epoch")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)

    # --- Panel 4c: Top-10 / bot-10 mean true label Q (per-state) ---
    # "If the planner picked the head's top-10 in each state, how good are
    # those picks really?" Compare to the oracle (literal best-10 per state)
    # to see how close to the achievable ceiling we are; the bottom-10 line
    # confirms the head also separates clearly-bad candidates downward.
    ax = fig.add_subplot(gs[3, :])
    if epochs.size and np.isfinite(ws_top10_pred_q).any():
        ax.plot(epochs, ws_top10_oracle_q, marker="^", markersize=3,
                linewidth=1.3, color="#000000", linestyle="--", alpha=0.7,
                label="Oracle top-10 (ceiling)")
        ax.plot(epochs, ws_top10_pred_q, marker="o", markersize=3,
                linewidth=1.5, color="#1D9E75",
                label="Head's top-10 picks: mean true Q")
        ax.plot(epochs, ws_bot10_pred_q, marker="v", markersize=3,
                linewidth=1.5, color="#C0392B",
                label="Head's bot-10 picks: mean true Q")
        ax.axhline(0, color="black", linestyle=":", linewidth=0.7, alpha=0.4)
        ax.legend(fontsize=8)
        # Annotate the gap (oracle − pred) as the head's "regret" budget
        if np.isfinite(ws_top10_pred_q).any() and np.isfinite(ws_top10_oracle_q).any():
            gap = float(np.nanmean(ws_top10_oracle_q - ws_top10_pred_q))
            ax.text(0.02, 0.97,
                    f"avg gap (oracle − pred top-10) = {gap:+.3f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    else:
        ax.text(0.5, 0.5, "ws_top10_pred_q not in history\n(retrain to populate)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean true label Q")
    ax.set_title("Top-10 / Bot-10 Picks: Mean True Label Q (per-state, averaged)")
    ax.grid(alpha=0.3)

    # --- Panel 5: Predicted vs True scatter (val) ---
    ax = fig.add_subplot(gs[4, 0])
    if val_preds is not None:
        p, t = val_preds["preds"], val_preds["targets"]
        ax.hexbin(t, p, gridsize=50, cmap="viridis", mincnt=1)
        lo, hi = float(min(p.min(), t.min())), float(max(p.max(), t.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="y=x")
        if p.std() > 1e-8 and t.std() > 1e-8:
            r = float(np.corrcoef(p, t)[0, 1])
            mse = float(np.mean((p - t) ** 2))
            ax.text(0.02, 0.97, f"r = {r:+.3f}\nMSE = {mse:.4f}\nN = {len(p)}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "val predictions not computed\n(rerun without --no-val-preds)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    ax.set_xlabel("Label Q (true)")
    ax.set_ylabel("Predicted Q")
    ax.set_title("Predicted vs Label Q on Val Set")
    ax.grid(alpha=0.3)

    # --- Panel 6: Residual histogram (val) ---
    ax = fig.add_subplot(gs[4, 1])
    if val_preds is not None:
        resid = val_preds["preds"] - val_preds["targets"]
        ax.hist(resid, bins=60, color="#3266AD", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        m, s = float(resid.mean()), float(resid.std())
        ax.text(0.02, 0.97, f"mean = {m:+.4f}\nstd = {s:.4f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    else:
        ax.text(0.5, 0.5, "val predictions not computed",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    ax.set_xlabel("Predicted - Label")
    ax.set_ylabel("Count")
    ax.set_title("Val Residual Distribution")
    ax.grid(alpha=0.3)

    # --- Panel 7: Per-state Spearman ρ histogram ---
    # The within-state ranking metric — does the head order candidates the
    # same way the labels do, *within each state*? Mean-shift errors don't
    # hurt this metric, so it isolates ranking ability from calibration.
    ax = fig.add_subplot(gs[5, 0])
    if val_preds is not None:
        rhos, top_k, top_q, t_stds = per_state_ranking(
            val_preds["preds"], val_preds["targets"], val_preds["state_idx"])
        if len(rhos):
            ax.hist(rhos, bins=30, color="#1D9E75", alpha=0.85, edgecolor="white")
            ax.axvline(float(rhos.mean()), color="black", linestyle="--",
                       linewidth=1.0,
                       label=f"mean ρ = {rhos.mean():+.3f}")
            ax.legend(fontsize=8)
            ax.text(0.02, 0.97,
                    f"states = {len(rhos)}\n"
                    f"median ρ = {float(np.median(rhos)):+.3f}\n"
                    f"frac > 0.5 = {float((rhos > 0.5).mean()):.2f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        else:
            ax.text(0.5, 0.5, "no usable val states",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="gray")
    else:
        rhos = top_k = top_q = t_stds = np.array([])
        ax.text(0.5, 0.5, "val predictions not computed",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("Count")
    ax.set_title("Per-State Spearman ρ (within-state ranking quality)")
    ax.set_xlim(-1.05, 1.05)
    ax.grid(alpha=0.3)

    # --- Panel 8: Top-K agreement histogram ---
    ax = fig.add_subplot(gs[5, 1])
    if val_preds is not None and len(top_k):
        ax.hist([top_k, top_q], bins=20, range=(0, 1),
                color=["#3266AD", "#E67E22"], alpha=0.85,
                label=[f"Top-10 (mean {top_k.mean():.2f})",
                       f"Top-25%  (mean {top_q.mean():.2f})"],
                edgecolor="white")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "val predictions not computed",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    ax.set_xlabel("Overlap fraction")
    ax.set_ylabel("Count")
    ax.set_title("Per-State Top-K Overlap (predicted vs true)")
    ax.grid(alpha=0.3)

    # --- Panel 9: Q vs log-pi gap landscape ---
    # The whole point of the identifier: find actions where Q (rollout truth)
    # exceeds beta·log_pi (policy preference) — the "gap" candidates the
    # planner should consider. Color by predicted Q so we can see whether the
    # head has learned to flag those high-Q low-π candidates.
    ax = fig.add_subplot(gs[6, :])
    if val_preds is not None:
        t = val_preds["targets"]
        lp = val_preds["log_pi"]
        p = val_preds["preds"]
        sc = ax.scatter(lp, t, c=p, cmap="coolwarm", s=4, alpha=0.6)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Predicted Q")
        if np.isfinite(beta):
            # Calibration line: gap = 0 ⇔ Q = beta·log_pi  (constant offset
            # absorbed by the head). Plotted as guidance, not as fit.
            xs = np.linspace(lp.min(), lp.max(), 50)
            ax.plot(xs, beta * xs + (t.mean() - beta * lp.mean()),
                    "k--", linewidth=1.0, alpha=0.6,
                    label=f"β·log π + const (β={beta:.3f})")
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "val predictions not computed",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    ax.set_xlabel("log π(a|s) (frozen policy)")
    ax.set_ylabel("Label Q (rollout)")
    ax.set_title("Val Gap Landscape: label Q vs log π, colored by predicted Q")
    ax.grid(alpha=0.3)

    # --- Panel 10: Q label distribution (val) ---
    ax = fig.add_subplot(gs[7, 0])
    if val_preds is not None:
        ax.hist(val_preds["targets"], bins=50, color="#534AB7",
                alpha=0.7, edgecolor="white", label="Label Q")
        ax.hist(val_preds["preds"], bins=50, color="#D85A30",
                alpha=0.55, edgecolor="white", label="Predicted Q")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "val predictions not computed",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    ax.set_xlabel("Q value")
    ax.set_ylabel("Count")
    ax.set_title("Val Q Distribution (label vs predicted)")
    ax.grid(alpha=0.3)

    # --- Panel 11: Per-state target std distribution ---
    # Reveals how many val states are "decision-laden" (high std => candidates
    # actually differ in value) vs "decided" (low std => any action is fine).
    # The ranking metrics above are dominated by the high-std tail.
    ax = fig.add_subplot(gs[7, 1])
    if val_preds is not None and len(t_stds):
        ax.hist(t_stds, bins=30, color="#8E44AD", alpha=0.85, edgecolor="white")
        ax.axvline(0.05, color="red", linestyle="--", linewidth=0.8,
                   alpha=0.6, label="0.05 (rough decision-laden cutoff)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "val predictions not computed",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    ax.set_xlabel("std(label Q) within state")
    ax.set_ylabel("Count")
    ax.set_title("Per-State Label-Q Spread (separability of candidates)")
    ax.grid(alpha=0.3)

    # --- Panel 12: Config / summary ---
    ax = fig.add_subplot(gs[8, :])
    ax.axis("off")
    lines = [
        "─── Run Summary ─────────────────────────────────────",
        f"Payload:           {payload_path}",
        f"Source checkpoint: {payload.get('checkpoint_used', '?')}",
        f"Data dir:          {payload.get('data_dir', '?')}",
        f"States/cands/games: {n_states} / {n_cands} / {n_games}",
        f"Beta (calibrated): {beta:.5f}",
        "",
        "─── Config ──────────────────────────────────────────",
    ]
    for k, v in config.items():
        lines.append(f"  {k}: {v}")
    if "val_game_uids" in payload:
        lines.append(f"  val_game_uids saved: {len(payload['val_game_uids'])} games")
    if "val_state_idx" in payload:
        lines.append(f"  val_state_idx saved: {len(payload['val_state_idx'])} states")
    if val_preds is None:
        lines.append("")
        lines.append("(diagnostic panels skipped — pass without --no-val-preds to compute)")
    ax.text(0.0, 1.0, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, family="monospace")

    for ax in fig.get_axes():
        ax.minorticks_on()
        ax.grid(which="minor", alpha=0.12, linewidth=0.5)

    out_path = path.parent.parent / "identifier_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    # ----- Scrollable Tk display -----
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"[plot] no display, skipping Tk window: {e}")
        return
    root.title("Identifier Head Training Dashboard")
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
