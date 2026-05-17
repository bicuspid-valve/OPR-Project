"""Phase 2: supervised training of the gap-identifier scalar Q(s, a) head.

Loads:
    - Frozen TacticalModel from --checkpoint (default ml_checkpoints/final_model.pt).
      Only the trunk is used; gradients are disabled on it. The hard convention
      is that this is the *same* checkpoint used to label the dataset — the
      manifest written by identifier_dataset.py records which one.
    - Sharded labeled dataset from --data-dir (output of identifier_dataset.py).

Trains:
    - IdentifierHead. Frozen trunk -> h cached once per unique state. Head
      consumes (h, raw entity slices for entities the action operates on,
      dest features, action discrete embeddings, head-active flags) -> tanh
      -> scalar Q in [-1, 1]. MSE loss against the rollout-based Q_proxy
      label.

Output:
    - --out path (default ml_checkpoints/identifier_head.pt) containing the
      head's state_dict plus metadata (checkpoint hash, train/val MSE curves,
      a calibrated beta = std(Q) / std(log pi) on the labeled set).

Diagnostics each epoch:
    - train MSE / val MSE
    - Pearson correlation of predicted Q against label Q on validation
    - top-decile-by-predicted-Q mean label Q vs bottom-decile (a coarse
      ranking-quality signal — a true ranking metric belongs in Phase 3)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_features import (
    DEST_FEATURE_DIM,
    MAX_UNITS_PER_SIDE,
    TACTICAL_UNIT_FEATURES,
)
from ml_model_tactical import (
    IdentifierHead,
    TacticalModel,
    _gather_unit_features,
)
from ml_training.checkpoint import _make_model, load_model_state_dict


DEFAULT_DATA_DIR = "ml_training/identifier_data"
DEFAULT_OUT_PATH = "ml_checkpoints/identifier_head.pt"
DEFAULT_CHECKPOINT = "ml_checkpoints/final_model.pt"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass
class _LoadedChunks:
    """All training data, fully materialized in memory.

    Per-state arrays are aligned by state_id; candidate arrays are aligned by
    candidate_id, with cand_state_idx[c] giving the state_id for candidate c.
    """
    # Per-state (S,)
    state_vec: torch.Tensor      # (S, 4016) float32
    alive_mask: torch.Tensor     # (S, 10) bool (currently unused at training)
    enemy_alive_mask: torch.Tensor  # (S, 10) bool (currently unused)
    round_num: torch.Tensor      # (S,) int32
    player_is_a: torch.Tensor    # (S,) bool
    game_uid: torch.Tensor       # (S,) int64

    # Per-candidate (C,)
    cand_state_idx: torch.Tensor       # (C,) int64
    cand_unit_idx: torch.Tensor        # (C,) int64
    cand_move_type: torch.Tensor       # (C,) int64
    cand_charge_target: torch.Tensor   # (C,) int64
    cand_shoot_target: torch.Tensor    # (C,) int64
    cand_advance_reachable: torch.Tensor  # (C,) bool
    cand_dest_active: torch.Tensor     # (C,) bool
    cand_charge_active: torch.Tensor   # (C,) bool
    cand_shoot_active: torch.Tensor    # (C,) bool
    cand_dest_features: torch.Tensor   # (C, 76) float32
    cand_log_pi: torch.Tensor          # (C,) float32
    cand_Q: torch.Tensor               # (C,) float32 — raw rollout Q label
    # (S,) per-state baseline subtracted from cand_Q to form the regression
    # target — Q label minus baseline = advantage. Populated post-load with
    # V_trunk(s) (the frozen trunk's value-head output) by the training
    # pipeline; load_chunks initializes to zeros as a placeholder.
    per_state_baseline: torch.Tensor


def load_chunks(data_dir: str) -> tuple[_LoadedChunks, dict]:
    """Concatenate every chunk_*.npz in data_dir into one big in-memory
    dataset. Returns (chunks, manifest)."""
    d = Path(data_dir)
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {data_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    state_vecs: list[np.ndarray] = []
    alive_masks: list[np.ndarray] = []
    enemy_alive_masks: list[np.ndarray] = []
    round_nums: list[np.ndarray] = []
    player_is_a: list[np.ndarray] = []
    game_uids: list[np.ndarray] = []

    cand_state_idx: list[np.ndarray] = []
    cand_unit_idx: list[np.ndarray] = []
    cand_move_type: list[np.ndarray] = []
    cand_charge: list[np.ndarray] = []
    cand_shoot: list[np.ndarray] = []
    cand_adv: list[np.ndarray] = []
    cand_dest_active: list[np.ndarray] = []
    cand_charge_active: list[np.ndarray] = []
    cand_shoot_active: list[np.ndarray] = []
    cand_dest_features: list[np.ndarray] = []
    cand_log_pi: list[np.ndarray] = []
    cand_Q: list[np.ndarray] = []

    state_offset = 0
    for entry in manifest["chunks"]:
        path = d / entry["path"]
        z = np.load(path)
        n_s = int(z["state_vec"].shape[0])

        state_vecs.append(z["state_vec"])
        alive_masks.append(z["alive_mask"])
        enemy_alive_masks.append(z["enemy_alive_mask"])
        round_nums.append(z["round_num"])
        player_is_a.append(z["player_is_a"])
        if "game_uid" in z.files:
            game_uids.append(z["game_uid"])
        else:
            # Old chunk without game_uid: each state is its own "game" — falls
            # back to state-level train/val split (correlated states leak).
            game_uids.append(np.arange(n_s, dtype=np.int64))

        # Shift cand_state_idx by the cumulative state offset
        cand_state_idx.append(z["cand_state_idx"].astype(np.int64) + state_offset)
        cand_unit_idx.append(z["cand_unit_idx"])
        cand_move_type.append(z["cand_move_type"])
        cand_charge.append(z["cand_charge_target_idx"])
        cand_shoot.append(z["cand_shoot_target_idx"])
        cand_adv.append(z["cand_advance_reachable"])
        cand_dest_active.append(z["cand_dest_active"])
        cand_charge_active.append(z["cand_charge_active"])
        cand_shoot_active.append(z["cand_shoot_active"])
        cand_dest_features.append(z["cand_dest_features"])
        cand_log_pi.append(z["cand_log_pi"])
        cand_Q.append(z["cand_Q"])

        state_offset += n_s

    state_vec_np = np.concatenate(state_vecs, axis=0)
    cand_state_idx_np = np.concatenate(cand_state_idx, axis=0).astype(np.int64)
    cand_Q_np = np.concatenate(cand_Q, axis=0)
    n_states_total = state_vec_np.shape[0]

    return _LoadedChunks(
        state_vec=torch.from_numpy(state_vec_np),
        alive_mask=torch.from_numpy(np.concatenate(alive_masks, axis=0)),
        enemy_alive_mask=torch.from_numpy(np.concatenate(enemy_alive_masks, axis=0)),
        round_num=torch.from_numpy(np.concatenate(round_nums, axis=0)),
        player_is_a=torch.from_numpy(np.concatenate(player_is_a, axis=0)),
        game_uid=torch.from_numpy(np.concatenate(game_uids, axis=0)),
        cand_state_idx=torch.from_numpy(cand_state_idx_np),
        cand_unit_idx=torch.from_numpy(np.concatenate(cand_unit_idx, axis=0).astype(np.int64)),
        cand_move_type=torch.from_numpy(np.concatenate(cand_move_type, axis=0).astype(np.int64)),
        cand_charge_target=torch.from_numpy(np.concatenate(cand_charge, axis=0).astype(np.int64)),
        cand_shoot_target=torch.from_numpy(np.concatenate(cand_shoot, axis=0).astype(np.int64)),
        cand_advance_reachable=torch.from_numpy(np.concatenate(cand_adv, axis=0)),
        cand_dest_active=torch.from_numpy(np.concatenate(cand_dest_active, axis=0)),
        cand_charge_active=torch.from_numpy(np.concatenate(cand_charge_active, axis=0)),
        cand_shoot_active=torch.from_numpy(np.concatenate(cand_shoot_active, axis=0)),
        cand_dest_features=torch.from_numpy(np.concatenate(cand_dest_features, axis=0)),
        cand_log_pi=torch.from_numpy(np.concatenate(cand_log_pi, axis=0)),
        cand_Q=torch.from_numpy(cand_Q_np),
        # Filled in by run_training_pipeline once the trunk is loaded.
        per_state_baseline=torch.zeros(n_states_total, dtype=torch.float32),
    ), manifest


# ---------------------------------------------------------------------------
# Trunk caching — one trunk forward per unique state, cached for all epochs
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_h(
    trunk_model: TacticalModel,
    state_vec: torch.Tensor,        # (S, 4016)
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:                  # (S, 512)
    """Run the frozen trunk on every state once; return cached h tensor."""
    trunk_model.eval()
    out: list[torch.Tensor] = []
    S = state_vec.shape[0]
    for i in range(0, S, batch_size):
        batch = state_vec[i : i + batch_size].to(device)
        h, _units, _round_onehot = trunk_model.trunk(batch)
        out.append(h.cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def precompute_V(
    trunk_model: TacticalModel,
    state_vec: torch.Tensor,        # (S, 4016)
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:                  # (S,)
    """Run the trunk's value head on every state to get V_trunk(s).

    Used as the per-state baseline for centering the Q regression target so
    the head learns advantage A(s,a) = Q(s,a) - V_trunk(s) — i.e. how much
    this action improves over the policy's expected continuation, rather than
    over the average random candidate. Mirrors the convention the value head
    uses in eval/planning forward: opponent_type=None and side=None, so both
    embeddings fall back to the mean over their respective categories.
    """
    trunk_model.eval()
    out: list[torch.Tensor] = []
    S = state_vec.shape[0]
    for i in range(0, S, batch_size):
        batch = state_vec[i : i + batch_size].to(device)
        h, _units, round_onehot = trunk_model.trunk(batch)
        opp_embed = trunk_model._get_opp_embed(h, None)
        side_embed = trunk_model._get_side_embed(h, None)
        V = trunk_model.value_head(h, round_onehot, opp_embed, side_embed)
        out.append(V.cpu())
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Train/val split (episode-level, by game_uid)
# ---------------------------------------------------------------------------

def split_train_val(
    chunks: _LoadedChunks,
    val_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (train_cand_idx, val_cand_idx). Splits by game_uid so all
    candidates from a given game stay together; keeps temporally-correlated
    states from leaking across the split."""
    rng = np.random.default_rng(seed)
    state_uid = chunks.game_uid.numpy()
    unique_uids = np.unique(state_uid)
    rng.shuffle(unique_uids)
    n_val = max(1, int(round(val_frac * len(unique_uids))))
    val_uids = set(unique_uids[:n_val].tolist())

    state_is_val = np.array(
        [u in val_uids for u in state_uid], dtype=np.bool_,
    )
    cand_state = chunks.cand_state_idx.numpy()
    cand_is_val = state_is_val[cand_state]

    train_idx = np.where(~cand_is_val)[0]
    val_idx = np.where(cand_is_val)[0]
    return torch.from_numpy(train_idx), torch.from_numpy(val_idx)


# ---------------------------------------------------------------------------
# Per-batch feature gather
# ---------------------------------------------------------------------------

def _build_batch(
    chunks: _LoadedChunks,
    h_cache: torch.Tensor,           # (S, 512) on device
    cand_idx: torch.Tensor,          # (B,) candidate indices into the dataset
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Materialize one minibatch: trunk output + raw entity slices + scalars."""
    cand_idx = cand_idx.to(device)
    state_idx = chunks.cand_state_idx.to(device).index_select(0, cand_idx)
    state_vec = chunks.state_vec.to(device).index_select(0, state_idx)

    h = h_cache.index_select(0, state_idx)

    unit_idx = chunks.cand_unit_idx.to(device).index_select(0, cand_idx)
    move_type = chunks.cand_move_type.to(device).index_select(0, cand_idx)
    charge_idx = chunks.cand_charge_target.to(device).index_select(0, cand_idx)
    shoot_idx = chunks.cand_shoot_target.to(device).index_select(0, cand_idx)
    adv = chunks.cand_advance_reachable.to(device).index_select(0, cand_idx).float()
    dest_active = chunks.cand_dest_active.to(device).index_select(0, cand_idx).float()
    charge_active = chunks.cand_charge_active.to(device).index_select(0, cand_idx).float()
    shoot_active = chunks.cand_shoot_active.to(device).index_select(0, cand_idx).float()
    dest_feat = chunks.cand_dest_features.to(device).index_select(0, cand_idx)
    log_pi = chunks.cand_log_pi.to(device).index_select(0, cand_idx)
    # Centered target: Q minus V_trunk(s). The head learns advantage
    # A(s,a) = Q(s,a) - V_trunk(s) — how much this action improves over the
    # policy's expected continuation, not over the average random candidate.
    Q_raw = chunks.cand_Q.to(device).index_select(0, cand_idx)
    state_baseline = chunks.per_state_baseline.to(device).index_select(0, state_idx)
    Q = Q_raw - state_baseline

    # Friendly slot 0..9 for selected unit; enemy slots are 10..19, so the
    # gather indices for charge/shoot targets are 10 + target_idx.
    unit_feat = _gather_unit_features(state_vec, unit_idx)
    charge_feat = _gather_unit_features(state_vec, charge_idx + MAX_UNITS_PER_SIDE)
    shoot_feat = _gather_unit_features(state_vec, shoot_idx + MAX_UNITS_PER_SIDE)
    # Zero out the slices that aren't active for this action so the head
    # doesn't have to learn that "active=0 means ignore the 200 numbers"
    charge_feat = charge_feat * charge_active.unsqueeze(-1)
    shoot_feat = shoot_feat * shoot_active.unsqueeze(-1)
    dest_feat = dest_feat * dest_active.unsqueeze(-1)

    active_flags = torch.stack([charge_active, shoot_active, dest_active,
                                move_type.float()], dim=-1)

    return dict(
        h=h, unit_feat=unit_feat,
        charge_feat=charge_feat, shoot_feat=shoot_feat,
        dest_feat=dest_feat,
        unit_idx=unit_idx, move_type=move_type,
        active_flags=active_flags,
        Q_target=Q, log_pi=log_pi,
    )


# ---------------------------------------------------------------------------
# Ranking loss — relevance-weighted pairwise logistic
# ---------------------------------------------------------------------------

def weighted_pairwise_loss(
    scores: torch.Tensor, targets: torch.Tensor,
) -> torch.Tensor:
    """Pairwise logistic ranking loss, weighted by max(0, max(t_i, t_j)).

    For each state (row of (M, K) inputs), compute all K² pairwise terms.
    Each pair (i, j) gets weight = max(0, max(t_i, t_j)) — so pairs of two
    bad candidates contribute zero gradient (we don't care about ordering
    among mediocre actions), while pairs involving at least one good
    candidate get full pressure (these are the orderings that matter for
    "is this action genuinely promising").

    Per-state normalization, then mean across states, so a single state with
    many good candidates can't dominate the batch loss.
    """
    M, K = scores.shape
    s_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)        # (M, K, K)
    t_diff = targets.unsqueeze(-1) - targets.unsqueeze(-2)
    sign_t = torch.sign(t_diff)
    # softplus(-sign(t)*(s_i-s_j)) = log(1 + exp(-sign*(s_i-s_j))) — pairwise logistic
    pair_loss = F.softplus(-sign_t * s_diff)

    # Relevance weight: only pairs where at least one cand has advantage > 0
    pair_weight = torch.clamp(
        torch.maximum(targets.unsqueeze(-1), targets.unsqueeze(-2)),
        min=0.0,
    )
    # Mask self-pairs (i == j) — both s_diff and t_diff are 0 there, garbage
    eye = torch.eye(K, device=scores.device, dtype=torch.bool)
    pair_weight = pair_weight.masked_fill(eye, 0.0)

    # Per-state weighted mean, then mean across states
    per_state_num = (pair_loss * pair_weight).sum(dim=(-1, -2))
    per_state_den = pair_weight.sum(dim=(-1, -2)) + 1e-8
    per_state_loss = per_state_num / per_state_den
    return per_state_loss.mean()


def _build_state_to_cands(
    chunks: _LoadedChunks, train_idx: torch.Tensor, k_per_state: int,
) -> list[np.ndarray]:
    """Group train candidates by their state. Returns a list of arrays, one
    per eligible state (states with >= k_per_state candidates), each holding
    the train-set candidate indices belonging to that state."""
    train_idx_np = train_idx.numpy()
    state_per_cand = chunks.cand_state_idx.numpy()[train_idx_np]
    order = np.argsort(state_per_cand, kind="stable")
    sorted_states = state_per_cand[order]
    sorted_cands = train_idx_np[order]
    # Find run-length boundaries
    boundaries = np.concatenate([
        [0],
        np.where(np.diff(sorted_states) != 0)[0] + 1,
        [len(sorted_states)],
    ])
    out: list[np.ndarray] = []
    for i in range(len(boundaries) - 1):
        run = sorted_cands[boundaries[i] : boundaries[i + 1]]
        if len(run) >= k_per_state:
            out.append(run)
    return out


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

K_PER_STATE = 16  # candidates per state in each ranking-loss group
MSE_AUX_WEIGHT = 0.3  # weight on the calibration MSE term — keeps unbounded
                      # head outputs anchored to advantage units


def train(
    head: IdentifierHead,
    chunks: _LoadedChunks,
    h_cache: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict:
    head.to(device)
    # AdamW for cleaner weight-decay semantics (decoupled from gradient).
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    rng = np.random.default_rng(seed)
    history: list[dict] = []

    # Build per-state candidate index once. Each train batch is M states ×
    # K candidates, so the ranking loss has well-formed within-state groups.
    state_groups = _build_state_to_cands(chunks, train_idx, K_PER_STATE)
    states_per_batch = max(1, batch_size // K_PER_STATE)
    print(f"[train] ranking-loss batching: "
          f"{states_per_batch} states × {K_PER_STATE} cands/state per batch, "
          f"{len(state_groups)} eligible train states "
          f"(loss = weighted_pairwise + {MSE_AUX_WEIGHT}·mse, "
          f"AdamW wd={weight_decay})")

    for epoch in range(epochs):
        # ---- Train ----
        head.train()
        perm = rng.permutation(len(state_groups))
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_rank_sum = 0.0
        train_n_states = 0
        t_epoch = time.time()
        for s_start in range(0, len(perm), states_per_batch):
            batch_state_ids = perm[s_start : s_start + states_per_batch]
            if len(batch_state_ids) < 2:
                # ListMLE on a single state still works, but a single-state
                # batch is too noisy for stable updates. Skip the trailing
                # partial batch.
                continue

            # Sample K candidates from each state (without replacement)
            batch_cand_arrays = []
            for sid in batch_state_ids:
                pool = state_groups[sid]
                chosen = rng.choice(pool, size=K_PER_STATE, replace=False)
                batch_cand_arrays.append(chosen)
            batch_cand = torch.from_numpy(
                np.concatenate(batch_cand_arrays).astype(np.int64)
            )
            M = len(batch_state_ids)
            K = K_PER_STATE

            batch = _build_batch(chunks, h_cache, batch_cand, device)
            optim.zero_grad(set_to_none=True)
            Q_pred = head(
                h=batch["h"], unit_feat=batch["unit_feat"],
                charge_feat=batch["charge_feat"], shoot_feat=batch["shoot_feat"],
                dest_feat=batch["dest_feat"], unit_idx=batch["unit_idx"],
                move_type=batch["move_type"], active_flags=batch["active_flags"],
            )
            Q_target = batch["Q_target"]

            scores_mk = Q_pred.view(M, K)
            targets_mk = Q_target.view(M, K)
            rank_loss = weighted_pairwise_loss(scores_mk, targets_mk)
            mse = F.mse_loss(Q_pred, Q_target)
            loss = rank_loss + MSE_AUX_WEIGHT * mse

            loss.backward()
            optim.step()
            train_loss_sum += float(loss.item()) * M
            train_rank_sum += float(rank_loss.item()) * M
            train_mse_sum += float(mse.item()) * M
            train_n_states += M

        train_mse = train_mse_sum / max(1, train_n_states)
        train_rank = train_rank_sum / max(1, train_n_states)
        train_total = train_loss_sum / max(1, train_n_states)

        # ---- Validate ----
        head.eval()
        val_preds: list[torch.Tensor] = []
        val_targets: list[torch.Tensor] = []
        # Per-cand state_idx for within-state ranking metrics. Pulled once,
        # not per-batch, since val_idx ordering is preserved through the loop.
        val_state_idx_np = chunks.cand_state_idx[val_idx].numpy()
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_cand = val_idx[i : i + batch_size]
                batch = _build_batch(chunks, h_cache, batch_cand, device)
                Q_pred = head(
                    h=batch["h"], unit_feat=batch["unit_feat"],
                    charge_feat=batch["charge_feat"], shoot_feat=batch["shoot_feat"],
                    dest_feat=batch["dest_feat"], unit_idx=batch["unit_idx"],
                    move_type=batch["move_type"], active_flags=batch["active_flags"],
                )
                val_preds.append(Q_pred.cpu())
                val_targets.append(batch["Q_target"].cpu())

        val_preds_t = torch.cat(val_preds, dim=0)
        val_targets_t = torch.cat(val_targets, dim=0)
        val_mse = F.mse_loss(val_preds_t, val_targets_t).item()

        # Diagnostic: pooled Pearson r + top/bottom-decile mean
        if val_preds_t.numel() >= 10:
            vp = val_preds_t.numpy()
            vt = val_targets_t.numpy()
            if vp.std() > 1e-8 and vt.std() > 1e-8:
                pearson = float(np.corrcoef(vp, vt)[0, 1])
            else:
                pearson = float("nan")
            order = np.argsort(vp)
            n_dec = max(1, len(vp) // 10)
            bot_mean = float(vt[order[:n_dec]].mean())
            top_mean = float(vt[order[-n_dec:]].mean())
        else:
            pearson, bot_mean, top_mean = float("nan"), float("nan"), float("nan")

        # Within-state ranking diagnostics: pooled Pearson is dominated by
        # between-state mean-Q variance, so it overstates the head's actual
        # ranking ability. Per-state Spearman ρ and top-K overlap measure the
        # only thing that matters for downstream planning — within a single
        # state, can the head pick the best candidates?
        ws_rhos: list[float] = []
        ws_top10: list[float] = []
        ws_top25pct: list[float] = []
        # Mean of true label Q at the head's top-10 / bottom-10 picks per
        # state, plus the oracle reference (mean of true label Q at the
        # actual best/worst 10 per state). Tells us how good the head's
        # top picks really are vs the best achievable.
        ws_top10_pred_q: list[float] = []
        ws_top10_oracle_q: list[float] = []
        ws_bot10_pred_q: list[float] = []
        if val_preds_t.numel() >= 4:
            unique_states = np.unique(val_state_idx_np)
            for s in unique_states:
                mask = val_state_idx_np == s
                if mask.sum() < 4:
                    continue
                p_s = vp[mask]
                t_s = vt[mask]
                # Skip degenerate states (decided games, all-equal Q)
                if t_s.std() < 1e-6 or p_s.std() < 1e-6:
                    continue
                rho, _ = spearmanr(t_s, p_s)
                if not np.isfinite(rho):
                    continue
                ws_rhos.append(float(rho))
                k10 = min(10, len(t_s))
                pred_top10_idx = np.argsort(-p_s)[:k10]
                pred_bot10_idx = np.argsort(p_s)[:k10]
                true_top10_idx = np.argsort(-t_s)[:k10]
                ws_top10.append(
                    len(set(true_top10_idx.tolist())
                        & set(pred_top10_idx.tolist())) / k10
                )
                ws_top10_pred_q.append(float(t_s[pred_top10_idx].mean()))
                ws_top10_oracle_q.append(float(t_s[true_top10_idx].mean()))
                ws_bot10_pred_q.append(float(t_s[pred_bot10_idx].mean()))
                kq = max(2, len(t_s) // 4)
                t_q = set(np.argsort(-t_s)[:kq].tolist())
                p_q = set(np.argsort(-p_s)[:kq].tolist())
                ws_top25pct.append(len(t_q & p_q) / kq)

        ws_rho_mean = float(np.mean(ws_rhos)) if ws_rhos else float("nan")
        ws_rho_med = float(np.median(ws_rhos)) if ws_rhos else float("nan")
        ws_top10_mean = float(np.mean(ws_top10)) if ws_top10 else float("nan")
        ws_top25pct_mean = float(np.mean(ws_top25pct)) if ws_top25pct else float("nan")
        ws_top10_pred_q_mean = (float(np.mean(ws_top10_pred_q))
                                if ws_top10_pred_q else float("nan"))
        ws_top10_oracle_q_mean = (float(np.mean(ws_top10_oracle_q))
                                  if ws_top10_oracle_q else float("nan"))
        ws_bot10_pred_q_mean = (float(np.mean(ws_bot10_pred_q))
                                if ws_bot10_pred_q else float("nan"))

        epoch_time = time.time() - t_epoch
        history.append(dict(
            epoch=epoch,
            train_mse=train_mse, train_rank=train_rank, train_total=train_total,
            val_mse=val_mse,
            val_pearson=pearson,
            val_top_decile_label=top_mean,
            val_bot_decile_label=bot_mean,
            ws_rho_mean=ws_rho_mean,
            ws_rho_median=ws_rho_med,
            ws_top10=ws_top10_mean,
            ws_top25pct=ws_top25pct_mean,
            ws_top10_pred_q=ws_top10_pred_q_mean,
            ws_top10_oracle_q=ws_top10_oracle_q_mean,
            ws_bot10_pred_q=ws_bot10_pred_q_mean,
            ws_n_states=len(ws_rhos),
            seconds=epoch_time,
        ))
        print(f"epoch {epoch:3d}  rank={train_rank:.4f} mse={train_mse:.5f}  "
              f"val_mse={val_mse:.5f}  val_r={pearson:+.3f}  "
              f"ws_ρ={ws_rho_mean:+.3f} (n={len(ws_rhos)})  "
              f"top10/top25%={ws_top10_mean:.2f}/{ws_top25pct_mean:.2f}  "
              f"top10 label Q (pred/oracle/bot)="
              f"{ws_top10_pred_q_mean:+.3f}/{ws_top10_oracle_q_mean:+.3f}/"
              f"{ws_bot10_pred_q_mean:+.3f}  "
              f"top10%/bot10% (flat) ="
              f"{top_mean:+.3f}/{bot_mean:+.3f}  "
              f"({epoch_time:.1f}s)")

    return {"history": history}


# ---------------------------------------------------------------------------
# Beta calibration
# ---------------------------------------------------------------------------

def calibrate_beta(chunks: _LoadedChunks, train_idx: torch.Tensor) -> float:
    """beta = std(advantage) / std(log pi) on the training set, where
    advantage = Q - V_trunk(s). Same transformation the loss uses, so beta
    lives on the head's output scale."""
    state_idx = chunks.cand_state_idx[train_idx].long().numpy()
    Q = chunks.cand_Q[train_idx].numpy() - chunks.per_state_baseline.numpy()[state_idx]
    lp = chunks.cand_log_pi[train_idx].numpy()
    sQ, sP = float(np.std(Q)), float(np.std(lp))
    return sQ / max(sP, 1e-6)


# ---------------------------------------------------------------------------
# Programmatic entry point — callable from main.py or other scripts
# ---------------------------------------------------------------------------

def run_training_pipeline(
    data_dir: str = DEFAULT_DATA_DIR,
    checkpoint: str = DEFAULT_CHECKPOINT,
    out_path: str = DEFAULT_OUT_PATH,
    *,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    val_frac: float = 0.1,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    """Load dataset + frozen trunk, train the head, save it, return the
    saved payload (for inspection / chaining).

    Refuses to run if the dataset's manifest checkpoint doesn't match
    `checkpoint`: log pi labels and the trunk h must come from the same pi.
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"[train] device: {dev}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    print(f"[train] loading dataset from {data_dir}")
    chunks, manifest = load_chunks(data_dir)
    n_states = chunks.state_vec.shape[0]
    n_cands = chunks.cand_Q.shape[0]
    n_games = int(chunks.game_uid.unique().numel())
    print(f"[train]   {n_states} states, {n_cands} candidates, "
          f"{n_games} unique games")

    if manifest.get("checkpoint") and manifest["checkpoint"] != checkpoint:
        raise RuntimeError(
            f"checkpoint mismatch: dataset labeled with {manifest['checkpoint']}, "
            f"but training requested {checkpoint}. The frozen trunk and "
            f"the labeled log pi values must come from the same model."
        )

    # Load frozen trunk
    print(f"[train] loading frozen trunk from {checkpoint}")
    trunk_model = _make_model("tactical")
    trunk_model.load_state_dict(load_model_state_dict(checkpoint), strict=False)
    trunk_model.to(dev)
    trunk_model.eval()
    for p in trunk_model.parameters():
        p.requires_grad_(False)

    # Precompute h cache
    print(f"[train] precomputing h cache for {n_states} states")
    h_cache = precompute_h(trunk_model, chunks.state_vec, dev).to(dev)
    print(f"[train]   h_cache shape: {tuple(h_cache.shape)}")

    # Precompute V_trunk(s) per state — used as the per-state baseline so
    # the head learns advantage = Q - V_trunk(s) instead of raw Q.
    print(f"[train] precomputing V_trunk baseline for {n_states} states")
    chunks.per_state_baseline = precompute_V(trunk_model, chunks.state_vec, dev)
    print(f"[train]   V_trunk: mean={float(chunks.per_state_baseline.mean()):+.3f}, "
          f"std={float(chunks.per_state_baseline.std()):.3f}, "
          f"range=[{float(chunks.per_state_baseline.min()):+.3f}, "
          f"{float(chunks.per_state_baseline.max()):+.3f}]")

    # Train/val split by game_uid
    train_idx, val_idx = split_train_val(chunks, val_frac, seed)
    val_state_idx = torch.unique(chunks.cand_state_idx[val_idx])
    val_game_uids = torch.unique(chunks.game_uid[val_state_idx])
    print(f"[train] train candidates: {len(train_idx)}, "
          f"val candidates: {len(val_idx)} "
          f"({len(val_state_idx)} states, {len(val_game_uids)} games)")
    if len(val_idx) == 0:
        raise RuntimeError(
            "validation set is empty — too few games for the requested val_frac. "
            "Lower val_frac or generate more states."
        )

    beta = calibrate_beta(chunks, train_idx)
    print(f"[train] calibrated beta (train-set std ratio): {beta:.4f}")

    head = IdentifierHead()
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[train] IdentifierHead params: {n_params}")

    out = train(
        head, chunks, h_cache, train_idx, val_idx, dev,
        epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, seed=seed,
    )

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": head.state_dict(),
        "checkpoint_used": checkpoint,
        "data_dir": data_dir,
        "n_states": n_states,
        "n_candidates": n_cands,
        "n_games": n_games,
        "beta_calibrated": beta,
        "history": out["history"],
        "config": dict(
            epochs=epochs, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay,
            val_frac=val_frac, seed=seed,
        ),
        # The actual held-out set as int64 tensors. game_uids are portable
        # (chunk-load-order independent) and can filter any future eval; the
        # state idx list is the literal "states the trainer never saw".
        "val_game_uids": val_game_uids.cpu(),
        "val_state_idx": val_state_idx.cpu(),
        # Saved so an inference-time consumer can de-center predictions back
        # into raw-Q space if needed (head outputs advantage = Q - V_trunk).
        # The identifier downstream gap = head_out - β·log_π is invariant to
        # the per-state offset, so most callers won't need this.
        "per_state_baseline": chunks.per_state_baseline.cpu(),
        "baseline_kind": "V_trunk",
        "target_is_centered": True,
    }
    torch.save(payload, out_path_p)
    print(f"[train] saved head to {out_path_p}")
    return payload


# ---------------------------------------------------------------------------
# Trunk fine-tuning: jointly train (h-producing trunk layers + identifier head)
# while keeping the original trunk checkpoint untouched.
# ---------------------------------------------------------------------------

DEFAULT_FT_TRUNK_OUT = "ml_checkpoints/final_model_id_finetuned.pt"
DEFAULT_FT_HEAD_OUT = "ml_checkpoints/identifier_head_finetuned.pt"

# Modules in TacticalModel that produce h. These are the only trunk params
# we unfreeze during fine-tuning. Everything else (policy heads, value head,
# CTDE embeddings, phase machinery) stays frozen — we only want h to adapt
# for action-conditioned Q prediction.
_FT_TRAINABLE_TRUNK_MODULES = ("unit_encoder", "stem", "core_block")


def _set_trunk_finetune_mode(trunk: TacticalModel) -> list[torch.nn.Parameter]:
    """Freeze all trunk params, then unfreeze the h-producing layers.
    Returns the list of trainable parameters."""
    for p in trunk.parameters():
        p.requires_grad_(False)
    trainable: list[torch.nn.Parameter] = []
    for name in _FT_TRAINABLE_TRUNK_MODULES:
        module = getattr(trunk, name)
        for p in module.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def _trunk_h_for_states(
    trunk: TacticalModel, state_vec_batch: torch.Tensor,
) -> torch.Tensor:
    """Run trunk forward on a batch of state vectors, return h.
    Gradients flow through the unfrozen h-producing layers only."""
    h, _units, _round_onehot = trunk.trunk(state_vec_batch)
    return h


def _build_batch_with_trunk(
    chunks: _LoadedChunks,
    trunk: TacticalModel,
    cand_idx: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Like _build_batch but computes h on-the-fly from the (fine-tunable)
    trunk instead of looking it up in a precomputed cache. Runs the trunk
    only on the *unique* states in this batch (typically M states for M*K
    candidates), then expands h to per-candidate shape via index_select.
    """
    cand_idx = cand_idx.to(device)
    state_idx = chunks.cand_state_idx.to(device).index_select(0, cand_idx)
    unique_states, inverse = torch.unique(state_idx, return_inverse=True)
    state_vec_unique = chunks.state_vec.to(device).index_select(0, unique_states)
    h_unique = _trunk_h_for_states(trunk, state_vec_unique)
    h = h_unique.index_select(0, inverse)

    state_vec = chunks.state_vec.to(device).index_select(0, state_idx)

    unit_idx = chunks.cand_unit_idx.to(device).index_select(0, cand_idx)
    move_type = chunks.cand_move_type.to(device).index_select(0, cand_idx)
    charge_idx = chunks.cand_charge_target.to(device).index_select(0, cand_idx)
    shoot_idx = chunks.cand_shoot_target.to(device).index_select(0, cand_idx)
    adv = chunks.cand_advance_reachable.to(device).index_select(0, cand_idx).float()
    dest_active = chunks.cand_dest_active.to(device).index_select(0, cand_idx).float()
    charge_active = chunks.cand_charge_active.to(device).index_select(0, cand_idx).float()
    shoot_active = chunks.cand_shoot_active.to(device).index_select(0, cand_idx).float()
    dest_feat = chunks.cand_dest_features.to(device).index_select(0, cand_idx)
    log_pi = chunks.cand_log_pi.to(device).index_select(0, cand_idx)
    Q_raw = chunks.cand_Q.to(device).index_select(0, cand_idx)
    state_baseline = chunks.per_state_baseline.to(device).index_select(0, state_idx)
    Q = Q_raw - state_baseline

    unit_feat = _gather_unit_features(state_vec, unit_idx)
    charge_feat = _gather_unit_features(state_vec, charge_idx + MAX_UNITS_PER_SIDE)
    shoot_feat = _gather_unit_features(state_vec, shoot_idx + MAX_UNITS_PER_SIDE)
    charge_feat = charge_feat * charge_active.unsqueeze(-1)
    shoot_feat = shoot_feat * shoot_active.unsqueeze(-1)
    dest_feat = dest_feat * dest_active.unsqueeze(-1)
    active_flags = torch.stack([charge_active, shoot_active, dest_active,
                                move_type.float()], dim=-1)
    return dict(
        h=h, unit_feat=unit_feat,
        charge_feat=charge_feat, shoot_feat=shoot_feat,
        dest_feat=dest_feat,
        unit_idx=unit_idx, move_type=move_type,
        active_flags=active_flags,
        Q_target=Q, log_pi=log_pi,
    )


def train_jointly(
    head: IdentifierHead,
    trunk_ft: TacticalModel,
    chunks: _LoadedChunks,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    lr_head: float,
    lr_trunk: float,
    weight_decay: float,
    seed: int,
) -> dict:
    """Joint training loop for (h-producing trunk layers + identifier head).
    Same loss and metrics as `train()`, but runs trunk forward in-loop so
    gradients flow back into the unfrozen trunk modules."""
    head.to(device)
    trunk_ft.to(device)

    trainable_trunk = [p for p in trunk_ft.parameters() if p.requires_grad]
    n_trunk_params = sum(p.numel() for p in trainable_trunk)
    n_head_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[ft] trainable params: trunk={n_trunk_params}, head={n_head_params}")

    optim = torch.optim.AdamW([
        {"params": head.parameters(), "lr": lr_head},
        {"params": trainable_trunk, "lr": lr_trunk},
    ], weight_decay=weight_decay)

    rng = np.random.default_rng(seed)
    history: list[dict] = []
    state_groups = _build_state_to_cands(chunks, train_idx, K_PER_STATE)
    states_per_batch = max(1, batch_size // K_PER_STATE)
    print(f"[ft] joint training: {states_per_batch} states × {K_PER_STATE} "
          f"cands/state, lr_head={lr_head} lr_trunk={lr_trunk}, "
          f"weight_decay={weight_decay}")

    for epoch in range(epochs):
        head.train()
        trunk_ft.train()
        # Re-freeze the non-h-producing modules every epoch in case .train()
        # flipped any BatchNorm/Dropout in frozen subtrees (defensive).
        for name, module in trunk_ft.named_children():
            if name not in _FT_TRAINABLE_TRUNK_MODULES:
                module.eval()

        perm = rng.permutation(len(state_groups))
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_rank_sum = 0.0
        train_n_states = 0
        t_epoch = time.time()
        for s_start in range(0, len(perm), states_per_batch):
            batch_state_ids = perm[s_start : s_start + states_per_batch]
            if len(batch_state_ids) < 2:
                continue
            batch_cand_arrays = []
            for sid in batch_state_ids:
                pool = state_groups[sid]
                chosen = rng.choice(pool, size=K_PER_STATE, replace=False)
                batch_cand_arrays.append(chosen)
            batch_cand = torch.from_numpy(
                np.concatenate(batch_cand_arrays).astype(np.int64)
            )
            M = len(batch_state_ids)
            K = K_PER_STATE

            batch = _build_batch_with_trunk(chunks, trunk_ft, batch_cand, device)
            optim.zero_grad(set_to_none=True)
            Q_pred = head(
                h=batch["h"], unit_feat=batch["unit_feat"],
                charge_feat=batch["charge_feat"], shoot_feat=batch["shoot_feat"],
                dest_feat=batch["dest_feat"], unit_idx=batch["unit_idx"],
                move_type=batch["move_type"], active_flags=batch["active_flags"],
            )
            Q_target = batch["Q_target"]
            scores_mk = Q_pred.view(M, K)
            targets_mk = Q_target.view(M, K)
            rank_loss = weighted_pairwise_loss(scores_mk, targets_mk)
            mse = F.mse_loss(Q_pred, Q_target)
            loss = rank_loss + MSE_AUX_WEIGHT * mse
            loss.backward()
            optim.step()
            train_loss_sum += float(loss.item()) * M
            train_rank_sum += float(rank_loss.item()) * M
            train_mse_sum += float(mse.item()) * M
            train_n_states += M

        train_mse = train_mse_sum / max(1, train_n_states)
        train_rank = train_rank_sum / max(1, train_n_states)
        train_total = train_loss_sum / max(1, train_n_states)

        # Validation: run trunk and head with grad off.
        head.eval()
        trunk_ft.eval()
        val_preds: list[torch.Tensor] = []
        val_targets: list[torch.Tensor] = []
        val_state_idx_np = chunks.cand_state_idx[val_idx].numpy()
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_cand = val_idx[i : i + batch_size]
                batch = _build_batch_with_trunk(chunks, trunk_ft, batch_cand, device)
                Q_pred = head(
                    h=batch["h"], unit_feat=batch["unit_feat"],
                    charge_feat=batch["charge_feat"], shoot_feat=batch["shoot_feat"],
                    dest_feat=batch["dest_feat"], unit_idx=batch["unit_idx"],
                    move_type=batch["move_type"], active_flags=batch["active_flags"],
                )
                val_preds.append(Q_pred.cpu())
                val_targets.append(batch["Q_target"].cpu())
        val_preds_t = torch.cat(val_preds, dim=0)
        val_targets_t = torch.cat(val_targets, dim=0)
        val_mse = F.mse_loss(val_preds_t, val_targets_t).item()

        if val_preds_t.numel() >= 10:
            vp = val_preds_t.numpy()
            vt = val_targets_t.numpy()
            if vp.std() > 1e-8 and vt.std() > 1e-8:
                pearson = float(np.corrcoef(vp, vt)[0, 1])
            else:
                pearson = float("nan")
            order = np.argsort(vp)
            n_dec = max(1, len(vp) // 10)
            bot_mean = float(vt[order[:n_dec]].mean())
            top_mean = float(vt[order[-n_dec:]].mean())
        else:
            pearson, bot_mean, top_mean = float("nan"), float("nan"), float("nan")

        ws_rhos: list[float] = []
        ws_top10: list[float] = []
        ws_top25pct: list[float] = []
        ws_top10_pred_q: list[float] = []
        ws_top10_oracle_q: list[float] = []
        ws_bot10_pred_q: list[float] = []
        if val_preds_t.numel() >= 4:
            unique_states = np.unique(val_state_idx_np)
            for s in unique_states:
                mask = val_state_idx_np == s
                if mask.sum() < 4:
                    continue
                p_s = vp[mask]
                t_s = vt[mask]
                if t_s.std() < 1e-6 or p_s.std() < 1e-6:
                    continue
                rho, _ = spearmanr(t_s, p_s)
                if not np.isfinite(rho):
                    continue
                ws_rhos.append(float(rho))
                k10 = min(10, len(t_s))
                pred_top10_idx = np.argsort(-p_s)[:k10]
                pred_bot10_idx = np.argsort(p_s)[:k10]
                true_top10_idx = np.argsort(-t_s)[:k10]
                ws_top10.append(
                    len(set(true_top10_idx.tolist()) & set(pred_top10_idx.tolist())) / k10
                )
                ws_top10_pred_q.append(float(t_s[pred_top10_idx].mean()))
                ws_top10_oracle_q.append(float(t_s[true_top10_idx].mean()))
                ws_bot10_pred_q.append(float(t_s[pred_bot10_idx].mean()))
                kq = max(2, len(t_s) // 4)
                t_q = set(np.argsort(-t_s)[:kq].tolist())
                p_q = set(np.argsort(-p_s)[:kq].tolist())
                ws_top25pct.append(len(t_q & p_q) / kq)

        ws_rho_mean = float(np.mean(ws_rhos)) if ws_rhos else float("nan")
        ws_rho_med = float(np.median(ws_rhos)) if ws_rhos else float("nan")
        ws_top10_mean = float(np.mean(ws_top10)) if ws_top10 else float("nan")
        ws_top25pct_mean = float(np.mean(ws_top25pct)) if ws_top25pct else float("nan")
        ws_top10_pred_q_mean = (float(np.mean(ws_top10_pred_q))
                                if ws_top10_pred_q else float("nan"))
        ws_top10_oracle_q_mean = (float(np.mean(ws_top10_oracle_q))
                                  if ws_top10_oracle_q else float("nan"))
        ws_bot10_pred_q_mean = (float(np.mean(ws_bot10_pred_q))
                                if ws_bot10_pred_q else float("nan"))

        epoch_time = time.time() - t_epoch
        history.append(dict(
            epoch=epoch,
            train_mse=train_mse, train_rank=train_rank, train_total=train_total,
            val_mse=val_mse,
            val_pearson=pearson,
            val_top_decile_label=top_mean,
            val_bot_decile_label=bot_mean,
            ws_rho_mean=ws_rho_mean,
            ws_rho_median=ws_rho_med,
            ws_top10=ws_top10_mean,
            ws_top25pct=ws_top25pct_mean,
            ws_top10_pred_q=ws_top10_pred_q_mean,
            ws_top10_oracle_q=ws_top10_oracle_q_mean,
            ws_bot10_pred_q=ws_bot10_pred_q_mean,
            ws_n_states=len(ws_rhos),
            seconds=epoch_time,
        ))
        print(f"[ft] ep {epoch:3d}  rank={train_rank:.4f} mse={train_mse:.5f}  "
              f"val_mse={val_mse:.5f}  val_r={pearson:+.3f}  "
              f"ws_ρ={ws_rho_mean:+.3f} (n={len(ws_rhos)})  "
              f"top10/top25%={ws_top10_mean:.2f}/{ws_top25pct_mean:.2f}  "
              f"top10 (pred/oracle/bot)="
              f"{ws_top10_pred_q_mean:+.3f}/{ws_top10_oracle_q_mean:+.3f}/"
              f"{ws_bot10_pred_q_mean:+.3f}  "
              f"({epoch_time:.1f}s)")

    return {"history": history}


def run_finetuning_pipeline(
    data_dir: str = DEFAULT_DATA_DIR,
    trunk_in: str = DEFAULT_CHECKPOINT,
    head_in: str = DEFAULT_OUT_PATH,
    trunk_out: str = DEFAULT_FT_TRUNK_OUT,
    head_out: str = DEFAULT_FT_HEAD_OUT,
    *,
    epochs: int = 20,
    batch_size: int = 512,
    lr_head: float = 3e-4,
    lr_trunk: float = 1e-5,
    weight_decay: float = 1e-3,
    val_frac: float = 0.1,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    """Joint fine-tuning pipeline. Loads the original trunk and head, makes
    a deep copy of the trunk, freezes its policy/value heads, unfreezes only
    the h-producing layers, then jointly trains (trunk_h_layers + head)
    against the same advantage targets used in the head-only pipeline.

    The original trunk file (`trunk_in`, default final_model.pt) is never
    written to. Outputs go to `trunk_out` and `head_out` — both default to
    distinct *_finetuned.pt paths under ml_checkpoints/.
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"[ft] device: {dev}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[ft] loading dataset from {data_dir}")
    chunks, manifest = load_chunks(data_dir)
    n_states = chunks.state_vec.shape[0]
    n_cands = chunks.cand_Q.shape[0]
    n_games = int(chunks.game_uid.unique().numel())
    print(f"[ft]   {n_states} states, {n_cands} candidates, {n_games} games")
    if manifest.get("checkpoint") and manifest["checkpoint"] != trunk_in:
        print(f"[ft] WARNING: manifest checkpoint ({manifest['checkpoint']}) "
              f"differs from trunk_in ({trunk_in}); proceeding anyway")

    # Original trunk — used ONLY to compute V_trunk(s) baseline. Never moves
    # off disk-state, never trained, the file is never overwritten.
    print(f"[ft] loading original (frozen) trunk from {trunk_in}")
    trunk_orig = _make_model("tactical")
    trunk_orig.load_state_dict(load_model_state_dict(trunk_in), strict=False)
    trunk_orig.eval()
    for p in trunk_orig.parameters():
        p.requires_grad_(False)
    trunk_orig.to(dev)

    # Compute the V_trunk baseline ONCE using the original (frozen) trunk so
    # the regression target stays stationary while the fine-tuned trunk's h
    # adapts to predict it.
    print(f"[ft] precomputing V_trunk baseline from frozen trunk")
    chunks.per_state_baseline = precompute_V(trunk_orig, chunks.state_vec, dev)
    print(f"[ft]   V_trunk: mean={float(chunks.per_state_baseline.mean()):+.3f}, "
          f"std={float(chunks.per_state_baseline.std()):.3f}")

    # Free the original trunk from device memory — done with it.
    del trunk_orig
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Deep-copy the trunk for fine-tuning. We need a fresh copy of the
    # weights: load_state_dict from disk again to avoid any in-place mutation
    # affecting the on-disk file's identity in our minds.
    print(f"[ft] loading fine-tuning trunk copy from {trunk_in}")
    trunk_ft = _make_model("tactical")
    trunk_ft.load_state_dict(load_model_state_dict(trunk_in), strict=False)
    trainable_trunk = _set_trunk_finetune_mode(trunk_ft)
    print(f"[ft]   unfrozen trunk modules: "
          f"{', '.join(_FT_TRAINABLE_TRUNK_MODULES)} "
          f"({sum(p.numel() for p in trainable_trunk):,} params)")

    # Load existing head (initialized from prior frozen-trunk training).
    print(f"[ft] loading head from {head_in}")
    head = IdentifierHead()
    head_payload = torch.load(head_in, map_location="cpu", weights_only=False)
    head.load_state_dict(head_payload["model_state_dict"])

    # Train/val split — same logic as head-only pipeline.
    train_idx, val_idx = split_train_val(chunks, val_frac, seed)
    val_state_idx = torch.unique(chunks.cand_state_idx[val_idx])
    val_game_uids = torch.unique(chunks.game_uid[val_state_idx])
    print(f"[ft] train candidates: {len(train_idx)}, "
          f"val candidates: {len(val_idx)} "
          f"({len(val_state_idx)} states, {len(val_game_uids)} games)")
    if len(val_idx) == 0:
        raise RuntimeError("validation set is empty — too few games for val_frac")

    out = train_jointly(
        head, trunk_ft, chunks, train_idx, val_idx, dev,
        epochs=epochs, batch_size=batch_size,
        lr_head=lr_head, lr_trunk=lr_trunk,
        weight_decay=weight_decay, seed=seed,
    )

    # Recalibrate beta on the (now post-fine-tuning) trained set's advantage.
    beta = calibrate_beta(chunks, train_idx)
    print(f"[ft] beta (recalibrated): {beta:.4f}")

    # Save fine-tuned trunk to a NEW path (original trunk_in untouched).
    trunk_out_p = Path(trunk_out)
    trunk_out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": trunk_ft.state_dict(),
        "source_checkpoint": trunk_in,
        "finetune_modules": list(_FT_TRAINABLE_TRUNK_MODULES),
        "finetune_lr_trunk": lr_trunk,
        "finetune_epochs": epochs,
    }, trunk_out_p)
    print(f"[ft] saved fine-tuned trunk to {trunk_out_p}")

    # Save fine-tuned head to a NEW path, with checkpoint_used pointing at
    # the new trunk so downstream consumers know they go together.
    head_out_p = Path(head_out)
    head_out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": head.state_dict(),
        "checkpoint_used": str(trunk_out_p),
        "trunk_source": trunk_in,
        "data_dir": data_dir,
        "n_states": n_states,
        "n_candidates": n_cands,
        "n_games": n_games,
        "beta_calibrated": beta,
        "history": out["history"],
        "config": dict(
            epochs=epochs, batch_size=batch_size,
            lr_head=lr_head, lr_trunk=lr_trunk,
            weight_decay=weight_decay,
            val_frac=val_frac, seed=seed,
            mode="joint_finetune",
        ),
        "val_game_uids": val_game_uids.cpu(),
        "val_state_idx": val_state_idx.cpu(),
        "per_state_baseline": chunks.per_state_baseline.cpu(),
        "baseline_kind": "V_trunk_original",
        "target_is_centered": True,
    }
    torch.save(payload, head_out_p)
    print(f"[ft] saved fine-tuned head to {head_out_p}")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: train the gap-identifier head.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Frozen TacticalModel checkpoint (trunk source). "
                             "Must match the manifest's checkpoint.")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH,
                        help="Output path for the trained head (.pt)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of games (not candidates) held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    run_training_pipeline(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
