"""Flat (vectorized) replay, PPO loss, auxiliary losses, and planning distillation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_features import (
    MAX_UNITS_PER_SIDE, TACTICAL_UNIT_FEATURES,
    extract_can_charge_mask, extract_is_shaken,
    MAX_DEST_CANDIDATES, DEST_FEATURE_DIM,
)
from ml_integration_tactical import compute_destination_features, is_phase_reencode_enabled
from ml_model_tactical import (
    TacticalModel, NUM_MOVE_TYPES, MOVE_MOVE, MOVE_CHARGE,
    NUM_OPPONENT_TYPES,
    PHASE_PRE_SELECT, PHASE_POST_SELECT, PHASE_POST_MOVETYPE, PHASE_POST_DEST,
    N_PHASES,
)

# Phase-reencode auxiliary per-phase value loss weight (Step 5). Per-phase V
# heads are trained as auxiliary regressors against the same GAE activation
# return as the main value head. Small weight keeps the main V loss dominant
# while giving per-phase heads gradient signal so they become usable as
# per-phase baselines in a future Step 5b.
_PER_PHASE_V_WEIGHT = 0.1

# One-shot diagnostic: prints phase-reencode flag state on the first replay
# call of a training run so we can confirm the expected branch is taken.
_FIRST_REPLAY_FLAG_DIAG_PRINTED = False

from ml_training.config import TacticalActivationRecord
from ml_training.entropy import EntropyTargetTuner


# ---------------------------------------------------------------------------
# Flat replay result
# ---------------------------------------------------------------------------

@dataclass
class FlatReplayResult:
    """Flat tensor output from batched tactical replay.

    Keeps everything as contiguous tensors so compute_loss_flat can operate
    without any Python-level per-step iteration.
    """
    log_probs: torch.Tensor    # (N,) — sum of log-probs across 5 heads
    entropies: torch.Tensor    # (N,) — mean entropy across active heads (for logging)
    values: torch.Tensor       # (N,) — value estimates
    n_episodes: int
    total_reward: float        # pre-computed sum of all rewards
    total_shoot_eff_reward: float = 0.0  # shooting efficiency shaping component (for logging)
    total_charge_eff_reward: float = 0.0  # charge efficiency shaping component (for logging)
    total_shoot_eff_count: int = 0       # number of activations that shot (for per-activation avg)
    total_charge_eff_count: int = 0      # number of activations that charged (for per-activation avg)
    # Per-head entropies for entropy target tuning
    unit_entropies: torch.Tensor | None = None     # (N,)
    move_entropies: torch.Tensor | None = None     # (N,)
    dest_entropies: torch.Tensor | None = None     # (N,) — destination mixture entropy
    charge_entropies: torch.Tensor | None = None   # (N,)
    shoot_entropies: torch.Tensor | None = None    # (N,)
    # Per-step masks for conditional heads
    is_move: torch.Tensor | None = None            # (N,) bool — dest active (non-charge, non-shaken)
    is_can_shoot: torch.Tensor | None = None       # (N,) bool — shoot active (advance-reachable dest)
    is_charge: torch.Tensor | None = None          # (N,) bool
    # Move-head was a real choice (not shaken AND ≥1 chargeable enemy). Used
    # to gate the move-head entropy target so masked-deterministic states
    # don't drag the aggregate entropy below the (now-infeasible) target.
    is_can_charge_any: torch.Tensor | None = None  # (N,) bool
    alive_mask: torch.Tensor | None = None         # (N, 10)
    enemy_alive_mask: torch.Tensor | None = None   # (N, 10)
    shoot_mask: torch.Tensor | None = None         # (N, 10)
    # Auxiliary head outputs — long-horizon (None when aux heads not present on model)
    aux_friendly_surv_alpha: torch.Tensor | None = None   # (N, 10)
    aux_friendly_surv_beta: torch.Tensor | None = None    # (N, 10)
    aux_enemy_surv_alpha: torch.Tensor | None = None      # (N, 10)
    aux_enemy_surv_beta: torch.Tensor | None = None       # (N, 10)
    aux_obj_control_logits: torch.Tensor | None = None    # (N, 5, 3)
    # Auxiliary head outputs — short-horizon (end-of-current-round)
    aux_friendly_surv_alpha_short: torch.Tensor | None = None   # (N, 10)
    aux_friendly_surv_beta_short: torch.Tensor | None = None    # (N, 10)
    aux_enemy_surv_alpha_short: torch.Tensor | None = None      # (N, 10)
    aux_enemy_surv_beta_short: torch.Tensor | None = None       # (N, 10)
    aux_obj_control_logits_short: torch.Tensor | None = None    # (N, 5, 3)
    # Activation countdown head outputs
    aux_friendly_act_remaining: torch.Tensor | None = None   # (N,) predicted friendly activations remaining
    aux_enemy_act_remaining: torch.Tensor | None = None       # (N,) predicted enemy activations remaining
    # Per-opponent-type mean value estimates (diagnostic)
    per_opp_type_mean_values: dict[str, float] | None = None
    # Logits for planning distillation loss
    unit_logits: torch.Tensor | None = None    # (N, 10) — raw logits after alive masking
    move_logits: torch.Tensor | None = None    # (N, 2) — move type logits conditioned on chosen unit
    charge_logits: torch.Tensor | None = None  # (N, 10) — charge target logits (masked)
    shoot_logits: torch.Tensor | None = None   # (N, 10) — shoot target logits (masked)
    dest_logits: torch.Tensor | None = None    # (N, mb_max_cands) — dest pointer logits (padded, -inf on mask)
    # Destination pointer: number of valid candidates per step (for normalised entropy)
    dest_n_valid: torch.Tensor | None = None  # (N,) — int, number of valid candidates
    # Per-phase value head outputs — only populated when the phase-reencode flag
    # is on. Shape (N, N_PHASES); columns indexed by PHASE_PRE_SELECT etc.
    # Trained in compute_loss_flat as auxiliary regressors against flat_returns.
    per_phase_values: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Pre-built replay data (avoids rebuilding tensors per minibatch)
# ---------------------------------------------------------------------------

@dataclass
class PreparedReplayData:
    """Pre-built tensors for efficient minibatched replay.

    Built once via ``prepare_replay_data`` before the PPO epoch loop.
    ``replay_from_prepared`` slices these by step indices per minibatch,
    avoiding the expensive Python data-prep that previously ran 12× per batch.
    """
    # Fixed-shape tensors on the target device, indexed by flat step index
    state_batch: torch.Tensor           # (N, FEAT_DIM)
    alive_mask: torch.Tensor            # (N, 10) bool
    enemy_alive_mask: torch.Tensor      # (N, 10) bool
    unit_indices: torch.Tensor          # (N,) long
    move_indices: torch.Tensor          # (N,) long
    dest_selected_indices: torch.Tensor  # (N,) long
    dest_is_ar: torch.Tensor            # (N,) bool
    post_move_rel: torch.Tensor         # (N, 30) float
    opp_type_indices: torch.Tensor      # (N,) long
    side_indices: torch.Tensor          # (N,) long
    charge_indices: torch.Tensor        # (N,) long
    shoot_indices: torch.Tensor         # (N,) long
    shoot_mask: torch.Tensor            # (N, 10) bool
    rewards: torch.Tensor               # (N,) float (for total_reward)
    total_shoot_eff_reward: float        # shooting efficiency shaping total (for logging)
    total_charge_eff_reward: float       # charge efficiency shaping total (for logging)
    total_shoot_eff_count: int           # activations that shot
    total_charge_eff_count: int          # activations that charged
    # Compact dest feature storage (avoids multi-GB padded arrays).
    # Per-minibatch, replay_from_prepared pads to the local max and transfers.
    dest_buffer: np.ndarray               # (total_cands, DEST_FEATURE_DIM) float32 — compact
    dest_offsets: np.ndarray              # (N,) int64 — offset into buffer per step
    dest_counts: np.ndarray               # (N,) int32 — number of valid candidates per step
    # Metadata
    n_steps: int
    n_episodes: int
    device: torch.device
    # Phase-reencode: per-step post-move state_vec for the POST_DEST encode.
    # None when the phase-reencode flag is off. When on, every row is populated —
    # activations that didn't produce a post-move state (charge, shaken, invalid
    # dest) get state_vec as their post-move value so the POST_DEST encode is a
    # no-op delta from the prior phase's input.
    state_vec_post_batch: torch.Tensor | None = None
    # Cover-aware expected wound frac per step (TERRAIN_SPEC §5.4/§5.5). When
    # present, replayed as ``expected_wound_frac_override`` to compute_shoot_logits
    # so sampling and replay see the same shoot key. None ⇒ replay falls back to
    # the cover-blind bucket lookup (used for older trajectories).
    cover_aware_dmg: torch.Tensor | None = None  # (N, 10) float


def prepare_replay_data(
    all_trajectories: list[list[TacticalActivationRecord]],
    device: torch.device = torch.device('cpu'),
) -> PreparedReplayData:
    """Build all replay tensors once for the entire batch.

    This is the expensive data-prep step (numpy stacking, alive-mask
    construction, destination-feature recomputation).  Calling it once
    before the PPO epoch loop and slicing per-minibatch via
    ``replay_from_prepared`` eliminates ~92% of the original Python
    overhead.
    """
    flat_steps: list[TacticalActivationRecord] = []
    for traj in all_trajectories:
        flat_steps.extend(traj)
    n_steps = len(flat_steps)
    n_units = MAX_UNITS_PER_SIDE

    if n_steps == 0:
        empty_f = torch.zeros(0, device=device)
        empty_b = torch.zeros(0, dtype=torch.bool, device=device)
        empty_l = torch.zeros(0, dtype=torch.long, device=device)
        return PreparedReplayData(
            state_batch=torch.zeros(0, 1, device=device),
            alive_mask=empty_b.unsqueeze(0), enemy_alive_mask=empty_b.unsqueeze(0),
            unit_indices=empty_l, move_indices=empty_l,
            dest_selected_indices=empty_l, dest_is_ar=empty_b,
            post_move_rel=torch.zeros(0, 30, device=device),
            opp_type_indices=empty_l, side_indices=empty_l,
            charge_indices=empty_l, shoot_indices=empty_l,
            shoot_mask=empty_b.unsqueeze(0), rewards=empty_f,
            dest_buffer=np.zeros((0, DEST_FEATURE_DIM), dtype=np.float32),
            dest_offsets=np.zeros(0, dtype=np.int64),
            dest_counts=np.zeros(0, dtype=np.int32),
            n_steps=0, n_episodes=len(all_trajectories), device=device,
        )

    # --- State vectors (one np.stack) ---
    state_np = np.stack([s.state_vec for s in flat_steps])
    state_batch = torch.from_numpy(state_np).to(device)

    # --- Post-move state vectors (Phase-reencode POST_DEST input) ---
    # Only materialised when the phase-reencode flag is on. Rows without a
    # stored state_vec_post fall back to the pre-move state_vec so the
    # POST_DEST encode has the same input as prior phases (h_dest == h_mt).
    if is_phase_reencode_enabled():
        state_post_np = np.stack([
            s.state_vec_post if s.state_vec_post is not None else s.state_vec
            for s in flat_steps
        ])
        state_post_batch = torch.from_numpy(state_post_np).to(device)
    else:
        state_post_batch = None

    # --- Alive masks (vectorized numpy) ---
    alive_np = np.zeros((n_steps, n_units), dtype=np.bool_)
    enemy_alive_np = np.zeros((n_steps, n_units), dtype=np.bool_)
    for i, s in enumerate(flat_steps):
        alive_np[i, :min(n_units, len(s.alive_mask))] = s.alive_mask[:n_units]
        enemy_alive_np[i, :min(n_units, len(s.enemy_alive_mask))] = s.enemy_alive_mask[:n_units]

    # --- Scalar per-step metadata (vectorized via numpy) ---
    unit_idx_np = np.array([s.unit_idx for s in flat_steps], dtype=np.int64)
    move_idx_np = np.array([s.move_type for s in flat_steps], dtype=np.int64)
    charge_idx_np = np.array([s.charge_target_idx for s in flat_steps], dtype=np.int64)
    shoot_idx_np = np.array([s.shoot_target_idx for s in flat_steps], dtype=np.int64)
    opp_type_np = np.array([s.opponent_type_idx for s in flat_steps], dtype=np.int64)
    reward_np = np.array([s.reward for s in flat_steps], dtype=np.float32)
    post_move_np = np.stack([s.post_move_rel for s in flat_steps]).astype(np.float32)

    # --- Side indices (vectorized) ---
    _side_map = {"A": 0, "B": 1}
    side_np = np.zeros(n_steps, dtype=np.int64)
    for i, s in enumerate(flat_steps):
        _dr = getattr(s, 'dest_recomp', None)
        if _dr:
            side_np[i] = _side_map.get(_dr.get('player', 'A'), 0)

    # --- Shoot mask ---
    if hasattr(flat_steps[0], 'shoot_mask') and flat_steps[0].shoot_mask is not None:
        shoot_mask_np = np.array([s.shoot_mask for s in flat_steps], dtype=np.bool_)
    else:
        shoot_mask_np = enemy_alive_np.copy()
    # Will apply is_shaken masking after tensor conversion

    # --- Cover-aware expected wound frac (§5.4/§5.5) ---
    # Build the batched tensor only when EVERY step has it recorded; mixed
    # batches (older trajectories without the field) fall back to the
    # cover-blind bucket lookup so we don't silently zero out their shoot
    # keys.
    cover_dmg_np: np.ndarray | None = None
    if all(getattr(s, 'cover_aware_dmg', None) is not None for s in flat_steps):
        cover_dmg_np = np.zeros((n_steps, MAX_UNITS_PER_SIDE), dtype=np.float32)
        for i, s in enumerate(flat_steps):
            cad = s.cover_aware_dmg
            cover_dmg_np[i, :len(cad)] = np.asarray(cad, dtype=np.float32)[:MAX_UNITS_PER_SIDE]

    # --- Destination features (precompute once, compact buffer storage) ---
    dest_selected_np = np.zeros(n_steps, dtype=np.int64)
    dest_is_ar_np = np.ones(n_steps, dtype=np.bool_)
    dest_counts_np = np.zeros(n_steps, dtype=np.int32)
    dest_offsets_np = np.zeros(n_steps, dtype=np.int64)

    # Collect per-step features into a flat list, then concatenate
    _feat_chunks: list[np.ndarray] = []
    _offset = 0

    for i, s in enumerate(flat_steps):
        if s.dest_candidates is not None and len(s.dest_candidates) > 0:
            n_cand = min(len(s.dest_candidates), MAX_DEST_CANDIDATES)
            dest_selected_np[i] = s.dest_selected_idx
            dest_counts_np[i] = n_cand
            dest_offsets_np[i] = _offset

            if (s.dest_advance_reachable is not None
                    and s.dest_selected_idx >= 0
                    and s.dest_selected_idx < len(s.dest_advance_reachable)):
                dest_is_ar_np[i] = s.dest_advance_reachable[s.dest_selected_idx]

            padded_ar = np.ones(n_cand, dtype=np.bool_)
            if s.dest_advance_reachable is not None:
                n_ar = min(len(s.dest_advance_reachable), n_cand)
                padded_ar[:n_ar] = s.dest_advance_reachable[:n_ar]

            if s.dest_features is not None and len(s.dest_features) > 0:
                feats = np.asarray(s.dest_features, dtype=np.float32)[:n_cand]
            elif s.dest_recomp is not None:
                rc = s.dest_recomp
                padded_cands = np.zeros((n_cand, 2), dtype=np.int32)
                padded_cands[:n_cand] = s.dest_candidates[:n_cand]
                padded_mask_rc = np.ones(n_cand, dtype=np.bool_)
                feats = np.asarray(compute_destination_features(
                    padded_cands, padded_mask_rc,
                    None, s.unit_idx, rc['player'],
                    None, rc['enemy_alive_mask'],
                    rc['fr_matchups'], rc['er_matchups'], rc['melee_matchups'],
                    rc['move_budget'],
                    enemy_cache=rc['enemy_cache'],
                    unit_centre=(rc['unit_cx'], rc['unit_cy']),
                    unit_alive_frac=rc['unit_alive_frac'],
                    advance_reachable=padded_ar,
                ), dtype=np.float32)[:n_cand]
            else:
                feats = np.zeros((n_cand, DEST_FEATURE_DIM), dtype=np.float32)
            _feat_chunks.append(feats)
            _offset += n_cand

    if _feat_chunks:
        dest_buffer = np.concatenate(_feat_chunks)  # (total_cands, DEST_FEATURE_DIM)
    else:
        dest_buffer = np.zeros((0, DEST_FEATURE_DIM), dtype=np.float32)
    del _feat_chunks

    # --- Move everything to device ---
    alive_t = torch.from_numpy(alive_np).to(device)
    enemy_alive_t = torch.from_numpy(enemy_alive_np).to(device)
    unit_idx_t = torch.from_numpy(unit_idx_np).to(device)
    move_idx_t = torch.from_numpy(move_idx_np).to(device)
    charge_idx_t = torch.from_numpy(charge_idx_np).to(device)
    shoot_idx_t = torch.from_numpy(shoot_idx_np).to(device)
    opp_type_t = torch.from_numpy(opp_type_np).to(device)
    side_t = torch.from_numpy(side_np).to(device)
    reward_t = torch.from_numpy(reward_np).to(device)
    post_move_t = torch.from_numpy(post_move_np).to(device)
    dest_sel_t = torch.from_numpy(dest_selected_np).to(device)
    dest_is_ar_t = torch.from_numpy(dest_is_ar_np).to(device)
    # dest_buffer stays on CPU (compact storage, per-minibatch padding in replay)

    # Apply is_shaken to shoot_mask
    is_shaken_np = extract_is_shaken(
        torch.from_numpy(state_np), torch.from_numpy(unit_idx_np)).numpy()
    shoot_mask_np = shoot_mask_np & ~np.expand_dims(is_shaken_np, -1)
    shoot_mask_t = torch.from_numpy(shoot_mask_np).to(device)
    cover_dmg_t = (torch.from_numpy(cover_dmg_np).to(device)
                   if cover_dmg_np is not None else None)

    return PreparedReplayData(
        state_batch=state_batch,
        alive_mask=alive_t, enemy_alive_mask=enemy_alive_t,
        unit_indices=unit_idx_t, move_indices=move_idx_t,
        dest_selected_indices=dest_sel_t, dest_is_ar=dest_is_ar_t,
        post_move_rel=post_move_t,
        opp_type_indices=opp_type_t, side_indices=side_t,
        charge_indices=charge_idx_t, shoot_indices=shoot_idx_t,
        shoot_mask=shoot_mask_t, rewards=reward_t,
        total_shoot_eff_reward=sum(s.shooting_efficiency_reward for s in flat_steps),
        total_charge_eff_reward=sum(s.charge_efficiency_reward for s in flat_steps),
        total_shoot_eff_count=sum(1 for s in flat_steps if s.shooting_efficiency_reward != 0.0),
        total_charge_eff_count=sum(1 for s in flat_steps if s.charge_efficiency_reward != 0.0),
        dest_buffer=dest_buffer,
        dest_offsets=dest_offsets_np, dest_counts=dest_counts_np,
        n_steps=n_steps, n_episodes=len(all_trajectories), device=device,
        state_vec_post_batch=state_post_batch,
        cover_aware_dmg=cover_dmg_t,
    )


def replay_from_prepared(
    model: TacticalModel,
    prepared: PreparedReplayData,
    step_indices: torch.Tensor,
    n_episodes: int = 0,
) -> FlatReplayResult:
    """Run model forward + log-prob computation on a slice of pre-built data.

    ``step_indices`` is a (M,) long tensor (on ``prepared.device``) that
    selects which steps from the prepared batch to include.  All heavy
    data-prep has already happened in ``prepare_replay_data``.
    """
    n_steps = step_indices.shape[0]
    n_units = MAX_UNITS_PER_SIDE

    if n_steps == 0:
        return FlatReplayResult(
            log_probs=torch.zeros(0, device=prepared.device),
            entropies=torch.zeros(0, device=prepared.device),
            values=torch.zeros(0, device=prepared.device),
            n_episodes=n_episodes, total_reward=0.0,
        )

    # --- Slice pre-built tensors (fast GPU indexing) ---
    sb = prepared.state_batch[step_indices]
    alive_batch = prepared.alive_mask[step_indices]
    enemy_alive_batch = prepared.enemy_alive_mask[step_indices]
    unit_indices = prepared.unit_indices[step_indices]
    move_indices = prepared.move_indices[step_indices]
    dest_selected_batch = prepared.dest_selected_indices[step_indices]
    dest_is_ar_batch = prepared.dest_is_ar[step_indices]
    post_move_rel_batch = prepared.post_move_rel[step_indices]
    opp_type_indices = prepared.opp_type_indices[step_indices]
    side_indices = prepared.side_indices[step_indices]
    charge_indices = prepared.charge_indices[step_indices]
    shoot_indices_t = prepared.shoot_indices[step_indices]
    shoot_mask_batch = prepared.shoot_mask[step_indices]
    cover_dmg_batch = (prepared.cover_aware_dmg[step_indices]
                       if prepared.cover_aware_dmg is not None else None)
    rewards_batch = prepared.rewards[step_indices]

    # --- Dest features: pad compact buffer to minibatch-local max, transfer ---
    step_idx_cpu = step_indices.cpu().numpy()
    mb_counts = prepared.dest_counts[step_idx_cpu]
    mb_max_cands = max(int(mb_counts.max()), 1) if len(mb_counts) > 0 else 1
    dest_feat_np = np.zeros((n_steps, mb_max_cands, DEST_FEATURE_DIM), dtype=np.float32)
    dest_mask_np = np.zeros((n_steps, mb_max_cands), dtype=np.bool_)
    for j in range(n_steps):
        nc = mb_counts[j]
        if nc > 0:
            off = prepared.dest_offsets[step_idx_cpu[j]]
            dest_feat_np[j, :nc] = prepared.dest_buffer[off:off + nc]
            dest_mask_np[j, :nc] = True
    dest_features_batch = torch.from_numpy(dest_feat_np).to(prepared.device)
    dest_mask_batch = torch.from_numpy(dest_mask_np).to(prepared.device)

    # === Trunk (single pass) or phase-reencode chain (4 encode calls) ===
    per_phase_values: torch.Tensor | None = None
    global _FIRST_REPLAY_FLAG_DIAG_PRINTED
    _flag = is_phase_reencode_enabled()
    _has_post = prepared.state_vec_post_batch is not None
    if not _FIRST_REPLAY_FLAG_DIAG_PRINTED:
        print(f"  [DIAG] first replay: phase_reencode_flag={_flag} "
              f"state_vec_post_batch_present={_has_post} "
              f"→ taking {'PHASED' if (_flag and _has_post) else 'LEGACY'} branch")
        _FIRST_REPLAY_FLAG_DIAG_PRINTED = True
    if _flag and _has_post:
        # Phase-reencode replay: h persists across phases, h0 recomputes each phase.
        h_pre, units_pre, round_onehot = model.encode(
            sb, phase=PHASE_PRE_SELECT, acting_unit_idx=None, h_prev=None,
        )
        h_sel, _units_sel, _ = model.encode(
            sb, phase=PHASE_POST_SELECT,
            acting_unit_idx=unit_indices, h_prev=h_pre,
        )
        h_mt, _units_mt, _ = model.encode(
            sb, phase=PHASE_POST_MOVETYPE,
            acting_unit_idx=unit_indices, h_prev=h_sel,
        )
        sb_post = prepared.state_vec_post_batch[step_indices]
        h_dest, units_dest, _ = model.encode(
            sb_post, phase=PHASE_POST_DEST,
            acting_unit_idx=unit_indices, h_prev=h_mt,
        )
        # Per-head h/units routing: pre-move h/units for unit/move/dest/value/aux
        # (they reason about the choice to make); post-move h/units for
        # charge/shoot (they reason about the state after committing).
        h = h_pre
        units = units_pre
        h_move = h_sel
        h_dest_query = h_mt
        h_pointer = h_dest
        units_pointer = units_dest
        # Per-phase value heads — auxiliary regressors trained against flat_returns
        # in compute_loss_flat (_PER_PHASE_V_WEIGHT).
        per_phase_values = torch.stack([
            model.per_phase_value(h_pre, PHASE_PRE_SELECT),
            model.per_phase_value(h_sel, PHASE_POST_SELECT),
            model.per_phase_value(h_mt, PHASE_POST_MOVETYPE),
            model.per_phase_value(h_dest, PHASE_POST_DEST),
        ], dim=-1)  # (N, N_PHASES)
    else:
        h, units, round_onehot = model.trunk(sb)
        h_move = h
        h_dest_query = h
        h_pointer = h
        units_pointer = units
    if torch.isnan(h).any() or torch.isinf(h).any():
        print("  WARNING: NaN/Inf in trunk output during replay — clamping")
        h = torch.nan_to_num(h, nan=0.0, posinf=50.0, neginf=-50.0)

    # === Unit selection head ===
    unit_logits = model.unit_selection_head(h)
    unit_logits = unit_logits.masked_fill(~alive_batch, float('-inf'))

    # === Extract unit features ===
    unit_features = units[:, :n_units, :].gather(
        1, unit_indices.unsqueeze(1).unsqueeze(2).expand(n_steps, 1, TACTICAL_UNIT_FEATURES),
    ).squeeze(1).detach()

    # === Move type head ===
    can_charge_batch = extract_can_charge_mask(sb, unit_indices)
    is_shaken_batch = extract_is_shaken(sb, unit_indices)

    h_uf = torch.cat([h_move, unit_features], dim=-1)
    move_logits_out = model.move_type_head(h_uf)
    no_chargeable = ~can_charge_batch.any(dim=-1)
    move_logits_out = move_logits_out.clone()
    move_logits_out[:, MOVE_CHARGE] = move_logits_out[:, MOVE_CHARGE].masked_fill(no_chargeable, float('-inf'))
    move_logits_out[:, MOVE_CHARGE] = move_logits_out[:, MOVE_CHARGE].masked_fill(is_shaken_batch, float('-inf'))

    move_onehot = F.one_hot(move_indices, NUM_MOVE_TYPES).float()
    h_uf_m = torch.cat([h_dest_query, unit_features, move_onehot], dim=-1)

    # === Destination pointer ===
    dest_logits = model.compute_dest_logits(h_uf_m, dest_features_batch, dest_mask_batch)

    # === Charge target pointer head ===
    charge_logits_out = model.compute_charge_logits(
        h_pointer, units_pointer, unit_indices, enemy_alive_batch, can_charge_batch,
    )

    # === Shoot target pointer head ===
    shoot_logits_out, shoot_primary, shoot_bias = model.compute_shoot_logits(
        h_pointer, units_pointer, unit_indices, post_move_rel_batch,
        enemy_alive_batch, shoot_range_mask=shoot_mask_batch,
        return_components=True,
        expected_wound_frac_override=cover_dmg_batch,
    )

    # === Value head ===
    opp_embed_batch = model.opponent_embedding(opp_type_indices)
    side_embed_batch = model.side_embedding(side_indices)
    values = model.value_head(h, round_onehot, opp_embed_batch, side_embed_batch)

    # Per-opponent-type mean value estimates (diagnostic)
    _opp_type_names = ["heuristic", "sp_mirror", "sp_hof", "sp_ml", "sp_random"]
    per_opp_type_mean_values: dict[str, float] = {}
    with torch.no_grad():
        for ot_idx, ot_name in enumerate(_opp_type_names):
            mask = opp_type_indices == ot_idx
            if mask.any():
                per_opp_type_mean_values[f"mean_value_{ot_name}"] = values[mask].mean().item()
        _mirror_mask = opp_type_indices == 1
        _a_mask = (side_indices == 0) & _mirror_mask
        _b_mask = (side_indices == 1) & _mirror_mask
        if _a_mask.any():
            per_opp_type_mean_values["mean_value_side_a"] = values[_a_mask].mean().item()
        if _b_mask.any():
            per_opp_type_mean_values["mean_value_side_b"] = values[_b_mask].mean().item()

        # Shoot head diagnostics: is the scalar pathway load-bearing vs. the
        # bias pathway? Ratio near 1 means logits are driven mostly by
        # expected_wound_frac × current_wound_frac; near 0 means the bias
        # pathway is bypassing the mechanical signal.
        _valid = shoot_mask_batch
        _n_valid = _valid.sum(dim=-1)
        _eligible = _n_valid >= 2
        if _eligible.any():
            _v = _valid.float()
            _n = _n_valid.clamp(min=1).float().unsqueeze(-1)
            _primary_mean = (shoot_primary * _v).sum(dim=-1, keepdim=True) / _n
            _total = shoot_primary + shoot_bias
            _total_mean = (_total * _v).sum(dim=-1, keepdim=True) / _n
            _primary_var = (((shoot_primary - _primary_mean) * _v) ** 2).sum(dim=-1) / _n.squeeze(-1)
            _total_var = (((_total - _total_mean) * _v) ** 2).sum(dim=-1) / _n.squeeze(-1)
            _ratio = _primary_var / _total_var.clamp(min=1e-8)
            per_opp_type_mean_values["shoot_primary_var_ratio"] = _ratio[_eligible].mean().item()
            per_opp_type_mean_values["shoot_primary_abs_mean"] = shoot_primary[_valid].abs().mean().item()
            per_opp_type_mean_values["shoot_bias_abs_mean"] = shoot_bias[_valid].abs().mean().item()

    # === Log-probs & entropies ===
    all_dead = ~alive_batch.any(dim=1, keepdim=True)
    safe_unit_logits = unit_logits.masked_fill(all_dead, 0.0)
    # Do NOT collapse -inf: masked dead slots must stay at zero probability
    # under softmax (otherwise log_softmax normalization leaks onto padded
    # slots, inflating entropy and biasing unit_lp).
    safe_unit_logits = torch.nan_to_num(safe_unit_logits, nan=0.0, posinf=50.0)
    unit_log_probs = torch.log_softmax(safe_unit_logits, dim=-1)
    unit_lp = unit_log_probs.gather(1, unit_indices.unsqueeze(1)).squeeze(1)
    unit_ent = torch.distributions.Categorical(logits=safe_unit_logits).entropy()

    move_logits_safe = torch.nan_to_num(move_logits_out, nan=0.0, posinf=50.0)
    move_dist = torch.distributions.Categorical(logits=move_logits_safe)
    move_lp = move_dist.log_prob(move_indices)
    move_ent = move_dist.entropy()

    has_dest = dest_mask_batch.any(dim=-1)
    safe_dest_logits = dest_logits.clone()
    safe_dest_logits[~has_dest] = 0.0
    safe_dest_logits = torch.nan_to_num(safe_dest_logits, nan=0.0, posinf=50.0)
    dest_log_probs = torch.log_softmax(safe_dest_logits, dim=-1)
    dest_lp = dest_log_probs.gather(1, dest_selected_batch.unsqueeze(1)).squeeze(1)
    dest_lp = dest_lp.masked_fill(~has_dest, 0.0)
    dest_probs = torch.softmax(safe_dest_logits, dim=-1)
    dest_ent = -(dest_probs * torch.log(dest_probs + 1e-8)).sum(dim=-1)
    dest_ent = dest_ent.masked_fill(~has_dest, 0.0)

    enemy_all_dead = ~enemy_alive_batch.any(dim=-1, keepdim=True)
    safe_charge_logits = charge_logits_out.masked_fill(enemy_all_dead, 0.0)
    safe_charge_logits = torch.nan_to_num(safe_charge_logits, nan=0.0, posinf=50.0)
    charge_log_probs = torch.log_softmax(safe_charge_logits, dim=-1)
    charge_lp = charge_log_probs.gather(1, charge_indices.unsqueeze(1)).squeeze(1)
    charge_ent = torch.distributions.Categorical(logits=safe_charge_logits).entropy()

    no_shootable = ~shoot_mask_batch.any(dim=-1, keepdim=True)
    safe_shoot_logits = shoot_logits_out.masked_fill(no_shootable, 0.0)
    safe_shoot_logits = torch.nan_to_num(safe_shoot_logits, nan=0.0, posinf=50.0)
    shoot_log_probs = torch.log_softmax(safe_shoot_logits, dim=-1)
    shoot_lp = shoot_log_probs.gather(1, shoot_indices_t.unsqueeze(1)).squeeze(1)
    shoot_lp = shoot_lp.masked_fill(no_shootable.squeeze(-1), 0.0)
    shoot_ent = torch.distributions.Categorical(logits=safe_shoot_logits).entropy()

    # === Combine log-probs ===
    total_lp = unit_lp + move_lp
    total_ent = unit_ent + move_ent
    n_heads = torch.full((n_steps,), 2.0, device=prepared.device)

    is_move = (move_indices == MOVE_MOVE) & ~is_shaken_batch
    _zero = torch.zeros_like(dest_lp)
    total_lp = total_lp + torch.where(is_move, dest_lp, _zero)
    total_ent = total_ent + torch.where(is_move, dest_ent, _zero)
    n_heads = n_heads + is_move.float()

    is_can_shoot = is_move & dest_is_ar_batch
    total_lp = total_lp + torch.where(is_can_shoot, shoot_lp, _zero)
    total_ent = total_ent + torch.where(is_can_shoot, shoot_ent, _zero)
    n_heads = n_heads + is_can_shoot.float()

    is_charge = move_indices == MOVE_CHARGE
    total_lp = total_lp + torch.where(is_charge, charge_lp, _zero)
    total_ent = total_ent + torch.where(is_charge, charge_ent, _zero)
    n_heads = n_heads + is_charge.float()

    # Move head is a real 2-way choice only when not shaken AND ≥1 chargeable
    # enemy. State_vec masking forces deterministic move otherwise — those
    # rows must not enter the move-entropy target average.
    is_can_charge_any = (~is_shaken_batch) & can_charge_batch.any(dim=-1)

    mean_ent = total_ent / n_heads.clamp(min=1.0)
    total_reward = rewards_batch.sum().item()
    total_shoot_eff_reward = prepared.total_shoot_eff_reward
    total_charge_eff_reward = prepared.total_charge_eff_reward
    total_shoot_eff_count = prepared.total_shoot_eff_count
    total_charge_eff_count = prepared.total_charge_eff_count

    # === Auxiliary prediction heads ===
    aux_fs_alpha = aux_fs_beta = aux_es_alpha = aux_es_beta = None
    aux_obj_logits = None
    aux_fs_alpha_short = aux_fs_beta_short = aux_es_alpha_short = aux_es_beta_short = None
    aux_obj_logits_short = None
    if hasattr(model, 'aux_friendly_survival_head'):
        fs_raw = model.aux_friendly_survival_head(h).view(n_steps, n_units, 2)
        aux_fs_alpha = F.softplus(fs_raw[..., 0]) + 0.01
        aux_fs_beta = F.softplus(fs_raw[..., 1]) + 0.01
        es_raw = model.aux_enemy_survival_head(h).view(n_steps, n_units, 2)
        aux_es_alpha = F.softplus(es_raw[..., 0]) + 0.01
        aux_es_beta = F.softplus(es_raw[..., 1]) + 0.01
        aux_obj_logits = model.aux_obj_control_head(h).view(n_steps, 5, 3)
        if hasattr(model, 'aux_friendly_survival_head_short'):
            fs_raw_s = model.aux_friendly_survival_head_short(h).view(n_steps, n_units, 2)
            aux_fs_alpha_short = F.softplus(fs_raw_s[..., 0]) + 0.01
            aux_fs_beta_short = F.softplus(fs_raw_s[..., 1]) + 0.01
            es_raw_s = model.aux_enemy_survival_head_short(h).view(n_steps, n_units, 2)
            aux_es_alpha_short = F.softplus(es_raw_s[..., 0]) + 0.01
            aux_es_beta_short = F.softplus(es_raw_s[..., 1]) + 0.01
            aux_obj_logits_short = model.aux_obj_control_head_short(h).view(n_steps, 5, 3)

    aux_f_act_rem = aux_e_act_rem = None
    if hasattr(model, 'aux_friendly_activations_head'):
        aux_f_act_rem = F.softplus(model.aux_friendly_activations_head(h).squeeze(-1))
        aux_e_act_rem = F.softplus(model.aux_enemy_activations_head(h).squeeze(-1))

    return FlatReplayResult(
        log_probs=total_lp, entropies=mean_ent, values=values,
        n_episodes=n_episodes, total_reward=total_reward,
        total_shoot_eff_reward=total_shoot_eff_reward,
        total_charge_eff_reward=total_charge_eff_reward,
        total_shoot_eff_count=total_shoot_eff_count,
        total_charge_eff_count=total_charge_eff_count,
        unit_entropies=unit_ent, move_entropies=move_ent,
        dest_entropies=dest_ent, charge_entropies=charge_ent,
        shoot_entropies=shoot_ent,
        is_move=is_move, is_can_shoot=is_can_shoot, is_charge=is_charge,
        is_can_charge_any=is_can_charge_any,
        alive_mask=alive_batch, enemy_alive_mask=enemy_alive_batch,
        shoot_mask=shoot_mask_batch,
        aux_friendly_surv_alpha=aux_fs_alpha, aux_friendly_surv_beta=aux_fs_beta,
        aux_enemy_surv_alpha=aux_es_alpha, aux_enemy_surv_beta=aux_es_beta,
        aux_obj_control_logits=aux_obj_logits,
        aux_friendly_surv_alpha_short=aux_fs_alpha_short,
        aux_friendly_surv_beta_short=aux_fs_beta_short,
        aux_enemy_surv_alpha_short=aux_es_alpha_short,
        aux_enemy_surv_beta_short=aux_es_beta_short,
        aux_obj_control_logits_short=aux_obj_logits_short,
        aux_friendly_act_remaining=aux_f_act_rem,
        aux_enemy_act_remaining=aux_e_act_rem,
        per_opp_type_mean_values=per_opp_type_mean_values,
        unit_logits=unit_logits, move_logits=move_logits_out,
        charge_logits=charge_logits_out, shoot_logits=shoot_logits_out,
        dest_logits=dest_logits,
        dest_n_valid=dest_mask_batch.sum(dim=-1),
        per_phase_values=per_phase_values,
    )


# ---------------------------------------------------------------------------
# Replay (legacy single-call interface, used by profiling scripts)
# ---------------------------------------------------------------------------

def replay_tactical_log_probs_flat(
    model: TacticalModel,
    all_trajectories: list[list[TacticalActivationRecord]],
) -> FlatReplayResult:
    """Replay trajectories through the tactical model and compute log-probs.

    Legacy interface — calls prepare_replay_data + replay_from_prepared
    in a single shot.  For PPO minibatching, use the two-step API directly
    to avoid redundant data preparation.
    """
    device = next(model.parameters()).device
    prepared = prepare_replay_data(all_trajectories, device=device)
    if prepared.n_steps == 0:
        return FlatReplayResult(
            log_probs=torch.zeros(0),
            entropies=torch.zeros(0),
            values=torch.zeros(0),
            n_episodes=len(all_trajectories),
            total_reward=0.0,
        )
    all_indices = torch.arange(prepared.n_steps, dtype=torch.long, device=device)
    return replay_from_prepared(model, prepared, all_indices, n_episodes=len(all_trajectories))


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------

def compute_loss_flat(
    flat_result: FlatReplayResult,
    flat_old_log_probs: torch.Tensor,
    flat_advantages: torch.Tensor,
    flat_returns: torch.Tensor,
    clip_epsilon: float,
    value_coeff: float,
    entropy_coeff: float,
    aux_coeff: float = 0.0,
    aux_ratio: float = 0.2,
    flat_steps: list[TacticalActivationRecord] | None = None,
    entropy_tuner: EntropyTargetTuner | None = None,
    planning_distill_max_weight: float = 0.0,
    planning_distill_ramp: float = 1.0,
    planning_distill_mode: str = "ce_chosen",
    mpo_eta: float = 1.0,
    flat_result_old: FlatReplayResult | None = None,
    kl_trust_region_beta: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Vectorized PPO loss — no Python loops over individual steps.

    All inputs are flat (N,) tensors aligned by step index.
    When aux_coeff > 0 and flat_steps is provided, computes auxiliary
    prediction losses (survival Beta NLL + objective control CE).

    If entropy_tuner is provided, uses per-head adaptive entropy targets
    instead of the flat entropy_coeff.  The alpha loss is computed and
    returned in metrics["alpha_loss"] for the caller to backprop separately.
    """
    n = flat_result.log_probs.shape[0]
    if n == 0:
        zero = torch.tensor(0.0)
        return zero, {
            "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
            "mean_entropy": 0.0, "mean_reward": 0.0, "aux_loss": 0.0,
            "alpha_loss": 0.0, "per_phase_value_loss": 0.0,
        }

    # Normalize advantages (zero-mean, unit-variance) for stable gradients
    adv_std, adv_mean = torch.std_mean(flat_advantages)
    if adv_std > 1e-8:
        flat_advantages = (flat_advantages - adv_mean) / adv_std

    # PPO clipped surrogate — fully vectorized
    # Clamp log-ratio to prevent exp() overflow (±5 → ratio in [~0.007, ~148])
    log_ratio = flat_result.log_probs - flat_old_log_probs
    # Diagnostic: print ratio stats on first call to help debug clip fraction issues
    if not hasattr(compute_loss_flat, '_diag_done'):
        compute_loss_flat._diag_done = True
        _lr = log_ratio.detach()
        _abs_lr = _lr.abs()
        _clipped = (_abs_lr > clip_epsilon).float().mean().item()
        print(f"  [DIAG] First-call log-ratio stats: "
              f"mean={_lr.mean().item():.4f} std={_lr.std().item():.4f} "
              f"min={_lr.min().item():.4f} max={_lr.max().item():.4f} "
              f"median={_lr.median().item():.4f} "
              f"|>0.5|={(_abs_lr > 0.5).float().mean().item():.3f} "
              f"|>1.0|={(_abs_lr > 1.0).float().mean().item():.3f} "
              f"|>2.0|={(_abs_lr > 2.0).float().mean().item():.3f} "
              f"clip_frac={_clipped:.3f}")
        # Show per-head breakdown
        _n = flat_result.log_probs.shape[0]
        _new_lps = flat_result.log_probs.detach()
        _old_lps = flat_old_log_probs.detach()
        print(f"  [DIAG] new_lp: mean={_new_lps.mean().item():.4f} std={_new_lps.std().item():.4f} "
              f"min={_new_lps.min().item():.4f} max={_new_lps.max().item():.4f}")
        print(f"  [DIAG] old_lp: mean={_old_lps.mean().item():.4f} std={_old_lps.std().item():.4f} "
              f"min={_old_lps.min().item():.4f} max={_old_lps.max().item():.4f}")
    log_ratio = log_ratio.clamp(-5.0, 5.0)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * flat_advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * flat_advantages
    mean_policy_loss = (-torch.min(surr1, surr2)).mean()

    # Clip fraction: how often the ratio was clipped
    clip_frac = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()

    # Value loss
    mean_value_loss = ((flat_result.values - flat_returns) ** 2).mean()

    # Per-phase value loss (Step 5): aux regressors trained against the same
    # flat_returns target as the main value head. Only attached when the
    # phase-reencode path populated them — otherwise stays at 0.
    per_phase_value_loss_val = 0.0
    # Per-phase breakdown for logging (diagnostic — not used by training).
    # Empty when flag is off. When on, two N_PHASES-long lists:
    #   per_phase_value_mean_per_phase[p] = mean V output at phase p across batch
    #   per_phase_value_loss_per_phase[p] = MSE between V@p and returns
    per_phase_value_mean_per_phase: list[float] = []
    per_phase_value_loss_per_phase: list[float] = []
    if flat_result.per_phase_values is not None:
        # flat_returns: (N,); per_phase_values: (N, N_PHASES)
        _ppv_target = flat_returns.detach().unsqueeze(-1)
        _pp_diff_sq = (flat_result.per_phase_values - _ppv_target) ** 2
        mean_per_phase_value_loss = _pp_diff_sq.mean()
        per_phase_value_loss_val = mean_per_phase_value_loss.item()
        with torch.no_grad():
            per_phase_value_mean_per_phase = (
                flat_result.per_phase_values.mean(dim=0).tolist()
            )
            per_phase_value_loss_per_phase = _pp_diff_sq.mean(dim=0).tolist()
    else:
        mean_per_phase_value_loss = None

    # Entropy (aggregate + per-head)
    mean_entropy = flat_result.entropies.mean()
    per_head_entropy = {}
    per_head_n = {}
    if flat_result.unit_entropies is not None:
        # Conditional heads are averaged only over samples where the head is active,
        # matching the alpha tuner's view (entropy.py). Unit is active for every
        # sample; move is active only when ≥1 enemy is chargeable and the unit
        # isn't shaken (otherwise charge is masked and entropy is forced to 0,
        # which would deflate the logged value below the tuner's effective
        # comparison). per_head_n carries the effective sample count so
        # minibatch aggregation can weight correctly.
        is_move = flat_result.is_move
        is_can_shoot = flat_result.is_can_shoot
        is_charge_m = flat_result.is_charge
        is_cc_any = flat_result.is_can_charge_any
        n_total = flat_result.unit_entropies.shape[0]
        for name, ent_t, gate in [
            ("unit", flat_result.unit_entropies, None),
            ("move", flat_result.move_entropies, is_cc_any),
            ("dest", flat_result.dest_entropies, is_move),
            ("charge", flat_result.charge_entropies, is_charge_m),
            ("shoot", flat_result.shoot_entropies, is_can_shoot),
        ]:
            if ent_t is None:
                per_head_entropy[name] = 0.0
                per_head_n[name] = 0
            elif gate is None:
                per_head_entropy[name] = ent_t.mean().item()
                per_head_n[name] = n_total
            else:
                n_gate = int(gate.sum().item())
                if n_gate == 0:
                    per_head_entropy[name] = 0.0
                    per_head_n[name] = 0
                else:
                    per_head_entropy[name] = ((ent_t * gate).sum() / n_gate).item()
                    per_head_n[name] = n_gate
    alpha_loss_val = 0.0

    if entropy_tuner is not None and flat_result.unit_entropies is not None:
        # Per-head adaptive entropy bonus
        entropy_bonus = entropy_tuner.compute_entropy_bonus(
            flat_result.unit_entropies,
            flat_result.move_entropies,
            flat_result.dest_entropies,
            flat_result.charge_entropies,
            flat_result.shoot_entropies,
            flat_result.is_move,
            flat_result.is_can_shoot,
            flat_result.is_charge,
            is_can_charge_any=flat_result.is_can_charge_any,
        )
        loss = mean_policy_loss + value_coeff * mean_value_loss - entropy_bonus

        # Alpha loss (caller backprops this separately through the alpha optimizer)
        alpha_loss = entropy_tuner.compute_alpha_loss(
            flat_result.unit_entropies,
            flat_result.move_entropies,
            flat_result.dest_entropies,
            flat_result.charge_entropies,
            flat_result.shoot_entropies,
            flat_result.is_move,
            flat_result.is_can_shoot,
            flat_result.is_charge,
            flat_result.alive_mask,
            flat_result.enemy_alive_mask,
            flat_result.shoot_mask,
            dest_n_valid=flat_result.dest_n_valid,
            is_can_charge_any=flat_result.is_can_charge_any,
        )
        alpha_loss_val = alpha_loss.item()
    else:
        # Legacy single-coefficient path
        loss = mean_policy_loss + value_coeff * mean_value_loss - entropy_coeff * mean_entropy
        alpha_loss = None

    # --- Per-phase value head auxiliary loss (phase-reencode only) ---
    if mean_per_phase_value_loss is not None:
        loss = loss + _PER_PHASE_V_WEIGHT * mean_per_phase_value_loss

    # --- Auxiliary prediction losses (adaptive coefficient) ---
    aux_loss_val = 0.0
    effective_aux_coeff = 0.0
    if (aux_coeff > 0
            and flat_steps is not None
            and flat_result.aux_friendly_surv_alpha is not None):
        _aux_loss = _compute_aux_loss(flat_result, flat_steps)
        if _aux_loss is not None:
            aux_loss_val = _aux_loss.item()
            policy_mag = abs(loss.item())
            raw_aux_mag = abs(aux_loss_val)
            # Scale so aux contributes at most aux_ratio of policy magnitude
            effective_aux_coeff = aux_ratio * policy_mag / max(raw_aux_mag, 1e-6)
            effective_aux_coeff = min(effective_aux_coeff, aux_coeff)
            loss = loss + effective_aux_coeff * _aux_loss

    # --- Planning distillation loss ---
    # The reported `distill_loss_val` is the unramped scalar so the raw
    # signal is visible in logs; the gradient contribution is scaled by
    # `planning_distill_ramp` (0 during warmup, 0->1 over the ramp window).
    distill_loss_val = 0.0
    distill_sub_losses: dict[str, float] = {}
    distill_peaks: dict[str, float] = {}
    if (planning_distill_max_weight > 0
            and planning_distill_ramp > 0
            and flat_steps is not None):
        _distill = _compute_planning_distill_loss(
            flat_result, flat_steps, planning_distill_max_weight,
            mode=planning_distill_mode, eta=mpo_eta,
        )
        if _distill is not None:
            _distill_tensor, distill_sub_losses, distill_peaks = _distill
            distill_loss_val = _distill_tensor.item()
            loss = loss + planning_distill_ramp * _distill_tensor

    # --- KL trust region against π_old ---
    # Active only when β > 0 and the snapshot forward result is available
    # (the trainer materialises model_old only when β > 0). β scheduling
    # is handled in the trainer; this function consumes the resolved β.
    kl_trust_region_loss_val = 0.0
    kl_trust_per_head: dict[str, float] = {}
    if kl_trust_region_beta > 0 and flat_result_old is not None:
        _kl_total, kl_trust_per_head = _compute_kl_trust_region(
            flat_result, flat_result_old)
        kl_trust_region_loss_val = _kl_total.item()
        loss = loss + kl_trust_region_beta * _kl_total

    # --- Planning activation metrics ---
    planning_activations = 0
    planning_improvements = 0
    planning_argmax_best = 0
    planning_value_deltas: list[float] = []
    if flat_steps is not None:
        for s in flat_steps:
            if s.was_planned:
                planning_activations += 1
                if s.planning_improved:
                    planning_improvements += 1
                    planning_value_deltas.append(s.planning_value_delta)
                else:
                    planning_argmax_best += 1

    weighted_aux = effective_aux_coeff * aux_loss_val
    non_aux_loss = loss.item() - weighted_aux - planning_distill_ramp * distill_loss_val
    metrics = {
        "loss": loss.item(),
        "policy_loss": mean_policy_loss.item(),
        "value_loss": mean_value_loss.item(),
        "mean_entropy": mean_entropy.item(),
        "mean_reward": flat_result.total_reward / max(flat_result.n_episodes, 1),
        "mean_shoot_eff_reward": flat_result.total_shoot_eff_reward / max(flat_result.total_shoot_eff_count, 1),
        "mean_charge_eff_reward": flat_result.total_charge_eff_reward / max(flat_result.total_charge_eff_count, 1),
        "aux_loss": aux_loss_val,
        "weighted_aux": weighted_aux,
        "non_aux_loss": non_aux_loss,
        "per_phase_value_loss": per_phase_value_loss_val,
        "per_phase_value_mean_per_phase": per_phase_value_mean_per_phase,
        "per_phase_value_loss_per_phase": per_phase_value_loss_per_phase,
        # Scalar per-phase keys — the PPO minibatch accumulator in loop.py drops
        # any non-(int|float) values, so the list-valued keys above never reach
        # the CSV. These scalar mirrors survive the accumulation and are what
        # _format_per_phase_value actually reads. Only populated when the flag
        # is on; absent otherwise so the CSV emits empty strings for flag-off
        # runs (distinguishable from a genuine 0.0).
        **({
            "per_phase_value_mean_pre":  per_phase_value_mean_per_phase[0],
            "per_phase_value_mean_sel":  per_phase_value_mean_per_phase[1],
            "per_phase_value_mean_mt":   per_phase_value_mean_per_phase[2],
            "per_phase_value_mean_dest": per_phase_value_mean_per_phase[3],
            "per_phase_value_loss_pre":  per_phase_value_loss_per_phase[0],
            "per_phase_value_loss_sel":  per_phase_value_loss_per_phase[1],
            "per_phase_value_loss_mt":   per_phase_value_loss_per_phase[2],
            "per_phase_value_loss_dest": per_phase_value_loss_per_phase[3],
        } if len(per_phase_value_mean_per_phase) == 4 else {}),
        "clip_frac": clip_frac,
        "per_head_entropy": per_head_entropy,
        "per_head_n": per_head_n,
        "alpha_loss": alpha_loss_val,
        "_alpha_loss_tensor": alpha_loss,  # for backprop (not serialized)
        "per_opp_type_mean_values": flat_result.per_opp_type_mean_values or {},
        "planning_distill_loss": distill_loss_val,
        "planning_distill_ramp": planning_distill_ramp,
        "planning_distill_sub": distill_sub_losses,
        "planning_distill_peaks": distill_peaks,
        "planning_activations": planning_activations,
        "planning_improvement_rate": (
            planning_improvements / planning_activations
            if planning_activations > 0 else 0.0
        ),
        "planning_mean_value_delta": (
            sum(planning_value_deltas) / len(planning_value_deltas)
            if planning_value_deltas else 0.0
        ),
        "planning_argmax_rate": (
            planning_argmax_best / planning_activations
            if planning_activations > 0 else 0.0
        ),
        "kl_trust_region_loss": kl_trust_region_loss_val,
        "kl_trust_region_beta": kl_trust_region_beta,
        "kl_trust_per_head": kl_trust_per_head,
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Auxiliary losses
# ---------------------------------------------------------------------------

def _compute_aux_loss_horizon(
    flat_result: FlatReplayResult,
    flat_steps: list[TacticalActivationRecord],
    use_short: bool,
) -> torch.Tensor | None:
    """Compute auxiliary loss for one horizon (short = end-of-round, long = end-of-game).

    Each horizon uses its own dedicated prediction heads. Returns a scalar
    tensor, or None if no valid targets or head outputs are available.
    """
    # Select the correct head outputs for this horizon
    if use_short:
        fs_alpha_all = flat_result.aux_friendly_surv_alpha_short
        fs_beta_all = flat_result.aux_friendly_surv_beta_short
        es_alpha_all = flat_result.aux_enemy_surv_alpha_short
        es_beta_all = flat_result.aux_enemy_surv_beta_short
        obj_logits_all = flat_result.aux_obj_control_logits_short
    else:
        fs_alpha_all = flat_result.aux_friendly_surv_alpha
        fs_beta_all = flat_result.aux_friendly_surv_beta
        es_alpha_all = flat_result.aux_enemy_surv_alpha
        es_beta_all = flat_result.aux_enemy_surv_beta
        obj_logits_all = flat_result.aux_obj_control_logits

    if fs_alpha_all is None:
        return None

    losses: list[torch.Tensor] = []

    # --- Survival losses (Beta NLL) ---
    fs_targets = []
    es_targets = []
    valid_surv = []
    for i, s in enumerate(flat_steps):
        fs = s.friendly_survival_target_short if use_short else s.friendly_survival_target
        es = s.enemy_survival_target_short if use_short else s.enemy_survival_target
        if fs is not None and es is not None:
            fs_targets.append(fs)
            es_targets.append(es)
            valid_surv.append(i)

    if valid_surv:
        idx = torch.tensor(valid_surv, dtype=torch.long)
        fs_t = torch.tensor(fs_targets, dtype=torch.float32)  # (M, 10)
        es_t = torch.tensor(es_targets, dtype=torch.float32)  # (M, 10)

        eps = 1e-3
        fs_t = fs_t.clamp(eps, 1.0 - eps)
        es_t = es_t.clamp(eps, 1.0 - eps)

        fs_alpha = fs_alpha_all[idx]  # (M, 10)
        fs_beta = fs_beta_all[idx]
        fs_dist = torch.distributions.Beta(fs_alpha.clamp(max=100.0), fs_beta.clamp(max=100.0))
        fs_nll = -fs_dist.log_prob(fs_t).mean()
        losses.append(fs_nll)

        es_alpha = es_alpha_all[idx]
        es_beta = es_beta_all[idx]
        es_dist = torch.distributions.Beta(es_alpha.clamp(max=100.0), es_beta.clamp(max=100.0))
        es_nll = -es_dist.log_prob(es_t).mean()
        losses.append(es_nll)

    # --- Objective control loss (cross-entropy) ---
    obj_targets = []
    valid_obj = []
    for i, s in enumerate(flat_steps):
        obj = s.obj_control_target_short if use_short else s.obj_control_target
        if obj is not None:
            obj_targets.append(obj)
            valid_obj.append(i)

    if valid_obj and obj_logits_all is not None:
        idx = torch.tensor(valid_obj, dtype=torch.long)
        obj_t = torch.tensor(obj_targets, dtype=torch.long)  # (M, 5)
        obj_logits = obj_logits_all[idx]  # (M, 5, 3)
        obj_ce = F.cross_entropy(obj_logits.reshape(-1, 3), obj_t.reshape(-1))
        losses.append(obj_ce)

    if not losses:
        return None
    return torch.stack(losses).mean()


def _compute_countdown_loss(
    flat_result: FlatReplayResult,
    flat_steps: list[TacticalActivationRecord],
) -> torch.Tensor | None:
    """MSE loss for activation countdown prediction heads.

    Targets are remaining friendly/enemy activations until game end,
    backfilled from the trajectory during collection.
    """
    if flat_result.aux_friendly_act_remaining is None:
        return None

    f_targets = []
    e_targets = []
    valid = []
    for i, s in enumerate(flat_steps):
        if s.friendly_activations_remaining is not None and s.enemy_activations_remaining is not None:
            f_targets.append(s.friendly_activations_remaining)
            e_targets.append(s.enemy_activations_remaining)
            valid.append(i)

    if not valid:
        return None

    idx = torch.tensor(valid, dtype=torch.long)
    f_t = torch.tensor(f_targets, dtype=torch.float32)
    e_t = torch.tensor(e_targets, dtype=torch.float32)

    f_pred = flat_result.aux_friendly_act_remaining[idx]
    e_pred = flat_result.aux_enemy_act_remaining[idx]

    f_mse = F.mse_loss(f_pred, f_t)
    e_mse = F.mse_loss(e_pred, e_t)
    return (f_mse + e_mse) * 0.5


def _compute_aux_loss(
    flat_result: FlatReplayResult,
    flat_steps: list[TacticalActivationRecord],
) -> torch.Tensor | None:
    """Compute combined auxiliary loss from all auxiliary heads.

    Includes short-horizon, long-horizon, and activation countdown losses.
    Each head's gradient is independent.

    Returns a scalar tensor, or None if no valid targets are available.
    """
    short_loss = _compute_aux_loss_horizon(flat_result, flat_steps, use_short=True)
    long_loss = _compute_aux_loss_horizon(flat_result, flat_steps, use_short=False)
    countdown_loss = _compute_countdown_loss(flat_result, flat_steps)

    losses = [l for l in (short_loss, long_loss, countdown_loss) if l is not None]
    if not losses:
        return None
    return torch.stack(losses).sum()


# ---------------------------------------------------------------------------
# Planning distillation loss
# ---------------------------------------------------------------------------

def _compute_kl_trust_region(
    flat_result: FlatReplayResult,
    flat_result_old: FlatReplayResult,
) -> tuple[torch.Tensor, dict]:
    """Per-head KL(π || π_old) summed across heads, gated by activation.

    π_old logits come from a frozen snapshot model (`model_old` in the
    trainer), refreshed once per outer batch iteration. The KL is
    averaged within each active head — using the same is_move /
    is_charge / is_can_shoot gates as the entropy tracker — and then
    summed across heads so per-head magnitudes stay commensurate.

    Returns (total_kl, per_head_kl_dict).
    """
    if flat_result.unit_logits is None or flat_result_old.unit_logits is None:
        device = flat_result.values.device
        return torch.zeros((), device=device), {}

    device = flat_result.unit_logits.device
    total = torch.zeros((), device=device)
    per_head: dict[str, float] = {}

    def _kl_per_row(logits: torch.Tensor, logits_old: torch.Tensor) -> torch.Tensor:
        # KL(π || π_old) row-wise. -inf logits come from identical legality
        # masks in both π and π_old, so log_p and log_p_old are both -inf at
        # those slots. Computing log_p - log_p_old directly gives NaN
        # (-inf − -inf), which contaminates gradients even when the forward
        # value is later cleaned with nan_to_num — the backward through the
        # NaN-producing op leaks NaN gradients into the logits. Use
        # torch.where to gate the subtraction at masked slots: at safe
        # slots compute the real KL term; at masked slots return 0 with a
        # zero-gradient branch.
        log_p = F.log_softmax(logits, dim=-1)
        log_p_old = F.log_softmax(logits_old, dim=-1)
        safe = torch.isfinite(log_p) & torch.isfinite(log_p_old)
        log_p_safe = torch.where(safe, log_p, torch.zeros_like(log_p))
        log_p_old_safe = torch.where(safe, log_p_old, torch.zeros_like(log_p_old))
        # At unsafe slots: log_p_safe.exp() = 1, (0 - 0) = 0 → contribution 0.
        # At safe slots: standard KL term. The torch.where above zeros the
        # gradient through log_p at unsafe positions.
        kl_terms = log_p_safe.exp() * (log_p_safe - log_p_old_safe)
        return kl_terms.sum(dim=-1)

    # Unit and move heads — always active.
    kl_unit = _kl_per_row(flat_result.unit_logits, flat_result_old.unit_logits).mean()
    total = total + kl_unit
    per_head["unit"] = kl_unit.item()

    if flat_result.move_logits is not None and flat_result_old.move_logits is not None:
        kl_move = _kl_per_row(flat_result.move_logits, flat_result_old.move_logits).mean()
        total = total + kl_move
        per_head["move"] = kl_move.item()

    # Conditional heads — gated by the same flags the entropy tuner uses.
    def _add_gated(
        logits: torch.Tensor | None, logits_old: torch.Tensor | None,
        gate: torch.Tensor | None, name: str,
    ) -> None:
        nonlocal total
        if logits is None or logits_old is None or gate is None:
            return
        n_gate = int(gate.sum().item())
        if n_gate == 0:
            return
        kl_per = _kl_per_row(logits, logits_old)
        kl_avg = (kl_per * gate.float()).sum() / n_gate
        total = total + kl_avg
        per_head[name] = kl_avg.item()

    _add_gated(flat_result.dest_logits, flat_result_old.dest_logits,
               flat_result.is_move, "dest")
    _add_gated(flat_result.charge_logits, flat_result_old.charge_logits,
               flat_result.is_charge, "charge")
    _add_gated(flat_result.shoot_logits, flat_result_old.shoot_logits,
               flat_result.is_can_shoot, "shoot")

    return total, per_head


def _kl_from_soft_target(
    target: torch.Tensor, log_probs: torch.Tensor,
) -> torch.Tensor:
    """KL(target || policy) with NaN/Inf-safe handling.

    Args:
        target: (P, K) soft target probabilities (0 for unevaluated slots)
        log_probs: (P, K) policy log-probabilities
    Returns:
        (P,) per-example KL divergence
    """
    raw_kl = target * (target.clamp(min=1e-8).log() - log_probs)
    return torch.nan_to_num(raw_kl, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=-1)


def _compute_planning_distill_loss(
    flat_result: FlatReplayResult,
    flat_steps: list[TacticalActivationRecord],
    max_weight: float,
    mode: str = "ce_chosen",
    eta: float = 1.0,
) -> tuple[torch.Tensor, dict, dict] | None:
    """Distillation loss from planned activations across all heads.

    Modes:
      "ce_chosen"    — hard cross-entropy on the planner's chosen candidate
                       index per head (one-hot through the KL machinery;
                       KL(one-hot || policy) == CE). Gated on planning_improved.
                       Weight = min(planning_value_delta, max_weight).
      "soft_kl"      — soft KL against softmax(per-candidate V), η=1.0 implicit
                       (η param ignored). Gated on planning_improved. Weight =
                       min(planning_value_delta, max_weight). Legacy.
      "mpo_marginal" — MPO marginalised loss. Soft KL against softmax(V/η),
                       NOT gated on planning_improved (every planned step
                       contributes). Per-step weight is uniform (1.0); the
                       softmax(V/η) target alone shapes the gradient.

    Returns (total_loss, sub_loss_dict, peaks_dict) or None if no eligible
    steps. peaks_dict has keys {tgt,pol}_<head> and agree_<head>; in
    "ce_chosen" mode the target is one-hot so tgt_<head> ≈ 1.0 and
    agree_<head> is the metric of interest.
    """
    if flat_result.unit_logits is None:
        return None

    # --- Collect per-step data for each head ---
    # Unit head
    unit_indices, unit_targets, unit_weights = [], [], []
    # Move type head
    move_indices, move_targets, move_weights = [], [], []
    # Charge target head
    charge_indices, charge_targets, charge_weights = [], [], []
    # Shoot target head
    shoot_indices, shoot_targets, shoot_weights = [], [], []
    # Destination pointer head — variable width; the model's dest_logits are
    # padded to the minibatch-local max candidate count, so we store raw
    # (idx, val) pairs here and build the soft target tensor after we know
    # that width.
    dest_indices: list[int] = []
    dest_sparse_targets: list[list[tuple[int, float]]] = []
    dest_weights: list[float] = []

    # Dest-mode-specific buffer: in ce_chosen we only need the per-step
    # selected-candidate index, not the (idx, val) sparse pairs.
    dest_selected: list[int] = []

    for i, s in enumerate(flat_steps):
        # Mode-dependent gate and per-step weight.
        # Legacy modes preserve the original (gate, weight) pair bit-exactly.
        if mode == "mpo_marginal":
            # Every planned step contributes — per-candidate weighting
            # comes from softmax(V/η) inside the target distribution, not
            # from planning_value_delta, so we don't apply value-delta
            # scaling here. We do use max_weight as a uniform per-step
            # scale so `planning_distill_max_weight` keeps its intuitive
            # role as the overall cap on distill influence (otherwise the
            # mpo_marginal contribution would be ~1/max_weight times the
            # legacy magnitude and would dominate the PPO surrogate).
            if not s.was_planned:
                continue
            w = max_weight
        else:
            if not s.was_planned or not s.planning_improved:
                continue
            w = min(s.planning_value_delta, max_weight)

        if mode == "ce_chosen":
            # --- Unit head: one-hot at chosen unit ---
            if 0 <= s.unit_idx < MAX_UNITS_PER_SIDE:
                target = torch.zeros(MAX_UNITS_PER_SIDE)
                target[s.unit_idx] = 1.0
                unit_indices.append(i)
                unit_targets.append(target)
                unit_weights.append(w)

            # --- Move head: one-hot at chosen move type ---
            if 0 <= s.move_type < NUM_MOVE_TYPES:
                target = torch.zeros(NUM_MOVE_TYPES)
                target[s.move_type] = 1.0
                move_indices.append(i)
                move_targets.append(target)
                move_weights.append(w)

            # --- Charge target head: only when chosen action is charge ---
            if (s.move_type == MOVE_CHARGE
                    and 0 <= s.charge_target_idx < MAX_UNITS_PER_SIDE):
                target = torch.zeros(MAX_UNITS_PER_SIDE)
                target[s.charge_target_idx] = 1.0
                charge_indices.append(i)
                charge_targets.append(target)
                charge_weights.append(w)

            # --- Shoot target head: only when chosen move shoots ---
            if (s.move_type == MOVE_MOVE
                    and 0 <= s.shoot_target_idx < MAX_UNITS_PER_SIDE):
                target = torch.zeros(MAX_UNITS_PER_SIDE)
                target[s.shoot_target_idx] = 1.0
                shoot_indices.append(i)
                shoot_targets.append(target)
                shoot_weights.append(w)

            # --- Dest head: only when chosen action moves to a valid dest ---
            if s.move_type == MOVE_MOVE and s.dest_selected_idx >= 0:
                dest_indices.append(i)
                dest_selected.append(s.dest_selected_idx)
                dest_weights.append(w)

        else:  # mode in ("soft_kl", "mpo_marginal") — soft target via softmax(V/η)
            # --- Unit head ---
            if s.planning_unit_values is not None and s.planning_unit_indices is not None:
                target = torch.full((MAX_UNITS_PER_SIDE,), float('-inf'))
                for idx, val in zip(s.planning_unit_indices, s.planning_unit_values):
                    target[idx] = val
                target = torch.softmax(target / eta, dim=0)
                unit_indices.append(i)
                unit_targets.append(target)
                unit_weights.append(w)

            # --- Move type head: ≥2 distinct move types for chosen unit ---
            if (s.planning_move_values is not None
                    and s.planning_move_indices is not None
                    and len(s.planning_move_indices) >= 2):
                target = torch.full((NUM_MOVE_TYPES,), float('-inf'))
                for idx, val in zip(s.planning_move_indices, s.planning_move_values):
                    target[idx] = val
                target = torch.softmax(target / eta, dim=0)
                move_indices.append(i)
                move_targets.append(target)
                move_weights.append(w)

            # --- Charge target head: chosen move is charge AND ≥2 distinct targets ---
            if (s.move_type == MOVE_CHARGE
                    and s.planning_charge_values is not None
                    and s.planning_charge_indices is not None
                    and len(s.planning_charge_indices) >= 2):
                target = torch.full((MAX_UNITS_PER_SIDE,), float('-inf'))
                for idx, val in zip(s.planning_charge_indices, s.planning_charge_values):
                    target[idx] = val
                target = torch.softmax(target / eta, dim=0)
                charge_indices.append(i)
                charge_targets.append(target)
                charge_weights.append(w)

            # --- Shoot target head: chosen move can shoot AND ≥2 distinct targets ---
            if (s.move_type == MOVE_MOVE
                    and s.planning_shoot_values is not None
                    and s.planning_shoot_indices is not None
                    and len(s.planning_shoot_indices) >= 2):
                target = torch.full((MAX_UNITS_PER_SIDE,), float('-inf'))
                for idx, val in zip(s.planning_shoot_indices, s.planning_shoot_values):
                    target[idx] = val
                target = torch.softmax(target / eta, dim=0)
                shoot_indices.append(i)
                shoot_targets.append(target)
                shoot_weights.append(w)

            # --- Destination pointer head: chosen move is MOVE AND ≥2 distinct dests ---
            if (s.move_type == MOVE_MOVE
                    and s.planning_dest_values is not None
                    and s.planning_dest_indices is not None
                    and len(s.planning_dest_indices) >= 2):
                dest_indices.append(i)
                dest_sparse_targets.append(list(zip(
                    s.planning_dest_indices, s.planning_dest_values)))
                dest_weights.append(w)

    if (not unit_indices and not move_indices and not charge_indices
            and not shoot_indices and not dest_indices):
        return None

    device = flat_result.unit_logits.device
    total_loss = torch.tensor(0.0, device=device)
    sub_losses: dict[str, float] = {}
    peaks: dict[str, float] = {}

    def _record_peaks(head: str, tgt: torch.Tensor, log_p: torch.Tensor) -> None:
        # Mean of per-row max probability for both the soft distill target
        # and the policy output. Compared against 1/N (uniform) externally.
        peaks[f"tgt_{head}"] = tgt.max(dim=-1).values.mean().item()
        peaks[f"pol_{head}"] = log_p.exp().max(dim=-1).values.mean().item()
        # Fraction of activations where the policy's argmax index matches
        # the target's argmax index. Separates "is the peak sharp enough"
        # from "is the peak on the right index." Low agreement + reasonable
        # peaks => policy is confidently picking the wrong option.
        peaks[f"agree_{head}"] = (
            tgt.argmax(dim=-1) == log_p.argmax(dim=-1)
        ).float().mean().item()

    # --- Unit selection KL ---
    if unit_indices:
        idx_t = torch.tensor(unit_indices, dtype=torch.long, device=device)
        tgt = torch.stack(unit_targets).to(device)
        w_t = torch.tensor(unit_weights, dtype=torch.float32, device=device)
        log_p = F.log_softmax(flat_result.unit_logits[idx_t], dim=-1)
        kl = _kl_from_soft_target(tgt, log_p)
        unit_loss = (w_t * kl).mean()
        total_loss = total_loss + unit_loss
        sub_losses["unit"] = unit_loss.item()
        _record_peaks("unit", tgt, log_p)

    # --- Move type KL ---
    if move_indices and flat_result.move_logits is not None:
        idx_t = torch.tensor(move_indices, dtype=torch.long, device=device)
        tgt = torch.stack(move_targets).to(device)
        w_t = torch.tensor(move_weights, dtype=torch.float32, device=device)
        log_p = F.log_softmax(flat_result.move_logits[idx_t], dim=-1)
        kl = _kl_from_soft_target(tgt, log_p)
        move_loss = (w_t * kl).mean()
        total_loss = total_loss + move_loss
        sub_losses["move"] = move_loss.item()
        _record_peaks("move", tgt, log_p)

    # --- Charge target KL ---
    if charge_indices and flat_result.charge_logits is not None:
        idx_t = torch.tensor(charge_indices, dtype=torch.long, device=device)
        tgt = torch.stack(charge_targets).to(device)
        w_t = torch.tensor(charge_weights, dtype=torch.float32, device=device)
        log_p = F.log_softmax(flat_result.charge_logits[idx_t], dim=-1)
        kl = _kl_from_soft_target(tgt, log_p)
        charge_loss = (w_t * kl).mean()
        total_loss = total_loss + charge_loss
        sub_losses["charge"] = charge_loss.item()
        _record_peaks("charge", tgt, log_p)

    # --- Shoot target KL ---
    if shoot_indices and flat_result.shoot_logits is not None:
        idx_t = torch.tensor(shoot_indices, dtype=torch.long, device=device)
        tgt = torch.stack(shoot_targets).to(device)
        w_t = torch.tensor(shoot_weights, dtype=torch.float32, device=device)
        log_p = F.log_softmax(flat_result.shoot_logits[idx_t], dim=-1)
        kl = _kl_from_soft_target(tgt, log_p)
        shoot_loss = (w_t * kl).mean()
        total_loss = total_loss + shoot_loss
        sub_losses["shoot"] = shoot_loss.item()
        _record_peaks("shoot", tgt, log_p)

    # --- Destination pointer KL ---
    # The dest head width is minibatch-local (padded to max valid candidates
    # across the minibatch). Build soft targets at that width; positions with
    # no planner sample get -inf (→ zero probability after softmax), and
    # positions beyond a given step's dest_n_valid are masked by the model's
    # own -inf, so KL stays well-behaved across padding.
    if dest_indices and flat_result.dest_logits is not None:
        idx_t = torch.tensor(dest_indices, dtype=torch.long, device=device)
        dest_logits_sel = flat_result.dest_logits[idx_t]  # (P, mb_max_cands)
        P, mb_w = dest_logits_sel.shape
        if mode == "ce_chosen":
            tgt = torch.zeros((P, mb_w), device=device)
            for row, sel_idx in enumerate(dest_selected):
                if 0 <= sel_idx < mb_w:
                    tgt[row, sel_idx] = 1.0
        else:
            tgt = torch.full((P, mb_w), float('-inf'), device=device)
            for row, pairs in enumerate(dest_sparse_targets):
                for d_idx, d_val in pairs:
                    if 0 <= d_idx < mb_w:
                        tgt[row, d_idx] = d_val
            tgt = torch.softmax(tgt / eta, dim=-1)
        w_t = torch.tensor(dest_weights, dtype=torch.float32, device=device)
        log_p = F.log_softmax(dest_logits_sel, dim=-1)
        kl = _kl_from_soft_target(tgt, log_p)
        dest_loss = (w_t * kl).mean()
        total_loss = total_loss + dest_loss
        sub_losses["dest"] = dest_loss.item()
        _record_peaks("dest", tgt, log_p)

    return total_loss, sub_losses, peaks
