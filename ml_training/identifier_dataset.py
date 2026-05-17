"""Phase 1: labeled dataset generator for the gap identifier.

Self-plays games with the frozen policy from final_model.pt, samples decision
states, draws stratified candidate actions per state, and labels each with
rollout-based Q and policy log-probability under the frozen policy.

Output: sharded .npz files in --output-dir, plus a manifest.json tracking
progress for resumability.

Per (s, a) record contains:
  state-level
    state_vec (4016,) float32, alive masks, round, player_is_a
  candidate-level
    discrete action indices (unit, move_type, charge_target, shoot_target)
    advance_reachable bit, head-active flags
    dest_features (76,) float32 for the chosen hex (zeros for charge actions)
    log_pi (frozen policy), Q (mean of M rollouts of player-action +
    opponent-activation + V at resulting state)

The Q label is in the side-to-move's perspective.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as _queue
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Make project-root scripts importable when invoking via `python -m
# ml_training.identifier_dataset` (the probe lives at project root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_features import (
    MAX_UNITS_PER_SIDE,
    encode_state_tactical,
    extract_can_charge_mask,
    extract_is_shaken,
)
from ml_integration_tactical import (
    _flip_x, _flip_y,
    _get_model_space_positions,
    compute_in_range_mask,
    compute_post_move_rel,
)
from ml_planning import (
    _build_masks,
    restore_game_state,
)
from ml_model_tactical import MOVE_CHARGE, MOVE_MOVE

# Reuse helpers from the Phase 0 probe (load_frozen_model, the rollout-based
# Q labeller, the policy log-pi computer, state collection, etc.). Keeping
# them in one place avoids divergence.
import probe_identifier_premise as probe  # type: ignore  # noqa: E402

DEFAULT_OUTPUT_DIR = "ml_training/identifier_data"
DEST_FEATURE_DIM = 76  # matches ml_features.DEST_FEATURE_DIM
STATE_VEC_DIM = 4016  # matches ml_features encoder output


# ---------------------------------------------------------------------------
# Stratified candidate sampling
# ---------------------------------------------------------------------------

def stratified_sample_candidates(
    my_units, opp_units, board, player: str,
    K: int, rng: random.Random,
) -> list[probe.CandidateAction]:
    """Sample up to K candidate actions stratified across the head structure.

    The 5 model heads are unit, move_type, dest, charge_target, shoot_target.
    Naive enumeration is dominated by destinations (50:1 over other heads).
    Stratification: per (alive_unit, action_type) bucket, allocate K/N_buckets
    samples; within a bucket, sample destinations and shoot targets randomly.

    Action types per unit:
        - "move":   move_type=MOVE_MOVE, sample dest from valid + optional shoot
        - "charge": move_type=MOVE_CHARGE, sample charge_target from chargeable

    Shaken units are skipped (their only legal action is hold-to-recover, which
    has no head structure for the identifier to exploit).
    """
    # Encode once for mask extraction (round_num doesn't affect can_charge/shaken)
    state_vec_for_masks = encode_state_tactical(my_units, opp_units, 1, board, player)

    buckets: list[dict] = []
    # The model's unit head is hard-sized to MAX_UNITS_PER_SIDE (10). Random
    # armies very occasionally generate >10 units (~0.5% of generations), and
    # without this cap the sampler emits candidates with unit_idx >= 10 that
    # crash compute_log_pi when it indexes the (10,)-shaped logits. The
    # policy itself can't act on those units anyway (argmax is over slots
    # 0..9), so dropping them here matches what _argmax_decision already does.
    for unit_idx, unit in enumerate(my_units[:MAX_UNITS_PER_SIDE]):
        if unit.models_alive <= 0 or unit.activated:
            continue
        if bool(extract_is_shaken(state_vec_for_masks, unit_idx).item()):
            continue

        cand_arr, mask, adv, feats = probe._per_unit_dest_arrays(
            unit, opp_units, board, player)
        valid_dest = np.where(mask)[0]
        if len(valid_dest) > 0:
            buckets.append({
                "type": "move", "unit_idx": unit_idx,
                "cand_arr": cand_arr, "mask": mask,
                "adv": adv, "feats": feats,
                "valid_dest": valid_dest,
            })

        can_charge = extract_can_charge_mask(state_vec_for_masks, unit_idx)
        if can_charge.any():
            chargeable = [
                c for c in range(MAX_UNITS_PER_SIDE)
                if can_charge[c]
                and c < len(opp_units)
                and opp_units[c].models_alive > 0
            ]
            if chargeable:
                buckets.append({
                    "type": "charge", "unit_idx": unit_idx,
                    "chargeable": chargeable,
                })

    if not buckets:
        return []

    # Allocate K samples uniformly across buckets
    base, remainder = divmod(K, len(buckets))
    quotas = [base + (1 if i < remainder else 0) for i in range(len(buckets))]

    # Catalog of alive enemies for shoot-target sampling
    alive_enemies = [
        i for i, e in enumerate(opp_units)
        if e.models_alive > 0 and i < MAX_UNITS_PER_SIDE
    ]

    candidates: list[probe.CandidateAction] = []
    for bucket, quota in zip(buckets, quotas):
        if bucket["type"] == "move":
            valid_dest_list = bucket["valid_dest"].tolist()
            cand_arr = bucket["cand_arr"]
            feats = bucket["feats"]
            mask = bucket["mask"]
            adv = bucket["adv"]

            # Full enumeration per user spec: every move candidate carries a
            # shoot target. For unreachable targets the in-range mask zeros
            # out the shoot head's contribution at log-pi time, and the game
            # engine just doesn't fire — so shoot_active is a derived flag,
            # not a sampling decision.
            for _ in range(quota):
                d_idx = int(rng.choice(valid_dest_list))
                shoot_t = int(rng.choice(alive_enemies)) if alive_enemies else 0
                candidates.append(probe.CandidateAction(
                    unit_idx=bucket["unit_idx"],
                    move_type=MOVE_MOVE,
                    dest_idx=d_idx,
                    dest_col=int(cand_arr[d_idx, 0]),
                    dest_row=int(cand_arr[d_idx, 1]),
                    charge_target_idx=0,
                    shoot_target_idx=shoot_t,
                    is_shaken=False,
                    advance_reachable=bool(adv[d_idx]),
                    dest_candidates=cand_arr,
                    dest_features=feats,
                    dest_mask=mask,
                ))
        else:  # charge
            chargeable = bucket["chargeable"]
            # No replacement — at most len(chargeable) distinct charge targets.
            # For quota > len(chargeable), repeat (the shaken-state, dice
            # variance still gives non-redundant rollouts at label time).
            for _ in range(quota):
                c_idx = int(rng.choice(chargeable))
                candidates.append(probe.CandidateAction(
                    unit_idx=bucket["unit_idx"],
                    move_type=MOVE_CHARGE,
                    dest_idx=-1, dest_col=None, dest_row=None,
                    charge_target_idx=c_idx, shoot_target_idx=0,
                    is_shaken=False, advance_reachable=False,
                    dest_candidates=None, dest_features=None, dest_mask=None,
                ))

    return candidates


# ---------------------------------------------------------------------------
# Chunk packing / writing
# ---------------------------------------------------------------------------

@dataclass
class _LabeledState:
    state_vec: np.ndarray            # (4016,) float32
    alive_mask: np.ndarray           # (10,) bool
    enemy_alive_mask: np.ndarray     # (10,) bool
    round_num: int
    player_is_a: bool
    game_uid: int                    # globally unique per (worker, game); states
                                     # from the same game share this id, used
                                     # for episode-level train/val splits
    candidates: list                 # list[_LabeledCandidate]


@dataclass
class _LabeledCandidate:
    unit_idx: int
    move_type: int
    charge_target_idx: int
    shoot_target_idx: int
    advance_reachable: bool
    dest_active: bool
    charge_active: bool
    shoot_active: bool
    dest_features: np.ndarray  # (76,) float32 — zeros if not dest_active
    log_pi: float
    Q: float


def _pack_chunk(states: list[_LabeledState]) -> dict:
    """Concatenate per-state arrays and per-candidate arrays into one npz dict."""
    S = len(states)
    state_vecs = np.stack([s.state_vec for s in states], axis=0).astype(np.float32)
    alive = np.stack([s.alive_mask for s in states], axis=0).astype(np.bool_)
    enemy_alive = np.stack([s.enemy_alive_mask for s in states], axis=0).astype(np.bool_)
    rounds = np.array([s.round_num for s in states], dtype=np.int32)
    player_is_a = np.array([s.player_is_a for s in states], dtype=np.bool_)
    game_uid = np.array([s.game_uid for s in states], dtype=np.int64)
    n_cands = np.array([len(s.candidates) for s in states], dtype=np.int32)

    # Flatten candidates
    total_C = int(n_cands.sum())
    cand_state_idx = np.empty(total_C, dtype=np.int32)
    cand_unit_idx = np.empty(total_C, dtype=np.int8)
    cand_move_type = np.empty(total_C, dtype=np.int8)
    cand_charge = np.empty(total_C, dtype=np.int8)
    cand_shoot = np.empty(total_C, dtype=np.int8)
    cand_adv = np.empty(total_C, dtype=np.bool_)
    cand_dest_active = np.empty(total_C, dtype=np.bool_)
    cand_charge_active = np.empty(total_C, dtype=np.bool_)
    cand_shoot_active = np.empty(total_C, dtype=np.bool_)
    cand_dest_feats = np.zeros((total_C, DEST_FEATURE_DIM), dtype=np.float32)
    cand_log_pi = np.empty(total_C, dtype=np.float32)
    cand_Q = np.empty(total_C, dtype=np.float32)

    write = 0
    for s_idx, s in enumerate(states):
        for c in s.candidates:
            cand_state_idx[write] = s_idx
            cand_unit_idx[write] = c.unit_idx
            cand_move_type[write] = c.move_type
            cand_charge[write] = c.charge_target_idx
            cand_shoot[write] = c.shoot_target_idx
            cand_adv[write] = c.advance_reachable
            cand_dest_active[write] = c.dest_active
            cand_charge_active[write] = c.charge_active
            cand_shoot_active[write] = c.shoot_active
            if c.dest_active:
                cand_dest_feats[write] = c.dest_features
            cand_log_pi[write] = c.log_pi
            cand_Q[write] = c.Q
            write += 1

    return dict(
        state_vec=state_vecs,
        alive_mask=alive,
        enemy_alive_mask=enemy_alive,
        round_num=rounds,
        player_is_a=player_is_a,
        game_uid=game_uid,
        n_candidates=n_cands,
        cand_state_idx=cand_state_idx,
        cand_unit_idx=cand_unit_idx,
        cand_move_type=cand_move_type,
        cand_charge_target_idx=cand_charge,
        cand_shoot_target_idx=cand_shoot,
        cand_advance_reachable=cand_adv,
        cand_dest_active=cand_dest_active,
        cand_charge_active=cand_charge_active,
        cand_shoot_active=cand_shoot_active,
        cand_dest_features=cand_dest_feats,
        cand_log_pi=cand_log_pi,
        cand_Q=cand_Q,
    )


# ---------------------------------------------------------------------------
# Manifest (for resumability + audit)
# ---------------------------------------------------------------------------

def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.json"


def _load_manifest(out_dir: Path) -> dict:
    p = _manifest_path(out_dir)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "checkpoint": probe.CHECKPOINT_PATH,
        "chunks": [],
        "states_written": 0,
        "candidates_written": 0,
    }


def _save_manifest(out_dir: Path, manifest: dict) -> None:
    p = _manifest_path(out_dir)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(p)


# ---------------------------------------------------------------------------
# Per-state labelling — collects candidates and labels them
# ---------------------------------------------------------------------------

def _shoot_in_range(
    cand: probe.CandidateAction,
    my_units, opp_units, player: str,
    enemy_alive_mask_t: torch.Tensor,
) -> bool:
    """True iff the candidate's chosen shoot target is in weapon range from
    the post-move position, mirroring the policy's shoot-head masking."""
    enemy_positions_ms = _get_model_space_positions(opp_units, player)
    post_x = float(cand.dest_col)
    post_y = float(cand.dest_row)
    if player == "B":
        post_x = _flip_x(post_x)
        post_y = _flip_y(post_y)
    post_move_rel = compute_post_move_rel(post_x, post_y, enemy_positions_ms)

    unit = my_units[cand.unit_idx]
    max_wr = max(
        (w.range_inches for w in unit.unit.weapons if not w.melee),
        default=0.0,
    )
    in_range_mask = compute_in_range_mask(
        post_move_rel, float(max_wr), enemy_alive_mask_t,
    )
    return bool(in_range_mask[cand.shoot_target_idx].item())


def _label_one_state(
    model,
    decision_state: probe.DecisionState,
    ctx: probe.GameContext,
    candidates_per_state: int,
    m_rollouts: int,
    rng: random.Random,
    game_uid: int = 0,
) -> _LabeledState | None:
    """Restore the snapshot, sample candidates, and label each."""
    restore_game_state(decision_state.snap, ctx.units_a, ctx.units_b, ctx.board)

    my_units = ctx.units_a if decision_state.current_is_a else ctx.units_b
    opp_units = ctx.units_b if decision_state.current_is_a else ctx.units_a
    player = "A" if decision_state.current_is_a else "B"
    my_fr = ctx.fr_a if decision_state.current_is_a else ctx.fr_b
    my_fm = ctx.fm_a if decision_state.current_is_a else ctx.fm_b
    opp_fr = ctx.fr_b if decision_state.current_is_a else ctx.fr_a
    opp_fm = ctx.fm_b if decision_state.current_is_a else ctx.fm_a
    my_pts = ctx.pts_a if decision_state.current_is_a else ctx.pts_b
    opp_pts = ctx.pts_b if decision_state.current_is_a else ctx.pts_a

    cands = stratified_sample_candidates(
        my_units, opp_units, ctx.board, player,
        K=candidates_per_state, rng=rng,
    )
    if not cands:
        return None

    # Encode the state once for storage (full encoding with matchups + points)
    state_vec = encode_state_tactical(
        my_units, opp_units, decision_state.round_num, ctx.board, player,
        friendly_ranged_matchups=my_fr, friendly_melee_matchups=my_fm,
        enemy_ranged_matchups=opp_fr, enemy_melee_matchups=opp_fm,
        total_friendly_points=my_pts, total_enemy_points=opp_pts,
    ).numpy().astype(np.float32)

    alive_mask_t, enemy_alive_mask_t = _build_masks(my_units, opp_units)
    alive_mask_np = alive_mask_t.numpy().astype(np.bool_)
    enemy_alive_mask_np = enemy_alive_mask_t.numpy().astype(np.bool_)

    labeled_cands: list[_LabeledCandidate] = []
    for cand in cands:
        # Restore before each scoring (compute_Q mutates state)
        restore_game_state(decision_state.snap, ctx.units_a, ctx.units_b, ctx.board)
        log_pi = probe.compute_log_pi(
            model, cand, my_units, opp_units, ctx.board, player,
            decision_state.round_num,
            my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts,
        )
        # Skip impossible-under-policy actions (mask collisions) — rare but real
        if not np.isfinite(log_pi):
            continue
        Q = probe.compute_Q(
            model, cand, decision_state.snap, ctx,
            current_is_a=decision_state.current_is_a,
            round_num=decision_state.round_num,
            m_rollouts=m_rollouts,
        )

        # Head-active flags: charge_active and dest_active follow directly
        # from move_type. shoot_active requires the in-range check at the
        # post-move position — same logic the policy's shoot head applies.
        is_charge = (cand.move_type == MOVE_CHARGE)
        dest_active = (not is_charge) and cand.dest_features is not None
        charge_active = is_charge
        shoot_active = False
        if (not is_charge) and cand.advance_reachable:
            shoot_active = _shoot_in_range(
                cand, my_units, opp_units, player, enemy_alive_mask_t,
            )

        if dest_active:
            dest_feats = cand.dest_features[cand.dest_idx].astype(np.float32)
        else:
            dest_feats = np.zeros(DEST_FEATURE_DIM, dtype=np.float32)

        labeled_cands.append(_LabeledCandidate(
            unit_idx=cand.unit_idx,
            move_type=cand.move_type,
            charge_target_idx=cand.charge_target_idx,
            shoot_target_idx=cand.shoot_target_idx,
            advance_reachable=cand.advance_reachable,
            dest_active=dest_active,
            charge_active=charge_active,
            shoot_active=shoot_active,
            dest_features=dest_feats,
            log_pi=float(log_pi),
            Q=float(Q),
        ))

    if not labeled_cands:
        return None

    return _LabeledState(
        state_vec=state_vec,
        alive_mask=alive_mask_np,
        enemy_alive_mask=enemy_alive_mask_np,
        round_num=decision_state.round_num,
        player_is_a=decision_state.current_is_a,
        game_uid=game_uid,
        candidates=labeled_cands,
    )


# Stratified sampler ambiguity: the sampler stores shoot_target_idx=0 with
# shoot_active=False. We need to propagate that flag from CandidateAction to
# the labeled record. Patch CandidateAction with an explicit field via a
# parallel lookup — simpler: re-derive from the sampler's bookkeeping.
#
# Pragmatic fix: store shoot_active as a bit on CandidateAction. Below we
# monkey-patch the dataclass; if the probe's CandidateAction doesn't have the
# field, we fall back to the heuristic (idx > 0 implies active).


# ---------------------------------------------------------------------------
# Multiprocessing: embarrassingly-parallel workers
# ---------------------------------------------------------------------------

@dataclass
class _WorkerResult:
    """Item pushed onto the result queue by workers.

    state == None is a sentinel meaning "this worker has exited."
    """
    state: _LabeledState | None
    worker_id: int


def _worker_loop(
    worker_id: int,
    n_states_target_per_worker: int,
    candidates_per_state: int,
    m_rollouts: int,
    games_per_collection_batch: int,
    seed: int,
    checkpoint_path: str,
    result_queue: "mp.Queue",
    stop_event: "mp.Event",
):
    """Worker process body.

    Plays self-play games independently, samples candidates per state, labels
    them, and pushes _LabeledState records onto the result queue. Exits when
    its per-worker target is reached or stop_event is set.
    """
    # Each worker reduces thread count to avoid BLAS over-subscription
    # (default threadpool * n_workers = thrashing on a single laptop).
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    rng_seed = seed + worker_id * 10_000 + 1
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed % (2**32 - 1))
    torch.manual_seed(rng_seed)

    try:
        model = probe.load_frozen_model(checkpoint_path)
    except Exception as e:
        print(f"[worker {worker_id}] failed to load model: {e}", file=sys.stderr)
        result_queue.put(_WorkerResult(None, worker_id))
        return

    states_done = 0
    batch_idx = 0
    try:
        while states_done < n_states_target_per_worker and not stop_event.is_set():
            try:
                states, contexts = probe.collect_decision_states(
                    model, n_games=games_per_collection_batch,
                )
            except Exception as e:
                print(f"[worker {worker_id}] state-collection error: {e}",
                      file=sys.stderr)
                continue
            if not states:
                continue
            # Globally-unique game id, packed into 64 bits:
            #   worker_id (16 bits) | batch_idx (32 bits) | game_idx_in_batch (16 bits)
            # States from the same game share the same uid (st.game_idx is
            # constant within a batch for a given game), enabling
            # episode-level train/val splits at training time.
            rng.shuffle(states)
            for st in states:
                if states_done >= n_states_target_per_worker or stop_event.is_set():
                    break
                ctx = contexts[st.game_idx]
                game_uid = (
                    ((worker_id + 1) & 0xFFFF) << 48
                    | (batch_idx & 0xFFFFFFFF) << 16
                    | (st.game_idx & 0xFFFF)
                )
                try:
                    labeled = _label_one_state(
                        model, st, ctx,
                        candidates_per_state=candidates_per_state,
                        m_rollouts=m_rollouts, rng=rng,
                        game_uid=game_uid,
                    )
                except Exception as e:
                    import traceback
                    print(f"[worker {worker_id}] state-label error: {e}\n"
                          f"{traceback.format_exc()}",
                          file=sys.stderr)
                    continue
                if labeled is None:
                    continue
                # Pushing may block if the queue is full — main is consuming
                # in real time, so this is the natural backpressure mechanism.
                result_queue.put(_WorkerResult(labeled, worker_id))
                states_done += 1
            batch_idx += 1
    finally:
        # Sentinel — main process counts these to know when all workers done
        result_queue.put(_WorkerResult(None, worker_id))


def run_labeling_parallel(
    n_states_target: int,
    candidates_per_state: int,
    m_rollouts: int,
    output_dir: str,
    chunk_size: int,
    games_per_collection_batch: int,
    seed: int,
    checkpoint_path: str,
    n_workers: int,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(out_dir)
    if manifest.get("checkpoint") and manifest["checkpoint"] != checkpoint_path:
        raise RuntimeError(
            f"manifest checkpoint mismatch: "
            f"existing {manifest['checkpoint']}, requested {checkpoint_path}. "
            f"Remove {out_dir} or use a different output_dir."
        )
    manifest["checkpoint"] = checkpoint_path

    states_written = manifest["states_written"]
    candidates_written = manifest["candidates_written"]
    chunk_idx = len(manifest["chunks"])
    print(f"[label] resuming: {states_written} states already written "
          f"({len(manifest['chunks'])} chunks).")

    target_remaining = n_states_target - states_written
    if target_remaining <= 0:
        print(f"[label] target {n_states_target} already met.")
        return

    print(f"[label] launching {n_workers} workers, "
          f"need {target_remaining} more states.")

    # Distribute remaining target across workers (round up so they collectively
    # hit at least n_states_target — the main loop stops at the exact target
    # and signals stop_event so over-shoot is bounded).
    per_worker = (target_remaining + n_workers - 1) // n_workers

    # spawn ensures each worker reinitializes torch cleanly (no fork-after-CUDA
    # issues, no copy-on-write bookkeeping landmines).
    ctx = mp.get_context("spawn")
    # Bound the queue so workers block on put when main can't keep up — natural
    # backpressure when the disk-write rate dominates.
    result_queue: mp.Queue = ctx.Queue(maxsize=max(8, n_workers * 4))
    stop_event = ctx.Event()

    workers: list[mp.Process] = []
    for i in range(n_workers):
        p = ctx.Process(
            target=_worker_loop,
            args=(i, per_worker, candidates_per_state, m_rollouts,
                  games_per_collection_batch, seed, checkpoint_path,
                  result_queue, stop_event),
            daemon=False,
        )
        p.start()
        workers.append(p)

    pending: list[_LabeledState] = []
    workers_alive = n_workers
    t0 = time.time()

    try:
        while workers_alive > 0 and states_written < n_states_target:
            try:
                result = result_queue.get(timeout=60.0)
            except _queue.Empty:
                alive_now = sum(1 for p in workers if p.is_alive())
                if alive_now < workers_alive:
                    print(f"[label]   {workers_alive - alive_now} worker(s) "
                          f"died unexpectedly; continuing with {alive_now}")
                    workers_alive = alive_now
                continue

            if result.state is None:
                workers_alive -= 1
                continue

            pending.append(result.state)
            states_written += 1
            candidates_written += len(result.state.candidates)

            if len(pending) >= chunk_size:
                _flush_chunk(out_dir, manifest, chunk_idx, pending,
                             states_written, candidates_written, t0)
                chunk_idx += 1
                pending = []

            if states_written >= n_states_target:
                stop_event.set()
                break

    except KeyboardInterrupt:
        print("[label] interrupted — signaling workers to stop")
        stop_event.set()
    finally:
        stop_event.set()
        # Drain any in-flight items so workers can finish their final put().
        # Without this they'd block on put() and never see the sentinel logic.
        drain_deadline = time.time() + 30.0
        while time.time() < drain_deadline and any(p.is_alive() for p in workers):
            try:
                result = result_queue.get(timeout=1.0)
            except _queue.Empty:
                continue
            if result.state is not None:
                pending.append(result.state)
                states_written += 1
                candidates_written += len(result.state.candidates)
                if len(pending) >= chunk_size:
                    _flush_chunk(out_dir, manifest, chunk_idx, pending,
                                 states_written, candidates_written, t0)
                    chunk_idx += 1
                    pending = []
        for p in workers:
            p.join(timeout=10.0)
            if p.is_alive():
                print(f"[label]   terminating stuck worker (pid {p.pid})")
                p.terminate()
                p.join(timeout=5.0)

        if pending:
            _flush_chunk(out_dir, manifest, chunk_idx, pending,
                         states_written, candidates_written, t0)

    elapsed = time.time() - t0
    print(f"[label] done. {states_written} states, {candidates_written} "
          f"candidates in {len(manifest['chunks'])} chunks. "
          f"Elapsed: {elapsed:.1f}s ({candidates_written/max(1.0,elapsed):.1f} cand/s).")


# ---------------------------------------------------------------------------
# Top-level run (single-process)
# ---------------------------------------------------------------------------

def run_labeling(
    n_states_target: int,
    candidates_per_state: int,
    m_rollouts: int,
    output_dir: str,
    chunk_size: int,
    games_per_collection_batch: int,
    seed: int,
    checkpoint_path: str,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[label] loading frozen model from {checkpoint_path}")
    model = probe.load_frozen_model(checkpoint_path)

    manifest = _load_manifest(out_dir)
    if manifest.get("checkpoint") and manifest["checkpoint"] != checkpoint_path:
        raise RuntimeError(
            f"manifest checkpoint mismatch: "
            f"existing {manifest['checkpoint']}, requested {checkpoint_path}. "
            f"Remove {out_dir} or use a different output_dir."
        )
    manifest["checkpoint"] = checkpoint_path

    states_written = manifest["states_written"]
    candidates_written = manifest["candidates_written"]
    print(f"[label] resuming from manifest: {states_written} states, "
          f"{candidates_written} candidates already written "
          f"({len(manifest['chunks'])} chunks).")

    pending: list[_LabeledState] = []
    chunk_idx = len(manifest["chunks"])
    batch_idx = 0
    t0 = time.time()

    while states_written < n_states_target:
        # 1) Collect a batch of decision states via short self-play games.
        n_games = games_per_collection_batch
        states, contexts = probe.collect_decision_states(model, n_games)
        if not states:
            print("[label] state collection returned 0 states — skipping batch")
            continue

        # 2) Label each collected state until we either fill the chunk or hit
        #    the dataset target. game_uid pairs (batch_idx, st.game_idx) so
        #    states from the same self-play game share the same uid for
        #    episode-level train/val splits at training time.
        rng.shuffle(states)
        for st in states:
            if states_written >= n_states_target:
                break
            ctx = contexts[st.game_idx]
            game_uid = (batch_idx << 16) | (st.game_idx & 0xFFFF)
            try:
                labeled = _label_one_state(
                    model, st, ctx,
                    candidates_per_state=candidates_per_state,
                    m_rollouts=m_rollouts, rng=rng,
                    game_uid=game_uid,
                )
            except Exception as e:
                print(f"[label]   state skipped due to error: {e}")
                continue
            if labeled is None:
                continue
            pending.append(labeled)
            states_written += 1
            candidates_written += len(labeled.candidates)

            if len(pending) >= chunk_size:
                _flush_chunk(out_dir, manifest, chunk_idx, pending,
                             states_written, candidates_written, t0)
                chunk_idx += 1
                pending = []
        batch_idx += 1

    # Final flush
    if pending:
        _flush_chunk(out_dir, manifest, chunk_idx, pending,
                     states_written, candidates_written, t0)

    print(f"[label] done. {states_written} states, {candidates_written} candidates"
          f" in {len(manifest['chunks'])} chunks. Elapsed: {time.time()-t0:.1f}s.")


def _flush_chunk(out_dir: Path, manifest: dict, chunk_idx: int,
                 pending: list[_LabeledState],
                 states_written: int, candidates_written: int,
                 t0: float):
    chunk_path = out_dir / f"chunk_{chunk_idx:05d}.npz"
    arrays = _pack_chunk(pending)
    np.savez(chunk_path, **arrays)
    manifest["chunks"].append({
        "path": chunk_path.name,
        "n_states": len(pending),
        "n_candidates": int(arrays["cand_log_pi"].shape[0]),
    })
    manifest["states_written"] = states_written
    manifest["candidates_written"] = candidates_written
    _save_manifest(out_dir, manifest)
    elapsed = time.time() - t0
    rate = candidates_written / max(1.0, elapsed)
    print(f"[label]   wrote {chunk_path.name}: "
          f"{len(pending)} states, {arrays['cand_log_pi'].shape[0]} candidates. "
          f"Total: {states_written} states, {candidates_written} candidates "
          f"({rate:.1f} cand/s, {elapsed:.0f}s elapsed).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: generate labeled (state, action) -> (log pi, Q) "
                    "dataset for the gap identifier."
    )
    parser.add_argument("--n-states", type=int, default=1000,
                        help="Target number of decision states to label "
                             "(default: 1000; 10k for full run)")
    parser.add_argument("--candidates-per-state", type=int, default=500,
                        help="Stratified candidates sampled per state")
    parser.add_argument("--rollouts", type=int, default=8,
                        help="Dice rollouts per (state, action) for Q label")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for sharded npz chunks")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="States per output chunk")
    parser.add_argument("--games-per-batch", type=int, default=20,
                        help="Self-play games to run per state-collection batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=probe.CHECKPOINT_PATH,
                        help="Frozen-policy checkpoint path")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel labeling workers (1 = single-process). "
                             "Each worker plays self-play games independently.")
    args = parser.parse_args()

    if args.workers > 1:
        run_labeling_parallel(
            n_states_target=args.n_states,
            candidates_per_state=args.candidates_per_state,
            m_rollouts=args.rollouts,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            games_per_collection_batch=args.games_per_batch,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            n_workers=args.workers,
        )
    else:
        run_labeling(
            n_states_target=args.n_states,
            candidates_per_state=args.candidates_per_state,
            m_rollouts=args.rollouts,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            games_per_collection_batch=args.games_per_batch,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
        )


if __name__ == "__main__":
    main()
