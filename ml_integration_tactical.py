"""ML tactical integration v2: wire TacticalModel outputs to game state per activation.

Destination pointer model: the model picks a movement type (hold/advance/rush/charge),
then a destination hex via pointer attention (for advance/rush) or charge target (for charge),
then a shooting target (for hold/advance).
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from board import Board, COLS, ROWS, OBJECTIVES, OBJ_SEIZE_RANGE, dist, dist_sq
from combat import evaluate_target, _shooter_cover_lookup
from models import UnitState
from ml_features import (
    MAX_UNITS_PER_SIDE,
    TACTICAL_UNIT_FEATURES,
    BOARD_DIAG,
    encode_state_tactical,
    precompute_damage,
    extract_can_charge_mask,
    extract_is_shaken,
    _flip_x,
    _flip_y,
    _get_model_objectives,
    DEST_FEATURE_DIM,
    DEST_EMBED_DIM,
    MAX_DEST_CANDIDATES,
    RANGE_THRESHOLDS,
    NUM_RANGE_THRESHOLDS,
)

from ml_model_tactical import (
    TacticalModel, TacticalModelOutput,
    NUM_MOVE_TYPES, MOVE_MOVE, MOVE_CHARGE,
    POST_MOVE_REL_FEATURES,
    PHASE_PRE_SELECT, PHASE_POST_SELECT, PHASE_POST_MOVETYPE, PHASE_POST_DEST,
    N_PHASES,
)
import fast_core as _fc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOVE_TYPE_NAMES = ["move", "charge"]

_INV_BOARD_DIAG = 1.0 / BOARD_DIAG

# ---------------------------------------------------------------------------
# Phase re-encode flag (commit-and-re-encode refactor).
# ---------------------------------------------------------------------------
# When False (default) apply_tactical_model takes the legacy single-trunk path
# (state_vec built once, trunk called 3–4× on the same input, post_move_rel as
# the only post-move signal). When True it takes the 4-phase path: encode()
# called once per phase with h persisted, POST_DEST using a post-move state_vec
# built via project_post_move_unit_state. The flag is toggled at process
# startup via set_phase_reencode_enabled() so the inference path is consistent
# across all call sites (game loop, sampling opponent, profiler tools).

_PHASE_REENCODE_ENABLED: bool = False


def set_phase_reencode_enabled(enabled: bool) -> None:
    """Toggle the phase-reencode inference path in apply_tactical_model."""
    global _PHASE_REENCODE_ENABLED
    _PHASE_REENCODE_ENABLED = bool(enabled)


def is_phase_reencode_enabled() -> bool:
    """Return the current phase-reencode flag (for tests / diagnostics)."""
    return _PHASE_REENCODE_ENABLED

# ---------------------------------------------------------------------------
# Precomputed per-hex objective lookups (game-global, computed once)
# ---------------------------------------------------------------------------
# Distance is rotation-invariant, so game-space coords work for both players.
# Player B reorders objectives (1↔2, 3↔4) so we store separate tables.

_OBJ_DIST_A: np.ndarray | None = None   # (COLS, ROWS, 5) float32 — normalised distance
_OBJ_IN_RANGE_A: np.ndarray | None = None  # (COLS, ROWS, 5) float32 — 0/1 seize flag
_OBJ_DIST_B: np.ndarray | None = None
_OBJ_IN_RANGE_B: np.ndarray | None = None


def _ensure_obj_lookup() -> None:
    """Build the objective lookup tables on first call."""
    global _OBJ_DIST_A, _OBJ_IN_RANGE_A, _OBJ_DIST_B, _OBJ_IN_RANGE_B
    if _OBJ_DIST_A is not None:
        return

    inv_bd = 1.0 / BOARD_DIAG
    # Player A objective order: [Centre, A-side, B-side, Home-A, Home-B]
    objs_a = _get_model_objectives("A")
    # Player B reorders: [Centre, B-side, A-side, Home-B, Home-A] (with flipped coords)
    objs_b = _get_model_objectives("B")

    dist_a = np.zeros((COLS, ROWS, 5), dtype=np.float32)
    inr_a = np.zeros((COLS, ROWS, 5), dtype=np.float32)
    dist_b = np.zeros((COLS, ROWS, 5), dtype=np.float32)
    inr_b = np.zeros((COLS, ROWS, 5), dtype=np.float32)

    for col in range(COLS):
        for row in range(ROWS):
            # Player A: model-space == game-space
            hx_a, hy_a = float(col), float(row)
            for oi, (ox, oy) in enumerate(objs_a):
                d = math.sqrt((hx_a - ox) ** 2 + (hy_a - oy) ** 2)
                dist_a[col, row, oi] = d * inv_bd
                inr_a[col, row, oi] = 1.0 if d <= OBJ_SEIZE_RANGE else 0.0

            # Player B: model-space = flipped game-space
            hx_b = (COLS - 1) - col
            hy_b = (ROWS - 1) - row
            for oi, (ox, oy) in enumerate(objs_b):
                d = math.sqrt((hx_b - ox) ** 2 + (hy_b - oy) ** 2)
                dist_b[col, row, oi] = d * inv_bd
                inr_b[col, row, oi] = 1.0 if d <= OBJ_SEIZE_RANGE else 0.0

    _OBJ_DIST_A = dist_a
    _OBJ_IN_RANGE_A = inr_a
    _OBJ_DIST_B = dist_b
    _OBJ_IN_RANGE_B = inr_b


# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------

_timing_encode_s: float = 0.0
_timing_forward_s: float = 0.0
_timing_calls: int = 0


def reset_timing() -> None:
    global _timing_encode_s, _timing_forward_s, _timing_calls
    _timing_encode_s = 0.0
    _timing_forward_s = 0.0
    _timing_calls = 0


def get_timing() -> dict:
    return {
        'encode_s': _timing_encode_s,
        'forward_s': _timing_forward_s,
        'calls': _timing_calls,
    }


# ---------------------------------------------------------------------------
# Coroutine-mode dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InferenceRequest:
    """Yielded by simulate_game when it needs an ML decision."""
    state_vec: torch.Tensor        # (2811,) pre-encoded
    alive_mask: torch.Tensor       # (10,) bool
    enemy_alive_mask: torch.Tensor # (10,) bool
    player: str                    # "A" or "B"
    friendly_positions: list[tuple[float, float]]  # model-space, 10 slots
    enemy_positions: list[tuple[float, float]]     # model-space, 10 slots
    advance_distances: list[float]                 # per friendly slot
    rush_distances: list[float]                    # per friendly slot
    max_weapon_ranges: list[float]                 # max ranged weapon range per friendly slot


@dataclass
class InferenceResult:
    """Fully-decoded decision sent back to simulate_game by the coordinator."""
    unit_idx: int                       # selected friendly slot
    move_type: int                      # 0=move, 1=charge
    dest_col: int                       # selected hex column (game-space)
    dest_row: int                       # selected hex row (game-space)
    charge_target_idx: int              # enemy slot
    shoot_target_idx: int               # enemy slot
    target_ranking: list[int]           # shoot target ranking (descending)
    value: float
    is_advance_reachable: bool = True   # whether selected dest is advance-reachable (can shoot)


# ---------------------------------------------------------------------------
# Two-phase coroutine protocol
# ---------------------------------------------------------------------------
# The two-phase variant lets the coroutine compute Dijkstra-reachable
# destination candidates between the model's unit-selection step and its
# destination-pointer step. The coordinator does:
#   1. Batched phase 1 across all games  — trunk, unit selection, move type,
#      charge target, value.
#   2. Each coroutine, on receiving its Phase1Result, computes Dijkstra
#      candidates for the selected unit and yields a Phase2Request.
#   3. Batched phase 2 across all games — destination pointer + shoot target,
#      using the trunk state cached on the coordinator from phase 1.
#
# This restores the same dest_features pipeline the non-coroutine path uses
# while preserving the cross-game inference batching of the coroutine path.

@dataclass
class Phase1Request:
    """Phase-1 yield: pre-decode state. The coordinator runs trunk + unit
    selection + move type + charge target + value in one batched pass."""
    state_vec: torch.Tensor
    alive_mask: torch.Tensor
    enemy_alive_mask: torch.Tensor
    player: str
    friendly_positions: list[tuple[float, float]]
    enemy_positions: list[tuple[float, float]]
    advance_distances: list[float]
    rush_distances: list[float]
    max_weapon_ranges: list[float]


@dataclass
class Phase1Result:
    """Phase-1 result sent back to the coroutine so it can pick the selected
    unit and compute its Dijkstra candidates."""
    unit_idx: int
    move_type: int
    charge_target_idx: int
    value: float


@dataclass
class Phase2Request:
    """Phase-2 yield: contains Dijkstra-derived destination candidates plus
    the data needed by the destination/shoot pointers and downstream decode."""
    unit_idx: int
    move_type: int
    charge_target_idx: int
    value: float
    dest_features: torch.Tensor          # (C, FEAT)
    dest_mask: torch.Tensor              # (C,) bool
    candidates: object                    # numpy (C, 2) int — (col, row)
    advance_reachable: object             # numpy (C,) bool
    friendly_positions: list[tuple[float, float]]
    enemy_positions: list[tuple[float, float]]
    max_weapon_ranges: list[float]
    player: str


@dataclass
class Phase2Result:
    """Final decoded decision returned to the coroutine, identical in shape
    to InferenceResult — the coroutine treats it as such."""
    unit_idx: int
    move_type: int
    dest_col: int
    dest_row: int
    charge_target_idx: int
    shoot_target_idx: int
    target_ranking: list[int]
    value: float
    is_advance_reachable: bool = True


# ---------------------------------------------------------------------------
# Post-move relative features
# ---------------------------------------------------------------------------

def project_post_move_unit_state(
    unit: UnitState,
    dest: tuple[int, int],
    is_rush: bool,
) -> UnitState:
    """Return a shallow copy of *unit* with model positions translated to *dest*.

    All model positions shift by (dest - round(centre(unit))) so formation shape
    is preserved; fatigued flips True when *is_rush*. Other fields (wounds,
    weapons, hero refs) share state with the original — this projection is
    intended for read-only feature re-encoding during the POST_DEST trunk pass
    and callers must not mutate the returned object.
    """
    new_unit = copy.copy(unit)
    cx, cy = unit.centre()
    dx = dest[0] - int(round(cx))
    dy = dest[1] - int(round(cy))
    new_unit.positions = [(c + dx, r + dy) for (c, r) in unit.alive_positions()]
    if is_rush:
        new_unit.fatigued = True
    return new_unit


def compute_post_move_rel(
    post_x: float,
    post_y: float,
    enemy_positions: list[tuple[float, float]],
) -> torch.Tensor:
    """Compute (sin θ, cos θ, dist) from post-move position to each of 10 enemy slots.

    Returns (30,) tensor.
    """
    if _fc.USE_C_EXT and _fc.is_available():
        arr = _fc.fast_compute_post_move_rel(
            post_x, post_y, enemy_positions, _INV_BOARD_DIAG)
        # arr is (30,) float32; torch.from_numpy shares memory — safe because
        # arr is a fresh copy from frombuffer+.copy() in the wrapper.
        return torch.from_numpy(arr)
    feats = torch.zeros(POST_MOVE_REL_FEATURES)
    for i, (ex, ey) in enumerate(enemy_positions):
        dx = ex - post_x
        dy = ey - post_y
        d = math.sqrt(dx * dx + dy * dy)
        base = i * 3
        if d < 1e-6:
            feats[base] = 0.0
            feats[base + 1] = 0.0
        else:
            feats[base] = dy / d       # sin θ
            feats[base + 1] = dx / d   # cos θ
        feats[base + 2] = d * _INV_BOARD_DIAG
    return feats


# ---------------------------------------------------------------------------
# Weapon range helpers (for in-range masking)
# ---------------------------------------------------------------------------

def _get_max_weapon_ranges(units: list[UnitState]) -> list[float]:
    """Get max ranged weapon range for each of up to 10 friendly unit slots.

    Dead/missing slots get 0.0.
    """
    ranges: list[float] = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(units) and units[i].models_alive > 0:
            max_r = max(
                (w.range_inches for w in units[i].unit.weapons if not w.melee),
                default=0.0,
            )
            ranges.append(float(max_r))
        else:
            ranges.append(0.0)
    return ranges


def compute_visibility_mask(
    shooter_pos: tuple[float, float],
    enemy_units: list[UnitState],
    board: Board | None,
) -> torch.Tensor:
    """Per-enemy-slot visibility mask from *shooter_pos* (game-space col, row).

    Returns a (MAX_UNITS_PER_SIDE,) bool tensor. Slot i is True iff at least
    one alive model of ``enemy_units[i]`` is visible from ``shooter_pos`` per
    TERRAIN_SPEC.md §4.4(1). When the board has no terrain, all alive enemy
    slots are True (no visibility constraint).
    """
    mask = torch.zeros(MAX_UNITS_PER_SIDE, dtype=torch.bool)
    has_terrain = board is not None and bool(board.terrain)
    if not has_terrain:
        for i in range(min(MAX_UNITS_PER_SIDE, len(enemy_units))):
            if enemy_units[i].models_alive > 0:
                mask[i] = True
        return mask

    from terrain_los import is_visible
    sx = int(round(shooter_pos[0]))
    sy = int(round(shooter_pos[1]))
    for i in range(min(MAX_UNITS_PER_SIDE, len(enemy_units))):
        e = enemy_units[i]
        if e.models_alive <= 0:
            continue
        for tp in e.alive_positions():
            if is_visible((sx, sy), tp, board.terrain):
                mask[i] = True
                break
    return mask


def compute_unit_visibility_arrays(
    unit: UnitState,
    candidates: np.ndarray,
    cand_mask: np.ndarray,
    enemy_units: list[UnitState],
    board: Board | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-destination and static visibility for a friendly unit.

    Returns
    -------
    dest_visibility : (MAX_DEST_CANDIDATES, MAX_UNITS_PER_SIDE) bool ndarray
        ``dest_visibility[c, e]`` is True iff at least one alive model of
        ``enemy_units[e]`` is visible from candidate dest hex ``candidates[c]``
        (in game-space). Padded rows (``cand_mask[c]=False``) are left zero.
    static_visibility : (MAX_UNITS_PER_SIDE,) bool ndarray
        Visibility from the unit's current centre to each enemy slot — used
        for activations that do not consume the destination pointer (charge/
        hold/shaken).

    When ``board`` has no terrain, both arrays are filled with the enemy-alive
    mask (no visibility constraint).
    """
    n_cand_full = MAX_DEST_CANDIDATES
    dest_vis = np.zeros((n_cand_full, MAX_UNITS_PER_SIDE), dtype=np.bool_)
    static_vis = np.zeros(MAX_UNITS_PER_SIDE, dtype=np.bool_)

    n_enemies = min(MAX_UNITS_PER_SIDE, len(enemy_units))
    alive = [(i, enemy_units[i].alive_positions())
             for i in range(n_enemies)
             if enemy_units[i].models_alive > 0]
    if not alive:
        return dest_vis, static_vis

    has_terrain = board is not None and bool(board.terrain)
    if not has_terrain:
        for ei, _ in alive:
            static_vis[ei] = True
            dest_vis[:, ei] = cand_mask  # valid dests see all alive enemies
        return dest_vis, static_vis

    # Static visibility from unit centre
    cx, cy = unit.centre()
    sx_s = int(round(cx))
    sy_s = int(round(cy))
    for ei, eps in alive:
        _, vmask = _shooter_cover_lookup(board, (sx_s, sy_s), eps)
        if any(vmask):
            static_vis[ei] = True

    # Per-destination visibility — only iterate valid candidates
    n_cand = candidates.shape[0]
    for ci in range(n_cand):
        if not cand_mask[ci]:
            continue
        sx = int(candidates[ci, 0])
        sy = int(candidates[ci, 1])
        for ei, eps in alive:
            _, vmask = _shooter_cover_lookup(board, (sx, sy), eps)
            if any(vmask):
                dest_vis[ci, ei] = True

    return dest_vis, static_vis


def compute_unit_expected_damage_arrays(
    unit: UnitState,
    candidates: np.ndarray,
    cand_mask: np.ndarray,
    enemy_units: list[UnitState],
    board: Board | None,
    expected_damage_table: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-destination and static cover-aware expected damage.

    For each candidate destination, compute the §5.5 game-time combination
    sum of E_damage[Y, U_def_j, cover_state(dest, U_def_j)] over the acting
    unit's currently alive shooter models, normalised by ``U_def_j``'s
    starting wounds and capped at 1.0. The destination is used as a single
    shooter-position proxy for every shooter (TERRAIN_SPEC §5.4).

    Returns
    -------
    dest_dmg : (MAX_DEST_CANDIDATES, MAX_UNITS_PER_SIDE) float32
        Expected wound fraction from each candidate dest to each enemy slot,
        scaled by ``unit.models_alive / unit.unit.models``. Invisible/dead/
        invalid combinations are 0.
    static_dmg : (MAX_UNITS_PER_SIDE,) float32
        Same computation from the unit's current centre — used when the
        activation does not consume the destination pointer (charge/hold/
        shaken).

    When ``expected_damage_table`` is None or the unit/enemy_units are not
    indexable, returns zero arrays (caller should fall back to the cover-
    blind path).
    """
    dest_dmg = np.zeros((MAX_DEST_CANDIDATES, MAX_UNITS_PER_SIDE), dtype=np.float32)
    static_dmg = np.zeros(MAX_UNITS_PER_SIDE, dtype=np.float32)

    if expected_damage_table is None or unit is None:
        return dest_dmg, static_dmg

    n_enemies = min(MAX_UNITS_PER_SIDE, len(enemy_units))
    alive_enemies = []
    for i in range(n_enemies):
        e = enemy_units[i]
        if e.models_alive > 0:
            alive_enemies.append((i, e, e.alive_positions()))
    if not alive_enemies:
        return dest_dmg, static_dmg

    atk_id = id(unit)
    n_atk_alive = unit.models_alive
    if n_atk_alive <= 0:
        return dest_dmg, static_dmg
    atk_alive_frac = n_atk_alive / max(unit.unit.models, 1)

    def _sum_for_pair(def_unit: UnitState, cover: bool) -> float:
        """Sum E_damage across surviving shooter models for one (atk, def) pair."""
        total = 0.0
        def_id = id(def_unit)
        for mi in range(n_atk_alive):
            entry = expected_damage_table.get((atk_id, mi, def_id))
            if entry is None:
                continue
            total += entry[1] if cover else entry[0]
        return total

    def _eval(shooter_sq: tuple[int, int]) -> dict[int, float]:
        """Returns {enemy_slot: wound_fraction} for one shooter square."""
        out: dict[int, float] = {}
        has_terrain = board is not None and bool(board.terrain)
        for ei, def_unit, eps in alive_enemies:
            if has_terrain:
                n_bad, vmask = _shooter_cover_lookup(board, shooter_sq, eps)
                if not any(vmask):
                    continue  # zero damage when fully blocked
                cover = (2 * n_bad > def_unit.models_alive)
            else:
                cover = False
            total = _sum_for_pair(def_unit, cover)
            if total <= 0:
                continue
            sw = max(_e_dmg_starting_wounds(def_unit.unit), 1)
            out[ei] = min(total / sw, 1.0) * atk_alive_frac
        return out

    # Static (from unit centre)
    cx, cy = unit.centre()
    static_map = _eval((int(round(cx)), int(round(cy))))
    for ei, val in static_map.items():
        static_dmg[ei] = val

    # Per-destination
    n_cand = candidates.shape[0]
    for ci in range(n_cand):
        if not cand_mask[ci]:
            continue
        sx = int(candidates[ci, 0])
        sy = int(candidates[ci, 1])
        dmg_map = _eval((sx, sy))
        for ei, val in dmg_map.items():
            dest_dmg[ci, ei] = val

    return dest_dmg, static_dmg


def _e_dmg_starting_wounds(unit) -> int:
    """Defender starting wounds — mirrors expected_damage_table._starting_wounds."""
    return (unit.tough if unit.tough else 1) * unit.models


def compute_in_range_mask(
    post_move_rel: torch.Tensor,
    max_weapon_range: float,
    enemy_alive_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute a bool mask of enemies within weapon range from post-move position.

    Uses centre-to-centre distance from post_move_rel features.

    Parameters
    ----------
    post_move_rel : (30,) — (sin θ, cos θ, normalised_dist) × 10 enemies
    max_weapon_range : max ranged weapon range of the selected unit (in cells)
    enemy_alive_mask : (10,) bool

    Returns
    -------
    (10,) bool — True for enemies that are alive AND in range.
    """
    # Extract normalised distances (every 3rd element starting at index 2)
    distances = post_move_rel[2::3] * BOARD_DIAG  # (10,) actual distance in cells
    in_range = distances <= max_weapon_range
    return in_range & enemy_alive_mask


def compute_in_range_mask_batched(
    post_move_rel_batch: torch.Tensor,
    max_weapon_ranges: torch.Tensor,
    enemy_alive_batch: torch.Tensor,
) -> torch.Tensor:
    """Batched version of compute_in_range_mask.

    Parameters
    ----------
    post_move_rel_batch : (N, 30)
    max_weapon_ranges : (N,) — max weapon range per sample
    enemy_alive_batch : (N, 10) bool

    Returns
    -------
    (N, 10) bool
    """
    distances = post_move_rel_batch[:, 2::3] * BOARD_DIAG  # (N, 10)
    in_range = distances <= max_weapon_ranges.unsqueeze(-1)
    return in_range & enemy_alive_batch


# ---------------------------------------------------------------------------
# Destination pointer: candidate set and feature computation
# ---------------------------------------------------------------------------


def compute_destination_candidates(
    unit: UnitState,
    board: Board,
    enemy_positions: set[tuple[int, int]],
    player: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute unified destination candidate set via Dijkstra (rush budget).

    Each candidate is flagged as advance-reachable (within advance budget,
    can shoot after moving) or rush-only (beyond advance budget, cannot shoot).
    The centroid (index 0) is always advance-reachable ("hold" equivalent).

    Returns:
        candidates: (MAX_DEST_CANDIDATES, 2) int array of (col, row) in game-space
        mask: (MAX_DEST_CANDIDATES,) bool array
        advance_reachable: (MAX_DEST_CANDIDATES,) bool array
    """
    cx, cy = unit.centre()
    centroid = (int(round(cx)), int(round(cy)))

    rush_budget = float(unit.unit.rush_distance)
    advance_budget = float(unit.unit.advance_distance)

    from movement import build_exclusion_grid
    exclusion_grid = build_exclusion_grid(enemy_positions)

    flying = bool(unit.unit.flying)
    strider = bool(unit.unit.strider)
    impassible = board.impassible_grid if board.terrain else None
    difficult = board.difficult_grid if board.terrain else None

    # Full candidate set: Dijkstra with rush budget
    reachable = _fc.fast_dijkstra_reachable_set(
        centroid, rush_budget, board.occupancy, enemy_positions,
        is_charge=False, flying=flying, strider=strider,
        exclusion_grid=exclusion_grid, cols=COLS, rows=ROWS,
        impassible_grid=impassible, difficult_grid=difficult,
    )  # (N, 2) int32

    # Advance-reachable subset: Dijkstra with advance budget
    if advance_budget >= rush_budget:
        # All rush-reachable cells are also advance-reachable
        adv_reachable_set: set[tuple[int, int]] | None = None  # means "all"
    elif advance_budget > 0:
        adv_cells = _fc.fast_dijkstra_reachable_set(
            centroid, advance_budget, board.occupancy, enemy_positions,
            is_charge=False, flying=flying, strider=strider,
            exclusion_grid=exclusion_grid, cols=COLS, rows=ROWS,
            impassible_grid=impassible, difficult_grid=difficult,
        )
        adv_reachable_set = set()
        for r in range(len(adv_cells)):
            adv_reachable_set.add((int(adv_cells[r, 0]), int(adv_cells[r, 1])))
    else:
        # advance_budget == 0: only centroid is advance-reachable
        adv_reachable_set = set()

    # Prepend centroid as candidate 0 (always valid — "stay put" option)
    centroid_arr = np.array([[centroid[0], centroid[1]]], dtype=np.int32)
    if len(reachable) > 0:
        not_centroid = ~((reachable[:, 0] == centroid[0]) & (reachable[:, 1] == centroid[1]))
        reachable = reachable[not_centroid]
        combined = np.concatenate([centroid_arr, reachable], axis=0)
    else:
        combined = centroid_arr

    n_valid = min(len(combined), MAX_DEST_CANDIDATES)

    # Return tight arrays (no padding to MAX_DEST_CANDIDATES — callers pad to
    # batch-max as needed, avoiding wasteful 4096-wide allocations).
    candidates = np.zeros((n_valid, 2), dtype=np.int32)
    mask = np.ones(n_valid, dtype=np.bool_)
    advance_reachable = np.zeros(n_valid, dtype=np.bool_)

    candidates[:] = combined[:n_valid]

    # Mark advance-reachable candidates
    if adv_reachable_set is None:
        # All candidates are advance-reachable
        advance_reachable[:] = True
    else:
        # Centroid (index 0) is always advance-reachable
        advance_reachable[0] = True
        for i in range(1, n_valid):
            c, r = int(combined[i, 0]), int(combined[i, 1])
            if (c, r) in adv_reachable_set:
                advance_reachable[i] = True

    return candidates, mask, advance_reachable


@dataclass
class _DestEnemyCache:
    """Cached enemy data for compute_destination_features, reusable across units
    within a single activation."""
    enemy_model_positions: list[tuple[float, float]]
    enemy_advance_dists: list[float]
    enemy_rush_dists: list[float]
    enemy_activated: list[bool]
    enemy_alive_fracs: list[float]


def build_dest_enemy_cache(
    enemy_units: list[UnitState],
    enemy_alive_mask: np.ndarray,
    player: str,
) -> _DestEnemyCache:
    """Build cached enemy data once per activation for reuse across units."""
    enemy_model_positions: list[tuple[float, float]] = []
    enemy_advance_dists: list[float] = []
    enemy_rush_dists: list[float] = []
    enemy_activated: list[bool] = []
    enemy_alive_fracs: list[float] = []
    sentinel = ((COLS - 1) / 2.0, (ROWS - 1) / 2.0)
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(enemy_units) and enemy_units[i].models_alive > 0:
            ex, ey = enemy_units[i].centre()
            if player == "B":
                ex = _flip_x(ex)
                ey = _flip_y(ey)
            enemy_model_positions.append((ex, ey))
            enemy_advance_dists.append(float(enemy_units[i].unit.advance_distance))
            enemy_rush_dists.append(float(enemy_units[i].unit.rush_distance))
            enemy_activated.append(enemy_units[i].activated)
            enemy_alive_fracs.append(
                enemy_units[i].models_alive / max(enemy_units[i].unit.models, 1))
        else:
            enemy_model_positions.append(sentinel)
            enemy_advance_dists.append(0.0)
            enemy_rush_dists.append(0.0)
            enemy_activated.append(False)
            enemy_alive_fracs.append(0.0)
    return _DestEnemyCache(
        enemy_model_positions=enemy_model_positions,
        enemy_advance_dists=enemy_advance_dists,
        enemy_rush_dists=enemy_rush_dists,
        enemy_activated=enemy_activated,
        enemy_alive_fracs=enemy_alive_fracs,
    )


_RANGE_THRESHOLDS_NP = np.array(RANGE_THRESHOLDS, dtype=np.float64)


def compute_destination_features(
    candidates: np.ndarray,            # (MAX_DEST_CANDIDATES, 2) game-space
    mask: np.ndarray,                  # (MAX_DEST_CANDIDATES,)
    unit: UnitState | None,
    unit_slot: int,
    player: str,
    enemy_units: list[UnitState] | None,
    enemy_alive_mask: np.ndarray,      # (10,) bool
    friendly_ranged_matchups: np.ndarray,  # (num_friendly, 10, 7) — unit's offensive damage
    enemy_ranged_matchups: np.ndarray,     # (num_enemy, 10, 7) — enemies' damage vs unit
    melee_matchups: np.ndarray,            # (num_enemy, 10) — enemies' melee damage vs unit
    move_budget: float,
    enemy_cache: _DestEnemyCache | None = None,
    *,
    unit_centre: tuple[float, float] | None = None,
    unit_alive_frac: float | None = None,
    advance_reachable: np.ndarray | None = None,  # (MAX_DEST_CANDIDATES,) bool
    board: Board | None = None,
    dest_expected_damage: np.ndarray | None = None,  # (MAX_DEST_CANDIDATES, MAX_UNITS_PER_SIDE)
) -> np.ndarray:
    """Compute per-hex features for all candidates (vectorized).

    Returns: (MAX_DEST_CANDIDATES, DEST_FEATURE_DIM) float32 array.
    All spatial features use model-space (Player B flipped).

    Feature [75] is the advance-reachable flag (1.0 = can shoot from here,
    0.0 = rush-only). Offensive value features [15:25] are zeroed for
    rush-only candidates.

    If *unit_centre* and *unit_alive_frac* are provided, *unit* may be None
    (used during PPO replay where UnitState objects are not available).
    """
    _ensure_obj_lookup()
    features = np.zeros((len(mask), DEST_FEATURE_DIM), dtype=np.float32)

    n_valid = int(mask.sum())
    if n_valid == 0:
        return features

    # Unit centroid in model-space
    if unit_centre is not None:
        cx, cy = unit_centre
    else:
        cx, cy = unit.centre()
        if player == "B":
            cx = _flip_x(cx)
            cy = _flip_y(cy)

    # Unit alive fraction
    if unit_alive_frac is None:
        unit_alive_frac = unit.models_alive / max(unit.unit.models, 1)

    # Use cached enemy data or compute fresh
    if enemy_cache is not None:
        ec = enemy_cache
    else:
        ec = build_dest_enemy_cache(enemy_units, enemy_alive_mask, player)

    # --- Vectorized setup ---
    # Valid candidate coords (game-space integers)
    cols_v = candidates[:n_valid, 0].astype(np.int32)  # (N,)
    rows_v = candidates[:n_valid, 1].astype(np.int32)  # (N,)

    # Model-space hex positions
    if player == "B":
        hx = (COLS - 1) - cols_v.astype(np.float64)
        hy = (ROWS - 1) - rows_v.astype(np.float64)
    else:
        hx = cols_v.astype(np.float64)
        hy = rows_v.astype(np.float64)

    # --- 3.1 Egocentric Spatial (5 features) [0:5] ---
    inv_budget = 1.0 / max(move_budget, 1e-6)
    dx = hx - cx  # (N,)
    dy = hy - cy
    norm_dist = np.sqrt(dx * dx + dy * dy)
    features[:n_valid, 0] = dx * inv_budget
    features[:n_valid, 1] = dy * inv_budget
    features[:n_valid, 2] = norm_dist * inv_budget
    valid_dir = norm_dist > 1e-6
    angles = np.arctan2(dy, dx)
    features[:n_valid, 3] = np.where(valid_dir, np.sin(angles), 0.0)
    features[:n_valid, 4] = np.where(valid_dir, np.cos(angles), 0.0)

    # --- 3.2 Objective Proximity (10 features) [5:15] ---
    obj_dist_tbl = _OBJ_DIST_A if player == "A" else _OBJ_DIST_B
    obj_inr_tbl = _OBJ_IN_RANGE_A if player == "A" else _OBJ_IN_RANGE_B
    # Fancy-index: (N, 5) from (COLS, ROWS, 5)
    obj_d = obj_dist_tbl[cols_v, rows_v]  # (N, 5)
    obj_i = obj_inr_tbl[cols_v, rows_v]   # (N, 5)
    features[:n_valid, 5:15:2] = obj_d
    features[:n_valid, 6:15:2] = obj_i

    # --- Enemy distance matrix (N, 10) ---
    enemy_pos = np.array(ec.enemy_model_positions, dtype=np.float64)  # (10, 2)
    # Broadcast: (N, 1) - (1, 10) => (N, 10)
    edx = hx[:, None] - enemy_pos[:, 0][None, :]  # (N, 10)
    edy = hy[:, None] - enemy_pos[:, 1][None, :]
    enemy_dists = np.sqrt(edx * edx + edy * edy)   # (N, 10)

    alive_mask_np = np.asarray(enemy_alive_mask, dtype=np.bool_)  # (10,)
    alive_fracs = np.array(ec.enemy_alive_fracs, dtype=np.float64)  # (10,)

    # Range bucket lookup: searchsorted gives index of first threshold >= dist
    # RANGE_THRESHOLDS is sorted ascending. searchsorted('right') gives the
    # index where dist would be inserted to keep sorted order.
    # For dist <= threshold[i], bucket = i. For dist > all, bucket = NUM_RANGE_THRESHOLDS-1.
    # np.searchsorted(thresholds, dist, side='left') gives first idx where thresh >= dist.
    # Clip to max bucket index.
    flat_dists = enemy_dists.ravel()  # (N*10,)
    flat_buckets = np.searchsorted(_RANGE_THRESHOLDS_NP, flat_dists, side='left')
    flat_buckets = np.minimum(flat_buckets, NUM_RANGE_THRESHOLDS - 1)
    buckets = flat_buckets.reshape(n_valid, MAX_UNITS_PER_SIDE)  # (N, 10)

    # --- 3.3 Offensive Value (10 features) [15:25] ---
    # Cover-aware path: when the caller precomputed per-(dest, enemy) cover-
    # aware expected damage (TERRAIN_SPEC §5.5), use it directly. Each row
    # already encodes "max wound fraction from this dest, with cover + LoS
    # gating + attacker alive frac applied." This is the value the spec
    # prescribes; the bucket-lookup path below is the cover-blind fallback.
    if dest_expected_damage is not None:
        off_vals = dest_expected_damage[:n_valid, :].astype(np.float32, copy=False)
        off_vals = off_vals.copy()
        off_vals[:, ~alive_mask_np] = 0.0
        features[:n_valid, 15:25] = off_vals
    elif unit_slot < len(friendly_ranged_matchups):
        # friendly_ranged_matchups[unit_slot, :, :] is (10, 7)
        fr_row = friendly_ranged_matchups[unit_slot]  # (10, 7)
        # Gather damage values: fr_row[ei, bucket[ci, ei]] for each (ci, ei)
        # Use advanced indexing: fr_row[arange(10), buckets] => (N, 10)
        ei_idx = np.arange(MAX_UNITS_PER_SIDE)
        off_vals = fr_row[ei_idx[None, :], buckets] * unit_alive_frac  # (N, 10)
        # Zero out dead enemies
        off_vals[:, ~alive_mask_np] = 0.0
        features[:n_valid, 15:25] = off_vals

    # --- 3.4 Per-Enemy Threat Features (50 features) [25:75] ---
    n_er = len(enemy_ranged_matchups) if enemy_ranged_matchups is not None else 0
    n_mm = len(melee_matchups) if melee_matchups is not None else 0
    enemy_adv = np.array(ec.enemy_advance_dists, dtype=np.float64)  # (10,)
    enemy_rush = np.array(ec.enemy_rush_dists, dtype=np.float64)    # (10,)
    enemy_act = np.array(ec.enemy_activated, dtype=np.float64)      # (10,)

    for ei in range(MAX_UNITS_PER_SIDE):
        if not alive_mask_np[ei]:
            continue
        base = 25 + ei * 5
        ed = enemy_dists[:, ei]         # (N,)
        eaf = alive_fracs[ei]
        bkt = buckets[:, ei]            # (N,)

        # +0: ranged_damage
        if ei < n_er:
            er_row = enemy_ranged_matchups[ei, unit_slot]  # (7,)
            features[:n_valid, base] = er_row[bkt] * eaf

        # +1: advance_shoot_damage
        if ei < n_er:
            eff_range = np.maximum(0.0, ed - enemy_adv[ei])
            adv_bkt = np.searchsorted(_RANGE_THRESHOLDS_NP, eff_range, side='left')
            adv_bkt = np.minimum(adv_bkt, NUM_RANGE_THRESHOLDS - 1)
            features[:n_valid, base + 1] = er_row[adv_bkt] * eaf

        # +2: can_charge
        charge_range = enemy_rush[ei] + 2.0
        can_charge = (ed < charge_range).astype(np.float32)
        features[:n_valid, base + 2] = can_charge

        # +3: melee_damage (where can_charge)
        if ei < n_mm:
            features[:n_valid, base + 3] = can_charge * melee_matchups[ei, unit_slot] * eaf

        # +4: has_activated
        features[:n_valid, base + 4] = enemy_act[ei]

    # --- 3.5 Advance-reachable flag (1 feature) [75] ---
    if advance_reachable is not None:
        ar_np = np.asarray(advance_reachable, dtype=np.bool_)
        features[:n_valid, 75] = ar_np[:n_valid].astype(np.float32)
        # Zero out offensive value features for rush-only candidates
        rush_only_mask = ~ar_np[:n_valid]
        if rush_only_mask.any():
            rush_only_indices = np.nonzero(rush_only_mask)[0]
            features[rush_only_indices, 15:25] = 0.0
    else:
        # Legacy fallback: assume all advance-reachable
        features[:n_valid, 75] = 1.0

    # --- 3.6 Per-hex terrain features (TERRAIN_SPEC.md §5.4) [76:82] ---
    # cover one-hot {sheltering, obscuring, blocking} at [76:79], movement
    # one-hot {open, difficult, impassible} at [79:82]. Movement defaults to
    # OPEN (so [79]=1) for cells not covered by terrain.
    if board is not None and board.terrain:
        from board import CoverType, MovementType
        for i in range(n_valid):
            piece = board.terrain_at_square.get(
                (int(cols_v[i]), int(rows_v[i])))
            if piece is None:
                features[i, 79] = 1.0  # OPEN
                continue
            ct = piece.cover_type
            mt = piece.movement_type
            if ct == CoverType.SHELTERING:
                features[i, 76] = 1.0
            elif ct == CoverType.OBSCURING:
                features[i, 77] = 1.0
            elif ct == CoverType.BLOCKING:
                features[i, 78] = 1.0
            if mt == MovementType.OPEN:
                features[i, 79] = 1.0
            elif mt == MovementType.DIFFICULT:
                features[i, 80] = 1.0
            elif mt == MovementType.IMPASSIBLE:
                features[i, 81] = 1.0
    else:
        # No terrain → all hexes are OPEN movement, no cover.
        features[:n_valid, 79] = 1.0

    return features


# ---------------------------------------------------------------------------
# Model-space position helpers
# ---------------------------------------------------------------------------

def _get_model_space_positions(
    units: list[UnitState], player: str,
) -> list[tuple[float, float]]:
    """Get model-space centre positions for up to 10 unit slots.

    Dead/missing slots get board centre sentinel.
    """
    sentinel = ((COLS - 1) / 2.0, (ROWS - 1) / 2.0)
    positions: list[tuple[float, float]] = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(units) and units[i].models_alive > 0:
            cx, cy = units[i].centre()
            if player == "B":
                cx = _flip_x(cx)
                cy = _flip_y(cy)
            positions.append((cx, cy))
        else:
            positions.append(sentinel)
    return positions


def _get_movement_budgets(
    units: list[UnitState],
) -> tuple[list[float], list[float]]:
    """Get advance and rush distances for up to 10 unit slots."""
    advance = []
    rush = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(units) and units[i].models_alive > 0:
            advance.append(float(units[i].unit.advance_distance))
            rush.append(float(units[i].unit.rush_distance))
        else:
            advance.append(0.0)
            rush.append(0.0)
    return advance, rush


# ---------------------------------------------------------------------------
# Decode + Execute (replaces old execute_ml_decision)
# ---------------------------------------------------------------------------

def _target_position(target: UnitState) -> tuple[int, int]:
    """Integer centre position of a target unit."""
    cx, cy = target.centre()
    return (int(round(cx)), int(round(cy)))


def execute_decoded_decision(
    unit: UnitState,
    enemies: list[UnitState],
    move_type: int,
    dest: tuple[float, float] | None,
    charge_target_idx: int,
    shoot_target_idx: int,
    is_advance_reachable: bool = True,
) -> tuple[str, tuple[int, int] | None, UnitState | None, str]:
    """Translate decoded model decision into (action, goal_position, charge_target, reason).

    Parameters
    ----------
    unit : the activated unit
    enemies : enemy unit list
    move_type : 0=move, 1=charge
    dest : (x, y) destination for move; None for charge
    charge_target_idx : enemy slot for charge
    shoot_target_idx : enemy slot for shooting (advance-reachable dest)
    is_advance_reachable : whether the chosen dest is advance-reachable (can shoot)
    """
    # Artillery: must hold
    if unit.unit.artillery:
        return ("hold", None, None, "artillery holds position")

    alive_enemies = [e for e in enemies if e.models_alive > 0]
    if not alive_enemies:
        return ("hold", None, None, "no enemies alive")

    if move_type == MOVE_CHARGE:
        if charge_target_idx < len(enemies) and enemies[charge_target_idx].models_alive > 0:
            target = enemies[charge_target_idx]
            tpos = _target_position(target)
            # Validate charge range (centre-to-centre)
            cx, cy = unit.centre()
            tx, ty = target.centre()
            threshold = unit.unit.rush_distance + 2
            if (cx - tx) ** 2 + (cy - ty) ** 2 < threshold * threshold:
                return ("charge", tpos, target, "model chose charge")
            # Out of charge range — rush toward target instead
            return ("rush", tpos, None, "charge target out of range, rushing toward it")
        # Fallback: charge failed (target dead), rush toward nearest enemy
        if alive_enemies:
            tpos = _target_position(alive_enemies[0])
            return ("rush", tpos, None, "charge target dead, rushing nearest")
        return ("hold", None, None, "charge target dead, no enemies")

    # MOVE_MOVE: dest determines hold/advance/rush
    if move_type == MOVE_MOVE:
        if dest is None:
            return ("hold", None, None, "move with no destination")
        goal = (int(round(dest[0])), int(round(dest[1])))
        # Check if dest is unit's current position (hold)
        cx, cy = unit.centre()
        centroid = (int(round(cx)), int(round(cy)))
        if goal == centroid:
            return ("hold", None, None, "model chose hold (dest=centroid)")
        if is_advance_reachable:
            return ("advance", goal, None, "model chose advance")
        else:
            return ("rush", goal, None, "model chose rush")

    return ("hold", None, None, f"unknown move_type: {move_type}")


# ---------------------------------------------------------------------------
# Coroutine-mode decode (used by _simulate_game_coroutine)
# ---------------------------------------------------------------------------

def decode_tactical_result(
    ir: InferenceResult,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    board: Board,
    player: str,
) -> tuple[UnitState | None, list[int], str, tuple[int, int] | None, UnitState | None, str]:
    """Unpack a fully-decoded InferenceResult into game-level decision.

    The coordinator has already done the full two-pass model decode (via
    batched_argmax_tactical). This function translates the decoded indices
    into game objects.  Destination is already in game-space (col, row).

    Returns (selected_unit, target_ranking, action, goal, charge_target, reason).
    """
    selected_idx = ir.unit_idx
    if selected_idx >= len(friendly_units) or friendly_units[selected_idx].models_alive <= 0:
        return None, [], "hold", None, None, "selected unit unavailable"
    selected_unit = friendly_units[selected_idx]

    move_type = ir.move_type

    dest = None
    if move_type == MOVE_MOVE:
        dest = (float(ir.dest_col), float(ir.dest_row))

    action, goal, charge_target, reason = execute_decoded_decision(
        selected_unit, enemy_units, move_type, dest, ir.charge_target_idx, ir.shoot_target_idx,
        is_advance_reachable=ir.is_advance_reachable,
    )

    return selected_unit, ir.target_ranking, action, goal, charge_target, reason


# ---------------------------------------------------------------------------
# Batched argmax tactical inference (for evolution / evaluation)
# ---------------------------------------------------------------------------

def batched_argmax_tactical(
    model: TacticalModel,
    requests: list[InferenceRequest],
) -> list[InferenceResult]:
    """Run batched two-pass argmax inference for multiple concurrent games.

    Pass 1: trunk → unit selection → move type → destination pointer → charge target
    Per-sample: compute post_move_rel from selected destination hex
    Pass 2 (batched): shoot target head with post-move features

    Returns one InferenceResult per request with fully-decoded decisions.

    NOTE: This function does NOT compute Dijkstra candidate sets because
    InferenceRequest does not carry unit/board state.  For batched evaluation
    (evolution), the caller should provide dest_features/dest_mask on the
    request, or this function falls back to the unit centroid for hold/charge
    and skips pointer for advance/rush when candidates are unavailable.
    Since batched_argmax is used in the hot eval path where we don't have
    Board objects, we run the pointer with empty candidates (producing
    centroid fallback).  For planning-quality inference, use
    apply_tactical_model which has full game state access.
    """
    n = len(requests)
    if n == 0:
        return []

    n_units = MAX_UNITS_PER_SIDE

    # Stack inputs
    state_batch = torch.stack([r.state_vec for r in requests])              # (N, feat)
    alive_batch = torch.stack([r.alive_mask for r in requests])             # (N, 10)
    enemy_alive_batch = torch.stack([r.enemy_alive_mask for r in requests]) # (N, 10)

    # Trunk
    h, units, _round_oh = model.trunk(state_batch)                            # (N, 512), (N, 20, 200), (N, 4)

    # Unit selection (argmax)
    unit_logits = model.unit_selection_head(h)                              # (N, 10)
    unit_logits = unit_logits.masked_fill(~alive_batch, float('-inf'))
    unit_indices = unit_logits.argmax(dim=-1)                               # (N,)

    # Extract per-sample unit features from unit embeddings
    unit_features = units[:, :n_units, :].gather(
        1, unit_indices.unsqueeze(1).unsqueeze(2).expand(n, 1, TACTICAL_UNIT_FEATURES),
    ).squeeze(1).detach()                                                   # (N, UF)

    # Extract can_charge mask for each sample's selected unit
    can_charge_batch = extract_can_charge_mask(state_batch, unit_indices)    # (N, 10)

    # Move type (argmax)
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf)                                # (N, 2)
    # Mask charge when no enemy is in charge range
    no_chargeable = ~can_charge_batch.any(dim=-1)                           # (N,)
    move_logits = move_logits.clone()
    move_logits[:, MOVE_CHARGE] = move_logits[:, MOVE_CHARGE].masked_fill(no_chargeable, float('-inf'))
    move_indices = move_logits.argmax(dim=-1)                               # (N,)
    move_onehot = F.one_hot(move_indices, NUM_MOVE_TYPES).float()           # (N, 2)

    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)            # (N, H+UF+2)

    # Destination pointer — always runs for MOVE_MOVE
    # Pad to batch-max candidate count (not global MAX_DEST_CANDIDATES)
    _batch_max_dc = 1
    if hasattr(requests[0], 'dest_features') and requests[0].dest_features is not None:
        for r in requests:
            _batch_max_dc = max(_batch_max_dc, r.dest_features.shape[0])
    dest_features_batch = torch.zeros(n, _batch_max_dc, DEST_FEATURE_DIM)
    dest_mask_batch = torch.zeros(n, _batch_max_dc, dtype=torch.bool)
    has_dest_data = False
    if hasattr(requests[0], 'dest_features') and requests[0].dest_features is not None:
        for i in range(n):
            nc = requests[i].dest_features.shape[0]
            dest_features_batch[i, :nc] = requests[i].dest_features
            dest_mask_batch[i, :nc] = requests[i].dest_mask[:nc]
        has_dest_data = True

    if has_dest_data:
        dest_logits = model.compute_dest_logits(h_uf_m, dest_features_batch, dest_mask_batch)
        dest_indices = dest_logits.argmax(dim=-1)  # (N,)
    else:
        dest_indices = torch.zeros(n, dtype=torch.long)  # fallback to centroid (idx 0)

    # Charge target (argmax) — mask by alive AND chargeable
    charge_logits = model.compute_charge_logits(
        h, units, unit_indices, enemy_alive_batch, can_charge_batch,
    )                                                                       # (N, 10)
    no_enemies = ~enemy_alive_batch.any(dim=-1)                             # (N,)
    charge_logits = charge_logits.masked_fill(no_enemies.unsqueeze(-1), 0.0)
    charge_indices = charge_logits.argmax(dim=-1)                           # (N,)

    # Value
    values = model.value_head(h, _round_oh)                                  # (N,) — head already squeezes

    # Per-sample: look up selected destination hex, compute post_move_rel
    unit_list = unit_indices.tolist()
    move_list = move_indices.tolist()
    dest_idx_list = dest_indices.tolist()

    pmr_tensors: list[torch.Tensor] = []
    dest_cols: list[int] = []
    dest_rows: list[int] = []
    adv_reachable_list: list[bool] = []
    for i in range(n):
        req = requests[i]
        uid = unit_list[i]
        mt = move_list[i]

        # Selected destination hex (game-space)
        is_ar = True  # advance-reachable flag for selected dest
        if mt == MOVE_MOVE and has_dest_data:
            didx = dest_idx_list[i]
            # Look up from request's candidate set
            if hasattr(req, 'dest_candidates') and req.dest_candidates is not None:
                dc = int(req.dest_candidates[didx, 0])
                dr = int(req.dest_candidates[didx, 1])
                if hasattr(req, 'dest_advance_reachable') and req.dest_advance_reachable is not None:
                    is_ar = bool(req.dest_advance_reachable[didx])
            else:
                # Fallback: centroid
                ucx, ucy = req.friendly_positions[uid]
                if req.player == "B":
                    ucx = _flip_x(ucx)
                    ucy = _flip_y(ucy)
                dc, dr = int(round(ucx)), int(round(ucy))
        else:
            # Charge: use unit centroid
            ucx, ucy = req.friendly_positions[uid]
            if req.player == "B":
                ucx = _flip_x(ucx)
                ucy = _flip_y(ucy)
            dc, dr = int(round(ucx)), int(round(ucy))

        dest_cols.append(dc)
        dest_rows.append(dr)
        adv_reachable_list.append(is_ar)

        # Compute post_move_rel in model-space
        px, py = float(dc), float(dr)
        if req.player == "B":
            px = _flip_x(px)
            py = _flip_y(py)
        if mt != MOVE_MOVE:
            px, py = req.friendly_positions[uid]

        pmr = compute_post_move_rel(px, py, req.enemy_positions)
        pmr_tensors.append(pmr)

    # Batched shoot pointer head with post-move features + in-range mask (pass 2)
    pmr_batch = torch.stack(pmr_tensors)                                    # (N, 30)
    max_wr_list = [requests[i].max_weapon_ranges[unit_list[i]] for i in range(n)]
    max_wr_t = torch.tensor(max_wr_list, dtype=torch.float32)
    shoot_mask_batch = compute_in_range_mask_batched(pmr_batch, max_wr_t, enemy_alive_batch)
    shoot_logits = model.compute_shoot_logits(
        h, units, unit_indices, pmr_batch, enemy_alive_batch,
        shoot_range_mask=shoot_mask_batch,
    )                                                                       # (N, 10)
    no_shootable = ~shoot_mask_batch.any(dim=-1)
    shoot_logits = shoot_logits.masked_fill(no_shootable.unsqueeze(-1), 0.0)
    shoot_indices = shoot_logits.argmax(dim=-1)                             # (N,)

    # Build results
    charge_list = charge_indices.tolist()
    shoot_list = shoot_indices.tolist()
    val_list = values.tolist()

    results: list[InferenceResult] = []
    for i in range(n):
        ranking = (
            list(range(n_units)) if no_shootable[i] else
            torch.argsort(shoot_logits[i], descending=True).tolist()
        )
        results.append(InferenceResult(
            unit_idx=unit_list[i],
            move_type=move_list[i],
            dest_col=dest_cols[i],
            dest_row=dest_rows[i],
            charge_target_idx=charge_list[i],
            shoot_target_idx=shoot_list[i],
            target_ranking=ranking,
            value=val_list[i],
            is_advance_reachable=adv_reachable_list[i],
        ))

    return results


# ---------------------------------------------------------------------------
# Two-phase batched inference (proper destination pointer support)
# ---------------------------------------------------------------------------

@torch.no_grad()
def batched_phase1_inference(
    model: TacticalModel,
    requests: list[Phase1Request],
) -> tuple[list[Phase1Result], list[dict]]:
    """Phase 1: trunk + unit selection + move type + charge target + value.

    Returns:
      - list of Phase1Result (decoded selections per request)
      - list of cache dicts to be retained by the coordinator until each
        request's matching Phase 2 call. Each cache holds the trunk tensors
        needed for the destination and shoot pointers in phase 2.
    """
    n = len(requests)
    if n == 0:
        return [], []

    n_units = MAX_UNITS_PER_SIDE

    state_batch = torch.stack([r.state_vec for r in requests])
    alive_batch = torch.stack([r.alive_mask for r in requests])
    enemy_alive_batch = torch.stack([r.enemy_alive_mask for r in requests])

    h, units, _round_oh = model.trunk(state_batch)

    # Unit selection
    unit_logits = model.unit_selection_head(h)
    unit_logits = unit_logits.masked_fill(~alive_batch, float('-inf'))
    unit_indices = unit_logits.argmax(dim=-1)

    unit_features = units[:, :n_units, :].gather(
        1, unit_indices.unsqueeze(1).unsqueeze(2).expand(n, 1, TACTICAL_UNIT_FEATURES),
    ).squeeze(1).detach()

    can_charge_batch = extract_can_charge_mask(state_batch, unit_indices)

    # Move type
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf)
    no_chargeable = ~can_charge_batch.any(dim=-1)
    move_logits = move_logits.clone()
    move_logits[:, MOVE_CHARGE] = move_logits[:, MOVE_CHARGE].masked_fill(
        no_chargeable, float('-inf'))
    move_indices = move_logits.argmax(dim=-1)
    move_onehot = F.one_hot(move_indices, NUM_MOVE_TYPES).float()

    # h_uf_m is needed by the destination pointer in phase 2.
    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)

    # Charge target
    charge_logits = model.compute_charge_logits(
        h, units, unit_indices, enemy_alive_batch, can_charge_batch,
    )
    no_enemies = ~enemy_alive_batch.any(dim=-1)
    charge_logits = charge_logits.masked_fill(no_enemies.unsqueeze(-1), 0.0)
    charge_indices = charge_logits.argmax(dim=-1)

    # Value
    values = model.value_head(h, _round_oh)

    unit_list = unit_indices.tolist()
    move_list = move_indices.tolist()
    charge_list = charge_indices.tolist()
    val_list = values.tolist()

    results: list[Phase1Result] = []
    caches: list[dict] = []
    for i in range(n):
        results.append(Phase1Result(
            unit_idx=unit_list[i],
            move_type=move_list[i],
            charge_target_idx=charge_list[i],
            value=val_list[i],
        ))
        caches.append({
            'h': h[i].detach().clone(),
            'units': units[i].detach().clone(),
            'h_uf_m': h_uf_m[i].detach().clone(),
            'enemy_alive': enemy_alive_batch[i].detach().clone(),
        })

    return results, caches


@torch.no_grad()
def batched_phase2_inference(
    model: TacticalModel,
    requests: list[Phase2Request],
    caches: list[dict],
) -> list[Phase2Result]:
    """Phase 2: destination pointer (over real Dijkstra candidates) + shoot
    target. Uses the trunk tensors cached in phase 1."""
    n = len(requests)
    if n == 0:
        return []

    n_units = MAX_UNITS_PER_SIDE

    # Re-batch cached trunk tensors
    h_batch = torch.stack([c['h'] for c in caches])
    units_batch = torch.stack([c['units'] for c in caches])
    h_uf_m_batch = torch.stack([c['h_uf_m'] for c in caches])
    enemy_alive_batch = torch.stack([c['enemy_alive'] for c in caches])

    # ----- Destination pointer -----
    # For MOVE_MOVE requests we received real Dijkstra candidates; for
    # MOVE_CHARGE we received a 1-row dummy. The pointer runs uniformly;
    # the result is only used for MOVE_MOVE.
    _batch_max_dc = max((r.dest_features.shape[0] for r in requests), default=1)
    dest_features_batch = torch.zeros(n, _batch_max_dc, DEST_FEATURE_DIM)
    dest_mask_batch = torch.zeros(n, _batch_max_dc, dtype=torch.bool)
    for i, r in enumerate(requests):
        nc = r.dest_features.shape[0]
        dest_features_batch[i, :nc] = r.dest_features
        dest_mask_batch[i, :nc] = r.dest_mask[:nc]

    dest_logits = model.compute_dest_logits(
        h_uf_m_batch, dest_features_batch, dest_mask_batch)
    dest_indices = dest_logits.argmax(dim=-1)
    dest_idx_list = dest_indices.tolist()

    # ----- Resolve destination + post-move position per request -----
    dest_cols: list[int] = []
    dest_rows: list[int] = []
    adv_reachable_list: list[bool] = []
    pmr_tensors: list[torch.Tensor] = []
    unit_indices_list: list[int] = []

    for i in range(n):
        r = requests[i]
        mt = r.move_type
        uid = r.unit_idx
        unit_indices_list.append(uid)

        if mt == MOVE_MOVE:
            didx = dest_idx_list[i]
            # candidates is numpy (C, 2)
            dc = int(r.candidates[didx, 0])
            dr = int(r.candidates[didx, 1])
            try:
                is_ar = bool(r.advance_reachable[didx])
            except (IndexError, TypeError):
                is_ar = True
        else:
            # Charge: post-move position is the unit centroid.
            ucx, ucy = r.friendly_positions[uid]
            if r.player == "B":
                ucx = _flip_x(ucx)
                ucy = _flip_y(ucy)
            dc, dr = int(round(ucx)), int(round(ucy))
            is_ar = True

        dest_cols.append(dc)
        dest_rows.append(dr)
        adv_reachable_list.append(is_ar)

        # post_move_rel uses model-space coords
        px, py = float(dc), float(dr)
        if r.player == "B":
            px = _flip_x(px)
            py = _flip_y(py)
        if mt != MOVE_MOVE:
            px, py = r.friendly_positions[uid]

        pmr = compute_post_move_rel(px, py, r.enemy_positions)
        pmr_tensors.append(pmr)

    # ----- Shoot pointer -----
    pmr_batch = torch.stack(pmr_tensors)
    unit_indices_t = torch.tensor(unit_indices_list, dtype=torch.long)
    max_wr_list = [requests[i].max_weapon_ranges[unit_indices_list[i]]
                   for i in range(n)]
    max_wr_t = torch.tensor(max_wr_list, dtype=torch.float32)
    shoot_mask_batch = compute_in_range_mask_batched(
        pmr_batch, max_wr_t, enemy_alive_batch)
    shoot_logits = model.compute_shoot_logits(
        h_batch, units_batch, unit_indices_t, pmr_batch, enemy_alive_batch,
        shoot_range_mask=shoot_mask_batch,
    )
    no_shootable = ~shoot_mask_batch.any(dim=-1)
    shoot_logits = shoot_logits.masked_fill(no_shootable.unsqueeze(-1), 0.0)
    shoot_indices = shoot_logits.argmax(dim=-1)
    shoot_list = shoot_indices.tolist()

    results: list[Phase2Result] = []
    for i in range(n):
        r = requests[i]
        ranking = (
            list(range(n_units)) if no_shootable[i] else
            torch.argsort(shoot_logits[i], descending=True).tolist()
        )
        results.append(Phase2Result(
            unit_idx=r.unit_idx,
            move_type=r.move_type,
            dest_col=dest_cols[i],
            dest_row=dest_rows[i],
            charge_target_idx=r.charge_target_idx,
            shoot_target_idx=shoot_list[i],
            target_ranking=ranking,
            value=r.value,
            is_advance_reachable=adv_reachable_list[i],
        ))

    return results


# ---------------------------------------------------------------------------
# Main integration entry point (argmax)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _apply_tactical_model_phased(
    model: TacticalModel,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    *,
    friendly_ranged_matchups=None,
    friendly_melee_matchups=None,
    enemy_ranged_matchups=None,
    enemy_melee_matchups=None,
    total_friendly_points: int | None = None,
    total_enemy_points: int | None = None,
) -> tuple[UnitState | None, list[int], str, tuple[int, int] | None, UnitState | None, str, dict]:
    """Phase-reencode inference path.

    Mirrors apply_tactical_model's signature and return shape but replaces the
    single-trunk-plus-forward-twice structure with four explicit model.encode()
    calls — one per phase (PRE_SELECT, POST_SELECT, POST_MOVETYPE, POST_DEST).
    h persists across phases via h_prev; h0 recomputes each phase from the
    current features. The POST_DEST encode consumes a post-move state_vec
    built via project_post_move_unit_state (for MOVE_MOVE, not shaken);
    otherwise the pre-move state_vec is reused.

    At identity init (FiLM γ=1/β=0, is_acting_embed zero) and with all
    continuation-phase iter counts set to 0, this path produces bit-identical
    head outputs to the legacy apply_tactical_model — the Step 6 ablation gate.
    """
    global _timing_encode_s, _timing_forward_s, _timing_calls

    # Build alive+unactivated mask
    alive_mask_list = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(friendly_units):
            us = friendly_units[i]
            alive_mask_list.append(us.models_alive > 0 and not us.activated)
        else:
            alive_mask_list.append(False)

    if not any(alive_mask_list):
        return None, [], "hold", None, None, "no units available", {}

    alive_mask = torch.tensor(alive_mask_list, dtype=torch.bool)
    enemy_alive_mask_list = [(i < len(enemy_units) and enemy_units[i].models_alive > 0)
                              for i in range(MAX_UNITS_PER_SIDE)]
    enemy_alive_mask = torch.tensor(enemy_alive_mask_list, dtype=torch.bool)
    enemy_alive_np = np.array(enemy_alive_mask_list, dtype=np.bool_)

    # Pre-move state encoding
    _t0 = time.perf_counter()
    state_vec = encode_state_tactical(
        friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=friendly_ranged_matchups,
        friendly_melee_matchups=friendly_melee_matchups,
        enemy_ranged_matchups=enemy_ranged_matchups,
        enemy_melee_matchups=enemy_melee_matchups,
        total_friendly_points=total_friendly_points,
        total_enemy_points=total_enemy_points,
    )
    _t1 = time.perf_counter()
    _timing_encode_s += _t1 - _t0

    friendly_positions = _get_model_space_positions(friendly_units, player)
    enemy_positions = _get_model_space_positions(enemy_units, player)

    if friendly_ranged_matchups is None or friendly_melee_matchups is None:
        friendly_ranged_matchups, friendly_melee_matchups = precompute_damage(
            [u.unit for u in friendly_units], [u.unit for u in enemy_units])
    if enemy_ranged_matchups is None or enemy_melee_matchups is None:
        enemy_ranged_matchups, enemy_melee_matchups = precompute_damage(
            [u.unit for u in enemy_units], [u.unit for u in friendly_units])

    state_batched = state_vec.unsqueeze(0)
    enemy_alive_batched = enemy_alive_mask.unsqueeze(0)

    _tf0 = time.perf_counter()

    # --- PHASE PRE_SELECT: unit selection ---
    h_pre, _units_pre, round_onehot = model.encode(
        state_batched, phase=PHASE_PRE_SELECT, acting_unit_idx=None, h_prev=None,
    )
    unit_logits = model.unit_selection_head(h_pre).squeeze(0)
    unit_logits = unit_logits.masked_fill(~alive_mask, float('-inf'))
    selected_idx = int(unit_logits.argmax().item())
    selected_unit = friendly_units[selected_idx]

    is_shaken = bool(extract_is_shaken(state_vec, selected_idx).item())
    can_charge_mask = extract_can_charge_mask(state_vec, selected_idx)

    # --- PHASE POST_SELECT: move-type head ---
    h_sel, units_sel, _ = model.encode(
        state_batched, phase=PHASE_POST_SELECT,
        acting_unit_idx=selected_idx, h_prev=h_pre,
    )
    unit_features = model._extract_unit_features(units_sel, selected_idx)
    h_uf = torch.cat([h_sel, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf).squeeze(0)
    if not can_charge_mask.any():
        move_logits = move_logits.clone()
        move_logits[MOVE_CHARGE] = float('-inf')
    if is_shaken:
        move_type = MOVE_MOVE
    else:
        move_type = int(move_logits.argmax().item())

    # --- PHASE POST_MOVETYPE: destination head ---
    h_mt, units_mt, _ = model.encode(
        state_batched, phase=PHASE_POST_MOVETYPE,
        acting_unit_idx=selected_idx, h_prev=h_sel,
    )

    unit_cx_ms, unit_cy_ms = friendly_positions[selected_idx]
    dest_col, dest_row = int(round(unit_cx_ms)), int(round(unit_cy_ms))
    dest_n_candidates = 0
    dest_top3: list[tuple[int, int, float]] = []
    dest_entropy = 0.0
    is_advance_reachable = True

    # Cover-aware expected damage arrays — used by both the dest-feature
    # offensive block [15:25] and the shoot head override (TERRAIN_SPEC §5.4).
    _e_dmg_table = getattr(board, 'expected_damage_table', None)
    _dest_dmg_arr = None
    _static_dmg_arr = None
    if move_type == MOVE_MOVE and not is_shaken:
        enemy_pos_set: set[tuple[int, int]] = set()
        for eu in enemy_units:
            if eu.models_alive > 0:
                for pos in eu.alive_positions():
                    enemy_pos_set.add(pos)

        candidates, cand_mask, adv_reachable = compute_destination_candidates(
            selected_unit, board, enemy_pos_set, player)
        _dest_dmg_arr, _static_dmg_arr = compute_unit_expected_damage_arrays(
            selected_unit, candidates, cand_mask, enemy_units, board, _e_dmg_table)
        budget = float(selected_unit.unit.rush_distance)
        dest_feats = compute_destination_features(
            candidates, cand_mask, selected_unit, selected_idx, player,
            enemy_units, enemy_alive_np,
            friendly_ranged_matchups, enemy_ranged_matchups, enemy_melee_matchups,
            budget, advance_reachable=adv_reachable, board=board,
            dest_expected_damage=_dest_dmg_arr)

        dest_features_t = torch.from_numpy(dest_feats).float()
        dest_mask_t = torch.from_numpy(cand_mask)

        move_onehot = F.one_hot(torch.tensor(move_type), NUM_MOVE_TYPES).float()
        uf_mt = model._extract_unit_features(units_mt, selected_idx).detach()
        h_uf_m = torch.cat([h_mt, uf_mt, move_onehot.unsqueeze(0)], dim=-1)
        dest_logits = model.compute_dest_logits(
            h_uf_m, dest_features_t.unsqueeze(0), dest_mask_t.unsqueeze(0)).squeeze(0)

        dest_idx = int(dest_logits.argmax().item())
        dest_col = int(candidates[dest_idx, 0])
        dest_row = int(candidates[dest_idx, 1])
        dest_n_candidates = int(cand_mask.sum())
        is_advance_reachable = bool(adv_reachable[dest_idx])
    else:
        # No move (charge / shaken): only the static damage row is needed for
        # the shoot head override.
        _, _static_dmg_arr = compute_unit_expected_damage_arrays(
            selected_unit, np.zeros((MAX_DEST_CANDIDATES, 2), dtype=np.int32),
            np.zeros(MAX_DEST_CANDIDATES, dtype=np.bool_),
            enemy_units, board, _e_dmg_table)

        dest_probs = torch.softmax(dest_logits, dim=-1)
        top_k = min(3, dest_n_candidates)
        top_vals, top_idxs = torch.topk(dest_probs, top_k)
        dest_top3 = [(int(candidates[ti, 0]), int(candidates[ti, 1]), tv.item())
                     for ti, tv in zip(top_idxs.tolist(), top_vals)]
        dest_entropy = -(dest_probs * torch.log(dest_probs + 1e-8)).sum().item()

    # post_move_rel: kept during transition (Steps 3–7) as the shoot head's side
    # channel alongside the POST_DEST h; removed in Step 8 once validated.
    if move_type == MOVE_MOVE and not is_shaken:
        px, py = float(dest_col), float(dest_row)
        if player == "B":
            px = _flip_x(px)
            py = _flip_y(py)
    else:
        px, py = unit_cx_ms, unit_cy_ms
    post_move_rel = compute_post_move_rel(px, py, enemy_positions)

    # --- PHASE POST_DEST: build post-move state_vec, encode, run charge + shoot heads ---
    if move_type == MOVE_MOVE and not is_shaken:
        is_rush = not is_advance_reachable
        post_unit = project_post_move_unit_state(
            selected_unit, (dest_col, dest_row), is_rush=is_rush)
        friendly_post = list(friendly_units)
        friendly_post[selected_idx] = post_unit

        _t2 = time.perf_counter()
        state_vec_post = encode_state_tactical(
            friendly_post, enemy_units, round_num, board, player,
            friendly_ranged_matchups=friendly_ranged_matchups,
            friendly_melee_matchups=friendly_melee_matchups,
            enemy_ranged_matchups=enemy_ranged_matchups,
            enemy_melee_matchups=enemy_melee_matchups,
            total_friendly_points=total_friendly_points,
            total_enemy_points=total_enemy_points,
        )
        _t3 = time.perf_counter()
        _timing_encode_s += _t3 - _t2
    else:
        state_vec_post = state_vec

    h_dest, units_dest, _ = model.encode(
        state_vec_post.unsqueeze(0), phase=PHASE_POST_DEST,
        acting_unit_idx=selected_idx, h_prev=h_mt,
    )

    charge_logits_b = model.compute_charge_logits(
        h_dest, units_dest, selected_idx,
        enemy_alive_batched, can_charge_mask.unsqueeze(0),
    ).squeeze(0)

    # Cover-aware expected wound frac for the chosen activation context.
    _ewf_override = None
    if _e_dmg_table is not None:
        if move_type == MOVE_MOVE and not is_shaken and _dest_dmg_arr is not None:
            _ewf_override = torch.from_numpy(_dest_dmg_arr[dest_idx].astype(np.float32)).unsqueeze(0)
        elif _static_dmg_arr is not None:
            _ewf_override = torch.from_numpy(_static_dmg_arr.astype(np.float32)).unsqueeze(0)

    shoot_logits_b = model.compute_shoot_logits(
        h_dest, units_dest, selected_idx,
        post_move_rel.unsqueeze(0), enemy_alive_batched,
        shoot_range_mask=None,
        expected_wound_frac_override=_ewf_override,
    ).squeeze(0)

    max_wr = max(
        (w.range_inches for w in selected_unit.unit.weapons if not w.melee),
        default=0.0,
    )
    shoot_range_mask = compute_in_range_mask(post_move_rel, float(max_wr), enemy_alive_mask)
    if move_type == MOVE_MOVE and not is_shaken:
        shooter_gs = (float(dest_col), float(dest_row))
    else:
        shooter_gs = selected_unit.centre()
    vis_mask = compute_visibility_mask(shooter_gs, enemy_units, board)
    shoot_range_mask = shoot_range_mask & vis_mask
    if is_shaken:
        shoot_range_mask = torch.zeros_like(shoot_range_mask)
    masked_shoot_logits = shoot_logits_b.masked_fill(~shoot_range_mask, float('-inf'))

    charge_target_idx = int(charge_logits_b.argmax().item()) if enemy_alive_mask.any() else 0
    shoot_target_idx = int(masked_shoot_logits.argmax().item()) if shoot_range_mask.any() else 0
    target_ranking = torch.argsort(masked_shoot_logits, descending=True).tolist()

    # Value: main V head reads h_pre (same GAE target as legacy). Per-phase V heads
    # are computed but not used at inference — they become load-bearing in Step 5.
    opp_embed = model._get_opp_embed(h_pre, None)
    side_embed = model._get_side_embed(h_pre, None)
    value = model.value_head(h_pre, round_onehot, opp_embed, side_embed).squeeze(0)

    _tf1 = time.perf_counter()
    _timing_forward_s += _tf1 - _tf0
    _timing_calls += 1

    # Execute decision
    dest = None
    if move_type == MOVE_MOVE:
        dest = (float(dest_col), float(dest_row))
    action, goal, charge_target, reason = execute_decoded_decision(
        selected_unit, enemy_units, move_type, dest, charge_target_idx, shoot_target_idx,
        is_advance_reachable=is_advance_reachable,
    )

    move_conf = torch.softmax(move_logits, dim=-1)
    assessment = {
        'value': value.item(),
        'selected_slot': selected_idx,
        'selected_name': selected_unit.unit.name,
        'unit_selection_logits': unit_logits.tolist(),
        'move_type': MOVE_TYPE_NAMES[move_type],
        'move_type_confidence': move_conf[move_type].item(),
        'move_type_probs': move_conf.tolist(),
        'dest_selected': (dest_col, dest_row),
        'dest_n_candidates': dest_n_candidates,
        'dest_top3': dest_top3,
        'dest_entropy': dest_entropy,
        'charge_target_idx': charge_target_idx,
        'charge_target_logits': charge_logits_b.tolist(),
        'shoot_target_idx': shoot_target_idx,
        'target_ranking': target_ranking,
        'target_scores': torch.softmax(masked_shoot_logits, dim=-1).tolist(),
        'action': action,
        'reason': reason,
        'friendly_names': [
            fu.unit.name if i < len(friendly_units) and friendly_units[i].models_alive > 0
            else None
            for i, fu in enumerate(friendly_units)
        ] + [None] * (MAX_UNITS_PER_SIDE - len(friendly_units)),
        'enemy_names': [
            eu.unit.name if i < len(enemy_units) and enemy_units[i].models_alive > 0
            else None
            for i, eu in enumerate(enemy_units)
        ] + [None] * (MAX_UNITS_PER_SIDE - len(enemy_units)),
    }

    # Auxiliary prediction heads — read h_pre (pre-move scope matches training targets).
    # One trunk call at most (aux shares h_pre with the main value head), down from
    # the legacy path's redundant third model.trunk() call.
    if hasattr(model, 'aux_friendly_survival_head'):
        fs_raw = model.aux_friendly_survival_head(h_pre).view(MAX_UNITS_PER_SIDE, 2)
        fs_alpha = F.softplus(fs_raw[:, 0]) + 0.01
        fs_beta = F.softplus(fs_raw[:, 1]) + 0.01
        fs_mean = (fs_alpha / (fs_alpha + fs_beta)).tolist()
        assessment['friendly_survival'] = fs_mean

        es_raw = model.aux_enemy_survival_head(h_pre).view(MAX_UNITS_PER_SIDE, 2)
        es_alpha = F.softplus(es_raw[:, 0]) + 0.01
        es_beta = F.softplus(es_raw[:, 1]) + 0.01
        es_mean = (es_alpha / (es_alpha + es_beta)).tolist()
        assessment['enemy_survival'] = es_mean

        obj_logits = model.aux_obj_control_head(h_pre).view(5, 3)
        obj_probs = torch.softmax(obj_logits, dim=-1).tolist()
        assessment['obj_control_probs'] = obj_probs

        if hasattr(model, 'aux_friendly_activations_head'):
            f_act = F.softplus(model.aux_friendly_activations_head(h_pre).squeeze(-1)).item()
            e_act = F.softplus(model.aux_enemy_activations_head(h_pre).squeeze(-1)).item()
            assessment['friendly_activations_remaining'] = f_act
            assessment['enemy_activations_remaining'] = e_act

    return selected_unit, target_ranking, action, goal, charge_target, reason, assessment


@torch.no_grad()
def apply_tactical_model(
    model: TacticalModel,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    *,
    friendly_ranged_matchups=None,
    friendly_melee_matchups=None,
    enemy_ranged_matchups=None,
    enemy_melee_matchups=None,
    total_friendly_points: int | None = None,
    total_enemy_points: int | None = None,
) -> tuple[UnitState | None, list[int], str, tuple[int, int] | None, UnitState | None, str, dict]:
    """Encode state, run tactical model, decode and execute one activation.

    Returns (selected_unit, target_ranking, action, goal, charge_target, reason, assessment).
    """
    if _PHASE_REENCODE_ENABLED:
        return _apply_tactical_model_phased(
            model, friendly_units, enemy_units, round_num, board, player,
            friendly_ranged_matchups=friendly_ranged_matchups,
            friendly_melee_matchups=friendly_melee_matchups,
            enemy_ranged_matchups=enemy_ranged_matchups,
            enemy_melee_matchups=enemy_melee_matchups,
            total_friendly_points=total_friendly_points,
            total_enemy_points=total_enemy_points,
        )

    global _timing_encode_s, _timing_forward_s, _timing_calls

    # Build alive+unactivated mask
    alive_mask_list = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(friendly_units):
            us = friendly_units[i]
            alive_mask_list.append(us.models_alive > 0 and not us.activated)
        else:
            alive_mask_list.append(False)

    if not any(alive_mask_list):
        return None, [], "hold", None, None, "no units available", {}

    alive_mask = torch.tensor(alive_mask_list, dtype=torch.bool)

    enemy_alive_mask_list = [(i < len(enemy_units) and enemy_units[i].models_alive > 0)
                              for i in range(MAX_UNITS_PER_SIDE)]
    enemy_alive_mask = torch.tensor(enemy_alive_mask_list, dtype=torch.bool)
    enemy_alive_np = np.array(enemy_alive_mask_list, dtype=np.bool_)

    # Encode state
    _t0 = time.perf_counter()
    state_vec = encode_state_tactical(
        friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=friendly_ranged_matchups,
        friendly_melee_matchups=friendly_melee_matchups,
        enemy_ranged_matchups=enemy_ranged_matchups,
        enemy_melee_matchups=enemy_melee_matchups,
        total_friendly_points=total_friendly_points,
        total_enemy_points=total_enemy_points,
    )
    _t1 = time.perf_counter()
    _timing_encode_s += _t1 - _t0

    # Get model-space positions for post-move computation
    friendly_positions = _get_model_space_positions(friendly_units, player)
    enemy_positions = _get_model_space_positions(enemy_units, player)

    # Precompute matchups if not provided (needed for destination features)
    if friendly_ranged_matchups is None or friendly_melee_matchups is None:
        friendly_ranged_matchups, friendly_melee_matchups = precompute_damage(
            [u.unit for u in friendly_units], [u.unit for u in enemy_units])
    if enemy_ranged_matchups is None or enemy_melee_matchups is None:
        enemy_ranged_matchups, enemy_melee_matchups = precompute_damage(
            [u.unit for u in enemy_units], [u.unit for u in friendly_units])

    # --- Forward pass 1: get unit + move_type ---
    _t2 = time.perf_counter()
    out = model(state_vec, alive_mask, enemy_alive_mask)
    _t3 = time.perf_counter()
    _timing_forward_s += _t3 - _t2
    _timing_calls += 1

    # Decode unit selection
    selected_idx = int(out.unit_logits.argmax().item())
    selected_unit = friendly_units[selected_idx]

    # Decode move type — Shaken units must hold (force MOVE_MOVE)
    is_shaken = extract_is_shaken(state_vec, selected_idx).item()
    if is_shaken:
        move_type = MOVE_MOVE
    else:
        move_type = int(out.move_logits.argmax().item())

    # --- Destination pointer (always for MOVE_MOVE) ---
    unit_cx_ms, unit_cy_ms = friendly_positions[selected_idx]
    dest_col, dest_row = int(round(unit_cx_ms)), int(round(unit_cy_ms))
    dest_n_candidates = 0
    dest_top3 = []
    dest_entropy = 0.0
    is_advance_reachable = True  # default for centroid / shaken

    _e_dmg_table = getattr(board, 'expected_damage_table', None)
    _dest_dmg_arr = None
    _static_dmg_arr = None
    if move_type == MOVE_MOVE and not is_shaken:
        # Build enemy position set for Dijkstra
        enemy_pos_set: set[tuple[int, int]] = set()
        for eu in enemy_units:
            if eu.models_alive > 0:
                for pos in eu.alive_positions():
                    enemy_pos_set.add(pos)

        candidates, cand_mask, adv_reachable = compute_destination_candidates(
            selected_unit, board, enemy_pos_set, player)
        _dest_dmg_arr, _static_dmg_arr = compute_unit_expected_damage_arrays(
            selected_unit, candidates, cand_mask, enemy_units, board, _e_dmg_table)
        budget = float(selected_unit.unit.rush_distance)
        dest_feats = compute_destination_features(
            candidates, cand_mask, selected_unit, selected_idx, player,
            enemy_units, enemy_alive_np,
            friendly_ranged_matchups, enemy_ranged_matchups, enemy_melee_matchups,
            budget, advance_reachable=adv_reachable, board=board,
            dest_expected_damage=_dest_dmg_arr)

        dest_features_t = torch.from_numpy(dest_feats).float()
        dest_mask_t = torch.from_numpy(cand_mask)

        # Run pointer
        move_onehot = F.one_hot(torch.tensor(move_type), NUM_MOVE_TYPES).float()
        h_trunk, units_trunk, _ = model.trunk(state_vec.unsqueeze(0))
        uf = model._extract_unit_features(units_trunk, selected_idx).detach()
        h_uf_m = torch.cat([h_trunk, uf, move_onehot.unsqueeze(0)], dim=-1)
        dest_logits = model.compute_dest_logits(
            h_uf_m, dest_features_t.unsqueeze(0), dest_mask_t.unsqueeze(0)).squeeze(0)

        dest_idx = int(dest_logits.argmax().item())
        dest_col = int(candidates[dest_idx, 0])
        dest_row = int(candidates[dest_idx, 1])
        dest_n_candidates = int(cand_mask.sum())
        is_advance_reachable = bool(adv_reachable[dest_idx])

        # Top-3 for diagnostics
        dest_probs = torch.softmax(dest_logits, dim=-1)
        top_k = min(3, dest_n_candidates)
        top_vals, top_idxs = torch.topk(dest_probs, top_k)
        dest_top3 = [(int(candidates[ti, 0]), int(candidates[ti, 1]), tv.item())
                     for ti, tv in zip(top_idxs.tolist(), top_vals)]
        dest_entropy = -(dest_probs * torch.log(dest_probs + 1e-8)).sum().item()

    # Compute post_move_rel from selected hex in model-space
    if move_type == MOVE_MOVE and not is_shaken:
        px, py = float(dest_col), float(dest_row)
        if player == "B":
            px = _flip_x(px)
            py = _flip_y(py)
    else:
        px, py = unit_cx_ms, unit_cy_ms
        # Static damage for non-move (charge/shaken) — needed for shoot head override.
        if _static_dmg_arr is None and _e_dmg_table is not None:
            _, _static_dmg_arr = compute_unit_expected_damage_arrays(
                selected_unit, np.zeros((MAX_DEST_CANDIDATES, 2), dtype=np.int32),
                np.zeros(MAX_DEST_CANDIDATES, dtype=np.bool_),
                enemy_units, board, _e_dmg_table)
    post_move_rel = compute_post_move_rel(px, py, enemy_positions)

    # Cover-aware expected wound frac override for the shoot head.
    _ewf_override = None
    if _e_dmg_table is not None:
        if move_type == MOVE_MOVE and not is_shaken and _dest_dmg_arr is not None:
            _ewf_override = torch.from_numpy(_dest_dmg_arr[dest_idx].astype(np.float32))
        elif _static_dmg_arr is not None:
            _ewf_override = torch.from_numpy(_static_dmg_arr.astype(np.float32))

    # --- Forward pass 2: re-run with post-move features for shooting head ---
    out2 = model(state_vec, alive_mask, enemy_alive_mask,
                 forced_unit_idx=selected_idx, post_move_rel=post_move_rel,
                 expected_wound_frac_override=_ewf_override)

    # Apply in-range mask to shoot logits
    max_wr = max(
        (w.range_inches for w in selected_unit.unit.weapons if not w.melee),
        default=0.0,
    )
    shoot_range_mask = compute_in_range_mask(post_move_rel, float(max_wr), enemy_alive_mask)
    if move_type == MOVE_MOVE and not is_shaken:
        shooter_gs = (float(dest_col), float(dest_row))
    else:
        shooter_gs = selected_unit.centre()
    vis_mask = compute_visibility_mask(shooter_gs, enemy_units, board)
    shoot_range_mask = shoot_range_mask & vis_mask
    if is_shaken:
        shoot_range_mask = torch.zeros_like(shoot_range_mask)
    masked_shoot_logits = out2.shoot_target_logits.masked_fill(~shoot_range_mask, float('-inf'))

    # Decode targets
    charge_target_idx = int(out2.charge_target_logits.argmax().item()) if enemy_alive_mask.any() else 0
    shoot_target_idx = int(masked_shoot_logits.argmax().item()) if shoot_range_mask.any() else 0

    target_ranking = torch.argsort(masked_shoot_logits, descending=True).tolist()

    # Compute destination in game-space
    dest = None
    if move_type == MOVE_MOVE:
        dest = (float(dest_col), float(dest_row))

    # Execute decision
    action, goal, charge_target, reason = execute_decoded_decision(
        selected_unit, enemy_units, move_type, dest, charge_target_idx, shoot_target_idx,
        is_advance_reachable=is_advance_reachable,
    )

    # Assessment for viewer / diagnostics
    move_conf = torch.softmax(out.move_logits, dim=-1)
    assessment = {
        'value': out.value.item(),
        'selected_slot': selected_idx,
        'selected_name': selected_unit.unit.name,
        'unit_selection_logits': out.unit_logits.tolist(),
        'move_type': MOVE_TYPE_NAMES[move_type],
        'move_type_confidence': move_conf[move_type].item(),
        'move_type_probs': move_conf.tolist(),
        'dest_selected': (dest_col, dest_row),
        'dest_n_candidates': dest_n_candidates,
        'dest_top3': dest_top3,
        'dest_entropy': dest_entropy,
        'charge_target_idx': charge_target_idx,
        'charge_target_logits': out2.charge_target_logits.tolist(),
        'shoot_target_idx': shoot_target_idx,
        'target_ranking': target_ranking,
        'target_scores': torch.softmax(masked_shoot_logits, dim=-1).tolist(),
        'action': action,
        'reason': reason,
        'friendly_names': [
            fu.unit.name if i < len(friendly_units) and friendly_units[i].models_alive > 0
            else None
            for i, fu in enumerate(friendly_units)
        ] + [None] * (MAX_UNITS_PER_SIDE - len(friendly_units)),
        'enemy_names': [
            eu.unit.name if i < len(enemy_units) and enemy_units[i].models_alive > 0
            else None
            for i, eu in enumerate(enemy_units)
        ] + [None] * (MAX_UNITS_PER_SIDE - len(enemy_units)),
    }

    # Auxiliary prediction heads (survival + objective control)
    if hasattr(model, 'aux_friendly_survival_head'):
        h_aux, _u, _ = model.trunk(state_vec.unsqueeze(0))  # (1, H)
        fs_raw = model.aux_friendly_survival_head(h_aux).view(MAX_UNITS_PER_SIDE, 2)
        fs_alpha = F.softplus(fs_raw[:, 0]) + 0.01
        fs_beta = F.softplus(fs_raw[:, 1]) + 0.01
        fs_mean = (fs_alpha / (fs_alpha + fs_beta)).tolist()
        assessment['friendly_survival'] = fs_mean

        es_raw = model.aux_enemy_survival_head(h_aux).view(MAX_UNITS_PER_SIDE, 2)
        es_alpha = F.softplus(es_raw[:, 0]) + 0.01
        es_beta = F.softplus(es_raw[:, 1]) + 0.01
        es_mean = (es_alpha / (es_alpha + es_beta)).tolist()
        assessment['enemy_survival'] = es_mean

        obj_logits = model.aux_obj_control_head(h_aux).view(5, 3)
        obj_probs = torch.softmax(obj_logits, dim=-1).tolist()
        assessment['obj_control_probs'] = obj_probs

        if hasattr(model, 'aux_friendly_activations_head'):
            f_act = F.softplus(model.aux_friendly_activations_head(h_aux).squeeze(-1)).item()
            e_act = F.softplus(model.aux_enemy_activations_head(h_aux).squeeze(-1)).item()
            assessment['friendly_activations_remaining'] = f_act
            assessment['enemy_activations_remaining'] = e_act

    return selected_unit, target_ranking, action, goal, charge_target, reason, assessment


# ---------------------------------------------------------------------------
# Sampling variant (for self-play opponents during training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_tactical_model_sampling(
    model: TacticalModel,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    *,
    friendly_ranged_matchups=None,
    friendly_melee_matchups=None,
    enemy_ranged_matchups=None,
    enemy_melee_matchups=None,
    total_friendly_points: int | None = None,
    total_enemy_points: int | None = None,
) -> tuple[UnitState | None, list[int], str, tuple[int, int] | None, UnitState | None, str, dict]:
    """Like apply_tactical_model but samples from distributions instead of argmax."""
    # Build masks
    alive_mask_list = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(friendly_units):
            us = friendly_units[i]
            alive_mask_list.append(us.models_alive > 0 and not us.activated)
        else:
            alive_mask_list.append(False)

    if not any(alive_mask_list):
        return None, [], "hold", None, None, "no units available", {}

    alive_mask = torch.tensor(alive_mask_list, dtype=torch.bool)
    enemy_alive_mask_list = [(i < len(enemy_units) and enemy_units[i].models_alive > 0)
                              for i in range(MAX_UNITS_PER_SIDE)]
    enemy_alive_mask = torch.tensor(enemy_alive_mask_list, dtype=torch.bool)
    enemy_alive_np = np.array(enemy_alive_mask_list, dtype=np.bool_)

    state_vec = encode_state_tactical(
        friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=friendly_ranged_matchups,
        friendly_melee_matchups=friendly_melee_matchups,
        enemy_ranged_matchups=enemy_ranged_matchups,
        enemy_melee_matchups=enemy_melee_matchups,
        total_friendly_points=total_friendly_points,
        total_enemy_points=total_enemy_points,
    )

    # Precompute matchups if needed
    if friendly_ranged_matchups is None or friendly_melee_matchups is None:
        friendly_ranged_matchups, friendly_melee_matchups = precompute_damage(
            [u.unit for u in friendly_units], [u.unit for u in enemy_units])
    if enemy_ranged_matchups is None or enemy_melee_matchups is None:
        enemy_ranged_matchups, enemy_melee_matchups = precompute_damage(
            [u.unit for u in enemy_units], [u.unit for u in friendly_units])

    friendly_positions = _get_model_space_positions(friendly_units, player)
    enemy_positions_ms = _get_model_space_positions(enemy_units, player)

    x = state_vec.unsqueeze(0)
    am = alive_mask.unsqueeze(0)

    # --- Trunk ---
    h, units, _ = model.trunk(x)

    # --- Unit selection (sample) ---
    unit_logits = model.unit_selection_head(h)
    unit_logits = unit_logits.masked_fill(~am, float('-inf'))
    unit_probs = torch.softmax(unit_logits, dim=-1).squeeze(0)
    selected_idx = int(torch.multinomial(unit_probs, 1).item())
    selected_unit = friendly_units[selected_idx]

    unit_features = model._extract_unit_features(units, selected_idx).detach()

    can_charge_mask = extract_can_charge_mask(state_vec, selected_idx)
    is_shaken = extract_is_shaken(state_vec, selected_idx).item()

    # --- Move type head (sample) ---
    h_uf = torch.cat([h, unit_features], dim=-1)
    move_logits = model.move_type_head(h_uf).squeeze(0)
    if not can_charge_mask.any():
        move_logits = move_logits.clone()
        move_logits[MOVE_CHARGE] = float('-inf')
    if is_shaken:
        move_logits = move_logits.clone()
        move_logits[MOVE_CHARGE] = float('-inf')
    move_probs = torch.softmax(move_logits, dim=-1)
    move_type = int(torch.multinomial(move_probs, 1).item())

    move_onehot = F.one_hot(torch.tensor(move_type), NUM_MOVE_TYPES).float().unsqueeze(0)
    h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)

    # --- Destination pointer (sample, MOVE_MOVE only) ---
    unit_cx_ms, unit_cy_ms = friendly_positions[selected_idx]
    dest_col, dest_row = int(round(unit_cx_ms)), int(round(unit_cy_ms))
    if player == "B":
        dest_col_gs = int(round(_flip_x(unit_cx_ms)))
        dest_row_gs = int(round(_flip_y(unit_cy_ms)))
    else:
        dest_col_gs, dest_row_gs = dest_col, dest_row
    _is_ar = True
    _e_dmg_table = getattr(board, 'expected_damage_table', None)
    _dest_dmg_arr = None
    _static_dmg_arr = None

    if move_type == MOVE_MOVE and not is_shaken:
        enemy_pos_set: set[tuple[int, int]] = set()
        for eu in enemy_units:
            if eu.models_alive > 0:
                for pos in eu.alive_positions():
                    enemy_pos_set.add(pos)

        candidates, cand_mask, adv_reachable = compute_destination_candidates(
            selected_unit, board, enemy_pos_set, player)
        _dest_dmg_arr, _static_dmg_arr = compute_unit_expected_damage_arrays(
            selected_unit, candidates, cand_mask, enemy_units, board, _e_dmg_table)
        budget = float(selected_unit.unit.rush_distance)
        dest_feats = compute_destination_features(
            candidates, cand_mask, selected_unit, selected_idx, player,
            enemy_units, enemy_alive_np,
            friendly_ranged_matchups, enemy_ranged_matchups, enemy_melee_matchups,
            budget, advance_reachable=adv_reachable, board=board,
            dest_expected_damage=_dest_dmg_arr)

        dest_features_t = torch.from_numpy(dest_feats).float().unsqueeze(0)
        dest_mask_t = torch.from_numpy(cand_mask).unsqueeze(0)

        dest_logits = model.compute_dest_logits(h_uf_m, dest_features_t, dest_mask_t).squeeze(0)
        dest_idx = int(torch.distributions.Categorical(logits=dest_logits).sample().item())
        dest_col_gs = int(candidates[dest_idx, 0])
        dest_row_gs = int(candidates[dest_idx, 1])
        _is_ar = bool(adv_reachable[dest_idx])
    else:
        if _e_dmg_table is not None:
            _, _static_dmg_arr = compute_unit_expected_damage_arrays(
                selected_unit, np.zeros((MAX_DEST_CANDIDATES, 2), dtype=np.int32),
                np.zeros(MAX_DEST_CANDIDATES, dtype=np.bool_),
                enemy_units, board, _e_dmg_table)

    # Compute post_move_rel from selected hex
    if move_type == MOVE_MOVE and not is_shaken:
        px, py = float(dest_col_gs), float(dest_row_gs)
        if player == "B":
            px = _flip_x(px)
            py = _flip_y(py)
    else:
        px, py = unit_cx_ms, unit_cy_ms
    post_move_rel = compute_post_move_rel(px, py, enemy_positions_ms).unsqueeze(0)

    # --- Charge target (sample) ---
    charge_logits = model.compute_charge_logits(
        h.squeeze(0), units.squeeze(0), selected_idx,
        enemy_alive_mask, can_charge_mask,
    )
    no_enemies = not enemy_alive_mask.any()
    no_chargeable = not (enemy_alive_mask & can_charge_mask).any()
    if no_enemies or no_chargeable:
        charge_target_idx = 0
    else:
        charge_probs = torch.softmax(charge_logits, dim=-1)
        charge_target_idx = int(torch.multinomial(charge_probs, 1).item())

    # --- Shoot target (sample) ---
    max_wr = max(
        (w.range_inches for w in selected_unit.unit.weapons if not w.melee),
        default=0.0,
    )
    shoot_range_mask = compute_in_range_mask(
        post_move_rel.squeeze(0), float(max_wr), enemy_alive_mask)
    if move_type == MOVE_MOVE and not is_shaken:
        shooter_gs = (float(dest_col_gs), float(dest_row_gs))
    else:
        shooter_gs = selected_unit.centre()
    vis_mask = compute_visibility_mask(shooter_gs, enemy_units, board)
    shoot_range_mask = shoot_range_mask & vis_mask
    if is_shaken:
        shoot_range_mask = torch.zeros_like(shoot_range_mask)
    # Cover-aware expected wound frac override.
    _ewf_override = None
    if _e_dmg_table is not None:
        if move_type == MOVE_MOVE and not is_shaken and _dest_dmg_arr is not None:
            _ewf_override = torch.from_numpy(_dest_dmg_arr[dest_idx].astype(np.float32))
        elif _static_dmg_arr is not None:
            _ewf_override = torch.from_numpy(_static_dmg_arr.astype(np.float32))
    shoot_logits = model.compute_shoot_logits(
        h.squeeze(0), units.squeeze(0), selected_idx,
        post_move_rel.squeeze(0), enemy_alive_mask,
        shoot_range_mask=shoot_range_mask,
        expected_wound_frac_override=_ewf_override,
    )
    no_shootable = not shoot_range_mask.any()
    if no_enemies or no_shootable:
        shoot_target_idx = 0
    else:
        shoot_probs = torch.softmax(shoot_logits, dim=-1)
        shoot_target_idx = int(torch.multinomial(shoot_probs, 1).item())

    target_ranking = torch.argsort(shoot_logits, descending=True).tolist()

    dest = None
    if move_type == MOVE_MOVE:
        dest = (float(dest_col_gs), float(dest_row_gs))

    action, goal, charge_target, reason = execute_decoded_decision(
        selected_unit, enemy_units, move_type, dest, charge_target_idx, shoot_target_idx,
        is_advance_reachable=_is_ar,
    )

    return selected_unit, target_ranking, action, goal, charge_target, reason, {}


# ---------------------------------------------------------------------------
# Target picking from ranking (used by combat resolution)
# ---------------------------------------------------------------------------

def pick_target_from_ranking(
    attacker: UnitState,
    enemies: list[UnitState],
    target_ranking: list[int],
    board: Board | None = None,
) -> UnitState | None:
    """Walk the ML target ranking and return the highest-ranked enemy in weapon range
    (and visible, when *board* carries terrain)."""
    for slot_idx in target_ranking:
        if slot_idx >= len(enemies):
            continue
        enemy = enemies[slot_idx]
        if enemy.models_alive <= 0:
            continue
        can_shoot, _, _ = evaluate_target(attacker, enemy, board)
        if can_shoot:
            return enemy
    return None
