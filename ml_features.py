"""ML feature extraction: encode game state as a fixed-size tensor for the tactical model."""
from __future__ import annotations

import math

import numpy as np
import torch

from board import COLS, ROWS, OBJECTIVES, Board
from models import ResolvedUnit, UnitState
import fast_core as _fc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_DIAG = math.sqrt(COLS * COLS + ROWS * ROWS)  # ~86.5
_INV_BOARD_DIAG = 1.0 / BOARD_DIAG
MAX_UNITS_PER_SIDE = 10

# Range thresholds for ranged damage features
RANGE_THRESHOLDS = [6, 9, 12, 18, 24, 30, 36]
NUM_RANGE_THRESHOLDS = len(RANGE_THRESHOLDS)

# 4 (round one-hot) + 5 (objective control) + 2 (points) = 11 global values
GLOBAL_FEATURES = 11

# Tactical model: egocentric (sin θ, cos θ, dist) for objectives and enemies
# Per-unit: 10 basic + 2 position + 15 obj-rel + 30 enemy-rel + 30 same-side-rel
#         + 70 ranged + 10 melee + 10 opp-post-advance-dist
#         + has_activated + fatigued + is_shaken = 180
_NUM_OBJECTIVES = 5
_TACTICAL_OBJ_REL = _NUM_OBJECTIVES * 3       # 15: (sin θ, cos θ, dist) per objective
_TACTICAL_OPP_REL = MAX_UNITS_PER_SIDE * 3    # 30: (sin θ, cos θ, dist) per opposing unit
_TACTICAL_SAME_REL = MAX_UNITS_PER_SIDE * 3   # 30: (sin θ, cos θ, dist) per same-side unit
_TACTICAL_BASE = 10 + 2 + _TACTICAL_OBJ_REL + _TACTICAL_OPP_REL + _TACTICAL_SAME_REL  # 87
_TACTICAL_OPP_POST_ADV = MAX_UNITS_PER_SIDE   # 10: post-advance distance per opposing unit
TACTICAL_UNIT_FEATURES = (_TACTICAL_BASE + NUM_RANGE_THRESHOLDS * MAX_UNITS_PER_SIDE
                          + MAX_UNITS_PER_SIDE + _TACTICAL_OPP_POST_ADV + 3)  # 180
# (87 + 70 ranged + 10 melee + 10 opp-post-advance + 3 tactical bools)
TACTICAL_TOTAL_FEATURES = MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES + GLOBAL_FEATURES  # 3611

# Tactical per-unit feature offsets
_TOFF_SCALARS = 0         # 10 scalars
_TOFF_POS = 10            # 2 absolute position (x, y)
_TOFF_OBJ_REL = 12        # 15: 5 objectives × (sin θ, cos θ, dist)
_TOFF_OPP_REL = 27        # 30: 10 opposing units × (sin θ, cos θ, dist)
_TOFF_SAME_REL = 57       # 30: 10 same-side units × (sin θ, cos θ, dist)
_TOFF_RANGED = 87         # 70: ranged matchup values (10 enemies × 7 thresholds)
_TOFF_MELEE = _TOFF_RANGED + MAX_UNITS_PER_SIDE * NUM_RANGE_THRESHOLDS  # 157: 10 melee values
_TOFF_OPP_POST_ADV = _TOFF_MELEE + MAX_UNITS_PER_SIDE  # 167: 10 post-advance distances
_TOFF_ACTIVATED = _TOFF_OPP_POST_ADV + MAX_UNITS_PER_SIDE  # 177: has_activated
_TOFF_FATIGUED = _TOFF_ACTIVATED + 1   # 178: fatigued
_TOFF_SHAKEN = _TOFF_FATIGUED + 1      # 179: is_shaken

# Normalisation ceilings
_MAX_TOUGH = 24
_MAX_MODELS = 10
_MAX_SPEED = 24.0

# Pre-allocated zero arrays for missing unit slots (never mutated)
_ZERO_RANGED_ROW = np.zeros((MAX_UNITS_PER_SIDE, NUM_RANGE_THRESHOLDS), dtype=np.float32)
_ZERO_MELEE_ROW = np.zeros(MAX_UNITS_PER_SIDE, dtype=np.float32)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def starting_wounds(unit: ResolvedUnit) -> int:
    """Total wounds a unit can absorb before being destroyed."""
    if unit.tough:
        return unit.tough * unit.models
    return unit.models


def _bane_block_prob(eff_def: float) -> float:
    """Block probability when attacker has Bane (re-roll defender nat6 saves)."""
    base_block = max((7 - eff_def) / 6.0, 0.0)
    # P(block|bane) = P(block and not nat6) + P(nat6) * P(block on reroll)
    return max(base_block - 1 / 6, 0.0) + (1 / 6) * base_block


def _expected_ranged_damage_at_range(
    attacker: ResolvedUnit, defender: ResolvedUnit, max_range: int,
) -> float:
    """Raw expected wounds from ranged weapons with range >= *max_range*.

    Only includes non-melee weapons whose range_inches >= max_range,
    simulating "what damage would this unit do if shooting from exactly
    max_range inches away?".

    The flat weapons list already has one entry per model carrying the weapon,
    so we must NOT multiply by attacker.models (that would double-count).

    Rules modelled: quality, reliable, stealth (unless unstoppable), AP, blast,
    deadly, defense, shielded, crack, rending, bane, unstoppable, regeneration,
    artillery (+1/-2 hit beyond 9"), relentless (extra nat6 hits beyond 9"),
    piercing spotter (+0.5 AP average).
    """
    beyond_9 = max_range > 9

    # --- Base hit quality (before per-weapon unstoppable check) ---
    base_quality = attacker.quality
    # Stealth: -1 to hit
    stealth_penalty = 1 if defender.stealth else 0
    # Artillery modifiers (only beyond 9")
    artillery_atk_bonus = 1 if (attacker.artillery and beyond_9) else 0
    artillery_def_penalty = 2 if (defender.artillery and beyond_9) else 0

    # Defender effective defense (shielded: +1)
    d_def = defender.defense + (1 if defender.shielded else 0)

    # Piercing spotter: average +0.5 AP
    spotter_ap = 0.5 if attacker.piercing_spotter else 0.0

    total = 0.0
    for w in attacker.weapons:
        if w.melee or w.range_inches < max_range:
            continue

        # Unstoppable ignores negative hit modifiers (stealth, artillery def)
        if w.unstoppable:
            quality = max(base_quality - artillery_atk_bonus, 2)
        else:
            quality = min(max(base_quality + stealth_penalty + artillery_def_penalty
                              - artillery_atk_bonus, 2), 6)
        hit_prob = (7 - quality) / 6.0

        p = 5 / 6 if w.reliable else hit_prob

        dice = w.attacks

        # Split hits into nat6 and normal (for crack/rending/relentless)
        nat6_hits = dice * (1 / 6)
        normal_hits = dice * max(p - 1 / 6, 0.0)

        # Relentless: each nat6 generates an extra normal hit (ranged, beyond 9")
        if attacker.relentless and beyond_9:
            normal_hits += nat6_hits

        # Blast multiplier
        if w.blast:
            blast_mult = min(w.blast, defender.models)
            nat6_hits *= blast_mult
            normal_hits *= blast_mult

        # --- Nat6 path: crack +2 AP, rending +4 AP ---
        nat6_ap = w.ap + spotter_ap
        if w.crack:
            nat6_ap += 2
        if w.rending:
            nat6_ap += 4

        eff_def_nat6 = min(d_def + nat6_ap, 7)
        block_nat6 = (_bane_block_prob(eff_def_nat6) if w.bane
                      else max((7 - eff_def_nat6) / 6.0, 1 / 6))
        wounds_nat6 = nat6_hits * (1 - block_nat6)

        # --- Normal path ---
        normal_ap = w.ap + spotter_ap
        eff_def_normal = min(d_def + normal_ap, 7)
        block_normal = (_bane_block_prob(eff_def_normal) if w.bane
                        else max((7 - eff_def_normal) / 6.0, 1 / 6))
        wounds_normal = normal_hits * (1 - block_normal)

        # Deadly multiplier
        if w.deadly:
            wounds_nat6 *= w.deadly
            wounds_normal *= w.deadly

        weapon_wounds = wounds_nat6 + wounds_normal

        # Regeneration: 5+ negates → 1/3 negated → multiply by 2/3
        # Bypassed by rending, unstoppable, bane
        if defender.regeneration and not (w.rending or w.unstoppable or w.bane):
            weapon_wounds *= (2 / 3)

        total += weapon_wounds
    return total


def _expected_melee_damage_raw(
    attacker: ResolvedUnit, defender: ResolvedUnit,
) -> float:
    """Raw expected wounds from melee weapons + impact, all models alive and in range.

    The flat weapons list already has one entry per model carrying the weapon,
    so we must NOT multiply by attacker.models (that would double-count).

    Assumes charge (first engagement): furious and thrust are active.

    Rules modelled: quality, reliable, AP, blast, deadly, defense, shielded,
    fortified (-1 AP), crack, rending, bane, unstoppable, regeneration,
    furious (extra nat6 hits on charge), thrust (+1 hit / +1 AP on charge),
    piercing spotter (+0.5 AP average), impact.
    """
    base_quality = attacker.quality
    base_hit_prob = (7 - base_quality) / 6.0

    # Defender effective defense (shielded: +1)
    d_def = defender.defense + (1 if defender.shielded else 0)

    # Piercing spotter: average +0.5 AP
    spotter_ap = 0.5 if attacker.piercing_spotter else 0.0

    total = 0.0
    for w in attacker.weapons:
        if not w.melee:
            continue

        # Thrust: +1 to hit on charge (quality -1, min 2)
        if w.thrust:
            thrust_quality = max(base_quality - 1, 2)
            p = 5 / 6 if w.reliable else (7 - thrust_quality) / 6.0
        else:
            p = 5 / 6 if w.reliable else base_hit_prob

        dice = w.attacks

        # Split hits into nat6 and normal
        nat6_hits = dice * (1 / 6)
        normal_hits = dice * max(p - 1 / 6, 0.0)

        # Furious: each nat6 generates an extra normal hit (assume charge)
        if attacker.furious:
            normal_hits += nat6_hits

        # Blast multiplier
        if w.blast:
            blast_mult = min(w.blast, defender.models)
            nat6_hits *= blast_mult
            normal_hits *= blast_mult

        # Base AP: weapon AP + thrust bonus, reduced by fortified
        base_ap = w.ap + (1 if w.thrust else 0)
        if defender.fortified:
            base_ap = max(base_ap - 1, 0)

        # --- Nat6 path: crack +2 AP, rending +4 AP ---
        nat6_ap = base_ap + spotter_ap
        if w.crack:
            nat6_ap += 2
        if w.rending:
            nat6_ap += 4

        eff_def_nat6 = min(d_def + nat6_ap, 7)
        block_nat6 = (_bane_block_prob(eff_def_nat6) if w.bane
                      else max((7 - eff_def_nat6) / 6.0, 1 / 6))
        wounds_nat6 = nat6_hits * (1 - block_nat6)

        # --- Normal path ---
        normal_ap = base_ap + spotter_ap
        eff_def_normal = min(d_def + normal_ap, 7)
        block_normal = (_bane_block_prob(eff_def_normal) if w.bane
                        else max((7 - eff_def_normal) / 6.0, 1 / 6))
        wounds_normal = normal_hits * (1 - block_normal)

        # Deadly multiplier
        if w.deadly:
            wounds_nat6 *= w.deadly
            wounds_normal *= w.deadly

        weapon_wounds = wounds_nat6 + wounds_normal

        # Regeneration: 5+ negates → multiply by 2/3
        # Bypassed by rending, unstoppable, bane
        if defender.regeneration and not (w.rending or w.unstoppable or w.bane):
            weapon_wounds *= (2 / 3)

        total += weapon_wounds

    # Impact contribution: impact × models × 5/6 (hits on 2+)
    # Impact has no AP, no special weapon rules
    if attacker.impact:
        impact_hits = attacker.impact * attacker.models * (5 / 6)
        eff_def_impact = d_def  # includes shielded
        block_prob = max((7 - eff_def_impact) / 6.0, 1 / 6)
        impact_wounds = impact_hits * (1 - block_prob)
        if defender.regeneration:
            impact_wounds *= (2 / 3)
        total += impact_wounds
    return total


def precompute_damage(
    friendly_resolved: list[ResolvedUnit],
    enemy_resolved: list[ResolvedUnit],
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-matchup kill proportions for each friendly vs each enemy.

    Returns (ranged_matchups, melee_matchups) as numpy arrays.

    ranged_matchups: shape (num_friendly, MAX_UNITS_PER_SIDE, NUM_RANGE_THRESHOLDS).
      Entry [i][j][k] is the expected fraction of enemy unit j that friendly
      unit i would kill when shooting from RANGE_THRESHOLDS[k] inches away
      (i.e. only weapons with range >= that threshold contribute).

    melee_matchups: shape (num_friendly, MAX_UNITS_PER_SIDE).
      Entry [i][j] is the expected fraction of enemy unit j killed in melee.

    All values are capped at 1.0.  Slots beyond len(enemy) are 0.0.

    Call once per side at game start; scale by models_alive / starting_models
    at encoding time.
    """
    n_friendly = len(friendly_resolved)
    n_enemy = min(len(enemy_resolved), MAX_UNITS_PER_SIDE)
    ranged_matchups = np.zeros((n_friendly, MAX_UNITS_PER_SIDE, NUM_RANGE_THRESHOLDS), dtype=np.float32)
    melee_matchups = np.zeros((n_friendly, MAX_UNITS_PER_SIDE), dtype=np.float32)
    for i, u in enumerate(friendly_resolved):
        for j in range(n_enemy):
            e = enemy_resolved[j]
            sw = max(starting_wounds(e), 1)
            inv_sw = 1.0 / sw
            for k, threshold in enumerate(RANGE_THRESHOLDS):
                ranged_matchups[i, j, k] = min(_expected_ranged_damage_at_range(u, e, threshold) * inv_sw, 1.0)
            melee_matchups[i, j] = min(_expected_melee_damage_raw(u, e) * inv_sw, 1.0)
    return ranged_matchups, melee_matchups


# ---------------------------------------------------------------------------
# Board-flip helpers
# ---------------------------------------------------------------------------

def _flip_y(y: float) -> float:
    """Flip a y-coordinate for Player B perspective."""
    return (ROWS - 1) - y


def _flip_x(x: float) -> float:
    """Flip an x-coordinate for Player B perspective (180° rotation)."""
    return (COLS - 1) - x


# Model-space objective positions.  Physical objectives are now 180°-
# rotationally symmetric, so paired objectives (A-side↔B-side, Home-A↔
# Home-B) map to each other under the flip.  Centre uses half-integer
# coords (35.5, 23.5) for self-symmetry on the even grid.
_MODEL_OBJECTIVES: list[tuple[float, float]] = [
    (35.5, 23.5),   # Centre — half-integer for self-symmetry under flip
    (18.0, 16.0),   # A-side   (physical 18, 16; flips to 53, 31 = B-side)
    (53.0, 31.0),   # B-side   (physical 53, 31; flips to 18, 16 = A-side)
    (36.0,  6.0),   # Home-A   (physical 36,  6; flips to 35, 41 = Home-B)
    (35.0, 41.0),   # Home-B   (physical 35, 41; flips to 36,  6 = Home-A)
]


def _get_model_objectives(player: str) -> list[tuple[float, float]]:
    """5 objective positions in the model's coordinate frame.

    Index 0=Centre, 1=My-side, 2=Enemy-side, 3=My-home, 4=Enemy-home.
    Uses half-integer positions so the 180° flip is perfectly symmetric.
    """
    if player == "A":
        return list(_MODEL_OBJECTIVES)
    # Player B: flip x and y (180° rotation) then remap 1↔2, 3↔4
    flipped = [(_flip_x(x), _flip_y(y)) for x, y in _MODEL_OBJECTIVES]
    return [flipped[0], flipped[2], flipped[1], flipped[4], flipped[3]]


def _objective_control_mapped(board: Board, player: str) -> list[float]:
    """5 objective-control values from model's perspective.

    +1.0 = friendly controls, −1.0 = enemy controls, 0.0 = neutral.
    """
    ctrl = board.objective_control
    friendly = player
    enemy = "B" if player == "A" else "A"

    def _val(idx: int) -> float:
        if ctrl[idx] == friendly:
            return 1.0
        if ctrl[idx] == enemy:
            return -1.0
        return 0.0

    if player == "A":
        order = [0, 1, 2, 3, 4]
    else:
        order = [0, 2, 1, 4, 3]
    return [_val(i) for i in order]


# ---------------------------------------------------------------------------
# Per-unit encoding (writes into pre-allocated numpy buffer)
# ---------------------------------------------------------------------------

def _survival_fraction(us: UnitState) -> float:
    unit = us.unit
    if unit.tough:
        total_start = unit.tough * unit.models
        total_rem = sum(
            unit.tough - w for w in us.wounds_per_model[: us.models_alive]
        )
        return total_rem / max(total_start, 1)
    return us.models_alive / max(unit.models, 1)


def _encode_unit_tactical_into(
    us: UnitState,
    is_friendly: bool,
    player: str,
    objectives: list[tuple[float, float]],
    ranged_matchups: np.ndarray,
    melee_matchups: np.ndarray,
    total_side_points: int,
    opposing_positions: list[tuple[float, float]],
    opposing_advance_distances: list[float],
    same_side_positions: list[tuple[float, float]],
    buf: np.ndarray,
    offset: int,
) -> None:
    """Write TACTICAL_UNIT_FEATURES (180) floats into buf for the v2 tactical model.

    Uses egocentric (sin θ, cos θ, dist) encoding for objectives, opposing units,
    and same-side units.

    Features layout:
      0-9     10 scalars (wounds, models, speed, survival, points_frac, fly, arty, fearless, fear, is_friendly)
      10-11   Absolute position (x, y) normalised
      12-26   5 objectives × (sin θ, cos θ, dist) = 15
      27-56   10 opposing units × (sin θ, cos θ, dist) = 30
      57-86   10 same-side units × (sin θ, cos θ, dist) = 30
      87-156  70 ranged matchup values (10 enemies × 7 thresholds)
      157-166 10 melee matchup values
      167-176 10 post-advance distances (how close each opposing unit could get)
      177     has_activated
      178     fatigued
      179     is_shaken
    """
    if us.models_alive <= 0:
        return  # buf is zero-initialized

    unit = us.unit

    # Position (flip x and y for Player B — 180° rotation)
    cx, cy = us.centre()
    if player == "B":
        cx = _flip_x(cx)
        cy = _flip_y(cy)

    # 0-9  Scalar features
    wound_count = unit.tough if unit.tough > 0 else 1
    buf[offset:offset + 10] = [
        wound_count / _MAX_TOUGH,
        unit.models / _MAX_MODELS,
        (0.0 if unit.artillery else float(unit.rush_distance)) / _MAX_SPEED,
        _survival_fraction(us),
        unit.points / max(total_side_points, 1),
        1.0 if unit.flying else 0.0,
        1.0 if unit.artillery else 0.0,
        1.0 if unit.fearless else 0.0,
        1.0 if unit.fear > 0 else 0.0,
        1.0 if is_friendly else 0.0,
    ]

    # 10-11  Absolute position
    buf[offset + _TOFF_POS] = (cx + 0.5) / float(COLS)
    buf[offset + _TOFF_POS + 1] = (cy + 0.5) / float(ROWS)

    # 12-26  Objectives: (sin θ, cos θ, dist) per objective
    o = offset + _TOFF_OBJ_REL
    for ox, oy in objectives:
        dx = ox - cx
        dy = oy - cy
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            buf[o] = 0.0       # sin θ
            buf[o + 1] = 0.0   # cos θ
        else:
            buf[o] = dy / d     # sin θ
            buf[o + 1] = dx / d # cos θ
        buf[o + 2] = d * _INV_BOARD_DIAG  # normalised distance
        o += 3

    # 27-56  Opposing units: (sin θ, cos θ, dist) per opposing unit
    # 167-176  Post-advance distances (computed in same loop)
    o = offset + _TOFF_OPP_REL
    pa = offset + _TOFF_OPP_POST_ADV
    for j, (ox, oy) in enumerate(opposing_positions):
        dx = ox - cx
        dy = oy - cy
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            buf[o] = 0.0
            buf[o + 1] = 0.0
        else:
            buf[o] = dy / d
            buf[o + 1] = dx / d
        buf[o + 2] = d * _INV_BOARD_DIAG
        o += 3
        buf[pa + j] = max(0.0, d - opposing_advance_distances[j]) * _INV_BOARD_DIAG

    # 57-86  Same-side units: (sin θ, cos θ, dist) per same-side unit
    o = offset + _TOFF_SAME_REL
    for sx, sy in same_side_positions:
        dx = sx - cx
        dy = sy - cy
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            buf[o] = 0.0
            buf[o + 1] = 0.0
        else:
            buf[o] = dy / d
            buf[o + 1] = dx / d
        buf[o + 2] = d * _INV_BOARD_DIAG
        o += 3

    # 87-156  Ranged matchup kill proportions
    scale = us.models_alive / max(unit.models, 1)
    ranged_start = offset + _TOFF_RANGED
    ranged_end = ranged_start + MAX_UNITS_PER_SIDE * NUM_RANGE_THRESHOLDS
    buf[ranged_start:ranged_end] = ranged_matchups.ravel() * scale

    # 157-166  Melee matchup kill proportions
    melee_start = offset + _TOFF_MELEE
    melee_end = melee_start + MAX_UNITS_PER_SIDE
    buf[melee_start:melee_end] = melee_matchups * scale

    # 177-179: tactical booleans written by encode_state_tactical caller


# ---------------------------------------------------------------------------
# Opposing-position helpers
# ---------------------------------------------------------------------------

# Sentinel for dead/missing opposing units: the true grid centre is
# self-symmetric under the 180° flip, so distances to it are identical
# for both sides.  (0, 0) leaked side information because it mapped to
# different physical corners depending on the player.
_DEAD_SENTINEL = ((COLS - 1) / 2.0, (ROWS - 1) / 2.0)   # (35.5, 23.5)


def _get_opposing_positions(
    opposing_units: list[UnitState], player: str,
) -> list[tuple[float, float]]:
    """Return centre positions for up to MAX_UNITS_PER_SIDE opposing units.

    Positions are in the model's coordinate frame (180°-rotated for Player B).
    Dead or missing slots get the grid-centre sentinel (side-symmetric).
    """
    positions: list[tuple[float, float]] = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(opposing_units) and opposing_units[i].models_alive > 0:
            cx, cy = opposing_units[i].centre()
            if player == "B":
                cx = _flip_x(cx)
                cy = _flip_y(cy)
            positions.append((cx, cy))
        else:
            positions.append(_DEAD_SENTINEL)
    return positions


# ---------------------------------------------------------------------------
# Tactical state encoding (per-activation model)
# ---------------------------------------------------------------------------

def encode_state_tactical(
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    *,
    friendly_ranged_matchups: np.ndarray | None = None,
    friendly_melee_matchups: np.ndarray | None = None,
    enemy_ranged_matchups: np.ndarray | None = None,
    enemy_melee_matchups: np.ndarray | None = None,
    total_friendly_points: int | None = None,
    total_enemy_points: int | None = None,
) -> torch.Tensor:
    """Encode the full game state into a fixed-size tensor for the tactical v2 model.

    Uses egocentric (sin θ, cos θ, dist) per-unit encoding for objectives and
    opposing units, plus has_activated, fatigued, and is_shaken flags.

    Parameters
    ----------
    friendly_units : units controlled by *player*
    enemy_units : opponent's units
    round_num : 1–4
    board : current Board state
    player : "A" or "B"
    friendly_ranged_matchups / friendly_melee_matchups : precomputed via precompute_damage()
    enemy_ranged_matchups / enemy_melee_matchups : precomputed for enemy side
    total_friendly_points / total_enemy_points : starting army-point totals
    """
    # Derive totals if not provided
    if total_friendly_points is None:
        total_friendly_points = sum(u.unit.points for u in friendly_units)
    if total_enemy_points is None:
        total_enemy_points = sum(u.unit.points for u in enemy_units)

    # Derive matchup matrices if not provided (slower path)
    if friendly_ranged_matchups is None or friendly_melee_matchups is None:
        fr, fm = precompute_damage(
            [u.unit for u in friendly_units],
            [u.unit for u in enemy_units],
        )
        friendly_ranged_matchups = friendly_ranged_matchups if friendly_ranged_matchups is not None else fr
        friendly_melee_matchups = friendly_melee_matchups if friendly_melee_matchups is not None else fm
    if enemy_ranged_matchups is None or enemy_melee_matchups is None:
        er, em = precompute_damage(
            [u.unit for u in enemy_units],
            [u.unit for u in friendly_units],
        )
        enemy_ranged_matchups = enemy_ranged_matchups if enemy_ranged_matchups is not None else er
        enemy_melee_matchups = enemy_melee_matchups if enemy_melee_matchups is not None else em

    objectives = _get_model_objectives(player)
    enemy_positions = _get_opposing_positions(enemy_units, player)
    friendly_positions = _get_opposing_positions(friendly_units, player)

    # Advance distances for post-advance feature (0.0 for dead/missing slots)
    enemy_advance_dists = [
        float(enemy_units[i].unit.advance_distance)
        if i < len(enemy_units) and enemy_units[i].models_alive > 0
        else 0.0
        for i in range(MAX_UNITS_PER_SIDE)
    ]
    friendly_advance_dists = [
        float(friendly_units[i].unit.advance_distance)
        if i < len(friendly_units) and friendly_units[i].models_alive > 0
        else 0.0
        for i in range(MAX_UNITS_PER_SIDE)
    ]

    buf = np.zeros(TACTICAL_TOTAL_FEATURES, dtype=np.float32)

    # --- Friendly slots (0–9): TACTICAL_UNIT_FEATURES each ---
    for i in range(MAX_UNITS_PER_SIDE):
        offset = i * TACTICAL_UNIT_FEATURES
        if i < len(friendly_units):
            us = friendly_units[i]
            rm = friendly_ranged_matchups[i] if i < len(friendly_ranged_matchups) else _ZERO_RANGED_ROW
            mm = friendly_melee_matchups[i] if i < len(friendly_melee_matchups) else _ZERO_MELEE_ROW
            _encode_unit_tactical_into(us, True, player, objectives, rm, mm,
                                       total_friendly_points, enemy_positions,
                                       enemy_advance_dists,
                                       friendly_positions, buf, offset)
            if us.models_alive > 0:
                if us.activated:
                    buf[offset + _TOFF_ACTIVATED] = 1.0
                if us.fatigued:
                    buf[offset + _TOFF_FATIGUED] = 1.0
                if us.shaken:
                    buf[offset + _TOFF_SHAKEN] = 1.0

    # --- Enemy slots (10–19): TACTICAL_UNIT_FEATURES each ---
    enemy_base = MAX_UNITS_PER_SIDE * TACTICAL_UNIT_FEATURES
    for i in range(MAX_UNITS_PER_SIDE):
        offset = enemy_base + i * TACTICAL_UNIT_FEATURES
        if i < len(enemy_units):
            us = enemy_units[i]
            rm = enemy_ranged_matchups[i] if i < len(enemy_ranged_matchups) else _ZERO_RANGED_ROW
            mm = enemy_melee_matchups[i] if i < len(enemy_melee_matchups) else _ZERO_MELEE_ROW
            _encode_unit_tactical_into(us, False, player, objectives, rm, mm,
                                       total_enemy_points, friendly_positions,
                                       friendly_advance_dists,
                                       enemy_positions, buf, offset)
            if us.models_alive > 0:
                if us.activated:
                    buf[offset + _TOFF_ACTIVATED] = 1.0
                if us.fatigued:
                    buf[offset + _TOFF_FATIGUED] = 1.0
                if us.shaken:
                    buf[offset + _TOFF_SHAKEN] = 1.0

    # --- Global features (11) ---
    g = MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES
    buf[g + round_num - 1] = 1.0
    buf[g + 4:g + 9] = _objective_control_mapped(board, player)
    alive_f = sum(u.unit.points for u in friendly_units if u.models_alive > 0)
    buf[g + 9] = alive_f / max(total_friendly_points, 1)
    alive_e = sum(u.unit.points for u in enemy_units if u.models_alive > 0)
    buf[g + 10] = alive_e / max(total_enemy_points, 1)

    assert len(buf) == TACTICAL_TOTAL_FEATURES
    return torch.from_numpy(buf)
