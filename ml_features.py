"""ML feature extraction: encode game state as a fixed-size tensor for the tactical model."""
from __future__ import annotations

import math

import numpy as np
import torch

from board import COLS, ROWS, OBJECTIVES, OBJ_SEIZE_RANGE, Board
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

# Tactical global features layout:
#   [0..3]   round one-hot               (4)
#   [4..8]   objective control           (5)
#   [9..13]  projected obj control       (5)
#   [14]     alive friendly fraction     (1)
#   [15]     alive enemy fraction        (1)
#   [16..18] deploy_phase one-hot {non_scout, scout, in_game}  (3) — NEW
#   [19]     is_my_turn_to_deploy         (1) — NEW
#   [20]     n_unplaced_self / MAX_UNITS_PER_SIDE   (1) — NEW
#   [21]     n_unplaced_opp  / MAX_UNITS_PER_SIDE   (1) — NEW
GLOBAL_FEATURES = 22
_GOFF_ROUND = 0
_GOFF_OBJ_CTRL = 4
_GOFF_OBJ_CTRL_PROJ = 9
_GOFF_ALIVE_F = 14
_GOFF_ALIVE_E = 15
_GOFF_DEPLOY_PHASE = 16   # 3 dims: 0=non_scout, 1=scout, 2=in_game
_GOFF_MY_DEPLOY_TURN = 19
_GOFF_N_UNPLACED_SELF = 20
_GOFF_N_UNPLACED_OPP = 21

# Tactical model: egocentric (sin θ, cos θ, dist) for objectives and enemies
# Per-unit: 10 basic + 2 position + 15 obj-rel + 30 enemy-rel + 30 same-side-rel
#         + 70 ranged + 10 melee + 10 opp-post-advance-dist
#         + 10 obj-reachability (5 obj × (can_advance, can_rush))
#         + 10 can_charge (1 per opposing unit)
#         + has_activated + fatigued + is_shaken + is_deployed = 201
_NUM_OBJECTIVES = 5
_TACTICAL_OBJ_REL = _NUM_OBJECTIVES * 3       # 15: (sin θ, cos θ, dist) per objective
_TACTICAL_OPP_REL = MAX_UNITS_PER_SIDE * 3    # 30: (sin θ, cos θ, dist) per opposing unit
_TACTICAL_SAME_REL = MAX_UNITS_PER_SIDE * 3   # 30: (sin θ, cos θ, dist) per same-side unit
_TACTICAL_BASE = 10 + 2 + _TACTICAL_OBJ_REL + _TACTICAL_OPP_REL + _TACTICAL_SAME_REL  # 87
_TACTICAL_OPP_POST_ADV = MAX_UNITS_PER_SIDE   # 10: post-advance distance per opposing unit
_TACTICAL_OBJ_REACH = _NUM_OBJECTIVES * 2     # 10: 5 objectives × (can_advance, can_rush)
_TACTICAL_CAN_CHARGE = MAX_UNITS_PER_SIDE  # 10: can_charge per opposing unit
TACTICAL_UNIT_FEATURES = (_TACTICAL_BASE + NUM_RANGE_THRESHOLDS * MAX_UNITS_PER_SIDE
                          + MAX_UNITS_PER_SIDE + _TACTICAL_OPP_POST_ADV
                          + _TACTICAL_OBJ_REACH + _TACTICAL_CAN_CHARGE + 4)  # 201
# (87 + 70 ranged + 10 melee + 10 opp-post-advance + 10 obj-reach + 10 can_charge
#  + 3 tactical bools + 1 is_deployed)
TACTICAL_TOTAL_FEATURES = MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES + GLOBAL_FEATURES  # 4042

# Tactical per-unit feature offsets
_TOFF_SCALARS = 0         # 10 scalars
_TOFF_POS = 10            # 2 absolute position (x, y)
_TOFF_OBJ_REL = 12        # 15: 5 objectives × (sin θ, cos θ, dist)
_TOFF_OPP_REL = 27        # 30: 10 opposing units × (sin θ, cos θ, dist)
_TOFF_SAME_REL = 57       # 30: 10 same-side units × (sin θ, cos θ, dist)
_TOFF_RANGED = 87         # 70: ranged matchup values (10 enemies × 7 thresholds)
_TOFF_MELEE = _TOFF_RANGED + MAX_UNITS_PER_SIDE * NUM_RANGE_THRESHOLDS  # 157: 10 melee values
_TOFF_OPP_POST_ADV = _TOFF_MELEE + MAX_UNITS_PER_SIDE  # 167: 10 post-advance distances
_TOFF_OBJ_REACH = _TOFF_OPP_POST_ADV + MAX_UNITS_PER_SIDE  # 177: 10 obj reachability
_TOFF_CAN_CHARGE = _TOFF_OBJ_REACH + _TACTICAL_OBJ_REACH  # 187: 10 can_charge flags
_TOFF_ACTIVATED = _TOFF_CAN_CHARGE + _TACTICAL_CAN_CHARGE  # 197: has_activated
_TOFF_FATIGUED = _TOFF_ACTIVATED + 1   # 198: fatigued
_TOFF_SHAKEN = _TOFF_FATIGUED + 1      # 199: is_shaken
_TOFF_IS_DEPLOYED = _TOFF_SHAKEN + 1   # 200: 1 if unit is on the board, else 0

# Deployment-action geometry: per-side egocentric grid of legal anchor cells.
# Non-scout phase mask: rows 0..11 (own DZ depth = 12).
# Scout phase mask:     rows 0..23 (own DZ + 12" forward).
# A single head of width DEPLOY_POS_GRID covers both, with phase-dependent mask.
DEPLOY_POS_DEPTH = 24       # max depth (egocentric) — covers scout zone
DEPLOY_POS_GRID = DEPLOY_POS_DEPTH * COLS  # 1728 cells
DEPLOY_POS_NONSCOUT_DEPTH = 12  # depth for non-scout phase mask

# Normalisation ceilings
_MAX_TOUGH = 24
_MAX_MODELS = 10
_MAX_SPEED = 24.0

# Destination pointer constants
# 75 base + 1 advance-reachable flag + 6 terrain (3 cover one-hot + 3 movement one-hot).
# Terrain block (TERRAIN_SPEC.md §5.4): per candidate hex, one-hot of cover_type
# {sheltering, obscuring, blocking} at [76:79] and movement_type
# {open, difficult, impassible} at [79:82]. The pre-existing reachable flag at
# [75] is already terrain-aware via the Dijkstra candidate computation.
DEST_FEATURE_DIM = 82
DEST_EMBED_DIM = 64         # embedding dimension for pointer cross-attention
MAX_DEST_CANDIDATES = 4096  # upper bound for candidate arrays; actual padding uses min(this, batch_max)

# Per-square terrain channel encoding — TERRAIN_SPEC.md §5.4 recommended
# layout. K = 6 board-shaped one-hot planes:
#   0: open (no terrain)
#   1: difficult-only (movement DIFFICULT, cover SHELTERING)  ← arbitrary; see encode_terrain_planes
#   2: impassible-only (movement IMPASSIBLE, cover SHELTERING — n/a today, future-proof)
#   3: sheltering (movement OPEN, cover SHELTERING)
#   4: obscuring (movement OPEN/DIFFICULT, cover OBSCURING)
#   5: blocking (movement IMPASSIBLE, cover BLOCKING)
TERRAIN_CHANNELS = 6

# Pre-allocated zero arrays for missing unit slots (never mutated)
_ZERO_RANGED_ROW = np.zeros((MAX_UNITS_PER_SIDE, NUM_RANGE_THRESHOLDS), dtype=np.float32)
_ZERO_MELEE_ROW = np.zeros(MAX_UNITS_PER_SIDE, dtype=np.float32)



# ---------------------------------------------------------------------------
# Terrain channels (TERRAIN_SPEC.md §5.4 per-square layout)
# ---------------------------------------------------------------------------

def encode_terrain_planes(board: Board) -> np.ndarray:
    """Return (TERRAIN_CHANNELS, ROWS, COLS) one-hot terrain planes.

    Each cell is hot in exactly one channel; the channel is chosen by
    cover_type (BLOCKING/OBSCURING/SHELTERING) when present, falling back to
    movement_type (IMPASSIBLE/DIFFICULT/OPEN). Cells with no terrain piece
    are hot in channel 0 (open).

    Intended to be consumed by a small spatial encoder (CNN) and concatenated
    into the global feature path of the tactical model. The model wiring is a
    follow-up — this function emits the correctly-shaped tensor today so
    downstream code can begin to depend on it.
    """
    from board import CoverType, MovementType
    planes = np.zeros((TERRAIN_CHANNELS, ROWS, COLS), dtype=np.float32)
    if not board.terrain:
        planes[0, :, :] = 1.0
        return planes
    planes[0, :, :] = 1.0  # default open
    for piece in board.terrain:
        ct = piece.cover_type
        mt = piece.movement_type
        if ct == CoverType.BLOCKING:
            ch = 5
        elif ct == CoverType.OBSCURING:
            ch = 4
        elif ct == CoverType.SHELTERING:
            ch = 3
        elif mt == MovementType.IMPASSIBLE:
            ch = 2
        elif mt == MovementType.DIFFICULT:
            ch = 1
        else:
            ch = 0
        for c, r in piece.squares():
            planes[0, r, c] = 0.0
            planes[ch, r, c] = 1.0
    return planes


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
    piercing spotter (+0.5 AP average). BB additions: Lacerate (block^2),
    Shred (+1/6 wound per def die), Smash (Blast(+3) vs def 5+ majority,
    ignores Regen), Indirect (ignores stealth/cover), Versatile Reach (+4"
    range), Versatile Attack EV pick at >9". ED additions: Clan Warrior
    (extra attack on nat 6, 5+ with Boost), Surge (+1 hit on nat 6 per
    weapon), Tear (AP+4 vs Tough(9)+), Puncture (AP+4 vs Tough(3-9), ignores
    Regen), Piercing Hunter (+1 AP at >9"), Ignores Cover (strips stealth).
    """
    beyond_9 = max_range > 9

    # Versatile Reach grants +4" range to ranged weapons; reflect by lowering
    # the effective max-range gate when computing per-weapon contribution.
    vr_bonus = 4 if attacker.versatile_reach else 0

    # --- Base hit quality (before per-weapon unstoppable check) ---
    base_quality = attacker.quality
    # Stealth: -1 to hit. Ignores Cover (ED) strips this for the whole unit.
    stealth_penalty = 0 if attacker.ignores_cover else (1 if defender.stealth else 0)
    # ED Piercing Hunter: +1 AP at >9".
    ph_ap = 1 if (attacker.piercing_hunter and beyond_9) else 0
    # ED Tear: +4 AP vs Tough(9)+. Puncture: +4 AP vs Tough(3-9), ignores Regen.
    tear_active = defender.tough >= 9
    puncture_active = 3 <= defender.tough <= 9
    # ED Clan Warrior multiplier: each nat 6 (or 5+ w/ boost) generates an
    # extra attack die. Approximate as a multiplier on attack count.
    if attacker.clan_warrior:
        cw_thresh = 5 if attacker.clan_warrior_boost else 6
        cw_trigger_p = (7 - cw_thresh) / 6.0   # 1/6 or 2/6
        cw_attack_mult = 1.0 + cw_trigger_p     # one extra die per triggering die
    else:
        cw_attack_mult = 1.0
    # Artillery modifiers (only beyond 9")
    artillery_atk_bonus = 1 if (attacker.artillery and beyond_9) else 0
    artillery_def_penalty = 2 if (defender.artillery and beyond_9) else 0

    # Defender effective defense (shielded: +1)
    d_def = defender.defense + (1 if defender.shielded else 0)

    # Smash: Blast(+3) when more than half of defender models have Defense 5+.
    smash_active = defender.defense >= 5  # 1 model unit or all models share Def

    # Piercing spotter: average +0.5 AP
    spotter_ap = 0.5 if attacker.piercing_spotter else 0.0

    # Versatile Attack: pre-compute the bonus for >9" engagements. We pick the
    # higher of AP(+1) or +1-to-hit by direct comparison below per-weapon.
    va_active = attacker.versatile_attack and beyond_9

    total = 0.0
    for w in attacker.weapons:
        if w.melee:
            continue
        if w.range_inches + vr_bonus < max_range:
            continue

        # Unstoppable ignores negative hit modifiers (stealth, artillery def).
        # Indirect ignores cover (we use the stealth penalty as the closest
        # proxy for cover in this sim).
        eff_stealth_penalty = stealth_penalty
        if w.indirect:
            eff_stealth_penalty = 0
        if w.unstoppable:
            quality = max(base_quality - artillery_atk_bonus, 2)
        else:
            quality = min(max(base_quality + eff_stealth_penalty + artillery_def_penalty
                              - artillery_atk_bonus, 2), 6)
        hit_prob = (7 - quality) / 6.0

        # Versatile Attack EV pick — try both options and take the higher.
        if va_active:
            # Option B: +1-to-hit (lower threshold, max 6)
            quality_hit = min(max(quality - 1, 2), 6)
            hit_prob_hit = (7 - quality_hit) / 6.0
            # Per-weapon EV is computed inline below; we just choose the
            # quality / ap pair that maximises expected wounds.
            # Approximate: pick by simple expected-wound proxy.
            ap_a = w.ap + 1
            ap_b = w.ap
            blk_a = max((7 - min(d_def + ap_a, 7)) / 6.0, 1 / 6)
            blk_b = max((7 - min(d_def + ap_b, 7)) / 6.0, 1 / 6)
            ev_a = hit_prob * (1.0 - blk_a)
            ev_b = hit_prob_hit * (1.0 - blk_b)
            if ev_b > ev_a:
                hit_prob = hit_prob_hit
                va_ap = 0
            else:
                va_ap = 1
        else:
            va_ap = 0

        p = 5 / 6 if w.reliable else hit_prob

        # Clan Warrior expands the effective attack count.
        dice = w.attacks * cw_attack_mult

        # Split hits into nat6 and normal (for crack/rending/relentless)
        nat6_hits = dice * (1 / 6)
        normal_hits = dice * max(p - 1 / 6, 0.0)

        # Relentless: each nat6 generates an extra normal hit (ranged, beyond 9")
        if attacker.relentless and beyond_9:
            normal_hits += nat6_hits

        # ED Surge: each nat 6 generates an extra normal hit (per-weapon).
        if w.surge:
            normal_hits += nat6_hits

        # Blast multiplier (Smash adds +3 to Blast vs def-5+ majority targets)
        eff_blast = w.blast
        smash_now = w.smash and smash_active
        if smash_now:
            eff_blast = (eff_blast or 0) + 3
        if eff_blast:
            blast_mult = min(eff_blast, defender.models)
            nat6_hits *= blast_mult
            normal_hits *= blast_mult

        # ED Tear / Puncture conditional AP.
        ed_extra_ap = ph_ap
        if w.tear and tear_active:
            ed_extra_ap += 4
        if w.puncture and puncture_active:
            ed_extra_ap += 4

        # --- Nat6 path: crack +2 AP, rending +4 AP ---
        nat6_ap = w.ap + spotter_ap + va_ap + ed_extra_ap
        if w.crack:
            nat6_ap += 2
        if w.rending:
            nat6_ap += 4

        eff_def_nat6 = min(d_def + nat6_ap, 7)
        block_nat6 = (_bane_block_prob(eff_def_nat6) if w.bane
                      else max((7 - eff_def_nat6) / 6.0, 1 / 6))
        # Lacerate: defender re-rolls successful blocks → block prob squared.
        if w.lacerate:
            block_nat6 = block_nat6 * block_nat6
        wounds_nat6 = nat6_hits * (1 - block_nat6)

        # --- Normal path ---
        normal_ap = w.ap + spotter_ap + va_ap + ed_extra_ap
        eff_def_normal = min(d_def + normal_ap, 7)
        block_normal = (_bane_block_prob(eff_def_normal) if w.bane
                        else max((7 - eff_def_normal) / 6.0, 1 / 6))
        if w.lacerate:
            block_normal = block_normal * block_normal
        wounds_normal = normal_hits * (1 - block_normal)

        # Shred: 1/6 of defense rolls are nat 1 → +1 raw wound each.
        # Approximate: total def dice ≈ (nat6_hits + normal_hits) (already
        # blast-multiplied); add (1/6) per die.
        if w.shred:
            wounds_normal += (nat6_hits + normal_hits) * (1.0 / 6.0)

        # Deadly multiplier (cap at model HP — overkill is wasted)
        if w.deadly:
            model_hp = defender.tough if defender.tough > 0 else 1
            effective_deadly = min(w.deadly, model_hp)
            wounds_nat6 *= effective_deadly
            wounds_normal *= effective_deadly

        weapon_wounds = wounds_nat6 + wounds_normal

        # Regeneration: 5+ negates → 1/3 negated → multiply by 2/3.
        # Bypassed by rending, unstoppable, bane, Smash, and Puncture.
        if (defender.regeneration
                and not (w.rending or w.unstoppable or w.bane or smash_now
                         or (w.puncture and puncture_active))):
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
    # ED: Melee Evasion on defender → -1 to hit; Precision Fighter on attacker → +1.
    # Vengeance markers are runtime state and aren't visible from ResolvedUnit
    # — we ignore them here; the live combat path applies them.
    base_quality_eff = base_quality
    if defender.melee_evasion:
        base_quality_eff += 1
    if attacker.precision_fighter:
        base_quality_eff -= 1
    base_quality_eff = max(2, min(6, base_quality_eff))
    base_hit_prob = (7 - base_quality_eff) / 6.0

    # Defender effective defense (shielded: +1)
    d_def = defender.defense + (1 if defender.shielded else 0)
    smash_active = defender.defense >= 5
    tear_active = defender.tough >= 9
    puncture_active = 3 <= defender.tough <= 9

    # Piercing spotter: average +0.5 AP
    spotter_ap = 0.5 if attacker.piercing_spotter else 0.0

    # Versatile Attack on charge: pick AP(+1) vs +1-to-hit by EV per weapon.
    va_active = attacker.versatile_attack

    # ED Clan Warrior melee multiplier.
    if attacker.clan_warrior:
        cw_thresh = 5 if attacker.clan_warrior_boost else 6
        cw_trigger_p = (7 - cw_thresh) / 6.0
        cw_attack_mult = 1.0 + cw_trigger_p
    else:
        cw_attack_mult = 1.0

    # ED Unpredictable Fighter — average across the d6 roll: 0.5 chance of
    # AP+1, 0.5 chance of +1 to hit. Approximate as half-bonus to each.
    uf_active = attacker.unpredictable_fighter

    total = 0.0
    for w in attacker.weapons:
        if not w.melee:
            continue

        # Thrust: +1 to hit on charge (quality -1, min 2)
        if w.thrust:
            thrust_quality = max(base_quality_eff - 1, 2)
            p = 5 / 6 if w.reliable else (7 - thrust_quality) / 6.0
        else:
            p = 5 / 6 if w.reliable else base_hit_prob

        # Versatile Attack: try +1-to-hit vs AP(+1), pick higher proxy.
        va_ap = 0
        if va_active:
            quality_hit = max((thrust_quality if w.thrust else base_quality_eff) - 1, 2)
            p_hit_va = 5 / 6 if w.reliable else (7 - quality_hit) / 6.0
            ap_a = w.ap + (1 if w.thrust else 0) + 1
            ap_b = w.ap + (1 if w.thrust else 0)
            blk_a = max((7 - min(d_def + ap_a, 7)) / 6.0, 1 / 6)
            blk_b = max((7 - min(d_def + ap_b, 7)) / 6.0, 1 / 6)
            ev_a = p * (1.0 - blk_a)
            ev_b = p_hit_va * (1.0 - blk_b)
            if ev_b > ev_a:
                p = p_hit_va
            else:
                va_ap = 1

        # Unpredictable Fighter average bonus: +0.5 AP and an effective
        # quality of base-0.5 (we simplify by averaging hit prob).
        uf_ap = 0.5 if uf_active else 0.0
        if uf_active:
            quality_h = max((thrust_quality if w.thrust else base_quality_eff) - 1, 2)
            p_uf_hit = 5 / 6 if w.reliable else (7 - quality_h) / 6.0
            p = 0.5 * p + 0.5 * p_uf_hit

        # ED Tear / Puncture conditional AP.
        ed_extra_ap = 0.0
        if w.tear and tear_active:
            ed_extra_ap += 4
        if w.puncture and puncture_active:
            ed_extra_ap += 4

        dice = w.attacks * cw_attack_mult

        # Split hits into nat6 and normal
        nat6_hits = dice * (1 / 6)
        normal_hits = dice * max(p - 1 / 6, 0.0)

        # Furious: each nat6 generates an extra normal hit (assume charge)
        if attacker.furious:
            normal_hits += nat6_hits

        # Surge in melee: +1 hit per nat 6 (per weapon).
        if w.surge:
            normal_hits += nat6_hits

        # Blast multiplier (Smash adds Blast(+3) vs def-5+ majority)
        eff_blast = w.blast
        smash_now = w.smash and smash_active
        if smash_now:
            eff_blast = (eff_blast or 0) + 3
        if eff_blast:
            blast_mult = min(eff_blast, defender.models)
            nat6_hits *= blast_mult
            normal_hits *= blast_mult

        # Base AP: weapon AP + thrust + Versatile Attack + Unpredictable
        # Fighter (averaged) + Tear/Puncture conditional, reduced by fortified.
        base_ap = w.ap + (1 if w.thrust else 0) + va_ap + uf_ap + ed_extra_ap
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
        if w.lacerate:
            block_nat6 = block_nat6 * block_nat6
        wounds_nat6 = nat6_hits * (1 - block_nat6)

        # --- Normal path ---
        normal_ap = base_ap + spotter_ap
        eff_def_normal = min(d_def + normal_ap, 7)
        block_normal = (_bane_block_prob(eff_def_normal) if w.bane
                        else max((7 - eff_def_normal) / 6.0, 1 / 6))
        if w.lacerate:
            block_normal = block_normal * block_normal
        wounds_normal = normal_hits * (1 - block_normal)

        # Shred extra wounds (1/6 chance per def die).
        if w.shred:
            wounds_normal += (nat6_hits + normal_hits) * (1.0 / 6.0)

        # Deadly multiplier (cap at model HP — overkill is wasted)
        if w.deadly:
            model_hp = defender.tough if defender.tough > 0 else 1
            effective_deadly = min(w.deadly, model_hp)
            wounds_nat6 *= effective_deadly
            wounds_normal *= effective_deadly

        weapon_wounds = wounds_nat6 + wounds_normal

        # Regeneration: 5+ negates → multiply by 2/3.
        # Bypassed by rending, unstoppable, bane, Smash, and Puncture.
        if (defender.regeneration
                and not (w.rending or w.unstoppable or w.bane or smash_now
                         or (w.puncture and puncture_active))):
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


def _projected_objective_control_mapped(
    board: Board,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    player: str,
) -> list[float]:
    """Projected objective control based on current unit positions.

    Same semantics as _objective_control_mapped (+1 friendly, -1 enemy, 0
    neutral), but computes what control *would* be if update_objectives ran
    right now, rather than reading the stale board.objective_control.

    If neither side is present the sticky value from board.objective_control
    is preserved (same rule as update_objectives).
    """
    threshold_sq = OBJ_SEIZE_RANGE * OBJ_SEIZE_RANGE

    if player == "A":
        a_units, b_units = friendly_units, enemy_units
    else:
        a_units, b_units = enemy_units, friendly_units

    projected: list[str] = list(board.objective_control)
    for oi, (oc, orow) in enumerate(OBJECTIVES):
        a_present = False
        for u in a_units:
            if u.models_alive <= 0 or u.shaken:
                continue
            for pos in u.alive_positions():
                dc = pos[0] - oc
                dr = pos[1] - orow
                if dc * dc + dr * dr <= threshold_sq:
                    a_present = True
                    break
            if a_present:
                break

        b_present = False
        for u in b_units:
            if u.models_alive <= 0 or u.shaken:
                continue
            for pos in u.alive_positions():
                dc = pos[0] - oc
                dr = pos[1] - orow
                if dc * dc + dr * dr <= threshold_sq:
                    b_present = True
                    break
            if b_present:
                break

        if a_present and b_present:
            projected[oi] = ""
        elif a_present:
            projected[oi] = "A"
        elif b_present:
            projected[oi] = "B"
        # else: keep sticky value

    friendly_tag = player
    enemy_tag = "B" if player == "A" else "A"

    def _val(idx: int) -> float:
        if projected[idx] == friendly_tag:
            return 1.0
        if projected[idx] == enemy_tag:
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
    """Write TACTICAL_UNIT_FEATURES (200) floats into buf for the v2 tactical model.

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
      177-186 10 objective reachability (5 obj × (can_advance, can_rush))
      187-196 10 can_charge (1 per opposing unit: 1.0 if charge is feasible)
      197     has_activated
      198     fatigued
      199     is_shaken
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

    # 177-186  Objective reachability: 5 objectives × (can_advance, can_rush)
    adv_budget = float(unit.advance_distance)
    rush_budget = float(unit.rush_distance)
    orb = offset + _TOFF_OBJ_REACH
    for ox, oy in objectives:
        dx = ox - cx
        dy = oy - cy
        d = math.sqrt(dx * dx + dy * dy)
        buf[orb] = 1.0 if d <= adv_budget + OBJ_SEIZE_RANGE else 0.0
        buf[orb + 1] = 1.0 if d <= rush_budget + OBJ_SEIZE_RANGE else 0.0
        orb += 2

    # 187-196  Can-charge: 1.0 if this unit can charge opposing unit j, else 0.0
    # Uses same logic as ai._can_charge: centre-to-centre dist < charge + 2.
    # Versatile Reach grants +2" charge via unit.charge_distance.
    charge_threshold = float(unit.charge_distance) + 2.0
    charge_threshold_sq = charge_threshold * charge_threshold
    cc = offset + _TOFF_CAN_CHARGE
    for j, (ox, oy) in enumerate(opposing_positions):
        if (ox, oy) == _DEAD_SENTINEL:
            continue
        dx = ox - cx
        dy = oy - cy
        if dx * dx + dy * dy < charge_threshold_sq:
            buf[cc + j] = 1.0

    # 197-199: tactical booleans written by encode_state_tactical caller


def _encode_unit_tactical_into_fast(
    us: UnitState,
    is_friendly: bool,
    player: str,
    objectives_np: np.ndarray,              # (10,) float64 — 5 × (x, y)
    opp_positions_np: np.ndarray,           # (20,) float64 — 10 × (x, y)
    opp_advance_np: np.ndarray,             # (10,) float64
    same_positions_np: np.ndarray,          # (20,) float64 — 10 × (x, y)
    ranged_matchups: np.ndarray,            # (10, 7) float32
    melee_matchups: np.ndarray,             # (10,) float32
    total_side_points: int,
    buf: np.ndarray,
    offset: int,
) -> None:
    """C-accelerated variant. Caller must skip dead units (models_alive <= 0).
    Writes the same 200-float block as `_encode_unit_tactical_into`.

    Shared-per-side arrays (objectives, positions, advance distances) are
    pre-built once by encode_state_tactical to avoid per-unit marshaling.
    """
    unit = us.unit
    cx, cy = us.centre()
    if player == "B":
        cx = _flip_x(cx)
        cy = _flip_y(cy)

    # Survival fraction (uses wounds_per_model for tough units)
    if unit.tough:
        total_start = unit.tough * unit.models
        total_rem = sum(
            unit.tough - w for w in us.wounds_per_model[: us.models_alive]
        )
        survival = total_rem / max(total_start, 1)
    else:
        survival = us.models_alive / max(unit.models, 1)

    wound_count = unit.tough if unit.tough > 0 else 1
    speed_val = 0.0 if unit.artillery else float(unit.rush_distance)
    points_frac = unit.points / max(total_side_points, 1)

    scalars = np.empty(15, dtype=np.float64)
    scalars[0] = wound_count
    scalars[1] = unit.models
    scalars[2] = speed_val
    scalars[3] = survival
    scalars[4] = points_frac
    scalars[5] = 1.0 if unit.flying else 0.0
    scalars[6] = 1.0 if unit.artillery else 0.0
    scalars[7] = 1.0 if unit.fearless else 0.0
    scalars[8] = 1.0 if unit.fear > 0 else 0.0
    scalars[9] = 1.0 if is_friendly else 0.0
    scalars[10] = cx
    scalars[11] = cy
    scalars[12] = float(unit.advance_distance)
    scalars[13] = float(unit.rush_distance)
    scalars[14] = us.models_alive

    _fc.fast_encode_unit_tactical(
        scalars,
        objectives_np,
        opp_positions_np,
        opp_advance_np,
        same_positions_np,
        ranged_matchups,
        melee_matchups,
        buf, offset,
        _INV_BOARD_DIAG,
        _MAX_TOUGH, _MAX_MODELS, _MAX_SPEED,
        OBJ_SEIZE_RANGE,
        _DEAD_SENTINEL,
        COLS, ROWS,
    )


# ---------------------------------------------------------------------------
# Can-charge mask extraction
# ---------------------------------------------------------------------------


def extract_can_charge_mask(
    state_vec: torch.Tensor,
    unit_idx: int | torch.Tensor,
) -> torch.Tensor:
    """Extract the can_charge flags for a friendly unit from the state vector.

    Parameters
    ----------
    state_vec : (feat,) or (N, feat) — encoded state
    unit_idx : int (single) or (N,) long tensor (batched)

    Returns
    -------
    (10,) or (N, 10) bool — True for each opposing unit that can be charged.
    """
    if isinstance(unit_idx, int):
        base = unit_idx * TACTICAL_UNIT_FEATURES + _TOFF_CAN_CHARGE
        return state_vec[..., base:base + MAX_UNITS_PER_SIDE] > 0.5
    # Batched: unit_idx is (N,) tensor
    bases = unit_idx.long() * TACTICAL_UNIT_FEATURES + _TOFF_CAN_CHARGE
    offsets = torch.arange(MAX_UNITS_PER_SIDE, device=state_vec.device)
    indices = bases.unsqueeze(1) + offsets.unsqueeze(0)  # (N, 10)
    return state_vec.gather(1, indices) > 0.5


def extract_is_shaken(
    state_vec: torch.Tensor,
    unit_idx: int | torch.Tensor,
) -> torch.Tensor:
    """Extract the is_shaken flag for a friendly unit from the state vector.

    Parameters
    ----------
    state_vec : (feat,) or (N, feat) — encoded state
    unit_idx : int (single) or (N,) long tensor (batched)

    Returns
    -------
    scalar bool or (N,) bool — True if the unit is Shaken.
    """
    if isinstance(unit_idx, int):
        idx = unit_idx * TACTICAL_UNIT_FEATURES + _TOFF_SHAKEN
        return state_vec[..., idx] > 0.5
    # Batched: unit_idx is (N,) tensor
    indices = unit_idx.long() * TACTICAL_UNIT_FEATURES + _TOFF_SHAKEN
    return state_vec.gather(1, indices.unsqueeze(1)).squeeze(1) > 0.5


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

    use_fast = _fc.USE_C_EXT and _fc.is_available()
    if use_fast:
        # Pre-build shared arrays once per side (avoids 20× marshaling).
        # Positions use _DEAD_SENTINEL for dead slots, so flatten directly.
        obj_np = np.asarray(objectives, dtype=np.float64).reshape(-1)
        enemy_pos_np = np.asarray(enemy_positions, dtype=np.float64).reshape(-1)
        friendly_pos_np = np.asarray(friendly_positions, dtype=np.float64).reshape(-1)
        enemy_adv_np = np.asarray(enemy_advance_dists, dtype=np.float64)
        friendly_adv_np = np.asarray(friendly_advance_dists, dtype=np.float64)

    # --- Friendly slots (0–9): TACTICAL_UNIT_FEATURES each ---
    for i in range(MAX_UNITS_PER_SIDE):
        offset = i * TACTICAL_UNIT_FEATURES
        if i < len(friendly_units):
            us = friendly_units[i]
            rm = friendly_ranged_matchups[i] if i < len(friendly_ranged_matchups) else _ZERO_RANGED_ROW
            mm = friendly_melee_matchups[i] if i < len(friendly_melee_matchups) else _ZERO_MELEE_ROW
            if use_fast and us.models_alive > 0:
                _encode_unit_tactical_into_fast(
                    us, True, player,
                    obj_np, enemy_pos_np, enemy_adv_np, friendly_pos_np,
                    rm, mm, total_friendly_points, buf, offset)
            else:
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
                buf[offset + _TOFF_IS_DEPLOYED] = 1.0

    # --- Enemy slots (10–19): TACTICAL_UNIT_FEATURES each ---
    enemy_base = MAX_UNITS_PER_SIDE * TACTICAL_UNIT_FEATURES
    for i in range(MAX_UNITS_PER_SIDE):
        offset = enemy_base + i * TACTICAL_UNIT_FEATURES
        if i < len(enemy_units):
            us = enemy_units[i]
            rm = enemy_ranged_matchups[i] if i < len(enemy_ranged_matchups) else _ZERO_RANGED_ROW
            mm = enemy_melee_matchups[i] if i < len(enemy_melee_matchups) else _ZERO_MELEE_ROW
            if use_fast and us.models_alive > 0:
                _encode_unit_tactical_into_fast(
                    us, False, player,
                    obj_np, friendly_pos_np, friendly_adv_np, enemy_pos_np,
                    rm, mm, total_enemy_points, buf, offset)
            else:
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
                buf[offset + _TOFF_IS_DEPLOYED] = 1.0

    # --- Global features ---
    g = MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES
    buf[g + _GOFF_ROUND + round_num - 1] = 1.0
    buf[g + _GOFF_OBJ_CTRL:g + _GOFF_OBJ_CTRL + 5] = _objective_control_mapped(board, player)
    buf[g + _GOFF_OBJ_CTRL_PROJ:g + _GOFF_OBJ_CTRL_PROJ + 5] = _projected_objective_control_mapped(
        board, friendly_units, enemy_units, player)
    alive_f = sum(u.unit.points for u in friendly_units if u.models_alive > 0)
    buf[g + _GOFF_ALIVE_F] = alive_f / max(total_friendly_points, 1)
    alive_e = sum(u.unit.points for u in enemy_units if u.models_alive > 0)
    buf[g + _GOFF_ALIVE_E] = alive_e / max(total_enemy_points, 1)
    # In-game state: deploy_phase = in_game (index 2 of the 3-hot). Remaining
    # deploy globals (is_my_deploy_turn, n_unplaced_*) stay at 0.
    buf[g + _GOFF_DEPLOY_PHASE + 2] = 1.0

    assert len(buf) == TACTICAL_TOTAL_FEATURES
    return torch.from_numpy(buf)


# ---------------------------------------------------------------------------
# Deployment-phase encoding
# ---------------------------------------------------------------------------

def _encode_unplaced_unit_into(
    us: UnitState,
    is_friendly: bool,
    ranged_matchups: np.ndarray,
    melee_matchups: np.ndarray,
    total_side_points: int,
    buf: np.ndarray,
    offset: int,
) -> None:
    """Write position-independent features for a unit that hasn't been placed yet.

    Fills the 10 stat scalars and the ranged/melee matchup blocks (which depend
    only on unit stats, not positions). is_deployed stays 0; every position-
    relative slot stays at the buffer's zero init."""
    unit = us.unit
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
    rs = offset + _TOFF_RANGED
    flat = ranged_matchups.reshape(-1)
    buf[rs:rs + flat.size] = flat
    ms = offset + _TOFF_MELEE
    buf[ms:ms + melee_matchups.size] = melee_matchups


_DEPLOY_PHASE_IDX = {"non_scout": 0, "scout": 1, "in_game": 2}


def encode_state_deploy(
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    board: Board,
    player: str,
    *,
    deploy_phase: str,
    is_my_turn: bool,
    friendly_ranged_matchups: np.ndarray | None = None,
    friendly_melee_matchups: np.ndarray | None = None,
    enemy_ranged_matchups: np.ndarray | None = None,
    enemy_melee_matchups: np.ndarray | None = None,
    total_friendly_points: int | None = None,
    total_enemy_points: int | None = None,
) -> torch.Tensor:
    """Encode a mid-deployment game state for the tactical model's deployment heads.

    Same tensor shape as encode_state_tactical (TACTICAL_TOTAL_FEATURES). Per-unit
    encoding for placed units (positions non-empty) mirrors encode_state_tactical,
    so the model sees the in-game-style relative geometry of whatever is on the
    board. Unplaced units get a stat-only encoding (no spurious zero-distance
    relations); is_deployed=0 distinguishes them.

    deploy_phase: one of "non_scout" or "scout" — drives the deploy_phase
        one-hot in the global block.
    is_my_turn: 1 if the *player* is about to deploy a unit right now.
    """
    if deploy_phase not in ("non_scout", "scout"):
        raise ValueError(f"deploy_phase must be 'non_scout' or 'scout', got {deploy_phase!r}")

    if total_friendly_points is None:
        total_friendly_points = sum(u.unit.points for u in friendly_units)
    if total_enemy_points is None:
        total_enemy_points = sum(u.unit.points for u in enemy_units)

    if friendly_ranged_matchups is None or friendly_melee_matchups is None:
        fr, fm = precompute_damage(
            [u.unit for u in friendly_units],
            [u.unit for u in enemy_units],
        )
        if friendly_ranged_matchups is None:
            friendly_ranged_matchups = fr
        if friendly_melee_matchups is None:
            friendly_melee_matchups = fm
    if enemy_ranged_matchups is None or enemy_melee_matchups is None:
        er, em = precompute_damage(
            [u.unit for u in enemy_units],
            [u.unit for u in friendly_units],
        )
        if enemy_ranged_matchups is None:
            enemy_ranged_matchups = er
        if enemy_melee_matchups is None:
            enemy_melee_matchups = em

    objectives = _get_model_objectives(player)
    # Positions: only placed enemies/friends contribute; everyone else uses
    # _DEAD_SENTINEL so the relative-geometry block is well-defined for the
    # placed units that look at them.
    def _positions(units: list[UnitState]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for i in range(MAX_UNITS_PER_SIDE):
            if i < len(units) and units[i].positions:
                cx, cy = units[i].centre()
                if player == "B":
                    cx = _flip_x(cx)
                    cy = _flip_y(cy)
                out.append((cx, cy))
            else:
                out.append(_DEAD_SENTINEL)
        return out

    enemy_positions = _positions(enemy_units)
    friendly_positions = _positions(friendly_units)
    enemy_advance_dists = [
        float(enemy_units[i].unit.advance_distance)
        if i < len(enemy_units) and enemy_units[i].positions
        else 0.0
        for i in range(MAX_UNITS_PER_SIDE)
    ]
    friendly_advance_dists = [
        float(friendly_units[i].unit.advance_distance)
        if i < len(friendly_units) and friendly_units[i].positions
        else 0.0
        for i in range(MAX_UNITS_PER_SIDE)
    ]

    buf = np.zeros(TACTICAL_TOTAL_FEATURES, dtype=np.float32)

    def _fill_side(units, ranged, melee, total_pts, base, is_friendly,
                   opp_positions, opp_adv, same_positions):
        for i in range(MAX_UNITS_PER_SIDE):
            offset = base + i * TACTICAL_UNIT_FEATURES
            if i >= len(units):
                continue
            us = units[i]
            rm = ranged[i] if i < len(ranged) else _ZERO_RANGED_ROW
            mm = melee[i] if i < len(melee) else _ZERO_MELEE_ROW
            if us.positions:
                _encode_unit_tactical_into(
                    us, is_friendly, player, objectives, rm, mm,
                    total_pts, opp_positions, opp_adv, same_positions,
                    buf, offset)
                buf[offset + _TOFF_IS_DEPLOYED] = 1.0
            elif us.models_alive > 0:
                _encode_unplaced_unit_into(
                    us, is_friendly, rm, mm, total_pts, buf, offset)

    _fill_side(friendly_units, friendly_ranged_matchups, friendly_melee_matchups,
               total_friendly_points, 0, True,
               enemy_positions, enemy_advance_dists, friendly_positions)
    enemy_base = MAX_UNITS_PER_SIDE * TACTICAL_UNIT_FEATURES
    _fill_side(enemy_units, enemy_ranged_matchups, enemy_melee_matchups,
               total_enemy_points, enemy_base, False,
               friendly_positions, friendly_advance_dists, enemy_positions)

    # --- Global features ---
    g = MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES
    # Objective control / projected control / alive-fraction blocks stay at 0
    # — they're undefined pre-deployment. Round one-hot stays at 0 too.
    buf[g + _GOFF_DEPLOY_PHASE + _DEPLOY_PHASE_IDX[deploy_phase]] = 1.0
    if is_my_turn:
        buf[g + _GOFF_MY_DEPLOY_TURN] = 1.0
    n_unplaced_self = sum(1 for u in friendly_units if not u.positions)
    n_unplaced_opp = sum(1 for u in enemy_units if not u.positions)
    buf[g + _GOFF_N_UNPLACED_SELF] = n_unplaced_self / float(MAX_UNITS_PER_SIDE)
    buf[g + _GOFF_N_UNPLACED_OPP] = n_unplaced_opp / float(MAX_UNITS_PER_SIDE)

    assert len(buf) == TACTICAL_TOTAL_FEATURES
    return torch.from_numpy(buf)


# ---------------------------------------------------------------------------
# Deployment action masks + index↔world coordinate mapping
# ---------------------------------------------------------------------------

def build_deploy_eligible_mask(
    units: list[UnitState],
    phase: str,
) -> torch.Tensor:
    """Per-friendly-slot bool mask of "right phase AND not yet placed AND
    actually exists in this slot". Length = MAX_UNITS_PER_SIDE."""
    want_scout = (phase == "scout")
    m = torch.zeros(MAX_UNITS_PER_SIDE, dtype=torch.bool)
    for i in range(MAX_UNITS_PER_SIDE):
        if i >= len(units):
            continue
        u = units[i]
        if u.positions:           # already placed
            continue
        if u.models_alive <= 0:   # not a real unit
            continue
        if bool(u.unit.scout) != want_scout:
            continue
        m[i] = True
    return m


def deploy_pos_idx_to_world(pos_idx: int, player: str) -> tuple[int, int]:
    """Map a deploy_pos_head logit index to a world (col, row) anchor.

    The head's grid is egocentric: index = depth * COLS + col_ego, depth
    counted from the deploying player's own back edge. For player A back-
    edge is row 0; for player B back-edge is row ROWS-1 and the egocentric
    column is also flipped to match _flip_x in the per-unit encoding."""
    depth = pos_idx // COLS
    col_ego = pos_idx % COLS
    if player == "A":
        return col_ego, depth
    return (COLS - 1) - col_ego, (ROWS - 1) - depth


def world_to_deploy_pos_idx(col: int, row: int, player: str) -> int:
    """Inverse of deploy_pos_idx_to_world."""
    if player == "A":
        depth, col_ego = row, col
    else:
        depth = (ROWS - 1) - row
        col_ego = (COLS - 1) - col
    return depth * COLS + col_ego


def build_deploy_legal_pos_mask(
    player: str,
    phase: str,
    board: Board,
    *,
    enemy_positions: set[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Bool mask of shape (DEPLOY_POS_GRID,) — True for legal anchor cells in
    the deploying player's own zone for the given phase.

    Rules:
      * Non-scout phase: own DZ depth = DEPLOY_POS_NONSCOUT_DEPTH (12 rows).
      * Scout phase: full DEPLOY_POS_DEPTH (24 rows) — own DZ + 12" forward.
      * Cell must be on the board, unoccupied, and inside the player's
        deployment zone for the phase (``Board.is_in_dz``). When the board
        carries map-driven DZ cell sets (``dz_*_cells``), this excludes
        walls and any irregular gaps; on a legacy empty board it reduces
        to the row-range check.
      * Scout phase: cell must not be in 1" exclusion of any enemy position
        (mirrors the per-step exclusion used by deploy_armies' scout phase).

    Multi-model "wholly within zone" fit is *not* enforced here — the
    spiral in _place_unit_at handles overflow via its fallback. A per-unit
    anchor-fit mask can layer on top of this for tighter training signal.
    """
    from movement import is_in_exclusion_zone

    if phase == "non_scout":
        depth_lim = DEPLOY_POS_NONSCOUT_DEPTH
        scout_zone = False
    elif phase == "scout":
        depth_lim = DEPLOY_POS_DEPTH
        scout_zone = True
    else:
        raise ValueError(f"phase must be 'non_scout' or 'scout', got {phase!r}")

    ep = enemy_positions if (phase == "scout") else None
    m = torch.zeros(DEPLOY_POS_GRID, dtype=torch.bool)
    for depth in range(depth_lim):
        for col_ego in range(COLS):
            world_col, world_row = deploy_pos_idx_to_world(depth * COLS + col_ego, player)
            if not board.is_free(world_col, world_row):
                continue
            if not board.is_in_dz(world_col, world_row, player, scout=scout_zone):
                continue
            if ep and is_in_exclusion_zone(world_col, world_row, ep):
                continue
            m[depth * COLS + col_ego] = True
    return m
