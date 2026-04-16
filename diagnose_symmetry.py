"""Diagnose value asymmetry by encoding the same board state from both
perspectives and comparing features that SHOULD be equivalent.

For a perfectly symmetric encoding of a board with units at known positions:
  encode(friendly=A_units, enemy=B_units, player="A")
should match (after swapping friendly↔enemy slots):
  encode(friendly=B_units, enemy=A_units, player="B")

Any feature mismatch pinpoints the encoding asymmetry.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch

from models import ArmyList, resolve_entry
from evolution import HallOfFame, resolve_army, _make_unit_states
from game import Board, UnitState, deploy_armies
from ml_features import (
    encode_state_tactical, precompute_damage,
    TACTICAL_UNIT_FEATURES, MAX_UNITS_PER_SIDE, TACTICAL_TOTAL_FEATURES,
    _get_model_objectives, _get_opposing_positions, _flip_x, _flip_y,
    _DEAD_SENTINEL,
    _TOFF_POS, _TOFF_OBJ_REL, _TOFF_OPP_REL, _TOFF_SAME_REL,
    _TOFF_RANGED, _TOFF_MELEE, _TOFF_OPP_POST_ADV, _TOFF_OBJ_REACH,
    _TOFF_CAN_CHARGE, _TOFF_ACTIVATED, _TOFF_FATIGUED, _TOFF_SHAKEN,
    NUM_RANGE_THRESHOLDS,
)


FEATURE_NAMES = {
    0: "wounds_norm", 1: "models_norm", 2: "speed_norm", 3: "survival_frac",
    4: "points_frac", 5: "flying", 6: "artillery", 7: "fearless",
    8: "fear", 9: "is_friendly",
    10: "pos_x", 11: "pos_y",
}
for i in range(5):
    FEATURE_NAMES[12 + i*3] = f"obj{i}_sin"
    FEATURE_NAMES[12 + i*3 + 1] = f"obj{i}_cos"
    FEATURE_NAMES[12 + i*3 + 2] = f"obj{i}_dist"
for i in range(10):
    FEATURE_NAMES[27 + i*3] = f"opp{i}_sin"
    FEATURE_NAMES[27 + i*3 + 1] = f"opp{i}_cos"
    FEATURE_NAMES[27 + i*3 + 2] = f"opp{i}_dist"
for i in range(10):
    FEATURE_NAMES[57 + i*3] = f"same{i}_sin"
    FEATURE_NAMES[57 + i*3 + 1] = f"same{i}_cos"
    FEATURE_NAMES[57 + i*3 + 2] = f"same{i}_dist"
for i in range(10):
    for k in range(7):
        FEATURE_NAMES[87 + i*7 + k] = f"ranged_{i}_t{k}"
for i in range(10):
    FEATURE_NAMES[157 + i] = f"melee_{i}"
for i in range(10):
    FEATURE_NAMES[167 + i] = f"post_adv_{i}"
for i in range(5):
    FEATURE_NAMES[177 + i*2] = f"obj{i}_adv_reach"
    FEATURE_NAMES[177 + i*2 + 1] = f"obj{i}_rush_reach"
for i in range(10):
    FEATURE_NAMES[187 + i] = f"can_charge_{i}"
FEATURE_NAMES[197] = "activated"
FEATURE_NAMES[198] = "fatigued"
FEATURE_NAMES[199] = "shaken"


def main():
    import fast_core
    fast_core.USE_C_EXT = fast_core.is_available()

    # Load HoF armies
    hof_path = Path(__file__).resolve().parent / "results" / "hall_of_fame_ml.json"
    hof = HallOfFame.load_from_json(hof_path)
    armies = [(e.army, resolve_army(e.army)) for e in hof.entries]

    # Set up a game state
    army_a, res_a = armies[0]
    army_b, res_b = armies[1]
    states_a = _make_unit_states(army_a, res_a, "A")
    states_b = _make_unit_states(army_b, res_b, "B")

    board = Board()
    deploy_armies(states_a, states_b, board)

    fr_a, fm_a = precompute_damage([u.unit for u in states_a],
                                   [u.unit for u in states_b])
    fr_b, fm_b = precompute_damage([u.unit for u in states_b],
                                   [u.unit for u in states_a])
    pts_a = sum(u.unit.points for u in states_a)
    pts_b = sum(u.unit.points for u in states_b)

    round_num = 1

    # ---- Encode from Player A's perspective ----
    vec_a = encode_state_tactical(
        states_a, states_b, round_num, board, "A",
        friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
        enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
        total_friendly_points=pts_a, total_enemy_points=pts_b,
    ).numpy()

    # ---- Encode from Player B's perspective ----
    vec_b = encode_state_tactical(
        states_b, states_a, round_num, board, "B",
        friendly_ranged_matchups=fr_b, friendly_melee_matchups=fm_b,
        enemy_ranged_matchups=fr_a, enemy_melee_matchups=fm_a,
        total_friendly_points=pts_b, total_enemy_points=pts_a,
    ).numpy()

    n_f = len(states_a)  # number of friendly units for A
    n_e = len(states_b)  # number of enemy units for A (friendly for B)

    print(f"Army A: {n_f} units, {pts_a} pts")
    print(f"Army B: {n_e} units, {pts_b} pts")
    print(f"State vector size: {len(vec_a)}")
    print()

    # ---- Check 1: Objectives should be identical for both perspectives ----
    print("=" * 60)
    print("CHECK 1: Model objectives")
    print("=" * 60)
    obj_a = _get_model_objectives("A")
    obj_b = _get_model_objectives("B")
    for i, (oa, ob) in enumerate(zip(obj_a, obj_b)):
        match = "OK" if oa == ob else "MISMATCH"
        print(f"  Obj {i}: A={oa}  B={ob}  {match}")
    print()

    # ---- Check 2: Position flipping ----
    print("=" * 60)
    print("CHECK 2: Position flipping symmetry")
    print("=" * 60)
    for i, us in enumerate(states_a):
        if us.models_alive <= 0:
            continue
        cx, cy = us.centre()
        fx, fy = _flip_x(cx), _flip_y(cy)
        print(f"  A unit {i} ({us.unit.name}): ({cx:.1f}, {cy:.1f}) -> flipped ({fx:.1f}, {fy:.1f})")
    for i, us in enumerate(states_b):
        if us.models_alive <= 0:
            continue
        cx, cy = us.centre()
        fx, fy = _flip_x(cx), _flip_y(cy)
        print(f"  B unit {i} ({us.unit.name}): ({cx:.1f}, {cy:.1f}) -> flipped ({fx:.1f}, {fy:.1f})")
    print()

    # ---- Check 3: Compare A's friendly slots with B's enemy slots ----
    # In vec_a, friendly slots 0..n_f-1 encode A's units from A's perspective.
    # In vec_b, enemy slots 0..n_f-1 encode A's units from B's perspective.
    # These should be identical EXCEPT:
    #   - is_friendly (feature 9): 1.0 in vec_a, 0.0 in vec_b
    #   - Position (features 10-11): should be flipped: a_pos + b_pos ≈ 1.0
    #   - Angle features: sin/cos should negate (180° rotation)
    #   - Opposing vs same-side slots swap
    #   - Matchup features swap (A's ranged_vs_B[i] != B's ranged_vs_A[i])
    print("=" * 60)
    print("CHECK 3: Feature comparison (A's unit i as friendly in A vs enemy in B)")
    print("=" * 60)

    # Compare A's unit 0 as seen by A (friendly slot 0) vs by B (enemy slot 0)
    for unit_i in range(min(n_f, 3)):  # first 3 units
        a_offset = unit_i * TACTICAL_UNIT_FEATURES
        b_offset = MAX_UNITS_PER_SIDE * TACTICAL_UNIT_FEATURES + unit_i * TACTICAL_UNIT_FEATURES

        a_feats = vec_a[a_offset:a_offset + TACTICAL_UNIT_FEATURES]
        b_feats = vec_b[b_offset:b_offset + TACTICAL_UNIT_FEATURES]

        print(f"\n  --- A unit {unit_i} ({states_a[unit_i].unit.name}) ---")

        # Scalars (0-9)
        print(f"  Scalars (0-9):")
        for f in range(10):
            name = FEATURE_NAMES.get(f, f"f{f}")
            av, bv = a_feats[f], b_feats[f]
            if f == 9:  # is_friendly
                expected_diff = True
                note = f" (expected: A=1.0 B=0.0)"
            else:
                expected_diff = False
                note = ""
            if abs(av - bv) > 1e-6 and not expected_diff:
                print(f"    {name}: A={av:.6f}  B={bv:.6f}  MISMATCH")
            elif expected_diff:
                print(f"    {name}: A={av:.6f}  B={bv:.6f}{note}")

        # Position (10-11): should satisfy a + b ≈ 1.0 (flipped)
        print(f"  Position (10-11):")
        for f in [10, 11]:
            name = FEATURE_NAMES.get(f, f"f{f}")
            av, bv = a_feats[f], b_feats[f]
            sym_check = av + bv
            status = "OK" if abs(sym_check - 1.0) < 0.02 else f"ASYM (sum={sym_check:.4f})"
            print(f"    {name}: A={av:.4f}  B={bv:.4f}  sum={sym_check:.4f}  {status}")

        # Objective relations (12-26): distances should match, angles should negate
        print(f"  Objective relations (12-26):")
        for oi in range(5):
            base = 12 + oi * 3
            a_sin, a_cos, a_dist = a_feats[base], a_feats[base+1], a_feats[base+2]
            b_sin, b_cos, b_dist = b_feats[base], b_feats[base+1], b_feats[base+2]
            dist_match = "OK" if abs(a_dist - b_dist) < 0.01 else f"MISMATCH ({a_dist:.4f} vs {b_dist:.4f})"
            sin_match = "OK" if abs(a_sin + b_sin) < 0.01 else f"MISMATCH (sum={a_sin+b_sin:.4f})"
            cos_match = "OK" if abs(a_cos + b_cos) < 0.01 else f"MISMATCH (sum={a_cos+b_cos:.4f})"
            print(f"    obj{oi}: dist {dist_match} | sin {sin_match} | cos {cos_match}")

    # ---- Check 4: Global features ----
    print(f"\n{'='*60}")
    print("CHECK 4: Global features")
    print("=" * 60)
    g = MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES
    global_a = vec_a[g:]
    global_b = vec_b[g:]
    global_names = ["round1", "round2", "round3", "round4",
                    "obj0_ctrl", "obj1_ctrl", "obj2_ctrl", "obj3_ctrl", "obj4_ctrl",
                    "obj0_proj", "obj1_proj", "obj2_proj", "obj3_proj", "obj4_proj",
                    "alive_frac_friendly", "alive_frac_enemy"]
    for i, name in enumerate(global_names):
        av, bv = global_a[i], global_b[i]
        if "ctrl" in name or "proj" in name:
            # These should negate between perspectives (friendly↔enemy)
            expected = "should negate" if av != 0 or bv != 0 else "both 0"
            print(f"  {name}: A={av:+.2f}  B={bv:+.2f}  (sum={av+bv:+.2f}, {expected})")
        elif "alive" in name:
            # alive_frac_friendly for A should be alive_frac_enemy for B
            print(f"  {name}: A={av:.4f}  B={bv:.4f}")
        else:
            match = "OK" if abs(av - bv) < 1e-6 else "MISMATCH"
            print(f"  {name}: A={av:.4f}  B={bv:.4f}  {match}")

    # ---- Check 5: MIRROR game — same army on both sides ----
    # This eliminates matchup asymmetry and isolates encoding issues
    print(f"\n{'='*60}")
    print("CHECK 5: MIRROR MATCH (same army both sides)")
    print("=" * 60)

    states_a2 = _make_unit_states(army_a, res_a, "A")
    states_b2 = _make_unit_states(army_a, res_a, "B")  # same army as B
    board2 = Board()
    deploy_armies(states_a2, states_b2, board2)

    fr_aa, fm_aa = precompute_damage([u.unit for u in states_a2],
                                     [u.unit for u in states_b2])
    pts_aa = sum(u.unit.points for u in states_a2)

    vec_mirror_a = encode_state_tactical(
        states_a2, states_b2, 1, board2, "A",
        friendly_ranged_matchups=fr_aa, friendly_melee_matchups=fm_aa,
        enemy_ranged_matchups=fr_aa, enemy_melee_matchups=fm_aa,
        total_friendly_points=pts_aa, total_enemy_points=pts_aa,
    ).numpy()

    vec_mirror_b = encode_state_tactical(
        states_b2, states_a2, 1, board2, "B",
        friendly_ranged_matchups=fr_aa, friendly_melee_matchups=fm_aa,
        enemy_ranged_matchups=fr_aa, enemy_melee_matchups=fm_aa,
        total_friendly_points=pts_aa, total_enemy_points=pts_aa,
    ).numpy()

    # In a mirror match with symmetric deployment, swapping friendly↔enemy
    # in the encoding should give identical vectors.
    # Compare: A's friendly slot i with B's friendly slot i
    # (both should describe the same "type" of unit in a mirror-symmetric position)
    print(f"\nMirror army: {len(states_a2)} units")

    # First just show positions
    print("\nUnit positions after deployment:")
    for i, us in enumerate(states_a2):
        if us.models_alive <= 0:
            continue
        cx, cy = us.centre()
        print(f"  A unit {i} ({us.unit.name}): ({cx:.1f}, {cy:.1f})")
    for i, us in enumerate(states_b2):
        if us.models_alive <= 0:
            continue
        cx, cy = us.centre()
        print(f"  B unit {i} ({us.unit.name}): ({cx:.1f}, {cy:.1f})")

    # In a perfect mirror, A's friendly slot i and B's friendly slot i
    # should have the same features (both describe the "i-th friendly unit")
    print(f"\nFriendly slot comparison (should be identical in mirror):")
    total_diffs = 0
    for unit_i in range(min(len(states_a2), len(states_b2))):
        a_offset = unit_i * TACTICAL_UNIT_FEATURES
        a_feats = vec_mirror_a[a_offset:a_offset + TACTICAL_UNIT_FEATURES]
        b_feats = vec_mirror_b[a_offset:a_offset + TACTICAL_UNIT_FEATURES]

        diffs = []
        for f in range(TACTICAL_UNIT_FEATURES):
            if abs(a_feats[f] - b_feats[f]) > 1e-4:
                diffs.append((f, a_feats[f], b_feats[f]))
                total_diffs += 1

        if diffs:
            print(f"\n  Unit {unit_i} ({states_a2[unit_i].unit.name}): {len(diffs)} feature mismatches")
            for f, av, bv in diffs[:20]:
                name = FEATURE_NAMES.get(f, f"f{f}")
                print(f"    {name} (f{f}): A={av:.6f}  B={bv:.6f}  diff={av-bv:+.6f}")
        else:
            print(f"  Unit {unit_i} ({states_a2[unit_i].unit.name}): OK (all features match)")

    # Global
    g = MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES
    ga = vec_mirror_a[g:]
    gb = vec_mirror_b[g:]
    print(f"\n  Global features:")
    for i, name in enumerate(global_names):
        if abs(ga[i] - gb[i]) > 1e-6:
            print(f"    {name}: A={ga[i]:.6f}  B={gb[i]:.6f}  MISMATCH")
    if all(abs(ga[i] - gb[i]) < 1e-6 for i in range(len(global_names))):
        print(f"    All global features match")

    print(f"\n  Total feature mismatches in mirror match: {total_diffs}")


if __name__ == "__main__":
    main()
