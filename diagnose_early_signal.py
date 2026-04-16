"""Investigate what creates the side-detection signal in early Round 1.

At activation 1, the probe is at chance (54%). By activation 3 it's 58%.
What features carry this emerging signal?

Tests:
1. Which feature groups carry the early signal?
2. Does who goes first correlate?
3. What are the logistic regression weights pointing at?
"""
from __future__ import annotations

import random
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from evolution import HallOfFame, resolve_army, _make_unit_states
from game import _simulate_game_impl
from ml_features import (
    encode_state_tactical, precompute_damage,
    TACTICAL_UNIT_FEATURES, MAX_UNITS_PER_SIDE,
)
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from ml_integration_tactical import apply_tactical_model

# Feature group offsets within each 200-dim unit slot
GROUPS = {
    "scalars(0-9)":       (0, 10),
    "position(10-11)":    (10, 12),
    "obj_rel(12-26)":     (12, 27),
    "opp_rel(27-56)":     (27, 57),
    "same_rel(57-86)":    (57, 87),
    "matchups(87-166)":   (87, 167),
    "post_adv(167-176)":  (167, 177),
    "obj_reach(177-186)": (177, 187),
    "can_charge(187-196)":(187, 197),
    "tac_flags(197-199)": (197, 200),
}


def extract_group(vec_np, group_start, group_end):
    """Extract a feature group from all 20 unit slots."""
    feats = []
    for slot in range(MAX_UNITS_PER_SIDE * 2):
        offset = slot * TACTICAL_UNIT_FEATURES
        feats.extend(vec_np[offset + group_start:offset + group_end])
    return feats


def extract_global(vec_np):
    """Extract the 16 global features."""
    g = MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES
    return vec_np[g:].tolist()


def main():
    model = TacticalModel()
    model.load_state_dict(load_model_state_dict(
        Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt"), strict=False)
    model.eval()

    hof = HallOfFame.load_from_json(
        Path(__file__).resolve().parent / "results" / "hall_of_fame_ml.json")
    armies = [(e.army, resolve_army(e.army)) for e in hof.entries]

    N = 200
    # Store per-activation data
    samples = []  # list of (vec_a_np, vec_b_np, act_num, who_is_active)

    print(f"Collecting Round 1 activations from {N} games...")

    for gi in range(N):
        (aa, ra), (ab, rb) = random.sample(armies, 2)
        sa = _make_unit_states(aa, ra, "A")
        sb = _make_unit_states(ab, rb, "B")
        fr_a, fm_a = precompute_damage([u.unit for u in sa], [u.unit for u in sb])
        fr_b, fm_b = precompute_damage([u.unit for u in sb], [u.unit for u in sa])
        pts_a = sum(u.unit.points for u in sa)
        pts_b = sum(u.unit.points for u in sb)

        collected = []
        counter = [0]

        def make_fn(mdl, out, ctr):
            def fn(my_u, opp_u, rn, board, player, mfr, mfm, ofr, ofm, mp, op):
                if rn == 1:
                    ctr[0] += 1
                    if player == "A":
                        ua, ub = my_u, opp_u
                        fra, fma, frb, fmb = mfr, mfm, ofr, ofm
                        pa, pb = mp, op
                    else:
                        ua, ub = opp_u, my_u
                        fra, fma, frb, fmb = ofr, ofm, mfr, mfm
                        pa, pb = op, mp
                    va = encode_state_tactical(ua, ub, rn, board, "A",
                        friendly_ranged_matchups=fra, friendly_melee_matchups=fma,
                        enemy_ranged_matchups=frb, enemy_melee_matchups=fmb,
                        total_friendly_points=pa, total_enemy_points=pb)
                    vb = encode_state_tactical(ub, ua, rn, board, "B",
                        friendly_ranged_matchups=frb, friendly_melee_matchups=fmb,
                        enemy_ranged_matchups=fra, enemy_melee_matchups=fma,
                        total_friendly_points=pb, total_enemy_points=pa)
                    out.append((va.numpy(), vb.numpy(), ctr[0], player))

                active, tr, action, goal, ct, reason, _ = apply_tactical_model(
                    mdl, my_u, opp_u, rn, board, player,
                    friendly_ranged_matchups=mfr, friendly_melee_matchups=mfm,
                    enemy_ranged_matchups=ofr, enemy_melee_matchups=ofm,
                    total_friendly_points=mp, total_enemy_points=op)
                return active, tr, action, goal, ct, reason
            return fn

        fn = make_fn(model, collected, counter)
        _simulate_game_impl(ra, rb, mode="objectives", states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model, _tactical_inference_fn=fn)
        samples.extend(collected)

        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{N}")

    print(f"Total Round 1 samples: {len(samples)}")

    # === Test 1: Per-feature-group probe at early activations ===
    print(f"\n{'='*70}")
    print(f"TEST 1: Which feature groups carry the early signal?")
    print(f"{'='*70}")

    for act_range, act_label in [((1, 2), "act 1-2"), ((3, 5), "act 3-5"),
                                  ((6, 10), "act 6-10"), ((11, 25), "act 11+")]:
        lo, hi = act_range
        subset = [(a, b, n, p) for a, b, n, p in samples if lo <= n <= hi]
        if len(subset) < 30:
            continue

        X_dict = {name: [] for name in list(GROUPS.keys()) + ["global", "full_input"]}
        y = []
        for va, vb, _, _ in subset:
            for name, (gs, ge) in GROUPS.items():
                X_dict[name].append(extract_group(va, gs, ge))
                X_dict[name].append(extract_group(vb, gs, ge))
            X_dict["global"].append(extract_global(va))
            X_dict["global"].append(extract_global(vb))
            X_dict["full_input"].append(va.tolist())
            X_dict["full_input"].append(vb.tolist())
            y.extend([0, 1])

        y = np.array(y)
        n = len(y)
        folds = min(5, n // 10)
        if folds < 2:
            continue

        print(f"\n  {act_label} (n={n}):")
        print(f"  {'Group':<22} {'Dims':>6} {'Accuracy':>10}")
        print(f"  {'-'*42}")

        results = []
        for name in list(GROUPS.keys()) + ["global", "full_input"]:
            X = np.array(X_dict[name])
            clf = LogisticRegression(max_iter=1000, C=1.0)
            scores = cross_val_score(clf, X, y, cv=folds)
            results.append((name, X.shape[1], scores.mean()))

        results.sort(key=lambda x: -x[2])
        for name, dims, acc in results:
            marker = " ***" if acc > 0.55 else ""
            print(f"  {name:<22} {dims:>6} {acc:>9.1%}{marker}")

    # === Test 2: Does who goes first matter? ===
    print(f"\n{'='*70}")
    print(f"TEST 2: Does who goes first correlate with probe accuracy?")
    print(f"{'='*70}")

    # At activation 1, who is active?
    act1_a_first = [(a, b) for a, b, n, p in samples if n == 1 and p == "A"]
    act1_b_first = [(a, b) for a, b, n, p in samples if n == 1 and p == "B"]
    print(f"  Activation 1: A goes first in {len(act1_a_first)} games, "
          f"B goes first in {len(act1_b_first)} games")

    # === Test 3: What do the probe weights point at? ===
    print(f"\n{'='*70}")
    print(f"TEST 3: Probe weight analysis (activations 3-5, full input)")
    print(f"{'='*70}")

    subset = [(a, b, n, p) for a, b, n, p in samples if 3 <= n <= 5]
    X = []
    y = []
    for va, vb, _, _ in subset:
        X.append(va)
        X.append(vb)
        y.extend([0, 1])
    X = np.array(X)
    y = np.array(y)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    weights = clf.coef_[0]

    # Find top features by absolute weight
    top_indices = np.argsort(np.abs(weights))[::-1][:30]

    print(f"  Top 30 features by |weight|:")
    print(f"  {'Index':>6} {'Slot':>5} {'Offset':>7} {'Feature':>20} {'Weight':>10}")
    print(f"  {'-'*55}")

    FEAT_NAMES = {
        0: "wounds", 1: "models", 2: "speed", 3: "survival",
        4: "pts_frac", 5: "flying", 6: "artillery", 7: "fearless",
        8: "fear", 9: "is_friendly", 10: "pos_x", 11: "pos_y",
        197: "activated", 198: "fatigued", 199: "shaken",
    }
    for i in range(5):
        FEAT_NAMES[12 + i*3] = f"obj{i}_sin"
        FEAT_NAMES[12 + i*3+1] = f"obj{i}_cos"
        FEAT_NAMES[12 + i*3+2] = f"obj{i}_dist"
    for i in range(10):
        FEAT_NAMES[27 + i*3] = f"opp{i}_sin"
        FEAT_NAMES[27 + i*3+1] = f"opp{i}_cos"
        FEAT_NAMES[27 + i*3+2] = f"opp{i}_dist"
    for i in range(10):
        FEAT_NAMES[57 + i*3] = f"same{i}_sin"
        FEAT_NAMES[57 + i*3+1] = f"same{i}_cos"
        FEAT_NAMES[57 + i*3+2] = f"same{i}_dist"

    g_offset = MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES
    GLOBAL_NAMES = {
        0: "round1", 1: "round2", 2: "round3", 3: "round4",
        4: "obj0_ctrl", 5: "obj1_ctrl", 6: "obj2_ctrl", 7: "obj3_ctrl", 8: "obj4_ctrl",
        9: "obj0_proj", 10: "obj1_proj", 11: "obj2_proj", 12: "obj3_proj", 13: "obj4_proj",
        14: "alive_f", 15: "alive_e",
    }

    for idx in top_indices:
        w = weights[idx]
        if idx >= g_offset:
            gi = idx - g_offset
            name = GLOBAL_NAMES.get(gi, f"g{gi}")
            print(f"  {idx:>6} {'glob':>5} {gi:>7} {name:>20} {w:>+10.4f}")
        else:
            slot = idx // TACTICAL_UNIT_FEATURES
            offset = idx % TACTICAL_UNIT_FEATURES
            slot_type = "F" if slot < 10 else "E"
            slot_idx = slot if slot < 10 else slot - 10
            name = FEAT_NAMES.get(offset, f"f{offset}")
            print(f"  {idx:>6} {slot_type}{slot_idx:>4} {offset:>7} {name:>20} {w:>+10.4f}")

    # Summary: aggregate weight by feature type
    print(f"\n  Weight magnitude by feature type (summed across all slots):")
    type_weights = defaultdict(float)
    for idx, w in enumerate(weights):
        if idx >= g_offset:
            type_weights["global"] += abs(w)
        else:
            offset = idx % TACTICAL_UNIT_FEATURES
            for name, (gs, ge) in GROUPS.items():
                if gs <= offset < ge:
                    type_weights[name] += abs(w)
                    break

    for name, tw in sorted(type_weights.items(), key=lambda x: -x[1]):
        print(f"    {name:<22} {tw:.4f}")


if __name__ == "__main__":
    main()
