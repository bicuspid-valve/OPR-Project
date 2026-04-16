"""Feature ablation study: systematically zero out feature groups in both
A and B encodings and measure the effect on V_A + V_B.

If zeroing out a feature group reduces the gap, that group is a major
contributor to the asymmetry.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch

from evolution import HallOfFame, resolve_army, _make_unit_states
from game import Board, deploy_armies
from ml_features import (
    encode_state_tactical, precompute_damage,
    TACTICAL_UNIT_FEATURES, MAX_UNITS_PER_SIDE,
    _TOFF_POS, _TOFF_OBJ_REL, _TOFF_OPP_REL, _TOFF_SAME_REL,
    _TOFF_RANGED, _TOFF_MELEE, _TOFF_OPP_POST_ADV, _TOFF_OBJ_REACH,
    _TOFF_CAN_CHARGE, _TOFF_ACTIVATED, _TOFF_FATIGUED, _TOFF_SHAKEN,
    NUM_RANGE_THRESHOLDS,
)
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel


def get_value(model, vec):
    with torch.no_grad():
        h, _, round_oh = model.trunk(vec.unsqueeze(0))
        opp_embed = model._get_opp_embed(h, None)
        return model.value_head(h, round_oh, opp_embed).item()


# Define feature groups within each 200-dim unit slot
FEATURE_GROUPS = {
    "scalars(0-9)": (0, 10),
    "position(10-11)": (10, 12),
    "obj_relations(12-26)": (12, 27),
    "opp_relations(27-56)": (27, 57),
    "same_relations(57-86)": (57, 87),
    "ranged_matchups(87-156)": (87, 157),
    "melee_matchups(157-166)": (157, 167),
    "post_adv_dist(167-176)": (167, 177),
    "obj_reachability(177-186)": (177, 187),
    "can_charge(187-196)": (187, 197),
    "tac_flags(197-199)": (197, 200),
}


def zero_feature_group_all_units(vec, group_start, group_end):
    """Zero out a feature group across all 20 unit slots."""
    v = vec.clone()
    for slot in range(MAX_UNITS_PER_SIDE * 2):
        offset = slot * TACTICAL_UNIT_FEATURES
        v[offset + group_start:offset + group_end] = 0.0
    return v


def zero_global_feature(vec, global_start, global_end):
    """Zero out specific global features."""
    v = vec.clone()
    g = MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES
    v[g + global_start:g + global_end] = 0.0
    return v


def main():
    import fast_core
    fast_core.USE_C_EXT = fast_core.is_available()

    model_path = Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt"
    sd = load_model_state_dict(model_path)
    model = TacticalModel()
    model.load_state_dict(sd, strict=False)
    model.eval()

    hof_path = Path(__file__).resolve().parent / "results" / "hall_of_fame_ml.json"
    hof = HallOfFame.load_from_json(hof_path)
    armies = [(e.army, resolve_army(e.army)) for e in hof.entries]

    N = 200
    # Collect baseline gaps and per-ablation gaps
    baseline_gaps = []
    ablation_gaps = {name: [] for name in FEATURE_GROUPS}
    ablation_gaps["global_obj_ctrl(4-8)"] = []
    ablation_gaps["global_proj_ctrl(9-13)"] = []
    ablation_gaps["global_alive_frac(14-15)"] = []

    for trial in range(N):
        (army_a, res_a), (army_b, res_b) = random.sample(armies, 2)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")
        board = Board()
        deploy_armies(sa, sb, board)

        fr_a, fm_a = precompute_damage([u.unit for u in sa], [u.unit for u in sb])
        fr_b, fm_b = precompute_damage([u.unit for u in sb], [u.unit for u in sa])
        pts_a = sum(u.unit.points for u in sa)
        pts_b = sum(u.unit.points for u in sb)

        vec_a = encode_state_tactical(
            sa, sb, 1, board, "A",
            friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
            enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
            total_friendly_points=pts_a, total_enemy_points=pts_b,
        )
        vec_b = encode_state_tactical(
            sb, sa, 1, board, "B",
            friendly_ranged_matchups=fr_b, friendly_melee_matchups=fm_b,
            enemy_ranged_matchups=fr_a, enemy_melee_matchups=fm_a,
            total_friendly_points=pts_b, total_enemy_points=pts_a,
        )

        # Baseline
        va = get_value(model, vec_a)
        vb = get_value(model, vec_b)
        baseline_gaps.append(va + vb)

        # Ablate each unit feature group
        for name, (gs, ge) in FEATURE_GROUPS.items():
            va_abl = get_value(model, zero_feature_group_all_units(vec_a, gs, ge))
            vb_abl = get_value(model, zero_feature_group_all_units(vec_b, gs, ge))
            ablation_gaps[name].append(va_abl + vb_abl)

        # Ablate global features
        va_abl = get_value(model, zero_global_feature(vec_a, 4, 9))
        vb_abl = get_value(model, zero_global_feature(vec_b, 4, 9))
        ablation_gaps["global_obj_ctrl(4-8)"].append(va_abl + vb_abl)

        va_abl = get_value(model, zero_global_feature(vec_a, 9, 14))
        vb_abl = get_value(model, zero_global_feature(vec_b, 9, 14))
        ablation_gaps["global_proj_ctrl(9-13)"].append(va_abl + vb_abl)

        va_abl = get_value(model, zero_global_feature(vec_a, 14, 16))
        vb_abl = get_value(model, zero_global_feature(vec_b, 14, 16))
        ablation_gaps["global_alive_frac(14-15)"].append(va_abl + vb_abl)

    baseline = np.mean(baseline_gaps)
    print(f"Feature ablation study ({N} board states at deployment)")
    print(f"{'='*65}")
    print(f"Baseline V_A + V_B: {baseline:+.4f}")
    print()
    print(f"{'Feature group':<30} {'Gap':>8} {'Delta':>8} {'Effect':>12}")
    print("-" * 65)

    results = []
    for name in list(FEATURE_GROUPS.keys()) + [
        "global_obj_ctrl(4-8)", "global_proj_ctrl(9-13)", "global_alive_frac(14-15)"
    ]:
        gap = np.mean(ablation_gaps[name])
        delta = gap - baseline
        results.append((name, gap, delta))

    # Sort by delta magnitude (most gap-reducing first)
    results.sort(key=lambda x: x[2])

    for name, gap, delta in results:
        direction = "REDUCES gap" if delta < -0.01 else "INCREASES gap" if delta > 0.01 else "minimal"
        print(f"  {name:<28} {gap:>+.4f} {delta:>+.4f}   {direction}")


if __name__ == "__main__":
    main()
