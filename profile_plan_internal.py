"""cProfile of plan_training_activation at user's exact params to find
what dominates the ~600ms cost — model fwd vs rollout sim vs pickle vs encode.
"""
from __future__ import annotations

import cProfile
import json
import os
import pstats
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from ml_model_tactical import TacticalModel
from ml_training import load_model_state_dict
from ml_features import (
    encode_state_tactical, precompute_damage, MAX_UNITS_PER_SIDE,
)
from ml_integration_tactical import (
    _get_model_space_positions, _get_movement_budgets, _get_max_weapon_ranges,
)
from ml_planning import plan_training_activation
from evolution import resolve_army, _make_unit_states, make_entry
from game import deploy_armies
from board import Board
from models import ArmyList

PARAMS = {
    "K_UNITS": 3, "C_SAMPLES_PER_UNIT": 3,
    "M_ROLLOUTS": 32, "N_LOOKAHEAD": 2,
    "SEQUENTIAL_HALVING": True, "SH_SCHEDULE": (8, 8, 16),
}


def load_army(entry):
    army = ArmyList()
    for e in entry["entries"]:
        ent = make_entry(e["template_id"], upgrades=e.get("upgrades", {}),
                         ai_role=e.get("ai_role", "killer"))
        ent.combat_preference = e.get("combat_preference", "ranged")
        army.entries.append(ent)
    return army


def setup_and_run(model, hof_data, n_iters=5):
    for _ in range(n_iters):
        a = load_army(random.choice(hof_data))
        b = load_army(random.choice(hof_data))
        res_a = resolve_army(a)
        res_b = resolve_army(b)
        sa = _make_unit_states(a, res_a, "A")
        sb = _make_unit_states(b, res_b, "B")
        board = Board()
        deploy_armies(sa, sb, board)
        fr_a, fm_a = precompute_damage([u.unit for u in sa], [u.unit for u in sb])
        fr_b, fm_b = precompute_damage([u.unit for u in sb], [u.unit for u in sa])
        pts_a = sum(u.unit.points for u in sa)
        pts_b = sum(u.unit.points for u in sb)

        alive_mask = torch.tensor(
            [(i < len(sa) and sa[i].models_alive > 0 and not sa[i].activated)
             for i in range(MAX_UNITS_PER_SIDE)], dtype=torch.bool)
        eam = torch.tensor(
            [(i < len(sb) and sb[i].models_alive > 0)
             for i in range(MAX_UNITS_PER_SIDE)], dtype=torch.bool)
        if not alive_mask.any():
            continue
        sv = encode_state_tactical(
            sa, sb, 2, board, "A",
            friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
            enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
            total_friendly_points=pts_a, total_enemy_points=pts_b)
        fp = _get_model_space_positions(sa, "A")
        ep = _get_model_space_positions(sb, "A")
        ad, rd = _get_movement_budgets(sa)
        mw = _get_max_weapon_ranges(sa)

        plan_training_activation(
            model, sv, alive_mask, eam, sa, sb, 2, board, "A",
            current_is_a=True, mode="objectives",
            friendly_positions=fp, enemy_positions=ep,
            advance_distances=ad, rush_distances=rd, max_weapon_ranges=mw,
            fr_a=fr_a, fm_a=fm_a, fr_b=fr_b, fm_b=fm_b,
            pts_a=pts_a, pts_b=pts_b,
            planning_params=PARAMS, opponent_type=0)


if __name__ == "__main__":
    random.seed(42)
    with open("results/hall_of_fame_ml.json") as f:
        hof = json.load(f)
    sd = load_model_state_dict("ml_checkpoints/final_model.pt")
    model = TacticalModel()
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Warm up
    setup_and_run(model, hof, n_iters=2)

    pr = cProfile.Profile()
    pr.enable()
    setup_and_run(model, hof, n_iters=10)
    pr.disable()

    stats = pstats.Stats(pr).strip_dirs().sort_stats("cumulative")
    print("\n=== TOP 40 cumulative ===")
    stats.print_stats(40)

    stats = pstats.Stats(pr).strip_dirs().sort_stats("tottime")
    print("\n=== TOP 30 tottime ===")
    stats.print_stats(30)
