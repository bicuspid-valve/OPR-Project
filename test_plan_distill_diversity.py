"""Diagnose why plan_dl_unit/move/charge/shoot are always 0 in training logs.

Runs plan_training_activation many times and counts how often each sub-head
gets ≥2 distinct options for the chosen unit — the prerequisite for per-head
distillation loss to be non-zero.

Usage: .venv/bin/python3 test_plan_distill_diversity.py
"""
from __future__ import annotations

import json
import os
import random
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from ml_model_tactical import TacticalModel, NUM_MOVE_TYPES
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


# ── Helpers (reused from profile_training_planning.py) ──────────────────────

def load_army_from_hof(entry: dict) -> ArmyList:
    army = ArmyList()
    for e in entry["entries"]:
        ent = make_entry(e["template_id"], upgrades=e.get("upgrades", {}),
                         ai_role=e.get("ai_role", "killer"))
        ent.combat_preference = e.get("combat_preference", "ranged")
        army.entries.append(ent)
    return army


def setup_game(hof_data):
    army_a = load_army_from_hof(random.choice(hof_data))
    army_b = load_army_from_hof(random.choice(hof_data))
    res_a = resolve_army(army_a)
    res_b = resolve_army(army_b)
    sa = _make_unit_states(army_a, res_a, "A")
    sb = _make_unit_states(army_b, res_b, "B")
    board = Board()
    deploy_armies(sa, sb, board)
    fr_a, fm_a = precompute_damage([u.unit for u in sa], [u.unit for u in sb])
    fr_b, fm_b = precompute_damage([u.unit for u in sb], [u.unit for u in sa])
    pts_a = sum(u.unit.points for u in sa)
    pts_b = sum(u.unit.points for u in sb)
    return sa, sb, board, fr_a, fm_a, fr_b, fm_b, pts_a, pts_b


def prepare_inputs(units_a, units_b, board, fr_a, fm_a, fr_b, fm_b,
                   pts_a, pts_b, round_num=2):
    alive_mask = torch.tensor(
        [(i < len(units_a) and units_a[i].models_alive > 0
          and not units_a[i].activated)
         for i in range(MAX_UNITS_PER_SIDE)], dtype=torch.bool)
    enemy_alive_mask = torch.tensor(
        [(i < len(units_b) and units_b[i].models_alive > 0)
         for i in range(MAX_UNITS_PER_SIDE)], dtype=torch.bool)
    if not alive_mask.any():
        return None
    state_vec = encode_state_tactical(
        units_a, units_b, round_num, board, "A",
        friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
        enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
        total_friendly_points=pts_a, total_enemy_points=pts_b,
    )
    friendly_pos = _get_model_space_positions(units_a, "A")
    enemy_pos = _get_model_space_positions(units_b, "A")
    adv_dists, rush_dists = _get_movement_budgets(units_a)
    max_wr = _get_max_weapon_ranges(units_a)
    return (state_vec, alive_mask, enemy_alive_mask, friendly_pos, enemy_pos,
            adv_dists, rush_dists, max_wr)


# ── Main diagnostic ─────────────────────────────────────────────────────────

def run_diagnostic(model, hof_data, n_trials=200):
    """Call plan_training_activation n_trials times, inspect sub-head diversity."""

    # Default training params (K=3 units, C=3 samples/unit)
    params_default = {"K_UNITS": 3, "C_SAMPLES_PER_UNIT": 3,
                      "M_ROLLOUTS": 4, "N_LOOKAHEAD": 3}
    # Higher C to test if more samples help
    params_high_c = {"K_UNITS": 3, "C_SAMPLES_PER_UNIT": 8,
                     "M_ROLLOUTS": 4, "N_LOOKAHEAD": 3}

    for label, params in [("Default (C=3)", params_default),
                          ("High-C (C=8)", params_high_c)]:
        print(f"\n{'='*70}")
        print(f"  {label}  —  {n_trials} planning calls")
        print(f"{'='*70}")

        total = 0
        improved = 0

        # Counts: how many times each sub-head has ≥2 distinct options
        move_diverse = 0      # ≥2 distinct move types for chosen unit
        charge_diverse = 0    # ≥2 distinct charge targets (when move=charge)
        shoot_diverse = 0     # ≥2 distinct shoot targets (when move=hold/advance)

        # Extra detail
        move_counts = []      # number of distinct move types per call
        charge_counts = []    # number of distinct charge targets (charge calls only)
        shoot_counts = []     # number of distinct shoot targets (hold/advance only)
        charge_calls = 0      # calls where chosen move was charge
        shoot_calls = 0       # calls where chosen move was hold/advance

        for trial in range(n_trials):
            game = setup_game(hof_data)
            inputs = prepare_inputs(*game)
            if inputs is None:
                continue
            sv, am, eam, fp, ep, ad, rd, mw = inputs
            ua, ub, bd = game[0], game[1], game[2]
            fr_a, fm_a, fr_b, fm_b, pts_a, pts_b = game[3:]

            result = plan_training_activation(
                model, sv, am, eam, ua, ub, 2, bd, "A",
                current_is_a=True, mode="objectives",
                friendly_positions=fp, enemy_positions=ep,
                advance_distances=ad, rush_distances=rd,
                max_weapon_ranges=mw,
                fr_a=fr_a, fm_a=fm_a, fr_b=fr_b, fm_b=fm_b,
                pts_a=pts_a, pts_b=pts_b,
                planning_params=params, opponent_type=0,
            )

            # Unpack — plan_training_activation returns a big tuple:
            # (unit_idx, move_type, angle, dist_frac, charge_target_idx,
            #  shoot_target_idx, target_ranking, post_move_rel, old_log_prob,
            #  value, shoot_mask,
            #  was_planned, planning_improved, planning_value_delta,
            #  planning_unit_values, planning_unit_indices,
            #  planning_move_values, planning_move_indices,
            #  planning_charge_values, planning_charge_indices,
            #  planning_shoot_values, planning_shoot_indices)
            total += 1
            was_planned = result[11]
            planning_improved = result[12]
            move_type = result[1]

            planning_move_values = result[16]
            planning_move_indices = result[17]
            planning_charge_values = result[18]
            planning_charge_indices = result[19]
            planning_shoot_values = result[20]
            planning_shoot_indices = result[21]

            if planning_improved:
                improved += 1

            # Move diversity
            n_move = len(planning_move_indices) if planning_move_indices else 0
            move_counts.append(n_move)
            if n_move >= 2:
                move_diverse += 1

            # Charge diversity (only relevant when move_type == charge=3)
            from ml_model_tactical import MOVE_CHARGE, MOVE_HOLD, MOVE_ADVANCE
            if move_type == MOVE_CHARGE:
                charge_calls += 1
                n_charge = len(planning_charge_indices) if planning_charge_indices else 0
                charge_counts.append(n_charge)
                if n_charge >= 2:
                    charge_diverse += 1

            # Shoot diversity (only relevant when move_type == hold/advance)
            if move_type in (MOVE_HOLD, MOVE_ADVANCE):
                shoot_calls += 1
                n_shoot = len(planning_shoot_indices) if planning_shoot_indices else 0
                shoot_counts.append(n_shoot)
                if n_shoot >= 2:
                    shoot_diverse += 1

        # ── Report ──
        print(f"\n  Total planning calls:           {total}")
        print(f"  Planning improved (non-argmax):  {improved}  "
              f"({improved/total*100:.1f}%)" if total else "")

        print(f"\n  --- Move type head ---")
        print(f"  Calls with ≥2 distinct moves:   {move_diverse} / {total}  "
              f"({move_diverse/total*100:.1f}%)" if total else "")
        if move_counts:
            from collections import Counter
            dist = Counter(move_counts)
            print(f"  Distribution of #distinct move types: "
                  f"{dict(sorted(dist.items()))}")

        print(f"\n  --- Charge target head ---")
        print(f"  Calls where chosen=charge:      {charge_calls} / {total}  "
              f"({charge_calls/total*100:.1f}%)" if total else "")
        if charge_calls:
            print(f"  Of those, ≥2 distinct targets:  {charge_diverse} / {charge_calls}  "
                  f"({charge_diverse/charge_calls*100:.1f}%)")
            dist = Counter(charge_counts)
            print(f"  Distribution of #distinct charge targets: "
                  f"{dict(sorted(dist.items()))}")
        else:
            print(f"  (no charge actions sampled)")

        print(f"\n  --- Shoot target head ---")
        print(f"  Calls where chosen=hold/advance: {shoot_calls} / {total}  "
              f"({shoot_calls/total*100:.1f}%)" if total else "")
        if shoot_calls:
            print(f"  Of those, ≥2 distinct targets:  {shoot_diverse} / {shoot_calls}  "
                  f"({shoot_diverse/shoot_calls*100:.1f}%)")
            dist = Counter(shoot_counts)
            print(f"  Distribution of #distinct shoot targets: "
                  f"{dict(sorted(dist.items()))}")
        else:
            print(f"  (no hold/advance actions sampled)")

        # Effective distill eligibility (mirroring loss.py logic)
        eligible = 0
        for mc in move_counts:
            if mc >= 2:
                eligible += 1
        print(f"\n  --- Combined eligibility for loss.py ---")
        print(f"  Steps where planning_improved AND ≥2 moves: "
              f"{min(improved, move_diverse)}")
        print(f"  Steps where planning_improved AND ≥2 charge: "
              f"{min(improved, charge_diverse)}")
        print(f"  Steps where planning_improved AND ≥2 shoot: "
              f"{min(improved, shoot_diverse)}")
        print(f"  (loss.py requires BOTH planning_improved=True AND ≥2 "
              f"distinct options)")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    # Load HoF armies
    hof_path = os.path.join("results", "hall_of_fame_ml.json")
    if not os.path.exists(hof_path):
        hof_path = os.path.join("results", "hall_of_fame.json")
    with open(hof_path) as f:
        hof_data = json.load(f)
    print(f"Loaded {len(hof_data)} HoF armies from {hof_path}")

    # Load model
    ckpt = os.path.join("ml_checkpoints", "final_model.pt")
    if not os.path.exists(ckpt):
        import glob as _glob
        ckpts = sorted(_glob.glob("ml_checkpoints/checkpoint_batch_*.pt"),
                       key=os.path.getmtime)
        ckpt = ckpts[-1] if ckpts else None
    if not ckpt:
        print("No checkpoints found!")
        sys.exit(1)

    sd = load_model_state_dict(ckpt)
    model = TacticalModel()
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Loaded model from {ckpt}")

    run_diagnostic(model, hof_data, n_trials=200)
