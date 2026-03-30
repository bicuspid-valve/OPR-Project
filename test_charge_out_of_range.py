"""Diagnose out-of-range charge declarations in ML vs ML games.

Tests the hypothesis that dead enemy units at the sentinel position (board centre)
cause false can_charge flags, allowing MOVE_CHARGE when no alive target is in range.
"""

import json
import math
import random
from pathlib import Path
from collections import Counter

import torch

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel, MOVE_CHARGE
from ml_features import (
    encode_state_tactical, extract_can_charge_mask,
    TACTICAL_UNIT_FEATURES, _TOFF_CAN_CHARGE, MAX_UNITS_PER_SIDE,
    _DEAD_SENTINEL,
)
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList
from game import simulate_game
import ml_integration_tactical as mli

_DIR = Path(__file__).resolve().parent


def load_army_from_hof(hof_entry: dict) -> ArmyList:
    army = ArmyList()
    for e in hof_entry["entries"]:
        entry = make_entry(
            e["template_id"],
            upgrades=e.get("upgrades", {}),
            ai_role=e.get("ai_role", "killer"),
        )
        entry.combat_preference = e.get("combat_preference", "ranged")
        army.entries.append(entry)
    return army


def run_test(num_games: int = 100):
    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof_ml = json.load(f)

    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"Dead sentinel position: {_DEAD_SENTINEL}")

    # Counters
    total_activations = 0
    total_charge_decl = 0
    valid_charges = 0
    out_of_range = 0
    target_dead_reason = 0

    # Root cause counters
    cause_dead_sentinel = 0   # chargeable indices all point to dead units
    cause_other = 0

    oor_sentinel_dists = []  # distance from charger to sentinel for each OOR event

    _original_apply = mli.apply_tactical_model

    def _instrumented_apply(
        model_arg, friendly_units, enemy_units, round_num, board, player, **kwargs
    ):
        nonlocal total_activations, total_charge_decl, valid_charges, out_of_range
        nonlocal target_dead_reason, cause_dead_sentinel, cause_other

        result = _original_apply(
            model_arg, friendly_units, enemy_units, round_num, board, player, **kwargs
        )
        active, target_ranking, action, goal, charge_target, reason, assessment = result
        total_activations += 1

        if active is None:
            return result

        if assessment.get('move_type', '') != 'charge':
            return result

        total_charge_decl += 1

        if "model chose charge" in reason:
            valid_charges += 1
        elif "charge target dead" in reason:
            target_dead_reason += 1
        elif "out of range" in reason:
            out_of_range += 1

            selected_slot = assessment.get('selected_slot', -1)

            # Re-derive can_charge_mask
            state_vec = encode_state_tactical(
                friendly_units, enemy_units, round_num, board, player,
                **{k: v for k, v in kwargs.items()
                   if k in ('friendly_ranged_matchups', 'friendly_melee_matchups',
                            'enemy_ranged_matchups', 'enemy_melee_matchups',
                            'total_friendly_points', 'total_enemy_points')}
            )
            can_charge_mask = extract_can_charge_mask(state_vec, selected_slot)
            chargeable_indices = [i for i in range(MAX_UNITS_PER_SIDE)
                                  if can_charge_mask[i].item()]

            # Check: are ALL chargeable indices pointing to dead/missing units?
            all_dead = True
            for ci in chargeable_indices:
                if ci < len(enemy_units) and enemy_units[ci].models_alive > 0:
                    all_dead = False
                    break

            if all_dead and chargeable_indices:
                cause_dead_sentinel += 1

                # Compute charger distance to sentinel (in model space)
                from ml_features import _flip_x, _flip_y
                cx, cy = active.centre()
                if player == "B":
                    cx = _flip_x(cx)
                    cy = _flip_y(cy)
                sx, sy = _DEAD_SENTINEL
                d_sentinel = math.sqrt((cx - sx)**2 + (cy - sy)**2)
                rush_d = active.unit.rush_distance
                threshold = rush_d + 2
                oor_sentinel_dists.append({
                    'dist_to_sentinel': d_sentinel,
                    'threshold': threshold,
                    'rush': rush_d,
                    'within': d_sentinel < threshold,
                    'num_dead_chargeable': len(chargeable_indices),
                })
            else:
                cause_other += 1

        return result

    mli.apply_tactical_model = _instrumented_apply

    wins = {"A": 0, "B": 0, "draw": 0}
    print(f"\nRunning {num_games} ML vs ML games...")
    print("-" * 60)

    for i in range(num_games):
        army_a = load_army_from_hof(random.choice(hof_ml))
        army_b = load_army_from_hof(random.choice(hof_ml))
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")
        result = simulate_game(
            res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
        )
        wins[result] += 1
        if (i + 1) % 25 == 0:
            print(f"  Game {i+1}/{num_games}  "
                  f"(charges: {total_charge_decl}, OOR: {out_of_range})")

    mli.apply_tactical_model = _original_apply

    print()
    print("=" * 70)
    print("CHARGE OUT-OF-RANGE ANALYSIS")
    print("=" * 70)
    print(f"Games: {num_games}  |  Activations: {total_activations}")
    print(f"Charge declarations: {total_charge_decl}")
    if total_charge_decl > 0:
        print(f"  Valid:        {valid_charges:>5}  ({valid_charges/total_charge_decl*100:.1f}%)")
        print(f"  Out-of-range: {out_of_range:>5}  ({out_of_range/total_charge_decl*100:.1f}%)")
        print(f"  Target dead:  {target_dead_reason:>5}  ({target_dead_reason/total_charge_decl*100:.1f}%)")

    print()
    print("-" * 70)
    print("ROOT CAUSE")
    print("-" * 70)
    print(f"  Dead-sentinel false positive: {cause_dead_sentinel} / {out_of_range} "
          f"({cause_dead_sentinel/max(out_of_range,1)*100:.1f}%)")
    print(f"  Other cause:                  {cause_other} / {out_of_range}")
    print()
    print(f"  Explanation: The can_charge encoding at ml_features.py:619-628 does NOT")
    print(f"  check if the opposing unit is alive. Dead units use the sentinel position")
    print(f"  at board centre {_DEAD_SENTINEL}. When a friendly unit is within")
    print(f"  rush_distance+2 of the sentinel, it gets can_charge=True for dead slots.")
    print(f"  This allows MOVE_CHARGE, but charge_target_logits are masked by")
    print(f"  enemy_alive_mask too, so all targets become -inf → argmax picks idx=0.")

    if oor_sentinel_dists:
        print()
        print(f"  Charger distance to sentinel ({_DEAD_SENTINEL}):")
        dists = [d['dist_to_sentinel'] for d in oor_sentinel_dists]
        thresholds = [d['threshold'] for d in oor_sentinel_dists]
        all_within = all(d['within'] for d in oor_sentinel_dists)
        print(f"    All within threshold: {all_within}")
        print(f"    Distance range: {min(dists):.1f} - {max(dists):.1f}")
        print(f"    Threshold range: {min(thresholds):.0f} - {max(thresholds):.0f}")

        # Number of dead slots marked chargeable
        dead_counts = Counter(d['num_dead_chargeable'] for d in oor_sentinel_dists)
        print(f"    Dead slots marked chargeable: {dict(dead_counts)}")

    print(f"\nGame results:  A={wins['A']}  B={wins['B']}  draws={wins['draw']}")
    print("=" * 70)


if __name__ == "__main__":
    run_test(100)
