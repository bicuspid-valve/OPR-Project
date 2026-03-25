"""Diagnose WHY argmax loses to sampling with random weights.

Questions:
1. Does argmax always pick the same unit slot?
2. Does it lock onto a specific role/stance/objective?
3. How peaked are the softmax distributions from random weights?
4. Does the "locked" choice correlate with unit quality?
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from collections import Counter
from ml_training import _make_model, _generate_army_pair
from ml_features import encode_state_tactical, MAX_UNITS_PER_SIDE
from ml_integration_tactical import ROLES, STANCES
from board import Board, OBJECTIVES
from ml_features import precompute_damage
from game import deploy_armies


def main():
    tac_model = _make_model("tactical")
    tac_model.eval()

    N_GAMES = 100
    N_ACTIVATIONS = 0

    # Track argmax choices
    argmax_unit_picks = Counter()
    sample_unit_picks = Counter()
    argmax_role_picks = Counter()
    sample_role_picks = Counter()
    argmax_stance_picks = Counter()
    sample_stance_picks = Counter()
    argmax_obj_picks = Counter()
    sample_obj_picks = Counter()

    # Track distribution stats
    unit_entropies = []
    unit_max_probs = []
    role_entropies = []
    stance_entropies = []
    obj_entropies = []

    # Track unit quality correlation
    argmax_unit_points = []
    argmax_unit_models_alive = []
    all_unit_points = []
    all_unit_models_alive = []

    # Track if argmax picks change across activations within a game
    per_game_argmax_units = []

    for game_i in range(N_GAMES):
        res_a, res_b, sa, sb, *_ = _generate_army_pair()
        units_a = [__import__('models', fromlist=['UnitState']).UnitState(ru) for ru in res_a]
        units_b = [__import__('models', fromlist=['UnitState']).UnitState(ru) for ru in res_b]
        for u in units_a:
            u.owner = "A"
        for u in units_b:
            u.owner = "B"

        board = Board()
        deploy_armies(units_a, units_b, board)

        fr_a, fm_a = precompute_damage([u.unit for u in units_a], [u.unit for u in units_b])
        fr_b, fm_b = precompute_damage([u.unit for u in units_b], [u.unit for u in units_a])
        pts_a = sum(u.unit.points for u in units_a)
        pts_b = sum(u.unit.points for u in units_b)

        game_argmax_units = []

        # Simulate multiple activations within round 1
        for u in units_a:
            u.activated = False

        for activation in range(len(units_a)):
            alive_mask_list = []
            for i in range(MAX_UNITS_PER_SIDE):
                if i < len(units_a):
                    us = units_a[i]
                    alive_mask_list.append(us.models_alive > 0 and not us.activated)
                else:
                    alive_mask_list.append(False)

            if not any(alive_mask_list):
                break

            alive_mask = torch.tensor(alive_mask_list, dtype=torch.bool)

            state_vec = encode_state_tactical(
                units_a, units_b, 1, board, "A",
                friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
                enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
                total_friendly_points=pts_a, total_enemy_points=pts_b,
            )

            with torch.no_grad():
                (unit_logits, role_probs, obj_probs, target_priority,
                 combat_pref, stance_probs, value) = tac_model(state_vec, alive_mask)

            N_ACTIVATIONS += 1

            # --- Argmax choices ---
            argmax_unit = int(unit_logits.argmax().item())
            argmax_role = int(role_probs.argmax().item())
            argmax_obj = int(obj_probs.argmax().item())
            argmax_stance = int(stance_probs.argmax().item())

            argmax_unit_picks[argmax_unit] += 1
            argmax_role_picks[argmax_role] += 1
            argmax_obj_picks[argmax_obj] += 1
            argmax_stance_picks[argmax_stance] += 1
            game_argmax_units.append(argmax_unit)

            # --- Sample choices (do 10 samples to see distribution) ---
            unit_probs = torch.softmax(unit_logits, dim=-1)
            for _ in range(10):
                s_unit = int(torch.multinomial(unit_probs, 1).item())
                s_role = int(torch.multinomial(role_probs, 1).item())
                s_obj = int(torch.multinomial(obj_probs, 1).item())
                s_stance = int(torch.multinomial(stance_probs, 1).item())
                sample_unit_picks[s_unit] += 1
                sample_role_picks[s_role] += 1
                sample_obj_picks[s_obj] += 1
                sample_stance_picks[s_stance] += 1

            # --- Distribution stats ---
            def entropy(probs):
                p = probs.clamp(min=1e-8)
                return -(p * p.log()).sum().item()

            unit_entropies.append(entropy(unit_probs))
            unit_max_probs.append(unit_probs.max().item())
            role_entropies.append(entropy(role_probs))
            stance_entropies.append(entropy(stance_probs))
            obj_entropies.append(entropy(obj_probs))

            # --- Unit quality for argmax pick ---
            if argmax_unit < len(units_a):
                picked = units_a[argmax_unit]
                argmax_unit_points.append(picked.unit.points)
                argmax_unit_models_alive.append(picked.models_alive)

            for u in units_a:
                if u.models_alive > 0 and not u.activated:
                    all_unit_points.append(u.unit.points)
                    all_unit_models_alive.append(u.models_alive)

            # Mark the argmax pick as activated so next activation sees different state
            units_a[argmax_unit].activated = True

        per_game_argmax_units.append(game_argmax_units)

    # --- Report ---
    print(f"Total activations analyzed: {N_ACTIVATIONS}")
    print()

    print("=== 1. UNIT SELECTION DISTRIBUTION ===")
    n_alive = sum(argmax_unit_picks.values())
    print(f"Argmax unit picks (n={n_alive}):")
    for slot in range(MAX_UNITS_PER_SIDE):
        a_count = argmax_unit_picks.get(slot, 0)
        s_count = sample_unit_picks.get(slot, 0)
        if a_count > 0 or s_count > 0:
            print(f"  Slot {slot}: argmax={a_count:4d} ({a_count/n_alive:.3f})  "
                  f"sample={s_count:4d} ({s_count/(n_alive*10):.3f})")

    print()
    print("=== 2. ROLE DISTRIBUTION ===")
    for i, role in enumerate(ROLES):
        a = argmax_role_picks.get(i, 0)
        s = sample_role_picks.get(i, 0)
        print(f"  {role:20s}: argmax={a/n_alive:.3f}  sample={s/(n_alive*10):.3f}")

    print()
    print("=== 3. STANCE DISTRIBUTION ===")
    for i, stance in enumerate(STANCES):
        a = argmax_stance_picks.get(i, 0)
        s = sample_stance_picks.get(i, 0)
        print(f"  {stance:20s}: argmax={a/n_alive:.3f}  sample={s/(n_alive*10):.3f}")

    print()
    print("=== 4. OBJECTIVE DISTRIBUTION ===")
    for i in range(5):
        a = argmax_obj_picks.get(i, 0)
        s = sample_obj_picks.get(i, 0)
        print(f"  Obj {i}: argmax={a/n_alive:.3f}  sample={s/(n_alive*10):.3f}")

    print()
    print("=== 5. SOFTMAX DISTRIBUTION STATS ===")
    n_units_alive_avg = np.mean([sum(1 for v in m if v) for m in
                                 [[True]*len(units_a)]])  # approximate
    print(f"  Unit selection:  entropy={np.mean(unit_entropies):.3f} "
          f"(max possible ~{np.log(5):.3f} for 5 units)  "
          f"max_prob={np.mean(unit_max_probs):.3f}")
    print(f"  Role (2 cls):    entropy={np.mean(role_entropies):.3f} "
          f"(max={np.log(2):.3f})")
    print(f"  Stance (3 cls):  entropy={np.mean(stance_entropies):.3f} "
          f"(max={np.log(3):.3f})")
    print(f"  Objective (5 cls): entropy={np.mean(obj_entropies):.3f} "
          f"(max={np.log(5):.3f})")

    print()
    print("=== 6. ARGMAX UNIT QUALITY ===")
    print(f"  Argmax picks avg points:       {np.mean(argmax_unit_points):.1f}")
    print(f"  All available avg points:      {np.mean(all_unit_points):.1f}")
    print(f"  Argmax picks avg models_alive: {np.mean(argmax_unit_models_alive):.2f}")
    print(f"  All available avg models_alive:{np.mean(all_unit_models_alive):.2f}")

    print()
    print("=== 7. DOES ARGMAX CHANGE ACROSS ACTIVATIONS? ===")
    # For first 10 games, show the sequence of argmax unit picks
    for i, seq in enumerate(per_game_argmax_units[:10]):
        unique = len(set(seq))
        print(f"  Game {i}: picks={seq}  unique={unique}/{len(seq)}")

    # Overall: how often does argmax pick the same first unit?
    first_picks = Counter(seq[0] for seq in per_game_argmax_units if seq)
    print(f"\n  First activation argmax pick distribution: {dict(first_picks)}")


if __name__ == "__main__":
    main()
