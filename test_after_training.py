"""Check argmax vs sampling distributions after 30 batches of training.

Compare the role/stance/objective concentration before and after training
to see if the distributions become peaked enough that argmax and sampling
converge.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from collections import Counter
from ml_training import _make_model, _generate_army_pair, TrainingConfig, run_training
from ml_features import encode_state_tactical, MAX_UNITS_PER_SIDE, precompute_damage
from ml_integration_tactical import ROLES, STANCES
from board import Board, OBJECTIVES
from game import deploy_armies


def analyze_model(model, label, n_games=100):
    """Run n_games and collect argmax vs sampling distribution stats."""
    model.eval()

    argmax_role = Counter()
    argmax_stance = Counter()
    argmax_obj = Counter()
    sample_role = Counter()
    sample_stance = Counter()
    sample_obj = Counter()

    unit_entropies = []
    role_entropies = []
    stance_entropies = []
    obj_entropies = []
    n_act = 0

    for _ in range(n_games):
        res_a, res_b, sa, sb, *_ = _generate_army_pair()
        from models import UnitState
        units_a = [UnitState(ru) for ru in res_a]
        units_b = [UnitState(ru) for ru in res_b]
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
                 combat_pref, stance_probs, value) = model(state_vec, alive_mask)

            n_act += 1

            # Argmax
            argmax_role[int(role_probs.argmax().item())] += 1
            argmax_stance[int(stance_probs.argmax().item())] += 1
            argmax_obj[int(obj_probs.argmax().item())] += 1

            # Sample (10x)
            for _ in range(10):
                sample_role[int(torch.multinomial(role_probs, 1).item())] += 1
                sample_stance[int(torch.multinomial(stance_probs, 1).item())] += 1
                sample_obj[int(torch.multinomial(obj_probs, 1).item())] += 1

            # Entropy
            def entropy(p):
                p = p.clamp(min=1e-8)
                return -(p * p.log()).sum().item()

            unit_probs = torch.softmax(unit_logits, dim=-1)
            unit_entropies.append(entropy(unit_probs))
            role_entropies.append(entropy(role_probs))
            stance_entropies.append(entropy(stance_probs))
            obj_entropies.append(entropy(obj_probs))

            units_a[int(unit_logits.argmax().item())].activated = True

    # Report
    print(f"\n--- {label} ({n_act} activations) ---")

    print(f"\n  Role distribution:")
    for i, role in enumerate(ROLES):
        a = argmax_role.get(i, 0) / n_act
        s = sample_role.get(i, 0) / (n_act * 10)
        print(f"    {role:20s}: argmax={a:.3f}  sample={s:.3f}")

    print(f"\n  Stance distribution:")
    for i, stance in enumerate(STANCES):
        a = argmax_stance.get(i, 0) / n_act
        s = sample_stance.get(i, 0) / (n_act * 10)
        print(f"    {stance:20s}: argmax={a:.3f}  sample={s:.3f}")

    print(f"\n  Objective distribution:")
    for i in range(5):
        a = argmax_obj.get(i, 0) / n_act
        s = sample_obj.get(i, 0) / (n_act * 10)
        print(f"    Obj {i}: argmax={a:.3f}  sample={s:.3f}")

    print(f"\n  Entropy (mean):")
    print(f"    Unit:      {np.mean(unit_entropies):.3f}  (max~{np.log(5):.3f})")
    print(f"    Role:      {np.mean(role_entropies):.3f}  (max={np.log(2):.3f})")
    print(f"    Stance:    {np.mean(stance_entropies):.3f}  (max={np.log(3):.3f})")
    print(f"    Objective: {np.mean(obj_entropies):.3f}  (max={np.log(5):.3f})")

    return {
        'role_entropy': np.mean(role_entropies),
        'stance_entropy': np.mean(stance_entropies),
        'obj_entropy': np.mean(obj_entropies),
    }


def main():
    # --- Random model (before training) ---
    random_model = _make_model("tactical")
    analyze_model(random_model, "RANDOM (untrained)")

    # --- Train for 30 batches ---
    print("\n" + "=" * 60)
    print("Training tactical model for 30 batches...")
    print("=" * 60)

    cfg = TrainingConfig(
        num_batches=30,
        batch_size=64,
        model_type="tactical",
        checkpoint_dir="ml_checkpoints_tactical_test",
        log_dir="ml_logs_tactical_test",
    )
    trained_model, metrics = run_training(config=cfg, verbose=True)

    # --- Trained model (after 30 batches) ---
    analyze_model(trained_model, "TRAINED (30 batches)")


if __name__ == "__main__":
    main()
