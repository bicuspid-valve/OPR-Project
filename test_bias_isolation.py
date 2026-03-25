"""Isolate the source of tactical SP-WR bias.

Test 1: A=sample, B=argmax (baseline, expect ~0.63)
Test 2: A=argmax, B=argmax (both deterministic)
Test 3: A=sample, B=sample (both stochastic)
Test 4: A=argmax, B=sample (reversed asymmetry)

If the bias is from sampling vs argmax:
  - Test 2 & 3 should be ~0.50
  - Test 4 should flip to ~0.37

If the bias is first-mover advantage:
  - All tests show A > 0.50 regardless of sampling mode
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import random
from ml_training import (
    _make_model, _generate_army_pair,
    _run_single_episode_tactical,
    sample_tactical_actions_no_grad,
)
from ml_integration_tactical import apply_tactical_model, ROLES, STANCES, remap_objective
from ml_features import MAX_UNITS_PER_SIDE
from board import OBJECTIVES


def apply_tactical_model_SAMPLING(
    model, friendly_units, enemy_units, round_num, board, player,
    **kwargs
):
    """Like apply_tactical_model but uses multinomial sampling instead of argmax."""
    from ml_features import encode_state_tactical

    alive_mask_list = []
    for i in range(MAX_UNITS_PER_SIDE):
        if i < len(friendly_units):
            us = friendly_units[i]
            alive_mask_list.append(us.models_alive > 0 and not us.activated)
        else:
            alive_mask_list.append(False)

    if not any(alive_mask_list):
        return None, [1.0] * MAX_UNITS_PER_SIDE, {}

    alive_mask = torch.tensor(alive_mask_list, dtype=torch.bool)

    state_vec = encode_state_tactical(
        friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=kwargs.get('friendly_ranged_matchups'),
        friendly_melee_matchups=kwargs.get('friendly_melee_matchups'),
        enemy_ranged_matchups=kwargs.get('enemy_ranged_matchups'),
        enemy_melee_matchups=kwargs.get('enemy_melee_matchups'),
        total_friendly_points=kwargs.get('total_friendly_points'),
        total_enemy_points=kwargs.get('total_enemy_points'),
    )

    with torch.no_grad():
        (unit_logits, role_probs, obj_probs, target_priority,
         combat_pref, stance_probs, _value) = model(state_vec, alive_mask)

    # Use sampling (same as Player A training path)
    unit_probs = torch.softmax(unit_logits, dim=-1)
    selected_idx = int(torch.multinomial(unit_probs, 1).item())
    selected_unit = friendly_units[selected_idx]

    target_multipliers = target_priority.tolist()

    role_idx = int(torch.multinomial(role_probs, 1).item())
    obj_idx = int(torch.multinomial(obj_probs, 1).item())
    combat_sample = int(torch.bernoulli(combat_pref).item())
    stance_idx = int(torch.multinomial(stance_probs, 1).item())

    selected_unit.ai_role = ROLES[role_idx]
    selected_unit.assigned_objective = remap_objective(obj_idx, player)
    selected_unit.combat_preference = "ranged" if combat_sample >= 1 else "melee"
    selected_unit.movement_stance = STANCES[stance_idx]

    return selected_unit, target_multipliers, {}


def argmax_tactical_actions_no_grad(
    unit_logits, role_probs, obj_probs, target_priority,
    combat_pref, stance_probs, alive_mask, friendly_units, player,
):
    """Like sample_tactical_actions_no_grad but uses argmax instead of sampling."""
    target_multipliers = target_priority.tolist()

    selected_unit_idx = int(unit_logits.argmax().item())
    role_idx = int(role_probs.argmax().item())
    obj_idx = int(obj_probs.argmax().item())
    combat_sample = 1 if combat_pref.item() >= 0.5 else 0
    stance_idx = int(stance_probs.argmax().item())

    # Dummy log prob (not used for evaluation)
    old_log_prob = 0.0

    us = friendly_units[selected_unit_idx]
    us.ai_role = ROLES[role_idx]
    us.assigned_objective = remap_objective(obj_idx, player)
    us.combat_preference = "ranged" if combat_sample >= 1 else "melee"
    us.movement_stance = STANCES[stance_idx]

    return target_multipliers, selected_unit_idx, role_idx, obj_idx, combat_sample, stance_idx, old_log_prob


def run_test(label, n_games, run_fn):
    wins = 0.0
    for i in range(n_games):
        result = run_fn()
        if result == "A":
            wins += 1
        elif result == "draw":
            wins += 0.5
        if (i + 1) % 50 == 0:
            print(f"  {label}: {i+1}/{n_games}, A-WR={wins/(i+1):.3f}")
    wr = wins / n_games
    print(f"  {label}: FINAL A-WR={wr:.3f}\n")
    return wr


def main():
    import ml_training
    import ml_integration_tactical

    N = 200

    tac_model = _make_model("tactical")
    tac_model.eval()

    # Save originals
    orig_sample = ml_training.sample_tactical_actions_no_grad
    orig_apply = ml_integration_tactical.apply_tactical_model

    # --- Test 1: Baseline (A=sample, B=argmax) ---
    print("=== Test 1: A=sample, B=argmax (baseline) ===")
    def game1():
        res_a, res_b, sa, sb, *_ = _generate_army_pair()
        sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sa]
        sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sb]
        _, result, _ = _run_single_episode_tactical(
            tac_model, tac_model, res_a, res_b, sa_data, sb_data, "selfplay", OBJECTIVES)
        return result
    wr1 = run_test("A=sample,B=argmax", N, game1)

    # --- Test 2: Both argmax (A=argmax, B=argmax) ---
    print("=== Test 2: A=argmax, B=argmax ===")
    ml_training.sample_tactical_actions_no_grad = argmax_tactical_actions_no_grad
    def game2():
        res_a, res_b, sa, sb, *_ = _generate_army_pair()
        sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sa]
        sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sb]
        _, result, _ = _run_single_episode_tactical(
            tac_model, tac_model, res_a, res_b, sa_data, sb_data, "selfplay", OBJECTIVES)
        return result
    wr2 = run_test("A=argmax,B=argmax", N, game2)
    ml_training.sample_tactical_actions_no_grad = orig_sample  # restore

    # --- Test 3: Both sample (A=sample, B=sample) ---
    print("=== Test 3: A=sample, B=sample ===")
    ml_integration_tactical.apply_tactical_model = apply_tactical_model_SAMPLING
    def game3():
        res_a, res_b, sa, sb, *_ = _generate_army_pair()
        sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sa]
        sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sb]
        _, result, _ = _run_single_episode_tactical(
            tac_model, tac_model, res_a, res_b, sa_data, sb_data, "selfplay", OBJECTIVES)
        return result
    wr3 = run_test("A=sample,B=sample", N, game3)
    ml_integration_tactical.apply_tactical_model = orig_apply  # restore

    # --- Test 4: Reversed (A=argmax, B=sample) ---
    print("=== Test 4: A=argmax, B=sample (reversed) ===")
    ml_training.sample_tactical_actions_no_grad = argmax_tactical_actions_no_grad
    ml_integration_tactical.apply_tactical_model = apply_tactical_model_SAMPLING
    def game4():
        res_a, res_b, sa, sb, *_ = _generate_army_pair()
        sa_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sa]
        sb_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in sb]
        _, result, _ = _run_single_episode_tactical(
            tac_model, tac_model, res_a, res_b, sa_data, sb_data, "selfplay", OBJECTIVES)
        return result
    wr4 = run_test("A=argmax,B=sample", N, game4)
    ml_training.sample_tactical_actions_no_grad = orig_sample
    ml_integration_tactical.apply_tactical_model = orig_apply

    print("=== Summary ===")
    print(f"  Test 1 - A=sample, B=argmax:  {wr1:.3f}")
    print(f"  Test 2 - A=argmax, B=argmax:  {wr2:.3f}")
    print(f"  Test 3 - A=sample, B=sample:  {wr3:.3f}")
    print(f"  Test 4 - A=argmax, B=sample:  {wr4:.3f}")
    print()
    if wr2 > 0.55 or wr3 > 0.55:
        print("  >> Bias persists even with symmetric sampling => first-mover or other structural advantage")
    elif abs(wr2 - 0.5) < 0.05 and abs(wr3 - 0.5) < 0.05:
        print("  >> Symmetric modes are ~0.50 => bias is from sampling vs argmax asymmetry")
    if wr4 < 0.45:
        print("  >> Reversed asymmetry flips bias => confirms sampling > argmax with random model")


if __name__ == "__main__":
    main()
