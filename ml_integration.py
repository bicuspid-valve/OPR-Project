"""ML integration: wire model outputs to game state, replacement for heuristic calls in game.py."""
from __future__ import annotations

import torch

from board import Board
from models import UnitState
from ml_features import MAX_UNITS_PER_SIDE, encode_state, precompute_damage
from ml_model import StrategicModel

# ---------------------------------------------------------------------------
# Constants (§3.4, §3.7)
# ---------------------------------------------------------------------------

ROLES = ["killer", "objective_holder"]
STANCES = ["kite", "normal", "aggressive"]

# Objective index remapping: model perspective → game perspective
# Player A: identity.  Player B: swap 1↔2 (A-side↔B-side), 3↔4 (Home-A↔Home-B)
_OBJ_REMAP_B = [0, 2, 1, 4, 3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remap_objective(model_obj_idx: int, player: str) -> int:
    """Convert model-perspective objective index to game-perspective index."""
    if player == "A":
        return model_obj_idx
    return _OBJ_REMAP_B[model_obj_idx]


def ml_activation_order(units: list[UnitState]) -> list[UnitState]:
    """Return unactivated alive units sorted by stored ML activation scores (descending)."""
    active = [u for u in units if u.models_alive > 0 and not u.activated]
    active.sort(key=lambda u: getattr(u, '_ml_activation_score', 0.0), reverse=True)
    return active


# ---------------------------------------------------------------------------
# Main integration entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_model_outputs(
    model: StrategicModel,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    *,
    friendly_ranged_matchups: list[list[list[float]]] | None = None,
    friendly_melee_matchups: list[list[float]] | None = None,
    enemy_ranged_matchups: list[list[list[float]]] | None = None,
    enemy_melee_matchups: list[list[float]] | None = None,
    total_friendly_points: int | None = None,
    total_enemy_points: int | None = None,
) -> list[float]:
    """Encode state, run ML forward pass, and write outputs to UnitState objects.

    Sets ai_role, assigned_objective, combat_preference, movement_stance,
    and _ml_activation_score on each living friendly unit.

    Returns target_multipliers — a list of 10 floats (one per enemy slot)
    for use by pick_target this round.
    """
    state_vec = encode_state(
        friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=friendly_ranged_matchups,
        friendly_melee_matchups=friendly_melee_matchups,
        enemy_ranged_matchups=enemy_ranged_matchups,
        enemy_melee_matchups=enemy_melee_matchups,
        total_friendly_points=total_friendly_points,
        total_enemy_points=total_enemy_points,
    )

    role_probs, obj_probs, target_priority, act_scores, combat_prefs, stance_probs, _value = model(state_vec)

    # Target multipliers (already exp(clamp(raw, -3, 3)) from model forward)
    target_multipliers = target_priority.tolist()

    # Write per-unit decisions to game state
    for i, us in enumerate(friendly_units):
        if i >= MAX_UNITS_PER_SIDE:
            break
        if us.models_alive <= 0:
            continue

        # Role: argmax → "killer" or "objective_holder"
        us.ai_role = ROLES[int(role_probs[i].argmax().item())]

        # Objective: argmax, remapped from model perspective to game perspective
        us.assigned_objective = remap_objective(int(obj_probs[i].argmax().item()), player)

        # Combat preference: sigmoid ≥ 0.5 → ranged
        us.combat_preference = "ranged" if combat_prefs[i].item() >= 0.5 else "melee"

        # Movement stance: argmax → "kite" / "normal" / "aggressive"
        us.movement_stance = STANCES[int(stance_probs[i].argmax().item())]

        # Activation priority score (used by ml_activation_order)
        us._ml_activation_score = act_scores[i].item()

    # Build assessment summary for viewer / diagnostics
    assessment: dict = {
        'value': _value.item(),
        'units': [],
    }
    for i, us in enumerate(friendly_units):
        if i >= MAX_UNITS_PER_SIDE:
            break
        if us.models_alive <= 0:
            continue
        role_idx = int(role_probs[i].argmax().item())
        obj_idx = int(obj_probs[i].argmax().item())
        stance_idx = int(stance_probs[i].argmax().item())
        assessment['units'].append({
            'slot': i,
            'name': us.unit.name,
            'role': ROLES[role_idx],
            'role_confidence': role_probs[i][role_idx].item(),
            'role_probs': role_probs[i].tolist(),
            'objective': remap_objective(obj_idx, player),
            'objective_confidence': obj_probs[i][obj_idx].item(),
            'objective_probs': obj_probs[i].tolist(),
            'combat_preference': "ranged" if combat_prefs[i].item() >= 0.5 else "melee",
            'combat_pref_prob': combat_prefs[i].item(),
            'stance': STANCES[stance_idx],
            'stance_confidence': stance_probs[i][stance_idx].item(),
            'stance_probs': stance_probs[i].tolist(),
            'target_priority': target_priority[i].item(),
            'activation_score': act_scores[i].item(),
        })

    return target_multipliers, assessment


# ---------------------------------------------------------------------------
# Sampling variant (for evaluation with stochastic action selection)
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_model_outputs_sampling(
    model: StrategicModel,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    *,
    friendly_ranged_matchups: list[list[list[float]]] | None = None,
    friendly_melee_matchups: list[list[float]] | None = None,
    enemy_ranged_matchups: list[list[list[float]]] | None = None,
    enemy_melee_matchups: list[list[float]] | None = None,
    total_friendly_points: int | None = None,
    total_enemy_points: int | None = None,
) -> list[float]:
    """Like apply_model_outputs but samples from distributions instead of argmax."""
    state_vec = encode_state(
        friendly_units, enemy_units, round_num, board, player,
        friendly_ranged_matchups=friendly_ranged_matchups,
        friendly_melee_matchups=friendly_melee_matchups,
        enemy_ranged_matchups=enemy_ranged_matchups,
        enemy_melee_matchups=enemy_melee_matchups,
        total_friendly_points=total_friendly_points,
        total_enemy_points=total_enemy_points,
    )

    role_probs, obj_probs, target_priority, act_scores, combat_prefs, stance_probs, _value = model(state_vec)

    target_multipliers = target_priority.tolist()

    for i, us in enumerate(friendly_units):
        if i >= MAX_UNITS_PER_SIDE:
            break
        if us.models_alive <= 0:
            continue

        # Sample from distributions instead of argmax
        us.ai_role = ROLES[int(torch.multinomial(role_probs[i], 1).item())]
        us.assigned_objective = remap_objective(int(torch.multinomial(obj_probs[i], 1).item()), player)
        us.combat_preference = "ranged" if int(torch.bernoulli(combat_prefs[i]).item()) else "melee"
        us.movement_stance = STANCES[int(torch.multinomial(stance_probs[i], 1).item())]
        us._ml_activation_score = act_scores[i].item()

    return target_multipliers, {}
