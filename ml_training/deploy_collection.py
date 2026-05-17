"""Model-controlled deployment: decision_fn factory + trajectory recording.

The returned callable plugs into ``game.deploy_armies`` via the
``decision_fn_a`` / ``decision_fn_b`` keyword args. It encodes the mid-
deployment state, calls ``TacticalModel.forward_deploy``, samples a
(unit, anchor) pair, and optionally appends a DeploymentRecord to a
caller-supplied list for PPO training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from game import DeployContext, _collect_enemy_positions
from ml_features import (
    encode_state_deploy,
    build_deploy_eligible_mask,
    build_deploy_legal_pos_mask,
    deploy_pos_idx_to_world,
    DEPLOY_POS_GRID,
)
from ml_training.config import DeploymentRecord


def make_model_deploy_decision_fn(
    model,
    *,
    player: str,
    opponent_type_idx: int = 0,
    side_idx: int = 0,
    record_into: list[DeploymentRecord] | None = None,
    deterministic: bool = False,
):
    """Return a callable usable as ``deploy_armies``'s decision_fn_a/b.

    Parameters
    ----------
    model : TacticalModel — must have forward_deploy.
    player : "A" or "B" — the deploying player. Drives egocentric flipping.
    opponent_type_idx, side_idx : value-head conditioning indices.
    record_into : if given, the function appends one DeploymentRecord per
        call so the trainer can later compute the PPO loss.
    deterministic : argmax over masked logits when True; sample otherwise.

    The returned callable is stateless w.r.t. the model parameters — each
    call does its own forward pass and re-encodes the (now updated) state.
    """
    model_device = next(model.parameters()).device

    def decision_fn(ctx: DeployContext):
        # Friendly slot ordering: eligible units FIRST so they always occupy
        # slots [0, len(eligible)) — within MAX_UNITS_PER_SIDE for any
        # reasonable army size. The legacy "placed first" ordering broke
        # whenever len(my_placed) reached MAX_UNITS_PER_SIDE: eligible units
        # got pushed past slot 10 and were sliced out, leaving the model
        # with no slot to choose. Placed and other-phase units still need
        # to appear in the state vector (the model uses their positions as
        # context for the next placement), they just take the trailing slots.
        all_friendly = ctx.eligible_units + ctx.my_placed + ctx.my_unplaced_other
        # build_deploy_eligible_mask iterates the first MAX_UNITS_PER_SIDE
        # entries and marks those unplaced + matching phase + alive. With
        # eligible-first ordering, the mask is True for slots [0,
        # min(len(eligible_units), MAX_UNITS_PER_SIDE)). For armies > 10
        # units that overflow into slot 10+, the trailing eligibles are
        # invisible to the model this turn — they fit in the window once
        # earlier picks shrink the list, so the loop still terminates.
        ord_eligible = build_deploy_eligible_mask(all_friendly, ctx.phase)

        all_opp = ctx.opp_placed + ctx.opp_unplaced

        x = encode_state_deploy(
            friendly_units=all_friendly,
            enemy_units=all_opp,
            board=ctx.board,
            player=player,
            deploy_phase=ctx.phase,
            is_my_turn=True,
        ).to(model_device)

        enemy_positions = _collect_enemy_positions(all_opp)
        legal_pos_mask = build_deploy_legal_pos_mask(
            player, ctx.phase, ctx.board, enemy_positions=enemy_positions,
        ).to(model_device)
        ord_eligible = ord_eligible.to(model_device)

        if not ord_eligible.any():
            raise RuntimeError("deploy decision_fn called with no eligible units")
        if not legal_pos_mask.any():
            raise RuntimeError("deploy decision_fn: no legal anchor cells")

        with torch.no_grad():
            unit_logits, pos_logits, value = model.forward_deploy(
                x, ord_eligible, legal_pos_mask,
                opponent_type=opponent_type_idx, side=side_idx,
            )

        unit_logp = F.log_softmax(unit_logits, dim=-1)
        pos_logp = F.log_softmax(pos_logits, dim=-1)
        if deterministic:
            unit_idx = int(torch.argmax(unit_logits).item())
            pos_idx = int(torch.argmax(pos_logits).item())
        else:
            unit_idx = int(torch.distributions.Categorical(logits=unit_logits).sample().item())
            pos_idx = int(torch.distributions.Categorical(logits=pos_logits).sample().item())

        log_prob = float(unit_logp[unit_idx].item() + pos_logp[pos_idx].item())

        chosen_unit = all_friendly[unit_idx]
        col_world, row_world = deploy_pos_idx_to_world(pos_idx, player)

        if record_into is not None:
            record_into.append(DeploymentRecord(
                state_vec=x.detach().cpu().numpy().copy(),
                phase=ctx.phase,
                eligible_mask=ord_eligible.detach().cpu().numpy().copy(),
                legal_pos_mask=legal_pos_mask.detach().cpu().numpy().copy(),
                unit_idx=unit_idx,
                pos_idx=pos_idx,
                old_log_prob=log_prob,
                old_value=float(value.item()),
                opponent_type_idx=opponent_type_idx,
                side_idx=side_idx,
            ))

        return chosen_unit, col_world, row_world

    return decision_fn
