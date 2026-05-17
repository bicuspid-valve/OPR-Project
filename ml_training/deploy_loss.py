"""PPO loss over DeploymentRecord trajectories.

Stateless — call once per minibatch of records with the model and
terminal returns. The records' state_vec, masks, and old log-prob/value
are stored from collection time; we re-encode by re-running the model
forward pass and compute the standard PPO clip surrogate.

Designed so it can be folded into the main training loop later: the
returned dict has the same shape as the per-head loss breakdowns
compute_loss_flat already produces.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ml_training.config import DeploymentRecord


def compute_deploy_loss(
    model,
    records: list[DeploymentRecord],
    returns: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    normalize_advantages: bool = True,
    device: torch.device | None = None,
) -> dict:
    """One PPO update step over a batch of DeploymentRecord trajectories.

    Parameters
    ----------
    model : TacticalModel
    records : list of DeploymentRecord, all from one training iteration's
        rollouts. Order matches ``returns``.
    returns : 1D float tensor of length len(records). Typically the
        Monte-Carlo terminal-outcome return (+1 / -1 / 0) from the
        deploying player's perspective.
    clip_eps, value_coef, entropy_coef : standard PPO hyperparameters.
    normalize_advantages : centre/scale advantages across the batch.

    Returns
    -------
    dict with keys: total_loss, policy_loss, value_loss, entropy,
        approx_kl, clip_frac. total_loss has grad; the rest are floats.
    """
    if not records:
        raise ValueError("compute_deploy_loss called with empty records list")
    if device is None:
        device = next(model.parameters()).device

    B = len(records)
    state_vecs = torch.from_numpy(np.stack([r.state_vec for r in records])).to(device)
    eligible_masks = torch.from_numpy(np.stack([r.eligible_mask for r in records])).to(device)
    legal_pos_masks = torch.from_numpy(np.stack([r.legal_pos_mask for r in records])).to(device)
    unit_idx = torch.tensor([r.unit_idx for r in records], dtype=torch.long, device=device)
    pos_idx = torch.tensor([r.pos_idx for r in records], dtype=torch.long, device=device)
    old_log_prob = torch.tensor([r.old_log_prob for r in records],
                                dtype=torch.float32, device=device)
    old_value = torch.tensor([r.old_value for r in records],
                             dtype=torch.float32, device=device)
    opp_idx = torch.tensor([r.opponent_type_idx for r in records],
                           dtype=torch.long, device=device)
    side_idx = torch.tensor([r.side_idx for r in records],
                            dtype=torch.long, device=device)
    returns = returns.to(device).float()

    unit_logits, pos_logits, value = model.forward_deploy(
        state_vecs, eligible_masks, legal_pos_masks,
        opponent_type=opp_idx, side=side_idx,
    )

    unit_logp_all = F.log_softmax(unit_logits, dim=-1)
    pos_logp_all = F.log_softmax(pos_logits, dim=-1)
    unit_logp = unit_logp_all.gather(-1, unit_idx.unsqueeze(-1)).squeeze(-1)
    pos_logp = pos_logp_all.gather(-1, pos_idx.unsqueeze(-1)).squeeze(-1)
    new_log_prob = unit_logp + pos_logp

    # Advantage = return - V_old (single-step / terminal). Standardise.
    advantages = returns - old_value
    if normalize_advantages and B > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    value_loss = F.mse_loss(value, returns)

    # Entropy: sum across the two action dimensions, mean over batch.
    # Logits at masked slots are -inf so softmax gives p=0 there; the entropy
    # contribution is 0 (p·log p → 0). But computing `probs * logp` naively
    # produces 0 * -inf = NaN, which poisons the *gradient* even when .where
    # replaces the forward NaN. Replace -inf log-probs with 0 before the
    # multiplication — p is already 0 at those slots, so the product stays 0.
    unit_probs = unit_logp_all.exp()
    pos_probs = pos_logp_all.exp()
    unit_logp_safe = torch.where(
        torch.isfinite(unit_logp_all), unit_logp_all, torch.zeros_like(unit_logp_all),
    )
    pos_logp_safe = torch.where(
        torch.isfinite(pos_logp_all), pos_logp_all, torch.zeros_like(pos_logp_all),
    )
    unit_ent = -(unit_probs * unit_logp_safe).sum(dim=-1)
    pos_ent = -(pos_probs * pos_logp_safe).sum(dim=-1)
    entropy = (unit_ent + pos_ent).mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy

    with torch.no_grad():
        approx_kl = (-log_ratio).mean().item()
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

    return {
        "total_loss": total,
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }
