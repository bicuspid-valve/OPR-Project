"""Per-head entropy target tuner (SAC-style adaptive entropy)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ml_training.config import TrainingConfig


class EntropyTargetTuner(nn.Module):
    """Maintains learnable log-alpha per policy head for adaptive entropy.

    Each head has an independent coefficient alpha_i = exp(log_alpha_i).
    The alpha loss drives entropy toward per-head targets:
        alpha_loss_i = -alpha_i * (entropy_i - target_i)

    For masked categorical heads (unit, charge, shoot), the target is
    computed dynamically as fraction * ln(num_valid_actions).
    """

    # Head names for logging/serialization
    HEAD_NAMES = ("unit", "move", "dest", "charge", "shoot")

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        # One log-alpha per head (initialized to ~0.01 effective alpha)
        init_val = math.log(0.01)
        self.log_alphas = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(init_val))
            for name in self.HEAD_NAMES
        })
        # Fixed targets for non-masked heads
        self.target_move = config.entropy_target_move
        self.target_dest = config.entropy_target_dest
        # Fraction for dynamic masked-categorical targets
        self.target_fraction = config.entropy_target_fraction

    # Alpha bounds: prevent runaway entropy bonus
    LOG_ALPHA_MIN = math.log(0.001)  # alpha >= 0.001
    LOG_ALPHA_MAX = math.log(0.1)    # alpha <= 0.1

    def get_alpha(self, head: str) -> torch.Tensor:
        """Return the positive alpha coefficient for a head."""
        clamped = self.log_alphas[head].clamp(self.LOG_ALPHA_MIN, self.LOG_ALPHA_MAX)
        return clamped.exp()

    def compute_entropy_bonus(
        self,
        unit_ent: torch.Tensor,      # (N,)
        move_ent: torch.Tensor,      # (N,)
        dest_ent: torch.Tensor,      # (N,)
        charge_ent: torch.Tensor,    # (N,)
        shoot_ent: torch.Tensor,     # (N,)
        is_adv_rush: torch.Tensor,   # (N,) bool — destination active
        is_hold_adv: torch.Tensor,   # (N,) bool — shoot active
        is_charge: torch.Tensor,     # (N,) bool — charge active
    ) -> torch.Tensor:
        """Compute the weighted entropy bonus for the policy loss.

        Returns a scalar: sum of alpha_i * mean_entropy_i across active heads.
        """
        # Detach alphas so the policy loss gradient doesn't flow through them —
        # only the separate alpha_loss should update the alpha parameters.
        bonus = self.get_alpha("unit").detach() * unit_ent.mean()
        bonus = bonus + self.get_alpha("move").detach() * move_ent.mean()

        # Destination: only active for advance/rush
        n_adv_rush = is_adv_rush.sum().clamp(min=1)
        bonus = bonus + self.get_alpha("dest").detach() * (dest_ent * is_adv_rush).sum() / n_adv_rush

        # Charge: only active for charge moves
        n_charge = is_charge.sum().clamp(min=1)
        bonus = bonus + self.get_alpha("charge").detach() * (charge_ent * is_charge).sum() / n_charge

        # Shoot: only active for hold/advance
        n_hold_adv = is_hold_adv.sum().clamp(min=1)
        bonus = bonus + self.get_alpha("shoot").detach() * (shoot_ent * is_hold_adv).sum() / n_hold_adv

        return bonus

    def compute_alpha_loss(
        self,
        unit_ent: torch.Tensor,       # (N,)
        move_ent: torch.Tensor,       # (N,)
        dest_ent: torch.Tensor,       # (N,)
        charge_ent: torch.Tensor,     # (N,)
        shoot_ent: torch.Tensor,      # (N,)
        is_adv_rush: torch.Tensor,    # (N,) bool
        is_hold_adv: torch.Tensor,    # (N,) bool
        is_charge: torch.Tensor,      # (N,) bool
        alive_mask: torch.Tensor,     # (N, 10) — for unit target
        enemy_alive_mask: torch.Tensor,  # (N, 10) — for charge target
        shoot_mask: torch.Tensor,     # (N, 10) — for shoot target
    ) -> torch.Tensor:
        """Compute the dual alpha loss that drives entropy toward targets.

        All entropy values are detached so the alpha loss doesn't affect the policy.
        """
        loss = torch.tensor(0.0)

        # Unit selection: target = fraction * ln(num_alive)
        n_alive = alive_mask.sum(dim=-1).clamp(min=1).float()
        unit_target = self.target_fraction * torch.log(n_alive)
        loss = loss + self.get_alpha("unit") * (unit_ent.detach() - unit_target).mean()

        # Move type: fixed target
        loss = loss + self.get_alpha("move") * (move_ent.detach() - self.target_move).mean()

        # Destination: fixed target, only for advance/rush steps
        if is_adv_rush.any():
            n_ar = is_adv_rush.sum().clamp(min=1)
            mean_dest_ent = (dest_ent.detach() * is_adv_rush).sum() / n_ar
            loss = loss + self.get_alpha("dest") * (mean_dest_ent - self.target_dest)

        # Charge target: target = fraction * ln(num_enemy_alive), only for charge steps
        if is_charge.any():
            n_enemy_alive = enemy_alive_mask.sum(dim=-1).clamp(min=1).float()
            charge_target = self.target_fraction * torch.log(n_enemy_alive)
            n_ch = is_charge.sum().clamp(min=1)
            mean_charge_ent = (charge_ent.detach() * is_charge).sum() / n_ch
            mean_charge_target = (charge_target * is_charge).sum() / n_ch
            loss = loss + self.get_alpha("charge") * (mean_charge_ent - mean_charge_target)

        # Shoot target: target = fraction * ln(num_in_range), only for hold/advance steps
        if is_hold_adv.any():
            n_shootable = shoot_mask.sum(dim=-1).clamp(min=1).float()
            shoot_target = self.target_fraction * torch.log(n_shootable)
            n_ha = is_hold_adv.sum().clamp(min=1)
            mean_shoot_ent = (shoot_ent.detach() * is_hold_adv).sum() / n_ha
            mean_shoot_target = (shoot_target * is_hold_adv).sum() / n_ha
            loss = loss + self.get_alpha("shoot") * (mean_shoot_ent - mean_shoot_target)

        return loss

    def alpha_summary(self) -> dict[str, float]:
        """Return current alpha values for logging."""
        return {name: self.log_alphas[name].exp().item() for name in self.HEAD_NAMES}
