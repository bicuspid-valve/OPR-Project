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
        # Destination pointer: target is a fraction of max entropy (normalised by ln(N_valid))
        self.target_dest_fraction = config.entropy_target_dest_fraction
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
        is_move: torch.Tensor,       # (N,) bool — destination active (non-charge, non-shaken)
        is_can_shoot: torch.Tensor,  # (N,) bool — shoot active (advance-reachable dest)
        is_charge: torch.Tensor,     # (N,) bool — charge active
    ) -> torch.Tensor:
        """Compute the weighted entropy bonus for the policy loss.

        Returns a scalar: sum of alpha_i * mean_entropy_i across active heads.
        """
        # Detach alphas so the policy loss gradient doesn't flow through them —
        # only the separate alpha_loss should update the alpha parameters.
        bonus = self.get_alpha("unit").detach() * unit_ent.mean()
        bonus = bonus + self.get_alpha("move").detach() * move_ent.mean()

        # Destination: active for move (non-charge, non-shaken)
        n_move = is_move.sum().clamp(min=1)
        bonus = bonus + self.get_alpha("dest").detach() * (dest_ent * is_move).sum() / n_move

        # Charge: only active for charge moves
        n_charge = is_charge.sum().clamp(min=1)
        bonus = bonus + self.get_alpha("charge").detach() * (charge_ent * is_charge).sum() / n_charge

        # Shoot: only active for advance-reachable destinations
        n_can_shoot = is_can_shoot.sum().clamp(min=1)
        bonus = bonus + self.get_alpha("shoot").detach() * (shoot_ent * is_can_shoot).sum() / n_can_shoot

        return bonus

    def compute_alpha_loss(
        self,
        unit_ent: torch.Tensor,       # (N,)
        move_ent: torch.Tensor,       # (N,)
        dest_ent: torch.Tensor,       # (N,)
        charge_ent: torch.Tensor,     # (N,)
        shoot_ent: torch.Tensor,      # (N,)
        is_move: torch.Tensor,        # (N,) bool — dest active (non-charge, non-shaken)
        is_can_shoot: torch.Tensor,   # (N,) bool — shoot active (advance-reachable dest)
        is_charge: torch.Tensor,      # (N,) bool
        alive_mask: torch.Tensor,     # (N, 10) — for unit target
        enemy_alive_mask: torch.Tensor,  # (N, 10) — for charge target
        shoot_mask: torch.Tensor,     # (N, 10) — for shoot target
        dest_n_valid: torch.Tensor | None = None,  # (N,) int — number of valid dest candidates
    ) -> torch.Tensor:
        """Compute the dual alpha loss that drives entropy toward targets.

        All entropy values are detached so the alpha loss doesn't affect the policy.
        """
        loss = torch.tensor(0.0)

        # Unit selection: target = fraction * ln(num_alive)
        n_alive = alive_mask.sum(dim=-1).clamp(min=1).float()
        unit_target = self.target_fraction * torch.log(n_alive)
        loss = loss + self.get_alpha("unit") * (unit_ent.detach() - unit_target).mean()

        # Move type: fixed target (2-way: move/charge)
        loss = loss + self.get_alpha("move") * (move_ent.detach() - self.target_move).mean()

        # Destination pointer: normalised entropy = dest_ent / ln(N_valid)
        # Target is target_dest_fraction (e.g. 0.25 = 25% of max entropy)
        if is_move.any() and dest_n_valid is not None:
            n_mv = is_move.sum().clamp(min=1)
            ln_n_valid = torch.log(dest_n_valid.float().clamp(min=1.0))  # (N,)
            normalised_ent = dest_ent.detach() / ln_n_valid.clamp(min=1.0)  # (N,)
            mean_norm_ent = (normalised_ent * is_move).sum() / n_mv
            loss = loss + self.get_alpha("dest") * (mean_norm_ent - self.target_dest_fraction)

        # Charge target: target = fraction * ln(num_enemy_alive), only for charge steps
        if is_charge.any():
            n_enemy_alive = enemy_alive_mask.sum(dim=-1).clamp(min=1).float()
            charge_target = self.target_fraction * torch.log(n_enemy_alive)
            n_ch = is_charge.sum().clamp(min=1)
            mean_charge_ent = (charge_ent.detach() * is_charge).sum() / n_ch
            mean_charge_target = (charge_target * is_charge).sum() / n_ch
            loss = loss + self.get_alpha("charge") * (mean_charge_ent - mean_charge_target)

        # Shoot target: target = fraction * ln(num_in_range), only for advance-reachable dest
        if is_can_shoot.any():
            n_shootable = shoot_mask.sum(dim=-1).clamp(min=1).float()
            shoot_target = self.target_fraction * torch.log(n_shootable)
            n_cs = is_can_shoot.sum().clamp(min=1)
            mean_shoot_ent = (shoot_ent.detach() * is_can_shoot).sum() / n_cs
            mean_shoot_target = (shoot_target * is_can_shoot).sum() / n_cs
            loss = loss + self.get_alpha("shoot") * (mean_shoot_ent - mean_shoot_target)

        return loss

    def alpha_summary(self) -> dict[str, float]:
        """Return current alpha values for logging."""
        return {name: self.log_alphas[name].exp().item() for name in self.HEAD_NAMES}
