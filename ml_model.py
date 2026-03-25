"""ML strategic model: PyTorch network for round-level wargame decisions."""
from __future__ import annotations

import torch
import torch.nn as nn

from ml_features import TOTAL_FEATURES, MAX_UNITS_PER_SIDE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRUNK_HIDDEN_1 = 256
TRUNK_HIDDEN_2 = 128
NUM_ROLES = 2          # killer, objective_holder
NUM_OBJECTIVES = 5     # centre, my-side, enemy-side, my-home, enemy-home
NUM_STANCES = 3        # kite, normal, aggressive
N_FRIENDLY = MAX_UNITS_PER_SIDE  # 10


class StrategicModel(nn.Module):
    """Multi-head policy network for round-level strategic decisions.

    Input:  (batch, 591) encoded game state
    Output: tuple of 7 tensors (see forward() docstring)
    """

    def __init__(self) -> None:
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(TOTAL_FEATURES, TRUNK_HIDDEN_1),
            nn.ReLU(),
            nn.Linear(TRUNK_HIDDEN_1, TRUNK_HIDDEN_2),
            nn.ReLU(),
        )

        # Output heads
        self.role_head = nn.Linear(TRUNK_HIDDEN_2, N_FRIENDLY * NUM_ROLES)          # 20
        self.objective_head = nn.Linear(TRUNK_HIDDEN_2, N_FRIENDLY * NUM_OBJECTIVES) # 50
        self.target_priority_head = nn.Linear(TRUNK_HIDDEN_2, N_FRIENDLY)            # 10
        self.activation_priority_head = nn.Linear(TRUNK_HIDDEN_2, N_FRIENDLY)        # 10
        self.combat_preference_head = nn.Linear(TRUNK_HIDDEN_2, N_FRIENDLY)          # 10
        self.movement_stance_head = nn.Linear(TRUNK_HIDDEN_2, N_FRIENDLY * NUM_STANCES)  # 30
        self.value_head = nn.Linear(TRUNK_HIDDEN_2, 1)                               # 1

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,  # role_probs:       (batch, 10, 2)
        torch.Tensor,  # obj_probs:        (batch, 10, 5)
        torch.Tensor,  # target_priority:  (batch, 10)  — multipliers in [exp(-3), exp(3)]
        torch.Tensor,  # activation_priority: (batch, 10) — raw scalars
        torch.Tensor,  # combat_pref:      (batch, 10)  — sigmoid probabilities
        torch.Tensor,  # stance_probs:     (batch, 10, 3)
        torch.Tensor,  # value:            (batch,) — state value estimate
    ]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 591) or (591,)

        Returns
        -------
        Tuple of 7 tensors with applied activations.
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        h = self.trunk(x)
        batch = x.size(0)

        # Role: softmax over 2 roles per friendly unit
        role_logits = self.role_head(h).view(batch, N_FRIENDLY, NUM_ROLES)
        role_probs = torch.softmax(role_logits, dim=-1)

        # Objective: softmax over 5 objectives per friendly unit
        obj_logits = self.objective_head(h).view(batch, N_FRIENDLY, NUM_OBJECTIVES)
        obj_probs = torch.softmax(obj_logits, dim=-1)

        # Target priority: clamp → exp → multipliers
        tp_raw = self.target_priority_head(h)
        target_priority = torch.exp(torch.clamp(tp_raw, -3.0, 3.0))

        # Activation priority: raw scalars
        activation_priority = self.activation_priority_head(h)

        # Combat preference: sigmoid (≥0.5 = ranged, <0.5 = melee)
        combat_pref = torch.sigmoid(self.combat_preference_head(h))

        # Movement stance: softmax over 3 stances per friendly unit
        stance_logits = self.movement_stance_head(h).view(batch, N_FRIENDLY, NUM_STANCES)
        stance_probs = torch.softmax(stance_logits, dim=-1)

        # Value estimate: single scalar per batch element
        value = self.value_head(h).squeeze(-1)

        if single:
            return (
                role_probs.squeeze(0),
                obj_probs.squeeze(0),
                target_priority.squeeze(0),
                activation_priority.squeeze(0),
                combat_pref.squeeze(0),
                stance_probs.squeeze(0),
                value.squeeze(0),
            )

        return role_probs, obj_probs, target_priority, activation_priority, combat_pref, stance_probs, value
