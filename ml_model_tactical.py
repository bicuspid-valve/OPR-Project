"""ML tactical model v2: PyTorch network for per-activation wargame decisions.

Trunk architecture: self-attention over 20 unit embeddings (180-dim, 5 heads),
then flatten + compress → 1024 → 512 + 2 residual layers.

Sequential head chain with free movement:
    h                            → unit_selection_head   → (10,)
    h ++ unit_feat(180)          → move_type_head        → (4,) hold/advance/rush/charge
    h ++ unit_feat ++ move(4)    → direction_head        → (3,) sin, cos, log_conc
                                 → distance_head         → (2,) alpha_raw, beta_raw
    h ++ unit_feat ++ move(4)    → charge_target_head    → (10,)
    h ++ unit_feat ++ move(4)
      ++ post_move_rel(30)       → shoot_target_head     → (10,)

unit_feat is the attended embedding for the selected unit (180-dim, enriched by
cross-unit attention) rather than raw input features.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_features import (
    TACTICAL_TOTAL_FEATURES, MAX_UNITS_PER_SIDE, TACTICAL_UNIT_FEATURES,
    GLOBAL_FEATURES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRUNK_STEM = 1024
TRUNK_WIDTH = 512
N_FRIENDLY = MAX_UNITS_PER_SIDE   # 10
N_ENEMY = MAX_UNITS_PER_SIDE      # 10
N_TOTAL_UNITS = N_FRIENDLY + N_ENEMY  # 20
NUM_MOVE_TYPES = 4                # 0=hold, 1=advance, 2=rush, 3=charge
POST_MOVE_REL_FEATURES = N_ENEMY * 3  # 30: (sin θ, cos θ, dist) per enemy from post-move pos
ATTN_DIM = TACTICAL_UNIT_FEATURES    # 180 — attend over raw feature dim
ATTN_HEADS = 5                       # 180 / 5 = 36 dims per head

# Movement type indices
MOVE_HOLD = 0
MOVE_ADVANCE = 1
MOVE_RUSH = 2
MOVE_CHARGE = 3


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TacticalModelOutput:
    unit_logits: torch.Tensor | None      # (10,) raw logits, masked; None from forward_per_unit
    move_logits: torch.Tensor             # (4,) hold/advance/rush/charge
    direction_params: torch.Tensor        # (3,) mean_sin, mean_cos, log_concentration
    distance_params: torch.Tensor         # (2,) alpha_raw, beta_raw
    charge_target_logits: torch.Tensor    # (10,) masked by enemy_alive_mask
    shoot_target_logits: torch.Tensor     # (10,) masked by enemy_alive + in range
    value: torch.Tensor                   # scalar


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TacticalModel(nn.Module):
    """Sequential-conditioned policy network for per-activation tactical decisions.

    Input:  (batch, TACTICAL_TOTAL_FEATURES) encoded game state
    Output: TacticalModelOutput

    Trunk: self-attention over 20 units (180-dim, 5 heads) → flatten →
           1024 → 512 + 2 residual layers.
    All heads always produce outputs (no conditional execution during forward).
    The integration layer decides which outputs to use based on move_type.
    """

    def __init__(self) -> None:
        super().__init__()

        # Self-attention over 20 unit embeddings (180-dim each)
        self.attn = nn.MultiheadAttention(
            embed_dim=ATTN_DIM, num_heads=ATTN_HEADS, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(ATTN_DIM)

        # Stem: flatten attended units + global → compress to trunk width
        attn_flat = N_TOTAL_UNITS * ATTN_DIM + GLOBAL_FEATURES  # 20*180 + 11 = 3611
        self.stem = nn.Sequential(
            nn.Linear(attn_flat, TRUNK_STEM),
            nn.LayerNorm(TRUNK_STEM),
            nn.ReLU(),
            nn.Linear(TRUNK_STEM, TRUNK_WIDTH),
            nn.LayerNorm(TRUNK_WIDTH),
            nn.ReLU(),
        )

        # Residual blocks (512 → 512 each)
        self.res_block_1 = nn.Sequential(
            nn.Linear(TRUNK_WIDTH, TRUNK_WIDTH),
            nn.LayerNorm(TRUNK_WIDTH),
            nn.ReLU(),
        )
        self.res_block_2 = nn.Sequential(
            nn.Linear(TRUNK_WIDTH, TRUNK_WIDTH),
            nn.LayerNorm(TRUNK_WIDTH),
            nn.ReLU(),
        )

        H = TRUNK_WIDTH  # 512
        UF = TACTICAL_UNIT_FEATURES  # 180

        # 1) Unit selection: h → 10 logits
        self.unit_selection_head = nn.Linear(H, N_FRIENDLY)

        # 2) Movement type: h + unit_feat → 4 logits
        self.move_type_head = nn.Linear(H + UF, NUM_MOVE_TYPES)

        # 3a) Direction: h + unit_feat + move_onehot → 3 (mean_sin, mean_cos, log_conc)
        self.direction_head = nn.Linear(H + UF + NUM_MOVE_TYPES, 3)

        # 3b) Distance: h + unit_feat + move_onehot → 2 (alpha_raw, beta_raw)
        self.distance_head = nn.Linear(H + UF + NUM_MOVE_TYPES, 2)

        # 3c) Charge target: h + unit_feat + move_onehot → 10 logits
        self.charge_target_head = nn.Linear(H + UF + NUM_MOVE_TYPES, N_ENEMY)

        # 4) Shooting target: h + unit_feat + move_onehot + post_move_rel → 10 logits
        self.shoot_target_head = nn.Linear(H + UF + NUM_MOVE_TYPES + POST_MOVE_REL_FEATURES, N_ENEMY)

        # Value head: h → scalar (not conditioned on actions)
        self.value_head = nn.Linear(H, 1)

        # Auxiliary prediction heads (trained with supervised targets, not used at inference)
        # Friendly survival: Beta(α, β) per friendly unit — 2 raw params each
        self.aux_friendly_survival_head = nn.Linear(H, N_FRIENDLY * 2)
        # Enemy survival: Beta(α, β) per enemy unit — 2 raw params each
        self.aux_enemy_survival_head = nn.Linear(H, N_ENEMY * 2)
        # Objective control: 5 objectives × 3 classes (friendly/enemy/neutral)
        self.aux_obj_control_head = nn.Linear(H, 5 * 3)

    def trunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run input through attention + stem + residual blocks.

        Returns
        -------
        h : (batch, 512) — global trunk representation
        u_attended : (batch, 20, 180) — per-unit attended embeddings
        """
        # Reshape flat input into unit blocks + global features
        unit_block = x[..., :N_TOTAL_UNITS * TACTICAL_UNIT_FEATURES]
        glob = x[..., N_TOTAL_UNITS * TACTICAL_UNIT_FEATURES:]  # (batch, 11)

        # (batch, 20, 180)
        units = unit_block.reshape(*x.shape[:-1], N_TOTAL_UNITS, ATTN_DIM)

        # Self-attention with residual + LayerNorm
        u_attn, _ = self.attn(units, units, units)
        u_attended = self.attn_norm(units + u_attn)  # (batch, 20, 180)

        # Flatten attended units + global → stem → residual blocks
        flat = torch.cat([u_attended.flatten(-2), glob], dim=-1)
        h = self.stem(flat)
        h = h + self.res_block_1(h)
        h = h + self.res_block_2(h)
        return h, u_attended

    def _extract_unit_features(self, u_attended: torch.Tensor, unit_idx: int) -> torch.Tensor:
        """Extract the attended embedding for friendly unit *unit_idx*.

        Parameters
        ----------
        u_attended : (batch, 20, 180) or (20, 180) — attended unit embeddings
        unit_idx : friendly unit slot (0–9)

        Works for both single and batched inputs.
        """
        return u_attended[..., unit_idx, :]

    def _run_conditioned_heads(
        self,
        h: torch.Tensor,
        unit_features: torch.Tensor,
        enemy_alive_mask: torch.Tensor | None,
        post_move_rel: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run move_type → direction/distance/charge_target → shoot_target chain.

        All heads run unconditionally; the caller decides which outputs matter
        based on the chosen move type.

        Parameters
        ----------
        post_move_rel : (batch, 30) or (30,) — post-move relative features.
            If None, zeros are used (argmax forward pass computes this externally).

        Returns (move_logits, direction_params, distance_params,
                 charge_target_logits, shoot_target_logits).
        """
        unit_features = unit_features.detach()

        # --- Movement type head ---
        h_uf = torch.cat([h, unit_features], dim=-1)
        move_logits = self.move_type_head(h_uf)

        # Conditioning: one-hot of argmax move type
        move_onehot = F.one_hot(
            move_logits.detach().argmax(dim=-1), NUM_MOVE_TYPES
        ).float()

        # --- Common input for direction/distance/charge ---
        h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)

        # --- Direction head ---
        direction_params = self.direction_head(h_uf_m)

        # --- Distance head ---
        distance_params = self.distance_head(h_uf_m)

        # --- Charge target head ---
        charge_target_logits = self.charge_target_head(h_uf_m)
        if enemy_alive_mask is not None:
            charge_target_logits = charge_target_logits.masked_fill(~enemy_alive_mask, float('-inf'))

        # --- Shooting target head ---
        if post_move_rel is None:
            post_move_rel = torch.zeros(
                *h.shape[:-1], POST_MOVE_REL_FEATURES,
                device=h.device, dtype=h.dtype,
            )
        shoot_input = torch.cat([h, unit_features, move_onehot, post_move_rel], dim=-1)
        shoot_target_logits = self.shoot_target_head(shoot_input)
        if enemy_alive_mask is not None:
            shoot_target_logits = shoot_target_logits.masked_fill(~enemy_alive_mask, float('-inf'))

        return move_logits, direction_params, distance_params, charge_target_logits, shoot_target_logits

    def forward(
        self,
        x: torch.Tensor,
        alive_mask: torch.Tensor | None = None,
        enemy_alive_mask: torch.Tensor | None = None,
        *,
        forced_unit_idx: int | None = None,
        post_move_rel: torch.Tensor | None = None,
    ) -> TacticalModelOutput:
        """Forward pass with sequential conditioning.

        Parameters
        ----------
        x : (batch, TACTICAL_TOTAL_FEATURES) or (TACTICAL_TOTAL_FEATURES,)
        alive_mask : bool, (batch, 10) or (10,) — friendly alive+unactivated
        enemy_alive_mask : bool, (batch, 10) or (10,) — enemy alive
        forced_unit_idx : if set, skip unit selection and use this index
        post_move_rel : (batch, 30) or (30,) — post-move relative features for shoot head.
            If None, zeros are used (caller should compute externally for argmax mode).
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
            if alive_mask is not None and alive_mask.dim() == 1:
                alive_mask = alive_mask.unsqueeze(0)
            if enemy_alive_mask is not None and enemy_alive_mask.dim() == 1:
                enemy_alive_mask = enemy_alive_mask.unsqueeze(0)
            if post_move_rel is not None and post_move_rel.dim() == 1:
                post_move_rel = post_move_rel.unsqueeze(0)

        h, u_attended = self.trunk(x)  # (batch, 512), (batch, 20, 180)

        # --- Unit selection ---
        unit_logits = self.unit_selection_head(h)  # (batch, 10)
        if alive_mask is not None:
            unit_logits = unit_logits.masked_fill(~alive_mask, float('-inf'))

        if forced_unit_idx is not None:
            chosen_idx = forced_unit_idx
        else:
            chosen_idx = unit_logits.detach()[0].argmax(dim=-1).item()

        unit_features = self._extract_unit_features(u_attended, chosen_idx)

        # --- Conditioned head chain ---
        move_logits, direction_params, distance_params, charge_logits, shoot_logits = (
            self._run_conditioned_heads(h, unit_features, enemy_alive_mask, post_move_rel)
        )

        # --- Value ---
        value = self.value_head(h).squeeze(-1)

        if single:
            return TacticalModelOutput(
                unit_logits=unit_logits.squeeze(0),
                move_logits=move_logits.squeeze(0),
                direction_params=direction_params.squeeze(0),
                distance_params=distance_params.squeeze(0),
                charge_target_logits=charge_logits.squeeze(0),
                shoot_target_logits=shoot_logits.squeeze(0),
                value=value.squeeze(0),
            )

        return TacticalModelOutput(
            unit_logits=unit_logits,
            move_logits=move_logits,
            direction_params=direction_params,
            distance_params=distance_params,
            charge_target_logits=charge_logits,
            shoot_target_logits=shoot_logits,
            value=value,
        )

    def forward_per_unit(
        self,
        h: torch.Tensor,
        u_attended: torch.Tensor,
        unit_indices: list[int],
        enemy_alive_mask: torch.Tensor,
        post_move_rel: torch.Tensor | None = None,
    ) -> list[TacticalModelOutput]:
        """Run conditioned heads for multiple candidate units from a single trunk pass.

        Parameters
        ----------
        h : (512,) — pre-computed trunk output (single, unbatched)
        u_attended : (20, 180) — attended unit embeddings from trunk
        unit_indices : which friendly unit slots to evaluate
        enemy_alive_mask : (10,) — enemy alive mask
        post_move_rel : (30,) — post-move relative features (or None for zeros)

        Returns
        -------
        One TacticalModelOutput per candidate unit (unit_logits=None).
        """
        results: list[TacticalModelOutput] = []
        value = self.value_head(h).squeeze(-1)  # scalar, shared across all candidates

        for uid in unit_indices:
            unit_features = self._extract_unit_features(u_attended, uid)

            move_logits, direction_params, distance_params, charge_logits, shoot_logits = (
                self._run_conditioned_heads(h, unit_features, enemy_alive_mask, post_move_rel)
            )

            results.append(TacticalModelOutput(
                unit_logits=None,
                move_logits=move_logits,
                direction_params=direction_params,
                distance_params=distance_params,
                charge_target_logits=charge_logits,
                shoot_target_logits=shoot_logits,
                value=value,
            ))

        return results
