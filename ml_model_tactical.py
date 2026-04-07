"""ML tactical model v2: PyTorch network for per-activation wargame decisions.

Trunk architecture: feedforward compression 4016 → 2048 → 1024 → 512
+ 2 residual layers.

Sequential head chain with destination pointer:
    h                            → unit_selection_head   → (10,)
    h ++ unit_feat(200)          → move_type_head        → (4,) hold/advance/rush/charge
    h ++ unit_feat ++ move(4)    → dest_query_proj       → (64,) query for pointer attention
    dest_features(512×75)        → dest_embed            → (512, 64) keys
                                   scaled dot-product attention → (512,) logits → masked softmax
    h ++ unit_feat ++ move(4)    → charge_target_head    → (10,)
    h ++ unit_feat ++ move(4)
      ++ post_move_rel(30)       → shoot_target_head     → (10,)

unit_feat is the raw 200-dim feature slice for the selected unit.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_features import (
    TACTICAL_TOTAL_FEATURES, MAX_UNITS_PER_SIDE, TACTICAL_UNIT_FEATURES,
    GLOBAL_FEATURES, extract_can_charge_mask,
    DEST_FEATURE_DIM, DEST_EMBED_DIM, MAX_DEST_CANDIDATES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRUNK_PRE_STEM = 2048
TRUNK_STEM = 1024
TRUNK_WIDTH = 512
N_FRIENDLY = MAX_UNITS_PER_SIDE   # 10
N_ENEMY = MAX_UNITS_PER_SIDE      # 10
N_TOTAL_UNITS = N_FRIENDLY + N_ENEMY  # 20
NUM_MOVE_TYPES = 4                # 0=hold, 1=advance, 2=rush, 3=charge
POST_MOVE_REL_FEATURES = N_ENEMY * 3  # 30: (sin θ, cos θ, dist) per enemy from post-move pos
NUM_ROUNDS = 4                       # game has 4 rounds (one-hot in global features)
NUM_OPPONENT_TYPES = 5               # heuristic, selfplay_mirror, selfplay_hof, selfplay_ml, selfplay_random
OPP_EMBED_DIM = 8                    # opponent-type embedding dimension

# Movement type indices
MOVE_HOLD = 0
MOVE_ADVANCE = 1
MOVE_RUSH = 2
MOVE_CHARGE = 3

# Offset of global features within the flat observation vector
_GLOBAL_OFFSET = N_TOTAL_UNITS * TACTICAL_UNIT_FEATURES  # 20 * 200 = 4000


# ---------------------------------------------------------------------------
# FiLM-conditioned value head
# ---------------------------------------------------------------------------

class FiLMValueHead(nn.Module):
    """Value head conditioned on game round via FiLM modulation.

    A small MLP maps the round one-hot to (gamma, beta) vectors that
    scale and shift the trunk features before the final value projection.
    This lets the value function learn that identical board states have
    different values at different stages of the game.

    When opp_embed_dim > 0, the opponent-type embedding is concatenated
    to the FiLM-modulated trunk before the final projection (CTDE pattern:
    the critic sees opponent type during training, the policy does not).
    """

    def __init__(self, trunk_dim: int = TRUNK_WIDTH, round_dim: int = NUM_ROUNDS,
                 opp_embed_dim: int = OPP_EMBED_DIM):
        super().__init__()
        self.opp_embed_dim = opp_embed_dim
        self.film_gen = nn.Sequential(
            nn.Linear(round_dim, trunk_dim),
            nn.ReLU(),
            nn.Linear(trunk_dim, trunk_dim * 2),  # gamma and beta
        )
        self.value_proj = nn.Linear(trunk_dim + opp_embed_dim, 1)

        # Init last film_gen layer so gamma ≈ 1, beta ≈ 0 (identity at start)
        final_layer = self.film_gen[-1]
        nn.init.zeros_(final_layer.weight)
        # bias: first half (gamma) = 1, second half (beta) = 0
        with torch.no_grad():
            final_layer.bias[:trunk_dim].fill_(1.0)
            final_layer.bias[trunk_dim:].fill_(0.0)

    def forward(self, h: torch.Tensor, round_onehot: torch.Tensor,
                opp_embed: torch.Tensor | None = None) -> torch.Tensor:
        """
        h:             (..., trunk_dim)
        round_onehot:  (..., NUM_ROUNDS)
        opp_embed:     (..., opp_embed_dim) or None
        Returns:       (...,) scalar value estimates
        """
        film_params = self.film_gen(round_onehot)
        gamma, beta = film_params.chunk(2, dim=-1)
        h_modulated = gamma * h + beta
        if opp_embed is not None and self.opp_embed_dim > 0:
            h_modulated = torch.cat([h_modulated, opp_embed], dim=-1)
        elif self.opp_embed_dim > 0:
            # No embedding provided — pad with zeros (should not happen in normal use)
            zeros = torch.zeros(*h_modulated.shape[:-1], self.opp_embed_dim,
                                device=h_modulated.device, dtype=h_modulated.dtype)
            h_modulated = torch.cat([h_modulated, zeros], dim=-1)
        return self.value_proj(h_modulated).squeeze(-1).tanh()


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TacticalModelOutput:
    unit_logits: torch.Tensor | None      # (10,) raw logits, masked; None from forward_per_unit
    move_logits: torch.Tensor             # (4,) hold/advance/rush/charge
    dest_logits: torch.Tensor | None      # (MAX_DEST_CANDIDATES,) — None for hold/charge in single inference
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

    Trunk: feedforward 4016 → 2048 → 1024 → 512 + 2 residual layers.
    All heads always produce outputs (no conditional execution during forward).
    The integration layer decides which outputs to use based on move_type.
    """

    def __init__(self) -> None:
        super().__init__()

        # Stem: flat input → compress to trunk width (smooth 2:1 reduction at each step)
        self.stem = nn.Sequential(
            nn.Linear(TACTICAL_TOTAL_FEATURES, TRUNK_PRE_STEM),
            nn.LayerNorm(TRUNK_PRE_STEM),
            nn.ReLU(),
            nn.Linear(TRUNK_PRE_STEM, TRUNK_STEM),
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
        UF = TACTICAL_UNIT_FEATURES  # 200

        # 1) Unit selection: h → 10 logits
        self.unit_selection_head = nn.Linear(H, N_FRIENDLY)

        # 2) Movement type: h + unit_feat → 4 logits
        self.move_type_head = nn.Linear(H + UF, NUM_MOVE_TYPES)

        # 3a) Destination pointer: cross-attention over candidate hex features
        self.dest_embed = nn.Sequential(
            nn.Linear(DEST_FEATURE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, DEST_EMBED_DIM),
        )
        self.dest_query_proj = nn.Linear(H + UF + NUM_MOVE_TYPES, DEST_EMBED_DIM)

        # 3b) Charge target: h + unit_feat + move_onehot → 10 logits
        self.charge_target_head = nn.Linear(H + UF + NUM_MOVE_TYPES, N_ENEMY)

        # 4) Shooting target: h + unit_feat + move_onehot + post_move_rel → 10 logits
        self.shoot_target_head = nn.Linear(H + UF + NUM_MOVE_TYPES + POST_MOVE_REL_FEATURES, N_ENEMY)

        # Opponent-type embedding (CTDE: value head only, not policy heads)
        self.opponent_embedding = nn.Embedding(NUM_OPPONENT_TYPES, OPP_EMBED_DIM)

        # Value head: h + round + opponent → scalar (FiLM-conditioned on game round)
        self.value_head = FiLMValueHead(H, opp_embed_dim=OPP_EMBED_DIM)

        # Auxiliary prediction heads (trained with supervised targets, not used at inference)
        # Long-horizon (end-of-game) heads
        # Friendly survival: Beta(α, β) per friendly unit — 2 raw params each
        self.aux_friendly_survival_head = nn.Linear(H, N_FRIENDLY * 2)
        # Enemy survival: Beta(α, β) per enemy unit — 2 raw params each
        self.aux_enemy_survival_head = nn.Linear(H, N_ENEMY * 2)
        # Objective control: 5 objectives × 3 classes (friendly/enemy/neutral)
        self.aux_obj_control_head = nn.Linear(H, 5 * 3)
        # Short-horizon (end-of-current-round) heads
        self.aux_friendly_survival_head_short = nn.Linear(H, N_FRIENDLY * 2)
        self.aux_enemy_survival_head_short = nn.Linear(H, N_ENEMY * 2)
        self.aux_obj_control_head_short = nn.Linear(H, 5 * 3)
        # Activation countdown heads: predict remaining activations until game end
        self.aux_friendly_activations_head = nn.Linear(H, 1)
        self.aux_enemy_activations_head = nn.Linear(H, 1)

    def trunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run input through feedforward stem + residual blocks.

        Returns
        -------
        h : (batch, 512) — global trunk representation
        units : (batch, 20, 200) — raw per-unit feature embeddings (reshaped from input)
        round_onehot : (batch, 4) — one-hot round indicator extracted from global features
        """
        # Extract round one-hot from global features
        glob = x[..., _GLOBAL_OFFSET:]  # (batch, 16)
        round_onehot = glob[..., :NUM_ROUNDS]  # (batch, 4) or (4,)

        # Reshape unit block for per-unit feature extraction by heads
        unit_block = x[..., :_GLOBAL_OFFSET]
        units = unit_block.reshape(*x.shape[:-1], N_TOTAL_UNITS, TACTICAL_UNIT_FEATURES)

        # Feedforward compression → residual blocks
        h = self.stem(x)
        h = h + self.res_block_1(h)
        h = h + self.res_block_2(h)
        return h, units, round_onehot

    def _get_opp_embed(self, h: torch.Tensor, opponent_type: int | None) -> torch.Tensor:
        """Build opponent-type embedding for value head conditioning.

        Parameters
        ----------
        h : (..., trunk_dim) — used only for shape/device inference
        opponent_type : index into NUM_OPPONENT_TYPES, or None for eval
            (mean embedding over all types)

        Returns (..., OPP_EMBED_DIM)
        """
        if opponent_type is not None:
            opp_embed = self.opponent_embedding(
                torch.tensor(opponent_type, device=h.device)
            )  # (OPP_EMBED_DIM,)
        else:
            # Eval / planning: use mean embedding (average over all types)
            opp_embed = self.opponent_embedding.weight.mean(dim=0)  # (OPP_EMBED_DIM,)
        # Broadcast to match h's leading dims
        if h.dim() > 1:
            opp_embed = opp_embed.unsqueeze(0).expand(h.shape[0], -1)
        return opp_embed

    def _extract_unit_features(self, units: torch.Tensor, unit_idx: int) -> torch.Tensor:
        """Extract the raw feature embedding for friendly unit *unit_idx*.

        Parameters
        ----------
        units : (batch, 20, 200) or (20, 200) — per-unit feature embeddings
        unit_idx : friendly unit slot (0–9)

        Works for both single and batched inputs.
        """
        return units[..., unit_idx, :]

    def compute_dest_logits(
        self,
        h_uf_m: torch.Tensor,
        dest_features: torch.Tensor,
        dest_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute destination pointer logits via scaled dot-product cross-attention.

        Parameters
        ----------
        h_uf_m : (..., TRUNK_WIDTH + UF + NUM_MOVE_TYPES) — query input
        dest_features : (..., MAX_DEST_CANDIDATES, DEST_FEATURE_DIM)
        dest_mask : (..., MAX_DEST_CANDIDATES) bool — True for valid candidates

        Returns (..., MAX_DEST_CANDIDATES) logits, invalid candidates masked to -inf.
        """
        dest_keys = self.dest_embed(dest_features)           # (..., MAX_DEST_CANDIDATES, DEST_EMBED_DIM)
        dest_query = self.dest_query_proj(h_uf_m)            # (..., DEST_EMBED_DIM)

        scale = DEST_EMBED_DIM ** 0.5
        # Scaled dot-product: query (unsqueeze to row) @ keys^T
        dest_logits = (dest_query.unsqueeze(-2) @ dest_keys.transpose(-1, -2)).squeeze(-2) / scale
        # (..., MAX_DEST_CANDIDATES)

        dest_logits = dest_logits.masked_fill(~dest_mask, float('-inf'))
        return dest_logits

    def _run_conditioned_heads(
        self,
        h: torch.Tensor,
        unit_features: torch.Tensor,
        enemy_alive_mask: torch.Tensor | None,
        post_move_rel: torch.Tensor | None,
        can_charge_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run move_type → charge_target → shoot_target chain.

        The destination pointer is NOT run here — it requires external candidate
        features and is handled by the caller (forward / integration layer).

        Parameters
        ----------
        post_move_rel : (batch, 30) or (30,) — post-move relative features.
            If None, zeros are used (argmax forward pass computes this externally).
        can_charge_mask : (batch, 10) or (10,) bool — True for chargeable enemies.

        Returns (move_logits, charge_target_logits, shoot_target_logits).
        """
        unit_features = unit_features.detach()

        # --- Movement type head ---
        h_uf = torch.cat([h, unit_features], dim=-1)
        move_logits = self.move_type_head(h_uf)

        # Mask charge option if no enemy is in charge range
        if can_charge_mask is not None:
            no_chargeable = ~can_charge_mask.any(dim=-1)  # (...,)
            move_logits = move_logits.clone()
            move_logits[..., MOVE_CHARGE] = move_logits[..., MOVE_CHARGE].masked_fill(
                no_chargeable, float('-inf'))

        # Conditioning: one-hot of argmax move type
        move_onehot = F.one_hot(
            move_logits.detach().argmax(dim=-1), NUM_MOVE_TYPES
        ).float()

        # --- Common input for charge ---
        h_uf_m = torch.cat([h, unit_features, move_onehot], dim=-1)

        # --- Charge target head ---
        charge_target_logits = self.charge_target_head(h_uf_m)
        if enemy_alive_mask is not None:
            charge_target_logits = charge_target_logits.masked_fill(~enemy_alive_mask, float('-inf'))
        if can_charge_mask is not None:
            charge_target_logits = charge_target_logits.masked_fill(~can_charge_mask, float('-inf'))

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

        return move_logits, charge_target_logits, shoot_target_logits

    def forward(
        self,
        x: torch.Tensor,
        alive_mask: torch.Tensor | None = None,
        enemy_alive_mask: torch.Tensor | None = None,
        *,
        forced_unit_idx: int | None = None,
        post_move_rel: torch.Tensor | None = None,
        opponent_type: int | None = None,
        dest_features: torch.Tensor | None = None,
        dest_mask: torch.Tensor | None = None,
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
        opponent_type : index into NUM_OPPONENT_TYPES for value head conditioning.
            None uses mean embedding (eval/planning default).
        dest_features : (batch, MAX_DEST_CANDIDATES, DEST_FEATURE_DIM) — per-hex features.
            If None, destination pointer is skipped (dest_logits=None).
        dest_mask : (batch, MAX_DEST_CANDIDATES) bool — True for valid candidates.
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
            if dest_features is not None and dest_features.dim() == 2:
                dest_features = dest_features.unsqueeze(0)
            if dest_mask is not None and dest_mask.dim() == 1:
                dest_mask = dest_mask.unsqueeze(0)

        h, units, round_onehot = self.trunk(x)

        # --- Unit selection ---
        unit_logits = self.unit_selection_head(h)  # (batch, 10)
        if alive_mask is not None:
            unit_logits = unit_logits.masked_fill(~alive_mask, float('-inf'))

        if forced_unit_idx is not None:
            chosen_idx = forced_unit_idx
        else:
            chosen_idx = unit_logits.detach()[0].argmax(dim=-1).item()

        unit_features = self._extract_unit_features(units, chosen_idx)

        # Extract can_charge mask for the selected unit from the state vector
        can_charge_mask = extract_can_charge_mask(x, chosen_idx)

        # --- Conditioned head chain (move, charge, shoot) ---
        move_logits, charge_logits, shoot_logits = (
            self._run_conditioned_heads(h, unit_features, enemy_alive_mask, post_move_rel,
                                        can_charge_mask)
        )

        # --- Destination pointer (only if features provided) ---
        dest_logits = None
        if dest_features is not None and dest_mask is not None:
            move_onehot = F.one_hot(
                move_logits.detach().argmax(dim=-1), NUM_MOVE_TYPES
            ).float()
            h_uf_m = torch.cat([h, unit_features.detach(), move_onehot], dim=-1)
            dest_logits = self.compute_dest_logits(h_uf_m, dest_features, dest_mask)

        # --- Value (round + opponent conditioned) ---
        opp_embed = self._get_opp_embed(h, opponent_type)
        value = self.value_head(h, round_onehot, opp_embed)

        if single:
            return TacticalModelOutput(
                unit_logits=unit_logits.squeeze(0),
                move_logits=move_logits.squeeze(0),
                dest_logits=dest_logits.squeeze(0) if dest_logits is not None else None,
                charge_target_logits=charge_logits.squeeze(0),
                shoot_target_logits=shoot_logits.squeeze(0),
                value=value.squeeze(0),
            )

        return TacticalModelOutput(
            unit_logits=unit_logits,
            move_logits=move_logits,
            dest_logits=dest_logits,
            charge_target_logits=charge_logits,
            shoot_target_logits=shoot_logits,
            value=value,
        )

    def forward_per_unit(
        self,
        h: torch.Tensor,
        units: torch.Tensor,
        unit_indices: list[int],
        enemy_alive_mask: torch.Tensor,
        post_move_rel: torch.Tensor | None = None,
        round_onehot: torch.Tensor | None = None,
        opponent_type: int | None = None,
        can_charge_masks: list[torch.Tensor] | None = None,
        dest_features_list: list[torch.Tensor] | None = None,
        dest_mask_list: list[torch.Tensor] | None = None,
    ) -> list[TacticalModelOutput]:
        """Run conditioned heads for multiple candidate units from a single trunk pass.

        Parameters
        ----------
        h : (512,) — pre-computed trunk output (single, unbatched)
        units : (20, 200) — per-unit feature embeddings from trunk
        unit_indices : which friendly unit slots to evaluate
        enemy_alive_mask : (10,) — enemy alive mask
        post_move_rel : (30,) — post-move relative features (or None for zeros)
        round_onehot : (4,) — one-hot round indicator for FiLM value head
        opponent_type : index into NUM_OPPONENT_TYPES for value head conditioning.
            None uses mean embedding (eval/planning default).
        can_charge_masks : per-unit (10,) bool masks, one per unit_indices entry.
        dest_features_list : per-unit (MAX_DEST_CANDIDATES, DEST_FEATURE_DIM) tensors.
        dest_mask_list : per-unit (MAX_DEST_CANDIDATES,) bool tensors.

        Returns
        -------
        One TacticalModelOutput per candidate unit (unit_logits=None).
        """
        results: list[TacticalModelOutput] = []
        if round_onehot is None:
            round_onehot = torch.zeros(NUM_ROUNDS, device=h.device, dtype=h.dtype)
        opp_embed = self._get_opp_embed(h, opponent_type)
        value = self.value_head(h, round_onehot, opp_embed)  # scalar, shared across all candidates

        for k, uid in enumerate(unit_indices):
            unit_features = self._extract_unit_features(units, uid)
            ccm = can_charge_masks[k] if can_charge_masks is not None else None

            move_logits, charge_logits, shoot_logits = (
                self._run_conditioned_heads(h, unit_features, enemy_alive_mask, post_move_rel, ccm)
            )

            # Destination pointer
            dest_logits = None
            if dest_features_list is not None and dest_mask_list is not None:
                move_onehot = F.one_hot(
                    move_logits.detach().argmax(dim=-1), NUM_MOVE_TYPES
                ).float()
                h_uf_m = torch.cat([h, unit_features.detach(), move_onehot], dim=-1)
                dest_logits = self.compute_dest_logits(
                    h_uf_m, dest_features_list[k], dest_mask_list[k])

            results.append(TacticalModelOutput(
                unit_logits=None,
                move_logits=move_logits,
                dest_logits=dest_logits,
                charge_target_logits=charge_logits,
                shoot_target_logits=shoot_logits,
                value=value,
            ))

        return results
