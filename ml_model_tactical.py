"""ML tactical model v2: PyTorch network for per-activation wargame decisions.

Trunk architecture: per-unit shared encoder (200 → 128 → 64) applied to all
20 units, then aggregated embeddings (20×64) + global (16) = 1296 → 512
+ recurrent core (shared-weight block applied N times, default 6).

Sequential head chain with three pointer heads:
    h                            → unit_selection_head   → (10,)
    h ++ unit_feat(200)          → move_type_head        → (2,) move/charge
    h ++ unit_feat ++ move(2)    → dest_query_proj       → (64,) query for pointer attention
    dest_features(512×76)        → dest_embed            → (512, 64) keys
                                   scaled dot-product attention → (512,) logits → masked softmax
    per-enemy charge candidates  → charge attention      → (10,) logits → masked softmax
    per-enemy shoot candidates   → shoot attention       → (10,) logits → masked softmax

The charge and shoot heads are PointerScoreHead instances: a shared MLP
(one hidden layer, 64 wide) is applied to each per-candidate feature
vector and produces a single scalar score per enemy slot. Weights are
shared across candidate slots, so the scoring function generalises
rather than memorising slot indices. Candidate feature vectors combine
the shared trunk h with matchup scalars and raw defender / attacker
statistics; the MLP can then weight precomputed expected damage
against board context.

The move head is a binary charge-or-not decision. Hold, advance, and rush
are unified: the destination pointer always runs for "move", and each
candidate hex carries an advance-reachable flag. Rush-only hexes have
offensive-damage features zeroed out (the unit cannot shoot after rushing).
The chosen hex's flag determines whether shooting occurs.

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
    RANGE_THRESHOLDS, NUM_RANGE_THRESHOLDS, BOARD_DIAG,
    _TOFF_RANGED, _TOFF_MELEE, _TOFF_ACTIVATED, _TOFF_FATIGUED,
    _TOFF_SHAKEN, _TOFF_OBJ_REL,
)
from board import OBJ_SEIZE_RANGE

# Model version tag — bump when head architecture changes so old checkpoints reject.
MODEL_VERSION = 3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRUNK_WIDTH = 512
UNIT_EMBED_DIM = 64
N_FRIENDLY = MAX_UNITS_PER_SIDE   # 10
N_ENEMY = MAX_UNITS_PER_SIDE      # 10
N_TOTAL_UNITS = N_FRIENDLY + N_ENEMY  # 20
NUM_MOVE_TYPES = 2                # 0=move, 1=charge
POST_MOVE_REL_FEATURES = N_ENEMY * 3  # 30: (sin θ, cos θ, dist) per enemy from post-move pos
NUM_ROUNDS = 4                       # game has 4 rounds (one-hot in global features)
NUM_OPPONENT_TYPES = 5               # heuristic, selfplay_mirror, selfplay_hof, selfplay_ml, selfplay_random
OPP_EMBED_DIM = 8                    # opponent-type embedding dimension
NUM_SIDES = 2                        # physical side A=0, B=1
SIDE_EMBED_DIM = 4                   # side embedding dimension (small — diagnostic, not load-bearing)

# Movement type indices (binary: move or charge)
MOVE_MOVE = 0     # unified hold/advance/rush — dest pointer picks hex, flag determines shoot
MOVE_CHARGE = 1

# Recurrent core
DEFAULT_CORE_ITERS = 6            # default recurrent iterations (hyperparameter)
CORE_INPUT_WIDTH = TRUNK_WIDTH * 2  # 1024: concatenation of h and h0

# Phase re-encode (commit-and-re-encode)
# Phases at which the trunk is re-run during a single activation. Each phase
# carries its own FiLM modulation on the shared core_block and its own
# iteration count. h persists across phases (flavor-b); h0 is recomputed from
# the stem each phase. is_acting is injected per-unit as a post-encoder
# additive embedding (not via these constants — see IsActingEmbedding).
PHASE_PRE_SELECT = 0   # no unit chosen; features pre-move
PHASE_POST_SELECT = 1  # acting unit committed; features pre-move (unchanged)
PHASE_POST_MOVETYPE = 2  # move/charge committed; features pre-move (unchanged)
PHASE_POST_DEST = 3    # destination committed; features rebuilt from post-move state
N_PHASES = 4

# Default per-phase iteration counts. Cold-start on PRE_SELECT (full DEFAULT_CORE_ITERS),
# warm continuation on each subsequent phase with h persisted.
DEFAULT_PHASE_ITERS = {
    PHASE_PRE_SELECT: DEFAULT_CORE_ITERS,
    PHASE_POST_SELECT: 2,
    PHASE_POST_MOVETYPE: 2,
    PHASE_POST_DEST: 2,
}

# Bottleneck dimension for per-continuation-phase adjustment blocks. The blocks
# produce a phase-specific delta added to the shared core_block's output.
# 64 is modest — the phase refinement is low-rank (a 10-dim "which unit is
# acting" one-hot plus related structure) — but expressive enough to capture
# per-phase shifts without letting the block memorise noise.
PHASE_ADJUSTMENT_BOTTLENECK = 64

# (The former residual-scale anneal machinery has been removed — per-phase
# differentiation is now provided by the phase_adjustment_blocks, and bootstrap
# no-op behaviour is handled by zero-initialising the adjustment blocks' final
# Linear rather than by a training-loop-level α ramp.)

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

    When side_embed_dim > 0, the physical-side embedding is also
    concatenated (CTDE: value head only, not policy heads). This allows
    the value function to calibrate independently for A and B sides,
    preventing miscalibrated advantages from driving a self-play feedback
    loop.  Divergence between the A and B value estimates at the same
    states serves as a diagnostic for game-engine asymmetry.
    """

    def __init__(self, trunk_dim: int = TRUNK_WIDTH, round_dim: int = NUM_ROUNDS,
                 opp_embed_dim: int = OPP_EMBED_DIM,
                 side_embed_dim: int = SIDE_EMBED_DIM):
        super().__init__()
        self.opp_embed_dim = opp_embed_dim
        self.side_embed_dim = side_embed_dim
        self.film_gen = nn.Sequential(
            nn.Linear(round_dim, trunk_dim),
            nn.ReLU(),
            nn.Linear(trunk_dim, trunk_dim * 2),  # gamma and beta
        )
        self.value_proj = nn.Linear(trunk_dim + opp_embed_dim + side_embed_dim, 1)

        # Init last film_gen layer so gamma ≈ 1, beta ≈ 0 (identity at start)
        final_layer = self.film_gen[-1]
        nn.init.zeros_(final_layer.weight)
        # bias: first half (gamma) = 1, second half (beta) = 0
        with torch.no_grad():
            final_layer.bias[:trunk_dim].fill_(1.0)
            final_layer.bias[trunk_dim:].fill_(0.0)

        # Zero-init value projection so pre_tanh starts at exactly 0, preventing
        # early-training overshoot into the tanh saturation regime.
        nn.init.zeros_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

    def forward(self, h: torch.Tensor, round_onehot: torch.Tensor,
                opp_embed: torch.Tensor | None = None,
                side_embed: torch.Tensor | None = None) -> torch.Tensor:
        """
        h:             (..., trunk_dim)
        round_onehot:  (..., NUM_ROUNDS)
        opp_embed:     (..., opp_embed_dim) or None
        side_embed:    (..., side_embed_dim) or None
        Returns:       (...,) scalar value estimates
        """
        film_params = self.film_gen(round_onehot)
        gamma, beta = film_params.chunk(2, dim=-1)
        h_modulated = gamma * h + beta
        if opp_embed is not None and self.opp_embed_dim > 0:
            h_modulated = torch.cat([h_modulated, opp_embed], dim=-1)
        elif self.opp_embed_dim > 0:
            zeros = torch.zeros(*h_modulated.shape[:-1], self.opp_embed_dim,
                                device=h_modulated.device, dtype=h_modulated.dtype)
            h_modulated = torch.cat([h_modulated, zeros], dim=-1)
        if side_embed is not None and self.side_embed_dim > 0:
            h_modulated = torch.cat([h_modulated, side_embed], dim=-1)
        elif self.side_embed_dim > 0:
            zeros = torch.zeros(*h_modulated.shape[:-1], self.side_embed_dim,
                                device=h_modulated.device, dtype=h_modulated.dtype)
            h_modulated = torch.cat([h_modulated, zeros], dim=-1)
        pre_tanh = self.value_proj(h_modulated).squeeze(-1)
        return torch.tanh(pre_tanh)


# ---------------------------------------------------------------------------
# Pointer-style target-scoring head
# ---------------------------------------------------------------------------


CHARGE_CAND_FEATURES = 96  # per-candidate raw features (excl. h context)
POINTER_HIDDEN = 64        # shared MLP hidden width
CHARGE_EMBED_DIM = 64      # key/query embedding for attention-style charge pointer
SHOOT_SLOT_DIM = 8         # shoot head per-slot addressing vector width (bias pathway)


class PointerScoreHead(nn.Module):
    """Shared-weight MLP applied to each candidate feature vector, producing
    one scalar score per slot. Invalid slots are masked to -inf before the
    caller applies softmax / argmax.
    """

    def __init__(self, in_dim: int, hidden: int = POINTER_HIDDEN):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        cand_feats: torch.Tensor,   # (..., K, in_dim)
        mask: torch.Tensor | None,  # (..., K) bool or None
    ) -> torch.Tensor:
        scores = self.mlp(cand_feats).squeeze(-1)  # (..., K)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return scores


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TacticalModelOutput:
    unit_logits: torch.Tensor | None      # (10,) raw logits, masked; None from forward_per_unit
    move_logits: torch.Tensor             # (2,) move/charge
    dest_logits: torch.Tensor | None      # (MAX_DEST_CANDIDATES,) — None for charge in single inference
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

    Trunk: per-unit shared encoder (200 → 64) + aggregation (1296 → 512) +
    recurrent core (shared-weight block applied n_iters times). All heads
    always produce outputs (no conditional execution during forward). The
    integration layer decides which outputs to use based on move_type.
    """

    def __init__(self) -> None:
        super().__init__()

        # Per-unit shared encoder: 200 → 64 (applied identically to all 20 units)
        self.unit_encoder = nn.Sequential(
            nn.Linear(TACTICAL_UNIT_FEATURES, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, UNIT_EMBED_DIM),
            nn.LayerNorm(UNIT_EMBED_DIM),
            nn.ReLU(),
        )

        # Stem: aggregated unit embeddings + global → trunk width
        AGG_DIM = N_TOTAL_UNITS * UNIT_EMBED_DIM + GLOBAL_FEATURES  # 1296
        self.stem = nn.Sequential(
            nn.Linear(AGG_DIM, TRUNK_WIDTH),
            nn.LayerNorm(TRUNK_WIDTH),
            nn.ReLU(),
        )

        # Recurrent core block (shared weights, applied n_iters times)
        self.core_block = nn.Sequential(
            nn.Linear(CORE_INPUT_WIDTH, TRUNK_WIDTH),  # 1024 → 512
            nn.LayerNorm(TRUNK_WIDTH),
            nn.ReLU(),
        )
        self.n_iters = DEFAULT_CORE_ITERS

        # Phase re-encode machinery (hybrid-block architecture).
        # Modules exist but continuation phases are zero-initialised: at fresh
        # init the new encode() path produces bit-identical output to trunk()
        # for phase=PHASE_PRE_SELECT, and for continuation phases the
        # adjustment blocks contribute 0 (their final Linear is zero).
        #
        # phase_adjustment_blocks: per-continuation-phase small bottleneck MLP
        #   producing an additive delta to the shared core_block's output.
        #   Structure:  Linear(1024 → 64) → LN → ReLU → Linear(64 → 512).
        #   The final Linear is zero-initialised so continuation phases start
        #   as no-ops and learn phase-specific refinements over training. Three
        #   blocks for POST_SELECT / POST_MOVETYPE / POST_DEST (PRE_SELECT has
        #   no adjustment block — it uses only the shared core_block).
        self.phase_adjustment_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(CORE_INPUT_WIDTH, PHASE_ADJUSTMENT_BOTTLENECK),
                nn.LayerNorm(PHASE_ADJUSTMENT_BOTTLENECK),
                nn.ReLU(),
                nn.Linear(PHASE_ADJUSTMENT_BOTTLENECK, TRUNK_WIDTH),
            )
            for _ in range(N_PHASES - 1)  # one per continuation phase
        ])
        for _adj in self.phase_adjustment_blocks:
            nn.init.zeros_(_adj[-1].weight)
            nn.init.zeros_(_adj[-1].bias)
        # is_acting_embed: 2-row embedding added to acting unit's post-encoder row.
        #   Row 0 = "not acting" (unused / zero), row 1 = "acting". Both zero-init so
        #   initial model is identity; row 1 learns an informative direction over training.
        self.is_acting_embed = nn.Embedding(2, UNIT_EMBED_DIM)
        nn.init.zeros_(self.is_acting_embed.weight)
        # per_phase_value_heads: auxiliary per-phase V heads used as baselines in
        #   per-phase policy-gradient terms (Step 5). Predict same activation return
        #   as the main FiLMValueHead. Zero-init so they start as no-ops.
        self.per_phase_value_heads = nn.ModuleList([
            nn.Linear(TRUNK_WIDTH, 1) for _ in range(N_PHASES)
        ])
        for _vh in self.per_phase_value_heads:
            nn.init.zeros_(_vh.weight)
            nn.init.zeros_(_vh.bias)

        H = TRUNK_WIDTH  # 512
        UF = TACTICAL_UNIT_FEATURES  # 200

        # 1) Unit selection: h → 10 logits
        self.unit_selection_head = nn.Linear(H, N_FRIENDLY)

        # 2) Movement type: h + unit_feat → 2 logits (move/charge)
        self.move_type_head = nn.Linear(H + UF, NUM_MOVE_TYPES)

        # 3a) Destination pointer: cross-attention over candidate hex features
        self.dest_embed = nn.Sequential(
            nn.Linear(DEST_FEATURE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, DEST_EMBED_DIM),
        )
        self.dest_query_proj = nn.Linear(H + UF + NUM_MOVE_TYPES, DEST_EMBED_DIM)

        # 3b) Charge target pointer: attention-style scoring (mirrors dest/shoot).
        self.charge_embed = nn.Sequential(
            nn.Linear(CHARGE_CAND_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, CHARGE_EMBED_DIM),
        )
        self.charge_query_proj = nn.Linear(H + UF, CHARGE_EMBED_DIM)

        # 4) Shoot target pointer: lightweight two-pathway head.
        #    Key (per candidate): 2-dim raw scalars [expected_wound_frac, current_wound_frac].
        #      No embedding MLP — the shaping gradient on expected_wound_frac makes
        #      a trivial linear query sufficient for mechanical target selection.
        #    Query (from h alone, no unit_feat concat): shoot_W_q(h), (H → 2).
        #    Bias (per candidate j): (shoot_W_b(h)) · shoot_slot_embed[j],
        #      a passive addressing pathway letting h arbitrate between otherwise
        #      mechanically-identical candidates or override the mechanical signal.
        #    logits_j = query · key_j + bias_j, then masked as before.
        # shoot_W_q has a bias term so q=[+1, 0] at init gives the initial policy
        # a prior of preferring high expected_wound_frac before any learning.
        self.shoot_W_q = nn.Linear(H, 2, bias=True)
        self.shoot_W_b = nn.Linear(H, SHOOT_SLOT_DIM, bias=False)
        self.shoot_slot_embed = nn.Embedding(N_ENEMY, SHOOT_SLOT_DIM)
        self.register_buffer(
            "_shoot_slot_ids", torch.arange(N_ENEMY), persistent=False,
        )
        # Zero-init bias pathway so the scalar pathway is load-bearing at step 1.
        # W_q weight: small so |h|-dependent noise is tiny at init; bias -> [+1, 0]
        # so untrained policy already ranks by expected_wound_frac.
        nn.init.zeros_(self.shoot_W_b.weight)
        with torch.no_grad():
            self.shoot_W_q.weight.normal_(mean=0.0, std=1e-3)
            self.shoot_W_q.bias.copy_(torch.tensor([1.0, 0.0]))

        # Range thresholds as non-trainable buffer for searchsorted on GPU.
        self.register_buffer(
            "_range_thresholds",
            torch.tensor(RANGE_THRESHOLDS, dtype=torch.float32),
            persistent=False,
        )

        # Opponent-type embedding (CTDE: value head only, not policy heads)
        self.opponent_embedding = nn.Embedding(NUM_OPPONENT_TYPES, OPP_EMBED_DIM)

        # Physical-side embedding (CTDE: value head only — diagnostic for A/B symmetry)
        self.side_embedding = nn.Embedding(NUM_SIDES, SIDE_EMBED_DIM)

        # Value head: h + round + opponent + side → scalar (FiLM-conditioned on game round)
        self.value_head = FiLMValueHead(H, opp_embed_dim=OPP_EMBED_DIM,
                                        side_embed_dim=SIDE_EMBED_DIM)

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
        """Run input through per-unit encoder + aggregation + recurrent core.

        Returns
        -------
        h : (batch, 512) — refined latent representation
        units : (batch, 20, 200) — raw per-unit features (heads still use these)
        round_onehot : (batch, 4) — one-hot round indicator extracted from global features
        """
        # Extract global features and round indicator
        glob = x[..., _GLOBAL_OFFSET:]  # (batch, 16)
        round_onehot = glob[..., :NUM_ROUNDS]  # (batch, 4) or (4,)

        # Reshape unit block for per-unit feature extraction by heads
        unit_block = x[..., :_GLOBAL_OFFSET]
        units = unit_block.reshape(*x.shape[:-1], N_TOTAL_UNITS, TACTICAL_UNIT_FEATURES)

        # Encode each unit through shared MLP
        unit_embeds = self.unit_encoder(units)  # (..., 20, 64)

        # Flatten embeddings and concatenate with global features
        unit_embeds_flat = unit_embeds.reshape(*x.shape[:-1], -1)  # (..., 1280)
        agg = torch.cat([unit_embeds_flat, glob], dim=-1)  # (..., 1296)

        # Stem
        h0 = self.stem(agg)  # (batch, 512)

        # Recurrent core: shared-weight block applied n_iters times
        h = h0
        for _ in range(self.n_iters):
            h = h + self.core_block(torch.cat([h, h0], dim=-1))

        return h, units, round_onehot

    def encode(
        self,
        x: torch.Tensor,
        phase: int = PHASE_PRE_SELECT,
        acting_unit_idx: int | torch.Tensor | None = None,
        h_prev: torch.Tensor | None = None,
        n_iters: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase-aware trunk encode for the commit-and-re-encode architecture.

        Runs per-unit encoder + aggregation + recurrent core, with two new
        signals layered on top of the base trunk() path:

        * Per-unit is_acting additive embedding added to the acting unit's
          post-encoder (UNIT_EMBED_DIM) row, before aggregation. Both rows
          zero-initialised so the untrained injection is a no-op at init.
        * Per-continuation-phase adjustment block producing a small
          bottleneck delta on top of the shared core_block's output. The
          adjustment block's final Linear is zero-initialised so continuation
          phases contribute exactly the same delta as PRE_SELECT at init
          (pure shared-block output); they learn phase-specific refinements
          over training.

        h persists across phases via *h_prev* (flavor-b: we don't restart
        from h0 each phase, we carry the previous phase's working memory).
        h0 is always recomputed from the current state's stem output so the
        anchor reflects any feature changes (e.g. post-move positions).

        Parameters
        ----------
        x : (batch, TACTICAL_TOTAL_FEATURES) or (TACTICAL_TOTAL_FEATURES,)
        phase : one of PHASE_PRE_SELECT / POST_SELECT / POST_MOVETYPE / POST_DEST
        acting_unit_idx : friendly slot (0..N_FRIENDLY-1), (B,) tensor, or None
            for PRE_SELECT where no unit is committed yet.
        h_prev : persistent h from the previous phase, or None for a cold start
            (h initialised to h0, matching trunk()'s behaviour).
        n_iters : override iteration count; defaults to DEFAULT_PHASE_ITERS[phase].

        Returns
        -------
        h : (..., TRUNK_WIDTH)
        units : (..., N_TOTAL_UNITS, TACTICAL_UNIT_FEATURES) — raw per-unit slice.
        round_onehot : (..., NUM_ROUNDS) — extracted from global features.
        """
        if n_iters is None:
            n_iters = DEFAULT_PHASE_ITERS[phase]

        glob = x[..., _GLOBAL_OFFSET:]
        round_onehot = glob[..., :NUM_ROUNDS]
        unit_block = x[..., :_GLOBAL_OFFSET]
        units = unit_block.reshape(*x.shape[:-1], N_TOTAL_UNITS, TACTICAL_UNIT_FEATURES)

        unit_embeds = self.unit_encoder(units)  # (..., 20, UNIT_EMBED_DIM)

        # is_acting additive injection on the acting unit's row.
        if acting_unit_idx is not None:
            acting_vec = self.is_acting_embed.weight[1]  # (UNIT_EMBED_DIM,)
            if isinstance(acting_unit_idx, int):
                # Out-of-place add to avoid in-place autograd issues on
                # the leaf tensor returned by unit_encoder.
                delta = torch.zeros_like(unit_embeds)
                delta[..., acting_unit_idx, :] = acting_vec
                unit_embeds = unit_embeds + delta
            else:
                one_hot = F.one_hot(
                    acting_unit_idx.long(), N_TOTAL_UNITS
                ).to(unit_embeds.dtype)  # (B, 20)
                delta = one_hot.unsqueeze(-1) * acting_vec  # (B, 20, UNIT_EMBED_DIM)
                unit_embeds = unit_embeds + delta

        unit_embeds_flat = unit_embeds.reshape(*x.shape[:-1], -1)
        agg = torch.cat([unit_embeds_flat, glob], dim=-1)
        h0 = self.stem(agg)

        h = h0 if h_prev is None else h_prev

        # Phase-specific adjustment: PRE_SELECT uses only the shared core_block;
        # continuation phases add a small bottleneck delta on top. Adjustment
        # blocks are zero-init, so at fresh init continuation phases produce
        # the same delta as PRE_SELECT (pure shared-block output).
        adj_block = (self.phase_adjustment_blocks[phase - 1]
                     if phase != PHASE_PRE_SELECT else None)

        for _ in range(n_iters):
            combined = torch.cat([h, h0], dim=-1)
            delta = self.core_block(combined)
            if adj_block is not None:
                delta = delta + adj_block(combined)
            h = h + delta

        return h, units, round_onehot

    def per_phase_value(self, h: torch.Tensor, phase: int) -> torch.Tensor:
        """Auxiliary per-phase value estimate. Zero-initialised — trained in Step 5
        as a phase-local baseline for PPO's policy-gradient term at that phase."""
        return self.per_phase_value_heads[phase](h).squeeze(-1)

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

    def _get_side_embed(self, h: torch.Tensor, side: str | int | None) -> torch.Tensor:
        """Build physical-side embedding for value head conditioning.

        Parameters
        ----------
        h : (..., trunk_dim) — used only for shape/device inference
        side : "A" (0), "B" (1), int index, or None for eval
            (mean embedding over both sides)

        Returns (..., SIDE_EMBED_DIM)
        """
        if side is not None:
            if isinstance(side, str):
                side = 0 if side == "A" else 1
            side_embed = self.side_embedding(
                torch.tensor(side, device=h.device)
            )
        else:
            # Eval: use mean embedding (average over A and B)
            side_embed = self.side_embedding.weight.mean(dim=0)
        if h.dim() > 1:
            side_embed = side_embed.unsqueeze(0).expand(h.shape[0], -1)
        return side_embed

    def _extract_unit_features(self, units: torch.Tensor, unit_idx: int) -> torch.Tensor:
        """Extract the raw feature embedding for friendly unit *unit_idx*.

        Parameters
        ----------
        units : (batch, 20, 200) or (20, 200) — per-unit feature embeddings
        unit_idx : friendly unit slot (0–9)

        Works for both single and batched inputs.
        """
        return units[..., unit_idx, :]

    def _index_units(
        self,
        units: torch.Tensor,          # (..., 20, UF)
        chosen_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        """Gather the (..., UF) feature slice for the selected friendly unit.

        Supports scalar index (single sample) or (B,) long tensor (batched).
        """
        if isinstance(chosen_idx, int):
            return units[..., chosen_idx, :]
        # Batched: chosen_idx is (B,)
        idx = chosen_idx.long().view(-1, 1, 1).expand(-1, 1, units.shape[-1])
        return units.gather(-2, idx).squeeze(-2)

    def _build_shoot_keys(
        self,
        units: torch.Tensor,          # (..., 20, UF)
        chosen_idx: int | torch.Tensor,
        post_move_rel: torch.Tensor,  # (..., 30)
    ) -> torch.Tensor:
        """Per-candidate shoot keys: (..., 10, 2).

        Layout per candidate (enemy slot j):
          [expected_wound_frac, current_wound_frac]

        expected_wound_frac = acting unit's ranged matchup (expected wounds as
        fraction of target's starting wounds, capped at 1.0) at the bucketed
        post-move distance to enemy j. Already scaled by
        models_alive/models at encoding time (see ml_features.precompute_damage
        and the scale applied in _encode_unit), so a depleted shooter yields
        proportionally lower values.

        current_wound_frac = enemy j's survival_frac (remaining wounds / max).
        """
        enemies = units[..., N_FRIENDLY:, :]  # (..., 10, UF)
        acting = self._index_units(units, chosen_idx)  # (..., UF)

        # Acting unit's ranged matchup row: (..., 10, NUM_RANGE_THRESHOLDS)
        ranged_row = acting[..., _TOFF_RANGED:_TOFF_RANGED + N_ENEMY * NUM_RANGE_THRESHOLDS]
        ranged_row = ranged_row.reshape(*acting.shape[:-1], N_ENEMY, NUM_RANGE_THRESHOLDS)

        # Bucketed expected_wound_frac at post-move distance.
        dist_norm = post_move_rel.reshape(
            *post_move_rel.shape[:-1], N_ENEMY, 3,
        )[..., 2]
        dist_inches = dist_norm * BOARD_DIAG
        thresholds = self._range_thresholds
        k = torch.searchsorted(thresholds, dist_inches, right=False).clamp(
            max=NUM_RANGE_THRESHOLDS - 1
        )
        expected_wound_frac = ranged_row.gather(-1, k.unsqueeze(-1)).squeeze(-1)

        current_wound_frac = enemies[..., 3]  # survival_frac

        return torch.stack([expected_wound_frac, current_wound_frac], dim=-1)

    def _build_charge_candidates(
        self,
        units: torch.Tensor,          # (..., 20, UF)
        chosen_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        """Per-candidate charge features: (..., 10, CHARGE_CAND_FEATURES=96).

        Layout per candidate (enemy slot j):
          [melee_dealt (1),            # acting vs enemy j
           melee_taken (1),            # enemy j vs acting (damage taken)
           acting_survival, acting_tough (2),
           sinθ, cosθ, dist_norm (3),
           models_proxy, survival_frac, points_frac, tough_norm,
           has_activated, is_fatigued, is_shaken (7),
           enemy_ranged_row (70),
           enemy_melee_row (10),
           min_obj_dist_norm, within_seize_range (2)]
        Total: 1 + 1 + 2 + 3 + 7 + 80 + 2 = 96.
        """
        enemies = units[..., N_FRIENDLY:, :]  # (..., 10, UF)
        acting = self._index_units(units, chosen_idx)  # (..., UF)

        # Acting unit's melee row: damage dealt to each enemy (..., 10).
        melee_dealt = acting[..., _TOFF_MELEE:_TOFF_MELEE + N_ENEMY]

        # Reverse: enemy j's melee damage against acting unit (at column chosen_idx).
        if isinstance(chosen_idx, int):
            melee_taken = enemies[..., _TOFF_MELEE + chosen_idx]
        else:
            col_idx = chosen_idx.long().view(-1, 1, 1).expand(-1, N_ENEMY, 1)
            enemy_melee_rows = enemies[..., _TOFF_MELEE:_TOFF_MELEE + N_FRIENDLY]
            melee_taken = enemy_melee_rows.gather(-1, col_idx).squeeze(-1)

        # Acting-unit scalars (broadcast across 10 candidate slots)
        acting_survival = acting[..., 3:4].expand(*acting.shape[:-1], N_ENEMY)
        acting_tough = acting[..., 0:1].expand(*acting.shape[:-1], N_ENEMY)

        # Per-enemy spatial (from acting unit's opp-rel block)
        opp_rel = acting[..., 27:27 + N_ENEMY * 3]
        opp_rel = opp_rel.reshape(*acting.shape[:-1], N_ENEMY, 3)
        sin_t = opp_rel[..., 0]
        cos_t = opp_rel[..., 1]
        dist_norm = opp_rel[..., 2]

        # Per-enemy defender scalars
        tough_norm = enemies[..., 0]
        starting_models_norm = enemies[..., 1]
        survival_frac = enemies[..., 3]
        points_frac = enemies[..., 4]
        has_activated = enemies[..., _TOFF_ACTIVATED]
        is_fatigued = enemies[..., _TOFF_FATIGUED]
        is_shaken = enemies[..., _TOFF_SHAKEN]
        models_proxy = starting_models_norm * survival_frac

        # Enemy's full offensive profile (70 ranged + 10 melee)
        enemy_ranged = enemies[..., _TOFF_RANGED:_TOFF_RANGED + N_FRIENDLY * NUM_RANGE_THRESHOLDS]
        enemy_melee = enemies[..., _TOFF_MELEE:_TOFF_MELEE + N_FRIENDLY]

        # Objective proximity
        obj_dists = enemies[..., _TOFF_OBJ_REL + 2:_TOFF_OBJ_REL + 15:3]  # (..., 10, 5)
        min_obj_dist = obj_dists.min(dim=-1).values
        within_seize = (min_obj_dist * BOARD_DIAG < OBJ_SEIZE_RANGE).to(acting.dtype)

        scalars = torch.stack([
            melee_dealt, melee_taken,
            acting_survival, acting_tough,
            sin_t, cos_t, dist_norm,
            models_proxy, survival_frac, points_frac, tough_norm,
            has_activated, is_fatigued, is_shaken,
            min_obj_dist, within_seize,
        ], dim=-1)  # (..., 10, 16)

        cand = torch.cat([scalars, enemy_ranged, enemy_melee], dim=-1)  # (..., 10, 96)
        return cand

    def compute_charge_logits(
        self,
        h: torch.Tensor,              # (..., H)
        units: torch.Tensor,          # (..., 20, UF)
        chosen_idx: int | torch.Tensor,
        enemy_alive_mask: torch.Tensor | None,  # (..., 10)
        can_charge_mask: torch.Tensor | None,   # (..., 10)
    ) -> torch.Tensor:
        """Attention-style charge pointer: return (..., 10) masked logits."""
        cand = self._build_charge_candidates(units.detach(), chosen_idx)
        unit_features = self._index_units(units, chosen_idx).detach()

        keys = self.charge_embed(cand)                                  # (..., 10, CHARGE_EMBED_DIM)
        query_in = torch.cat([h, unit_features], dim=-1)                # (..., H + UF)
        query = self.charge_query_proj(query_in)                        # (..., CHARGE_EMBED_DIM)

        scale = CHARGE_EMBED_DIM ** 0.5
        logits = (query.unsqueeze(-2) @ keys.transpose(-1, -2)).squeeze(-2) / scale

        mask = None
        if enemy_alive_mask is not None:
            mask = enemy_alive_mask
        if can_charge_mask is not None:
            mask = can_charge_mask if mask is None else (mask & can_charge_mask)
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        return logits

    def compute_shoot_logits(
        self,
        h: torch.Tensor,              # (..., H)
        units: torch.Tensor,          # (..., 20, UF)
        chosen_idx: int | torch.Tensor,
        post_move_rel: torch.Tensor,  # (..., 30)
        enemy_alive_mask: torch.Tensor | None,  # (..., 10)
        shoot_range_mask: torch.Tensor | None = None,  # (..., 10)
        *,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lightweight two-pathway shoot pointer: return (..., 10) masked logits.

        Scalar pathway:  primary_j = (W_q h) · key_j
          key_j = [expected_wound_frac_j, current_wound_frac_j] (2-dim, no MLP)
        Bias pathway:    bias_j    = (W_b h) · slot_embed[j]
          slot_embed acts as a passive addressing vector letting h write
          per-slot biases independent of the key scalars.
        logits_j = primary_j + bias_j, then masked as before.

        Set return_components=True to also receive (primary, bias) for
        diagnostics (both pre-mask).
        """
        keys = self._build_shoot_keys(units.detach(), chosen_idx, post_move_rel)  # (..., 10, 2)

        query = self.shoot_W_q(h)                                                 # (..., 2)
        primary = (query.unsqueeze(-2) @ keys.transpose(-1, -2)).squeeze(-2)      # (..., 10)

        slot_vecs = self.shoot_slot_embed(self._shoot_slot_ids)                   # (10, SHOOT_SLOT_DIM)
        b_vec = self.shoot_W_b(h)                                                 # (..., SHOOT_SLOT_DIM)
        bias = b_vec @ slot_vecs.T                                                # (..., 10)

        logits = primary + bias

        mask = None
        if enemy_alive_mask is not None:
            mask = enemy_alive_mask
        if shoot_range_mask is not None:
            mask = shoot_range_mask if mask is None else (mask & shoot_range_mask)
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))

        if return_components:
            return logits, primary, bias
        return logits

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
        units: torch.Tensor,
        chosen_idx: int | torch.Tensor,
        enemy_alive_mask: torch.Tensor | None,
        post_move_rel: torch.Tensor | None,
        can_charge_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run move_type → charge pointer → shoot pointer chain.

        The destination pointer is NOT run here — it requires external
        candidate features and is handled by the caller.

        Parameters
        ----------
        h : (..., H) — trunk output.
        units : (..., 20, UF) — reshaped per-unit features from trunk.
        chosen_idx : selected friendly unit slot (scalar int, or (B,) tensor).
        post_move_rel : (..., 30) — post-move relative features for shoot head.
            If None, zeros are used.
        can_charge_mask : (..., 10) bool — True for chargeable enemies.

        Returns (move_logits, charge_target_logits, shoot_target_logits).
        """
        unit_features = self._index_units(units, chosen_idx).detach()

        # --- Movement type head ---
        h_uf = torch.cat([h, unit_features], dim=-1)
        move_logits = self.move_type_head(h_uf)

        # Mask charge option if no enemy is in charge range
        if can_charge_mask is not None:
            no_chargeable = ~can_charge_mask.any(dim=-1)  # (...,)
            move_logits = move_logits.clone()
            move_logits[..., MOVE_CHARGE] = move_logits[..., MOVE_CHARGE].masked_fill(
                no_chargeable, float('-inf'))

        # --- Charge pointer ---
        charge_target_logits = self.compute_charge_logits(
            h, units, chosen_idx, enemy_alive_mask, can_charge_mask,
        )

        # --- Shoot pointer ---
        if post_move_rel is None:
            post_move_rel = torch.zeros(
                *h.shape[:-1], POST_MOVE_REL_FEATURES,
                device=h.device, dtype=h.dtype,
            )
        shoot_target_logits = self.compute_shoot_logits(
            h, units, chosen_idx, post_move_rel, enemy_alive_mask,
            shoot_range_mask=None,
        )

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
            self._run_conditioned_heads(
                h, units, chosen_idx, enemy_alive_mask, post_move_rel,
                can_charge_mask,
            )
        )

        # --- Destination pointer (only if features provided) ---
        dest_logits = None
        if dest_features is not None and dest_mask is not None:
            move_onehot = F.one_hot(
                move_logits.detach().argmax(dim=-1), NUM_MOVE_TYPES
            ).float()
            h_uf_m = torch.cat([h, unit_features.detach(), move_onehot], dim=-1)
            dest_logits = self.compute_dest_logits(h_uf_m, dest_features, dest_mask)

        # --- Value (round + opponent + side conditioned) ---
        opp_embed = self._get_opp_embed(h, opponent_type)
        side_embed = self._get_side_embed(h, None)  # forward() doesn't know the side; use mean
        value = self.value_head(h, round_onehot, opp_embed, side_embed)

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
        side_embed = self._get_side_embed(h, None)  # eval path — use mean
        value = self.value_head(h, round_onehot, opp_embed, side_embed)  # scalar, shared across all candidates

        for k, uid in enumerate(unit_indices):
            unit_features = self._extract_unit_features(units, uid)
            ccm = can_charge_masks[k] if can_charge_masks is not None else None

            move_logits, charge_logits, shoot_logits = (
                self._run_conditioned_heads(
                    h, units, uid, enemy_alive_mask, post_move_rel, ccm,
                )
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
