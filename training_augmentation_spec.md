# Training Augmentation Spec — Planning, Aux Horizons, Opponent Conditioning

This spec covers three training-time improvements, ordered by implementation dependency:

1. **Opponent-Conditioned Value Head** — value head receives an opponent-type embedding (training only)
2. **Multi-Horizon Auxiliary Predictions** — add end-of-next-round supervision targets to existing aux heads
3. **Planning-Augmented Training** — stochastic planning during training with gated distillation

Features 1–2 are independent of each other and can be implemented in either order. Feature 3 depends on the existing `ml_planning.py` infrastructure and benefits from features 1–2 being in place (better value head → better planning evaluations), but has no hard code dependency on them.

---

## Feature 1: Opponent-Conditioned Value Head

### 1.1 Motivation

The value head currently estimates V(s) by averaging over all opponent types. This produces noisy advantage estimates: beating a weak heuristic and beating a strong self-play checkpoint both look like positive reward, but the advantage signal should be very different. Conditioning the value head on opponent type (while keeping all policy heads unconditioned) follows the Centralized Training, Decentralized Execution (CTDE) pattern — the policy learns robust play, but the critic has sharper baselines.

### 1.2 Opponent Type Encoding

Five categorical opponent types, matching the existing training schedule:

| Index | Type | When |
|-------|------|------|
| 0 | `heuristic` | vs heuristic AI |
| 1 | `selfplay_mirror` | mirror self-play (both sides = current model) |
| 2 | `selfplay_hof` | vs checkpoint from hall_of_fame.json armies |
| 3 | `selfplay_ml` | vs checkpoint from hall_of_fame_ml.json armies |
| 4 | `selfplay_random` | vs checkpoint with random armies |

### 1.3 Architecture Changes

**File: `ml_model_tactical.py`**

Add a learnable embedding table and modify the value head:

```python
NUM_OPPONENT_TYPES = 5
OPP_EMBED_DIM = 8

# In __init__:
self.opponent_embedding = nn.Embedding(NUM_OPPONENT_TYPES, OPP_EMBED_DIM)
# Replace:  self.value_head = nn.Linear(H, 1)
# With:
self.value_head = nn.Linear(H + OPP_EMBED_DIM, 1)
```

No other heads are modified. The opponent embedding is concatenated to the trunk output `h` only for the value head forward path.

### 1.4 Forward Pass Changes

**File: `ml_model_tactical.py`**

Add an `opponent_type` parameter to `forward()` and `forward_per_unit()`:

```python
def forward(
    self,
    x: torch.Tensor,
    alive_mask: torch.Tensor | None,
    enemy_alive_mask: torch.Tensor | None,
    *,
    forced_unit_idx: int | None = None,
    post_move_rel: torch.Tensor | None = None,
    opponent_type: int | None = None,  # NEW — index into NUM_OPPONENT_TYPES
) -> TacticalModelOutput:
```

Value computation becomes:

```python
if opponent_type is not None:
    opp_embed = self.opponent_embedding(
        torch.tensor([opponent_type], device=h.device)
    ).expand(h.shape[0], -1)  # (batch, OPP_EMBED_DIM)
    value_input = torch.cat([h, opp_embed], dim=-1)
else:
    # Eval / planning: use mean embedding (average over all types)
    opp_embed = self.opponent_embedding.weight.mean(dim=0, keepdim=True)
    opp_embed = opp_embed.expand(h.shape[0], -1)
    value_input = torch.cat([h, opp_embed], dim=-1)
value = self.value_head(value_input).squeeze(-1)
```

Using the mean embedding at eval time (rather than zeros) ensures the value head operates in a similar region of its input space to training, avoiding a distributional shift.

### 1.5 Data Pipeline Changes

**File: `ml_training.py`**

The `TacticalActivationRecord` dataclass gains a new field:

```python
@dataclass
class TacticalActivationRecord:
    # ... existing fields ...
    opponent_type_idx: int = 0  # NEW — index into NUM_OPPONENT_TYPES
```

**Mapping logic** (in `_run_single_episode_tactical` or the worker that builds records):

```python
_OPP_TYPE_MAP = {
    "heuristic": 0,
    "selfplay_mirror": 1,
    "selfplay_hof": 2,
    "selfplay_ml": 3,
    "selfplay_random": 4,
}

def _get_opponent_type_idx(opp_type: str, army_type: str) -> int:
    if opp_type == "heuristic":
        return 0
    if opp_type == "selfplay_mirror":
        return 1
    # For checkpoint-based self-play, use army_type to distinguish
    if army_type == "hof":
        return 2
    if army_type == "hof_ml":
        return 3
    return 4  # random
```

The opponent type index must be stored per-activation (not per-game) because all activations in a game share the same opponent, but the replay system operates on flat activation records.

### 1.6 Replay Path Changes

**File: `ml_training.py`**

In `replay_log_probs_batch` (the flat PPO replay function), pass `opponent_type` through to the model's forward call. Since all activations in a single game share the same opponent type, this is just reading the stored `opponent_type_idx` from each `TacticalActivationRecord` and passing it to the model.

For batched replay, construct an `opponent_type_batch` tensor `(N,)` of ints from the flat steps and pass it to the batched forward. The model's value head indexes the embedding table per-sample.

### 1.7 Planning Path

**File: `ml_planning.py`**

Planning uses the value head to score candidates. During training-time planning (Feature 3), pass the current game's opponent type through to planning rollout value evaluations. During eval-time planning (existing code), pass `opponent_type=None` so the mean-embedding fallback applies.

### 1.8 Checkpoint Compatibility

Old checkpoints lack `opponent_embedding` and have `value_head` with input dim `H` instead of `H + OPP_EMBED_DIM`. Load with `strict=False`. The embedding initialises randomly (fine — it will be trained quickly). The value head's weight matrix will be the wrong shape; handle this in `load_model_state_dict`:

```python
# If loading old checkpoint without opponent conditioning:
if "opponent_embedding.weight" not in state_dict:
    # Don't load value_head from checkpoint — let it reinitialise
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("value_head.")}
```

This means the value head resets when upgrading from an old checkpoint. This is acceptable — the value head trains quickly (a few hundred batches), and the policy heads are unaffected.

### 1.9 Logging

Log per-opponent-type mean value estimates as a diagnostic: `mean_value_heuristic`, `mean_value_sp_mirror`, etc. This confirms the conditioning is working (values against heuristic should be consistently higher than against strong SP opponents).

---

## Feature 2: Multi-Horizon Auxiliary Predictions

### 2.1 Motivation

The existing aux heads (survival Beta NLL + objective control CE) predict end-of-game state only. Adding a shorter-horizon target — end-of-current-round — gives the trunk richer gradient signal about immediate consequences of actions. Dedicated short-horizon heads allow each horizon to specialise its predictions, while the shared trunk still benefits from both gradient signals.

### 2.2 Approach: Separate Heads, Blended Loss

**File: `ml_model_tactical.py`**

Three new prediction heads are added alongside the existing long-horizon (end-of-game) heads:

```python
# Long-horizon (end-of-game) — existing
self.aux_friendly_survival_head = nn.Linear(H, N_FRIENDLY * 2)
self.aux_enemy_survival_head = nn.Linear(H, N_ENEMY * 2)
self.aux_obj_control_head = nn.Linear(H, 5 * 3)

# Short-horizon (end-of-current-round) — NEW
self.aux_friendly_survival_head_short = nn.Linear(H, N_FRIENDLY * 2)
self.aux_enemy_survival_head_short = nn.Linear(H, N_ENEMY * 2)
self.aux_obj_control_head_short = nn.Linear(H, 5 * 3)
```

Each set of heads produces its own predictions from the shared trunk output `h`. The total auxiliary loss sums both horizons:

```
aux_loss = loss(short_heads(h), end_of_current_round_target)
         + loss(long_heads(h), end_of_game_target)
```

Since each horizon has its own dedicated heads, there is no need to discount either loss — the gradients through the head weights are independent, and both contribute to the shared trunk. The short-horizon target has lower variance (closer in time, fewer intervening stochastic events), so it provides a cleaner gradient signal for learning local state dynamics. The long-horizon target ensures the trunk still learns features relevant to winning the whole game.

### 2.3 Data Collection Changes

**File: `ml_training.py`**

During episode collection in `_run_single_episode_tactical`, the code already iterates through rounds and has access to `units_a`, `units_b`, `board` at each round boundary. After each round's activations complete (and `board.update_objectives()` has been called), snapshot the survival fractions and objective control:

```python
@dataclass
class RoundSnapshot:
    friendly_survival: list[float]  # length 10, per-unit survival fraction
    enemy_survival: list[float]     # length 10
    obj_control: list[int]          # length 5, 0=friendly 1=enemy 2=neutral
```

Build one `RoundSnapshot` at the end of each round (after objectives update). Store these in a list indexed by round number (0-indexed: round 1 end = index 0, round 4 end = index 3).

### 2.4 Target Assignment

For each `TacticalActivationRecord` created during round R:

- **Short-horizon target**: `RoundSnapshot` from end of round R (i.e., the state after the current round finishes). For activations in round 4 (the last round), the short-horizon target equals the end-of-game target.
- **Long-horizon target**: `RoundSnapshot` from end of round 4 (existing behaviour, same as current `friendly_survival_target` / `enemy_survival_target` / `obj_control_target`).

New fields on `TacticalActivationRecord`:

```python
@dataclass
class TacticalActivationRecord:
    # ... existing fields ...
    # Existing end-of-game targets (renamed for clarity):
    friendly_survival_target: list[float] | None = None     # end-of-game
    enemy_survival_target: list[float] | None = None        # end-of-game
    obj_control_target: list[int] | None = None             # end-of-game
    # NEW short-horizon targets:
    friendly_survival_target_short: list[float] | None = None  # end-of-current-round
    enemy_survival_target_short: list[float] | None = None     # end-of-current-round
    obj_control_target_short: list[int] | None = None          # end-of-current-round
```

### 2.5 Loss Computation Changes

**File: `ml_training.py`**

Modify `_compute_aux_loss` to compute both horizon losses from their respective head outputs and blend them. A helper `_compute_aux_loss_horizon` handles one horizon at a time, selecting the correct head outputs and targets based on a `use_short` flag:

```python
def _compute_aux_loss_horizon(flat_result, flat_steps, use_short: bool):
    # Select head outputs: short-horizon heads or long-horizon heads
    if use_short:
        fs_alpha = flat_result.aux_friendly_surv_alpha_short[idx]
        # ... (analogous for beta, enemy, obj)
    else:
        fs_alpha = flat_result.aux_friendly_surv_alpha[idx]
        # ...
    # Compute Beta NLL + obj CE against the matching targets
    ...

def _compute_aux_loss(flat_result, flat_steps):
    short_loss = _compute_aux_loss_horizon(flat_result, flat_steps, use_short=True)
    long_loss = _compute_aux_loss_horizon(flat_result, flat_steps, use_short=False)
    # Sum both (gracefully handles None if one horizon has no valid targets)
    return short_loss + long_loss
```

Each horizon's loss is computed against its own dedicated head outputs. The short-horizon heads learn to predict end-of-current-round state; the long-horizon heads learn to predict end-of-game state. Both sets of gradients flow through the shared trunk.

`FlatReplayResult` gains five new fields for the short-horizon head outputs (`aux_friendly_surv_alpha_short`, `aux_friendly_surv_beta_short`, `aux_enemy_surv_alpha_short`, `aux_enemy_surv_beta_short`, `aux_obj_control_logits_short`), computed in `replay_tactical_log_probs_flat` from the `_short` heads.

### 2.6 Snapshot Helper

```python
def _make_round_snapshot(
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    board: Board,
    player: str,
) -> RoundSnapshot:
    fs = [_survival_fraction(friendly_units[i]) if i < len(friendly_units)
          and friendly_units[i].models_alive > 0 else 0.0
          for i in range(MAX_UNITS_PER_SIDE)]

    es = [_survival_fraction(enemy_units[i]) if i < len(enemy_units)
          and enemy_units[i].models_alive > 0 else 0.0
          for i in range(MAX_UNITS_PER_SIDE)]

    friendly_tag = player
    enemy_tag = "B" if player == "A" else "A"
    # Remap objective indices for player perspective (same as _objective_control_mapped)
    order = [0, 1, 2, 3, 4] if player == "A" else [0, 2, 1, 4, 3]
    obj = []
    for idx in order:
        ctrl = board.objective_control[idx]
        if ctrl == friendly_tag:
            obj.append(0)  # friendly
        elif ctrl == enemy_tag:
            obj.append(1)  # enemy
        else:
            obj.append(2)  # neutral
    return RoundSnapshot(fs, es, obj)
```

### 2.7 Integration Point

In `_run_single_episode_tactical`, after each round's activations complete and `board.update_objectives()` is called, call `_make_round_snapshot()` and store it. After the game ends, walk through all activation records and assign both short-horizon and long-horizon targets from the appropriate snapshots.

---

## Feature 3: Planning-Augmented Training

### 3.1 Overview

During training, each activation has a small probability (`planning_rate`, default 0.05) of being resolved via Monte Carlo planning instead of direct policy sampling. This follows the Expert Iteration (ExIt) principle: occasionally make search-quality decisions to reach better game states, and train the policy to match. A gated KL distillation loss on the unit selection head accelerates learning from planned activations.

### 3.2 New Hyperparameters

**File: `ml_training.py`, `TrainingConfig`**

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...
    # Planning-augmented training
    planning_rate: float = 0.05           # probability of planning per activation
    planning_rate_end: float | None = None  # if set, anneal planning_rate linearly to this value
    planning_distill_max_weight: float = 0.1  # max weight for distillation KL loss
    training_planning_K: int = 3          # candidate units (reduced from eval default 6)
    training_planning_C: int = 3          # action samples per unit (reduced from 4)
    training_planning_M: int = 4          # rollouts per candidate (same as eval)
    training_planning_N: int = 3          # lookahead activations (reduced from 4)
```

These reduced defaults give 3 × 3 × 4 = 36 rollouts per planned activation (vs 96 at eval), each simulating 3 activations. At ~0.1ms per activation, that's ~11ms per planned activation. With 5% of ~40 activations per game = ~2 planned activations, this adds ~22ms per game (~15-25% overhead depending on baseline game time).

### 3.3 Policy Argmax as Baseline Candidate

When planning triggers, the first candidate evaluated is always the **policy's own argmax action tuple**. This is the action the model would have taken without planning. The remaining K-1 candidate units are sampled from the unit selection distribution (excluding the argmax unit if it was already included).

For the argmax candidate: run the full conditioned head chain in argmax mode (same as eval), producing a complete action tuple (unit, move_type, direction, distance, charge_target, shoot_target). This becomes candidate 0.

For the remaining candidates: sample K-1 additional units from the unit selection logits (top-K excluding the argmax), then sample C action tuples per unit from the policy (via the existing sampling path).

All candidates (including the argmax) go through M rollouts with N-step lookahead, producing an average value per candidate.

### 3.4 Planning Decision and Gating

After rollouts, compare the best non-argmax candidate's value against the argmax candidate's value:

```python
argmax_value = candidates[0].avg_value
best_search_idx = argmax(c.avg_value for c in candidates[1:]) + 1
best_search_value = candidates[best_search_idx].avg_value

planning_improved = best_search_value > argmax_value
planning_value_delta = max(best_search_value - argmax_value, 0.0)

if planning_improved:
    chosen_candidate = candidates[best_search_idx]
else:
    chosen_candidate = candidates[0]  # stick with policy argmax
```

The chosen candidate's action is executed in the game. Whether or not planning improved, the activation record stores all the information needed for the distillation loss.

### 3.5 Data Stored Per Planned Activation

New fields on `TacticalActivationRecord`:

```python
@dataclass
class TacticalActivationRecord:
    # ... existing fields ...
    # Planning augmentation data (None for non-planned activations)
    was_planned: bool = False
    planning_improved: bool = False
    planning_value_delta: float = 0.0
    planning_unit_values: list[float] | None = None   # per-candidate-unit avg rollout values
    planning_unit_indices: list[int] | None = None     # which unit slots were evaluated
```

`planning_unit_values` and `planning_unit_indices` together define the distillation target for the unit selection head: a softmax distribution over the evaluated units, weighted by their rollout values.

### 3.6 Distillation Loss

**File: `ml_training.py`**

For planned activations where `planning_improved == True`, compute a KL divergence loss pushing the unit selection head toward the planning-derived target distribution:

```python
def _compute_planning_distill_loss(
    flat_result: FlatReplayResult,
    flat_steps: list[TacticalActivationRecord],
    max_weight: float,
) -> torch.Tensor | None:
    """Gated distillation loss from planned activations.

    Only applies to activations where planning found a better action than
    the policy argmax. Weight scales with the value gap.
    """
    planned_indices = []
    target_dists = []
    weights = []

    for i, s in enumerate(flat_steps):
        if not s.was_planned or not s.planning_improved:
            continue
        if s.planning_unit_values is None or s.planning_unit_indices is None:
            continue

        # Build soft target over 10 unit slots from planning values
        target = torch.full((MAX_UNITS_PER_SIDE,), float('-inf'))
        for idx, val in zip(s.planning_unit_indices, s.planning_unit_values):
            target[idx] = val
        target = torch.softmax(target, dim=0)  # non-evaluated slots → 0

        planned_indices.append(i)
        target_dists.append(target)
        weights.append(min(s.planning_value_delta, max_weight))

    if not planned_indices:
        return None

    idx = torch.tensor(planned_indices, dtype=torch.long)
    targets = torch.stack(target_dists)                          # (P, 10)
    w = torch.tensor(weights, dtype=torch.float32)               # (P,)

    # Get policy's unit selection log-probs for these activations
    unit_log_probs = F.log_softmax(flat_result.unit_logits[idx], dim=-1)  # (P, 10)

    # KL(target || policy) = sum(target * (log(target) - log(policy)))
    kl = (targets * (targets.clamp(min=1e-8).log() - unit_log_probs)).sum(dim=-1)  # (P,)

    return (w * kl).mean()
```

### 3.7 Integration with Total Loss

In `compute_loss_flat`, after the existing aux loss:

```python
# Planning distillation loss
distill_loss_val = 0.0
if config.planning_rate > 0 and flat_steps is not None:
    _distill = _compute_planning_distill_loss(
        flat_result, flat_steps, config.planning_distill_max_weight
    )
    if _distill is not None:
        distill_loss_val = _distill.item()
        loss = loss + _distill
```

The distillation loss does not go through the `aux_ratio` adaptive scaling — it has its own weighting via `planning_value_delta`, which naturally decays as the policy improves.

### 3.8 FlatReplayResult Changes

The existing `FlatReplayResult` dataclass needs `unit_logits` exposed so the distillation loss can compute KL divergence against the planning-derived target distribution. Currently the unit logits are computed inside `replay_tactical_log_probs_flat` but not returned. Add:

```python
@dataclass
class FlatReplayResult:
    # ... existing fields ...
    unit_logits: torch.Tensor | None = None  # (N, 10) — raw logits after alive masking
```

In `replay_tactical_log_probs_flat`, after computing `unit_logits` (already done for log-prob computation), store it on the result:

```python
# Already exists in the function:
unit_logits = model.unit_selection_head(h)
unit_logits = unit_logits.masked_fill(~alive_batch, float('-inf'))

# Add to returned FlatReplayResult:
return FlatReplayResult(
    # ... existing fields ...
    unit_logits=unit_logits,  # NEW
)
```

### 3.9 Episode Collection: Planning Trigger

**File: `ml_training.py`**

Planning triggers are evaluated per-activation for **Player A only** (the training player). Player B (opponent) never uses training-time planning regardless of type.

In `_run_single_episode_tactical`, immediately before the existing `sample_tactical_actions_no_grad` call for Player A, add:

```python
use_planning = (
    config.planning_rate > 0
    and random.random() < config.planning_rate
)

if use_planning:
    # Run single-threaded planning (§3.10)
    (sel_idx, move_type_a, sampled_angle_a, sampled_frac_a,
     charge_tgt_a, shoot_tgt_a, _a_tac_target_ranking,
     pmr_a, old_lp, value_est, shoot_mask_a,
     _was_planned, _planning_improved, _planning_value_delta,
     _planning_unit_values, _planning_unit_indices
    ) = plan_training_activation(
        model, state_vec, alive_mask, enemy_alive_mask,
        units_a, units_b, round_num, board, "A",
        current_is_a=current_is_a,
        mode=mode,
        friendly_positions=a_friendly_pos,
        enemy_positions=a_enemy_pos,
        advance_distances=a_adv_dists,
        rush_distances=a_rush_dists,
        max_weapon_ranges=a_max_wr,
        fr_a=fr_a, fm_a=fm_a, fr_b=fr_b, fm_b=fm_b,
        pts_a=pts_a, pts_b=pts_b,
        planning_params={
            "K_UNITS": config.training_planning_K,
            "C_SAMPLES_PER_UNIT": config.training_planning_C,
            "M_ROLLOUTS": config.training_planning_M,
            "N_LOOKAHEAD": config.training_planning_N,
        },
        opponent_type=opponent_type_idx,  # for value head conditioning
    )
else:
    # Existing path: sample from policy
    (sel_idx, move_type_a, ...) = sample_tactical_actions_no_grad(...)
    _was_planned = False
    _planning_improved = False
    _planning_value_delta = 0.0
    _planning_unit_values = None
    _planning_unit_indices = None
```

The `TacticalActivationRecord` is then built with the planning fields populated:

```python
step = TacticalActivationRecord(
    # ... existing fields ...
    was_planned=_was_planned,
    planning_improved=_planning_improved,
    planning_value_delta=_planning_value_delta,
    planning_unit_values=_planning_unit_values,
    planning_unit_indices=_planning_unit_indices,
)
```

**Critical: `old_log_prob` is always `π_policy(a_chosen | s)`** — the probability the base policy assigns to whatever action was actually taken, whether chosen by policy sampling or by planning. This is computed by running the policy forward on the chosen action, not by any planning-derived quantity. For planned activations, this means running a forward pass through the policy's heads with the planning-chosen action to get its log-prob under the current policy. The `plan_training_activation` function (§3.10) is responsible for returning this value.

### 3.10 Single-Threaded Planning for Training

**File: `ml_planning.py`**

Add a new entry point for training-time planning that runs single-threaded (no multiprocessing pool) and includes the policy-argmax baseline candidate:

```python
@torch.no_grad()
def plan_training_activation(
    model: TacticalModel,
    state_vec: torch.Tensor,
    alive_mask: torch.Tensor,
    enemy_alive_mask: torch.Tensor,
    friendly_units: list[UnitState],
    enemy_units: list[UnitState],
    round_num: int,
    board: Board,
    player: str,
    *,
    current_is_a: bool,
    mode: str,
    friendly_positions: list[tuple[float, float]],
    enemy_positions: list[tuple[float, float]],
    advance_distances: list[float],
    rush_distances: list[float],
    max_weapon_ranges: list[float] | None = None,
    fr_a=None, fm_a=None, fr_b=None, fm_b=None,
    pts_a: int = 0, pts_b: int = 0,
    planning_params: dict | None = None,
    opponent_type: int | None = None,
) -> tuple:
    """Training-time planning with policy-argmax baseline.

    Runs single-threaded (no worker pool). Returns the same action tuple
    format as sample_tactical_actions_no_grad, plus planning metadata.

    Returns:
        (unit_idx, move_type, angle, dist_frac, charge_target, shoot_target,
         target_ranking, post_move_rel, old_log_prob, value, shoot_mask,
         was_planned, planning_improved, planning_value_delta,
         planning_unit_values, planning_unit_indices)
    """
```

The function proceeds as follows:

1. **Policy argmax candidate (candidate 0):** Run full forward pass in argmax mode. Record the action tuple and compute `old_log_prob` for this action under the sampling distribution (for PPO compatibility). This is always the first candidate.

2. **Sampled candidates (candidates 1..K-1):** Sample K-1 additional unit indices from the top of the unit selection distribution (excluding the argmax unit). For each, sample C action tuples from the policy via the conditioned heads.

3. **Rollouts:** For each candidate (including the argmax), run M rollouts with N-step lookahead using `_run_rollout_chunk_sequential` (the existing single-threaded rollout worker). Pass `opponent_type` through so the value head evaluations during rollouts are conditioned.

4. **Selection:** Compare argmax candidate value against best non-argmax value. Choose the winner.

5. **Log-prob of chosen action:** If the chosen action differs from the argmax (planning improved), compute `π_policy(a_chosen | s)` via a forward pass through the sampling path with the chosen action forced. This ensures PPO's importance ratio is well-defined.

6. **Return:** All standard action fields plus `(was_planned=True, planning_improved, planning_value_delta, planning_unit_values, planning_unit_indices)`.

The `planning_unit_values` list contains the average rollout value for each evaluated unit, and `planning_unit_indices` contains the corresponding unit slot indices. Together these define the distillation target.

### 3.11 Interaction with Coroutine-Based Episode Collection

The current training episode collection uses a coroutine/generator pattern where `_run_single_episode_tactical` yields `_TacticalInferenceRequest` objects for batched inference across multiple games. Training-time planning is **not compatible with this batching pattern** — planning requires running full rollout simulations from within a single game's activation, which blocks the coroutine.

**Recommended approach: Planning activations bypass the coroutine.** When `use_planning` triggers, the coroutine does not yield an inference request. Instead, it calls `plan_training_activation` directly (which runs synchronously in the worker process) and gets back the full action tuple. The coroutine continues with the planned action as if inference had completed normally. Non-planned activations continue to yield for batched inference as before.

This means planned activations incur sequential overhead within their worker process, but since only ~5% of activations trigger planning, the impact on batching efficiency is minimal.

### 3.12 Planning Rate Scheduling

The planning rate can optionally be scheduled. Two useful patterns:

**Constant rate (default):** `planning_rate = 0.05` throughout training. Simple and effective.

**Annealed rate:** Start higher (e.g., 0.10) when the policy is weak and planning provides the most value, decay to lower (e.g., 0.02) as the policy improves and planning's marginal benefit diminishes. This naturally reduces overhead as training progresses.

```python
if config.planning_rate_end is not None:
    effective_rate = config.planning_rate + progress * (config.planning_rate_end - config.planning_rate)
else:
    effective_rate = config.planning_rate
```

For the initial implementation, use a constant rate. The `planning_rate_end` field is a placeholder for future use.

### 3.13 Logging and Metrics

Add the following per-batch metrics:

| Metric | Description |
|--------|-------------|
| `planning_activations` | Count of planned activations in batch |
| `planning_improvement_rate` | Fraction of planned activations where planning beat the policy argmax |
| `planning_mean_value_delta` | Mean value gap (V_planned - V_argmax) across planned activations where planning improved |
| `planning_distill_loss` | Distillation loss magnitude (0 if no planned activations improved) |
| `planning_overhead_ms` | Total wall-clock time spent on training planning in batch |

The `planning_improvement_rate` is the key diagnostic. If it stays near 1.0 throughout training, the policy has significant room to improve and planning is consistently finding better actions. If it drops toward 0, the policy has converged to planning-quality decisions (or planning budget is insufficient). A healthy trajectory is a gradual decline from ~0.8 to ~0.3 over the course of training.

---

## Files Modified

| File | Features | Changes |
|------|----------|---------|
| `ml_model_tactical.py` | 1, 2 | Add `opponent_embedding`, modify `value_head` input dim, add `opponent_type` param to `forward()` and `forward_per_unit()`; add short-horizon aux heads (`aux_*_head_short`) |
| `ml_training.py` | 1, 2, 3 | New fields on `TacticalActivationRecord`; opponent type mapping; round snapshot collection; planning trigger in episode collection; distillation loss; new `TrainingConfig` fields; updated metrics logging |
| `ml_planning.py` | 1, 3 | Pass `opponent_type` through to value head in rollouts; new `plan_training_activation()` entry point for single-threaded training planning |
| `result_plotting.py` | 3 | Add planning metrics panels (improvement rate, distill loss, overhead) |

## Files Unchanged

| File | Reason |
|------|--------|
| `ai.py` | Heuristic agent path unaffected |
| `game.py` | No changes — planning is called from `ml_training.py`, not `game.py` |
| `board.py` | No changes |
| `models.py` | No changes |
| `combat.py` | No changes |
| `ml_model.py` | Strategic model unaffected |
| `ml_features.py` | No changes |
| `evolution.py` | Optimizer unaffected |

## Checkpoint Compatibility

- **Feature 1 (opponent conditioning):** Old checkpoints lack `opponent_embedding` and have wrong `value_head` shape. Load with `strict=False`, drop old `value_head.*` keys, let both reinitialise. Value head trains quickly (~100-200 batches). Policy heads are unaffected.
- **Feature 2 (multi-horizon aux):** Old checkpoints lack the `aux_*_head_short` layers. Load with `strict=False`; the new short-horizon heads initialise randomly and train quickly. The replay path guards with `hasattr`, so training gracefully falls back to long-horizon-only loss until the short heads are present. New `TacticalActivationRecord` fields have defaults, so in-progress training data is compatible.
- **Feature 3:** No model architecture changes beyond Features 1–2. New `TacticalActivationRecord` fields have defaults, so in-progress training data is compatible.

## Implementation Order

Recommended sequence for Claude Code tasks:

**Task 1: Opponent-Conditioned Value Head (Feature 1)**
- Modify `ml_model_tactical.py`: add embedding, modify value head, update forward signatures
- Modify `ml_training.py`: add `opponent_type_idx` to `TacticalActivationRecord`, mapping logic, pass through replay path
- Add checkpoint compat handling in `load_model_state_dict`
- Update `ml_planning.py` to accept and pass `opponent_type`

**Task 2: Multi-Horizon Auxiliary Targets (Feature 2)**
- Add `RoundSnapshot` dataclass and `_make_round_snapshot()` to `ml_training.py`
- Modify episode collection to capture per-round snapshots
- Add short-horizon target fields to `TacticalActivationRecord`
- Modify aux loss computation to blend both horizons

**Task 3: Training-Time Planning (Feature 3)**
- Add `plan_training_activation()` to `ml_planning.py`
- Add planning fields to `TacticalActivationRecord`
- Add planning trigger in episode collection
- Add `_compute_planning_distill_loss()` to `ml_training.py`
- Wire distillation loss into `compute_loss_flat`
- Add `unit_logits` to `FlatReplayResult`
- Add planning config fields to `TrainingConfig`
- Add planning metrics logging

**Task 4: Metrics and Plotting**
- Add per-opponent-type value estimates to CSV logging
- Add planning metrics to CSV logging
- Update `result_plotting.py` with new panels
