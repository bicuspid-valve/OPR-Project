# Tactical Planning Model — Architecture & Integration Spec

This spec covers four areas of work:

1. **Architecture changes** — restructure TacticalModel to use sequentially conditioned output heads and categorical target selection
2. **Execution logic** — deterministic action resolution from ML outputs, replacing the current heuristic dispatch in `ai.py`
3. **Monte Carlo planning** — eval-time search using rollouts and the value head
4. **Training changes** — sequential sampling, conditioned replay, updated record format

Steps 1 and 2 are prerequisites for steps 3 and 4, but are independently useful (step 2 replaces the current hybrid ML+heuristic action dispatch with a cleaner fully-ML-driven path). Step 4 can be implemented in parallel with step 3.

---

## Step 1: Architecture Changes to TacticalModel

### 1.1 Overview

Replace the current flat multi-head architecture (all heads read the same trunk output) with a sequential chain where each decision head is conditioned on the outputs of preceding heads. Replace the continuous exp-clamped target priority multipliers with a categorical distribution over enemy slots.

**Current architecture:**
```
state_vec (771) → trunk (256→128) → all heads in parallel from h
```

**New architecture:**
```
state_vec (771) → trunk (256→128) → h

h                                → unit_selection_head  → unit logits (10)
h ++ unit_features (38)          → priority_head         → priority logits (2)
h ++ unit_features ++ priority (2) → objective_head      → objective logits (5)
h ++ unit_features ++ priority (2) → target_head         → target logits (10)
h ++ unit_features ++ priority (2) ++ target_ranking (10) ++ objective (5) → engagement_head → engagement logits (4)
```

Where `++` denotes concatenation and all conditioning inputs are **detached** during forward (no gradient flows backward through sampled discrete choices — the policy gradient handles credit assignment).

### 1.2 Constants

```python
TRUNK_HIDDEN_1 = 256
TRUNK_HIDDEN_2 = 128
TACTICAL_UNIT_FEATURES = 38  # existing, unchanged
N_FRIENDLY = 10
N_ENEMY = 10
NUM_PRIORITIES = 2       # 0=objective, 1=killer
NUM_OBJECTIVES = 5       # centre, my-side, enemy-side, my-home, enemy-home
NUM_ENGAGEMENTS = 4      # 0=melee, 1=ranged_aggressive, 2=ranged_kite, 3=ranged_hold
```

### 1.3 Head Input Dimensions

| Head | Input | Dim |
|------|-------|-----|
| `unit_selection_head` | h | 128 |
| `priority_head` | h ++ unit_features | 128 + 38 = 166 |
| `objective_head` | h ++ unit_features ++ priority_onehot | 166 + 2 = 168 |
| `target_head` | h ++ unit_features ++ priority_onehot | 168 |
| `engagement_head` | h ++ unit_features ++ priority_onehot ++ target_ranking_softmax ++ objective_onehot | 168 + 10 + 5 = 183 |

Each head is a single `nn.Linear(input_dim → output_dim)` layer. No hidden layers within heads (trunk provides representation capacity).

### 1.4 Target Head

The target head replaces the old exp-clamped continuous multipliers. It outputs logits over 10 enemy slots, masked by alive enemies (dead slots get -inf before softmax). The resulting probability distribution is interpreted as a **priority ranking**: argmax = #1 target, second-highest = #2, etc.

This ranking is used by the execution logic (step 2) to find the highest-ranked *reachable* enemy for the chosen engagement type. The model is not masked by reachability — it ranks all alive enemies by desirability and the execution logic handles the "can I actually reach them" question.

```python
target_logits = self.target_head(h2)                          # (batch, 10)
target_logits = target_logits.masked_fill(~enemy_alive_mask, float('-inf'))
target_probs = torch.softmax(target_logits, dim=-1)           # (batch, 10)
```

### 1.5 Objective Head

The objective head fires for all activations (not just priority=objective) because killer-priority units still need an objective assignment as a tiebreaker for positioning when no enemies are reachable. However, the objective output only materially affects execution when priority=objective.

### 1.6 Engagement Types

The four engagement types replace the old `combat_preference` × `movement_stance` combination:

| Index | Name | Meaning |
|-------|------|---------|
| 0 | `melee` | Charge if possible, otherwise rush toward target |
| 1 | `ranged_aggressive` | Advance toward target and shoot, rush if out of range |
| 2 | `ranged_kite` | Maximise distance from nearest enemy while keeping target in range |
| 3 | `ranged_hold` | Hold position and shoot; do nothing if no target in range |

No masking is applied to engagement types. A melee-only unit selecting `ranged_hold` will simply hold and do nothing (no ranged weapons to fire). The model learns from reward that this wastes the activation.

### 1.7 Value Head

Unchanged from current architecture. Reads from trunk output `h` (128-dim), outputs a scalar state value estimate. Not conditioned on action choices — it estimates game-level win probability from the current state.

```python
self.value_head = nn.Linear(TRUNK_HIDDEN_2, 1)
```

### 1.8 Forward Pass Signature

```python
def forward(
    self,
    x: torch.Tensor,                      # (batch, 771) or (771,)
    alive_mask: torch.Tensor | None,       # (batch, 10) or (10,) — friendly alive+unactivated
    enemy_alive_mask: torch.Tensor | None, # (batch, 10) or (10,) — enemy alive
    *,
    # For conditioned heads during search/eval (pre-selected choices):
    forced_unit_idx: int | None = None,
    forced_priority: int | None = None,
) -> TacticalModelOutput:
```

The `forced_*` parameters support the planning loop (step 3): when searching over candidate actions, the planner can force a specific unit selection and query the downstream heads for that unit's decisions.

### 1.9 Output Dataclass

```python
@dataclass
class TacticalModelOutput:
    unit_logits: torch.Tensor | None   # (10,) raw logits, masked by alive_mask; None when returned by forward_per_unit
    priority_logits: torch.Tensor    # (2,)
    objective_logits: torch.Tensor   # (5,)
    target_logits: torch.Tensor      # (10,) masked by enemy_alive_mask
    engagement_logits: torch.Tensor  # (4,)
    value: torch.Tensor              # scalar
```

### 1.10 Batched Forward for Search

For planning (step 3), we need to efficiently query the per-unit heads for multiple candidate units from a single trunk pass. Add a method:

```python
def forward_per_unit(
    self,
    h: torch.Tensor,                       # (128,) — pre-computed trunk output
    x: torch.Tensor,                       # (771,) — full state vec (for extracting unit features)
    unit_indices: list[int],               # which unit slots to evaluate
    enemy_alive_mask: torch.Tensor,        # (10,)
) -> list[TacticalModelOutput]:
```

This runs the trunk once, then loops over `unit_indices`, extracting each unit's 38-float feature slice and running the conditioned head chain. Returns one `TacticalModelOutput` per candidate unit (with `unit_logits` set to `None` since unit selection is already decided).

**Feature vector layout for unit extraction:** In the 771-float tactical state vector, friendly unit `i`'s 38 features occupy `x[i * 38 : i * 38 + 38]` (offsets 0–379 are the friendly block). Enemy unit `j`'s features occupy `x[380 + j * 38 : 380 + j * 38 + 38]` (offsets 380–759 are the enemy block). Global features are at offsets 760–770.

### 1.11 `enemy_alive_mask` Construction

The new `forward()` signature requires an `enemy_alive_mask` tensor alongside the existing `alive_mask`. This mask is `True` for enemy slots where `models_alive > 0` (no activation-status filter — enemies may or may not have activated). It must be built and passed at every call site:

- `apply_tactical_model` in `ml_integration_tactical.py` (eval path)
- `apply_tactical_model_sampling` in `ml_integration_tactical.py` (self-play opponent path)
- `_run_single_episode_tactical` in `ml_training.py` (training episode loop, Player A forward pass)
- The coroutine batching path: `InferenceRequest` must carry `enemy_alive_mask`, and the batch runner must stack it alongside `alive_mask`
- `plan_activation` in `ml_planning.py` (planning rollouts)

Construction pattern (mirrors `alive_mask` but for enemies, without the `not activated` filter):

```python
enemy_alive_mask = torch.tensor(
    [(i < len(enemy_units) and enemy_units[i].models_alive > 0)
     for i in range(MAX_UNITS_PER_SIDE)],
    dtype=torch.bool,
)
```

---

## Step 2: Execution Logic

### 2.1 Overview

Replace the current `choose_action_and_goal` + `pick_target` dispatch in `ai.py` with a deterministic execution function that translates ML outputs into (action, goal_position, charge_target) tuples. This function is pure game logic — no ML, no heuristics, no scoring. It takes the model's decisions as authoritative and resolves them against the game state.

### 2.2 Input

```python
def execute_ml_decision(
    unit: UnitState,
    enemies: list[UnitState],
    priority: str,              # "objective" or "killer"
    target_ranking: list[int],  # enemy slot indices sorted by priority (highest first)
    objective_idx: int,         # index into OBJECTIVES list
    engagement: str,            # "melee" | "ranged_aggressive" | "ranged_kite" | "ranged_hold"
    board: Board,
) -> tuple[str, tuple[int,int] | None, UnitState | None, str]:
    """Returns (action, goal_position, charge_target, reason)."""
```

`target_ranking` is the full ordering of alive enemy slots by the model's target probabilities, highest first. The execution logic walks this list to find the highest-ranked enemy that satisfies the reachability constraints for the chosen engagement type.

### 2.3 Execution Cases

The logic is a priority × engagement matrix. In all cases below, "highest priority target" means the first alive enemy in `target_ranking` that satisfies the relevant reachability constraint. "Assigned objective" means `OBJECTIVES[objective_idx]`.

---

#### Case 1: `melee` + `objective`

**If** a charge path exists that ends within seize range (3") of the assigned objective AND in base contact with the #1 ranked target:
→ `("charge", objective_position, target_unit, "melee+obj: charge onto objective hitting target")`

**Else:**
→ `("rush", objective_position, None, "melee+obj: rushing toward objective")`

*Implementation note:* "charge path that ends within seize range of objective" requires checking that the charge endpoint (base contact with target) is within 3" of the objective marker. This is a geometric check, not a pathfinding change — if the target happens to be near the objective, the standard charge resolution will land the unit there.

---

#### Case 2: `ranged_aggressive` + `objective` / `ranged_kite` + `objective` / `ranged_hold` + `objective`

All three ranged engagement types behave identically when priority is `objective`:

**If** the assigned objective is within advance range (6") OR the unit is already on the objective:
→ Advance onto the objective (or hold if already there). Shoot the highest-ranked target that is in weapon range from the objective position.
→ `("advance", objective_position, None, "ranged+obj: advancing onto objective")` or `("hold", None, None, "ranged+obj: holding on objective")`

**If** the assigned objective is NOT within advance range:
→ `("rush", objective_position, None, "ranged+obj: rushing toward objective")`

*Note:* If the unit advances/holds on the objective but no enemies are in weapon range, it still moves to/stays on the objective and simply doesn't shoot. The objective is the priority.

---

#### Case 3: `melee` + `killer`

**If** the #1 ranked target is in charge range:
→ `("charge", target_position, target_unit, "melee+killer: charging priority target")`

**Else:**
→ `("rush", target_position, None, "melee+killer: rushing toward priority target")`

*Note:* Unlike the heuristic agent, this does not walk down the ranking to find a chargeable target. The model selected this target; if it's not in charge range, the unit rushes toward it. The model learns to account for reachability through reward.

---

#### Case 4: `ranged_hold` + `killer`

Shoot the highest-ranked target that is in weapon range from the current position. Do not move.

**If** any ranked target is in range:
→ `("hold", None, None, "ranged_hold+killer: shooting priority target in range")`

**If** no ranked target is in range:
→ `("hold", None, None, "ranged_hold+killer: no target in range, holding")`

*Note:* This is the only case where the execution logic walks the target ranking to find a reachable target — because "hold" means the unit can't move to improve its position, so shooting the best *available* target is strictly better than shooting nothing.

---

#### Case 5: `ranged_aggressive` + `killer`

**If** the #1 ranked target is within advance (6") + max weapon range:
→ `("advance", toward_target, None, "ranged_aggressive+killer: advancing and shooting")`

**If** the #1 ranked target is NOT within advance + weapon range:
→ `("rush", target_position, None, "ranged_aggressive+killer: rushing toward target")`

---

#### Case 6: `ranged_kite` + `killer`

**If** the #1 ranked target is within advance (6") + max weapon range:
→ Find the point P such that:
  - P is within advance range (6") of the unit's current position
  - The #1 ranked target is within max weapon range of P
  - The distance from P to the nearest enemy unit is maximised

→ `("advance", P, None, "ranged_kite+killer: kiting to optimal firing position")`

**If** the #1 ranked target is NOT within advance + max weapon range:
→ Find the point P such that:
  - P is within rush range (12") of the unit's current position
  - The distance from P to the nearest enemy unit is maximised

→ `("rush", P, None, "ranged_kite+killer: retreating to safety")`

*Note:* In the retreat case, there is no attraction toward the target — the unit is purely maximising distance from enemies. This is intentional; if the model wanted to close on the target it would have picked `ranged_aggressive`.

---

#### Case 7: No enemies alive

→ `("hold", None, None, "no enemies alive")`

---

### 2.4 Target Selection for Shooting

After movement is resolved, the game loop currently calls `pick_target()` to determine which enemy to shoot. For ML-driven units, this is replaced by reading the target ranking:

```python
def pick_target_from_ranking(
    attacker: UnitState,
    enemies: list[UnitState],
    target_ranking: list[int],
) -> UnitState | None:
    """Walk the ML target ranking and return the highest-ranked enemy in weapon range."""
    for slot_idx in target_ranking:
        if slot_idx >= len(enemies):
            continue
        enemy = enemies[slot_idx]
        if enemy.models_alive <= 0:
            continue
        can_shoot, _, _ = evaluate_target(attacker, enemy)
        if can_shoot:
            return enemy
    return None
```

This replaces `pick_target_killer`, `pick_target_holder`, `pick_target_clearer` for ML units. The heuristic versions remain for heuristic-agent games.

### 2.5 Integration with Game Loop

In `game.py`, the ML tactical path currently calls `apply_tactical_model` → sets unit attributes → falls through to `choose_action_and_goal` and `pick_target`. The new flow:

1. Call the new model forward pass → get `TacticalModelOutput`
2. Decode: argmax (eval) or sample (training) each head to get `(unit_idx, priority, objective_idx, target_ranking, engagement)`
3. Call `execute_ml_decision(...)` → get `(action, goal, charge_target, reason)`
4. Call `pick_target_from_ranking(...)` for shooting resolution
5. Execute movement and combat as normal

The `UnitState` fields `ai_role`, `combat_preference`, `movement_stance`, and `assigned_objective` are no longer set by the ML path. The execution function reads the model outputs directly. These fields remain on `UnitState` for the heuristic agent path.

---

## Step 3: Monte Carlo Planning (Eval-Time Only)

### 3.1 Overview

At evaluation time, instead of taking the model's argmax decisions directly, run a Monte Carlo search: sample candidate action tuples, simulate each forward through the game, and pick the candidate with the best expected outcome. This is applied per-activation — every time the ML agent needs to decide what to do with a unit, it searches before committing.

Planning is NOT used during training. Training uses the standard sample-from-policy approach so that gradients remain well-defined.

### 3.2 State Snapshot and Restore

Planning requires forking the game state to simulate hypothetical actions. Implement lightweight snapshot/restore for the mutable game objects:

**UnitState snapshot:** Copy all mutable fields — `models_alive`, `wounds_per_model` (list copy), `shaken`, `morale_checked`, `activated`, `fatigued`, `positions` (list copy), `weapons_per_model` (list-of-lists copy), `_removed_positions` (list copy), `ai_role`, `combat_preference`, `assigned_objective`, `movement_stance`, `owner`, `hero_model_index`. The `unit` field (ResolvedUnit) is frozen/immutable and shared.

**Board snapshot:** Copy `occupancy` (bytearray copy) and `objective_control` (list copy).

Implement as:
```python
def snapshot_game_state(
    units_a: list[UnitState],
    units_b: list[UnitState],
    board: Board,
) -> GameSnapshot:
    ...

def restore_game_state(snapshot: GameSnapshot) -> tuple[list[UnitState], list[UnitState], Board]:
    ...
```

Use shallow copies of immutable data and explicit copies of mutable containers. Avoid `deepcopy` — it's slow and copies the frozen ResolvedUnit objects unnecessarily.

### 3.3 Planning Loop

```
function plan_activation(model, game_state, friendly_units, enemy_units, ...):
    # 1. One trunk pass
    h, unit_logits, value = model.trunk_and_unit_head(state_vec, alive_mask)

    # 2. Select top-K candidate units by probability
    unit_probs = softmax(unit_logits)
    candidate_units = top_k(unit_probs, K_UNITS)  # e.g. K_UNITS = 6

    # 3. For each candidate unit, query conditioned heads
    for unit_idx in candidate_units:
        per_unit_output = model.forward_per_unit(h, state_vec, unit_idx, enemy_alive_mask)

        # Sample C action tuples from the per-unit distributions
        for c in range(C_SAMPLES_PER_UNIT):  # e.g. C_SAMPLES_PER_UNIT = 4
            priority = sample(per_unit_output.priority_logits)
            objective = sample(per_unit_output.objective_logits)
            target_ranking = argsort(per_unit_output.target_logits, descending=True)
            engagement = sample(per_unit_output.engagement_logits)

            # 4. For each candidate, run M rollouts
            total_value = 0
            for m in range(M_ROLLOUTS):  # e.g. M_ROLLOUTS = 4
                snapshot = snapshot_game_state(...)
                apply_candidate_action(snapshot, unit_idx, priority, objective, target_ranking, engagement)
                # Simulate forward N activations using base policy (no planning)
                simulate_forward(snapshot, N_LOOKAHEAD_ACTIVATIONS, model)  # e.g. N = 4
                # Evaluate resulting state
                total_value += model.value_head(encode_state(snapshot))
            
            avg_value = total_value / M_ROLLOUTS
            candidates.append((unit_idx, priority, objective, target_ranking, engagement, avg_value))

    # 5. Pick the best candidate
    best = max(candidates, key=lambda c: c[-1])
    return best
```

### 3.4 Default Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `K_UNITS` | 6 | Candidate units to evaluate (from top of unit_selection distribution) |
| `C_SAMPLES_PER_UNIT` | 4 | Action tuples sampled per candidate unit |
| `M_ROLLOUTS` | 4 | Rollouts per candidate (for dice averaging) |
| `N_LOOKAHEAD` | 4 | Activations simulated forward before value-head evaluation |

Total work per activation: 6 × 4 × 4 = 96 rollouts, each simulating 4 activations. At ~0.1ms per activation, that's ~38ms per planning decision. A full game (~40 activations for the planning side) takes ~1.5 seconds. These parameters are tunable at eval time.

### 3.5 Forward Simulation

During rollouts, both sides use the base policy (argmax, no planning) for all decisions after the candidate action. The forward simulation:

1. Applies the candidate action to the current activation
2. Alternates activations between sides for N_LOOKAHEAD steps
3. The opponent side uses the same model with argmax decoding (or a separate opponent model/heuristic)
4. After N_LOOKAHEAD activations, calls `model.value_head(encode_state(...))` on the resulting state

If the game ends before N_LOOKAHEAD activations are exhausted (all enemies destroyed, or round 4 ends), use the actual game outcome (+1 win, -1 loss, 0 draw) instead of the value head.

### 3.6 Integration

Planning wraps the existing `apply_tactical_model` call site. Add a flag to `simulate_game`:

```python
def simulate_game(..., ml_planning: bool = False, planning_params: dict | None = None):
```

When `ml_planning=True`, each ML activation calls `plan_activation()` instead of the direct forward-pass-and-decode path. The planning parameters (K, C, M, N) are passed via `planning_params` with the defaults from §3.4.

Planning is eval-only. Training always uses the non-planning path with sampling.

---

## Step 4: Training Changes

### 4.1 Sequential Sampling

During training, actions are sampled from each head in sequence, with conditioning applied at each step. The log-probability of the full action is the sum of log-probs at each stage:

```
log_prob = log_prob(unit_idx | h)
         + log_prob(priority | h, unit)
         + log_prob(objective | h, unit, priority)    # always included
         + log_prob(target_ranking | h, unit, priority)  # see §4.2
         + log_prob(engagement | h, unit, priority, target, objective)
```

### 4.2 Target Ranking Log-Probability

The target head outputs a categorical distribution over 10 enemy slots. During training, we sample a single target slot (the unit's primary target — the one it will attempt to engage). The log-prob of this sample is included in the REINFORCE loss.

We do NOT compute a log-probability over the full ranking (that would be a Plackett-Luce model, which is more complex). The model is trained to put high probability on the best target; the ranking emerges from the learned distribution at eval time.

### 4.3 Conditioning with Detached Samples

When computing the forward pass for training, sampled discrete choices are detached before being used as conditioning inputs:

```python
priority_onehot = F.one_hot(sampled_priority, NUM_PRIORITIES).float().detach()
```

This ensures gradients flow only through the REINFORCE policy gradient, not through the conditioning path. Each head's gradient depends on the reward signal, not on how its input was sampled.

### 4.4 Entropy Bonus

Compute entropy for each head and include the mean in the loss as before. The sequential structure doesn't change the entropy calculation — each head's entropy is computed independently from its own output distribution.

### 4.5 Replay Log-Probs (PPO Path)

For the PPO replay path (`replay_log_probs_batch`), the same sequential conditioning applies. Stored experience records need to include the sampled values for all five decisions (unit_idx, priority, objective, target_slot, engagement) so that conditioning can be reconstructed during replay.

Rename the existing `TacticalTrajectoryStep` class (in `ml_training.py`) to `TacticalActivationRecord` and update its fields to store the new decision format:
```python
@dataclass
class TacticalActivationRecord:
    state_vec: torch.Tensor
    alive_mask: torch.Tensor
    enemy_alive_mask: torch.Tensor
    unit_idx: int
    priority: int            # 0=objective, 1=killer
    objective: int           # 0-4
    target_slot: int         # 0-9 (primary target)
    engagement: int          # 0-3
    old_log_prob: float
    reward: float
```

---

## Migration Notes

### Backward Compatibility

- The heuristic agent path in `ai.py` is unchanged. `pick_target_killer`, `pick_target_holder`, `choose_action_and_goal` etc. remain for heuristic and evolution games.
- The old `TacticalModel` class is replaced, not extended. No need to maintain both.
- The strategic model (`ml_model.py`) is unaffected.

### Files Modified

| File | Changes |
|------|---------|
| `ml_model_tactical.py` | Replace `TacticalModel` with new conditioned-head architecture |
| `ml_integration_tactical.py` | Update `apply_tactical_model` to use new output format and `enemy_alive_mask`; add `execute_ml_decision`; add `pick_target_from_ranking`; update `apply_tactical_model_sampling` to produce `target_ranking` and route through `execute_ml_decision` instead of setting `UnitState` fields; update `InferenceRequest`, `InferenceResult`, and `decode_tactical_result` for the coroutine batching path to match the new output format |
| `ml_features.py` | No structural changes — `enemy_alive_mask` is computed in the integration layer (see §1.11) |
| `ml_training.py` | Rename `TacticalTrajectoryStep` → `TacticalActivationRecord` with new fields; update `sample_tactical_actions_no_grad` for sequential conditioned sampling; update `replay_tactical_log_probs_batch` for conditioned replay; update `_run_single_episode_tactical` to use `execute_ml_decision` + `pick_target_from_ranking` instead of `choose_action_and_goal` + `pick_target` for Player A activations, and to build/pass `enemy_alive_mask` |
| `game.py` | Wire new execution path for ML activations; remove `_batched_tactical` / `plan_round_tactical` call site; add `ml_planning` flag and planning wrapper |
| `ai.py` | No changes (heuristic path preserved) |
| `models.py` | No changes |
| `board.py` | No changes |

### Deprecated / Removed

| Item | Location | Action |
|------|----------|--------|
| `plan_round_tactical` | `ml_integration_tactical.py` | **Remove.** The batched round-planning fast path assumed all heads read from the same trunk output in parallel. The new conditioned-head architecture requires per-activation sequential decoding, which is incompatible with the pre-planning approach. Evolution games should use the per-activation path instead. |

### New Files

| File | Purpose |
|------|---------|
| `ml_planning.py` | `snapshot_game_state`, `restore_game_state`, `plan_activation`, `simulate_forward` |
