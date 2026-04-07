# Destination Pointer Head — V1 Implementation Spec

This spec adds a discrete destination pointer head to the tactical model. The pointer replaces the continuous direction (von Mises) and distance (Beta) heads with a categorical distribution over Dijkstra-reachable grid cells, selected via scaled dot-product cross-attention.

The pointer fires only on advance and rush actions. Hold and charge skip it (hold stays put, charge destination is determined by the charge target).

---

## 1. Action Sequence

**Current:**
```
h → unit_selection → move_type → direction (von Mises) → distance (Beta) → charge_target → shoot_target
```

**New:**
```
h → unit_selection → move_type → destination_pointer (advance/rush only) → charge_target → shoot_target
```

The destination pointer replaces both `direction_head` and `distance_head`. All other heads are unchanged.

---

## 2. Candidate Set Generation

### 2.1 Dijkstra Reachable Set

For advance or rush, run Dijkstra from the unit's centroid position with the appropriate movement budget:

- **Advance:** `budget = unit.advance_distance` (half of rush)
- **Rush:** `budget = unit.rush_distance`

Return the full set of reachable cells rather than the single best cell. A cell is reachable if:
- Dijkstra path cost ≤ budget (octile metric, √2 diagonal cost)
- Cell is not occupied (occupancy grid)
- Cell is not in an exclusion zone (within 1" of an enemy model), unless the unit started adjacent to enemies
- Cell is within board boundaries

This requires a new C extension function (§6).

**Empty set fallback:** The unit's current centroid position is always included as candidate index 0, even if it would normally be filtered by the occupancy grid (the unit occupies its own cells). This guarantees the candidate set is never empty, preventing undefined behaviour in softmax, and gives the model an explicit "stay put" option through the advance/rush path. The `compute_destination_candidates` function (§5.1) unconditionally prepends the centroid after receiving the Dijkstra results.

### 2.2 Candidate Set Size

Typical sizes: ~100 candidates for a 6" advance, ~400 for a 12" rush. Near board edges or surrounded by obstacles, sets may be smaller.

### 2.3 Padding for Batched Training

Pad all candidate sets to a fixed maximum size `MAX_DEST_CANDIDATES = 512`. Candidates beyond the actual set are masked with `-inf` before softmax. The padding constant is chosen to accommodate the largest possible rush (12" on an open board ≈ 409 cells) with margin for safety.

Store per-activation:
- `dest_candidates`: `(MAX_DEST_CANDIDATES, 2)` int array of (col, row) coordinates, zero-padded
- `dest_mask`: `(MAX_DEST_CANDIDATES,)` bool array, True for valid candidates
- `dest_features`: `(MAX_DEST_CANDIDATES, DEST_FEATURE_DIM)` float array, zero-padded
- `dest_selected_idx`: int index into the candidate array (the sampled/argmax choice)

---

## 3. Per-Hex Features

Each candidate hex is featurized with `DEST_FEATURE_DIM = 75` features, grouped into four categories. All spatial features use the model's egocentric coordinate frame (Player B positions are flipped, matching the existing encoding in `ml_features.py`).

### 3.1 Egocentric Spatial (5 features)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | dx | Signed column offset from unit centroid, normalised by move budget |
| 1 | dy | Signed row offset from unit centroid, normalised by move budget |
| 2 | norm_dist | Euclidean distance / move budget (0.0 = at unit, 1.0 = max range) |
| 3 | sin_angle | sin(atan2(dy, dx)) — direction from unit to hex |
| 4 | cos_angle | cos(atan2(dy, dx)) — direction from unit to hex |

### 3.2 Objective Proximity (10 features)

For each of the 5 objectives (centre, my-side, enemy-side, my-home, enemy-home):

| Offset | Feature | Description |
|--------|---------|-------------|
| +0 | obj_dist | Euclidean distance from candidate hex to objective, normalised by `BOARD_DIAG` |
| +1 | obj_seize | Binary: 1.0 if distance ≤ `OBJ_SEIZE_RANGE` (3"), else 0.0 |

Features at indices 5–14.

### 3.3 Offensive Value (10 features)

For each of the 10 enemy slots, the expected ranged damage fraction the active unit would inflict on that enemy if shooting from this candidate hex. Looked up from the precomputed `ranged_matchups` table at the range bucket corresponding to the Euclidean distance from the candidate hex to the enemy's current position. Scaled by `unit.models_alive / unit.unit.models`.

Dead/absent enemy slots are zero-padded.

Features at indices 15–24 (one value per enemy slot, same ordering as the trunk's enemy block).

This is target-head-neutral: the pointer sees offensive potential against *all* enemies simultaneously, and the shoot target head (which fires after the pointer) can then pick the best target from the selected position. No circular dependency with the shoot head.

### 3.4 Per-Enemy Threat Features (50 features)

For each of the 10 enemy slots, 5 features encoding the threat that enemy poses to a unit standing on this hex. Dead/absent enemy slots are zero-padded (the model learns to ignore them trivially).

| Offset per enemy | Feature | Description |
|------------------|---------|-------------|
| +0 | ranged_damage | Expected damage fraction if this enemy shoots the hex from its current position. Looked up from precomputed `ranged_matchups` at the appropriate range bucket for the distance from hex to enemy. Scaled by `enemy.models_alive / enemy.unit.models`. |
| +1 | advance_shoot_damage | Expected damage fraction if this enemy advances toward the hex first, then shoots. Effective range = max(0, distance_to_hex - enemy.advance_distance). Same matchup table lookup at the reduced range. |
| +2 | can_charge | Binary: 1.0 if the enemy's charge range (rush_distance + 2") reaches this hex from its current position, else 0.0. |
| +3 | melee_damage | Expected melee damage fraction if this enemy charges the hex. Uses precomputed `melee_matchups`. 0.0 if `can_charge` is 0. Scaled by `enemy.models_alive / enemy.unit.models`. |
| +4 | has_activated | Binary: 1.0 if this enemy has already activated this round (reduced future threat), else 0.0. Same value for all candidate hexes — but having it per-enemy-per-hex allows the model to learn threat discounting directly in the hex embedding MLP. |

Features at indices 25–74 (enemy slot 0 at 25–29, slot 1 at 30–34, ..., slot 9 at 70–74).

Enemy slot ordering is consistent with the trunk's unit encoding: slot i in the threat features corresponds to the same enemy as slot i in the trunk's enemy block. This is stable within a game — the model does not need to learn slot-identity correspondence.

Total per-hex features: **75**.

### 3.5 Feature Computation

All features are computed from data already available at inference time: the precomputed ranged/melee matchup tables (`ranged_matchups`, `melee_matchups`), unit positions, objective positions, and the `activated` flag per enemy. No new simulations are required.

The feature computation function should operate on the full padded candidate array and return a `(MAX_DEST_CANDIDATES, DEST_FEATURE_DIM)` tensor. Invalid candidates (beyond the mask) get zeros.

---

## 4. Model Architecture Changes

### 4.1 New Components in `TacticalModel`

```python
DEST_FEATURE_DIM = 75
DEST_EMBED_DIM = 64
MAX_DEST_CANDIDATES = 512

# Per-hex feature embedding MLP
self.dest_embed = nn.Sequential(
    nn.Linear(DEST_FEATURE_DIM, 64),
    nn.ReLU(),
    nn.Linear(64, DEST_EMBED_DIM),
)

# Query projection: trunk + unit_feat + move_onehot → DEST_EMBED_DIM
self.dest_query_proj = nn.Linear(TRUNK_WIDTH + TACTICAL_UNIT_FEATURES + NUM_MOVE_TYPES, DEST_EMBED_DIM)
```

### 4.2 Removed Components

Remove `direction_head` and `distance_head`. Remove all von Mises and Beta distribution logic from the model.

### 4.3 Forward Pass (Destination Pointer)

**Single-activation inference (sampling, planning):** The pointer fires after `move_type_head` only when move_type is advance or rush. For hold and charge, the pointer is skipped entirely — no candidate set is computed and `dest_logits` is None.

**Batched forward pass (PPO replay, batched argmax):** The pointer always runs for all activations in the batch, regardless of move_type. Hold/charge activations have all-zero `dest_features` and an all-False `dest_mask`, which produces meaningless logits that are masked to `-inf`. The destination log-prob and entropy for these activations are set to 0 when computing the loss. This avoids the complexity of splitting the batch into pointer/non-pointer subsets for negligible compute savings.

```python
# dest_features: (batch, MAX_DEST_CANDIDATES, DEST_FEATURE_DIM)
# dest_mask:     (batch, MAX_DEST_CANDIDATES) bool

# Embed candidate hexes
dest_keys = self.dest_embed(dest_features)           # (batch, MAX_DEST_CANDIDATES, 64)

# Build query from trunk context + unit features + move type
query_input = torch.cat([h, unit_features, move_onehot], dim=-1)
dest_query = self.dest_query_proj(query_input)        # (batch, 64)

# Scaled dot-product attention
scale = DEST_EMBED_DIM ** 0.5
dest_logits = (dest_query.unsqueeze(1) @ dest_keys.transpose(-1, -2)).squeeze(1) / scale
# dest_logits: (batch, MAX_DEST_CANDIDATES)

# Mask invalid candidates
dest_logits = dest_logits.masked_fill(~dest_mask, float('-inf'))

# Guard: for hold/charge rows where dest_mask is all-False, all logits are -inf.
# log_softmax produces NaN for these rows. Before computing log-prob, build a
# boolean mask of which rows are advance/rush (have valid candidates) and zero
# out the destination log-prob contribution for hold/charge rows.
has_dest = dest_mask.any(dim=-1)  # (batch,) — True for advance/rush
```

During sampling: `dest_idx = Categorical(logits=dest_logits).sample()`

During argmax (eval/evolution): `dest_idx = dest_logits.argmax(dim=-1)`

The selected hex coordinates are looked up from `dest_candidates[dest_idx]`.

### 4.4 Two-Pass Inference (Shoot Target Conditioning)

The destination pointer's features are fully target-independent — offensive value features (§3.3) encode damage against all enemies, not a selected target. The pointer runs in a single pass with no circular dependency.

The two-pass structure is retained solely for the shoot target head, which needs `post_move_rel` computed from the selected destination:

- **Pass 1:** Run trunk → unit selection → move type → destination pointer → charge target. The pointer selects a destination hex using features that depend only on board state, not on any downstream head output.
- **Pass 2:** From the selected destination hex, compute `post_move_rel` (30 floats: sin/cos/dist per enemy from the selected hex position). Feed to shoot_target_head as before. The shoot head is unchanged.

### 4.5 `TacticalModelOutput` Changes

```python
@dataclass
class TacticalModelOutput:
    unit_logits: torch.Tensor | None        # (10,)
    move_logits: torch.Tensor               # (4,)
    dest_logits: torch.Tensor | None        # (MAX_DEST_CANDIDATES,) — None for hold/charge
    charge_target_logits: torch.Tensor      # (10,)
    shoot_target_logits: torch.Tensor       # (10,)
    value: torch.Tensor                     # scalar
```

Removed: `direction_params`, `distance_params`.

---

## 5. Integration Layer Changes (`ml_integration_tactical.py`)

**Coordinate frame convention:** Candidate sets and selected destinations are in **game-space** (col, row) throughout the integration layer and simulator interface. The Player B flip to model-space happens only inside `compute_destination_features` when computing egocentric and relational features. This matches the existing pattern where `post_move_rel` is computed in model-space but positions are stored and passed to `execute_movement` in game-space.

### 5.1 Candidate Set Computation

New function `compute_destination_candidates`:

```python
def compute_destination_candidates(
    unit: UnitState,
    move_type: int,
    board: Board,
    enemy_positions: set[tuple[int, int]],
    player: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute padded destination candidate set.

    Returns:
        candidates: (MAX_DEST_CANDIDATES, 2) int array of (col, row) in game-space
        mask: (MAX_DEST_CANDIDATES,) bool array
    """
```

This calls the new C extension function (§6) to get the Dijkstra reachable set, filters by occupancy and exclusion zones, pads to `MAX_DEST_CANDIDATES`, and returns.

### 5.2 Per-Hex Feature Computation

New function `compute_destination_features`:

```python
def compute_destination_features(
    candidates: np.ndarray,            # (MAX_DEST_CANDIDATES, 2)
    mask: np.ndarray,                  # (MAX_DEST_CANDIDATES,)
    unit: UnitState,
    unit_slot: int,                    # friendly slot index (for matchup table lookup)
    player: str,
    enemy_units: list[UnitState],
    enemy_alive_mask: np.ndarray,
    friendly_ranged_matchups: np.ndarray,  # precomputed: unit's damage vs enemies (offensive)
    enemy_ranged_matchups: np.ndarray,     # precomputed: enemies' damage vs unit (threat)
    melee_matchups: np.ndarray,            # precomputed: enemies' melee damage vs unit (threat)
    move_budget: float,
) -> np.ndarray:
    """Compute per-hex features for all candidates.

    Returns: (MAX_DEST_CANDIDATES, DEST_FEATURE_DIM) float32 array
    """
```

All coordinates are converted to model-space (Player B flipped) before computing egocentric features, matching the trunk's coordinate frame.

### 5.3 Post-Move Rel Computation

After the pointer selects a destination hex, compute `post_move_rel` from the selected hex's game-space coordinates. This replaces the `compute_post_move_position` call that previously used sampled angle + distance.

```python
# Selected hex in game-space
dest_col, dest_row = candidates[dest_idx]
# Convert to model-space for post_move_rel
dest_x, dest_y = float(dest_col), float(dest_row)
if player == "B":
    dest_x = _flip_x(dest_x)
    dest_y = _flip_y(dest_y)
post_move_rel = compute_post_move_rel(dest_x, dest_y, enemy_positions)
```

### 5.4 `execute_decoded_decision` Changes

The `dest` parameter changes from a continuous `(float, float)` to a discrete `(int, int)` grid coordinate for advance/rush. The function already rounds to int, so the only change is that the input is exact rather than approximate.

### 5.5 Simulator Interface

The selected hex is the literal destination of the lead model. It is passed as the `goal` parameter to `execute_movement`. The existing coherency-leashed multi-model placement, pathfinding, and kite nudge logic are all unchanged. The pointer output replaces heuristic goal selection, not movement execution.

---

## 6. C Extension Changes (`_fast_core.c`)

### 6.1 New Function: `c_dijkstra_reachable_set`

```c
static PyObject* py_dijkstra_reachable_set(PyObject* self, PyObject* args);
```

**Signature:**
```c
c_dijkstra_reachable_set(
    start_col, start_row,       // unit centroid
    budget,                      // movement budget (float)
    occupancy,                   // bytearray (COLS * ROWS)
    exclusion_grid,              // bytearray (COLS * ROWS)
    enemy_positions,             // flat int32 buffer
    n_enemies,
    cols, rows,
    is_charge, flying, already_adjacent
) -> bytes
```

**Returns:** A flat `int32` buffer of `[col0, row0, col1, row1, ...]` containing all reachable, unoccupied, legal cells within the movement budget. This avoids per-element Python object allocation (400 tuples → 400 PyObject allocations). The Python wrapper reshapes the buffer into a numpy array.

**Implementation:** Reuse the existing Dijkstra loop from `py_pathfind_move`. After the exploration phase, iterate over all cells with `dist_arr[idx] < INF_COST`, apply the same occupancy/exclusion/enemy filters, and write passing cells into a pre-allocated `int32` result buffer. Return the buffer as a Python bytes object.

### 6.2 Python Wrapper (`fast_core.py`)

```python
def fast_dijkstra_reachable_set(
    start: tuple[int, int],
    budget: float,
    occupancy: bytearray,
    enemy_positions: set[tuple[int, int]],
    is_charge: bool = False,
    flying: bool = False,
    exclusion_grid: bytearray | None = None,
    cols: int = 72,
    rows: int = 48,
) -> np.ndarray:
    """Returns (N, 2) int32 numpy array of reachable (col, row) coordinates."""
```

The wrapper calls the C function, receives a flat bytes buffer, and reshapes:
```python
raw = _fast_core.c_dijkstra_reachable_set(...)
return np.frombuffer(raw, dtype=np.int32).reshape(-1, 2)
```

Fallback: pure-Python implementation using the existing Dijkstra loop in `movement.py`, modified to collect all reachable cells and return as a numpy array.

### 6.3 Register in Module Definition

Add to `FastCoreMethods[]`:
```c
{"c_dijkstra_reachable_set", py_dijkstra_reachable_set, METH_VARARGS,
 "Return all reachable cells within movement budget via Dijkstra."},
```

---

## 7. Training Changes (`ml_training.py`)

### 7.1 `TacticalActivationRecord` Changes

```python
@dataclass
class TacticalActivationRecord:
    state_vec: list[float]
    alive_mask: list[bool]
    enemy_alive_mask: list[bool]
    unit_idx: int
    move_type: int
    # Destination pointer (replaces sampled_angle, sampled_distance_frac)
    dest_candidates: list[list[int]]        # (N, 2) — actual candidates (unpadded for storage)
    dest_mask: list[bool]                   # (N,) — all True (unpadded)
    dest_features: list[list[float]]        # (N, DEST_FEATURE_DIM) — unpadded
    dest_selected_idx: int                  # index into candidates
    charge_target_idx: int
    shoot_target_idx: int
    shoot_mask: list[bool]
    post_move_rel: list[float]
    old_log_prob: float
    old_value: float
```

Removed: `sampled_angle`, `sampled_distance_frac`.

Records store unpadded candidate data to minimise serialisation size. Padding to `MAX_DEST_CANDIDATES` happens at replay time.

### 7.2 Sampling (`sample_tactical_actions_no_grad`)

After move_type is sampled:
1. If move_type is advance or rush, compute candidate set via `fast_dijkstra_reachable_set`
2. Compute per-hex features via `compute_destination_features`
3. Pad to `MAX_DEST_CANDIDATES`
4. Run destination pointer forward pass → sample from categorical
5. Look up selected hex coordinates
6. Compute `post_move_rel` from selected hex
7. Continue to charge_target and shoot_target heads as before

If move_type is hold or charge, skip steps 1–4. Set `dest_candidates`, `dest_features`, `dest_mask` to empty and `dest_selected_idx` to -1.

### 7.3 Log-Probability

The total log-prob per activation becomes:

```
log_prob = log_prob(unit_idx | h)
         + log_prob(move_type | h, unit_feat)
         + log_prob(dest_idx | h, unit_feat, move_type, dest_features)   # advance/rush only
         + log_prob(charge_target | h, unit_feat, move_type)
         + log_prob(shoot_target | h, unit_feat, move_type, post_move_rel)
```

For hold/charge, the destination log-prob term is 0.

### 7.4 Batched Replay (`replay_tactical_log_probs_batch`)

For PPO replay:
1. Reconstruct padded candidate tensors from stored records: pad `dest_candidates`, `dest_features`, `dest_mask` to `(N, MAX_DEST_CANDIDATES, ...)`.
2. Run trunk and upstream heads in batch as before.
3. For the destination pointer: batch-compute embeddings and attention, mask by `dest_mask`, compute log-prob of stored `dest_selected_idx`.
4. For activations where move_type was hold/charge, the destination log-prob is 0 (masked out).
5. Downstream heads (charge, shoot) continue as before.

### 7.5 Entropy

The destination pointer is a masked categorical over variable-sized candidate sets. Entropy is computed from the masked softmax distribution:

```python
dest_probs = softmax(dest_logits.masked_fill(~dest_mask, -inf))
dest_entropy = -(dest_probs * log(dest_probs + eps)).sum(dim=-1)
```

For hold/charge activations, destination entropy is 0.

### 7.6 Entropy Tuning

Add a new entry to `EntropyTargetTuner` for the destination head. Because candidate set sizes vary widely (~100 for advance, ~400 for rush), the entropy is **normalised by ln(N_valid)** before applying the alpha coefficient. This converts the entropy to a ratio in [0, 1] (where 0 = fully collapsed, 1 = uniform) so a single learned alpha works across all candidate set sizes.

```python
normalised_dest_entropy = dest_entropy / ln(N_valid).clamp(min=1.0)
alpha_loss_dest = -alpha_dest * (normalised_dest_entropy - target_fraction)
```

- New hyperparameter: `entropy_target_dest_fraction: float = 0.25` (target 25% of max entropy, consistent with other masked categoricals)
- Learnable `log_alpha_dest` parameter
- The normalisation is applied only for the entropy tuner's alpha loss; the raw (unnormalised) entropy is still used in the policy loss as the entropy bonus term

---

## 8. Planning Changes (`ml_planning.py`)

### 8.1 Candidate Action Generation

Replace direction/distance sampling with destination pointer sampling:
1. Compute candidate set for the selected unit and move_type
2. Compute per-hex features
3. Run pointer forward pass
4. Sample destination from the categorical distribution

The rest of the planning pipeline (rollout simulation, value evaluation) is unchanged. The destination hex feeds directly into `execute_decoded_decision` as before.

### 8.2 Batched Argmax (`batched_argmax_tactical`)

Replace the direction/distance argmax decode with destination pointer argmax:
1. Per-request: compute candidate set, features, padding
2. Batch the padded feature tensors across requests
3. Run pointer attention in batch
4. Argmax per request
5. Look up selected hex coordinates

---

## 9. Diagnostics and Viewer

### 9.1 Assessment Dict

Update the assessment dictionary produced by `apply_tactical_model`:

```python
assessment = {
    ...
    'dest_selected': (col, row),                    # selected hex coordinates
    'dest_n_candidates': int,                       # size of candidate set
    'dest_top3': [(col, row, prob), ...],           # top 3 candidates by probability
    'dest_entropy': float,                          # pointer entropy
    ...
}
```

Removed: `direction_angle`, `direction_concentration`, `distance_frac`, `distance_alpha`, `distance_beta`.

### 9.2 Viewer

The viewer can optionally render the candidate set as a coloured overlay on the grid, with colour intensity proportional to the pointer's probability distribution. The selected hex is highlighted. This is a viewer-only change and not required for the core implementation.

---

## 10. Files Modified

| File | Changes |
|------|---------|
| `_fast_core.c` | Add `c_dijkstra_reachable_set` function |
| `fast_core.py` | Add `fast_dijkstra_reachable_set` wrapper with Python fallback |
| `build_fast_core.py` | No changes (auto-discovers functions from module definition) |
| `ml_model_tactical.py` | Remove `direction_head`, `distance_head`. Add `dest_embed`, `dest_query_proj`. Update `TacticalModelOutput`. Update `forward` / `forward_per_unit`. |
| `ml_integration_tactical.py` | Add `compute_destination_candidates`, `compute_destination_features`. Update `apply_tactical_model`, `decode_tactical_result`, `batched_argmax_tactical`. Remove von Mises / Beta decode helpers. Update `post_move_rel` computation to use selected hex. |
| `ml_training.py` | Update `TacticalActivationRecord` fields. Update `sample_tactical_actions_no_grad`. Update `replay_tactical_log_probs_batch` for padded pointer replay. Add destination entropy to `EntropyTargetTuner`. Remove direction/distance log-prob and entropy computation. |
| `ml_planning.py` | Update candidate action generation to use pointer sampling instead of direction/distance sampling. Update `batched_argmax_tactical` calls. |
| `ml_features.py` | Add `DEST_FEATURE_DIM`, `MAX_DEST_CANDIDATES` constants. Optionally add `compute_destination_features` here (or in `ml_integration_tactical.py` — placement TBD based on import graph). |

### New Constants

```python
# ml_features.py or ml_model_tactical.py
DEST_FEATURE_DIM = 75
DEST_EMBED_DIM = 64
MAX_DEST_CANDIDATES = 512
```

---

## 11. Scope Boundary

**In scope:**
- Destination pointer head for advance/rush actions (leader model positioning)
- C extension for Dijkstra reachable set
- Padded batched training replay
- Entropy tuning for the pointer head
- Planning integration

**Out of scope (future work):**
- Per-model positioning within the unit (V2: lightweight adjustment head per non-leader model)
- Cross-attention threat aggregation replacing flat per-enemy features (V2)
- Hierarchical coarse-then-fine pointer for higher grid resolution (V2+)
- Grid resolution increase (0.2" cells)
- Hex grid migration (rejected)
- Spellcasting heads

---

## 12. Migration Notes

### 12.1 Heuristic Agent

The heuristic agent in `ai.py` is unaffected. It does not use the tactical model.

### 12.2 Evolution Games

Evolution games using `batched_argmax_tactical` will use the pointer's argmax path. Performance may initially regress until the pointer is trained.

### 12.3 Training

This is a breaking architecture change. Training restarts from scratch.
