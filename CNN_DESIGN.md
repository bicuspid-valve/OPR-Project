# Spatial-CNN Design

Design document for wiring a spatial convolutional encoder into the tactical
model so it can reason about *where* terrain, friendly units, and enemy units
sit on the board — not just per-unit MLP encodings. The CNN output is fused
into the trunk so all downstream heads (unit selection, move type, dest
pointer, charge, shoot, value, dest features) benefit.

---

## Goal

Today the trunk's only spatial signal is the per-candidate terrain one-hot at
indices `[76:82]` of the destination feature block (the cover/movement type of
the destination hex itself). The trunk has **no global view of the board** —
no idea where walls cluster, no idea where the enemy formation sits, no way
to reason about "all the not-yet-activated enemies are massed on the right."

A small spatial CNN over an 11-channel board tensor fixes that. The CNN
output (~64-dim embedding) is concatenated into the trunk's aggregation tensor
before the stem MLP, so it propagates to every head.

---

## Prerequisite: flow inversion at POST_DEST

This spec depends on an architectural change to how POST_DEST is computed.
Today, the integration layer runs the full inference chain (PRE_SELECT →
POST_SELECT → POST_MOVETYPE → dest sample → POST_DEST encode → shoot/charge
logits) *before* the engine executes the chosen move. The POST_DEST encode
uses [project_post_move_unit_state](ml_integration_tactical.py): a centroid-
delta translation of the acting unit's models that approximates where they'd
land. The real per-model landing cells from the engine's stateful pathfinder
([execute_movement](movement.py)) are never seen by the model.

**Under this spec the order is inverted.** Once the dest is sampled, the
inference layer triggers engine execution mid-call, then runs POST_DEST with
the **actual** per-model positions the engine produced:

```
[was]                                  [is]
PRE_SELECT → POST_SELECT               PRE_SELECT → POST_SELECT
  → POST_MOVETYPE → dest sample          → POST_MOVETYPE → dest sample
  → POST_DEST encode using               → ENGINE EXECUTES MOVE
       project_post_move_unit_state       → POST_DEST encode using
  → shoot/charge logits                       engine-truth positions
  → return                              → shoot/charge logits
ENGINE EXECUTES MOVE                     → return
ENGINE continues (combat)              ENGINE continues (combat)
```

Both the per-unit feature path (`encode_state_tactical` at POST_DEST) and
the CNN's POST_DEST plane consume the same `UnitState.positions` snapshot
captured immediately after `execute_movement` / `execute_charge_movement`
(and, for charges, after `execute_counter_charge` — see below). Internal
consistency at POST_DEST: every view of post-move state agrees.

**Consequences:**

- `project_post_move_unit_state` is deleted. Three callers in
  `ml_integration_tactical.py` updated to use engine-truth positions.
- The integration layer owns the engine call between dest sampling and
  POST_DEST encode. `_episode_tactical_generator` in
  `ml_training/collection.py` either yields the engine call back to the
  outer driver, or hands the engine handle to the integration layer
  directly. (Implementation detail — either works.)
- Batched sampling (`_batched_sample_tactical_no_grad`) interleaves a per-
  game engine call between the phase-1 batched forward (unit / move / dest)
  and the phase-2 batched forward (POST_DEST). Per-game engine work in a
  Python loop between the two GPU launches; cost is dominated by GPU time,
  not the loop.
- Charge activations change semantically: today
  `state_vec_post = state_vec` (POST_DEST sees pre-charge state). Under
  this spec POST_DEST sees post-charge + post-counter-charge state, before
  any impact damage. The shoot head outputs remain ignored for charges, so
  this only affects the value head and charge head's view at POST_DEST —
  arguably more correct than before.
- Per-candidate dest features (the dest pointer's "how much damage if I
  went here" scored for thousands of candidates) stay on the TERRAIN_SPEC
  §5.4 single-point shooter proxy. Engine truth would require running the
  pathfinder per candidate, which is intractable. Approximation at scoring,
  truth at re-encode: the dest pointer commits to a dest based on
  approximate scoring, then engine truth enters the picture for the
  committed dest only.

The CNN sections below assume this flow inversion is in place.

---

## Scope of spatial reasoning

This CNN is for **coarse global spatial pattern recognition** — formation
shapes, where clusters sit, terrain density, "is the enemy massed on one side
of the board." It is *not* a replacement for the analytical line-of-sight and
cover pipeline, and architectural choices below (resolution, pooling, output
size) are made against the coarse-pattern goal, not against LoS substitution.

Things the CNN should be expected to represent:

- Cluster locations and shapes at the 5–15 cell scale.
- Terrain density and rough wall placement.
- Side-of-board imbalances and formation gestalt ("enemies massed on the
  right", "friendly line bunched in the centre").
- Local "unit adjacent to wall" / "wall on the north side of this unit"
  relationships within the post-conv receptive field (~9 cells).

Things the CNN should *not* be expected to represent — these stay in the
analytical pipeline:

- True per-shot LoS from an arbitrary shooter to an arbitrary target. That's
  a global ray query along a specific segment, handled by `terrain_los.py`.
- Per-target cover state for shooting. Already folded into the cover-aware
  `expected_wound_frac` (TERRAIN_SPEC §5.4 / §5.5) the shoot head consumes.
- "Is unit U hiding from enemy E behind wall W." A relational query along a
  specific segment; the CNN has no mechanism to express it and doesn't need
  to — the analytical pipeline already answers it exactly.
- Sub-cell-precision LoS edge cases ("barely slipped into sight"). The CNN
  operates at full-cell resolution.

In short: the CNN gives the trunk a *gestalt* view of the board. Precise
geometric queries are answered analytically and arrive at the trunk via
existing per-unit and per-destination features.

**Approximation at scoring, truth at re-encode.** Under the flow inversion
described above, every consumer at POST_DEST — the CNN plane *and* the per-
unit feature path that feeds the trunk — sees the engine's actual per-model
landing cells. There is no internal mismatch at POST_DEST.

The per-candidate dest-feature pipeline that fed the dest pointer head, by
contrast, uses the candidate dest as a single shooter-position proxy for
every model in the acting unit (TERRAIN_SPEC §5.4, see also
[ml_integration_tactical.py:454–455](ml_integration_tactical.py)). That
approximation is intentional and stays — running the real pathfinder per
candidate dest is intractable when there can be thousands of candidates per
activation. So: the dest pointer commits based on approximate scoring;
once it commits, the engine runs the real movement; from that point on,
every model view sees truth.

---

## Channel layout (12 channels, shape `(12, 48, 72)`)

| Ch | Content | Recompute frequency |
|----|---------|---------------------|
| 0 | Open (no terrain piece) | Once per game (cache on `Board`) |
| 1 | Difficult + Sheltering | Once per game |
| 2 | Difficult + Obscuring | Once per game |
| 3 | Open + Sheltering | Once per game |
| 4 | Open + Obscuring | Once per game |
| 5 | Impassible + Blocking | Once per game |
| 6 | Friendly, not yet activated (1.0 at each alive model position) | Per activation |
| 7 | Friendly, already activated | Per activation |
| 8 | Enemy, not yet activated | Per activation |
| 9 | Enemy, already activated | Per activation |
| 10 | Friendly shaken (overlay; both NA and AC) | Per activation |
| 11 | Enemy shaken (overlay; both NA and AC) | Per activation |

Channels 0–5 are a **closed enum of the six valid (movement, cover) tuples
the game supports**. Exactly one of channels 0–5 is hot per cell. Any other
combination is invalid input and the helper raises rather than silently
mapping it. New piece types that introduce a 7th combination require an
explicit schema bump (new channel + checkpoint break).

Channels 6–11 are unit overlays. Binary `1.0` marks at every alive model
position. Multiple models in a unit each contribute their own cell, so a
tight cluster shows up as multiple adjacent hot cells; the CNN's first conv
layer captures the clustering. The engine maintains a one-model-per-cell
invariant ([board.py](board.py) — `occupancy` is a binary `bytearray`, not
a count, and `execute_movement` enforces it via collision resolution), so
binary marks faithfully represent unit state and no per-cell counting is
needed.

The shaken channels are side-keyed (10 = friendly, 11 = enemy) because the
tactical valence of a shaken cluster is opposite by side — shaken friendly
NA is a wasted upcoming slot; shaken enemy NA is a weakened opponent. A
single overlay channel would force the CNN to recover side via correlation
with channels 6–9; an extra channel for ~144 extra conv1 params is a much
better trade.

### Player perspective

Rendered in the acting model's model-space — Player B's view is the 180° flip
of Player A's. Cache **two** terrain bases on `Board`: `terrain_planes_A` and
`terrain_planes_B`. Unit/shaken channels (6–10) are rendered live in player
coordinates each activation.

### Post-move re-render

Between phases POST_MOVETYPE and POST_DEST, the acting unit's models move
from start positions to their engine-determined landing cells. Only channels
6/7 (and possibly 10/11) change. Cheapest implementation: full re-render of
channels 6–11 — ~120 grid writes, negligible cost.

**Post-move positions are the engine's actual output, captured immediately
after `execute_movement` / `execute_charge_movement` (and
`execute_counter_charge` for charges) returns.** The flow inversion described
in "Prerequisite: flow inversion at POST_DEST" makes this possible — the
engine runs *before* POST_DEST encode, not after — and the same snapshot
feeds both the CNN plane and the per-unit feature path.

The dest pointer head selects a single `(dest_col, dest_row)` *goal*, but
the per-model landing cells the engine produces are not a translation of the
start formation: the leader pathfinds toward the goal under movement budget
+ terrain + occupancy, subsequent models target a leashed point clamped to
coherency distance of already-placed teammates, and collisions get shoved to
adjacent free cells ([movement.py:254–365](movement.py)). There is no
closed-form way to reconstruct the resulting formation from `(unit_idx,
dest)` alone — capturing from the engine is the only correct option.

The sampler stores the snapshot on the activation record; the replay path
reads it back directly. See the replay section and risk #2 for capture-point
timing.

---

## Architecture

### `TerrainCNN` module (new, in `ml_model_tactical.py`)

```python
class TerrainCNN(nn.Module):
    def __init__(self, in_channels=12, out_dim=64, n_heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),  # 48x72 → 24x36
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),  # 24x36 → 12x18
            nn.ReLU(),
        )
        # Learned 2D positional embedding on the post-conv map so that
        # otherwise-identical conv features at different board locations are
        # distinguishable to the attention pool.
        self.pos_embed = nn.Parameter(torch.zeros(1, 32, 12, 18))
        nn.init.normal_(self.pos_embed, std=0.02)
        # Multi-query attention pool: each of n_heads learned queries scores
        # all 216 spatial tokens; softmax-weighted sums produce n_heads
        # 32-dim summaries that are concatenated and projected to out_dim.
        # Different queries can specialise (e.g. "where are enemy clusters",
        # "where is dense terrain", "where is the friendly formation centroid").
        self.n_heads = n_heads
        self.queries = nn.Parameter(torch.empty(n_heads, 32))
        nn.init.normal_(self.queries, std=0.02)
        self.out_proj = nn.Linear(n_heads * 32, out_dim)
        # Zero-init the final projection so the untrained CNN contributes 0
        # to the trunk's agg — matches repo convention for new heads (see
        # phase_adjustment_blocks, per_phase_value_heads, deploy heads) and
        # avoids injecting random spatial noise into the trunk at step 1.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        # planes: (B, 12, 48, 72)  →  (B, out_dim)
        feat = self.conv(planes) + self.pos_embed       # (B, 32, 12, 18)
        B = feat.shape[0]
        tokens = feat.flatten(2).transpose(1, 2)        # (B, 216, 32)
        scores = tokens @ self.queries.T                # (B, 216, n_heads)
        attn = scores.softmax(dim=1)                    # softmax over 216 spatial positions
        pooled = attn.transpose(1, 2) @ tokens          # (B, n_heads, 32)
        return self.out_proj(pooled.reshape(B, -1))     # (B, out_dim)
```

Output: `(B, 64)` per sample. Total parameters: roughly 30k (conv stack
~15k, pos_embed ~7k, queries ~0.1k, out_proj ~8k) — still trivial vs the
trunk's hundreds of thousands.

**Why attention pool over average pool.** A plain `AdaptiveAvgPool2d(1)`
collapses the 12×18 post-conv map to a single 32-vector by averaging across
all 216 spatial positions — which destroys *where* anything is. The
motivating example ("all the not-yet-activated enemies are massed on the
right") cannot survive that collapse. Multi-query attention preserves
position via the learned positional embedding and lets each query specialise
to a different "what to look at" pattern. Cost is ~2× the parameter count of
the avg-pool variant (~30k vs ~16k); both are negligible against the trunk.

CoordConv (appending normalized x/y channels and keeping avg-pool) is a
cheaper alternative that also fixes the position-loss problem, but the
single averaged summary still has to encode every region of interest into
one vector. Attention with K=4 queries gives the trunk K disentangled views
of the board for ~20k extra parameters, which is the right trade here.

### Fusion point — trunk's `agg`

In `ml_model_tactical.py` around line 453 (current `trunk()` implementation):

```python
agg = torch.cat([unit_embeds_flat, glob, terrain_embed], dim=-1)
# was (1280 + 16) = 1296, now (1280 + 16 + 64) = 1360
```

Bump the `AGG_DIM` constant by `terrain_embed_dim`. `self.stem` grows from
`Linear(1296, 512)` to `Linear(1360, 512)`.

`encode()` (the phased trunk) gets the same concat at the same point in each
of its four phases — but the CNN forward only needs to run **twice per
activation** (pre-move embed reused across PRE_SELECT / POST_SELECT /
POST_MOVETYPE; post-move embed used at POST_DEST).

### CNN forward as a separate model method (cache by contract)

To prevent callers from accidentally re-running the CNN three times,
**`encode/forward/trunk` do not take raw planes** — they take a precomputed
`terrain_embed` vector. Running the CNN is a separate, explicit step:

```python
class TacticalModel(nn.Module):
    def encode_terrain(self, planes: torch.Tensor) -> torch.Tensor:
        """Run the CNN on (B, 12, 48, 72) planes; return (B, terrain_embed_dim).
        Call this exactly twice per activation: once with the pre-move planes,
        once with the post-move planes. Pass the two embeds through phase
        encodes appropriately."""
        return self.terrain_cnn(planes)

    def forward(self, x, ..., terrain_embed: torch.Tensor | None = None) -> ...
    def trunk(self,    x,    terrain_embed: torch.Tensor | None = None) -> ...
    def encode(self,   x, phase, ..., terrain_embed: torch.Tensor | None = None) -> ...
```

`terrain_embed=None` substitutes a zero vector — back-compat for paths that
don't have planes yet, and the trivial "CNN turned off" state at fresh init
(since `out_proj` is zero-init).

Canonical call pattern for an activation:

```python
pre_embed = model.encode_terrain(pre_planes)

h0, units, ro = model.encode(state, phase=PHASE_PRE_SELECT,
                             terrain_embed=pre_embed, ...)
h1, _, _      = model.encode(state, phase=PHASE_POST_SELECT,
                             terrain_embed=pre_embed, h_prev=h0)
h2, _, _      = model.encode(state, phase=PHASE_POST_MOVETYPE,
                             terrain_embed=pre_embed, h_prev=h1)

# ... engine executes the chosen move (flow inversion section) ...

post_embed = model.encode_terrain(post_planes)
h3, units_dest, _ = model.encode(state_post, phase=PHASE_POST_DEST,
                                 terrain_embed=post_embed, h_prev=h2)
```

Two `encode_terrain` calls per activation, deliberately — the API makes it
hard to do three. The "compute once per (pre, post) pair" contract is
enforced at the call-site level rather than relying on caller discipline.

---

## Plane construction helpers — `ml_features.py`

### Update `encode_terrain_planes`

`encode_terrain_planes(board)` already produces a 6-channel `(6, ROWS, COLS)`
tensor, but its channel semantics are the wrong schema — it picks a channel
via cover-type-with-movement-fallback and includes a speculative
"impassible+sheltering" slot that no real piece occupies. Rewrite it to
emit the closed enum defined in the channel layout table:

```python
# (movement_type, cover_type) → channel
_TERRAIN_CHANNEL = {
    (MovementType.OPEN,        None):                  0,  # open
    (MovementType.DIFFICULT,   CoverType.SHELTERING):  1,
    (MovementType.DIFFICULT,   CoverType.OBSCURING):   2,
    (MovementType.OPEN,        CoverType.SHELTERING):  3,
    (MovementType.OPEN,        CoverType.OBSCURING):   4,
    (MovementType.IMPASSIBLE,  CoverType.BLOCKING):    5,
}
```

Empty cells get channel 0 hot by default. Any `(mt, ct)` combination not in
the table raises `ValueError` — silent fallback would corrupt the CNN's
input distribution. Tests should cover at least one piece of each of the
five non-empty types, and one invalid-combo failure case.

### New helpers needed

```python
def encode_unit_planes(
    unit_positions: np.ndarray,   # (N_TOTAL_UNITS, MAX_MODELS_PER_UNIT, 2) int16
                                  #   — (col, row) per model; (-1, -1) for dead / unused
    unit_flags: np.ndarray,        # (N_TOTAL_UNITS,) uint8
                                  #   — bit 0 activated, bit 1 shaken, bit 2 side (0=A, 1=B)
    player: str,                   # "A" or "B" — for model-space flip
) -> np.ndarray:
    """Returns (5, 48, 72) float32 — channels 6..10 (friendly_na, friendly_ac,
    enemy_na, enemy_ac, shaken). The helper has no concept of pre vs post-move;
    the caller passes whichever snapshot is appropriate. Post-move snapshots
    are obtained from the engine after execute_movement; pre-move snapshots
    are read from UnitState.positions at request-build time."""

def assemble_board_planes(
    terrain_base: np.ndarray,    # (6, 48, 72)
    unit_planes: np.ndarray,     # (5, 48, 72)
) -> np.ndarray:
    """Returns (12, 48, 72) by stacking terrain + units."""
```

### Cached on `Board`

At game start (after `set_terrain`), populate:

- `board.terrain_planes_A: np.ndarray (6, 48, 72)`
- `board.terrain_planes_B: np.ndarray (6, 48, 72)` (180° flip of A)

One-time cost: ~80 µs each. Lifetime: full game (terrain is immutable per
`TERRAIN_SPEC.md §2.1`).

**Training-run lifecycle.** The training loop at
[loop.py:87–101](ml_training/loop.py) loads `map_data` once at startup and
ships it to every worker via existing IPC, where each worker calls
`apply_map(Board(), map_data, ...)` at init ([collection.py:185–188](ml_training/collection.py)).
One training run = one map. The terrain-plane caches therefore live
naturally on the per-process `Board` instance and are populated as a side-
effect of `apply_map`. The main process *and* every worker get their own
copy; neither ships a plane tensor across the IPC boundary. Replay reads
the main-process copy directly.

**Future hook: multi-map training.** If curriculum / map-randomization is
ever added, the per-process single-`Board` assumption breaks. The minimal
extension is a `map_id: int` on `TacticalActivationRecord` plus a
`dict[int, (planes_A, planes_B)]` on the run context. No v1 cost; recorded
here so the future migration is a one-paragraph diff, not a redesign.

---

## Model surface changes

Add a new method `encode_terrain(planes) → embed` and thread a precomputed
`terrain_embed` kwarg (NOT raw planes) through the existing entry points.
See "CNN forward as a separate model method" above for the rationale.

```python
def encode_terrain(self, planes: torch.Tensor) -> torch.Tensor:
    """planes: (B, 12, 48, 72)  →  (B, terrain_embed_dim).
    Run the CNN once. Caller is responsible for invoking this exactly twice
    per activation (pre-move planes, post-move planes) and threading the
    resulting embeds through phase encodes."""

def forward(self, x, alive_mask=None, enemy_alive_mask=None, *,
            forced_unit_idx=None, post_move_rel=None, opponent_type=None,
            dest_features=None, dest_mask=None,
            expected_wound_frac_override=None,
            terrain_embed: torch.Tensor | None = None,  # (B, terrain_embed_dim)
) -> TacticalModelOutput
```

When `terrain_embed` is `None` → use a zero vector (back-compat / no-terrain
fallback; identical to `out_proj`'s zero-init state).

Mirror change in:

- `TacticalModel.trunk(x, terrain_embed=None)`
- `TacticalModel.encode(x, phase, ..., terrain_embed=None)` — each phase
  threads the kwarg through.
- `TacticalModel._run_conditioned_heads(...)` — already takes h/units; doesn't
  need the embed directly because the CNN feeds into the trunk's `h`.

---

## Call-site changes

### Inference (`ml_integration_tactical.py`)

Three entry points need the wiring:

- `apply_tactical_model` (non-phased)
- `_apply_tactical_model_phased`
- `apply_tactical_model_sampling`

For each:

1. Build `pre_planes = assemble_board_planes(board.terrain_planes_<player>,
   encode_unit_planes(positions_pre, flags, player))`.
2. `pre_embed = model.encode_terrain(pre_planes.unsqueeze(0))`.
3. Pass `terrain_embed=pre_embed` to PRE_SELECT / POST_SELECT /
   POST_MOVETYPE encodes (trunk + unit / move / dest pointer).
4. Engine executes the chosen move (per the flow inversion).
5. Build `post_planes` from the post-engine `UnitState.positions`.
6. `post_embed = model.encode_terrain(post_planes.unsqueeze(0))`.
7. Pass `terrain_embed=post_embed` to the POST_DEST encode (charge / shoot
   logits).

Exactly two `encode_terrain` calls per activation. The model's interior
phase encodes never see raw planes.

### Training-time sampling (`ml_training/collection.py` + `ml_training/sampling.py`)

**Request payload** — add a lightweight unit-positions sidecar to
`_TacticalInferenceRequest`, *not* the full plane tensor. Per-request the
sidecar carries the **pre-move snapshot only**; the post-move snapshot is
captured by the sampler later, after the engine has actually executed the
chosen move.

```python
unit_positions_pre: np.ndarray | None = None
    # (N_TOTAL_UNITS, MAX_MODELS_PER_UNIT, 2) int16
    # (col, row) per model; (-1, -1) for dead / unused slots.
unit_flags: np.ndarray | None = None
    # (N_TOTAL_UNITS,) uint8 — bit 0 activated, bit 1 shaken, bit 2 side
side: int                                  # already implied by request context
```

Per-request payload: ~820 bytes for the pre-move sidecar
(N_TOTAL_UNITS=20 · MAX_MODELS_PER_UNIT=10 · 2 · 2 + 20). Still tiny vs the
~162 KB of a full plane tensor. Workers already have
`board.terrain_planes_A/_B` from worker-init `apply_map`, so the central
batch consumer assembles planes from the sidecar locally.

`_batched_sample_tactical_no_grad` flow:

1. Look up per-request `terrain_planes_<side>` from the (process-local)
   `Board` cache.
2. Render pre-move unit planes for each request from `unit_positions_pre` +
   `unit_flags`; stack with the terrain base → `pre_batch: (N, 12, 48, 72)`.
3. **One CNN forward**: `pre_embed_batch = model.encode_terrain(pre_batch)`
   → `(N, terrain_embed_dim)`.
4. Pass `terrain_embed=pre_embed_batch` to `model.trunk` / `model()` for
   unit + move + charge + dest pointer pass.
5. **Engine executes the chosen move** (per the flow inversion):
   `execute_movement` / `execute_charge_movement` for each game in batch.
6. **Capture `unit_positions_post`** by snapshotting `UnitState.positions`
   across all 20 units immediately after the engine returns. Differs from
   pre only in the acting unit's rows; capturing the full 20-unit array
   keeps the renderer uniform.
7. Render post-move unit planes from `unit_positions_post` + `unit_flags`;
   stack with terrain → `post_batch: (N, 12, 48, 72)`.
8. **Second CNN forward**: `post_embed_batch = model.encode_terrain(post_batch)`.
9. Pass `terrain_embed=post_embed_batch` to the POST_DEST encode + shoot
   pass.
10. Both `unit_positions_pre` and `unit_positions_post` are written to the
    `TacticalActivationRecord` for replay.

Two `encode_terrain` calls per batch — pre-batch and post-batch — same
contract as single-game inference, just batched.

Note: charge moves use `execute_charge_movement` rather than
`execute_movement`; the post-snapshot capture point applies to both.

### Replay (`ml_training/loss.py`)

Replay re-runs the trunk on stored state — it needs the same planes that were
used at sampling time. Re-render at replay time from compact per-step state;
storing per-step plane tensors directly would cost ~162 KB/step (41 MB/256-
step minibatch) and dominate wall-clock cost.

**The terrain bases are run-context state, not per-trajectory.** Because the
training loop is single-map per run (see "Cached on `Board`" above), the
main process already holds:

- `terrain_planes_A: np.ndarray (6, 48, 72)`
- `terrain_planes_B: np.ndarray (6, 48, 72)`

Both built once at startup. `prepare_replay_data` reads them directly from
the run context; no sidecar list parallel to `all_trajectories`, no per-game
keying. The existing `side_np` array on `PreparedReplayData`
([loss.py:233–237](ml_training/loss.py)) indexes between A and B at replay
time.

**Per-step state stored on `TacticalActivationRecord`:**

```python
unit_positions_pre:  np.ndarray | None = None
    # (N_TOTAL_UNITS, MAX_MODELS_PER_UNIT, 2) int16
    # snapshot taken at request-build time, BEFORE execute_movement.
unit_positions_post: np.ndarray | None = None
    # (N_TOTAL_UNITS, MAX_MODELS_PER_UNIT, 2) int16
    # snapshot taken AFTER execute_movement / execute_charge_movement returns.
    # Differs from pre only in the acting unit's rows.
unit_flags: np.ndarray | None = None
    # (N_TOTAL_UNITS,) uint8 — activated / shaken / side bits.
    # Invariant within an activation: no model dies between pre and post
    # (movement does not cause casualties; deaths happen in shooting / melee
    # / morale, which occur outside the pre→post window). Same flags apply
    # to both snapshots.
```

~1.6 KB per step total. Captured at the moments pinned in risk #2 below, so
sampler and replay see bit-identical state.

**`prepare_replay_data` flow:**

1. Pull `terrain_A`, `terrain_B` from the run context (passed in as args,
   not embedded per-step).
2. Build `terrain_per_step` by indexing with `side_np`:
   `torch.where(side_np[:, None, None, None] == 0, terrain_A, terrain_B)`
   → `(N, 6, 48, 72)`.
3. Render pre-move unit planes vectorised across all steps from stacked
   `unit_positions_pre` / `unit_flags` → `(N, 5, 48, 72)`.
4. Concat → `terrain_planes_pre: (N, 12, 48, 72)`.
5. Render post-move unit planes the same way from `unit_positions_post`
   (no special-case override — the post snapshot already has the acting
   unit at its actual post-move positions).
6. Concat → `terrain_planes_post: (N, 12, 48, 72)`.

**New fields on `PreparedReplayData`:**

```python
terrain_planes_pre: torch.Tensor | None = None   # (N, 12, 48, 72)
terrain_planes_post: torch.Tensor | None = None  # (N, 12, 48, 72)
```

`replay_from_prepared` slices both pre and post plane tensors by minibatch
step indices, then runs **two `encode_terrain` calls per minibatch** to
produce `pre_embed_batch` and `post_embed_batch`, and threads them through
the phase encodes:

- `pre_embed_batch` → PRE_SELECT / POST_SELECT / POST_MOVETYPE encodes.
- `post_embed_batch` → POST_DEST encode.

Same two-call-per-batch contract as the sampler — replay and sampler share
the API pattern, so any drift between them is in the *inputs* (positions,
flags, side), never in the call count.

### Other call sites (lower priority, graceful no-op without)

- `play_viewer.py` interactive activation paths.
- `ml_planning.py` (only used if `planning_rate > 0`).
- `batched_argmax_tactical` / `batched_phase1_inference` /
  `batched_phase2_inference` — used by evolution + benchmark, not the active
  `ml_train`. Plumb them once but rank lower.

---

## File-by-file change inventory

| File | Approx LOC | Nature of change |
|------|------------|------------------|
| `ml_features.py` | +75 | Rewrite `encode_terrain_planes` to the 6-type closed-enum schema (raises on invalid combos); add `encode_unit_planes`, `assemble_board_planes`, constants (`UNIT_CHANNELS=6`, `TOTAL_PLANE_CHANNELS=12`). |
| `board.py` | +20 | `terrain_planes_A`, `terrain_planes_B` fields; populated in `set_terrain`. |
| `ml_model_tactical.py` | +80 | `TerrainCNN` class; trunk concat; `forward` / `trunk` / `encode` accept new kwarg; bump `AGG_DIM`. |
| `ml_integration_tactical.py` | +180 / −40 | **Flow inversion at POST_DEST** (see prerequisite section): three inference paths (`apply_tactical_model`, `_apply_tactical_model_phased`, `apply_tactical_model_sampling`) restructured so the engine executes the chosen move between dest sampling and POST_DEST encode; per-unit features and CNN plane both built from engine-truth `UnitState.positions`. Delete `project_post_move_unit_state` and update its callers. Also build pre/post CNN planes and pass them to the model. |
| `ml_training/config.py` | +12 | `unit_positions_pre` (request + record), `unit_positions_post` + `unit_flags` (record) fields. |
| `ml_training/collection.py` | +35 / −20 | Three request-build sites populate `unit_positions_pre` at the death-sync moment. Existing engine call sites (charge / move at lines 1016, 1054 etc.) refactored: the integration layer owns the engine call between dest sampling and POST_DEST; the episode generator yields back to the inference layer for it rather than calling the engine itself for moves. Post-move capture writes `unit_positions_post` onto the record. |
| `ml_training/sampling.py` | +90 | Single + batched samplers assemble planes from sidecars + worker-cached terrain bases. `_batched_sample_tactical_no_grad` interleaves a per-game engine call between the phase-1 batched forward (unit / move / dest) and the phase-2 batched forward (POST_DEST), capturing engine-truth post-positions per game before stacking the post-plane batch. |
| `ml_training/loss.py` | +55 | `PreparedReplayData` carries pre/post tensors; `prepare_replay_data` reads run-context terrain bases + stacked per-step pre/post snapshots, vectorised render; `replay_from_prepared` passes them through. No override-after-the-fact logic. |
| `ml_training/loop.py` | +5 | Pass `terrain_A` / `terrain_B` from the startup-loaded `Board` into `prepare_replay_data`. |
| `play_viewer.py`, `ml_planning.py` | +40 | Flow-inversion plumbing for interactive activation paths and the planning-rate path; same engine-mid-inference pattern as the main integration layer. |

**Estimated total**: ~610 LOC across 11 files (net of ~60 LOC deleted from
`project_post_move_unit_state` and old call-site logic).
**Engineering time**: ~22–35h focused — ~16–22h implementation plus ~4–8h
for lightweight smoke checks at each MVP milestone and the likely-but-not-
certain debugging if first-PPO-batch drift surfaces. High end gated on
that bisection (see risk #2 canary).

---

## Risks & gotchas

1. **Phase re-encode CNN cache — enforced by the API.** The model's
   `encode/forward/trunk` accept a precomputed `terrain_embed` vector, not
   raw planes (see "CNN forward as a separate model method"). Callers run
   `model.encode_terrain()` exactly twice per activation — once on
   pre-planes, once on post-planes — and pass the resulting embeds through
   phase encodes. The "compute once per (pre, post) pair" contract is
   enforced at the call-site level rather than relying on caller
   discipline. Sanity check during code review: every activation path
   should call `encode_terrain` exactly twice; more = wasted FLOPs, fewer
   = a phase is silently running on the wrong embed.

2. **Replay/sample drift.** If the unit-plane render at replay time doesn't
   exactly match what the sampler saw, PPO ratios diverge. Two capture
   points must be pinned exactly:
   - `unit_positions_pre`: captured *after* the prior activation's death-
     sync, *before* the current activation begins — at the moment the
     `_TacticalInferenceRequest` is built. Source: `UnitState.positions`
     across all 20 units.
   - `unit_positions_post`: captured *immediately after* all movement-phase
     engine calls for the current activation return, *before* any combat
     damage:
       - Move activations: after `execute_movement` returns.
       - Charge activations: after BOTH `execute_charge_movement` AND
         `execute_counter_charge` return (counter-charge moves the target
         and is part of the pre-combat movement; capturing before it would
         leave the target's positions stale).
       - Hold / shaken activations: equal to `unit_positions_pre` (no move
         happens). The sidecar can either store both copies or set
         `unit_positions_post = unit_positions_pre` by reference.

   The pre→post window contains only movement (no casualties), so
   `unit_flags` is shared between the two snapshots. Any future engine
   change that introduces casualties during the movement step (e.g.
   overwatch fire) would invalidate this invariant — flag it loudly.

   **Canary**: the first PPO batch after wiring replay is the canary. If
   ratios are ≠ 1 within float precision on the very first batch (no
   policy updates yet — sampler weights == replay weights), there's a
   bit-level mismatch between sampler and replay. Bisection recipe: turn
   the CNN feature flag off; if ratios snap back, the mismatch is in the
   CNN plane render. If they don't, turn the flow inversion off; if they
   snap back, the mismatch is in the engine-truth POST_DEST pipeline. If
   neither helps, look at the side-flip, padding, and snapshot-timing
   pieces in turn.

3. **Player perspective.** A and B see different planes for the same physical
   board state. The replay path must thread side info correctly (`side_np`
   on `PreparedReplayData` already exists; use it to pick the right terrain
   base).

4. **Multiprocessing pickle size — solved by design.** Workers cache
   `board.terrain_planes_A/_B` at init from the `map_data` they already
   receive; `_TacticalInferenceRequest` carries only the ~62-byte
   `unit_positions` / `unit_flags` sidecar; planes are assembled where
   they're consumed. No 162 KB plane tensors cross the pickle boundary.
   The only thing to watch is that any new request-build site remembers to
   populate the sidecar — easy to enforce with a dataclass `__post_init__`
   assertion when the CNN feature flag is on.

5. **Checkpoint incompatibility.** Adding the CNN + bumping `AGG_DIM` changes
   the trunk's parameter shapes. Old `final_model.pt` won't load. The active
   `ml_train(...)` call already has `restart_training=True`, so no real loss
   — just be aware that mid-run resume across the cutover isn't possible.

6. **No-terrain games.** When `board.terrain` is empty (no map applied), the
   cached `terrain_planes_*` are all-zero on channels 1–5 and all-one on
   channel 0. The CNN still trains on the unit channels alone. Fine, but
   worth a smoke test.

---

## Minimum viable order of operations

Each step is independently testable, so the work is naturally checkpointable
and rollback-friendly. Validation is **lightweight smoke by default** — one
quick check per milestone, eyeball the result, move on. Escalate to targeted
unit tests only if a smoke check surfaces unexpected behaviour at that
boundary.

1. `ml_features.encode_unit_planes` + `assemble_board_planes`.
   *Smoke*: render one plane tensor for a hand-crafted board — does the
   terrain look right? Are unit cells where they should be? Does the A/B
   flip mirror correctly?
2. `Board.terrain_planes_A/B` caching.
   *Smoke*: print both planes for the loaded map; A vs B should be 180° flips.
3. **Flow inversion at POST_DEST** (prerequisite — see top of doc). Land
   this independently of the CNN, behind its own feature flag if useful:
   restructure the three integration-layer entry points to call the engine
   between dest sampling and POST_DEST encode; delete
   `project_post_move_unit_state`; verify per-unit features at POST_DEST
   now reflect engine-truth positions.
   *Smoke*: run `quick_ml_check2.py`, confirm no crashes and value head
   outputs are within sane range of a pre-change baseline.
   *Acceptance*: training run to ~50 batches with the new flow only (no CNN
   yet) — value and loss curves should not regress materially vs main.
4. `TerrainCNN` module + trunk concat (with a feature flag to disable for
   A/B testing).
   *Smoke*: forward shape check; verify zero-init on `out_proj` makes the
   CNN contribute exactly zero to `agg` at step 1, so trunk output is
   bit-identical to the no-CNN baseline at fresh init.
5. Single-game inference (`apply_tactical_model`) with the flag on.
   *Smoke*: viewer game with `quick_ml_check2.py` runs without errors.
6. Training-time sampling (`collection.py` + `sampling.py`) — capture
   `unit_positions_pre` at request build and `unit_positions_post`
   immediately after the engine call now owned by the inference layer.
   *Smoke*: collect a few episodes, print one activation's pre/post
   snapshots side-by-side — only the acting unit's rows should differ, and
   the differences should match what `execute_movement` actually did.
7. Replay (`loss.py`).
   *Smoke* (slightly tighter — silent-corruption risk): sample one step,
   replay it, print policy logits and value from both passes. They should
   agree to a few decimals. This is the canary for sampler/replay drift —
   if it fails, see risk #2 for the bisection recipe.
8. Run a short training session (~50 batches).
   *Smoke*: watch loss + value curves; abort if anything looks wild.
   Compare to the step-3 baseline (flow inversion alone) to isolate the
   CNN's contribution.
9. Optional: revisit batched inference path for evolution / benchmarks;
   thread the flow inversion through `play_viewer.py` and `ml_planning.py`.

**Escalation**: if any smoke step surfaces unexpected behaviour, add
targeted unit tests at that boundary before continuing — don't push past a
warning sign.

---

## What this does *not* solve

- Per-shot LoS subtleties beyond the cover-aware E_damage signal already
  wired (see `TERRAIN_SPEC.md §5.4/§5.5`).
- Per-unit value-function awareness of nearby terrain (the CNN feeds the
  trunk, not the per-unit MLP — so individual unit features still don't see
  "I'm in cover" except via the destination-cover one-hot at `[76:82]`).
- Anything outside the spatial domain — e.g. weapon synergies between units.
  Those live in the trunk's MLP and stay there.
