# Terrain System — Implementation Spec

## 1. Overview

Adds terrain as a first-class feature of the game state. Terrain is a list of axis-aligned rectangular grid regions, each tagged with a **cover type** (how it affects shooting) and a **movement type** (how it affects movement). Terrain affects movement pathfinding, line of sight, cover bonuses, and target legality.

The game board is the existing 72×48 grid ([board.py:12-13](board.py#L12-L13)). One grid square = 1 inch. Each model occupies exactly one square. All coordinates are integer `(col, row)` pairs unless noted; continuous coordinates are used only inside the geometric obscure check.

This spec is engine-and-ML-features-inclusive. Adopting it invalidates current ML checkpoints; retraining from scratch is required.

---

## 2. Terrain Representation

### 2.1 Data model

A **terrain piece** is:

```
TerrainPiece:
    x_lo: int            # inclusive grid column bounds
    x_hi: int            # inclusive
    y_lo: int            # inclusive grid row bounds
    y_hi: int            # inclusive
    cover_type: {SHELTERING, OBSCURING, BLOCKING}
    movement_type: {OPEN, DIFFICULT, IMPASSIBLE}
```

The grid squares belonging to a piece `X` are exactly `{(c, r) : x_lo ≤ c ≤ x_hi, y_lo ≤ r ≤ y_hi}`. In continuous coordinates the piece occupies the closed rectangle `[x_lo, x_hi+1] × [y_lo, y_hi+1]`.

The board owns a list `Board.terrain: list[TerrainPiece]`. This list is set at deployment time and is immutable for the duration of a game.

### 2.2 Validity constraints

A terrain configuration is **valid** iff:

1. `0 ≤ x_lo ≤ x_hi < COLS` and `0 ≤ y_lo ≤ y_hi < ROWS`.
2. No two pieces share a grid square. Pieces *may* share an edge (i.e. one piece's `x_hi = other.x_lo - 1`).
3. Every piece with `cover_type = BLOCKING` has `movement_type = IMPASSIBLE`. (Blocking is always impassible.)

Validation is enforced at board construction; invalid configurations raise.

### 2.3 Convenience lookups

Two derived structures, built once when terrain is set:

- `terrain_at_square: dict[(col, row) -> TerrainPiece | None]` — fast per-square lookup.
- `impassible_grid: bytearray` of size `COLS * ROWS`, with `1` for every square belonging to a piece with `movement_type = IMPASSIBLE`.

### 2.4 Objective interaction

Objectives may sit inside any terrain type, including IMPASSIBLE. Seizing requires a friendly model within `OBJ_SEIZE_RANGE` of the objective coordinate, not occupancy of the square itself, so an objective in impassible terrain remains contestable as long as some squares in its seize zone are accessible. Terrain configurations are assumed to satisfy this; no validation is enforced.

---

## 3. Movement Semantics

### 3.1 Path definition

A **path** is a sequence of squares `s_0, s_1, …, s_n` where `s_0` is the model's starting square, `s_n` is its destination, and consecutive squares are 4- or 8-adjacent. Step cost is `1.0` (cardinal) or `√2` (diagonal), consistent with [movement.py:17-20](movement.py#L17-L20). **Path length** = sum of step costs in inches.

A path **enters** terrain piece `X` iff *any* `s_i` with `i ≥ 1` belongs to `X`. The starting square `s_0` is **never** considered "entered" — a model starting in difficult terrain is not penalized for being there.

### 3.2 Movement type effects

For each move action (advance / rush / charge):

- **OPEN** — no effect.
- **DIFFICULT** — if the path enters any difficult piece, the total path length must be ≤ 6.0 inches. The cap applies to all move types (advance, rush, charge). If a unit's normal budget is already ≤ 6", the cap is a no-op.
- **IMPASSIBLE** — the path must contain no square belonging to an impassible piece (except as overridden by `flying`, see §3.4).

Multiple difficult pieces on the path do **not** stack; the cap remains 6.0.

### 3.3 Pathfinding implementation

The current `_greedy_move` ([movement.py:83](movement.py#L83)) uses per-cell Dijkstra (each cell has one best cost). The difficult-terrain cap is a *per-path* property, so the state space must be extended:

> Search over `(square, has_entered_difficult: bool)` pairs. From any state, expand to neighbors with the existing step cost, except: a neighbor square in an impassible piece is rejected outright (unless flying); a neighbor square in a difficult piece sets the next state's `has_entered_difficult = True`. Whenever `has_entered_difficult` is true, the budget for that state is `min(remaining_budget, 6.0 − cost_so_far)`.

Equivalent two-tier formulation (easier to implement): explore freely up to `min(unit_budget, 6.0)` regardless of terrain, **and** additionally explore up to `unit_budget` along paths that stay entirely on non-difficult squares. The reachable set is the union. Both implementations must produce the same reachable set.

The model's chosen destination must additionally satisfy: the destination square is not in an impassible piece. This applies to **all** units including those with `flying` — flying permits passing *through* impassible terrain during path traversal but does not allow ending a move there (see §3.4). The destination square must also be unoccupied (existing rule).

Both the C path ([fast_core.c](fast_core.c)) and the Python fallback ([movement.py:123-204](movement.py#L123-L204)) must be updated. A parity test (§6) asserts they produce identical reachable-square sets and final destinations on terrain-bearing boards.

### 3.4 Ability overrides

- **`flying`** (already plumbed) — ignores both DIFFICULT and IMPASSIBLE for the purposes of *path traversal*. The destination square must still not lie in an impassible piece. `flying` already bypasses enemy occupancy ([movement.py:173](movement.py#L173)).
- **`strider`** (defined at [models.py:155](models.py#L155), currently unused) — ignores DIFFICULT only. The 6" cap and impassible rules are unaffected. `strider` must be plumbed through `execute_movement` ([movement.py:207](movement.py#L207)) → `_greedy_move` → `_fc.fast_pathfind_move` analogously to `flying`. Both extension and Python fallback updated.

Flying and strider are movement-only. Neither affects LOS, cover, or shooting in any way.

---

## 4. Line of Sight and Cover

This section defines the geometric and rule-level predicates used during shooting resolution. All geometric work is done in continuous coordinates; a model in grid square `(c, r)` occupies the closed square with corners `(c, r), (c+1, r), (c, r+1), (c+1, r+1)`.

LOS and cover predicates gate shooting only. Charge declarations do **not** require LOS to the target unit — path legality (§3) and charge range are sufficient. Once units are engaged in melee, combat resolves regardless of LOS between the engaged models: a successful charge that leaves attacker and defender on opposite sides of a thin BLOCKING piece (or any other LOS-breaking geometry) still resolves melee normally.

### 4.1 Primitive: line crosses terrain

A line segment `L = [P, Q]` **crosses** terrain piece `X` iff `L` intersects the *open interior* of `X`'s continuous rectangle, `(x_lo, x_hi+1) × (y_lo, y_hi+1)`.

Equivalently: there exists `t ∈ (0, 1)` such that `P + t(Q − P)` lies strictly inside that rectangle. Touching an edge or a corner does **not** count as crossing.

Reference implementation: clip `L` against the closed rectangle using Liang-Barsky; the segment crosses iff the clipped interval has nonzero length and lies strictly inside the rectangle (i.e. not entirely on a boundary edge). A small epsilon (`1e-9`) guards floating-point boundary cases; the rule is "strictly inside" but degenerate-on-edge inputs must be classified as **not crossing**.

### 4.2 Predicates: per-piece obscure and combined visibility

Two predicates are derived from the 16 corner-pair line-crossing tests between shooter `Y`'s 4 corners and target `A`'s 4 corners. Both predicates share the same 16 line-vs-piece crossings; implementations should compute them once and reuse.

**Per-piece "obscured by X from Y" (boolean).** For each corner `Z` of `Y`'s square, check whether all four segments `[Z, A_corner_i]` (`i = 1..4`) do **not** cross `X`. If any such `Z` exists, `A` is **not obscured by X from Y**; otherwise `A` is **obscured by X from Y**. This predicate is used only for cover (§4.3).

**Combined visibility "A is visible to Y" (boolean).** `A` is **visible to Y** iff there exists at least one pair `(Z, A_corner_j)` — over `Z` ∈ `Y`'s 4 corners × `A_corner_j` ∈ `A`'s 4 corners, 16 pairs total — such that the segment `[Z, A_corner_j]` crosses *no* piece with `cover_type = BLOCKING`. Visibility is determined by the combined geometry across all BLOCKING pieces: two pieces with a gap between them block sight only if every one of the 16 lines is crossed by at least one BLOCKING piece. The per-piece obscure predicate is **not** used for visibility — visibility is a function of the combined BLOCKING set.

Notes:

- Degenerate case: `Y == A`. Cannot occur in practice (occupancy is exclusive), but the algorithm is well-defined: every corner-pair segment is zero-length and crosses no rectangle interior, so `A` is visible and not obscured by any piece.
- Degenerate case: `Y` and `A` share a corner (adjacent squares). The zero-length-projection segments from a shared corner to itself do not cross any rectangle interior; the combined-visibility check succeeds trivially via that pair if no BLOCKING piece sits between the rest.

### 4.3 Per-model cover and visibility predicates

**Visibility.** `A` is **visible to Y** iff the combined-visibility predicate from §4.2 holds — i.e. some `(Z, A_corner_j)` line is clear of every BLOCKING piece. Determined by the combined BLOCKING geometry, not piece-by-piece.

**Cover.** For each terrain piece `X`, evaluate independently per §4.2's per-piece obscure and the wholly-within check. The result is summarised per `(Y, A)`:

| `X` cover type | "wholly within X" (A's square ∈ X) | "obscured by X from Y" (§4.2 per-piece) |
|---|---|---|
| **SHELTERING** | `A` is **in cover** (regardless of Y) | (no effect via obscure check) |
| **OBSCURING** | `A` is **in cover** | `A` is **in cover** |
| **BLOCKING** | n/a — impassible, A can't be inside | `A` is **in cover** (if still visible per the combined rule above) |

`A` is **in cover from Y** iff at least one of the rows above triggers "in cover" for some piece `X` (cover does not stack — the flag is binary). Cover is moot when `A` is not visible to `Y`.

Notes on the geometry of "wholly within":

- Each model occupies one square; "wholly within X" reduces to "A's square ∈ X."
- Because pieces don't overlap, a model is wholly within at most one piece.

Notes on shooter position:

- A shooter `Y` whose square is in an OBSCURING piece `X` but on `X`'s outer edge (some corner of `Y` lies on `X`'s boundary) can shoot outward without lines crossing `X`'s interior — `X` does not impose cover on the target. A shooter `Y` whose square is in `X` but *not* on the outer edge has all corners strictly inside `X`'s interior, so every outbound segment immediately crosses `X` → target is in cover by `X`.
- A shooter `Y` inside a SHELTERING piece `X` (which has no obscure-passing-through rule) does not impart cover via `X` to anyone — sheltering only protects models inside it from incoming shots.

### 4.4 Unit-level shooting rules

For attacker unit `U_atk` shooting target unit `U_def`:

1. **Target declaration.** `U_atk` may declare `U_def` as a target iff at least one model in `U_atk` has at least one model in `U_def` that is visible to it. (Existing range constraints from [combat.py:244](combat.py#L244) still apply.)

2. **Per-shooter resolution.** For each shooter `Y ∈ U_atk` who actually fires:
   - If **every** model in `U_def` is not visible to `Y`, `Y` cannot fire.
   - Otherwise let `m_cover = |{A ∈ U_def : A is in cover from Y or not visible to Y}|` and `m_total = |U_def|`.
     - If `2 · m_cover > m_total` (strict majority), `U_def` is **in cover from Y**: every defense roll triggered by `Y`'s shots this activation gets **+1**.
     - Otherwise no bonus.
   - Different shooters in `U_atk` may yield different cover states for the same `U_def`; each is resolved independently.

3. **Per-shot targeting.** A shooter `Y`'s individual shots can only resolve against models in `U_def` that are visible to `Y`. (Models not visible to `Y` are skipped when selecting per-shot targets.)

The cover bonus does not stack across pieces, shooters, or other sources. The check uses strict majority — exactly half does **not** confer cover, matching the convention used elsewhere ([combat.py:43](combat.py#L43)).

Cover and stealth stack: the +1 defense from cover and the −1 to hit from stealth apply independently to their respective rolls, since they modify opposite sides of the resolution. No special case is required — the engine already composes them correctly via `_effective_defense` and the per-weapon hit math.

### 4.5 "Ignores Cover" shooting abilities

Out of scope for this spec other than the integration hook: certain weapons/units already have ignore-cover semantics (see `Ignores Cover` flag at [combat.py:754](combat.py#L754)). The +1 defense bonus from §4.4(2) is **suppressed** when the firing weapon has that flag set. Visibility is not affected — "ignores cover" does not let a shooter target an invisible model.

---

## 5. Engine Integration Points

### 5.1 Board

[board.py](board.py): add `terrain: list[TerrainPiece]`, `terrain_at_square`, and `impassible_grid` to `Board`. Validate per §2.2 in `__post_init__` or a dedicated `set_terrain(...)` method called during deployment.

### 5.2 Movement

- [movement.py:83](movement.py#L83) `_greedy_move`: extend state with `has_entered_difficult` per §3.3. Reject impassible neighbors unless `flying`. Plumb `strider` through the signature.
- [movement.py:207](movement.py#L207) `execute_movement`: accept and forward `strider`.
- [fast_core.c](fast_core.c) `fast_pathfind_move`: mirror all of the above. Maintain Python/C parity.
- [game.py:497](game.py#L497) and surrounding action dispatch: pass `strider=...` from `ResolvedUnit` alongside the existing `flying=...`.

### 5.3 Combat

- [combat.py:9](combat.py#L9) `_effective_defense`: takes an optional `cover_bonus: int = 0` and adds it to the returned value. Callers compute the bonus from §4.4 and pass it in.
- [combat.py:146](combat.py#L146) `evaluate_target` / [combat.py:244](combat.py#L244) `can_shoot_any`: gate target legality on §4.4(1).
- [combat.py:286](combat.py#L286) `resolve_shooting`: before each shooter `Y` fires, compute per-`Y` cover state for the target unit (one boolean) and the visible-models set; pass these into the per-shot loop. Apply the +1 cover bonus via `_effective_defense` for every defense roll while resolving `Y`'s shots. Suppress when the firing weapon has `ignores_cover`.

A single helper module is appropriate, e.g. `terrain_los.py`, exposing:

```
def line_crosses_terrain(p: tuple[float,float], q: tuple[float,float], X: TerrainPiece) -> bool
def obscured_by(Y_sq, A_sq, X) -> bool                          # per-piece obscure (§4.2)
def is_visible(Y_sq, A_sq, terrain) -> bool                     # combined-blocking (§4.2)
def is_in_cover(Y_sq, A_sq, terrain) -> bool
def shooter_cover_state(Y_sq, target_squares, terrain) -> (n_cover_or_invisible, visible_mask)
```

A C-accelerated equivalent should live in `_fast_core` to match the existing pattern; per-shooter LOS is on the inner shooting loop and will be hot.

### 5.4 ML features

[ml_features.py](ml_features.py): add a terrain channel to the feature tensor. Two layouts; pick one in implementation:

- **Per-square channels (recommended).** Add `K` board-shaped scalar planes (e.g. `K = 6`: one-hot of `{open, difficult-only, impassible-only, sheltering, obscuring, blocking}`). Inflates the input by `K · COLS · ROWS` elements alongside the existing 4016. This is dense and uniform; good for the destination feature head.
- **Per-piece list.** Append the terrain list as a fixed-length array of `(x_lo, x_hi, y_lo, y_hi, cover_id, movement_id)` tuples padded to a max count. Compact but harder for the model to spatially correlate.

The destination feature head ([ml_features.py:70-72](ml_features.py#L70-L72), `DEST_FEATURE_DIM = 76`) should additionally surface, per candidate hex: cover type of that hex, movement type of that hex, whether the hex is reachable under the current move action given terrain.

The shoot pointer head's per-enemy keys (`_build_shoot_keys` in [ml_model_tactical.py:618](ml_model_tactical.py#L618)) currently expose `expected_wound_frac` as a cover-blind range-bucketed lookup against `friendly_ranged_matchups`. Replace this scalar with a unit-level cover-aware expected damage, computed via §5.5's "Game-time combination" formula — i.e. for each enemy slot `j`, sum `E_damage[Y, U_def_j, cover_state(Y, U_def_j)]` over the shooters `Y` in the acting unit `U_atk`. Key shape is unchanged (still 2-dim `[expected_wound_frac, current_wound_frac]`); the cover-blind value is not retained alongside — it is strictly dominated.

**Shooter post-move squares (centroid proxy).** The dest head picks one destination hex per unit, not per model. For the shoot-head cover lookups, use the acting unit's chosen destination as the proxy shooter position for *every* `Y ∈ U_atk`. Cover collapses to a single binary per `(U_atk_destination, U_def_j)` pair, and the aggregation simplifies to `n_attacker_shooters · E_damage[attacker_class, U_def_j, cover]` per attacker equivalence class. The shoot head's role is per-target ranking, not precise damage prediction — the precision loss vs. a per-model formation placement is small and the plumbing is trivial.

Adopting this spec invalidates all existing checkpoints in [ml_checkpoints/](ml_checkpoints/) and siblings. Retrain from scratch.

### 5.5 Pre-game expected ranged damage table

At game start (after deployment, once terrain is set and unit rosters are final), precompute expected ranged damage for every pair `(Y, U_def)` of `(attacker_model, target_unit)` across the two armies. For each pair, compute **two** values:

- `E_damage[Y, U_def, cover=False]` — expected damage `U_def` takes from one full activation of `Y`'s ranged weapons against `U_def`, assuming `Y` is in range of and has LOS to every model in `U_def`, and `U_def` gets **no** +1 cover bonus from `Y`.
- `E_damage[Y, U_def, cover=True]` — same, but with the +1 cover bonus from §4.4(2) applied to `U_def`'s defense rolls.

Keying on `(model, unit)` rather than `(model, model)` matches the granularity of the game-time cover decision: per §4.4(2), cover is a single binary state for the whole target unit from the perspective of one shooter, so one number per shooter per branch is exactly what gets consumed. Combining model-shoots-unit values into a unit-shoots-unit estimate is a straightforward sum over shooters; combining model-shoots-model values would require an additional within-unit aggregation step that recapitulates the per-shot targeting / wound allocation logic.

Range and per-target-model visibility are board-state dependent and are **not** baked into the table — the values condition on "`Y` is firing into `U_def`, full legality satisfied." Cover *is* baked in (both branches) because it's the only terrain-derived modifier and dispatching on a single bool at lookup time is cheap.

**Internal target-selection / wound-allocation model.** Each `E_damage` value integrates over `Y`'s individual shots resolving against models in `U_def`. The policy used inside the precomputation must match the policy used by `resolve_shooting` ([combat.py:286](combat.py#L286)) so the precomputed expectation is consistent with what would actually happen at the table. The same policy is used in both cover branches — the only difference between them is the +1 to defense.

**Granularity / dedup.** On the attacker side, identical models within `U_atk` (same weapons, stats, shooting-relevant abilities) collapse to one equivalence class — compute once per `(attacker_class, U_def)` and reuse for each instance. The attacker-equivalence key is `(weapon profiles, attack stat, quality, special rules touching ranged output)`. On the defender side the *unit* is the unit (no per-model collapsing on that side — the value already integrates over the unit's composition).

**Game-time combination.** Different shooters `Y ∈ U_atk` may see different cover states on the same `U_def`. Sum shooter-by-shooter:

```
total = 0
for Y in U_atk firing this activation:
    cover = shooter_cover_state(Y, U_def, terrain)        # binary, §4.4(2)
    total += E_damage[Y, U_def, cover]
```

This handles the mixed case the unit-level cover rule produces (e.g. half the attacking unit shoots from a position where `U_def` is majority-obscured, the other half shoots clean).

**Storage and invalidation.** Stored on the `Game` (or equivalent top-level state) keyed by `(attacker_model_id, defender_unit_id)`. Built once and immutable for the duration of the game — model loadouts and unit composition do not currently change mid-game. If that assumption is broken later (mid-game weapon swaps, attached heroes joining units, casualties altering unit composition in a way the table needs to track), the affected rows are invalidated. Casualties within `U_def` during the game are *not* a table-invalidation event — the precomputed value is a planning estimate against the unit's starting composition; live resolution still rolls actual dice.

**Used by.** Tactical-planning heuristics (see [tactical_planning_spec.md](tactical_planning_spec.md)) and any shooting-evaluation code that needs an expectation rather than an actual roll — specifically, the ML shoot pointer head's `expected_wound_frac` key (see §5.4). **Not** used by `resolve_shooting` itself.

**Build dependency.** Consumes the §5.6 visibility/cover table for `is_visible` and `shooter_cover_state` lookups; build §5.6 first.

**Ignores Cover.** Per §4.5 the +1 is suppressed on weapons with `ignores_cover`. For attacker models whose entire ranged profile carries `ignores_cover`, the two branches collapse — store one value and flag the pair. For mixed-profile attackers (some weapons ignore cover, others don't), keep both branches: the `cover=True` branch applies the +1 only to the defense rolls from non-ignoring weapons.

### 5.6 Precomputed visibility/cover table

Per-shooter LOS and cover are deterministic functions of terrain and the two squares involved, and terrain is immutable for the duration of a game (§2.1). At deployment time, after terrain is set and **before** the §5.5 E_damage table is built, precompute:

```
vis_cover_table[shooter_sq, target_sq] ∈ {OPEN, COVER, NO_LOS}
```

Indexed by **ordered** square pairs — cover is not symmetric. A shooter wholly inside an OBSCURING piece imposes cover on its target via §4.3's wholly-within rule, but the reverse arrangement does not. Geometric line-crossings *are* symmetric and can be deduplicated internally during the build, but the published table exposes ordered pairs.

Whole-board entries (no range gate). Storage at 1 byte per pair is `COLS² · ROWS² · 1 ≈ 12 MB` on a 72×48 board — small enough that the simplicity of "always look it up" beats any savings from a range gate. (A 36" gate would also miss Versatile Reach's +4" extension, which a whole-board table sidesteps.)

Each entry encodes:

- `OPEN` — target visible from shooter, no cover bonus from terrain.
- `COVER` — target visible from shooter, +1 defense from cover applies (§4.4).
- `NO_LOS` — target not visible from shooter, per the §4.2 combined-blocking rule: every one of the 16 `(shooter_corner, target_corner)` lines is crossed by at least one BLOCKING piece.

The §5.3 helper module's signatures are unchanged; their implementations switch from live geometric compute to table lookup. `is_visible(Y_sq, A_sq) := table[Y_sq, A_sq] != NO_LOS`; `is_in_cover(Y_sq, A_sq) := table[Y_sq, A_sq] == COVER`; `shooter_cover_state` aggregates over the target unit's model squares.

**Build cost.** ~12M ordered-pair evaluations, each O(|terrain|) segment-vs-rectangle tests. AABB pre-culling per terrain piece reduces the inner loop to relevant pieces only. Expected build time in C with pre-culling: under one second per terrain layout.

**Disk cache.** Terrain layouts are per-game (§2.1), but training rollouts reuse a fixed bank of boards across many episodes. Cache the built table on disk keyed by a stable hash of the terrain layout (sorted tuple of `(x_lo, x_hi, y_lo, y_hi, cover_type, movement_type)` tuples). Repeat games on the same layout incur zero rebuild cost.

---

## 6. Tests to Add

New test files alongside the existing `test_*.py` suite:

- `test_terrain_validation.py` — overlap detection, bounds checks, blocking-must-be-impassible.
- `test_terrain_movement.py` — open passes through, difficult caps at 6", impassible blocks, flying traverses difficult and impassible during path but cannot end on impassible, strider ignores only difficult, destination-square impassible rejection (including for flying units), 6" cap across multiple difficult pieces, starting-square-in-difficult does not trigger cap.
- `test_terrain_los.py` — per-piece obscure boolean, combined-blocking visibility (single-BLOCKING fully invisible; two BLOCKING pieces with a corner-pair gap remain visible; two BLOCKING pieces that together cover all 16 corner-pair lines block sight even when neither alone does), models on outer edge of obscuring shoot freely outward, models in interior of obscuring impose cover, sheltering does not pass cover through, BLOCKING obscure ⇒ cover when target still visible.
- `test_terrain_cover.py` — per-shooter unit-level cover (strict majority), invisible-to-all-of-attacker blocks target declaration, ignores-cover suppresses +1.
- `test_fast_core_terrain_parity.py` — C and Python pathfinding produce identical reachable sets on a bank of terrain-bearing boards. Extends the existing `test_fast_core_ports.py` pattern.
- `test_precomputed_visibility_cover.py` — §5.6 table entries match live geometric compute on a bank of randomized terrain layouts; ordered-pair asymmetry preserved where expected (shooter-inside-OBSCURING vs. target-inside-OBSCURING); disk-cache round-trip via terrain hash returns the same table.
- `test_ml_features_terrain.py` — terrain channels are present, correctly populated, and survive board mirroring (existing symmetry tests).
- `test_expected_damage_table.py` — §5.5 table built at game start, both cover branches populated for every `(attacker_model, defender_unit)` pair, identical-attacker-model dedup hits the same entry, `cover=True` value matches `cover=False` recomputed with defense+1, ignores-cover weapons produce equal branches, shooter-by-shooter sum reproduces a direct unit-vs-unit expected damage calculation when all shooters see the same cover state.

---

## 7. Open Questions / Future Extensions

- **Hero models with different bases.** Current spec assumes uniform 1×1 bases. If multi-square heroes or different base sizes are added later, all "wholly within" and corner-set definitions need revisiting.
- **Partial cover model.** Currently cover is binary (+1 defense). A future refinement could distinguish partial (e.g. +1) from full (e.g. +2 or save reroll). The §4.4 plumbing already passes a numeric bonus, so the extension is low-cost.
