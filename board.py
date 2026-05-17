"""Board state: 72x48 grid, occupancy tracking, objectives, and control."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum


# ===================================================================
# CONSTANTS
# ===================================================================

COLS = 72
ROWS = 48

# Deployment zones
DEPLOY_A_MIN_ROW = 0
DEPLOY_A_MAX_ROW = 11
DEPLOY_B_MIN_ROW = 36
DEPLOY_B_MAX_ROW = 47

# Forward deployment row for non-Scout units
DEPLOY_A_FRONT_ROW = 11
DEPLOY_B_FRONT_ROW = 36

# Scout forward deployment rows (12" further forward)
SCOUT_A_ROW = 23
SCOUT_B_ROW = 24

# Objective positions (col, row)
OBJECTIVES: list[tuple[float, float]] = [
    (35.5, 23.5),  # Centre — half-integer for 180° self-symmetry on even grid
    (18, 16),  # A-side
    (53, 31),  # B-side  (180° symmetric: 71-18=53, 47-16=31)
    (36, 6),   # Home-A (centre of A's deployment zone)
    (35, 41),  # Home-B  (180° symmetric: 71-36=35, 47-6=41)
]

# Home objective indices per player
HOME_OBJ_A = 3
HOME_OBJ_B = 4

OBJ_SEIZE_RANGE = 3.0  # centre-to-centre distance for objective control


# ===================================================================
# TERRAIN
# ===================================================================

class CoverType(IntEnum):
    SHELTERING = 0
    OBSCURING = 1
    BLOCKING = 2


class MovementType(IntEnum):
    OPEN = 0
    DIFFICULT = 1
    IMPASSIBLE = 2


@dataclass(frozen=True)
class TerrainPiece:
    """Axis-aligned rectangular grid region with cover and movement tags.

    Bounds are inclusive on the grid: squares (c, r) with x_lo <= c <= x_hi and
    y_lo <= r <= y_hi belong to the piece. In continuous coordinates it occupies
    the closed rectangle [x_lo, x_hi+1] x [y_lo, y_hi+1].
    """
    x_lo: int
    x_hi: int
    y_lo: int
    y_hi: int
    cover_type: CoverType
    movement_type: MovementType

    def contains_square(self, col: int, row: int) -> bool:
        return self.x_lo <= col <= self.x_hi and self.y_lo <= row <= self.y_hi

    def squares(self):
        for c in range(self.x_lo, self.x_hi + 1):
            for r in range(self.y_lo, self.y_hi + 1):
                yield (c, r)

    def hash_key(self) -> tuple:
        return (self.x_lo, self.x_hi, self.y_lo, self.y_hi,
                int(self.cover_type), int(self.movement_type))


def terrain_layout_hash(pieces: list[TerrainPiece]) -> str:
    """Order-independent SHA-256 of a terrain layout. Used as a disk-cache key
    for the §5.6 visibility/cover table."""
    import hashlib
    items = sorted(p.hash_key() for p in pieces)
    h = hashlib.sha256()
    for it in items:
        h.update(repr(it).encode())
    return h.hexdigest()


def _validate_terrain(pieces: list[TerrainPiece]) -> None:
    """Per spec §2.2 — raises ValueError on invalid configurations."""
    seen: dict[tuple[int, int], int] = {}
    for i, p in enumerate(pieces):
        if not (0 <= p.x_lo <= p.x_hi < COLS):
            raise ValueError(
                f"terrain[{i}]: x bounds out of range "
                f"(x_lo={p.x_lo}, x_hi={p.x_hi}, COLS={COLS})")
        if not (0 <= p.y_lo <= p.y_hi < ROWS):
            raise ValueError(
                f"terrain[{i}]: y bounds out of range "
                f"(y_lo={p.y_lo}, y_hi={p.y_hi}, ROWS={ROWS})")
        if (p.cover_type == CoverType.BLOCKING
                and p.movement_type != MovementType.IMPASSIBLE):
            raise ValueError(
                f"terrain[{i}]: BLOCKING pieces must be IMPASSIBLE")
        for sq in p.squares():
            if sq in seen:
                raise ValueError(
                    f"terrain[{i}] overlaps terrain[{seen[sq]}] at square {sq}")
            seen[sq] = i


# ===================================================================
# DISTANCE HELPERS
# ===================================================================

def dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean distance centre-to-centre."""
    dc = a[0] - b[0]
    dr = a[1] - b[1]
    return math.sqrt(dc * dc + dr * dr)


def dist_sq(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Squared Euclidean distance (avoids sqrt for comparisons)."""
    dc = a[0] - b[0]
    dr = a[1] - b[1]
    return dc * dc + dr * dr


# ===================================================================
# BOARD STATE
# ===================================================================

@dataclass
class Board:
    """Tracks which squares are occupied and objective control state."""
    # Flat occupancy grid: occupancy[row * COLS + col] = 1 if occupied
    occupancy: bytearray = field(default_factory=bytearray)
    # Per-board objective markers. Defaults to module-level OBJECTIVES for
    # legacy callers; map_loader.apply_map overwrites this from MapData.
    objectives: list[tuple[float, float]] = field(default_factory=lambda: list(OBJECTIVES))
    # Objective control: index -> "A", "B", or "" (neutral/uncontrolled)
    objective_control: list[str] = field(default_factory=list)
    # Terrain (set at deployment, immutable thereafter — see set_terrain).
    terrain: list[TerrainPiece] = field(default_factory=list)
    # Derived from terrain — lazily built on set_terrain.
    terrain_at_square: dict[tuple[int, int], TerrainPiece] = field(default_factory=dict)
    impassible_grid: bytearray = field(default_factory=bytearray)
    difficult_grid: bytearray = field(default_factory=bytearray)
    # Precomputed visibility/cover lookup table (§5.6). None when terrain
    # is empty; populated on set_terrain via vis_cover_table.build_or_load.
    vis_cover_table: object | None = None
    # Per-map deployment-zone cell sets. None ⇒ fall back to the legacy
    # DEPLOY_*_MIN/MAX row-range check (empty-board layout).
    dz_a_cells: frozenset[tuple[int, int]] | None = None
    dz_b_cells: frozenset[tuple[int, int]] | None = None
    scout_a_cells: frozenset[tuple[int, int]] | None = None
    scout_b_cells: frozenset[tuple[int, int]] | None = None

    def __post_init__(self):
        if not self.occupancy:
            self.occupancy = bytearray(COLS * ROWS)
        if not self.objective_control:
            self.objective_control = [""] * len(self.objectives)
        if not self.impassible_grid:
            self.impassible_grid = bytearray(COLS * ROWS)
        if not self.difficult_grid:
            self.difficult_grid = bytearray(COLS * ROWS)
        if self.terrain:
            # Copy the list; rebuild derived structures from scratch.
            pieces = list(self.terrain)
            self.terrain = []
            self.set_terrain(pieces)

    def set_terrain(self, pieces: list[TerrainPiece],
                    *, build_vis_cover: bool = True) -> None:
        """Install terrain layout. Validates per §2.2 and rebuilds derived state.

        Called once at deployment. Calling again replaces the layout
        (intended for tests / layout swaps; mid-game terrain mutation is
        outside spec). Set ``build_vis_cover=False`` to skip the §5.6
        precomputed visibility/cover table build (combat will fall back to
        live geometric compute via :mod:`terrain_los`); useful for tests on
        many random layouts where the table build cost dominates."""
        _validate_terrain(pieces)
        self.terrain = list(pieces)
        self.terrain_at_square = {}
        self.impassible_grid = bytearray(COLS * ROWS)
        self.difficult_grid = bytearray(COLS * ROWS)
        for piece in self.terrain:
            for c, r in piece.squares():
                self.terrain_at_square[(c, r)] = piece
                idx = r * COLS + c
                if piece.movement_type == MovementType.IMPASSIBLE:
                    self.impassible_grid[idx] = 1
                elif piece.movement_type == MovementType.DIFFICULT:
                    self.difficult_grid[idx] = 1
        # Build (or load from disk cache) the visibility/cover table — §5.6.
        if self.terrain and build_vis_cover:
            from vis_cover_table import build_or_load
            self.vis_cover_table = build_or_load(self.terrain)
        else:
            self.vis_cover_table = None

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < COLS and 0 <= row < ROWS

    def is_free(self, col: int, row: int) -> bool:
        return 0 <= col < COLS and 0 <= row < ROWS and not self.occupancy[row * COLS + col]

    def is_occupied(self, col: int, row: int) -> bool:
        return bool(self.occupancy[row * COLS + col])

    def place(self, col: int, row: int):
        self.occupancy[row * COLS + col] = 1

    def remove(self, col: int, row: int):
        self.occupancy[row * COLS + col] = 0

    def move_model(self, old_col: int, old_row: int, new_col: int, new_row: int):
        self.occupancy[old_row * COLS + old_col] = 0
        self.occupancy[new_row * COLS + new_col] = 1

    def is_in_dz(self, col: int, row: int, player: str,
                 *, scout: bool = False) -> bool:
        """True iff (col, row) is a legal deployment square for ``player``.

        Uses the per-Board ``dz_*_cells`` / ``scout_*_cells`` sets when set
        (map-driven layouts); otherwise falls back to the legacy DEPLOY_*
        row-range constants. ``scout=True`` allows the 12" forward extension."""
        if 0 <= col < COLS and 0 <= row < ROWS:
            pass
        else:
            return False
        is_a = (player == "A")
        cells = (self.dz_a_cells if is_a else self.dz_b_cells)
        scout_cells = (self.scout_a_cells if is_a else self.scout_b_cells)
        if cells is not None:
            if (col, row) in cells:
                return True
            if scout and scout_cells is not None and (col, row) in scout_cells:
                return True
            return False
        # Legacy fallback: row-range check.
        if is_a:
            lo, hi = DEPLOY_A_MIN_ROW, DEPLOY_A_MAX_ROW
            if scout:
                hi = max(hi, SCOUT_A_ROW)
        else:
            lo, hi = DEPLOY_B_MIN_ROW, DEPLOY_B_MAX_ROW
            if scout:
                lo = min(lo, SCOUT_B_ROW)
        return lo <= row <= hi

    def update_objectives(self, units_a: list, units_b: list):
        """Update objective control at end of round per §2.1.

        Tags are based on each unit's ``owner`` attribute so that "A" always
        means physical side A regardless of which positional argument each
        side was passed in. The training generator swaps positional args
        when the learning model is placed on side B, so relying on
        positional order would mis-tag captures in that case.
        """
        threshold_sq = OBJ_SEIZE_RANGE * OBJ_SEIZE_RANGE

        for oi, (oc, orow) in enumerate(self.objectives):
            a_present = False
            b_present = False

            for u in list(units_a) + list(units_b):
                if u.destroyed or u.shaken:
                    continue
                for pos in u.alive_positions():
                    if dist_sq(pos, (oc, orow)) <= threshold_sq:
                        if u.owner == "A":
                            a_present = True
                        elif u.owner == "B":
                            b_present = True
                        break
                if a_present and b_present:
                    break

            if a_present and not b_present:
                self.objective_control[oi] = "A"
            elif b_present and not a_present:
                self.objective_control[oi] = "B"
            elif a_present and b_present:
                # Contested → becomes neutral
                self.objective_control[oi] = ""
            # If neither present, control stays as-is (previously seized)

    def count_objectives(self, player: str) -> int:
        return sum(1 for c in self.objective_control if c == player)
