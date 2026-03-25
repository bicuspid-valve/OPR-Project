"""Board state: 72x48 grid, occupancy tracking, objectives, and control."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


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
OBJECTIVES: list[tuple[int, int]] = [
    (36, 24),  # Centre
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
    # Objective control: index -> "A", "B", or "" (neutral/uncontrolled)
    objective_control: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.occupancy:
            self.occupancy = bytearray(COLS * ROWS)
        if not self.objective_control:
            self.objective_control = [""] * len(OBJECTIVES)

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

    def update_objectives(self, units_a: list, units_b: list):
        """Update objective control at end of round per §2.1."""
        threshold_sq = OBJ_SEIZE_RANGE * OBJ_SEIZE_RANGE

        for oi, (oc, orow) in enumerate(OBJECTIVES):
            a_present = False
            b_present = False

            for u in units_a:
                if u.destroyed or u.shaken:
                    continue
                for pos in u.alive_positions():
                    if dist_sq(pos, (oc, orow)) <= threshold_sq:
                        a_present = True
                        break
                if a_present:
                    break

            for u in units_b:
                if u.destroyed or u.shaken:
                    continue
                for pos in u.alive_positions():
                    if dist_sq(pos, (oc, orow)) <= threshold_sq:
                        b_present = True
                        break
                if b_present:
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
