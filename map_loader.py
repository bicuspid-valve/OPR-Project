"""Load tile-grid map JSONs (maps/map*.json) into engine TerrainPiece objects.

The map format is a 72x48 grid of string tile labels. Each tile is mapped to
either (a) a (CoverType, MovementType) pair that becomes a TerrainPiece, or
(b) a structural label (deployment zone, objective marker, open) that does
not contribute to the terrain list. Connected same-category regions are
decomposed into a non-overlapping cover of axis-aligned rectangles so the
result satisfies Board._validate_terrain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import board as _board_module
from board import (
    COLS, ROWS,
    CoverType, MovementType, TerrainPiece,
)


# Scout deployment extends an extra 12" forward of the regular DZ front.
SCOUT_FORWARD_ROWS = 12


# Tile → terrain semantics. None means "structural, not terrain".
# `ruin+objective` is a ruin tile that also marks an objective square; it
# contributes a terrain piece AND an entry to the `objective_tiles` overlay.
# `objective_center` marks the centre of an objective cluster.
TILE_TO_TERRAIN: dict[str, tuple[CoverType, MovementType] | None] = {
    "open": None,
    "deployment": None,
    "deployment+objective": None,
    "objective": None,
    "objective_center": None,
    "wall": (CoverType.BLOCKING, MovementType.IMPASSIBLE),
    "forest": (CoverType.OBSCURING, MovementType.DIFFICULT),
    "water": (CoverType.SHELTERING, MovementType.DIFFICULT),
    "ruin": (CoverType.SHELTERING, MovementType.OPEN),
    "ruin+objective": (CoverType.SHELTERING, MovementType.OPEN),
}


@dataclass
class MapData:
    """Parsed map: terrain pieces plus structural overlays the engine may
    optionally consume (objectives, deployment-zone cells)."""
    terrain: list[TerrainPiece]
    objectives: list[tuple[float, float]]        # objective centres (col, row); floats for half-integer centres
    objective_tiles: list[tuple[int, int]]       # every cell tagged as part of an objective blob
    deployment_a: list[tuple[int, int]]          # (col, row) cells in zone A
    deployment_b: list[tuple[int, int]]          # (col, row) cells in zone B
    dz_objective_tiles: list[tuple[int, int]]    # `deployment+objective` cells (DZ + obj overlay)
    width: int
    height: int


def load_map(path: str | Path,
             tile_overrides: dict[str, tuple[CoverType, MovementType] | None]
             | None = None) -> MapData:
    """Read a map JSON and return its parsed MapData.

    `tile_overrides` lets a caller swap individual tile mappings (e.g. treat
    a specific map's `water` as IMPASSIBLE)."""
    data = json.loads(Path(path).read_text())
    grid: list[list[str]] = data["grid"]
    h = data["height"]
    w = data["width"]
    declared_objectives = data.get("objectives") or []
    if h != ROWS or w != COLS:
        raise ValueError(
            f"map {path} is {w}x{h}; engine expects {COLS}x{ROWS}")

    mapping = dict(TILE_TO_TERRAIN)
    if tile_overrides:
        mapping.update(tile_overrides)

    # --- Split deployment_a / deployment_b by which half of the board the
    # cell falls in. The engine's existing convention is rows 0..ROWS/2-1
    # belong to player A and rows ROWS/2..ROWS-1 to player B.
    half = ROWS // 2
    deployment_a: list[tuple[int, int]] = []
    deployment_b: list[tuple[int, int]] = []
    dz_objective_tiles: list[tuple[int, int]] = []
    objective_markers: list[tuple[int, int]] = []
    objective_tiles: list[tuple[int, int]] = []
    cell_terrain: list[list[tuple[CoverType, MovementType] | None]] = [
        [None] * w for _ in range(h)
    ]

    for r in range(h):
        for c in range(w):
            tile = grid[r][c]
            if tile == "deployment":
                (deployment_a if r < half else deployment_b).append((c, r))
            elif tile == "deployment+objective":
                # In the deployment zone AND inside an objective's seize disc.
                # Counted in the DZ overlay (so the viewer paints it with the
                # DZ colour) and tracked separately so the viewer can mark
                # them subtly without an obj outline. Engine still treats the
                # cell as obj-controlled via OBJECTIVES + seize range.
                (deployment_a if r < half else deployment_b).append((c, r))
                dz_objective_tiles.append((c, r))
            elif tile == "objective_center":
                objective_markers.append((c, r))
                objective_tiles.append((c, r))
            elif tile == "objective":
                objective_tiles.append((c, r))
            elif tile == "ruin+objective":
                objective_tiles.append((c, r))
            entry = mapping.get(tile)
            if entry is not None:
                cell_terrain[r][c] = entry

    # --- Greedy maximal-rectangle decomposition per (cover, movement) class.
    # Iterating per class keeps adjacent-but-different terrain regions from
    # being merged into a single rectangle (which would be wrong).
    pieces: list[TerrainPiece] = []
    used = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if used[r][c] or cell_terrain[r][c] is None:
                continue
            entry = cell_terrain[r][c]

            # Extend right as far as the same (cover, movement) class runs.
            x_hi = c
            while (x_hi + 1 < w and not used[r][x_hi + 1]
                   and cell_terrain[r][x_hi + 1] == entry):
                x_hi += 1

            # Extend down as long as every column in [c, x_hi] is still the
            # same class on the next row.
            y_hi = r
            while y_hi + 1 < h:
                ok = True
                for cc in range(c, x_hi + 1):
                    if (used[y_hi + 1][cc]
                            or cell_terrain[y_hi + 1][cc] != entry):
                        ok = False
                        break
                if not ok:
                    break
                y_hi += 1

            for rr in range(r, y_hi + 1):
                for cc in range(c, x_hi + 1):
                    used[rr][cc] = True

            cover, movement = entry
            pieces.append(TerrainPiece(
                x_lo=c, x_hi=x_hi, y_lo=r, y_hi=y_hi,
                cover_type=cover, movement_type=movement,
            ))

    # Top-level "objectives" JSON list (if non-empty) overrides what the grid
    # could express via in-tile `objective_center` markers. Map authors use
    # this to declare half-integer centres (e.g. (35.5, 23.5)) and to control
    # the order so HOME_OBJ_A=3 / HOME_OBJ_B=4 still index the home positions.
    final_objectives: list[tuple[float, float]]
    if declared_objectives:
        final_objectives = [(float(x), float(y)) for x, y in declared_objectives]
    else:
        final_objectives = [(float(c), float(r)) for c, r in objective_markers]

    return MapData(
        terrain=pieces,
        objectives=final_objectives,
        objective_tiles=objective_tiles,
        deployment_a=deployment_a,
        deployment_b=deployment_b,
        dz_objective_tiles=dz_objective_tiles,
        width=w,
        height=h,
    )


def _scout_cells_for_player(dz_cells: list[tuple[int, int]],
                             player: str,
                             impassible: frozenset[tuple[int, int]]
                             ) -> frozenset[tuple[int, int]]:
    """Build the legal scout-deployment cell set: DZ ∪ 12 rows forward of
    the DZ front edge, clipped to the board and excluding IMPASSIBLE squares.

    "Forward" = +row for A (deploys top), -row for B (deploys bottom).
    The 1" enemy-exclusion check is applied at placement time in
    _place_unit_at; this set only enforces the geometric scout-zone bound."""
    if not dz_cells:
        return frozenset()
    by_col: dict[int, tuple[int, int]] = {}
    for c, r in dz_cells:
        lo, hi = by_col.get(c, (r, r))
        by_col[c] = (min(lo, r), max(hi, r))
    is_a = (player == "A")
    out: set[tuple[int, int]] = set(dz_cells)
    for c, (lo, hi) in by_col.items():
        if is_a:
            r_lo = hi + 1
            r_hi = min(ROWS - 1, hi + SCOUT_FORWARD_ROWS)
        else:
            r_lo = max(0, lo - SCOUT_FORWARD_ROWS)
            r_hi = lo - 1
        for r in range(r_lo, r_hi + 1):
            if (c, r) not in impassible:
                out.add((c, r))
    return frozenset(out)


def apply_map(board, map_data: MapData, *,
              build_vis_cover: bool = True,
              install_objectives_globally: bool = True) -> None:
    """Install a parsed map onto a Board: terrain, objectives, DZ cells.

    When ``install_objectives_globally`` is True (default), the module-level
    :data:`board.OBJECTIVES` list is mutated in place so legacy importers
    (``from board import OBJECTIVES``) see the map's objective set. Pass
    False to confine changes to this Board instance only (useful for tests)."""
    board.set_terrain(map_data.terrain, build_vis_cover=build_vis_cover)

    # Per-board state — visible to engine code that uses ``board.objectives``
    # and ``board.is_in_dz``.
    board.objectives = [tuple(o) for o in map_data.objectives]
    board.objective_control = [""] * len(board.objectives)

    impassible = frozenset(
        (c, r)
        for p in map_data.terrain
        if p.movement_type == MovementType.IMPASSIBLE
        for c, r in p.squares()
    )
    board.dz_a_cells = frozenset(map_data.deployment_a) - impassible
    board.dz_b_cells = frozenset(map_data.deployment_b) - impassible
    board.scout_a_cells = _scout_cells_for_player(map_data.deployment_a, "A", impassible)
    board.scout_b_cells = _scout_cells_for_player(map_data.deployment_b, "B", impassible)

    # Mutate module-level OBJECTIVES in place so legacy importers see the
    # new values. (Cannot rebind: `from board import OBJECTIVES` callers
    # would keep their original binding. Mutating the underlying list works
    # because they all share the same list object.)
    if install_objectives_globally:
        _board_module.OBJECTIVES.clear()
        _board_module.OBJECTIVES.extend(board.objectives)


if __name__ == "__main__":
    import sys
    from board import Board
    for path in sys.argv[1:] or ["maps/map1.json", "maps/map2.json"]:
        m = load_map(path)
        b = Board()
        # Validate without paying the §5.6 vis-cover table build cost.
        b.set_terrain(m.terrain, build_vis_cover=False)
        by_kind: dict[tuple[str, str], int] = {}
        for p in m.terrain:
            k = (p.cover_type.name, p.movement_type.name)
            by_kind[k] = by_kind.get(k, 0) + 1
        print(f"{path}: {len(m.terrain)} pieces, "
              f"{len(m.objectives)} objective markers, "
              f"DZ-A={len(m.deployment_a)} cells, "
              f"DZ-B={len(m.deployment_b)} cells")
        for k, n in sorted(by_kind.items()):
            print(f"    {k[0]:>10} + {k[1]:>10}: {n} pieces")
