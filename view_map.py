"""Standalone viewer for a parsed map: terrain, deployment zones, and
objectives. Reuses the game viewer's pixel scale (CELL=12) and y-flip
(row 0 at bottom) so positions visually match what an in-game replay
would show.

Usage:
    python view_map.py maps/map1.json
    python view_map.py maps/map2.json
"""
from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

from board import COLS, ROWS, CoverType, MovementType
from map_loader import load_map, MapData


CELL = 12
BOARD_W = COLS * CELL
BOARD_H = ROWS * CELL


# Per-(cover, movement) style. Tiles that map_loader doesn't currently
# produce are included so an override / future map still renders cleanly.
TERRAIN_STYLES: dict[tuple[CoverType, MovementType], dict[str, str]] = {
    (CoverType.BLOCKING,  MovementType.IMPASSIBLE):
        {"fill": "#2a2a2a", "outline": "#6e6e6e",
         "label": "Wall  —  blocking + impassable"},
    (CoverType.OBSCURING, MovementType.DIFFICULT):
        {"fill": "#2d5a2d", "outline": "#5fa45f",
         "label": "Forest  —  obscuring + difficult"},
    (CoverType.SHELTERING, MovementType.DIFFICULT):
        {"fill": "#1e4d6b", "outline": "#56a8d4",
         "label": "Water  —  sheltering + difficult"},
    (CoverType.SHELTERING, MovementType.OPEN):
        {"fill": "#8a6a3a", "outline": "#d4a866",
         "label": "Ruin  —  sheltering + open"},
    (CoverType.OBSCURING, MovementType.OPEN):
        {"fill": "#5a7a3a", "outline": "#9fc36c",
         "label": "Obscuring + open"},
    (CoverType.SHELTERING, MovementType.IMPASSIBLE):
        {"fill": "#5a7a8a", "outline": "#a8c8d4",
         "label": "Sheltering + impassable"},
    (CoverType.OBSCURING, MovementType.IMPASSIBLE):
        {"fill": "#5a3a7a", "outline": "#a888d4",
         "label": "Obscuring + impassable"},
}

DZ_A_FILL = "#a8b8e0"
DZ_B_FILL = "#e0a8a8"
OBJ_TILE_OUTLINE = "#a07c14"
OBJ_CENTRE_FILL = "#f0c020"
GRID_FILL = "#8a8a8a"
BOARD_BG = "#cfcfcf"
LEGEND_BG = "#2d2d2d"


def _cell_to_pixels(col: int, row: int) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) pixel rect for cell (col, row), y-flipped
    so row 0 is at the bottom (matches viewer.py)."""
    x0 = col * CELL
    y0 = (ROWS - 1 - row) * CELL
    return x0, y0, x0 + CELL, y0 + CELL


def render_map(canvas: tk.Canvas, m: MapData) -> None:
    canvas.delete("all")

    # Deployment zone tint (rendered first; terrain overlays it where they
    # overlap, which is correct — a wall in the DZ should look like a wall).
    for col, row in m.deployment_a:
        x0, y0, x1, y1 = _cell_to_pixels(col, row)
        canvas.create_rectangle(x0, y0, x1, y1, fill=DZ_A_FILL, outline="")
    for col, row in m.deployment_b:
        x0, y0, x1, y1 = _cell_to_pixels(col, row)
        canvas.create_rectangle(x0, y0, x1, y1, fill=DZ_B_FILL, outline="")

    # Terrain rectangles.
    for piece in m.terrain:
        style = TERRAIN_STYLES.get(
            (piece.cover_type, piece.movement_type),
            {"fill": "#7e7e3a", "outline": "#bfbf66",
             "label": f"{piece.cover_type.name} + {piece.movement_type.name}"})
        x0 = piece.x_lo * CELL
        x1 = (piece.x_hi + 1) * CELL
        # y_hi is the higher-row edge of the rect; after the flip it becomes
        # the *top* pixel. y_lo becomes the *bottom* pixel.
        y0 = (ROWS - 1 - piece.y_hi) * CELL
        y1 = (ROWS - piece.y_lo) * CELL
        canvas.create_rectangle(x0, y0, x1, y1,
                                fill=style["fill"],
                                outline=style["outline"], width=1)

    # Objective tiles (excluding centres — those get a distinct marker).
    centres = set(m.objectives)
    for col, row in m.objective_tiles:
        if (col, row) in centres:
            continue
        x0, y0, x1, y1 = _cell_to_pixels(col, row)
        canvas.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2,
                                outline=OBJ_TILE_OUTLINE, width=1)

    # `deployment+objective` cells: the DZ colour is the dominant overlay
    # (already painted above). Add the same gold square outline that regular
    # objective tiles get so these read as "objective square + DZ fill".
    for col, row in m.dz_objective_tiles:
        x0, y0, x1, y1 = _cell_to_pixels(col, row)
        canvas.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2,
                                outline=OBJ_TILE_OUTLINE, width=1)

    # Objective centres (filled circle on top of everything).
    for col, row in m.objectives:
        x0, y0, x1, y1 = _cell_to_pixels(col, row)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        r = CELL // 2 - 1
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                           fill=OBJ_CENTRE_FILL, outline="black")

    # Grid lines every 6" — drawn last so they overlay all fills lightly.
    for c in range(0, COLS + 1, 6):
        x = c * CELL
        canvas.create_line(x, 0, x, BOARD_H, fill=GRID_FILL)
    for r in range(0, ROWS + 1, 6):
        y = (ROWS - r) * CELL
        canvas.create_line(0, y, BOARD_W, y, fill=GRID_FILL)


def _terrain_counts(m: MapData) -> dict[tuple[CoverType, MovementType], int]:
    counts: dict[tuple[CoverType, MovementType], int] = {}
    for p in m.terrain:
        k = (p.cover_type, p.movement_type)
        counts[k] = counts.get(k, 0) + 1
    return counts


def build_legend(parent: ttk.Frame, m: MapData, map_path: str) -> ttk.Frame:
    frame = ttk.Frame(parent)

    title = ttk.Label(frame, text=f"Map: {map_path}",
                      font=("Consolas", 11, "bold"))
    title.pack(anchor=tk.W, pady=(0, 4))

    summary_lines = [
        f"Grid: {m.width} x {m.height}",
        f"Terrain pieces: {len(m.terrain)}",
        f"Deployment cells: A={len(m.deployment_a)}  B={len(m.deployment_b)}",
        f"Objective tiles: {len(m.objective_tiles)} "
        f"(centres: {len(m.objectives)})",
    ]
    ttk.Label(frame, text="\n".join(summary_lines),
              font=("Consolas", 10), justify=tk.LEFT).pack(
        anchor=tk.W, pady=(0, 8))

    ttk.Label(frame, text="Legend", font=("Consolas", 11, "bold")).pack(
        anchor=tk.W, pady=(2, 4))

    counts = _terrain_counts(m)
    # Terrain styles present in this map, plus the structural overlays.
    rows: list[tuple[str, str, str, str]] = []
    for key, n in sorted(counts.items(),
                         key=lambda kv: TERRAIN_STYLES.get(
                             kv[0], {}).get("label", "")):
        style = TERRAIN_STYLES.get(
            key, {"fill": "#7e7e3a", "outline": "#bfbf66",
                  "label": f"{key[0].name} + {key[1].name}"})
        rows.append(("rect", style["fill"], style["outline"],
                     f"{style['label']}  ({n} pieces)"))
    if m.deployment_a:
        rows.append(("rect", DZ_A_FILL, "", "Player A deployment zone"))
    if m.deployment_b:
        rows.append(("rect", DZ_B_FILL, "", "Player B deployment zone"))
    if m.objective_tiles:
        rows.append(("outline", BOARD_BG, OBJ_TILE_OUTLINE, "Objective tile"))
    if m.dz_objective_tiles:
        rows.append(("dz_obj", DZ_A_FILL, OBJ_TILE_OUTLINE,
                     "Objective tile inside deployment zone"))
    if m.objectives:
        rows.append(("dot", OBJ_CENTRE_FILL, "black", "Objective centre"))

    swatch_h = 22
    label_font = tkfont.Font(family="Consolas", size=10)
    text_x = 40
    right_pad = 12
    canvas_w = text_x + max(
        (label_font.measure(text) for *_, text in rows), default=0) + right_pad
    legend_canvas = tk.Canvas(
        frame, width=canvas_w, height=swatch_h * len(rows) + 6,
        bg=LEGEND_BG, highlightthickness=0)
    legend_canvas.pack(anchor=tk.W)

    for i, (kind, fill, outline, text) in enumerate(rows):
        y = 4 + i * swatch_h
        if kind == "rect":
            legend_canvas.create_rectangle(
                8, y, 32, y + 14, fill=fill,
                outline=outline if outline else fill, width=1)
        elif kind == "outline":
            legend_canvas.create_rectangle(
                8, y, 32, y + 14, fill=fill, outline="")
            legend_canvas.create_rectangle(
                10, y + 2, 30, y + 12, fill="", outline=outline, width=1)
        elif kind == "dot":
            legend_canvas.create_rectangle(
                8, y, 32, y + 14, fill=LEGEND_BG, outline="")
            legend_canvas.create_oval(
                16, y + 3, 24, y + 11, fill=fill, outline=outline)
        elif kind == "dz_obj":
            legend_canvas.create_rectangle(
                8, y, 32, y + 14, fill=fill, outline="")
            legend_canvas.create_rectangle(
                10, y + 2, 30, y + 12, fill="", outline=outline, width=1)
        legend_canvas.create_text(text_x, y + 7, text=text, fill="white",
                                  anchor=tk.W, font=label_font)

    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map", help="Path to map JSON (e.g. maps/map1.json)")
    args = ap.parse_args()

    m = load_map(args.map)

    root = tk.Tk()
    root.title(f"Map Viewer — {args.map}")
    root.configure(bg="#1e1e1e")

    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    canvas = tk.Canvas(main_frame, width=BOARD_W + 1, height=BOARD_H + 1,
                       bg=BOARD_BG, highlightthickness=0)
    canvas.pack(side=tk.LEFT)
    render_map(canvas, m)

    legend = build_legend(main_frame, m, args.map)
    legend.pack(side=tk.LEFT, padx=(12, 0), anchor=tk.N)

    root.mainloop()


if __name__ == "__main__":
    main()
