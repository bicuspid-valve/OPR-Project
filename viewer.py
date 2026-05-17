"""Graphical game replay viewer using tkinter."""
from __future__ import annotations

import colorsys
import tkinter as tk
from tkinter import ttk

import torch
import torch.nn.functional as F

from board import COLS, ROWS, OBJECTIVES, CoverType, MovementType

# Cell size in pixels
CELL = 12
BOARD_W = COLS * CELL
BOARD_H = ROWS * CELL


# Per-(cover, movement) fills/outlines for terrain pieces. Colours track
# view_map.py but the wall fill is lifted for visibility against the
# game viewer's dark canvas background.
TERRAIN_STYLES: dict[tuple[CoverType, MovementType], tuple[str, str]] = {
    (CoverType.BLOCKING,  MovementType.IMPASSIBLE):  ("#4a4a4a", "#8e8e8e"),  # wall
    (CoverType.OBSCURING, MovementType.DIFFICULT):   ("#2d5a2d", "#5fa45f"),  # forest
    (CoverType.SHELTERING, MovementType.DIFFICULT):  ("#1e4d6b", "#56a8d4"),  # water
    (CoverType.SHELTERING, MovementType.OPEN):       ("#8a6a3a", "#d4a866"),  # ruin
}
# Faint deployment-zone tints. Drawn under terrain so a wall in the DZ
# still reads as a wall.
DZ_A_FILL = "#2a3550"
DZ_B_FILL = "#502a2a"
# Subtle outline on objective overlay tiles (matches view_map's gold).
OBJ_TILE_OUTLINE = "#a07c14"

# Base hues for each player (HSV).  Player A = blue range, Player B = red range.
# Each distinct template_id gets a unique hue within the player's range.
_PLAYER_A_HUE_RANGE = (0.55, 0.72)  # blue-cyan-purple
_PLAYER_B_HUE_RANGE = (0.95, 1.12)  # red-orange (wraps around 1.0)

_PLAYER_A_SAT = 0.70
_PLAYER_B_SAT = 0.70
_BRIGHTNESS = 0.78


def _generate_palette(n: int, hue_range: tuple[float, float],
                      saturation: float, value: float) -> list[str]:
    """Generate n visually distinct colors spread across a hue range."""
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        hue = hue_range[0] + t * (hue_range[1] - hue_range[0])
        hue = hue % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def _assign_colors_by_template(owners: list[str],
                               template_ids: list[str]) -> list[str]:
    """Assign colors so same template_id within a player share the same color."""
    # Collect unique template_ids per player
    a_templates: list[str] = []
    b_templates: list[str] = []
    for owner, tid in zip(owners, template_ids):
        if owner == "A" and tid not in a_templates:
            a_templates.append(tid)
        elif owner == "B" and tid not in b_templates:
            b_templates.append(tid)

    a_palette = _generate_palette(max(len(a_templates), 1),
                                  _PLAYER_A_HUE_RANGE, _PLAYER_A_SAT, _BRIGHTNESS)
    b_palette = _generate_palette(max(len(b_templates), 1),
                                  _PLAYER_B_HUE_RANGE, _PLAYER_B_SAT, _BRIGHTNESS)

    a_map = {tid: a_palette[i] for i, tid in enumerate(a_templates)}
    b_map = {tid: b_palette[i] for i, tid in enumerate(b_templates)}

    colors = []
    for owner, tid in zip(owners, template_ids):
        if owner == "A":
            colors.append(a_map[tid])
        else:
            colors.append(b_map[tid])
    return colors


OBJ_COLORS = {"A": "#2166ac", "B": "#d6604d", "": "#888888"}
OBJ_NAMES = ["Centre", "A-side", "B-side", "Home-A", "Home-B"]

# Info box dimensions
_INFO_W = 260
_INFO_PAD = 8
_INFO_LINE_H = 14
_INFO_BG = "#1a1a2e"
_INFO_BORDER = "#5c5c8a"
_INFO_TEXT = "#e0e0e0"
_INFO_HEADER = "#ffffff"


class GameViewer:
    def __init__(self, frames: list[dict], labels: list[str], owners: list[str],
                 mode: str = "objectives", unit_points: list[int] | None = None,
                 unit_info: list[dict] | None = None,
                 parent: tk.Tk | tk.Toplevel | None = None,
                 ai_suggest_fn=None,
                 map_data=None):
        self.frames = frames
        self.labels = labels
        self.owners = owners
        self.mode = mode
        self.unit_points = unit_points or [0] * len(labels)
        self.unit_info = unit_info or []
        self.current = 0
        self.n_units = len(labels)
        self._selected_unit: int | None = None  # index of clicked unit
        self._parent = parent  # if provided, use Toplevel instead of Tk
        self._ai_suggest_fn = ai_suggest_fn  # callback(frame_idx) -> dict
        self._next_suggest_fn = None  # set externally by PlayViewer
        # MapData (terrain + DZ cells + objectives) for the game just played.
        # When set, _render paints the terrain pieces and DZ tints; objective
        # positions are read from map_data.objectives instead of the legacy
        # module global. None ⇒ legacy empty-board look.
        self.map_data = map_data

        # Extract template_ids from unit_info (fall back to index-based)
        template_ids = []
        for i in range(self.n_units):
            if i < len(self.unit_info) and 'template_id' in self.unit_info[i]:
                template_ids.append(self.unit_info[i]['template_id'])
            else:
                template_ids.append(str(i))

        # Assign colors by template_id
        self.colors = _assign_colors_by_template(owners, template_ids)

        self._build_ui()
        self._render()

    def _build_ui(self):
        if self._parent is not None:
            self.root = tk.Toplevel(self._parent)
        else:
            self.root = tk.Tk()
        self.root.title("Grid Tactical Simulator \u2014 Game Replay")
        self.root.configure(bg="#1e1e1e")

        # Main layout: board on left, info panel on right
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: board canvas
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        self.canvas = tk.Canvas(left_frame, width=BOARD_W + 1, height=BOARD_H + 1,
                                bg="#2d2d2d", highlightthickness=0)
        self.canvas.pack(padx=5, pady=5)

        # Bind click on canvas
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Navigation buttons
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(pady=5)

        self.btn_prev = ttk.Button(nav_frame, text="< Previous", command=self._prev)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.frame_label = ttk.Label(nav_frame, text="0 / 0", font=("Consolas", 11))
        self.frame_label.pack(side=tk.LEFT, padx=10)

        self.btn_next = ttk.Button(nav_frame, text="Next >", command=self._next)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        if self._ai_suggest_fn is not None:
            self.btn_ai_suggest = ttk.Button(
                nav_frame, text="AI Suggestion",
                command=self._on_ai_suggest)
            self.btn_ai_suggest.pack(side=tk.LEFT, padx=15)

        # Activation log (scrolling text) — under the board
        log_label = ttk.Label(left_frame, text="Activation Log:",
                              font=("Consolas", 10, "bold"))
        log_label.pack(anchor=tk.W, padx=5, pady=(10, 0))

        log_frame = ttk.Frame(left_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.log_text = tk.Text(log_frame, width=72, height=12,
                                bg="#1e1e1e", fg="white",
                                font=("Consolas", 9), wrap=tk.WORD,
                                state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Right: info panel
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Round / description at top
        self.desc_var = tk.StringVar(value="")
        desc_label = ttk.Label(right_frame, textvariable=self.desc_var,
                               font=("Consolas", 10), wraplength=350, justify=tk.LEFT)
        desc_label.pack(anchor=tk.W, pady=(5, 10))

        # Objective status
        self.obj_var = tk.StringVar(value="")
        obj_label = ttk.Label(right_frame, textvariable=self.obj_var,
                              font=("Consolas", 10), justify=tk.LEFT)
        obj_label.pack(anchor=tk.W, pady=(0, 10))

        # NN Assessment panel (only shown when ml_assessment data is present)
        self._has_ml = any(f.get('ml_assessment') for f in self.frames)
        if self._has_ml:
            # Detect tactical vs strategic from first assessment
            first_assess = next((f['ml_assessment'] for f in self.frames if f.get('ml_assessment')), {})
            if 'planning_candidates' in first_assess:
                nn_label = "MC Planning Assessment (Player A)"
            elif 'move_type' in first_assess:
                nn_label = "NN Tactical Assessment (Player A)"
            elif 'engagement' in first_assess:
                nn_label = "NN Tactical Assessment (Player A)"
            else:
                nn_label = "NN Assessment (Player A)"
            self.nn_frame = ttk.LabelFrame(right_frame, text=nn_label)
            self.nn_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
            nn_inner = tk.Frame(self.nn_frame)
            nn_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)
            self.nn_assessment_text = tk.Text(
                nn_inner, font=("Consolas", 9), wrap=tk.NONE,
                height=28, state=tk.DISABLED, relief=tk.FLAT,
                bg=ttk.Style().lookup("TLabelframe", "background") or "#f0f0f0",
            )
            nn_scrollbar = ttk.Scrollbar(nn_inner, orient=tk.VERTICAL,
                                         command=self.nn_assessment_text.yview)
            self.nn_assessment_text.configure(yscrollcommand=nn_scrollbar.set)
            self.nn_assessment_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            nn_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Attention Focus panel (shown when ml_assessment has attention data)
        self._has_attn = any(
            f.get('ml_assessment', {}).get('attention_weights')
            for f in self.frames
        )
        if self._has_attn:
            self.attn_frame = ttk.LabelFrame(right_frame, text="Attention Focus")
            self.attn_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
            attn_inner = tk.Frame(self.attn_frame)
            attn_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)
            self.attn_text = tk.Text(
                attn_inner, font=("Consolas", 9), wrap=tk.NONE,
                height=14, state=tk.DISABLED, relief=tk.FLAT,
                bg=ttk.Style().lookup("TLabelframe", "background") or "#f0f0f0",
            )
            attn_scrollbar = ttk.Scrollbar(attn_inner, orient=tk.VERTICAL,
                                           command=self.attn_text.yview)
            self.attn_text.configure(yscrollcommand=attn_scrollbar.set)
            self.attn_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            attn_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ML Features panel (shown when a unit is selected)
        self.ml_frame = ttk.LabelFrame(right_frame, text="ML Input Features")
        self.ml_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
        self.ml_features_var = tk.StringVar(value="(click a unit to inspect)")
        self.ml_features_label = ttk.Label(self.ml_frame,
                                            textvariable=self.ml_features_var,
                                            font=("Consolas", 9), justify=tk.LEFT)
        self.ml_features_label.pack(anchor=tk.W, padx=5, pady=3)

        # Side-by-side: Unit Key + Combat Stats
        panels_frame = ttk.Frame(right_frame)
        panels_frame.pack(fill=tk.X, anchor=tk.W)

        # Unit key (left panel)
        key_panel = ttk.Frame(panels_frame)
        key_panel.pack(side=tk.LEFT, anchor=tk.NW)

        key_label = ttk.Label(key_panel, text="Unit Key:", font=("Consolas", 10, "bold"))
        key_label.pack(anchor=tk.W)

        self.key_canvas = tk.Canvas(key_panel, width=280,
                                    height=max(20 * self.n_units + 10, 100),
                                    bg="#2d2d2d", highlightthickness=0)
        self.key_canvas.pack(anchor=tk.W)

        # Draw unit key
        for i in range(self.n_units):
            y = 5 + i * 20
            self.key_canvas.create_rectangle(5, y, 17, y + 12,
                                             fill=self.colors[i], outline="")
            self.key_canvas.create_text(22, y + 6, text=self.labels[i],
                                        fill="white", anchor=tk.W,
                                        font=("Consolas", 9))

        # Combat stats (right panel)
        stats_panel = ttk.Frame(panels_frame)
        stats_panel.pack(side=tk.LEFT, anchor=tk.NW, padx=(10, 0))

        stats_label = ttk.Label(stats_panel, text="Combat Stats:",
                                font=("Consolas", 10, "bold"))
        stats_label.pack(anchor=tk.W)

        self.stats_var = tk.StringVar(value="(no combat this activation)")
        stats_display = ttk.Label(stats_panel, textvariable=self.stats_var,
                                  font=("Consolas", 9), justify=tk.LEFT,
                                  wraplength=250)
        stats_display.pack(anchor=tk.W)

        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())

    def _on_canvas_click(self, event):
        """Handle click on the board canvas — find which unit (if any) was clicked."""
        # Convert pixel coords to grid coords
        col = event.x // CELL
        row = ROWS - 1 - (event.y // CELL)  # flip Y back

        frame = self.frames[self.current]
        clicked_unit = None
        for ui in range(self.n_units):
            for pos_col, pos_row in frame['positions'][ui]:
                if pos_col == col and pos_row == row:
                    clicked_unit = ui
                    break
            if clicked_unit is not None:
                break

        if clicked_unit is not None:
            self._selected_unit = clicked_unit
        else:
            self._selected_unit = None

        # Redraw to show/hide info box
        self._render()

    def _prev(self):
        if self.current > 0:
            self.current -= 1
            self._selected_unit = None
            self._render()

    def _next(self):
        if self.current < len(self.frames) - 1:
            self.current += 1
            self._selected_unit = None
            self._render()

    def _render(self):
        frame = self.frames[self.current]
        self.canvas.delete("all")

        # Map-driven overlays: deployment-zone tint + terrain pieces +
        # objective-tile outlines. Painted before the grid so the grid sits
        # on top and stays legible. No-ops when no MapData was supplied.
        if self.map_data is not None:
            for col, row in self.map_data.deployment_a:
                x = col * CELL
                y = (ROWS - 1 - row) * CELL
                self.canvas.create_rectangle(x, y, x + CELL, y + CELL,
                                             fill=DZ_A_FILL, outline="")
            for col, row in self.map_data.deployment_b:
                x = col * CELL
                y = (ROWS - 1 - row) * CELL
                self.canvas.create_rectangle(x, y, x + CELL, y + CELL,
                                             fill=DZ_B_FILL, outline="")
            for piece in self.map_data.terrain:
                style = TERRAIN_STYLES.get(
                    (piece.cover_type, piece.movement_type),
                    ("#7e7e3a", "#bfbf66"))
                x0 = piece.x_lo * CELL
                x1 = (piece.x_hi + 1) * CELL
                # y_hi → top pixel after flip; y_lo → bottom.
                y0 = (ROWS - 1 - piece.y_hi) * CELL
                y1 = (ROWS - piece.y_lo) * CELL
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill=style[0], outline=style[1], width=1)
            # Objective-tile overlay (the cluster squares around each obj
            # centre). Skip the centres themselves — they get the standard
            # circle marker further down.
            centres = {(int(round(c)), int(round(r))) for c, r in self.map_data.objectives}
            for col, row in self.map_data.objective_tiles:
                if (col, row) in centres:
                    continue
                x = col * CELL
                y = (ROWS - 1 - row) * CELL
                self.canvas.create_rectangle(x + 2, y + 2, x + CELL - 2, y + CELL - 2,
                                             outline=OBJ_TILE_OUTLINE, width=1)
            for col, row in self.map_data.dz_objective_tiles:
                x = col * CELL
                y = (ROWS - 1 - row) * CELL
                self.canvas.create_rectangle(x + 2, y + 2, x + CELL - 2, y + CELL - 2,
                                             outline=OBJ_TILE_OUTLINE, width=1)

        # Draw grid lines (sparse — every 6 inches)
        for c in range(0, COLS + 1, 6):
            x = c * CELL
            self.canvas.create_line(x, 0, x, BOARD_H, fill="#3a3a3a")
        for r in range(0, ROWS + 1, 6):
            y = (ROWS - r) * CELL  # flip Y so row 0 is at bottom
            self.canvas.create_line(0, y, BOARD_W, y, fill="#3a3a3a")

        # Legacy DZ row dividers — only on the empty-board layout. Map-driven
        # layouts express the DZ via the tinted cells above, so the hardcoded
        # row=12/row=36 lines don't represent map2's irregular DZ.
        if self.map_data is None:
            for row_line in [12, 36]:
                y = (ROWS - row_line) * CELL
                self.canvas.create_line(0, y, BOARD_W, y, fill="#555555", dash=(4, 4))

        # Draw objectives (objectives mode only). Prefer the map's objective
        # list when available so map2's centres (which differ from the legacy
        # constants for side objectives) are rendered correctly even if the
        # module-level OBJECTIVES wasn't mutated in this process.
        obj_ctrl = frame['objectives']
        objectives_to_draw = (self.map_data.objectives
                              if self.map_data is not None else OBJECTIVES)
        if self.mode != "kill_points":
            for oi, (oc, orow) in enumerate(objectives_to_draw):
                if oi >= len(obj_ctrl):
                    break
                x = oc * CELL + CELL // 2
                y = (ROWS - orow) * CELL - CELL // 2
                r = CELL * 2
                color = OBJ_COLORS.get(obj_ctrl[oi], "#888888")
                self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                        outline=color, width=2, dash=(3, 3))
                obj_label = OBJ_NAMES[oi][0] if oi < 3 else OBJ_NAMES[oi][-1]
                self.canvas.create_text(x, y, text=obj_label,
                                        fill=color, font=("Consolas", 8, "bold"))

        # Draw models
        for ui in range(self.n_units):
            positions = frame['positions'][ui]
            color = self.colors[ui]
            is_selected = (ui == self._selected_unit)
            for col, row in positions:
                x = col * CELL
                y = (ROWS - 1 - row) * CELL  # flip Y
                self.canvas.create_rectangle(x + 1, y + 1, x + CELL - 1, y + CELL - 1,
                                             fill=color, outline="")
                # Highlight selected unit's models with a white border
                if is_selected:
                    self.canvas.create_rectangle(x, y, x + CELL, y + CELL,
                                                 outline="#ffffff", width=2)

        # Highlight top-attended units on the board
        attn_ranking = self._get_attention_ranking(frame)
        if attn_ranking is not None:
            top_friendly_idx, top_enemy_idx = attn_ranking
            # Draw gold outline for top-attended friendly unit
            if top_friendly_idx is not None and top_friendly_idx < self.n_units:
                for col, row in frame['positions'][top_friendly_idx]:
                    x = col * CELL
                    y = (ROWS - 1 - row) * CELL
                    self.canvas.create_rectangle(
                        x - 1, y - 1, x + CELL + 1, y + CELL + 1,
                        outline="#ffd700", width=2)
            # Draw magenta outline for top-attended enemy unit
            if top_enemy_idx is not None and top_enemy_idx < self.n_units:
                for col, row in frame['positions'][top_enemy_idx]:
                    x = col * CELL
                    y = (ROWS - 1 - row) * CELL
                    self.canvas.create_rectangle(
                        x - 1, y - 1, x + CELL + 1, y + CELL + 1,
                        outline="#ff00ff", width=2)

        # Draw unit info box if a unit is selected
        if self._selected_unit is not None:
            self._draw_info_box(frame)

        # Update frame counter
        self.frame_label.configure(
            text=f"Frame {self.current + 1} / {len(self.frames)}  |  Round {frame['round']}")

        # Update description
        self.desc_var.set(frame['description'])

        # Update scoring display
        if self.mode == "kill_points":
            a_kp = 0
            b_kp = 0
            for ui in range(self.n_units):
                if frame['alive'][ui] <= 0:
                    if self.owners[ui] == "B":
                        a_kp += self.unit_points[ui]
                    else:
                        b_kp += self.unit_points[ui]
            self.obj_var.set(f"Kill Points:\n  Player A: {a_kp}pts\n  Player B: {b_kp}pts")
        else:
            obj_parts = []
            for oi, ctrl in enumerate(obj_ctrl):
                status = f"Player {ctrl}" if ctrl else "Neutral"
                obj_parts.append(f"  {OBJ_NAMES[oi]}: {status}")
            self.obj_var.set("Objectives:\n" + "\n".join(obj_parts))

        # Update NN assessment display
        if self._has_ml:
            ml_assessment = frame.get('ml_assessment')
            self.nn_assessment_text.configure(state=tk.NORMAL)
            self.nn_assessment_text.delete("1.0", tk.END)
            if ml_assessment:
                self.nn_assessment_text.insert("1.0", self._format_nn_assessment(ml_assessment))
            else:
                self.nn_assessment_text.insert("1.0", "(pre-game — no assessment yet)")
            self.nn_assessment_text.configure(state=tk.DISABLED)

        # Update attention focus panel
        if self._has_attn:
            self.attn_text.configure(state=tk.NORMAL)
            self.attn_text.delete("1.0", tk.END)
            ml_assessment = frame.get('ml_assessment')
            if ml_assessment and ml_assessment.get('attention_weights'):
                self.attn_text.insert("1.0", self._format_attention(ml_assessment))
            else:
                self.attn_text.insert("1.0", "(no attention data this frame)")
            self.attn_text.configure(state=tk.DISABLED)

        # Update activation log — show all descriptions up to current frame
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        for i, f in enumerate(self.frames[:self.current + 1]):
            desc = f.get('description', '')
            if desc:
                rnd = f.get('round', 0)
                prefix = f"[R{rnd}] " if rnd > 0 else "[--] "
                self.log_text.insert(tk.END, prefix + desc + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

        # Update ML features panel
        if self._selected_unit is not None:
            self.ml_features_var.set(self._format_ml_features(frame))
        else:
            self.ml_features_var.set("(click a unit to inspect)")

        # Update combat stats
        combat_stats = frame.get('combat_stats')
        if combat_stats:
            combat_type = combat_stats.get('combat_type', 'shooting')

            if combat_type == 'melee':
                lines = self._format_melee_stats(combat_stats)
            else:
                lines = self._format_shooting_stats(combat_stats)
            self.stats_var.set("\n".join(lines))
        else:
            self.stats_var.set("(no combat this activation)")

    def _draw_info_box(self, frame: dict):
        """Draw a floating info box on the canvas for the selected unit."""
        ui = self._selected_unit
        if ui is None:
            return

        alive = frame['alive'][ui]

        # Build info lines
        lines: list[tuple[str, str]] = []  # (text, color)
        lines.append((self.labels[ui], _INFO_HEADER))
        lines.append((f"Owner: Player {self.owners[ui]}", _INFO_TEXT))
        lines.append((f"Models alive: {alive}", _INFO_TEXT))

        # Status flags
        status_parts = []
        if frame.get('activated', [False] * len(self.labels))[ui]:
            status_parts.append("Activated")
        if frame.get('shaken', [False] * len(self.labels))[ui]:
            status_parts.append("Shaken")
        if frame.get('fatigued', [False] * len(self.labels))[ui]:
            status_parts.append("Fatigued")
        if status_parts:
            lines.append((", ".join(status_parts), "#ffcc44"))
        else:
            lines.append(("Ready", "#66ff66"))

        if ui < len(self.unit_info):
            info = self.unit_info[ui]
            lines.append((f"Points: {info['points']}  |  Q{info['quality']}+  D{info['defense']}+",
                          _INFO_TEXT))
            lines.append((f"Role: {info['ai_role']}  ({info['combat_preference']})",
                          _INFO_TEXT))
            if info['special']:
                lines.append(("Rules: " + ", ".join(info['special']), "#aaccff"))
            lines.append(("", _INFO_TEXT))  # spacer
            lines.append(("Weapons:", _INFO_HEADER))
            for wline in info['weapons']:
                lines.append(("  " + wline, _INFO_TEXT))
        else:
            lines.append((f"Points: {self.unit_points[ui]}", _INFO_TEXT))

        # Calculate box dimensions
        n_lines = len(lines)
        box_h = _INFO_PAD * 2 + n_lines * _INFO_LINE_H + 4

        # Position the box near the unit's centroid, but keep it on-screen
        positions = frame['positions'][ui]
        if positions:
            avg_col = sum(c for c, r in positions) / len(positions)
            avg_row = sum(r for c, r in positions) / len(positions)
            anchor_x = int(avg_col * CELL) + CELL
            anchor_y = int((ROWS - 1 - avg_row) * CELL) - box_h - CELL
        else:
            anchor_x = 20
            anchor_y = 20

        # Clamp to canvas bounds
        if anchor_x + _INFO_W > BOARD_W:
            anchor_x = BOARD_W - _INFO_W - 4
        if anchor_x < 4:
            anchor_x = 4
        if anchor_y < 4:
            anchor_y = 4
        if anchor_y + box_h > BOARD_H:
            anchor_y = BOARD_H - box_h - 4

        # Draw background
        self.canvas.create_rectangle(anchor_x, anchor_y,
                                     anchor_x + _INFO_W, anchor_y + box_h,
                                     fill=_INFO_BG, outline=_INFO_BORDER, width=2)

        # Draw text lines
        ty = anchor_y + _INFO_PAD + 2
        for text, color in lines:
            if text:
                font = ("Consolas", 8, "bold") if color == _INFO_HEADER else ("Consolas", 8)
                self.canvas.create_text(anchor_x + _INFO_PAD, ty,
                                        text=text, fill=color, anchor=tk.NW, font=font)
            ty += _INFO_LINE_H

    def _get_attention_ranking(self, frame: dict) -> tuple[int | None, int | None] | None:
        """Return (top_friendly_unit_index, top_enemy_unit_index) in viewer unit indices.

        Attention weights slots 0-9 = friendly (Player A), 10-19 = enemy (Player B).
        Maps these back to the viewer's unit index list.
        Returns None if no attention data available.
        """
        ml_assessment = frame.get('ml_assessment')
        if not ml_assessment or not ml_assessment.get('attention_weights'):
            return None

        weights = ml_assessment['attention_weights']  # (20,) list
        friendly_names = ml_assessment.get('friendly_names', [])
        enemy_names = ml_assessment.get('enemy_names', [])

        # Find top friendly slot (0-9) among alive units
        top_friendly_slot = None
        top_friendly_w = -1.0
        for i in range(10):
            if i < len(friendly_names) and friendly_names[i] is not None:
                if weights[i] > top_friendly_w:
                    top_friendly_w = weights[i]
                    top_friendly_slot = i

        # Find top enemy slot (10-19) among alive units
        top_enemy_slot = None
        top_enemy_w = -1.0
        for i in range(10):
            if i < len(enemy_names) and enemy_names[i] is not None:
                if weights[10 + i] > top_enemy_w:
                    top_enemy_w = weights[10 + i]
                    top_enemy_slot = i

        # Map slots to viewer unit indices. Player A units come first, then B.
        a_indices = [ui for ui in range(self.n_units) if self.owners[ui] == "A"]
        b_indices = [ui for ui in range(self.n_units) if self.owners[ui] == "B"]

        top_friendly_idx = (a_indices[top_friendly_slot]
                            if top_friendly_slot is not None and top_friendly_slot < len(a_indices)
                            else None)
        top_enemy_idx = (b_indices[top_enemy_slot]
                         if top_enemy_slot is not None and top_enemy_slot < len(b_indices)
                         else None)

        return top_friendly_idx, top_enemy_idx

    def _format_attention(self, assessment: dict) -> str:
        """Format attention weights into a ranked list for the attention panel."""
        weights = assessment['attention_weights']  # (20,) list
        friendly_names = assessment.get('friendly_names', [])
        enemy_names = assessment.get('enemy_names', [])

        # Build friendly ranked list (slots 0-9)
        friendly_entries = []
        for i in range(10):
            name = friendly_names[i] if i < len(friendly_names) else None
            if name is not None:
                friendly_entries.append((weights[i], name))
        friendly_entries.sort(key=lambda e: e[0], reverse=True)

        # Build enemy ranked list (slots 10-19)
        enemy_entries = []
        for i in range(10):
            name = enemy_names[i] if i < len(enemy_names) else None
            if name is not None:
                enemy_entries.append((weights[10 + i], name))
        enemy_entries.sort(key=lambda e: e[0], reverse=True)

        lines = []
        lines.append("Friendly Units (by attention):")
        for w, name in friendly_entries:
            bar = "\u2588" * max(1, int(w * 80))
            lines.append(f"  {w:.3f}  {name}  {bar}")

        lines.append("")
        lines.append("Enemy Units (by attention):")
        for w, name in enemy_entries:
            bar = "\u2588" * max(1, int(w * 80))
            lines.append(f"  {w:.3f}  {name}  {bar}")

        lines.append("")
        lines.append("Board: gold = top friendly, magenta = top enemy")

        return "\n".join(lines)

    def _format_ml_features(self, frame: dict) -> str:
        """Format ML input features for the selected unit."""
        ui = self._selected_unit
        if ui is None or ui >= len(self.unit_info):
            return "(no data)"
        info = self.unit_info[ui]
        ml = info.get('ml_features')
        if ml is None:
            return "(no ML data — game ran without ML model)"

        alive = frame['alive'][ui]
        starting_models = info['models']
        tough = info.get('tough', 0)
        # Compute survival fraction from frame data
        if tough and 'wounds' in frame:
            wounds = frame['wounds'][ui]
            total_start = tough * starting_models
            total_rem = sum(tough - w for w in wounds)
            survival = total_rem / max(total_start, 1)
        else:
            survival = alive / max(starting_models, 1)

        ap_labels = ["AP0", "AP1", "AP2", "AP3", "AP4+"]
        ap_str = ", ".join(l for l, v in zip(ap_labels, ml['ap_flags']) if v)
        deadly_labels = ["Non-Deadly", "Deadly(3)", "Deadly(6+)"]
        deadly_str = ", ".join(l for l, v in zip(deadly_labels, ml['deadly_flags']) if v)

        ability_parts = []
        for key, label in [('flying', 'Flying'), ('artillery', 'Artillery'),
                           ('stealth', 'Stealth'), ('fearless', 'Fearless'),
                           ('fear', 'Fear')]:
            if ml[key]:
                ability_parts.append(label)
        abilities_str = ", ".join(ability_parts) if ability_parts else "None"

        lines = [
            f"{self.labels[ui]}  (Player {self.owners[ui]})",
            f"",
            f"Toughness:    {ml['toughness']:.4f}  (raw: {int(ml['toughness'] * 24)})",
            f"Model Count:  {ml['model_count']:.4f}  (raw: {starting_models})",
            f"Defense:      {ml['defense']:.4f}  (raw: {info['defense']}+)",
            f"Ranged Dmg:   {ml['ranged_dmg']:.4f}",
            f"Melee Dmg:    {ml['melee_dmg']:.4f}  ({', '.join(f'{v:.4f}' for v in ml.get('melee_dmg_list', []))})",
            f"Speed:        {ml['speed']:.4f}  (raw: {int(ml['speed'] * 24)}\")",
            f"Survival:     {survival:.4f}  ({alive}/{starting_models} models"
            f"{', ' + str(total_rem) + '/' + str(total_start) + ' HP' if tough else ''})",
            f"Points Frac:  {ml['points_frac']:.4f}  (raw: {info['points']}pts)",
            f"",
            f"AP:     {ap_str}",
            f"Deadly: {deadly_str}",
            f"Abilities: {abilities_str}",
        ]

        # Tactical-model dynamic state flags (change per-frame)
        activated = frame.get('activated', [False] * len(self.labels))
        shaken = frame.get('shaken', [False] * len(self.labels))
        fatigued = frame.get('fatigued', [False] * len(self.labels))
        state_flags = []
        if ui < len(activated) and activated[ui]:
            state_flags.append("Activated")
        if ui < len(shaken) and shaken[ui]:
            state_flags.append("Shaken")
        if ui < len(fatigued) and fatigued[ui]:
            state_flags.append("Fatigued")
        lines.append(f"State Flags: {', '.join(state_flags) if state_flags else 'Ready'}")

        # Per-enemy ranged damage at each range threshold
        ranged_matchups = ml.get('ranged_matchups', [])
        enemy_names = ml.get('enemy_names', [])
        if ranged_matchups:
            thresholds = [6, 9, 12, 18, 24, 30, 36]
            lines.append("")
            lines.append(f"Ranged Damage vs Enemies  (thresholds: {thresholds}\")")
            for j, row in enumerate(ranged_matchups):
                if j >= len(enemy_names):
                    break
                # Skip padded zero-only entries beyond actual enemies
                if all(v == 0.0 for v in row):
                    continue
                name = enemy_names[j] if j < len(enemy_names) else f"Enemy {j}"
                vals = " ".join(f"{v:.3f}" for v in row)
                lines.append(f"  vs {name}: [{vals}]")

        # Per-enemy melee damage
        melee_dmg_list = ml.get('melee_dmg_list', [])
        if melee_dmg_list and enemy_names:
            lines.append("")
            lines.append("Melee Damage vs Enemies:")
            for j, v in enumerate(melee_dmg_list):
                if j >= len(enemy_names):
                    break
                if v == 0.0:
                    continue
                name = enemy_names[j] if j < len(enemy_names) else f"Enemy {j}"
                lines.append(f"  vs {name}: {v:.3f}")

        return "\n".join(lines)

    def _format_planning_assessment(self, assessment: dict, obj_names: list[str]) -> str:
        """Format Monte Carlo planning candidates for display."""
        candidates = assessment['planning_candidates']
        if not candidates:
            return "Planning: no candidates"

        # Sort by value descending
        sorted_cands = sorted(candidates, key=lambda c: c['value'], reverse=True)

        lines = [f"MC Planning — {len(candidates)} candidates"]
        lines.append("")

        for i, c in enumerate(sorted_cands):
            marker = " *" if c.get('selected') else ""
            target_str = f"  tgt:{c['top_target']}" if c.get('top_target') else ""
            move_type = c.get('move_type', '?')
            dir_angle = c.get('direction_angle', 0)
            dist_frac = c.get('distance_frac', 0)
            lines.append(
                f"{'>' if c.get('selected') else ' '} {c['unit_name']}  "
                f"{c['action']}  {move_type}  "
                f"dir:{dir_angle:.1f}  dist:{dist_frac:.2f}  "
                f"val:{c['value']:+.3f}{target_str}{marker}"
            )
            if c.get('reason'):
                lines.append(f"    {c['reason']}")
            if i < len(sorted_cands) - 1:
                lines.append("")

        return "\n".join(lines)

    def _format_nn_assessment(self, assessment: dict) -> str:
        """Format the neural network's assessment for display.

        Handles strategic (per-unit list), tactical (single-unit), and
        planning (Monte Carlo candidate list) formats.
        """
        obj_names = ["Centre", "A-side", "B-side", "Home-A", "Home-B"]
        role_names = ["killer", "obj_holder"]
        stance_names = ["kite", "normal", "aggressive"]

        if 'planning_candidates' in assessment:
            return self._format_planning_assessment(assessment, obj_names)

        value = assessment['value']
        lines = [f"State Value: {value:+.3f}"]
        lines.append("")

        if 'units' in assessment:
            # Strategic model: all units listed by activation order
            sorted_units = sorted(assessment['units'],
                                  key=lambda u: u['activation_score'], reverse=True)
            for u in sorted_units:
                name = u['name']
                role = u['role']
                obj = obj_names[u['objective']]
                stance = u['stance']
                cpref = u['combat_preference']
                act = u['activation_score']
                lines.append(f"{name}:")
                lines.append(f"  {role} -> {obj}  |  {cpref}/{stance}  |  act:{act:.1f}")

                # Full probability distributions
                role_probs = u.get('role_probs')
                if role_probs:
                    rp = "  role: " + "  ".join(f"{role_names[j]}:{p:.0%}" for j, p in enumerate(role_probs))
                    lines.append(rp)
                obj_probs = u.get('objective_probs')
                if obj_probs:
                    op = "  obj:  " + "  ".join(f"{obj_names[j]}:{p:.0%}" for j, p in enumerate(obj_probs))
                    lines.append(op)
                stance_probs = u.get('stance_probs')
                if stance_probs:
                    sp = "  stance: " + "  ".join(f"{stance_names[j]}:{p:.0%}" for j, p in enumerate(stance_probs))
                    lines.append(sp)
                cpref_prob = u.get('combat_pref_prob')
                if cpref_prob is not None:
                    lines.append(f"  combat: ranged:{cpref_prob:.0%}  melee:{1-cpref_prob:.0%}")
                tp = u.get('target_priority')
                if tp is not None:
                    lines.append(f"  target mult: {tp:.2f}")

        elif 'move_type' in assessment:
            # Tactical v2 model: free-movement with unit selection
            name = assessment.get('selected_name', '?')
            move_type = assessment.get('move_type', '?')
            move_conf = assessment.get('move_type_confidence', 0)
            lines.append(f"Selected: {name}  (slot {assessment.get('selected_slot', '?')})")

            # Unit selection logits
            unit_logits = assessment.get('unit_selection_logits')
            friendly_names = assessment.get('friendly_names', [])
            if unit_logits:
                logits_t = torch.tensor(unit_logits)
                probs = F.softmax(logits_t[logits_t != float('-inf')].float(),
                                  dim=-1) if (logits_t != float('-inf')).any() else logits_t
                prob_parts = []
                pi = 0
                for si, lv in enumerate(unit_logits):
                    if lv != float('-inf'):
                        p = probs[pi].item() if pi < len(probs) else 0.0
                        marker = " *" if si == assessment.get('selected_slot') else ""
                        label = friendly_names[si] if si < len(friendly_names) and friendly_names[si] else f"slot {si}"
                        prob_parts.append(f"  {label}: {p:.0%}{marker}")
                        pi += 1
                if prob_parts:
                    lines.append("Unit Selection:")
                    lines.extend(prob_parts)

            lines.append("")
            # Move type with full distribution (binary: move / charge)
            move_probs = assessment.get('move_type_probs')
            move_names = ['move', 'charge']
            if move_probs:
                lines.append("Move Type:")
                for mi, mn in enumerate(move_names):
                    if mi >= len(move_probs):
                        break
                    marker = " *" if mn == move_type else ""
                    lines.append(f"  {mn}: {move_probs[mi]:.0%}{marker}")
            else:
                lines.append(f"Move: {move_type}  ({move_conf:.0%} confidence)")

            # Destination pointer selection (unified hold/advance/rush)
            if move_type == 'move':
                dest_sel = assessment.get('dest_selected')
                n_cand = assessment.get('dest_n_candidates')
                dest_entropy = assessment.get('dest_entropy')
                if dest_sel is not None:
                    dc, dr = dest_sel
                    detail = f"  destination: ({dc}, {dr})"
                    if n_cand is not None:
                        detail += f"  [{n_cand} candidates]"
                    if dest_entropy is not None:
                        detail += f"  entropy:{dest_entropy:.2f}"
                    lines.append(detail)
                top3 = assessment.get('dest_top3')
                if top3:
                    lines.append("  top hexes:")
                    for tc, tr, tp in top3:
                        marker = " *" if dest_sel is not None and (tc, tr) == tuple(dest_sel) else ""
                        lines.append(f"    ({tc}, {tr}): {tp:.0%}{marker}")

            # Charge target with logits
            if move_type == 'charge':
                ct_idx = assessment.get('charge_target_idx', 0)
                enemy_names = assessment.get('enemy_names', [])
                ct_name = enemy_names[ct_idx] if ct_idx < len(enemy_names) and enemy_names[ct_idx] else f"slot {ct_idx}"
                lines.append(f"  charge target: {ct_name}")
                ct_logits = assessment.get('charge_target_logits')
                if ct_logits:
                    ct_t = torch.tensor(ct_logits)
                    valid = ct_t != float('-inf')
                    if valid.any():
                        ct_probs = F.softmax(ct_t[valid].float(), dim=-1)
                        pi = 0
                        for si, lv in enumerate(ct_logits):
                            if lv != float('-inf'):
                                p = ct_probs[pi].item()
                                en = enemy_names[si] if si < len(enemy_names) and enemy_names[si] else f"slot {si}"
                                marker = " *" if si == ct_idx else ""
                                lines.append(f"    {en}: {p:.0%}{marker}")
                                pi += 1

            # Action/reason from execution logic
            action = assessment.get('action')
            reason = assessment.get('reason')
            if action:
                lines.append(f"  action: {action}" + (f"  ({reason})" if reason else ""))

            # Shooting target scores (for hold/advance)
            target_scores = assessment.get('target_scores')
            enemy_names = assessment.get('enemy_names', [])
            if target_scores and enemy_names:
                entries = [
                    (score, ename)
                    for score, ename in zip(target_scores, enemy_names)
                    if ename is not None
                ]
                entries.sort(key=lambda e: e[0], reverse=True)
                if entries:
                    lines.append("")
                    shoot_idx = assessment.get('shoot_target_idx')
                    shoot_name = enemy_names[shoot_idx] if shoot_idx is not None and shoot_idx < len(enemy_names) and enemy_names[shoot_idx] else None
                    header = "Shoot Target Priority"
                    if shoot_name:
                        header += f"  (selected: {shoot_name})"
                    lines.append(header + ":")
                    for score, ename in entries:
                        marker = " <" if ename == shoot_name else ""
                        lines.append(f"  {ename}: {score:.2f}{marker}")

            # --- Auxiliary predictions ---
            self._format_aux_predictions(lines, assessment, obj_names)

        else:
            # Legacy tactical model: single selected unit per activation
            name = assessment.get('selected_name', '?')
            priority = assessment.get('priority', assessment.get('role', '?'))
            obj_raw = assessment.get('objective', 0)
            obj = obj_names[obj_raw] if isinstance(obj_raw, int) and obj_raw < len(obj_names) else str(obj_raw)
            engagement = assessment.get('engagement', assessment.get('stance', '?'))
            lines.append(f"Selected: {name}")
            lines.append(f"  {priority} -> {obj}  |  {engagement}")

            # Action/reason from execution logic
            action = assessment.get('action')
            reason = assessment.get('reason')
            if action:
                lines.append(f"  action: {action}" + (f"  ({reason})" if reason else ""))

            # Confidence scores
            pri_conf = assessment.get('priority_confidence', assessment.get('role_confidence'))
            obj_conf = assessment.get('objective_confidence')
            eng_conf = assessment.get('engagement_confidence', assessment.get('stance_confidence'))
            if pri_conf is not None:
                lines.append(f"  confidence: pri {pri_conf:.0%}  obj {obj_conf:.0%}  eng {eng_conf:.0%}")

            # Target priority scores (sorted by score descending)
            target_scores = assessment.get('target_scores', assessment.get('target_priority'))
            enemy_names = assessment.get('enemy_names')
            if target_scores and enemy_names:
                lines.append("")
                lines.append("Target Priority:")
                entries = [
                    (score, ename)
                    for score, ename in zip(target_scores, enemy_names)
                    if ename is not None
                ]
                entries.sort(key=lambda e: e[0], reverse=True)
                for score, ename in entries:
                    lines.append(f"  {ename}: {score:.2f}")

        return "\n".join(lines)

    def _format_aux_predictions(self, lines: list[str], assessment: dict,
                                obj_names: list[str]) -> None:
        """Append auxiliary prediction head outputs (survival, obj control) to lines."""
        friendly_survival = assessment.get('friendly_survival')
        enemy_survival = assessment.get('enemy_survival')
        obj_control = assessment.get('obj_control_probs')
        friendly_names = assessment.get('friendly_names', [])
        enemy_names = assessment.get('enemy_names', [])

        if not (friendly_survival or enemy_survival or obj_control):
            return

        lines.append("")
        lines.append("--- Auxiliary Predictions ---")

        if friendly_survival:
            lines.append("Friendly Survival:")
            for i, sv in enumerate(friendly_survival):
                fname = friendly_names[i] if i < len(friendly_names) and friendly_names[i] else None
                if fname is None:
                    continue
                lines.append(f"  {fname}: {sv:.0%}")

        if enemy_survival:
            lines.append("Enemy Survival:")
            for i, sv in enumerate(enemy_survival):
                ename = enemy_names[i] if i < len(enemy_names) and enemy_names[i] else None
                if ename is None:
                    continue
                lines.append(f"  {ename}: {sv:.0%}")

        if obj_control:
            ctrl_names = ["friendly", "enemy", "neutral"]
            lines.append("Objective Control:")
            for oi, probs in enumerate(obj_control):
                best = max(range(3), key=lambda j: probs[j])
                detail = "  ".join(f"{ctrl_names[j]}:{probs[j]:.0%}" for j in range(3))
                lines.append(f"  {obj_names[oi]}: {detail}")

        f_act = assessment.get('friendly_activations_remaining')
        e_act = assessment.get('enemy_activations_remaining')
        if f_act is not None and e_act is not None:
            lines.append(f"Activations Remaining:  friendly: {f_act:.1f}  enemy: {e_act:.1f}")

    def _format_shooting_stats(self, stats: dict) -> list[str]:
        mod = stats['hit_modifier']
        mod_str = f" (modifier: {mod:+d})" if mod != 0 else ""
        atk_rules = stats.get('attacker_rules', [])
        def_rules = stats.get('defender_rules', [])
        lines = [
            f"Attacker Quality: {stats['attacker_quality']}+{mod_str}",
        ]
        if atk_rules:
            lines.append(f"  [{', '.join(atk_rules)}]")
        lines.append(f"Defender Defense: {stats['defender_defense']}+")
        if def_rules:
            lines.append(f"  [{', '.join(def_rules)}]")
        lines.append("")
        for w in stats.get('attacker_weapons', []):
            count = w.get('count', 1)
            count_str = f"{count}x " if count > 1 else ""
            stat_str = f"{count_str}{w['name']}  {w['range']}\"  A{w['attacks']}"
            if w['abilities']:
                stat_str += "  " + ", ".join(w['abilities'])
            lines.append(stat_str)
        lines.append("")
        lines.append(f"Attacks: {stats['total_attacks']}")
        lines.append(f"Hits: {stats['total_hits']}")
        lines.append(f"Failed Def Rolls: {stats['total_wounds']}")
        return lines

    def _format_melee_stats(self, stats: dict) -> list[str]:
        lines = ["MELEE COMBAT"]
        # Impact info
        impact_info = stats.get('impact_info')
        if impact_info:
            lines.append(impact_info)

        # Charger attack stats
        if 'attacker_quality' in stats:
            fatigued = stats.get('fatigued', False)
            quality_str = f"Attacker Quality: {stats['attacker_quality']}+"
            if fatigued:
                quality_str += " (Fatigued: 6+ only)"
            lines.append(quality_str)
            lines.append(f"Defender Defense: {stats['defender_defense']}+")
            lines.append("")
            for w in stats.get('attacker_weapons', []):
                count = w.get('count', 1)
                count_str = f"{count}x " if count > 1 else ""
                stat_str = f"{count_str}{w['name']}  A{w['attacks']}"
                if w['abilities']:
                    stat_str += "  " + ", ".join(w['abilities'])
                lines.append(stat_str)
            lines.append("")
            lines.append(f"Attacks: {stats['total_attacks']}")
            lines.append(f"Hits: {stats['total_hits']}")
            lines.append(f"Failed Def Rolls: {stats['total_wounds']}")

        lines.append("")
        charger_w = stats.get('charger_wounds', 0)
        defender_w = stats.get('defender_wounds', 0)
        lines.append(f"Charger wounds dealt: {charger_w}")
        lines.append(f"Defender wounds dealt: {defender_w}")
        if charger_w > defender_w:
            lines.append("Charger wins melee")
        elif defender_w > charger_w:
            lines.append("Defender wins melee")
        else:
            lines.append("Melee tied")
        return lines

    # ─── AI Suggestion ────────────────────────────────────────────

    def _on_ai_suggest(self):
        """Run the AI suggestion callback in a background thread."""
        import threading
        self.btn_ai_suggest.configure(state=tk.DISABLED, text="Thinking...")
        self.root.update_idletasks()
        frame_idx = self.current

        def worker():
            try:
                result = self._ai_suggest_fn(frame_idx)
            except Exception as exc:
                result = {'error': f"Error running AI suggestion:\n{exc}"}
            self.root.after(
                0, lambda: self._show_ai_suggestion(result, frame_idx))

        threading.Thread(target=worker, daemon=True).start()

    def _show_ai_suggestion(self, data, frame_idx: int):
        """Display the AI suggestion in a visual viewer window."""
        self.btn_ai_suggest.configure(state=tk.NORMAL, text="AI Suggestion")

        if not data or 'error' in data:
            msg = (data.get('error', 'Unknown error')
                   if data else 'No data returned')
            popup = tk.Toplevel(self.root)
            popup.title("AI Suggestion")
            popup.configure(bg="#1e1e1e")
            ttk.Label(popup, text=msg,
                      font=("Consolas", 10)).pack(padx=20, pady=20)
            ttk.Button(popup, text="Close",
                       command=popup.destroy).pack(pady=(0, 10))
            popup.transient(self.root)
            popup.grab_set()
            return

        data['mode'] = self.mode

        def reroll_fn():
            return self._ai_suggest_fn(frame_idx)

        viewer = AISuggestionViewer(
            data, self.root, reroll_fn,
            next_fn=self._next_suggest_fn)
        viewer.run()

    def run(self):
        if self._parent is not None:
            # Modal: block the parent window until replay is closed
            self.root.grab_set()
            self.root.wait_window()
        else:
            self.root.mainloop()


def show_game(frames: list[dict], labels: list[str], owners: list[str],
              mode: str = "objectives", unit_points: list[int] | None = None,
              unit_info: list[dict] | None = None,
              parent: tk.Tk | tk.Toplevel | None = None,
              map_data=None):
    """Launch the game viewer window. Pass ``map_data`` (a MapData instance)
    to render the map's terrain and irregular deployment zones beneath the
    units; omit for the legacy empty-board look."""
    viewer = GameViewer(frames, labels, owners, mode=mode, unit_points=unit_points,
                        unit_info=unit_info, parent=parent, map_data=map_data)
    viewer.run()


# ===================================================================
# AI SUGGESTION VIEWER
# ===================================================================

class AISuggestionViewer:
    """Lightweight viewer showing the AI's suggested move with
    before/after board states, combat results, re-roll, and next-suggestion."""

    def __init__(self, data: dict, parent, reroll_fn, next_fn=None):
        self.frames = data['frames']       # list of frame dicts
        self.labels = data['labels']
        self.owners = data['owners']
        self.colors = data['colors']
        self.summary = data['summary']
        self.active_idx = data.get('active_idx')
        self.mode = data.get('mode', 'objectives')
        self.n_units = len(self.labels)
        self._reroll_fn = reroll_fn
        self._next_fn = next_fn
        self._parent = parent
        self.current = len(self.frames) - 1  # start on last (after) frame
        self._game_over = False

        self._build_ui()
        self._render()

    def _build_ui(self):
        self.root = tk.Toplevel(self._parent)
        self.root.title("AI Suggestion \u2014 What would the AI do?")
        self.root.configure(bg="#1e1e1e")

        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ── Left column: board + buttons ──
        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.BOTH)

        self.canvas = tk.Canvas(
            left, width=BOARD_W + 1, height=BOARD_H + 1,
            bg="#2d2d2d", highlightthickness=0)
        self.canvas.pack(padx=5, pady=5)

        # Single row: nav + actions
        btn_frame = ttk.Frame(left)
        btn_frame.pack(pady=3)

        self.btn_prev = ttk.Button(
            btn_frame, text="<", width=3, command=self._prev)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.frame_label = ttk.Label(
            btn_frame, text="", font=("Consolas", 9, "bold"))
        self.frame_label.pack(side=tk.LEFT, padx=4)

        self.btn_next_frame = ttk.Button(
            btn_frame, text=">", width=3, command=self._next)
        self.btn_next_frame.pack(side=tk.LEFT, padx=2)

        ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=6, fill=tk.Y)

        self.btn_reroll = ttk.Button(
            btn_frame, text="Re-roll", command=self._reroll)
        self.btn_reroll.pack(side=tk.LEFT, padx=3)

        if self._next_fn is not None:
            self.btn_next_suggest = ttk.Button(
                btn_frame, text="Next Suggestion",
                command=self._next_suggestion)
            self.btn_next_suggest.pack(side=tk.LEFT, padx=3)

        ttk.Button(btn_frame, text="Close",
                   command=self.root.destroy).pack(side=tk.LEFT, padx=3)

        # ── Right column: summary, description, stats ──
        right = ttk.Frame(outer)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        ttk.Label(right, text="AI Suggestion",
                  font=("Consolas", 11, "bold")).pack(
                      anchor=tk.W, pady=(5, 3))

        self.summary_text = tk.Text(
            right, font=("Consolas", 9), bg="#1a1a2e", fg="#e0e0e0",
            height=7, width=32, wrap=tk.WORD, relief=tk.FLAT,
            padx=6, pady=4)
        self.summary_text.insert("1.0", self.summary)
        self.summary_text.configure(state=tk.DISABLED)
        self.summary_text.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(right, text="Action Result",
                  font=("Consolas", 10, "bold")).pack(
                      anchor=tk.W, pady=(0, 2))

        self.desc_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.desc_var,
                  font=("Consolas", 9), wraplength=250,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 8))

        self.stats_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.stats_var,
                  font=("Consolas", 9), justify=tk.LEFT
                  ).pack(anchor=tk.W)

        # Keyboard nav
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())

    def _prev(self):
        if self.current > 0:
            self.current -= 1
            self._render()

    def _next(self):
        if self.current < len(self.frames) - 1:
            self.current += 1
            self._render()

    def _show(self, idx: int):
        self.current = max(0, min(idx, len(self.frames) - 1))
        self._render()

    def _render(self):
        frame = self.frames[self.current]
        self.canvas.delete("all")

        # Grid lines (every 6")
        for c in range(0, COLS + 1, 6):
            x = c * CELL
            self.canvas.create_line(x, 0, x, BOARD_H, fill="#3a3a3a")
        for r in range(0, ROWS + 1, 6):
            y = (ROWS - r) * CELL
            self.canvas.create_line(0, y, BOARD_W, y, fill="#3a3a3a")

        # Deployment zone lines
        for row_line in [12, 36]:
            y = (ROWS - row_line) * CELL
            self.canvas.create_line(
                0, y, BOARD_W, y, fill="#555555", dash=(4, 4))

        # Objectives
        obj_ctrl = frame.get('objectives', [])
        if self.mode != "kill_points":
            for oi, (oc, orow) in enumerate(OBJECTIVES):
                x = oc * CELL + CELL // 2
                y = (ROWS - orow) * CELL - CELL // 2
                rad = CELL * 2
                ctrl = obj_ctrl[oi] if oi < len(obj_ctrl) else ""
                color = OBJ_COLORS.get(ctrl, "#888888")
                self.canvas.create_oval(
                    x - rad, y - rad, x + rad, y + rad,
                    outline=color, width=2, dash=(3, 3))
                lbl = OBJ_NAMES[oi][0] if oi < 3 else OBJ_NAMES[oi][-1]
                self.canvas.create_text(
                    x, y, text=lbl,
                    fill=color, font=("Consolas", 8, "bold"))

        # Unit models — highlight the active unit for this frame
        frame_active = frame.get('_active_idx', self.active_idx)
        for ui in range(self.n_units):
            positions = frame['positions'][ui]
            color = self.colors[ui]
            is_active = (ui == frame_active)
            for col, row in positions:
                px = col * CELL
                py = (ROWS - 1 - row) * CELL
                self.canvas.create_rectangle(
                    px + 1, py + 1, px + CELL - 1, py + CELL - 1,
                    fill=color, outline="")
                if is_active:
                    self.canvas.create_rectangle(
                        px, py, px + CELL, py + CELL,
                        outline="#ffff00", width=2)

        # Frame counter
        self.frame_label.configure(
            text=f"{self.current + 1} / {len(self.frames)}")

        # Enable/disable nav buttons
        self.btn_prev.configure(
            state=tk.NORMAL if self.current > 0 else tk.DISABLED)
        self.btn_next_frame.configure(
            state=tk.NORMAL
            if self.current < len(self.frames) - 1
            else tk.DISABLED)

        # Description
        self.desc_var.set(frame.get('description', ''))

        # Combat stats
        cs = frame.get('combat_stats')
        if cs:
            self.stats_var.set(self._format_combat(cs))
        else:
            self.stats_var.set("")

    @staticmethod
    def _format_combat(cs: dict) -> str:
        lines: list[str] = []
        ct = cs.get('combat_type', 'shooting')
        if ct == 'melee':
            if cs.get('impact_info'):
                lines.append(cs['impact_info'])
            cw = cs.get('charger_wounds', 0)
            dw = cs.get('defender_wounds', 0)
            lines.append(f"Wounds dealt: {cw}")
            lines.append(f"Wounds received: {dw}")
            if cw > dw:
                lines.append("Charger wins melee")
            elif dw > cw:
                lines.append("Defender wins melee")
            else:
                lines.append("Melee tied")
        else:
            lines.append(
                f"Hits: {cs.get('total_hits', '?')} | "
                f"Failed Def Rolls: {cs.get('total_wounds', '?')}")
            lines.append(
                f"Saves: {cs.get('armor_saves', 0)} armour, "
                f"{cs.get('invulnerable_saves', 0)} invuln")
        return "\n".join(lines)

    def _reroll(self):
        import threading
        self.btn_reroll.configure(state=tk.DISABLED, text="Rolling...")
        self.root.update_idletasks()

        def worker():
            try:
                data = self._reroll_fn()
            except Exception as exc:
                data = {'error': str(exc)}
            self.root.after(0, lambda: self._apply_reroll(data))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_reroll(self, data):
        self.btn_reroll.configure(state=tk.NORMAL, text="Re-roll Dice")
        if not data or 'error' in data:
            return
        self.frames = data['frames']
        self.summary = data['summary']
        self.active_idx = data.get('active_idx', self.active_idx)
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", self.summary)
        self.summary_text.configure(state=tk.DISABLED)
        self.current = len(self.frames) - 1
        self._render()

    def _next_suggestion(self):
        """Advance the simulation: B moves, then show A's next suggestion."""
        import threading
        self.btn_next_suggest.configure(
            state=tk.DISABLED, text="Simulating...")
        self.root.update_idletasks()

        def worker():
            try:
                data = self._next_fn()
            except Exception as exc:
                data = {'error': str(exc)}
            self.root.after(0, lambda: self._apply_next(data))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_next(self, data):
        self.btn_next_suggest.configure(
            state=tk.NORMAL, text="Next Suggestion")

        if not data:
            return

        if 'error' in data:
            # Might still have intermediate frames to show
            extra = data.get('frames', [])
            if extra:
                self.frames.extend(extra)
                self.current = len(self.frames) - 1
                self._render()
            # Disable further advancement
            self.btn_next_suggest.configure(state=tk.DISABLED)
            self._game_over = True
            self.desc_var.set(data['error'])
            return

        # Append new frames (B action + A before/after)
        new_frames = data['frames']
        self.frames.extend(new_frames)
        self.summary = data['summary']
        self.active_idx = data.get('active_idx', self.active_idx)
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", self.summary)
        self.summary_text.configure(state=tk.DISABLED)
        self.current = len(self.frames) - 1
        self._render()

    def run(self):
        self.root.transient(self._parent)
        self.root.grab_set()
        self.root.wait_window()
