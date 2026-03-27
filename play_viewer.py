"""Interactive play mode: human vs tactical ML model using tkinter GUI."""
from __future__ import annotations

import colorsys
import copy
import json
import math
import random
import tkinter as tk
from tkinter import ttk
from pathlib import Path

from board import Board, COLS, ROWS, OBJECTIVES
from models import ResolvedUnit, UnitState
from combat import (
    resolve_shooting, check_morale, can_shoot_any, models_in_range,
    resolve_melee, resolve_impact, check_melee_morale,
)
from movement import (
    execute_movement, execute_charge_movement, execute_counter_charge,
    post_melee_separation, consolidation_move,
)
from ai import (
    pick_target, choose_action_and_goal, activation_order,
    assign_objectives, reassign_roles, _can_charge,
)
from game import (
    deploy_armies, _sync_dead_models, _collect_enemy_positions,
    _make_unit_labels, _snapshot, _base_name, _kite_range_params,
)
from ml_planning import snapshot_game_state, restore_game_state
from simulation import (
    execute_activation, get_ai_decision,
    check_game_over as sim_check_game_over,
    check_round_over as sim_check_round_over,
    start_round as sim_start_round,
    end_round as sim_end_round,
    ActivationResult,
)

# ── Visual constants (matching viewer.py) ──────────────────────────
CELL = 12
BOARD_W = COLS * CELL
BOARD_H = ROWS * CELL

_PLAYER_A_HUE_RANGE = (0.55, 0.72)
_PLAYER_B_HUE_RANGE = (0.95, 1.12)
_PLAYER_A_SAT = 0.70
_PLAYER_B_SAT = 0.70
_BRIGHTNESS = 0.78

OBJ_COLORS = {"A": "#2166ac", "B": "#d6604d", "": "#888888"}
OBJ_NAMES = ["Centre", "A-side", "B-side", "Home-A", "Home-B"]

_INFO_W = 260
_INFO_PAD = 8
_INFO_LINE_H = 14
_INFO_BG = "#1a1a2e"
_INFO_BORDER = "#5c5c8a"
_INFO_TEXT = "#e0e0e0"
_INFO_HEADER = "#ffffff"

# Highlight colors
_HIGHLIGHT_FRIENDLY = "#44ff44"
_HIGHLIGHT_ENEMY = "#ffaa00"
_HIGHLIGHT_CHARGE = "#ff4444"
_GHOST_ALPHA = 0.45  # conceptual; we'll mix with bg
_RANGE_COLORS = ["#88ccff", "#ffcc44", "#ff8866", "#aa88ff", "#66ffaa"]

# ── Phases ─────────────────────────────────────────────────────────
PHASE_SELECT_UNIT = "select_unit"
PHASE_CHOOSE_ACTION = "choose_action"
PHASE_CHARGE_TARGET = "charge_target"
PHASE_MOVE_TARGET = "move_target"
PHASE_SHOOT_TARGET = "shoot_target"
PHASE_CONFIRM = "confirm"
PHASE_RESOLUTION = "resolution"
PHASE_GAME_OVER = "game_over"
PHASE_BETWEEN_ROUNDS = "between_rounds"


# ── Helpers ────────────────────────────────────────────────────────

def _generate_palette(n: int, hue_range: tuple[float, float],
                      saturation: float, value: float) -> list[str]:
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
        colors.append(a_map[tid] if owner == "A" else b_map[tid])
    return colors


def _ghost_color(base_hex: str) -> str:
    """Mix a base color with the dark background for a ghost/preview look."""
    r = int(base_hex[1:3], 16)
    g = int(base_hex[3:5], 16)
    b = int(base_hex[5:7], 16)
    bg = 0x2d  # bg color component
    a = _GHOST_ALPHA
    r2 = int(r * a + bg * (1 - a))
    g2 = int(g * a + bg * (1 - a))
    b2 = int(b * a + bg * (1 - a))
    return f"#{r2:02x}{g2:02x}{b2:02x}"


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _unit_centroid(unit: UnitState) -> tuple[float, float]:
    positions = unit.alive_positions()
    if not positions:
        return (0.0, 0.0)
    cx = sum(c for c, r in positions) / len(positions)
    cy = sum(r for c, r in positions) / len(positions)
    return (cx, cy)


# ── Snapshot / restore helpers for preview ─────────────────────────

def _save_unit_positions(unit: UnitState) -> list[tuple[int, int]]:
    return list(unit.positions)


def _restore_unit_positions(unit: UnitState, saved: list[tuple[int, int]],
                            board: Board):
    """Remove current positions from board, restore saved ones."""
    for c, r in unit.alive_positions():
        board.remove(c, r)
    unit.positions = saved
    for c, r in unit.alive_positions():
        board.place(c, r)


# ===================================================================
# PRE-GAME DIALOG
# ===================================================================

class PreGameDialog:
    """Dialog to configure planning and army selection before starting the game."""

    ARMY_HOF_ML = "Random HoF_ml list"
    ARMY_RANDOM = "Random list (\u22651980 pts)"
    ARMY_IMPORTED = "Imported list"

    def __init__(self, parent: tk.Tk | None = None):
        self.result: dict | None = None
        self._imported_lists = self._find_imported_lists()

        self.win = tk.Toplevel(parent) if parent else tk.Tk()
        self.win.title("Game Setup")
        self.win.configure(bg="#1e1e1e")
        self.win.resizable(False, False)

        frame = ttk.Frame(self.win, padding=20)
        frame.pack()

        row = 0
        ttk.Label(frame, text="Play vs Tactical ML Model",
                  font=("Consolas", 14, "bold")).grid(row=row, column=0, columnspan=2, pady=(0, 15))

        # --- AI Planning ---
        row += 1
        self.planning_var = tk.BooleanVar(value=False)
        self.plan_check = ttk.Checkbutton(frame, text="Enable MC Planning for AI",
                                          variable=self.planning_var,
                                          command=self._toggle_params)
        self.plan_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)

        params = [("K (candidate units):", "4"),
                  ("C (action samples):", "4"),
                  ("M (rollouts):", "32"),
                  ("N (lookahead):", "6")]

        self.param_entries: list[ttk.Entry] = []
        for i, (label, default) in enumerate(params):
            row += 1
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=(20, 5), pady=2)
            e = ttk.Entry(frame, width=6)
            e.insert(0, default)
            e.configure(state=tk.DISABLED)
            e.grid(row=row, column=1, sticky=tk.W, pady=2)
            self.param_entries.append(e)

        # --- Army Selection ---
        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(12, 8))

        row += 1
        ttk.Label(frame, text="Your Army",
                  font=("Consolas", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

        army_choices = [self.ARMY_HOF_ML, self.ARMY_RANDOM]
        if self._imported_lists:
            army_choices.append(self.ARMY_IMPORTED)

        row += 1
        self.army_var = tk.StringVar(value=self.ARMY_HOF_ML)
        self.army_combo = ttk.Combobox(frame, textvariable=self.army_var,
                                       values=army_choices, state="readonly", width=28)
        self.army_combo.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        self.army_combo.bind("<<ComboboxSelected>>", self._on_army_choice)

        # Imported list file dropdown (hidden until "Imported list" is selected)
        row += 1
        self._imported_row = row
        self._imported_label = ttk.Label(frame, text="List file:")
        self._imported_label.grid(row=row, column=0, sticky=tk.W, padx=(20, 5), pady=2)
        self._imported_label.grid_remove()

        self.imported_file_var = tk.StringVar(
            value=self._imported_lists[0] if self._imported_lists else "")
        self.imported_file_combo = ttk.Combobox(
            frame, textvariable=self.imported_file_var,
            values=self._imported_lists, state="readonly", width=22)
        self.imported_file_combo.grid(row=row, column=1, sticky=tk.W, pady=2)
        self.imported_file_combo.grid_remove()

        # --- Start ---
        row += 1
        ttk.Button(frame, text="Start Game", command=self._start).grid(
            row=row, column=0, columnspan=2, pady=(15, 0))

        self.win.grab_set()
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

    @staticmethod
    def _find_imported_lists() -> list[str]:
        """Scan the Imported Lists folder for .txt files."""
        folder = Path(__file__).parent / "Imported Lists"
        if not folder.is_dir():
            return []
        return sorted(p.name for p in folder.glob("*.txt"))

    def _toggle_params(self):
        state = tk.NORMAL if self.planning_var.get() else tk.DISABLED
        for e in self.param_entries:
            e.configure(state=state)

    def _on_army_choice(self, _event=None):
        if self.army_var.get() == self.ARMY_IMPORTED:
            self._imported_label.grid()
            self.imported_file_combo.grid()
        else:
            self._imported_label.grid_remove()
            self.imported_file_combo.grid_remove()

    def _start(self):
        planning = self.planning_var.get()
        params = None
        if planning:
            try:
                params = {
                    'K_UNITS': int(self.param_entries[0].get()),
                    'C_SAMPLES_PER_UNIT': int(self.param_entries[1].get()),
                    'M_ROLLOUTS': int(self.param_entries[2].get()),
                    'N_LOOKAHEAD': int(self.param_entries[3].get()),
                }
            except ValueError:
                params = None

        army_choice = self.army_var.get()
        imported_file = None
        if army_choice == self.ARMY_IMPORTED:
            imported_file = self.imported_file_var.get()
            if not imported_file:
                return  # nothing selected yet

        self.result = {
            'planning': planning,
            'planning_params': params,
            'army_choice': army_choice,
            'imported_file': imported_file,
        }
        self.win.destroy()

    def _cancel(self):
        self.result = None
        self.win.destroy()

    def run(self) -> dict | None:
        self.win.wait_window()
        return self.result


# ===================================================================
# PLAY VIEWER
# ===================================================================

class PlayViewer:
    """Interactive game viewer where human plays Player A against ML Player B."""

    def __init__(self, units_a: list[UnitState], units_b: list[UnitState],
                 board: Board, labels: list[str], owners: list[str],
                 ml_model, mode: str = "objectives",
                 ml_planning: bool = False, planning_params: dict | None = None):
        self.units_a = units_a
        self.units_b = units_b
        self.board = board
        self.all_units = units_a + units_b
        self.labels = labels
        self.owners = owners
        self.ml_model = ml_model
        self.mode = mode
        self.ml_planning = ml_planning
        self.planning_params = planning_params
        self.n_units = len(labels)

        # Template IDs for color assignment
        template_ids = [u.unit.template_id for u in self.all_units]
        self.colors = _assign_colors_by_template(owners, template_ids)

        # Build unit index lookup
        self.unit_to_idx: dict[int, int] = {id(u): i for i, u in enumerate(self.all_units)}

        # ML precomputed data
        from ml_features import precompute_damage
        self._fr_a, self._fm_a = precompute_damage([u.unit for u in units_a],
                                                    [u.unit for u in units_b])
        self._fr_b, self._fm_b = precompute_damage([u.unit for u in units_b],
                                                    [u.unit for u in units_a])
        self._pts_a = sum(u.unit.points for u in units_a)
        self._pts_b = sum(u.unit.points for u in units_b)

        # Import tactical model functions
        from ml_integration_tactical import (
            apply_tactical_model, pick_target_from_ranking,
        )
        self._apply_tactical = apply_tactical_model
        self._pick_target_ranking = pick_target_from_ranking
        if ml_planning:
            from ml_planning import plan_activation as _plan
            self._plan_activation = _plan
        else:
            self._plan_activation = None

        # Game state
        self.round_num = 0  # 0-indexed internally, display as 1-indexed
        self.a_first = random.random() < 0.5
        self.a_finished_first = self.a_first
        self.current_is_a = self.a_first
        self.a_done = False
        self.b_done = False
        self.game_over = False

        # Frame history (for replay within resolution phase)
        self.frames: list[dict] = []
        self.game_snapshots: list = []  # GameSnapshot per frame (for AI suggestion in replay)
        self.frame_idx = 0

        # Phase state
        self.phase = PHASE_SELECT_UNIT
        self._selected_unit_idx: int | None = None  # index in all_units
        self._active_unit: UnitState | None = None
        self._chosen_action: str | None = None
        self._move_goal: tuple[int, int] | None = None
        self._charge_target: UnitState | None = None
        self._shoot_target: UnitState | None = None
        self._preview_positions: list[tuple[int, int]] | None = None
        self._preview_path: tuple[tuple[float, float], tuple[int, int]] | None = None
        self._info_click_unit: int | None = None  # for info box display on any click
        self._saved_positions_before_move: list[tuple[int, int]] | None = None
        self._pending_ai_frames: list[dict] = []
        self._ai_turn_pending: bool = False

        # Build UI
        self._build_ui()

        # Record deployment frame
        self._record_frame("Deployment complete", round_display=0)

        # Start first round
        self._start_round()

    # ───────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("OPR Tactical — Interactive Play")
        self.root.configure(bg="#1e1e1e")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ── Left side: board + controls ──
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        self.canvas = tk.Canvas(left_frame, width=BOARD_W + 1, height=BOARD_H + 1,
                                bg="#2d2d2d", highlightthickness=0)
        self.canvas.pack(padx=5, pady=5)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Status bar
        self.status_var = tk.StringVar(value="Select a unit to activate")
        status_label = ttk.Label(left_frame, textvariable=self.status_var,
                                 font=("Consolas", 11), wraplength=BOARD_W)
        status_label.pack(anchor=tk.W, padx=5, pady=(5, 2))

        # Button bar
        self.btn_frame = ttk.Frame(left_frame)
        self.btn_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Activation log
        log_label = ttk.Label(left_frame, text="Activation Log:",
                              font=("Consolas", 10, "bold"))
        log_label.pack(anchor=tk.W, padx=5, pady=(5, 0))

        log_frame = ttk.Frame(left_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.log_text = tk.Text(log_frame, width=72, height=10,
                                bg="#1e1e1e", fg="white",
                                font=("Consolas", 9), wrap=tk.WORD,
                                state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Right side: info panels ──
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Round & turn info
        self.round_var = tk.StringVar(value="")
        ttk.Label(right_frame, textvariable=self.round_var,
                  font=("Consolas", 11, "bold")).pack(anchor=tk.W, pady=(5, 5))

        # Objective status
        self.obj_var = tk.StringVar(value="")
        ttk.Label(right_frame, textvariable=self.obj_var,
                  font=("Consolas", 10), justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 10))

        # Combat stats
        stats_frame = ttk.LabelFrame(right_frame, text="Combat Stats")
        stats_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
        self.stats_var = tk.StringVar(value="(no combat yet)")
        ttk.Label(stats_frame, textvariable=self.stats_var,
                  font=("Consolas", 9), justify=tk.LEFT, wraplength=350).pack(anchor=tk.W, padx=5, pady=3)

        # AI assessment panel
        ai_frame = ttk.LabelFrame(right_frame, text="AI Assessment (Player B)")
        ai_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
        ai_inner = tk.Frame(ai_frame)
        ai_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)
        self.ai_text = tk.Text(ai_inner, font=("Consolas", 9), wrap=tk.NONE,
                               height=8, state=tk.DISABLED, relief=tk.FLAT,
                               bg=ttk.Style().lookup("TLabelframe", "background") or "#f0f0f0")
        ai_scroll = ttk.Scrollbar(ai_inner, orient=tk.VERTICAL,
                                  command=self.ai_text.yview)
        self.ai_text.configure(yscrollcommand=ai_scroll.set)
        self.ai_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ai_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Unit key with status
        key_frame = ttk.LabelFrame(right_frame, text="Units")
        key_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 5))

        key_height = max(18 * self.n_units + 10, 100)
        self.key_canvas = tk.Canvas(key_frame, width=360, height=key_height,
                                    bg="#2d2d2d", highlightthickness=0)
        self.key_canvas.pack(anchor=tk.W, padx=3, pady=3)

        self.root.bind("<Left>", lambda e: self._on_key_prev())
        self.root.bind("<Right>", lambda e: self._on_key_next())
        self.root.bind("r", lambda e: self._on_key_replay())
        self.root.bind("R", lambda e: self._on_key_replay())

    # ───────────────────────────────────────────────────────────────
    # RENDERING
    # ───────────────────────────────────────────────────────────────

    def _render(self):
        """Full redraw of board, panels, and buttons."""
        self.canvas.delete("all")

        # Grid lines
        for c in range(0, COLS + 1, 6):
            x = c * CELL
            self.canvas.create_line(x, 0, x, BOARD_H, fill="#3a3a3a")
        for r in range(0, ROWS + 1, 6):
            y = (ROWS - r) * CELL
            self.canvas.create_line(0, y, BOARD_W, y, fill="#3a3a3a")

        # Deployment zone boundaries
        for row_line in [12, 36]:
            y = (ROWS - row_line) * CELL
            self.canvas.create_line(0, y, BOARD_W, y, fill="#555555", dash=(4, 4))

        # Objectives
        if self.mode != "kill_points":
            for oi, (oc, orow) in enumerate(OBJECTIVES):
                x = oc * CELL + CELL // 2
                y = (ROWS - orow) * CELL - CELL // 2
                r = CELL * 2
                color = OBJ_COLORS.get(self.board.objective_control[oi], "#888888")
                self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                        outline=color, width=2, dash=(3, 3))
                obj_label = OBJ_NAMES[oi][0] if oi < 3 else OBJ_NAMES[oi][-1]
                self.canvas.create_text(x, y, text=obj_label,
                                        fill=color, font=("Consolas", 8, "bold"))

        # Draw models
        for ui in range(self.n_units):
            unit = self.all_units[ui]
            positions = unit.alive_positions()
            color = self.colors[ui]
            is_selected = (ui == self._selected_unit_idx)
            is_info = (ui == self._info_click_unit)

            # Determine if unit should be highlighted
            highlight = self._get_unit_highlight(ui)

            for col, row in positions:
                x = col * CELL
                y = (ROWS - 1 - row) * CELL
                self.canvas.create_rectangle(x + 1, y + 1, x + CELL - 1, y + CELL - 1,
                                             fill=color, outline="")
                if is_selected or is_info:
                    self.canvas.create_rectangle(x, y, x + CELL, y + CELL,
                                                 outline="#ffffff", width=2)
                elif highlight:
                    self.canvas.create_rectangle(x, y, x + CELL, y + CELL,
                                                 outline=highlight, width=2, dash=(3, 2))

        # Draw ghost preview positions
        if self._preview_positions and self._active_unit is not None:
            ghost_c = _ghost_color(self.colors[self.unit_to_idx[id(self._active_unit)]])
            for col, row in self._preview_positions:
                x = col * CELL
                y = (ROWS - 1 - row) * CELL
                self.canvas.create_rectangle(x + 1, y + 1, x + CELL - 1, y + CELL - 1,
                                             fill=ghost_c, outline="#aaaaaa", dash=(2, 2))

        # Draw preview path line
        if self._preview_path:
            start, end = self._preview_path
            sx = start[0] * CELL + CELL // 2
            sy = (ROWS - 1 - start[1]) * CELL + CELL // 2
            ex = end[0] * CELL + CELL // 2
            ey = (ROWS - 1 - end[1]) * CELL + CELL // 2
            self.canvas.create_line(sx, sy, ex, ey, fill="#aaaaaa", dash=(4, 3), width=1)

        # Draw range circles (shooting phase)
        if self.phase == PHASE_SHOOT_TARGET and self._active_unit is not None:
            self._draw_range_circles()

        # Draw movement budget circle (move target phase)
        if self.phase == PHASE_MOVE_TARGET and self._active_unit is not None:
            self._draw_move_budget_circle()

        # Draw info box for clicked unit
        if self._info_click_unit is not None:
            self._draw_info_box(self._info_click_unit)

        # Update panels
        self._update_round_info()
        self._update_objectives()
        self._update_unit_key()
        self._update_log()
        self._update_buttons()

    def _get_unit_highlight(self, ui: int) -> str | None:
        """Return highlight color for a unit given current phase, or None."""
        unit = self.all_units[ui]
        owner = self.owners[ui]

        if self.phase == PHASE_SELECT_UNIT:
            # Highlight non-activated friendly units
            if owner == "A" and unit.models_alive > 0 and not unit.activated:
                return _HIGHLIGHT_FRIENDLY

        elif self.phase == PHASE_CHARGE_TARGET:
            # Highlight chargeable enemies
            if owner == "B" and unit.models_alive > 0 and self._active_unit is not None:
                if _can_charge(self._active_unit, unit):
                    return _HIGHLIGHT_CHARGE

        elif self.phase == PHASE_SHOOT_TARGET:
            # Highlight shootable enemies
            if owner == "B" and unit.models_alive > 0 and self._active_unit is not None:
                if can_shoot_any(self._active_unit, unit):
                    return _HIGHLIGHT_ENEMY

        return None

    def _draw_range_circles(self):
        """Draw range circles for the active unit's weapons."""
        if self._active_unit is None:
            return
        # If we have a preview position (advance), use that as the centre
        if self._preview_positions:
            cx = sum(c for c, r in self._preview_positions) / len(self._preview_positions)
            cy = sum(r for c, r in self._preview_positions) / len(self._preview_positions)
        else:
            cx, cy = _unit_centroid(self._active_unit)
        # Collect unique ranges from alive models' ranged weapons
        ranges: set[int] = set()
        for mi in range(self._active_unit.models_alive):
            for w in self._active_unit.weapons_per_model[mi]:
                if not w.melee and w.range_inches > 0:
                    ranges.add(w.range_inches)
        for i, rng in enumerate(sorted(ranges)):
            color = _RANGE_COLORS[i % len(_RANGE_COLORS)]
            px = cx * CELL + CELL // 2
            py = (ROWS - 1 - cy) * CELL + CELL // 2
            r_px = rng * CELL
            self.canvas.create_oval(px - r_px, py - r_px, px + r_px, py + r_px,
                                    outline=color, width=1, dash=(4, 4))
            # Label
            self.canvas.create_text(px + r_px + 3, py, text=f'{rng}"',
                                    fill=color, anchor=tk.W, font=("Consolas", 7))

    def _draw_move_budget_circle(self):
        """Draw the movement budget radius around active unit."""
        if self._active_unit is None or self._chosen_action is None:
            return
        cx, cy = _unit_centroid(self._active_unit)
        if self._chosen_action == "advance":
            budget = self._active_unit.unit.advance_distance
        else:
            budget = self._active_unit.unit.rush_distance
        px = cx * CELL + CELL // 2
        py = (ROWS - 1 - cy) * CELL + CELL // 2
        r_px = budget * CELL
        self.canvas.create_oval(px - r_px, py - r_px, px + r_px, py + r_px,
                                outline="#66ff66", width=1, dash=(6, 3))
        self.canvas.create_text(px + r_px + 3, py, text=f'{budget}"',
                                fill="#66ff66", anchor=tk.W, font=("Consolas", 8))

    def _draw_info_box(self, ui: int):
        """Draw a floating info box on the canvas for unit ui."""
        unit = self.all_units[ui]
        ru = unit.unit

        lines: list[tuple[str, str]] = []
        lines.append((self.labels[ui], _INFO_HEADER))
        lines.append((f"Owner: Player {self.owners[ui]}", _INFO_TEXT))
        lines.append((f"Models alive: {unit.models_alive} / {ru.models}", _INFO_TEXT))
        lines.append((f"Points: {ru.points}  |  Q{ru.quality}+  D{ru.defense}+", _INFO_TEXT))

        # Status flags
        status_parts = []
        if unit.activated:
            status_parts.append("Activated")
        if unit.shaken:
            status_parts.append("Shaken")
        if unit.fatigued:
            status_parts.append("Fatigued")
        if ru.tough:
            # Show wounds on alive models
            wound_strs = []
            for mi in range(unit.models_alive):
                w = unit.wounds_per_model[mi]
                if w > 0:
                    wound_strs.append(f"{w}/{ru.tough}")
            if wound_strs:
                status_parts.append(f"Wounds: {', '.join(wound_strs)}")
            else:
                status_parts.append(f"Tough({ru.tough})")
        if status_parts:
            lines.append((", ".join(status_parts), "#ffcc44"))
        else:
            lines.append(("Ready", "#66ff66"))

        # Special rules
        special = []
        for attr, label in [('scout', 'Scout'), ('stealth', 'Stealth'),
                            ('fast', 'Fast'), ('flying', 'Flying'),
                            ('relentless', 'Relentless'), ('artillery', 'Artillery'),
                            ('fearless', 'Fearless'), ('regeneration', 'Regeneration'),
                            ('furious', 'Furious'), ('impact', None)]:
            val = getattr(ru, attr, None)
            if val:
                if attr == 'impact':
                    special.append(f"Impact({val})")
                else:
                    special.append(label)
        if special:
            lines.append(("Rules: " + ", ".join(special), "#aaccff"))

        lines.append(("", _INFO_TEXT))
        lines.append(("Weapons:", _INFO_HEADER))
        seen: dict[str, int] = {}
        for w in ru.weapons:
            seen[w.name] = seen.get(w.name, 0) + 1
        for wname, count in seen.items():
            w = next(w for w in ru.weapons if w.name == wname)
            prefix = f"{count}x " if count > 1 else ""
            range_str = f'{w.range_inches}"' if w.range_inches > 0 else "melee"
            abilities = []
            if w.ap: abilities.append(f"AP({w.ap})")
            if w.blast: abilities.append(f"Blast({w.blast})")
            if w.deadly: abilities.append(f"Deadly({w.deadly})")
            if w.rending: abilities.append("Rending")
            if w.reliable: abilities.append("Reliable")
            ab = f" [{', '.join(abilities)}]" if abilities else ""
            lines.append((f"  {prefix}{wname} ({range_str}, A{w.attacks}{ab})", _INFO_TEXT))

        # Calculate box dimensions
        n_lines = len(lines)
        box_h = _INFO_PAD * 2 + n_lines * _INFO_LINE_H + 4

        positions = unit.alive_positions()
        if positions:
            avg_col = sum(c for c, r in positions) / len(positions)
            avg_row = sum(r for c, r in positions) / len(positions)
            anchor_x = int(avg_col * CELL) + CELL
            anchor_y = int((ROWS - 1 - avg_row) * CELL) - box_h - CELL
        else:
            anchor_x, anchor_y = 20, 20

        # Clamp
        if anchor_x + _INFO_W > BOARD_W:
            anchor_x = BOARD_W - _INFO_W - 4
        if anchor_x < 4:
            anchor_x = 4
        if anchor_y < 4:
            anchor_y = 4
        if anchor_y + box_h > BOARD_H:
            anchor_y = BOARD_H - box_h - 4

        self.canvas.create_rectangle(anchor_x, anchor_y,
                                     anchor_x + _INFO_W, anchor_y + box_h,
                                     fill=_INFO_BG, outline=_INFO_BORDER, width=2)
        ty = anchor_y + _INFO_PAD + 2
        for text, color in lines:
            if text:
                font = ("Consolas", 8, "bold") if color == _INFO_HEADER else ("Consolas", 8)
                self.canvas.create_text(anchor_x + _INFO_PAD, ty,
                                        text=text, fill=color, anchor=tk.NW, font=font)
            ty += _INFO_LINE_H

    def _update_round_info(self):
        a_activated = sum(1 for u in self.units_a if u.activated)
        b_activated = sum(1 for u in self.units_b if u.activated)
        a_total = sum(1 for u in self.units_a if u.models_alive > 0)
        b_total = sum(1 for u in self.units_b if u.models_alive > 0)
        if self.game_over:
            self.round_var.set("Game Over")
        elif self.phase == PHASE_BETWEEN_ROUNDS:
            self.round_var.set(f"Round {self.round_num + 1} complete")
        elif self.current_is_a:
            self.round_var.set(f"Round {self.round_num + 1}  |  Your turn ({a_activated}/{a_total} activated)")
        else:
            self.round_var.set(f"Round {self.round_num + 1}  |  AI turn ({b_activated}/{b_total} activated)")

    def _update_objectives(self):
        if self.mode == "kill_points":
            a_kp = sum(u.unit.points for u in self.units_b if u.models_alive <= 0)
            b_kp = sum(u.unit.points for u in self.units_a if u.models_alive <= 0)
            self.obj_var.set(f"Kill Points:\n  You: {a_kp}pts\n  AI: {b_kp}pts")
        else:
            parts = []
            for oi, ctrl in enumerate(self.board.objective_control):
                status = f"Player {ctrl}" if ctrl else "Neutral"
                parts.append(f"  {OBJ_NAMES[oi]}: {status}")
            self.obj_var.set("Objectives:\n" + "\n".join(parts))

    def _update_unit_key(self):
        self.key_canvas.delete("all")
        for i in range(self.n_units):
            unit = self.all_units[i]
            y = 3 + i * 18
            color = self.colors[i]
            alive = unit.models_alive
            total = unit.unit.models

            if alive <= 0:
                color = "#555555"

            self.key_canvas.create_rectangle(5, y, 15, y + 12,
                                             fill=color, outline="")

            status_parts = []
            if alive <= 0:
                status_parts.append("DEAD")
            else:
                status_parts.append(f"{alive}/{total}")
                if unit.activated:
                    status_parts.append("act")
                if unit.shaken:
                    status_parts.append("shk")
                if unit.fatigued:
                    status_parts.append("ftg")
                if unit.unit.tough:
                    total_wounds = sum(unit.wounds_per_model[mi] for mi in range(alive))
                    if total_wounds > 0:
                        status_parts.append(f"w:{total_wounds}")

            text = f"{self.labels[i]}  [{', '.join(status_parts)}]"
            text_color = "#777777" if alive <= 0 else "white"
            self.key_canvas.create_text(20, y + 6, text=text,
                                        fill=text_color, anchor=tk.W,
                                        font=("Consolas", 8))

    def _update_log(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        for f in self.frames:
            desc = f.get('description', '')
            if desc:
                rnd = f.get('round', 0)
                prefix = f"[R{rnd}] " if rnd > 0 else "[--] "
                self.log_text.insert(tk.END, prefix + desc + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _update_buttons(self):
        """Rebuild the button bar for the current phase."""
        for w in self.btn_frame.winfo_children():
            w.destroy()

        if self.phase == PHASE_SELECT_UNIT:
            if len(self.frames) > 1:
                ttk.Button(self.btn_frame, text="Replay", command=self._open_replay).pack(side=tk.LEFT, padx=3)

        elif self.phase == PHASE_CHOOSE_ACTION:
            can_charge = False
            if self._active_unit is not None:
                for enemy in self.units_b:
                    if enemy.models_alive > 0 and _can_charge(self._active_unit, enemy):
                        can_charge = True
                        break
            ttk.Button(self.btn_frame, text="Hold", command=lambda: self._choose_action("hold")).pack(side=tk.LEFT, padx=3)
            ttk.Button(self.btn_frame, text="Advance", command=lambda: self._choose_action("advance")).pack(side=tk.LEFT, padx=3)
            ttk.Button(self.btn_frame, text="Rush", command=lambda: self._choose_action("rush")).pack(side=tk.LEFT, padx=3)
            charge_btn = ttk.Button(self.btn_frame, text="Charge", command=lambda: self._choose_action("charge"))
            charge_btn.pack(side=tk.LEFT, padx=3)
            if not can_charge:
                charge_btn.configure(state=tk.DISABLED)
            ttk.Button(self.btn_frame, text="Cancel", command=self._cancel_to_select).pack(side=tk.LEFT, padx=10)

        elif self.phase == PHASE_CHARGE_TARGET:
            ttk.Button(self.btn_frame, text="Back", command=self._back_to_action).pack(side=tk.LEFT, padx=3)

        elif self.phase == PHASE_MOVE_TARGET:
            if self._preview_positions is not None:
                ttk.Button(self.btn_frame, text="Back", command=self._back_to_action).pack(side=tk.LEFT, padx=3)
                if self._chosen_action == "rush":
                    ttk.Button(self.btn_frame, text="Confirm Rush", command=self._confirm_action).pack(side=tk.LEFT, padx=3)
                else:
                    ttk.Button(self.btn_frame, text="Confirm Move", command=self._move_to_shooting).pack(side=tk.LEFT, padx=3)
            else:
                ttk.Button(self.btn_frame, text="Back", command=self._back_to_action).pack(side=tk.LEFT, padx=3)

        elif self.phase == PHASE_SHOOT_TARGET:
            ttk.Button(self.btn_frame, text="Skip (no shot)", command=self._skip_shooting).pack(side=tk.LEFT, padx=3)
            back_target = PHASE_MOVE_TARGET if self._chosen_action in ("advance",) else PHASE_CHOOSE_ACTION
            ttk.Button(self.btn_frame, text="Back", command=lambda: self._go_back_from_shooting()).pack(side=tk.LEFT, padx=3)

        elif self.phase == PHASE_CONFIRM:
            ttk.Button(self.btn_frame, text="Back", command=self._back_from_confirm).pack(side=tk.LEFT, padx=3)
            ttk.Button(self.btn_frame, text="Confirm", command=self._confirm_action).pack(side=tk.LEFT, padx=3)

        elif self.phase == PHASE_RESOLUTION:
            if self._pending_ai_frames or self._ai_turn_pending:
                ttk.Button(self.btn_frame, text="Next (AI turn)", command=self._show_next_ai_frame).pack(side=tk.LEFT, padx=3)
            else:
                ttk.Button(self.btn_frame, text="Continue", command=self._after_resolution).pack(side=tk.LEFT, padx=3)

        elif self.phase == PHASE_BETWEEN_ROUNDS:
            ttk.Button(self.btn_frame, text="Start Next Round", command=self._advance_round).pack(side=tk.LEFT, padx=3)
            ttk.Button(self.btn_frame, text="Replay", command=self._open_replay).pack(side=tk.LEFT, padx=3)

        elif self.phase == PHASE_GAME_OVER:
            ttk.Button(self.btn_frame, text="Replay", command=self._open_replay).pack(side=tk.LEFT, padx=3)
            ttk.Button(self.btn_frame, text="Close", command=self.root.destroy).pack(side=tk.LEFT, padx=3)

    # ───────────────────────────────────────────────────────────────
    # GAME FLOW
    # ───────────────────────────────────────────────────────────────

    def _start_round(self):
        """Begin a new round: reset activations, set turn order."""
        for u in self.units_a:
            u.activated = False
            u.fatigued = False
        for u in self.units_b:
            u.activated = False
            u.fatigued = False

        if self.round_num == 0:
            self.current_is_a = self.a_first
        else:
            self.current_is_a = self.a_finished_first

        self.a_done = False
        self.b_done = False

        # Reassign roles for heuristic side (human doesn't need this but
        # the AI side uses heuristic fallbacks for role assignment)
        if self.mode != "kill_points":
            assign_objectives(self.units_a)
            reassign_roles(self.units_b)

        # If AI goes first, defer AI turn until user clicks "Next (AI turn)"
        if not self.current_is_a:
            self._pending_ai_frames = []
            self._ai_turn_pending = True
            self.phase = PHASE_RESOLUTION
            self.status_var.set("AI goes first this round. Click Next to see the AI's action.")
            self._render()
        else:
            self.phase = PHASE_SELECT_UNIT
            self.status_var.set("Select a unit to activate")
            self._clear_selection()
            self._render()

    def _record_frame(self, description: str, round_display: int | None = None,
                      combat_stats: dict | None = None,
                      ml_assessment: dict | None = None):
        """Record a frame (snapshot) for the log."""
        snap = _snapshot(self.all_units, self.board.objective_control)
        snap['description'] = description
        snap['round'] = round_display if round_display is not None else self.round_num + 1
        if combat_stats:
            snap['combat_stats'] = combat_stats
        if ml_assessment:
            snap['ml_assessment'] = ml_assessment
        self.frames.append(snap)
        self.game_snapshots.append(
            snapshot_game_state(self.units_a, self.units_b, self.board))

    def _check_game_over(self) -> bool:
        """Check if the game should end."""
        a_alive = any(u.models_alive > 0 for u in self.units_a)
        b_alive = any(u.models_alive > 0 for u in self.units_b)
        if not a_alive or not b_alive:
            return True
        return False

    def _end_game(self):
        """Handle end of game."""
        self.game_over = True
        if self.mode == "kill_points":
            a_kp = sum(u.unit.points for u in self.units_b if u.models_alive <= 0)
            b_kp = sum(u.unit.points for u in self.units_a if u.models_alive <= 0)
            if a_kp > b_kp:
                result_text = f"You win! ({a_kp} vs {b_kp} kill points)"
            elif b_kp > a_kp:
                result_text = f"You lose. ({a_kp} vs {b_kp} kill points)"
            else:
                result_text = f"Draw! ({a_kp} vs {b_kp} kill points)"
        else:
            self.board.update_objectives(self.units_a, self.units_b)
            a_objs = self.board.count_objectives("A")
            b_objs = self.board.count_objectives("B")
            if a_objs > b_objs:
                result_text = f"You win! ({a_objs} vs {b_objs} objectives)"
            elif b_objs > a_objs:
                result_text = f"You lose. ({a_objs} vs {b_objs} objectives)"
            else:
                result_text = f"Draw! ({a_objs} vs {b_objs} objectives)"

        self._record_frame(f"Game Over -- {result_text}")
        self.phase = PHASE_GAME_OVER
        self.status_var.set(f"Game Over -- {result_text}")
        self._render()

    def _check_round_over(self) -> bool:
        """Check if the current round is over (both sides done)."""
        a_can = any(u.models_alive > 0 and not u.activated for u in self.units_a)
        b_can = any(u.models_alive > 0 and not u.activated for u in self.units_b)
        if not a_can:
            self.a_done = True
            if not self.b_done:
                self.a_finished_first = True
        if not b_can:
            self.b_done = True
            if not self.a_done:
                self.a_finished_first = False
        return self.a_done and self.b_done

    def _end_round(self):
        """Handle end-of-round scoring and transition."""
        if self.mode != "kill_points":
            self.board.update_objectives(self.units_a, self.units_b)

        if self.mode == "kill_points":
            a_kp = sum(u.unit.points for u in self.units_b if u.models_alive <= 0)
            b_kp = sum(u.unit.points for u in self.units_a if u.models_alive <= 0)
            desc = f"End of Round {self.round_num + 1} -- Kill Points: You: {a_kp}pts, AI: {b_kp}pts"
        else:
            obj_parts = []
            for oi, ctrl in enumerate(self.board.objective_control):
                if ctrl:
                    obj_parts.append(f"{OBJ_NAMES[oi]}: Player {ctrl}")
                else:
                    obj_parts.append(f"{OBJ_NAMES[oi]}: Neutral")
            desc = f"End of Round {self.round_num + 1} -- {', '.join(obj_parts)}"

        self._record_frame(desc)

        if self.round_num >= 3:
            # Game over after round 4
            self._end_game()
        else:
            self.phase = PHASE_BETWEEN_ROUNDS
            self.status_var.set(desc)
            self._render()

    def _advance_round(self):
        """Move to the next round."""
        self.round_num += 1
        self._start_round()

    # ───────────────────────────────────────────────────────────────
    # HUMAN ACTIVATION PHASES
    # ───────────────────────────────────────────────────────────────

    def _clear_selection(self):
        self._selected_unit_idx = None
        self._active_unit = None
        self._chosen_action = None
        self._move_goal = None
        self._charge_target = None
        self._shoot_target = None
        self._preview_positions = None
        self._preview_path = None
        self._info_click_unit = None
        self._saved_positions_before_move: list[tuple[int, int]] | None = None

    def _on_canvas_click(self, event):
        """Handle click on the board canvas."""
        col = event.x // CELL
        row = ROWS - 1 - (event.y // CELL)

        # Find which unit was clicked
        clicked_unit = None
        for ui in range(self.n_units):
            unit = self.all_units[ui]
            for pos_col, pos_row in unit.alive_positions():
                if pos_col == col and pos_row == row:
                    clicked_unit = ui
                    break
            if clicked_unit is not None:
                break

        if self.phase == PHASE_SELECT_UNIT:
            if clicked_unit is not None and self.owners[clicked_unit] == "A":
                unit = self.all_units[clicked_unit]
                if unit.models_alive > 0 and not unit.activated:
                    self._selected_unit_idx = clicked_unit
                    self._active_unit = unit
                    self._info_click_unit = clicked_unit
                    self.phase = PHASE_CHOOSE_ACTION
                    self.status_var.set(f"{self.labels[clicked_unit]} selected — choose an action")
                    self._render()
                    return
            # Show info on any clicked unit
            self._info_click_unit = clicked_unit
            self._render()

        elif self.phase == PHASE_CHOOSE_ACTION:
            # Allow viewing unit info while choosing action
            self._info_click_unit = clicked_unit
            self._render()

        elif self.phase == PHASE_CHARGE_TARGET:
            if clicked_unit is not None and self.owners[clicked_unit] == "B":
                target = self.all_units[clicked_unit]
                if target.models_alive > 0 and self._active_unit is not None:
                    if _can_charge(self._active_unit, target):
                        self._charge_target = target
                        self._info_click_unit = clicked_unit
                        # Preview charge position
                        self._preview_charge(target)
                        self.phase = PHASE_CONFIRM
                        target_label = self.labels[clicked_unit]
                        self.status_var.set(f"Charging {target_label} — confirm or go back")
                        self._render()
                        return
            self._info_click_unit = clicked_unit
            self._render()

        elif self.phase == PHASE_MOVE_TARGET:
            if 0 <= col < COLS and 0 <= row < ROWS:
                goal = (col, row)
                self._move_goal = goal
                self._preview_movement(goal)
                note = ""
                if self._chosen_action == "rush":
                    note = " (unit cannot attack due to rushing)"
                self.status_var.set(f"Move to ({col}, {row}){note} — confirm or click another position")
                self._render()
                return

        elif self.phase == PHASE_SHOOT_TARGET:
            if clicked_unit is not None and self.owners[clicked_unit] == "B":
                target = self.all_units[clicked_unit]
                if target.models_alive > 0 and self._active_unit is not None:
                    if can_shoot_any(self._active_unit, target):
                        self._shoot_target = target
                        self._info_click_unit = clicked_unit
                        self.phase = PHASE_CONFIRM
                        self._build_confirm_status()
                        self._render()
                        return
            self._info_click_unit = clicked_unit
            self._render()

        elif self.phase == PHASE_RESOLUTION:
            self._info_click_unit = clicked_unit
            self._render()

        elif self.phase == PHASE_GAME_OVER:
            self._info_click_unit = clicked_unit
            self._render()

    def _choose_action(self, action: str):
        """Handle action button click."""
        self._chosen_action = action
        if action == "hold":
            if self._active_unit is not None and self._active_unit.shaken:
                # Shaken: auto-recover, no shooting
                self.phase = PHASE_CONFIRM
                self.status_var.set(f"{self.labels[self._selected_unit_idx]} will hold (recovering from Shaken)")
                self._shoot_target = None
            else:
                self.phase = PHASE_SHOOT_TARGET
                self.status_var.set("Select a shooting target (or skip)")
        elif action == "charge":
            self.phase = PHASE_CHARGE_TARGET
            self.status_var.set("Select an enemy to charge")
        else:
            # advance or rush
            self.phase = PHASE_MOVE_TARGET
            if action == "advance":
                budget = self._active_unit.unit.advance_distance
            else:
                budget = self._active_unit.unit.rush_distance
            self.status_var.set(f"Click the board to set movement target ({action}: {budget}\" range)")
        self._preview_positions = None
        self._preview_path = None
        self._render()

    def _cancel_to_select(self):
        self._clear_selection()
        self.phase = PHASE_SELECT_UNIT
        self.status_var.set("Select a unit to activate")
        self._render()

    def _back_to_action(self):
        self._preview_positions = None
        self._preview_path = None
        self._move_goal = None
        self._charge_target = None
        self.phase = PHASE_CHOOSE_ACTION
        self.status_var.set(f"{self.labels[self._selected_unit_idx]} selected — choose an action")
        self._render()

    def _move_to_shooting(self):
        """After confirming movement (advance), execute the move on the board
        so that shooting range checks use the new position, then proceed to
        shooting phase.  The old position is saved for undo on Back."""
        unit = self._active_unit
        if unit is None:
            return
        # Actually execute the advance on the board
        self._saved_positions_before_move = _save_unit_positions(unit)
        budget = unit.unit.advance_distance
        enemy_positions = _collect_enemy_positions(self.units_b)
        execute_movement(unit, self._move_goal, budget, self.board, enemy_positions,
                         flying=unit.unit.flying)
        # Clear preview (unit is now really there)
        self._preview_positions = None
        self._preview_path = None

        if unit.shaken:
            # Shaken: auto-recover, no shooting
            self.phase = PHASE_CONFIRM
            self._shoot_target = None
            self.status_var.set(f"{self.labels[self._selected_unit_idx]} will advance (recovering from Shaken, no shooting)")
        else:
            self.phase = PHASE_SHOOT_TARGET
            self.status_var.set("Select a shooting target (or skip)")
        self._render()

    def _skip_shooting(self):
        self._shoot_target = None
        self.phase = PHASE_CONFIRM
        self._build_confirm_status()
        self._render()

    def _go_back_from_shooting(self):
        if self._chosen_action == "advance":
            # Undo the advance that was applied in _move_to_shooting
            if self._saved_positions_before_move is not None:
                _restore_unit_positions(self._active_unit, self._saved_positions_before_move, self.board)
                self._saved_positions_before_move = None
            self.phase = PHASE_MOVE_TARGET
            budget = self._active_unit.unit.advance_distance
            self.status_var.set(f"Click the board to set movement target (advance: {budget}\" range)")
            self._preview_positions = None
            self._preview_path = None
        else:
            # hold
            self.phase = PHASE_CHOOSE_ACTION
            self.status_var.set(f"{self.labels[self._selected_unit_idx]} selected — choose an action")
        self._render()

    def _back_from_confirm(self):
        if self._chosen_action == "charge":
            self._charge_target = None
            self._preview_positions = None
            self._preview_path = None
            self.phase = PHASE_CHARGE_TARGET
            self.status_var.set("Select an enemy to charge")
        elif self._chosen_action == "rush":
            self.phase = PHASE_MOVE_TARGET
            budget = self._active_unit.unit.rush_distance
            self.status_var.set(f"Click the board to set movement target (rush: {budget}\" range)")
            self._preview_positions = None
            self._preview_path = None
        elif self._shoot_target is not None:
            # Back from confirm to shooting — unit is still at advanced position, keep it
            self._shoot_target = None
            self.phase = PHASE_SHOOT_TARGET
            self.status_var.set("Select a shooting target (or skip)")
        elif self._chosen_action == "advance":
            # Back from confirm (advance, no shot) to move target — undo the advance
            if self._saved_positions_before_move is not None:
                _restore_unit_positions(self._active_unit, self._saved_positions_before_move, self.board)
                self._saved_positions_before_move = None
            self.phase = PHASE_MOVE_TARGET
            budget = self._active_unit.unit.advance_distance
            self.status_var.set(f"Click the board to set movement target (advance: {budget}\" range)")
            self._preview_positions = None
            self._preview_path = None
        else:
            # hold with no shot
            self.phase = PHASE_CHOOSE_ACTION
            self.status_var.set(f"{self.labels[self._selected_unit_idx]} selected — choose an action")
        self._render()

    def _build_confirm_status(self):
        """Build the confirmation summary text."""
        label = self.labels[self._selected_unit_idx]
        action = self._chosen_action

        if action == "charge" and self._charge_target:
            target_idx = self.unit_to_idx[id(self._charge_target)]
            target_label = self.labels[target_idx]
            self.status_var.set(f"{label} will charge {target_label}")
        elif action == "rush":
            self.status_var.set(f"{label} will rush to ({self._move_goal[0]}, {self._move_goal[1]}) (no attacks)")
        elif action == "advance":
            if self._shoot_target:
                shoot_idx = self.unit_to_idx[id(self._shoot_target)]
                shoot_label = self.labels[shoot_idx]
                self.status_var.set(f"{label} will advance and shoot {shoot_label}")
            else:
                self.status_var.set(f"{label} will advance (no shooting)")
        elif action == "hold":
            if self._shoot_target:
                shoot_idx = self.unit_to_idx[id(self._shoot_target)]
                shoot_label = self.labels[shoot_idx]
                self.status_var.set(f"{label} will hold and shoot {shoot_label}")
            else:
                self.status_var.set(f"{label} will hold (no shooting)")

    # ───────────────────────────────────────────────────────────────
    # PREVIEW HELPERS
    # ───────────────────────────────────────────────────────────────

    def _preview_movement(self, goal: tuple[int, int]):
        """Preview movement to a goal position (non-destructive)."""
        unit = self._active_unit
        if unit is None:
            return
        # Save state
        saved_pos = _save_unit_positions(unit)
        budget = unit.unit.advance_distance if self._chosen_action == "advance" else unit.unit.rush_distance
        enemy_positions = _collect_enemy_positions(self.units_b)

        # Execute movement on real board (we'll restore after)
        execute_movement(unit, goal, budget, self.board, enemy_positions,
                         flying=unit.unit.flying)

        # Capture preview
        self._preview_positions = list(unit.alive_positions())
        end = _unit_centroid(unit)
        # Compute start centroid from saved positions
        if saved_pos:
            sx = sum(c for c, r in saved_pos[:unit.models_alive]) / max(unit.models_alive, 1)
            sy = sum(r for c, r in saved_pos[:unit.models_alive]) / max(unit.models_alive, 1)
        else:
            sx, sy = 0, 0
        self._preview_path = ((sx, sy), (end[0], end[1]))

        # Restore
        _restore_unit_positions(unit, saved_pos, self.board)

    def _preview_charge(self, target: UnitState):
        """Preview charge movement toward a target."""
        unit = self._active_unit
        if unit is None:
            return
        saved_pos = _save_unit_positions(unit)
        saved_target_pos = _save_unit_positions(target)
        enemy_positions = _collect_enemy_positions(self.units_b)

        execute_charge_movement(unit, target, self.board, enemy_positions)

        self._preview_positions = list(unit.alive_positions())
        centroid = _unit_centroid(unit)
        if saved_pos:
            sx = sum(c for c, r in saved_pos[:unit.models_alive]) / max(unit.models_alive, 1)
            sy = sum(r for c, r in saved_pos[:unit.models_alive]) / max(unit.models_alive, 1)
        else:
            sx, sy = 0, 0
        self._preview_path = ((sx, sy), (centroid[0], centroid[1]))

        # Restore both units
        _restore_unit_positions(unit, saved_pos, self.board)
        _restore_unit_positions(target, saved_target_pos, self.board)

    # ───────────────────────────────────────────────────────────────
    # EXECUTION
    # ───────────────────────────────────────────────────────────────

    def _confirm_action(self):
        """Execute the confirmed human action."""
        unit = self._active_unit
        if unit is None:
            return

        unit.activated = True
        active_label = self.labels[self._selected_unit_idx]
        action = self._chosen_action
        desc_parts = []
        combat_stats = None

        if action == "charge" and self._charge_target is not None:
            charge_target = self._charge_target
            target_idx = self.unit_to_idx[id(charge_target)]
            target_label = self.labels[target_idx]

            pre_centre = unit.centre()
            enemy_positions = _collect_enemy_positions(self.units_b)
            execute_charge_movement(unit, charge_target, self.board, enemy_positions)
            post_centre = unit.centre()
            move_dist = _dist(pre_centre, post_centre)
            execute_counter_charge(charge_target, unit, self.board)

            desc_parts.append(f"{active_label} charges {target_label} {move_dist:.0f}\"")

            # Impact
            impact_info = ""
            if unit.unit.impact > 0:
                imp = resolve_impact(unit, charge_target)
                _sync_dead_models(charge_target, self.board)
                if imp['impact_hits'] > 0:
                    impact_info = f"Impact: {imp['impact_hits']} hits, {imp['impact_wounds']} wounds"

            # Charger swings
            charger_wounds = 0
            if charge_target.models_alive > 0:
                combat_stats = resolve_melee(unit, charge_target, is_charge=True, recorded=True)
                charger_wounds = combat_stats['wounds_dealt'] if combat_stats else 0
                _sync_dead_models(charge_target, self.board)

            # Defender strikes back
            defender_wounds = 0
            if unit.models_alive > 0 and charge_target.models_alive > 0:
                def_stats = resolve_melee(charge_target, unit, is_strike_back=True, recorded=True)
                defender_wounds = def_stats['wounds_dealt'] if def_stats else 0
                _sync_dead_models(unit, self.board)

            melee_parts = []
            if impact_info:
                melee_parts.append(impact_info)
            melee_parts.append(f"Melee: {charger_wounds} wounds dealt, {defender_wounds} received")

            # Melee morale
            if unit.models_alive > 0 and charge_target.models_alive > 0:
                check_melee_morale(unit, charger_wounds, defender_wounds)
                check_melee_morale(charge_target, defender_wounds, charger_wounds)
                _sync_dead_models(unit, self.board)
                _sync_dead_models(charge_target, self.board)

            if charge_target.models_alive <= 0:
                melee_parts.append(f"{target_label} destroyed!")
            if unit.models_alive <= 0:
                melee_parts.append(f"{active_label} destroyed!")

            unit.fatigued = True
            if charge_target.models_alive > 0:
                charge_target.fatigued = True

            # Post-melee
            if unit.models_alive > 0 and charge_target.models_alive > 0:
                enemy_positions = _collect_enemy_positions(self.units_b)
                post_melee_separation(unit, charge_target, self.board, enemy_positions)
            elif unit.models_alive > 0:
                consolidation_move(unit, self.board, self.units_b, OBJECTIVES, self.mode)
            elif charge_target.models_alive > 0:
                consolidation_move(charge_target, self.board, self.units_a, OBJECTIVES, self.mode)

            desc_parts.append("-- " + ", ".join(melee_parts))

            if combat_stats is None:
                combat_stats = {}
            combat_stats['combat_type'] = 'melee'
            combat_stats['charger_wounds'] = charger_wounds
            combat_stats['defender_wounds'] = defender_wounds
            if impact_info:
                combat_stats['impact_info'] = impact_info

        elif action in ("advance", "rush") and self._move_goal is not None:
            if action == "advance" and self._saved_positions_before_move is not None:
                # Movement was already executed in _move_to_shooting;
                # compute distance from saved start position
                sx = sum(c for c, r in self._saved_positions_before_move) / max(len(self._saved_positions_before_move), 1)
                sy = sum(r for c, r in self._saved_positions_before_move) / max(len(self._saved_positions_before_move), 1)
                post_centre = unit.centre()
                move_dist = _dist((sx, sy), post_centre)
                self._saved_positions_before_move = None  # consumed
            else:
                # Rush (or advance that wasn't pre-applied) — execute now
                budget = unit.unit.advance_distance if action == "advance" else unit.unit.rush_distance
                pre_centre = unit.centre()
                enemy_positions = _collect_enemy_positions(self.units_b)
                execute_movement(unit, self._move_goal, budget, self.board, enemy_positions,
                                 flying=unit.unit.flying)
                post_centre = unit.centre()
                move_dist = _dist(pre_centre, post_centre)

            action_verb = "Advances" if action == "advance" else "Rushes"
            desc_parts.append(f"{active_label} {action_verb} {move_dist:.0f}\"")

            if action == "rush":
                desc_parts.append("(unit cannot attack due to rushing)")
            elif action == "advance":
                if unit.shaken:
                    unit.shaken = False
                    desc_parts.append("(was Shaken, recovers)")
                elif self._shoot_target is not None:
                    target = self._shoot_target
                    target_idx = self.unit_to_idx[id(target)]
                    target_label = self.labels[target_idx]
                    before = target.models_alive
                    combat_stats = resolve_shooting(unit, target, recorded=True)
                    check_morale(target)
                    _sync_dead_models(target, self.board)
                    killed = before - target.models_alive
                    if killed > 0:
                        if target.models_alive <= 0:
                            desc_parts.append(f"and shoots {target_label}, destroying the unit!")
                        else:
                            desc_parts.append(f"and shoots {target_label}, killing {killed} model{'s' if killed != 1 else ''}")
                    else:
                        desc_parts.append(f"and shoots {target_label}, no casualties")
                else:
                    desc_parts.append("(no targets shot)")

        elif action == "hold":
            desc_parts.append(f"{active_label} Holds")
            if unit.shaken:
                unit.shaken = False
                desc_parts.append("(was Shaken, recovers)")
            elif self._shoot_target is not None:
                target = self._shoot_target
                target_idx = self.unit_to_idx[id(target)]
                target_label = self.labels[target_idx]
                before = target.models_alive
                combat_stats = resolve_shooting(unit, target, recorded=True)
                check_morale(target)
                _sync_dead_models(target, self.board)
                killed = before - target.models_alive
                if killed > 0:
                    if target.models_alive <= 0:
                        desc_parts.append(f"and shoots {target_label}, destroying the unit!")
                    else:
                        desc_parts.append(f"and shoots {target_label}, killing {killed} model{'s' if killed != 1 else ''}")
                else:
                    desc_parts.append(f"and shoots {target_label}, no casualties")
            else:
                desc_parts.append("(no targets shot)")

        # Record human frame
        description = " ".join(desc_parts)
        self._record_frame(description, combat_stats=combat_stats)

        # Update combat stats panel
        if combat_stats:
            self.stats_var.set(self._format_combat_stats(combat_stats))
        else:
            self.stats_var.set("(no combat this activation)")

        # Check game over
        if self._check_game_over():
            self._end_game()
            return

        # Now run AI turn
        self.current_is_a = False
        self._preview_positions = None
        self._preview_path = None

        if self._check_round_over():
            self._end_round()
            return

        # Defer AI activation until user clicks "Next (AI turn)"
        self._pending_ai_frames = []
        self._ai_turn_pending = True

        self.phase = PHASE_RESOLUTION
        self.status_var.set(description)
        self._render()

    def _run_ai_activation(self):
        """Run one AI (Player B) activation and queue the frame."""
        my_units = self.units_b
        opp_units = self.units_a

        # Check if AI has activatable units
        activatable = [u for u in my_units if u.models_alive > 0 and not u.activated]
        if not activatable:
            self.b_done = True
            if not self.a_done:
                self.a_finished_first = False
            return

        active, target_ranking, ml_action, ml_goal, \
            ml_charge_target, ml_reason, ml_assessment = \
            get_ai_decision(
                self.ml_model, my_units, opp_units,
                self.round_num + 1, self.board, "B",
                apply_tactical_fn=self._apply_tactical,
                plan_fn=self._plan_activation,
                use_planning=self.ml_planning,
                planning_params=self.planning_params,
                fr_friendly=self._fr_b, fm_friendly=self._fm_b,
                fr_enemy=self._fr_a, fm_enemy=self._fm_a,
                pts_friendly=self._pts_b, pts_enemy=self._pts_a,
                units_a=self.units_a, units_b=self.units_b,
                fr_a=self._fr_a, fm_a=self._fm_a,
                fr_b=self._fr_b, fm_b=self._fm_b,
                pts_a=self._pts_a, pts_b=self._pts_b,
                mode=self.mode,
            )

        if active is None:
            self.b_done = True
            if not self.a_done:
                self.a_finished_first = False
            return

        def resolve_target(act, opp):
            return self._pick_target_ranking(act, opp, target_ranking)

        result = execute_activation(
            active, ml_action, ml_goal, ml_charge_target, ml_reason,
            my_units=my_units, opp_units=opp_units,
            board=self.board,
            labels=self.labels, unit_to_idx=self.unit_to_idx,
            mode=self.mode,
            resolve_shoot_target=resolve_target,
        )

        self._record_frame(result.description,
                           combat_stats=result.combat_stats,
                           ml_assessment=ml_assessment)
        self._pending_ai_frames.append({
            'description': result.description,
            'combat_stats': result.combat_stats,
            'ml_assessment': ml_assessment,
        })

    def _show_next_ai_frame(self):
        """Run deferred AI activation (if pending) then show the AI frame."""
        if self._ai_turn_pending:
            self._ai_turn_pending = False
            self._run_ai_activation()

        if self._pending_ai_frames:
            frame_info = self._pending_ai_frames.pop(0)
            self.status_var.set(frame_info['description'])
            if frame_info.get('combat_stats'):
                self.stats_var.set(self._format_combat_stats(frame_info['combat_stats']))
            else:
                self.stats_var.set("(no combat this activation)")
            # Update AI assessment
            if frame_info.get('ml_assessment'):
                self._update_ai_assessment(frame_info['ml_assessment'])
            self._render()
            return

        # No more AI frames, continue
        self._after_resolution()

    def _after_resolution(self):
        """After viewing resolution frames, determine what's next."""
        if self._check_game_over():
            self._end_game()
            return

        if self._check_round_over():
            self._end_round()
            return

        # Switch back to human
        self.current_is_a = True

        # Check if human has activatable units
        a_can = any(u.models_alive > 0 and not u.activated for u in self.units_a)
        if not a_can:
            self.a_done = True
            if not self.b_done:
                self.a_finished_first = True
            # Defer next AI activation — one per button click
            b_can = any(u.models_alive > 0 and not u.activated for u in self.units_b)
            if not b_can:
                self.b_done = True
                self._end_round()
                return
            self._pending_ai_frames = []
            self._ai_turn_pending = True
            self.phase = PHASE_RESOLUTION
            self.status_var.set("AI is finishing its activations. Click Next to continue.")
            self._render()
            return

        # Check if AI has units — if not, just let human keep going
        b_can = any(u.models_alive > 0 and not u.activated for u in self.units_b)
        if not b_can:
            self.b_done = True
            if not self.a_done:
                self.a_finished_first = False

        self._clear_selection()
        self.phase = PHASE_SELECT_UNIT
        self.status_var.set("Select a unit to activate")
        self._render()

    def _update_ai_assessment(self, assessment: dict):
        """Update the AI assessment text panel."""
        self.ai_text.configure(state=tk.NORMAL)
        self.ai_text.delete("1.0", tk.END)
        text = self._format_ai_assessment(assessment)
        self.ai_text.insert("1.0", text)
        self.ai_text.configure(state=tk.DISABLED)

    def _format_ai_assessment(self, assessment: dict) -> str:
        """Format AI assessment for display."""
        if 'planning_candidates' in assessment:
            candidates = assessment['planning_candidates']
            if not candidates:
                return "Planning: no candidates"
            sorted_cands = sorted(candidates, key=lambda c: c['value'], reverse=True)
            lines = [f"MC Planning — {len(candidates)} candidates", ""]
            for c in sorted_cands:
                prefix = ">" if c.get('selected') else " "
                marker = " *" if c.get('selected') else ""
                target_str = f"  tgt:{c['top_target']}" if c.get('top_target') else ""
                move = c.get('move_type', '?')
                lines.append(
                    f"{prefix} {c['unit_name']}  {c['action']}"
                    f"  ({move})  val:{c['value']:+.3f}{target_str}{marker}"
                )
                if c.get('reason'):
                    lines.append(f"    {c['reason']}")
                lines.append("")
            return "\n".join(lines)

        # Tactical assessment (non-planning)
        value = assessment.get('value', 0)
        lines = [f"State Value: {value:+.3f}", ""]
        name = assessment.get('selected_name', '?')
        move_type = assessment.get('move_type', '?')
        confidence = assessment.get('move_type_confidence')
        conf_str = f"  ({confidence:.0%})" if confidence is not None else ""
        lines.append(f"Selected: {name}")
        action = assessment.get('action')
        reason = assessment.get('reason')
        if action:
            lines.append(f"  action: {action}  move: {move_type}{conf_str}")
            if reason:
                lines.append(f"  ({reason})")
        target_scores = assessment.get('target_scores')
        enemy_names = assessment.get('enemy_names')
        if target_scores and enemy_names:
            lines.append("")
            lines.append("Target Priority:")
            entries = [(s, n) for s, n in zip(target_scores, enemy_names) if n is not None]
            entries.sort(key=lambda e: e[0], reverse=True)
            for score, ename in entries:
                lines.append(f"  {ename}: {score:.2f}")
        return "\n".join(lines)

    def _format_combat_stats(self, stats: dict) -> str:
        """Format combat stats for the panel."""
        combat_type = stats.get('combat_type', 'shooting')
        if combat_type == 'melee':
            lines = ["MELEE COMBAT"]
            impact_info = stats.get('impact_info')
            if impact_info:
                lines.append(impact_info)
            if 'attacker_quality' in stats:
                lines.append(f"Attacker Quality: {stats['attacker_quality']}+")
                lines.append(f"Defender Defense: {stats['defender_defense']}+")
                lines.append(f"Attacks: {stats.get('total_attacks', '?')}")
                lines.append(f"Hits: {stats.get('total_hits', '?')}")
                lines.append(f"Failed Def Rolls: {stats.get('total_wounds', '?')}")
            lines.append("")
            lines.append(f"Charger wounds dealt: {stats.get('charger_wounds', 0)}")
            lines.append(f"Defender wounds dealt: {stats.get('defender_wounds', 0)}")
            return "\n".join(lines)
        else:
            mod = stats.get('hit_modifier', 0)
            mod_str = f" (modifier: {mod:+d})" if mod != 0 else ""
            lines = [f"Attacker Quality: {stats.get('attacker_quality', '?')}+{mod_str}"]
            lines.append(f"Defender Defense: {stats.get('defender_defense', '?')}+")
            lines.append("")
            for w in stats.get('attacker_weapons', []):
                count = w.get('count', 1)
                count_str = f"{count}x " if count > 1 else ""
                stat_str = f"{count_str}{w['name']}  {w['range']}\"  A{w['attacks']}"
                if w.get('abilities'):
                    stat_str += "  " + ", ".join(w['abilities'])
                lines.append(stat_str)
            lines.append("")
            lines.append(f"Attacks: {stats.get('total_attacks', '?')}")
            lines.append(f"Hits: {stats.get('total_hits', '?')}")
            lines.append(f"Failed Def Rolls: {stats.get('total_wounds', '?')}")
            return "\n".join(lines)

    # ───────────────────────────────────────────────────────────────
    # REPLAY
    # ───────────────────────────────────────────────────────────────

    def _build_replay_metadata(self) -> tuple[list[int], list[dict]]:
        """Build unit_points and unit_info lists for GameViewer."""
        unit_points = [u.unit.points for u in self.all_units]
        unit_info = []
        for ui, u in enumerate(self.all_units):
            ru = u.unit
            weapons = []
            seen: dict[str, int] = {}
            for w in ru.weapons:
                seen[w.name] = seen.get(w.name, 0) + 1
            for wname, count in seen.items():
                w = next(w for w in ru.weapons if w.name == wname)
                prefix = f"{count}x " if count > 1 else ""
                rng = f'{w.range_inches}"' if w.range_inches > 0 else "melee"
                abilities = []
                if w.ap: abilities.append(f"AP({w.ap})")
                if w.blast: abilities.append(f"Blast({w.blast})")
                if w.deadly: abilities.append(f"Deadly({w.deadly})")
                if w.rending: abilities.append("Rending")
                if w.reliable: abilities.append("Reliable")
                ab = f" [{', '.join(abilities)}]" if abilities else ""
                weapons.append(f"{prefix}{wname} ({rng}, A{w.attacks}{ab})")
            special = []
            for attr, label in [('scout', 'Scout'), ('stealth', 'Stealth'),
                                ('fast', 'Fast'), ('flying', 'Flying'),
                                ('relentless', 'Relentless'), ('artillery', 'Artillery'),
                                ('fearless', 'Fearless'), ('regeneration', 'Regeneration'),
                                ('furious', 'Furious'), ('impact', None)]:
                val = getattr(ru, attr, None)
                if val:
                    special.append(f"Impact({val})" if attr == 'impact' else label)
            unit_info.append({
                'template_id': ru.template_id,
                'points': ru.points,
                'quality': ru.quality,
                'defense': ru.defense,
                'models': ru.models,
                'tough': getattr(ru, 'tough', 0),
                'ai_role': getattr(u, 'ai_role', ''),
                'combat_preference': getattr(ru, 'combat_preference', ''),
                'special': special,
                'weapons': weapons,
            })
        return unit_points, unit_info

    # ── Shared helpers for AI suggestion callbacks ───────────────

    def _ai_decision_kwargs(self, player: str) -> dict:
        """Build the keyword arguments for get_ai_decision."""
        if player == "A":
            fr_f, fm_f = self._fr_a, self._fm_a
            fr_e, fm_e = self._fr_b, self._fm_b
            pts_f, pts_e = self._pts_a, self._pts_b
        else:
            fr_f, fm_f = self._fr_b, self._fm_b
            fr_e, fm_e = self._fr_a, self._fm_a
            pts_f, pts_e = self._pts_b, self._pts_a
        return dict(
            apply_tactical_fn=self._apply_tactical,
            plan_fn=self._plan_activation,
            use_planning=self.ml_planning,
            planning_params=self.planning_params,
            fr_friendly=fr_f, fm_friendly=fm_f,
            fr_enemy=fr_e, fm_enemy=fm_e,
            pts_friendly=pts_f, pts_enemy=pts_e,
            units_a=self.units_a, units_b=self.units_b,
            fr_a=self._fr_a, fm_a=self._fm_a,
            fr_b=self._fr_b, fm_b=self._fm_b,
            pts_a=self._pts_a, pts_b=self._pts_b,
            mode=self.mode,
        )

    def _run_one_ai_activation(self, player: str, round_num: int):
        """Run one AI activation for *player* and return (result, active_idx,
        target_ranking, ml_assessment).  Returns (None, …) if no units."""
        if player == "A":
            friendly, enemy = self.units_a, self.units_b
        else:
            friendly, enemy = self.units_b, self.units_a

        active, target_ranking, action, goal, charge_target, reason, \
            ml_assessment = get_ai_decision(
                self.ml_model, friendly, enemy, round_num,
                self.board, player, **self._ai_decision_kwargs(player))

        if active is None:
            return None, None, None, None

        def resolve_target(act, opp):
            return self._pick_target_ranking(act, opp, target_ranking)

        result = execute_activation(
            active, action, goal, charge_target, reason,
            my_units=friendly, opp_units=enemy,
            board=self.board,
            labels=self.labels, unit_to_idx=self.unit_to_idx,
            mode=self.mode,
            resolve_shoot_target=resolve_target,
        )
        active_idx = self.unit_to_idx[id(active)]
        return result, active_idx, target_ranking, ml_assessment

    def _build_summary(self, active_idx, action, goal, reason,
                       charge_target, target_ranking, opp_units):
        """Build a short summary string for the AI suggestion popup."""
        al = self.labels[active_idx]
        lines = [f"Activate: {al}",
                 f"Action:   {action.capitalize()}"]
        if goal:
            lines.append(f"Move to:  ({goal[0]}, {goal[1]})")
        if action == "charge" and charge_target is not None:
            ci = self.unit_to_idx[id(charge_target)]
            lines.append(f"Charge:   {self.labels[ci]}")
        lines.append(f"Reason:   {reason}")
        if target_ranking and action not in ("rush", "charge"):
            ranked = []
            for ri in target_ranking:
                if ri < len(opp_units) and opp_units[ri].models_alive > 0:
                    ranked.append(
                        self.labels[self.unit_to_idx[id(opp_units[ri])]])
            if ranked:
                lines.append(f"Shoot:    {ranked[0]}")
        return "\n".join(lines)

    # ── Replay viewer ─────────────────────────────────────────

    def _open_replay(self):
        """Open the replay viewer as a modal window showing frames collected so far."""
        from viewer import GameViewer
        if not self.frames:
            return
        unit_points, unit_info = self._build_replay_metadata()

        snapshots = list(self.game_snapshots)

        # Persistent simulation state for "next suggestion" chaining.
        sim = {
            'tip': None,           # GameSnapshot after last A action
            'round_num': 0,
            'a_finished_first': True,
        }

        # ── Initial suggestion callback ──

        def ai_suggest_fn(frame_idx: int) -> dict:
            if frame_idx < 0 or frame_idx >= len(snapshots):
                return {'error': 'No snapshot available for this frame.'}

            live_snap = snapshot_game_state(
                self.units_a, self.units_b, self.board)
            try:
                restore_game_state(
                    snapshots[frame_idx],
                    self.units_a, self.units_b, self.board)

                round_num = self.frames[frame_idx].get('round', 1)

                # Get A's decision
                active, tr, action, goal, ct, reason, assess = \
                    get_ai_decision(
                        self.ml_model, self.units_a, self.units_b,
                        round_num, self.board, "A",
                        **self._ai_decision_kwargs("A"))

                if active is None:
                    return {'error':
                            'No activatable Player A units at this point.'}

                active_idx = self.unit_to_idx[id(active)]
                al = self.labels[active_idx]

                # Before frame
                before = _snapshot(
                    self.all_units, self.board.objective_control)
                before['description'] = f"Before: {al} ready"
                before['round'] = round_num

                # Execute
                def resolve_tgt(act, opp):
                    return self._pick_target_ranking(act, opp, tr)

                result = execute_activation(
                    active, action, goal, ct, reason,
                    my_units=self.units_a, opp_units=self.units_b,
                    board=self.board,
                    labels=self.labels, unit_to_idx=self.unit_to_idx,
                    mode=self.mode,
                    resolve_shoot_target=resolve_tgt,
                )

                # After frame
                after = _snapshot(
                    self.all_units, self.board.objective_control)
                after['description'] = result.description
                after['round'] = round_num
                if result.combat_stats:
                    after['combat_stats'] = result.combat_stats

                # Save simulation tip
                sim['tip'] = snapshot_game_state(
                    self.units_a, self.units_b, self.board)
                sim['round_num'] = round_num

                summary = self._build_summary(
                    active_idx, action, goal, reason, ct, tr,
                    self.units_b)

                return {
                    'frames': [before, after],
                    'labels': list(self.labels),
                    'owners': list(self.owners),
                    'colors': list(self.colors),
                    'summary': summary,
                    'active_idx': active_idx,
                }
            finally:
                restore_game_state(
                    live_snap,
                    self.units_a, self.units_b, self.board)

        # ── Next-suggestion callback ──

        def next_suggest_fn() -> dict:
            if sim['tip'] is None:
                return {'error': 'Click AI Suggestion first.'}

            live_snap = snapshot_game_state(
                self.units_a, self.units_b, self.board)
            try:
                restore_game_state(
                    sim['tip'],
                    self.units_a, self.units_b, self.board)

                round_num = sim['round_num']
                a_finished_first = sim['a_finished_first']
                frames: list[dict] = []
                max_iter = 30  # safety limit

                for _ in range(max_iter):
                    if sim_check_game_over(self.units_a, self.units_b):
                        return {'error': 'Game over.',
                                'frames': frames}

                    a_done, b_done, aff, round_over = \
                        sim_check_round_over(
                            self.units_a, self.units_b)
                    if aff is not None:
                        a_finished_first = aff

                    if round_over:
                        desc = sim_end_round(
                            self.board, self.units_a, self.units_b,
                            round_num, self.mode)
                        ef = _snapshot(self.all_units,
                                       self.board.objective_control)
                        ef['description'] = desc
                        ef['round'] = round_num + 1
                        frames.append(ef)
                        round_num += 1
                        if round_num >= 4:
                            return {'error': 'Game over (4 rounds).',
                                    'frames': frames}
                        sim_start_round(
                            self.units_a, self.units_b,
                            round_num, self.mode,
                            a_finished_first=a_finished_first)
                        continue

                    # ── B activation ──
                    if not b_done:
                        res_b, bidx, _, _ = \
                            self._run_one_ai_activation(
                                "B", round_num)
                        if res_b is not None:
                            bf = _snapshot(
                                self.all_units,
                                self.board.objective_control)
                            bf['description'] = res_b.description
                            bf['round'] = round_num
                            if res_b.combat_stats:
                                bf['combat_stats'] = \
                                    res_b.combat_stats
                            bf['_active_idx'] = bidx
                            frames.append(bf)

                            if sim_check_game_over(
                                    self.units_a, self.units_b):
                                return {'error': 'Game over.',
                                        'frames': frames}

                    # ── A suggestion ──
                    if not a_done:
                        active, tr, action, goal, ct, reason, _ = \
                            get_ai_decision(
                                self.ml_model,
                                self.units_a, self.units_b,
                                round_num, self.board, "A",
                                **self._ai_decision_kwargs("A"))

                        if active is None:
                            # A is done this round; loop to handle
                            continue

                        aidx = self.unit_to_idx[id(active)]
                        al = self.labels[aidx]

                        before = _snapshot(
                            self.all_units,
                            self.board.objective_control)
                        before['description'] = \
                            f"Before: {al} ready"
                        before['round'] = round_num

                        def _resolve(act, opp, _tr=tr):
                            return self._pick_target_ranking(
                                act, opp, _tr)

                        result = execute_activation(
                            active, action, goal, ct, reason,
                            my_units=self.units_a,
                            opp_units=self.units_b,
                            board=self.board,
                            labels=self.labels,
                            unit_to_idx=self.unit_to_idx,
                            mode=self.mode,
                            resolve_shoot_target=_resolve,
                        )

                        after = _snapshot(
                            self.all_units,
                            self.board.objective_control)
                        after['description'] = result.description
                        after['round'] = round_num
                        if result.combat_stats:
                            after['combat_stats'] = \
                                result.combat_stats

                        sim['tip'] = snapshot_game_state(
                            self.units_a, self.units_b, self.board)
                        sim['round_num'] = round_num
                        sim['a_finished_first'] = a_finished_first

                        summary = self._build_summary(
                            aidx, action, goal, reason, ct, tr,
                            self.units_b)

                        return {
                            'frames': frames + [before, after],
                            'labels': list(self.labels),
                            'owners': list(self.owners),
                            'colors': list(self.colors),
                            'summary': summary,
                            'active_idx': aidx,
                        }

                return {'error': 'Could not advance simulation.'}
            finally:
                restore_game_state(
                    live_snap,
                    self.units_a, self.units_b, self.board)

        viewer = GameViewer(
            self.frames, self.labels, self.owners,
            mode=self.mode, unit_points=unit_points, unit_info=unit_info,
            parent=self.root, ai_suggest_fn=ai_suggest_fn,
        )
        viewer._next_suggest_fn = next_suggest_fn
        # Jump to the last frame
        viewer.current = len(self.frames) - 1
        viewer._render()
        viewer.run()

    # ───────────────────────────────────────────────────────────────
    # KEYBOARD HANDLERS
    # ───────────────────────────────────────────────────────────────

    def _on_key_prev(self):
        pass  # Could implement frame stepping if desired

    def _on_key_next(self):
        if self.phase == PHASE_RESOLUTION:
            if self._pending_ai_frames or self._ai_turn_pending:
                self._show_next_ai_frame()
            else:
                self._after_resolution()

    def _on_key_replay(self):
        if self.phase in (PHASE_SELECT_UNIT, PHASE_BETWEEN_ROUNDS, PHASE_GAME_OVER):
            if len(self.frames) > 1:
                self._open_replay()

    # ───────────────────────────────────────────────────────────────
    # RUN
    # ───────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ===================================================================
# ENTRY POINT
# ===================================================================

def _load_army_from_data(data) -> 'ArmyList':
    """Load an ArmyList from a JSON dict (HoF entry or imported list)."""
    from models import ArmyList, ArmyListEntry
    entries = []
    for e in data['entries']:
        entry = ArmyListEntry(
            template_id=e['template_id'],
            chosen_upgrades=e.get('upgrades', {}),
            ai_role=e.get('ai_role', 'killer'),
            combat_preference=e.get('combat_preference', 'ranged'),
            attached_to=e.get('attached_to', -1),
        )
        entries.append(entry)
    army = ArmyList(entries=entries, fitness=data.get('fitness', 0))
    return army


def _load_player_army(config: dict, hof: list[dict]) -> tuple['ArmyList', str]:
    """Build the human player's army based on dialog selection.

    Returns (army, description_string).
    """
    from models import ArmyList, ArmyListEntry, compute_entry_cost
    from evolution import generate_random_army

    choice = config['army_choice']

    if choice == PreGameDialog.ARMY_HOF_ML:
        idx = random.randrange(len(hof))
        army_data = hof[idx]
        army = _load_army_from_data(army_data)
        desc = f"HoF_ml rank #{army_data['rank']}"
        return army, desc

    if choice == PreGameDialog.ARMY_RANDOM:
        MIN_PTS = 1980
        for _ in range(200):
            army = generate_random_army(mode="objectives")
            # compute_entry_cost fills computed_cost on each entry
            for e in army.entries:
                compute_entry_cost(e)
            if army.total_cost >= MIN_PTS:
                desc = f"Random ({army.total_cost}pts)"
                return army, desc
        # Fallback: return last attempt even if under budget
        desc = f"Random ({army.total_cost}pts)"
        return army, desc

    if choice == PreGameDialog.ARMY_IMPORTED:
        from import_list import convert_list
        list_dir = Path(__file__).parent / "Imported Lists"
        filepath = list_dir / config['imported_file']
        data, warnings = convert_list(filepath)
        for w in warnings:
            print(w)
        army = _load_army_from_data(data)
        # Recompute costs via resolve so computed_cost is accurate
        for e in army.entries:
            compute_entry_cost(e)
        desc = f"Imported: {config['imported_file']} ({army.total_cost}pts)"
        return army, desc

    raise ValueError(f"Unknown army choice: {choice}")


def play_interactive():
    """Launch an interactive game: human (Player A) vs tactical ML model (Player B)."""
    import torch
    from ml_model_tactical import TacticalModel
    from ml_training import load_model_state_dict
    from evolution import resolve_army, _make_unit_states

    # Load hall of fame (needed for AI army + optional player choice)
    hof_path = Path(__file__).parent / "results" / "hall_of_fame_ml.json"
    if not hof_path.exists():
        print("Error: hall_of_fame_ml.json not found in results/")
        return

    with open(hof_path) as f:
        hof = json.load(f)

    if len(hof) < 2:
        print("Error: need at least 2 armies in hall of fame")
        return

    # Load tactical model
    checkpoint_path = Path(__file__).parent / "ml_checkpoints" / "final_model.pt"
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Pre-game dialog
    root_tmp = tk.Tk()
    root_tmp.withdraw()
    dialog = PreGameDialog(root_tmp)
    config = dialog.run()
    root_tmp.destroy()

    if config is None:
        print("Game cancelled.")
        return

    ml_planning = config['planning']
    planning_params = config['planning_params']

    # --- Player army (A) ---
    army_a, army_a_desc = _load_player_army(config, hof)

    # --- AI army (B): always a random HoF_ml list ---
    ai_idx = random.randrange(len(hof))
    army_b_data = hof[ai_idx]
    army_b = _load_army_from_data(army_b_data)

    army_a_resolved = resolve_army(army_a)
    army_b_resolved = resolve_army(army_b)

    states_a = _make_unit_states(army_a, army_a_resolved, "A")
    states_b = _make_unit_states(army_b, army_b_resolved, "B")

    # Deploy
    board = Board()
    deploy_armies(states_a, states_b, board)

    labels = _make_unit_labels(states_a, states_b)
    owners = [u.owner for u in states_a + states_b]

    from main import format_army
    print(f"Your army ({army_a.total_cost}pts, {army_a_desc}):")
    print(format_army(army_a))
    print(f"\nAI army ({army_b.total_cost}pts, HoF_ml rank #{army_b_data['rank']}):")
    print(format_army(army_b))
    print(f"\nML Planning: {'ON' if ml_planning else 'OFF'}")
    if planning_params:
        print(f"Planning params: K={planning_params['K_UNITS']}, C={planning_params['C_SAMPLES_PER_UNIT']}, M={planning_params['M_ROLLOUTS']}, N={planning_params['N_LOOKAHEAD']}")
    print("\nLaunching game...")

    viewer = PlayViewer(
        states_a, states_b, board, labels, owners,
        ml_model=model, mode="objectives",
        ml_planning=ml_planning, planning_params=planning_params,
    )
    viewer.run()


if __name__ == "__main__":
    play_interactive()
