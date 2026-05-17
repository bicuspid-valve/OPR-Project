"""Training metrics, army loading, and army pair generation."""
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from models import ResolvedUnit, UnitState

# Try to import army generation utilities (optional — only needed for real training)
try:
    from evolution import generate_random_army, resolve_army, _make_unit_states
    _HAS_EVOLUTION = True
except Exception:
    _HAS_EVOLUTION = False


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Rolling metrics for monitoring training progress."""
    heuristic_results: deque = field(default_factory=lambda: deque(maxlen=200))
    selfplay_results: deque = field(default_factory=lambda: deque(maxlen=200))
    heuristic_hof_results: deque = field(default_factory=lambda: deque(maxlen=200))
    heuristic_hof_ml_results: deque = field(default_factory=lambda: deque(maxlen=200))
    heuristic_random_results: deque = field(default_factory=lambda: deque(maxlen=200))
    selfplay_hof_results: deque = field(default_factory=lambda: deque(maxlen=200))
    selfplay_hof_ml_results: deque = field(default_factory=lambda: deque(maxlen=200))
    selfplay_random_results: deque = field(default_factory=lambda: deque(maxlen=200))
    batch_logs: list[dict] = field(default_factory=list)
    # Per-physical-side win rate tracking (A/B symmetry monitoring)
    a_side_results: deque = field(default_factory=lambda: deque(maxlen=200))
    b_side_results: deque = field(default_factory=lambda: deque(maxlen=200))

    def record_game(self, result: str, opponent_type: str,
                    army_type: str = "random",
                    physical_side: str | None = None) -> None:
        # result is main-perspective: "main" = main model won, "opp" = lost, "draw" = draw.
        # Physical "A"/"B" is transformed in _run_games_batched_tactical before dispatch.
        win = 1.0 if result == "main" else (0.5 if result == "draw" else 0.0)
        if opponent_type == "heuristic":
            self.heuristic_results.append(win)
            if army_type == "hof":
                self.heuristic_hof_results.append(win)
            elif army_type == "hof_ml":
                self.heuristic_hof_ml_results.append(win)
            else:
                self.heuristic_random_results.append(win)
        elif opponent_type == "selfplay":
            # Checkpoint opponents only. Mirror matches are excluded: both sides
            # are logged, so the pair always averages to 0.5 and would bias the
            # metric toward 0.5. Mirrors feed the per-side deques below instead.
            self.selfplay_results.append(win)
            if army_type == "hof":
                self.selfplay_hof_results.append(win)
            elif army_type == "hof_ml":
                self.selfplay_hof_ml_results.append(win)
            else:
                self.selfplay_random_results.append(win)
        # Track per-physical-side win rate (mirror self-play only —
        # cleanest signal for game-engine symmetry)
        if physical_side and opponent_type in ("selfplay_mirror", "mirror_b"):
            if physical_side == "A":
                self.a_side_results.append(win)
            elif physical_side == "B":
                self.b_side_results.append(win)

    @property
    def heuristic_win_rate(self) -> float:
        if not self.heuristic_results:
            return 0.5
        return sum(self.heuristic_results) / len(self.heuristic_results)

    @property
    def selfplay_win_rate(self) -> float:
        if not self.selfplay_results:
            return 0.5
        return sum(self.selfplay_results) / len(self.selfplay_results)

    def _wr(self, dq: deque) -> float:
        if not dq:
            return 0.5
        return sum(dq) / len(dq)

    @property
    def heuristic_hof_win_rate(self) -> float:
        return self._wr(self.heuristic_hof_results)

    @property
    def heuristic_hof_ml_win_rate(self) -> float:
        return self._wr(self.heuristic_hof_ml_results)

    @property
    def heuristic_random_win_rate(self) -> float:
        return self._wr(self.heuristic_random_results)

    @property
    def selfplay_hof_win_rate(self) -> float:
        return self._wr(self.selfplay_hof_results)

    @property
    def selfplay_hof_ml_win_rate(self) -> float:
        return self._wr(self.selfplay_hof_ml_results)

    @property
    def selfplay_random_win_rate(self) -> float:
        return self._wr(self.selfplay_random_results)

    @property
    def a_side_win_rate(self) -> float:
        return self._wr(self.a_side_results)

    @property
    def b_side_win_rate(self) -> float:
        return self._wr(self.b_side_results)

    def log_batch(self, batch_num: int, loss_metrics: dict,
                  heuristic_fraction: float) -> dict:
        entry = {
            "batch": batch_num,
            "heuristic_win_rate": round(self.heuristic_win_rate, 4),
            "selfplay_win_rate": round(self.selfplay_win_rate, 4),
            "heuristic_hof_wr": round(self.heuristic_hof_win_rate, 4),
            "heuristic_hof_ml_wr": round(self.heuristic_hof_ml_win_rate, 4),
            "heuristic_random_wr": round(self.heuristic_random_win_rate, 4),
            "selfplay_hof_wr": round(self.selfplay_hof_win_rate, 4),
            "selfplay_hof_ml_wr": round(self.selfplay_hof_ml_win_rate, 4),
            "selfplay_random_wr": round(self.selfplay_random_win_rate, 4),
            "heuristic_fraction": round(heuristic_fraction, 2),
            **{k: round(v, 6) for k, v in loss_metrics.items() if isinstance(v, (int, float))},
        }
        self.batch_logs.append(entry)
        return entry


# ---------------------------------------------------------------------------
# Army list helpers for training
# ---------------------------------------------------------------------------

def _load_hof_armies_from_file(filename: str) -> list:
    """Load army lists from results/<filename>.

    Returns a list of ArmyList objects, or an empty list if the file
    is missing or the evolution module is unavailable.
    """
    if not _HAS_EVOLUTION:
        return []
    try:
        from evolution import make_entry, _attached_hero_count
        from ml_features import MAX_UNITS_PER_SIDE
        from models import ArmyList
        hof_path = Path(__file__).resolve().parent.parent / "results" / filename
        if not hof_path.exists():
            return []
        with open(hof_path) as f:
            hof_data = json.load(f)
        armies = []
        skipped = 0
        for entry_data in hof_data:
            army = ArmyList()
            for e in entry_data["entries"]:
                entry = make_entry(
                    e["template_id"],
                    upgrades=e.get("upgrades", {}),
                    ai_role=e.get("ai_role", "killer"),
                )
                entry.combat_preference = e.get("combat_preference", "ranged")
                entry.attached_to = e.get("attached_to", -1)
                army.entries.append(entry)
            # Drop legacy HoF armies that exceed the unit slot count — older
            # evolution runs without enforce_forceorg occasionally produced
            # 11+ effective-unit armies, which the model can't represent.
            effective_units = len(army.entries) - _attached_hero_count(army)
            if effective_units > MAX_UNITS_PER_SIDE:
                skipped += 1
                continue
            armies.append(army)
        if skipped:
            print(f"[hof] {filename}: skipped {skipped} legacy armies "
                  f"with >{MAX_UNITS_PER_SIDE} effective units")
        return armies
    except Exception:
        return []


def _load_hof_armies() -> list:
    """Load army lists from results/hall_of_fame.json."""
    return _load_hof_armies_from_file("hall_of_fame.json")


def _load_hof_ml_armies() -> list:
    """Load army lists from results/hall_of_fame_ml.json."""
    return _load_hof_armies_from_file("hall_of_fame_ml.json")


def _generate_army_pair(
    opp_type: str = "heuristic",
    hof_armies: list | None = None,
    hof_ml_armies: list | None = None,
    return_attach_data: bool = False,
) -> tuple:
    """Generate a pair of armies for a training game.

    Army selection depends on opponent type:

    **vs heuristic** (player B is heuristic):
    - Player B (heuristic) always gets a hall_of_fame.json list.
    - Player A (ML) gets a hall_of_fame.json list 50% / hall_of_fame_ml.json 50%.
    - Falls back to random if the required HoF files are unavailable.

    **vs selfplay** (both players are ML):
    - Both players get the same list *type*: random 50%, hall_of_fame.json 25%,
      hall_of_fame_ml.json 25%.
    - Falls back to random when a selected HoF source is unavailable.

    Returns (resolved_a, resolved_b, states_a, states_b, army_type).
    army_type is "hof", "hof_ml", or "random".
    """
    if not _HAS_EVOLUTION:
        raise RuntimeError(
            "evolution module not available — cannot generate random armies. "
            "Use run_training_batch() with pre-built armies instead."
        )

    if opp_type == "heuristic":
        # Player B (heuristic) always from hall_of_fame.json
        if hof_armies:
            army_b = random.choice(hof_armies)
        else:
            army_b = generate_random_army(mode="objectives", enforce_forceorg=True)

        # Player A (ML): 50% hall_of_fame.json, 50% hall_of_fame_ml.json
        if random.random() < 0.5:
            if hof_armies:
                army_a = random.choice(hof_armies)
                army_type = "hof"
            else:
                army_a = generate_random_army(mode="objectives", enforce_forceorg=True)
                army_type = "random"
        else:
            if hof_ml_armies:
                army_a = random.choice(hof_ml_armies)
                army_type = "hof_ml"
            else:
                army_a = generate_random_army(mode="objectives", enforce_forceorg=True)
                army_type = "random"
    else:
        # Self-play: both get same type — random 50%, hof 25%, hof_ml 25%
        roll = random.random()
        if roll < 0.5:
            army_a = generate_random_army(mode="objectives", enforce_forceorg=True)
            army_b = generate_random_army(mode="objectives", enforce_forceorg=True)
            army_type = "random"
        elif roll < 0.75:
            if hof_armies:
                army_a = random.choice(hof_armies)
                army_b = random.choice(hof_armies)
                army_type = "hof"
            else:
                army_a = generate_random_army(mode="objectives", enforce_forceorg=True)
                army_b = generate_random_army(mode="objectives", enforce_forceorg=True)
                army_type = "random"
        else:
            if hof_ml_armies:
                army_a = random.choice(hof_ml_armies)
                army_b = random.choice(hof_ml_armies)
                army_type = "hof_ml"
            else:
                army_a = generate_random_army(mode="objectives", enforce_forceorg=True)
                army_b = generate_random_army(mode="objectives", enforce_forceorg=True)
                army_type = "random"

    res_a = resolve_army(army_a)
    res_b = resolve_army(army_b)
    states_a = _make_unit_states(army_a, res_a, "A")
    states_b = _make_unit_states(army_b, res_b, "B")
    if return_attach_data:
        attach_a = _entries_attach_data(army_a)
        attach_b = _entries_attach_data(army_b)
        return res_a, res_b, states_a, states_b, army_type, attach_a, attach_b
    return res_a, res_b, states_a, states_b, army_type


def _entries_attach_data(army) -> list[tuple[int, bool]]:
    """Per-entry (attached_to, is_hero) data for an ArmyList. Used by the
    training generator to redo hero merging on a fresh UnitState list — the
    spec sends ``res_a`` (per-entry ResolvedUnits) and ``states_a_data``
    (per-state, post-merge), and without this attach data the generator
    cannot tell which res_a entries are heroes that should be merged into
    a host. Returns a list of (attached_to, is_hero) tuples, one per entry,
    aligned with ``resolve_army(army)``."""
    from templates import get_templates_dict
    td = get_templates_dict()
    out: list[tuple[int, bool]] = []
    for e in army.entries:
        tpl = td.get(e.template_id)
        is_hero = bool(tpl and tpl.hero)
        out.append((int(e.attached_to), is_hero))
    return out
