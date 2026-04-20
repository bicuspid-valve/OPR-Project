"""Tests for tactical model input calculations.

Verifies that encode_state_tactical produces correct feature values,
the has_activated flag works properly, alive masks are built correctly,
damage matchups scale appropriately, and player-perspective transforms
(coordinate flipping, objective remapping) are consistent.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from board import Board, COLS, ROWS, OBJECTIVES
from models import ResolvedUnit, UnitState, Weapon
from ml_features import (
    TACTICAL_TOTAL_FEATURES,
    TACTICAL_UNIT_FEATURES,
    GLOBAL_FEATURES,
    MAX_UNITS_PER_SIDE,
    BOARD_DIAG,
    _MAX_TOUGH,
    _MAX_MODELS,
    _MAX_SPEED,
    _TOFF_ACTIVATED,
    _TOFF_FATIGUED,
    _TOFF_SHAKEN,
    starting_wounds,
    precompute_damage,
    encode_state_tactical,
    _flip_y,
    _get_model_objectives,
    _objective_control_mapped,
    _survival_fraction,
)
from ml_integration_tactical import (
    apply_tactical_model,
    project_post_move_unit_state,
    is_phase_reencode_enabled,
    set_phase_reencode_enabled,
)
import ml_model_tactical as _mmt
from ml_model_tactical import (
    TacticalModel,
    PHASE_PRE_SELECT, PHASE_POST_SELECT, PHASE_POST_MOVETYPE, PHASE_POST_DEST,
    N_PHASES, DEFAULT_PHASE_ITERS, DEFAULT_CORE_ITERS, TRUNK_WIDTH, UNIT_EMBED_DIM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remap_objective(obj_idx: int, player: str) -> int:
    """Remap objective index for player perspective (0→0, 1↔2, 3↔4 for B)."""
    if player == "A":
        return obj_idx
    return {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}[obj_idx]


# ---------------------------------------------------------------------------
# Fixture helpers (same pattern as test_ml_features.py)
# ---------------------------------------------------------------------------

def _make_weapon(*, name="Gun", range_inches=24, attacks=1, ap=0,
                 deadly=0, melee=False, reliable=False, blast=0) -> Weapon:
    return Weapon(name=name, range_inches=range_inches, attacks=attacks,
                  ap=ap, deadly=deadly, melee=melee, reliable=reliable,
                  blast=blast)


def _make_resolved(*, name="TestUnit", models=5, quality=4, defense=4,
                   tough=0, points=100, weapons=None, impact=0,
                   flying=False, artillery=False, stealth=False,
                   fearless=False, fear=0, fast=False, teleport=False) -> ResolvedUnit:
    if weapons is None:
        weapons = [_make_weapon() for _ in range(models)]
    wpm = [[] for _ in range(models)]
    for i, w in enumerate(weapons):
        wpm[i % models].append(w)
    return ResolvedUnit(
        template_id="test", name=name, models=models, quality=quality,
        defense=defense, tough=tough, points=points, weapons=weapons,
        weapons_per_model=wpm, flying=flying, artillery=artillery,
        stealth=stealth, fearless=fearless, fear=fear, fast=fast,
        teleport=teleport, impact=impact,
    )


def _make_unit_state(resolved: ResolvedUnit, owner: str = "A",
                     positions: list[tuple[int, int]] | None = None,
                     activated: bool = False,
                     fatigued: bool = False) -> UnitState:
    us = UnitState(unit=resolved, owner=owner)
    us.activated = activated
    us.fatigued = fatigued
    if positions is not None:
        us.positions = list(positions)
    else:
        us.positions = [(10 + i, 5) for i in range(resolved.models)]
    return us


def _simple_game_state(player: str = "A"):
    """Build a minimal 2-unit-per-side game for testing."""
    f1 = _make_resolved(name="Archers", models=5, quality=4, defense=4,
                        points=100, weapons=[
                            _make_weapon(attacks=1, ap=0) for _ in range(5)
                        ])
    f2 = _make_resolved(name="Swords", models=3, quality=4, defense=3,
                        points=80, weapons=[
                            _make_weapon(name="Blade", melee=True,
                                         range_inches=0, attacks=2)
                            for _ in range(3)
                        ])
    e1 = _make_resolved(name="EnemyGuns", models=5, quality=4, defense=4,
                        points=120, weapons=[
                            _make_weapon(attacks=2, ap=1) for _ in range(5)
                        ])
    e2 = _make_resolved(name="EnemyTank", models=1, quality=3, defense=2,
                        points=200, tough=6, weapons=[
                            _make_weapon(attacks=6, ap=3, deadly=3),
                        ])

    if player == "A":
        f_pos = [[(10 + j, 5) for j in range(5)],
                 [(20 + j, 5) for j in range(3)]]
        e_pos = [[(10 + j, 40) for j in range(5)],
                 [(30, 42)]]
    else:
        f_pos = [[(10 + j, 40) for j in range(5)],
                 [(20 + j, 42) for j in range(3)]]
        e_pos = [[(10 + j, 5) for j in range(5)],
                 [(30, 5)]]

    fu = [_make_unit_state(f1, owner=player, positions=f_pos[0]),
          _make_unit_state(f2, owner=player, positions=f_pos[1])]
    enemy_owner = "B" if player == "A" else "A"
    eu = [_make_unit_state(e1, owner=enemy_owner, positions=e_pos[0]),
          _make_unit_state(e2, owner=enemy_owner, positions=e_pos[1])]
    board = Board()
    return fu, eu, board


# ===========================================================================
# 1. BASIC SHAPE & DTYPE
# ===========================================================================

class TestTacticalShape:
    def test_output_size(self):
        fu, eu, board = _simple_game_state()
        vec = encode_state_tactical(fu, eu, 1, board, "A")
        assert vec.shape == (TACTICAL_TOTAL_FEATURES,)
        assert vec.shape[0] == 4016

    def test_dtype_float32(self):
        fu, eu, board = _simple_game_state()
        vec = encode_state_tactical(fu, eu, 1, board, "A")
        assert vec.dtype == torch.float32

    def test_feature_count_constant(self):
        """TACTICAL_TOTAL_FEATURES = 20 * 200 + 16 = 4016."""
        assert TACTICAL_UNIT_FEATURES == 200  # 87 base + 70 ranged + 10 melee + 10 post-adv + 10 obj-reach + 10 can-charge + 3 bools
        assert TACTICAL_TOTAL_FEATURES == MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES + GLOBAL_FEATURES
        assert TACTICAL_TOTAL_FEATURES == 4016


# ===========================================================================
# 2. HAS_ACTIVATED FLAG (the key difference from strategic encoding)
# ===========================================================================

class TestHasActivatedFlag:
    def _activated_index(self, slot: int) -> int:
        """Index of the has_activated feature for a given unit slot."""
        return slot * TACTICAL_UNIT_FEATURES + _TOFF_ACTIVATED

    def test_not_activated(self):
        """Alive, unactivated unit should have has_activated = 0.0."""
        r = _make_resolved(models=3, points=80)
        us = _make_unit_state(r, activated=False)
        eu_r = _make_resolved(models=1, points=50)
        eu = _make_unit_state(eu_r, owner="B", positions=[(30, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        idx = self._activated_index(0)
        assert vec[idx].item() == 0.0

    def test_activated(self):
        """Alive, activated unit should have has_activated = 1.0."""
        r = _make_resolved(models=3, points=80)
        us = _make_unit_state(r, activated=True)
        eu_r = _make_resolved(models=1, points=50)
        eu = _make_unit_state(eu_r, owner="B", positions=[(30, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        idx = self._activated_index(0)
        assert vec[idx].item() == 1.0

    def test_dead_unit_has_activated_zero(self):
        """Dead units should have has_activated = 0.0."""
        r = _make_resolved(models=3, points=80)
        us = _make_unit_state(r, activated=True)
        us.models_alive = 0
        us.positions.clear()
        eu_r = _make_resolved(models=1, points=50)
        eu = _make_unit_state(eu_r, owner="B", positions=[(30, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        idx = self._activated_index(0)
        assert vec[idx].item() == 0.0

    def test_enemy_activated_flag(self):
        """Enemy units should also get has_activated encoded."""
        fu_r = _make_resolved(models=1, points=50)
        fu = _make_unit_state(fu_r, positions=[(10, 5)])
        eu_r = _make_resolved(models=3, points=80)
        eu = _make_unit_state(eu_r, owner="B", positions=[(10, 40), (11, 40), (12, 40)],
                              activated=True)
        board = Board()
        vec = encode_state_tactical([fu], [eu], 1, board, "A")
        # Enemy slot 0 = index 10 in slots
        enemy_idx = self._activated_index(MAX_UNITS_PER_SIDE)  # slot 10
        assert vec[enemy_idx].item() == 1.0

    def test_mixed_activation_states(self):
        """Multiple friendly units with different activation states."""
        r1 = _make_resolved(name="Unit1", models=3, points=80)
        r2 = _make_resolved(name="Unit2", models=2, points=60)
        r3 = _make_resolved(name="Unit3", models=4, points=90)

        us1 = _make_unit_state(r1, activated=True)
        us2 = _make_unit_state(r2, activated=False,
                               positions=[(20, 5), (21, 5)])
        us3 = _make_unit_state(r3, activated=True,
                               positions=[(30 + i, 5) for i in range(4)])

        eu_r = _make_resolved(models=1, points=50)
        eu = _make_unit_state(eu_r, owner="B", positions=[(30, 40)])
        board = Board()

        vec = encode_state_tactical([us1, us2, us3], [eu], 1, board, "A")
        assert vec[self._activated_index(0)].item() == 1.0  # us1 activated
        assert vec[self._activated_index(1)].item() == 0.0  # us2 not activated
        assert vec[self._activated_index(2)].item() == 1.0  # us3 activated

    def test_empty_slot_has_activated_zero(self):
        """Padded (empty) slots should have has_activated = 0.0."""
        r = _make_resolved(models=1, points=50)
        fu = _make_unit_state(r, positions=[(10, 5)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([fu], [eu], 1, board, "A")
        # Friendly slot 1 (empty) should have has_activated = 0.0
        assert vec[self._activated_index(1)].item() == 0.0


# ===========================================================================
# 2b. FATIGUED FLAG
# ===========================================================================

class TestFatiguedFlag:
    def _fatigued_index(self, slot: int) -> int:
        """Index of the fatigued feature for a given unit slot."""
        return slot * TACTICAL_UNIT_FEATURES + _TOFF_FATIGUED

    def test_not_fatigued(self):
        """Alive, non-fatigued unit should have fatigued = 0.0."""
        r = _make_resolved(models=3, points=80)
        us = _make_unit_state(r, fatigued=False)
        eu_r = _make_resolved(models=1, points=50)
        eu = _make_unit_state(eu_r, owner="B", positions=[(30, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        assert vec[self._fatigued_index(0)].item() == 0.0

    def test_fatigued(self):
        """Alive, fatigued unit should have fatigued = 1.0."""
        r = _make_resolved(models=3, points=80)
        us = _make_unit_state(r, fatigued=True)
        eu_r = _make_resolved(models=1, points=50)
        eu = _make_unit_state(eu_r, owner="B", positions=[(30, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        assert vec[self._fatigued_index(0)].item() == 1.0

    def test_dead_unit_fatigued_zero(self):
        """Dead units should have fatigued = 0.0."""
        r = _make_resolved(models=3, points=80)
        us = _make_unit_state(r, fatigued=True)
        us.models_alive = 0
        us.positions.clear()
        eu_r = _make_resolved(models=1, points=50)
        eu = _make_unit_state(eu_r, owner="B", positions=[(30, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        assert vec[self._fatigued_index(0)].item() == 0.0

    def test_enemy_fatigued_flag(self):
        """Enemy units should also get fatigued encoded."""
        fu_r = _make_resolved(models=1, points=50)
        fu = _make_unit_state(fu_r, positions=[(10, 5)])
        eu_r = _make_resolved(models=3, points=80)
        eu = _make_unit_state(eu_r, owner="B", positions=[(10, 40), (11, 40), (12, 40)],
                              fatigued=True)
        board = Board()
        vec = encode_state_tactical([fu], [eu], 1, board, "A")
        enemy_idx = self._fatigued_index(MAX_UNITS_PER_SIDE)  # slot 10
        assert vec[enemy_idx].item() == 1.0

    def test_empty_slot_fatigued_zero(self):
        """Padded (empty) slots should have fatigued = 0.0."""
        r = _make_resolved(models=1, points=50)
        fu = _make_unit_state(r, positions=[(10, 5)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([fu], [eu], 1, board, "A")
        assert vec[self._fatigued_index(1)].item() == 0.0


# ===========================================================================
# 3. PER-UNIT FEATURE VALUES (features 0–36, same as strategic)
# ===========================================================================

class TestPerUnitFeatures:
    def test_wound_count_normalisation(self):
        """Feature 0: wound count / MAX_TOUGH."""
        r = _make_resolved(models=1, tough=6, points=100)
        us = _make_unit_state(r, positions=[(10, 5)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        assert vec[0].item() == pytest.approx(6 / _MAX_TOUGH)

    def test_non_tough_wound_count(self):
        """Non-tough units get wound_count = 1."""
        r = _make_resolved(models=5, tough=0, points=100)
        us = _make_unit_state(r)
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        assert vec[0].item() == pytest.approx(1 / _MAX_TOUGH)

    def test_model_count_normalisation(self):
        """Feature 1: models / MAX_MODELS."""
        r = _make_resolved(models=5, points=100)
        us = _make_unit_state(r)
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        assert vec[1].item() == pytest.approx(5 / _MAX_MODELS)

    def test_speed_normalisation(self):
        """Feature 2: rush_distance / MAX_SPEED, 0 for artillery."""
        r = _make_resolved(models=1, points=50)
        us = _make_unit_state(r, positions=[(10, 5)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        expected_speed = r.rush_distance / _MAX_SPEED
        assert vec[2].item() == pytest.approx(expected_speed)

    def test_artillery_speed_zero(self):
        """Artillery units should have speed = 0."""
        r = _make_resolved(models=1, artillery=True, points=50)
        us = _make_unit_state(r, positions=[(10, 5)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        assert vec[2].item() == 0.0

    def test_survival_fraction_full(self):
        """Feature 3: survival fraction = 1.0 at full strength."""
        r = _make_resolved(models=5, points=100)
        us = _make_unit_state(r)
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        assert vec[3].item() == pytest.approx(1.0)

    def test_survival_fraction_damaged(self):
        """Feature 3: survival fraction after casualties."""
        r = _make_resolved(models=5, points=100)
        us = _make_unit_state(r, positions=[(10 + i, 5) for i in range(3)])
        us.models_alive = 3
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        assert vec[3].item() == pytest.approx(3 / 5)

    def test_points_fraction(self):
        """Feature 4: unit points / total side points."""
        r1 = _make_resolved(models=3, points=150)
        r2 = _make_resolved(models=2, points=50)
        us1 = _make_unit_state(r1)
        us2 = _make_unit_state(r2, positions=[(20, 5), (21, 5)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([us1, us2], [eu], 1, board, "A")
        # us1 points fraction = 150 / 200
        assert vec[4].item() == pytest.approx(150 / 200)
        # us2 points fraction = 50 / 200
        us2_start = TACTICAL_UNIT_FEATURES  # slot 1 start
        assert vec[us2_start + 4].item() == pytest.approx(50 / 200)

    def test_ability_flags(self):
        """Features 5-8: flying, artillery, fearless, fear."""
        r = _make_resolved(models=1, flying=True, fearless=True, fear=2, points=50)
        us = _make_unit_state(r, positions=[(10, 5)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        assert vec[5].item() == 1.0  # flying
        assert vec[6].item() == 0.0  # not artillery
        assert vec[7].item() == 1.0  # fearless
        assert vec[8].item() == 1.0  # fear > 0

    def test_is_friendly_flag(self):
        """Feature 9: 1.0 for friendly, 0.0 for enemy."""
        fu, eu, board = _simple_game_state()
        vec = encode_state_tactical(fu, eu, 1, board, "A")
        # Friendly slot 0 feature 9
        assert vec[9].item() == 1.0
        # Enemy slot 0 feature 9 (slot 10)
        enemy_start = MAX_UNITS_PER_SIDE * TACTICAL_UNIT_FEATURES
        assert vec[enemy_start + 9].item() == 0.0

    def test_position_normalisation(self):
        """Features 10-11: x/COLS, y/ROWS (with flip for Player B)."""
        r = _make_resolved(models=1, points=50)
        us = _make_unit_state(r, positions=[(36, 24)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        assert vec[10].item() == pytest.approx(36 / COLS)
        assert vec[11].item() == pytest.approx(24 / ROWS)

    def test_position_flip_player_b(self):
        """Player B's y-coordinates should be flipped."""
        r = _make_resolved(models=1, points=50)
        us = _make_unit_state(r, positions=[(36, 40)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="A", positions=[(10, 5)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "B")
        assert vec[10].item() == pytest.approx(36 / COLS)
        expected_y = _flip_y(40) / ROWS  # (47-40)/48 = 7/48
        assert vec[11].item() == pytest.approx(expected_y)

    def test_objective_distances(self):
        """Features 12-16: distance to 5 objectives, normalised by BOARD_DIAG."""
        r = _make_resolved(models=1, points=50)
        pos = (36, 24)  # centre of board
        us = _make_unit_state(r, positions=[pos])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")

        objectives = _get_model_objectives("A")
        for i, (ox, oy) in enumerate(objectives):
            d = math.sqrt((pos[0] - ox) ** 2 + (pos[1] - oy) ** 2)
            expected = d / BOARD_DIAG
            assert vec[12 + i].item() == pytest.approx(expected, abs=1e-5), \
                f"Objective {i} distance mismatch"


# ===========================================================================
# 4. DAMAGE MATCHUP FEATURES (features 17-36)
# ===========================================================================

class TestDamageMatchupFeatures:
    def test_ranged_matchup_nonzero(self):
        """Ranged kill proportion features should be nonzero for ranged units."""
        r = _make_resolved(models=5, quality=4, points=100, weapons=[
            _make_weapon(attacks=2, ap=1) for _ in range(5)
        ])
        e_r = _make_resolved(models=5, quality=4, defense=4, points=100)
        us = _make_unit_state(r)
        eu = _make_unit_state(e_r, owner="B",
                              positions=[(10 + i, 40) for i in range(5)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        # Feature 27 = first ranged matchup value (6" threshold vs enemy slot 0)
        assert vec[27].item() > 0.0

    def test_melee_matchup_zero_for_ranged_only(self):
        """Melee kill proportions should be 0 for ranged-only units."""
        r = _make_resolved(models=5, quality=4, points=100, weapons=[
            _make_weapon(attacks=2, ap=1) for _ in range(5)
        ])
        e_r = _make_resolved(models=5, quality=4, defense=4, points=100)
        us = _make_unit_state(r)
        eu = _make_unit_state(e_r, owner="B",
                              positions=[(10 + i, 40) for i in range(5)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        # Features 97-106 = melee matchups (should all be 0 for ranged-only)
        for i in range(10):
            assert vec[97 + i].item() == 0.0

    def test_damage_scales_with_casualties(self):
        """Matchup values should scale with models_alive / starting_models."""
        r = _make_resolved(models=5, quality=4, points=100, weapons=[
            _make_weapon(attacks=2, ap=1) for _ in range(5)
        ])
        e_r = _make_resolved(models=3, quality=4, defense=4, points=80)
        us_full = _make_unit_state(r, positions=[(i, 5) for i in range(5)])
        eu = _make_unit_state(e_r, owner="B",
                              positions=[(i, 40) for i in range(3)])
        board = Board()

        vec_full = encode_state_tactical([us_full], [eu], 1, board, "A")

        # Kill 2 models
        us_half = _make_unit_state(r, positions=[(i, 5) for i in range(3)])
        us_half.models_alive = 3
        vec_half = encode_state_tactical([us_half], [eu], 1, board, "A")

        # Feature 27 = first ranged matchup (6" threshold vs enemy slot 0)
        full_val = vec_full[27].item()
        half_val = vec_half[27].item()
        assert half_val < full_val
        assert half_val == pytest.approx(full_val * 3 / 5, rel=1e-5)

    def test_unused_enemy_slots_zero(self):
        """Matchup values for empty enemy slots should be 0."""
        r = _make_resolved(models=5, quality=4, points=100, weapons=[
            _make_weapon(attacks=2, ap=1) for _ in range(5)
        ])
        e_r = _make_resolved(models=1, points=50)
        us = _make_unit_state(r)
        eu = _make_unit_state(e_r, owner="B", positions=[(30, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        # Ranged matchups for enemy slots 1-9 should be all zero
        # Each enemy slot has 7 threshold values; enemy slot 1 starts at 27 + 7 = 34
        for slot in range(1, 10):
            for t in range(7):
                assert vec[27 + slot * 7 + t].item() == 0.0


# ===========================================================================
# 5. GLOBAL FEATURES (round one-hot, obj control, points remaining)
# ===========================================================================

class TestGlobalFeatures:
    def _global_start(self) -> int:
        return MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES

    @pytest.mark.parametrize("round_num", [1, 2, 3, 4])
    def test_round_one_hot(self, round_num):
        fu, eu, board = _simple_game_state()
        vec = encode_state_tactical(fu, eu, round_num, board, "A")
        start = self._global_start()
        for r in range(4):
            expected = 1.0 if (r + 1) == round_num else 0.0
            assert vec[start + r].item() == pytest.approx(expected)

    def test_objective_control_neutral(self):
        """All objectives neutral → all control values = 0."""
        fu, eu, board = _simple_game_state()
        vec = encode_state_tactical(fu, eu, 1, board, "A")
        start = self._global_start() + 4  # after round one-hot
        for i in range(5):
            assert vec[start + i].item() == pytest.approx(0.0)

    def test_objective_control_friendly(self):
        """Friendly-controlled objective → +1.0."""
        fu, eu, board = _simple_game_state()
        board.objective_control[0] = "A"  # centre
        vec = encode_state_tactical(fu, eu, 1, board, "A")
        start = self._global_start() + 4
        assert vec[start].item() == pytest.approx(1.0)

    def test_objective_control_enemy(self):
        """Enemy-controlled objective → -1.0."""
        fu, eu, board = _simple_game_state()
        board.objective_control[0] = "B"  # centre controlled by enemy
        vec = encode_state_tactical(fu, eu, 1, board, "A")
        start = self._global_start() + 4
        assert vec[start].item() == pytest.approx(-1.0)

    def test_points_remaining_full(self):
        """Full strength → both points remaining = 1.0."""
        fu, eu, board = _simple_game_state()
        vec = encode_state_tactical(fu, eu, 1, board, "A")
        assert vec[-2].item() == pytest.approx(1.0)
        assert vec[-1].item() == pytest.approx(1.0)

    def test_points_remaining_after_casualties(self):
        """After killing one enemy unit, enemy points should decrease."""
        fu, eu, board = _simple_game_state()
        total_e = sum(u.unit.points for u in eu)
        # Kill enemy unit 0
        eu[0].positions.clear()
        eu[0].models_alive = 0
        vec = encode_state_tactical(fu, eu, 1, board, "A")
        alive_e = eu[1].unit.points
        assert vec[-1].item() == pytest.approx(alive_e / total_e)
        # Friendly still at full
        assert vec[-2].item() == pytest.approx(1.0)


# ===========================================================================
# 6. PLAYER-PERSPECTIVE CONSISTENCY
# ===========================================================================

class TestPlayerPerspective:
    def test_symmetric_encoding(self):
        """Encoding from A's perspective should be structurally identical
        to encoding from B's perspective (with appropriate flips)."""
        fu_a, eu_a, board_a = _simple_game_state("A")
        fu_b, eu_b, board_b = _simple_game_state("B")
        vec_a = encode_state_tactical(fu_a, eu_a, 1, board_a, "A")
        vec_b = encode_state_tactical(fu_b, eu_b, 1, board_b, "B")
        # Both should be the same size
        assert vec_a.shape == vec_b.shape == (TACTICAL_TOTAL_FEATURES,)

    def test_objective_control_remapped_for_b(self):
        """Player B should see swapped objective control (indices 1↔2, 3↔4)."""
        fu, eu, board = _simple_game_state()
        board.objective_control[1] = "A"  # A-side controlled by A
        board.objective_control[2] = "B"  # B-side controlled by B

        ctrl = _objective_control_mapped(board, "A")
        assert ctrl[1] == 1.0   # A's my-side = A-side → +1
        assert ctrl[2] == -1.0  # A's enemy-side = B-side → -1

        ctrl_b = _objective_control_mapped(board, "B")
        assert ctrl_b[1] == 1.0   # B's my-side = B-side → +1
        assert ctrl_b[2] == -1.0  # B's enemy-side = A-side → -1

    def test_remap_objective_player_a(self):
        """Player A: identity mapping."""
        for i in range(5):
            assert remap_objective(i, "A") == i

    def test_remap_objective_player_b(self):
        """Player B: 0→0, 1↔2, 3↔4."""
        assert remap_objective(0, "B") == 0
        assert remap_objective(1, "B") == 2
        assert remap_objective(2, "B") == 1
        assert remap_objective(3, "B") == 4
        assert remap_objective(4, "B") == 3


# ===========================================================================
# 7. DEAD UNIT ENCODING
# ===========================================================================

class TestDeadUnitEncoding:
    def test_dead_unit_all_zeros(self):
        """Dead units should produce all-zero features (including has_activated)."""
        r = _make_resolved(models=3, points=80)
        us = _make_unit_state(r)
        us.models_alive = 0
        us.positions.clear()

        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(30, 40)])
        board = Board()
        vec = encode_state_tactical([us], [eu], 1, board, "A")
        # All 38 features for slot 0 should be zero
        slot0 = vec[:TACTICAL_UNIT_FEATURES]
        assert (slot0 == 0.0).all()


# ===========================================================================
# 8. PADDING / EMPTY SLOTS
# ===========================================================================

class TestPadding:
    def test_empty_friendly_slots_zeroed(self):
        """Unused friendly slots beyond actual units should be all zeros."""
        r = _make_resolved(models=1, points=50)
        fu = _make_unit_state(r, positions=[(10, 5)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([fu], [eu], 1, board, "A")
        # Slots 1..9 should be all zeros
        for slot in range(1, MAX_UNITS_PER_SIDE):
            start = slot * TACTICAL_UNIT_FEATURES
            end = start + TACTICAL_UNIT_FEATURES
            assert (vec[start:end] == 0.0).all(), f"Friendly slot {slot} not zeroed"

    def test_empty_enemy_slots_zeroed(self):
        """Unused enemy slots should be all zeros."""
        fu = _make_unit_state(_make_resolved(models=1, points=50),
                              positions=[(10, 5)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state_tactical([fu], [eu], 1, board, "A")
        enemy_base = MAX_UNITS_PER_SIDE * TACTICAL_UNIT_FEATURES
        for slot in range(1, MAX_UNITS_PER_SIDE):
            start = enemy_base + slot * TACTICAL_UNIT_FEATURES
            end = start + TACTICAL_UNIT_FEATURES
            assert (vec[start:end] == 0.0).all(), f"Enemy slot {slot} not zeroed"


# ===========================================================================
# 9. ALIVE MASK CONSTRUCTION (from ml_integration_tactical)
# ===========================================================================

class TestAliveMask:
    def test_alive_unactivated_mask(self):
        """Alive + unactivated units should be True in mask."""
        model = TacticalModel()
        r1 = _make_resolved(models=3, points=80)
        r2 = _make_resolved(models=2, points=60)
        us1 = _make_unit_state(r1, activated=False)
        us2 = _make_unit_state(r2, activated=True,
                               positions=[(20, 5), (21, 5)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(30, 40)])
        board = Board()

        selected, _, assessment = apply_tactical_model(
            model, [us1, us2], [eu], 1, board, "A")
        # Only us1 is alive+unactivated, so it must be selected
        assert selected is us1
        assert assessment['selected_slot'] == 0

    def test_all_activated_returns_none(self):
        """If all units are activated, should return None."""
        model = TacticalModel()
        r = _make_resolved(models=3, points=80)
        us = _make_unit_state(r, activated=True)
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(30, 40)])
        board = Board()

        selected, mults, _ = apply_tactical_model(
            model, [us], [eu], 1, board, "A")
        assert selected is None
        assert mults == [1.0] * MAX_UNITS_PER_SIDE

    def test_dead_units_excluded(self):
        """Dead units should not be selectable."""
        model = TacticalModel()
        r1 = _make_resolved(models=3, points=80)
        r2 = _make_resolved(models=2, points=60)
        us1 = _make_unit_state(r1, activated=False)
        us1.models_alive = 0
        us1.positions.clear()
        us2 = _make_unit_state(r2, activated=False,
                               positions=[(20, 5), (21, 5)])
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(30, 40)])
        board = Board()

        selected, _, assessment = apply_tactical_model(
            model, [us1, us2], [eu], 1, board, "A")
        assert selected is us2
        assert assessment['selected_slot'] == 1


# ===========================================================================
# 10. MODEL OUTPUT SHAPES & APPLICATION
# ===========================================================================

class TestModelOutputs:
    def test_model_forward_shapes(self):
        """TacticalModel forward pass should produce correctly shaped outputs."""
        model = TacticalModel()
        x = torch.randn(TACTICAL_TOTAL_FEATURES)
        mask = torch.tensor([True] * 5 + [False] * 5)
        outputs = model(x, mask)
        assert len(outputs) == 7
        unit_logits, role_probs, obj_probs, target_prio, combat_pref, stance, value = outputs
        assert unit_logits.shape == (10,)
        assert role_probs.shape == (2,)
        assert obj_probs.shape == (5,)
        assert target_prio.shape == (10,)
        assert combat_pref.shape == ()
        assert stance.shape == (3,)
        assert value.shape == ()

    def test_masked_logits_negative_inf(self):
        """Masked-out unit slots should have -inf logits."""
        model = TacticalModel()
        x = torch.randn(TACTICAL_TOTAL_FEATURES)
        mask = torch.tensor([True, True, False, False, False,
                             False, False, False, False, False])
        unit_logits, *_ = model(x, mask)
        assert unit_logits[0].item() != float('-inf')
        assert unit_logits[1].item() != float('-inf')
        for i in range(2, 10):
            assert unit_logits[i].item() == float('-inf')

    def test_role_probs_sum_to_one(self):
        """Role probabilities should sum to 1."""
        model = TacticalModel()
        x = torch.randn(TACTICAL_TOTAL_FEATURES)
        _, role_probs, *_ = model(x)
        assert role_probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_obj_probs_sum_to_one(self):
        """Objective probabilities should sum to 1."""
        model = TacticalModel()
        x = torch.randn(TACTICAL_TOTAL_FEATURES)
        _, _, obj_probs, *_ = model(x)
        assert obj_probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_stance_probs_sum_to_one(self):
        """Stance probabilities should sum to 1."""
        model = TacticalModel()
        x = torch.randn(TACTICAL_TOTAL_FEATURES)
        *_, stance, _ = model(x)
        assert stance.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_combat_pref_in_0_1(self):
        """Combat preference should be sigmoid output in [0, 1]."""
        model = TacticalModel()
        x = torch.randn(TACTICAL_TOTAL_FEATURES)
        *_, combat_pref, _, _ = model(x)
        assert 0.0 <= combat_pref.item() <= 1.0

    def test_target_priority_positive(self):
        """Target priority multipliers should be positive (exp of clamped values)."""
        model = TacticalModel()
        x = torch.randn(TACTICAL_TOTAL_FEATURES)
        _, _, _, target_prio, *_ = model(x)
        assert (target_prio > 0).all()

    def test_assessment_dict_keys(self):
        """apply_tactical_model should return a complete assessment dict."""
        model = TacticalModel()
        r = _make_resolved(models=3, points=80)
        us = _make_unit_state(r, activated=False)
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(30, 40)])
        board = Board()

        _, _, assessment = apply_tactical_model(
            model, [us], [eu], 1, board, "A")
        expected_keys = {
            'value', 'selected_slot', 'selected_name',
            'unit_selection_logits', 'role', 'role_confidence',
            'objective', 'objective_confidence',
            'combat_preference', 'combat_pref_prob',
            'stance', 'stance_confidence',
        }
        assert set(assessment.keys()) == expected_keys

    @pytest.mark.skip(reason="ROLES/STANCES removed from ml_integration_tactical")
    def test_role_in_valid_set(self):
        """Assigned role should be one of the valid roles."""
        pass

    @pytest.mark.skip(reason="ROLES/STANCES removed from ml_integration_tactical")
    def test_stance_in_valid_set(self):
        """Assigned stance should be one of the valid stances."""
        pass


# ===========================================================================
# 11. BATCH FORWARD PASS
# ===========================================================================

class TestBatchForward:
    def test_batch_output_shapes(self):
        """Batch forward pass should produce correctly shaped outputs."""
        model = TacticalModel()
        batch = 4
        x = torch.randn(batch, TACTICAL_TOTAL_FEATURES)
        mask = torch.ones(batch, 10, dtype=torch.bool)
        outputs = model(x, mask)
        unit_logits, role_probs, obj_probs, target_prio, combat_pref, stance, value = outputs
        assert unit_logits.shape == (batch, 10)
        assert role_probs.shape == (batch, 2)
        assert obj_probs.shape == (batch, 5)
        assert target_prio.shape == (batch, 10)
        assert combat_pref.shape == (batch,)
        assert stance.shape == (batch, 3)
        assert value.shape == (batch,)


# ===========================================================================
# 12. POST-MOVE UNIT STATE PROJECTION (phase-reencode foundation)
# ===========================================================================

class TestProjectPostMoveUnitState:
    """project_post_move_unit_state builds the ephemeral UnitState used to
    produce a post-move state_vec for the POST_DEST trunk re-encode, without
    mutating the canonical unit list.
    """

    def test_positions_translate_to_dest(self):
        r = _make_resolved(models=3, points=80)
        unit = _make_unit_state(r, positions=[(10, 5), (11, 5), (12, 5)])
        # centre ≈ (11, 5); dest (20, 20) → delta (+9, +15)
        projected = project_post_move_unit_state(unit, (20, 20), is_rush=False)
        assert projected.positions == [(19, 20), (20, 20), (21, 20)]

    def test_formation_preserved(self):
        r = _make_resolved(models=4, points=80)
        orig = [(5, 5), (5, 6), (6, 5), (6, 6)]
        unit = _make_unit_state(r, positions=orig)
        projected = project_post_move_unit_state(unit, (20, 20), is_rush=False)
        # Pairwise offsets preserved
        orig_offsets = [(p[0] - orig[0][0], p[1] - orig[0][1]) for p in orig]
        new_offsets = [(p[0] - projected.positions[0][0],
                        p[1] - projected.positions[0][1]) for p in projected.positions]
        assert orig_offsets == new_offsets

    def test_original_unit_not_mutated(self):
        r = _make_resolved(models=3, points=80)
        orig_positions = [(10, 5), (11, 5), (12, 5)]
        unit = _make_unit_state(r, positions=list(orig_positions))
        unit.fatigued = False
        _ = project_post_move_unit_state(unit, (20, 20), is_rush=True)
        assert unit.positions == orig_positions
        assert unit.fatigued is False

    def test_rush_sets_fatigued(self):
        r = _make_resolved(models=2, points=50)
        unit = _make_unit_state(r, positions=[(10, 5), (11, 5)])
        unit.fatigued = False
        projected = project_post_move_unit_state(unit, (20, 20), is_rush=True)
        assert projected.fatigued is True

    def test_advance_preserves_fatigue(self):
        r = _make_resolved(models=2, points=50)
        unit = _make_unit_state(r, positions=[(10, 5), (11, 5)])
        unit.fatigued = False
        projected = project_post_move_unit_state(unit, (15, 10), is_rush=False)
        assert projected.fatigued is False

    def test_hold_in_place_no_translation(self):
        r = _make_resolved(models=3, points=80)
        positions = [(10, 5), (11, 5), (12, 5)]
        unit = _make_unit_state(r, positions=list(positions))
        # centre=(11,5) rounded → dest equals current centre
        projected = project_post_move_unit_state(unit, (11, 5), is_rush=False)
        assert projected.positions == positions

    def test_state_vec_only_differs_in_position_dependent_fields(self):
        """Encoding the projected unit vs. a separately-moved unit should yield
        identical state_vecs — the projection is feature-equivalent to moving.
        """
        fu, eu, board = _simple_game_state("A")
        # Move unit 0 to (25, 20) via projection, then encode
        projected_fu = [project_post_move_unit_state(fu[0], (25, 20), is_rush=False)] + fu[1:]
        vec_via_projection = encode_state_tactical(projected_fu, eu, 1, board, "A")

        # Move via direct mutation on a fresh copy, then encode
        fu_mut = list(_simple_game_state("A")[0])
        cx, cy = fu_mut[0].centre()
        dx = 25 - int(round(cx))
        dy = 20 - int(round(cy))
        fu_mut[0].positions = [(c + dx, r + dy) for (c, r) in fu_mut[0].positions]
        vec_via_mutation = encode_state_tactical(fu_mut, eu, 1, board, "A")

        assert torch.equal(vec_via_projection, vec_via_mutation)

    def test_state_vec_differs_from_pre_move(self):
        """Post-move state_vec should differ from pre-move in at least the
        position-dependent features of the projected unit.
        """
        fu, eu, board = _simple_game_state("A")
        pre_vec = encode_state_tactical(fu, eu, 1, board, "A")

        projected_fu = [project_post_move_unit_state(fu[0], (25, 20), is_rush=False)] + fu[1:]
        post_vec = encode_state_tactical(projected_fu, eu, 1, board, "A")

        assert not torch.equal(pre_vec, post_vec)

        # The moved unit's (x, y) slot (per-unit features 10, 11) must differ.
        slot0_x = 0 * TACTICAL_UNIT_FEATURES + 10
        slot0_y = 0 * TACTICAL_UNIT_FEATURES + 11
        assert pre_vec[slot0_x].item() != post_vec[slot0_x].item()
        assert pre_vec[slot0_y].item() != post_vec[slot0_y].item()


# ===========================================================================
# 13. PHASE-AWARE ENCODE (Step 2 identity gate)
# ===========================================================================

class TestPhaseAwareEncode:
    """encode() with identity-initialised FiLM, zero is_acting embedding, and
    zero per-phase value heads must reproduce the current trunk() exactly when
    called with PHASE_PRE_SELECT / acting_unit_idx=None / h_prev=None.

    These tests are the Step-6 ablation gate in miniature: if any of them break,
    the encode() split has drifted from bit-identity and later PPO replay
    equivalence cannot hold.
    """

    def test_encode_matches_trunk_at_init(self):
        torch.manual_seed(0)
        model = TacticalModel()
        model.eval()
        x = torch.randn(2, TACTICAL_TOTAL_FEATURES)
        with torch.no_grad():
            h_trunk, units_trunk, ro_trunk = model.trunk(x)
            h_enc, units_enc, ro_enc = model.encode(
                x,
                phase=PHASE_PRE_SELECT,
                acting_unit_idx=None,
                h_prev=None,
                n_iters=DEFAULT_CORE_ITERS,
            )
        assert torch.allclose(h_trunk, h_enc, atol=0, rtol=0)
        assert torch.equal(units_trunk, units_enc)
        assert torch.equal(ro_trunk, ro_enc)

    def test_encode_unbatched_matches_trunk(self):
        torch.manual_seed(1)
        model = TacticalModel()
        model.eval()
        x = torch.randn(TACTICAL_TOTAL_FEATURES)
        with torch.no_grad():
            h_trunk, _, _ = model.trunk(x)
            h_enc, _, _ = model.encode(x, n_iters=DEFAULT_CORE_ITERS)
        assert torch.allclose(h_trunk, h_enc, atol=0, rtol=0)

    def test_adjustment_blocks_zero_init(self):
        """Each adjustment block's final Linear is zero-initialised so at fresh
        init continuation phases contribute nothing on top of the shared core_block."""
        model = TacticalModel()
        # There's one adjustment block per continuation phase (N_PHASES - 1).
        assert len(model.phase_adjustment_blocks) == N_PHASES - 1
        for adj in model.phase_adjustment_blocks:
            # Final layer is the 4th element: Linear(BOTTLENECK → TRUNK_WIDTH)
            assert torch.all(adj[-1].weight == 0.0)
            assert torch.all(adj[-1].bias == 0.0)

    def test_is_acting_embed_zero_at_init(self):
        model = TacticalModel()
        assert model.is_acting_embed.num_embeddings == 2
        assert model.is_acting_embed.embedding_dim == UNIT_EMBED_DIM
        assert torch.all(model.is_acting_embed.weight == 0.0)

    def test_per_phase_value_heads_zero_at_init(self):
        model = TacticalModel()
        assert len(model.per_phase_value_heads) == N_PHASES
        for vh in model.per_phase_value_heads:
            assert torch.all(vh.weight == 0.0)
            assert torch.all(vh.bias == 0.0)
        # Output at any h is exactly 0 at init.
        h = torch.randn(3, TRUNK_WIDTH)
        for p in range(N_PHASES):
            assert torch.all(model.per_phase_value(h, p) == 0.0)

    def test_is_acting_injection_is_noop_at_init(self):
        """is_acting_embed.weight = 0 ⇒ encoding with any acting_unit_idx must
        match encoding with acting_unit_idx=None bit-for-bit at init."""
        torch.manual_seed(2)
        model = TacticalModel()
        model.eval()
        x = torch.randn(2, TACTICAL_TOTAL_FEATURES)
        with torch.no_grad():
            h_none, _, _ = model.encode(x, acting_unit_idx=None)
            h_int, _, _ = model.encode(x, acting_unit_idx=3)
            h_tensor, _, _ = model.encode(
                x, acting_unit_idx=torch.tensor([3, 7])
            )
        assert torch.allclose(h_int, h_none, atol=0, rtol=0)
        assert torch.allclose(h_tensor, h_none, atol=0, rtol=0)

    def test_is_acting_injection_changes_h_when_trained(self):
        """After writing a non-zero row-1 into is_acting_embed, encoding with
        an acting unit must differ from encoding without one."""
        torch.manual_seed(3)
        model = TacticalModel()
        model.eval()
        with torch.no_grad():
            model.is_acting_embed.weight[1].normal_(mean=0.0, std=0.1)
        x = torch.randn(1, TACTICAL_TOTAL_FEATURES)
        with torch.no_grad():
            h_none, _, _ = model.encode(x, acting_unit_idx=None)
            h_act, _, _ = model.encode(x, acting_unit_idx=5)
        assert not torch.allclose(h_act, h_none)

    def test_adjustment_block_wired_in(self):
        """After writing non-zero weights into the final Linear of an adjustment
        block, that phase's encode() must diverge from the pre-perturbation
        output — confirms the adjustment path is actually added into h."""
        torch.manual_seed(4)
        model = TacticalModel()
        model.eval()
        x = torch.randn(1, TACTICAL_TOTAL_FEATURES)
        with torch.no_grad():
            h_before, _, _ = model.encode(x, phase=PHASE_POST_SELECT)
            # Perturb POST_SELECT's adjustment block's final Linear bias.
            adj = model.phase_adjustment_blocks[PHASE_POST_SELECT - 1]
            adj[-1].bias.fill_(0.01)
            h_after, _, _ = model.encode(x, phase=PHASE_POST_SELECT)
        assert not torch.allclose(h_before, h_after)

    def test_h_prev_persistence_respected(self):
        """Passing h_prev should start the recurrent loop from that h, not h0,
        and produce a different output from a cold start."""
        torch.manual_seed(5)
        model = TacticalModel()
        model.eval()
        x = torch.randn(1, TACTICAL_TOTAL_FEATURES)
        with torch.no_grad():
            h_cold, _, _ = model.encode(
                x, phase=PHASE_POST_SELECT, n_iters=2, h_prev=None,
            )
            # Arbitrary h_prev — same shape as h_cold but different values
            fake_prev = torch.randn_like(h_cold)
            h_warm, _, _ = model.encode(
                x, phase=PHASE_POST_SELECT, n_iters=2, h_prev=fake_prev,
            )
        assert not torch.allclose(h_cold, h_warm)


# ===========================================================================
# 14. PHASE-REENCODE INFERENCE PATH (Step 3 flag + ablation gate)
# ===========================================================================

class TestPhaseReencodeInference:
    """apply_tactical_model behaviour with the phase_reencode flag toggled.

    Ablation gate: flag=True with FiLM γ=1/β=0, is_acting_embed zero, and all
    continuation-phase iter counts set to 0 must reproduce the legacy head
    choices (selected unit, move type, charge target, shoot target) on any
    input that keeps state_vec_post == state_vec. The shaken-unit path
    guarantees no post-move state rebuild, so it's the cleanest gate.
    """

    def teardown_method(self, _method):
        # Always leave the process-wide flag off between tests.
        set_phase_reencode_enabled(False)

    def test_flag_default_false(self):
        assert is_phase_reencode_enabled() is False

    def test_set_get_flag_round_trip(self):
        set_phase_reencode_enabled(True)
        assert is_phase_reencode_enabled() is True
        set_phase_reencode_enabled(False)
        assert is_phase_reencode_enabled() is False

    def _build_shaken_scenario(self):
        """Scenario: friendly unit 0 is shaken → is_shaken forces MOVE_MOVE
        with hold-in-place destination, so state_vec_post == state_vec in the
        phased path and the ablation gate has a clean comparison target."""
        fu, eu, board = _simple_game_state("A")
        fu[0].shaken = True
        # Shaken units also get activated=False implicitly; leave other flags.
        return fu, eu, board

    def test_legacy_path_runs(self):
        """Sanity: flag=False + fresh model + shaken scenario returns a valid tuple."""
        torch.manual_seed(0)
        model = TacticalModel()
        model.eval()
        fu, eu, board = self._build_shaken_scenario()
        result = apply_tactical_model(model, fu, eu, 1, board, "A")
        assert len(result) == 7
        selected_unit, target_ranking, action, goal, charge_target, reason, assessment = result
        assert selected_unit is not None
        assert isinstance(assessment, dict)
        assert 'value' in assessment

    def test_phased_runs_with_default_iters(self):
        """Flag=True with default iters {6, 2, 2, 2} should execute end-to-end."""
        torch.manual_seed(1)
        model = TacticalModel()
        model.eval()
        fu, eu, board = self._build_shaken_scenario()
        set_phase_reencode_enabled(True)
        try:
            result = apply_tactical_model(model, fu, eu, 1, board, "A")
        finally:
            set_phase_reencode_enabled(False)
        assert len(result) == 7
        selected_unit, _, action, _, _, _, assessment = result
        assert selected_unit is not None
        assert action in ("hold", "advance", "rush", "charge")

    def _set_continuation_iters(self, n: int) -> dict[int, int]:
        """Set DEFAULT_PHASE_ITERS[continuation phases] = n, return original dict.

        With n=0 the continuation phases skip the core_block loop entirely and
        return h_prev unchanged → h_sel = h_mt = h_dest = h_pre → the phased
        path degenerates to legacy (under shaken/charge scenarios where
        state_vec_post == state_vec).
        """
        original = dict(_mmt.DEFAULT_PHASE_ITERS)
        _mmt.DEFAULT_PHASE_ITERS[_mmt.PHASE_POST_SELECT] = n
        _mmt.DEFAULT_PHASE_ITERS[_mmt.PHASE_POST_MOVETYPE] = n
        _mmt.DEFAULT_PHASE_ITERS[_mmt.PHASE_POST_DEST] = n
        return original

    def _restore_continuation_iters(self, original: dict[int, int]) -> None:
        _mmt.DEFAULT_PHASE_ITERS.clear()
        _mmt.DEFAULT_PHASE_ITERS.update(original)

    def test_ablation_gate_shaken_matches_legacy(self):
        """Ablation: flag=True with continuation iters=0 + zero is_acting_embed
        + zero-init adjustment blocks + shaken unit (forces state_vec_post ==
        state_vec) must produce the same head choices as flag=False on the
        same model."""
        torch.manual_seed(2)
        model = TacticalModel()
        model.eval()
        fu, eu, board = self._build_shaken_scenario()

        # Legacy path outputs
        result_legacy = apply_tactical_model(model, fu, eu, 1, board, "A")
        (leg_unit, _, leg_action, leg_goal, leg_charge_tgt, _, leg_assess) = result_legacy

        original = self._set_continuation_iters(0)
        try:
            set_phase_reencode_enabled(True)
            # Rebuild scenario because apply_tactical_model may mutate engine state
            fu2, eu2, board2 = self._build_shaken_scenario()
            result_phased = apply_tactical_model(model, fu2, eu2, 1, board2, "A")
        finally:
            self._restore_continuation_iters(original)
            set_phase_reencode_enabled(False)
        (ph_unit, _, ph_action, ph_goal, ph_charge_tgt, _, ph_assess) = result_phased

        # Selected unit slot + action class + target identity must match
        assert leg_assess['selected_slot'] == ph_assess['selected_slot']
        assert leg_assess['move_type'] == ph_assess['move_type']
        assert leg_assess['charge_target_idx'] == ph_assess['charge_target_idx']
        assert leg_assess['shoot_target_idx'] == ph_assess['shoot_target_idx']
        assert leg_action == ph_action

    def test_ablation_gate_logits_close(self):
        """Same ablation as above, but compare logits numerically — they
        should be equal to float tolerance (all iters-0 continuation passes
        produce h_sel == h_mt == h_dest == h_pre == legacy h)."""
        torch.manual_seed(3)
        model = TacticalModel()
        model.eval()
        fu, eu, board = self._build_shaken_scenario()
        result_legacy = apply_tactical_model(model, fu, eu, 1, board, "A")
        leg_assess = result_legacy[6]

        original = self._set_continuation_iters(0)
        try:
            set_phase_reencode_enabled(True)
            fu2, eu2, board2 = self._build_shaken_scenario()
            result_phased = apply_tactical_model(model, fu2, eu2, 1, board2, "A")
        finally:
            self._restore_continuation_iters(original)
            set_phase_reencode_enabled(False)
        ph_assess = result_phased[6]

        leg_unit_logits = torch.tensor(leg_assess['unit_selection_logits'])
        ph_unit_logits = torch.tensor(ph_assess['unit_selection_logits'])
        # Masked slots become -inf on both sides; compare unmasked only.
        finite = torch.isfinite(leg_unit_logits) & torch.isfinite(ph_unit_logits)
        assert torch.allclose(
            leg_unit_logits[finite], ph_unit_logits[finite], atol=1e-5, rtol=1e-5,
        )
        assert abs(leg_assess['value'] - ph_assess['value']) < 1e-5


# ===========================================================================
# 15. COLLECTION POST-MOVE STATE_VEC (Step 4)
# ===========================================================================

class TestCollectionPostMoveStateVec:
    """_maybe_build_post_move_state_vec is collection.py's entry point for
    producing the POST_DEST re-encode input during trajectory storage. It
    must (a) no-op when the phase-reencode flag is off, (b) no-op on
    activations where state_vec_post would equal state_vec (charge, shaken,
    invalid dest), and (c) produce a float32 (4016,) array otherwise.
    """

    def teardown_method(self, _method):
        set_phase_reencode_enabled(False)

    def _scenario(self):
        fu, eu, board = _simple_game_state("A")
        # Unit 0: 5 Archers at (10,5)..(14,5); make (25, 20) the destination.
        return fu, eu, board

    def _call(self, *, move_type, dest_idx, advance_reachable, shaken=False):
        from ml_training.collection import _maybe_build_post_move_state_vec
        import numpy as np
        fu, eu, board = self._scenario()
        if shaken:
            fu[0].shaken = True
        candidates = np.array([[25, 20], [18, 10]], dtype=np.int32)
        dest_ar = [advance_reachable, True]
        return _maybe_build_post_move_state_vec(
            units_friendly=fu, units_enemy=eu,
            round_num=1, board=board, model_side="A",
            sel_idx=0, move_type=move_type,
            dest_candidates=candidates,
            dest_selected_idx=dest_idx,
            dest_advance_reachable=dest_ar,
            fr_matchups=None, fm_matchups=None,
            er_matchups=None, em_matchups=None,
            pts_friendly=None, pts_enemy=None,
        )

    def test_flag_off_returns_none(self):
        assert is_phase_reencode_enabled() is False
        from ml_model_tactical import MOVE_MOVE
        out = self._call(move_type=MOVE_MOVE, dest_idx=0, advance_reachable=True)
        assert out is None

    def test_flag_on_move_returns_array(self):
        from ml_model_tactical import MOVE_MOVE
        set_phase_reencode_enabled(True)
        out = self._call(move_type=MOVE_MOVE, dest_idx=0, advance_reachable=True)
        assert out is not None
        assert out.dtype == np.float32
        assert out.shape == (TACTICAL_TOTAL_FEATURES,)

    def test_flag_on_charge_returns_none(self):
        from ml_model_tactical import MOVE_CHARGE
        set_phase_reencode_enabled(True)
        out = self._call(move_type=MOVE_CHARGE, dest_idx=0, advance_reachable=True)
        assert out is None

    def test_flag_on_shaken_returns_none(self):
        from ml_model_tactical import MOVE_MOVE
        set_phase_reencode_enabled(True)
        out = self._call(move_type=MOVE_MOVE, dest_idx=0, advance_reachable=True, shaken=True)
        assert out is None

    def test_flag_on_invalid_dest_returns_none(self):
        from ml_model_tactical import MOVE_MOVE
        set_phase_reencode_enabled(True)
        out = self._call(move_type=MOVE_MOVE, dest_idx=-1, advance_reachable=True)
        assert out is None

    def test_post_move_differs_from_pre_move(self):
        """Post-move state_vec must differ from the pre-move state_vec when
        the unit actually moves — otherwise the POST_DEST re-encode is pointless."""
        from ml_model_tactical import MOVE_MOVE
        set_phase_reencode_enabled(True)
        out = self._call(move_type=MOVE_MOVE, dest_idx=0, advance_reachable=True)
        fu, eu, board = self._scenario()
        pre_vec = encode_state_tactical(fu, eu, 1, board, "A")
        assert not np.array_equal(out, pre_vec.numpy())

    def test_rush_flag_inferred_from_advance_reachable(self):
        """When advance_reachable=False, the helper should flag the move as
        rush → projected unit has fatigued=True, which shows up in the
        fatigued index of the state_vec. Verify via direct delta."""
        from ml_model_tactical import MOVE_MOVE
        from ml_features import _TOFF_FATIGUED, TACTICAL_UNIT_FEATURES
        set_phase_reencode_enabled(True)
        out_advance = self._call(move_type=MOVE_MOVE, dest_idx=0, advance_reachable=True)
        out_rush = self._call(move_type=MOVE_MOVE, dest_idx=0, advance_reachable=False)
        fatigued_idx = 0 * TACTICAL_UNIT_FEATURES + _TOFF_FATIGUED
        assert out_advance[fatigued_idx] == 0.0
        assert out_rush[fatigued_idx] == 1.0


# ===========================================================================
# 16. PHASE-REENCODE REPLAY (Step 5)
# ===========================================================================

def _make_minimal_charge_record(seed: int = 0, state_vec: np.ndarray | None = None):
    """Build a minimal MOVE_CHARGE TacticalActivationRecord for replay testing.

    Charge activations have no destination, so prepare_replay_data's dest
    feature recomputation path isn't exercised. state_vec_post is left None —
    the prepare step will fall back to state_vec for the POST_DEST encode.
    """
    from ml_training.config import TacticalActivationRecord as _TAR
    from ml_model_tactical import MOVE_CHARGE as _MOVE_CHARGE
    rng = np.random.default_rng(seed)
    if state_vec is None:
        state_vec = rng.standard_normal(TACTICAL_TOTAL_FEATURES).astype(np.float32)
    return _TAR(
        state_vec=state_vec,
        alive_mask=[True] + [False] * 9,
        enemy_alive_mask=[True] + [False] * 9,
        unit_idx=0,
        move_type=_MOVE_CHARGE,
        dest_candidates=np.zeros((0, 2), dtype=np.int32),
        post_move_rel=np.zeros(30, dtype=np.float32),
        old_log_prob=0.0, old_value=0.0,
        reward=0.0,
        charge_target_idx=0,
        shoot_target_idx=0,
        shoot_mask=[False] * 10,
    )


class TestPhaseReencodeReplay:
    """replay_from_prepared with the phase-reencode flag toggled.

    Ablation gate: flag=True with continuation iters=0 + FiLM identity + zero
    is_acting embedding + state_vec_post == state_vec (charge activations) must
    reproduce the legacy replay's total log_probs to float tolerance.
    """

    def teardown_method(self, _method):
        set_phase_reencode_enabled(False)

    def test_prepared_state_vec_post_none_when_flag_off(self):
        from ml_training.loss import prepare_replay_data
        traj = [_make_minimal_charge_record()]
        prepared = prepare_replay_data([traj], device=torch.device('cpu'))
        assert prepared.state_vec_post_batch is None

    def test_prepared_state_vec_post_populated_when_flag_on(self):
        from ml_training.loss import prepare_replay_data
        set_phase_reencode_enabled(True)
        traj = [_make_minimal_charge_record()]
        prepared = prepare_replay_data([traj], device=torch.device('cpu'))
        assert prepared.state_vec_post_batch is not None
        assert prepared.state_vec_post_batch.shape == prepared.state_batch.shape
        # Charge activation → fallback to state_vec.
        assert torch.equal(prepared.state_vec_post_batch, prepared.state_batch)

    def test_flag_off_replay_has_no_per_phase_values(self):
        from ml_training.loss import prepare_replay_data, replay_from_prepared
        torch.manual_seed(0)
        model = TacticalModel()
        model.eval()
        traj = [_make_minimal_charge_record()]
        prepared = prepare_replay_data([traj], device=torch.device('cpu'))
        idx = torch.arange(prepared.n_steps)
        with torch.no_grad():
            out = replay_from_prepared(model, prepared, idx, n_episodes=1)
        assert out.per_phase_values is None

    def test_flag_on_replay_populates_per_phase_values(self):
        from ml_training.loss import prepare_replay_data, replay_from_prepared
        torch.manual_seed(1)
        model = TacticalModel()
        model.eval()
        set_phase_reencode_enabled(True)
        traj = [_make_minimal_charge_record()]
        prepared = prepare_replay_data([traj], device=torch.device('cpu'))
        idx = torch.arange(prepared.n_steps)
        with torch.no_grad():
            out = replay_from_prepared(model, prepared, idx, n_episodes=1)
        assert out.per_phase_values is not None
        assert out.per_phase_values.shape == (1, N_PHASES)
        # Per-phase V heads are zero-init → outputs are exactly 0 at fresh model.
        assert torch.all(out.per_phase_values == 0.0)

    def test_ablation_gate_replay_log_probs_match_legacy(self):
        """Flag=True with continuation iters 0 + identity FiLM + zero is_acting
        + charge activation (state_vec_post == state_vec) must match flag=False
        log_probs to float tolerance."""
        from ml_training.loss import prepare_replay_data, replay_from_prepared
        torch.manual_seed(2)
        model = TacticalModel()
        model.eval()

        traj = [_make_minimal_charge_record(seed=42)]

        # Flag off (legacy)
        prepared_legacy = prepare_replay_data([traj], device=torch.device('cpu'))
        idx = torch.arange(prepared_legacy.n_steps)
        with torch.no_grad():
            out_legacy = replay_from_prepared(
                model, prepared_legacy, idx, n_episodes=1,
            )
        legacy_lp = out_legacy.log_probs.clone()
        legacy_values = out_legacy.values.clone()

        # Flag on with ablation (continuation iters = 0 so h passes through)
        original_iters = dict(_mmt.DEFAULT_PHASE_ITERS)
        try:
            _mmt.DEFAULT_PHASE_ITERS[_mmt.PHASE_POST_SELECT] = 0
            _mmt.DEFAULT_PHASE_ITERS[_mmt.PHASE_POST_MOVETYPE] = 0
            _mmt.DEFAULT_PHASE_ITERS[_mmt.PHASE_POST_DEST] = 0
            set_phase_reencode_enabled(True)
            prepared_phased = prepare_replay_data([traj], device=torch.device('cpu'))
            with torch.no_grad():
                out_phased = replay_from_prepared(
                    model, prepared_phased, idx, n_episodes=1,
                )
        finally:
            _mmt.DEFAULT_PHASE_ITERS.clear()
            _mmt.DEFAULT_PHASE_ITERS.update(original_iters)
            set_phase_reencode_enabled(False)

        assert torch.allclose(legacy_lp, out_phased.log_probs, atol=1e-5, rtol=1e-5)
        assert torch.allclose(legacy_values, out_phased.values, atol=1e-5, rtol=1e-5)

    def test_compute_loss_flat_reports_per_phase_value_loss(self):
        from ml_training.loss import (
            prepare_replay_data, replay_from_prepared, compute_loss_flat,
        )
        torch.manual_seed(3)
        model = TacticalModel()
        model.eval()
        set_phase_reencode_enabled(True)
        traj = [_make_minimal_charge_record(seed=7)]
        prepared = prepare_replay_data([traj], device=torch.device('cpu'))
        idx = torch.arange(prepared.n_steps)
        with torch.no_grad():
            out = replay_from_prepared(model, prepared, idx, n_episodes=1)
        old_lp = torch.zeros(prepared.n_steps)
        advantages = torch.zeros(prepared.n_steps)
        returns_ = torch.zeros(prepared.n_steps)
        _loss, metrics = compute_loss_flat(
            out, old_lp, advantages, returns_,
            clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.0,
            aux_coeff=0.0,
        )
        # Per-phase V heads zero-init ⇒ outputs exactly 0 ⇒ MSE of (0 - 0) = 0.
        assert 'per_phase_value_loss' in metrics
        assert metrics['per_phase_value_loss'] == 0.0

    def test_flag_on_replay_nonzero_per_phase_loss_after_perturbation(self):
        """Wire check: perturb a per-phase V head so its output is non-zero,
        and confirm that per_phase_value_loss reflects it in compute_loss_flat."""
        from ml_training.loss import (
            prepare_replay_data, replay_from_prepared, compute_loss_flat,
        )
        torch.manual_seed(4)
        model = TacticalModel()
        model.eval()
        # Force per-phase V head 0 (PRE_SELECT) to predict 1.0 regardless of h.
        with torch.no_grad():
            model.per_phase_value_heads[0].bias.fill_(1.0)
        set_phase_reencode_enabled(True)
        traj = [_make_minimal_charge_record(seed=11)]
        prepared = prepare_replay_data([traj], device=torch.device('cpu'))
        idx = torch.arange(prepared.n_steps)
        with torch.no_grad():
            out = replay_from_prepared(model, prepared, idx, n_episodes=1)
        old_lp = torch.zeros(prepared.n_steps)
        advantages = torch.zeros(prepared.n_steps)
        returns_ = torch.zeros(prepared.n_steps)
        _loss, metrics = compute_loss_flat(
            out, old_lp, advantages, returns_,
            clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.0,
            aux_coeff=0.0,
        )
        # PRE_SELECT predicts 1.0, others predict 0.0, targets are 0.0.
        # MSE over 4 phases = (1^2 + 0 + 0 + 0) / 4 = 0.25.
        assert abs(metrics['per_phase_value_loss'] - 0.25) < 1e-6


# ===========================================================================
# 17. FULL-STACK GRADIENT-EQUIVALENCE ABLATION GATE (Step 6)
# ===========================================================================

class TestPhaseReencodeGradientEquivalence:
    """Full-stack ablation: after ONE backward pass through compute_loss_flat,
    the flag-on+ablation-settings path must produce gradients on all *shared*
    parameters that match the legacy flag-off gradients bit-for-bit (or within
    float-arithmetic tolerance).

    The new modules (phase_adjustment_blocks, is_acting_embed,
    per_phase_value_heads) are expected to move independently — they didn't
    exist in the legacy path. The gate proves that flipping the flag on under
    ablation settings does not perturb the gradient flow through any
    previously-trained parameter, so resuming training from an existing
    checkpoint is safe.

    Multi-step divergence is by-design: once the new modules drift from
    zero-init, subsequent steps will no longer match legacy. So this test is
    intentionally single-step.
    """

    # Parameter name prefixes that belong exclusively to the new machinery
    # (added in Steps 2 & 5). All other parameters must have matching grads.
    _NEW_PARAM_PREFIXES = ("phase_adjustment_blocks", "is_acting_embed", "per_phase_value_heads")

    def teardown_method(self, _method):
        set_phase_reencode_enabled(False)

    def _is_new_param(self, name: str) -> bool:
        return any(name.startswith(p) for p in self._NEW_PARAM_PREFIXES)

    def _make_trajectory(self, seed: int):
        """A small charge-only trajectory whose replay exercises every head
        chain but keeps state_vec_post == state_vec so the ablation gate holds.

        All 10 enemy slots are alive — this matters because the charge head
        masks logits at dead enemies with -inf, which turn into gradient-free
        -50 after nan_to_num. A single-alive-enemy record would collapse the
        charge softmax to a point mass with near-zero gradient, preventing
        the POST_MOVETYPE / POST_DEST adjustment blocks from receiving signal
        via the charge path.
        """
        import numpy as np
        from ml_training.config import TacticalActivationRecord as _TAR
        from ml_model_tactical import MOVE_CHARGE as _MOVE_CHARGE
        rng = np.random.default_rng(seed)
        records = []
        for i in range(3):
            state_vec = rng.standard_normal(TACTICAL_TOTAL_FEATURES).astype(np.float32)
            records.append(_TAR(
                state_vec=state_vec,
                alive_mask=[True] * 10,
                enemy_alive_mask=[True] * 10,  # all enemies alive ⇒ charge gradient flows
                unit_idx=0,
                move_type=_MOVE_CHARGE,
                dest_candidates=np.zeros((0, 2), dtype=np.int32),
                post_move_rel=np.zeros(30, dtype=np.float32),
                old_log_prob=0.0, old_value=0.0,
                reward=0.0,
                charge_target_idx=3,
                shoot_target_idx=0,
                shoot_mask=[False] * 10,
            ))
        return records

    def _run_backward(self, *, flag_on: bool, ablation: bool = True):
        """Run one forward + backward pass, return {param_name: grad_clone} dict.

        ``flag_on=True, ablation=True`` ⇒ phased encode with continuation
        iters forced to 0 so h passes through unchanged; output should match
        ``flag_on=False`` bit-for-bit on shared parameters.
        ``flag_on=True, ablation=False`` ⇒ default continuation iters — used
        to confirm the new modules actually carry gradient.
        """
        from ml_training.loss import (
            prepare_replay_data, replay_from_prepared, compute_loss_flat,
        )
        torch.manual_seed(1234)
        model = TacticalModel()
        model.train()

        if flag_on:
            if ablation:
                _mmt.DEFAULT_PHASE_ITERS[_mmt.PHASE_POST_SELECT] = 0
                _mmt.DEFAULT_PHASE_ITERS[_mmt.PHASE_POST_MOVETYPE] = 0
                _mmt.DEFAULT_PHASE_ITERS[_mmt.PHASE_POST_DEST] = 0
            set_phase_reencode_enabled(True)

        traj = self._make_trajectory(seed=7)
        prepared = prepare_replay_data([traj], device=torch.device('cpu'))
        idx = torch.arange(prepared.n_steps)
        result = replay_from_prepared(model, prepared, idx, n_episodes=1)
        # Nonzero returns so per-phase V heads actually get gradient and
        # policy/value losses don't all degenerate to zero.
        returns_ = torch.full((prepared.n_steps,), 0.5)
        advantages = torch.full((prepared.n_steps,), 0.3)
        old_lp = torch.zeros(prepared.n_steps)
        loss, _metrics = compute_loss_flat(
            result, old_lp, advantages, returns_,
            clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01,
            aux_coeff=0.0,
        )
        model.zero_grad()
        loss.backward()
        grads = {n: p.grad.clone() if p.grad is not None else None
                 for n, p in model.named_parameters()}
        return grads, loss.item()

    def test_shared_param_grads_match_legacy(self):
        original_iters = dict(_mmt.DEFAULT_PHASE_ITERS)
        grads_off, loss_off = self._run_backward(flag_on=False)
        try:
            grads_on, loss_on = self._run_backward(flag_on=True)
        finally:
            _mmt.DEFAULT_PHASE_ITERS.clear()
            _mmt.DEFAULT_PHASE_ITERS.update(original_iters)
            set_phase_reencode_enabled(False)

        # All parameters present in both snapshots.
        assert set(grads_off.keys()) == set(grads_on.keys())

        shared_mismatches: list[tuple[str, float]] = []
        for name, g_off in grads_off.items():
            if self._is_new_param(name):
                continue  # These are phase-reencode-only; no legacy counterpart.
            g_on = grads_on[name]
            if g_off is None and g_on is None:
                continue
            assert g_off is not None and g_on is not None, (
                f"param {name}: one side has no grad, other does"
            )
            if not torch.allclose(g_off, g_on, atol=1e-5, rtol=1e-5):
                diff = (g_off - g_on).abs().max().item()
                shared_mismatches.append((name, diff))
        assert not shared_mismatches, (
            f"shared parameter grad mismatch under ablation: {shared_mismatches}"
        )

    def test_new_params_get_gradient_when_wired(self):
        """Confirm the new modules actually receive gradient under *full*
        phase-reencode settings (not ablation) — if any of them is zero, the
        new code-path isn't being exercised end-to-end.

        phase_adjustment_blocks[p]: final Linear is zero-init, so it gets
          gradient from the phase's encode loop and will drift away from zero.
        is_acting_embed.weight[1]: gradient flows because it's added to the
          acting unit's post-encoder row and that propagates through the stem
          and the core-block iterations of the continuation phases.
        per_phase_value_heads.weight: gradient flows from per_phase_value_loss.

        Note: under ablation (continuation iters = 0) both the adjustment
        blocks and is_acting_embed get no gradient because the continuation
        loop is skipped. That's by design — the ablation gate's whole point is
        to degenerate to the legacy trunk.
        """
        grads_on, _ = self._run_backward(flag_on=True, ablation=False)

        # phase_adjustment_blocks[p].final_Linear: zero-init and in the graph
        # at full continuation iters — must receive gradient so it starts to
        # move away from zero over training.
        for p in range(N_PHASES - 1):
            final_weight_grad = grads_on[f'phase_adjustment_blocks.{p}.3.weight']
            assert final_weight_grad is not None
            assert final_weight_grad.abs().sum().item() > 0, (
                f"phase_adjustment_blocks[{p}] final Linear grad is zero — "
                "adjustment block not in gradient graph"
            )

        # is_acting_embed.weight[1] (row 1 = acting): receives gradient via the
        # acting unit's post-encoder row in POST_SELECT/POST_MOVETYPE/POST_DEST.
        is_acting_grad = grads_on['is_acting_embed.weight']
        assert is_acting_grad is not None
        assert is_acting_grad[1].abs().sum().item() > 0, (
            "is_acting_embed.weight[1] grad is zero — additive injection not wired"
        )

        # per_phase_value_heads[p].weight: receives gradient from
        # per_phase_value_loss whenever returns ≠ 0.
        for p in range(N_PHASES):
            w_grad = grads_on[f'per_phase_value_heads.{p}.weight']
            assert w_grad is not None
            assert w_grad.abs().sum().item() > 0, (
                f"per_phase_value_heads[{p}].weight grad is zero — loss not attached"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
