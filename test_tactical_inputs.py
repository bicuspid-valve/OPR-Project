"""Tests for tactical model input calculations.

Verifies that encode_state_tactical produces correct feature values,
the has_activated flag works properly, alive masks are built correctly,
damage matchups scale appropriately, and player-perspective transforms
(coordinate flipping, objective remapping) are consistent.
"""
from __future__ import annotations

import math
import pytest
import torch

from board import Board, COLS, ROWS, OBJECTIVES
from models import ResolvedUnit, UnitState, Weapon
from ml_features import (
    TACTICAL_TOTAL_FEATURES,
    TACTICAL_UNIT_FEATURES,
    UNIT_FEATURES,
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
    remap_objective,
    ROLES,
    STANCES,
)
from ml_model_tactical import TacticalModel


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
        assert vec.shape[0] == 3611

    def test_dtype_float32(self):
        fu, eu, board = _simple_game_state()
        vec = encode_state_tactical(fu, eu, 1, board, "A")
        assert vec.dtype == torch.float32

    def test_feature_count_constant(self):
        """TACTICAL_TOTAL_FEATURES = 20 * 180 + 11 = 3611."""
        assert TACTICAL_UNIT_FEATURES == 180  # 87 base + 70 ranged + 10 melee + 10 post-adv + 3 bools
        assert TACTICAL_TOTAL_FEATURES == MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES + GLOBAL_FEATURES
        assert TACTICAL_TOTAL_FEATURES == 3611


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

    def test_role_in_valid_set(self):
        """Assigned role should be one of the valid roles."""
        model = TacticalModel()
        r = _make_resolved(models=3, points=80)
        us = _make_unit_state(r, activated=False)
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(30, 40)])
        board = Board()
        _, _, assessment = apply_tactical_model(
            model, [us], [eu], 1, board, "A")
        assert assessment['role'] in ROLES

    def test_stance_in_valid_set(self):
        """Assigned stance should be one of the valid stances."""
        model = TacticalModel()
        r = _make_resolved(models=3, points=80)
        us = _make_unit_state(r, activated=False)
        eu = _make_unit_state(_make_resolved(models=1, points=50),
                              owner="B", positions=[(30, 40)])
        board = Board()
        _, _, assessment = apply_tactical_model(
            model, [us], [eu], 1, board, "A")
        assert assessment['stance'] in STANCES


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
# 12. CONSISTENCY BETWEEN STRATEGIC AND TACTICAL ENCODING
# ===========================================================================

class TestStrategicTacticalConsistency:
    def test_base_features_match(self):
        """The first 37 features per unit should match between strategic and tactical."""
        from ml_features import encode_state
        fu, eu, board = _simple_game_state()
        vec_strat = encode_state(fu, eu, 1, board, "A")
        vec_tact = encode_state_tactical(fu, eu, 1, board, "A")

        # Compare first unit's base features (0-36)
        for i in range(UNIT_FEATURES):
            strat_val = vec_strat[i].item()
            tact_val = vec_tact[i].item()
            assert tact_val == pytest.approx(strat_val, abs=1e-6), \
                f"Feature {i} mismatch: strategic={strat_val}, tactical={tact_val}"

    def test_global_features_match(self):
        """Global features should be identical between strategic and tactical."""
        from ml_features import encode_state, TOTAL_FEATURES
        fu, eu, board = _simple_game_state()
        vec_strat = encode_state(fu, eu, 2, board, "A")
        vec_tact = encode_state_tactical(fu, eu, 2, board, "A")

        strat_globals = vec_strat[MAX_UNITS_PER_SIDE * 2 * UNIT_FEATURES:]
        tact_globals = vec_tact[MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES:]

        assert strat_globals.shape == tact_globals.shape == (GLOBAL_FEATURES,)
        for i in range(GLOBAL_FEATURES):
            assert tact_globals[i].item() == pytest.approx(strat_globals[i].item(), abs=1e-6), \
                f"Global feature {i} mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
