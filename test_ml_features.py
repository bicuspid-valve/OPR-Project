"""§8.1 Feature extraction tests for ml_features.py."""
from __future__ import annotations

import math
import pytest

from board import Board, COLS, ROWS, OBJECTIVES
from models import ResolvedUnit, UnitState, Weapon
from ml_features import (
    TOTAL_FEATURES,
    UNIT_FEATURES,
    BOARD_DIAG,
    starting_wounds,
    precompute_damage,
    encode_state,
    _flip_y,
    _get_model_objectives,
    _survival_fraction,
)


# ---------------------------------------------------------------------------
# Fixture helpers
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
                     positions: list[tuple[int, int]] | None = None) -> UnitState:
    us = UnitState(unit=resolved, owner=owner)
    if positions is not None:
        us.positions = list(positions)
    else:
        us.positions = [(10 + i, 5) for i in range(resolved.models)]
    return us


# ---------------------------------------------------------------------------
# starting_wounds
# ---------------------------------------------------------------------------

class TestStartingWounds:
    def test_non_tough(self):
        unit = _make_resolved(models=5, tough=0)
        assert starting_wounds(unit) == 5

    def test_tough_single(self):
        unit = _make_resolved(models=1, tough=6)
        assert starting_wounds(unit) == 6

    def test_tough_multi(self):
        unit = _make_resolved(models=3, tough=3)
        assert starting_wounds(unit) == 9


# ---------------------------------------------------------------------------
# precompute_damage
# ---------------------------------------------------------------------------

class TestPrecomputeDamage:
    def test_ranged_nonzero(self):
        """A unit with ranged weapons should produce nonzero ranged matchup values."""
        attacker = _make_resolved(models=5, quality=4, weapons=[
            _make_weapon(range_inches=24, attacks=1, ap=1)
            for _ in range(5)
        ])
        defender = _make_resolved(models=5, quality=4, defense=4, points=100)
        r_matchups, m_matchups = precompute_damage([attacker], [defender])
        # r_matchups[0][0] is a list of NUM_RANGE_THRESHOLDS floats
        # 24" weapons → nonzero at thresholds <= 24, zero at 30 and 36
        assert r_matchups[0][0][0] > 0.0   # 6" threshold
        assert r_matchups[0][0][4] > 0.0   # 24" threshold
        assert r_matchups[0][0][5] == 0.0   # 30" threshold — out of range
        assert m_matchups[0][0] == 0.0  # no melee weapons
        # Unused enemy slots should be zero at all thresholds
        assert (r_matchups[0][1] == 0.0).all()

    def test_melee_with_impact(self):
        """A melee unit with impact should have nonzero melee matchup values."""
        attacker = _make_resolved(models=3, quality=4, impact=3, weapons=[
            _make_weapon(name="Sword", melee=True, range_inches=0, attacks=2)
            for _ in range(3)
        ])
        defender = _make_resolved(models=5, quality=4, defense=4, points=100)
        r_matchups, m_matchups = precompute_damage([attacker], [defender])
        # All ranged thresholds should be zero (melee-only unit)
        assert (r_matchups[0][0] == 0.0).all()
        assert m_matchups[0][0] > 0.0

    def test_kill_proportion_capped_at_1(self):
        """Kill proportions should be capped at 1.0."""
        # Massive firepower vs fragile target
        attacker = _make_resolved(models=5, quality=3, weapons=[
            _make_weapon(range_inches=24, attacks=6, ap=3, deadly=3)
            for _ in range(5)
        ])
        defender = _make_resolved(models=1, defense=6, points=50, tough=0)
        r_matchups, _ = precompute_damage([attacker], [defender])
        # Capped at 1.0 at all thresholds within range
        assert r_matchups[0][0][0] == 1.0  # 6" threshold
        assert r_matchups[0][0][4] == 1.0  # 24" threshold

    def test_stealth_reduces_ranged_matchup(self):
        """Stealth on defender should reduce ranged kill proportion."""
        attacker = _make_resolved(models=5, quality=4, weapons=[
            _make_weapon(range_inches=24, attacks=2, ap=1)
            for _ in range(5)
        ])
        normal = _make_resolved(models=5, quality=4, defense=4, points=100)
        stealthy = _make_resolved(models=5, quality=4, defense=4, points=100,
                                  stealth=True)
        r_normal, _ = precompute_damage([attacker], [normal])
        r_stealth, _ = precompute_damage([attacker], [stealthy])
        # Compare at 6" threshold (index 0) — stealth should reduce damage
        assert r_stealth[0][0][0] < r_normal[0][0][0]

    def test_range_thresholds(self):
        """Units with mixed weapon ranges should show different damage at different thresholds."""
        # 3 short-range weapons (12") + 1 long-range weapon (30")
        attacker = _make_resolved(models=4, quality=4, weapons=[
            _make_weapon(range_inches=12, attacks=2, ap=1),
            _make_weapon(range_inches=12, attacks=2, ap=1),
            _make_weapon(range_inches=12, attacks=2, ap=1),
            _make_weapon(range_inches=30, attacks=1, ap=1),
        ])
        defender = _make_resolved(models=5, quality=4, defense=4, points=100)
        r_matchups, _ = precompute_damage([attacker], [defender])
        thresholds = r_matchups[0][0]  # 7 values
        # At 6" and 12": all 4 weapons contribute
        assert thresholds[0] > 0  # 6"
        assert thresholds[2] > 0  # 12"
        # At 18": only the 30" weapon contributes → much less damage
        assert thresholds[3] < thresholds[2]  # 18" < 12"
        # At 30": still the sniper
        assert thresholds[5] > 0  # 30"
        # At 36": nothing in range
        assert thresholds[6] == 0.0  # 36"


# ---------------------------------------------------------------------------
# encode_state — basic shape and value tests
# ---------------------------------------------------------------------------

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


class TestEncodeStateShape:
    def test_output_size(self):
        """Encode a known game state and verify feature count."""
        fu, eu, board = _simple_game_state()
        vec = encode_state(fu, eu, 1, board, "A")
        assert vec.shape == (TOTAL_FEATURES,)
        assert vec.shape[0] == 2151

    def test_dtype_float32(self):
        fu, eu, board = _simple_game_state()
        vec = encode_state(fu, eu, 1, board, "A")
        assert vec.dtype == __import__("torch").float32


class TestDeadUnitsZero:
    def test_dead_unit_features_all_zero(self):
        """Dead units should produce all-zero feature vectors."""
        r = _make_resolved(models=3, points=80, weapons=[
            _make_weapon() for _ in range(3)
        ])
        us = _make_unit_state(r, owner="A")
        # Kill all models
        us.positions.clear()
        us.models_alive = 0
        us.wounds_per_model.clear()

        enemy_r = _make_resolved(models=1, points=50)
        eu = _make_unit_state(enemy_r, owner="B", positions=[(30, 40)])

        board = Board()
        vec = encode_state([us], [eu], 1, board, "A")
        # Slot 0 features (indices 0..36) should all be zero
        slot0 = vec[:UNIT_FEATURES]
        assert (slot0 == 0.0).all()


class TestBoardFlip:
    def test_y_coordinate_flipped_for_player_b(self):
        """Player B's units should have mirrored y-coordinates."""
        fu, eu, board = _simple_game_state("A")
        vec_a = encode_state(fu, eu, 1, board, "A")

        fu_b, eu_b, board_b = _simple_game_state("B")
        vec_b = encode_state(fu_b, eu_b, 1, board_b, "B")

        # Feature index 11 (y position) within first unit slot
        y_idx = 11  # 0-indexed within 37 features
        y_a = vec_a[y_idx].item()
        y_b = vec_b[y_idx].item()

        # Player A units at row 5, Player B units at row 40
        # After flip for B: row 40 → 47-40=7, normalised: 7/48
        # Player A row 5, normalised: 5/48
        # Both should be in the "friendly near top" range
        assert y_a == pytest.approx(5.0 / ROWS)
        assert y_b == pytest.approx(7.0 / ROWS)

    def test_objective_indices_swapped_for_player_b(self):
        """Player B should see swapped objective indices (1↔2, 3↔4)."""
        objs_a = _get_model_objectives("A")
        objs_b = _get_model_objectives("B")

        # Centre should be at same x, flipped y
        assert objs_a[0][0] == objs_b[0][0]  # same x
        assert objs_b[0][1] == pytest.approx(_flip_y(OBJECTIVES[0][1]))

        # My-side for A = A-side objective, my-side for B = B-side flipped
        # A's my-side: (18, 16). B's my-side: B-side (54, 32) flipped → (54, 15)
        assert objs_a[1] == (18.0, 16.0)
        assert objs_b[1] == pytest.approx((54.0, _flip_y(32)))

        # My-home for A = Home-A (36,6). My-home for B = Home-B (36,42) flipped
        assert objs_a[3] == (36.0, 6.0)
        assert objs_b[3] == pytest.approx((36.0, _flip_y(42)))

    def test_objective_control_swapped_for_player_b(self):
        """Objective control should be remapped for Player B."""
        fu, eu, board = _simple_game_state("A")
        # Give A control of A-side (idx 1) and B control of B-side (idx 2)
        board.objective_control[1] = "A"
        board.objective_control[2] = "B"

        vec_a = encode_state(fu, eu, 1, board, "A")
        # Global features start at 20*29 = 580
        # Round one-hot: 4 values, then 5 objective control values
        ctrl_start = 20 * UNIT_FEATURES + 4
        # From A's perspective: idx1 (my-side) = +1, idx2 (enemy-side) = -1
        assert vec_a[ctrl_start + 1].item() == pytest.approx(1.0)
        assert vec_a[ctrl_start + 2].item() == pytest.approx(-1.0)

        # Now encode from B's perspective (swap friendly/enemy units)
        vec_b = encode_state(eu, fu, 1, board, "B")
        # From B's perspective: my-side = B-side (game idx 2, B controls) → +1
        # enemy-side = A-side (game idx 1, A controls) → -1
        assert vec_b[ctrl_start + 1].item() == pytest.approx(1.0)
        assert vec_b[ctrl_start + 2].item() == pytest.approx(-1.0)


class TestDamageScaling:
    def test_damage_scales_with_casualties(self):
        """Matchup damage features should decrease when models are killed."""
        r = _make_resolved(models=5, quality=4, defense=4, points=100,
                           weapons=[_make_weapon(attacks=2, ap=1)
                                    for _ in range(5)])
        e_r = _make_resolved(models=3, quality=4, defense=4, points=80,
                             weapons=[_make_weapon() for _ in range(3)])

        us_full = _make_unit_state(r, owner="A",
                                   positions=[(i, 5) for i in range(5)])
        eu = _make_unit_state(e_r, owner="B",
                              positions=[(i, 40) for i in range(3)])
        board = Board()

        vec_full = encode_state([us_full], [eu], 1, board, "A")

        # Kill 2 models
        us_half = _make_unit_state(r, owner="A",
                                   positions=[(i, 5) for i in range(3)])
        us_half.models_alive = 3
        vec_half = encode_state([us_half], [eu], 1, board, "A")

        # Feature 27 (first ranged matchup at 6" threshold vs enemy slot 0)
        # should scale with survival
        ranged_full = vec_full[27].item()
        ranged_half = vec_half[27].item()
        assert ranged_half < ranged_full
        assert ranged_half == pytest.approx(ranged_full * 3 / 5, rel=1e-5)


class TestRoundOneHot:
    @pytest.mark.parametrize("round_num", [1, 2, 3, 4])
    def test_round_one_hot(self, round_num):
        fu, eu, board = _simple_game_state()
        vec = encode_state(fu, eu, round_num, board, "A")
        oh_start = 20 * UNIT_FEATURES  # 580
        for r in range(4):
            expected = 1.0 if (r + 1) == round_num else 0.0
            assert vec[oh_start + r].item() == pytest.approx(expected)


class TestPointsRemaining:
    def test_full_strength(self):
        fu, eu, board = _simple_game_state()
        vec = encode_state(fu, eu, 1, board, "A")
        # Friendly and enemy points remaining (last 2 values)
        assert vec[-2].item() == pytest.approx(1.0)
        assert vec[-1].item() == pytest.approx(1.0)

    def test_after_casualties(self):
        fu, eu, board = _simple_game_state()
        total_e = sum(u.unit.points for u in eu)
        # Kill all of enemy unit 0
        eu[0].positions.clear()
        eu[0].models_alive = 0
        vec = encode_state(fu, eu, 1, board, "A")
        alive_e = eu[1].unit.points
        assert vec[-1].item() == pytest.approx(alive_e / total_e)


class TestSurvivalFraction:
    def test_full_health(self):
        r = _make_resolved(models=5, tough=0)
        us = _make_unit_state(r)
        assert _survival_fraction(us) == pytest.approx(1.0)

    def test_half_dead_non_tough(self):
        r = _make_resolved(models=4, tough=0)
        us = _make_unit_state(r)
        us.positions = us.positions[:2]
        us.models_alive = 2
        assert _survival_fraction(us) == pytest.approx(0.5)

    def test_tough_wounded(self):
        r = _make_resolved(models=1, tough=6)
        us = _make_unit_state(r)
        us.wounds_per_model = [2]
        assert _survival_fraction(us) == pytest.approx(4 / 6)

    def test_tough_multi_model(self):
        r = _make_resolved(models=3, tough=3)
        us = _make_unit_state(r, positions=[(0, 0), (1, 0), (2, 0)])
        us.wounds_per_model = [1, 0, 2]
        # total remaining = (3-1) + (3-0) + (3-2) = 2+3+1 = 6, starting = 9
        assert _survival_fraction(us) == pytest.approx(6 / 9)


class TestPaddingAndTruncation:
    def test_fewer_than_10_units_zero_padded(self):
        """Unused unit slots should be all zeros."""
        fu, eu, board = _simple_game_state()  # 2 friendly, 2 enemy
        vec = encode_state(fu, eu, 1, board, "A")
        # Friendly slot 2 should be zeros
        slot2 = vec[2 * UNIT_FEATURES: 3 * UNIT_FEATURES]
        assert (slot2 == 0.0).all()
        # Enemy slot 12 should be zeros
        slot12 = vec[12 * UNIT_FEATURES: 13 * UNIT_FEATURES]
        assert (slot12 == 0.0).all()


class TestAbilityFlags:
    def test_flying_flag(self):
        r = _make_resolved(models=1, flying=True, points=50)
        us = _make_unit_state(r, positions=[(10, 5)])
        e_r = _make_resolved(models=1, points=50)
        eu = _make_unit_state(e_r, owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state([us], [eu], 1, board, "A")
        # Feature 5 = flying
        assert vec[5].item() == 1.0

    def test_artillery_speed_zero(self):
        r = _make_resolved(models=1, artillery=True, points=50)
        us = _make_unit_state(r, positions=[(10, 5)])
        e_r = _make_resolved(models=1, points=50)
        eu = _make_unit_state(e_r, owner="B", positions=[(10, 40)])
        board = Board()
        vec = encode_state([us], [eu], 1, board, "A")
        # Feature 2 = speed, should be 0 for artillery
        assert vec[2].item() == 0.0
        # Feature 6 = artillery flag
        assert vec[6].item() == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
