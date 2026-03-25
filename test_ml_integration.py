"""§8.3 Integration tests for ml_integration.py and game.py ML path."""
from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

from board import Board, COLS, ROWS, OBJECTIVES
from models import ResolvedUnit, UnitState, Weapon
from ml_features import MAX_UNITS_PER_SIDE, precompute_damage, encode_state
from ml_model import StrategicModel
from ml_integration import (
    ROLES, STANCES, remap_objective, ml_activation_order, apply_model_outputs,
)
from ai import pick_target, _base_target_score, choose_action_and_goal
from game import simulate_game


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
                     positions: list[tuple[int, int]] | None = None,
                     ai_role: str = "killer",
                     combat_preference: str = "ranged") -> UnitState:
    us = UnitState(unit=resolved, owner=owner)
    us.ai_role = ai_role
    us.combat_preference = combat_preference
    if positions is not None:
        us.positions = list(positions)
    else:
        us.positions = [(10 + i, 5) for i in range(resolved.models)]
    return us


def _build_test_armies():
    """Build two small armies (3 units each) for integration testing."""
    # Player A
    a1 = _make_resolved(name="A_Archers", models=5, quality=4, defense=4,
                        points=100, weapons=[
                            _make_weapon(attacks=1, ap=0) for _ in range(5)
                        ])
    a2 = _make_resolved(name="A_Swords", models=3, quality=4, defense=3,
                        points=80, weapons=[
                            _make_weapon(name="Blade", melee=True,
                                         range_inches=0, attacks=2)
                            for _ in range(3)
                        ])
    a3 = _make_resolved(name="A_Lancers", models=3, quality=3, defense=3,
                        points=120, weapons=[
                            _make_weapon(attacks=2, ap=1) for _ in range(3)
                        ])

    # Player B
    b1 = _make_resolved(name="B_Gunners", models=5, quality=4, defense=4,
                        points=110, weapons=[
                            _make_weapon(attacks=2, ap=1) for _ in range(5)
                        ])
    b2 = _make_resolved(name="B_Tank", models=1, quality=3, defense=2,
                        points=200, tough=6, weapons=[
                            _make_weapon(attacks=6, ap=3, deadly=3),
                        ])
    b3 = _make_resolved(name="B_Scouts", models=3, quality=4, defense=5,
                        points=60, weapons=[
                            _make_weapon(attacks=1, ap=0) for _ in range(3)
                        ])
    return [a1, a2, a3], [b1, b2, b3]


@pytest.fixture
def model():
    torch.manual_seed(42)
    return StrategicModel()


@pytest.fixture
def test_game_state():
    """Build a positioned game state ready for ML integration testing."""
    army_a, army_b = _build_test_armies()

    units_a = []
    for i, ru in enumerate(army_a):
        us = _make_unit_state(ru, owner="A",
                              positions=[(10 + j, 5 + i * 3)
                                         for j in range(ru.models)])
        units_a.append(us)

    units_b = []
    for i, ru in enumerate(army_b):
        us = _make_unit_state(ru, owner="B",
                              positions=[(10 + j, 40 + i * 2)
                                         for j in range(ru.models)])
        units_b.append(us)

    board = Board()
    return units_a, units_b, board


# ---------------------------------------------------------------------------
# 1. Full game with ML model (Player A) vs heuristic (Player B)
# ---------------------------------------------------------------------------

class TestFullGameMLvsHeuristic:
    def test_game_completes(self):
        """Run a full game with ML Player A, heuristic Player B — should finish without error."""
        army_a, army_b = _build_test_armies()
        torch.manual_seed(7)
        model = StrategicModel()
        result = simulate_game(army_a, army_b, mode="objectives",
                               ml_model_a=model)
        assert result in ("A", "B", "draw")

    def test_game_completes_ml_both_sides(self):
        """Both players using ML should complete without error."""
        army_a, army_b = _build_test_armies()
        torch.manual_seed(7)
        model_a = StrategicModel()
        model_b = StrategicModel()
        result = simulate_game(army_a, army_b, mode="objectives",
                               ml_model_a=model_a, ml_model_b=model_b)
        assert result in ("A", "B", "draw")

    def test_game_completes_kill_points_mode(self):
        """ML should also work in kill_points mode."""
        army_a, army_b = _build_test_armies()
        torch.manual_seed(7)
        model = StrategicModel()
        result = simulate_game(army_a, army_b, mode="kill_points",
                               ml_model_a=model)
        assert result in ("A", "B", "draw")


# ---------------------------------------------------------------------------
# 2. Verify reassign_roles() is NOT called for ML-controlled side
# ---------------------------------------------------------------------------

class TestReassignRolesSkipped:
    def test_reassign_roles_not_called_for_ml_player(self):
        """When ML controls Player A, reassign_roles should not be called for A's units."""
        army_a, army_b = _build_test_armies()
        torch.manual_seed(42)
        model = StrategicModel()

        calls = []
        original_reassign = __import__('ai').reassign_roles

        def tracking_reassign(units):
            player = units[0].owner if units else "?"
            calls.append(player)
            original_reassign(units)

        with patch('game.reassign_roles', side_effect=tracking_reassign):
            simulate_game(army_a, army_b, mode="objectives",
                          ml_model_a=model)

        # reassign_roles should only be called for Player B (4 rounds)
        assert "A" not in calls
        # B should have been called (up to 4 times, may be fewer if game ends early)
        assert all(c == "B" for c in calls)
        assert len(calls) > 0

    def test_assign_objectives_not_called_for_ml_player(self):
        """assign_objectives should be skipped for ML-controlled sides."""
        army_a, army_b = _build_test_armies()
        torch.manual_seed(42)
        model = StrategicModel()

        calls = []
        original_assign = __import__('ai').assign_objectives

        def tracking_assign(units):
            player = units[0].owner if units else "?"
            calls.append(player)
            original_assign(units)

        with patch('game.assign_objectives', side_effect=tracking_assign):
            simulate_game(army_a, army_b, mode="objectives",
                          ml_model_a=model)

        assert "A" not in calls
        assert "B" in calls


# ---------------------------------------------------------------------------
# 3. Verify activation order matches model's priority output
# ---------------------------------------------------------------------------

class TestActivationOrder:
    def test_ml_activation_order_matches_scores(self, model, test_game_state):
        """Units should be ordered by activation priority score descending."""
        units_a, units_b, board = test_game_state

        target_mults, _ = apply_model_outputs(
            model, units_a, units_b, 1, board, "A",
        )

        # Get the stored scores
        scores = [(getattr(u, '_ml_activation_score', 0.0), u) for u in units_a
                  if u.models_alive > 0]
        scores.sort(key=lambda x: x[0], reverse=True)
        expected_order = [u for _, u in scores]

        ordered = ml_activation_order(units_a)
        assert ordered == expected_order

    def test_ml_activation_order_skips_activated(self, model, test_game_state):
        """Activated units should be excluded from the order."""
        units_a, units_b, board = test_game_state
        apply_model_outputs(model, units_a, units_b, 1, board, "A")

        units_a[0].activated = True
        ordered = ml_activation_order(units_a)
        assert units_a[0] not in ordered
        assert len(ordered) == len(units_a) - 1


# ---------------------------------------------------------------------------
# 4. Verify roles and objectives written to UnitState match model output
# ---------------------------------------------------------------------------

class TestModelOutputsApplied:
    def test_roles_are_valid(self, model, test_game_state):
        """All living units should have a valid role after apply_model_outputs."""
        units_a, units_b, board = test_game_state
        apply_model_outputs(model, units_a, units_b, 1, board, "A")

        for u in units_a:
            if u.models_alive > 0:
                assert u.ai_role in ROLES

    def test_objectives_are_valid(self, model, test_game_state):
        """All living units should have a valid objective (0–4) after apply."""
        units_a, units_b, board = test_game_state
        apply_model_outputs(model, units_a, units_b, 1, board, "A")

        for u in units_a:
            if u.models_alive > 0:
                assert 0 <= u.assigned_objective <= 4

    def test_combat_preference_valid(self, model, test_game_state):
        units_a, units_b, board = test_game_state
        apply_model_outputs(model, units_a, units_b, 1, board, "A")

        for u in units_a:
            if u.models_alive > 0:
                assert u.combat_preference in ("ranged", "melee")

    def test_movement_stance_valid(self, model, test_game_state):
        units_a, units_b, board = test_game_state
        apply_model_outputs(model, units_a, units_b, 1, board, "A")

        for u in units_a:
            if u.models_alive > 0:
                assert u.movement_stance in STANCES

    def test_dead_units_not_modified(self, model, test_game_state):
        """Dead units should retain their original attributes."""
        units_a, units_b, board = test_game_state
        # Kill unit 0
        units_a[0].positions.clear()
        units_a[0].models_alive = 0
        old_role = units_a[0].ai_role

        apply_model_outputs(model, units_a, units_b, 1, board, "A")
        assert units_a[0].ai_role == old_role

    def test_objective_remap_player_b(self, model, test_game_state):
        """Player B objectives should be remapped (1↔2, 3↔4) from model perspective."""
        units_a, units_b, board = test_game_state
        # Swap sides for B perspective
        apply_model_outputs(model, units_b, units_a, 1, board, "B")

        # Verify the remap function
        assert remap_objective(0, "B") == 0  # Centre stays
        assert remap_objective(1, "B") == 2  # My-side → B-side (game idx 2)
        assert remap_objective(2, "B") == 1  # Enemy-side → A-side (game idx 1)
        assert remap_objective(3, "B") == 4  # My-home → Home-B (game idx 4)
        assert remap_objective(4, "B") == 3  # Enemy-home → Home-A (game idx 3)


# ---------------------------------------------------------------------------
# 5. Verify target priority multipliers are applied during pick_target
# ---------------------------------------------------------------------------

class TestTargetMultipliers:
    def test_multipliers_change_target_selection(self):
        """High multiplier on a specific enemy should shift target selection."""
        attacker = _make_resolved(models=5, quality=4, defense=4, points=100,
                                  weapons=[_make_weapon(attacks=1, ap=0, range_inches=48)
                                           for _ in range(5)])
        # Two enemies: one high-value, one low-value
        enemy_good = _make_resolved(name="Good", models=5, quality=4, defense=4,
                                    points=200, weapons=[_make_weapon() for _ in range(5)])
        enemy_bad = _make_resolved(name="Bad", models=1, quality=4, defense=2,
                                   points=50, tough=3,
                                   weapons=[_make_weapon()])

        att_us = _make_unit_state(attacker, owner="A",
                                  positions=[(i, 5) for i in range(5)])
        eg_us = _make_unit_state(enemy_good, owner="B",
                                 positions=[(i, 20) for i in range(5)])
        eb_us = _make_unit_state(enemy_bad, owner="B",
                                 positions=[(30, 20)])

        enemies = [eg_us, eb_us]

        # Without multipliers: should pick whatever has better score/points ratio
        base_target = pick_target(att_us, enemies)

        # With huge multiplier on enemy_bad (index 1): should override
        mults = [0.01, 20.0] + [1.0] * 8  # suppress enemy 0, boost enemy 1
        boosted_target = pick_target(att_us, enemies, target_multipliers=mults)

        assert boosted_target is eb_us

    def test_multipliers_returned_by_apply(self, model, test_game_state):
        """apply_model_outputs should return a list of 10 multiplier floats."""
        units_a, units_b, board = test_game_state
        mults, assessment = apply_model_outputs(model, units_a, units_b, 1, board, "A")

        assert isinstance(mults, list)
        assert len(mults) == MAX_UNITS_PER_SIDE
        # All multipliers in valid range [exp(-3), exp(3)]
        for m in mults:
            assert math.exp(-3) - 1e-6 <= m <= math.exp(3) + 1e-6


# ---------------------------------------------------------------------------
# 6. Kite stance: ranged killer holds at max range
# ---------------------------------------------------------------------------

class TestKiteStance:
    def test_kite_holds_when_in_full_volley_range(self):
        """A ranged killer with kite stance should hold when target is in full volley range."""
        # Unit with 24" range weapons
        attacker = _make_resolved(models=3, quality=4, defense=4, points=100,
                                  weapons=[_make_weapon(range_inches=24, attacks=2)
                                           for _ in range(3)])
        enemy = _make_resolved(models=3, quality=4, defense=4, points=100,
                               weapons=[_make_weapon() for _ in range(3)])

        # Place attacker 20" from enemy (within 24" range)
        att_us = _make_unit_state(attacker, owner="A",
                                  positions=[(10, 10), (11, 10), (12, 10)],
                                  ai_role="killer", combat_preference="ranged")
        att_us.movement_stance = "kite"

        enemy_us = _make_unit_state(enemy, owner="B",
                                    positions=[(10, 30), (11, 30), (12, 30)])
        board = Board()

        action, goal, charge_target, reason = choose_action_and_goal(
            att_us, [enemy_us], board, mode="objectives")

        assert action == "hold"
        assert "kite" in reason.lower()

    def test_kite_advances_toward_max_range_when_out_of_range(self):
        """A ranged killer with kite stance should advance (not rush) toward max range position."""
        attacker = _make_resolved(models=1, quality=4, defense=4, points=100,
                                  weapons=[_make_weapon(range_inches=24, attacks=2)])
        enemy = _make_resolved(models=1, quality=4, defense=4, points=100,
                               weapons=[_make_weapon()])

        # Place 40" apart (beyond 24" range)
        att_us = _make_unit_state(attacker, owner="A",
                                  positions=[(36, 5)],
                                  ai_role="killer", combat_preference="ranged")
        att_us.movement_stance = "kite"

        enemy_us = _make_unit_state(enemy, owner="B",
                                    positions=[(36, 45)])
        board = Board()

        action, goal, charge_target, reason = choose_action_and_goal(
            att_us, [enemy_us], board, mode="objectives")

        # Should advance, not rush
        assert action == "advance"
        assert "kite" in reason.lower()


# ---------------------------------------------------------------------------
# 7. Aggressive stance: rushes toward target even when in range
# ---------------------------------------------------------------------------

class TestAggressiveStance:
    def test_aggressive_advances_when_in_range(self):
        """A ranged killer with aggressive stance should advance toward target even in full volley range."""
        attacker = _make_resolved(models=3, quality=4, defense=4, points=100,
                                  weapons=[_make_weapon(range_inches=24, attacks=2)
                                           for _ in range(3)])
        enemy = _make_resolved(models=3, quality=4, defense=4, points=100,
                               weapons=[_make_weapon() for _ in range(3)])

        # Place within range (20" apart)
        att_us = _make_unit_state(attacker, owner="A",
                                  positions=[(10, 10), (11, 10), (12, 10)],
                                  ai_role="killer", combat_preference="ranged")
        att_us.movement_stance = "aggressive"

        enemy_us = _make_unit_state(enemy, owner="B",
                                    positions=[(10, 30), (11, 30), (12, 30)])
        board = Board()

        action, goal, charge_target, reason = choose_action_and_goal(
            att_us, [enemy_us], board, mode="objectives")

        assert action == "advance"
        assert "aggressive" in reason.lower()

    def test_aggressive_rushes_when_partially_in_range(self):
        """Aggressive stance should rush toward target when some weapons not in range."""
        # Unit with short-range (12") and long-range (48") weapons
        attacker = _make_resolved(models=1, quality=4, defense=4, points=100,
                                  weapons=[
                                      _make_weapon(range_inches=48, attacks=2),
                                      _make_weapon(name="Short", range_inches=12, attacks=4),
                                  ])
        enemy = _make_resolved(models=1, quality=4, defense=4, points=100,
                               weapons=[_make_weapon()])

        # Place 30" apart — 48" weapon in range but 12" weapon not (not full volley)
        att_us = _make_unit_state(attacker, owner="A",
                                  positions=[(36, 5)],
                                  ai_role="killer", combat_preference="ranged")
        att_us.movement_stance = "aggressive"

        enemy_us = _make_unit_state(enemy, owner="B",
                                    positions=[(36, 35)])
        board = Board()

        action, goal, charge_target, reason = choose_action_and_goal(
            att_us, [enemy_us], board, mode="objectives")

        assert action == "rush"
        assert "aggressive" in reason.lower()


# ---------------------------------------------------------------------------
# 8. Movement stance ignored for objective_holders and melee units
# ---------------------------------------------------------------------------

class TestStanceIgnored:
    def test_stance_ignored_for_objective_holder(self):
        """Objective holders should use holder logic regardless of stance."""
        attacker = _make_resolved(models=3, quality=4, defense=4, points=100,
                                  weapons=[_make_weapon(range_inches=24, attacks=2)
                                           for _ in range(3)])
        enemy = _make_resolved(models=3, quality=4, defense=4, points=100,
                               weapons=[_make_weapon() for _ in range(3)])

        att_us = _make_unit_state(attacker, owner="A",
                                  positions=[(10, 10), (11, 10), (12, 10)],
                                  ai_role="objective_holder",
                                  combat_preference="ranged")
        att_us.movement_stance = "aggressive"
        att_us.assigned_objective = 0  # Centre objective

        enemy_us = _make_unit_state(enemy, owner="B",
                                    positions=[(10, 30), (11, 30), (12, 30)])
        board = Board()

        action, goal, charge_target, reason = choose_action_and_goal(
            att_us, [enemy_us], board, mode="objectives")

        # Should be holder logic (rush/advance to objective), not aggressive killer logic
        assert "aggressive" not in reason.lower()

    def test_stance_ignored_for_melee_preference(self):
        """Melee-preference units should use melee logic regardless of stance."""
        attacker = _make_resolved(models=3, quality=4, defense=4, points=100,
                                  weapons=[
                                      _make_weapon(name="Sword", melee=True,
                                                   range_inches=0, attacks=2)
                                      for _ in range(3)
                                  ])
        enemy = _make_resolved(models=3, quality=4, defense=4, points=100,
                               weapons=[_make_weapon() for _ in range(3)])

        att_us = _make_unit_state(attacker, owner="A",
                                  positions=[(10, 10), (11, 10), (12, 10)],
                                  ai_role="killer",
                                  combat_preference="melee")
        att_us.movement_stance = "kite"

        enemy_us = _make_unit_state(enemy, owner="B",
                                    positions=[(10, 30), (11, 30), (12, 30)])
        board = Board()

        action, goal, charge_target, reason = choose_action_and_goal(
            att_us, [enemy_us], board, mode="objectives")

        # Should not be kiting — melee units go melee path
        assert "kite" not in reason.lower()


# ---------------------------------------------------------------------------
# Objective remap roundtrip
# ---------------------------------------------------------------------------

class TestObjectiveRemap:
    def test_remap_identity_for_player_a(self):
        for i in range(5):
            assert remap_objective(i, "A") == i

    def test_remap_roundtrip_player_b(self):
        """Remapping twice should return to original for Player B."""
        from ml_integration import _OBJ_REMAP_B
        for i in range(5):
            assert _OBJ_REMAP_B[_OBJ_REMAP_B[i]] == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
