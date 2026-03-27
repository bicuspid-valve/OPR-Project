"""§8.1 Feature extraction tests for ml_features.py."""
from __future__ import annotations

import pytest

from models import ResolvedUnit, UnitState, Weapon
from ml_features import (
    starting_wounds,
    precompute_damage,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
