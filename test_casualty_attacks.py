"""Test: verify that resolve_shooting respects models_alive after casualties.

Creates a unit, kills some models via apply_wounds, and checks that
the returned total_attacks matches the reduced model count.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from models import UnitState, ResolvedUnit, Weapon

NUM_TRIALS = 500


def _make_unit(models: int, weapon_attacks: int = 2,
               weapon_range: int = 24, quality: int = 3,
               defense: int = 4, tough: int = 0) -> ResolvedUnit:
    """Build a simple ranged unit."""
    w = Weapon(name="Test Gun", range_inches=weapon_range, attacks=weapon_attacks,
               melee=False, ap=0, blast=0, deadly=0, crack=False, rending=False,
               reliable=False, takedown=False, unstoppable=False, bane=False)
    weapons = [w] * models
    wpm = [[w] for _ in range(models)]
    return ResolvedUnit(
        template_id="test", name="Test Unit", models=models,
        quality=quality, defense=defense, tough=tough,
        weapons=weapons, weapons_per_model=wpm, points=100,
    )


def _make_state(ru: ResolvedUnit, distance: int = 5) -> UnitState:
    """Create a UnitState with models placed in range of (50, 50)."""
    us = UnitState(unit=ru)
    us.positions = [(50 - distance, 50 + i) for i in range(ru.models)]
    us.weapons_per_model = [list(mw) for mw in ru.weapons_per_model]
    return us


def _make_target() -> UnitState:
    """Single tough target at (50, 50) that won't die easily."""
    ru = ResolvedUnit(
        template_id="target", name="Target", models=1,
        quality=5, defense=2, tough=20,
        weapons=[Weapon(name="Fist", range_inches=0, attacks=1, melee=True,
                        ap=0, blast=0, deadly=0, crack=False, rending=False,
                        reliable=False, takedown=False, unstoppable=False,
                        bane=False)],
        weapons_per_model=[[Weapon(name="Fist", range_inches=0, attacks=1,
                                   melee=True, ap=0, blast=0, deadly=0,
                                   crack=False, rending=False, reliable=False,
                                   takedown=False, unstoppable=False,
                                   bane=False)]],
        points=500,
    )
    us = UnitState(unit=ru)
    us.positions = [(50, 50)]
    us.weapons_per_model = [list(mw) for mw in ru.weapons_per_model]
    us.wounds_per_model = [0]
    return us


def test_full_strength():
    """10 models, no casualties -> should always get 10 * A2 = 20 attacks."""
    from combat import resolve_shooting

    ru = _make_unit(models=10, weapon_attacks=2)
    failures = 0
    for _ in range(NUM_TRIALS):
        attacker = _make_state(ru)
        target = _make_target()
        stats = resolve_shooting(attacker, target, recorded=True)
        if stats is None:
            failures += 1
            continue
        if stats['total_attacks'] != 20:
            failures += 1
            print(f"  FAIL full_strength: expected 20 attacks, got {stats['total_attacks']}")
    print(f"[full_strength] {NUM_TRIALS - failures}/{NUM_TRIALS} passed "
          f"(expected total_attacks=20)")


def test_after_casualties():
    """10 models, kill 4 via apply_wounds -> should get 6 * A2 = 12 attacks."""
    from combat import resolve_shooting

    ru = _make_unit(models=10, weapon_attacks=2)
    failures = 0
    for _ in range(NUM_TRIALS):
        attacker = _make_state(ru)
        # Kill 4 models
        attacker.apply_wounds(4)
        assert attacker.models_alive == 6, \
            f"Expected 6 alive, got {attacker.models_alive}"
        assert len(attacker.weapons_per_model) == 6, \
            f"Expected 6 wpm entries, got {len(attacker.weapons_per_model)}"
        target = _make_target()
        stats = resolve_shooting(attacker, target, recorded=True)
        if stats is None:
            failures += 1
            continue
        if stats['total_attacks'] != 12:
            failures += 1
            print(f"  FAIL after_casualties: expected 12, got {stats['total_attacks']}  "
                  f"models_alive={attacker.models_alive}  "
                  f"len(wpm)={len(attacker.weapons_per_model)}")
    print(f"[after_casualties] {NUM_TRIALS - failures}/{NUM_TRIALS} passed "
          f"(expected total_attacks=12)")


def test_after_heavy_casualties():
    """10 models, kill 8 -> should get 2 * A2 = 4 attacks."""
    from combat import resolve_shooting

    ru = _make_unit(models=10, weapon_attacks=2)
    failures = 0
    for _ in range(NUM_TRIALS):
        attacker = _make_state(ru)
        attacker.apply_wounds(8)
        assert attacker.models_alive == 2
        assert len(attacker.weapons_per_model) == 2
        target = _make_target()
        stats = resolve_shooting(attacker, target, recorded=True)
        if stats is None:
            failures += 1
            continue
        if stats['total_attacks'] != 4:
            failures += 1
            print(f"  FAIL heavy_casualties: expected 4, got {stats['total_attacks']}")
    print(f"[heavy_casualties] {NUM_TRIALS - failures}/{NUM_TRIALS} passed "
          f"(expected total_attacks=4)")


def test_weapon_summary_matches_attacks():
    """Verify the weapon summary count * attacks == total_attacks."""
    from combat import resolve_shooting

    ru = _make_unit(models=10, weapon_attacks=2)
    failures = 0
    for kills in range(0, 10):
        for _ in range(50):
            attacker = _make_state(ru)
            if kills > 0:
                attacker.apply_wounds(kills)
            target = _make_target()
            stats = resolve_shooting(attacker, target, recorded=True)
            if stats is None:
                continue
            # Sum attacks from weapon summary
            summary_attacks = sum(
                w['count'] * w['attacks'] for w in stats['attacker_weapons']
            )
            if summary_attacks != stats['total_attacks']:
                failures += 1
                print(f"  FAIL summary_mismatch: kills={kills} "
                      f"summary_attacks={summary_attacks} "
                      f"total_attacks={stats['total_attacks']} "
                      f"models_alive={attacker.models_alive}")
    print(f"[weapon_summary_matches] {500 - failures}/500 passed")


def test_snapshot_restore_then_shoot():
    """Snapshot, kill models, restore, kill different amount, shoot — verify attacks."""
    from combat import resolve_shooting
    from ml_planning import snapshot_game_state, restore_game_state
    from board import Board

    ru = _make_unit(models=10, weapon_attacks=2)
    board = Board()
    failures = 0

    for _ in range(NUM_TRIALS):
        attacker = _make_state(ru)
        target = _make_target()
        units_a = [attacker]
        units_b = [target]

        # Snapshot at full strength
        snap = snapshot_game_state(units_a, units_b, board)

        # Kill 4 models
        attacker.apply_wounds(4)
        assert attacker.models_alive == 6

        # Restore to full strength
        restore_game_state(snap, units_a, units_b, board)
        assert attacker.models_alive == 10, \
            f"After restore: expected 10, got {attacker.models_alive}"
        assert len(attacker.weapons_per_model) == 10, \
            f"After restore: expected 10 wpm, got {len(attacker.weapons_per_model)}"

        # Now kill 7 and shoot
        attacker.apply_wounds(7)
        assert attacker.models_alive == 3

        # Re-place surviving models in range
        attacker.positions = [(45, 50 + i) for i in range(3)]
        target_fresh = _make_target()
        units_b[0] = target_fresh

        stats = resolve_shooting(attacker, target_fresh, recorded=True)
        if stats is None:
            continue
        if stats['total_attacks'] != 6:  # 3 * A2
            failures += 1
            print(f"  FAIL snapshot_restore: expected 6, got {stats['total_attacks']} "
                  f"models_alive={attacker.models_alive} "
                  f"len(wpm)={len(attacker.weapons_per_model)}")

    print(f"[snapshot_restore] {NUM_TRIALS - failures}/{NUM_TRIALS} passed "
          f"(expected total_attacks=6 after restore+kill)")


if __name__ == "__main__":
    test_full_strength()
    test_after_casualties()
    test_after_heavy_casualties()
    test_weapon_summary_matches_attacks()
    test_snapshot_restore_then_shoot()
