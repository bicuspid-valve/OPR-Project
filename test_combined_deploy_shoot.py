"""Targeted test: deploy a combined unit, inflict casualties, then shoot.

Checks whether models_alive, len(weapons_per_model), and len(positions)
stay in sync after deployment + apply_wounds, and whether resolve_shooting
uses the correct (reduced) attack count.
"""
from __future__ import annotations
import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from models import UnitState, ResolvedUnit, Weapon, ArmyListEntry, resolve_entry
from templates import get_templates_dict
from combat import resolve_shooting
from board import Board
from game import deploy_armies


def _find_combined_templates():
    """Return a few combined template IDs for testing."""
    td = get_templates_dict()
    return [tid for tid, tpl in td.items() if tpl.is_combined]


def test_invariants_after_deploy_and_wounds():
    """Test that combined units maintain state invariants through deploy + casualties."""
    combined_ids = _find_combined_templates()
    print(f"Found {len(combined_ids)} combined templates")

    td = get_templates_dict()
    failures = 0
    tests = 0

    for tid in combined_ids[:10]:  # test first 10
        tpl = td[tid]
        entry = ArmyListEntry(template_id=tid, chosen_upgrades={})
        resolved = resolve_entry(entry)

        for trial in range(20):
            tests += 1
            board = Board()

            # Create attacker (combined) and a simple target
            attacker = UnitState(resolved)
            attacker.owner = "A"

            # Simple 1-model target
            target_ru = ResolvedUnit(
                template_id="target", name="Target", models=1,
                quality=5, defense=2, tough=20,
                weapons=[], weapons_per_model=[[]],
                points=500,
            )
            target = UnitState(target_ru)
            target.owner = "B"
            target.wounds_per_model = [0]

            # Deploy
            deploy_armies([attacker], [target], board)

            starting_alive = attacker.models_alive
            starting_wpm = len(attacker.weapons_per_model)
            starting_pos = len(attacker.positions)

            # Verify post-deploy invariants
            if starting_alive != starting_wpm:
                print(f"  FAIL {tid} trial {trial}: post-deploy "
                      f"alive={starting_alive} != wpm={starting_wpm}")
                failures += 1
                continue
            if starting_alive != starting_pos:
                print(f"  FAIL {tid} trial {trial}: post-deploy "
                      f"alive={starting_alive} != positions={starting_pos}")
                failures += 1
                continue

            # Kill some models
            kills = random.randint(1, max(1, starting_alive - 1))
            attacker.apply_wounds(kills)

            expected_alive = starting_alive - kills
            actual_alive = attacker.models_alive
            actual_wpm = len(attacker.weapons_per_model)
            actual_pos = len(attacker.positions)

            if actual_alive != expected_alive:
                print(f"  FAIL {tid} trial {trial}: after {kills} kills "
                      f"alive={actual_alive} expected={expected_alive}")
                failures += 1
                continue
            if actual_wpm != expected_alive:
                print(f"  FAIL {tid} trial {trial}: after {kills} kills "
                      f"wpm={actual_wpm} expected={expected_alive}")
                failures += 1
                continue
            if actual_pos != expected_alive:
                print(f"  FAIL {tid} trial {trial}: after {kills} kills "
                      f"positions={actual_pos} expected={expected_alive}")
                failures += 1
                continue

            # Now shoot and verify attack count
            # Place target in range
            if target.models_alive > 0 and attacker.models_alive > 0:
                stats = resolve_shooting(attacker, target, recorded=True)
                if stats is not None:
                    # Compute expected max attacks
                    expected_attacks = 0
                    for mw in attacker.weapons_per_model:
                        for w in mw:
                            if not w.melee:
                                expected_attacks += w.attacks
                    if stats['total_attacks'] > expected_attacks:
                        print(f"  FAIL {tid} trial {trial}: "
                              f"total_attacks={stats['total_attacks']} > "
                              f"expected_max={expected_attacks} "
                              f"(alive={actual_alive}/{resolved.models})")
                        failures += 1

    print(f"\n[combined_deploy_shoot] {tests - failures}/{tests} passed")
    return failures


def test_game_loop_invariant():
    """Simulate the full game loop pattern: alternating activations with
    one unit shooting another, then the damaged unit shooting back.

    This mimics the exact scenario: B shoots A's combined unit, then
    A's combined unit shoots B.
    """
    from combat import resolve_shooting, check_morale
    from game import deploy_armies, _sync_dead_models

    td = get_templates_dict()
    combined_ids = [tid for tid, tpl in td.items() if tpl.is_combined]

    failures = 0
    tests = 0

    for tid in combined_ids[:10]:
        tpl = td[tid]
        entry = ArmyListEntry(template_id=tid, chosen_upgrades={})
        resolved = resolve_entry(entry)

        for trial in range(50):
            tests += 1
            board = Board()

            # A's combined unit
            unit_a = UnitState(resolved)
            unit_a.owner = "A"

            # B's shooter (simple 5-model unit with A3 guns)
            b_weapon = Weapon(name="Gun", range_inches=24, attacks=3,
                              melee=False, ap=2, blast=0, deadly=0,
                              crack=False, rending=False, reliable=False,
                              takedown=False, unstoppable=False, bane=False)
            b_resolved = ResolvedUnit(
                template_id="shooter", name="Shooter", models=5,
                quality=3, defense=4,
                weapons=[b_weapon] * 5,
                weapons_per_model=[[b_weapon] for _ in range(5)],
                points=200,
            )
            unit_b = UnitState(b_resolved)
            unit_b.owner = "B"

            deploy_armies([unit_a], [unit_b], board)

            a_alive_before_shot = unit_a.models_alive

            # B shoots A (simulating B's activation)
            resolve_shooting(unit_b, unit_a)
            check_morale(unit_a)
            _sync_dead_models(unit_a, board)

            a_alive_after_shot = unit_a.models_alive
            a_wpm_after = len(unit_a.weapons_per_model)
            a_pos_after = len(unit_a.positions)

            if a_alive_after_shot != a_wpm_after:
                print(f"  FAIL {tid} trial {trial}: after B shoots, "
                      f"alive={a_alive_after_shot} wpm={a_wpm_after}")
                failures += 1
                continue
            if a_alive_after_shot != a_pos_after:
                print(f"  FAIL {tid} trial {trial}: after B shoots, "
                      f"alive={a_alive_after_shot} positions={a_pos_after}")
                failures += 1
                continue

            if unit_a.models_alive <= 0 or unit_b.models_alive <= 0:
                continue

            # A shoots B back (simulating A's activation)
            stats = resolve_shooting(unit_a, unit_b, recorded=True)
            if stats is None:
                continue

            # Verify total_attacks is consistent with alive models
            expected_max = 0
            for mw in unit_a.weapons_per_model:
                for w in mw:
                    if not w.melee:
                        expected_max += w.attacks

            if stats['total_attacks'] > expected_max:
                casualties = a_alive_before_shot - a_alive_after_shot
                print(f"  BUG {tid} trial {trial}: "
                      f"alive={a_alive_after_shot}/{resolved.models} "
                      f"({casualties} killed) but "
                      f"total_attacks={stats['total_attacks']} > "
                      f"max={expected_max}")
                failures += 1

    print(f"\n[game_loop_invariant] {tests - failures}/{tests} passed")
    return failures


if __name__ == "__main__":
    f1 = test_invariants_after_deploy_and_wounds()
    print()
    f2 = test_game_loop_invariant()
    if f1 == 0 and f2 == 0:
        print("\nAll tests passed — bug does not reproduce at this level.")
        print("Bug is likely in the activation selection / game loop orchestration.")
