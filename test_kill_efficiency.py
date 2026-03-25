"""Test: which units are most points-efficient at killing a shielded Great Elemental.

Target: Great Elemental with great_spear_shield upgrade (380pts)
  Q3+, D2+, Tough(12), Fearless, Fear(2), Fortified
  Weapons: Great Spear (6A AP4 melee), Stomp (4A melee), 1 Shardgun

For each unit template (with various upgrades), we simulate:
  - Ranged: all models at close range (12"), fire once
  - Melee: all models in melee range (2"), charge once
  - Combined: shoot then charge
We measure average wounds dealt per 100 points spent.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models import ArmyListEntry, UnitState, ResolvedUnit, resolve_entry, Weapon
from templates import get_templates_dict
from combat import resolve_shooting, resolve_melee, resolve_impact

NUM_TRIALS = 5000


def make_target() -> tuple[ArmyListEntry, ResolvedUnit]:
    """Create the shielded Great Elemental target."""
    entry = ArmyListEntry(
        template_id="great_elemental",
        chosen_upgrades={"melee_swap": "great_spear_shield"},
    )
    resolved = resolve_entry(entry)
    return entry, resolved


def make_target_state(resolved: ResolvedUnit) -> UnitState:
    """Create a fresh UnitState for the target, placed at (10,10)."""
    state = UnitState(unit=resolved)
    state.positions = [(10, 10)]
    state.weapons_per_model = [list(resolved.weapons)]
    return state


def make_attacker_state(resolved: ResolvedUnit, distance: int = 2) -> UnitState:
    """Create a fresh UnitState for attacker, placed `distance` squares from target."""
    state = UnitState(unit=resolved)
    # Place models in a line starting at (10 - distance, 10)
    positions = []
    for i in range(resolved.models):
        col = 10 - distance
        row = 10 - (resolved.models // 2) + i
        positions.append((col, row))
    state.positions = positions
    state.weapons_per_model = [list(mw) for mw in resolved.weapons_per_model]
    return state


def test_unit(template_id: str, upgrades: dict[str, str],
              label: str) -> dict:
    """Run NUM_TRIALS of shooting + melee against the target. Return stats."""
    entry = ArmyListEntry(template_id=template_id, chosen_upgrades=upgrades)
    resolved = resolve_entry(entry)
    cost = entry.computed_cost

    _, target_resolved = make_target()

    ranged_wounds_total = 0
    melee_wounds_total = 0
    combined_wounds_total = 0

    for _ in range(NUM_TRIALS):
        # --- Ranged only (at 10" = within 12" range) ---
        t_state = make_target_state(target_resolved)
        a_state = make_attacker_state(resolved, distance=10)
        before = sum(t_state.wounds_per_model) if t_state.wounds_per_model else (t_state.unit.models - t_state.models_alive)
        resolve_shooting(a_state, t_state)
        after = sum(t_state.wounds_per_model) if t_state.wounds_per_model else (t_state.unit.models - t_state.models_alive)
        ranged_wounds_total += (after - before)

        # --- Melee only (adjacent, charging) ---
        t_state = make_target_state(target_resolved)
        a_state = make_attacker_state(resolved, distance=1)
        before = sum(t_state.wounds_per_model) if t_state.wounds_per_model else (t_state.unit.models - t_state.models_alive)
        if resolved.impact:
            resolve_impact(a_state, t_state)
        resolve_melee(a_state, t_state, is_charge=True)
        after = sum(t_state.wounds_per_model) if t_state.wounds_per_model else (t_state.unit.models - t_state.models_alive)
        # If target destroyed, count full tough value
        if t_state.models_alive <= 0:
            after = target_resolved.tough
        melee_wounds_total += (after - before)

        # --- Combined: shoot at 10" then melee at 1" ---
        t_state = make_target_state(target_resolved)
        a_state = make_attacker_state(resolved, distance=10)
        resolve_shooting(a_state, t_state)
        # Move to melee range
        a_state.positions = [(10 - 1, 10 - (resolved.models // 2) + i)
                             for i in range(a_state.models_alive)]
        before_combined = sum(t_state.wounds_per_model) if t_state.wounds_per_model else 0
        if t_state.models_alive <= 0:
            before_combined = target_resolved.tough
        # Only charge if target still alive
        if t_state.models_alive > 0:
            if resolved.impact:
                resolve_impact(a_state, t_state)
            resolve_melee(a_state, t_state, is_charge=True)
        after_combined = sum(t_state.wounds_per_model) if t_state.wounds_per_model else 0
        if t_state.models_alive <= 0:
            after_combined = target_resolved.tough
        combined_wounds_total += after_combined

    avg_ranged = ranged_wounds_total / NUM_TRIALS
    avg_melee = melee_wounds_total / NUM_TRIALS
    avg_combined = combined_wounds_total / NUM_TRIALS

    return {
        'label': label,
        'cost': cost,
        'avg_ranged_wounds': avg_ranged,
        'avg_melee_wounds': avg_melee,
        'avg_combined_wounds': avg_combined,
        'ranged_per_100pts': avg_ranged / cost * 100,
        'melee_per_100pts': avg_melee / cost * 100,
        'combined_per_100pts': avg_combined / cost * 100,
    }


def main():
    td = get_templates_dict()

    # Build test cases: (template_id, upgrades, label)
    test_cases: list[tuple[str, dict, str]] = []

    # --- Infantry ---
    test_cases.append(("protectors", {}, "Protectors (base)"))
    test_cases.append(("protectors", {"gun_platform": "shard_cannon"}, "Protectors + Shard Cannon"))
    test_cases.append(("protectors", {"gun_platform": "missile_launcher"}, "Protectors + Missile Launcher"))
    test_cases.append(("protectors", {"gun_platform": "burst_laser"}, "Protectors + Burst Laser"))
    test_cases.append(("protectors", {"gun_platform": "shatter_cannon"}, "Protectors + Shatter Cannon"))
    test_cases.append(("protectors", {"gun_platform": "laser_cannon"}, "Protectors + Laser Cannon"))

    test_cases.append(("strikers", {}, "Strikers (base)"))
    test_cases.append(("strikers", {"specialist": "fusion_rifle"}, "Strikers + Fusion Rifle"))

    test_cases.append(("acolytes", {}, "Acolytes (base)"))
    test_cases.append(("acolytes", {"melee_swap": "energy_spear"}, "Acolytes + Energy Spear"))

    test_cases.append(("retributors", {}, "Retributors (base)"))
    test_cases.append(("scorchers", {}, "Scorchers (base)"))
    test_cases.append(("snipers", {}, "Snipers (base)"))

    test_cases.append(("gliders", {}, "Gliders (base)"))
    test_cases.append(("shifters", {}, "Shifters (base)"))
    test_cases.append(("revenants", {}, "Revenants (base)"))
    test_cases.append(("stingers", {}, "Stingers (base)"))

    test_cases.append(("vanquishers", {}, "Vanquishers (base)"))
    test_cases.append(("vanquishers",
                        {"rocket_1": "impact_rocket_1", "rocket_2": "impact_rocket_2", "rocket_3": "impact_rocket_3"},
                        "Vanquishers (all Impact Rockets)"))
    test_cases.append(("vanquishers", {"psy_marker": "psy_marker"}, "Vanquishers + Psy-Marker"))
    test_cases.append(("vanquishers",
                        {"rocket_1": "impact_rocket_1", "rocket_2": "impact_rocket_2",
                         "rocket_3": "impact_rocket_3", "psy_marker": "psy_marker"},
                        "Vanquishers (all Impact + Psy)"))

    test_cases.append(("scorchers", {"weapon_swap": "fusion_rifles_all"}, "Scorchers (all Fusion Rifles)"))
    test_cases.append(("scorchers", {"fusion_pike_swap": "fusion_pike"}, "Scorchers + Fusion Pike"))

    # --- Elites ---
    test_cases.append(("elemental_protectors", {}, "Elemental Protectors (base)"))
    test_cases.append(("elemental_strikers", {}, "Elemental Strikers (base)"))
    test_cases.append(("elemental_strikers", {"weapon_swap_all": "dual_energy_swords"}, "Elemental Strikers + Dual Swords"))

    # --- Vehicles ---
    test_cases.append(("ag_apc", {}, "AG APC (base)"))
    test_cases.append(("heavy_jetbike", {}, "Heavy Jetbike (base)"))
    test_cases.append(("combat_walker", {}, "Combat Walker (base)"))
    test_cases.append(("support_artillery", {}, "Support Artillery (base)"))
    test_cases.append(("ag_tank", {}, "AG Tank (base)"))
    test_cases.append(("ag_tank", {"main_weapon_all": "prism_cannon"}, "AG Tank (Prism Cannon)"))
    test_cases.append(("ag_tank", {"main_weapon_all": "spinner_cannon"}, "AG Tank (Spinner Cannon)"))
    test_cases.append(("ag_tank", {"main_weapon_1": "rapid_laser_1"}, "AG Tank + Rapid Laser"))

    # --- Jetbikes ---
    test_cases.append(("jetbike_protectors", {}, "Jetbike Protectors (base)"))
    test_cases.append(("jetbike_strikers", {}, "Jetbike Strikers (base)"))

    # --- Monsters ---
    test_cases.append(("great_elemental", {"melee_swap": "great_spear_shield"},
                        "Great Elemental (Spear+Shield)"))
    test_cases.append(("great_elemental", {"melee_swap": "dual_great_sword"},
                        "Great Elemental (Dual Swords)"))
    test_cases.append(("great_elemental", {"melee_swap": "dual_great_axe"},
                        "Great Elemental (Dual Axes)"))
    test_cases.append(("great_elemental", {"melee_swap": "great_axe_shield"},
                        "Great Elemental (Axe+Shield)"))
    test_cases.append(("great_elemental", {"melee_swap": "great_sword_shield"},
                        "Great Elemental (Sword+Shield)"))

    # Great Elemental with heavy ranged
    test_cases.append(("great_elemental",
                        {"melee_swap": "great_spear_shield", "heavy_ranged": "rapid_shard_cannon_ge"},
                        "GE Spear+Shield + Rapid Shard Cannon"))
    test_cases.append(("great_elemental",
                        {"melee_swap": "great_spear_shield", "heavy_ranged": "rapid_shatter_cannon_ge"},
                        "GE Spear+Shield + Rapid Shatter"))
    test_cases.append(("great_elemental",
                        {"melee_swap": "great_spear_shield", "heavy_ranged": "rapid_laser_cannon_ge"},
                        "GE Spear+Shield + Rapid Laser"))

    test_cases.append(("titan_elemental", {}, "Titan Elemental (base)"))
    test_cases.append(("titan_elemental", {"melee_swap": "titan_axe_shield"},
                        "Titan Elemental (Axe+Shield)"))

    # Check if elemental_avatar exists
    if "elemental_avatar" in td:
        test_cases.append(("elemental_avatar", {}, "Elemental Avatar (base)"))

    # Run tests
    print(f"Testing {len(test_cases)} unit configurations vs Shielded Great Elemental")
    print(f"Target: Great Elemental (great_spear_shield) — Q3+ D2+ T12 Fearless Fortified Fear(2)")
    print(f"Trials per unit: {NUM_TRIALS}")
    print("=" * 110)

    results = []
    for tid, upgrades, label in test_cases:
        try:
            r = test_unit(tid, upgrades, label)
            results.append(r)
            print(f"  {label:45s} {r['cost']:4d}pts | "
                  f"Ranged: {r['avg_ranged_wounds']:5.2f} ({r['ranged_per_100pts']:5.2f}/100pts) | "
                  f"Melee: {r['avg_melee_wounds']:5.2f} ({r['melee_per_100pts']:5.2f}/100pts) | "
                  f"Combined: {r['avg_combined_wounds']:5.2f} ({r['combined_per_100pts']:5.2f}/100pts)")
        except Exception as e:
            print(f"  {label:45s} ERROR: {e}")

    # Rankings
    print("\n" + "=" * 110)
    print("RANKINGS BY COMBINED WOUNDS PER 100 POINTS (shoot + charge)")
    print("=" * 110)
    by_combined = sorted(results, key=lambda r: r['combined_per_100pts'], reverse=True)
    for i, r in enumerate(by_combined, 1):
        print(f"  {i:2d}. {r['label']:45s} {r['cost']:4d}pts | "
              f"{r['avg_combined_wounds']:5.2f} wounds | "
              f"{r['combined_per_100pts']:5.2f} wounds/100pts")

    print("\n" + "=" * 110)
    print("RANKINGS BY RANGED WOUNDS PER 100 POINTS (shooting only)")
    print("=" * 110)
    by_ranged = sorted(results, key=lambda r: r['ranged_per_100pts'], reverse=True)
    for i, r in enumerate(by_ranged[:15], 1):
        print(f"  {i:2d}. {r['label']:45s} {r['cost']:4d}pts | "
              f"{r['avg_ranged_wounds']:5.2f} wounds | "
              f"{r['ranged_per_100pts']:5.2f} wounds/100pts")

    print("\n" + "=" * 110)
    print("RANKINGS BY MELEE WOUNDS PER 100 POINTS (charge only)")
    print("=" * 110)
    by_melee = sorted(results, key=lambda r: r['melee_per_100pts'], reverse=True)
    for i, r in enumerate(by_melee[:15], 1):
        print(f"  {i:2d}. {r['label']:45s} {r['cost']:4d}pts | "
              f"{r['avg_melee_wounds']:5.2f} wounds | "
              f"{r['melee_per_100pts']:5.2f} wounds/100pts")


if __name__ == "__main__":
    main()
