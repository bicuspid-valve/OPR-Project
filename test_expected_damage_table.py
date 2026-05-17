"""TERRAIN_SPEC.md §6 — pre-game expected damage table tests.

Validates table coverage, attacker-class dedup, cover branch correctness,
ignores_cover collapse, and shooter-by-shooter sum reproducing a direct
unit-vs-unit calculation when cover state is uniform.
Run: .venv/bin/python test_expected_damage_table.py
"""
from __future__ import annotations

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models import Weapon, ResolvedUnit, UnitState
from expected_damage_table import (
    build_table, lookup, expected_damage_per_model,
)


def _check(name: str, ok: bool, detail: str = "") -> int:
    if ok:
        print(f"  [OK]   {name}")
        return 0
    print(f"  [FAIL] {name} {detail}")
    return 1


def _make_unit(name: str, q: int, d: int, models: int,
               weapons: list[Weapon],
               ignores_cover: bool = False) -> UnitState:
    u = ResolvedUnit(
        template_id=name, name=name, models=models,
        quality=q, defense=d,
        weapons=weapons * models,
        weapons_per_model=[list(weapons)] * models,
        ignores_cover=ignores_cover,
    )
    return UnitState(
        u, models_alive=models, wounds_per_model=[1] * models,
        positions=[(i, i) for i in range(models)],
        weapons_per_model=[list(weapons)] * models,
        owner='A',
    )


def test_table_built_for_all_pairs() -> int:
    fails = 0
    gun = Weapon("rifle", range_inches=24, attacks=2, ap=0)
    a1 = _make_unit("a1", 4, 4, 3, [gun])
    a2 = _make_unit("a2", 4, 4, 2, [gun])
    b1 = _make_unit("b1", 4, 4, 5, [])
    b2 = _make_unit("b2", 4, 4, 4, [])
    table = build_table([a1, a2], [b1, b2])
    # Expected entries: each attacker model × each opposing unit
    # a1 has 3 models, a2 has 2 models; b1 has 5 models, b2 has 4 models
    # So: a1×{b1,b2} = 3×2=6, a2×{b1,b2} = 2×2=4, b1×{a1,a2}=5×2=10, b2×{a1,a2}=4×2=8
    # Total = 6+4+10+8 = 28 entries
    fails += _check("table built for all (model, opposing-unit) pairs",
                    len(table) == 28, f"got {len(table)} entries")
    # Each entry has a tuple of two floats
    for k, v in table.items():
        if not (isinstance(v, tuple) and len(v) == 2):
            fails += _check(f"entry {k} is (no_cover, cover) tuple",
                            False, f"got {v!r}")
            break
    return fails


def test_attacker_class_dedup() -> int:
    """Identical models within a unit should hit the same cached numbers."""
    gun = Weapon("rifle", range_inches=24, attacks=2, ap=0)
    a = _make_unit("a", 4, 4, 5, [gun])  # 5 identical models
    b = _make_unit("b", 4, 4, 5, [])
    table = build_table([a], [b])
    v0 = lookup(table, a, 0, b, False)
    v1 = lookup(table, a, 1, b, False)
    v2 = lookup(table, a, 4, b, False)
    return _check("identical attacker models dedup to same value",
                  v0 == v1 == v2, f"got {v0}, {v1}, {v2}")


def test_cover_reduces_expected_damage() -> int:
    gun = Weapon("rifle", range_inches=24, attacks=2, ap=0)
    a = _make_unit("a", 2, 4, 1, [gun])  # quality 2 → always hits
    b = _make_unit("b", 4, 4, 5, [])
    table = build_table([a], [b])
    no_cover = lookup(table, a, 0, b, False)
    with_cover = lookup(table, a, 0, b, True)
    return _check("cover reduces expected damage",
                  with_cover < no_cover,
                  f"no={no_cover} cover={with_cover}")


def test_unit_ignores_cover_collapses_branches() -> int:
    gun = Weapon("rifle", range_inches=24, attacks=2, ap=0)
    a = _make_unit("a", 4, 4, 1, [gun], ignores_cover=True)
    b = _make_unit("b", 4, 4, 5, [])
    table = build_table([a], [b])
    no = lookup(table, a, 0, b, False)
    co = lookup(table, a, 0, b, True)
    return _check("unit ignores_cover collapses cover/no-cover branches",
                  no == co, f"no={no} cover={co}")


def test_weapon_ignores_cover_collapses() -> int:
    gun = Weapon("rifle_ic", range_inches=24, attacks=2, ap=0,
                 ignores_cover=True)
    a = _make_unit("a", 4, 4, 1, [gun])
    b = _make_unit("b", 4, 4, 5, [])
    table = build_table([a], [b])
    no = lookup(table, a, 0, b, False)
    co = lookup(table, a, 0, b, True)
    return _check("per-weapon ignores_cover collapses branches",
                  no == co, f"no={no} cover={co}")


def test_shooter_by_shooter_sum() -> int:
    """Sum over shooters reproduces a direct unit-vs-unit calc when all
    shooters see the same (uniform) cover state."""
    fails = 0
    gun = Weapon("rifle", range_inches=24, attacks=2, ap=0)
    a = _make_unit("a", 3, 4, 5, [gun])
    b = _make_unit("b", 4, 4, 5, [])
    table = build_table([a], [b])
    # Sum the per-shooter no-cover values.
    summed_no = sum(lookup(table, a, mi, b, False)
                    for mi in range(a.unit.models))
    # Direct calc: 5 models × per-model expected_damage
    per_model = expected_damage_per_model(a.unit, [gun], b.unit, cover=False)
    direct = 5 * per_model
    # Both are capped at defender total wounds, so cap is the relevant comparison
    cap = b.unit.models * (b.unit.tough or 1)
    expected = min(direct, cap)
    fails += _check("shooter sum equals 5 × per-model damage (capped)",
                    abs(summed_no - expected) < 1e-6,
                    f"summed={summed_no} expected={expected}")
    return fails


if __name__ == "__main__":
    total = 0
    print("\n--- test_table_built_for_all_pairs ---")
    total += test_table_built_for_all_pairs()
    print("\n--- test_attacker_class_dedup ---")
    total += test_attacker_class_dedup()
    print("\n--- test_cover_reduces_expected_damage ---")
    total += test_cover_reduces_expected_damage()
    print("\n--- test_unit_ignores_cover_collapses_branches ---")
    total += test_unit_ignores_cover_collapses_branches()
    print("\n--- test_weapon_ignores_cover_collapses ---")
    total += test_weapon_ignores_cover_collapses()
    print("\n--- test_shooter_by_shooter_sum ---")
    total += test_shooter_by_shooter_sum()
    print(f"\n=== {total} failure(s) ===")
    sys.exit(1 if total else 0)
