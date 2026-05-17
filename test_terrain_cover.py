"""TERRAIN_SPEC.md §6 — per-shooter unit-level cover & ignores_cover tests.

Covers §4.4: strict-majority cover bonus, invisible-to-all blocks target
declaration, ignores_cover (per-weapon and unit-level) suppresses +1.
Run: python3 test_terrain_cover.py
"""
from __future__ import annotations

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import models as mm
import combat as cm
from board import Board, TerrainPiece, CoverType, MovementType
from models import Weapon, ResolvedUnit, UnitState


def _check(name: str, ok: bool, detail: str = "") -> int:
    if ok:
        print(f"  [OK]   {name}")
        return 0
    print(f"  [FAIL] {name} {detail}")
    return 1


def _force_dice(value: int) -> None:
    mm._dice_pool = np.full(20000, value, dtype=np.int8)
    mm._dice_idx = 0


def _make_atk(quality=2, defense=4, ignores_cover=False, weapon=None):
    if weapon is None:
        weapon = Weapon("rifle", range_inches=24, attacks=2, ap=0)
    atk_u = ResolvedUnit(
        template_id="t", name="atk", models=5, quality=quality, defense=defense,
        weapons=[weapon] * 5,
        weapons_per_model=[[weapon]] * 5,
        ignores_cover=ignores_cover,
    )
    return UnitState(
        atk_u, models_alive=5, wounds_per_model=[1] * 5,
        positions=[(10, 10), (10, 11), (10, 12), (10, 13), (10, 14)],
        weapons_per_model=[[weapon]] * 5, owner='A',
    )


def _make_def(positions=None):
    def_u = ResolvedUnit(
        template_id="t", name="def", models=5, quality=4, defense=4,
        weapons=[], weapons_per_model=[[]] * 5,
    )
    if positions is None:
        positions = [(20, 10), (20, 11), (20, 12), (20, 13), (20, 14)]
    return UnitState(
        def_u, models_alive=5, wounds_per_model=[1] * 5,
        positions=positions,
        weapons_per_model=[[]] * 5, owner='B',
    )


def test_majority_cover_applies() -> int:
    fails = 0
    # Sheltering piece covering ALL 5 defender models (5/5 in cover, majority).
    b = Board()
    b.set_terrain([
        TerrainPiece(18, 25, 8, 16, CoverType.SHELTERING, MovementType.OPEN)], build_vis_cover=False)
    _force_dice(3)  # def 4+ → block fails (3 < 4) without cover
    d_no = _make_def(); cm.resolve_shooting(_make_atk(), d_no, board=Board())
    _force_dice(3)
    d_yes = _make_def(); cm.resolve_shooting(_make_atk(), d_yes, board=b)
    fails += _check("baseline (no cover) deals wounds",
                    d_no.models_alive < 5)
    fails += _check("majority cover prevents wounds (def 4+ → 3+ saves)",
                    d_yes.models_alive == 5)
    return fails


def test_strict_majority_required() -> int:
    # Cover only 2 of 5 models — NOT a strict majority (2*2=4 < 5).
    fails = 0
    b = Board()
    b.set_terrain([
        TerrainPiece(18, 25, 12, 16, CoverType.SHELTERING, MovementType.OPEN)], build_vis_cover=False)
    # Defenders at rows 10..14; piece covers y=12..16, so models at (20,12),
    # (20,13), (20,14) are inside (3 of 5 — strict majority).
    _force_dice(3)
    d3 = _make_def(); cm.resolve_shooting(_make_atk(), d3, board=b)
    fails += _check("3 of 5 in cover (strict majority) blocks wounds",
                    d3.models_alive == 5)

    # Now shrink piece to cover only 2 of 5: y=13..14
    b2 = Board()
    b2.set_terrain([
        TerrainPiece(18, 25, 13, 14, CoverType.SHELTERING, MovementType.OPEN)], build_vis_cover=False)
    _force_dice(3)
    d2 = _make_def(); cm.resolve_shooting(_make_atk(), d2, board=b2)
    fails += _check("2 of 5 in cover (no majority) does NOT confer cover",
                    d2.models_alive < 5)
    return fails


def test_unit_ignores_cover_suppresses() -> int:
    b = Board()
    b.set_terrain([
        TerrainPiece(18, 25, 8, 16, CoverType.SHELTERING, MovementType.OPEN)], build_vis_cover=False)
    _force_dice(3)
    d = _make_def()
    cm.resolve_shooting(_make_atk(ignores_cover=True), d, board=b)
    return _check("unit-level ignores_cover overrides cover bonus",
                  d.models_alive < 5, f"alive={d.models_alive}")


def test_weapon_ignores_cover_suppresses() -> int:
    b = Board()
    b.set_terrain([
        TerrainPiece(18, 25, 8, 16, CoverType.SHELTERING, MovementType.OPEN)], build_vis_cover=False)
    _force_dice(3)
    w = Weapon("rifle_ic", range_inches=24, attacks=2, ap=0,
               ignores_cover=True)
    d = _make_def()
    cm.resolve_shooting(_make_atk(weapon=w), d, board=b)
    return _check("per-weapon ignores_cover overrides cover bonus",
                  d.models_alive < 5, f"alive={d.models_alive}")


def test_target_declaration_visibility() -> int:
    # All defender models behind a BLOCKING piece → can_shoot_any False.
    b = Board()
    b.set_terrain([
        TerrainPiece(13, 17, 5, 25, CoverType.BLOCKING, MovementType.IMPASSIBLE)], build_vis_cover=False)
    atk = _make_atk()
    d = _make_def()
    can_shoot = cm.can_shoot_any(atk, d, board=b)
    return _check("can_shoot_any False when no defender model is visible",
                  not can_shoot)


def test_resolve_shooting_skips_invisible_shooter() -> int:
    # All defenders behind BLOCKING — every attacker model sees nothing →
    # resolve_shooting deals zero wounds (no shooter has a visible target).
    b = Board()
    b.set_terrain([
        TerrainPiece(13, 17, 5, 25, CoverType.BLOCKING, MovementType.IMPASSIBLE)], build_vis_cover=False)
    _force_dice(3)
    d = _make_def()
    cm.resolve_shooting(_make_atk(), d, board=b)
    return _check("invisible defender → zero wounds dealt",
                  d.models_alive == 5)


if __name__ == "__main__":
    total = 0
    print("\n--- test_majority_cover_applies ---")
    total += test_majority_cover_applies()
    print("\n--- test_strict_majority_required ---")
    total += test_strict_majority_required()
    print("\n--- test_unit_ignores_cover_suppresses ---")
    total += test_unit_ignores_cover_suppresses()
    print("\n--- test_weapon_ignores_cover_suppresses ---")
    total += test_weapon_ignores_cover_suppresses()
    print("\n--- test_target_declaration_visibility ---")
    total += test_target_declaration_visibility()
    print("\n--- test_resolve_shooting_skips_invisible_shooter ---")
    total += test_resolve_shooting_skips_invisible_shooter()
    print(f"\n=== {total} failure(s) ===")
    sys.exit(1 if total else 0)
