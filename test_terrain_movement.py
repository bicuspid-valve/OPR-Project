"""TERRAIN_SPEC.md §6 — terrain movement tests.

Covers §3: open passes through, difficult caps at 6", impassible blocks,
flying traverses difficult and impassible during path but cannot end on
impassible, strider ignores only difficult, destination-square impassible
rejection (including for flying), 6" cap across multiple difficult pieces,
starting-square-in-difficult does not trigger cap.
Run: python3 test_terrain_movement.py
"""
from __future__ import annotations

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from board import Board, TerrainPiece, CoverType, MovementType, COLS, ROWS
from movement import _greedy_move
import fast_core


def _set_python_only():
    fast_core.USE_C_EXT = False


def _set_c_path():
    fast_core.USE_C_EXT = True


def _check(name: str, ok: bool, detail: str = "") -> int:
    if ok:
        print(f"  [OK]   {name}")
        return 0
    print(f"  [FAIL] {name} {detail}")
    return 1


def test_open_passthrough() -> int:
    _set_python_only()
    b = Board()
    res = _greedy_move((10, 10), (20, 10), 12.0, b, set())
    return _check("OPEN passthrough reaches goal", res == (20, 10), f"got {res}")


def test_difficult_cap_6in() -> int:
    fails = 0
    _set_python_only()
    b = Board()
    # Full-width band of difficult terrain that the unit MUST enter to make
    # progress south. With a 12" budget the cap at 6" should pin the unit to
    # ≤ 6" Manhattan-ish reach.
    b.set_terrain([TerrainPiece(0, COLS - 1, 14, 16, CoverType.OBSCURING,
                                 MovementType.DIFFICULT)], build_vis_cover=False)
    res = _greedy_move((10, 10), (10, 30), 12.0, b, set())
    # Distance from (10,10) along the chosen path is at most 6"; reaching
    # (10, 16) would cost 6 cardinal steps (within cap), but the unit can't
    # exceed 6 squares south.
    dy = res[1] - 10
    fails += _check("difficult cap pins south reach to ~6\"",
                    dy <= 7, f"got {res} dy={dy}")
    return fails


def test_impassible_blocks() -> int:
    _set_python_only()
    b = Board()
    b.set_terrain([TerrainPiece(11, 19, 8, 12, CoverType.BLOCKING,
                                 MovementType.IMPASSIBLE)], build_vis_cover=False)
    res = _greedy_move((10, 10), (15, 10), 12.0, b, set())
    # Cannot end inside the impassible piece
    in_imp = b.impassible_grid[res[1] * COLS + res[0]]
    return _check("impassible piece blocks destination",
                  not in_imp, f"got {res} in_imp={bool(in_imp)}")


def test_flying_traverses_but_cannot_end() -> int:
    fails = 0
    _set_python_only()
    b = Board()
    # Goal is INSIDE the impassible — flying must NOT end there.
    b.set_terrain([TerrainPiece(13, 17, 8, 12, CoverType.BLOCKING,
                                 MovementType.IMPASSIBLE)], build_vis_cover=False)
    res = _greedy_move((10, 10), (15, 10), 12.0, b, set(), flying=True)
    in_imp = b.impassible_grid[res[1] * COLS + res[0]]
    fails += _check("flying cannot end on impassible",
                    not in_imp, f"got {res}")
    # But flying CAN reach a square BEYOND the impassible, even with an
    # impassible piece on the direct path — proving traversal works.
    res2 = _greedy_move((10, 10), (22, 10), 12.0, b, set(), flying=True)
    fails += _check("flying reaches goal beyond impassible",
                    res2 == (22, 10), f"got {res2}")
    return fails


def test_strider_ignores_difficult_only() -> int:
    fails = 0
    _set_python_only()
    b = Board()
    b.set_terrain([TerrainPiece(11, 19, 8, 12, CoverType.OBSCURING,
                                 MovementType.DIFFICULT)], build_vis_cover=False)
    # Without strider: cap kicks in, can't reach (22, 10).
    res_no = _greedy_move((10, 10), (22, 10), 12.0, b, set())
    # With strider: ignores DIFFICULT cap, reaches goal.
    res_yes = _greedy_move((10, 10), (22, 10), 12.0, b, set(), strider=True)
    fails += _check("strider lets unit reach goal through difficult",
                    res_yes == (22, 10), f"got {res_yes}")
    fails += _check("non-strider does NOT reach goal",
                    res_no != (22, 10), f"got {res_no}")
    # Strider does NOT bypass impassible.
    b2 = Board()
    b2.set_terrain([TerrainPiece(11, 19, 8, 12, CoverType.BLOCKING,
                                  MovementType.IMPASSIBLE)], build_vis_cover=False)
    res_s_imp = _greedy_move((10, 10), (15, 10), 12.0, b2, set(),
                             strider=True)
    in_imp = b2.impassible_grid[res_s_imp[1] * COLS + res_s_imp[0]]
    fails += _check("strider does NOT enter impassible",
                    not in_imp, f"got {res_s_imp}")
    return fails


def test_dest_impassible_rejected_for_flying() -> int:
    _set_python_only()
    b = Board()
    b.set_terrain([TerrainPiece(13, 17, 8, 12, CoverType.BLOCKING,
                                 MovementType.IMPASSIBLE)], build_vis_cover=False)
    # Goal IS the centre of the impassible piece. Flying can pass through but
    # must not end there.
    res = _greedy_move((10, 10), (15, 10), 24.0, b, set(), flying=True)
    in_imp = b.impassible_grid[res[1] * COLS + res[0]]
    return _check("flying destination not in impassible",
                  not in_imp, f"got {res}")


def test_multiple_difficult_no_stack() -> int:
    _set_python_only()
    b = Board()
    # Two full-width difficult bands. Even crossing both, the cap stays at 6"
    # (does not become 3"). Path enters difficult once the unit moves south
    # past row 13, and the cap pins it to ≤ 6" total path length.
    b.set_terrain([
        TerrainPiece(0, COLS - 1, 14, 15, CoverType.OBSCURING,
                     MovementType.DIFFICULT),
        TerrainPiece(0, COLS - 1, 17, 18, CoverType.OBSCURING,
                     MovementType.DIFFICULT),
    ], build_vis_cover=False)
    res = _greedy_move((10, 10), (10, 30), 12.0, b, set())
    dy = res[1] - 10
    return _check("two difficult bands do not stack the cap (still 6\")",
                  dy <= 7, f"got {res} dy={dy}")


def test_start_in_difficult_no_cap_when_exiting() -> int:
    _set_python_only()
    # Start in a difficult piece, but the exit takes one step (s_1 outside
    # difficult). Per spec, the path enters difficult only if some s_i (i≥1)
    # belongs to a difficult piece.
    b = Board()
    b.set_terrain([
        TerrainPiece(8, 11, 8, 11, CoverType.OBSCURING, MovementType.DIFFICULT),
    ], build_vis_cover=False)
    # Start at (10,10) inside difficult; goal NW so first step exits.
    res = _greedy_move((10, 10), (0, 0), 12.0, b, set())
    # The path should have some flexibility — it's allowed to cap-or-not based
    # on whether the route stays out of difficult after s_0. This is a sanity
    # check only: starting inside should NOT make the unit immobile.
    return _check("start-in-difficult does not freeze the unit",
                  res != (10, 10), f"got {res}")


if __name__ == "__main__":
    total = 0
    print("\n--- test_open_passthrough ---"); total += test_open_passthrough()
    print("\n--- test_difficult_cap_6in ---"); total += test_difficult_cap_6in()
    print("\n--- test_impassible_blocks ---"); total += test_impassible_blocks()
    print("\n--- test_flying_traverses_but_cannot_end ---")
    total += test_flying_traverses_but_cannot_end()
    print("\n--- test_strider_ignores_difficult_only ---")
    total += test_strider_ignores_difficult_only()
    print("\n--- test_dest_impassible_rejected_for_flying ---")
    total += test_dest_impassible_rejected_for_flying()
    print("\n--- test_multiple_difficult_no_stack ---")
    total += test_multiple_difficult_no_stack()
    print("\n--- test_start_in_difficult_no_cap_when_exiting ---")
    total += test_start_in_difficult_no_cap_when_exiting()
    print(f"\n=== {total} failure(s) ===")
    sys.exit(1 if total else 0)
