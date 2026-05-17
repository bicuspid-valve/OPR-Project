"""Bit-compare C ports against the pure-Python reference implementations
for build_exclusion_grid, compute_post_move_rel, and _encode_unit_tactical_into.

Run: .venv/bin/python test_fast_core_ports.py
"""
from __future__ import annotations

import os
import random
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch


def test_build_exclusion_grid():
    from board import COLS, ROWS
    import fast_core
    import movement as mv

    rng = random.Random(0)
    failures = 0
    for trial in range(10):
        n = rng.randint(0, 30)
        enemies = set()
        while len(enemies) < n:
            enemies.add((rng.randrange(COLS), rng.randrange(ROWS)))
        py = mv._build_exclusion_grid_py(enemies)
        c = fast_core.fast_build_exclusion_grid(enemies, COLS, ROWS)
        if bytes(py) != bytes(c):
            diff = sum(1 for a, b in zip(py, c) if a != b)
            print(f"  [FAIL] build_exclusion_grid trial {trial}: {diff} cells differ")
            failures += 1
        else:
            print(f"  [OK]   build_exclusion_grid trial {trial}: n={n}")
    # Edge cases: empty, corner enemies
    for case, enemies in [
        ("empty", set()),
        ("corners", {(0, 0), (COLS - 1, 0), (0, ROWS - 1), (COLS - 1, ROWS - 1)}),
        ("dense", {(c, r) for c in range(5) for r in range(5)}),
    ]:
        py = mv._build_exclusion_grid_py(enemies)
        c = fast_core.fast_build_exclusion_grid(enemies, COLS, ROWS)
        if bytes(py) != bytes(c):
            print(f"  [FAIL] build_exclusion_grid {case}")
            failures += 1
        else:
            print(f"  [OK]   build_exclusion_grid {case}")
    return failures


def test_compute_post_move_rel():
    from ml_features import BOARD_DIAG
    import fast_core

    inv_diag = 1.0 / BOARD_DIAG
    rng = random.Random(1)
    failures = 0
    for trial in range(10):
        post_x = rng.uniform(0, 72)
        post_y = rng.uniform(0, 48)
        enemies = [(rng.uniform(0, 72), rng.uniform(0, 48)) for _ in range(10)]
        # Pure Python reference
        import math
        ref = np.zeros(30, dtype=np.float32)
        for i, (ex, ey) in enumerate(enemies):
            dx = ex - post_x; dy = ey - post_y
            d = math.sqrt(dx * dx + dy * dy)
            base = i * 3
            if d >= 1e-6:
                ref[base] = dy / d
                ref[base + 1] = dx / d
            ref[base + 2] = d * inv_diag
        c_arr = fast_core.fast_compute_post_move_rel(post_x, post_y, enemies, inv_diag)
        if not np.allclose(ref, c_arr, atol=1e-6):
            diff = np.max(np.abs(ref - c_arr))
            print(f"  [FAIL] compute_post_move_rel trial {trial}: max diff {diff}")
            failures += 1
        else:
            print(f"  [OK]   compute_post_move_rel trial {trial}")
    # Edge: enemy AT post-move position (d < 1e-6)
    c_arr = fast_core.fast_compute_post_move_rel(
        10.0, 10.0, [(10.0, 10.0)] + [(0.0, 0.0)] * 9, inv_diag)
    if c_arr[0] == 0.0 and c_arr[1] == 0.0 and c_arr[2] < 1e-6:
        print(f"  [OK]   compute_post_move_rel coincident enemy")
    else:
        print(f"  [FAIL] compute_post_move_rel coincident: {c_arr[:3]}")
        failures += 1
    return failures


def test_encode_unit_tactical():
    """Set up a real game and compare the C-encoded state_vec vs Python ref."""
    import json
    import fast_core
    from ml_features import (
        encode_state_tactical, _encode_unit_tactical_into,
        _encode_unit_tactical_into_fast,
        TACTICAL_TOTAL_FEATURES, TACTICAL_UNIT_FEATURES, MAX_UNITS_PER_SIDE,
        _TOFF_ACTIVATED, _TOFF_FATIGUED, _TOFF_SHAKEN,
        _ZERO_RANGED_ROW, _ZERO_MELEE_ROW,
        _get_model_objectives, _get_opposing_positions, precompute_damage,
        _objective_control_mapped, _projected_objective_control_mapped,
        GLOBAL_FEATURES,
    )
    from board import Board
    from game import deploy_armies
    from evolution import resolve_army, _make_unit_states, make_entry
    from models import ArmyList

    hof_path = "results/hall_of_fame_ml.json"
    if not os.path.exists(hof_path):
        hof_path = "results/hall_of_fame.json"
    with open(hof_path) as f:
        hof = json.load(f)

    def _load(entry):
        army = ArmyList()
        for e in entry["entries"]:
            ent = make_entry(e["template_id"], upgrades=e.get("upgrades", {}),
                             ai_role=e.get("ai_role", "killer"))
            ent.combat_preference = e.get("combat_preference", "ranged")
            army.entries.append(ent)
        return army

    failures = 0
    max_diff = 0.0
    rng = random.Random(42)
    for trial in range(6):
        a = _load(rng.choice(hof))
        b = _load(rng.choice(hof))
        res_a = resolve_army(a); res_b = resolve_army(b)
        sa = _make_unit_states(a, res_a, "A")
        sb = _make_unit_states(b, res_b, "B")
        board = Board()
        deploy_armies(sa, sb, board)
        fr_a, fm_a = precompute_damage([u.unit for u in sa], [u.unit for u in sb])
        fr_b, fm_b = precompute_damage([u.unit for u in sb], [u.unit for u in sa])
        pts_a = sum(u.unit.points for u in sa)
        pts_b = sum(u.unit.points for u in sb)

        for player in ("A", "B"):
            if player == "A":
                units_self, units_other = sa, sb
                rm_self, mm_self = fr_a, fm_a
                rm_other, mm_other = fr_b, fm_b
                pts_self, pts_other = pts_a, pts_b
            else:
                units_self, units_other = sb, sa
                rm_self, mm_self = fr_b, fm_b
                rm_other, mm_other = fr_a, fm_a
                pts_self, pts_other = pts_b, pts_a

            fast_core.USE_C_EXT = True
            sv_c = encode_state_tactical(
                units_self, units_other, 2, board, player,
                friendly_ranged_matchups=rm_self, friendly_melee_matchups=mm_self,
                enemy_ranged_matchups=rm_other, enemy_melee_matchups=mm_other,
                total_friendly_points=pts_self, total_enemy_points=pts_other,
            )
            fast_core.USE_C_EXT = False
            sv_py = encode_state_tactical(
                units_self, units_other, 2, board, player,
                friendly_ranged_matchups=rm_self, friendly_melee_matchups=mm_self,
                enemy_ranged_matchups=rm_other, enemy_melee_matchups=mm_other,
                total_friendly_points=pts_self, total_enemy_points=pts_other,
            )
            fast_core.USE_C_EXT = True

            diff = (sv_c - sv_py).abs().max().item()
            max_diff = max(max_diff, diff)
            # float32 rounding order can differ on sin/cos — allow tiny tolerance.
            if diff > 2e-5:
                # Locate first offending index for diagnosis
                bad = (sv_c - sv_py).abs() > 2e-5
                idx = int(bad.nonzero()[0].item()) if bad.any() else -1
                print(f"  [FAIL] encode_state trial={trial} player={player} "
                      f"max_diff={diff:.2e} first_bad_idx={idx} "
                      f"py={sv_py[idx].item():.6f} c={sv_c[idx].item():.6f}")
                failures += 1
            else:
                print(f"  [OK]   encode_state trial={trial} player={player} "
                      f"max_diff={diff:.2e}")
    print(f"  overall max diff: {max_diff:.2e}")
    return failures


if __name__ == "__main__":
    total = 0
    print("\n--- test_build_exclusion_grid ---")
    total += test_build_exclusion_grid()
    print("\n--- test_compute_post_move_rel ---")
    total += test_compute_post_move_rel()
    print("\n--- test_encode_unit_tactical ---")
    total += test_encode_unit_tactical()
    print(f"\n=== {total} failure(s) ===")
    sys.exit(1 if total else 0)
