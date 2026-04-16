"""Check deployment symmetry by measuring the average position offset
between A's units and B's flipped units across many deployments."""
from __future__ import annotations

import random
from pathlib import Path
import numpy as np

from evolution import HallOfFame, resolve_army, _make_unit_states
from game import Board, deploy_armies
from board import COLS, ROWS

def _flip(cx, cy):
    return (COLS - 1) - cx, (ROWS - 1) - cy

def main():
    import fast_core
    fast_core.USE_C_EXT = fast_core.is_available()

    hof_path = Path(__file__).resolve().parent / "results" / "hall_of_fame_ml.json"
    hof = HallOfFame.load_from_json(hof_path)
    armies = [(e.army, resolve_army(e.army)) for e in hof.entries]

    N = 500
    # For each army, deploy as mirror match and measure position deltas
    all_deltas_x = []
    all_deltas_y = []
    per_unit_deltas = {}  # unit_index -> list of (dx, dy)

    for trial in range(N):
        army, res = random.choice(armies)
        sa = _make_unit_states(army, res, "A")
        sb = _make_unit_states(army, res, "B")
        board = Board()
        deploy_armies(sa, sb, board)

        n_units = min(len(sa), len(sb))
        for i in range(n_units):
            if sa[i].models_alive <= 0 or sb[i].models_alive <= 0:
                continue
            a_cx, a_cy = sa[i].centre()
            b_cx, b_cy = sb[i].centre()
            # Flip A's position — if symmetric, should equal B's position
            fa_cx, fa_cy = _flip(a_cx, a_cy)
            dx = b_cx - fa_cx
            dy = b_cy - fa_cy
            all_deltas_x.append(dx)
            all_deltas_y.append(dy)
            if i not in per_unit_deltas:
                per_unit_deltas[i] = []
            per_unit_deltas[i].append((dx, dy))

    all_dx = np.array(all_deltas_x)
    all_dy = np.array(all_deltas_y)

    print(f"Deployment symmetry check ({N} mirror deployments)")
    print(f"Total unit measurements: {len(all_dx)}")
    print(f"\nOverall position delta (B_actual - flip(A_actual)):")
    print(f"  dx: mean={all_dx.mean():+.4f}  std={all_dx.std():.4f}  (0 = perfect symmetry)")
    print(f"  dy: mean={all_dy.mean():+.4f}  std={all_dy.std():.4f}")
    print(f"  |d|: mean={np.sqrt(all_dx**2 + all_dy**2).mean():.4f}")
    print()

    # Check which units are most asymmetric
    print("Per-unit-slot average delta:")
    for i in sorted(per_unit_deltas.keys()):
        deltas = per_unit_deltas[i]
        dxs = [d[0] for d in deltas]
        dys = [d[1] for d in deltas]
        mx, my = np.mean(dxs), np.mean(dys)
        sx, sy = np.std(dxs), np.std(dys)
        n = len(deltas)
        if abs(mx) > 0.01 or abs(my) > 0.01:
            print(f"  Slot {i}: dx={mx:+.3f}±{sx:.3f}  dy={my:+.3f}±{sy:.3f}  (n={n})  ** ASYMMETRIC **")
        else:
            print(f"  Slot {i}: dx={mx:+.3f}±{sx:.3f}  dy={my:+.3f}±{sy:.3f}  (n={n})")

    # Fraction with non-zero delta
    nonzero = np.sum(np.abs(all_dx) > 0.01) + np.sum(np.abs(all_dy) > 0.01)
    print(f"\n  Measurements with non-zero delta: {nonzero}/{2*len(all_dx)} ({100*nonzero/(2*len(all_dx)):.1f}%)")


if __name__ == "__main__":
    main()
