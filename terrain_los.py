"""Line-of-sight and cover predicates for terrain (spec §4).

All geometric work is done in continuous coordinates: a model in grid square
(c, r) occupies the closed square with corners (c, r), (c+1, r), (c, r+1),
(c+1, r+1). Predicates expressed strictly in terms of grid squares + a terrain
list — this module has no dependency on Board state, so it can be used both at
deployment time (vis_cover_table build) and during live combat resolution.
"""
from __future__ import annotations

from board import CoverType, TerrainPiece


_EPS = 1e-9


def _square_corners(sq: tuple[int, int]) -> tuple:
    c, r = sq
    return (
        (float(c), float(r)),
        (float(c + 1), float(r)),
        (float(c), float(r + 1)),
        (float(c + 1), float(r + 1)),
    )


def line_crosses_terrain(p: tuple[float, float],
                         q: tuple[float, float],
                         X: TerrainPiece) -> bool:
    """True iff segment [p, q] crosses the open interior of X's continuous
    rectangle (x_lo, x_hi+1) x (y_lo, y_hi+1). Touching an edge or corner does
    not count as crossing — see §4.1."""
    x_min = float(X.x_lo)
    x_max = float(X.x_hi + 1)
    y_min = float(X.y_lo)
    y_max = float(X.y_hi + 1)

    px, py = p
    qx, qy = q
    dx = qx - px
    dy = qy - py

    # Liang-Barsky against the closed rectangle.
    p_arr = (-dx, dx, -dy, dy)
    q_arr = (px - x_min, x_max - px, py - y_min, y_max - py)

    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p_arr, q_arr):
        if abs(pi) < _EPS:
            # Segment is parallel to this slab; if outside, no intersection.
            if qi < -_EPS:
                return False
            continue
        t = qi / pi
        if pi < 0:
            if t > t1:
                return False
            if t > t0:
                t0 = t
        else:
            if t < t0:
                return False
            if t < t1:
                t1 = t

    if t1 - t0 <= _EPS:
        # Clipped to a single point at most — touching only.
        return False

    # The clipped interval [t0, t1] is non-degenerate. Check whether it lies
    # strictly inside the open rectangle (i.e., at least one interior point).
    # A midpoint test is sufficient because the segment is straight.
    tm = 0.5 * (t0 + t1)
    mx = px + tm * dx
    my = py + tm * dy
    if (x_min + _EPS < mx < x_max - _EPS
            and y_min + _EPS < my < y_max - _EPS):
        return True
    # Otherwise the clipped sub-segment lies entirely on a boundary edge.
    return False


def obscured_by(Y_sq: tuple[int, int],
                A_sq: tuple[int, int],
                X: TerrainPiece) -> bool:
    """Per-piece "A is obscured by X from Y" — §4.2.

    For each corner Z of Y, check whether the four segments [Z, A_corner_i] all
    cross X. If some Z exists where none of the four segments cross X, A is
    NOT obscured by X. Otherwise A IS obscured.
    """
    y_corners = _square_corners(Y_sq)
    a_corners = _square_corners(A_sq)
    for z in y_corners:
        any_crosses = False
        for a in a_corners:
            if line_crosses_terrain(z, a, X):
                any_crosses = True
                break
        if not any_crosses:
            # This Z has at least one clear line to A — not obscured.
            return False
    return True


def is_visible(Y_sq: tuple[int, int],
               A_sq: tuple[int, int],
               terrain: list[TerrainPiece]) -> bool:
    """Combined-blocking visibility — §4.2.

    A is visible to Y iff there exists at least one (Z, A_corner_j) pair
    over Y's 4 corners x A's 4 corners (16 pairs) such that the segment
    [Z, A_corner_j] crosses NO BLOCKING piece.
    """
    blocking = [p for p in terrain if p.cover_type == CoverType.BLOCKING]
    if not blocking:
        return True
    y_corners = _square_corners(Y_sq)
    a_corners = _square_corners(A_sq)
    for z in y_corners:
        for a in a_corners:
            blocked = False
            for X in blocking:
                if line_crosses_terrain(z, a, X):
                    blocked = True
                    break
            if not blocked:
                return True
    return False


def is_in_cover(Y_sq: tuple[int, int],
                A_sq: tuple[int, int],
                terrain: list[TerrainPiece]) -> bool:
    """A is in cover from Y, per §4.3.

    Cover applies if any piece X satisfies any of:
      - SHELTERING: A's square is wholly within X (regardless of Y).
      - OBSCURING: A's square is wholly within X, OR A is obscured by X from Y.
      - BLOCKING: A is obscured by X from Y (only meaningful when still visible).
    Cover is a binary flag — does not stack.
    """
    for X in terrain:
        ct = X.cover_type
        if ct == CoverType.SHELTERING:
            if X.contains_square(*A_sq):
                return True
        elif ct == CoverType.OBSCURING:
            if X.contains_square(*A_sq):
                return True
            if obscured_by(Y_sq, A_sq, X):
                return True
        elif ct == CoverType.BLOCKING:
            # A is impassible-bound, so wholly-within is impossible per §2.2 + §3.
            if obscured_by(Y_sq, A_sq, X):
                return True
    return False


def shooter_cover_state(Y_sq: tuple[int, int],
                        target_squares: list[tuple[int, int]],
                        terrain: list[TerrainPiece]) -> tuple[int, list[bool]]:
    """Aggregate per-shooter state over a defending unit's model squares.

    Returns (n_cover_or_invisible, visible_mask) where visible_mask[i] is
    True iff target_squares[i] is visible to Y. n_cover_or_invisible counts
    target models that are either not visible or in cover from Y.
    Used by §4.4 majority cover computation.
    """
    n_bad = 0
    visible: list[bool] = [False] * len(target_squares)
    for i, A_sq in enumerate(target_squares):
        vis = is_visible(Y_sq, A_sq, terrain)
        visible[i] = vis
        if not vis:
            n_bad += 1
        elif is_in_cover(Y_sq, A_sq, terrain):
            n_bad += 1
    return n_bad, visible
