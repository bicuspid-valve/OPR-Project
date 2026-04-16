"""Quick profile: run 100 ML-vs-ML games (HoF armies, no planning)
and measure value-prediction symmetry at each activation boundary.

For every activation we record V from both perspectives at the same
board state. In a perfectly calibrated zero-sum value head:
  V_A + V_B ≈ 0
The "symmetry gap" is V_A + V_B.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch

from models import ArmyList, resolve_entry
from evolution import HallOfFame, resolve_army, _make_unit_states
from game import _simulate_game_impl, UnitState
from ml_features import encode_state_tactical, precompute_damage
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from ml_integration_tactical import apply_tactical_model


def get_value(model, units_a, units_b, round_num, board, player,
              fr_a, fm_a, fr_b, fm_b, pts_a, pts_b):
    """Compute value estimate for `player` at current board state."""
    if player == "A":
        friendly, enemy = units_a, units_b
        fr, fm = fr_a, fm_a
        er, em = fr_b, fm_b
        pts_f, pts_e = pts_a, pts_b
    else:
        friendly, enemy = units_b, units_a
        fr, fm = fr_b, fm_b
        er, em = fr_a, fm_a
        pts_f, pts_e = pts_b, pts_a

    state_vec = encode_state_tactical(
        friendly, enemy, round_num, board, player,
        friendly_ranged_matchups=fr, friendly_melee_matchups=fm,
        enemy_ranged_matchups=er, enemy_melee_matchups=em,
        total_friendly_points=pts_f, total_enemy_points=pts_e,
    )
    with torch.no_grad():
        h, _units, round_oh = model.trunk(state_vec.unsqueeze(0))
        opp_embed = model._get_opp_embed(h, None)
        value = model.value_head(h, round_oh, opp_embed).item()
    return value


def main():
    import fast_core
    fast_core.USE_C_EXT = fast_core.is_available()

    # Load model
    model_path = Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt"
    sd = load_model_state_dict(model_path)
    model = TacticalModel()
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Model loaded: {model_path.name}")

    # Load HoF armies
    hof_path = Path(__file__).resolve().parent / "results" / "hall_of_fame_ml.json"
    hof = HallOfFame.load_from_json(hof_path)
    print(f"Hall of Fame: {len(hof.entries)} armies")

    if len(hof.entries) < 2:
        print("Need at least 2 HoF armies!")
        return

    armies = [(e.army, resolve_army(e.army)) for e in hof.entries]

    N_GAMES = 100
    all_gaps = []       # V_A + V_B at same board state (should be ~0)
    all_va = []         # V from A's perspective
    all_vb = []         # V from B's perspective
    per_round_gaps = {r: [] for r in range(1, 5)}  # gaps by game round

    t0 = time.time()
    for game_i in range(N_GAMES):
        # Pick two random HoF armies
        (army_a, res_a), (army_b, res_b) = random.sample(armies, 2)
        states_a = _make_unit_states(army_a, res_a, "A")
        states_b = _make_unit_states(army_b, res_b, "B")

        # We'll use _simulate_game_impl with a custom _tactical_inference_fn
        # that wraps apply_tactical_model but also captures value from both sides.
        # The tricky part: _tactical_inference_fn only receives per-side data,
        # but we need both sides. We capture states_a/states_b via closure,
        # and the board is passed directly.

        # Precompute damage (these are computed inside _simulate_game_impl too,
        # but we need them in our closure)
        game_gaps = []
        game_va = []
        game_vb = []

        def make_inference_fn(mdl, gaps_out, va_out, vb_out):
            """Create an inference callback that also records values."""
            # These will be set by _simulate_game_impl when it creates unit states
            # We capture them via the states passed in.
            # But _simulate_game_impl creates its OWN states from army lists...
            # We need to access the actual units_a/units_b inside the game loop.
            # The callback receives (my_units, opp_units, ...) so we can
            # reconstruct both sides from those.

            def inference_fn(my_units, opp_units, round_num, board, player,
                             my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts):
                # Determine units_a, units_b from perspective
                if player == "A":
                    ua, ub = my_units, opp_units
                    fr_a, fm_a = my_fr, my_fm
                    fr_b, fm_b = opp_fr, opp_fm
                    pts_a, pts_b = my_pts, opp_pts
                else:
                    ua, ub = opp_units, my_units
                    fr_a, fm_a = opp_fr, opp_fm
                    fr_b, fm_b = my_fr, my_fm
                    pts_a, pts_b = opp_pts, my_pts

                # Compute value from BOTH perspectives at this board state
                va = get_value(mdl, ua, ub, round_num, board, "A",
                               fr_a, fm_a, fr_b, fm_b, pts_a, pts_b)
                vb = get_value(mdl, ua, ub, round_num, board, "B",
                               fr_a, fm_a, fr_b, fm_b, pts_a, pts_b)
                gap = va + vb
                gaps_out.append((round_num, gap))
                va_out.append(va)
                vb_out.append(vb)

                # Now do the actual ML decision
                active, tr, action, goal, ct, reason, _ = apply_tactical_model(
                    mdl, my_units, opp_units, round_num, board, player,
                    friendly_ranged_matchups=my_fr, friendly_melee_matchups=my_fm,
                    enemy_ranged_matchups=opp_fr, enemy_melee_matchups=opp_fm,
                    total_friendly_points=my_pts, total_enemy_points=opp_pts,
                )
                return active, tr, action, goal, ct, reason

            return inference_fn

        inference_fn = make_inference_fn(model, game_gaps, game_va, game_vb)

        # Run game using existing game loop — model handles both sides
        _simulate_game_impl(
            res_a, res_b, mode="objectives",
            states_a=states_a, states_b=states_b,
            ml_model_a=model, ml_model_b=model,
            _tactical_inference_fn=inference_fn,
        )

        for rn, gap in game_gaps:
            all_gaps.append(gap)
            per_round_gaps[rn].append(gap)
        all_va.extend(game_va)
        all_vb.extend(game_vb)

        if (game_i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {game_i+1}/{N_GAMES} games  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    all_gaps = np.array(all_gaps)
    all_va = np.array(all_va)
    all_vb = np.array(all_vb)

    print(f"\n{'='*60}")
    print(f"VALUE SYMMETRY PROFILE  ({N_GAMES} games, {len(all_gaps)} measurements)")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s ({elapsed/N_GAMES:.2f}s/game)")
    print()

    # Symmetry gap: V_A + V_B at same board state (should be ~0)
    print("Symmetry gap  (V_A + V_B at same state,  ideal = 0):")
    print(f"  Mean:    {all_gaps.mean():+.4f}")
    print(f"  Median:  {np.median(all_gaps):+.4f}")
    print(f"  Std:     {all_gaps.std():.4f}")
    print(f"  |gap|:   {np.abs(all_gaps).mean():.4f}  (mean absolute)")
    print(f"  Min:     {all_gaps.min():+.4f}")
    print(f"  Max:     {all_gaps.max():+.4f}")
    print()

    # Per-round breakdown
    print("Per-round symmetry gap (mean ± std):")
    for rn in range(1, 5):
        rg = np.array(per_round_gaps[rn]) if per_round_gaps[rn] else np.array([0.0])
        print(f"  Round {rn}: {rg.mean():+.4f} ± {rg.std():.4f}  (n={len(rg)})")
    print()

    # Distribution of gaps
    bins = [(-2, -0.5), (-0.5, -0.2), (-0.2, -0.1), (-0.1, 0.1),
            (0.1, 0.2), (0.2, 0.5), (0.5, 2)]
    print("  Gap distribution:")
    for lo, hi in bins:
        count = np.sum((all_gaps >= lo) & (all_gaps < hi))
        pct = 100 * count / len(all_gaps)
        bar = '#' * int(pct / 2)
        print(f"    [{lo:+.1f}, {hi:+.1f}):  {count:5d}  ({pct:5.1f}%)  {bar}")
    print()

    # Per-side value stats
    print(f"Player A values (n={len(all_va)}):")
    print(f"  Mean: {all_va.mean():+.4f}  Std: {all_va.std():.4f}")
    print(f"Player B values (n={len(all_vb)}):")
    print(f"  Mean: {all_vb.mean():+.4f}  Std: {all_vb.std():.4f}")
    print()

    # Systematic bias check
    both = np.concatenate([all_va, all_vb])
    print(f"All values combined (n={len(both)}):")
    print(f"  Mean: {both.mean():+.4f}  (>0 = systematic optimism)")
    print(f"  Fraction positive: {(both > 0).mean():.1%}")


if __name__ == "__main__":
    main()
