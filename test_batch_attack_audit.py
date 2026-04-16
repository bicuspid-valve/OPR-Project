"""Batch simulation audit: check that total_attacks is consistent with alive
model counts across many recorded games.

For every shooting activation frame, we verify:
1. The weapon summary in combat_stats implies the same total as total_attacks
2. The attacker's alive count in the snapshot is < starting models but
   total_attacks equals the FULL-STRENGTH total (the bug the user observed)
"""
from __future__ import annotations

import random
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from evolution import HallOfFame, resolve_army, _make_unit_states
from game import simulate_game_recorded
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from pathlib import Path


def main():
    model_path = Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt"
    sd = load_model_state_dict(model_path)
    model = TacticalModel()
    model.load_state_dict(sd, strict=False)
    model.eval()

    hof = HallOfFame.load_from_json(
        Path(__file__).resolve().parent / "results" / "hall_of_fame_ml.json")
    armies = [(e.army, resolve_army(e.army)) for e in hof.entries]

    N_GAMES = 200
    total_shooting_frames = 0
    summary_vs_total_mismatches = 0
    full_strength_despite_casualties = 0
    violations = []

    t0 = time.time()
    for game_i in range(N_GAMES):
        (army_a, res_a), (army_b, res_b) = random.sample(armies, 2)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")
        all_units_start = sa + sb

        # Record starting models and max ranged attacks per unit
        n_units = len(all_units_start)
        starting_models = [u.unit.models for u in all_units_start]
        # Max ranged attacks at full strength (from weapons_per_model)
        starting_ranged_attacks = []
        for u in all_units_start:
            total = 0
            for mw in u.unit.weapons_per_model:
                for w in mw:
                    if not w.melee:
                        total += w.attacks
            starting_ranged_attacks.append(total)

        result, frames, labels, owners, unit_pts, unit_info = \
            simulate_game_recorded(
                res_a, res_b, states_a=sa, states_b=sb,
                ml_model_a=model, ml_model_b=model,
            )

        prev_activated = [False] * n_units

        for fi, frame in enumerate(frames):
            cs = frame.get('combat_stats')
            if cs is None:
                prev_activated = list(frame.get('activated', prev_activated))
                continue
            # Only check shooting (not melee)
            if cs.get('combat_type') == 'melee':
                prev_activated = list(frame.get('activated', prev_activated))
                continue
            if 'attacker_weapons' not in cs:
                prev_activated = list(frame.get('activated', prev_activated))
                continue

            total_shooting_frames += 1

            # Check 1: weapon summary total vs total_attacks
            summary_attacks = sum(
                w['count'] * w['attacks'] for w in cs['attacker_weapons'])
            actual_attacks = cs.get('total_attacks', 0)
            if summary_attacks != actual_attacks:
                summary_vs_total_mismatches += 1

            # Use debug fields injected by resolve_shooting
            dbg_alive = cs.get('_dbg_attacker_alive')
            dbg_wpm = cs.get('_dbg_attacker_wpm')
            dbg_name = cs.get('_dbg_attacker_name', '?')
            dbg_starting = cs.get('_dbg_attacker_starting', 0)

            if dbg_alive is not None and dbg_starting > 0:
                dbg_max_ranged = cs.get('_dbg_attacker_max_ranged', 0)
                dbg_full_ranged = cs.get('_dbg_attacker_full_ranged', 0)

                # Check 2: total_attacks exceeds what alive models can produce
                if actual_attacks > dbg_max_ranged and dbg_max_ranged > 0:
                    full_strength_despite_casualties += 1
                    violations.append({
                        'game': game_i,
                        'frame': fi,
                        'unit': dbg_name,
                        'alive_at_shoot': dbg_alive,
                        'wpm_at_shoot': dbg_wpm,
                        'starting': dbg_starting,
                        'total_attacks': actual_attacks,
                        'max_from_alive': dbg_max_ranged,
                        'full_ranged': dbg_full_ranged,
                        'description': frame.get('description', ''),
                    })

            prev_activated = list(frame.get('activated', prev_activated))

        if (game_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{game_i + 1}/{N_GAMES}] {elapsed:.1f}s  "
                  f"shooting_frames={total_shooting_frames}  "
                  f"violations={full_strength_despite_casualties}")

    elapsed = time.time() - t0
    print(f"\n=== RESULTS ({N_GAMES} games, {elapsed:.1f}s) ===")
    print(f"Total shooting frames checked: {total_shooting_frames}")
    print(f"Weapon summary vs total_attacks mismatches: {summary_vs_total_mismatches}")
    print(f"Full-strength attacks despite casualties: {full_strength_despite_casualties}")

    if violations:
        print(f"\n--- First 20 violations ---")
        for v in violations[:20]:
            print(f"  Game {v['game']} frame {v['frame']}: {v['unit']}  "
                  f"alive={v['alive_at_shoot']}/{v['starting']}  "
                  f"attacks={v['total_attacks']} max_from_alive={v['max_from_alive']} "
                  f"full={v['full_ranged']}")
    else:
        print("\nNo violations found — bug does not reproduce in batch simulation.")


if __name__ == "__main__":
    main()
