"""Profile internals of game simulation to find hot paths."""
from __future__ import annotations

import os
import time
import statistics
import cProfile
import pstats
import io

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ml_training import _generate_army_pair, _run_games_batched_tactical
from ml_model_tactical import TacticalModel
from ml_features import encode_state_tactical, precompute_damage
from game import simulate_game, deploy_armies, _collect_enemy_positions, _sync_dead_models
from ai import (
    pick_target, choose_action_and_goal, activation_order,
    assign_objectives, reassign_roles,
)
from combat import resolve_shooting, check_morale, resolve_melee, resolve_impact, check_melee_morale
from movement import (
    execute_movement, execute_charge_movement, execute_counter_charge,
    post_melee_separation, consolidation_move, build_exclusion_grid,
)
from board import Board, OBJECTIVES
from models import UnitState
import random


def timed_game_with_breakdown():
    """Run one game with fine-grained timing of each subsystem."""
    res_a, res_b, states_a, states_b, *_ = _generate_army_pair()

    board = Board()

    timers = {
        "deploy": 0.0,
        "assign_objectives": 0.0,
        "reassign_roles": 0.0,
        "activation_order": 0.0,
        "choose_action": 0.0,
        "collect_enemy_pos": 0.0,
        "build_exclusion": 0.0,
        "execute_movement": 0.0,
        "charge_movement": 0.0,
        "counter_charge": 0.0,
        "resolve_shooting": 0.0,
        "resolve_melee": 0.0,
        "resolve_impact": 0.0,
        "check_morale": 0.0,
        "melee_morale": 0.0,
        "post_melee_sep": 0.0,
        "consolidation": 0.0,
        "sync_dead": 0.0,
        "pick_target": 0.0,
        "obj_update": 0.0,
    }
    counts = {k: 0 for k in timers}

    units_a = states_a
    units_b = states_b

    t0 = time.perf_counter()
    deploy_armies(units_a, units_b, board)
    timers["deploy"] += time.perf_counter() - t0
    counts["deploy"] += 1

    t0 = time.perf_counter()
    assign_objectives(units_a)
    assign_objectives(units_b)
    timers["assign_objectives"] += time.perf_counter() - t0
    counts["assign_objectives"] += 1

    a_first = random.random() < 0.5
    a_finished_first = a_first

    total_activations = 0

    for round_num in range(4):
        for u in units_a + units_b:
            u.activated = False
            u.fatigued = False

        current_is_a = a_first if round_num == 0 else a_finished_first

        t0 = time.perf_counter()
        reassign_roles(units_a)
        reassign_roles(units_b)
        timers["reassign_roles"] += time.perf_counter() - t0
        counts["reassign_roles"] += 1

        a_done = False
        b_done = False
        a_finished_first = True

        while True:
            if current_is_a:
                my_units, opp_units = units_a, units_b
            else:
                my_units, opp_units = units_b, units_a

            t0 = time.perf_counter()
            ordered = activation_order(my_units, enemies=opp_units, mode="objectives")
            timers["activation_order"] += time.perf_counter() - t0
            counts["activation_order"] += 1
            active = ordered[0] if ordered else None

            if active is None:
                if current_is_a:
                    a_done = True
                    if not b_done:
                        a_finished_first = True
                else:
                    b_done = True
                    if not a_done:
                        a_finished_first = False
                if a_done and b_done:
                    break
                current_is_a = not current_is_a
                continue

            active.activated = True
            total_activations += 1

            t0 = time.perf_counter()
            action, goal, charge_target, _reason = choose_action_and_goal(
                active, opp_units, board, mode="objectives")
            timers["choose_action"] += time.perf_counter() - t0
            counts["choose_action"] += 1

            if action == "charge" and charge_target is not None:
                t0 = time.perf_counter()
                enemy_positions = _collect_enemy_positions(opp_units)
                timers["collect_enemy_pos"] += time.perf_counter() - t0
                counts["collect_enemy_pos"] += 1

                t0 = time.perf_counter()
                execute_charge_movement(active, charge_target, board, enemy_positions)
                timers["charge_movement"] += time.perf_counter() - t0
                counts["charge_movement"] += 1

                t0 = time.perf_counter()
                execute_counter_charge(charge_target, active, board)
                timers["counter_charge"] += time.perf_counter() - t0
                counts["counter_charge"] += 1

                if active.unit.impact > 0:
                    t0 = time.perf_counter()
                    resolve_impact(active, charge_target)
                    timers["resolve_impact"] += time.perf_counter() - t0
                    counts["resolve_impact"] += 1
                    t0 = time.perf_counter()
                    _sync_dead_models(charge_target, board)
                    timers["sync_dead"] += time.perf_counter() - t0
                    counts["sync_dead"] += 1

                charger_wounds = 0
                if charge_target.models_alive > 0:
                    t0 = time.perf_counter()
                    charger_wounds = resolve_melee(active, charge_target, is_charge=True) or 0
                    timers["resolve_melee"] += time.perf_counter() - t0
                    counts["resolve_melee"] += 1
                    t0 = time.perf_counter()
                    _sync_dead_models(charge_target, board)
                    timers["sync_dead"] += time.perf_counter() - t0
                    counts["sync_dead"] += 1

                defender_wounds = 0
                if active.models_alive > 0 and charge_target.models_alive > 0:
                    t0 = time.perf_counter()
                    defender_wounds = resolve_melee(charge_target, active, is_strike_back=True) or 0
                    timers["resolve_melee"] += time.perf_counter() - t0
                    counts["resolve_melee"] += 1
                    t0 = time.perf_counter()
                    _sync_dead_models(active, board)
                    timers["sync_dead"] += time.perf_counter() - t0
                    counts["sync_dead"] += 1

                if active.models_alive > 0 and charge_target.models_alive > 0:
                    t0 = time.perf_counter()
                    check_melee_morale(active, charger_wounds, defender_wounds)
                    check_melee_morale(charge_target, defender_wounds, charger_wounds)
                    timers["melee_morale"] += time.perf_counter() - t0
                    counts["melee_morale"] += 1
                    t0 = time.perf_counter()
                    _sync_dead_models(active, board)
                    _sync_dead_models(charge_target, board)
                    timers["sync_dead"] += time.perf_counter() - t0
                    counts["sync_dead"] += 1

                active.fatigued = True
                if charge_target.models_alive > 0:
                    charge_target.fatigued = True

                if active.models_alive > 0 and charge_target.models_alive > 0:
                    t0 = time.perf_counter()
                    enemy_positions = _collect_enemy_positions(opp_units)
                    post_melee_separation(active, charge_target, board, enemy_positions)
                    timers["post_melee_sep"] += time.perf_counter() - t0
                    counts["post_melee_sep"] += 1
                elif active.models_alive > 0:
                    t0 = time.perf_counter()
                    consolidation_move(active, board, opp_units, OBJECTIVES, "objectives")
                    timers["consolidation"] += time.perf_counter() - t0
                    counts["consolidation"] += 1
                elif charge_target.models_alive > 0:
                    t0 = time.perf_counter()
                    consolidation_move(charge_target, board, my_units, OBJECTIVES, "objectives")
                    timers["consolidation"] += time.perf_counter() - t0
                    counts["consolidation"] += 1

            elif action in ("advance", "rush") and goal is not None:
                t0 = time.perf_counter()
                budget = active.unit.advance_distance if action == "advance" else active.unit.rush_distance
                enemy_positions = _collect_enemy_positions(opp_units)
                timers["collect_enemy_pos"] += time.perf_counter() - t0
                counts["collect_enemy_pos"] += 1

                t0 = time.perf_counter()
                execute_movement(active, goal, budget, board, enemy_positions,
                                 flying=active.unit.flying)
                timers["execute_movement"] += time.perf_counter() - t0
                counts["execute_movement"] += 1

                if action != "rush":
                    if active.shaken:
                        active.shaken = False
                    else:
                        t0 = time.perf_counter()
                        target = pick_target(active, opp_units)
                        timers["pick_target"] += time.perf_counter() - t0
                        counts["pick_target"] += 1
                        if target is not None:
                            t0 = time.perf_counter()
                            resolve_shooting(active, target)
                            timers["resolve_shooting"] += time.perf_counter() - t0
                            counts["resolve_shooting"] += 1
                            t0 = time.perf_counter()
                            check_morale(target)
                            timers["check_morale"] += time.perf_counter() - t0
                            counts["check_morale"] += 1
                            t0 = time.perf_counter()
                            _sync_dead_models(target, board)
                            timers["sync_dead"] += time.perf_counter() - t0
                            counts["sync_dead"] += 1

            elif action == "hold":
                if active.shaken:
                    active.shaken = False
                else:
                    t0 = time.perf_counter()
                    target = pick_target(active, opp_units)
                    timers["pick_target"] += time.perf_counter() - t0
                    counts["pick_target"] += 1
                    if target is not None:
                        t0 = time.perf_counter()
                        resolve_shooting(active, target)
                        timers["resolve_shooting"] += time.perf_counter() - t0
                        counts["resolve_shooting"] += 1
                        t0 = time.perf_counter()
                        check_morale(target)
                        timers["check_morale"] += time.perf_counter() - t0
                        counts["check_morale"] += 1
                        t0 = time.perf_counter()
                        _sync_dead_models(target, board)
                        timers["sync_dead"] += time.perf_counter() - t0
                        counts["sync_dead"] += 1

            opp_alive = any(u.models_alive > 0 for u in opp_units)
            if not opp_alive:
                break
            current_is_a = not current_is_a

        t0 = time.perf_counter()
        board.update_objectives(units_a, units_b)
        timers["obj_update"] += time.perf_counter() - t0
        counts["obj_update"] += 1

    return timers, counts, total_activations


def main():
    N = 30
    print(f"=== Game Internals Profiler ({N} games) ===\n")

    all_timers = None
    all_counts = None
    game_times = []
    total_activations_list = []

    for i in range(N):
        t0 = time.perf_counter()
        timers, counts, activations = timed_game_with_breakdown()
        game_time = time.perf_counter() - t0
        game_times.append(game_time)
        total_activations_list.append(activations)

        if all_timers is None:
            all_timers = dict(timers)
            all_counts = dict(counts)
        else:
            for k in timers:
                all_timers[k] += timers[k]
                all_counts[k] += counts[k]

    total_game_time = sum(game_times)
    avg_game = statistics.mean(game_times)
    avg_activations = statistics.mean(total_activations_list)

    print(f"Avg game time: {avg_game*1000:.1f}ms | Avg activations/game: {avg_activations:.0f}")
    print(f"Total time across {N} games: {total_game_time:.2f}s\n")

    # Sort by total time
    sorted_timers = sorted(all_timers.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Subsystem':<25} {'Total':>8} {'%':>6} {'Calls':>7} {'Avg/call':>10}")
    print("-" * 60)
    for name, total in sorted_timers:
        pct = 100 * total / total_game_time if total_game_time > 0 else 0
        call_count = all_counts[name]
        avg_per_call = (total / call_count * 1000) if call_count > 0 else 0
        print(f"  {name:<23} {total*1000:>7.1f}ms {pct:>5.1f}% {call_count:>7} {avg_per_call:>8.3f}ms")

    accounted = sum(all_timers.values())
    overhead = total_game_time - accounted
    print(f"  {'(overhead/unmeasured)':<23} {overhead*1000:>7.1f}ms {100*overhead/total_game_time:>5.1f}%")

    # Also run cProfile on a few games for call-level detail
    print(f"\n\n=== cProfile: 10 full training episodes ===\n")
    model = TacticalModel()
    model.eval()

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(10):
        res_a, res_b, states_a, states_b, *_ = _generate_army_pair()
        states_a_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_a]
        states_b_data = [(u.ai_role, u.combat_preference, u.assigned_objective) for u in states_b]
        game_specs = [(res_a, res_b, states_a_data, states_b_data, "heuristic", -1, "random")]
        _run_games_batched_tactical(model, game_specs, {})
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(40)
    print(s.getvalue())


if __name__ == "__main__":
    main()
