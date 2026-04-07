"""CLI entry point: output formatting, frequency analysis, and run loop."""
from __future__ import annotations

import copy
import csv
import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

from models import ArmyList, resolve_entry
from templates import get_templates_dict
from concurrent.futures import ProcessPoolExecutor

from evolution import (
    POINTS_BUDGET, POPULATION_SIZE, GENERATIONS, GAMES_PER_MATCHUP,
    SWISS_ROUNDS, TIME_LIMIT, _WORKER_COUNT,
    HOF_EVAL_INTERVAL,
    generate_random_army, evaluate_population, next_generation,
    resolve_army, _make_unit_states, forceorg_limits,
    HallOfFame, _init_evo_worker, _init_evo_ml_worker, _g_evo_ml_model,
)
from game import simulate_game
from ml_training import TrainingConfig, run_training, load_model_state_dict

ROLE_LABELS = {
    "killer": "Killer",
    "objective_clearer": "Obj-Clearer",
    "objective_holder": "Obj-Holder",
    "home_objective_holder": "Home-Holder",
}


# ===================================================================
# DISPLAY & OUTPUT
# ===================================================================

def format_army(army: ArmyList, mode: str = "objectives",
                enforce_forceorg: bool = False) -> str:
    lines = [f"=== Army List ({army.total_cost}pts) ==="]
    td = get_templates_dict()

    for entry in sorted(army.entries, key=lambda e: e.computed_cost, reverse=True):
        resolved = resolve_entry(entry)
        tpl = td[entry.template_id]
        upgrade_strs = []
        for slot in tpl.upgrade_slots:
            if slot.id in entry.chosen_upgrades:
                opt_id = entry.chosen_upgrades[slot.id]
                label = opt_id
                for opt in slot.options:
                    if opt.id == opt_id:
                        label = opt.description or opt.id
                        break
                upgrade_strs.append(label)
        name = f"{tpl.name} [{tpl.size}]"
        if upgrade_strs:
            name += " + " + " + ".join(upgrade_strs)

        dots = "." * max(1, 45 - len(name))
        pref = entry.combat_preference
        pref_tag = " (M)" if pref == "melee" else ""
        hero_tag = ""
        if entry.attached_to >= 0:
            hero_tag = f" -> #{entry.attached_to}"
        if mode == "kill_points":
            lines.append(f"  {name} {dots} {resolved.points}pts{pref_tag}{hero_tag}")
        else:
            role_label = ROLE_LABELS.get(entry.ai_role, entry.ai_role)
            lines.append(f"  {name} {dots} {resolved.points}pts  [{role_label}]{pref_tag}{hero_tag}")
    total_models = sum(td[e.template_id].size for e in army.entries)
    lines.append(f"  ---")
    lines.append(f"  Total: {army.total_cost}pts | Units: {len(army.entries)} | Models: {total_models}")
    lines.append(f"  Win rate: {army.fitness:.3f}")

    if enforce_forceorg:
        limits = forceorg_limits(POINTS_BUDGET)
        hero_count = sum(1 for e in army.entries if td[e.template_id].hero)
        from evolution import _attached_hero_count
        attached_heroes = _attached_hero_count(army)
        effective_entries = len(army.entries) - attached_heroes
        largest_cost = max((e.computed_cost for e in army.entries), default=0)
        lines.append(f"  [Force Org: {hero_count} heroes/{limits['max_heroes']} max | "
                     f"{effective_entries} entries/{limits['max_entries']} max | "
                     f"largest unit {largest_cost}pts/{limits['max_unit_cost']} max]")

    return "\n".join(lines)


def unit_frequency_analysis(population: list[ArmyList], top_n: int = 30,
                            mode: str = "objectives"):
    ranked = sorted(population, key=lambda a: a.fitness, reverse=True)[:top_n]

    # Unit type frequency
    unit_counts: dict[str, list[int]] = {}  # template_id -> [count per list]
    upgrade_counts: dict[str, int] = {}     # "template + upgrade" -> count of lists

    # Role distribution
    role_totals: dict[str, int] = {"killer": 0, "objective_clearer": 0, "objective_holder": 0, "home_objective_holder": 0}
    total_units = 0
    # Unit-role mapping: "template_id" -> {role -> count}
    unit_role_counts: dict[str, dict[str, int]] = {}

    for army in ranked:
        seen: dict[str, int] = {}
        for entry in army.entries:
            seen[entry.template_id] = seen.get(entry.template_id, 0) + 1
            tpl = get_templates_dict()[entry.template_id]
            for slot_id, opt_id in entry.chosen_upgrades.items():
                label = opt_id
                for slot in tpl.upgrade_slots:
                    if slot.id == slot_id:
                        for opt in slot.options:
                            if opt.id == opt_id:
                                label = opt.description or opt.id
                                break
                        break
                key = f"{tpl.name} + {label}"
                upgrade_counts[key] = upgrade_counts.get(key, 0) + 1

            # Track roles
            role_totals[entry.ai_role] = role_totals.get(entry.ai_role, 0) + 1
            total_units += 1
            if entry.template_id not in unit_role_counts:
                unit_role_counts[entry.template_id] = {}
            urc = unit_role_counts[entry.template_id]
            urc[entry.ai_role] = urc.get(entry.ai_role, 0) + 1

        for tid in get_templates_dict():
            if tid not in unit_counts:
                unit_counts[tid] = []
            unit_counts[tid].append(seen.get(tid, 0))

    lines = [f"\n=== Unit Frequency (Top {top_n} Lists) ==="]
    templates_dict = get_templates_dict()
    freq_data = []
    for tid, counts in unit_counts.items():
        present = sum(1 for c in counts if c > 0)
        avg_copies = sum(counts) / len(counts) if counts else 0
        freq_data.append((templates_dict[tid].name, present, top_n, avg_copies, tid))

    freq_data.sort(key=lambda x: x[1], reverse=True)
    for name, present, total, avg, _ in freq_data:
        dots = "." * max(1, 30 - len(name))
        lines.append(f"  {name} {dots} {present}/{total} ({100*present/total:.0f}%)  avg {avg:.1f} copies")

    lines.append(f"\n=== Upgrade Frequency (Top {top_n} Lists) ===")
    for key, count in sorted(upgrade_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {key} .... {count}/{top_n}")

    # Role distribution (objectives mode only)
    if mode != "kill_points":
        lines.append(f"\n=== Role Distribution (Top {top_n} Lists) ===")
        if total_units > 0:
            for role in ["killer", "objective_clearer", "objective_holder", "home_objective_holder"]:
                count = role_totals.get(role, 0)
                avg_per_list = count / top_n
                pct = 100 * count / total_units
                label = ROLE_LABELS.get(role, role)
                dots = "." * max(1, 20 - len(label))
                lines.append(f"  {label} {dots} avg {avg_per_list:.1f} units/list ({pct:.0f}%)")

        # Unit-role breakdown (which units are most commonly each role)
        lines.append(f"\n=== Unit-Role Assignments (Top {top_n} Lists) ===")
        for name, present, total, avg, tid in freq_data:
            if present == 0:
                continue
            urc = unit_role_counts.get(tid, {})
            total_appearances = sum(urc.values())
            if total_appearances == 0:
                continue
            role_parts = []
            for role in ["killer", "objective_clearer", "objective_holder", "home_objective_holder"]:
                rc = urc.get(role, 0)
                if rc > 0:
                    rpct = 100 * rc / total_appearances
                    role_parts.append(f"{ROLE_LABELS[role]}:{rpct:.0f}%")
            if role_parts:
                lines.append(f"  {name}: {', '.join(role_parts)}")

    never = [name for name, present, _, _, _ in freq_data if present == 0]
    if never:
        lines.append(f"\n=== Never Selected ===")
        for name in never:
            lines.append(f"  {name} (0/{top_n})")

    return "\n".join(lines)


# ===================================================================
# BASELINE EVALUATION
# ===================================================================

def _play_single_game(args):
    """Worker for parallel baseline games."""
    army_a, army_b, res_a, res_b, mode, *rest = args
    use_ml = rest[0] if rest else False
    ml_batch_tactical = rest[1] if len(rest) > 1 else True
    ml_kw = {}
    if use_ml and _g_evo_ml_model is not None:
        ml_kw = {'ml_model_a': _g_evo_ml_model, 'ml_model_b': _g_evo_ml_model,
                 'ml_batch_tactical': ml_batch_tactical}
    sa = _make_unit_states(army_a, res_a, "A")
    sb = _make_unit_states(army_b, res_b, "B")
    return simulate_game(res_a, res_b, mode=mode, states_a=sa, states_b=sb, **ml_kw)


def _win_rate_vs_baseline(army: ArmyList, resolved_army: list,
                          baseline: list[ArmyList],
                          baseline_resolved: list[list],
                          mode: str,
                          pool: ProcessPoolExecutor | None = None,
                          sample_size: int = 0,
                          use_ml: bool = False,
                          ml_batch_tactical: bool = True) -> float:
    """Play army against baseline members. Return win rate.
    If sample_size > 0, play against a random sample instead of all."""
    n = len(baseline)
    if 0 < sample_size < n:
        indices = random.sample(range(n), sample_size)
    else:
        indices = list(range(n))

    work = [(army, baseline[bi], resolved_army, baseline_resolved[bi], mode,
             use_ml, ml_batch_tactical)
            for bi in indices]

    if pool is not None:
        results = list(pool.map(_play_single_game, work, chunksize=10))
    else:
        with ProcessPoolExecutor(max_workers=_WORKER_COUNT) as _pool:
            results = list(_pool.map(_play_single_game, work, chunksize=10))

    wins = 0.0
    for result in results:
        if result == "A":
            wins += 1
        elif result == "draw":
            wins += 0.5
    return wins / len(indices)


def _mean_fitness_vs_baseline(population: list[ArmyList],
                              baseline: list[ArmyList],
                              baseline_resolved: list[list],
                              mode: str,
                              opponents: int = 5,
                              pool: ProcessPoolExecutor | None = None,
                              sample_size: int = 0,
                              use_ml: bool = False,
                              ml_batch_tactical: bool = True) -> float:
    """Each member plays 1 game against `opponents` random baseline members.
    If sample_size > 0, only evaluate a random sample of the population."""
    if 0 < sample_size < len(population):
        pop_sample = random.sample(population, sample_size)
    else:
        pop_sample = population

    # Build all work items up front
    work = []
    # Track which games belong to which army (army_index, num_games)
    army_game_counts: list[int] = []
    for army in pop_sample:
        res = resolve_army(army)
        sample = random.sample(range(len(baseline)), min(opponents, len(baseline)))
        army_game_counts.append(len(sample))
        for bi in sample:
            work.append((army, baseline[bi], res, baseline_resolved[bi], mode,
                         use_ml, ml_batch_tactical))

    if pool is not None:
        results = list(pool.map(_play_single_game, work, chunksize=10))
    else:
        with ProcessPoolExecutor(max_workers=_WORKER_COUNT) as _pool:
            results = list(_pool.map(_play_single_game, work, chunksize=10))

    total = 0.0
    idx = 0
    for num_games in army_game_counts:
        wins = 0.0
        for _ in range(num_games):
            result = results[idx]
            if result == "A":
                wins += 1
            elif result == "draw":
                wins += 0.5
            idx += 1
        total += wins / num_games
    return total / len(pop_sample)


ML_ROLES = {"killer", "home_objective_holder"}

_ML_ROLE_REMAP = {
    "objective_clearer": "killer",
    "objective_holder": "home_objective_holder",
}


def _restrict_ml_roles(army: ArmyList):
    """Remap unit roles to only killer and home_objective_holder for ML mode."""
    for entry in army.entries:
        if entry.ai_role not in ML_ROLES:
            entry.ai_role = _ML_ROLE_REMAP.get(entry.ai_role, "killer")


# ===================================================================
# MAIN
# ===================================================================

def run_list_evolution(graphic: bool = False, mode: str = "objectives",
         enforce_forceorg: bool = False, use_ml: bool = False,
         ml_batch_tactical: bool = True,
         restart_evolution: bool = True,
         use_c_ext: bool = True):
    start_time = time.time()

    # --- C extension setup ---
    import fast_core
    c_available = fast_core.is_available()
    c_active = use_c_ext and c_available
    fast_core.USE_C_EXT = c_active
    c_label = "ON" if c_active else ("OFF (not compiled)" if use_c_ext and not c_available else "OFF")

    # --- ML model setup ---
    ml_model_path: str | None = None
    ml_model_type: str | None = None
    if use_ml:
        model_path = Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"ML model not found: {model_path}")
        from ml_model_tactical import TacticalModel
        sd = load_model_state_dict(model_path)
        test_model = TacticalModel()
        test_model.load_state_dict(sd, strict=False)
        ml_model_type = "tactical"
        del test_model
        ml_model_path = str(model_path)
        print(f"ML model loaded: {ml_model_type} ({model_path.name})")

    mode_label = "Kill Points" if mode == "kill_points" else "Objectives"
    forceorg_label = " | Force Org: ON" if enforce_forceorg else ""
    batch_tag = " batched" if ml_batch_tactical else ""
    ml_label = f" | ML: {ml_model_type}{batch_tag}" if use_ml else ""
    print(f"High Elf Fleets -- Grid Tactical Evolutionary Optimizer ({mode_label}{forceorg_label}{ml_label})")
    print(f"C extension: {c_label}")
    print(f"Population: {POPULATION_SIZE} | Generations: {GENERATIONS} | "
          f"Budget: {POINTS_BUDGET}pts")
    print(f"Games per matchup: {GAMES_PER_MATCHUP} | "
          f"Swiss rounds: {SWISS_ROUNDS}")
    if TIME_LIMIT is not None:
        print(f"Time limit: {TIME_LIMIT} minutes")
    print("=" * 70)

    # Initialize
    population = [generate_random_army(mode=mode, enforce_forceorg=enforce_forceorg)
                  for _ in range(POPULATION_SIZE)]
    if use_ml:
        for army in population:
            _restrict_ml_roles(army)

    # Freeze gen-0 baseline for fitness measurement
    baseline = copy.deepcopy(population)
    baseline_resolved = [resolve_army(a) for a in baseline]

    log_rows: list[dict] = []
    gen_times: list[float] = []
    gen_snapshots: list[list[ArmyList]] = []  # snapshots of populations for baseline mixing
    if restart_evolution:
        hall_of_fame = HallOfFame()
    else:
        hof_file = "hall_of_fame_ml.json" if use_ml else "hall_of_fame.json"
        hof_path = Path(__file__).parent / "results" / hof_file
        hall_of_fame = HallOfFame.load_from_json(hof_path)
        if hall_of_fame.entries:
            print(f"Loaded Hall of Fame: {len(hall_of_fame.entries)} entries from {hof_file} "
                  f"(top fitness {hall_of_fame.entries[0].fitness:.3f})")
        else:
            print(f"No existing Hall of Fame found at {hof_path}, starting fresh.")

    pool_kwargs: dict = {'max_workers': _WORKER_COUNT}
    if use_ml:
        pool_kwargs['initializer'] = _init_evo_ml_worker
        pool_kwargs['initargs'] = (ml_model_path, ml_model_type, use_c_ext)
    else:
        pool_kwargs['initializer'] = _init_evo_worker
        pool_kwargs['initargs'] = (use_c_ext,)

    with ProcessPoolExecutor(**pool_kwargs) as pool:
      for gen in range(1, GENERATIONS + 1):
        gen_start = time.time()
        evaluate_population(population, mode=mode, pool=pool, use_ml=use_ml,
                            ml_batch_tactical=ml_batch_tactical,
                            bench=(gen == 1),
                            ml_coroutine_batch=(use_ml and not ml_batch_tactical))
        gen_time = time.time() - gen_start

        # Snapshot population for baseline mixing (first 25 generations)
        if gen <= 25:
            gen_snapshots.append(copy.deepcopy(population))

        # After generation 25, rebuild baseline as even mix from first 25 generations
        if gen == 25:
            per_gen = max(1, len(baseline) // 25)
            mixed: list[ArmyList] = []
            for snapshot in gen_snapshots:
                mixed.extend(random.sample(snapshot, min(per_gen, len(snapshot))))
            # Trim or pad to match original baseline size
            if len(mixed) > len(baseline):
                mixed = mixed[:len(baseline)]
            elif len(mixed) < len(baseline):
                # Fill remaining slots by cycling through snapshots
                i = 0
                while len(mixed) < len(baseline):
                    extra = gen_snapshots[i % len(gen_snapshots)]
                    mixed.append(random.choice(extra))
                    i += 1
            baseline = mixed
            baseline_resolved = [resolve_army(a) for a in baseline]
            print(f"--- Baseline updated: now an even mix of lists from generations 1-25 ---")

        # Best fitness: top performer from intra-population tournament,
        # then measured as win rate vs a sample of baseline members
        gen_best = max(population, key=lambda a: a.fitness)
        best_resolved = resolve_army(gen_best)
        best_vs_baseline = _win_rate_vs_baseline(
            gen_best, best_resolved, baseline, baseline_resolved, mode,
            pool=pool, sample_size=20, use_ml=use_ml,
            ml_batch_tactical=ml_batch_tactical)

        # Mean fitness: a sample of population members play 3 random baseline opponents
        mean_vs_baseline = _mean_fitness_vs_baseline(
            population, baseline, baseline_resolved, mode, opponents=3,
            pool=pool, sample_size=20, use_ml=use_ml,
            ml_batch_tactical=ml_batch_tactical)

        # Hall of Fame evaluation
        hof_info = None
        if gen % HOF_EVAL_INTERVAL == 0:
            hof_info = hall_of_fame.try_promote(population, mode=mode,
                                                pool=pool, generation=gen,
                                                ml_coroutine_batch=(use_ml and not ml_batch_tactical))
            print(f"--- Hall of Fame: {hof_info['candidates']} evaluated, "
                  f"{hof_info['promoted']} promoted, {hof_info['demoted']} demoted | "
                  f"{hof_info['size']} members | top {hof_info['top_fitness']:.3f} ---")

        log_rows.append({
            'generation': gen,
            'best_fitness': round(best_vs_baseline, 4),
            'mean_fitness': round(mean_vs_baseline, 4),
            'best_cost': gen_best.total_cost,
            'best_units': len(gen_best.entries),
            'time_sec': round(gen_time, 1),
            'hof_size': len(hall_of_fame.entries),
            'hof_top_fitness': round(hall_of_fame.entries[0].fitness, 4) if hall_of_fame.entries else '',
        })

        gen_times.append(gen_time)
        recent_avg = sum(gen_times[-5:]) / len(gen_times[-5:])
        gens_remaining = GENERATIONS - gen
        est_remaining = recent_avg * gens_remaining
        if TIME_LIMIT is not None:
            time_limit_remaining = TIME_LIMIT * 60 - (time.time() - start_time)
            est_remaining = min(est_remaining, max(0, time_limit_remaining))
        eft = datetime.now() + timedelta(seconds=est_remaining)
        eft_str = eft.strftime("%H:%M")

        print(f"Gen {gen:03d} | Best: {best_vs_baseline:.3f} | "
              f"Mean: {mean_vs_baseline:.3f} | "
              f"{gen_time:.1f}s | EFT {eft_str}")

        if gen % 25 == 0:
            print(format_army(gen_best, mode=mode, enforce_forceorg=enforce_forceorg))
            print()

        elapsed = time.time() - start_time
        if TIME_LIMIT is not None and elapsed >= TIME_LIMIT * 60:
            print(f"\nTIME LIMIT reached ({TIME_LIMIT} min) after generation {gen}.")
            break

        population = next_generation(population, mode=mode,
                                     enforce_forceorg=enforce_forceorg)
        if use_ml:
            for army in population:
                _restrict_ml_roles(army)

    total_time = time.time() - start_time

    # Final output
    print("\n" + "=" * 70)
    print(f"OPTIMIZATION COMPLETE — {total_time:.0f}s total")
    print("=" * 70)

    # Use Hall of Fame as the source for end-of-run stats when available,
    # fall back to final generation if HoF is empty.
    if hall_of_fame.entries:
        hof_armies = [e.army for e in hall_of_fame.entries]
        stats_source = hof_armies
        stats_label = "Hall of Fame"
    else:
        stats_source = sorted(population, key=lambda a: a.fitness, reverse=True)
        stats_label = "final generation"

    # Top 5 distinct
    print(f"\n--- TOP 5 LISTS ({stats_label}) ---")
    seen_sigs: set[tuple] = set()
    shown = 0
    for army in stats_source:
        sig = (army.total_cost, len(army.entries),
               tuple(sorted(e.template_id for e in army.entries)))
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            print(format_army(army, mode=mode, enforce_forceorg=enforce_forceorg))
            print()
            shown += 1
            if shown >= 5:
                break

    # Frequency analysis (top 30 from HoF or population)
    freq = unit_frequency_analysis(stats_source, top_n=min(30, len(stats_source)),
                                   mode=mode)
    print(freq)

    # --- Write results to files ---
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    # Generation log CSV
    log_path = out_dir / "evo_generations.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader()
        w.writerows(log_rows)

    # Best army JSON (top army from stats source)
    top_army = stats_source[0]
    best_path = out_dir / "evo_best_army.json"
    best_data = {
        'total_cost': top_army.total_cost,
        'fitness': top_army.fitness,
        'entries': [
            {
                'template_id': e.template_id,
                'upgrades': e.chosen_upgrades,
                'ai_role': e.ai_role,
                'combat_preference': e.combat_preference,
                'attached_to': e.attached_to,
                'cost': e.computed_cost,
            }
            for e in top_army.entries
        ],
    }
    with open(best_path, "w") as f:
        json.dump(best_data, f, indent=2)

    # Frequency analysis text
    freq_path = out_dir / "evo_frequency.txt"
    with open(freq_path, "w") as f:
        f.write(freq)

    # Hall of Fame JSON
    if hall_of_fame.entries:
        hof_path = out_dir / ("hall_of_fame_ml.json" if use_ml else "hall_of_fame.json")
        hof_data = [
            {
                'rank': i + 1,
                'fitness': entry.fitness,
                'generation_added': entry.generation_added,
                'total_cost': entry.army.total_cost,
                'entries': [
                    {
                        'template_id': e.template_id,
                        'upgrades': e.chosen_upgrades,
                        'ai_role': e.ai_role,
                        'combat_preference': e.combat_preference,
                        'attached_to': e.attached_to,
                        'cost': e.computed_cost,
                    }
                    for e in entry.army.entries
                ],
            }
            for i, entry in enumerate(hall_of_fame.entries)
        ]
        with open(hof_path, "w") as f:
            json.dump(hof_data, f, indent=2)
    else:
        hof_path = None

    result_paths = f"\nResults written to:\n  {log_path}\n  {best_path}\n  {freq_path}"
    if hof_path:
        result_paths += f"\n  {hof_path}"
    print(result_paths)

    if graphic:
        ml_model = None
        if use_ml and ml_model_path is not None:
            from ml_model_tactical import TacticalModel
            ml_model = TacticalModel()
            ml_model.load_state_dict(
                load_model_state_dict(ml_model_path), strict=False)
            ml_model.eval()
        _run_showcase_game(top_army, mode=mode, enforce_forceorg=enforce_forceorg,
                           ml_model=ml_model)


def _run_showcase_game(top_army: ArmyList, mode: str = "objectives",
                       enforce_forceorg: bool = False, ml_model=None):
    """Play the top army vs a random army (replay until top wins), then show viewer.
    ml_model: if provided, Player A uses this ML model instead of heuristic AI.
    """
    from evolution import generate_random_army, resolve_army, _make_unit_states
    from game import simulate_game_recorded

    top_resolved = resolve_army(top_army)

    print("\n--- SHOWCASE GAME ---")
    ai_type = "ML model" if ml_model is not None else "heuristic"
    print(f"Playing top list ({ai_type}) vs random opponent (replaying until top list wins)...")

    attempts = 0
    while True:
        attempts += 1
        opponent = generate_random_army(mode=mode, enforce_forceorg=enforce_forceorg)
        opp_resolved = resolve_army(opponent)

        sa = _make_unit_states(top_army, top_resolved, "A")
        sb = _make_unit_states(opponent, opp_resolved, "B")

        result, frames, labels, owners, unit_points, unit_info = simulate_game_recorded(
            top_resolved, opp_resolved, mode=mode, states_a=sa, states_b=sb,
            ml_model_a=ml_model)

        if result == "A":
            print(f"Top list wins after {attempts} attempt{'s' if attempts != 1 else ''}!")
            break

        if attempts >= 100:
            print(f"Could not find a win in 100 attempts, showing last game.")
            break

    print(f"Launching game viewer ({len(frames)} frames)...")
    from viewer import show_game
    show_game(frames, labels, owners, mode=mode, unit_points=unit_points,
             unit_info=unit_info)


def ml_train(num_batches: int = 20, batch_size: int = 128, verbose: bool = True,
             time_limit: float | None = None,
             model_type: str = "tactical",
             use_c_ext: bool = True,
             restart_training: bool = False,
             entropy_coeff_start: float = 0.01,
             entropy_coeff_end: float = 0.01,
             memory_max: str | None = None,
             memory_swap_max: str | None = None,
             worker_count: int | None = 6,
             device: str = "auto",
             planning_rate: float = 0.0,
             planning_rate_end: float | None = None,
             minibatch_size: int = 64,
             blend_ratio: float = 0.0):
    """Run a short ML training run and print summary stats.

    use_c_ext: if True (default), use the compiled C extension for hot loops
               in movement, combat, and feature encoding.  Set False to force
               pure-Python paths.  Requires running build_fast_core.py first.
    memory_max: if set (e.g. "12G"), re-exec under systemd-run with a cgroup
                memory limit so the training cannot freeze the system.
    memory_swap_max: swap limit for the cgroup (e.g. "2G"), requires memory_max.
    worker_count: number of multiprocessing pool workers (default: cpu_count // 2).
    planning_rate: probability of planning per activation (0 = disabled, 0.05 typical).
    planning_rate_end: if set, anneal planning_rate linearly to this value over training.
    """
    # --- Re-exec under cgroup memory limit if requested ---
    if memory_max is not None and os.environ.get("_ML_TRAIN_CGROUP") != "1":
        import subprocess, sys
        cmd = ["systemd-run", "--user", "--scope",
               "-p", f"MemoryMax={memory_max}"]
        if memory_swap_max is not None:
            cmd += ["-p", f"MemorySwapMax={memory_swap_max}"]
        env = os.environ.copy()
        env["_ML_TRAIN_CGROUP"] = "1"
        cmd += [sys.executable] + sys.argv
        print(f"Re-launching under cgroup: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)

    import fast_core
    c_available = fast_core.is_available()
    c_active = use_c_ext and c_available
    c_label = "ON" if c_active else ("OFF (not compiled)" if use_c_ext and not c_available else "OFF")

    print("=" * 70)
    print(f"ML Game Player — Short Training Run ({model_type})")
    print(f"Batches: {num_batches} | Batch size: {batch_size} (games per batch)")
    print(f"Total games: {num_batches * batch_size}")
    print(f"C extension: {c_label}")
    if worker_count is not None:
        print(f"Workers: {worker_count}")
    if memory_max is not None:
        mem_label = f"MemoryMax={memory_max}"
        if memory_swap_max is not None:
            mem_label += f", MemorySwapMax={memory_swap_max}"
        print(f"Cgroup limits: {mem_label}")
    if time_limit is not None:
        print(f"Time limit: {time_limit} minutes")
    print("=" * 70)

    start = time.time()
    config = TrainingConfig(
        num_batches=num_batches,
        batch_size=batch_size,
        time_limit=time_limit,
        checkpoint_dir="ml_checkpoints",
        model_type=model_type,
        use_c_ext=use_c_ext,
        entropy_coeff_start=entropy_coeff_start,
        entropy_coeff_end=entropy_coeff_end,
        worker_count=worker_count,
        device=device,
        planning_rate=planning_rate,
        planning_rate_end=planning_rate_end,
        ppo_minibatch_games=minibatch_size,
        unit_local_advantage_blend=blend_ratio,
    )
    model, metrics = run_training(config=config, verbose=verbose,
                                   restart=restart_training)
    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE — {elapsed:.1f}s")
    print(f"  Heuristic win rate: {metrics.heuristic_win_rate:.3f}")
    print(f"  Self-play win rate: {metrics.selfplay_win_rate:.3f}")
    if metrics.batch_logs:
        last = metrics.batch_logs[-1]
        print(f"  Final loss: {last['loss']:.4f} | Entropy: {last['mean_entropy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    #ml_train(num_batches=300000, batch_size=512, time_limit=(180), model_type="tactical", use_c_ext=True, restart_training=True, memory_max="14G", memory_swap_max="40G", worker_count = 4, planning_rate = 0, minibatch_size = 128, blend_ratio = 0.25)
    ml_train(num_batches=300000, batch_size=512, time_limit=(300), model_type="tactical", use_c_ext=True, restart_training=False, memory_max="14G", memory_swap_max="40G", worker_count = 4, planning_rate = 0, minibatch_size = 128, blend_ratio = 0.25)
    #ml_train(num_batches=3, batch_size=256, time_limit=2, model_type="tactical", use_c_ext=False)  # pure Python
    #run_list_evolution(graphic=True, mode="objectives", enforce_forceorg=True, use_ml=True, ml_batch_tactical=False, restart_evolution=False, use_c_ext=True)
    #from play_viewer import play_interactive
    #play_interactive()