"""Planning parameter sweep: ML+Planning vs ML (no planning).

Both sides use the same tactical model with random hall_of_fame_ml.json armies.
Tests increasingly expensive planning parameter combinations (K, C, M, N),
50 games each, stopping after the first test that finishes past the 90-minute mark.

Tracks: win rate (planning side), draw rate, avg time per game.
"""

import json
import random
import time
from pathlib import Path

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList
from game import simulate_game

_DIR = Path(__file__).resolve().parent
TOTAL_TIME_LIMIT = 90 * 60  # set to 0 for now to ensure that only baseline test is run
GAMES_PER_TEST = 50


def load_army_from_hof(hof_entry: dict) -> ArmyList:
    army = ArmyList()
    for e in hof_entry["entries"]:
        entry = make_entry(
            e["template_id"],
            upgrades=e.get("upgrades", {}),
            ai_role=e.get("ai_role", "killer"),
        )
        entry.combat_preference = e.get("combat_preference", "ranged")
        army.entries.append(entry)
    return army


def run_test(label, planning_params, model, hof_ml_data, num_games=GAMES_PER_TEST):
    """Run num_games of ML+Planning (A) vs ML (B). Returns results dict."""
    wins = {"A": 0, "B": 0, "draw": 0}
    game_times = []

    for i in range(num_games):
        army_a = load_army_from_hof(random.choice(hof_ml_data))
        army_b = load_army_from_hof(random.choice(hof_ml_data))
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")

        t0 = time.time()
        result = simulate_game(
            res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
            ml_planning="A", planning_params=planning_params,
        )
        dt = time.time() - t0
        game_times.append(dt)
        wins[result] += 1
        print(f"  Game {i+1}/{num_games}: {dt:.1f}s", end="\r")

    avg_time = sum(game_times) / len(game_times)
    total_time = sum(game_times)
    plan_wr = (wins["A"] + 0.5 * wins["draw"]) / num_games

    return {
        "label": label,
        "params": planning_params,
        "wins": wins,
        "plan_win_rate": plan_wr,
        "avg_time": avg_time,
        "total_time": total_time,
        "game_times": game_times,
    }


def print_result(r):
    p = r["params"]
    n = sum(r["wins"].values())
    print(f"\n{'='*70}")
    print(f"  {r['label']}")
    print(f"  K={p['K_UNITS']}, C={p['C_SAMPLES_PER_UNIT']}, "
          f"M={p['M_ROLLOUTS']}, N={p.get('N_LOOKAHEAD', 4)}")
    print(f"  {n} games | avg {r['avg_time']:.2f}s/game | total {r['total_time']:.1f}s")
    print(f"  ML+Planning: {r['wins']['A']:>3} wins ({r['wins']['A']/n*100:.1f}%)")
    print(f"  ML (no plan):{r['wins']['B']:>3} wins ({r['wins']['B']/n*100:.1f}%)")
    print(f"  Draws:       {r['wins']['draw']:>3}      ({r['wins']['draw']/n*100:.1f}%)")
    print(f"  Planning win rate (draws=0.5): {r['plan_win_rate']*100:.1f}%")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Parameter combinations, ordered from cheapest to most expensive.
# Cost proxy: K * C * M * N
# ---------------------------------------------------------------------------
PARAM_COMBOS = [
    # --- Very cheap (cost <= 64) ---
    ("K2 C2 M4 N2",   {"K_UNITS": 2, "C_SAMPLES_PER_UNIT": 2, "M_ROLLOUTS": 4,  "N_LOOKAHEAD": 2}),   # 32
    ("K2 C2 M4 N4",   {"K_UNITS": 2, "C_SAMPLES_PER_UNIT": 2, "M_ROLLOUTS": 4,  "N_LOOKAHEAD": 4}),   # 64
    # --- Cheap (cost 64-256) ---
    ("K4 C2 M4 N4",   {"K_UNITS": 4, "C_SAMPLES_PER_UNIT": 2, "M_ROLLOUTS": 4,  "N_LOOKAHEAD": 4}),   # 128
    ("K4 C4 M4 N4",   {"K_UNITS": 4, "C_SAMPLES_PER_UNIT": 4, "M_ROLLOUTS": 4,  "N_LOOKAHEAD": 4}),   # 256
    # --- Medium (cost 256-1024) ---
    ("K4 C4 M16 N4",  {"K_UNITS": 4, "C_SAMPLES_PER_UNIT": 4, "M_ROLLOUTS": 16, "N_LOOKAHEAD": 4}),   # 1024
    ("K4 C4 M4 N8",   {"K_UNITS": 4, "C_SAMPLES_PER_UNIT": 4, "M_ROLLOUTS": 4,  "N_LOOKAHEAD": 8}),   # 512
    ("K6 C4 M4 N4",   {"K_UNITS": 6, "C_SAMPLES_PER_UNIT": 4, "M_ROLLOUTS": 4,  "N_LOOKAHEAD": 4}),   # 384
    # --- Expensive (cost 1024-4096) ---
    ("K4 C10 M16 N4", {"K_UNITS": 4, "C_SAMPLES_PER_UNIT": 10, "M_ROLLOUTS": 16, "N_LOOKAHEAD": 4}),  # 2560
    ("K6 C4 M16 N4",  {"K_UNITS": 6, "C_SAMPLES_PER_UNIT": 4, "M_ROLLOUTS": 16, "N_LOOKAHEAD": 4}),   # 1536
    ("K4 C4 M64 N4",  {"K_UNITS": 4, "C_SAMPLES_PER_UNIT": 4, "M_ROLLOUTS": 64, "N_LOOKAHEAD": 4}),   # 4096
    # --- Very expensive (cost > 4096) ---
    ("K4 C10 M64 N4", {"K_UNITS": 4, "C_SAMPLES_PER_UNIT": 10, "M_ROLLOUTS": 64, "N_LOOKAHEAD": 4}),  # 10240
    ("K6 C10 M64 N4", {"K_UNITS": 6, "C_SAMPLES_PER_UNIT": 10, "M_ROLLOUTS": 64, "N_LOOKAHEAD": 4}),  # 15360
]


if __name__ == "__main__":
    # Load hall of fame
    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof_ml_data = json.load(f)

    # Load model
    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model_label = "tactical"
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded {model_label} model from {checkpoint_path}")

    # --- Baseline: ML vs ML (no planning on either side) ---
    print(f"\n{'#'*70}")
    print(f"# BASELINE: ML vs ML (no planning), {GAMES_PER_TEST} games")
    print(f"{'#'*70}")
    baseline_wins = {"A": 0, "B": 0, "draw": 0}
    baseline_times = []
    for i in range(GAMES_PER_TEST):
        army_a = load_army_from_hof(random.choice(hof_ml_data))
        army_b = load_army_from_hof(random.choice(hof_ml_data))
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")
        t0 = time.time()
        result = simulate_game(
            res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
        )
        dt = time.time() - t0
        baseline_times.append(dt)
        baseline_wins[result] += 1
        print(f"  Game {i+1}/{GAMES_PER_TEST}: {dt:.1f}s", end="\r")

    baseline_avg = sum(baseline_times) / len(baseline_times)
    baseline_total = sum(baseline_times)
    print(f"\n  Baseline: avg {baseline_avg:.2f}s/game | total {baseline_total:.1f}s")
    print(f"  A: {baseline_wins['A']} | B: {baseline_wins['B']} | Draw: {baseline_wins['draw']}")
    print(f"  (Expect ~50/50 since both sides are identical ML)")

    # --- Planning parameter sweep ---
    global_start = time.time()
    # Subtract baseline time from budget
    elapsed_before_sweep = baseline_total
    results = []
    stopped_early = False

    print(f"\n{'#'*70}")
    print(f"# PLANNING PARAMETER SWEEP: ML+Planning (A) vs ML (B)")
    print(f"# {GAMES_PER_TEST} games per configuration, 90-min total budget")
    print(f"# Budget remaining: {(TOTAL_TIME_LIMIT - elapsed_before_sweep)/60:.1f} min")
    print(f"{'#'*70}")

    for label, params in PARAM_COMBOS:
        elapsed = time.time() - global_start + elapsed_before_sweep
        remaining = TOTAL_TIME_LIMIT - elapsed
        cost = (params["K_UNITS"] * params["C_SAMPLES_PER_UNIT"]
                * params["M_ROLLOUTS"] * params.get("N_LOOKAHEAD", 4))

        print(f"\n--- {label} (cost proxy: {cost}) ---")
        print(f"  Time elapsed: {elapsed/60:.1f} min | remaining: {remaining/60:.1f} min")

        if remaining <= 0:
            print(f"  SKIPPED — time budget exhausted.")
            stopped_early = True
            break

        r = run_test(label, params, model, hof_ml_data)
        results.append(r)
        print_result(r)

        # Check if we've exceeded the time limit (stop after first overrun)
        elapsed = time.time() - global_start + elapsed_before_sweep
        if elapsed > TOTAL_TIME_LIMIT:
            print(f"\n*** Time budget exceeded ({elapsed/60:.1f} min). Stopping. ***")
            stopped_early = True
            break

    # --- Summary table ---
    total_elapsed = time.time() - global_start + elapsed_before_sweep
    print(f"\n\n{'#'*70}")
    print(f"# SUMMARY — {len(results)} configurations tested in {total_elapsed/60:.1f} min")
    if stopped_early:
        print(f"# (stopped early — time budget reached)")
    print(f"{'#'*70}\n")

    print(f"{'Config':<22} {'Cost':>6} {'Avg s/game':>10} {'Plan WR%':>9} "
          f"{'W':>4} {'L':>4} {'D':>4} {'Total s':>8}")
    print("-" * 75)
    # Baseline row
    print(f"{'ML vs ML (baseline)':<22} {'—':>6} {baseline_avg:>10.2f} "
          f"{'—':>9} {baseline_wins['A']:>4} {baseline_wins['B']:>4} "
          f"{baseline_wins['draw']:>4} {baseline_total:>8.1f}")
    for r in results:
        p = r["params"]
        cost = (p["K_UNITS"] * p["C_SAMPLES_PER_UNIT"]
                * p["M_ROLLOUTS"] * p.get("N_LOOKAHEAD", 4))
        n = sum(r["wins"].values())
        print(f"{r['label']:<22} {cost:>6} {r['avg_time']:>10.2f} "
              f"{r['plan_win_rate']*100:>8.1f}% {r['wins']['A']:>4} "
              f"{r['wins']['B']:>4} {r['wins']['draw']:>4} {r['total_time']:>8.1f}")

    # Save results to JSON for later analysis
    output = {
        "baseline": {
            "wins": baseline_wins,
            "avg_time": baseline_avg,
            "total_time": baseline_total,
        },
        "tests": [
            {
                "label": r["label"],
                "params": r["params"],
                "wins": r["wins"],
                "plan_win_rate": r["plan_win_rate"],
                "avg_time": r["avg_time"],
                "total_time": r["total_time"],
            }
            for r in results
        ],
        "total_elapsed_min": total_elapsed / 60,
        "games_per_test": GAMES_PER_TEST,
        "stopped_early": stopped_early,
    }
    out_path = _DIR / "results" / "planning_param_sweep.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
