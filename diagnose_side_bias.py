"""Diagnose side-A vs side-B bias in ML vs ML games.

Runs many games with no planning, both sides identical ML model + random
hall_of_fame_ml.json armies. Tracks detailed per-game stats to find the
source of any observed asymmetry.
"""

import json
import random
import copy
import time
from pathlib import Path
from collections import defaultdict

from ml_model import StrategicModel
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList
from game import simulate_game

_DIR = Path(__file__).resolve().parent


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


if __name__ == "__main__":
    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof_ml_data = json.load(f)

    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    state_dict = load_model_state_dict(checkpoint_path)
    is_tactical = any(k.startswith("unit_selection_head") for k in state_dict)
    if is_tactical:
        model = TacticalModel()
    else:
        model = StrategicModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    NUM_GAMES = 200

    # Accumulators
    wins = {"A": 0, "B": 0, "draw": 0}
    tablings = {"A_tables_B": 0, "B_tables_A": 0, "mutual": 0, "neither": 0}

    # Per-game stats lists (indexed by game)
    stats = []

    print(f"Running {NUM_GAMES} games: ML vs ML (no planning), random ML HoF armies...\n")
    t_start = time.time()

    for i in range(NUM_GAMES):
        hof_a = random.choice(hof_ml_data)
        hof_b = random.choice(hof_ml_data)
        army_a = load_army_from_hof(hof_a)
        army_b = load_army_from_hof(hof_b)
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")

        # Record initial state
        a_init_models = sum(u.models_alive for u in sa)
        b_init_models = sum(u.models_alive for u in sb)
        a_init_points = sum(u.unit.points for u in sa)
        b_init_points = sum(u.unit.points for u in sb)
        a_num_units = len(sa)
        b_num_units = len(sb)

        result = simulate_game(
            res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
        )
        wins[result] += 1

        # Post-game state (sa, sb are mutated)
        a_surv_models = sum(u.models_alive for u in sa)
        b_surv_models = sum(u.models_alive for u in sb)
        a_surv_units = sum(1 for u in sa if u.models_alive > 0)
        b_surv_units = sum(1 for u in sb if u.models_alive > 0)
        a_surv_points = sum(u.unit.points for u in sa if u.models_alive > 0)
        b_surv_points = sum(u.unit.points for u in sb if u.models_alive > 0)

        # Kills
        a_models_killed = b_init_models - b_surv_models  # A killed these B models
        b_models_killed = a_init_models - a_surv_models  # B killed these A models
        a_units_destroyed = b_num_units - b_surv_units
        b_units_destroyed = a_num_units - a_surv_units
        a_points_destroyed = b_init_points - b_surv_points
        b_points_destroyed = a_init_points - a_surv_points

        # Tabling
        a_tabled_b = b_surv_models == 0
        b_tabled_a = a_surv_models == 0
        if a_tabled_b and b_tabled_a:
            tablings["mutual"] += 1
        elif a_tabled_b:
            tablings["A_tables_B"] += 1
        elif b_tabled_a:
            tablings["B_tables_A"] += 1
        else:
            tablings["neither"] += 1

        stats.append({
            "result": result,
            "a_init_pts": a_init_points,
            "b_init_pts": b_init_points,
            "a_init_models": a_init_models,
            "b_init_models": b_init_models,
            "a_num_units": a_num_units,
            "b_num_units": b_num_units,
            "a_models_killed": a_models_killed,
            "b_models_killed": b_models_killed,
            "a_units_destroyed": a_units_destroyed,
            "b_units_destroyed": b_units_destroyed,
            "a_pts_destroyed": a_points_destroyed,
            "b_pts_destroyed": b_points_destroyed,
            "a_surv_models": a_surv_models,
            "b_surv_models": b_surv_models,
            "a_tabled_b": a_tabled_b,
            "b_tabled_a": b_tabled_a,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{NUM_GAMES} games done...", end="\r")

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/NUM_GAMES:.2f}s/game)\n")

    # === ANALYSIS ===
    N = NUM_GAMES

    print("=" * 60)
    print("GAME OUTCOMES")
    print("=" * 60)
    print(f"  A wins:  {wins['A']:>4} ({wins['A']/N*100:.1f}%)")
    print(f"  B wins:  {wins['B']:>4} ({wins['B']/N*100:.1f}%)")
    print(f"  Draws:   {wins['draw']:>4} ({wins['draw']/N*100:.1f}%)")

    print(f"\n  Tablings:")
    print(f"    A tables B:  {tablings['A_tables_B']:>4}")
    print(f"    B tables A:  {tablings['B_tables_A']:>4}")
    print(f"    Mutual:      {tablings['mutual']:>4}")
    print(f"    Neither:     {tablings['neither']:>4}")

    # Outcome breakdown: tabling wins vs objective wins
    a_wins_by_tabling = sum(1 for s in stats if s["result"] == "A" and s["a_tabled_b"])
    a_wins_by_obj = sum(1 for s in stats if s["result"] == "A" and not s["a_tabled_b"])
    b_wins_by_tabling = sum(1 for s in stats if s["result"] == "B" and s["b_tabled_a"])
    b_wins_by_obj = sum(1 for s in stats if s["result"] == "B" and not s["b_tabled_a"])
    print(f"\n  Win breakdown:")
    print(f"    A wins by tabling:    {a_wins_by_tabling:>4}")
    print(f"    A wins by objectives: {a_wins_by_obj:>4}")
    print(f"    B wins by tabling:    {b_wins_by_tabling:>4}")
    print(f"    B wins by objectives: {b_wins_by_obj:>4}")

    print(f"\n{'=' * 60}")
    print("ARMY COMPOSITION (initial)")
    print("=" * 60)
    avg_a_pts = sum(s["a_init_pts"] for s in stats) / N
    avg_b_pts = sum(s["b_init_pts"] for s in stats) / N
    avg_a_models = sum(s["a_init_models"] for s in stats) / N
    avg_b_models = sum(s["b_init_models"] for s in stats) / N
    avg_a_units = sum(s["a_num_units"] for s in stats) / N
    avg_b_units = sum(s["b_num_units"] for s in stats) / N
    print(f"  Avg points:  A={avg_a_pts:.1f}  B={avg_b_pts:.1f}")
    print(f"  Avg models:  A={avg_a_models:.1f}  B={avg_b_models:.1f}")
    print(f"  Avg units:   A={avg_a_units:.1f}  B={avg_b_units:.1f}")

    # Check: do games where A has more points correlate with A winning?
    a_higher_pts = sum(1 for s in stats if s["a_init_pts"] > s["b_init_pts"] and s["result"] == "A")
    b_higher_pts = sum(1 for s in stats if s["b_init_pts"] > s["a_init_pts"] and s["result"] == "B")
    a_has_more = sum(1 for s in stats if s["a_init_pts"] > s["b_init_pts"])
    b_has_more = sum(1 for s in stats if s["b_init_pts"] > s["a_init_pts"])
    equal_pts = sum(1 for s in stats if s["a_init_pts"] == s["b_init_pts"])
    print(f"\n  A has more initial pts: {a_has_more} games")
    print(f"  B has more initial pts: {b_has_more} games")
    print(f"  Equal pts:             {equal_pts} games")

    # Among games with equal army points, who wins?
    eq_games = [s for s in stats if s["a_init_pts"] == s["b_init_pts"]]
    if eq_games:
        eq_a = sum(1 for s in eq_games if s["result"] == "A")
        eq_b = sum(1 for s in eq_games if s["result"] == "B")
        eq_d = sum(1 for s in eq_games if s["result"] == "draw")
        print(f"  Among equal-pts games ({len(eq_games)}): A={eq_a} B={eq_b} D={eq_d}")

    print(f"\n{'=' * 60}")
    print("COMBAT STATS (averages)")
    print("=" * 60)
    avg_a_mk = sum(s["a_models_killed"] for s in stats) / N
    avg_b_mk = sum(s["b_models_killed"] for s in stats) / N
    avg_a_ud = sum(s["a_units_destroyed"] for s in stats) / N
    avg_b_ud = sum(s["b_units_destroyed"] for s in stats) / N
    avg_a_pd = sum(s["a_pts_destroyed"] for s in stats) / N
    avg_b_pd = sum(s["b_pts_destroyed"] for s in stats) / N
    print(f"  Avg models killed:    A kills {avg_a_mk:.1f}  B kills {avg_b_mk:.1f}")
    print(f"  Avg units destroyed:  A kills {avg_a_ud:.1f}  B kills {avg_b_ud:.1f}")
    print(f"  Avg points destroyed: A kills {avg_a_pd:.1f}  B kills {avg_b_pd:.1f}")

    # Kill efficiency: points destroyed / own points
    avg_a_eff = sum(s["a_pts_destroyed"] / s["a_init_pts"] for s in stats) / N
    avg_b_eff = sum(s["b_pts_destroyed"] / s["b_init_pts"] for s in stats) / N
    print(f"  Kill efficiency (pts_destroyed/own_pts): A={avg_a_eff:.3f}  B={avg_b_eff:.3f}")

    print(f"\n{'=' * 60}")
    print("SURVIVAL STATS (averages)")
    print("=" * 60)
    avg_a_sm = sum(s["a_surv_models"] for s in stats) / N
    avg_b_sm = sum(s["b_surv_models"] for s in stats) / N
    print(f"  Avg surviving models: A={avg_a_sm:.1f}  B={avg_b_sm:.1f}")

    # Conditional on non-tabling games
    non_tab = [s for s in stats if not s["a_tabled_b"] and not s["b_tabled_a"]]
    if non_tab:
        nt_a_sm = sum(s["a_surv_models"] for s in non_tab) / len(non_tab)
        nt_b_sm = sum(s["b_surv_models"] for s in non_tab) / len(non_tab)
        nt_a_pd = sum(s["a_pts_destroyed"] for s in non_tab) / len(non_tab)
        nt_b_pd = sum(s["b_pts_destroyed"] for s in non_tab) / len(non_tab)
        nt_aw = sum(1 for s in non_tab if s["result"] == "A")
        nt_bw = sum(1 for s in non_tab if s["result"] == "B")
        nt_d = sum(1 for s in non_tab if s["result"] == "draw")
        print(f"\n  Non-tabling games ({len(non_tab)}):")
        print(f"    Results: A={nt_aw} B={nt_bw} D={nt_d}")
        print(f"    Avg surviving models: A={nt_a_sm:.1f}  B={nt_b_sm:.1f}")
        print(f"    Avg pts destroyed: A={nt_a_pd:.1f}  B={nt_b_pd:.1f}")

    print(f"\n{'=' * 60}")
    print("UNIT COUNT ASYMMETRY")
    print("=" * 60)
    # Does having more units (activations) help?
    more_units_wins = sum(1 for s in stats
                         if s["a_num_units"] > s["b_num_units"] and s["result"] == "A")
    fewer_units_wins = sum(1 for s in stats
                          if s["a_num_units"] < s["b_num_units"] and s["result"] == "A")
    a_more_units = sum(1 for s in stats if s["a_num_units"] > s["b_num_units"])
    a_fewer_units = sum(1 for s in stats if s["a_num_units"] < s["b_num_units"])
    same_units = sum(1 for s in stats if s["a_num_units"] == s["b_num_units"])

    print(f"  A has more units:  {a_more_units} games (A wins {more_units_wins})")
    print(f"  B has more units:  {a_fewer_units} games (A wins {fewer_units_wins})")
    print(f"  Same unit count:   {same_units} games")

    # Among same-unit-count games
    same_unit_games = [s for s in stats if s["a_num_units"] == s["b_num_units"]]
    if same_unit_games:
        su_a = sum(1 for s in same_unit_games if s["result"] == "A")
        su_b = sum(1 for s in same_unit_games if s["result"] == "B")
        su_d = sum(1 for s in same_unit_games if s["result"] == "draw")
        print(f"  Same-unit-count results ({len(same_unit_games)}): A={su_a} B={su_b} D={su_d}")

    print(f"\n{'=' * 60}")
    print("FIRST MOVER (estimated from unit count parity)")
    print("=" * 60)
    # Can't directly observe coin flip, but we can note that the coin flip
    # is 50/50. Let's just report overall.
    print("  (Coin flip is random.random() < 0.5, should be 50/50)")
    print("  To test first-mover effect, we'd need to instrument game.py")

    print()