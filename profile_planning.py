"""Quick profiler for plan_activation — identifies where time is spent."""
from __future__ import annotations

import json
import os
import pickle
import random
import statistics
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch

from ml_model_tactical import TacticalModel
from ml_training import load_model_state_dict
from ml_features import encode_state_tactical, precompute_damage, MAX_UNITS_PER_SIDE
from ml_planning import (
    plan_activation, snapshot_game_state, restore_game_state,
    simulate_forward,
    DEFAULT_N_LOOKAHEAD,
)
from evolution import make_entry, resolve_army, _make_unit_states
from game import deploy_armies
from board import Board, OBJECTIVES
from models import ArmyList


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


def setup_game(hof_data, model):
    """Set up a random game state and return everything needed for planning."""
    army_a = load_army_from_hof(random.choice(hof_data))
    army_b = load_army_from_hof(random.choice(hof_data))
    res_a = resolve_army(army_a)
    res_b = resolve_army(army_b)
    sa = _make_unit_states(army_a, res_a, "A")
    sb = _make_unit_states(army_b, res_b, "B")
    board = Board()
    deploy_armies(sa, sb, board)

    fr_a, fm_a = precompute_damage([u.unit for u in sa], [u.unit for u in sb])
    fr_b, fm_b = precompute_damage([u.unit for u in sb], [u.unit for u in sa])
    pts_a = sum(u.unit.points for u in sa)
    pts_b = sum(u.unit.points for u in sb)

    return sa, sb, board, fr_a, fm_a, fr_b, fm_b, pts_a, pts_b


def profile_plan_activation(model, hof_data, params, n_trials=5, n_warmup=1):
    """Profile full plan_activation calls."""
    print(f"\n{'='*60}")
    print(f"plan_activation profiling — K={params.get('K_UNITS',6)}, "
          f"C={params.get('C_SAMPLES_PER_UNIT',4)}, "
          f"M={params.get('M_ROLLOUTS',4)}, "
          f"N={params.get('N_LOOKAHEAD',4)}, "
          f"workers={params.get('NUM_WORKERS',6)}")
    print(f"{'='*60}")

    # Warmup (pool startup, JIT, etc.)
    for _ in range(n_warmup):
        sa, sb, board, fr_a, fm_a, fr_b, fm_b, pts_a, pts_b = setup_game(hof_data, model)
        plan_activation(
            model, sa, sb, 1, board, "A",
            units_a=sa, units_b=sb,
            current_is_a=True, mode="objectives",
            friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
            enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
            total_friendly_points=pts_a, total_enemy_points=pts_b,
            fr_a=fr_a, fm_a=fm_a, fr_b=fr_b, fm_b=fm_b,
            pts_a=pts_a, pts_b=pts_b,
            planning_params=params,
        )
    print(f"  ({n_warmup} warmup trial(s) done)")

    times = []
    for i in range(n_trials):
        sa, sb, board, fr_a, fm_a, fr_b, fm_b, pts_a, pts_b = setup_game(hof_data, model)

        t0 = time.perf_counter()
        result = plan_activation(
            model, sa, sb, 1, board, "A",
            units_a=sa, units_b=sb,
            current_is_a=True, mode="objectives",
            friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
            enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
            total_friendly_points=pts_a, total_enemy_points=pts_b,
            fr_a=fr_a, fm_a=fm_a, fr_b=fr_b, fm_b=fm_b,
            pts_a=pts_a, pts_b=pts_b,
            planning_params=params,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        n_candidates = len(result[-1])
        print(f"  Trial {i+1}: {elapsed:.3f}s  ({n_candidates} candidates)")

    print(f"  Mean: {statistics.mean(times):.3f}s | "
          f"Min: {min(times):.3f}s | Max: {max(times):.3f}s")
    return statistics.mean(times)


def profile_components(model, hof_data):
    """Break down time by component."""
    print(f"\n{'='*60}")
    print("COMPONENT BREAKDOWN")
    print(f"{'='*60}")

    sa, sb, board, fr_a, fm_a, fr_b, fm_b, pts_a, pts_b = setup_game(hof_data, model)

    # 1. State encoding
    encode_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        encode_state_tactical(
            sa, sb, 1, board, "A",
            friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
            enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
            total_friendly_points=pts_a, total_enemy_points=pts_b,
        )
        encode_times.append(time.perf_counter() - t0)
    print(f"\n1. encode_state_tactical: {statistics.mean(encode_times)*1000:.2f}ms avg")

    # 2. Model forward pass (trunk + all heads)
    state_vec = encode_state_tactical(
        sa, sb, 1, board, "A",
        friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
        enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
        total_friendly_points=pts_a, total_enemy_points=pts_b,
    )
    alive_mask = torch.tensor(
        [(i < len(sa) and sa[i].models_alive > 0) for i in range(MAX_UNITS_PER_SIDE)],
        dtype=torch.bool,
    )
    enemy_alive_mask = torch.tensor(
        [(i < len(sb) and sb[i].models_alive > 0) for i in range(MAX_UNITS_PER_SIDE)],
        dtype=torch.bool,
    )

    # Trunk only
    trunk_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            x = state_vec.unsqueeze(0)
            h, _u, _aw, _ = model.trunk(x)
            h = h.squeeze(0)
        trunk_times.append(time.perf_counter() - t0)
    print(f"2. Trunk forward: {statistics.mean(trunk_times)*1000:.3f}ms avg")

    # Full forward (both passes)
    full_fwd_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(state_vec, alive_mask, enemy_alive_mask)
        full_fwd_times.append(time.perf_counter() - t0)
    print(f"3. Full model forward: {statistics.mean(full_fwd_times)*1000:.3f}ms avg")

    # 3. Snapshot/restore
    snap_times = []
    for _ in range(200):
        t0 = time.perf_counter()
        snap = snapshot_game_state(sa, sb, board)
        snap_times.append(time.perf_counter() - t0)
    print(f"4. Snapshot: {statistics.mean(snap_times)*1000:.3f}ms avg")

    restore_times = []
    snap = snapshot_game_state(sa, sb, board)
    for _ in range(200):
        t0 = time.perf_counter()
        restore_game_state(snap, sa, sb, board)
        restore_times.append(time.perf_counter() - t0)
    print(f"5. Restore: {statistics.mean(restore_times)*1000:.3f}ms avg")

    # 4. Pickle serialization (for parallel workers)
    pickle_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        data = pickle.dumps((
            list(sa), list(sb), board,
            fr_a, fm_a, fr_b, fm_b, pts_a, pts_b,
        ))
        pickle_times.append(time.perf_counter() - t0)
    pickle_size = len(data)
    print(f"6. Pickle serialize: {statistics.mean(pickle_times)*1000:.2f}ms avg ({pickle_size/1024:.1f} KB)")

    unpickle_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        pickle.loads(data)
        unpickle_times.append(time.perf_counter() - t0)
    print(f"7. Pickle deserialize: {statistics.mean(unpickle_times)*1000:.2f}ms avg")

    # 5. simulate_forward (N=4 lookahead)
    sim_times = []
    for _ in range(20):
        snap = snapshot_game_state(sa, sb, board)
        t0 = time.perf_counter()
        with torch.no_grad():
            simulate_forward(
                sa, sb, board, model, 4, True, 1, "objectives",
                fr_a, fm_a, fr_b, fm_b, pts_a, pts_b,
            )
        sim_times.append(time.perf_counter() - t0)
        restore_game_state(snap, sa, sb, board)
    print(f"8. simulate_forward (N=4): {statistics.mean(sim_times)*1000:.2f}ms avg")

    # 6. precompute_damage
    dmg_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        precompute_damage([u.unit for u in sa], [u.unit for u in sb])
        dmg_times.append(time.perf_counter() - t0)
    print(f"9. precompute_damage: {statistics.mean(dmg_times)*1000:.2f}ms avg")

    # Summary: estimate for one candidate (M rollouts)
    M = 4
    N = 4
    per_rollout = (
        statistics.mean(snap_times)
        + statistics.mean(sim_times)
        + statistics.mean(restore_times)
    )
    per_candidate = M * per_rollout
    print(f"\n--- Estimates ---")
    print(f"  Per rollout (snap + sim_forward + restore): {per_rollout*1000:.2f}ms")
    print(f"  Per candidate (M={M} rollouts): {per_candidate*1000:.1f}ms")
    for n_cands in [4, 8, 16, 24]:
        seq = n_cands * per_candidate
        print(f"  {n_cands:2d} candidates sequential: {seq*1000:.0f}ms ({seq:.2f}s)")


if __name__ == "__main__":
    _DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(_DIR, "results", "hall_of_fame_ml.json")) as f:
        hof_data = json.load(f)

    checkpoint_path = os.path.join(_DIR, "ml_checkpoints", "final_model.pt")
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded tactical model from {checkpoint_path}")

    # Component breakdown first
    profile_components(model, hof_data)

    # Full plan_activation with the params from quick_ml_check3
    profile_plan_activation(model, hof_data, {
        "K_UNITS": 4,
        "C_SAMPLES_PER_UNIT": 4,
        "M_ROLLOUTS": 4,
        "N_LOOKAHEAD": 4,
        "NUM_WORKERS": 6,
    })

    # Sequential for comparison
    profile_plan_activation(model, hof_data, {
        "K_UNITS": 4,
        "C_SAMPLES_PER_UNIT": 4,
        "M_ROLLOUTS": 4,
        "N_LOOKAHEAD": 4,
        "NUM_WORKERS": 1,
    })
