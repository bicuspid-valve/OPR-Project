"""Profile training-time planning to measure batchable vs non-batchable time.

Measures where time is spent in plan_training_activation to estimate how much
speedup batching across games would provide.

Usage: .venv/bin/python3 profile_training_planning.py
"""
from __future__ import annotations

import json
import os
import pickle
import random
import statistics
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from ml_model_tactical import (
    TacticalModel, NUM_MOVE_TYPES,
    MOVE_MOVE, MOVE_CHARGE,
)
from ml_training import load_model_state_dict, sample_tactical_actions_no_grad
from ml_features import encode_state_tactical, precompute_damage, MAX_UNITS_PER_SIDE, extract_can_charge_mask
from ml_integration_tactical import (
    _get_model_space_positions, _get_movement_budgets, _get_max_weapon_ranges,
    execute_decoded_decision, compute_post_move_rel, compute_in_range_mask,
    decode_destination_params, decode_destination_argmax, compute_post_move_position,
)
from ml_planning import plan_training_activation, _run_chunk_batched
from evolution import resolve_army, _make_unit_states, make_entry
from game import deploy_armies
from board import Board
from models import ArmyList


def load_army_from_hof(entry: dict) -> ArmyList:
    army = ArmyList()
    for e in entry["entries"]:
        ent = make_entry(e["template_id"], upgrades=e.get("upgrades", {}),
                         ai_role=e.get("ai_role", "killer"))
        ent.combat_preference = e.get("combat_preference", "ranged")
        army.entries.append(ent)
    return army


def setup_game(hof_data):
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


def prepare_planning_inputs(units_a, units_b, board, fr_a, fm_a, fr_b, fm_b,
                            pts_a, pts_b, round_num=2):
    """Prepare all inputs needed for plan_training_activation."""
    alive_mask = torch.tensor(
        [(i < len(units_a) and units_a[i].models_alive > 0
          and not units_a[i].activated)
         for i in range(MAX_UNITS_PER_SIDE)], dtype=torch.bool)
    enemy_alive_mask = torch.tensor(
        [(i < len(units_b) and units_b[i].models_alive > 0)
         for i in range(MAX_UNITS_PER_SIDE)], dtype=torch.bool)
    if not alive_mask.any():
        return None
    state_vec = encode_state_tactical(
        units_a, units_b, round_num, board, "A",
        friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
        enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
        total_friendly_points=pts_a, total_enemy_points=pts_b,
    )
    friendly_pos = _get_model_space_positions(units_a, "A")
    enemy_pos = _get_model_space_positions(units_b, "A")
    adv_dists, rush_dists = _get_movement_budgets(units_a)
    max_wr = _get_max_weapon_ranges(units_a)
    return (state_vec, alive_mask, enemy_alive_mask, friendly_pos, enemy_pos,
            adv_dists, rush_dists, max_wr)


# ── Test 1: full plan_training_activation timing ──────────────────────────

def test_full_planning(model, hof_data, n_trials=10):
    print(f"\n{'='*70}")
    print(f"TEST 1: Full plan_training_activation (K=3, C=3, M=4, N=3)")
    print(f"{'='*70}")

    params = {"K_UNITS": 3, "C_SAMPLES_PER_UNIT": 3,
              "M_ROLLOUTS": 4, "N_LOOKAHEAD": 3}
    times = []
    for i in range(n_trials):
        game = setup_game(hof_data)
        inputs = prepare_planning_inputs(*game)
        if inputs is None:
            continue
        sv, am, eam, fp, ep, ad, rd, mw = inputs
        ua, ub, bd = game[0], game[1], game[2]
        fr_a, fm_a, fr_b, fm_b, pts_a, pts_b = game[3:]

        t0 = time.perf_counter()
        plan_training_activation(
            model, sv, am, eam, ua, ub, 2, bd, "A",
            current_is_a=True, mode="objectives",
            friendly_positions=fp, enemy_positions=ep,
            advance_distances=ad, rush_distances=rd, max_weapon_ranges=mw,
            fr_a=fr_a, fm_a=fm_a, fr_b=fr_b, fm_b=fm_b,
            pts_a=pts_a, pts_b=pts_b,
            planning_params=params, opponent_type=0,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Trial {i+1}: {elapsed*1000:.1f} ms")

    avg = statistics.mean(times)
    print(f"  Average: {avg*1000:.1f} ms  (min={min(times)*1000:.1f}, max={max(times)*1000:.1f})")
    return avg


# ── Test 2: component breakdown ───────────────────────────────────────────

def test_components(model, hof_data, n_trials=10):
    print(f"\n{'='*70}")
    print(f"TEST 2: Component breakdown")
    print(f"{'='*70}")

    K, C, M, N = 3, 3, 4, 3
    results = {k: [] for k in [
        "state_encode", "trunk_pass", "candidate_gen",
        "pickle_dump", "pickle_load_each", "rollouts",
        "logprob", "normal_sample",
    ]}
    meta = {"n_candidates": [], "n_pickle_loads": [], "pickle_kb": []}

    for trial in range(n_trials):
        game = setup_game(hof_data)
        ua, ub, bd, fr_a, fm_a, fr_b, fm_b, pts_a, pts_b = game

        # State encoding
        t0 = time.perf_counter()
        state_vec = encode_state_tactical(
            ua, ub, 2, bd, "A",
            friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
            enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
            total_friendly_points=pts_a, total_enemy_points=pts_b,
        )
        results["state_encode"].append(time.perf_counter() - t0)

        alive_mask = torch.tensor(
            [(i < len(ua) and ua[i].models_alive > 0 and not ua[i].activated)
             for i in range(MAX_UNITS_PER_SIDE)], dtype=torch.bool)
        eam = torch.tensor(
            [(i < len(ub) and ub[i].models_alive > 0)
             for i in range(MAX_UNITS_PER_SIDE)], dtype=torch.bool)
        if not alive_mask.any():
            continue

        fp = _get_model_space_positions(ua, "A")
        ep = _get_model_space_positions(ub, "A")
        ad, rd = _get_movement_budgets(ua)
        mw = _get_max_weapon_ranges(ua)

        # Trunk pass
        t0 = time.perf_counter()
        with torch.no_grad():
            x = state_vec.unsqueeze(0)
            am = alive_mask.unsqueeze(0)
            h, u_att, round_oh = model.trunk(x)
            unit_logits = model.unit_selection_head(h)
            unit_logits = unit_logits.masked_fill(~am, float('-inf'))
            unit_probs = torch.softmax(unit_logits, dim=-1).squeeze(0)
        results["trunk_pass"].append(time.perf_counter() - t0)

        # Candidate generation
        t0 = time.perf_counter()
        with torch.no_grad():
            argmax_unit = int(unit_probs.argmax().item())
            num_alive = int(alive_mask.sum().item())
            k = min(K, num_alive)
            _, top_idx = torch.topk(unit_probs, k)
            cand_units = top_idx.tolist()
            if argmax_unit in cand_units:
                cand_units.remove(argmax_unit)
            cand_units = [argmax_unit] + cand_units[:K - 1]

            candidates = []
            h_b = h
            for ui, uid in enumerate(cand_units):
                unit = ua[uid]
                uf = model._extract_unit_features(u_att.squeeze(0), uid).detach()
                uf_b = uf.unsqueeze(0)
                ccm = extract_can_charge_mask(state_vec, uid)  # (10,) bool
                h_uf = torch.cat([h_b, uf_b], dim=-1)
                ml = model.move_type_head(h_uf).squeeze(0)
                if not ccm.any():
                    ml = ml.clone(); ml[MOVE_CHARGE] = float('-inf')
                mp = torch.softmax(ml, dim=-1)
                for si in range(C):
                    is_am = (ui == 0 and si == 0)
                    mt = int(mp.argmax().item()) if is_am else int(torch.multinomial(mp, 1).item())
                    moh = F.one_hot(torch.tensor(mt), NUM_MOVE_TYPES).float().unsqueeze(0)
                    h_uf_m = torch.cat([h_b, uf_b, moh], dim=-1)
                    dest_r = model.destination_head(h_uf_m).squeeze(0)
                    sa, sf = decode_destination_argmax(dest_r)
                    cx, cy = fp[uid]
                    if mt == 1: px,py = compute_post_move_position(cx,cy,sa,sf*ad[uid])
                    elif mt == 2: px,py = compute_post_move_position(cx,cy,sa,sf*rd[uid])
                    else: px,py = cx,cy
                    cl = model.compute_charge_logits(h_b.squeeze(0), u_att.squeeze(0), uid, eam, ccm)
                    ct = int(cl.argmax().item()) if is_am else (int(torch.multinomial(torch.softmax(cl,-1),1).item()) if eam.any() else 0)
                    pmr = compute_post_move_rel(px, py, ep)
                    wr = max((w.range_inches for w in unit.unit.weapons if not w.melee), default=0.0)
                    sm = compute_in_range_mask(pmr, float(wr), eam)
                    sl = model.compute_shoot_logits(h_b.squeeze(0), u_att.squeeze(0), uid, pmr, eam, shoot_range_mask=sm)
                    st = int(sl.argmax().item()) if is_am else (int(torch.multinomial(torch.softmax(sl,-1),1).item()) if sm.any() else 0)
                    rank = torch.argsort(sl, descending=True).tolist()
                    dest = (px,py) if mt in (1,2) else None
                    act, goal, ct_u, reason = execute_decoded_decision(unit, ub, mt, dest, ct, st)
                    ct_i = -1
                    if ct_u:
                        for j,eu in enumerate(ub):
                            if eu is ct_u: ct_i = j; break
                    candidates.append((uid,mt,sa,sf,rank,ct,st,act,goal,ct_i,reason))
        results["candidate_gen"].append(time.perf_counter() - t0)
        meta["n_candidates"].append(len(candidates))

        # Pickle dump
        t0 = time.perf_counter()
        state_bytes = pickle.dumps((list(ua), list(ub), bd, fr_a, fm_a, fr_b, fm_b, pts_a, pts_b))
        results["pickle_dump"].append(time.perf_counter() - t0)
        meta["pickle_kb"].append(len(state_bytes) / 1024)

        # Pickle load (per-rollout cost)
        n_loads = len(candidates) * M
        meta["n_pickle_loads"].append(n_loads)
        t0 = time.perf_counter()
        for _ in range(n_loads):
            pickle.loads(state_bytes)
        t_all_loads = time.perf_counter() - t0
        results["pickle_load_each"].append(t_all_loads / n_loads)

        # Rollout evaluation
        t0 = time.perf_counter()
        _run_chunk_batched((
            state_bytes, candidates, M, N, 2, "objectives", "A", True, True,
        ), model_override=model)
        results["rollouts"].append(time.perf_counter() - t0)

        # Log-prob computation
        t0 = time.perf_counter()
        with torch.no_grad():
            ch = candidates[0]
            uf2 = model._extract_unit_features(u_att.squeeze(0), ch[0]).detach()
            uf2_b = uf2.unsqueeze(0)
            h_uf2 = torch.cat([h_b, uf2_b], dim=-1)
            model.move_type_head(h_uf2)
            moh2 = F.one_hot(torch.tensor(ch[1]), NUM_MOVE_TYPES).float().unsqueeze(0)
            h_uf_m2 = torch.cat([h_b, uf2_b, moh2], dim=-1)
            model.destination_head(h_uf_m2)
            model.compute_charge_logits(h_b.squeeze(0), u_att.squeeze(0), ch[0], eam, extract_can_charge_mask(state_vec, ch[0]))
            pmr2 = compute_post_move_rel(fp[ch[0]][0], fp[ch[0]][1], ep)
            model.compute_shoot_logits(h_b.squeeze(0), u_att.squeeze(0), ch[0], pmr2, eam)
            model.value_head(h, round_oh, model._get_opp_embed(h, 0))
        results["logprob"].append(time.perf_counter() - t0)

        # Normal sample (baseline)
        t0 = time.perf_counter()
        sample_tactical_actions_no_grad(
            model, state_vec, alive_mask, eam, fp, ep, ad, rd, mw)
        results["normal_sample"].append(time.perf_counter() - t0)

    # Print results
    print(f"\n  {'Component':<30} {'Avg (ms)':>10} {'Category':>15}")
    print(f"  {'-'*55}")

    batchable_total = 0
    unbatchable_total = 0
    total = 0

    rows = [
        ("trunk_pass",     "BATCHABLE"),
        ("candidate_gen",  "BATCHABLE"),
        ("pickle_dump",    "per-game"),
        ("rollouts",       "mixed"),
        ("logprob",        "BATCHABLE"),
        ("normal_sample",  "baseline"),
    ]
    for key, cat in rows:
        vals = results[key]
        if not vals:
            continue
        avg_ms = statistics.mean(vals) * 1000
        print(f"  {key:<30} {avg_ms:>10.2f} {cat:>15}")
        if cat == "BATCHABLE":
            batchable_total += statistics.mean(vals)
        if key == "rollouts":
            total += statistics.mean(vals)
        if key in ("trunk_pass", "candidate_gen", "logprob"):
            total += statistics.mean(vals)

    # Pickle stats
    avg_load = statistics.mean(results["pickle_load_each"]) * 1000
    avg_n_loads = statistics.mean(meta["n_pickle_loads"])
    avg_pickle_kb = statistics.mean(meta["pickle_kb"])
    avg_n_cands = statistics.mean(meta["n_candidates"])
    total_pickle_ms = avg_load * avg_n_loads

    print(f"\n  Pickle: {avg_pickle_kb:.0f} KB, {avg_load:.2f} ms/load × {avg_n_loads:.0f} loads = {total_pickle_ms:.1f} ms total")
    print(f"  Candidates: {avg_n_cands:.1f} avg")

    # Estimate rollout breakdown: subtract pickle from rollout total
    avg_rollout = statistics.mean(results["rollouts"]) * 1000
    rollout_minus_pickle = avg_rollout - total_pickle_ms
    print(f"\n  Rollout breakdown:")
    print(f"    Total rollout:          {avg_rollout:.1f} ms")
    print(f"    Pickle loads:           {total_pickle_ms:.1f} ms  (per-game)")
    print(f"    Model fwd + game sim:   {rollout_minus_pickle:.1f} ms")

    return results, meta


# ── Test 3: batched trunk pass scaling ────────────────────────────────────

def test_batched_trunk_scaling(model, hof_data):
    print(f"\n{'='*70}")
    print(f"TEST 3: Trunk pass scaling with batch size")
    print(f"{'='*70}")

    # Prepare many game states
    all_states = []
    for _ in range(100):
        game = setup_game(hof_data)
        ua, ub, bd, fr_a, fm_a, fr_b, fm_b, pts_a, pts_b = game
        sv = encode_state_tactical(
            ua, ub, 2, bd, "A",
            friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
            enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
            total_friendly_points=pts_a, total_enemy_points=pts_b,
        )
        all_states.append(sv)

    for batch_size in [1, 4, 16, 42, 85]:
        batch = torch.stack(all_states[:batch_size])
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            with torch.no_grad():
                h, u_att, _ = model.trunk(batch)
                model.unit_selection_head(h)
            times.append(time.perf_counter() - t0)
        avg = statistics.mean(times) * 1000
        per_item = avg / batch_size
        print(f"  Batch {batch_size:3d}: {avg:7.2f} ms total, {per_item:6.3f} ms/item")


# ── Test 4: projected speedup ─────────────────────────────────────────────

def estimate_speedup(results, meta):
    print(f"\n{'='*70}")
    print(f"PROJECTED SPEEDUP")
    print(f"{'='*70}")

    N_GAMES = 42  # typical Player A requests per coordinator round

    avg_trunk = statistics.mean(results["trunk_pass"]) * 1000
    avg_cand = statistics.mean(results["candidate_gen"]) * 1000
    avg_lp = statistics.mean(results["logprob"]) * 1000
    avg_rollout = statistics.mean(results["rollouts"]) * 1000
    avg_load = statistics.mean(results["pickle_load_each"]) * 1000
    avg_n_loads = statistics.mean(meta["n_pickle_loads"])
    avg_sample = statistics.mean(results["normal_sample"]) * 1000

    total_pickle = avg_load * avg_n_loads
    rollout_fwd_sim = avg_rollout - total_pickle

    total_per_activation = avg_trunk + avg_cand + avg_rollout + avg_lp

    print(f"\n  Current per planned activation:  {total_per_activation:.1f} ms")
    print(f"    Trunk:                          {avg_trunk:.2f} ms")
    print(f"    Candidate gen:                  {avg_cand:.1f} ms")
    print(f"    Rollouts:                       {avg_rollout:.1f} ms")
    print(f"      (pickle: {total_pickle:.1f}, fwd+sim: {rollout_fwd_sim:.1f})")
    print(f"    Log-prob:                       {avg_lp:.2f} ms")

    print(f"\n  Normal sample (no planning):      {avg_sample:.2f} ms")

    # With batching across N_GAMES:
    # trunk + cand_gen + logprob: amortize to ~1/N_GAMES
    batched_trunk = avg_trunk / N_GAMES
    batched_cand = avg_cand / N_GAMES
    batched_lp = avg_lp / N_GAMES

    # Rollout: pickle is per-game, game sim is per-game,
    # but model fwd within rollouts could be batched
    # Conservative: assume 40% of rollout_fwd_sim is model fwd (batchable)
    for model_frac in [0.3, 0.5, 0.7]:
        rollout_model = rollout_fwd_sim * model_frac
        rollout_sim = rollout_fwd_sim * (1 - model_frac)
        batched_rollout_model = rollout_model / N_GAMES
        batched_per_act = (batched_trunk + batched_cand + batched_lp
                           + total_pickle + batched_rollout_model + rollout_sim)

        speedup = total_per_activation / batched_per_act

        print(f"\n  Batched ({N_GAMES} games), rollout model fraction={model_frac:.0%}:")
        print(f"    Per activation:  {batched_per_act:.1f} ms  ({speedup:.1f}x speedup)")

    # Training batch impact
    print(f"\n  --- Training batch impact (rate=0.10, 512 games, 6 workers) ---")
    games_per_worker = 512 / 6
    a_activations_per_game = 20
    total_a_activations = games_per_worker * a_activations_per_game
    planned = 0.10 * total_a_activations
    normal = total_a_activations - planned

    current_planning_ms = planned * total_per_activation
    current_normal_ms = normal * avg_sample
    current_total = current_planning_ms + current_normal_ms

    baseline_ms = total_a_activations * avg_sample

    print(f"    Baseline (no planning):  {baseline_ms/1000:.1f} s")
    print(f"    Current (sequential):    {current_total/1000:.1f} s  ({current_total/baseline_ms:.1f}x)")

    for model_frac in [0.3, 0.5, 0.7]:
        rollout_model = rollout_fwd_sim * model_frac
        rollout_sim = rollout_fwd_sim * (1 - model_frac)
        batched_rollout_model = rollout_model / N_GAMES
        batched_per_act = (batched_trunk + batched_cand + batched_lp
                           + total_pickle + batched_rollout_model + rollout_sim)
        batched_planning_ms = planned * batched_per_act
        batched_total = batched_planning_ms + current_normal_ms
        print(f"    Batched (fwd={model_frac:.0%}):       {batched_total/1000:.1f} s  ({batched_total/baseline_ms:.1f}x)")


if __name__ == "__main__":
    print(f"C extension: {'ON' if fast_core.USE_C_EXT else 'OFF'}")

    hof_path = os.path.join("results", "hall_of_fame_ml.json")
    if not os.path.exists(hof_path):
        hof_path = os.path.join("results", "hall_of_fame.json")
    with open(hof_path) as f:
        hof_data = json.load(f)
    print(f"Loaded {len(hof_data)} HoF armies from {hof_path}")

    ckpt = os.path.join("ml_checkpoints", "final_model.pt")
    if not os.path.exists(ckpt):
        import glob
        ckpts = sorted(glob.glob("ml_checkpoints/checkpoint_batch_*.pt"),
                       key=os.path.getmtime)
        if ckpts:
            ckpt = ckpts[-1]
        else:
            print("No checkpoints found!")
            exit(1)

    sd = load_model_state_dict(ckpt)
    model = TacticalModel()
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Loaded model from {ckpt}")

    test_full_planning(model, hof_data, n_trials=10)
    results, meta = test_components(model, hof_data, n_trials=10)
    test_batched_trunk_scaling(model, hof_data)
    estimate_speedup(results, meta)
