"""Linear probe: can a linear classifier predict A-side vs B-side
from the model's internal representations?

Collects activations at multiple layers from the same board states
encoded from both perspectives, then fits logistic regression.
50% accuracy = no side info leaked. >>50% = model encodes side identity.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from evolution import HallOfFame, resolve_army, _make_unit_states
from game import Board, deploy_armies, _simulate_game_impl
from ml_features import (
    encode_state_tactical, precompute_damage,
    TACTICAL_UNIT_FEATURES, MAX_UNITS_PER_SIDE,
)
from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from ml_integration_tactical import apply_tactical_model


def collect_activations(model, vec):
    """Run vec through the model and return activations at each layer."""
    with torch.no_grad():
        x = vec.unsqueeze(0)

        # 1. Raw input features (flattened unit block + global)
        _GLOBAL_OFFSET = MAX_UNITS_PER_SIDE * 2 * TACTICAL_UNIT_FEATURES
        glob = x[..., _GLOBAL_OFFSET:]
        unit_block = x[..., :_GLOBAL_OFFSET]
        units = unit_block.reshape(1, 20, TACTICAL_UNIT_FEATURES)

        # 2. Per-unit embeddings (after shared encoder)
        unit_embeds = model.unit_encoder(units)  # (1, 20, 64)

        # 3. Aggregated + global (stem input)
        unit_embeds_flat = unit_embeds.reshape(1, -1)  # (1, 1280)
        agg = torch.cat([unit_embeds_flat, glob], dim=-1)  # (1, 1296)

        # 4. After stem
        h0 = model.stem(agg)  # (1, 512)

        # 5. After recurrent core (final h)
        h = h0
        for _ in range(model.n_iters):
            h = h + model.core_block(torch.cat([h, h0], dim=-1))

    return {
        'input': vec.numpy(),
        'unit_embeds_flat': unit_embeds_flat.squeeze(0).numpy(),
        'unit_embeds_friendly': unit_embeds[0, :10].reshape(-1).numpy(),  # 640-dim
        'unit_embeds_enemy': unit_embeds[0, 10:].reshape(-1).numpy(),     # 640-dim
        'stem_out': h0.squeeze(0).numpy(),
        'trunk_out': h.squeeze(0).numpy(),
    }


def main():
    model_path = Path(__file__).resolve().parent / "ml_checkpoints" / "final_model.pt"
    sd = load_model_state_dict(model_path)
    model = TacticalModel()
    model.load_state_dict(sd, strict=False)
    model.eval()

    hof = HallOfFame.load_from_json(
        Path(__file__).resolve().parent / "results" / "hall_of_fame_ml.json")
    armies = [(e.army, resolve_army(e.army)) for e in hof.entries]

    # Collect states from actual games (not just deployment)
    N_GAMES = 50
    activations = {k: [] for k in [
        'input', 'unit_embeds_flat', 'unit_embeds_friendly',
        'unit_embeds_enemy', 'stem_out', 'trunk_out',
    ]}
    labels = []  # 0 = A, 1 = B
    round_nums = []  # round number for each sample (0 = deployment)

    print(f"Collecting activations from {N_GAMES} games...")

    for gi in range(N_GAMES):
        (army_a, res_a), (army_b, res_b) = random.sample(armies, 2)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")

        fr_a, fm_a = precompute_damage([u.unit for u in sa], [u.unit for u in sb])
        fr_b, fm_b = precompute_damage([u.unit for u in sb], [u.unit for u in sa])
        pts_a = sum(u.unit.points for u in sa)
        pts_b = sum(u.unit.points for u in sb)

        # Collect deployment state (round 0) before any activations
        # Use copies so the game simulation can redeploy fresh
        import copy
        sa_dep = copy.deepcopy(sa)
        sb_dep = copy.deepcopy(sb)
        board_deploy = Board()
        deploy_armies(sa_dep, sb_dep, board_deploy)
        vec_a_dep = encode_state_tactical(
            sa_dep, sb_dep, 1, board_deploy, "A",
            friendly_ranged_matchups=fr_a, friendly_melee_matchups=fm_a,
            enemy_ranged_matchups=fr_b, enemy_melee_matchups=fm_b,
            total_friendly_points=pts_a, total_enemy_points=pts_b)
        vec_b_dep = encode_state_tactical(
            sb_dep, sa_dep, 1, board_deploy, "B",
            friendly_ranged_matchups=fr_b, friendly_melee_matchups=fm_b,
            enemy_ranged_matchups=fr_a, enemy_melee_matchups=fm_a,
            total_friendly_points=pts_b, total_enemy_points=pts_a)
        act_a_dep = collect_activations(model, vec_a_dep)
        act_b_dep = collect_activations(model, vec_b_dep)
        for k in activations:
            activations[k].append(act_a_dep[k])
            activations[k].append(act_b_dep[k])
        labels.extend([0, 1])
        round_nums.extend([0, 0])

        # Collect states at various points during the game via the inference callback
        states_collected = []

        def make_inference_fn(mdl, states_out):
            def fn(my_units, opp_units, round_num, board, player,
                   my_fr, my_fm, opp_fr, opp_fm, my_pts, opp_pts):
                # Only sample ~25% of activations to keep dataset manageable
                if random.random() < 0.25:
                    if player == "A":
                        ua, ub = my_units, opp_units
                        fra, fma = my_fr, my_fm
                        frb, fmb = opp_fr, opp_fm
                        pa, pb = my_pts, opp_pts
                    else:
                        ua, ub = opp_units, my_units
                        fra, fma = opp_fr, opp_fm
                        frb, fmb = my_fr, my_fm
                        pa, pb = opp_pts, my_pts

                    # Encode from BOTH perspectives at the same board state
                    vec_a = encode_state_tactical(
                        ua, ub, round_num, board, "A",
                        friendly_ranged_matchups=fra, friendly_melee_matchups=fma,
                        enemy_ranged_matchups=frb, enemy_melee_matchups=fmb,
                        total_friendly_points=pa, total_enemy_points=pb)
                    vec_b = encode_state_tactical(
                        ub, ua, round_num, board, "B",
                        friendly_ranged_matchups=frb, friendly_melee_matchups=fmb,
                        enemy_ranged_matchups=fra, enemy_melee_matchups=fma,
                        total_friendly_points=pb, total_enemy_points=pa)
                    states_out.append((vec_a, vec_b, round_num))

                # Still need to do the actual decision
                active, tr, action, goal, ct, reason, _ = apply_tactical_model(
                    mdl, my_units, opp_units, round_num, board, player,
                    friendly_ranged_matchups=my_fr, friendly_melee_matchups=my_fm,
                    enemy_ranged_matchups=opp_fr, enemy_melee_matchups=opp_fm,
                    total_friendly_points=my_pts, total_enemy_points=opp_pts)
                return active, tr, action, goal, ct, reason
            return fn

        inference_fn = make_inference_fn(model, states_collected)
        _simulate_game_impl(
            res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
            ml_model_a=model, ml_model_b=model,
            _tactical_inference_fn=inference_fn)

        # Process collected states
        for vec_a, vec_b, rn in states_collected:
            act_a = collect_activations(model, vec_a)
            act_b = collect_activations(model, vec_b)
            for k in activations:
                activations[k].append(act_a[k])
                activations[k].append(act_b[k])
            labels.extend([0, 1])  # A=0, B=1
            round_nums.extend([rn, rn])

        if (gi + 1) % 10 == 0:
            print(f"  {gi+1}/{N_GAMES} games, {len(labels)} samples so far")

    labels = np.array(labels)
    rounds = np.array(round_nums)
    n = len(labels)
    print(f"\nTotal samples: {n} ({(labels==0).sum()} A, {(labels==1).sum()} B)")

    # --- Overall probes ---
    print(f"\n{'='*65}")
    print(f"LINEAR PROBE RESULTS (5-fold cross-validation)")
    print(f"{'='*65}")
    print(f"{'Layer':<30} {'Dims':>6} {'Accuracy':>10} {'vs chance':>10}")
    print("-" * 65)

    for layer_name in ['input', 'unit_embeds_friendly', 'unit_embeds_enemy',
                       'unit_embeds_flat', 'stem_out', 'trunk_out']:
        X = np.array(activations[layer_name])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X, labels, cv=5, scoring='accuracy')
        mean_acc = scores.mean()
        std_acc = scores.std()
        print(f"  {layer_name:<28} {X.shape[1]:>6} {mean_acc:>9.1%} ± {std_acc:.1%}"
              f"  {'+' if mean_acc > 0.55 else ''}{(mean_acc - 0.5)*100:>+6.1f}pp")

    # --- Position-only probes ---
    print()
    print("Position-only probes:")
    pos_feats_all = []
    for inp in activations['input']:
        pf = []
        for slot in range(20):
            offset = slot * TACTICAL_UNIT_FEATURES
            pf.extend([inp[offset + 10], inp[offset + 11]])
        pos_feats_all.append(pf)

    X_pos = np.array(pos_feats_all)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, X_pos, labels, cv=5, scoring='accuracy')
    print(f"  {'positions_all (40d)':<28} {X_pos.shape[1]:>6} {scores.mean():>9.1%} ± {scores.std():.1%}"
          f"  {'+' if scores.mean() > 0.55 else ''}{(scores.mean() - 0.5)*100:>+6.1f}pp")

    # --- Per-round probes (trunk_out) ---
    print(f"\n{'='*65}")
    print(f"PER-ROUND PROBE (trunk_out, 512d)")
    print(f"{'='*65}")
    print(f"  {'Round':<10} {'N':>6} {'Accuracy':>10} {'vs chance':>10}")
    print("-" * 45)

    X_trunk = np.array(activations['trunk_out'])
    X_pos_all = np.array(pos_feats_all)

    for rn in range(1, 5):
        mask = rounds == rn
        n_rn = mask.sum()
        if n_rn < 20:
            continue
        X_rn = X_trunk[mask]
        y_rn = labels[mask]
        clf = LogisticRegression(max_iter=1000, C=1.0)
        folds = min(5, n_rn // 10)
        if folds < 2:
            continue
        scores = cross_val_score(clf, X_rn, y_rn, cv=folds, scoring='accuracy')
        print(f"  Round {rn:<5} {n_rn:>6} {scores.mean():>9.1%} ± {scores.std():.1%}"
              f"  {'+' if scores.mean() > 0.55 else ''}{(scores.mean() - 0.5)*100:>+6.1f}pp")

    # --- Per-round probes (positions only) ---
    print(f"\n{'='*65}")
    print(f"PER-ROUND PROBE (positions only, 40d)")
    print(f"{'='*65}")
    print(f"  {'Round':<10} {'N':>6} {'Accuracy':>10} {'vs chance':>10}")
    print("-" * 45)

    for rn in range(1, 5):
        mask = rounds == rn
        n_rn = mask.sum()
        if n_rn < 20:
            continue
        X_rn = X_pos_all[mask]
        y_rn = labels[mask]
        clf = LogisticRegression(max_iter=1000, C=1.0)
        folds = min(5, n_rn // 10)
        if folds < 2:
            continue
        scores = cross_val_score(clf, X_rn, y_rn, cv=folds, scoring='accuracy')
        print(f"  Round {rn:<5} {n_rn:>6} {scores.mean():>9.1%} ± {scores.std():.1%}"
              f"  {'+' if scores.mean() > 0.55 else ''}{(scores.mean() - 0.5)*100:>+6.1f}pp")


    # --- Nonlinear (MLP) per-round probes ---
    print(f"\n{'='*65}")
    print(f"PER-ROUND PROBE — MLP (trunk_out, 512d)")
    print(f"  (tests whether nonlinear side info exists)")
    print(f"{'='*65}")
    print(f"  {'Round':<10} {'N':>6} {'Linear':>10} {'MLP':>10} {'Gap':>8}")
    print("-" * 55)

    for rn in range(1, 5):
        mask = rounds == rn
        n_rn = mask.sum()
        if n_rn < 40:
            continue
        X_rn = X_trunk[mask]
        y_rn = labels[mask]
        folds = min(5, n_rn // 10)
        if folds < 2:
            continue

        lin_clf = LogisticRegression(max_iter=1000, C=1.0)
        lin_scores = cross_val_score(lin_clf, X_rn, y_rn, cv=folds, scoring='accuracy')

        mlp_clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                          early_stopping=True, random_state=42))
        mlp_scores = cross_val_score(mlp_clf, X_rn, y_rn, cv=folds, scoring='accuracy')

        gap = mlp_scores.mean() - lin_scores.mean()
        print(f"  Round {rn:<5} {n_rn:>6} {lin_scores.mean():>9.1%} {mlp_scores.mean():>9.1%}"
              f"  {gap:>+7.1%}")

    # --- Nonlinear (MLP) per-round probes on raw input ---
    print(f"\n{'='*65}")
    print(f"PER-ROUND PROBE — MLP (raw input, 4016d)")
    print(f"{'='*65}")
    print(f"  {'Round':<10} {'N':>6} {'Linear':>10} {'MLP':>10} {'Gap':>8}")
    print("-" * 55)

    X_input = np.array(activations['input'])
    for rn in range(1, 5):
        mask = rounds == rn
        n_rn = mask.sum()
        if n_rn < 40:
            continue
        X_rn = X_input[mask]
        y_rn = labels[mask]
        folds = min(5, n_rn // 10)
        if folds < 2:
            continue

        lin_clf = LogisticRegression(max_iter=1000, C=1.0)
        lin_scores = cross_val_score(lin_clf, X_rn, y_rn, cv=folds, scoring='accuracy')

        mlp_clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                          early_stopping=True, random_state=42))
        mlp_scores = cross_val_score(mlp_clf, X_rn, y_rn, cv=folds, scoring='accuracy')

        gap = mlp_scores.mean() - lin_scores.mean()
        print(f"  Round {rn:<5} {n_rn:>6} {lin_scores.mean():>9.1%} {mlp_scores.mean():>9.1%}"
              f"  {gap:>+7.1%}")


if __name__ == "__main__":
    main()
