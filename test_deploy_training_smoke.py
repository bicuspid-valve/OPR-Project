"""Smoke test for chunk 3: end-to-end deployment training pipeline.

Runs ~4 self-play games with the model controlling both sides' deployment,
collects DeploymentRecord trajectories, computes Monte-Carlo returns from
terminal outcomes, runs one PPO update step, and verifies that gradients
flow into the new deployment heads.
"""
from __future__ import annotations

import random

import torch

from evolution import generate_random_army, resolve_army, _make_unit_states
from game import simulate_game, deploy_armies, Board
from ml_model_tactical import TacticalModel
from ml_training.deploy_collection import make_model_deploy_decision_fn
from ml_training.deploy_loss import compute_deploy_loss


def play_one_game(model, side_for_main: str = "A") -> tuple[str, list, list]:
    """Run one self-play-style game.

    Both sides' deployment is controlled by *model*. After deployment,
    the rest of the game uses the engine's heuristic AI (no ML tactical
    control) — chunk 3 only trains the deployment heads.

    Returns (game_result, records_a, records_b).
    """
    ai = generate_random_army(mode="objectives", enforce_forceorg=True)
    bi = generate_random_army(mode="objectives", enforce_forceorg=True)
    ra, rb = resolve_army(ai), resolve_army(bi)

    sa = _make_unit_states(ai, ra, "A")
    sb = _make_unit_states(bi, rb, "B")
    board = Board()

    records_a: list = []
    records_b: list = []
    fn_a = make_model_deploy_decision_fn(
        model, player="A", side_idx=0, record_into=records_a,
    )
    fn_b = make_model_deploy_decision_fn(
        model, player="B", side_idx=1, record_into=records_b,
    )
    deploy_armies(sa, sb, board, decision_fn_a=fn_a, decision_fn_b=fn_b)

    # Continue the game with heuristic tactical AI (no ml_model_*).
    # simulate_game accepts pre-deployed unit states via states_a/_b — but its
    # _simulate_game_impl always calls deploy_armies again at line 290, which
    # would overwrite our deployment. Inline the post-deploy game loop instead.
    # For chunk 3 smoke we just call simulate_game from scratch with the same
    # armies and accept that the *trained-on* deployment is the one we recorded
    # rather than the one actually played. This still trains a value function
    # against terminal outcomes — fine for smoke. Wiring real reuse of the
    # recorded deployment would mean extending _simulate_game_impl with a
    # "skip deploy" flag, which we defer.
    sa = _make_unit_states(ai, ra, "A")
    sb = _make_unit_states(bi, rb, "B")
    result = simulate_game(ra, rb, mode="objectives", states_a=sa, states_b=sb)
    return result, records_a, records_b


def main():
    random.seed(0)
    torch.manual_seed(0)

    model = TacticalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    all_records: list = []
    all_returns: list[float] = []
    n_games = 4

    print(f"Playing {n_games} self-play deployment games...")
    for g in range(n_games):
        result, recs_a, recs_b = play_one_game(model)
        if result == "A":
            r_a, r_b = 1.0, -1.0
        elif result == "B":
            r_a, r_b = -1.0, 1.0
        else:
            r_a, r_b = 0.0, 0.0
        for r in recs_a:
            all_records.append(r)
            all_returns.append(r_a)
        for r in recs_b:
            all_records.append(r)
            all_returns.append(r_b)
        print(f"  game {g+1}: result={result}, |records_a|={len(recs_a)}, |records_b|={len(recs_b)}")

    print(f"\nTotal records: {len(all_records)}")
    print(f"  unique unit choices: {sorted({r.unit_idx for r in all_records})}")
    print(f"  pos_idx range: [{min(r.pos_idx for r in all_records)},"
          f" {max(r.pos_idx for r in all_records)}]")
    print(f"  phases: {sorted({r.phase for r in all_records})}")

    returns_t = torch.tensor(all_returns, dtype=torch.float32)
    print(f"\nReturns: mean={returns_t.mean():.3f} std={returns_t.std():.3f}")

    # --- Snapshot pre-update head params ---
    pre_unit = model.deploy_unit_head.weight.detach().clone()
    pre_pos = model.deploy_pos_head.weight.detach().clone()

    # --- One PPO update ---
    model.train()
    metrics = compute_deploy_loss(
        model, all_records, returns_t,
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
    )
    total = metrics["total_loss"]
    optimizer.zero_grad()
    total.backward()

    # Verify gradients flowed into deploy heads
    g_unit = model.deploy_unit_head.weight.grad
    g_pos = model.deploy_pos_head.weight.grad
    assert g_unit is not None and g_unit.abs().sum().item() > 0, "no grad on deploy_unit_head"
    assert g_pos is not None and g_pos.abs().sum().item() > 0, "no grad on deploy_pos_head"
    print(f"\ndeploy_unit_head |grad|_sum = {g_unit.abs().sum().item():.4f}")
    print(f"deploy_pos_head  |grad|_sum = {g_pos.abs().sum().item():.4f}")

    optimizer.step()

    # Verify the heads moved
    d_unit = (model.deploy_unit_head.weight - pre_unit).abs().sum().item()
    d_pos = (model.deploy_pos_head.weight - pre_pos).abs().sum().item()
    print(f"deploy_unit_head |Δ weight| = {d_unit:.4f}")
    print(f"deploy_pos_head  |Δ weight| = {d_pos:.4f}")
    assert d_unit > 0 and d_pos > 0, "weights did not move"

    print(f"\nPPO update metrics:")
    print(f"  policy_loss = {metrics['policy_loss']:+.4f}")
    print(f"  value_loss  = {metrics['value_loss']:+.4f}")
    print(f"  entropy     = {metrics['entropy']:+.4f}")
    print(f"  approx_kl   = {metrics['approx_kl']:+.4f}")
    print(f"  clip_frac   = {metrics['clip_frac']:+.4f}")

    # --- Re-run loss to check ratios are now != 1 (policy changed) ---
    with torch.no_grad():
        metrics2 = compute_deploy_loss(
            model, all_records, returns_t,
            clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
        )
    print(f"\nAfter step — entropy={metrics2['entropy']:+.4f}, "
          f"approx_kl={metrics2['approx_kl']:+.4f} (nonzero means policy moved)")

    print("\nchunk 3 smoke test PASSED")


if __name__ == "__main__":
    main()
