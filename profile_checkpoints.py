"""Profile: final model & newest checkpoint vs pool — argmax AND sampling.

Compares argmax vs sampling win rates to test whether entropy/confidence
differences explain inflated self-play metrics.

Pre-loads all models to avoid race conditions with concurrent training.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import ArmyList, UnitState
from evolution import make_entry, resolve_army, _make_unit_states
from game import simulate_game
from ml_model_tactical import TacticalModel
from ml_training.checkpoint import load_model_state_dict


GAMES_PER_MATCHUP = 5
CHECKPOINT_DIR = Path(__file__).resolve().parent / "ml_checkpoints"


def load_model(path: Path) -> TacticalModel:
    sd = load_model_state_dict(path)
    m = TacticalModel()
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def load_hof(filename: str) -> list[ArmyList]:
    path = Path(__file__).resolve().parent / "results" / filename
    with open(path) as f:
        hof_data = json.load(f)
    armies = []
    for entry_data in hof_data:
        army = ArmyList()
        for e in entry_data["entries"]:
            entry = make_entry(
                e["template_id"],
                upgrades=e.get("upgrades", {}),
                ai_role=e.get("ai_role", "killer"),
            )
            entry.combat_preference = e.get("combat_preference", "ranged")
            army.entries.append(entry)
        armies.append(army)
    return armies


def play_matchup(model_a, model_b, armies, n_games, sampling=False):
    """Play n_games. Returns (a_wins, b_wins, draws)."""
    a_wins = b_wins = draws = 0
    for _ in range(n_games):
        army_a = random.choice(armies)
        army_b = random.choice(armies)
        res_a = resolve_army(army_a)
        res_b = resolve_army(army_b)
        sa = _make_unit_states(army_a, res_a, "A")
        sb = _make_unit_states(army_b, res_b, "B")
        result = simulate_game(
            res_a, res_b, mode="objectives",
            states_a=sa, states_b=sb,
            ml_model_a=model_a, ml_model_b=model_b,
            ml_sampling=sampling,
        )
        if result == "A":
            a_wins += 1
        elif result == "B":
            b_wins += 1
        else:
            draws += 1
    return a_wins, b_wins, draws


def run_sweep(label, test_model, pool_models, armies, sampling=False):
    mode = "SAMPLING" if sampling else "ARGMAX"
    print(f"\n--- {label} ({mode}) vs checkpoint pool ---")
    print(f"{'Opponent':<30s} {'W':>3s} {'L':>3s} {'D':>3s} {'WR':>7s}")
    print("-" * 50)

    total_w = total_l = total_d = 0
    t0 = time.time()

    for opp_batch, opp_model in pool_models:
        w, l, d = play_matchup(test_model, opp_model, armies, GAMES_PER_MATCHUP,
                               sampling=sampling)
        wr = (w + 0.5 * d) / GAMES_PER_MATCHUP
        total_w += w
        total_l += l
        total_d += d
        print(f"  vs batch {opp_batch:>6d}          {w:>3d} {l:>3d} {d:>3d} {wr:>7.3f}")

    total_games = total_w + total_l + total_d
    overall_wr = (total_w + 0.5 * total_d) / total_games if total_games else 0
    elapsed = time.time() - t0
    print("-" * 50)
    print(f"  TOTAL                       {total_w:>3d} {total_l:>3d} {total_d:>3d} {overall_wr:>7.3f}  ({elapsed:.1f}s)")
    return overall_wr


def main():
    random.seed(42)
    torch.manual_seed(42)

    ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint_batch_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    print(f"Found {len(ckpts)} checkpoints")
    if not ckpts:
        return

    pool_paths = ckpts[-20:]
    newest_ckpt_path = pool_paths[-1]

    final_path = CHECKPOINT_DIR / "final_model.pt"
    has_final = final_path.exists()

    hof = load_hof("hall_of_fame.json")
    hof_ml = load_hof("hall_of_fame_ml.json")
    armies = hof + hof_ml
    print(f"Loaded {len(armies)} armies")

    print(f"\nPre-loading all models...")
    final_model = load_model(final_path) if has_final else None
    newest_model = load_model(newest_ckpt_path)
    pool_models = []
    for p in pool_paths:
        batch_num = int(p.stem.split("_")[-1])
        try:
            pool_models.append((batch_num, load_model(p)))
        except FileNotFoundError:
            print(f"  batch {batch_num} MISSING")
    loaded = f"{len(pool_models)} pool + newest"
    if has_final:
        loaded += " + final"
    print(f"Loaded {loaded}")

    print(f"\n{'='*60}")
    print(f"{GAMES_PER_MATCHUP} games per matchup")
    print(f"{'='*60}")

    results = {}
    results["NEWEST_argmax"] = run_sweep("NEWEST", newest_model, pool_models, armies, sampling=False)
    if has_final:
        results["FINAL_argmax"] = run_sweep("FINAL", final_model, pool_models, armies, sampling=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:<25s} {v:.3f}")


if __name__ == "__main__":
    main()
