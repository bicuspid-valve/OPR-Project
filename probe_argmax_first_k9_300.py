"""Focused probe: K=9 (argmax + 8 random), 300 games for tighter SE.

Previous run gave K=9 planner_wr = 0.450 ± 0.050 — not significantly
distinguishable from 0.50 (mirror baseline). With 300 games, SE drops
to ~0.029, so we can detect a ~3pp shift from 0.50.
"""
import json
import math
import random
import multiprocessing as mp
from pathlib import Path

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList
from game import simulate_game

_DIR = Path(__file__).resolve().parent

K = 9
NUM_GAMES = 300
NUM_WORKERS = 6


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


_WORKER_MODEL = None
_WORKER_HOF_ML = None


def _worker_init(checkpoint_path, hof_ml_data):
    global _WORKER_MODEL, _WORKER_HOF_ML
    import torch
    torch.set_num_threads(1)
    random.seed()
    sd = load_model_state_dict(checkpoint_path)
    m = TacticalModel()
    m.load_state_dict(sd, strict=False)
    m.eval()
    _WORKER_MODEL = m
    _WORKER_HOF_ML = hof_ml_data


def _play(_idx):
    aa = load_army_from_hof(random.choice(_WORKER_HOF_ML))
    ab = load_army_from_hof(random.choice(_WORKER_HOF_ML))
    ra = resolve_army(aa); rb = resolve_army(ab)
    sa = _make_unit_states(aa, ra, "A"); sb = _make_unit_states(ab, rb, "B")
    side = "A" if random.random() < 0.5 else "B"
    pp = {
        "K_INDEPENDENT_UNIFORM": K,
        "UNIFORM_ALT_SAMPLING": True,
        "INCLUDE_ARGMAX_FIRST": True,
        "M_ROLLOUTS": 16,
        "N_LOOKAHEAD": 2,
        "NUM_WORKERS": 1,
    }
    r = simulate_game(
        ra, rb, mode="objectives", states_a=sa, states_b=sb,
        ml_model_a=_WORKER_MODEL, ml_model_b=_WORKER_MODEL,
        ml_planning=side, planning_params=pp,
    )
    if r == "draw":
        return "draw"
    return "planner" if r == side else "policy"


def main():
    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof = json.load(f)
    ckpt = _DIR / "ml_checkpoints" / "final_model.pt"
    print(f"checkpoint: {ckpt}")
    print(f"K={K}, games={NUM_GAMES}, workers={NUM_WORKERS}")
    print("INCLUDE_ARGMAX_FIRST=True (1 argmax + 8 random)")
    print()

    wins = {"planner": 0, "policy": 0, "draw": 0}
    with mp.Pool(NUM_WORKERS, initializer=_worker_init, initargs=(ckpt, hof)) as pool:
        done = 0
        for r in pool.imap_unordered(_play, range(NUM_GAMES)):
            wins[r] += 1
            done += 1
            if done % 25 == 0 or done == NUM_GAMES:
                p = wins["planner"] / done
                se = math.sqrt(p * (1 - p) / done)
                print(f"  {done:>4d}/{NUM_GAMES}  planner={wins['planner']:>3} "
                      f"policy={wins['policy']:>3} draws={wins['draw']:>3}  "
                      f"wr={p:.3f}±{se:.3f}", flush=True)

    n = sum(wins.values())
    p = wins["planner"] / n
    se = math.sqrt(p * (1 - p) / n)
    z_vs_50 = (p - 0.5) / se if se > 0 else 0.0
    print()
    print(f"Final: n={n}  planner={wins['planner']}  policy={wins['policy']}  draws={wins['draw']}")
    print(f"planner_wr = {p:.4f} ± {se:.4f}")
    print(f"vs 0.500: z = {z_vs_50:+.2f}  ({'NOT ' if abs(z_vs_50) < 1.96 else ''}significant at 95%)")

    # Also reported relative to mirror "expected 50% of decisive games"
    decisive = wins["planner"] + wins["policy"]
    if decisive > 0:
        p_dec = wins["planner"] / decisive
        se_dec = math.sqrt(p_dec * (1 - p_dec) / decisive)
        z_dec = (p_dec - 0.5) / se_dec
        print(f"Of decisive games (excl. draws): planner_wr = {p_dec:.4f} ± {se_dec:.4f}  z={z_dec:+.2f}")


if __name__ == "__main__":
    main()
