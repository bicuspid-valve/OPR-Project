"""Probe: planning(argmax + 39 policy) vs planning(argmax + 9 policy + 30 uniform).

Both sides plan every activation with 40 candidates total — same compute.
Only candidate composition differs.

Hypothesis: V wants candidate *diversity*, not *quality*. Pure policy
samples are tightly clustered around argmax (policy is sharp); V can't
find winners in a tight cluster. Hybrid (some policy + some uniform)
gives V both baseline-quality candidates and the diverse alternatives
it needs to find genuine improvements over argmax.

Sides are randomized per game; result is the mixed-pool side's win rate.
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

NUM_GAMES = 100
NUM_WORKERS = 6

# Both pools have 40 total candidates (1 argmax + 39 sampled).
PURE_POLICY_PARAMS = {
    "K_INDEPENDENT_UNIFORM": 20,
    "INCLUDE_ARGMAX_FIRST": True,
    "N_POLICY_SAMPLES": 19,   # all 39 non-argmax slots are policy-sampled
    "M_ROLLOUTS": 16,
    "N_LOOKAHEAD": 2,
    "NUM_WORKERS": 1,
}
MIXED_PARAMS = {
    "K_INDEPENDENT_UNIFORM": 20,
    "INCLUDE_ARGMAX_FIRST": True,
    "N_POLICY_SAMPLES": 4,    # 9 policy-sampled, then 30 uniform
    "M_ROLLOUTS": 16,
    "N_LOOKAHEAD": 2,
    "NUM_WORKERS": 1,
}


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

    # Coin-flip: which physical side gets the mixed pool.
    mixed_side = "A" if random.random() < 0.5 else "B"
    if mixed_side == "A":
        params_a, params_b = MIXED_PARAMS, PURE_POLICY_PARAMS
    else:
        params_a, params_b = PURE_POLICY_PARAMS, MIXED_PARAMS

    # ml_planning=True means planner runs on *both* sides; the per-side
    # params are taken from planning_params_a / planning_params_b if
    # supported, otherwise we pass a single dict and have to use a
    # different mechanism. simulate_game accepts a single planning_params
    # dict — for asymmetric per-side params we need a different approach.
    # Workaround: monkey-patch via two separate planning_params on the
    # wrapper. simulate_game passes the same dict to both sides' planner
    # calls, so we need the per-side flag dispatched inside the planner.
    #
    # Cleanest: use ml_planning="A" for one side and a DIFFERENT
    # asymmetric mechanism for the other, but that path doesn't exist.
    # Instead we run TWO games with the same setup — one with mixed=A,
    # pure=B; one swapped — and aggregate. Done above via mixed_side.
    #
    # But we still need to pass per-side params. simulate_game forwards
    # planning_params to BOTH sides' planner calls. To get asymmetric
    # behaviour we route via a small extension: pass a special marker
    # dict that the planner can interpret per-side via the `player`
    # argument it already receives.
    #
    # Simplest workable hack without modifying game.py: pass a wrapper
    # dict and have the planner pick the right sub-dict based on `player`.

    planning_params = {
        "_per_side_params": {"A": params_a, "B": params_b},
    }
    result = simulate_game(
        ra, rb, mode="objectives", states_a=sa, states_b=sb,
        ml_model_a=_WORKER_MODEL, ml_model_b=_WORKER_MODEL,
        ml_planning=True,  # plan for both sides
        planning_params=planning_params,
    )
    if result == "draw":
        return "draw"
    return "mixed" if result == mixed_side else "pure"


def main():
    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof = json.load(f)
    ckpt = _DIR / "ml_checkpoints" / "final_model.pt"
    print(f"checkpoint: {ckpt}")
    print(f"games: {NUM_GAMES}, workers: {NUM_WORKERS}")
    print(f"side A pool (mixed coin-flip): 1 argmax + 9 policy + 30 uniform")
    print(f"side B pool (mixed coin-flip): 1 argmax + 39 policy")
    print()

    wins = {"mixed": 0, "pure": 0, "draw": 0}
    with mp.Pool(NUM_WORKERS, initializer=_worker_init, initargs=(ckpt, hof)) as pool:
        done = 0
        for r in pool.imap_unordered(_play, range(NUM_GAMES)):
            wins[r] += 1
            done += 1
            if done % 5 == 0 or done == NUM_GAMES:
                p = wins["mixed"] / done
                se = math.sqrt(p * (1 - p) / done)
                print(f"  {done:>4d}/{NUM_GAMES}  mixed={wins['mixed']:>3} "
                      f"pure={wins['pure']:>3} draws={wins['draw']:>3}  "
                      f"mixed_wr={p:.3f}±{se:.3f}", flush=True)

    n = sum(wins.values())
    p = wins["mixed"] / n
    se = math.sqrt(p * (1 - p) / n)
    z = (p - 0.5) / se if se > 0 else 0.0
    print()
    print(f"Final: n={n}  mixed={wins['mixed']}  pure={wins['pure']}  draws={wins['draw']}")
    print(f"mixed_wr (incl. draws) = {p:.4f} ± {se:.4f}  vs 0.500: z = {z:+.2f}")
    decisive = wins["mixed"] + wins["pure"]
    if decisive > 0:
        p_dec = wins["mixed"] / decisive
        se_dec = math.sqrt(p_dec * (1 - p_dec) / decisive)
        z_dec = (p_dec - 0.5) / se_dec
        print(f"Decisive games (n={decisive}): mixed_wr = {p_dec:.4f} ± {se_dec:.4f}  z={z_dec:+.2f}")


if __name__ == "__main__":
    main()
