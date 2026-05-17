"""Probe: standard policy vs policy with pp_v_dest dest-override.

Side X: standard inference EXCEPT when MOVE_MOVE is selected, the
        destination is replaced by the pp_v_dest top-1 of 100 uniform-
        random legal dest candidates. Charge & shoot are re-derived
        from the new post-move state.
Side Y: pure standard inference (policy's dest_logits argmax).

Tests whether pp_v_dest is a useful cheap dest-selection mechanism
relative to the policy's own dest head. Different question from the
earlier rollout-vs-pp_v_dest calibration probe.
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
NUM_DESTS = 100  # X side: pick best of N random by pp_v_dest


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

    # Coin-flip: which physical side gets the X (dest-override) treatment.
    x_side = "A" if random.random() < 0.5 else "B"
    pp = {
        "DEST_OVERRIDE_PP_V_N": NUM_DESTS,
        "NUM_WORKERS": 1,
    }
    result = simulate_game(
        ra, rb, mode="objectives", states_a=sa, states_b=sb,
        ml_model_a=_WORKER_MODEL, ml_model_b=_WORKER_MODEL,
        ml_planning=x_side,           # plan_activation fires only for X
        planning_params=pp,
    )
    if result == "draw":
        return "draw"
    return "X" if result == x_side else "Y"


def main():
    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof = json.load(f)
    ckpt = _DIR / "ml_checkpoints" / "final_model.pt"
    print(f"checkpoint: {ckpt}")
    print(f"games: {NUM_GAMES}, workers: {NUM_WORKERS}")
    print(f"X = standard inference + dest override (pp_v_dest top-1 of {NUM_DESTS} random)")
    print(f"Y = standard inference (policy dest_logits argmax)")
    print()

    wins = {"X": 0, "Y": 0, "draw": 0}
    with mp.Pool(NUM_WORKERS, initializer=_worker_init, initargs=(ckpt, hof)) as pool:
        done = 0
        for r in pool.imap_unordered(_play, range(NUM_GAMES)):
            wins[r] += 1
            done += 1
            if done % 5 == 0 or done == NUM_GAMES:
                p = wins["X"] / done
                se = math.sqrt(p * (1 - p) / done)
                print(f"  {done:>4d}/{NUM_GAMES}  X={wins['X']:>3} "
                      f"Y={wins['Y']:>3} draws={wins['draw']:>3}  "
                      f"X_wr={p:.3f}±{se:.3f}", flush=True)

    n = sum(wins.values())
    p = wins["X"] / n
    se = math.sqrt(p * (1 - p) / n)
    z = (p - 0.5) / se if se > 0 else 0.0
    print()
    print(f"Final: n={n}  X={wins['X']}  Y={wins['Y']}  draws={wins['draw']}")
    print(f"X_wr (incl. draws) = {p:.4f} ± {se:.4f}  vs 0.500: z = {z:+.2f}")
    decisive = wins["X"] + wins["Y"]
    if decisive > 0:
        p_dec = wins["X"] / decisive
        se_dec = math.sqrt(p_dec * (1 - p_dec) / decisive)
        z_dec = (p_dec - 0.5) / se_dec if se_dec > 0 else 0.0
        print(f"Decisive games (n={decisive}): X_wr = {p_dec:.4f} ± {se_dec:.4f}  z={z_dec:+.2f}")


if __name__ == "__main__":
    main()
