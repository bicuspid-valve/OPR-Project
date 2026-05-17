"""Probe: ML policy vs ML planner with K candidates where the FIRST is
the policy's joint argmax and the remaining K-1 are uniform-random.

Differs from probe_planner_vs_policy.py only in INCLUDE_ARGMAX_FIRST=True.

Question: how much does access to K-1 random alternatives improve over
playing pure policy? V can never do worse than picking candidate 0
(argmax) and recovering policy behaviour, so:

  K=1: only argmax → planner plays pure policy → ~50% mirror baseline.
  K>1: planner plays V-best of {argmax} ∪ {K-1 random}.

  > 50%: V usefully identifies improvements over argmax in random
         alternatives. The gap above 50% measures the marginal
         contribution of planning over pure policy.
  ≈ 50%: V is roughly neutral — most random alternatives are worse,
         V keeps argmax; rare improvements wash out against rare V
         mistakes.
  < 50%: V is doing net harm — its mistakes (preferring a random
         action that's actually worse than argmax) outweigh its
         successes. Suggests V's per-state ranking is too noisy
         relative to the argmax-vs-random separation.
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

K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NUM_GAMES_PER_K = 100
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
    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    _WORKER_MODEL = model
    _WORKER_HOF_ML = hof_ml_data


def _play_one_game(args):
    K, _idx = args
    army_a = load_army_from_hof(random.choice(_WORKER_HOF_ML))
    army_b = load_army_from_hof(random.choice(_WORKER_HOF_ML))
    res_a = resolve_army(army_a)
    res_b = resolve_army(army_b)
    sa = _make_unit_states(army_a, res_a, "A")
    sb = _make_unit_states(army_b, res_b, "B")

    planner_side = "A" if random.random() < 0.5 else "B"
    planning_params = {
        "K_INDEPENDENT_UNIFORM": K,
        "UNIFORM_ALT_SAMPLING": True,
        "INCLUDE_ARGMAX_FIRST": True,
        "M_ROLLOUTS": 16,
        "N_LOOKAHEAD": 2,
        "NUM_WORKERS": 1,
    }
    result = simulate_game(
        res_a, res_b, mode="objectives",
        states_a=sa, states_b=sb,
        ml_model_a=_WORKER_MODEL, ml_model_b=_WORKER_MODEL,
        ml_planning=planner_side,
        planning_params=planning_params,
    )
    if result == "draw":
        return "draw"
    return "planner" if result == planner_side else "policy"


def main():
    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof_ml_data = json.load(f)
    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    print(f"checkpoint: {checkpoint_path}")
    print(f"K values: {K_VALUES}, games per K: {NUM_GAMES_PER_K}, workers: {NUM_WORKERS}")
    print(f"INCLUDE_ARGMAX_FIRST = True (slot 0 = policy argmax, slots 1..K-1 = random)")
    print()

    results = {}
    with mp.Pool(
        processes=NUM_WORKERS, initializer=_worker_init,
        initargs=(checkpoint_path, hof_ml_data),
    ) as pool:
        for K in K_VALUES:
            wins = {"planner": 0, "policy": 0, "draw": 0}
            args_list = [(K, i) for i in range(NUM_GAMES_PER_K)]
            done = 0
            for r in pool.imap_unordered(_play_one_game, args_list):
                wins[r] += 1
                done += 1
                if done % 10 == 0 or done == NUM_GAMES_PER_K:
                    print(f"  K={K:2d}: {done}/{NUM_GAMES_PER_K}", end="\r", flush=True)
            results[K] = wins
            n = sum(wins.values())
            wr = wins["planner"] / n
            se = math.sqrt(wr * (1 - wr) / n) if n > 0 else 0.0
            print(f"  K={K:2d}: planner {wins['planner']:>3}/{n}  "
                  f"({wr:.3f}±{se:.3f})  policy {wins['policy']:>3}, "
                  f"draws {wins['draw']:>3}                ")

    print()
    print(f"{'K':>3s}  {'n':>5s}  {'planner_wr':>13s}  {'policy_wr':>10s}  {'draws':>6s}")
    print("-" * 50)
    for K in K_VALUES:
        wins = results[K]
        n = sum(wins.values())
        wr = wins["planner"] / n
        se = math.sqrt(wr * (1 - wr) / n) if n > 0 else 0.0
        wr_str = f"{wr:.3f}±{se:.3f}"
        print(f"{K:>3d}  {n:>5d}  {wr_str:>13s}  "
              f"{wins['policy']/n:>10.3f}  {wins['draw']:>6d}")

    csv_path = _DIR / "ml_logs" / "planner_argmax_first.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("K,n,planner_wins,policy_wins,draws,planner_wr,se\n")
        for K in K_VALUES:
            wins = results[K]
            n = sum(wins.values())
            wr = wins["planner"] / n
            se = math.sqrt(wr * (1 - wr) / n) if n > 0 else 0.0
            f.write(f"{K},{n},{wins['planner']},{wins['policy']},{wins['draw']},"
                    f"{wr:.6f},{se:.6f}\n")
    print(f"\nSaved CSV to {csv_path}")


if __name__ == "__main__":
    main()
