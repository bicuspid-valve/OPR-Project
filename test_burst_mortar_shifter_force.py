"""Test: ML vs ML (both ML HoF). 500 games normal + 500 games forced.
Normal batch: side A always carries a Burst Mortar (baseline).
Forced batch: randomise which side (A or B) carries the Burst Mortar; that side's
burst mortar is forced to shoot any in-range shifter unit with 2+ models.
Compare win-rate from the mortar side's perspective.
"""

import json
import random
import multiprocessing as mp
from pathlib import Path

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from ml_training import load_model_state_dict
from ml_model_tactical import TacticalModel
from evolution import make_entry, resolve_army, _make_unit_states
from models import ArmyList
from game import simulate_game

_DIR = Path(__file__).resolve().parent

# Worker globals
_WORKER_MODEL = None
_WORKER_HOF_MORTAR = None  # Armies with Burst Mortar (for the mortar side)
_WORKER_HOF_ANY = None     # Any ML HoF army (for the non-mortar side)


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


def entry_has_burst_mortar(hof_entry: dict) -> bool:
    for x in hof_entry["entries"]:
        if x["template_id"] == "support_artillery":
            mw = x.get("upgrades", {}).get("main_weapon")
            if mw not in ("heavy_mortar", "aa_cannon"):
                return True
    return False


def _unit_has_burst_mortar(u) -> bool:
    for w in u.unit.weapons:
        if w.name == "Burst Mortar":
            return True
    return False


# Per-game hot state: which side (if any) has forced-targeting enabled.
_FORCE_SIDE = ""  # "" = off, "A" or "B" = force that side's burst mortars


def _install_force_patch():
    """Replace ml_integration_tactical.pick_target_from_ranking with a wrapper that,
    when _FORCE_SIDE is set, forces any Burst Mortar on that side to target an
    in-range shifter unit with 2+ models (if one exists)."""
    import ml_integration_tactical as mit
    from combat import evaluate_target

    if getattr(mit, "_force_patched", False):
        return

    _orig = mit.pick_target_from_ranking

    def _patched(attacker, enemies, target_ranking):
        if _FORCE_SIDE and attacker.owner == _FORCE_SIDE and _unit_has_burst_mortar(attacker):
            for e in enemies:
                if (e.models_alive >= 2
                        and e.unit.template_id == "shifters"):
                    can_shoot, _, _ = evaluate_target(attacker, e)
                    if can_shoot:
                        return e
        return _orig(attacker, enemies, target_ranking)

    mit.pick_target_from_ranking = _patched
    mit._force_patched = True


def _worker_init(checkpoint_path, hof_mortar_data, hof_any_data):
    global _WORKER_MODEL, _WORKER_HOF_MORTAR, _WORKER_HOF_ANY
    import torch
    torch.set_num_threads(1)
    random.seed()

    _install_force_patch()

    state_dict = load_model_state_dict(checkpoint_path)
    model = TacticalModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    _WORKER_MODEL = model
    _WORKER_HOF_MORTAR = hof_mortar_data
    _WORKER_HOF_ANY = hof_any_data


def _play_one_game(args):
    """args = (force_mode: 0|1, mortar_side: 'A'|'B')
    Returns dict with result ('A'/'B'/'draw'/'skip'), force_mode, mortar_side."""
    global _FORCE_SIDE
    force_mode, mortar_side = args
    _FORCE_SIDE = mortar_side if force_mode else ""

    # Mortar-side army drawn from filtered pool, other side from full pool.
    mortar_army = load_army_from_hof(random.choice(_WORKER_HOF_MORTAR))
    other_army = load_army_from_hof(random.choice(_WORKER_HOF_ANY))
    if mortar_side == "A":
        army_a, army_b = mortar_army, other_army
    else:
        army_a, army_b = other_army, mortar_army

    res_a = resolve_army(army_a)
    res_b = resolve_army(army_b)
    sa = _make_unit_states(army_a, res_a, "A")
    sb = _make_unit_states(army_b, res_b, "B")

    # Safety check: did the mortar side actually end up with a Burst Mortar?
    mortar_states = sa if mortar_side == "A" else sb
    if not any(_unit_has_burst_mortar(u) for u in mortar_states):
        return {"result": "skip", "force_mode": force_mode, "mortar_side": mortar_side}

    result = simulate_game(
        res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
        ml_model_a=_WORKER_MODEL, ml_model_b=_WORKER_MODEL,
    )
    return {"result": result, "force_mode": force_mode, "mortar_side": mortar_side}


def _summarise(label, wins, mortar_wins):
    total = wins["A"] + wins["B"] + wins["draw"]
    if total == 0:
        print(f"  {label}: no games")
        return
    # Side-A perspective
    a = wins["A"]; b = wins["B"]; d = wins["draw"]
    wr_a_half = (a + 0.5 * d) / total * 100
    # Mortar-side perspective
    m_w, m_l, m_d = mortar_wins["win"], mortar_wins["loss"], mortar_wins["draw"]
    m_tot = m_w + m_l + m_d
    wr_m_half = (m_w + 0.5 * m_d) / m_tot * 100 if m_tot else 0.0
    print(f"  {label} ({total} games):")
    print(f"    Side A wins: {a:>3} ({a/total*100:5.1f}%)   "
          f"Side B wins: {b:>3} ({b/total*100:5.1f}%)   "
          f"Draws: {d:>3} ({d/total*100:5.1f}%)")
    print(f"    A win-rate (draws=0.5): {wr_a_half:5.1f}%")
    print(f"    Mortar-side win-rate (draws=0.5): {wr_m_half:5.1f}%  "
          f"[W/L/D = {m_w}/{m_l}/{m_d}]")


if __name__ == "__main__":
    NUM_GAMES_PER_COND = 500
    NUM_WORKERS = 8

    with open(_DIR / "results" / "hall_of_fame_ml.json") as f:
        hof_ml_data = json.load(f)

    hof_mortar_data = [e for e in hof_ml_data if entry_has_burst_mortar(e)]
    hof_any_data = hof_ml_data
    print(f"ML HoF: {len(hof_ml_data)} armies. {len(hof_mortar_data)} have a Burst Mortar.")
    if not hof_mortar_data:
        raise SystemExit("No ML HoF armies with Burst Mortar — nothing to test.")

    checkpoint_path = _DIR / "ml_checkpoints" / "final_model.pt"
    print(f"Using model checkpoint: {checkpoint_path}")

    # Build task list. Mortar side is randomised in BOTH conditions so the only
    # difference is the force rule.
    rng = random.Random(12345)
    tasks = []
    for fm in (0, 1):
        for _ in range(NUM_GAMES_PER_COND):
            side = "A" if rng.random() < 0.5 else "B"
            tasks.append((fm, side))
    rng.shuffle(tasks)

    for fm, label in ((0, "Normal"), (1, "Forced")):
        a = sum(1 for t in tasks if t[0] == fm and t[1] == "A")
        b = sum(1 for t in tasks if t[0] == fm and t[1] == "B")
        print(f"{label}-batch side assignment: A={a}  B={b}")

    wins = {0: {"A": 0, "B": 0, "draw": 0},
            1: {"A": 0, "B": 0, "draw": 0}}
    mortar_wins = {0: {"win": 0, "loss": 0, "draw": 0},
                   1: {"win": 0, "loss": 0, "draw": 0}}
    # Per-batch breakdown by mortar side
    by_side = {0: {"A": {"win": 0, "loss": 0, "draw": 0},
                   "B": {"win": 0, "loss": 0, "draw": 0}},
               1: {"A": {"win": 0, "loss": 0, "draw": 0},
                   "B": {"win": 0, "loss": 0, "draw": 0}}}
    skipped = {0: 0, 1: 0}

    print(f"\nRunning {2 * NUM_GAMES_PER_COND} games ({NUM_GAMES_PER_COND} normal + "
          f"{NUM_GAMES_PER_COND} forced) on {NUM_WORKERS} workers...\n")

    done = {0: 0, 1: 0}

    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=_worker_init,
        initargs=(checkpoint_path, hof_mortar_data, hof_any_data),
    ) as pool:
        pending = [pool.apply_async(_play_one_game, args=(t,)) for t in tasks]

        while pending:
            still = []
            for fut in pending:
                if fut.ready():
                    out = fut.get()
                    r = out["result"]
                    fm = out["force_mode"]
                    ms = out["mortar_side"]
                    if r == "skip":
                        skipped[fm] += 1
                        pending.append(pool.apply_async(_play_one_game, args=((fm, ms),)))
                    else:
                        wins[fm][r] += 1
                        if r == ms:
                            mortar_wins[fm]["win"] += 1
                            bucket = "win"
                        elif r == "draw":
                            mortar_wins[fm]["draw"] += 1
                            bucket = "draw"
                        else:
                            mortar_wins[fm]["loss"] += 1
                            bucket = "loss"
                        by_side[fm][ms][bucket] += 1
                        done[fm] += 1
                        tot = done[0] + done[1]
                        grand = 2 * NUM_GAMES_PER_COND
                        print(f"  progress: {tot}/{grand}  "
                              f"(normal {done[0]}/{NUM_GAMES_PER_COND}, "
                              f"forced {done[1]}/{NUM_GAMES_PER_COND}, "
                              f"skipped {skipped[0] + skipped[1]})", end="\r")
                else:
                    still.append(fut)
            pending = still
            if pending:
                pending[0].wait(timeout=1.0)

    print()
    print(f"\nSkipped (no burst mortar on mortar-side after resolution): "
          f"normal={skipped[0]}  forced={skipped[1]}")
    print()
    print("=" * 72)
    print("Results")
    print("=" * 72)
    _summarise("Normal   (mortar on A, no force)         ", wins[0], mortar_wins[0])
    _summarise("Forced   (random mortar side -> shifter) ", wins[1], mortar_wins[1])

    n_n = sum(wins[0].values())
    n_f = sum(wins[1].values())
    m_n = sum(mortar_wins[0].values())
    m_f = sum(mortar_wins[1].values())
    a_n = (wins[0]["A"] + 0.5 * wins[0]["draw"]) / n_n * 100 if n_n else 0
    a_f = (wins[1]["A"] + 0.5 * wins[1]["draw"]) / n_f * 100 if n_f else 0
    mrt_n = (mortar_wins[0]["win"] + 0.5 * mortar_wins[0]["draw"]) / m_n * 100 if m_n else 0
    mrt_f = (mortar_wins[1]["win"] + 0.5 * mortar_wins[1]["draw"]) / m_f * 100 if m_f else 0
    print()
    print(f"Side-A win-rate    (draws=0.5):  normal {a_n:5.1f}%  ->  "
          f"forced {a_f:5.1f}%   (delta {a_f - a_n:+.1f} pp)")
    print(f"Mortar-side win-rate (draws=0.5):  normal {mrt_n:5.1f}%  ->  "
          f"forced {mrt_f:5.1f}%   (delta {mrt_f - mrt_n:+.1f} pp)")

    # Per-side breakdown of mortar-side win-rate for both batches
    print()
    print("Breakdown by mortar side (win-rate from mortar-side POV):")
    for fm, label in ((0, "normal"), (1, "forced")):
        for side in ("A", "B"):
            d = by_side[fm][side]
            n = d["win"] + d["loss"] + d["draw"]
            if n == 0:
                print(f"  {label}, mortar on {side}: no games")
                continue
            wr = (d["win"] + 0.5 * d["draw"]) / n * 100
            print(f"  {label}, mortar on {side}  ({n:3d} games):  "
                  f"W/L/D = {d['win']}/{d['loss']}/{d['draw']}   win-rate {wr:5.1f}%")
