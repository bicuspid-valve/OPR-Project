"""Test: ML vs ML (both ML HoF). 1500 games = 500 per condition.
Mortar side is randomised A/B in every batch. Three conditions:
  - normal:  no forcing (ML picks targets)
  - shifter: mortar forced to shoot any in-range shifter unit with 2+ models
  - efficient: mortar forced to shoot the in-range target with max expected
               points-worth of damage (expected_kills * points-per-model)
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


# Per-game hot state: which side (if any) has forced-targeting enabled, and which mode.
_FORCE_SIDE = ""   # "" = off, "A" or "B" = force that side's burst mortars
_FORCE_MODE = ""   # "shifter" | "efficient" | ""


def _expected_points_damage(attacker, target, evaluate_target) -> float:
    """Expected points-worth of damage dealt to `target` this volley.
    Uses evaluate_target's damage_score (= expected_wounds / target.points),
    converts back to expected_wounds, divides by tough to estimate kills,
    multiplies by points-per-model."""
    can_shoot, damage_score, _full = evaluate_target(attacker, target)
    if not can_shoot:
        return -1.0
    pts = max(target.unit.points, 1)
    expected_wounds = damage_score * pts
    wounds_per_kill = max(target.unit.tough, 1)
    models = max(target.unit.models, 1)
    expected_kills = expected_wounds / wounds_per_kill
    points_per_model = pts / models
    # Cap kills at alive models so we don't overcount
    expected_kills = min(expected_kills, target.models_alive)
    return expected_kills * points_per_model


def _install_force_patch():
    """Replace ml_integration_tactical.pick_target_from_ranking with a wrapper that,
    when _FORCE_SIDE is set, overrides Burst Mortar target selection on that side
    according to _FORCE_MODE."""
    import ml_integration_tactical as mit
    from combat import evaluate_target

    if getattr(mit, "_force_patched", False):
        return

    _orig = mit.pick_target_from_ranking

    def _patched(attacker, enemies, target_ranking):
        if (_FORCE_SIDE and attacker.owner == _FORCE_SIDE
                and _unit_has_burst_mortar(attacker)):
            if _FORCE_MODE == "shifter":
                for e in enemies:
                    if (e.models_alive >= 2
                            and e.unit.template_id == "shifters"):
                        can_shoot, _, _ = evaluate_target(attacker, e)
                        if can_shoot:
                            return e
            elif _FORCE_MODE == "efficient":
                best = None
                best_val = -1.0
                for e in enemies:
                    if e.models_alive <= 0:
                        continue
                    val = _expected_points_damage(attacker, e, evaluate_target)
                    if val > best_val:
                        best_val = val
                        best = e
                if best is not None:
                    return best
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
    """args = (cond: 'normal'|'shifter'|'efficient', mortar_side: 'A'|'B')
    Returns dict with result ('A'/'B'/'draw'/'skip'), cond, mortar_side."""
    global _FORCE_SIDE, _FORCE_MODE
    cond, mortar_side = args
    if cond == "normal":
        _FORCE_SIDE = ""
        _FORCE_MODE = ""
    else:
        _FORCE_SIDE = mortar_side
        _FORCE_MODE = cond  # "shifter" or "efficient"

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
        return {"result": "skip", "cond": cond, "mortar_side": mortar_side}

    result = simulate_game(
        res_a, res_b, mode="objectives", states_a=sa, states_b=sb,
        ml_model_a=_WORKER_MODEL, ml_model_b=_WORKER_MODEL,
    )
    return {"result": result, "cond": cond, "mortar_side": mortar_side}


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
    print(f"  {label} (actual: {total} games):")
    print(f"    Side A wins: {a:>3} ({a/total*100:5.1f}%)   "
          f"Side B wins: {b:>3} ({b/total*100:5.1f}%)   "
          f"Draws: {d:>3} ({d/total*100:5.1f}%)")
    print(f"    A win-rate (draws=0.5): {wr_a_half:5.1f}%")
    print(f"    Mortar-side win-rate (draws=0.5): {wr_m_half:5.1f}%  "
          f"[W/L/D = {m_w}/{m_l}/{m_d}]")


if __name__ == "__main__":
    NUM_GAMES_PER_COND = 2000
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

    # Build task list. Mortar side is randomised in ALL conditions.
    CONDS = ("normal", "shifter", "efficient")
    rng = random.Random(12345)
    tasks = []
    for cond in CONDS:
        for _ in range(NUM_GAMES_PER_COND):
            side = "A" if rng.random() < 0.5 else "B"
            tasks.append((cond, side))
    rng.shuffle(tasks)

    for cond in CONDS:
        a = sum(1 for t in tasks if t[0] == cond and t[1] == "A")
        b = sum(1 for t in tasks if t[0] == cond and t[1] == "B")
        print(f"{cond:>9}-batch side assignment: A={a}  B={b}")

    wins = {c: {"A": 0, "B": 0, "draw": 0} for c in CONDS}
    mortar_wins = {c: {"win": 0, "loss": 0, "draw": 0} for c in CONDS}
    by_side = {c: {"A": {"win": 0, "loss": 0, "draw": 0},
                   "B": {"win": 0, "loss": 0, "draw": 0}} for c in CONDS}
    skipped = {c: 0 for c in CONDS}
    done = {c: 0 for c in CONDS}

    GRAND = len(CONDS) * NUM_GAMES_PER_COND
    print(f"\nRunning {GRAND} games ({NUM_GAMES_PER_COND} per condition) "
          f"on {NUM_WORKERS} workers...\n")

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
                    cond = out["cond"]
                    ms = out["mortar_side"]
                    if r == "skip":
                        skipped[cond] += 1
                        pending.append(pool.apply_async(_play_one_game, args=((cond, ms),)))
                    else:
                        wins[cond][r] += 1
                        if r == ms:
                            mortar_wins[cond]["win"] += 1
                            bucket = "win"
                        elif r == "draw":
                            mortar_wins[cond]["draw"] += 1
                            bucket = "draw"
                        else:
                            mortar_wins[cond]["loss"] += 1
                            bucket = "loss"
                        by_side[cond][ms][bucket] += 1
                        done[cond] += 1
                        tot = sum(done.values())
                        progress_parts = "  ".join(
                            f"{c} {done[c]}/{NUM_GAMES_PER_COND}" for c in CONDS)
                        print(f"  progress: {tot}/{GRAND}  "
                              f"({progress_parts}  "
                              f"skipped {sum(skipped.values())})", end="\r")
                else:
                    still.append(fut)
            pending = still
            if pending:
                pending[0].wait(timeout=1.0)

    print()
    skip_parts = "  ".join(f"{c}={skipped[c]}" for c in CONDS)
    print(f"\nSkipped (no burst mortar on mortar-side after resolution): {skip_parts}")
    print()
    print("=" * 72)
    print("Results")
    print("=" * 72)
    LABELS = {
        "normal":    "Normal    (no force)                  ",
        "shifter":   "Shifter   (force -> 2+ model shifter) ",
        "efficient": "Efficient (force -> max points damage)",
    }
    for c in CONDS:
        _summarise(LABELS[c], wins[c], mortar_wins[c])

    # Delta table: mortar-side win rate vs. normal baseline
    def _wr(mw):
        t = sum(mw.values())
        return (mw["win"] + 0.5 * mw["draw"]) / t * 100 if t else 0.0

    base = _wr(mortar_wins["normal"])
    print()
    print(f"Mortar-side win-rate (draws=0.5):")
    for c in CONDS:
        wr = _wr(mortar_wins[c])
        delta = wr - base
        tag = " (baseline)" if c == "normal" else f"   (delta {delta:+.1f} pp)"
        print(f"  {c:>9}: {wr:5.1f}%{tag}")

    # Per-side breakdown
    print()
    print("Breakdown by mortar side (win-rate from mortar-side POV):")
    for c in CONDS:
        for side in ("A", "B"):
            d = by_side[c][side]
            n = d["win"] + d["loss"] + d["draw"]
            if n == 0:
                print(f"  {c:>9}, mortar on {side}: no games")
                continue
            wr = (d["win"] + 0.5 * d["draw"]) / n * 100
            print(f"  {c:>9}, mortar on {side}  ({n:3d} games):  "
                  f"W/L/D = {d['win']}/{d['loss']}/{d['draw']}   win-rate {wr:5.1f}%")
