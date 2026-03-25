"""Test side asymmetry in game simulation.

Tests:
1. Heuristic vs Heuristic (no ML) — is there a raw A-side advantage?
2. Same ML model on both sides — does A still win more?
3. ML(A) vs Heuristic(B) then ML(B) vs Heuristic(A) — side-swap comparison
4. Old checkpoint vs new checkpoint from both sides — is strength real?
"""

import sys, os, time, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Ensure project is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evolution import generate_random_army, resolve_army, _make_unit_states
from game import simulate_game
from ml_model_tactical import TacticalModel
from ml_training import load_model_state_dict


def load_model(path: str) -> TacticalModel:
    model = TacticalModel()
    model.load_state_dict(load_model_state_dict(path))
    model.eval()
    return model


def run_games_serial(n_games: int, ml_model_a=None, ml_model_b=None,
                     army_specs: list | None = None) -> dict:
    """Run n_games and return result counts.

    army_specs: list of (army_a, res_a, army_b, res_b) tuples.
    States are rebuilt fresh each game to avoid mutation issues.
    """
    import copy
    counts = {"A": 0, "B": 0, "draw": 0}
    for i in range(n_games):
        if army_specs:
            army_a, res_a, army_b, res_b = army_specs[i % len(army_specs)]
        else:
            army_a = generate_random_army(mode="objectives")
            army_b = generate_random_army(mode="objectives")
            res_a = resolve_army(army_a)
            res_b = resolve_army(army_b)

        # Fresh states every game (simulate_game mutates them)
        sa = _make_unit_states(copy.deepcopy(army_a), copy.deepcopy(res_a), "A")
        sb = _make_unit_states(copy.deepcopy(army_b), copy.deepcopy(res_b), "B")
        ra = copy.deepcopy(res_a)
        rb = copy.deepcopy(res_b)

        result = simulate_game(ra, rb, mode="objectives",
                               states_a=sa, states_b=sb,
                               ml_model_a=ml_model_a, ml_model_b=ml_model_b)
        counts[result] += 1
    return counts


def print_results(label: str, counts: dict, n: int):
    a_rate = (counts["A"] + 0.5 * counts["draw"]) / n
    print(f"  {label}")
    print(f"    A wins: {counts['A']:4d} ({100*counts['A']/n:.1f}%)")
    print(f"    B wins: {counts['B']:4d} ({100*counts['B']/n:.1f}%)")
    print(f"    Draws:  {counts['draw']:4d} ({100*counts['draw']/n:.1f}%)")
    print(f"    A win rate (draws=0.5): {a_rate:.3f}")
    print()


def main():
    N = 200  # games per test
    ckpt_dir = Path(__file__).resolve().parent / "ml_checkpoints"

    new_path = str(ckpt_dir / "final_model.pt")
    old_path = str(ckpt_dir / "checkpoint_batch_003400.pt")

    print(f"Loading models...")
    model_new = load_model(new_path)
    model_old = load_model(old_path)
    print(f"  New: {Path(new_path).name}")
    print(f"  Old: {Path(old_path).name}")
    print(f"  Games per test: {N}")
    print()

    # Pre-generate armies so we can reuse them for side-swap tests
    print("Generating armies...")
    army_specs = []
    for _ in range(N):
        a = generate_random_army(mode="objectives")
        b = generate_random_army(mode="objectives")
        ra, rb = resolve_army(a), resolve_army(b)
        army_specs.append((a, ra, b, rb))
    print()

    # ── Test 1: Heuristic vs Heuristic ──
    print("=" * 60)
    print("TEST 1: Heuristic(A) vs Heuristic(B) — raw side advantage")
    print("=" * 60)
    t = time.time()
    counts = run_games_serial(N, ml_model_a=None, ml_model_b=None,
                              army_specs=army_specs)
    print_results("Heuristic vs Heuristic", counts, N)
    print(f"  ({time.time()-t:.1f}s)")

    # ── Test 2: Same new model on both sides ──
    print("=" * 60)
    print("TEST 2: Same model(A) vs Same model(B) — ML side advantage")
    print("=" * 60)
    t = time.time()
    counts = run_games_serial(N, ml_model_a=model_new, ml_model_b=model_new,
                              army_specs=army_specs)
    print_results("Same ML model both sides", counts, N)
    print(f"  ({time.time()-t:.1f}s)")

    # ── Test 3: ML vs Heuristic, then swap sides ──
    print("=" * 60)
    print("TEST 3: ML vs Heuristic — side swap comparison")
    print("=" * 60)

    # 3a: ML as player A
    t = time.time()
    counts_a = run_games_serial(N, ml_model_a=model_new, ml_model_b=None,
                                army_specs=army_specs)
    print_results("ML=A vs Heuristic=B", counts_a, N)

    # 3b: ML as player B (swap armies so same matchup, different sides)
    swapped_specs = [(b, rb, a, ra) for (a, ra, b, rb) in army_specs]
    counts_b = run_games_serial(N, ml_model_a=None, ml_model_b=model_new,
                                army_specs=swapped_specs)
    print_results("Heuristic=A vs ML=B", counts_b, N)

    # ML's combined winrate from both sides
    ml_wins_as_a = counts_a["A"] + 0.5 * counts_a["draw"]
    ml_wins_as_b = counts_b["B"] + 0.5 * counts_b["draw"]
    combined = (ml_wins_as_a + ml_wins_as_b) / (2 * N)
    print(f"  ML combined win rate (both sides): {combined:.3f}")
    print(f"  Side A advantage: {(counts_a['A'] + 0.5*counts_a['draw'])/N - combined:.3f}")
    print(f"  ({time.time()-t:.1f}s)")

    # ── Test 4: New checkpoint vs Old checkpoint, both sides ──
    print()
    print("=" * 60)
    print("TEST 4: New model vs Old model — strength from both sides")
    print("=" * 60)

    # 4a: New=A, Old=B
    t = time.time()
    counts_4a = run_games_serial(N, ml_model_a=model_new, ml_model_b=model_old,
                                 army_specs=army_specs)
    print_results("New=A vs Old=B", counts_4a, N)

    # 4b: Old=A, New=B
    counts_4b = run_games_serial(N, ml_model_a=model_old, ml_model_b=model_new,
                                 army_specs=army_specs)
    print_results("Old=A vs New=B", counts_4b, N)

    new_wins_as_a = counts_4a["A"] + 0.5 * counts_4a["draw"]
    new_wins_as_b = counts_4b["B"] + 0.5 * counts_4b["draw"]
    new_combined = (new_wins_as_a + new_wins_as_b) / (2 * N)
    print(f"  New model combined win rate (both sides): {new_combined:.3f}")
    a_advantage = (counts_4a["A"] + 0.5*counts_4a["draw"])/N - new_combined
    print(f"  Side A advantage: {a_advantage:.3f}")
    print(f"  ({time.time()-t:.1f}s)")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
