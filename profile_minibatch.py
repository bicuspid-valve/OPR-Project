"""Quick profile: ppo_minibatch_games=64 vs 128 batch times."""
import os
import time

# Skip cgroup re-exec for profiling
os.environ["_ML_TRAIN_CGROUP"] = "1"

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from ml_training import TrainingConfig, run_training

def main():
    NUM_BATCHES = 5  # just enough to get stable timing

    base_kwargs = dict(
        batch_size=512,
        model_type="tactical",
        use_c_ext=True,
        worker_count=6,
        planning_rate=0.02,
        num_batches=NUM_BATCHES,
        checkpoint_dir="ml_checkpoints",  # resume from existing
    )

    results = {}

    for minibatch_games in [64, 128]:
        print("\n" + "=" * 70)
        print(f"PROFILING: ppo_minibatch_games={minibatch_games}")
        print("=" * 70)

        config = TrainingConfig(
            **base_kwargs,
            ppo_minibatch_games=minibatch_games,
        )
        t0 = time.time()
        model, metrics = run_training(config=config, verbose=True, restart=False)
        elapsed = time.time() - t0

        # Extract per-batch times from metrics
        batch_times = []
        for log in metrics.batch_logs:
            if "batch_time" in log:
                batch_times.append(log["batch_time"])

        avg = elapsed / NUM_BATCHES
        results[minibatch_games] = {
            "total": elapsed,
            "avg": avg,
            "batch_times": batch_times,
        }
        print(f"\n>>> minibatch={minibatch_games}: {elapsed:.1f}s total, {avg:.1f}s/batch")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    for mb, r in results.items():
        bt_str = ", ".join(f"{t:.1f}" for t in r["batch_times"]) if r["batch_times"] else "N/A"
        print(f"  ppo_minibatch_games={mb:>3d}: avg {r['avg']:.1f}s/batch  ({bt_str})")

    if 64 in results and 128 in results:
        diff = results[128]["avg"] - results[64]["avg"]
        pct = 100 * diff / results[64]["avg"]
        print(f"\n  Delta: {diff:+.1f}s/batch ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
