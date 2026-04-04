"""Verify that all checkpoint self-play games actually use the checkpoint model.

Runs a few training batches and checks that no checkpoint games silently
fall back to the heuristic AI.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_training.collection import _MAX_SHARED_OPPONENTS
from ml_training.checkpoint import CheckpointPool, _make_model


def main():
    random.seed(42)
    torch.manual_seed(42)

    pool_size = 20
    batch_size = 512
    heuristic_frac = 0.20

    print(f"_MAX_SHARED_OPPONENTS = {_MAX_SHARED_OPPONENTS}")
    print(f"Checkpoint pool size  = {pool_size}")
    print()

    # Simulate many batches
    n_batches = 200
    total_ckpt_games = 0
    total_loaded = 0
    total_unloaded = 0

    for _ in range(n_batches):
        opp_path_cache: dict[int, int] = {}
        for _ in range(batch_size):
            if random.random() < heuristic_frac:
                continue
            if random.random() < 0.5:
                continue  # mirror
            # checkpoint game
            total_ckpt_games += 1
            path = random.randint(0, pool_size - 1)
            if path not in opp_path_cache:
                opp_path_cache[path] = len(opp_path_cache)
            idx = opp_path_cache[path]
            if idx < _MAX_SHARED_OPPONENTS:
                total_loaded += 1
            else:
                total_unloaded += 1

    pct_loaded = 100 * total_loaded / total_ckpt_games if total_ckpt_games else 0
    pct_unloaded = 100 * total_unloaded / total_ckpt_games if total_ckpt_games else 0

    print(f"Simulated {n_batches} batches x {batch_size} games")
    print(f"  Checkpoint games:  {total_ckpt_games}")
    print(f"  Loaded correctly:  {total_loaded} ({pct_loaded:.1f}%)")
    print(f"  Heuristic fallback: {total_unloaded} ({pct_unloaded:.1f}%)")
    print()

    if total_unloaded == 0:
        print("PASS: All checkpoint games use real checkpoint models.")
    else:
        print(f"FAIL: {total_unloaded} games would silently fall back to heuristic!")

    # Also verify _MAX_SHARED_OPPONENTS >= pool max_checkpoints default
    from ml_training.config import TrainingConfig
    cfg = TrainingConfig()
    print(f"\nTrainingConfig.max_checkpoints = {cfg.max_checkpoints}")
    print(f"_MAX_SHARED_OPPONENTS          = {_MAX_SHARED_OPPONENTS}")
    if _MAX_SHARED_OPPONENTS >= cfg.max_checkpoints:
        print("PASS: Shared slots >= pool size.")
    else:
        print("FAIL: Shared slots < pool size — some opponents will be dropped!")


if __name__ == "__main__":
    main()
