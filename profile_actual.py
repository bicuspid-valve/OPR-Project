"""Profile actual run_training with timing patches on key phases."""
from __future__ import annotations
import os, sys, time, statistics
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch

# ---- Timing collector ----
_phase_times: dict[str, list[float]] = {}
def _rec(name, dt):
    _phase_times.setdefault(name, []).append(dt)

# ---- Patch pool.map to time episode collection ----
import multiprocessing.pool as _mp_pool
_orig_map = _mp_pool.Pool.map
def _timed_map(self, func, iterable, chunksize=None):
    t0 = time.perf_counter()
    r = _orig_map(self, func, iterable, chunksize)
    _rec("pool_map", time.perf_counter() - t0)
    return r
_mp_pool.Pool.map = _timed_map

# ---- Patch replay_tactical_log_probs_flat ----
import ml_training.loss as _lmod
_orig_replay = _lmod.replay_tactical_log_probs_flat
def _timed_replay(*a, **kw):
    t0 = time.perf_counter()
    r = _orig_replay(*a, **kw)
    _rec("replay_fwd", time.perf_counter() - t0)
    return r
_lmod.replay_tactical_log_probs_flat = _timed_replay

# ---- Patch compute_loss_flat ----
_orig_loss = _lmod.compute_loss_flat
def _timed_loss(*a, **kw):
    t0 = time.perf_counter()
    r = _orig_loss(*a, **kw)
    _rec("compute_loss", time.perf_counter() - t0)
    return r
_lmod.compute_loss_flat = _timed_loss

# ---- Patch compute_gae ----
import ml_training.gae as _gmod
_orig_gae = _gmod.compute_gae
def _timed_gae(*a, **kw):
    t0 = time.perf_counter()
    r = _orig_gae(*a, **kw)
    _rec("gae", time.perf_counter() - t0)
    return r
_gmod.compute_gae = _timed_gae

# ---- Patch loss.backward ----
_orig_backward = torch.Tensor.backward
def _timed_backward(self, *a, **kw):
    t0 = time.perf_counter()
    r = _orig_backward(self, *a, **kw)
    _rec("backward", time.perf_counter() - t0)
    return r
torch.Tensor.backward = _timed_backward

# ---- Patch optimizer.step ----
_orig_step = torch.optim.Adam.step
def _timed_step(self, *a, **kw):
    t0 = time.perf_counter()
    r = _orig_step(self, *a, **kw)
    _rec("optim_step", time.perf_counter() - t0)
    return r
torch.optim.Adam.step = _timed_step

# ---- Patch references in loop module (which has already-bound imports) ----
import ml_training.loop as _loop_mod
_loop_mod.replay_tactical_log_probs_flat = _timed_replay
_loop_mod.compute_loss_flat = _timed_loss
_loop_mod.compute_gae = _timed_gae

# ---- Now import and run ----
from ml_training import TrainingConfig, run_training

def main():
    config = TrainingConfig(
        num_batches=5,
        batch_size=512,
        time_limit=None,
        checkpoint_dir="ml_checkpoints_profile",
        model_type="tactical",
        use_c_ext=True,
        worker_count=6,
        planning_rate=0.0,  # disable for profiling (normally 0.03)
    )

    print(f"Profiling: {config.num_batches} batches × {config.batch_size} games")
    print(f"PPO epochs: {config.ppo_epochs}, minibatch: {config.ppo_minibatch_games}")
    print()

    t0 = time.perf_counter()
    model, metrics = run_training(config=config, verbose=True, restart=True)
    total = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print("TIMING BREAKDOWN")
    print("=" * 70)
    print(f"Total: {total:.2f}s  ({config.num_batches} batches, avg {total/config.num_batches:.2f}s/batch)\n")

    for name in ["pool_map", "gae", "replay_fwd", "compute_loss", "backward", "optim_step"]:
        ts = _phase_times.get(name, [])
        if ts:
            s = sum(ts)
            print(f"  {name:<20} {s:>8.2f}s  ({s/total*100:>5.1f}%)  calls={len(ts):>5}  avg={statistics.mean(ts)*1000:.1f}ms")

    # Derived
    pool = sum(_phase_times.get("pool_map", []))
    replay = sum(_phase_times.get("replay_fwd", []))
    loss = sum(_phase_times.get("compute_loss", []))
    bwd = sum(_phase_times.get("backward", []))
    opt = sum(_phase_times.get("optim_step", []))
    gae = sum(_phase_times.get("gae", []))
    ppo = replay + loss + bwd + opt
    other = total - pool - gae - ppo

    print(f"\nSUMMARY:")
    print(f"  Episode collection:  {pool:>8.2f}s  ({pool/total*100:>5.1f}%)")
    print(f"  GAE:                 {gae:>8.2f}s  ({gae/total*100:>5.1f}%)")
    print(f"  PPO update total:    {ppo:>8.2f}s  ({ppo/total*100:>5.1f}%)")
    print(f"    replay fwd:        {replay:>8.2f}s")
    print(f"    loss compute:      {loss:>8.2f}s")
    print(f"    backward:          {bwd:>8.2f}s")
    print(f"    optimizer step:    {opt:>8.2f}s")
    print(f"  Other (spec build,   {other:>8.2f}s  ({other/total*100:>5.1f}%)")
    print(f"   weight sync, etc.)")
    print(f"\n  Games/sec: {config.num_batches * config.batch_size / total:.1f}")

if __name__ == "__main__":
    main()
