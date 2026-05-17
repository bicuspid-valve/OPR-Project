"""Phase 2 plumbing smoke test.

Runs run_training() for a tiny number of batches under three configs to
prove that the new loss machinery works end-to-end without crashing:

  (A) Legacy ce_chosen mode, no KL trust region — same as before MPO.
  (B) New mpo_marginal mode, kl_trust_region_beta = 0 — exercises the
      ungated soft-KL path with η scaling, but no model_old forward.
  (C) New mpo_marginal mode, kl_trust_region_beta > 0 — exercises the
      full pipeline including the per-batch model_old refresh, the
      no-grad model_old forward, and the KL trust region term.

Each config runs for `NUM_BATCHES` batches in an isolated checkpoint
directory so neither config sees the others' state. The probe asserts:
  - run_training returns without raising
  - The reported `loss` is finite for every batch
  - For (C) only: kl_trust_region_loss > 0 at least once

Usage: .venv/bin/python3 probe_mpo_phase2.py
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import traceback
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import fast_core
fast_core.USE_C_EXT = fast_core.is_available()

from ml_training.config import TrainingConfig
from ml_training.loop import run_training

NUM_BATCHES = 2


def _make_cfg(label: str, **overrides) -> TrainingConfig:
    base = dict(
        num_batches=NUM_BATCHES,
        batch_size=4,
        ppo_epochs=1,
        ppo_minibatch_games=2,
        # Planning must be on for any distill loss to fire. Set it high
        # so the rollouts always plan and we get planning data on every
        # activation.
        planning_rate=1.0,
        planning_warmup_batches=0,
        planning_distill_ramp_batches=0,
        planning_distill_max_weight=0.1,
        # Cheap planning so the test runs in seconds, not minutes.
        training_planning_K=2,
        training_planning_C=2,
        training_planning_M=2,
        training_planning_N=1,
        training_planning_sequential_halving=False,
        checkpoint_interval=1000,  # don't save during smoke test
        checkpoint_dir=f"ml_checkpoints_phase2_{label}",
        log_dir=f"ml_logs_phase2_{label}",
        worker_count=1,
        device="cpu",
    )
    base.update(overrides)
    return TrainingConfig(**base)


def _clean(label: str) -> None:
    for d in (f"ml_checkpoints_phase2_{label}", f"ml_logs_phase2_{label}"):
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)


def _run(label: str, cfg: TrainingConfig) -> tuple[bool, str]:
    """Run training; return (ok, message)."""
    _clean(label)
    print(f"\n{'='*60}")
    print(f"  CONFIG ({label}): planning_distill_mode={cfg.planning_distill_mode!r}, "
          f"β={cfg.kl_trust_region_beta}, η={cfg.mpo_eta}")
    print(f"{'='*60}")
    try:
        model, metrics = run_training(config=cfg, verbose=False)
    except Exception:
        traceback.print_exc()
        return False, "run_training raised"

    # Check the metrics object — batch_logs is a list of per-batch dicts.
    logs = list(getattr(metrics, "batch_logs", []) or [])
    if not logs:
        return False, "no batch_logs recorded"
    losses = [d.get("loss") for d in logs if d.get("loss") is not None]
    if not losses:
        return False, "no loss values in batch_logs"
    bad = [(i, x) for i, x in enumerate(losses) if not math.isfinite(x)]
    if bad:
        return False, f"non-finite loss at batches: {bad}"

    kl_vals = [d.get("kl_trust_region_loss", 0.0) for d in logs]
    distill_vals = [d.get("planning_distill_loss", 0.0) for d in logs]
    print(f"  ✓ ran for {len(losses)} batches")
    print(f"    losses:        {[f'{x:+.4f}' for x in losses]}")
    print(f"    distill_loss:  {[f'{x:+.4f}' for x in distill_vals]}")
    print(f"    kl_trust_loss: {[f'{x:+.4f}' for x in kl_vals]}")

    if cfg.kl_trust_region_beta > 0 and all(abs(x) < 1e-9 for x in kl_vals):
        return False, ("KL trust region was active but kl_trust_region_loss "
                       "stayed at 0 — model_old wiring is dormant")
    return True, "ok"


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    cfg_a = _make_cfg(
        "ce_chosen",
        planning_distill_mode="ce_chosen",
        kl_trust_region_beta=0.0,
    )
    results.append(("(A) ce_chosen, β=0", *_run("ce_chosen", cfg_a)))

    cfg_b = _make_cfg(
        "mpo_no_kl",
        planning_distill_mode="mpo_marginal",
        kl_trust_region_beta=0.0,
        mpo_eta=1.0,
    )
    results.append(("(B) mpo_marginal, β=0, η=1", *_run("mpo_no_kl", cfg_b)))

    cfg_c = _make_cfg(
        "mpo_with_kl",
        planning_distill_mode="mpo_marginal",
        kl_trust_region_beta=0.1,
        mpo_eta=1.0,
    )
    results.append(("(C) mpo_marginal, β=0.1, η=1", *_run("mpo_with_kl", cfg_c)))

    # (D) Switch test: 1 batch in legacy mode, then flip to mpo_marginal
    # with the trust region active. Verifies mpo_switch_batch routing —
    # batch 1 should report β=0 in metrics, batches 2-3 should report β>0.
    cfg_d = _make_cfg(
        "mpo_switch",
        num_batches=3,
        planning_distill_mode="ce_chosen",
        kl_trust_region_beta=1.0,
        mpo_eta=1.0,
        mpo_switch_batch=1,
    )
    print(f"\n{'='*60}")
    print(f"  CONFIG (mpo_switch): mpo_switch_batch=1, β_post=1.0")
    print(f"{'='*60}")
    _clean("mpo_switch")
    try:
        _, metrics = run_training(config=cfg_d, verbose=False)
    except Exception:
        traceback.print_exc()
        results.append(("(D) switch at batch 1", False, "run_training raised"))
    else:
        logs = list(getattr(metrics, "batch_logs", []) or [])
        # Per-batch β read from the metrics dict — exposed in compute_loss_flat.
        betas = [d.get("kl_trust_region_beta", 0.0) for d in logs]
        print(f"  per-batch β: {betas}")
        if len(betas) < 3:
            results.append(("(D) switch at batch 1", False,
                            f"expected 3 batches, got {len(betas)}"))
        elif betas[0] != 0.0:
            results.append(("(D) switch at batch 1", False,
                            f"pre-switch batch 1 should have β=0, got {betas[0]}"))
        elif any(b == 0.0 for b in betas[1:]):
            results.append(("(D) switch at batch 1", False,
                            f"post-switch batches should have β>0, got {betas[1:]}"))
        else:
            print(f"  ✓ pre-switch β=0, post-switch β>0")
            results.append(("(D) switch at batch 1", True, "ok"))
        _clean("mpo_switch")

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    failed = 0
    for name, ok, msg in results:
        marker = "✓" if ok else "✗"
        print(f"  {marker}  {name}: {msg}")
        if not ok:
            failed += 1

    # Cleanup test artefacts unless something failed (in which case leave
    # them for inspection).
    if failed == 0:
        for label in ("ce_chosen", "mpo_no_kl", "mpo_with_kl"):
            _clean(label)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
