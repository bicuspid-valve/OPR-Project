"""Checkpoint management, model loading, opponent scheduling, EMA baseline."""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn

from ml_model_tactical import TacticalModel, DEFAULT_CORE_ITERS


# ---------------------------------------------------------------------------
# Baseline (exponential moving average)
# ---------------------------------------------------------------------------

class EMABaseline:
    """Exponential moving average baseline for advantage computation."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.value = 0.0
        self._initialized = False

    def update(self, game_reward: float) -> None:
        if not self._initialized:
            self.value = game_reward
            self._initialized = True
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * game_reward

    def get(self) -> float:
        return self.value


# ---------------------------------------------------------------------------
# Opponent scheduling
# ---------------------------------------------------------------------------

def get_heuristic_fraction(win_rate: float) -> float:
    """Determine heuristic opponent fraction from rolling win rate."""
    if win_rate < 0.55:
        return 0.50
    elif win_rate <= 0.65:
        return 0.30
    else:
        return 0.20


# ---------------------------------------------------------------------------
# Model factory & loading
# ---------------------------------------------------------------------------

def _make_model(model_type: str = "tactical") -> nn.Module:
    """Create a fresh model instance."""
    return TacticalModel()


def load_model_state_dict(path) -> dict:
    """Load a model state dict from a checkpoint file.

    Handles migration from older checkpoints that lack the side embedding
    by zero-padding the value_proj weight matrix.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Migrate value_proj if it's the old size (no side embedding)
    from ml_model_tactical import SIDE_EMBED_DIM
    vp_key = "value_head.value_proj.weight"
    if vp_key in sd:
        old_w = sd[vp_key]
        model = TacticalModel()
        expected_in = model.value_head.value_proj.in_features
        if old_w.shape[1] < expected_in:
            pad = expected_in - old_w.shape[1]
            sd[vp_key] = torch.cat([old_w, torch.zeros(1, pad)], dim=1)
        del model

    return sd


# ---------------------------------------------------------------------------
# Checkpoint pool
# ---------------------------------------------------------------------------

class CheckpointPool:
    """Manages a pool of past model checkpoints for self-play."""

    def __init__(self, max_size: int = 20, save_dir: str = "ml_checkpoints",
                 model_type: str = "tactical",
                 seed_existing: int = 0):
        self.max_size = max_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[Path] = []
        self.model_type = model_type

        # Optionally seed the pool with the N newest existing checkpoints
        if seed_existing > 0:
            existing = sorted(
                self.save_dir.glob("checkpoint_batch_*.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            for p in existing[-seed_existing:]:
                self.entries.append(p)

    def save(self, model: nn.Module, batch_num: int) -> None:
        """Save a checkpoint and add to pool, evicting oldest if full."""
        path = self.save_dir / f"checkpoint_batch_{batch_num:06d}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "n_iters": getattr(model, "n_iters", DEFAULT_CORE_ITERS),
        }, path)
        self.entries.append(path)
        # Evict oldest if over capacity
        while len(self.entries) > self.max_size:
            old = self.entries.pop(0)
            if old.exists():
                old.unlink()

    def sample_opponent(self) -> nn.Module | None:
        """Load a random checkpoint as an opponent. Returns None if pool is empty."""
        if not self.entries:
            return None
        path = random.choice(self.entries)
        if not path.exists():
            self.entries.remove(path)
            return None
        opponent = _make_model(self.model_type)
        ckpt = torch.load(path, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            opponent.load_state_dict(ckpt["model_state_dict"], strict=False)
            opponent.n_iters = ckpt.get("n_iters", DEFAULT_CORE_ITERS)
        else:
            # Legacy bare state_dict
            opponent.load_state_dict(ckpt, strict=False)
        opponent.eval()
        return opponent

    def sample_opponent_state_dict(self) -> dict | None:
        """Load a random checkpoint's state_dict (for passing to worker processes)."""
        if not self.entries:
            return None
        path = random.choice(self.entries)
        if not path.exists():
            self.entries.remove(path)
            return None
        ckpt = torch.load(path, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        return ckpt

    def sample_opponent_path(self) -> Path | None:
        """Return a random checkpoint path (without loading). Returns None if empty."""
        if not self.entries:
            return None
        path = random.choice(self.entries)
        if not path.exists():
            self.entries.remove(path)
            return None
        return path

    def load_state_dict(self, path: Path) -> dict:
        """Load a checkpoint's state_dict from the given path."""
        ckpt = torch.load(path, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        return ckpt

    def __len__(self) -> int:
        return len(self.entries)
