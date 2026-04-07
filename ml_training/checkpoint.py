"""Checkpoint management, model loading, opponent scheduling, EMA baseline."""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn

from ml_model_tactical import TacticalModel


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

    Handles both the legacy format (raw state_dict) and the new format
    (dict with 'model_state_dict' and 'batch_num' keys).

    For old checkpoints without opponent conditioning, drops value_head.*
    keys (shape mismatch) so they reinitialise cleanly with strict=False.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    # Checkpoint compat: old checkpoints lack opponent_embedding and have
    # value_head.value_proj with wrong input dim (H vs H+OPP_EMBED_DIM).
    # Drop value_head keys so they reinitialise; policy heads are unaffected.
    if "opponent_embedding.weight" not in state_dict:
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("value_head.")}
    # Checkpoint compat: old checkpoints have direction_head/distance_head or
    # destination_head instead of dest_embed/dest_query_proj. Drop them so
    # the new pointer layers reinitialise.
    if "direction_head.weight" in state_dict or "destination_head.weight" in state_dict:
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("direction_head.")
                      and not k.startswith("distance_head.")
                      and not k.startswith("destination_head.")}
    return state_dict


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
        torch.save(model.state_dict(), path)
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
        opponent.load_state_dict(torch.load(path, weights_only=True), strict=False)
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
        return torch.load(path, weights_only=True)

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
        return torch.load(path, weights_only=True)

    def __len__(self) -> int:
        return len(self.entries)
