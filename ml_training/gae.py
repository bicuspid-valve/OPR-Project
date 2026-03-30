"""Generalized Advantage Estimation (GAE)."""
from __future__ import annotations

from ml_training.config import TacticalActivationRecord


def compute_gae(
    all_trajectories: list[list[TacticalActivationRecord]],
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[list[list[float]], list[list[float]]]:
    """Compute GAE advantages and returns for all episodes.

    Uses old_value from collection time (stored in TacticalActivationRecord).
    Returns (all_advantages, all_returns).
    """
    all_advantages: list[list[float]] = []
    all_returns: list[list[float]] = []

    for trajectory in all_trajectories:
        T = len(trajectory)
        advantages = [0.0] * T
        returns = [0.0] * T
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = trajectory[t + 1].old_value if t < T - 1 else 0.0
            delta = trajectory[t].reward + gamma * next_value - trajectory[t].old_value
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae
            returns[t] = last_gae + trajectory[t].old_value
        all_advantages.append(advantages)
        all_returns.append(returns)

    return all_advantages, all_returns
