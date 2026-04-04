"""Generalized Advantage Estimation (GAE)."""
from __future__ import annotations

from collections import defaultdict

from ml_training.config import TacticalActivationRecord


def compute_gae(
    all_trajectories: list[list[TacticalActivationRecord]],
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    unit_local_blend: float = 0.0,
) -> tuple[list[list[float]], list[list[float]]]:
    """Compute GAE advantages and returns for all episodes.

    Uses old_value from collection time (stored in TacticalActivationRecord).

    If unit_local_blend > 0, computes a second per-unit-chain GAE where each
    unit's activations form a sub-trajectory (bootstrapping from the unit's
    next activation rather than the next global step), then blends:
        A = (1 - unit_local_blend) * A_global + unit_local_blend * A_unit_local

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

        # Unit-local GAE: group by unit_idx and run GAE along each chain
        if unit_local_blend > 0.0:
            unit_chains: dict[int, list[int]] = defaultdict(list)
            for t, step in enumerate(trajectory):
                unit_chains[step.unit_idx].append(t)

            unit_local_adv = [0.0] * T
            for chain in unit_chains.values():
                chain_len = len(chain)
                last_gae_u = 0.0
                for i in reversed(range(chain_len)):
                    t = chain[i]
                    if i < chain_len - 1:
                        next_value = trajectory[chain[i + 1]].old_value
                    else:
                        # Bootstrap from global state after unit's last action
                        # (not 0.0) so sacrificial plays aren't penalised for dying
                        last_t = chain[i]
                        next_value = trajectory[last_t + 1].old_value if last_t < T - 1 else 0.0
                    delta = (trajectory[t].reward
                             + gamma * next_value
                             - trajectory[t].old_value)
                    last_gae_u = delta + gamma * gae_lambda * last_gae_u
                    unit_local_adv[t] = last_gae_u

            blend = unit_local_blend
            for t in range(T):
                advantages[t] = (1.0 - blend) * advantages[t] + blend * unit_local_adv[t]

        all_advantages.append(advantages)
        all_returns.append(returns)

    return all_advantages, all_returns
