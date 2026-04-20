"""Simulate a marketplace with SUTVA violation (spillover within time blocks).

Model (per user i in time block b):
    Y_i = alpha + tau * T_i - gamma * mean(T_j for j != i in block b) + noise

- `tau` is the direct effect of treating a user.
- `gamma` is the spillover: treating others consumes a shared resource, lowering
  conversion for everyone else in the same block (think: driver pool, ad
  inventory, limited-supply promotions).

When spillover is present (gamma > 0), user-level A/B tests overstate the true
policy effect, because the control group experiences spillover from the 50% of
treated users in the same block. Switchback (block-level randomization) breaks
the interference by assigning entire blocks to one arm.

True policy effect (counterfactual contrast of "treat all" vs "treat none"):
    Delta = tau - gamma
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Market:
    user_id: np.ndarray
    block_id: np.ndarray
    treatment: np.ndarray
    y: np.ndarray


def simulate_market(
    n_blocks: int = 200,
    users_per_block: int = 50,
    tau: float = 0.30,
    gamma: float = 0.20,
    alpha: float = 1.0,
    noise: float = 0.5,
    design: str = "user",
    seed: int = 7,
) -> Market:
    """Simulate a marketplace under a given randomization design.

    Parameters
    ----------
    design
        ``"user"`` — each user independently assigned T=1 w.p. 0.5 (standard A/B).
        ``"switchback"`` — each *block* assigned T=1 w.p. 0.5; all users in a
        block share the same treatment.
    """
    rng = np.random.default_rng(seed)

    if design == "user":
        treatment = rng.binomial(1, 0.5, size=n_blocks * users_per_block).astype(float)
    elif design == "switchback":
        block_assignment = rng.binomial(1, 0.5, size=n_blocks).astype(float)
        treatment = np.repeat(block_assignment, users_per_block)
    else:
        raise ValueError(f"unknown design: {design!r}")

    block_id = np.repeat(np.arange(n_blocks), users_per_block)
    user_id = np.arange(len(block_id))

    sum_by_block = np.bincount(block_id, weights=treatment, minlength=n_blocks)
    block_sum_per_row = sum_by_block[block_id]
    others_mean = (block_sum_per_row - treatment) / max(users_per_block - 1, 1)

    y = alpha + tau * treatment - gamma * others_mean + rng.normal(0, noise, size=len(treatment))
    return Market(user_id=user_id, block_id=block_id, treatment=treatment, y=y)


def true_policy_effect(tau: float, gamma: float) -> float:
    """Counterfactual contrast of treat-all vs treat-none markets."""
    return tau - gamma
