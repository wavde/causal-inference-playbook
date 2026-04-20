"""
Simulate a streaming A/B test with continuous arrival of users.

Each user gets an outcome Y ~ N(mu + tau*T, sigma^2) where T in {0,1} is
the random assignment.
"""

from __future__ import annotations

import numpy as np


def simulate_stream(
    n_per_arm: int = 5000,
    true_effect: float = 0.0,
    sigma: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_treated, y_control) of length n_per_arm each."""
    rng = np.random.default_rng(seed)
    y_c = rng.normal(0.0, sigma, size=n_per_arm)
    y_t = rng.normal(true_effect, sigma, size=n_per_arm)
    return y_t, y_c
