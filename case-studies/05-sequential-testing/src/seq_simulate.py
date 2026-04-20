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


def simulate_stream_with_covariate(
    n_per_arm: int = 5000,
    true_effect: float = 0.0,
    sigma: float = 1.0,
    correlation: float = 0.7,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Streaming A/B with a pre-experiment covariate X correlated with Y.

    Each user has pre-period covariate X_i ~ N(0, 1) and outcome
        Y_i = tau * T_i + rho * X_i + sqrt(1 - rho^2) * eps_i
    scaled by ``sigma``. Correlation between X and Y(0) is ``rho``.

    Returns (y_t, y_c, x_t, x_c): treated/control outcomes and their
    pre-experiment covariate values, suitable for ``msprt_cuped``.
    """
    rng = np.random.default_rng(seed)
    x_t = rng.normal(0.0, 1.0, size=n_per_arm)
    x_c = rng.normal(0.0, 1.0, size=n_per_arm)
    eps_t = rng.normal(0.0, 1.0, size=n_per_arm)
    eps_c = rng.normal(0.0, 1.0, size=n_per_arm)
    resid_scale = np.sqrt(max(0.0, 1.0 - correlation**2))
    y_c = sigma * (correlation * x_c + resid_scale * eps_c)
    y_t = true_effect + sigma * (correlation * x_t + resid_scale * eps_t)
    return y_t, y_c, x_t, x_c
