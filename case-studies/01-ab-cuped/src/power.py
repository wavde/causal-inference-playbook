"""
Power and sample-size helpers for A/B tests with and without CUPED.

For a two-sample test of means with significance ``alpha`` (two-sided) and
desired power ``power``, the required per-arm sample size at effect size
``mde`` is approximately

    n = 2 * (z_{1-alpha/2} + z_{power})^2 * sigma^2 / mde^2.

CUPED scales variance by (1 - rho^2), so the required sample size scales
by the same factor. This module wraps that arithmetic with helpers for
the three questions analysts are asked every week:

1. "How much can we speed up this experiment with CUPED?" -> n_cuped / n_naive
2. "At n users per arm and correlation rho, what's our MDE?" -> mde_vs_rho
3. "Plot how MDE shrinks as rho grows." -> plot_mde_vs_rho

Everything here is a thin wrapper over the standard two-sample normal
approximation — we deliberately keep it simple so it can be used for
back-of-the-envelope planning conversations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class PowerCalc:
    n_per_arm: int
    mde: float
    sigma: float
    rho: float
    alpha: float
    power: float
    uses_cuped: bool

    def __repr__(self) -> str:
        tag = f" (CUPED, rho={self.rho:.2f})" if self.uses_cuped else ""
        return (
            f"PowerCalc{tag}: n/arm={self.n_per_arm:,}, MDE={self.mde:.4f}, "
            f"sigma={self.sigma:.3f}, alpha={self.alpha}, power={self.power}"
        )


def required_n_per_arm(
    mde: float,
    sigma: float,
    rho: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Per-arm sample size for a two-sided two-sample test of means.

    With ``rho > 0`` the formula uses CUPED-adjusted variance
    ``sigma^2 * (1 - rho^2)``.

    Parameters
    ----------
    mde : minimum detectable effect on the outcome scale (must be > 0).
    sigma : per-observation standard deviation of the raw outcome.
    rho : correlation between outcome and pre-experiment covariate. 0 = no CUPED.
    alpha : two-sided Type-I error.
    power : 1 - Type-II error.
    """
    if mde <= 0 or sigma <= 0:
        raise ValueError("mde and sigma must be positive")
    if not -1 <= rho <= 1:
        raise ValueError("rho must be in [-1, 1]")
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    variance = sigma ** 2 * (1 - rho ** 2)
    n = 2.0 * (z_alpha + z_power) ** 2 * variance / (mde ** 2)
    return int(np.ceil(n))


def mde_at_n(
    n_per_arm: int,
    sigma: float,
    rho: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Minimum detectable effect at a fixed per-arm sample size."""
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    variance = sigma ** 2 * (1 - rho ** 2)
    return float((z_alpha + z_power) * np.sqrt(2.0 * variance / n_per_arm))


def cuped_power_summary(
    mde: float,
    sigma: float,
    rho: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[str, object]:
    """
    Side-by-side summary of sample-size savings from CUPED.

    Returns a dict with per-arm sample sizes under naive vs CUPED,
    the absolute saving, and the percent reduction.
    """
    n_naive = required_n_per_arm(mde, sigma, rho=0.0, alpha=alpha, power=power)
    n_cuped = required_n_per_arm(mde, sigma, rho=rho, alpha=alpha, power=power)
    return {
        "n_naive_per_arm": n_naive,
        "n_cuped_per_arm": n_cuped,
        "absolute_saving": n_naive - n_cuped,
        "percent_reduction": float(1 - n_cuped / n_naive) if n_naive > 0 else 0.0,
        "variance_reduction_factor": 1 - rho ** 2,
        "rho": rho,
        "mde": mde,
        "sigma": sigma,
    }


def mde_vs_rho(
    n_per_arm: int,
    sigma: float,
    rhos: np.ndarray | list[float] | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
) -> np.ndarray:
    """
    Vector of MDEs across a grid of correlations.

    Returns a 2-column array ``[[rho, mde], ...]`` suitable for plotting
    or turning into a pandas DataFrame.
    """
    if rhos is None:
        rhos = np.linspace(0, 0.95, 20)
    rhos = np.asarray(rhos, dtype=float)
    out = np.empty((len(rhos), 2))
    for i, rho in enumerate(rhos):
        out[i, 0] = rho
        out[i, 1] = mde_at_n(n_per_arm, sigma, rho=float(rho), alpha=alpha, power=power)
    return out


def plot_mde_vs_rho(
    n_per_arm: int,
    sigma: float,
    rhos: np.ndarray | list[float] | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    ax=None,
):
    """
    Quick matplotlib helper showing MDE shrinking with rho. Caller owns the
    figure (we don't call plt.show()). Optional ``ax`` lets you compose
    into a larger figure.

    Returns the Axes object.
    """
    # Lazy import so this module is importable in environments without matplotlib.
    import matplotlib.pyplot as plt

    data = mde_vs_rho(n_per_arm, sigma, rhos=rhos, alpha=alpha, power=power)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data[:, 0], data[:, 1], lw=2.2, color="#1f77b4")
    ax.scatter(data[:, 0], data[:, 1], s=18, color="#1f77b4")
    ax.set_xlabel("Corr(Y, pre-experiment covariate)  rho")
    ax.set_ylabel(f"MDE at n={n_per_arm:,}/arm, alpha={alpha}, power={power}")
    ax.set_title("CUPED: MDE shrinks with covariate correlation")
    ax.grid(True, alpha=0.3)
    return ax
