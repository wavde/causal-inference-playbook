"""
Sequential / always-valid testing for online experiments.

Three estimators compared:

1. **Naive fixed-horizon z-test.** Standard A/B inference; valid only if you
   look exactly once, at the pre-specified end. Peeking inflates Type-I error.

2. **mSPRT (mixture Sequential Probability Ratio Test).** Robbins (1970);
   productionized by Optimizely / Johari et al. (2017). Uses a Gaussian prior
   over effect sizes to construct a likelihood ratio that is a martingale
   under H0. Stopping rule: reject when log-likelihood-ratio >= log(1/alpha).
   Always-valid: Type-I error stays <= alpha at *any* stopping time.

3. **Pocock alpha-spending.** Group-sequential design with K planned looks
   and a constant per-look critical value chosen so total Type-I error = alpha.
   Useful when looks are pre-planned (clinical-trial style).

The mSPRT formula here matches the one in
    github.com/wavde/experiment-toolkit/src/experiment_toolkit/sequential.py
which fixed an off-by-2 bug in an earlier version.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SequentialResult:
    rejected: bool
    stopped_at: int           # sample size per arm at stop time
    final_estimate: float
    method: str

    def __repr__(self) -> str:
        return (
            f"{self.method}(rejected={self.rejected}, n_per_arm_at_stop={self.stopped_at}, "
            f"estimate={self.final_estimate:+.4f})"
        )


def fixed_horizon_zscore(
    y_t: np.ndarray, y_c: np.ndarray
) -> tuple[float, float, float]:
    """Standard two-sample z (returns z, p, delta_hat). Valid only at horizon end."""
    delta = float(y_t.mean() - y_c.mean())
    se = float(np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c)))
    z = delta / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p, delta


def msprt_log_likelihood_ratio(
    delta_hat: float,
    n_per_arm: int,
    sigma: float,
    tau: float,
) -> float:
    """
    Log-likelihood ratio for mSPRT with Gaussian mixing prior N(0, tau^2)
    over the true effect, vs. H0: effect = 0.

    For two-sample with equal n per arm, delta_hat ~ N(delta, V) with
        V = 2 * sigma^2 / n_per_arm

    log_lambda = 0.5 * log(V / (V + tau^2)) + (delta_hat^2 / (2V)) * tau^2/(V+tau^2)
    """
    V = 2 * sigma**2 / n_per_arm
    shrinkage = tau**2 / (V + tau**2)
    return 0.5 * np.log(V / (V + tau**2)) + (delta_hat**2 / (2 * V)) * shrinkage


def msprt_sequential_test(
    y_t: np.ndarray,
    y_c: np.ndarray,
    alpha: float = 0.05,
    tau: float | None = None,
    sigma: float | None = None,
    look_every: int = 50,
    min_n: int = 100,
) -> SequentialResult:
    """
    Walk through the data looking every `look_every` users per arm.
    Reject (and stop) the first time log-LR >= log(1/alpha).

    `tau`: prior SD for the effect. If None, defaults to sigma_hat (a sensible
    "we expect effects on the order of 1 sigma" prior).

    `sigma`: known/assumed per-observation SD. If None, estimated from the data.
    Using a *fixed* sigma is required for the mSPRT to be a true martingale
    under H0 -- estimating sigma online introduces bias. We follow the standard
    practical approach of estimating sigma at the first look and then freezing it.
    """
    n_max = min(len(y_t), len(y_c))
    threshold = np.log(1 / alpha)

    # Initialize sigma from the first look (then freeze).
    init_n = max(min_n, look_every)
    if sigma is None:
        pooled = np.concatenate([y_t[:init_n], y_c[:init_n]])
        sigma = float(pooled.std(ddof=1))
    if tau is None:
        tau = sigma

    n = init_n
    while n <= n_max:
        delta_hat = float(y_t[:n].mean() - y_c[:n].mean())
        log_lr = msprt_log_likelihood_ratio(delta_hat, n, sigma, tau)
        if log_lr >= threshold:
            return SequentialResult(
                rejected=True, stopped_at=n,
                final_estimate=delta_hat, method="mSPRT",
            )
        n += look_every

    # Never crossed threshold -> fail to reject.
    delta_hat = float(y_t[:n_max].mean() - y_c[:n_max].mean())
    return SequentialResult(
        rejected=False, stopped_at=n_max,
        final_estimate=delta_hat, method="mSPRT",
    )


def pocock_critical_value(alpha: float, K: int) -> float:
    """
    Per-look two-sided Z critical value for Pocock's design with K equally-
    spaced looks and overall Type-I error alpha. Numerically inverted via
    bisection on the joint Normal multiple-testing distribution.
    """
    # Approximate the per-look |Z| critical value c such that
    # P(max_{k=1..K} |Z_k| > c) = alpha under H0,
    # where (Z_1, ..., Z_K) are jointly Normal with corr(Z_i, Z_j) = sqrt(i/j).
    # Use the closed-form approximation from Pocock (1977) tables via simulation.
    from numpy.random import default_rng
    rng = default_rng(0)
    n_sim = 200_000
    incr = rng.standard_normal((n_sim, K))
    # Cumulative test statistic at each look (running mean scaled by sqrt(k))
    cumsum = incr.cumsum(axis=1)
    Z = cumsum / np.sqrt(np.arange(1, K + 1))
    max_abs = np.max(np.abs(Z), axis=1)
    return float(np.quantile(max_abs, 1 - alpha))


def pocock_sequential_test(
    y_t: np.ndarray,
    y_c: np.ndarray,
    alpha: float = 0.05,
    K: int = 5,
    sigma: float | None = None,
) -> SequentialResult:
    """
    Pocock group-sequential test with K equally-spaced looks.
    """
    n_max = min(len(y_t), len(y_c))
    look_sizes = np.linspace(n_max / K, n_max, K).astype(int)
    crit = pocock_critical_value(alpha, K)

    if sigma is None:
        pooled = np.concatenate([y_t[: look_sizes[0]], y_c[: look_sizes[0]]])
        sigma = float(pooled.std(ddof=1))

    for n in look_sizes:
        delta_hat = float(y_t[:n].mean() - y_c[:n].mean())
        se = sigma * np.sqrt(2 / n)
        z = delta_hat / se if se > 0 else 0.0
        if abs(z) >= crit:
            return SequentialResult(
                rejected=True, stopped_at=int(n),
                final_estimate=delta_hat, method=f"Pocock(K={K})",
            )

    delta_hat = float(y_t[:n_max].mean() - y_c[:n_max].mean())
    return SequentialResult(
        rejected=False, stopped_at=n_max,
        final_estimate=delta_hat, method=f"Pocock(K={K})",
    )


def naive_peeking_test(
    y_t: np.ndarray,
    y_c: np.ndarray,
    alpha: float = 0.05,
    look_every: int = 50,
    min_n: int = 100,
) -> SequentialResult:
    """
    Demonstration estimator: applies the *fixed-horizon* z-test repeatedly.
    Rejects the first time p < alpha. This is the WRONG thing to do; included
    here so we can quantify the Type-I error inflation it causes.
    """
    n_max = min(len(y_t), len(y_c))
    n = max(min_n, look_every)
    while n <= n_max:
        _, p, delta_hat = fixed_horizon_zscore(y_t[:n], y_c[:n])
        if p < alpha:
            return SequentialResult(
                rejected=True, stopped_at=n,
                final_estimate=delta_hat, method="naive_peeking",
            )
        n += look_every
    delta_hat = float(y_t[:n_max].mean() - y_c[:n_max].mean())
    return SequentialResult(
        rejected=False, stopped_at=n_max,
        final_estimate=delta_hat, method="naive_peeking",
    )
