"""
CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

Reference: Deng, Xu, Kohavi, Walker (2013),
"Improving the Sensitivity of Online Controlled Experiments by Utilizing
Pre-Experiment Data."

Core idea:
    Y_cuped = Y - theta * (X - mean(X))
    where X is a pre-experiment covariate correlated with Y,
    and theta = cov(Y, X) / var(X).

Under A/B randomization, X is independent of the treatment assignment, so
subtracting theta*(X - mean(X)) does not bias the treatment effect estimate
but reduces variance by a factor of (1 - rho^2), where rho = corr(Y, X).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ExperimentResult:
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    n_treatment: int
    n_control: int
    method: str

    def __repr__(self) -> str:
        return (
            f"{self.method}: ATE={self.estimate:+.4f} "
            f"(95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], "
            f"SE={self.std_error:.4f}, p={self.p_value:.4f}, "
            f"n_t={self.n_treatment}, n_c={self.n_control})"
        )


def compute_theta(y: np.ndarray, x: np.ndarray) -> float:
    """Optimal CUPED coefficient: cov(Y,X)/var(X)."""
    var_x = np.var(x, ddof=1)
    if var_x == 0:
        return 0.0
    return float(np.cov(y, x, ddof=1)[0, 1] / var_x)


def apply_cuped(y: np.ndarray, x: np.ndarray, theta: float | None = None) -> np.ndarray:
    """Return the CUPED-adjusted outcome Y - theta * (X - mean(X))."""
    if theta is None:
        theta = compute_theta(y, x)
    return y - theta * (x - np.mean(x))


def _welch_ttest(y_t: np.ndarray, y_c: np.ndarray, method: str) -> ExperimentResult:
    mean_t, mean_c = np.mean(y_t), np.mean(y_c)
    var_t = np.var(y_t, ddof=1) / len(y_t)
    var_c = np.var(y_c, ddof=1) / len(y_c)
    se = float(np.sqrt(var_t + var_c))
    estimate = float(mean_t - mean_c)
    # Welch-Satterthwaite df
    df_num = (var_t + var_c) ** 2
    df_den = var_t**2 / (len(y_t) - 1) + var_c**2 / (len(y_c) - 1)
    df = df_num / df_den if df_den > 0 else len(y_t) + len(y_c) - 2
    t_crit = stats.t.ppf(0.975, df)
    p_value = 2 * (1 - stats.t.cdf(abs(estimate / se), df)) if se > 0 else 1.0
    return ExperimentResult(
        estimate=estimate,
        std_error=se,
        ci_low=estimate - t_crit * se,
        ci_high=estimate + t_crit * se,
        p_value=float(p_value),
        n_treatment=len(y_t),
        n_control=len(y_c),
        method=method,
    )


def naive_ab_test(df: pd.DataFrame, outcome: str, treatment: str) -> ExperimentResult:
    """Standard difference-in-means with Welch's t-test."""
    y_t = df.loc[df[treatment] == 1, outcome].to_numpy()
    y_c = df.loc[df[treatment] == 0, outcome].to_numpy()
    return _welch_ttest(y_t, y_c, method="naive_welch")


def cuped_ab_test(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariate: str,
) -> ExperimentResult:
    """
    A/B test with CUPED variance reduction.

    Theta is estimated on the pooled sample (valid under randomization).
    """
    y = df[outcome].to_numpy()
    x = df[covariate].to_numpy()
    theta = compute_theta(y, x)
    y_adj = apply_cuped(y, x, theta)
    assigned = df[treatment].to_numpy()
    return _welch_ttest(
        y_adj[assigned == 1],
        y_adj[assigned == 0],
        method=f"cuped(theta={theta:.3f})",
    )


def variance_reduction(
    df: pd.DataFrame, outcome: str, covariate: str
) -> float:
    """Expected variance reduction factor = 1 - corr(Y, X)^2."""
    rho = float(np.corrcoef(df[outcome], df[covariate])[0, 1])
    return 1 - rho**2
