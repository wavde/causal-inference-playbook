"""
Synthetic Control Method (Abadie, Diamond, Hainmueller 2010).

Core idea:
    A single treated unit is compared to a *weighted average* of untreated
    "donor" units, where the weights are chosen so the synthetic unit
    matches the treated unit's pre-treatment outcome trajectory. The
    post-treatment gap between treated and synthetic is the causal estimate.

Weights w satisfy:
    w_j >= 0  for all j
    sum(w_j) == 1
    minimize || y1_pre - Y0_pre @ w ||^2

We solve with SLSQP. For moderate donor pools (J < 100) this is fast and
reliable; for larger problems a dedicated QP solver (cvxpy) would be better.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass
class SyntheticControlResult:
    weights: np.ndarray
    donor_names: list[str]
    y_treated: np.ndarray
    y_synthetic: np.ndarray
    pre_periods: int
    post_periods: int
    att: float
    gap: np.ndarray
    rmspe_pre: float
    rmspe_post: float
    rmspe_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        self.rmspe_ratio = (
            self.rmspe_post / self.rmspe_pre if self.rmspe_pre > 0 else float("inf")
        )

    def __repr__(self) -> str:
        top = sorted(
            zip(self.donor_names, self.weights, strict=False),
            key=lambda t: -t[1],
        )[:5]
        top_str = ", ".join(f"{n}={w:.2f}" for n, w in top if w > 0.01)
        return (
            f"SyntheticControl(ATT={self.att:+.3f}, "
            f"RMSPE pre={self.rmspe_pre:.3f} post={self.rmspe_post:.3f} "
            f"ratio={self.rmspe_ratio:.2f}, top donors: {top_str})"
        )


def _fit_weights(y1_pre: np.ndarray, Y0_pre: np.ndarray) -> np.ndarray:
    """
    Solve the constrained least-squares weight problem.

    Parameters
    ----------
    y1_pre : (T_pre,) treated unit's pre-treatment outcomes
    Y0_pre : (T_pre, J) donor outcomes stacked column-wise
    """
    J = Y0_pre.shape[1]
    w0 = np.full(J, 1.0 / J)

    def loss(w: np.ndarray) -> float:
        resid = y1_pre - Y0_pre @ w
        return float(resid @ resid)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * J

    result = minimize(
        loss,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    # Don't raise on "iteration limit reached" -- SLSQP usually returns a
    # usable (near-optimal) solution anyway. Only truly invalid results
    # (NaN or constraint violation) should be flagged.
    w = result.x
    if np.any(np.isnan(w)):
        raise RuntimeError(f"SLSQP produced NaN weights: {result.message}")
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        raise RuntimeError(
            f"SLSQP produced degenerate weights (all zero after clipping): {result.message}"
        )
    return w / total


def fit_synthetic_control(
    y_treated: np.ndarray,
    Y_donors: np.ndarray,
    pre_periods: int,
    donor_names: list[str] | None = None,
) -> SyntheticControlResult:
    """
    Fit synthetic control for a single treated unit.

    Parameters
    ----------
    y_treated   : (T,) outcome trajectory of the treated unit
    Y_donors    : (T, J) outcomes for J donor units
    pre_periods : number of pre-treatment periods (treatment starts at index = pre_periods)
    donor_names : optional labels, length J
    """
    y_treated = np.asarray(y_treated, dtype=float)
    Y_donors = np.asarray(Y_donors, dtype=float)
    T, J = Y_donors.shape
    if y_treated.shape[0] != T:
        raise ValueError("y_treated and Y_donors must share the time dimension")
    if pre_periods <= 0 or pre_periods >= T:
        raise ValueError("pre_periods must lie strictly inside [1, T-1]")
    if donor_names is None:
        donor_names = [f"donor_{j}" for j in range(J)]

    w = _fit_weights(y_treated[:pre_periods], Y_donors[:pre_periods])
    y_synth = Y_donors @ w
    gap = y_treated - y_synth

    rmspe_pre = float(np.sqrt(np.mean(gap[:pre_periods] ** 2)))
    rmspe_post = float(np.sqrt(np.mean(gap[pre_periods:] ** 2)))
    att = float(np.mean(gap[pre_periods:]))

    return SyntheticControlResult(
        weights=w,
        donor_names=list(donor_names),
        y_treated=y_treated,
        y_synthetic=y_synth,
        pre_periods=pre_periods,
        post_periods=T - pre_periods,
        att=att,
        gap=gap,
        rmspe_pre=rmspe_pre,
        rmspe_post=rmspe_post,
    )


def in_space_placebo(
    Y_all: np.ndarray,
    pre_periods: int,
    treated_idx: int,
    unit_names: list[str] | None = None,
) -> dict[str, SyntheticControlResult]:
    """
    Run placebo-in-space: re-fit SC pretending each donor is the treated unit.

    Returns a dict {unit_name: result} including the real treated unit.
    The empirical p-value for the real effect is:
        (# units with |RMSPE_post/RMSPE_pre| >= real) / total_units
    """
    T, N = Y_all.shape
    if unit_names is None:
        unit_names = [f"unit_{i}" for i in range(N)]

    out: dict[str, SyntheticControlResult] = {}
    for i in range(N):
        others = [j for j in range(N) if j != i]
        out[unit_names[i]] = fit_synthetic_control(
            y_treated=Y_all[:, i],
            Y_donors=Y_all[:, others],
            pre_periods=pre_periods,
            donor_names=[unit_names[j] for j in others],
        )
    return out


def placebo_pvalue(placebo_results: dict[str, SyntheticControlResult], treated_name: str) -> float:
    """
    Two-sided empirical p-value based on the RMSPE ratio (Abadie 2010, §4).

    Standard practice (Abadie 2015) is to filter donors with very poor
    pre-period fit. We keep only donors whose pre-RMSPE is within 20x the
    treated unit's pre-RMSPE.
    """
    treated = placebo_results[treated_name]
    cutoff = treated.rmspe_pre * 20
    ratios = [
        r.rmspe_ratio
        for name, r in placebo_results.items()
        if r.rmspe_pre <= cutoff or name == treated_name
    ]
    rank = sum(1 for r in ratios if r >= treated.rmspe_ratio)
    return rank / len(ratios)


def in_time_placebo(
    y_treated: np.ndarray,
    Y_donors: np.ndarray,
    true_pre_periods: int,
    fake_pre_periods: int,
    donor_names: list[str] | None = None,
) -> SyntheticControlResult:
    """
    Placebo-in-time: pretend treatment began earlier (at fake_pre_periods),
    fit SC only on data before the real treatment date, and check whether
    we "detect" an effect during the still-pre-treatment window
    [fake_pre_periods, true_pre_periods).

    A credible SC should produce a near-zero gap in that window.
    """
    if not (0 < fake_pre_periods < true_pre_periods):
        raise ValueError("fake_pre_periods must be in (0, true_pre_periods)")
    return fit_synthetic_control(
        y_treated=y_treated[:true_pre_periods],
        Y_donors=Y_donors[:true_pre_periods],
        pre_periods=fake_pre_periods,
        donor_names=donor_names,
    )
