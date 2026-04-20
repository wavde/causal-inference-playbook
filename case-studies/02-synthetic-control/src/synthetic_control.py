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


def placebo_pvalue(
    placebo_results: dict[str, SyntheticControlResult],
    treated_name: str,
    rmspe_cutoff_multiplier: float = 20.0,
) -> float:
    """
    Two-sided empirical p-value based on the RMSPE ratio (Abadie 2010, §4).

    Standard practice (Abadie 2015) is to filter donors with very poor
    pre-period fit. ``rmspe_cutoff_multiplier`` controls how lenient that
    filter is; donors whose pre-RMSPE exceeds ``cutoff_multiplier`` times the
    treated unit's pre-RMSPE are dropped. The default (20x) is permissive and
    preserves legacy behavior; Abadie (2010) uses the equivalent of ~2.24x
    (a 5x MSPE threshold).
    """
    treated = placebo_results[treated_name]
    cutoff = treated.rmspe_pre * rmspe_cutoff_multiplier
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


@dataclass
class ATTBootstrapCI:
    """Block-bootstrap CI for the post-period synthetic-control ATT."""

    att: float
    ci_low: float
    ci_high: float
    se: float
    p_value: float
    alpha: float
    block_length: int
    n_bootstrap: int

    def __repr__(self) -> str:
        return (
            f"ATTBootstrapCI(ATT={self.att:+.3f}, "
            f"{int((1 - self.alpha) * 100)}% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}], "
            f"SE={self.se:.3f}, p={self.p_value:.4f}, "
            f"block_len={self.block_length}, B={self.n_bootstrap})"
        )


def block_bootstrap_att_ci(
    result: SyntheticControlResult,
    block_length: int = 3,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> ATTBootstrapCI:
    """
    Moving-block bootstrap confidence interval for the synthetic-control ATT.

    Idea (Politis-Romano 1994 moving-block bootstrap adapted to SC):
    the pre-period gap ``y_treated - y_synthetic`` is a trajectory of fitting
    residuals. If the donor-weight fit is honest, those residuals
    characterise the remaining noise in the counterfactual. Under the null
    of no treatment effect, post-treatment residuals would be draws from the
    same process. We mimic that by resampling overlapping blocks of
    pre-period residuals to construct a null distribution for the post-period
    average gap, then invert to get a CI and a two-sided p-value for the
    observed ATT.

    Parameters
    ----------
    result : fitted SyntheticControlResult (from fit_synthetic_control).
    block_length : length of the moving blocks (3-6 is typical; use longer
        blocks when pre-period residuals are strongly autocorrelated).
    n_bootstrap : number of bootstrap replicates.
    alpha : confidence level (default 5%).
    seed : RNG seed.

    Returns
    -------
    ATTBootstrapCI with point estimate, CI, bootstrap SE, and the p-value
    from the null distribution.

    Caveats
    -------
    This assumes the pre-period residuals are (approximately) stationary
    and that the donor weights are stable across bootstrap replicates.
    For more conservative inference, prefer Abadie's placebo-permutation
    test (``in_space_placebo`` + ``placebo_pvalue``) or Chernozhukov et al.
    (2021) conformal inference for SC.
    """
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    pre_gap = np.asarray(result.gap[: result.pre_periods], dtype=float)
    n_pre = len(pre_gap)
    if block_length > n_pre:
        raise ValueError("block_length exceeds number of pre-periods")
    t_post = int(result.post_periods)
    if t_post == 0:
        raise ValueError("result has zero post-periods; nothing to bootstrap")

    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(t_post / block_length))
    max_start = n_pre - block_length + 1

    null_atts = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        starts = rng.integers(0, max_start, size=n_blocks_needed)
        blocks = [pre_gap[s : s + block_length] for s in starts]
        samples = np.concatenate(blocks)[:t_post]
        null_atts[b] = float(samples.mean())

    se = float(null_atts.std(ddof=1))
    # Two-sided CI: ATT +/- quantile(|null|, 1 - alpha). The null dist is
    # centered near 0 so |.| quantile gives a symmetric two-sided half-width.
    half_width = float(np.quantile(np.abs(null_atts), 1 - alpha))
    p_value = float((np.abs(null_atts) >= abs(result.att)).mean())

    return ATTBootstrapCI(
        att=float(result.att),
        ci_low=float(result.att - half_width),
        ci_high=float(result.att + half_width),
        se=se,
        p_value=p_value,
        alpha=alpha,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
    )
