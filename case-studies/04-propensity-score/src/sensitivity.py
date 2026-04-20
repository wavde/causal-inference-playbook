"""
Sensitivity analysis for unmeasured confounding.

Observational causal estimates (PSM, IPW, AIPW) assume
*no unmeasured confounders* (conditional unconfoundedness). That assumption
is untestable in the data. The right response is to quantify **how strong**
an unmeasured confounder would have to be to overturn the finding.

Two complementary tools are implemented here:

1. ``e_value`` — VanderWeele & Ding (2017) E-value. The minimum strength of
   association (on the risk-ratio scale) that an unmeasured confounder would
   need with both treatment and outcome to explain away the observed effect.
   For a continuous outcome we use the Chinn / VanderWeele approximation
   via Cohen's d (``d = ATT / sd_outcome``).

2. ``rosenbaum_wilcoxon_bounds`` and ``rosenbaum_gamma_threshold`` —
   Rosenbaum (2002) bounds for 1-to-1 matched pairs. We sweep the
   sensitivity parameter Gamma (the odds by which a hidden confounder could
   differentially assign treatment within a matched pair) and report the
   worst-case Wilcoxon signed-rank p-value, plus the Gamma at which
   p crosses the specified alpha.

These two answer complementary questions:

  * E-value is *scale-free* and summary-level: "how strong would U have to be?"
  * Rosenbaum bounds use the actual matched-pair data and ask: "at what
    Gamma does my statistical conclusion fail?"

References
----------
- VanderWeele & Ding (2017). "Sensitivity Analysis in Observational
  Research: Introducing the E-value." *Annals of Internal Medicine*.
- Chinn (2000). "A simple method for converting an odds ratio to effect
  size for use in meta-analysis." *Statistics in Medicine*.
- Rosenbaum (2002). *Observational Studies*, 2nd ed., ch. 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class EValueResult:
    """E-value summary for a point estimate and (optionally) its CI bound."""

    rr: float
    e_value_point: float
    e_value_ci: float | None
    scale: str

    def __repr__(self) -> str:
        parts = [f"EValue(RR={self.rr:.3f}, point={self.e_value_point:.2f}"]
        if self.e_value_ci is not None:
            parts.append(f", CI={self.e_value_ci:.2f}")
        parts.append(f", scale={self.scale})")
        return "".join(parts)


def _e_from_rr(rr: float) -> float:
    """E-value formula on the risk-ratio scale (VanderWeele-Ding 2017)."""
    if rr < 1:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def e_value(
    estimate: float,
    sd_outcome: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    outcome_type: str = "continuous",
) -> EValueResult:
    """
    Compute the E-value for an observational causal estimate.

    Parameters
    ----------
    estimate : ATT / ATE point estimate. Units depend on ``outcome_type``.
    sd_outcome : required when ``outcome_type='continuous'``. The SD of the
        outcome among the control group (preferred) or overall.
    ci_low, ci_high : CI bounds on the same scale as ``estimate``. If
        provided, we also compute the E-value for the CI bound closest to
        the null, per VanderWeele-Ding's recommended practice.
    outcome_type : 'continuous' (applies Chinn approximation, converting
        standardized mean difference d = estimate / sd_outcome into a
        risk-ratio via RR = exp(0.91 * d)) or 'risk_ratio' (treat
        ``estimate`` as an RR directly).

    Returns
    -------
    EValueResult with the implied RR, point E-value, and CI E-value.

    An E-value of 1.5 means "an unmeasured confounder would need associations
    of 1.5-fold with both T and Y to explain the finding." Typical rule of
    thumb: an E-value below 2 for a small study is weak; above 3 is strong.
    """
    if outcome_type == "continuous":
        if sd_outcome is None or sd_outcome <= 0:
            raise ValueError("sd_outcome required for continuous outcomes")
        d = estimate / sd_outcome
        rr = float(np.exp(0.91 * d))
        scale = "continuous (Chinn RR approx)"
        ci_rr = None
        if ci_low is not None and ci_high is not None:
            # Take the CI bound closest to the null (d=0 -> rr=1).
            if estimate > 0:
                d_ci = ci_low / sd_outcome
            else:
                d_ci = ci_high / sd_outcome
            ci_rr = float(np.exp(0.91 * d_ci))
    elif outcome_type == "risk_ratio":
        rr = float(estimate)
        scale = "risk_ratio"
        ci_rr = None
        if ci_low is not None and ci_high is not None:
            ci_rr = float(ci_low if rr > 1 else ci_high)
    else:
        raise ValueError(f"outcome_type must be 'continuous' or 'risk_ratio', got {outcome_type!r}")

    e_point = _e_from_rr(rr)
    e_ci: float | None = None
    if ci_rr is not None:
        # If the CI crosses the null (on RR scale: includes 1), E-value is 1.
        if (rr > 1 and ci_rr <= 1) or (rr < 1 and ci_rr >= 1):
            e_ci = 1.0
        else:
            e_ci = _e_from_rr(ci_rr)

    return EValueResult(rr=rr, e_value_point=e_point, e_value_ci=e_ci, scale=scale)


def _wilcoxon_signed_rank_one_sided_p(
    diffs: np.ndarray, gamma: float, alternative: str = "greater"
) -> float:
    """
    Worst-case (upper) one-sided p-value for the Wilcoxon signed-rank
    statistic under Rosenbaum (2002, ch. 4) sensitivity model with
    parameter Gamma >= 1.

    Under Gamma, the probability that a matched pair has a positive-sign
    contribution is bounded by p_plus = gamma / (1 + gamma). The signed-rank
    sum S has mean n_p * sum_ranks / n and variance
    sum r_i^2 * p_plus * (1 - p_plus) under this bound, from which we form
    a normal-approximation p-value.

    `alternative='greater'` returns the p-value for H1: treatment raised Y.
    `alternative='less'` returns the p-value for H1: treatment lowered Y.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0.0]
    if len(d) == 0:
        return 1.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0

    if alternative == "greater":
        S_obs = float(ranks[d > 0].sum())
        p_plus = gamma / (1.0 + gamma)
    elif alternative == "less":
        S_obs = float(ranks[d < 0].sum())
        p_plus = gamma / (1.0 + gamma)
    else:
        raise ValueError("alternative must be 'greater' or 'less'")

    mean = p_plus * ranks.sum()
    var = p_plus * (1.0 - p_plus) * (ranks ** 2).sum()
    if var <= 0:
        return 1.0 if S_obs <= mean else 0.0
    z = (S_obs - mean) / np.sqrt(var)
    return float(1.0 - norm.cdf(z))


def rosenbaum_wilcoxon_bounds(
    pair_differences: np.ndarray,
    gammas: np.ndarray | list[float] | None = None,
    alternative: str = "greater",
) -> pd.DataFrame:
    """
    Sweep Rosenbaum's sensitivity parameter Gamma and return the worst-case
    one-sided Wilcoxon signed-rank p-value at each Gamma.

    Parameters
    ----------
    pair_differences : array of matched-pair outcome differences
        (Y_treated - Y_control).
    gammas : grid of Gamma values to evaluate (all >= 1). Default: 1 to 3
        in 0.25 steps.
    alternative : 'greater' (H1: treatment increases Y) or 'less'.

    Returns
    -------
    DataFrame with columns [gamma, p_upper_bound].
    """
    if gammas is None:
        gammas = np.round(np.arange(1.0, 3.01, 0.25), 2)
    rows = []
    for g in gammas:
        p = _wilcoxon_signed_rank_one_sided_p(
            np.asarray(pair_differences), float(g), alternative=alternative
        )
        rows.append({"gamma": float(g), "p_upper_bound": p})
    return pd.DataFrame(rows)


def rosenbaum_gamma_threshold(
    pair_differences: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "greater",
    gamma_max: float = 10.0,
    resolution: float = 0.01,
) -> float:
    """
    Binary search for the smallest Gamma at which the worst-case Wilcoxon
    signed-rank p-value crosses ``alpha``. This is the "sensitivity
    threshold" commonly reported in observational studies:

        "A hidden confounder would need to create ~Gamma_crit-fold greater
         odds of treatment within matched pairs to overturn the finding at
         alpha=5%."

    Returns ``gamma_max`` if p <= alpha even at gamma_max (very robust
    finding), or 1.0 if the finding is already non-significant at Gamma=1.
    """
    p_at_one = _wilcoxon_signed_rank_one_sided_p(
        np.asarray(pair_differences), 1.0, alternative=alternative
    )
    if p_at_one > alpha:
        return 1.0

    p_at_max = _wilcoxon_signed_rank_one_sided_p(
        np.asarray(pair_differences), gamma_max, alternative=alternative
    )
    if p_at_max <= alpha:
        return float(gamma_max)

    lo, hi = 1.0, gamma_max
    while hi - lo > resolution:
        mid = 0.5 * (lo + hi)
        p_mid = _wilcoxon_signed_rank_one_sided_p(
            np.asarray(pair_differences), mid, alternative=alternative
        )
        if p_mid <= alpha:
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


def matched_pair_differences(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    caliper_sd: float | None = 0.2,
) -> np.ndarray:
    """
    Convenience wrapper: run 1-NN propensity matching and return the array
    of matched-pair outcome differences (Y_treated - Y_matched_control)
    suitable for Rosenbaum-bound analysis.
    """
    # Local import to avoid a mandatory top-level coupling.
    from psm import _match_indices, estimate_propensity

    propensity = estimate_propensity(X, T)
    logit_e = np.log(propensity / (1 - propensity))
    caliper = caliper_sd * np.std(logit_e) if caliper_sd is not None else None
    t_idx, c_idx, _ = _match_indices(propensity, T, caliper)
    return Y[t_idx] - Y[c_idx]
