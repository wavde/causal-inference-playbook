"""
Difference-in-Differences (DiD) and event-study estimators.

Two specifications are provided:

1. **2x2 DiD** (canonical Card-Krueger style): OLS of outcome on
   unit FE + period FE + (treated x post). Coefficient on the
   interaction is the ATT under parallel trends.

2. **Event-study DiD**: OLS of outcome on unit FE + period FE +
   sum of (treated x relative_time_k) for k != -1. Each k coefficient
   is the dynamic ATT at lag k. Pre-treatment lags should be ~0 if
   parallel trends hold.

Cluster-robust standard errors (clustered by unit) are reported throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class DiDResult:
    att: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    n_obs: int
    n_units: int

    def __repr__(self) -> str:
        return (
            f"DiD(ATT={self.att:+.3f}, SE={self.std_error:.3f}, "
            f"95% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}], "
            f"p={self.p_value:.4f}, n_obs={self.n_obs}, n_units={self.n_units})"
        )


@dataclass
class EventStudyResult:
    coefs: pd.DataFrame   # columns: relative_time, coef, se, ci_low, ci_high, p_value
    parallel_trends_pvalue: float
    n_obs: int
    n_units: int

    def __repr__(self) -> str:
        return (
            f"EventStudy(n_leads_lags={len(self.coefs)}, "
            f"parallel_trends p={self.parallel_trends_pvalue:.4f})"
        )


def _ols_clustered(
    y: np.ndarray,
    X: pd.DataFrame,
    cluster: np.ndarray,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS with cluster-robust standard errors (clustered by `cluster`)."""
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const)
    return model.fit(cov_type="cluster", cov_kwds={"groups": cluster})


def did_2x2(
    df: pd.DataFrame,
    outcome: str = "y",
    unit: str = "unit",
    period: str = "period",
    treated: str = "treated_unit",
    post: str = "post",
) -> DiDResult:
    """
    Canonical two-way-FE DiD. `df` is long-format with one row per (unit, period).

    Required columns:
      - outcome (y)
      - treated_unit: 1 if unit is ever treated, 0 otherwise
      - post: 1 if period is post-treatment, 0 otherwise

    Returns ATT estimate and cluster-robust inference.
    """
    work = df.copy()
    work["did"] = work[treated] * work[post]

    unit_dummies = pd.get_dummies(work[unit], prefix="u", drop_first=True, dtype=float)
    period_dummies = pd.get_dummies(work[period], prefix="t", drop_first=True, dtype=float)
    X = pd.concat([work[["did"]], unit_dummies, period_dummies], axis=1)

    fit = _ols_clustered(work[outcome].to_numpy(), X, work[unit].to_numpy())
    att = float(fit.params["did"])
    se = float(fit.bse["did"])
    p = float(fit.pvalues["did"])
    ci_low, ci_high = (float(x) for x in fit.conf_int().loc["did"])

    return DiDResult(
        att=att,
        std_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p,
        n_obs=len(work),
        n_units=int(work[unit].nunique()),
    )


def event_study(
    df: pd.DataFrame,
    outcome: str = "y",
    unit: str = "unit",
    period: str = "period",
    treated: str = "treated_unit",
    treatment_period: int | None = None,
    omit_relative_time: int = -1,
) -> EventStudyResult:
    """
    Event-study DiD with leads and lags.

    `treatment_period`: the calendar period when treatment turns on for treated
    units (assumed common across treated units; staggered treatment requires
    Sun-Abraham/Callaway-Sant'Anna estimators not implemented here).

    `omit_relative_time`: the relative-time bin to omit as the reference
    (default -1, i.e. the period just before treatment).

    Returns coefficients on (treated x relative_time = k) for all k != omit.
    Pre-treatment leads should be ~0 under parallel trends; their joint
    F-test is reported as `parallel_trends_pvalue`.
    """
    if treatment_period is None:
        treated_mask = (df[treated] == 1)
        if "post" not in df.columns:
            raise ValueError("provide treatment_period or include a 'post' column")
        first_post = df.loc[treated_mask & (df["post"] == 1), period].min()
        treatment_period = int(first_post)

    work = df.copy()
    work["rel_time"] = work[period] - treatment_period

    rel_times = sorted(work["rel_time"].unique())
    rel_times_to_use = [k for k in rel_times if k != omit_relative_time]

    interaction_cols = []
    for k in rel_times_to_use:
        col = f"D_k{k:+d}"
        work[col] = ((work[treated] == 1) & (work["rel_time"] == k)).astype(float)
        interaction_cols.append(col)

    unit_dummies = pd.get_dummies(work[unit], prefix="u", drop_first=True, dtype=float)
    period_dummies = pd.get_dummies(work[period], prefix="t", drop_first=True, dtype=float)
    X = pd.concat([work[interaction_cols], unit_dummies, period_dummies], axis=1)
    fit = _ols_clustered(work[outcome].to_numpy(), X, work[unit].to_numpy())

    rows = []
    for k, col in zip(rel_times_to_use, interaction_cols, strict=True):
        rows.append(
            {
                "relative_time": k,
                "coef": float(fit.params[col]),
                "se": float(fit.bse[col]),
                "ci_low": float(fit.conf_int().loc[col, 0]),
                "ci_high": float(fit.conf_int().loc[col, 1]),
                "p_value": float(fit.pvalues[col]),
            }
        )
    coefs = pd.DataFrame(rows).sort_values("relative_time").reset_index(drop=True)

    # Parallel trends F-test: joint significance of all pre-treatment leads.
    # Use a restriction matrix R (n_pre × n_params) picking out the pre-period
    # interaction columns rather than a string formula, which is fragile with
    # signed column names like ``D_k-5``.
    pre_cols = [
        f"D_k{k:+d}"
        for k in rel_times_to_use
        if k < omit_relative_time
    ]
    if pre_cols:
        param_index = list(fit.params.index)
        R = np.zeros((len(pre_cols), len(param_index)))
        for i, col in enumerate(pre_cols):
            R[i, param_index.index(col)] = 1.0
        pt_test = fit.f_test(R)
        pt_pvalue = float(np.asarray(pt_test.pvalue).item())
    else:
        pt_pvalue = float("nan")

    return EventStudyResult(
        coefs=coefs,
        parallel_trends_pvalue=pt_pvalue,
        n_obs=len(work),
        n_units=int(work[unit].nunique()),
    )
