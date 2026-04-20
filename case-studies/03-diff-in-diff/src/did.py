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


@dataclass
class CSResult:
    """Result of Callaway-Sant'Anna staggered DiD."""

    group_time_att: pd.DataFrame   # columns: cohort, period, att, se, ci_low, ci_high, n_treated, n_control
    event_study: pd.DataFrame      # columns: relative_time, att, se, ci_low, ci_high
    overall_att: float
    overall_se: float
    overall_ci_low: float
    overall_ci_high: float
    n_units: int
    n_bootstrap: int

    def __repr__(self) -> str:
        return (
            f"CS(overall ATT={self.overall_att:+.3f}, "
            f"95% CI [{self.overall_ci_low:+.3f}, {self.overall_ci_high:+.3f}], "
            f"n_units={self.n_units}, boot={self.n_bootstrap})"
        )


def _cs_group_time_point_estimates(
    wide: pd.DataFrame,
    cohorts: list[int],
    periods: list[int],
) -> dict[tuple[int, int], float]:
    """
    Callaway-Sant'Anna ATT(g, t) with never-treated control, outcome-regression
    identifier reducing to a simple 2x2 DiD between baseline g-1 and period t.

    ATT(g, t) = mean[Y_t - Y_{g-1} | cohort=g] - mean[Y_t - Y_{g-1} | cohort=0]

    `wide` is a unit-by-period matrix with a 'cohort' column.
    """
    never = wide[wide["cohort"] == 0]
    out: dict[tuple[int, int], float] = {}
    for g in cohorts:
        if g <= min(periods):
            continue
        baseline = g - 1
        if baseline not in wide.columns:
            continue
        treated_g = wide[wide["cohort"] == g]
        for t in periods:
            if t < g or t not in wide.columns:
                continue
            diff_treated = (treated_g[t] - treated_g[baseline]).mean()
            diff_control = (never[t] - never[baseline]).mean()
            out[(g, t)] = float(diff_treated - diff_control)
    return out


def cs_staggered_att(
    df: pd.DataFrame,
    outcome: str = "y",
    unit: str = "unit",
    period: str = "period",
    cohort: str = "cohort",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> CSResult:
    """
    Callaway & Sant'Anna (2021) staggered difference-in-differences.

    Estimates group-time average treatment effects ATT(g, t) using a
    never-treated control group and the outcome-regression (unconditional)
    identifier. Standard errors come from a **unit cluster bootstrap**.

    This estimator is robust to treatment-effect heterogeneity across cohorts,
    unlike two-way fixed-effects DiD which is biased in that setting
    (Goodman-Bacon 2021, de Chaisemartin & D'Haultfoeuille 2020).

    Parameters
    ----------
    df : long-format panel with columns [unit, period, outcome, cohort].
        ``cohort`` must be the treatment period for treated units and 0
        for never-treated units.
    n_bootstrap : number of cluster-bootstrap replications for inference.
    alpha : confidence level (default 5%).

    Returns
    -------
    CSResult with:
      * group_time_att : one row per (g, t) with t >= g
      * event_study   : ATT(e) aggregated by relative time e = t - g
      * overall_att   : simple average across group-time cells with t >= g
    """
    work = df[[unit, period, outcome, cohort]].copy()
    wide = work.pivot(index=[unit, cohort], columns=period, values=outcome).reset_index()

    cohorts = sorted([int(c) for c in wide[cohort].unique() if c > 0])
    periods = sorted([c for c in wide.columns if isinstance(c, int | np.integer)])

    point_gt = _cs_group_time_point_estimates(wide, cohorts, periods)
    if not point_gt:
        raise ValueError("No identifiable (g, t) cells — check cohort/period coding.")

    rng = np.random.default_rng(seed)
    units = wide[unit].to_numpy()
    n = len(units)

    boot_gt: dict[tuple[int, int], list[float]] = {k: [] for k in point_gt}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resamp = wide.iloc[idx].reset_index(drop=True)
        b = _cs_group_time_point_estimates(resamp, cohorts, periods)
        for k in boot_gt:
            if k in b:
                boot_gt[k].append(b[k])

    z = 1.959963984540054  # ~N^{-1}(0.975); alpha is wired to 5% conventions
    if alpha != 0.05:
        from scipy.stats import norm
        z = float(norm.ppf(1 - alpha / 2))

    gt_rows = []
    for (g, t), point in point_gt.items():
        boots = np.asarray(boot_gt[(g, t)])
        se = float(boots.std(ddof=1)) if len(boots) > 1 else float("nan")
        gt_rows.append(
            {
                "cohort": g,
                "period": t,
                "relative_time": t - g,
                "att": point,
                "se": se,
                "ci_low": point - z * se,
                "ci_high": point + z * se,
                "n_treated": int((wide[cohort] == g).sum()),
                "n_control": int((wide[cohort] == 0).sum()),
            }
        )
    gt_df = pd.DataFrame(gt_rows).sort_values(["cohort", "period"]).reset_index(drop=True)

    # Event-study aggregation: average ATT(g, g+e) across cohorts, equal-weighted
    # by cohort (Callaway-Sant'Anna "simple" aggregation). Bootstrap SE aggregates
    # the same way across each replicate.
    es_points: dict[int, list[float]] = {}
    for (g, t), point in point_gt.items():
        es_points.setdefault(t - g, []).append(point)

    es_boots: dict[int, list[list[float]]] = {}
    for (g, t), boots in boot_gt.items():
        es_boots.setdefault(t - g, []).append(boots)

    es_rows = []
    for e in sorted(es_points):
        att_e = float(np.mean(es_points[e]))
        replicate_means = np.mean(
            np.vstack([np.asarray(b) for b in es_boots[e]]), axis=0
        )
        se_e = float(replicate_means.std(ddof=1)) if len(replicate_means) > 1 else float("nan")
        es_rows.append(
            {
                "relative_time": e,
                "att": att_e,
                "se": se_e,
                "ci_low": att_e - z * se_e,
                "ci_high": att_e + z * se_e,
            }
        )
    es_df = pd.DataFrame(es_rows)

    # Overall ATT: equal-weighted mean of ATT(g, t) for t >= g.
    overall_point = float(np.mean(list(point_gt.values())))
    all_boot_matrix = np.vstack(
        [np.asarray(boot_gt[k]) for k in point_gt if len(boot_gt[k]) > 1]
    )
    overall_replicates = all_boot_matrix.mean(axis=0)
    overall_se = float(overall_replicates.std(ddof=1))

    return CSResult(
        group_time_att=gt_df,
        event_study=es_df,
        overall_att=overall_point,
        overall_se=overall_se,
        overall_ci_low=overall_point - z * overall_se,
        overall_ci_high=overall_point + z * overall_se,
        n_units=int(wide[unit].nunique()),
        n_bootstrap=n_bootstrap,
    )
