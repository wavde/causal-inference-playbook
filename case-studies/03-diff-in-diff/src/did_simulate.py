"""
Simulate a panel for difference-in-differences.

Scenario: Spotify rolls out 'Daily Mix v2' in some countries (treated)
on month T0 and not others (control). Outcome: minutes listened per
DAU per day (index, baseline 100).

Default DGP satisfies parallel trends. A `parallel_trend_violation`
flag injects diverging unit-specific trends to demo PT failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_did_panel(
    n_units: int = 30,
    n_treated: int = 10,
    n_periods: int = 20,
    treatment_period: int = 12,
    true_effect: float = 2.0,
    parallel_trend_violation: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Return long-format panel: columns [unit, period, y, treated_unit, post, treated].

    - `treated_unit`: 1 for the first n_treated units, 0 otherwise (time-invariant)
    - `post`: 1 for periods >= treatment_period (time-varying)
    - `treated`: treated_unit * post (DiD interaction)

    `parallel_trend_violation` (default 0): add this much extra slope per
    period to treated units only, breaking parallel trends.
    """
    rng = np.random.default_rng(seed)

    unit_intercepts = rng.normal(100, 8, size=n_units)
    common_trend = 0.4

    Y = np.zeros((n_periods, n_units))
    t = np.arange(n_periods)
    common_shocks = rng.normal(0, 1.0, size=n_periods).cumsum() * 0.3

    for i in range(n_units):
        is_treated = i < n_treated
        trend = common_trend + (parallel_trend_violation if is_treated else 0.0)
        level = unit_intercepts[i] + trend * t
        noise = rng.normal(0, 1.0, size=n_periods)
        Y[:, i] = level + common_shocks + noise

    Y[treatment_period:, :n_treated] += true_effect

    rows = []
    for i in range(n_units):
        is_treated = int(i < n_treated)
        for p in range(n_periods):
            post = int(p >= treatment_period)
            rows.append(
                {
                    "unit": f"country_{i:02d}",
                    "period": p,
                    "y": float(Y[p, i]),
                    "treated_unit": is_treated,
                    "post": post,
                    "treated": is_treated * post,
                }
            )
    return pd.DataFrame(rows)


def simulate_staggered_panel(
    cohort_sizes: dict[int, int] | None = None,
    n_never_treated: int = 20,
    n_periods: int = 20,
    cohort_effects: dict[int, float] | None = None,
    dynamic_slope: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Staggered-adoption panel where units turn on at different periods and
    treatment effects can differ across cohorts. Designed to expose the
    Goodman-Bacon / de Chaisemartin bias in TWFE and demonstrate the value
    of Callaway-Sant'Anna group-time ATTs.

    Parameters
    ----------
    cohort_sizes
        Mapping of ``treatment_period -> n_units_in_that_cohort``.
        Default is three cohorts treated at periods 6, 10, 14.
    n_never_treated
        Units that are never treated (serve as clean controls for CS).
    cohort_effects
        Mapping of ``treatment_period -> per-period-since-treatment ATT``.
        When cohorts have different effect sizes, TWFE is biased because
        already-treated units become implicit controls for later cohorts.
        Default gives cohorts effects of 1.0, 3.0, 5.0 respectively.
    dynamic_slope
        Extra ATT per period since treatment (0 = static effect).
    seed
        RNG seed.

    Returns
    -------
    Long-format DataFrame with columns:
    ``unit, period, y, cohort, treated_unit, post, treated, rel_time``.
    ``cohort`` is the treatment period (0 means never-treated).
    """
    if cohort_sizes is None:
        cohort_sizes = {6: 15, 10: 15, 14: 15}
    if cohort_effects is None:
        cohort_effects = {6: 1.0, 10: 3.0, 14: 5.0}

    rng = np.random.default_rng(seed)
    rows = []
    common_shocks = rng.normal(0, 1.0, size=n_periods).cumsum() * 0.3
    common_trend = 0.4

    unit_id = 0

    def add_unit(cohort: int, effect: float) -> None:
        nonlocal unit_id
        intercept = rng.normal(100, 8)
        noise = rng.normal(0, 1.0, size=n_periods)
        for p in range(n_periods):
            rel_time = p - cohort if cohort > 0 else -9999
            is_post = cohort > 0 and p >= cohort
            te = 0.0
            if is_post:
                te = effect + dynamic_slope * rel_time
            y = intercept + common_trend * p + common_shocks[p] + noise[p] + te
            rows.append(
                {
                    "unit": f"u_{unit_id:03d}",
                    "period": p,
                    "y": float(y),
                    "cohort": cohort,
                    "treated_unit": int(cohort > 0),
                    "post": int(is_post),
                    "treated": int(is_post),
                    "rel_time": int(rel_time) if cohort > 0 else 0,
                }
            )
        unit_id += 1

    for g, n in cohort_sizes.items():
        for _ in range(n):
            add_unit(g, cohort_effects[g])
    for _ in range(n_never_treated):
        add_unit(0, 0.0)

    return pd.DataFrame(rows)
