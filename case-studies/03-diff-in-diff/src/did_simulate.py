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
