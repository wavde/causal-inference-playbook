"""
Simulate a panel for synthetic control.

Scenario: streaming platform tests a price increase in *one country*.
    - N = 20 countries (1 treated + 19 donors)
    - T = 30 months of monthly active subscriber index (baseline 100)
    - Each country has its own intercept + growth trend
    - All countries share two latent factors (regional trends)
    - Treatment lowers the treated country's subscribers by a true ATT post-period

Parameters are tuned so a well-fit synthetic control should recover an ATT
close to the true effect, while placebo donors show much smaller gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_panel(
    n_units: int = 20,
    n_periods: int = 30,
    pre_periods: int = 20,
    treated_idx: int = 0,
    true_effect: float = -5.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Return a long-format panel: columns [unit, period, y, treated].

    `treated` is 1 only for the treated unit in post periods.
    """
    rng = np.random.default_rng(seed)

    unit_intercepts = rng.normal(100, 5, size=n_units)
    unit_trends = rng.normal(0.3, 0.15, size=n_units)

    t = np.arange(n_periods)
    f1 = np.sin(2 * np.pi * t / 12)
    f2 = np.cumsum(rng.normal(0, 0.3, size=n_periods))
    loadings1 = rng.normal(2, 0.8, size=n_units)
    loadings2 = rng.normal(1, 0.5, size=n_units)

    Y = np.zeros((n_periods, n_units))
    for i in range(n_units):
        level = unit_intercepts[i] + unit_trends[i] * t
        factor = loadings1[i] * f1 + loadings2[i] * f2
        noise = rng.normal(0, 1.0, size=n_periods)
        Y[:, i] = level + factor + noise

    Y[pre_periods:, treated_idx] += true_effect

    rows = []
    for i in range(n_units):
        for p in range(n_periods):
            rows.append(
                {
                    "unit": f"country_{i:02d}",
                    "period": p,
                    "y": float(Y[p, i]),
                    "treated": int(i == treated_idx and p >= pre_periods),
                }
            )
    return pd.DataFrame(rows)


def panel_to_matrix(
    df: pd.DataFrame, treated_name: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pivot long-format panel into (y_treated, Y_donors, donor_names)."""
    wide = df.pivot(index="period", columns="unit", values="y").sort_index()
    y_treated = wide[treated_name].to_numpy()
    donors = [c for c in wide.columns if c != treated_name]
    Y_donors = wide[donors].to_numpy()
    return y_treated, Y_donors, donors
