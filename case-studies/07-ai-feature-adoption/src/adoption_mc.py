"""
Monte Carlo comparison of estimators against the known ATT.

Every replication draws a fresh user base from one scenario, runs each
estimator and its placebo, and records whether the 95% CI covers the true
ATT. Seeds are derived from (seed, scenario index, replication), so any
subset of replications is reproducible on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from adoption_estimators import ESTIMATORS, e_values, estimate_with_placebo
from adoption_simulate import SCENARIOS, population_truth, simulate


def run_scenario(name: str, n_reps: int, n_users: int, seed: int = 0) -> pd.DataFrame:
    """One row per (replication, estimator)."""
    sc = SCENARIOS[name]
    sc_idx = list(SCENARIOS).index(name)
    truth = population_truth(sc)
    rows = []
    for rep in range(n_reps):
        rng = np.random.default_rng([seed, sc_idx, rep])
        d = simulate(sc, n_users, rng)
        sd_y = float(d.y_post[d.T == 0].std(ddof=1))
        for est_name in ESTIMATORS:
            main, placebo = estimate_with_placebo(est_name, d)
            ev_point, ev_ci = e_values(main, sd_y)
            rows.append(
                {
                    "scenario": name,
                    "estimator": est_name,
                    "rep": rep,
                    "true_att": truth["att"],
                    "est": main.est,
                    "se": main.se,
                    "covers": main.covers(truth["att"]),
                    "rejects_zero": main.rejects_zero(),
                    "placebo_est": placebo.est,
                    "placebo_flags": placebo.rejects_zero(),
                    "e_value": ev_point,
                    "e_value_ci": ev_ci,
                }
            )
    return pd.DataFrame(rows)


def summarize(draws: pd.DataFrame) -> pd.DataFrame:
    """Bias, spread, coverage, rejection and diagnostic rates per scenario x estimator."""
    out = []
    for (sc, est), g in draws.groupby(["scenario", "estimator"], sort=False):
        passes = ~g["placebo_flags"]
        out.append(
            {
                "scenario": sc,
                "estimator": est,
                "true_att": g["true_att"].iloc[0],
                "mean_est": g["est"].mean(),
                "bias": g["est"].mean() - g["true_att"].iloc[0],
                "emp_sd": g["est"].std(ddof=1),
                "mean_se": g["se"].mean(),
                "coverage": g["covers"].mean(),
                "reject_zero": g["rejects_zero"].mean(),
                "placebo_flag": g["placebo_flags"].mean(),
                "median_e_value": g["e_value"].median(),
                "median_e_value_ci": g["e_value_ci"].median(),
                # The decision rule: act only when the placebo passes and the CI excludes 0.
                "acts": (passes & g["rejects_zero"]).mean(),
                # ...and how often acting happens on a CI that misses the truth.
                "acts_wrong": (passes & g["rejects_zero"] & ~g["covers"]).mean(),
            }
        )
    return pd.DataFrame(out)
