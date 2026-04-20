"""
Simulate observational data with confounding for propensity score methods.

Scenario: e-commerce platform wants to estimate the effect of installing
the mobile app (treatment T=1) on annual spend (outcome Y).

Selection: heavy users, premium members, and younger users are more likely
to install the app -- so naive (T=1 vs T=0) comparisons overstate the effect.

Covariates X (5 dims, all observed):
    age (years), prior_orders, premium_member (0/1),
    region_score (continuous), days_since_signup

True ATE = +50 (in dollars). Naive comparison overstates this; PSM/IPW
should recover the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "age",
    "prior_orders",
    "premium_member",
    "region_score",
    "days_since_signup",
]


def simulate_observational(
    n: int = 5000,
    true_effect: float = 50.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns (X, T, Y, df_long) where df_long is a tidy DataFrame for inspection.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(35, 10, size=n).clip(18, 80)
    prior_orders = rng.poisson(5, size=n).astype(float)
    premium = rng.binomial(1, 0.3, size=n).astype(float)
    region = rng.normal(0, 1, size=n)
    days_signup = rng.gamma(shape=2.0, scale=180, size=n)

    X = np.column_stack([age, prior_orders, premium, region, days_signup])

    # Selection model: younger, heavier, premium users select into app install
    age_z = (age - 35) / 10
    orders_z = (prior_orders - 5) / 3
    days_z = (days_signup - 360) / 180
    logit = (
        -0.5
        + (-0.4) * age_z
        + 0.6 * orders_z
        + 0.8 * premium
        + 0.2 * region
        + 0.1 * days_z
    )
    p_treat = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, p_treat).astype(int)

    # Outcome model: confounders independently drive spend; treatment adds true_effect.
    Y = (
        300
        + 8 * age_z
        + 25 * orders_z
        + 60 * premium
        + 15 * region
        + 5 * days_z
        + true_effect * T
        + rng.normal(0, 30, size=n)
    )

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["T"] = T
    df["Y"] = Y
    df["propensity_true"] = p_treat
    return X, T, Y, df
