"""
Simulate an A/B experiment with a pre-experiment covariate.

Scenario: streaming platform tests a new recommendation algorithm.
    - Outcome Y: minutes watched during the 14-day experiment.
    - Pre-covariate X: minutes watched in the 14 days *before* the experiment.
    - True treatment effect: +3 minutes on average.

The pre-period covariate X is strongly correlated with Y (users who watched
a lot before tend to watch a lot after). CUPED exploits this correlation to
shrink the standard error of the treatment effect estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_experiment(
    n: int = 20_000,
    true_effect: float = 3.0,
    correlation: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns: user_id, treatment, pre_minutes, minutes.

    Parameters
    ----------
    n : total users (split 50/50)
    true_effect : ATE in outcome units (minutes)
    correlation : target Pearson correlation between pre and post outcomes
    """
    rng = np.random.default_rng(seed)

    pre = rng.normal(loc=120, scale=40, size=n)

    noise_scale = 40 * np.sqrt(1 - correlation**2)
    base_post = correlation * (pre - 120) + 120
    noise = rng.normal(0, noise_scale, size=n)

    treatment = rng.integers(0, 2, size=n)
    post = base_post + noise + true_effect * treatment

    return pd.DataFrame(
        {
            "user_id": np.arange(n),
            "treatment": treatment,
            "pre_minutes": pre,
            "minutes": post,
        }
    )
