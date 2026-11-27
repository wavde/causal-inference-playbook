"""
Simulated users of an opt-in AI writing assistant.

Each user has three periods of a weekly engagement outcome (sessions per week):

    t = 0  pre-period 1 (two periods before launch)
    t = 1  pre-period 2 (the period just before launch)
    t = 2  post-period  (after launch; adopters have the feature)

Outcome model:

    y_it = 10 + 1.0*tenure + 1.5*activity + gamma_u*U + lambda_v*V*t
           + kappa_s*S*[t=2] + tau_i*T_i*[t=2] + eps_it,    eps_it ~ N(0, 2^2)

    tau_i = tau0 + tau1 * activity              (heterogeneous effect)

Adoption model (logistic):

    logit P(T=1) = -1 + 0.5*tenure + 0.8*activity + c_u*U + c_v*V + c_s*S

``tenure`` and ``activity`` are observed. ``U`` (a stable "power user" trait),
``V`` (a user-specific growth trend) and ``S`` (a shock that only shows up in
the post-period, e.g. starting a writing-heavy project) are not. Each scenario
switches on one unobserved selection channel, so the ground truth and the
mechanism of any bias are known by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

NOISE_SD = 2.0
FEATURE_NAMES = ["tenure", "activity"]


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    tau0: float = 1.0
    tau1: float = 0.5
    gamma_u: float = 0.0
    c_u: float = 0.0
    lambda_v: float = 0.0
    c_v: float = 0.0
    kappa_s: float = 0.0
    c_s: float = 0.0


_BASE = [
    Scenario("observables", "Selection on observed X only"),
    Scenario("fixed_trait", "Selection on a stable unobserved trait U", gamma_u=2.0, c_u=1.0),
    Scenario("pre_trend", "Selection on an unobserved growth trend V", lambda_v=0.5, c_v=1.0),
    Scenario("post_shock", "Selection on an unobserved post-period shock S", kappa_s=1.0, c_s=1.0),
]
SCENARIOS: dict[str, Scenario] = {s.name: s for s in _BASE}
SCENARIOS["null_pre_trend"] = replace(
    SCENARIOS["pre_trend"], name="null_pre_trend", label="Null effect + growth trend V",
    tau0=0.0, tau1=0.0,
)
SCENARIOS["null_post_shock"] = replace(
    SCENARIOS["post_shock"], name="null_post_shock", label="Null effect + post-period shock S",
    tau0=0.0, tau1=0.0,
)


@dataclass
class AdoptionData:
    X: np.ndarray       # (n, 2) observed covariates: tenure, activity
    T: np.ndarray       # (n,) adoption indicator
    y_pre1: np.ndarray  # t = 0
    y_pre2: np.ndarray  # t = 1
    y_post: np.ndarray  # t = 2
    tau: np.ndarray     # (n,) individual treatment effect (known by construction)

    def to_long(self, first: str = "y_pre2", second: str = "y_post") -> pd.DataFrame:
        """Two-period long panel in the format expected by ``did.did_2x2``."""
        n = len(self.T)
        unit = np.arange(n)
        zeros, ones = np.zeros(n, dtype=int), np.ones(n, dtype=int)
        return pd.DataFrame(
            {
                "unit": np.r_[unit, unit],
                "period": np.r_[zeros, ones],
                "y": np.r_[getattr(self, first), getattr(self, second)],
                "treated_unit": np.r_[self.T, self.T],
                "post": np.r_[zeros, ones],
            }
        )


def _draw_users(sc: Scenario, n: int, rng: np.random.Generator):
    tenure = rng.normal(size=n)
    activity = 0.4 * tenure + np.sqrt(1 - 0.4**2) * rng.normal(size=n)
    U, V, S = rng.normal(size=(3, n))
    logit = -1.0 + 0.5 * tenure + 0.8 * activity + sc.c_u * U + sc.c_v * V + sc.c_s * S
    p = 1.0 / (1.0 + np.exp(-logit))
    tau = sc.tau0 + sc.tau1 * activity
    return tenure, activity, U, V, S, p, tau


def simulate(sc: Scenario, n: int, rng: np.random.Generator) -> AdoptionData:
    tenure, activity, U, V, S, p, tau = _draw_users(sc, n, rng)
    T = (rng.uniform(size=n) < p).astype(int)
    base = 10.0 + 1.0 * tenure + 1.5 * activity + sc.gamma_u * U
    eps = rng.normal(scale=NOISE_SD, size=(3, n))
    return AdoptionData(
        X=np.column_stack([tenure, activity]),
        T=T,
        y_pre1=base + eps[0],
        y_pre2=base + sc.lambda_v * V + eps[1],
        y_post=base + 2 * sc.lambda_v * V + sc.kappa_s * S + tau * T + eps[2],
        tau=tau,
    )


def population_truth(sc: Scenario, n: int = 2_000_000, seed: int = 12345) -> dict[str, float]:
    """
    Population ATT and ATE, known by construction.

    ATE = tau0 exactly because E[activity] = 0. ATT = E[tau | T=1] is computed
    by weighting a large draw of individual effects by each user's adoption
    probability, which avoids the extra noise of sampling T.
    """
    rng = np.random.default_rng(seed)
    *_, p, tau = _draw_users(sc, n, rng)
    return {
        "att": float(np.sum(p * tau) / np.sum(p)),
        "ate": float(sc.tau0),
        "adopt_rate": float(p.mean()),
    }
