"""
Estimators for the opt-in adoption case, each paired with its own placebo.

Point estimates reuse the playbook's existing implementations:

  * ``psm.estimate_propensity`` (case 04) for the propensity model, and the
    same AIPW ATT formula as ``psm.aipw_att``. That function returns only a
    point estimate, so ``aipw_att_se`` refits the same nuisances and adds an
    influence-function standard error. A test checks that both agree.
  * ``did.did_2x2`` (case 03) defines the DiD estimand. With two periods it
    equals the difference in mean first differences, which is what the Monte
    Carlo uses because two-way fixed effects with thousands of unit dummies
    is too slow to run thousands of times. A test checks they agree.
  * ``sensitivity.e_value`` (case 04) for the E-value.

Placebo ("fake launch") test: run the same estimator one period earlier,
pretending adoption happened between pre-period 1 and pre-period 2. The true
effect there is zero, so a significant placebo estimate flags bias that the
estimator's assumptions failed to remove.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

_CASES = Path(__file__).resolve().parents[2]
for _sub in ("04-propensity-score", "03-diff-in-diff"):
    _p = str(_CASES / _sub / "src")
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adoption_simulate import AdoptionData  # noqa: E402
from psm import estimate_propensity  # noqa: E402
from sensitivity import e_value  # noqa: E402

Z95 = float(norm.ppf(0.975))
ESTIMATORS = ["naive", "aipw_x", "aipw_x_pre", "did"]
ESTIMATOR_LABELS = {
    "naive": "Naive adopter vs non-adopter",
    "aipw_x": "AIPW on X",
    "aipw_x_pre": "AIPW on X + both pre-period outcomes",
    "did": "DiD (pre-period 2 to post)",
}


@dataclass
class Estimate:
    est: float
    se: float

    @property
    def ci_low(self) -> float:
        return self.est - Z95 * self.se

    @property
    def ci_high(self) -> float:
        return self.est + Z95 * self.se

    @property
    def z(self) -> float:
        return self.est / self.se

    def covers(self, value: float) -> bool:
        return self.ci_low <= value <= self.ci_high

    def rejects_zero(self) -> bool:
        return abs(self.z) > Z95


def diff_in_means(y: np.ndarray, T: np.ndarray) -> Estimate:
    """Difference in means with a Welch (unequal-variance) standard error."""
    y1, y0 = y[T == 1], y[T == 0]
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return Estimate(float(y1.mean() - y0.mean()), float(se))


def aipw_att_se(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Estimate:
    """
    AIPW (doubly robust) ATT, identical to ``psm.aipw_att``, plus an
    influence-function standard error that treats the fitted propensity and
    outcome models as fixed (the usual first-order approximation for AIPW).
    """
    e = estimate_propensity(X, T)
    mu0 = LinearRegression().fit(X[T == 0], Y[T == 0]).predict(X)
    resid = Y - mu0
    phi = T * resid - (1 - T) * (e / (1 - e)) * resid
    n, n_t = len(T), T.sum()
    att = phi.sum() / n_t
    psi = (phi - T * att) / (n_t / n)
    return Estimate(float(att), float(np.sqrt(np.sum(psi**2)) / n))


def did_first_diff(y_before: np.ndarray, y_after: np.ndarray, T: np.ndarray) -> Estimate:
    """Two-period DiD as a difference in mean changes (same point estimate as did_2x2)."""
    return diff_in_means(y_after - y_before, T)


def estimate_with_placebo(name: str, d: AdoptionData) -> tuple[Estimate, Estimate]:
    """Return (main estimate, placebo estimate) for one estimator."""
    X, T = d.X, d.T
    if name == "naive":
        return diff_in_means(d.y_post, T), diff_in_means(d.y_pre2, T)
    if name == "aipw_x":
        return aipw_att_se(X, T, d.y_post), aipw_att_se(X, T, d.y_pre2)
    if name == "aipw_x_pre":
        main = aipw_att_se(np.column_stack([X, d.y_pre1, d.y_pre2]), T, d.y_post)
        placebo = aipw_att_se(np.column_stack([X, d.y_pre1]), T, d.y_pre2)
        return main, placebo
    if name == "did":
        return did_first_diff(d.y_pre2, d.y_post, T), did_first_diff(d.y_pre1, d.y_pre2, T)
    raise ValueError(f"unknown estimator {name!r}")


def e_values(est: Estimate, sd_outcome: float) -> tuple[float, float]:
    """(point E-value, CI E-value) via the playbook's ``sensitivity.e_value``."""
    ev = e_value(est.est, sd_outcome=sd_outcome, ci_low=est.ci_low, ci_high=est.ci_high)
    return ev.e_value_point, float(ev.e_value_ci)
