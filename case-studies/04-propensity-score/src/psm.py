"""
Propensity score methods for observational causal inference.

When randomization isn't possible, we assume *conditional unconfoundedness*:

    (Y(0), Y(1)) ⫫ T  |  X

i.e., conditional on observed covariates X, treatment assignment is as good
as random. This module estimates the propensity score e(x) = P(T=1 | X=x)
and uses it for:

  - 1-nearest-neighbor matching on the (logit) propensity, with caliper
  - Inverse-probability weighting (IPW)
  - Augmented IPW / doubly-robust (AIPW) ATT estimator

Standard errors come from a paired bootstrap by default (Abadie-Imbens 2008
showed standard bootstrap is invalid for *fixed-bandwidth* matching, but
paired bootstrap of the matched ATT remains a widely used pragmatic choice).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


@dataclass
class MatchingResult:
    att: float
    std_error: float
    ci_low: float
    ci_high: float
    n_treated: int
    n_matched: int
    n_unmatched_caliper: int
    method: str

    def __repr__(self) -> str:
        return (
            f"{self.method}(ATT={self.att:+.3f}, SE={self.std_error:.3f}, "
            f"95% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}], "
            f"matched={self.n_matched}/{self.n_treated}, "
            f"caliper_drops={self.n_unmatched_caliper})"
        )


def estimate_propensity(
    X: np.ndarray,
    T: np.ndarray,
    C: float = 1.0,
) -> np.ndarray:
    """Logistic regression P(T=1 | X). Returns propensity in (0,1)."""
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X, T)
    e = model.predict_proba(X)[:, 1]
    # Clip away from 0/1 for numerical stability of logit and IPW.
    return np.clip(e, 1e-4, 1 - 1e-4)


def standardized_mean_diff(
    X: np.ndarray, T: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Per-covariate standardized mean difference (SMD) between treated and control.

    SMD < 0.1 (in absolute value) is the conventional threshold for "balanced".
    """
    if weights is None:
        weights = np.ones_like(T, dtype=float)
    w_t = weights[T == 1]
    w_c = weights[T == 0]
    Xt = X[T == 1]
    Xc = X[T == 0]

    mu_t = np.average(Xt, axis=0, weights=w_t)
    mu_c = np.average(Xc, axis=0, weights=w_c)
    var_t = np.average((Xt - mu_t) ** 2, axis=0, weights=w_t)
    var_c = np.average((Xc - mu_c) ** 2, axis=0, weights=w_c)
    pooled_sd = np.sqrt((var_t + var_c) / 2)
    pooled_sd = np.where(pooled_sd == 0, 1, pooled_sd)
    return (mu_t - mu_c) / pooled_sd


def _match_indices(
    propensity: np.ndarray,
    T: np.ndarray,
    caliper: float | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    For each treated unit, find the nearest control on the *logit* propensity.

    Returns
    -------
    treated_idx : indices of treated units that found a match within caliper
    control_idx : matched control index for each retained treated unit (with replacement)
    n_dropped   : how many treated units were dropped due to caliper
    """
    logit_e = np.log(propensity / (1 - propensity))
    treated_idx_all = np.where(T == 1)[0]
    control_idx_all = np.where(T == 0)[0]

    nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nbrs.fit(logit_e[control_idx_all].reshape(-1, 1))
    dist, idx = nbrs.kneighbors(logit_e[treated_idx_all].reshape(-1, 1))
    dist = dist.ravel()
    matched_control = control_idx_all[idx.ravel()]

    if caliper is not None:
        # Caliper on logit-propensity scale (Rosenbaum-Rubin: 0.2 * SD(logit_e))
        keep = dist <= caliper
        n_dropped = int((~keep).sum())
        return treated_idx_all[keep], matched_control[keep], n_dropped
    return treated_idx_all, matched_control, 0


def psm_att(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    caliper_sd: float | None = 0.2,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> MatchingResult:
    """
    1-NN propensity score matching ATT with paired bootstrap SE.

    `caliper_sd`: caliper as a multiple of SD(logit propensity).
    Set to None for no caliper.
    """
    propensity = estimate_propensity(X, T)
    logit_e = np.log(propensity / (1 - propensity))
    caliper = caliper_sd * np.std(logit_e) if caliper_sd is not None else None

    t_idx, c_idx, dropped = _match_indices(propensity, T, caliper)
    if len(t_idx) == 0:
        raise RuntimeError("All treated units dropped by caliper")

    diffs = Y[t_idx] - Y[c_idx]
    att = float(np.mean(diffs))

    # Paired bootstrap on matched (treated, control) pairs.
    rng = np.random.default_rng(seed)
    boot_atts = np.empty(n_bootstrap)
    n = len(diffs)
    for b in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        boot_atts[b] = diffs[sample].mean()
    se = float(boot_atts.std(ddof=1))
    ci_low, ci_high = (float(x) for x in np.quantile(boot_atts, [0.025, 0.975]))

    return MatchingResult(
        att=att,
        std_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
        n_treated=int((T == 1).sum()),
        n_matched=len(t_idx),
        n_unmatched_caliper=dropped,
        method="psm_1nn",
    )


def ipw_att(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
    """
    Inverse-probability-weighted ATT (Hirano-Imbens-Ridder 2003 estimator).

        ATT = (1/n_t) * sum_i [ T_i * Y_i - (1 - T_i) * (e_i / (1 - e_i)) * Y_i ]
    """
    e = estimate_propensity(X, T)
    n_t = (T == 1).sum()
    treated = (T * Y).sum()
    control = ((1 - T) * (e / (1 - e)) * Y).sum()
    return float((treated - control) / n_t)


def aipw_att(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray
) -> float:
    """
    Augmented IPW (doubly robust) ATT.

    Consistent if *either* the propensity model or the outcome model is correct.
    Outcome model: linear regression of Y on X among controls only.
    """
    from sklearn.linear_model import LinearRegression

    e = estimate_propensity(X, T)
    mu0_model = LinearRegression().fit(X[T == 0], Y[T == 0])
    mu0_hat = mu0_model.predict(X)

    n_t = (T == 1).sum()
    contribution = (
        T * (Y - mu0_hat) - (1 - T) * (e / (1 - e)) * (Y - mu0_hat)
    )
    return float(contribution.sum() / n_t)


def balance_table(
    X: np.ndarray,
    T: np.ndarray,
    feature_names: list[str] | None = None,
    matched_t_idx: np.ndarray | None = None,
    matched_c_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Pre/post-matching balance: SMD per covariate.
    """
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(X.shape[1])]

    pre = standardized_mean_diff(X, T)

    if matched_t_idx is not None and matched_c_idx is not None:
        X_matched = np.vstack([X[matched_t_idx], X[matched_c_idx]])
        T_matched = np.concatenate(
            [np.ones(len(matched_t_idx)), np.zeros(len(matched_c_idx))]
        )
        post = standardized_mean_diff(X_matched, T_matched)
    else:
        post = np.full_like(pre, np.nan)

    return pd.DataFrame(
        {"covariate": feature_names, "smd_unmatched": pre, "smd_matched": post}
    )
