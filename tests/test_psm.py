"""Tests for PSM implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-studies" / "04-propensity-score" / "src"))

from psm import (  # noqa: E402
    _match_indices,
    aipw_att,
    balance_table,
    estimate_propensity,
    ipw_att,
    psm_att,
    standardized_mean_diff,
)
from psm_simulate import FEATURE_NAMES, simulate_observational  # noqa: E402


def test_propensity_in_unit_interval():
    X, T, Y, _ = simulate_observational(n=1000, seed=0)
    e = estimate_propensity(X, T)
    assert e.shape == T.shape
    assert (e > 0).all() and (e < 1).all()


def test_naive_estimator_is_biased():
    X, T, Y, _ = simulate_observational(n=5000, true_effect=50.0, seed=0)
    naive = float(Y[T == 1].mean() - Y[T == 0].mean())
    # Selection on premium/heavy users should inflate the naive estimate
    assert naive > 60.0


def test_psm_recovers_true_att():
    X, T, Y, _ = simulate_observational(n=5000, true_effect=50.0, seed=0)
    result = psm_att(X, T, Y, caliper_sd=0.2, n_bootstrap=200, seed=0)
    assert abs(result.att - 50.0) < 8.0


def test_ipw_recovers_true_att():
    X, T, Y, _ = simulate_observational(n=5000, true_effect=50.0, seed=0)
    est = ipw_att(X, T, Y)
    assert abs(est - 50.0) < 8.0


def test_aipw_recovers_true_att():
    X, T, Y, _ = simulate_observational(n=5000, true_effect=50.0, seed=0)
    est = aipw_att(X, T, Y)
    assert abs(est - 50.0) < 8.0


def test_psm_unbiased_across_seeds():
    estimates = []
    for s in range(20):
        X, T, Y, _ = simulate_observational(n=3000, true_effect=50.0, seed=s)
        estimates.append(psm_att(X, T, Y, n_bootstrap=50, seed=s).att)
    assert abs(float(np.mean(estimates)) - 50.0) < 5.0


def test_matching_improves_balance():
    X, T, Y, _ = simulate_observational(n=5000, seed=0)
    e = estimate_propensity(X, T)
    logit_e = np.log(e / (1 - e))
    caliper = 0.2 * np.std(logit_e)
    t_idx, c_idx, _ = _match_indices(e, T, caliper=caliper)
    bal = balance_table(X, T, FEATURE_NAMES, matched_t_idx=t_idx, matched_c_idx=c_idx)
    # Average |SMD| should drop substantially after matching
    pre_avg = bal["smd_unmatched"].abs().mean()
    post_avg = bal["smd_matched"].abs().mean()
    assert post_avg < pre_avg / 2


def test_smd_zero_when_groups_identical():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    T = np.r_[np.ones(100), np.zeros(100)]
    # Make T groups identical by tiling
    X[:100] = X[100:]
    smd = standardized_mean_diff(X, T)
    assert np.allclose(smd, 0)


def test_caliper_drops_when_too_tight():
    X, T, Y, _ = simulate_observational(n=2000, seed=0)
    result = psm_att(X, T, Y, caliper_sd=0.001, n_bootstrap=20, seed=0)
    assert result.n_unmatched_caliper > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
