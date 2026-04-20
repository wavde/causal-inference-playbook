"""Tests for sequential testing implementations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-studies" / "05-sequential-testing" / "src"))

from sequential import (  # noqa: E402
    fixed_horizon_zscore,
    msprt_log_likelihood_ratio,
    msprt_sequential_test,
    naive_peeking_test,
    pocock_critical_value,
    pocock_sequential_test,
)
from seq_simulate import simulate_stream  # noqa: E402


def test_fixed_horizon_basic():
    y_t, y_c = simulate_stream(n_per_arm=2000, true_effect=0.5, seed=0)
    z, p, delta = fixed_horizon_zscore(y_t, y_c)
    assert p < 0.001
    assert abs(delta - 0.5) < 0.1


def test_msprt_log_lr_zero_at_zero_estimate():
    """log-LR should be log(sqrt(V/(V+tau^2))) <= 0 when delta_hat=0."""
    log_lr = msprt_log_likelihood_ratio(delta_hat=0.0, n_per_arm=100, sigma=1.0, tau=0.1)
    assert log_lr < 0


def test_msprt_log_lr_increases_with_estimate_magnitude():
    base = msprt_log_likelihood_ratio(0.0, 100, 1.0, 0.1)
    bigger = msprt_log_likelihood_ratio(0.3, 100, 1.0, 0.1)
    assert bigger > base


def test_naive_peeking_inflates_type1_error():
    """With repeated looks, naive z-test's false-positive rate should exceed 0.05."""
    rejs = 0
    n_sims = 200
    for s in range(n_sims):
        y_t, y_c = simulate_stream(n_per_arm=2000, true_effect=0.0, seed=s)
        if naive_peeking_test(y_t, y_c, alpha=0.05, look_every=50).rejected:
            rejs += 1
    rate = rejs / n_sims
    # Should be well above 0.05 -- this is the bug we're warning about
    assert rate > 0.10


def test_msprt_controls_type1_error():
    """mSPRT's empirical rejection rate under H0 should be <= alpha (with slack)."""
    rejs = 0
    n_sims = 300
    for s in range(n_sims):
        y_t, y_c = simulate_stream(n_per_arm=3000, true_effect=0.0, seed=s)
        if msprt_sequential_test(y_t, y_c, alpha=0.05, look_every=100).rejected:
            rejs += 1
    rate = rejs / n_sims
    # Allow a bit of slack since alpha-control is asymptotic in continuous time;
    # in finite discrete looks rate may sit slightly above alpha but should
    # not approach the naive-peeking inflation level (10%+).
    assert rate <= 0.10


def test_msprt_has_power_under_h1():
    rejs = 0
    n_sims = 100
    for s in range(n_sims):
        y_t, y_c = simulate_stream(n_per_arm=5000, true_effect=0.15, seed=s)
        if msprt_sequential_test(y_t, y_c, alpha=0.05, look_every=100).rejected:
            rejs += 1
    rate = rejs / n_sims
    assert rate > 0.6


def test_pocock_critical_value_increases_with_K():
    c1 = pocock_critical_value(0.05, K=1)
    c5 = pocock_critical_value(0.05, K=5)
    c10 = pocock_critical_value(0.05, K=10)
    # K=1 should give ~1.96; more looks => higher per-look threshold
    assert abs(c1 - 1.96) < 0.05
    assert c5 > c1
    assert c10 > c5


def test_pocock_controls_type1_error():
    rejs = 0
    n_sims = 200
    for s in range(n_sims):
        y_t, y_c = simulate_stream(n_per_arm=2000, true_effect=0.0, seed=s)
        if pocock_sequential_test(y_t, y_c, alpha=0.05, K=5).rejected:
            rejs += 1
    assert rejs / n_sims <= 0.10


def test_msprt_stops_earlier_than_fixed_horizon_under_h1():
    msprt_stops = []
    for s in range(50):
        y_t, y_c = simulate_stream(n_per_arm=5000, true_effect=0.3, seed=s)
        msprt_stops.append(msprt_sequential_test(y_t, y_c, alpha=0.05, look_every=50).stopped_at)
    # Average stop should be substantially less than the full horizon (5000)
    assert float(np.mean(msprt_stops)) < 4000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
