"""Tests for CUPED implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-studies" / "01-ab-cuped" / "src"))

from cuped import apply_cuped, compute_theta, cuped_ab_test, naive_ab_test  # noqa: E402
from cuped_simulate import simulate_experiment  # noqa: E402


def test_theta_zero_when_uncorrelated():
    rng = np.random.default_rng(0)
    y = rng.normal(size=10_000)
    x = rng.normal(size=10_000)
    assert abs(compute_theta(y, x)) < 0.05


def test_cuped_preserves_mean_under_randomization():
    df = simulate_experiment(n=5_000, true_effect=0.0, seed=1)
    y_adj = apply_cuped(df["minutes"].to_numpy(), df["pre_minutes"].to_numpy())
    # Adjusted mean should be close to original mean
    assert abs(y_adj.mean() - df["minutes"].mean()) < 0.01


def test_cuped_agrees_with_naive_pointwise():
    """For any given sample, CUPED and naive should give similar point estimates
    (they estimate the same ATE; they just differ in variance). The difference
    should be small relative to the naive SE."""
    df = simulate_experiment(n=20_000, true_effect=3.0, seed=42)
    naive = naive_ab_test(df, "minutes", "treatment")
    cuped = cuped_ab_test(df, "minutes", "treatment", "pre_minutes")
    assert abs(cuped.estimate - naive.estimate) < naive.std_error


def test_cuped_unbiased_across_seeds():
    """Averaging across many simulated experiments, CUPED should recover the
    true effect. This is the honest unbiasedness check."""
    estimates = []
    for seed in range(100):
        df = simulate_experiment(n=5_000, true_effect=3.0, seed=seed)
        estimates.append(
            cuped_ab_test(df, "minutes", "treatment", "pre_minutes").estimate
        )
    mean_est = float(np.mean(estimates))
    # 100 runs at n=5000, SE per run ~0.8 for naive; mean-of-means SE ~0.08
    assert abs(mean_est - 3.0) < 0.15


def test_cuped_reduces_standard_error():
    df = simulate_experiment(n=20_000, true_effect=3.0, correlation=0.7, seed=42)
    naive = naive_ab_test(df, "minutes", "treatment")
    cuped = cuped_ab_test(df, "minutes", "treatment", "pre_minutes")
    # With rho=0.7, expected SE ratio is sqrt(1 - 0.49) ~= 0.71
    assert cuped.std_error < naive.std_error
    assert cuped.std_error / naive.std_error < 0.85


def test_no_gain_when_covariate_uncorrelated():
    df = simulate_experiment(n=10_000, true_effect=1.0, correlation=0.0, seed=3)
    naive = naive_ab_test(df, "minutes", "treatment")
    cuped = cuped_ab_test(df, "minutes", "treatment", "pre_minutes")
    # SEs should be approximately equal when covariate is uninformative
    ratio = cuped.std_error / naive.std_error
    assert 0.9 < ratio < 1.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
