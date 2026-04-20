"""Tests for DiD implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-studies" / "03-diff-in-diff" / "src"))

from did import did_2x2, event_study  # noqa: E402
from did_simulate import simulate_did_panel  # noqa: E402


def test_did_recovers_true_att():
    panel = simulate_did_panel(
        n_units=30, n_treated=10, n_periods=20,
        treatment_period=12, true_effect=2.0, seed=0,
    )
    result = did_2x2(panel)
    assert abs(result.att - 2.0) < 0.6
    assert result.ci_low < 2.0 < result.ci_high


def test_did_zero_effect_null():
    panel = simulate_did_panel(
        n_units=30, n_treated=10, n_periods=20,
        treatment_period=12, true_effect=0.0, seed=1,
    )
    result = did_2x2(panel)
    assert abs(result.att) < 0.6
    # Null should not be rejected at alpha=0.05
    assert result.p_value > 0.05


def test_did_unbiased_across_seeds():
    estimates = []
    for s in range(50):
        panel = simulate_did_panel(
            n_units=30, n_treated=10, n_periods=20,
            treatment_period=12, true_effect=2.0, seed=s,
        )
        estimates.append(did_2x2(panel).att)
    assert abs(float(np.mean(estimates)) - 2.0) < 0.15


def test_event_study_pre_treatment_coefs_near_zero():
    panel = simulate_did_panel(
        n_units=30, n_treated=10, n_periods=20,
        treatment_period=12, true_effect=2.0, seed=0,
    )
    es = event_study(panel, treatment_period=12)
    pre = es.coefs[es.coefs["relative_time"] < -1]
    # Pre-treatment leads should be near 0 (excluding the omitted -1 base)
    assert pre["coef"].abs().mean() < 0.6
    # Parallel-trends F-test should not reject
    assert es.parallel_trends_pvalue > 0.05


def test_event_study_post_coefs_near_true_effect():
    panel = simulate_did_panel(
        n_units=30, n_treated=10, n_periods=20,
        treatment_period=12, true_effect=2.0, seed=0,
    )
    es = event_study(panel, treatment_period=12)
    post = es.coefs[es.coefs["relative_time"] >= 0]
    # Each post-treatment dynamic effect should hover around 2.0
    assert abs(post["coef"].mean() - 2.0) < 0.6


def test_event_study_detects_parallel_trends_violation():
    panel = simulate_did_panel(
        n_units=30, n_treated=10, n_periods=20,
        treatment_period=12, true_effect=2.0,
        parallel_trend_violation=0.4, seed=0,
    )
    es = event_study(panel, treatment_period=12)
    # With diverging trends, the joint test on leads should reject.
    assert es.parallel_trends_pvalue < 0.05


def test_did_under_pt_violation_is_biased():
    """Sanity check: when PT fails, 2x2 DiD picks up the trend gap as 'effect'."""
    panel = simulate_did_panel(
        n_units=30, n_treated=10, n_periods=20,
        treatment_period=12, true_effect=2.0,
        parallel_trend_violation=0.5, seed=0,
    )
    result = did_2x2(panel)
    # Should be substantially above 2.0 because of the diverging trend
    assert result.att > 2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
