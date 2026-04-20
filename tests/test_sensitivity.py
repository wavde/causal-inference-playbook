"""Tests for sensitivity analysis (E-value and Rosenbaum bounds)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-studies" / "04-propensity-score" / "src"))

from psm_simulate import simulate_observational  # noqa: E402
from sensitivity import (  # noqa: E402
    _wilcoxon_signed_rank_one_sided_p,
    e_value,
    matched_pair_differences,
    rosenbaum_gamma_threshold,
    rosenbaum_wilcoxon_bounds,
)


def test_e_value_risk_ratio_of_two():
    """VanderWeele-Ding (2017) example: RR=2 -> E-value = 2 + sqrt(2) ~ 3.41."""
    r = e_value(2.0, outcome_type="risk_ratio")
    assert abs(r.e_value_point - (2 + np.sqrt(2))) < 1e-6


def test_e_value_null_rr_is_one():
    r = e_value(1.0, outcome_type="risk_ratio")
    assert r.e_value_point == 1.0


def test_e_value_symmetric_under_inversion():
    """E-value is the same for RR=0.5 as for RR=2."""
    r1 = e_value(2.0, outcome_type="risk_ratio")
    r2 = e_value(0.5, outcome_type="risk_ratio")
    assert abs(r1.e_value_point - r2.e_value_point) < 1e-6


def test_e_value_continuous_grows_with_effect():
    small = e_value(0.1, sd_outcome=1.0)
    large = e_value(1.0, sd_outcome=1.0)
    assert small.e_value_point < large.e_value_point
    assert small.e_value_point > 1.0


def test_e_value_ci_crossing_null_returns_one():
    """When the CI crosses the null, the CI E-value should be exactly 1.0."""
    r = e_value(0.3, sd_outcome=1.0, ci_low=-0.1, ci_high=0.7)
    assert r.e_value_ci == 1.0


def test_rosenbaum_gamma_one_matches_standard_wilcoxon():
    """At Gamma=1, Rosenbaum's bound is the ordinary Wilcoxon p-value."""
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.5, 1.0, size=80)   # clearly positive
    p = _wilcoxon_signed_rank_one_sided_p(diffs, gamma=1.0, alternative="greater")
    # With n=80 and mean +0.5, a one-sided Wilcoxon on H1: greater should
    # be well below 0.01.
    assert p < 0.01


def test_rosenbaum_p_monotonic_in_gamma():
    """As Gamma grows, the worst-case p-value should weakly increase."""
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.5, 1.0, size=100)
    bounds = rosenbaum_wilcoxon_bounds(diffs, gammas=[1.0, 1.5, 2.0, 3.0])
    ps = bounds["p_upper_bound"].to_numpy()
    assert np.all(np.diff(ps) >= -1e-12)


def test_rosenbaum_threshold_large_for_strong_effect():
    """A strong true effect should survive large Gamma."""
    rng = np.random.default_rng(0)
    diffs = rng.normal(1.0, 1.0, size=300)
    g = rosenbaum_gamma_threshold(diffs, alpha=0.05, alternative="greater")
    assert g > 2.0


def test_rosenbaum_threshold_small_for_weak_effect():
    """A weak/null effect gives threshold close to 1.0."""
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.02, 1.0, size=50)
    g = rosenbaum_gamma_threshold(diffs, alpha=0.05, alternative="greater")
    assert g < 1.2


def test_sensitivity_pipeline_on_simulated_data():
    """End-to-end: simulate, match, run both sensitivity analyses."""
    X, T, Y, _ = simulate_observational(n=500, true_effect=1.0, seed=0)
    diffs = matched_pair_differences(X, T, Y, caliper_sd=0.2)
    assert len(diffs) > 50
    # E-value on matched ATT:
    att = float(diffs.mean())
    sd_y = float(Y[T == 0].std(ddof=1))
    ev = e_value(att, sd_outcome=sd_y)
    assert ev.e_value_point > 1.0
    # Rosenbaum threshold should be > 1.0 for a +1.0 true effect.
    g = rosenbaum_gamma_threshold(diffs, alpha=0.05, alternative="greater")
    assert g >= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
