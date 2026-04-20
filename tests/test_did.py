"""Tests for DiD implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-studies" / "03-diff-in-diff" / "src"))

from did import cs_staggered_att, did_2x2, event_study  # noqa: E402
from did_simulate import simulate_did_panel, simulate_staggered_panel  # noqa: E402


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


def test_cs_recovers_cohort_att_under_heterogeneous_effects():
    """
    Callaway-Sant'Anna group-time ATTs should each recover the true cohort
    effect, even when effects differ sharply across cohorts.
    """
    panel = simulate_staggered_panel(
        cohort_sizes={6: 30, 10: 30, 14: 30},
        n_never_treated=30,
        n_periods=20,
        cohort_effects={6: 1.0, 10: 3.0, 14: 5.0},
        seed=0,
    )
    result = cs_staggered_att(panel, n_bootstrap=100, seed=0)
    gt = result.group_time_att
    for g, true_eff in [(6, 1.0), (10, 3.0), (14, 5.0)]:
        cohort_means = gt[gt["cohort"] == g]["att"].mean()
        assert abs(cohort_means - true_eff) < 0.7, (
            f"CS ATT for cohort {g} = {cohort_means:.2f}, expected ~{true_eff}"
        )


def test_twfe_is_biased_under_heterogeneous_staggered_effects():
    """
    Under staggered adoption with *dynamic, heterogeneous* effects, naive TWFE
    suffers from negative-weight / forbidden-comparison bias
    (Goodman-Bacon 2021). CS is unbiased in the same setting. Here we add a
    strong dynamic slope so the bias is clearly visible.
    """
    panel = simulate_staggered_panel(
        cohort_sizes={4: 30, 8: 30, 12: 30},
        n_never_treated=30,
        n_periods=20,
        cohort_effects={4: 1.0, 8: 3.0, 12: 5.0},
        dynamic_slope=0.5,
        seed=0,
    )
    twfe = did_2x2(panel)
    cs = cs_staggered_att(panel, n_bootstrap=50, seed=0)
    # CS remains close to the true equal-weighted ATT. TWFE is meaningfully
    # further away because already-treated units contaminate the comparison.
    assert abs(twfe.att - cs.overall_att) > 0.5


def test_cs_event_study_shape_and_no_pre_cells():
    panel = simulate_staggered_panel(seed=0)
    result = cs_staggered_att(panel, n_bootstrap=50, seed=0)
    es = result.event_study
    # All relative times in the event study should be >= 0 since CS
    # identifies ATT(g, t) only for t >= g.
    assert (es["relative_time"] >= 0).all()
    assert not es["att"].isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])