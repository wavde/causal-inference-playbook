"""Tests for the switchback case study."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-studies" / "06-switchback" / "src"))

from switch_simulate import simulate_market, true_policy_effect  # noqa: E402
from switchback import naive_ab_estimate, switchback_estimate  # noqa: E402


def test_truth_formula():
    assert true_policy_effect(0.3, 0.2) == pytest.approx(0.1)


def test_simulate_user_design_shape():
    m = simulate_market(n_blocks=10, users_per_block=20, design="user", seed=0)
    assert len(m.y) == 200
    assert set(np.unique(m.treatment).tolist()) <= {0.0, 1.0}


def test_simulate_switchback_block_level():
    m = simulate_market(n_blocks=10, users_per_block=20, design="switchback", seed=0)
    for b in np.unique(m.block_id):
        assert len(np.unique(m.treatment[m.block_id == b])) == 1


def test_switchback_estimate_rejects_user_design():
    m = simulate_market(n_blocks=10, users_per_block=20, design="user", seed=0)
    with pytest.raises(ValueError, match="block-level"):
        switchback_estimate(m.y, m.treatment, m.block_id)


def test_naive_ab_overstates_when_spillover_exists():
    """Naive A/B should overshoot the true policy effect on average."""
    tau, gamma = 0.30, 0.20
    truth = true_policy_effect(tau, gamma)
    points = []
    for seed in range(50):
        m = simulate_market(design="user", tau=tau, gamma=gamma, seed=seed)
        points.append(naive_ab_estimate(m.y, m.treatment).point)
    mean_naive = float(np.mean(points))
    assert mean_naive > truth + 0.05, (
        f"expected naive A/B to overstate truth={truth:.2f}, got mean={mean_naive:.3f}"
    )
    assert abs(mean_naive - tau) < 0.03, (
        f"naive A/B should converge to tau={tau:.2f}, got {mean_naive:.3f}"
    )


def test_switchback_is_unbiased_for_policy_effect():
    tau, gamma = 0.30, 0.20
    truth = true_policy_effect(tau, gamma)
    points = []
    for seed in range(50):
        m = simulate_market(design="switchback", tau=tau, gamma=gamma, seed=seed)
        points.append(switchback_estimate(m.y, m.treatment, m.block_id).point)
    mean_sb = float(np.mean(points))
    assert abs(mean_sb - truth) < 0.03, (
        f"expected switchback to recover truth={truth:.2f}, got mean={mean_sb:.3f}"
    )


def test_switchback_ci_coverage():
    tau, gamma = 0.30, 0.20
    truth = true_policy_effect(tau, gamma)
    hits = 0
    n_reps = 200
    for seed in range(n_reps):
        m = simulate_market(design="switchback", tau=tau, gamma=gamma, seed=seed)
        est = switchback_estimate(m.y, m.treatment, m.block_id)
        if est.ci_low <= truth <= est.ci_high:
            hits += 1
    coverage = hits / n_reps
    assert 0.90 <= coverage <= 0.99, f"expected ~95% coverage, got {coverage:.1%}"
