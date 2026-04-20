"""Tests for synthetic control implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-studies" / "02-synthetic-control" / "src"))

from sc_simulate import panel_to_matrix, simulate_panel  # noqa: E402
from synthetic_control import (  # noqa: E402
    _fit_weights,
    fit_synthetic_control,
    in_space_placebo,
    in_time_placebo,
    placebo_pvalue,
)


def test_weights_are_valid_simplex():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20)
    Y = rng.normal(size=(20, 10))
    w = _fit_weights(y, Y)
    assert np.all(w >= -1e-9)
    assert abs(w.sum() - 1.0) < 1e-6


def test_perfect_fit_when_treated_is_a_donor_copy():
    """If the treated unit equals donor 3 exactly, SC should load w=1 on donor 3."""
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(25, 5))
    y1 = Y[:, 3].copy()
    w = _fit_weights(y1[:20], Y[:20])
    assert w[3] > 0.95


def test_recovers_true_att_on_simulated_panel():
    panel = simulate_panel(
        n_units=20, n_periods=30, pre_periods=20,
        treated_idx=0, true_effect=-5.0, seed=0,
    )
    y1, Y0, donors = panel_to_matrix(panel, "country_00")
    result = fit_synthetic_control(y1, Y0, pre_periods=20, donor_names=donors)
    # True effect = -5; expect recovered within ~2 units given noise.
    assert abs(result.att - (-5.0)) < 2.0
    # Post RMSPE should be meaningfully larger than pre RMSPE.
    assert result.rmspe_ratio > 1.5


def test_zero_effect_gives_small_att():
    panel = simulate_panel(
        n_units=20, n_periods=30, pre_periods=20,
        treated_idx=0, true_effect=0.0, seed=0,
    )
    y1, Y0, donors = panel_to_matrix(panel, "country_00")
    result = fit_synthetic_control(y1, Y0, pre_periods=20, donor_names=donors)
    assert abs(result.att) < 2.0


def test_in_space_placebo_ranks_true_effect_highly():
    panel = simulate_panel(
        n_units=20, n_periods=30, pre_periods=20,
        treated_idx=0, true_effect=-8.0, seed=0,
    )
    wide = panel.pivot(index="period", columns="unit", values="y").sort_index()
    Y_all = wide.to_numpy()
    names = list(wide.columns)
    placebos = in_space_placebo(
        Y_all, pre_periods=20, treated_idx=0, unit_names=names,
    )
    p = placebo_pvalue(placebos, "country_00")
    # With a large true effect (-8), the treated unit should be among the
    # top few RMSPE ratios => small empirical p-value.
    assert p <= 0.15


def test_in_time_placebo_gives_small_att():
    """With no treatment applied, an in-time placebo should find ~0 effect."""
    panel = simulate_panel(
        n_units=20, n_periods=30, pre_periods=20,
        treated_idx=0, true_effect=0.0, seed=2,
    )
    y1, Y0, donors = panel_to_matrix(panel, "country_00")
    placebo = in_time_placebo(
        y_treated=y1, Y_donors=Y0,
        true_pre_periods=20, fake_pre_periods=15, donor_names=donors,
    )
    assert abs(placebo.att) < 2.0


def test_shape_validation():
    rng = np.random.default_rng(0)
    y = rng.normal(size=10)
    Y = rng.normal(size=(12, 5))  # mismatched T
    with pytest.raises(ValueError):
        fit_synthetic_control(y, Y, pre_periods=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
