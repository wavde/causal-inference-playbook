"""Tests for the CUPED power helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-studies" / "01-ab-cuped" / "src"))

from power import (  # noqa: E402
    cuped_power_summary,
    mde_at_n,
    mde_vs_rho,
    required_n_per_arm,
)


def test_required_n_at_rho_zero_matches_classical_formula():
    """At rho=0, our helper must match the standard two-sample z formula."""
    # alpha=0.05, power=0.8 -> z_a=1.96, z_p=0.84, (z_a+z_p)^2 ~ 7.84.
    # n ~ 2 * 7.84 * 1 / 0.1^2 = 1568.
    n = required_n_per_arm(mde=0.1, sigma=1.0, rho=0.0)
    assert 1560 <= n <= 1580


def test_required_n_scales_by_one_minus_rho_squared():
    """CUPED: required n should scale by (1 - rho^2)."""
    n_naive = required_n_per_arm(mde=0.1, sigma=1.0, rho=0.0)
    n_cuped = required_n_per_arm(mde=0.1, sigma=1.0, rho=0.7)
    ratio = n_cuped / n_naive
    expected = 1 - 0.7 ** 2
    # Allow slack for integer ceiling.
    assert abs(ratio - expected) < 0.02


def test_mde_at_n_is_inverse_of_required_n():
    """mde_at_n(required_n_per_arm(mde, ...)) should approximately recover mde."""
    mde = 0.1
    n = required_n_per_arm(mde=mde, sigma=1.0, rho=0.5)
    mde_back = mde_at_n(n_per_arm=n, sigma=1.0, rho=0.5)
    assert abs(mde_back - mde) < 0.002


def test_cuped_power_summary_percent_reduction():
    s = cuped_power_summary(mde=0.1, sigma=1.0, rho=0.6)
    # rho=0.6 -> variance factor 1 - 0.36 = 0.64, so ~36% sample-size cut
    assert abs(s["percent_reduction"] - 0.36) < 0.02
    assert s["n_cuped_per_arm"] < s["n_naive_per_arm"]


def test_mde_vs_rho_is_monotonically_decreasing():
    out = mde_vs_rho(n_per_arm=1000, sigma=1.0,
                     rhos=np.linspace(0, 0.9, 10))
    mdes = out[:, 1]
    assert np.all(np.diff(mdes) <= 1e-9)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        required_n_per_arm(mde=0.0, sigma=1.0)
    with pytest.raises(ValueError):
        required_n_per_arm(mde=0.1, sigma=-1.0)
    with pytest.raises(ValueError):
        required_n_per_arm(mde=0.1, sigma=1.0, rho=1.5)
    with pytest.raises(ValueError):
        mde_at_n(n_per_arm=-5, sigma=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
