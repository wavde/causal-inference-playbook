"""Known-answer tests for the opt-in AI feature adoption case (case 07)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "case-studies" / "07-ai-feature-adoption" / "src"))

from adoption_estimators import (  # noqa: E402
    aipw_att_se,
    did_first_diff,
    estimate_with_placebo,
)
from adoption_mc import run_scenario, summarize  # noqa: E402
from adoption_simulate import SCENARIOS, population_truth, simulate  # noqa: E402
from did import did_2x2  # noqa: E402
from psm import aipw_att  # noqa: E402

N_REPS = 40
N_USERS = 4000


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    names = ["observables", "fixed_trait", "pre_trend", "post_shock"]
    draws = pd.concat([run_scenario(n, N_REPS, N_USERS, seed=0) for n in names])
    return summarize(draws).set_index(["scenario", "estimator"])


def test_aipw_point_matches_case_04_implementation():
    d = simulate(SCENARIOS["fixed_trait"], 2000, np.random.default_rng(1))
    X = np.column_stack([d.X, d.y_pre1, d.y_pre2])
    assert aipw_att_se(X, d.T, d.y_post).est == pytest.approx(aipw_att(X, d.T, d.y_post), abs=1e-10)


def test_first_difference_matches_case_03_did_2x2():
    d = simulate(SCENARIOS["pre_trend"], 400, np.random.default_rng(2))
    twfe = did_2x2(d.to_long("y_pre2", "y_post"))
    assert did_first_diff(d.y_pre2, d.y_post, d.T).est == pytest.approx(twfe.att, abs=1e-8)


def test_truth_att_exceeds_ate_when_heavy_users_adopt_and_gain_more():
    truth = population_truth(SCENARIOS["observables"])
    assert truth["ate"] == 1.0
    assert truth["att"] > 1.2
    assert population_truth(SCENARIOS["null_post_shock"])["att"] == 0.0


def test_run_scenario_is_deterministic():
    a = run_scenario("pre_trend", 2, 500, seed=0)
    b = run_scenario("pre_trend", 2, 500, seed=0)
    pd.testing.assert_frame_equal(a, b)


def test_observables_adjusted_estimators_unbiased_naive_biased(summary):
    for est in ["aipw_x", "aipw_x_pre", "did"]:
        assert abs(summary.loc[("observables", est), "bias"]) < 0.06
        assert summary.loc[("observables", est), "coverage"] >= 0.85
        assert summary.loc[("observables", est), "placebo_flag"] <= 0.20
    assert summary.loc[("observables", "naive"), "bias"] > 1.0


def test_fixed_trait_did_unbiased_and_placebo_catches_aipw(summary):
    assert abs(summary.loc[("fixed_trait", "did"), "bias"]) < 0.06
    assert summary.loc[("fixed_trait", "did"), "placebo_flag"] <= 0.20
    # Pre-period outcomes only partly proxy the trait, and the placebo sees the residual.
    assert summary.loc[("fixed_trait", "aipw_x_pre"), "bias"] > 0.3
    assert summary.loc[("fixed_trait", "aipw_x_pre"), "placebo_flag"] >= 0.9


def test_pre_trend_biases_did_upward_and_placebo_flags_it(summary):
    for est in ["aipw_x_pre", "did"]:
        assert summary.loc[("pre_trend", est), "bias"] > 0.2
        assert summary.loc[("pre_trend", est), "placebo_flag"] >= 0.8


def test_post_shock_bias_is_invisible_to_the_placebo(summary):
    for est in ["aipw_x_pre", "did"]:
        assert summary.loc[("post_shock", est), "bias"] > 0.4
        assert summary.loc[("post_shock", est), "placebo_flag"] <= 0.20


def test_placebo_estimates_are_centred_on_zero_without_pre_period_selection():
    d = simulate(SCENARIOS["post_shock"], 20000, np.random.default_rng(3))
    _, placebo = estimate_with_placebo("did", d)
    assert abs(placebo.z) < 3.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
