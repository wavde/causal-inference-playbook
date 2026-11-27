"""Known-answer tests for case 08: staggered rollout vs fixed holdout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "case-studies" / "08-rollout-vs-holdout" / "src"
sys.path.insert(0, str(SRC))

from rollout import (  # noqa: E402
    Population,
    RolloutDesign,
    adoption_rate,
    analyze,
    simulate_panel,
    true_itt,
)

POP = Population(n=6_000)
HOLDOUT = RolloutDesign("holdout", ((0.9, 0), (0.1, None)))
STAGGERED = RolloutDesign("staggered", ((0.6, 0), (0.4, 2)))
HYBRID = RolloutDesign("hybrid", ((0.75, 0), (0.2, 2), (0.05, None)))
D = np.arange(POP.post)
NOVELTY = 0.4 * 0.6**D


def _draws(design, tau, reps, seed):
    rng = np.random.default_rng(seed)
    out: dict[str, list] = {}
    for _ in range(reps):
        for k, e in analyze(simulate_panel(POP, design, tau, rng), design, POP).items():
            out.setdefault(k, []).append((e.estimate, e.se))
    return {k: np.array(v) for k, v in out.items()}


def test_designs_withhold_the_same_share():
    for d in (HOLDOUT, STAGGERED, HYBRID):
        assert d.withheld_share(8) == pytest.approx(0.10)


def test_adoption_rate_matches_simulation():
    rng = np.random.default_rng(0)
    panel = simulate_panel(Population(n=200_000), HOLDOUT, np.zeros(8), rng)
    assert panel.adopted.mean() == pytest.approx(adoption_rate(POP), abs=0.005)


def test_true_itt_for_novelty():
    t = true_itt(POP, NOVELTY)
    p = adoption_rate(POP)
    assert t["short"] == pytest.approx(p * (0.4 + 0.24) / 2)
    assert t["long"] == pytest.approx(p * 0.4 * (0.6**6 + 0.6**7) / 2)


def test_identification_by_design():
    rng = np.random.default_rng(1)
    for design, keys in ((HOLDOUT, {"short", "long", "shape"}), (STAGGERED, {"short"}),
                         (HYBRID, {"short", "long", "shape"})):
        assert set(analyze(simulate_panel(POP, design, NOVELTY, rng), design, POP)) == keys


@pytest.mark.parametrize("design", [HOLDOUT, STAGGERED, HYBRID], ids=lambda d: d.name)
def test_estimates_unbiased_with_calibrated_se(design):
    truth = true_itt(POP, NOVELTY)
    draws = _draws(design, NOVELTY, reps=150, seed=2)
    for k, a in draws.items():
        sd = a[:, 0].std(ddof=1)
        assert a[:, 0].mean() == pytest.approx(truth[k], abs=4 * sd / np.sqrt(len(a))), k
        assert a[:, 1].mean() == pytest.approx(sd, rel=0.2), k


def test_staggered_is_more_precise_for_short_run_but_blind_to_long_run():
    stag = _draws(STAGGERED, NOVELTY, reps=60, seed=3)
    hold = _draws(HOLDOUT, NOVELTY, reps=60, seed=4)
    assert stag["short"][:, 1].mean() < 0.75 * hold["short"][:, 1].mean()
    assert "long" not in stag
    truth = true_itt(POP, NOVELTY)
    # The short-run read overstates the long-run effect of a fading feature.
    assert stag["short"][:, 0].mean() > truth["long"] + 0.05


def test_validation():
    with pytest.raises(ValueError, match="sum to 1"):
        RolloutDesign("bad", ((0.5, 0), (0.4, None)))
    with pytest.raises(ValueError, match="week 0"):
        RolloutDesign("bad", ((0.5, 1), (0.5, None)))
    with pytest.raises(ValueError):
        true_itt(POP, np.zeros(3))


def test_run_smoke():
    out = subprocess.run([sys.executable, str(SRC / "run.py"), "--smoke"],
                         capture_output=True, text=True, check=True)
    assert "wrote" in out.stdout
