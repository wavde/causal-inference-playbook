"""Estimators for marketplace experiments with interference.

Two estimators, same data:

- ``naive_ab_estimate`` — compare treated vs control users directly. Biased
  upward when there is positive spillover (gamma > 0), because the control
  group in each block is absorbing the negative externality of the treated
  users it shares a block with.
- ``switchback_estimate`` — aggregate to block means, then compare all-treated
  blocks vs all-control blocks. Consistent for the true policy effect
  ``tau - gamma`` under block-level randomization. SEs are computed at the
  block level (one observation per block) so they honor the randomization unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Estimate:
    point: float
    se: float
    ci_low: float
    ci_high: float

    def __repr__(self) -> str:
        return (
            f"Estimate(point={self.point:+.4f}, se={self.se:.4f}, "
            f"ci=[{self.ci_low:+.4f}, {self.ci_high:+.4f}])"
        )


def _welch(y1: np.ndarray, y0: np.ndarray) -> Estimate:
    m1, m0 = y1.mean(), y0.mean()
    v1, v0 = y1.var(ddof=1), y0.var(ddof=1)
    se = np.sqrt(v1 / len(y1) + v0 / len(y0))
    point = m1 - m0
    df = (v1 / len(y1) + v0 / len(y0)) ** 2 / (
        (v1 / len(y1)) ** 2 / (len(y1) - 1) + (v0 / len(y0)) ** 2 / (len(y0) - 1)
    )
    crit = stats.t.ppf(0.975, df)
    return Estimate(point=point, se=se, ci_low=point - crit * se, ci_high=point + crit * se)


def naive_ab_estimate(y: np.ndarray, treatment: np.ndarray) -> Estimate:
    """User-level Welch test. Biased under SUTVA violation."""
    y1 = y[treatment == 1]
    y0 = y[treatment == 0]
    if len(y1) < 2 or len(y0) < 2:
        raise ValueError("need at least 2 users in each arm")
    return _welch(y1, y0)


def switchback_estimate(
    y: np.ndarray, treatment: np.ndarray, block_id: np.ndarray
) -> Estimate:
    """Block-level Welch test.

    Assumes block-level assignment (``design='switchback'`` in the simulator):
    every user in a block shares the same treatment. Collapses to one row per
    block so that SEs reflect the true randomization unit.
    """
    blocks = np.unique(block_id)
    block_means = np.array([y[block_id == b].mean() for b in blocks])
    block_t = np.array([treatment[block_id == b][0] for b in blocks])
    if not np.all(
        [np.all(treatment[block_id == b] == block_t[i]) for i, b in enumerate(blocks)]
    ):
        raise ValueError(
            "switchback_estimate requires block-level assignment; "
            "got mixed treatment within at least one block"
        )
    y1 = block_means[block_t == 1]
    y0 = block_means[block_t == 0]
    if len(y1) < 2 or len(y0) < 2:
        raise ValueError("need at least 2 blocks in each arm")
    return _welch(y1, y0)
