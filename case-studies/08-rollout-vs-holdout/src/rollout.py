"""Staggered rollout vs fixed holdout for an opt-in AI feature.

Case 07 concluded that an observational adopter comparison is rarely usable
and that a randomized holdout or staggered rollout should be the default. This
module asks which of those to run when withheld traffic is the budget.

Model
-----
``n`` users, ``pre`` pre-launch weeks and ``post`` weeks after launch. Users
are randomized to *access groups*. Each group gets access to the feature at a
given week after launch, or never within the window (a holdout). With access,
user ``i`` opts in with probability ``sigmoid(adopt_logit + adopt_slope * a_i / user_sd)``,
so heavier users adopt more (the case 07 selection). Adopters gain ``tau(d)``
in week ``d`` after their access starts:

    y_it = a_i + g_t + A_i * 1[access] * tau(d_it) + e_it

Here ``a_i ~ N(0, user_sd^2)``, ``e_it ~ N(0, 1)`` and ``g_t`` is a common
weekly shock. Access is randomized, so intent-to-treat effects of access are
identified without modeling adoption. The true ITT at duration ``d`` is
``E[A] * tau(d)``.

Estimands
---------
- **Short-run ITT**: the average over weeks 0 to ``short_weeks - 1`` after
  access.
- **Long-run ITT**: the average over the last ``long_weeks`` weeks of the
  window.

Estimators
----------
Every contrast compares the mean outcome over a window of weeks between a
group that has had access since launch and a group that had no access during
the window. It is CUPED-adjusted with each user's pre-launch mean (Deng et al.
2013) and uses a Wald CI.

The **shape test** is a difference-in-differences between those two groups:
(late window minus early window) for treated users minus the same for
holdout users. It estimates long-run minus short-run ITT, and user effects
cancel.

The cost of a design is the share of user-weeks without access during the
window. Designs are compared at equal cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy import integrate, stats

Z_95 = float(stats.norm.ppf(0.975))


@dataclass(frozen=True)
class RolloutDesign:
    """Access groups as ``(share, access_week)`` pairs.

    ``access_week`` is the number of weeks after launch at which the group
    gets access, or ``None`` for no access within the window.
    """

    name: str
    groups: tuple[tuple[float, int | None], ...]

    def __post_init__(self) -> None:
        shares = [s for s, _ in self.groups]
        if any(s <= 0 for s in shares) or abs(sum(shares) - 1.0) > 1e-9:
            raise ValueError("group shares must be positive and sum to 1")
        weeks = [w for _, w in self.groups]
        if 0 not in weeks:
            raise ValueError("one group must get access at launch (week 0)")
        if any(w is not None and w < 0 for w in weeks):
            raise ValueError("access weeks must be non-negative")

    def withheld_share(self, post: int) -> float:
        """Share of post-launch user-weeks without access."""
        return float(sum(s * (post if w is None else min(w, post)) for s, w in self.groups) / post)


@dataclass(frozen=True)
class Population:
    n: int = 10_000
    pre: int = 4
    post: int = 8
    user_sd: float = 1.5
    adopt_logit: float = -0.9
    adopt_slope: float = 0.5
    short_weeks: int = 2
    long_weeks: int = 2

    def __post_init__(self) -> None:
        if self.short_weeks + self.long_weeks > self.post:
            raise ValueError("short and long windows must fit inside the post period")

    @property
    def long_start(self) -> int:
        return self.post - self.long_weeks


def _p_adopt(pop: Population, a: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-(pop.adopt_logit + pop.adopt_slope * a / pop.user_sd)))


@cache
def adoption_rate(pop: Population) -> float:
    """E[P(adopt)] over the user-effect distribution (numerical integration)."""
    lim = 10 * pop.user_sd
    value, _ = integrate.quad(
        lambda a: _p_adopt(pop, a) * stats.norm.pdf(a, scale=pop.user_sd), -lim, lim
    )
    return float(value)


def true_itt(pop: Population, tau: np.ndarray) -> dict[str, float]:
    """True short-run and long-run ITT of access since launch."""
    tau = np.asarray(tau, dtype=float)
    if len(tau) != pop.post:
        raise ValueError("tau must have one value per post-launch week")
    p = adoption_rate(pop)
    short = p * float(tau[: pop.short_weeks].mean())
    long = p * float(tau[pop.long_start :].mean())
    return {"short": short, "long": long, "shape": long - short, "adoption": p}


@dataclass(frozen=True)
class Panel:
    y_pre: np.ndarray
    y_post: np.ndarray
    access_week: np.ndarray
    adopted: np.ndarray


def simulate_panel(
    pop: Population, design: RolloutDesign, tau: np.ndarray, rng: np.random.Generator
) -> Panel:
    """Simulate one rollout. ``access_week`` uses -1 for "never in window"."""
    tau = np.asarray(tau, dtype=float)
    if len(tau) != pop.post:
        raise ValueError("tau must have one value per post-launch week")
    shares = np.array([s for s, _ in design.groups])
    weeks = np.array([-1 if w is None else w for _, w in design.groups])
    access = weeks[rng.choice(len(shares), size=pop.n, p=shares)]
    a = rng.normal(0.0, pop.user_sd, pop.n)
    adopted = rng.random(pop.n) < _p_adopt(pop, a)
    g = rng.normal(0.0, 0.5, pop.pre + pop.post)
    y_pre = a[:, None] + g[None, : pop.pre] + rng.standard_normal((pop.n, pop.pre))
    t = np.arange(pop.post)[None, :]
    has = (access[:, None] >= 0) & (t >= access[:, None])
    d = np.clip(t - access[:, None], 0, pop.post - 1)
    lift = np.where(has & adopted[:, None], tau[d], 0.0)
    y_post = a[:, None] + g[None, pop.pre :] + lift + rng.standard_normal((pop.n, pop.post))
    return Panel(y_pre, y_post, access, adopted)


@dataclass(frozen=True)
class Estimate:
    estimate: float
    se: float

    def covers(self, truth: float) -> bool:
        return abs(self.estimate - truth) <= Z_95 * self.se

    def significant(self) -> bool:
        return abs(self.estimate) > Z_95 * self.se


def _diff(y: np.ndarray, treated: np.ndarray, control: np.ndarray) -> Estimate:
    if treated.sum() < 2 or control.sum() < 2:
        raise ValueError("each side of the contrast needs at least 2 users")
    yt, yc = y[treated], y[control]
    est = float(yt.mean() - yc.mean())
    se = float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))
    return Estimate(est, se)


def window_contrast(
    panel: Panel, weeks: slice, control_from_week: int | None, cuped: bool = True
) -> Estimate:
    """ITT of access-since-launch over ``weeks``.

    The control group is users without access at any point in the window.
    These are the holdout users (``control_from_week=None``) or every user
    whose access starts at or after ``control_from_week``.
    """
    treated = panel.access_week == 0
    if control_from_week is None:
        control = panel.access_week == -1
    else:
        if control_from_week < weeks.stop:
            raise ValueError("control users must lack access for the whole window")
        control = (panel.access_week == -1) | (panel.access_week >= control_from_week)
    y = panel.y_post[:, weeks].mean(axis=1)
    if cuped:
        x = panel.y_pre.mean(axis=1)
        theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
        y = y - theta * (x - x.mean())
    return _diff(y, treated, control)


def shape_test(panel: Panel, pop: Population) -> Estimate:
    """Long-run minus short-run ITT. Differences each user's late and early
    windows, then compares access-since-launch users with holdout users."""
    late = panel.y_post[:, pop.long_start :].mean(axis=1)
    early = panel.y_post[:, : pop.short_weeks].mean(axis=1)
    return _diff(late - early, panel.access_week == 0, panel.access_week == -1)


def analyze(panel: Panel, design: RolloutDesign, pop: Population) -> dict[str, Estimate]:
    """All estimands the design identifies.

    - ``short``: short-run ITT, using every user without access in the short
      window as control.
    - ``long``: long-run ITT, only if the design has a holdout.
    - ``shape``: long minus short, only if the design has a holdout.
    """
    out: dict[str, Estimate] = {}
    has_holdout = any(w is None for _, w in design.groups)
    delayed = [w for _, w in design.groups if w is not None and w > 0]
    if delayed and min(delayed) >= pop.short_weeks:
        out["short"] = window_contrast(panel, slice(0, pop.short_weeks), pop.short_weeks)
    elif has_holdout:
        out["short"] = window_contrast(panel, slice(0, pop.short_weeks), None)
    if has_holdout:
        out["long"] = window_contrast(panel, slice(pop.long_start, pop.post), None)
        out["shape"] = shape_test(panel, pop)
    return out
