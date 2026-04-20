"""Generate the hero chart shown in README.md.

A 2x2 panel summarizing the four methodological pillars of the playbook:
  (1) CUPED variance reduction — power gain at fixed sample size.
  (2) Synthetic control — treated unit vs synthetic counterfactual.
  (3) Difference-in-differences — event-study coefficients with 95% CIs.
  (4) Propensity score matching — covariate overlap before vs after weighting.

All data are simulated with a fixed seed so the chart is deterministic.

Run:  python scripts/make_hero.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RNG = np.random.default_rng(7)
OUT = Path(__file__).resolve().parents[1] / "docs" / "hero.png"


def panel_cuped(ax: plt.Axes) -> None:
    n_sims = 500
    n_per_arm = 1500
    rho = 0.7
    effect = 0.04

    def reject_rate(use_cuped: bool) -> float:
        reject = 0
        for _ in range(n_sims):
            x_c = RNG.normal(size=n_per_arm)
            y_c = rho * x_c + np.sqrt(1 - rho**2) * RNG.normal(size=n_per_arm)
            x_t = RNG.normal(size=n_per_arm)
            y_t = rho * x_t + np.sqrt(1 - rho**2) * RNG.normal(size=n_per_arm) + effect
            if use_cuped:
                x_all = np.concatenate([x_c, x_t])
                y_all = np.concatenate([y_c, y_t])
                theta = np.cov(x_all, y_all)[0, 1] / np.var(x_all)
                y_c = y_c - theta * (x_c - x_all.mean())
                y_t = y_t - theta * (x_t - x_all.mean())
            tstat, _ = stats.ttest_ind(y_t, y_c)
            if abs(tstat) > 1.96:
                reject += 1
        return reject / n_sims

    naive = reject_rate(False)
    cuped = reject_rate(True)
    ax.bar(["t-test", "CUPED"], [naive * 100, cuped * 100], color=["#c7c7c7", "#1f77b4"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Power (%)")
    ax.set_title(f"CUPED variance reduction\n+{(cuped - naive) * 100:.0f}pp power at fixed N")
    for i, v in enumerate([naive, cuped]):
        ax.text(i, v * 100 + 2, f"{v * 100:.0f}%", ha="center", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)


def panel_synth(ax: plt.Axes) -> None:
    T = 40
    t_treat = 25
    donors = RNG.normal(0, 1, size=(5, T)).cumsum(axis=1) * 0.3
    weights = np.array([0.4, 0.25, 0.15, 0.15, 0.05])
    counterfactual = weights @ donors
    effect = np.where(np.arange(T) >= t_treat,
                      -0.8 * (np.arange(T) - t_treat + 1) / 10, 0)
    treated = counterfactual + effect + RNG.normal(0, 0.1, size=T)

    ts = np.arange(T)
    ax.plot(ts, treated, color="#d62728", lw=2, label="Treated unit")
    ax.plot(ts, counterfactual, color="#1f77b4", lw=2, ls="--", label="Synthetic control")
    ax.axvline(t_treat, color="black", lw=1, alpha=0.5)
    ax.fill_between(ts, treated, counterfactual, where=ts >= t_treat,
                    alpha=0.2, color="#d62728")
    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome")
    ax.set_title("Synthetic control\nTreated vs synthetic counterfactual")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)


def panel_did(ax: plt.Axes) -> None:
    event_times = np.arange(-5, 6)
    coefs = np.where(
        event_times < 0,
        RNG.normal(0, 0.05, size=len(event_times)),
        0.3 * (event_times + 1) / (event_times + 2) + RNG.normal(0, 0.04, size=len(event_times)),
    )
    coefs[event_times == -1] = 0.0
    se = np.full_like(coefs, 0.08)
    ax.errorbar(event_times, coefs, yerr=1.96 * se, fmt="o", color="#2ca02c",
                capsize=4, lw=1.5, markersize=6)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Event time (periods from treatment)")
    ax.set_ylabel("Coefficient (95% CI)")
    ax.set_title("Event-study DiD\nParallel pre-trends, post-treatment effect")
    ax.grid(True, alpha=0.3)


def panel_psm(ax: plt.Axes) -> None:
    n = 2000
    x = RNG.normal(size=n)
    logit = -0.8 + 1.5 * x
    p = 1 / (1 + np.exp(-logit))
    t = RNG.binomial(1, p)
    treated_x = x[t == 1]
    control_x = x[t == 0]
    bins = np.linspace(-4, 4, 40)
    ax.hist(treated_x, bins=bins, alpha=0.6, color="#d62728",
            label="Treated", density=True)
    ax.hist(control_x, bins=bins, alpha=0.6, color="#1f77b4",
            label="Control", density=True)
    w_control = 1 / (1 - p[t == 0])
    weighted_density, _ = np.histogram(control_x, bins=bins,
                                       weights=w_control, density=True)
    centers = 0.5 * (bins[1:] + bins[:-1])
    ax.plot(centers, weighted_density, color="black", lw=2, label="Control (IPW)")
    ax.set_xlabel("Covariate X")
    ax.set_ylabel("Density")
    ax.set_title("Propensity score weighting\nOverlap before vs after IPW")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panel_cuped(axes[0, 0])
    panel_synth(axes[0, 1])
    panel_did(axes[1, 0])
    panel_psm(axes[1, 1])
    fig.suptitle("causal-inference-playbook — four methods, four case studies",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
