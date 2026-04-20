"""Hero chart: switchback recovers the true policy effect; naive A/B doesn't."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from switch_simulate import simulate_market, true_policy_effect  # noqa: E402
from switchback import naive_ab_estimate, switchback_estimate  # noqa: E402

OUT = Path(__file__).resolve().parent / "hero.png"


def main() -> None:
    tau, gamma = 0.30, 0.20
    truth = true_policy_effect(tau, gamma)
    n_reps = 200

    naive_points, naive_los, naive_his = [], [], []
    sb_points, sb_los, sb_his = [], [], []
    for seed in range(n_reps):
        mu = simulate_market(design="user", tau=tau, gamma=gamma, seed=seed)
        ms = simulate_market(design="switchback", tau=tau, gamma=gamma, seed=seed)
        n = naive_ab_estimate(mu.y, mu.treatment)
        s = switchback_estimate(ms.y, ms.treatment, ms.block_id)
        naive_points.append(n.point)
        naive_los.append(n.ci_low)
        naive_his.append(n.ci_high)
        sb_points.append(s.point)
        sb_los.append(s.ci_low)
        sb_his.append(s.ci_high)

    order = np.argsort(naive_points)
    idx = np.arange(n_reps)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, points, los, his, title, color in [
        (axes[0], naive_points, naive_los, naive_his, "Naive A/B (user-level)", "#d62728"),
        (axes[1], sb_points, sb_los, sb_his, "Switchback (block-level)", "#2ca02c"),
    ]:
        p = np.array(points)[order]
        lo = np.array(los)[order]
        hi = np.array(his)[order]
        ax.errorbar(idx, p, yerr=[p - lo, hi - p], fmt="o", color=color,
                    markersize=3, lw=0.7, alpha=0.6, capsize=0)
        ax.axhline(truth, color="black", lw=1.5, ls="--",
                   label=f"Truth = {truth:.2f}")
        ax.set_title(f"{title}\nmean estimate = {np.mean(points):+.3f}")
        ax.set_xlabel("Replication (sorted)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)

    axes[0].set_ylabel("Estimated effect (95% CI)")
    fig.suptitle(
        "Case 06 — 200 replications: naive A/B is confidently wrong, "
        "switchback is honest",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
