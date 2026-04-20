"""Hero chart: propensity score overlap before and after IPW."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from psm import estimate_propensity  # noqa: E402
from psm_simulate import simulate_observational  # noqa: E402

OUT = Path(__file__).resolve().parent / "hero.png"


def main() -> None:
    X, T, Y, _ = simulate_observational(n=5000, seed=7)
    ps = estimate_propensity(X, T)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 1, 40)

    axes[0].hist(ps[T == 1], bins=bins, alpha=0.6, color="#d62728", density=True,
                 label=f"Treated (n={(T == 1).sum()})")
    axes[0].hist(ps[T == 0], bins=bins, alpha=0.6, color="#1f77b4", density=True,
                 label=f"Control (n={(T == 0).sum()})")
    axes[0].set_xlabel("Propensity score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Raw: treated have higher propensity\n(confounded comparison)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    ps_clip = np.clip(ps, 0.01, 0.99)
    w = np.where(T == 1, 1.0 / ps_clip, 1.0 / (1 - ps_clip))
    axes[1].hist(ps_clip[T == 1], bins=bins, alpha=0.6, color="#d62728",
                 weights=w[T == 1], density=True, label="Treated (IPW)")
    axes[1].hist(ps_clip[T == 0], bins=bins, alpha=0.6, color="#1f77b4",
                 weights=w[T == 0], density=True, label="Control (IPW)")
    axes[1].set_xlabel("Propensity score")
    axes[1].set_title("After IPW: distributions aligned\n(rebalanced pseudo-populations)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Case 04 — Propensity score weighting rebalances treated and control",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
