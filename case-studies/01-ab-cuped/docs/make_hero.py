"""Hero chart: CUPED variance reduction in action."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from cuped_simulate import simulate_experiment  # noqa: E402
from cuped import apply_cuped  # noqa: E402

OUT = Path(__file__).resolve().parent / "hero.png"


def main() -> None:
    df = simulate_experiment(n=10000, true_effect=0.0, correlation=0.8, seed=7)
    y = df["minutes"].to_numpy()
    x = df["pre_minutes"].to_numpy()
    t = df["treatment"].to_numpy()

    y_adj = apply_cuped(y, x)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins_raw = np.linspace(y.min(), y.max(), 50)
    bins_adj = np.linspace(y_adj.min(), y_adj.max(), 50)

    axes[0].hist(y[t == 0], bins=bins_raw, alpha=0.55, color="#1f77b4", label="Control")
    axes[0].hist(y[t == 1], bins=bins_raw, alpha=0.55, color="#d62728", label="Treatment")
    axes[0].set_title(f"Raw outcome (minutes watched)\nstd = {y.std():.1f}")
    axes[0].set_xlabel("Minutes")
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    var_reduction = (1 - y_adj.var() / y.var()) * 100
    axes[1].hist(y_adj[t == 0], bins=bins_adj, alpha=0.55, color="#1f77b4", label="Control")
    axes[1].hist(y_adj[t == 1], bins=bins_adj, alpha=0.55, color="#d62728", label="Treatment")
    axes[1].set_title(
        f"CUPED-adjusted (ρ=0.8)\nstd = {y_adj.std():.1f}  —  "
        f"{var_reduction:.0f}% variance reduction"
    )
    axes[1].set_xlabel("Y - θ·(X - mean X)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Case 01 — CUPED soaks up pre-period signal, shrinking variance by ~60%",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
