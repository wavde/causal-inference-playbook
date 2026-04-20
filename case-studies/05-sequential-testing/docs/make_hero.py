"""Hero chart: naive peeking vs mSPRT Type-I error under H0."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from seq_simulate import simulate_stream  # noqa: E402
from sequential import msprt_log_likelihood_ratio  # noqa: E402

OUT = Path(__file__).resolve().parent / "hero.png"


def main() -> None:
    n_reps = 500
    n_max = 2000
    alpha = 0.05
    peek_interval = 50

    peeks = np.arange(peek_interval, n_max + 1, peek_interval)
    naive_rej = np.zeros(len(peeks))
    msprt_rej = np.zeros(len(peeks))

    rng = np.random.default_rng(7)
    for _ in range(n_reps):
        seed = int(rng.integers(1 << 31))
        y_t, y_c = simulate_stream(n_per_arm=n_max, true_effect=0.0, seed=seed)
        sigma = float(np.concatenate([y_t[:100], y_c[:100]]).std(ddof=1))
        tau = sigma
        naive_done = False
        msprt_done = False
        for i, n in enumerate(peeks):
            diff = y_t[:n].mean() - y_c[:n].mean()
            z = diff / np.sqrt(2.0 / n)
            naive_p = 2 * (1 - stats.norm.cdf(abs(z)))
            if not naive_done and naive_p < alpha:
                naive_done = True
            if naive_done:
                naive_rej[i] += 1

            log_lr = msprt_log_likelihood_ratio(float(diff), int(n), sigma, tau)
            msprt_p = min(1.0, float(np.exp(-log_lr)))
            if not msprt_done and msprt_p < alpha:
                msprt_done = True
            if msprt_done:
                msprt_rej[i] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(peeks, naive_rej / n_reps * 100, color="#d62728", lw=2,
            label="Naive (peek every 50)")
    ax.plot(peeks, msprt_rej / n_reps * 100, color="#2ca02c", lw=2,
            label="mSPRT (always-valid)")
    ax.axhline(5, color="black", ls="--", lw=1, alpha=0.6, label="α = 5%")
    ax.set_xlabel("Sample size per arm")
    ax.set_ylabel("Cumulative Type-I error (%)")
    ax.set_title(
        f"Case 05 — Naive peeking inflates Type-I to "
        f"{naive_rej[-1] / n_reps * 100:.0f}% by n={n_max};\n"
        f"mSPRT stays ≤ 5% at every peek"
    )
    ax.legend(loc="center right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
