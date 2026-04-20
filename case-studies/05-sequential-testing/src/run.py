"""
Run the sequential-testing case study end-to-end.

Demonstrates:
  1. Naive peeking inflates Type-I error to ~30% with frequent looks.
  2. mSPRT controls Type-I at the nominal alpha at *any* stopping time.
  3. Under H1, mSPRT stops faster on average than the fixed-horizon test.

Usage:
    cd case-studies/05-sequential-testing
    python src/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from sequential import (  # noqa: E402
    fixed_horizon_zscore,
    msprt_sequential_test,
    naive_peeking_test,
    pocock_sequential_test,
)
from seq_simulate import simulate_stream  # noqa: E402


def empirical_rejection_rate(
    test_fn,
    n_sims: int = 500,
    n_per_arm: int = 3000,
    true_effect: float = 0.0,
    seed_offset: int = 0,
    **kwargs,
) -> tuple[float, float]:
    """Return (rejection_rate, mean_stop_n)."""
    rejected = 0
    stop_ns = []
    for s in range(n_sims):
        y_t, y_c = simulate_stream(
            n_per_arm=n_per_arm, true_effect=true_effect, seed=seed_offset + s
        )
        result = test_fn(y_t, y_c, **kwargs)
        if result.rejected:
            rejected += 1
        stop_ns.append(result.stopped_at)
    return rejected / n_sims, float(np.mean(stop_ns))


def main() -> None:
    n_sims = 500
    n_per_arm = 3000
    alpha = 0.05

    print(f"Setup: {n_sims} simulated experiments, n_per_arm={n_per_arm}, alpha={alpha}\n")

    print("=== Type-I error under H0 (true_effect = 0) ===")
    print("                                  rej_rate   mean_stop_n")

    # Fixed horizon (look exactly once at the end)
    fh_rejs = 0
    for s in range(n_sims):
        y_t, y_c = simulate_stream(n_per_arm=n_per_arm, true_effect=0.0, seed=s)
        _, p, _ = fixed_horizon_zscore(y_t, y_c)
        if p < alpha:
            fh_rejs += 1
    print(f"  Fixed horizon (1 look)         {fh_rejs/n_sims:>8.3f}    {n_per_arm}")

    rate, mean_n = empirical_rejection_rate(
        naive_peeking_test, n_sims=n_sims, n_per_arm=n_per_arm,
        true_effect=0.0, alpha=alpha, look_every=100,
    )
    print(f"  Naive peeking (every 100 obs)  {rate:>8.3f}    {mean_n:.0f}    <-- INFLATED")

    rate, mean_n = empirical_rejection_rate(
        msprt_sequential_test, n_sims=n_sims, n_per_arm=n_per_arm,
        true_effect=0.0, alpha=alpha, look_every=100,
    )
    print(f"  mSPRT                          {rate:>8.3f}    {mean_n:.0f}    <-- controlled")

    rate, mean_n = empirical_rejection_rate(
        pocock_sequential_test, n_sims=n_sims, n_per_arm=n_per_arm,
        true_effect=0.0, alpha=alpha, K=5,
    )
    print(f"  Pocock (K=5 looks)             {rate:>8.3f}    {mean_n:.0f}    <-- controlled")

    print("\n=== Power & expected stop time under H1 (true_effect = 0.1 sigma) ===")
    print("                                  rej_rate   mean_stop_n")

    fh_rejs = 0
    for s in range(n_sims):
        y_t, y_c = simulate_stream(n_per_arm=n_per_arm, true_effect=0.1, seed=s + 1000)
        _, p, _ = fixed_horizon_zscore(y_t, y_c)
        if p < alpha:
            fh_rejs += 1
    print(f"  Fixed horizon (1 look)         {fh_rejs/n_sims:>8.3f}    {n_per_arm}")

    rate, mean_n = empirical_rejection_rate(
        msprt_sequential_test, n_sims=n_sims, n_per_arm=n_per_arm,
        true_effect=0.1, alpha=alpha, look_every=100, seed_offset=1000,
    )
    print(f"  mSPRT                          {rate:>8.3f}    {mean_n:.0f}")

    rate, mean_n = empirical_rejection_rate(
        pocock_sequential_test, n_sims=n_sims, n_per_arm=n_per_arm,
        true_effect=0.1, alpha=alpha, K=5, seed_offset=1000,
    )
    print(f"  Pocock (K=5)                   {rate:>8.3f}    {mean_n:.0f}")


if __name__ == "__main__":
    main()
