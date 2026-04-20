"""
Run a single simulated experiment and print naive vs CUPED results.

Usage:
    python -m case_studies.01_ab_cuped.src.run
    # or from the case study directory:
    python src/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cuped import cuped_ab_test, naive_ab_test, variance_reduction  # noqa: E402
from simulate import simulate_experiment  # noqa: E402


def main() -> None:
    df = simulate_experiment(n=20_000, true_effect=3.0, correlation=0.7, seed=42)
    print(f"Sample size: {len(df):,}  (treated={df['treatment'].sum():,})")

    naive = naive_ab_test(df, outcome="minutes", treatment="treatment")
    cuped = cuped_ab_test(
        df, outcome="minutes", treatment="treatment", covariate="pre_minutes"
    )

    print()
    print(naive)
    print(cuped)

    vr = variance_reduction(df, outcome="minutes", covariate="pre_minutes")
    se_ratio = cuped.std_error / naive.std_error
    print()
    print(f"Expected variance reduction (1 - rho^2): {vr:.3f}")
    print(f"Observed SE ratio (CUPED / naive):        {se_ratio:.3f}")
    print(f"Equivalent sample-size multiplier:        {1 / se_ratio**2:.2f}x")


if __name__ == "__main__":
    main()
