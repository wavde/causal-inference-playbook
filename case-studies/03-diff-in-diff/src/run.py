"""
Run the DiD case study end-to-end.

Usage:
    cd case-studies/03-diff-in-diff
    python src/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from did import did_2x2, event_study  # noqa: E402
from did_simulate import simulate_did_panel  # noqa: E402


def main() -> None:
    panel = simulate_did_panel(
        n_units=30,
        n_treated=10,
        n_periods=20,
        treatment_period=12,
        true_effect=2.0,
        seed=0,
    )
    print(
        f"Panel: {panel['unit'].nunique()} units x "
        f"{panel['period'].nunique()} periods  "
        f"(treated={panel['treated_unit'].sum() // panel['period'].nunique()})\n"
    )

    print("--- 2x2 DiD ---")
    result = did_2x2(panel)
    print(result)

    print("\n--- Event study ---")
    es = event_study(panel, treatment_period=12)
    print(es)
    print()
    print(es.coefs.to_string(index=False))

    # Show what happens when parallel trends are violated.
    print("\n--- Same model, with parallel trends VIOLATED (diverging trends) ---")
    bad_panel = simulate_did_panel(
        n_units=30, n_treated=10, n_periods=20, treatment_period=12,
        true_effect=2.0, parallel_trend_violation=0.3, seed=0,
    )
    bad = did_2x2(bad_panel)
    bad_es = event_study(bad_panel, treatment_period=12)
    print(bad)
    print(f"Parallel-trends F-test p-value: {bad_es.parallel_trends_pvalue:.4f}")
    print("(Lower p-value => leads are jointly nonzero => PT likely violated.)")


if __name__ == "__main__":
    main()
