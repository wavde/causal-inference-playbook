"""
Run the synthetic control case study end-to-end.

Usage:
    cd case-studies/02-synthetic-control
    python src/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simulate import panel_to_matrix, simulate_panel  # noqa: E402
from synthetic_control import (  # noqa: E402
    fit_synthetic_control,
    in_space_placebo,
    in_time_placebo,
    placebo_pvalue,
)


def main() -> None:
    pre = 20
    treated_idx = 0
    true_effect = -5.0

    panel = simulate_panel(
        n_units=20,
        n_periods=30,
        pre_periods=pre,
        treated_idx=treated_idx,
        true_effect=true_effect,
        seed=0,
    )
    treated_name = f"country_{treated_idx:02d}"
    y1, Y0, donors = panel_to_matrix(panel, treated_name=treated_name)

    print(
        f"Panel: {panel['unit'].nunique()} units x "
        f"{panel['period'].nunique()} periods "
        f"({pre} pre, {panel['period'].nunique() - pre} post)"
    )
    print(f"Treated: {treated_name}   True ATT: {true_effect:+.2f}\n")

    result = fit_synthetic_control(y1, Y0, pre_periods=pre, donor_names=donors)
    print(result)

    wide = panel.pivot(index="period", columns="unit", values="y").sort_index()
    Y_all = wide.to_numpy()
    unit_names = list(wide.columns)
    placebos = in_space_placebo(
        Y_all, pre_periods=pre, treated_idx=treated_idx, unit_names=unit_names
    )
    pval = placebo_pvalue(placebos, treated_name)
    print(f"\nPlacebo-in-space p-value (RMSPE-ratio rank): {pval:.3f}")

    placebo_time = in_time_placebo(
        y_treated=y1,
        Y_donors=Y0,
        true_pre_periods=pre,
        fake_pre_periods=pre - 5,
        donor_names=donors,
    )
    print(f"Placebo-in-time ATT (should be ~0): {placebo_time.att:+.3f}")


if __name__ == "__main__":
    main()
