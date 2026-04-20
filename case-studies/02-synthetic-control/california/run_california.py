"""Replicate Abadie, Diamond, Hainmueller (2010) — California Proposition 99.

Data source
-----------
State-year panel of per-capita cigarette sales (packs/person/year) for 39
continental US states, 1970-2000. Prop 99 raised California's cigarette tax
by $0.25/pack in 1989; the paper's headline result is that post-1989 per-
capita consumption was ~20 packs lower than the synthetic California would
have been absent the tax.

Runs entirely against the existing `synthetic_control.py` solver.

Usage
-----
    python case-studies/02-synthetic-control/california/run_california.py

First run fetches the dataset (~0.1 MB) to ``california/data/smoking.csv``.
Subsequent runs are fully offline.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_CSV = DATA_DIR / "smoking.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/OscarEngelbrektson/"
    "SyntheticControlMethods/master/examples/datasets/smoking_data.csv"
)

TREATMENT_YEAR = 1989
TREATED_STATE = "California"

# States that raised cigarette taxes by $0.50+ or passed major tobacco control
# measures between 1989-2000, per Abadie (2010) §5, excluded from the donor
# pool. We keep the dataset's canonical 38-state donor pool and drop nothing
# additional — the SC solver handles the selection via weights that collapse
# to near-zero for poor matches.

SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from synthetic_control import (  # noqa: E402
    fit_synthetic_control,
    in_space_placebo,
    placebo_pvalue,
)


def _download() -> pd.DataFrame:
    if not DATA_CSV.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Fetching smoking dataset -> {DATA_CSV}")
        urllib.request.urlretrieve(DATA_URL, DATA_CSV)
    return pd.read_csv(DATA_CSV)


def _build_panel(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, int]:
    df = df.copy()
    df["year"] = df["year"].astype(int)
    pivot = df.pivot(index="year", columns="state", values="cigsale").sort_index()
    years = pivot.index.to_numpy()
    if TREATED_STATE not in pivot.columns:
        raise RuntimeError(f"{TREATED_STATE} missing from panel")
    y_treated = pivot[TREATED_STATE].to_numpy(dtype=float)
    donor_cols = [c for c in pivot.columns if c != TREATED_STATE]
    Y_donors = pivot[donor_cols].to_numpy(dtype=float)
    pre_periods = int(np.sum(years < TREATMENT_YEAR))
    return y_treated, Y_donors, donor_cols, years, pre_periods


def main() -> None:
    df = _download()
    y_treated, Y_donors, donor_names, years, pre_periods = _build_panel(df)

    print(f"\nDataset: {len(years)} years ({years.min()}-{years.max()}), "
          f"{Y_donors.shape[1]} donor states, {pre_periods} pre-treatment years.\n")

    result = fit_synthetic_control(
        y_treated=y_treated,
        Y_donors=Y_donors,
        pre_periods=pre_periods,
        donor_names=donor_names,
    )

    print("Synthetic California fit:")
    print(f"  {result}\n")

    nonzero = sorted(
        [(n, w) for n, w in zip(donor_names, result.weights, strict=False) if w > 0.01],
        key=lambda t: -t[1],
    )
    print("Donor weights (>1%):")
    for name, w in nonzero:
        print(f"  {name:<20s} {w:.3f}")

    print("\nTreated - synthetic gap by year:")
    for year, gap in zip(years, result.gap, strict=False):
        marker = "  " if year < TREATMENT_YEAR else "* "
        print(f"  {marker}{year}   {gap:+7.2f}")

    print(
        f"\nATT (avg {TREATMENT_YEAR}-{years.max()}): {result.att:+.2f} packs/capita/year"
    )
    print(f"Peak gap by {years.max()}: {result.gap[-1]:+.2f} packs/capita")
    print("\nAbadie (2010) published ATT: ~-19 packs/capita; peak by 2000: ~-26.")

    _unit_names = [TREATED_STATE] + donor_names
    Y_all = np.column_stack([y_treated, Y_donors])
    placebos = in_space_placebo(
        Y_all=Y_all,
        pre_periods=pre_periods,
        treated_idx=0,
        unit_names=_unit_names,
    )
    pval_loose = placebo_pvalue(placebos, TREATED_STATE)
    pval_abadie = placebo_pvalue(placebos, TREATED_STATE, rmspe_cutoff_multiplier=2.24)
    print(f"\nIn-space placebo p-value (20x pre-RMSPE filter): {pval_loose:.3f}")
    print(f"In-space placebo p-value (Abadie 2010 5x MSPE ≈ 2.24x RMSPE filter): {pval_abadie:.3f}")


if __name__ == "__main__":
    main()
