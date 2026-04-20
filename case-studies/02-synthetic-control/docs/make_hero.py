"""Hero chart: California synthetic control trajectory (Abadie 2010 replication)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
CALIFORNIA_DATA = Path(__file__).resolve().parents[1] / "california" / "data" / "smoking.csv"
sys.path.insert(0, str(SRC))

from synthetic_control import fit_synthetic_control  # noqa: E402

OUT = Path(__file__).resolve().parent / "hero.png"

TREATED_STATE = "California"
TREATMENT_YEAR = 1989


def main() -> None:
    if not CALIFORNIA_DATA.exists():
        raise SystemExit(
            f"Missing {CALIFORNIA_DATA}. Run california/run_california.py first."
        )

    df = pd.read_csv(CALIFORNIA_DATA)
    df["year"] = df["year"].astype(int)
    pivot = df.pivot(index="year", columns="state", values="cigsale").sort_index()
    years = pivot.index.to_numpy()
    y_treated = pivot[TREATED_STATE].to_numpy(dtype=float)
    donor_cols = [c for c in pivot.columns if c != TREATED_STATE]
    Y_donors = pivot[donor_cols].to_numpy(dtype=float)
    pre = int(np.sum(years < TREATMENT_YEAR))

    result = fit_synthetic_control(y_treated, Y_donors, pre_periods=pre,
                                   donor_names=donor_cols)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(years, y_treated, color="#d62728", lw=2.2, label="California (actual)")
    ax.plot(years, result.y_synthetic, color="#1f77b4", lw=2.2, ls="--",
            label="Synthetic California")
    ax.fill_between(years, y_treated, result.y_synthetic,
                    where=years >= TREATMENT_YEAR, alpha=0.2, color="#d62728")
    ax.axvline(TREATMENT_YEAR, color="black", lw=1, alpha=0.6)
    ax.annotate(
        "Prop 99\n(Jan 1989)",
        xy=(TREATMENT_YEAR, 80), xytext=(1975, 55),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", alpha=0.6),
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Cigarette sales (packs/capita/year)")
    ax.set_title(
        f"Case 02 — Abadie (2010) replication: California vs synthetic control\n"
        f"ATT (1989-2000) = {result.att:+.1f} packs/capita/year   "
        f"(published: ~-19)"
    )
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
