"""Hero chart: event-study DiD coefficients with 95% CIs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from did import event_study  # noqa: E402
from did_simulate import simulate_did_panel  # noqa: E402

OUT = Path(__file__).resolve().parent / "hero.png"


def main() -> None:
    df = simulate_did_panel(
        n_units=40,
        n_treated=15,
        n_periods=20,
        treatment_period=12,
        true_effect=2.5,
        seed=7,
    )
    result = event_study(df, treatment_period=12)

    coefs = result.coefs
    et = coefs["relative_time"].to_numpy()
    c = coefs["coef"].to_numpy()
    lo = coefs["ci_low"].to_numpy()
    hi = coefs["ci_high"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(et, c, yerr=[c - lo, hi - c], fmt="o", color="#2ca02c",
                capsize=4, lw=1.6, markersize=7, label="Coefficient (95% CI)")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Event time (periods from treatment)")
    ax.set_ylabel("DiD coefficient")
    ax.set_title(
        f"Case 03 — Event-study DiD\n"
        f"Flat pre-trends (parallel-trends F-test p={result.parallel_trends_pvalue:.2f}), "
        f"~+2.5 unit jump at t=0"
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
