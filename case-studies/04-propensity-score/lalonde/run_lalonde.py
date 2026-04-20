"""
LaLonde (1986) replication — the canonical observational-causal benchmark.

Pipeline
--------
1. Fetch the three Dehejia-Wahba text files (treated, NSW control, CPS control)
   and cache them under ``data/`` (gitignored) so subsequent runs are offline.
2. Compute the experimental benchmark ATT by running a simple difference of
   means on the NSW treated vs NSW control panel. The canonical answer is
   ~$1794 (1978 dollars).
3. Throw away the NSW control group. Combine NSW treated with the CPS
   non-experimental control sample. This is the hostile observational setting
   where LaLonde showed classical econometric adjustments fail.
4. Estimate the ATT by:
     - Naive difference in means
     - 1-NN propensity score matching with caliper (reuses ``psm.py``)
     - Inverse-probability weighting
     - Augmented IPW (doubly-robust)
5. Print a comparison table against the experimental benchmark. A well-behaved
   estimator should land within a few hundred dollars of ~$1794; the naive
   estimator will be wildly negative.

Run
---
    python lalonde/run_lalonde.py

Data source: http://users.nber.org/~rdehejia/nswdata2.html
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

sys.path.insert(0, str(SRC))
from psm import aipw_att, ipw_att, psm_att  # noqa: E402

BASE_URL = "http://users.nber.org/~rdehejia/data"
FILES = {
    "nsw_treated": "nswre74_treated.txt",
    "nsw_control": "nswre74_control.txt",
    "cps_control": "cps_controls.txt",
}
COLS = [
    "treat", "age", "educ", "black", "hispanic",
    "married", "nodegree", "re74", "re75", "re78",
]


def _fetch(name: str, filename: str) -> pd.DataFrame:
    cached = DATA / filename
    if not cached.exists():
        url = f"{BASE_URL}/{filename}"
        print(f"Fetching {url} -> {cached.name}")
        urllib.request.urlretrieve(url, cached)
    return pd.read_csv(cached, sep=r"\s+", header=None, names=COLS, engine="python")


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (_fetch("nsw_treated", FILES["nsw_treated"]),
            _fetch("nsw_control", FILES["nsw_control"]),
            _fetch("cps_control", FILES["cps_control"]))


COVARIATES = ["age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75"]


def experimental_benchmark(treated: pd.DataFrame, control: pd.DataFrame) -> float:
    return float(treated["re78"].mean() - control["re78"].mean())


def build_observational_panel(
    treated: pd.DataFrame, cps: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    panel = pd.concat([treated, cps], ignore_index=True)
    X = panel[COVARIATES].to_numpy(dtype=float)
    T = panel["treat"].to_numpy(dtype=int)
    Y = panel["re78"].to_numpy(dtype=float)
    return X, T, Y


def main() -> None:
    treated, nsw_control, cps = load_panels()
    print(f"NSW treated: n={len(treated)}")
    print(f"NSW control: n={len(nsw_control)}")
    print(f"CPS control: n={len(cps)}")

    benchmark = experimental_benchmark(treated, nsw_control)
    print(f"\nExperimental benchmark ATT (NSW only): ${benchmark:,.0f}")
    print("  (LaLonde 1986 / Dehejia-Wahba 1999 canonical answer ~ $1,794)")

    X, T, Y = build_observational_panel(treated, cps)
    print(f"\nObservational panel: n_treated={T.sum()}, n_control={(1 - T).sum()}")

    naive = float(Y[T == 1].mean() - Y[T == 0].mean())

    match_res = psm_att(X, T, Y)
    ipw = ipw_att(X, T, Y)
    aipw = aipw_att(X, T, Y)

    print("\n--- ATT estimates on the observational (treated + CPS) panel ---")
    print(f"  Naive diff-in-means : ${naive:>10,.0f}")
    print(f"  1-NN PSM (caliper)  : ${match_res.att:>10,.0f}")
    print(f"  IPW                 : ${ipw:>10,.0f}")
    print(f"  AIPW (doubly robust): ${aipw:>10,.0f}")
    print(f"  Experimental target : ${benchmark:>10,.0f}")

    print("\nInterpretation:")
    print(" - Naive is badly negative: CPS controls earn far more than program")
    print("   participants in 1978, so uncorrected differences are misleading.")
    print(" - PSM / IPW / AIPW should land within a few hundred dollars of the")
    print("   experimental benchmark when covariates are sufficient — exactly")
    print("   the failure mode LaLonde highlighted and Dehejia-Wahba repaired.")


if __name__ == "__main__":
    main()
