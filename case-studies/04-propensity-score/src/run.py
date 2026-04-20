"""
Run the PSM case study end-to-end.

Usage:
    cd case-studies/04-propensity-score
    python src/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from psm import (  # noqa: E402
    _match_indices,
    aipw_att,
    balance_table,
    estimate_propensity,
    ipw_att,
    psm_att,
)
from psm_simulate import FEATURE_NAMES, simulate_observational  # noqa: E402


def main() -> None:
    true_effect = 50.0
    X, T, Y, _ = simulate_observational(n=5000, true_effect=true_effect, seed=0)
    print(f"Sample: n={len(T):,}  treated={int(T.sum()):,}  true ATT={true_effect:+.1f}\n")

    naive = float(Y[T == 1].mean() - Y[T == 0].mean())
    print(f"Naive (treated - control)        : {naive:+.2f}   (biased)")

    psm = psm_att(X, T, Y, caliper_sd=0.2, n_bootstrap=300, seed=0)
    print(psm)

    ipw = ipw_att(X, T, Y)
    aipw = aipw_att(X, T, Y)
    print(f"IPW ATT                          : {ipw:+.3f}")
    print(f"AIPW (doubly-robust) ATT         : {aipw:+.3f}")

    # Balance diagnostics
    e = estimate_propensity(X, T)
    logit_e = np.log(e / (1 - e))
    caliper = 0.2 * np.std(logit_e)
    t_idx, c_idx, dropped = _match_indices(e, T, caliper=caliper)
    bal = balance_table(X, T, FEATURE_NAMES, matched_t_idx=t_idx, matched_c_idx=c_idx)
    print("\nCovariate balance (SMD; |SMD|<0.1 = balanced):")
    print(bal.to_string(index=False))


if __name__ == "__main__":
    main()
