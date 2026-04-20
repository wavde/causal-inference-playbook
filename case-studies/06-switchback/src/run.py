"""Run the switchback case study end-to-end.

Shows that under spillover (SUTVA violation):
  - naive user-level A/B overstates the true policy effect
  - switchback (block-level randomization) recovers it

Usage:
    cd case-studies/06-switchback
    python src/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from switch_simulate import simulate_market, true_policy_effect
from switchback import naive_ab_estimate, switchback_estimate


def main() -> None:
    tau, gamma = 0.30, 0.20
    truth = true_policy_effect(tau, gamma)

    print(f"True policy effect (treat-all vs treat-none): tau - gamma = {truth:+.3f}\n")

    user_design = simulate_market(design="user", tau=tau, gamma=gamma, seed=7)
    est_naive = naive_ab_estimate(user_design.y, user_design.treatment)
    print("Naive A/B (user-level randomization):")
    print(f"  {est_naive}")
    print(f"  bias vs truth: {est_naive.point - truth:+.3f}  ({est_naive.point / truth:.1f}x)\n")

    sb_design = simulate_market(design="switchback", tau=tau, gamma=gamma, seed=7)
    est_sb = switchback_estimate(sb_design.y, sb_design.treatment, sb_design.block_id)
    print("Switchback (block-level randomization, block-clustered SE):")
    print(f"  {est_sb}")
    print(f"  bias vs truth: {est_sb.point - truth:+.3f}")

    print("\nBottom line: when spillover is present (gamma > 0), user-level")
    print("A/B tests overstate the true policy effect. Switchback breaks the")
    print("interference by making every user in a block share the same arm.")

    print("\nCoverage check (500 replications):")
    _coverage_simulation(tau, gamma, truth, n_reps=500)


def _coverage_simulation(
    tau: float, gamma: float, truth: float, n_reps: int
) -> None:
    naive_hits = 0
    sb_hits = 0
    naive_points = []
    sb_points = []
    for seed in range(n_reps):
        user_design = simulate_market(design="user", tau=tau, gamma=gamma, seed=seed)
        sb_design = simulate_market(design="switchback", tau=tau, gamma=gamma, seed=seed)
        n = naive_ab_estimate(user_design.y, user_design.treatment)
        s = switchback_estimate(sb_design.y, sb_design.treatment, sb_design.block_id)
        naive_points.append(n.point)
        sb_points.append(s.point)
        if n.ci_low <= truth <= n.ci_high:
            naive_hits += 1
        if s.ci_low <= truth <= s.ci_high:
            sb_hits += 1
    print(f"  naive   mean point = {np.mean(naive_points):+.3f}   "
          f"95% CI coverage = {naive_hits / n_reps:.1%}")
    print(f"  switch. mean point = {np.mean(sb_points):+.3f}   "
          f"95% CI coverage = {sb_hits / n_reps:.1%}")


if __name__ == "__main__":
    main()
