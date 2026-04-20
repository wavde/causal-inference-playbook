# Case 06: Switchback Testing (Marketplace Experiments)

> **Claim.** User-level A/B tests are biased when users in the same market share inventory. Switchback (block-level randomization) removes the bias by breaking interference between arms.

## The business problem

You're an analyst at a marketplace — rides, ads, listings, promo codes, whatever has a shared supply side. PM wants to ship a new ranking algo. You run a standard user-level A/B and see **+30%** on conversion. You ship it, then realize the production lift was closer to **+10%**. What happened?

**You violated SUTVA.** The treated users were taking inventory (drivers, ad slots, promo budget) away from the control users sharing the same market at the same time. Your control group's observed conversion was *depressed by the treatment itself*, so the gap you measured wasn't the gap you'd get by treating everyone.

## The model

Each user `i` in time block `b` converts with:

```
Y_i = alpha + tau * T_i  -  gamma * mean(T_j for j != i in block b)  +  noise
```

- `tau` = direct treatment benefit.
- `gamma` = spillover cost borne by everyone in the block when others are treated.

The **true policy effect** — what you get if you treat the whole market vs no one — is `tau - gamma`. That's what the PM actually wants to know.

With `tau = 0.30`, `gamma = 0.20`, truth is **+0.10**.

## Two estimators

| Design | What it measures | Bias |
|---|---|---|
| **User-level A/B** (`naive_ab_estimate`) | `tau` alone. Control is depressed by spillover from the 50% treated in the same block, and treated isn't. So the gap is `tau`, not `tau - gamma`. | **+gamma** (overstates) |
| **Switchback** (`switchback_estimate`) | `tau - gamma`. Entire blocks are 100% treated or 100% control, so there's no within-block mixing. | ≈ 0 |

## How to reproduce

```bash
cd case-studies/06-switchback
python src/run.py
```

Example output (seed=7, 200 blocks × 50 users):

```
True policy effect (treat-all vs treat-none): tau - gamma = +0.100

Naive A/B (user-level randomization):
  bias vs truth ≈ +0.20  (~3× overstated)

Switchback (block-level randomization, block-clustered SE):
  bias vs truth ≈ 0.00

Coverage check (500 replications):
  naive   95% CI coverage ≈ 0%   (confidently wrong)
  switch. 95% CI coverage ≈ 95%  (honest)
```

The naive test is 3× off and its 95% CI essentially *never* covers the truth — a textbook confidently-wrong test. Switchback recovers the truth with the expected ~95% coverage.

## When to reach for switchback

- Shared supply (rides, drivers, inventory, budget, slots).
- Network effects (social features, pricing, search ranking where user actions affect each other's experience).
- Anytime your control group isn't independent of your treatment group.

## When *not* to

- Switchback costs precision — the effective sample size is the number of **blocks**, not users. You'll need more total exposure to reach the same power.
- Pick block length carefully: too short → carryover from the previous block contaminates, too long → too few blocks to cluster on.
- If there's no interference (pure individual-level outcome, e.g., email click-through), a user-level A/B is more efficient and just as unbiased.

## What I'd do with more time

- Carryover modeling: add a decaying effect from the previous block and show how it biases switchback when blocks are too short.
- Optimal block-length choice via bias-variance tradeoff — see Hu & Wager (2022), "Switchback Experiments under Geometric Mixing."
- Adaptive designs: alternate deterministically and use regression-based estimators that model time trends.

## References

- Bojinov, Simchi-Levi, Zhao (2023). *Design and Analysis of Switchback Experiments.* Management Science.
- Hu, Wager (2022). *Switchback Experiments under Geometric Mixing.* arXiv:2209.00197.
- Chamandy (2016). *Experimentation in a Ridesharing Marketplace.* Lyft Engineering blog.
