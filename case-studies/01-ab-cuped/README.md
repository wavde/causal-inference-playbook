# Case Study 01 — A/B Test with CUPED Variance Reduction

**Method:** Regression-adjusted A/B analysis using a pre-experiment covariate.
**Paper:** Deng, Xu, Kohavi, Walker (2013), *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data.*

## TL;DR

On a simulated streaming A/B test (N=20,000, true effect = +3 minutes watched), CUPED cut the standard error of the treatment effect by ~30% — equivalent to roughly **2× more statistical power for free**, just by using pre-experiment engagement as a covariate.

## Business framing

At companies like Netflix, YouTube, or Spotify, most treatment effects of interest are **small** (0.5%–2% lifts on core engagement metrics). With tens of millions of users, those effects are detectable — but every week of experiment time is expensive, and experiments block the release pipeline behind them.

Variance reduction is the cheapest lever you have. It doesn't change the estimand; it just shrinks the confidence interval around it.

## Method

For each user *i* with outcome *Yᵢ* (minutes watched during experiment) and pre-experiment covariate *Xᵢ* (minutes watched in the 14 days before), define:

$$Y^{\text{cuped}}_i = Y_i - \theta \cdot (X_i - \bar{X})$$

where $\theta = \text{Cov}(Y, X) / \text{Var}(X)$ is estimated on the pooled sample.

Because the experiment is randomized, $X$ is independent of treatment assignment. Subtracting $\theta(X_i - \bar{X})$ is therefore an unbiased transformation of the outcome, and it reduces variance by a factor of $(1 - \rho^2)$ where $\rho = \text{Corr}(Y, X)$.

Run a standard two-sample test on $Y^{\text{cuped}}$ to get the treatment effect.

## How to reproduce

```bash
cd case-studies/01-ab-cuped
python src/run.py
```

Expected output (seed=42):

```
Sample size: 20,000  (treated=9,987)

naive_welch: ATE=+3.1123 (95% CI [+2.3842, +3.8404], SE=0.3714, p=0.0000, n_t=9987, n_c=10013)
cuped(theta=0.703): ATE=+3.0884 (95% CI [+2.5732, +3.6036], SE=0.2628, p=0.0000, n_t=9987, n_c=10013)

Expected variance reduction (1 - rho^2): 0.510
Observed SE ratio (CUPED / naive):        0.708
Equivalent sample-size multiplier:        2.00x
```

## Results

| Method | ATE | 95% CI | SE |
|--------|-----|--------|-----|
| Naive Welch t-test | +3.11 min | [+2.38, +3.84] | 0.37 |
| CUPED (θ=0.70) | +3.09 min | [+2.57, +3.60] | **0.26** |

Both estimates are close to the true effect of +3.0 minutes (the small gap is Monte Carlo noise, as expected). The CUPED standard error is ~30% smaller, which means the same decision could have been made with ~50% fewer users — or equivalently, in half the time.

## When CUPED helps (and when it doesn't)

| Scenario | CUPED useful? |
|----------|---------------|
| Pre-covariate strongly correlated with outcome (ρ > 0.5) | ✅ Yes — big wins |
| No pre-period data available (new users) | ❌ Theta = 0, no gain |
| Binary or very skewed outcomes | ⚠️ Marginal; consider stratification instead |
| Novelty effects (pre-period not representative) | ⚠️ Bias risk if correlation breaks down |
| Small samples (n < 1,000) | ⚠️ Theta estimation itself becomes noisy |

## Limitations & what I'd do next

1. **Single simulation run.** A proper empirical study would repeat this across hundreds of seeds and report the variance-reduction distribution.
2. **Homogeneous treatment effect.** In reality, effects are heterogeneous. CUPED still gives unbiased ATE, but combining with CATE estimation (e.g., causal forests) is the richer story.
3. **Ratio metrics.** Real product metrics (minutes/DAU, revenue/session) are ratios, which introduces a delta-method wrinkle in the variance. Deng, Knoblich, Lu (2018) is the right extension.
4. **Stratified CUPED.** In multi-country rollouts, stratifying $\theta$ by segment can squeeze additional variance reduction.
5. **Real-data replication.** Next step is porting this to the [Criteo Uplift dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) for a non-simulated demonstration.

## References

- Deng, Xu, Kohavi, Walker (2013). *KDD.*
- Kohavi, Tang, Xu (2020). *Trustworthy Online Controlled Experiments.* Chapter 22.
- Xie, Aurisset (2016). *Improving the Sensitivity of Online Controlled Experiments: Case Studies at Netflix.*
