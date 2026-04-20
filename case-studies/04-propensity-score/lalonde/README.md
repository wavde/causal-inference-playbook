# LaLonde (1986) Replication

The canonical benchmark for observational causal inference: the National
Supported Work (NSW) job training experiment.

## Why this exists

The simulated PSM example in [`../`](../) validates that the code works under
known ground truth. This replication validates it against **the real data
LaLonde used to show classical econometric adjustments fail** — then
demonstrates that modern PSM / IPW / AIPW recover the experimental benchmark.

## The setup

- **Treated:** 185 participants in the NSW job training program.
- **Experimental control:** 260 randomized controls from the same NSW study.
  Diff-in-means ≈ **$1,794** (1978 earnings). This is the causal "truth."
- **Observational control:** 15,992 non-experimental CPS controls. Swapping
  the NSW control for CPS is the hostile setting LaLonde used.

## Results

Run the script:

```bash
cd case-studies/04-propensity-score
python lalonde/run_lalonde.py
```

On the observational (NSW-treated + CPS) panel you should see:

| Estimator | ATT | vs experimental benchmark |
|---|---:|---|
| Naive diff-in-means | ≈ −$8,500 | catastrophically wrong |
| 1-NN PSM (caliper) | within a few hundred $ of benchmark | ✅ |
| IPW | within a few hundred $ of benchmark | ✅ |
| AIPW (doubly-robust) | within a few hundred $ of benchmark | ✅ |
| Experimental target | ≈ **$1,794** | |

Exact numbers depend on the propensity-model specification; Dehejia-Wahba
(1999) and follow-up work show that reasonable specifications recover the
benchmark to within the sampling error of the original experiment.

## Why this matters for a portfolio

- **Real data, not simulation.** The canonical stress test for observational
  estimators — every senior causal-inference practitioner knows it.
- **A testable claim.** "Naive is ~$8.5K wrong; our PSM is within sampling
  error of the experimental benchmark."
- **Narrow, contained scope.** One script, one table, one conclusion.

## Data source

[Dehejia-Wahba replication files](http://users.nber.org/~rdehejia/nswdata2.html) —
`nswre74_treated.txt`, `nswre74_control.txt`, `cps_controls.txt`.

Files are downloaded on first run and cached in `data/` (gitignored).

## References

- LaLonde, R. (1986). *Evaluating the Econometric Evaluations of Training
  Programs with Experimental Data.* **American Economic Review**, 76(4).
- Dehejia, R. & Wahba, S. (1999). *Causal Effects in Nonexperimental Studies:
  Reevaluating the Evaluation of Training Programs.* **JASA**, 94(448).
- Smith, J. & Todd, P. (2005). *Does Matching Overcome LaLonde's Critique?*
  **Journal of Econometrics**, 125(1–2).
