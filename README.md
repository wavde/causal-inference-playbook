# Causal Inference Playbook

> Worked case studies in experimentation and quasi-experimental methods, each paired with a memo-style writeup.

![CI](https://github.com/wavde/causal-inference-playbook/actions/workflows/ci.yml/badge.svg)

![hero](docs/hero.png)

A reference set of end-to-end case studies covering the methods analysts reach for when a clean A/B test is not available, or when a clean A/B test is available but variance has to be wrung out of it. Each case pairs working code with a short memo that states the question, the method, the results, and the limitations.

Intended audience: practitioners who want a comparable reference implementation of each method, reviewers who want to see the diagnostics alongside the estimate, and anyone preparing for an interview loop that covers causal methods in depth.

## What each method earns its place for

- **Variance reduction** (CUPED, stratification), for the regime where MDEs are small and sample size is the binding constraint.
- **Quasi-experiments** (synthetic control, difference-in-differences), for interventions that cannot be randomised.
- **Sequential tests**, because stakeholders look at dashboards on their own schedule.
- **Memo-writing**, because the analysis only matters through the decision it informs.

## Case studies

| # | Case study | Method | Status |
|---|-----------|--------|--------|
| 01 | [A/B test with CUPED](case-studies/01-ab-cuped/) | Regression-adjusted variance reduction | Complete |
| 02 | [Synthetic control](case-studies/02-synthetic-control/) | Abadie weighted donors with placebo inference | Complete |
| 03 | [Difference-in-differences](case-studies/03-diff-in-diff/) | TWFE, event study, Callaway-Sant'Anna | Complete |
| 04 | [Propensity score matching](case-studies/04-propensity-score/) | PSM, IPW, doubly-robust AIPW, LaLonde replication | Complete |
| 05 | [Sequential testing](case-studies/05-sequential-testing/) | mSPRT, Pocock alpha-spending | Complete |
| 06 | [Switchback (marketplace)](case-studies/06-switchback/) | Block-level randomisation under SUTVA violation | Complete |

Each case study uses the same layout:

```
case-studies/NN-name/
  src/          # implementation, simulator, reproducer
  README.md     # framing, method, results, limitations
```

Shared tests are in `tests/`; CI configuration is in `.github/workflows/`.

## Companion repo: paid-media measurement

The synthetic-control method in case 02 and the DiD method in case 03 are the foundations for paid-media geo-lift measurement. A worked paid-media application lives in a sibling repo, [**paid-media-playbook**](https://github.com/wavde/paid-media-playbook): DMA-level geo lift, multi-touch attribution, incrementality, MMM, and web attribution with cookie-loss and iOS ATT accounting.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
pytest -q
```

Each case README has its own reproducer command.

## Out of scope

This repository does not cover structural causal models, instrumental-variable designs, mediation analysis, or Bayesian causal machinery. It also does not ship any production pipeline; the data is simulated or, in the LaLonde case, public.

## References

- Kohavi, Tang, Xu. *Trustworthy Online Controlled Experiments.*
- Deng et al. (2013). Improving the sensitivity of online controlled experiments using pre-experiment data (CUPED).
- Abadie, Diamond, Hainmueller (2010). Synthetic control methods.
- Goodman-Bacon (2021). Difference-in-differences with variation in treatment timing.
- Callaway & Sant'Anna (2021). Difference-in-differences with multiple time periods.
- Rosenbaum & Rubin (1983). The central role of the propensity score.
- Johari, Pekelis, Walsh (2017). Peeking at A/B tests (mSPRT).
- Howard, Ramdas, McAuliffe, Sekhon (2021). Time-uniform confidence sequences.

## License

MIT. See [LICENSE](LICENSE).
