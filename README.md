# Causal Inference Playbook

> A hands-on tour of experimentation and causal methods used in senior analytics work — each case study paired with a memo-style writeup.

![CI](https://github.com/wavde/causal-inference-playbook/actions/workflows/ci.yml/badge.svg)

## Why this repo

Most analytics portfolios stop at "ran a t-test on A/B data." Senior analytics work at FAANG-scale companies looks different:

- **Variance reduction** (CUPED, stratification) because 1% MDEs require ruthless efficiency
- **Quasi-experiments** (synthetic control, diff-in-diff) because you can't always randomize
- **Sequential tests** because business stakeholders peek at dashboards
- **Memo-writing** because the analysis is only as good as the decision it drives

This repo is a running collection of case studies, each with code *and* a written memo.

## Case Studies

| # | Case Study | Method | Status |
|---|-----------|--------|--------|
| 01 | [A/B test with CUPED](case-studies/01-ab-cuped/) | Regression-adjusted variance reduction | ✅ Complete |
| 02 | [Synthetic control](case-studies/02-synthetic-control/) | Abadie-style synthetic control | 🚧 Planned |
| 03 | [Difference-in-differences](case-studies/03-diff-in-diff/) | DiD + parallel trends diagnostic | 🚧 Planned |
| 04 | [Propensity score matching](case-studies/04-propensity-score/) | PSM + doubly-robust estimation | 🚧 Planned |
| 05 | [Sequential testing](case-studies/05-sequential-testing/) | mSPRT / always-valid p-values | 🚧 Planned |

## Repo layout

```
case-studies/
  01-ab-cuped/        # fully worked example — use this as the quality bar
    src/              # reusable code
    docs/             # memo-style writeup
    notebook.ipynb    # reproducible analysis
    README.md
  02-...              # planned
tests/                # unit tests for shared utilities
.github/workflows/    # CI
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pytest -q                     # runs smoke tests
```

Each case study's README documents how to reproduce its analysis.

## References

- Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments*
- Deng et al. (2013) — *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (CUPED)
- Abadie, Diamond, Hainmueller (2010) — *Synthetic Control Methods*

## License

MIT — see [LICENSE](LICENSE).
