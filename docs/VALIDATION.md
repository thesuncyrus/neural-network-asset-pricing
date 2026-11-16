# Validation

Executed 2026-11-16 with Python 3.12, PyTorch 2.15.0 on CPU, pandas 2.2.3 and scikit-learn 1.9.1. Exact top-level dependencies are in requirements.txt.

The complete recovered historical notebook ran on 252 monthly files covering 2001–2021, with all four models evaluated for test years 2011–2021. It also completed the original separate factor-interpretation refit and final 90/10 portfolio calculations. Newly exported annual metrics are in results/replay-2026-11-16; saved original outputs are retained separately. See RESULTS_PROVENANCE.md for the comparison and export boundary. Raw equity records and logs remain private.

`python -m pytest -q`: **7 passed** for the separately corrected modular workflow. Checks cover schema/coverage, duplicate keys, temporal separation, target-independent imputation, decile ties, ranking-loss stability, constant-target R², and an artificial-data backtest with all four models and matching reloaded-checkpoint predictions. Artificial scores are not research findings.
