# Result provenance and corrected execution

`results/` and the website figures derive from saved aggregate outputs in the supplied notebook. The 2001–2021 source panel was not available in the uploaded earlier-year archives. A full historical retraining has therefore not been verified.

| Saved notebook comparison | SAE | Ridge | MLP | PCA + Ridge |
| --- | ---: | ---: | ---: | ---: |
| Mean annual Rank IC | 0.080381 | 0.009429 | 0.004059 | 0.003583 |
| Mean annual R² (test-year mean benchmark) | -0.036972 | -0.010766 | -0.012341 | -0.008230 |
| Monthly decile Sharpe | 0.121458 | 0.346690 | 0.292342 | 0.183439 |

SAE leads this saved ranking comparison; Ridge leads the decile portfolio comparison. Monthly Sharpe is not annualized. Results are gross of costs. Earlier quintile experiments are not the source of the decile table.

## Current runnable implementation

- `analysis.py`, the root notebook and `predict.py` are the executable path.
- `notebooks/Historical_Decile_Analysis.ipynb` retains aggregate saved outputs from the earlier portfolio copy, with record-level previews removed.
- `original_code/MGMT475_Project_final.ipynb` preserves uploaded code/markdown with execution outputs removed; its earlier quintile calculations remain historical source only.
- `models.py` extracts the original model architecture/losses into importable functions, with stable softplus ranking loss and a correct constant-target R² guard.
- Feature imputation now precedes removal of unavailable targets, so future target availability does not determine a month's feature median.
- MLP validation MSE is weighted by batch size. Undefined SAE validation Rank IC fails clearly instead of silently accepting an arbitrary final epoch.
- Input validation checks month coverage, names, feature count, duplicate keys and finite values. Small valid samples are no longer silently skipped by arbitrary row-count thresholds.
- Factor interpretation reuses the last evaluated encoder instead of training an unrecorded second model. It remains a weight-product approximation, not an exact ReLU marginal effect.
- Decile code rejects overlapping legs caused by collapsed prediction cutoffs and reports actual leg sizes.

These fixes can change fitted results. They are not a claim that a new run exactly matches archived values. The seven-test suite includes training all four models, exports, checkpoint reload and prediction equality on artificial monthly data; synthetic scores are not used in the portfolio.
