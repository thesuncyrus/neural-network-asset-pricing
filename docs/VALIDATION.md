# Validation

Verified on 2026-11-16 with Python 3.12 and CPU PyTorch 2.5.1. Exact top-level versions are in `requirements.txt`.

`python -m pytest -q`: **7 passed**. Coverage includes panel/schema/month coverage, duplicate keys, file/date mismatch, target-independent imputation, temporal ordering, exact decile leg size, collapsed prediction ties, numerically stable grouped ranking loss, constant-target R², and a complete artificial-data backtest of all four models. Checkpoints were reloaded and inference predictions matched the backtest predictions.

The integration test writes annual metrics, splits, decile paths, loadings, metadata and local predictions to a temporary directory. Its figures and scores are not historical results. The full 2001–2021 training run remains unverified because the required panel was not supplied.
