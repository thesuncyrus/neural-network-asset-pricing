# Data setup

The historical experiment requires the 252 monthly Parquet files `200101_0.parquet` through `202112_0.parquet`. Nested folders are supported. Each file's actual dates must match its filename. Required fields: `permno`, `month`, `ret_excess` and exactly 94 numeric `characteristic_*` columns. Additional macro/industry fields are retained for auditing but excluded from models.

Duplicate security-month keys, missing months, duplicate files, infinite measurements and incorrect feature counts are rejected. Missing targets are dropped after target-independent monthly feature imputation. The underlying panel is not included; `dataset.pdf` is documentation, not data. The 252 required monthly files are now available locally and are used only for private validation.

From the root directory:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python analysis.py --data-dir /path/to/monthly-parquet --output-dir artifacts
```

The full neural backtest can take substantial CPU time. It writes model checkpoints, train scalers, predictions, split audits, annual metrics, portfolios, loadings and run metadata to ignored `artifacts/`. The CLI hashes every input file. Training/testing defaults are fixed in `TrainConfig` and `analysis.py`.

For inference with an evaluated checkpoint:

```sh
python predict.py --data data/new-month.parquet --checkpoint artifacts/sae-2021.pt --scaler artifacts/scaler-2021.joblib --output artifacts/private-predictions.csv
```

Provide complete contemporaneous cross-sections for median imputation. Inference does not require returns. Only load trusted local checkpoint/scaler files. Keep all security-level records and models out of public commits until redistribution rights are established.

Historical replay: `python run_historical.py --data-dir /private/GatingModelData2 --output-dir artifacts/historical`. This path preserves the original missing-target filter before feature imputation and the notebook RNG order. The corrected modular path above is a separate experiment.
