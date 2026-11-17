# Equity panel inputs
The analysis expects 252 monthly Parquet files, 200101_0.parquet through 202112_0.parquet. Nested folders are supported. Required fields are permno, month, ret_excess and 94 numeric characteristic_* columns; the notebook also reads the year partition. Macro and industry fields are excluded from modeling.

The supplied panel combines CRSP/Compustat information accessed through WRDS with firm characteristics. Permission to redistribute the merged panel has not been established, so it is not included in the public repository. Users must obtain an authorized copy through their institution. Public characteristics alone are not the complete model input.

Place an authorized panel in data/GatingModelData2 or pass its directory explicitly. `run_notebook.py` executes the notebook; `analysis.py` is the modular runner. The modular loader validates month coverage, duplicate security-month keys, feature counts and finite values. See README.md for preprocessing differences between the two paths.

For inference: `python predict.py --data /path/to/new-month.parquet --checkpoint artifacts/modular/sae-2021.pt --scaler artifacts/modular/scaler-2021.joblib --output artifacts/private-predictions.csv`. Supply complete monthly cross-sections. Inference does not require realized returns.

Sources: [WRDS terms](https://wrds-www.wharton.upenn.edu/users/tou/) and [public firm-characteristic data](https://dachxiu.chicagobooth.edu/).
