"""Expanding-window experiment with auditable splits, artifacts and metrics."""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
import sys
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from data import load_panel, panel_paths, prepare_panel, temporal_split
from portfolio import long_short_decile_returns, portfolio_summary
from models import (TrainConfig, seed_everything, train_one_model, predict_df,
                    fit_predict_ridge, fit_predict_pca_ridge, train_predict_mlp,
                    oos_r2, rank_ic_by_month, encoder_effective_weights, monthly_factor_premia)

def run_backtest(panel, out, cfg=None, first_year=2011, last_year=2021, seed=42, mlp_epochs=6):
    cfg = cfg or TrainConfig()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    panel.isna().mean().rename("missing_fraction").to_csv(out / "missingness.csv")
    data, features = prepare_panel(panel)
    data.groupby("year").size().rename("rows").to_csv(out / "rows-by-year.csv")
    predictions, metrics, splits = [], [], []
    for year in range(first_year, last_year + 1):
        tr, va, te = temporal_split(data, year)
        if len(features) < cfg.latent_dim or len(tr) < cfg.latent_dim:
            raise ValueError("Too few features/observations for requested PCA dimensions.")
        splits.append(dict(test_year=year, train_rows=len(tr), validation_rows=len(va), test_rows=len(te),
                           train_end=str(tr.month.max().date()), validation_start=str(va.month.min().date()),
                           validation_end=str(va.month.max().date()), test_start=str(te.month.min().date())))
        sae, scaler = train_one_model(tr, va, features, "ret_excess", cfg)
        pred_sae, z = predict_df(sae, scaler, te, features)
        frame = te[["permno", "month", "year", "ret_excess"]].copy()
        frame["pred_sae"] = pred_sae
        frame["pred_ridge"] = fit_predict_ridge(tr, te, features, alpha=5.)
        frame["pred_pca_ridge"] = fit_predict_pca_ridge(tr, te, features, n_components=cfg.latent_dim, alpha=5.)
        frame["pred_mlp"] = train_predict_mlp(tr, va, te, features, epochs=mlp_epochs,
                                             hidden_dim=64, batch_size=cfg.batch_size)
        for name, col in [("SAE", "pred_sae"), ("Ridge", "pred_ridge"), ("PCA+Ridge", "pred_pca_ridge"), ("MLP", "pred_mlp")]:
            if not np.isfinite(frame[col]).all():
                raise ValueError(f"Non-finite {name} predictions in {year}.")
            metrics.append(dict(test_year=year, model=name, mse=mean_squared_error(frame.ret_excess, frame[col]),
                                oos_r2=oos_r2(frame.ret_excess, frame[col]), rank_ic=rank_ic_by_month(frame, col)))
        predictions.append(frame)
        torch.save({"state_dict": {k: v.cpu() for k, v in sae.state_dict().items()},
                    "features": features, "config": asdict(cfg)}, out / f"sae-{year}.pt")
        joblib.dump(scaler, out / f"scaler-{year}.joblib")
        if year == last_year:
            effective = encoder_effective_weights(sae)
            loadings = []
            for k, weights in enumerate(effective):
                for j in np.argsort(-np.abs(weights))[:10]:
                    loadings.append(dict(factor=k, characteristic=features[j], weight=weights[j], abs_weight=abs(weights[j])))
            pd.DataFrame(loadings).to_csv(out / "factor-loadings.csv", index=False)
            monthly_factor_premia(te, z, te.ret_excess.to_numpy(), te.month.to_numpy()).to_csv(out / f"factor-premia-{year}.csv", index=False)
    oos = pd.concat(predictions, ignore_index=True)
    oos.to_parquet(out / "private-oos-predictions.parquet", index=False)
    pd.DataFrame(splits).to_csv(out / "split-audit.csv", index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "metrics-by-year.csv", index=False)
    summary = metrics.groupby("model").agg(mse_mean=("mse", "mean"), oos_r2_mean=("oos_r2", "mean"),
        rank_ic_mean=("rank_ic", "mean"), years=("test_year", "nunique"))
    summary.to_csv(out / "model-comparison.csv")
    portfolios = []
    for col in ["pred_sae", "pred_ridge", "pred_pca_ridge", "pred_mlp"]:
        path = long_short_decile_returns(oos, pred_col=col)
        path.to_csv(out / f"portfolio-path-{col}.csv", index=False)
        portfolios.append(dict(model=col, **portfolio_summary(path)))
    pd.DataFrame(portfolios).to_csv(out / "decile-portfolios.csv", index=False)
    metadata = dict(seed=seed, configuration=asdict(cfg), mlp_epochs=mlp_epochs,
        initial_rows=len(panel), observed_target_rows=len(data), features=features,
        first_test_year=first_year, last_test_year=last_year, python=sys.version,
        platform=platform.platform(), device="cuda" if torch.cuda.is_available() else "cpu",
        packages={p: importlib.metadata.version(p) for p in ["torch", "numpy", "pandas", "scikit-learn", "pyarrow"]},
        imputation="Contemporaneous feature medians before filtering missing targets",
        r2_benchmark="test-year mean", portfolio_quantiles=[.1, .9], sharpe="monthly",
        loadings="last evaluated SAE encoder; linear-weight product approximation")
    (out / "run-info.json").write_text(json.dumps(metadata, indent=2))
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    panel = load_panel(args.data_dir)
    result = run_backtest(panel, args.output_dir, seed=args.seed)
    manifest = []
    for path in panel_paths(args.data_dir):
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha))
    pd.DataFrame(manifest).to_csv(Path(args.output_dir) / "input-manifest.csv", index=False)
    print(result)

if __name__ == "__main__":
    main()
