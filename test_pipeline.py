"""Synthetic fixtures verify mechanics, never historical performance claims."""
import numpy as np
import pandas as pd
import pytest
import torch
import joblib
from predict import predict_months
from data import validate_panel, load_panel, prepare_panel, temporal_split
from models import TrainConfig, pairwise_rank_loss, oos_r2
from portfolio import long_short_decile_returns, portfolio_summary
from analysis import run_backtest

def panel(n=40):
    rng = np.random.default_rng(9)
    dates = pd.date_range("2008-01-01", "2011-12-01", freq="MS")
    d = pd.DataFrame({"month": np.repeat(dates, n), "permno": np.tile(np.arange(n), len(dates))})
    for j in range(8):
        d[f"characteristic_{j}"] = rng.normal(size=len(d))
    d["ret_excess"] = .01 * d.characteristic_0 + rng.normal(0, .02, len(d))
    return validate_panel(d, "2008-01", "2011-12", 8)

def test_schema_and_duplicate_month():
    d = panel()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_panel(pd.concat([d, d.iloc[[0]]]), "2008-01", "2011-12", 8)
    with pytest.raises(ValueError, match="coverage"):
        validate_panel(d[d.month != d.month.min()], "2008-01", "2011-12", 8)
    d.loc[0, "characteristic_0"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_panel(d, "2008-01", "2011-12", 8)

def test_file_month_mismatch(tmp_path):
    d = panel(); d = d[d.month == pd.Timestamp("2008-01-01")]
    d.to_parquet(tmp_path / "200802_0.parquet")
    with pytest.raises(ValueError, match="disagree"):
        load_panel(tmp_path, "2008-02", "2008-02", 8)

def test_imputation_does_not_condition_on_target():
    d = panel()
    ids = d.index[d.month == d.month.min()]
    d.loc[ids, "characteristic_0"] = np.arange(len(ids), dtype=float)
    d.loc[ids[0], "characteristic_0"] = np.nan
    d.loc[ids[-20:], "ret_excess"] = np.nan
    filled, _ = prepare_panel(d)
    assert filled.loc[ids[0], "characteristic_0"] == np.median(np.arange(1, len(ids)))
    train, valid, test = temporal_split(filled, 2011)
    assert train.month.max() < valid.month.min() < test.month.min()

def test_deciles_and_ties():
    d = pd.DataFrame({"year": 2021, "month": pd.Timestamp("2021-01-01"),
                      "pred": np.arange(100), "ret_excess": np.arange(100)/1000})
    p = long_short_decile_returns(d)
    assert p.n_long.iloc[0] == p.n_short.iloc[0] == 10
    assert p.long_short.iloc[0] == pytest.approx(.09)
    assert portfolio_summary(p)["n_months"] == 1
    d["pred"] = 1
    with pytest.raises(ValueError, match="Overlapping"):
        long_short_decile_returns(d)

def test_rank_loss_is_stable_and_group_local():
    torch.manual_seed(12)
    y = torch.arange(8, dtype=torch.float32)
    pred = (-y * 1e6).requires_grad_()
    group = torch.zeros(8, dtype=torch.long)
    loss = pairwise_rank_loss(y, pred, group)
    assert torch.isfinite(loss)
    loss.backward(); assert torch.isfinite(pred.grad).all()
    assert pairwise_rank_loss(y, pred, torch.arange(8)).item() == 0

def test_r_squared_definition():
    y = np.arange(5.)
    assert oos_r2(y, y) == 1
    assert oos_r2(y, y+100) < 0
    assert np.isnan(oos_r2(np.ones(5), np.ones(5)))

def test_end_to_end_backtest(tmp_path):
    torch.set_num_threads(1)
    cfg = TrainConfig(epochs=1, batch_size=256, hidden_dim=8, latent_dim=2, dropout=0.)
    result = run_backtest(panel(), tmp_path, cfg, first_year=2011, last_year=2011, mlp_epochs=1)
    assert len(result) == 4 and (result.years == 1).all()
    predictions = pd.read_parquet(tmp_path / "private-oos-predictions.parquet")
    assert len(predictions) == 480 and predictions.year.eq(2011).all()
    assert (tmp_path / "sae-2011.pt").exists()
    metrics = pd.read_csv(tmp_path / "decile-portfolios.csv")
    assert len(metrics) == 4 and metrics.n_months.eq(12).all()
    assert (tmp_path / "run-info.json").exists()
    checkpoint = torch.load(tmp_path / "sae-2011.pt", weights_only=True)
    scaler = joblib.load(tmp_path / "scaler-2011.joblib")
    d = panel(); d = d[d.year == 2011]
    predicted = predict_months(d.drop(columns="ret_excess"), checkpoint, scaler)
    np.testing.assert_allclose(predicted.pred_sae, predictions.pred_sae, rtol=1e-5, atol=1e-7)
