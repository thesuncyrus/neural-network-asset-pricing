"""Monthly panel validation and target-independent feature preparation."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def validate_panel(data, start="2001-01", end="2021-12", n_features=94):
    required = {"permno", "month", "ret_excess"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    data = data.copy()
    data["month"] = pd.to_datetime(data["month"], errors="raise")
    if data[["permno", "month"]].isna().any().any():
        raise ValueError("Missing security or month identifiers.")
    period = data.month.dt.to_period("M")
    if pd.DataFrame({"id": data.permno, "month": period}).duplicated().any():
        raise ValueError("Duplicate security-month records.")
    expected = pd.period_range(start, end, freq="M")
    missing = expected.difference(period.unique())
    extra = pd.PeriodIndex(period.unique()).difference(expected)
    if len(missing) or len(extra):
        raise ValueError(f"Sample coverage mismatch; missing={list(missing)}, extra={list(extra)}")
    features = [c for c in data if c.startswith("characteristic_")]
    if len(features) != n_features:
        raise ValueError(f"Expected {n_features} characteristics; got {len(features)}.")
    for col in features + ["ret_excess"]:
        if not pd.api.types.is_numeric_dtype(data[col]) or np.isinf(data[col].to_numpy(dtype=float)).any():
            raise ValueError(f"Non-numeric or infinite values in {col}")
    data["month"] = period.dt.to_timestamp()
    data["year"] = data.month.dt.year
    return data.sort_values(["month", "permno"]).reset_index(drop=True)

def panel_paths(directory, start="2001-01", end="2021-12"):
    lo, hi = int(start.replace("-", "")), int(end.replace("-", ""))
    paths = []
    for p in sorted(Path(directory).rglob("*.parquet")):
        match = re.fullmatch(r"(\d{6})_0\.parquet", p.name)
        if match and lo <= int(match[1]) <= hi:
            paths.append(p)
    if not paths:
        raise ValueError(f"No YYYYMM_0.parquet files for {start} through {end}.")
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("Duplicate monthly files in nested directories.")
    return paths

def load_panel(data_directory, start="2001-01", end="2021-12", n_features=94):
    frames = []
    for p in panel_paths(data_directory, start, end):
        frame = pq.ParquetFile(p).read().to_pandas()
        if "month" not in frame or not (pd.to_datetime(frame.month).dt.strftime("%Y%m") == p.name[:6]).all():
            raise ValueError(f"File month and row dates disagree: {p.name}")
        frames.append(frame)
    return validate_panel(pd.concat(frames, ignore_index=True), start, end, n_features)

def prepare_panel(data):
    features = [c for c in data if c.startswith("characteristic_")]
    data = data.copy()
    # Compute monthly feature medians before filtering future target availability.
    medians = data.groupby("month")[features].transform("median")
    data[features] = data[features].fillna(medians).fillna(0.)
    data = data.dropna(subset=["ret_excess"]).copy()
    data["ym_id"] = data.year * 100 + data.month.dt.month
    return data, features

def temporal_split(data, test_year):
    train = data[data.year <= test_year - 2].copy()
    valid = data[data.year == test_year - 1].copy()
    test = data[data.year == test_year].copy()
    if any(d.empty for d in (train, valid, test)):
        raise ValueError(f"Empty partition for {test_year}.")
    if not train.month.max() < valid.month.min() <= valid.month.max() < test.month.min():
        raise ValueError("Temporal partitions overlap.")
    return train, valid, test
