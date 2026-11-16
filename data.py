from pathlib import Path
import argparse
import pandas as pd
import pyarrow.parquet as pq

def load_panel(data_directory):
    paths = sorted(Path(data_directory).rglob("*.parquet"))
    # Monthly files are named YYYYMM_0.parquet; only use the course sample.
    paths = [p for p in paths if p.name[:6].isdigit() and 200101 <= int(p.name[:6]) <= 202112]
    if not paths:
        raise ValueError("Provide the 2001–2021 monthly Parquet files.")
    frames = [pq.ParquetFile(p).read().to_pandas() for p in paths]
    data = pd.concat(frames, ignore_index=True)
    data["month"] = pd.to_datetime(data["month"])
    data["year"] = data["month"].dt.year
    if data.duplicated(["permno", "month"]).any():
        raise ValueError("Duplicate security-month records in the data directory.")
    months = data["month"].dt.to_period("M")
    expected = pd.period_range("2001-01", "2021-12", freq="M")
    missing = expected.difference(months.unique())
    if len(missing):
        raise ValueError(f"Missing months in the 2001–2021 sample: {list(missing)}")
    return data
