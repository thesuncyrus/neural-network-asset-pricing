"""Score complete monthly cross-sections with a saved SAE and training scaler."""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
from models import SupervisedAutoencoder

def predict_months(frame, checkpoint, scaler):
    features = checkpoint["features"]
    required = ["permno", "month"] + features
    if not set(required).issubset(frame):
        raise ValueError("Missing identifiers or trained characteristics.")
    d = frame.copy(); d["month"] = pd.to_datetime(d.month).dt.to_period("M").dt.to_timestamp()
    if d[["permno", "month"]].isna().any().any() or d.duplicated(["permno", "month"]).any():
        raise ValueError("Invalid or duplicate security-month keys.")
    if not all(pd.api.types.is_numeric_dtype(d[c]) for c in features) or np.isinf(d[features].to_numpy()).any():
        raise ValueError("Non-numeric/infinite characteristics.")
    d[features] = d[features].fillna(d.groupby("month")[features].transform("median")).fillna(0.)
    cfg = checkpoint["config"]
    model = SupervisedAutoencoder(len(features), cfg["hidden_dim"], cfg["latent_dim"], cfg["dropout"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    values = scaler.transform(d[features].to_numpy(dtype=np.float32))
    outputs = []
    with torch.no_grad():
        for start in range(0, len(d), 8192):
            pred, _, _ = model(torch.tensor(values[start:start+8192], dtype=torch.float32))
            outputs.append(pred.numpy().ravel())
    if not outputs:
        raise ValueError("Empty prediction input.")
    result = d[["permno", "month"]].copy(); result["pred_sae"] = np.concatenate(outputs)
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--scaler", required=True); p.add_argument("--output", required=True)
    a = p.parse_args()
    # Only load artifacts you trust; joblib is not a safe format for untrusted files.
    result = predict_months(pd.read_parquet(a.data), torch.load(a.checkpoint, map_location="cpu", weights_only=True), joblib.load(a.scaler))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False)
