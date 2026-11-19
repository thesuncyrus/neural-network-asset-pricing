"""Model and evaluation functions extracted from the supplied project notebook.
Importing this module does not read data or start training.
"""
import random
from config import TrainConfig
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
target_col = "ret_excess"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fill_features_cs_median(data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    data = data.copy()
    # per-month median for each feature
    med = data.groupby("month")[feature_cols].transform("median")
    data[feature_cols] = data[feature_cols].fillna(med)
    # any remaining NaNs (month had all NaN): fill with 0
    data[feature_cols] = data[feature_cols].fillna(0.0)
    return data

class PanelDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, gid: np.ndarray=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        if gid is None:
            self.gid = None
        else:
            self.gid = torch.tensor(gid, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.gid is None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx], self.y[idx], self.gid[idx]

def make_loaders(train_df, val_df, feature_cols, target_col,
                 gid_col="ym_id", batch_size=4096, num_workers=0):

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_col].to_numpy(dtype=np.float32)
    g_train = train_df[gid_col].to_numpy(dtype=np.int64)

    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df[target_col].to_numpy(dtype=np.float32)
    g_val = val_df[gid_col].to_numpy(dtype=np.int64)

    train_ds = PanelDataset(X_train, y_train, g_train)
    val_ds = PanelDataset(X_val, y_val, g_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(DEVICE=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(DEVICE=="cuda"))
    return train_loader, val_loader

class SupervisedAutoencoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Predictor head (simple linear = "factor returns" mapping)
        self.predictor = nn.Linear(latent_dim, 1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.predictor(z)
        x_hat = self.decoder(z)
        return y_hat, x_hat, z


def pairwise_rank_loss(y_true, y_pred, group_id, n_pairs_per_group=256):
    """
    y_true, y_pred: shape [B]
    group_id: shape [B] month-id (same month -> same group)
    """
    loss_terms = []
    unique_g = torch.unique(group_id)
    for g in unique_g:
        idx = torch.where(group_id == g)[0]
        n = idx.numel()
        if n < 4:
            continue

        # sample random pairs inside this month
        i = idx[torch.randint(0, n, (n_pairs_per_group,), device=idx.device)]
        j = idx[torch.randint(0, n, (n_pairs_per_group,), device=idx.device)]

        yt_i, yt_j = y_true[i], y_true[j]
        yp_i, yp_j = y_pred[i], y_pred[j]

        s = torch.sign(yt_i - yt_j)  # +1 if i>j, -1 if i<j, 0 if tie
        mask = (s != 0)
        if mask.sum() < 10:
            continue

        diff = (yp_i - yp_j)[mask]
        s = s[mask]

        # logistic pairwise ranking loss
        loss_terms.append(F.softplus(-s * diff).mean())

    if len(loss_terms) == 0:
        return torch.tensor(0.0, device=y_true.device)
    return torch.stack(loss_terms).mean()

def train_one_model(train_df, val_df, feature_cols, target_col, cfg: TrainConfig):
    # Scale features (fit on train only!)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32))

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[feature_cols] = X_train
    val_scaled[feature_cols] = X_val

    train_loader, val_loader = make_loaders(
        train_scaled, val_scaled, feature_cols, target_col,
        gid_col="ym_id", batch_size=cfg.batch_size
    )

    model = SupervisedAutoencoder(
        in_dim=len(feature_cols),
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    best_ic = -1e9
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb, gb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).view(-1)
            gb = gb.to(DEVICE)

            optimizer.zero_grad()
            y_hat, x_hat, _ = model(xb)
            y_hat = y_hat.view(-1)

            loss_pred = mse(y_hat, yb)
            loss_rec = mse(x_hat, xb)
            loss_rank = pairwise_rank_loss(yb, y_hat, gb, n_pairs_per_group=256)

            loss = cfg.alpha_mse * loss_pred + cfg.beta_rank * loss_rank + cfg.lambda_rec * loss_rec
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation Rank IC (the metric you care about)
        model.eval()
        val_preds = []
        val_true = []
        val_year = []
        val_month = []

        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(DEVICE)
                y_hat, _, _ = model(xb)
                val_preds.append(y_hat.detach().cpu().numpy().reshape(-1))
                val_true.append(yb.detach().cpu().numpy().reshape(-1))

        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)

        tmp = val_scaled[["year", "month"]].copy()
        tmp["pred"] = val_preds
        tmp[target_col] = val_true
        val_ic = rank_ic_by_month(tmp, pred_col="pred", true_col=target_col)

        tr = float(np.mean(train_losses))
        print(f"Epoch {epoch:02d} | train loss {tr:.6f} | val rankIC {val_ic:.6f}")

        # Early stopping on Rank IC (maximize)
        if val_ic > best_ic + 1e-6:
            best_ic = val_ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print("Early stopping on rank IC.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_state is None:
        raise ValueError("No finite validation Rank IC; inspect validation predictions and targets.")
    return model, scaler

def predict_df(model, scaler, data_df, feature_cols, batch_size=8192):
    model.eval()
    X = scaler.transform(data_df[feature_cols].to_numpy(dtype=np.float32))
    ds = PanelDataset(
    X,
    data_df[target_col].to_numpy(dtype=np.float32),
    data_df["ym_id"].to_numpy(dtype=np.int64)
)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds = []
    zs = []
    with torch.no_grad():
        for xb, _, _ in loader:
            xb = xb.to(DEVICE)
            y_hat, _, z = model(xb)
            preds.append(y_hat.detach().cpu().numpy().reshape(-1))
            zs.append(z.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    zs = np.concatenate(zs, axis=0)
    return preds, zs

def oos_r2(y_true, y_pred):
    # Out-of-sample R^2 relative to predicting the mean of y_true
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else np.nan

def rank_ic_by_month(df_monthly: pd.DataFrame, pred_col="pred", true_col=target_col):
    # Spearman correlation each (year, month), then average
    ics = []
    for (y, m), g in df_monthly.groupby(["year", "month"]):
        g = g.dropna(subset=[pred_col, true_col])
        if len(g) < 3 or g[true_col].nunique() < 3:
            continue
        ic = g[[pred_col, true_col]].corr(method="spearman").iloc[0, 1]
        if pd.notnull(ic):
            ics.append(ic)
    return float(np.mean(ics)) if len(ics) else np.nan

def fit_predict_ridge(train_df, test_df, feature_cols, alpha=5.0):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    ytr = train_df[target_col].to_numpy(dtype=np.float32)
    Xte = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))

    model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return pred

def fit_predict_pca_ridge(train_df, test_df, feature_cols, n_components=8, alpha=5.0):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    ytr = train_df[target_col].to_numpy(dtype=np.float32)
    Xte = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))

    pca = PCA(n_components=n_components, random_state=42)
    Ztr = pca.fit_transform(Xtr)
    Zte = pca.transform(Xte)

    model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
    model.fit(Ztr, ytr)
    pred = model.predict(Zte)
    return pred

class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_predict_mlp(train_df, val_df, test_df, feature_cols,
                      hidden_dim=64, dropout=0.1, lr=1e-3, epochs=10, batch_size=4096):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    Xva = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32))
    Xte = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))

    ytr = train_df[target_col].to_numpy(dtype=np.float32)
    yva = val_df[target_col].to_numpy(dtype=np.float32)

    train_ds = PanelDataset(Xtr, ytr)
    val_ds = PanelDataset(Xva, yva)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLPRegressor(len(feature_cols), hidden_dim=hidden_dim, dropout=dropout).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    patience = 2

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            yhat = model(xb)
            loss = mse(yhat, yb)
            loss.backward()
            opt.step()

        model.eval()
        losses = []
        weights = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                yhat = model(xb)
                losses.append(mse(yhat, yb).item())
                weights.append(len(yb))
        va = float(np.average(losses, weights=weights))
        print(f"MLP Epoch {ep:02d} | val {va:.6f}")

        if va < best_val - 1e-6:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # predict test
    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
        pred = model(Xte_t).detach().cpu().numpy().reshape(-1)
    return pred

def encoder_effective_weights(model: SupervisedAutoencoder) -> np.ndarray:
    # encoder is [Linear, ReLU, Dropout, Linear]
    lin1 = model.encoder[0]
    lin2 = model.encoder[3]
    W1 = lin1.weight.detach().cpu().numpy()  # hidden x in
    W2 = lin2.weight.detach().cpu().numpy()  # latent x hidden
    W_eff = W2 @ W1  # latent x in
    return W_eff

def monthly_factor_premia(test_df: pd.DataFrame, z: np.ndarray, y: np.ndarray, months: np.ndarray):
    """
    For each month, run cross-sectional regression: y = a + b'z
    Return b_t time series (factor premia).
    """
    res = []
    df_tmp = pd.DataFrame({"month": months, "y": y})
    for j in range(z.shape[1]):
        df_tmp[f"z{j}"] = z[:, j]

    for m, g in df_tmp.groupby("month"):
        if len(g) < 200:
            continue
        X = g[[f"z{j}" for j in range(z.shape[1])]].to_numpy()
        yy = g["y"].to_numpy()
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, yy)
        b = reg.coef_
        res.append([m] + list(b))

    cols = ["month"] + [f"factor_{j}" for j in range(z.shape[1])]
    return pd.DataFrame(res, columns=cols).sort_values("month")
