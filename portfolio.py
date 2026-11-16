"""Equal-weight monthly top/bottom deciles with explicit tie handling."""
import numpy as np
import pandas as pd

def long_short_decile_returns(df_in, pred_col="pred", ret_col="ret_excess",
                             top_q=.9, bot_q=.1, min_n=30):
    if not 0 < bot_q < top_q < 1 or min_n < 2:
        raise ValueError("Invalid portfolio quantiles or minimum sample size.")
    if df_in[["year", "month"]].isna().any().any():
        raise ValueError("Missing portfolio calendar keys.")
    if "permno" in df_in and df_in.duplicated(["permno", "month"]).any():
        raise ValueError("Duplicate securities in a portfolio month.")
    rows = []
    for (year, month), g in df_in.groupby(["year", "month"], sort=True):
        g = g[np.isfinite(g[pred_col]) & np.isfinite(g[ret_col])]
        if len(g) < min_n:
            continue
        lo, hi = g[pred_col].quantile([bot_q, top_q])
        if lo >= hi:
            raise ValueError(f"Overlapping portfolio legs from tied predictions in {month}.")
        long = g[g[pred_col] >= hi][ret_col]
        short = g[g[pred_col] <= lo][ret_col]
        rows.append((year, month, long.mean(), short.mean(), long.mean()-short.mean(), len(long), len(short)))
    out = pd.DataFrame(rows, columns=["year", "month", "long", "short", "long_short", "n_long", "n_short"])
    out["cum_long_short"] = (1 + out.long_short).cumprod() - 1
    return out

def portfolio_summary(path):
    if path.empty:
        raise ValueError("No eligible portfolio months.")
    mu, sd = path.long_short.mean(), path.long_short.std(ddof=1)
    return dict(avg_monthly_long_short=mu, std_long_short=sd,
                sharpe_monthly=mu/sd if sd > 0 else np.nan,
                cum_long_short_end=path.cum_long_short.iloc[-1], n_months=len(path))
