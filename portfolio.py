import numpy as np
import pandas as pd

target_col = "ret_excess"

# Equal-weight top/bottom 10% portfolios, formed independently each month.
# Quantile ties are included; realized leg sizes may slightly exceed 10%.

def long_short_decile_returns(df_in: pd.DataFrame, pred_col="pred", ret_col=target_col,
                              top_q=0.9, bot_q=0.1, min_n=30):
    out = []
    for (y, m), g in df_in.groupby(["year", "month"]):
        g = g.dropna(subset=[pred_col, ret_col])
        if len(g) < min_n:
            continue

        q_hi = g[pred_col].quantile(top_q)
        q_lo = g[pred_col].quantile(bot_q)

        long = g.loc[g[pred_col] >= q_hi, ret_col].mean()
        short = g.loc[g[pred_col] <= q_lo, ret_col].mean()
        ls = long - short
        out.append((y, m, long, short, ls))

    res = pd.DataFrame(out, columns=["year", "month", "long", "short", "long_short"])
    res = res.sort_values(["year", "month"]).reset_index(drop=True)
    res["cum_long_short"] = (1.0 + res["long_short"]).cumprod() - 1.0
    return res