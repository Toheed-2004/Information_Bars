"""
Triple-barrier labeling (De Prado AFML Ch.3).
  +1  upper barrier hit first  (profit target)
  -1  lower barrier hit first  (stop loss)
   0  neither hit in horizon   (HOLD)

Barriers are volatility-scaled: width = rolling_sigma * multiplier.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def label_bars(
    df: pd.DataFrame,
    profit_target: float = 2.0,
    stop_loss: float = 1.5,
    max_holding_bars: int = 10,
    vol_lookback: int = 10,
) -> pd.Series:
    """
    Parameters
    ----------
    df               : DataFrame with close/high/low, sorted ascending.
    profit_target    : Upper barrier = close * (1 + sigma * profit_target)
    stop_loss        : Lower barrier = close * (1 - sigma * stop_loss)
    max_holding_bars : Vertical barrier length.
    vol_lookback     : Rolling window for sigma.

    Returns
    -------
    pd.Series of int8 {-1, 0, +1}, same index as df, name='label'.
    """
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    n  = len(c)

    log_r = np.log(c / np.concatenate([[c[0]], c[:-1]]))
    vol   = (pd.Series(log_r)
             .rolling(vol_lookback, min_periods=1)
             .std()
             .bfill()
             .to_numpy())

    labels = np.zeros(n, dtype=np.int8)

    for i in range(n):
        end = min(i + max_holding_bars, n - 1)
        if end <= i:
            continue
        ub = c[i] * (1.0 + vol[i] * profit_target)
        lb = c[i] * (1.0 - vol[i] * stop_loss)
        fh = h[i+1:end+1]
        fl = lo[i+1:end+1]
        if len(fh) == 0:
            continue
        ui = int(np.argmax(fh >= ub))
        li = int(np.argmax(fl <= lb))
        hu = bool(fh[ui] >= ub)
        hl = bool(fl[li] <= lb)
        if   hu and not hl: labels[i] =  1
        elif hl and not hu: labels[i] = -1
        elif hu and hl:     labels[i] =  1 if ui <= li else -1

    return pd.Series(labels, index=df.index, name="label", dtype=np.int8)
