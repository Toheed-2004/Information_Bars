"""
Fractional differencing (De Prado AFML Ch.5).
Finds minimum d making series stationary while preserving maximum memory.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple


def _weights(d: float, size: int, thresh: float = 1e-5, max_window: int = 200) -> np.ndarray:
    w = [1.0]
    for k in range(1, min(size, max_window)):
        w.append(-w[-1] * (d - k + 1) / k)
        if abs(w[-1]) < thresh:
            break
    return np.array(w[::-1], dtype=np.float64)


def frac_diff(series: pd.Series, d: float, thresh: float = 1e-5, max_window: int = 200) -> pd.Series:
    w   = _weights(d, len(series), thresh, max_window=max_window)
    wid = len(w)
    arr = series.to_numpy(np.float64)
    out = np.full(len(arr), np.nan)
    for i in range(wid - 1, len(arr)):
        out[i] = float(np.dot(w, arr[i - wid + 1: i + 1]))
    return pd.Series(out, index=series.index)


def find_min_d(
    series: pd.Series,
    d_min: float = 0.1,
    d_max: float = 1.0,
    d_step: float = 0.1,
    adf_sig: float = 0.05,
    thresh: float = 1e-5,
) -> Tuple[float, pd.Series]:
    """
    Return (d, differenced_series) for minimum d that passes ADF.
    Falls back to d=1.0 if nothing passes.
    """
    from statsmodels.tsa.stattools import adfuller
    for d in np.arange(d_min, d_max + 1e-9, d_step):
        d = round(float(d), 2)
        fd = frac_diff(series, d, thresh).dropna()
        if len(fd) < 20:
            continue
        try:
            pval = adfuller(fd, maxlag=1, autolag=None)[1]
            if pval <= adf_sig:
                full = frac_diff(series, d, thresh)
                return d, full
        except Exception:
            continue
    return d_max, frac_diff(series, d_max, thresh)
