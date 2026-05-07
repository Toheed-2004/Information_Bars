"""
features/engineer.py
--------------------
Feature engineering for the information bars research.

Design principle
----------------
Each bar type uses ALL its naturally available columns.
Calendar bars have only OHLCV — that is their structural limitation,
not something artificially imposed. Tick bars have buy_sell_imbalance,
precise vwap, tick density. Minute bars have estimated microstructure.

Only pure metadata is dropped (exchange, symbol, timestamps, index col).

Added on top of existing bar columns
-------------------------------------
RSI(14), MACD histogram (price-normalised), ATR(14) normalised,
Bollinger position and width, EMA(9/21/50) ratios, log returns at
1/3/5/10 horizons, volume z-score, lagged {rsi, macd_hist, log_ret_1,
buy_sell_imbalance} at 1/2/3 bars, plus fractionally differenced
close_fdiff and volume_fdiff when provided.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

_META = {"Unnamed: 0", "exchange", "symbol", "created_at",
         "datetime_start", "datetime_end"}


def engineer(
    df: pd.DataFrame,
    close_fdiff: Optional[pd.Series] = None,
    volume_fdiff: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df           : Bar DataFrame, datetime index, sorted ascending.
    close_fdiff  : Fractionally differenced close (optional).
    volume_fdiff : Fractionally differenced volume (optional).

    Returns
    -------
    pd.DataFrame of float32, NaN rows dropped.
    """
    out = df.copy()

    # 1. Drop metadata
    out.drop(columns=[c for c in _META if c in out.columns],
             inplace=True, errors="ignore")

    # 2. Encode string/category columns as dummies
    for col in out.select_dtypes(include=["object","category","string"]).columns:
        try:
            dummies = pd.get_dummies(out[col], prefix=col,
                                     drop_first=False, dtype=np.float32)
            out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
        except Exception:
            out.drop(columns=[col], inplace=True, errors="ignore")

    c  = df["close"].astype(float)
    h  = df["high"].astype(float)
    lo = df["low"].astype(float)
    v  = df["volume"].astype(float) if "volume" in df.columns else None

    # 3. Fractionally differenced series
    if close_fdiff is not None:
        out["close_fdiff"] = close_fdiff.reindex(out.index)
    if volume_fdiff is not None:
        out["volume_fdiff"] = volume_fdiff.reindex(out.index)

    # 4. RSI(14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    out["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) / 100

    # 5. MACD histogram (price-normalised, removes level dependency)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (macd - sig) / (c + 1e-10)

    # 6. ATR(14) normalised
    prev  = c.shift(1)
    tr    = pd.concat([h - lo, (h - prev).abs(), (lo - prev).abs()], axis=1).max(1)
    out["atr_norm"] = tr.rolling(14, min_periods=1).mean() / (c + 1e-10)

    # 7. Bollinger Bands
    mid = c.rolling(20, min_periods=1).mean()
    std = c.rolling(20, min_periods=1).std()
    out["bb_pos"]   = (c - (mid - 2*std)) / (4*std + 1e-10)
    out["bb_width"] = 4*std / (mid + 1e-10)

    # 8. EMA ratios
    for w in (9, 21, 50):
        out[f"ema_r{w}"] = c / (c.ewm(span=w, adjust=False).mean() + 1e-10) - 1

    # 9. Log returns at multiple horizons
    for h_ in (1, 3, 5, 10):
        out[f"ret_{h_}"] = np.log(c / (c.shift(h_) + 1e-10))

    # 10. Volume z-score
    if v is not None:
        vm = v.rolling(20, min_periods=1).mean()
        vs = v.rolling(20, min_periods=1).std()
        out["vol_z"] = (v - vm) / (vs + 1e-10)

    # 11. Lag features for key signals (prevents look-ahead, adds memory)
    lag_cols = [col for col in ["rsi", "macd_hist", "ret_1", "buy_sell_imbalance"]
                if col in out.columns]
    for col in lag_cols:
        for lag in (1, 2, 3):
            out[f"{col}_l{lag}"] = out[col].shift(lag)

    # 12. Final
    out = out.select_dtypes(include=[np.number])
    out.dropna(inplace=True)
    return out.astype(np.float32)
