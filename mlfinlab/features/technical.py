"""
mlfinlab.features.technical
==============================
Classical technical analysis features adapted for bar-type-agnostic use.

All functions accept a pandas Series / DataFrame indexed by datetime and
return pandas objects with the same index.  They can be applied to any
bar type (time, tick, volume, dollar, imbalance, etc.).

Features
--------
rsi              Relative Strength Index
macd             MACD + signal + histogram
bollinger_bands  Bollinger bands (upper, mid, lower, bandwidth, %B)
atr              Average True Range (normalized and raw)
vwap             Rolling VWAP from OHLCV bars
zscore           Rolling Z-score of any series
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI using EWM (equivalent to Wilder smoothing).

    Parameters
    ----------
    close : pd.Series
    period : int

    Returns
    -------
    pd.Series  RSI in [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD, signal line, and histogram.

    Parameters
    ----------
    close : pd.Series
    fast, slow, signal : int

    Returns
    -------
    pd.DataFrame  columns: ``macd``, ``signal``, ``histogram``.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands and derived metrics.

    Parameters
    ----------
    close : pd.Series
    period : int
    std_dev : float  Number of standard deviations for the bands.

    Returns
    -------
    pd.DataFrame  columns: ``upper``, ``mid``, ``lower``,
                            ``bandwidth``, ``pct_b``.
    """
    mid = close.rolling(period).mean()
    sigma = close.rolling(period).std(ddof=0)
    upper = mid + std_dev * sigma
    lower = mid - std_dev * sigma
    bandwidth = (upper - lower) / (mid + 1e-12)
    pct_b = (close - lower) / (upper - lower + 1e-12)
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_mid": mid,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_pct_b": pct_b,
        },
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Average True Range
# ---------------------------------------------------------------------------

def atr(
    bars: pd.DataFrame,
    period: int = 14,
    normalized: bool = True,
) -> pd.DataFrame:
    """Compute ATR (raw) and optionally the normalized ATR (NATR = ATR/close).

    Parameters
    ----------
    bars : pd.DataFrame  OHLCV with lower-case column names.
    period : int
    normalized : bool  Also return NATR.

    Returns
    -------
    pd.DataFrame  columns: ``atr`` [, ``natr``].
    """
    df = bars.copy()
    df.columns = [c.lower() for c in df.columns]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_series = tr.ewm(com=period - 1, adjust=False).mean()
    out = pd.DataFrame({"atr": atr_series}, index=bars.index)
    if normalized:
        out["natr"] = out["atr"] / (df["close"] + 1e-12)
    return out


# ---------------------------------------------------------------------------
# Rolling VWAP
# ---------------------------------------------------------------------------

def vwap(
    bars: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Rolling Volume-Weighted Average Price.

    Typical price = (H + L + C) / 3.

    Parameters
    ----------
    bars : pd.DataFrame  OHLCV.
    window : int

    Returns
    -------
    pd.Series
    """
    df = bars.copy()
    df.columns = [c.lower() for c in df.columns]
    tp = (df["high"] + df["low"] + df["close"]) / 3
    dollar_vol = tp * df["volume"]
    rolling_vwap = dollar_vol.rolling(window).sum() / (
        df["volume"].rolling(window).sum() + 1e-12
    )
    return rolling_vwap.rename(f"vwap_{window}")


# ---------------------------------------------------------------------------
# Rolling Z-score
# ---------------------------------------------------------------------------

def zscore(
    series: pd.Series,
    window: int = 20,
    demean: bool = True,
) -> pd.Series:
    """Compute rolling Z-score.

    Z_t = (x_t − μ_t) / σ_t   where μ and σ are rolling statistics.

    Parameters
    ----------
    series : pd.Series
    window : int
    demean : bool  Subtract rolling mean (True) or only scale by std.

    Returns
    -------
    pd.Series
    """
    mu = series.rolling(window).mean() if demean else 0.0
    sigma = series.rolling(window).std(ddof=1) + 1e-12
    return ((series - mu) / sigma).rename(f"zscore_{window}")
