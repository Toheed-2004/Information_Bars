"""
mlfinlab.data.synthetic
========================
Generate realistic synthetic OHLCV data for module testing.

Produces bars with:
  * GBM-driven close prices
  * Consistent OHLC (open = prev close + gap, H/L from intra-bar noise)
  * Lognormal volume correlated weakly with volatility
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(
    n_bars: int = 1000,
    freq: str = "1h",
    start: str = "2020-01-01",
    mu: float = 0.0001,
    sigma: float = 0.01,
    vol_scale: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV bar data.

    Parameters
    ----------
    n_bars   : int   Number of bars.
    freq     : str   Bar frequency (pandas offset alias).
    start    : str   Start datetime string.
    mu       : float Drift per bar.
    sigma    : float Volatility per bar.
    vol_scale: float Controls intra-bar high-low range relative to sigma.
    seed     : int   Random seed.

    Returns
    -------
    pd.DataFrame  columns: open, high, low, close, volume
                  index: DatetimeIndex
    """
    rng = np.random.default_rng(seed)

    idx = pd.date_range(start=start, periods=n_bars, freq=freq)

    # --- close prices via GBM
    log_ret = rng.normal(mu, sigma, n_bars)
    log_p = np.cumsum(log_ret)
    close = 100.0 * np.exp(log_p)

    # --- open = previous close with tiny gap
    open_ = np.empty(n_bars)
    open_[0] = close[0] * np.exp(rng.normal(0, sigma * 0.1))
    open_[1:] = close[:-1] * np.exp(rng.normal(0, sigma * 0.1, n_bars - 1))

    # --- high/low: extend open/close by random intra-bar move
    intra = np.abs(rng.normal(0, sigma * vol_scale, n_bars))
    oc_max = np.maximum(open_, close)
    oc_min = np.minimum(open_, close)
    high = oc_max * np.exp(intra)
    low = oc_min / np.exp(intra)

    # --- volume: log-normal, slightly anti-correlated with volatility
    vol_factor = 1 + 3 * np.abs(log_ret) / sigma
    volume = (
        rng.lognormal(mean=10, sigma=0.5, size=n_bars) * vol_factor
    ).astype(int)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return df
