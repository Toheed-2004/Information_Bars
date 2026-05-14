"""
mlfinlab.features.microstructural
====================================
Market-microstructure feature engineering from OHLCV bars.

Microstructural features capture the *information content* of price
formation and are largely orthogonal to classical technical indicators,
making them high-value inputs for financial ML models.

Features implemented
--------------------
bar_features          Rich feature set extracted from a single bar DataFrame.
roll_spread           Roll (1984) effective bid-ask spread estimator.
amihud_lambda         Amihud (2002) price-impact / illiquidity measure.
kyle_lambda           Kyle λ (OLS regression of Δprice on signed volume).
corwin_schultz_spread Corwin & Schultz (2012) high-low spread estimator.

References
----------
Roll, R. (1984). "A simple implicit measure of the bid-ask spread."
    *Journal of Finance*, 39(4), 1127-1139.
Amihud, Y. (2002). "Illiquidity and stock returns."
    *Journal of Financial Markets*, 5(1), 31-56.
Corwin, S. A. & Schultz, P. (2012). "A simple way to estimate bid-ask
    spreads from daily high and low prices."
    *Journal of Finance*, 67(2), 719-760.
de Prado, M. L. (2018). *Advances in Financial Machine Learning*, Ch.18-19.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# Bar-level feature extraction
# ---------------------------------------------------------------------------

def bar_features(
    bars: pd.DataFrame,
    log_price: bool = True,
) -> pd.DataFrame:
    """Extract a rich feature set from an OHLCV bar DataFrame.

    Parameters
    ----------
    bars : pd.DataFrame
        DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``volume``.  Index must be a DatetimeIndex.
    log_price : bool
        Compute log-price features in addition to raw price features.

    Returns
    -------
    pd.DataFrame  Feature matrix with the same DatetimeIndex.

    Features produced
    -----------------
    ``hl_spread``          (high − low) / close       – intra-bar range
    ``co_return``          log(close / open)           – intra-bar momentum
    ``oc_return``          log(open  / prev_close)     – gap return
    ``body_ratio``         |close − open| / (high − low + ε)
    ``upper_shadow``       (high − max(o,c)) / (high − low + ε)
    ``lower_shadow``       (min(o,c) − low) / (high − low + ε)
    ``log_volume``         log(volume + 1)
    ``log_dollar_volume``  log(volume * close + 1)
    ``vwap``               volume-weighted average price proxy
    ``ret_1`` … ``ret_5``  lagged log-returns
    ``vol_5`` … ``vol_20`` rolling std of log-returns
    ``rsi_14``             Relative Strength Index (14 bars)
    ``cs_spread``          Corwin-Schultz bid-ask spread
    ``amihud``             Amihud illiquidity
    ``autocorr_10``        10-bar return autocorrelation
    """
    df = bars.copy()
    df.columns = [c.lower() for c in df.columns]

    eps = 1e-9
    feat = pd.DataFrame(index=df.index)

    # --- intra-bar geometry
    hl = df["high"] - df["low"]
    feat["hl_spread"] = hl / (df["close"] + eps)
    feat["co_return"] = np.log(df["close"] / (df["open"] + eps))
    feat["oc_return"] = np.log(df["open"] / (df["close"].shift(1) + eps))
    feat["body_ratio"] = np.abs(df["close"] - df["open"]) / (hl + eps)
    feat["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (hl + eps)
    feat["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (hl + eps)

    # --- volume
    feat["log_volume"] = np.log1p(df["volume"])
    feat["log_dollar_volume"] = np.log1p(df["volume"] * df["close"])
    # Rolling VWAP (20-bar window) expressed as deviation from close.
    # The raw VWAP level is non-stationary (tracks price); deviation is not.
    # vwap_dev = (vwap - close) / close  -> stationary, bar-type-agnostic.
    tp = (df["high"] + df["low"] + df["close"]) / 3
    dv = tp * df["volume"]
    rolling_vwap = dv.rolling(20).sum() / (df["volume"].rolling(20).sum() + eps)
    feat["vwap_dev"] = (rolling_vwap - df["close"]) / (df["close"] + eps)

    # --- log-returns and lags
    log_ret = np.log(df["close"] / df["close"].shift(1))
    for lag in range(1, 6):
        feat[f"ret_{lag}"] = log_ret.shift(lag - 1)

    # --- rolling volatility
    for w in [5, 10, 20]:
        feat[f"vol_{w}"] = log_ret.rolling(w).std()

    # --- RSI
    feat["rsi_14"] = _rsi_series(df["close"], 14)

    # --- microstructure
    feat["cs_spread"] = corwin_schultz_spread(df)
    # Amihud illiquidity: |ret| / dollar_volume.
    # Raw value is ~1e-12 for BTC because dollar volumes are huge.
    # Multiply by 1e9 to bring to human-readable scale (~1e-3).
    # This is purely cosmetic rescaling; relative ordering across bar
    # types is unchanged and the feature is still meaningful.
    feat["amihud"] = amihud_lambda(df["close"], df["volume"]) * 1e9

    # --- autocorrelation of returns
    feat["autocorr_10"] = log_ret.rolling(20).apply(
        lambda x: x.autocorr(10) if len(x) >= 11 else np.nan, raw=False
    )

    if log_price:
        feat["log_close"] = np.log(df["close"])
        feat["log_open"] = np.log(df["open"])
        feat["log_high"] = np.log(df["high"])
        feat["log_low"] = np.log(df["low"])

    return feat


# ---------------------------------------------------------------------------
# Roll (1984) effective spread
# ---------------------------------------------------------------------------

def roll_spread(close: pd.Series) -> pd.Series:
    """Estimate the effective bid-ask spread using Roll's (1984) method.

    Spread = 2 * sqrt(-cov(ΔP_t, ΔP_{t-1}))

    When the covariance is positive (momentum), NaN is returned for
    that rolling window.

    Parameters
    ----------
    close : pd.Series
        Close prices.

    Returns
    -------
    pd.Series  Rolling effective spread estimate.
    """
    dP = close.diff()
    cov = dP.rolling(20).cov(dP.shift(1))
    spread = 2 * np.sqrt(np.maximum(-cov, 0))
    return spread.rename("roll_spread")


# ---------------------------------------------------------------------------
# Amihud (2002) illiquidity
# ---------------------------------------------------------------------------

def amihud_lambda(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute Amihud's (2002) illiquidity ratio.

    λ = |r_t| / dollar_volume_t

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Bar volume.
    window : int
        Rolling window (bars).

    Returns
    -------
    pd.Series  Rolling mean Amihud λ.
    """
    ret = np.abs(np.log(close / close.shift(1)))
    dollar_vol = (close * volume).replace(0, np.nan)
    illiq = ret / dollar_vol
    return illiq.rolling(window).mean().rename("amihud_lambda")


# ---------------------------------------------------------------------------
# Kyle (1985) lambda
# ---------------------------------------------------------------------------

def kyle_lambda(
    close: pd.Series,
    volume: pd.Series,
    buy_volume: pd.Series | None = None,
    window: int = 20,
) -> pd.Series:
    """Estimate Kyle's λ via OLS of Δprice on signed order-flow.

    Δprice_t = α + λ * signed_volume_t + ε_t

    When *buy_volume* is unavailable, the tick-rule is used to classify
    volume as buy (+) or sell (-).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Bar volume.
    buy_volume : pd.Series, optional
        Fraction of volume classified as buyer-initiated (0–1), or
        actual buy volume.  If None, tick rule is applied.
    window : int
        Rolling estimation window.

    Returns
    -------
    pd.Series  Rolling Kyle λ estimates.
    """
    price_change = close.diff()

    if buy_volume is None:
        # tick rule: +1 for uptick, -1 for downtick
        tick = np.sign(price_change).replace(0, np.nan).ffill().fillna(1)
        signed_vol = tick * volume
    else:
        sell_volume = volume - buy_volume
        signed_vol = buy_volume - sell_volume

    lambdas: list = []
    for i in range(window, len(close) + 1):
        sv = signed_vol.iloc[i - window : i].values
        dp = price_change.iloc[i - window : i].values
        mask = ~(np.isnan(sv) | np.isnan(dp))
        if mask.sum() < 5:
            lambdas.append(np.nan)
            continue
        slope, *_ = linregress(sv[mask], dp[mask])
        lambdas.append(slope)

    result = pd.Series(
        [np.nan] * window + lambdas[:-1] if len(lambdas) > 0 else [np.nan] * len(close),
        index=close.index,
        name="kyle_lambda",
    )
    # align length
    return result.reindex(close.index)


# ---------------------------------------------------------------------------
# Corwin & Schultz (2012) high-low spread
# ---------------------------------------------------------------------------

def corwin_schultz_spread(bars: pd.DataFrame) -> pd.Series:
    """Estimate the bid-ask spread from daily high-low prices.

    Derivation:
      β = E[ln(H/L)]²  over two consecutive single-day windows
      γ = E[ln(H_{2d}/L_{2d})]²  over each two-day window
      α = (√(2β) − √β) / (3 − 2√2) − √(γ / (3 − 2√2))
      Spread = 2 * (e^α − 1) / (1 + e^α)

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV DataFrame with lower-case columns ``high``, ``low``.

    Returns
    -------
    pd.Series  Per-bar bid-ask spread estimate (0 when formula produces
               complex numbers).
    """
    high = bars["high"]
    low = bars["low"]

    log_hl = np.log(high / low)
    beta = log_hl ** 2 + log_hl.shift(1) ** 2

    high2 = np.maximum(high, high.shift(1))
    low2 = np.minimum(low, low.shift(1))
    gamma = np.log(high2 / low2) ** 2

    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = alpha.clip(lower=0)

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.rename("cs_spread")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")