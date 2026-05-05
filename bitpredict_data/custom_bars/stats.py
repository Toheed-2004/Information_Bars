"""
bars/stats.py — Bar quality statistics calculator.

Reads all bars for every active (exchange, symbol, bar_type) combination
and computes quality statistics, upserting results into
data_bars.bars_quality_stats.

Statistics are grouped into five categories:
  1. Coverage        — how much data exists and at what rate
  2. Sampling quality — how uniform bar sizes and durations are (core value
                        proposition of alternative bars vs time bars)
  3. Return distribution — moments and entropy of bar returns
  4. ML / IID quality — autocorrelation, variance ratio, stationarity proxy,
                        and effective sample size (directly relevant to AI)
  5. OHLC integrity  — data validity checks

All array computations are fully vectorised via NumPy; no Python-level loops
over individual rows.

Usage:
    python main.py stats
    python -m bitpredict.data.custom_bars.stats
"""

import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bitpredict.common.logging import get_logger


load_dotenv()
logger = get_logger(__name__)

# Minimum bars required for any stat to be meaningful
MIN_BARS = 50


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(val) -> Optional[float]:
    """Return a finite Python float, or None (for DB NULL) on NaN/inf/None."""
    if val is None:
        return None
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _col(df: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    """Extract a column as float64 array, filling missing column with default."""
    if name in df.columns:
        return df[name].to_numpy(dtype=np.float64)
    return np.full(len(df), default, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_bar_stats(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    bar_type: str,
) -> Dict[str, Any]:
    """Compute quality statistics from a bars DataFrame.

    All heavy lifting is vectorised — numpy operates on entire arrays at once.

    Parameters
    ----------
    df : pd.DataFrame
        All rows from one bar table (e.g. data_bars.bybit_btc_dollar).
    exchange, symbol, bar_type : str
        Identifiers written into the output dict.

    Returns
    -------
    dict  ready for ``upsert_bar_stats``, or empty dict if too few bars.
    """
    n = len(df)
    if n < MIN_BARS:
        logger.warning(
            "Skipping %s/%s/%s — only %d bars (minimum %d required)",
            exchange, symbol, bar_type, n, MIN_BARS,
        )
        return {}

    # ── Extract arrays (single pass) ─────────────────────────────────────────
    bar_size   = _col(df, "bar_size")
    duration   = _col(df, "duration_minutes")
    bar_return = _col(df, "bar_return")
    open_      = _col(df, "open")
    high_      = _col(df, "high")
    low_       = _col(df, "low")
    close_     = _col(df, "close")
    cp         = _col(df, "close_position")   # close position in [0,1]
    pr         = _col(df, "price_range")
    tc         = _col(df, "tick_count", default=1.0)

    # ── 1. Coverage ───────────────────────────────────────────────────────────
    start_col = "datetime_start" if "datetime_start" in df.columns else "datetime"
    end_col   = "datetime_end"   if "datetime_end"   in df.columns else "datetime"

    date_range_start = pd.to_datetime(df[start_col]).min()
    date_range_end   = pd.to_datetime(df[end_col]).max()
    calendar_days    = max(1, (date_range_end - date_range_start).days)

    # Bars-per-day: count bars per calendar date (vectorised groupby equivalent)
    end_dates = pd.to_datetime(df[end_col]).dt.normalize()   # midnight of each bar's end date
    bpd_counts = end_dates.value_counts()
    mean_bpd = float(bpd_counts.mean())
    std_bpd  = float(bpd_counts.std()) if len(bpd_counts) > 1 else 0.0

    # ── 2. Sampling quality ───────────────────────────────────────────────────
    bs = bar_size[np.isfinite(bar_size) & (bar_size > 0)]
    if len(bs) < 5:
        bs = bar_size[np.isfinite(bar_size)]

    if len(bs) > 0:
        bs_mean = float(np.mean(bs))
        bs_std  = float(np.std(bs))
        bs_cv   = bs_std / bs_mean if bs_mean > 0 else np.nan
        bs_p5, bs_p25, bs_p50, bs_p75, bs_p95 = (
            float(v) for v in np.percentile(bs, [5, 25, 50, 75, 95])
        )
    else:
        bs_mean = bs_std = bs_cv = np.nan
        bs_p5 = bs_p25 = bs_p50 = bs_p75 = bs_p95 = np.nan

    dur = duration[np.isfinite(duration) & (duration > 0)]
    if len(dur) > 0:
        dur_mean = float(np.mean(dur))
        dur_std  = float(np.std(dur))
        dur_cv   = dur_std / dur_mean if dur_mean > 0 else np.nan
        dur_p95  = float(np.percentile(dur, 95))
    else:
        dur_mean = dur_std = dur_cv = dur_p95 = np.nan

    tc_finite = tc[np.isfinite(tc)]
    tc_mean = float(np.mean(tc_finite)) if len(tc_finite) > 0 else np.nan

    # ── 3. Return distribution ────────────────────────────────────────────────
    ret = bar_return[np.isfinite(bar_return)]
    n_ret = len(ret)

    if n_ret > 0:
        ret_mean = float(np.mean(ret))
        ret_std  = float(np.std(ret))
    else:
        ret_mean = ret_std = np.nan

    if n_ret >= 4:
        mu        = np.mean(ret)
        residuals = ret - mu
        m2 = np.mean(residuals ** 2)
        m3 = np.mean(residuals ** 3)
        m4 = np.mean(residuals ** 4)
        ret_skew = float(m3 / m2 ** 1.5) if m2 > 0 else 0.0
        ret_kurt = float(m4 / m2 ** 2)   if m2 > 0 else 0.0
    else:
        ret_skew = ret_kurt = np.nan

    # Shannon entropy of the return distribution (bits)
    if n_ret >= 20:
        n_bins = min(50, max(10, n_ret // 5))
        hist, _ = np.histogram(ret, bins=n_bins)
        hist = hist[hist > 0].astype(np.float64)
        probs = hist / hist.sum()
        ret_entropy = float(-np.dot(probs, np.log2(probs)))
    else:
        ret_entropy = np.nan

    # ── 4. ML / IID quality ───────────────────────────────────────────────────
    # Lag-1 autocorrelation of returns  (independence test)
    if n_ret >= 10:
        r  = np.corrcoef(ret[:-1], ret[1:])[0, 1]
        return_autocorr = float(r) if np.isfinite(r) else 0.0

        abs_ret = np.abs(ret)
        ar = np.corrcoef(abs_ret[:-1], abs_ret[1:])[0, 1]
        abs_return_autocorr = float(ar) if np.isfinite(ar) else 0.0
    else:
        return_autocorr = abs_return_autocorr = np.nan

    # Variance ratio — random walk test
    # VR(q) = Var(q-bar return) / (q * Var(1-bar return))
    # VR ≈ 1.0 → random walk (no autocorrelation) → best for ML labels
    # VR > 1  → momentum / positive serial dependence
    # VR < 1  → mean-reversion / negative serial dependence
    var1 = np.var(ret) if n_ret >= 10 else 0.0

    if var1 > 0 and n_ret >= 20:
        # Lag-2: overlapping 2-bar returns (max power, standard Lo-MacKinlay)
        ret2 = ret[:-1] + ret[1:]
        vr2 = float(np.var(ret2) / (2.0 * var1))

        # Lag-5: non-overlapping windows (avoids double-counting bias)
        if n_ret >= 15:
            n5 = (n_ret // 5) * 5
            ret5 = ret[:n5].reshape(-1, 5).sum(axis=1)
            vr5 = float(np.var(ret5) / (5.0 * var1))
        else:
            vr5 = np.nan
    else:
        vr2 = vr5 = np.nan

    # Rolling volatility CV — distribution stationarity proxy
    # Divide bars into non-overlapping windows, compute std per window,
    # then CV of those window stds.  Low CV → stable regime → better for ML.
    if n_ret >= 30:
        win = max(10, n_ret // 10)          # ~10 windows
        n_complete = (n_ret // win) * win
        windows = ret[:n_complete].reshape(-1, win)
        wstds = np.std(windows, axis=1)     # one std per window (vectorised)
        wstd_mean = np.mean(wstds)
        rolling_vol_cv = float(np.std(wstds) / wstd_mean) if wstd_mean > 0 else np.nan
    else:
        rolling_vol_cv = np.nan

    # Effective sample size  ESS ≈ N / (1 + 2|ρ₁|)
    # Accounts for serial dependence reducing the number of truly independent
    # observations available for ML training.
    if n_ret >= 10 and np.isfinite(return_autocorr):
        eff_n = max(1, int(n_ret / max(1.0, 1.0 + 2.0 * abs(return_autocorr))))
    else:
        eff_n = n_ret

    # ── 5. OHLC integrity ─────────────────────────────────────────────────────
    # All conditions that must hold for a valid OHLC bar (vectorised bitmask)
    valid = (
        np.isfinite(open_)  & np.isfinite(high_) &
        np.isfinite(low_)   & np.isfinite(close_) &
        (open_ > 0)         & (close_ > 0) &
        (high_ >= np.maximum(open_, close_)) &
        (low_  <= np.minimum(open_, close_)) &
        (high_ >= low_)
    )
    pct_valid = float(np.mean(valid)) * 100.0

    cp_finite = cp[np.isfinite(cp)]
    cp_mean = float(np.mean(cp_finite)) if len(cp_finite) > 0 else np.nan
    cp_std  = float(np.std(cp_finite))  if len(cp_finite) > 0 else np.nan

    pr_finite = pr[np.isfinite(pr) & (pr >= 0)]
    pr_mean = float(np.mean(pr_finite)) if len(pr_finite) > 0 else np.nan

    # ── 6. Quality scores (0–100) ─────────────────────────────────────────────
    # Sampling score — bar_size_cv is the headline metric; lower = more uniform
    if np.isfinite(bs_cv):
        if bs_cv <= 0.15:
            sampling_score = 100.0
        elif bs_cv <= 0.40:
            sampling_score = 100.0 - ((bs_cv - 0.15) / 0.25) * 35.0
        else:
            sampling_score = max(0.0, 65.0 - ((bs_cv - 0.40) / 0.60) * 65.0)
    else:
        sampling_score = 50.0

    # ML score — autocorrelation penalty + entropy reward (each 0–50)
    if np.isfinite(return_autocorr):
        # abs autocorr near 0 → fully independent → 50pts
        autocorr_component = max(0.0, (1.0 - abs(return_autocorr)) * 50.0)
    else:
        autocorr_component = 25.0

    if np.isfinite(ret_entropy):
        # 6 bits ≈ well-distributed; cap at 50pts
        entropy_component = min(50.0, (ret_entropy / 6.0) * 50.0)
    else:
        entropy_component = 25.0

    ml_score = autocorr_component + entropy_component

    # Integrity score — already 0–100 from pct_valid
    integrity_score = pct_valid

    # Composite (weights: sampling 40%, ML 40%, integrity 20%)
    quality_score = sampling_score * 0.40 + ml_score * 0.40 + integrity_score * 0.20

    # ── Assemble output dict ──────────────────────────────────────────────────
    return {
        "exchange": exchange,
        "symbol":   symbol,
        "bar_type": bar_type,
        # coverage
        "total_bars":        n,
        "date_range_start":  date_range_start.to_pydatetime() if hasattr(date_range_start, "to_pydatetime") else date_range_start,
        "date_range_end":    date_range_end.to_pydatetime()   if hasattr(date_range_end,   "to_pydatetime") else date_range_end,
        "calendar_days":     calendar_days,
        "mean_bars_per_day": _f(mean_bpd),
        "std_bars_per_day":  _f(std_bpd),
        # sampling
        "bar_size_mean":  _f(bs_mean),
        "bar_size_std":   _f(bs_std),
        "bar_size_cv":    _f(bs_cv),
        "bar_size_p5":    _f(bs_p5),
        "bar_size_p25":   _f(bs_p25),
        "bar_size_p50":   _f(bs_p50),
        "bar_size_p75":   _f(bs_p75),
        "bar_size_p95":   _f(bs_p95),
        "duration_mean":  _f(dur_mean),
        "duration_std":   _f(dur_std),
        "duration_cv":    _f(dur_cv),
        "duration_p95":   _f(dur_p95),
        "tick_count_mean": _f(tc_mean),
        # return distribution
        "return_mean":     _f(ret_mean),
        "return_std":      _f(ret_std),
        "return_skew":     _f(ret_skew),
        "return_kurtosis": _f(ret_kurt),
        "return_entropy":  _f(ret_entropy),
        # ML / IID quality
        "return_autocorr_lag1":     _f(return_autocorr),
        "abs_return_autocorr_lag1": _f(abs_return_autocorr),
        "variance_ratio_lag2":      _f(vr2),
        "variance_ratio_lag5":      _f(vr5),
        "rolling_vol_cv":           _f(rolling_vol_cv),
        "eff_sample_size":          eff_n,
        # OHLC integrity
        "pct_valid_bars":      _f(pct_valid),
        "close_position_mean": _f(cp_mean),
        "close_position_std":  _f(cp_std),
        "price_range_mean":    _f(pr_mean),
        # scores
        "sampling_score":  _f(sampling_score),
        "ml_score":        _f(ml_score),
        "integrity_score": _f(integrity_score),
        "quality_score":   _f(quality_score),
        # metadata
        "bars_used":   n,
        "computed_at": datetime.now(timezone.utc),
    }
