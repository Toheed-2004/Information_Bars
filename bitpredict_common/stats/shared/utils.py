"""
Shared utility functions — single source of truth for helpers used by
both custom/ and vectorbt_pro/ stats modules.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Crypto annualization factor — 365.25 calendar days per year
ANN_FACTOR = 365


def _auto_detect_ann_factor(timestamps: np.ndarray) -> float:
    """
    Auto-detect annualization factor from a timestamps array.
    Uses VBT Pro's methodology. Falls back to ANN_FACTOR for short series.
    """
    if len(timestamps) <= 1:
        return ANN_FACTOR
    index = pd.to_datetime(timestamps)
    offset = index[0] + pd.offsets.YearBegin() - index[0]
    first_date = index[0] + offset
    last_date  = index[-1] + offset
    next_year  = last_date + pd.offsets.YearBegin()
    ratio = (last_date.value - first_date.value) / (next_year.value - first_date.value)
    if ratio == 0:
        return ANN_FACTOR
    ann = len(index) / ratio / (next_year.year - first_date.year)
    return float(ann)


def _bh_per_trade_returns(ledger: pd.DataFrame) -> np.ndarray:
    """Buy-and-hold return per trade: exit_price / entry_price - 1."""
    entry_px = ledger['avg_entry_price'].values.astype(float)
    exit_px  = ledger['avg_exit_price'].values.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(entry_px > 0, exit_px / entry_px - 1, 0.0)


def _max_consecutive_numpy(mask: np.ndarray) -> int:
    """Find maximum consecutive True values in a boolean array."""
    if len(mask) == 0:
        return 0
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0 or len(ends) == 0:
        return 0
    lengths = ends - starts
    return int(np.max(lengths)) if len(lengths) > 0 else 0


def _rolling_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Return a 2-D array of rolling windows using numpy stride tricks.
    Shape: (n - window + 1, window).  No copies, O(1) memory overhead.
    """
    if len(arr) < window:
        return np.empty((0, window), dtype=arr.dtype)
    return sliding_window_view(arr, window)


def _calculate_avg_return(returns: np.ndarray) -> float:
    """Mean of non-zero returns (QuantStats convention)."""
    if len(returns) == 0:
        return 0.0
    non_zero = returns[returns != 0]
    return float(np.mean(non_zero)) if len(non_zero) > 0 else 0.0


def _calculate_geometric_mean(returns: np.ndarray) -> float:
    """Geometric mean: prod(1+r)^(1/n) - 1."""
    if len(returns) == 0:
        return 0.0
    gross = np.maximum(1 + returns, 1e-10)
    try:
        return float(np.prod(gross) ** (1 / len(gross)) - 1)
    except (OverflowError, ValueError):
        return 0.0
