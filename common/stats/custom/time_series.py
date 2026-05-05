import numpy as np
from typing import Dict, Any
from ..shared.utils import _rolling_windows

def _get_empty_time_series_analysis() -> Dict[str, float]:
    return {
        'rolling_return_mean_pct': 0.0, 'rolling_return_std_pct': 0.0,
        'rolling_volatility_mean': 0.0, 'rolling_volatility_std': 0.0,
        'rolling_volatility_mean_pct': 0.0, 'rolling_volatility_std_pct': 0.0,
        'rolling_sharpe_mean': 0.0, 'rolling_sharpe_std': 0.0,
        'rolling_max_dd_mean': 0.0, 'rolling_max_dd_std': 0.0,
        'volatility_of_volatility': 0.0, 'sharpe_consistency': 0.0,
        'return_consistency': 0.0, 'lag1_autocorrelation': 0.0,
        'trend_strength': 0.0, 'rolling_window_size': 0,
        'rolling_periods_count': 0, 'total_analysis_periods': 0
    }

def _calculate_time_series_analysis(
    portfolio_returns: np.ndarray,
    balance_array: np.ndarray,
    timestamps: np.ndarray,
    ann_factor: float
) -> Dict[str, Any]:
    """Vectorized time series analysis using sliding_window_view."""

    if len(portfolio_returns) == 0 or len(balance_array) == 0:
        return _get_empty_time_series_analysis()

    n = len(portfolio_returns)
    window_size = max(5, min(30, n // 3))

    # Rolling windows over returns (shape: num_windows × window_size)
    ret_windows = _rolling_windows(portfolio_returns, window_size)  # (m, w)
    num_windows = len(ret_windows)

    if num_windows == 0:
        return _get_empty_time_series_analysis()

    # Rolling means and stds
    roll_means = np.mean(ret_windows, axis=1)          # (m,)
    roll_stds = np.std(ret_windows, axis=1, ddof=1)    # (m,)

    rolling_returns = roll_means * ann_factor
    rolling_vols = roll_stds * np.sqrt(ann_factor)
    # Sharpe per window (0 where std==0)
    rolling_sharpes = np.where(roll_stds > 0, roll_means / roll_stds * np.sqrt(ann_factor), 0.0)

    # Rolling max drawdown — requires iterating over balance sub-windows
    bal_windows = _rolling_windows(balance_array, window_size)  # (m, w)
    cummax = np.maximum.accumulate(bal_windows, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dd_windows = np.where(cummax > 0, (bal_windows - cummax) / cummax * 100, 0.0)
    rolling_mdd = np.min(dd_windows, axis=1)

    def _smean(a): return float(np.mean(a)) if len(a) > 0 else 0.0
    def _sstd(a): return float(np.std(a, ddof=1)) if len(a) > 1 else 0.0

    rrm = _smean(rolling_returns) * 100
    rrs = _sstd(rolling_returns) * 100
    rvm = _smean(rolling_vols)
    rvs = _sstd(rolling_vols)
    rsm = _smean(rolling_sharpes)
    rss = _sstd(rolling_sharpes)
    rmm = _smean(rolling_mdd)
    rms = _sstd(rolling_mdd)

    # Lag-1 autocorrelation
    lag1 = 0.0
    if n > 1:
        c = np.corrcoef(portfolio_returns[:-1], portfolio_returns[1:])[0, 1]
        lag1 = float(c) if not np.isnan(c) else 0.0

    # Trend strength
    trend = 0.0
    if len(balance_array) > 1 and balance_array[0] != 0:
        x = np.arange(len(balance_array), dtype=float)
        slope = np.polyfit(x, balance_array, 1)[0]
        trend = float(slope / balance_array[0] * len(balance_array))

    return {
        'rolling_return_mean_pct': rrm,
        'rolling_return_std_pct': rrs,
        'rolling_volatility_mean': rvm,
        'rolling_volatility_mean_pct': rvm * 100,
        'rolling_volatility_std': rvs,
        'rolling_volatility_std_pct': rvs * 100,
        'rolling_sharpe_mean': rsm,
        'rolling_sharpe_std': rss,
        'rolling_max_dd_mean': rmm,
        'rolling_max_dd_std': rms,
        'volatility_of_volatility': float(rvs / rvm) if rvm > 0 else 0.0,
        'sharpe_consistency': float(1.0 - (rss / abs(rsm))) if rsm != 0 else 0.0,
        'return_consistency': float(1.0 - (rrs / abs(rrm))) if rrm != 0 else 0.0,
        'lag1_autocorrelation': lag1,
        'trend_strength': trend,
        'rolling_window_size': int(window_size),
        'rolling_periods_count': int(num_windows),
        'total_analysis_periods': int(n),
    }
