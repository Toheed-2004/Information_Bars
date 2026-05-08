"""
Fully vectorized time series analysis for ALL strategies at once.
Operates on daily returns arrays with pure numpy - no pandas after input.
No per-strategy loops.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from bitpredict.common.stats.custom_1.utils import BatchedReturns
from bitpredict.common.stats.custom_1.config import ANN_FACTOR, SQRT_ANN


def calculate_time_series_analysis(batched: BatchedReturns) -> np.ndarray:
    """
    Batched time series analysis for ALL strategies in a single pass.
    Zero per-strategy Python loops.

    Parameters
    ----------
    batched : BatchedReturns
        Daily returns and balances with shape (max_days, n_strats) [TRANSPOSED]

    Returns
    -------
    Structured array (n_strats,) with all time series metrics
    """
    max_days, n_strats = batched.daily_returns_2d.shape

    daily_returns  = batched.daily_returns_2d   # (max_days, n_strats)
    daily_balances = batched.daily_balances_2d  # (max_days, n_strats)
    valid_mask     = batched.valid_mask_2d      # (max_days, n_strats)
    n_days         = batched.n_days_per_strat   # (n_strats,)

    has_data = n_days > 0

    # ── Window size (per strategy, then a common minimum for batched ops) ─
    window_sizes  = np.maximum(5, np.minimum(30, n_days // 3)).astype(np.int32)
    common_window = int(np.min(window_sizes[has_data])) if has_data.any() else 5
    num_windows   = np.maximum(n_days - common_window + 1, 0)  # (n_strats,)

    # ── Rolling window metrics ────────────────────────────────────────────
    # sliding_window_view requires a uniform axis length, so we use max_days.
    # We'll mask out-of-range windows afterwards.
    # Shape: (max_days - common_window + 1, n_strats, common_window)
    max_windows = max_days - common_window + 1

    (rolling_return_means, rolling_return_stds,
     rolling_vol_means,    rolling_vol_stds,
     rolling_sharpe_means, rolling_sharpe_stds,
     rolling_mdd_means,    rolling_mdd_stds,
     num_windows_arr) = _rolling_stats_batched(
        daily_returns, daily_balances, valid_mask,
        n_days, common_window, max_windows, n_strats,
    )

    # ── Derived metrics ───────────────────────────────────────────────────
    vol_of_vol = np.where(
        rolling_vol_means > 0,
        rolling_vol_stds / rolling_vol_means,
        0.0,
    )
    sharpe_consistency = np.where(
        np.abs(rolling_sharpe_means) > 0,
        1.0 - rolling_sharpe_stds / np.abs(rolling_sharpe_means),
        0.0,
    )
    return_consistency = np.where(
        np.abs(rolling_return_means) > 0,
        1.0 - rolling_return_stds / np.abs(rolling_return_means),
        0.0,
    )

    # ── Lag-1 autocorrelation — vectorized ───────────────────────────────
    lag1_autocorr = _lag1_autocorr_batched(daily_returns, valid_mask, n_days, n_strats)

    # ── Trend strength — vectorized OLS slope ────────────────────────────
    trend_strength = _trend_strength_batched(daily_balances, n_days, n_strats, max_days)

    # ── Build structured array ────────────────────────────────────────────
    dtype = [
        ('rolling_return_mean_pct',     'f8'), ('rolling_return_std_pct',     'f8'),
        ('rolling_volatility_mean_pct', 'f8'), ('rolling_volatility_std_pct', 'f8'),
        ('rolling_sharpe_mean',         'f8'), ('rolling_sharpe_std',         'f8'),
        ('rolling_max_dd_mean_pct',     'f8'), ('rolling_max_dd_std_pct',     'f8'),
        ('volatility_of_volatility',    'f8'), ('sharpe_consistency',         'f8'),
        ('return_consistency',          'f8'), ('lag1_autocorrelation',       'f8'),
        ('trend_strength',              'f8'), ('rolling_window_size',        'i4'),
        ('rolling_periods_count',       'i4'), ('total_analysis_periods',     'i4'),
    ]

    result = np.zeros(n_strats, dtype=dtype)
    result['rolling_return_mean_pct']     = rolling_return_means
    result['rolling_return_std_pct']      = rolling_return_stds
    result['rolling_volatility_mean_pct'] = rolling_vol_means   * 100
    result['rolling_volatility_std_pct']  = rolling_vol_stds    * 100
    result['rolling_sharpe_mean']         = rolling_sharpe_means
    result['rolling_sharpe_std']          = rolling_sharpe_stds
    result['rolling_max_dd_mean_pct']     = rolling_mdd_means
    result['rolling_max_dd_std_pct']      = rolling_mdd_stds
    result['volatility_of_volatility']    = vol_of_vol
    result['sharpe_consistency']          = sharpe_consistency
    result['return_consistency']          = return_consistency
    result['lag1_autocorrelation']        = lag1_autocorr
    result['trend_strength']              = trend_strength
    result['rolling_window_size']         = window_sizes
    result['rolling_periods_count']       = num_windows_arr
    result['total_analysis_periods']      = n_days.astype(np.int32)

    return result


# ── Rolling stats ─────────────────────────────────────────────────────────────

def _rolling_stats_batched(
    daily_returns:  np.ndarray,   # (max_days, n_strats)
    daily_balances: np.ndarray,   # (max_days, n_strats)
    valid_mask:     np.ndarray,   # (max_days, n_strats)
    n_days:         np.ndarray,   # (n_strats,)
    window:         int,
    max_windows:    int,
    n_strats:       int,
) -> tuple:
    """
    Compute all rolling aggregates across strategies with no Python loop.
    Uses sliding_window_view over the time axis, then masks invalid windows.
    """
    if max_windows <= 0:
        z = np.zeros(n_strats)
        return z, z, z, z, z, z, z, z, np.zeros(n_strats, dtype=np.int32)

    # ── Sliding windows over full padded arrays ───────────────────────────
    # ret_wins : (max_windows, n_strats, window)
    ret_wins = sliding_window_view(daily_returns,  window, axis=0)
    bal_wins = sliding_window_view(daily_balances, window, axis=0)
    val_wins = sliding_window_view(valid_mask.astype(np.float32), window, axis=0)

    # Mask out padding days inside each window
    ret_wins = np.where(val_wins, ret_wins, 0.0)

    # valid count per window — (max_windows, n_strats)
    win_n   = val_wins.sum(axis=2)
    win_n_s = np.maximum(win_n, 1.0)

    # ── Window-level stats ────────────────────────────────────────────────
    # means / stds
    win_sum    = ret_wins.sum(axis=2)                          # (max_windows, n_strats)
    win_sum_sq = (ret_wins ** 2).sum(axis=2)
    win_means  = win_sum / win_n_s
    win_var    = np.where(
        win_n > 1,
        (win_sum_sq - win_sum ** 2 / win_n_s) / np.maximum(win_n - 1, 1.0),
        0.0,
    )
    win_stds = np.sqrt(np.maximum(win_var, 0.0))

    # annualised return / vol / sharpe per window
    roll_ret_ann  = win_means * ANN_FACTOR                     # (max_windows, n_strats)
    roll_vol_ann  = win_stds  * SQRT_ANN
    roll_sharpe   = np.where(win_stds > 0, win_means / win_stds * SQRT_ANN, 0.0)

    # rolling max drawdown per window
    # cummax along the window axis
    cummax  = np.maximum.accumulate(bal_wins, axis=2)          # (max_windows, n_strats, window)
    dd_wins = np.where(
        (cummax > 0) & (val_wins > 0),
        (bal_wins - cummax) / cummax * 100,
        0.0,
    )
    roll_mdd = dd_wins.min(axis=2)                             # (max_windows, n_strats)

    # ── Window validity mask ──────────────────────────────────────────────
    # Window w is valid for strategy s iff w < n_days[s] - window + 1
    w_idx    = np.arange(max_windows)[:, np.newaxis]           # (max_windows, 1)
    win_valid = w_idx < num_windows_per_strat(n_days, window)[np.newaxis, :]
    # reshape for broadcast: (max_windows, n_strats)
    win_valid = (w_idx < (n_days - window + 1)[np.newaxis, :]) & (n_days > 0)[np.newaxis, :]

    num_valid = win_valid.sum(axis=0).astype(np.int32)         # (n_strats,)
    nv_f      = np.maximum(num_valid.astype(np.float64), 1.0)

    # ── Aggregate: mean / ddof-1 std over valid windows per strategy ──────
    def _agg(x):
        # x : (max_windows, n_strats)
        x_m  = np.where(win_valid, x, 0.0)
        mean = x_m.sum(axis=0) / nv_f
        sq   = np.where(win_valid, (x - mean[np.newaxis, :]) ** 2, 0.0)
        std  = np.sqrt(np.where(
            num_valid > 1,
            sq.sum(axis=0) / np.maximum(num_valid - 1, 1),
            0.0,
        ))
        return mean, std

    ret_mean,    ret_std    = _agg(roll_ret_ann)
    vol_mean,    vol_std    = _agg(roll_vol_ann)
    sharpe_mean, sharpe_std = _agg(roll_sharpe)
    mdd_mean,    mdd_std    = _agg(roll_mdd)

    # original code multiplied ret_mean by 100 a second time at assignment
    return (
        ret_mean    * 100, ret_std    * 100,
        vol_mean,          vol_std,
        sharpe_mean,       sharpe_std,
        mdd_mean,          mdd_std,
        num_valid,
    )


def num_windows_per_strat(n_days, window):
    """Helper — kept out of hot path."""
    return np.maximum(n_days - window + 1, 0)


# ── Lag-1 autocorrelation — vectorized ───────────────────────────────────────

def _lag1_autocorr_batched(
    daily_returns: np.ndarray,   # (max_days, n_strats)
    valid_mask:    np.ndarray,   # (max_days, n_strats)
    n_days:        np.ndarray,   # (n_strats,)
    n_strats:      int,
) -> np.ndarray:
    """
    Manual Pearson correlation between r[t] and r[t+1] for all strategies.

    Pearson(x, y) = (n*Σxy - Σx*Σy) / sqrt((n*Σx² - (Σx)²)(n*Σy² - (Σy)²))

    x = r[:-1],  y = r[1:]  — both masked to valid days.
    """
    max_days = daily_returns.shape[0]

    # Pair mask: both day d and d+1 must be valid
    v      = valid_mask.astype(np.float64)                     # (max_days, n_strats)
    pair_v = v[:-1] * v[1:]                                    # (max_days-1, n_strats)

    x = daily_returns[:-1] * pair_v                            # (max_days-1, n_strats)
    y = daily_returns[1:]  * pair_v

    n   = pair_v.sum(axis=0)                                   # (n_strats,)
    sx  = x.sum(axis=0)
    sy  = y.sum(axis=0)
    sxy = (x * y * pair_v).sum(axis=0)

    # Use pre-masked x/y — no need to re-mask
    sx2 = (x * x).sum(axis=0)
    sy2 = (y * y).sum(axis=0)

    num  = n * sxy - sx * sy
    den  = np.sqrt(
        np.maximum(n * sx2 - sx ** 2, 0.0) *
        np.maximum(n * sy2 - sy ** 2, 0.0)
    )
    corr = np.where((den > 0) & (n_days > 1), num / den, 0.0)
    return np.where(np.isfinite(corr), corr, 0.0)


# ── Trend strength — vectorized OLS slope ────────────────────────────────────

def _trend_strength_batched(
    daily_balances: np.ndarray,   # (max_days, n_strats)
    n_days:         np.ndarray,   # (n_strats,)
    n_strats:       int,
    max_days:       int,
) -> np.ndarray:
    """
    OLS slope of balance vs. time index, normalised by first balance.

    slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    trend_strength = slope / balance[0] * n
    """
    day_idx = np.arange(max_days, dtype=np.float64)            # (max_days,)

    # Valid mask
    valid   = (day_idx[:, np.newaxis] < n_days[np.newaxis, :]) # (max_days, n_strats)
    x       = np.where(valid, day_idx[:, np.newaxis], 0.0)     # (max_days, n_strats)
    y       = np.where(valid, daily_balances, 0.0)

    n   = n_days.astype(np.float64)
    sx  = x.sum(axis=0)                                        # (n_strats,)
    sy  = y.sum(axis=0)
    sxy = (x * y).sum(axis=0)
    sx2 = (x * x).sum(axis=0)

    denom = n * sx2 - sx ** 2
    slope = np.where(denom != 0.0, (n * sxy - sx * sy) / denom, 0.0)

    first_bal = daily_balances[0]                              # (n_strats,)
    trend_strength = np.where(
        (first_bal != 0.0) & (n > 1),
        slope / first_bal * n,
        0.0,
    )
    return trend_strength