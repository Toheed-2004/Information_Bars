"""
Fully vectorized long/short analysis for ALL strategies at once.
Operates directly on LedgerArrays with pure numpy - no pandas after input.
"""

import numpy as np
from bitpredict.common.stats.custom_1.utils import LedgerArrays, COL_ACC_RET


def calculate_long_short_analysis(stacked: LedgerArrays) -> np.ndarray:
    """
    Batched long/short analysis for ALL strategies in a single pass.
    
    Parameters
    ----------
    stacked : LedgerArrays
        Pre-stacked ledger arrays with shape (n_strats, max_trades, n_cols)
    
    Returns
    -------
    Structured array (n_strats,) with all long/short metrics
    """
    n_strats = len(stacked.names)
    max_trades = stacked.numeric_3d.shape[1]
    
    # Extract arrays
    acc_ret = stacked.numeric_3d[:, :, COL_ACC_RET]  # (n_strats, max_trades)
    sign = stacked.sign_2d  # (n_strats, max_trades) - 1.0 for long, -1.0 for short
    lengths = stacked.lengths  # (n_strats,)
    entry_ts = stacked.datetime_3d[:, :, 0]  # (n_strats, max_trades)
    exit_ts = stacked.datetime_3d[:, :, 1]  # (n_strats, max_trades)
    
    # Create validity mask
    valid_mask = np.arange(max_trades)[None, :] < lengths[:, None]  # (n_strats, max_trades)
    
    # ── Direction masks (vectorized) ──────────────────────────────────────
    long_mask = (sign == 1.0) & valid_mask  # (n_strats, max_trades)
    short_mask = (sign == -1.0) & valid_mask  # (n_strats, max_trades)
    
    # ── Trade counts (vectorized) ─────────────────────────────────────────
    long_count = np.sum(long_mask, axis=1)  # (n_strats,)
    short_count = np.sum(short_mask, axis=1)  # (n_strats,)
    total_count = lengths.astype(np.float64)  # (n_strats,)
    
    long_pct = np.where(total_count > 0, long_count / total_count * 100, 0.0)
    short_pct = np.where(total_count > 0, short_count / total_count * 100, 0.0)
    
    # ── Win/Loss counts (vectorized) ──────────────────────────────────────
    win_mask = acc_ret > 0  # (n_strats, max_trades)
    loss_mask = acc_ret < 0  # (n_strats, max_trades)
    
    long_wins = np.sum(long_mask & win_mask, axis=1)  # (n_strats,)
    long_losses = np.sum(long_mask & loss_mask, axis=1)  # (n_strats,)
    short_wins = np.sum(short_mask & win_mask, axis=1)  # (n_strats,)
    short_losses = np.sum(short_mask & loss_mask, axis=1)  # (n_strats,)
    
    # ── Win rates (vectorized) ────────────────────────────────────────────
    long_win_rate = np.where(long_count > 0, long_wins / long_count * 100, 0.0)
    short_win_rate = np.where(short_count > 0, short_wins / short_count * 100, 0.0)
    
    # ── Duration analysis (vectorized) ────────────────────────────────────
    # Duration in days: (exit_ts - entry_ts) / nanoseconds_per_day
    NS_PER_DAY = np.int64(86_400_000_000_000)
    durations_days = (exit_ts - entry_ts).astype(np.float64) / NS_PER_DAY  # (n_strats, max_trades)
    
    # Masked mean for long/short durations
    long_durations = np.where(long_mask, durations_days, np.nan)
    short_durations = np.where(short_mask, durations_days, np.nan)
    
    long_avg_duration_days = np.nanmean(long_durations, axis=1)  # (n_strats,)
    short_avg_duration_days = np.nanmean(short_durations, axis=1)  # (n_strats,)
    
    # Replace NaN with 0.0 for strategies with no long/short trades
    long_avg_duration_days = np.where(np.isnan(long_avg_duration_days), 0.0, long_avg_duration_days)
    short_avg_duration_days = np.where(np.isnan(short_avg_duration_days), 0.0, short_avg_duration_days)
    
    # ── PnL analysis (vectorized) ─────────────────────────────────────────
    # Masked PnL arrays
    long_pnl = np.where(long_mask, acc_ret, np.nan)  # (n_strats, max_trades)
    short_pnl = np.where(short_mask, acc_ret, np.nan)  # (n_strats, max_trades)
    
    # Total PnL
    long_total_pnl = np.nansum(long_pnl, axis=1)  # (n_strats,)
    short_total_pnl = np.nansum(short_pnl, axis=1)  # (n_strats,)
    
    # Average PnL
    long_avg_pnl = np.nanmean(long_pnl, axis=1)  # (n_strats,)
    short_avg_pnl = np.nanmean(short_pnl, axis=1)  # (n_strats,)
    
    # Replace NaN with 0.0
    long_avg_pnl = np.where(np.isnan(long_avg_pnl), 0.0, long_avg_pnl)
    short_avg_pnl = np.where(np.isnan(short_avg_pnl), 0.0, short_avg_pnl)
    
    # Best/Worst trades
    long_best = np.where(long_count > 0, np.nanmax(long_pnl, axis=1), 0.0)
    long_worst = np.where(long_count > 0, np.nanmin(long_pnl, axis=1), 0.0)
    short_best = np.where(short_count > 0, np.nanmax(short_pnl, axis=1), 0.0)
    short_worst = np.where(short_count > 0, np.nanmin(short_pnl, axis=1), 0.0)
    
    # ── Build structured array (VBT-compatible naming) ────────────────────
    dtype = [
        ('long_trades_count', 'i8'), ('short_trades_count', 'i8'),
        ('long_trades_pct', 'f8'), ('short_trades_pct', 'f8'),
        ('long_winning_trades', 'i8'), ('long_losing_trades', 'i8'),
        ('short_winning_trades', 'i8'), ('short_losing_trades', 'i8'),
        ('long_win_rate_pct', 'f8'), ('short_win_rate_pct', 'f8'),  # VBT naming
        ('long_avg_duration_days', 'f8'), ('short_avg_duration_days', 'f8'),
        ('long_total_pnl_pct', 'f8'), ('short_total_pnl_pct', 'f8'),
        ('long_avg_pnl_pct', 'f8'), ('short_avg_pnl_pct', 'f8'),
        ('long_best_trade_pct', 'f8'), ('long_worst_trade_pct', 'f8'),  # VBT naming
        ('short_best_trade_pct', 'f8'), ('short_worst_trade_pct', 'f8'),  # VBT naming
    ]
    
    result = np.zeros(n_strats, dtype=dtype)
    result['long_trades_count'] = long_count.astype(np.int64)
    result['short_trades_count'] = short_count.astype(np.int64)
    result['long_trades_pct'] = long_pct
    result['short_trades_pct'] = short_pct
    result['long_winning_trades'] = long_wins.astype(np.int64)
    result['long_losing_trades'] = long_losses.astype(np.int64)
    result['short_winning_trades'] = short_wins.astype(np.int64)
    result['short_losing_trades'] = short_losses.astype(np.int64)
    result['long_win_rate_pct'] = long_win_rate
    result['short_win_rate_pct'] = short_win_rate
    result['long_avg_duration_days'] = long_avg_duration_days
    result['short_avg_duration_days'] = short_avg_duration_days
    result['long_total_pnl_pct'] = long_total_pnl
    result['short_total_pnl_pct'] = short_total_pnl
    result['long_avg_pnl_pct'] = long_avg_pnl
    result['short_avg_pnl_pct'] = short_avg_pnl
    result['long_best_trade_pct'] = long_best
    result['long_worst_trade_pct'] = long_worst
    result['short_best_trade_pct'] = short_best
    result['short_worst_trade_pct'] = short_worst
    
    return result
