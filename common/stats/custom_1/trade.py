"""
Fully vectorized trade analysis for custom_1.
Operates directly on LedgerArrays — batched computation for all strategies.
No per-strategy loop except streak detection (inherently sequential).
"""
import numpy as np
from bitpredict.common.stats.custom_1.config import NS_PER_DAY


# ── Streak detection — unavoidably per-strategy ───────────────────────────────

def _max_consecutive_batched(mask_2d: np.ndarray, trade_valid: np.ndarray) -> np.ndarray:
    """
    Max consecutive True values for each row of mask_2d.

    Parameters
    ----------
    mask_2d     : (n_strats, max_trades) bool
    trade_valid : (n_strats, max_trades) bool  — which cells are real trades

    Returns
    -------
    (n_strats,) int32
    """
    n_strats = mask_2d.shape[0]
    out = np.zeros(n_strats, dtype=np.int32)
    for s in range(n_strats):
        m = mask_2d[s, trade_valid[s]]
        if len(m) == 0:
            continue
        padded = np.empty(len(m) + 2, dtype=np.int8)
        padded[0] = 0
        padded[1:-1] = m.view(np.int8)
        padded[-1] = 0
        d = np.diff(padded)
        starts = np.flatnonzero(d == 1)
        ends   = np.flatnonzero(d == -1)
        if starts.size:
            out[s] = int((ends - starts).max())
    return out


# ── Main entry point ──────────────────────────────────────────────────────────

def calculate_trade_analysis(stacked) -> np.ndarray:
    """
    Calculate trade analysis for ALL strategies in batched mode.

    Parameters
    ----------
    stacked : LedgerArrays namedtuple

    Returns
    -------
    Structured array (n_strats,) with all trade metrics
    """
    n_strats   = len(stacked.names)
    lengths    = stacked.lengths                               # (n_strats,)
    max_trades = stacked.numeric_3d.shape[1]

    # ── Valid mask ────────────────────────────────────────────────────────
    trade_idx   = np.arange(max_trades)[np.newaxis, :]         # (1, max_trades)
    trade_valid = trade_idx < lengths[:, np.newaxis]           # (n_strats, max_trades)
    n_valid     = lengths.astype(np.float64)                   # (n_strats,) float for division

    # ── Raw arrays ───────────────────────────────────────────────────────
    returns_raw = stacked.numeric_3d[:, :, 6] / 100.0         # (n_strats, max_trades)
    acc_ret_raw = stacked.numeric_3d[:, :, 5] / 100.0         # (n_strats, max_trades)
    balance_raw = stacked.numeric_3d[:, :, 7]                 # (n_strats, max_trades)
    entry_ts    = stacked.datetime_3d[:, :, 0]                # (n_strats, max_trades)
    exit_ts     = stacked.datetime_3d[:, :, 1]                # (n_strats, max_trades)

    # Zero out padding so it doesn't affect sums/means
    r   = np.where(trade_valid, returns_raw, 0.0)              # (n_strats, max_trades)
    bal = np.where(trade_valid, balance_raw, 0.0)
    dur = np.where(trade_valid,
                   (exit_ts - entry_ts) / NS_PER_DAY, 0.0)    # (n_strats, max_trades)

    # ── Win / loss masks ──────────────────────────────────────────────────
    win_mask  = trade_valid & (returns_raw > 0)                # (n_strats, max_trades)
    loss_mask = trade_valid & (returns_raw < 0)

    n_win  = win_mask.sum(axis=1).astype(np.float64)           # (n_strats,)
    n_loss = loss_mask.sum(axis=1).astype(np.float64)

    # ── Basic rates ───────────────────────────────────────────────────────
    has_trades = lengths > 0
    win_rate  = np.where(has_trades, n_win  / n_valid * 100.0, 0.0)
    loss_rate = np.where(has_trades, n_loss / n_valid * 100.0, 0.0)

    # ── Best / worst ──────────────────────────────────────────────────────
    r_nan = np.where(trade_valid, returns_raw, np.nan)
    best_trade  = np.where(has_trades, np.nanmax(r_nan, axis=1) * 100, 0.0)
    worst_trade = np.where(has_trades, np.nanmin(r_nan, axis=1) * 100, 0.0)

    # ── Avg win / loss returns ────────────────────────────────────────────
    win_sum  = np.where(win_mask,  r, 0.0).sum(axis=1)        # (n_strats,)
    loss_sum = np.where(loss_mask, r, 0.0).sum(axis=1)
    avg_win  = np.where(n_win  > 0, win_sum  / np.maximum(n_win,  1) * 100, 0.0)
    avg_loss = np.where(n_loss > 0, loss_sum / np.maximum(n_loss, 1) * 100, 0.0)

    # ── Durations ─────────────────────────────────────────────────────────
    win_dur_sum  = np.where(win_mask,  dur, 0.0).sum(axis=1)
    loss_dur_sum = np.where(loss_mask, dur, 0.0).sum(axis=1)
    avg_win_dur  = np.where(n_win  > 0, win_dur_sum  / np.maximum(n_win,  1), 0.0)
    avg_loss_dur = np.where(n_loss > 0, loss_dur_sum / np.maximum(n_loss, 1), 0.0)

    dur_sum = dur.sum(axis=1)
    avg_dur = np.where(has_trades, dur_sum / n_valid, 0.0)

    # ddof=1 std via: var = (sum_sq - sum^2/n) / (n-1)
    dur_sum_sq = (dur ** 2).sum(axis=1)
    dur_var    = np.where(
        n_valid > 1,
        (dur_sum_sq - dur_sum ** 2 / np.maximum(n_valid, 1)) / np.maximum(n_valid - 1, 1),
        0.0
    )
    dur_std = np.sqrt(np.maximum(dur_var, 0.0))

    # ── Total PnL ─────────────────────────────────────────────────────────
    # initial capital = balance[first_trade] / (1 + acc_ret[first_trade])
    first_acc = acc_ret_raw[:, 0]                              # (n_strats,)
    first_bal = balance_raw[:, 0]
    denom     = 1.0 + first_acc
    initial   = np.where(denom != 0.0, first_bal / denom, first_bal)

    # last valid balance per strategy — gather with fancy index
    last_idx  = np.maximum(lengths - 1, 0).astype(int)        # (n_strats,)
    last_bal  = balance_raw[np.arange(n_strats), last_idx]
    total_pnl = np.where(initial != 0.0, (last_bal / initial - 1) * 100, 0.0)

    # ── SQN / std / avg return ────────────────────────────────────────────
    r_sum    = r.sum(axis=1)
    r_sum_sq = (r ** 2).sum(axis=1)
    avg_ret  = np.where(has_trades, r_sum / n_valid * 100, 0.0)

    r_var = np.where(
        n_valid > 1,
        (r_sum_sq - r_sum ** 2 / np.maximum(n_valid, 1)) / np.maximum(n_valid - 1, 1),
        0.0
    )
    r_std = np.sqrt(np.maximum(r_var, 0.0))
    trade_std = r_std * 100

    mean_r = np.where(has_trades, r_sum / n_valid, 0.0)
    sqn    = np.where(r_std > 0, mean_r / r_std * np.sqrt(n_valid), 0.0)

    # ── Edge ratio ────────────────────────────────────────────────────────
    edge_ratio = np.where(avg_loss != 0.0, avg_win / np.abs(avg_loss), 0.0)

    # ── Geometric mean — log trick avoids overflow ────────────────────────
    gross    = np.where(trade_valid, np.maximum(1.0 + returns_raw, 1e-10), 1.0)
    log_sum  = np.log(gross).sum(axis=1)                       # (n_strats,)
    geo_mean = np.where(has_trades, np.expm1(log_sum / np.maximum(n_valid, 1)) * 100, 0.0)

    # ── Profit factor ─────────────────────────────────────────────────────
    gross_wins   = win_sum                                     # already computed
    gross_losses = np.abs(loss_sum)
    profit_factor = np.where(gross_losses > 0, gross_wins / gross_losses, 0.0)

    # ── Win/loss ratio ────────────────────────────────────────────────────
    win_loss_ratio = np.where(n_loss > 0, n_win / np.maximum(n_loss, 1), 0.0)

    # ── Outlier ratios (IQR) — vectorized percentile ──────────────────────
    # np.nanpercentile operates on axis=1 over NaN-masked array
    q1 = np.nanpercentile(r_nan, 25, axis=1)                  # (n_strats,)
    q3 = np.nanpercentile(r_nan, 75, axis=1)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    outlier_win  = np.where(has_trades,
        np.where(trade_valid, returns_raw > upper[:, np.newaxis], False).sum(axis=1) / n_valid, 0.0)
    outlier_loss = np.where(has_trades,
        np.where(trade_valid, returns_raw < lower[:, np.newaxis], False).sum(axis=1) / n_valid, 0.0)

    # ── Streaks — per-strategy loop is unavoidable ────────────────────────
    max_wins   = _max_consecutive_batched(win_mask,  trade_valid)
    max_losses = _max_consecutive_batched(loss_mask, trade_valid)

    # ── Assemble structured output ────────────────────────────────────────
    dtype = [
        ('total_trades',                    'i4'),
        ('win_rate_pct',                    'f8'), ('loss_rate_pct',                'f8'),
        ('best_trade_pct',                  'f8'), ('worst_trade_pct',              'f8'),
        ('winning_trades',                  'i4'), ('losing_trades',                'i4'),
        ('avg_winning_trade_pct',           'f8'), ('avg_losing_trade_pct',         'f8'),
        ('avg_winning_trade_duration_days', 'f8'), ('avg_losing_trade_duration_days','f8'),
        ('consecutive_wins',                'i4'), ('consecutive_losses',           'i4'),
        ('max_winning_streak',              'i4'), ('max_losing_streak',            'i4'),
        ('avg_duration_trades',             'f8'), ('trade_duration_std',           'f8'),
        ('total_pnl_pct',                   'f8'), ('trade_return_std',             'f8'),
        ('avg_return_all_trades',           'f8'), ('geometric_mean_returns',       'f8'),
        ('sqn',                             'f8'), ('edge_ratio',                   'f8'),
        ('win_loss_ratio',                  'f8'), ('outlier_win_ratio',            'f8'),
        ('outlier_loss_ratio',              'f8'), ('expectancy',                   'f8'),
        ('profit_factor',                   'f8'), ('mfe_pct',                      'f8'),
        ('mae_pct',                         'f8'),
    ]

    result = np.zeros(n_strats, dtype=dtype)
    result['total_trades']                     = lengths.astype(np.int32)
    result['win_rate_pct']                     = win_rate
    result['loss_rate_pct']                    = loss_rate
    result['best_trade_pct']                   = best_trade
    result['worst_trade_pct']                  = worst_trade
    result['winning_trades']                   = n_win.astype(np.int32)
    result['losing_trades']                    = n_loss.astype(np.int32)
    result['avg_winning_trade_pct']            = avg_win
    result['avg_losing_trade_pct']             = avg_loss
    result['avg_winning_trade_duration_days']  = avg_win_dur
    result['avg_losing_trade_duration_days']   = avg_loss_dur
    result['consecutive_wins']                 = max_wins
    result['consecutive_losses']               = max_losses
    result['max_winning_streak']               = max_wins
    result['max_losing_streak']                = max_losses
    result['avg_duration_trades']              = avg_dur
    result['trade_duration_std']               = dur_std
    result['total_pnl_pct']                    = total_pnl
    result['trade_return_std']                 = trade_std
    result['avg_return_all_trades']            = avg_ret
    result['geometric_mean_returns']           = geo_mean
    result['sqn']                              = sqn
    result['edge_ratio']                       = edge_ratio
    result['win_loss_ratio']                   = win_loss_ratio
    result['outlier_win_ratio']                = outlier_win
    result['outlier_loss_ratio']               = outlier_loss
    result['expectancy']                       = avg_ret        # same definition as original
    result['profit_factor']                    = profit_factor
    result['mfe_pct']                          = 0.0
    result['mae_pct']                          = 0.0

    return result