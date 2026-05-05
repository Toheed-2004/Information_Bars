import pandas as pd
import numpy as np
from typing import Dict, Any

from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    LEDGER_ENTRY_DATETIME_COL,
    LEDGER_EXIT_DATETIME_COL,
    LEDGER_DIRECTION_COL,
    ENTRY_REGIME_LABEL_COL,
    ENTRY_REGIME_TREND_COL,
    ENTRY_REGIME_VOLATILITY_COL,
    ENTRY_REGIME_MOMENTUM_COL,
)


def calculate_regime_performance(ledger: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute performance stats grouped by each regime dimension.
    All return values are in percentage points (e.g. -0.23 means -0.23%).
    Sharpe and Sortino are annualised using per-group trade frequency (crypto 24/7, 365.25 days).
    """
    total_trades = len(ledger)

    # Sort by exit_datetime once — drawdown order matters
    if LEDGER_EXIT_DATETIME_COL in ledger.columns:
        ledger = ledger.sort_values(LEDGER_EXIT_DATETIME_COL).reset_index(drop=True)

    results = {}
    results['by_regime_label'] = _compute_stats(ledger, ENTRY_REGIME_LABEL_COL, total_trades)
    results['by_trend'] = _compute_stats(ledger, ENTRY_REGIME_TREND_COL, total_trades)
    results['by_volatility'] = _compute_stats(ledger, ENTRY_REGIME_VOLATILITY_COL, total_trades)
    results['by_momentum'] = _compute_stats(ledger, ENTRY_REGIME_MOMENTUM_COL, total_trades)

    # Direction × trend composite (e.g. "Long_in_BULL")
    ledger = ledger.copy()
    ledger['_dir_trend'] = (
        ledger[LEDGER_DIRECTION_COL].astype(str)
        + '_in_'
        + ledger[ENTRY_REGIME_TREND_COL].astype(str)
    )
    results['by_direction_trend'] = _compute_stats(ledger, '_dir_trend', total_trades)

    return results


def _compute_stats(ledger: pd.DataFrame, group_col: str, total_trades: int) -> Dict[str, Dict[str, Any]]:
    """
    Vectorized computation of all regime stats for every group at once.

    Metrics returned per group:
        trade_count, pct_of_total_trades,
        win_rate_pct, avg_return_pct, median_return_pct, total_return_pct,
        profit_factor, return_skewness,
        sharpe_ratio (annualised), sortino_ratio (annualised), calmar_ratio,
        avg_trade_duration_days, max_drawdown_pct, max_consecutive_losses
    """
    if group_col not in ledger.columns:
        return {}

    returns = ledger[LEDGER_PNL_COL].values.astype(float)
    groups = ledger[group_col].values.astype(str)

    unique_groups, group_ids = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)

    # -----------------------------------------------------------------------
    # Fully vectorised stats (bincount)
    # -----------------------------------------------------------------------
    trade_counts = np.bincount(group_ids, minlength=n_groups)

    wins = (returns > 0).astype(float)
    win_counts = np.bincount(group_ids, weights=wins, minlength=n_groups)
    win_rates = np.divide(win_counts * 100.0, trade_counts,
                          out=np.zeros(n_groups), where=trade_counts > 0)

    total_returns = np.bincount(group_ids, weights=returns, minlength=n_groups)
    avg_returns = np.divide(total_returns, trade_counts,
                            out=np.zeros(n_groups), where=trade_counts > 0)

    gross_profit = np.where(returns > 0, returns, 0.0)
    gross_loss = np.where(returns < 0, -returns, 0.0)
    sum_profit = np.bincount(group_ids, weights=gross_profit, minlength=n_groups)
    sum_loss = np.bincount(group_ids, weights=gross_loss, minlength=n_groups)
    profit_factors = np.divide(sum_profit, sum_loss,
                               out=np.zeros(n_groups), where=sum_loss > 0)

    # Population variance: E[x²] - E[x]²
    sum_sq = np.bincount(group_ids, weights=returns ** 2, minlength=n_groups)
    e_x2 = np.divide(sum_sq, trade_counts, out=np.zeros(n_groups), where=trade_counts > 0)
    variance = np.maximum(e_x2 - avg_returns ** 2, 0.0)
    std_returns = np.sqrt(variance)

    # Trade duration (days) using nanosecond timestamps
    has_datetimes = (LEDGER_EXIT_DATETIME_COL in ledger.columns
                     and LEDGER_ENTRY_DATETIME_COL in ledger.columns)
    if has_datetimes:
        exit_dt = pd.to_datetime(ledger[LEDGER_EXIT_DATETIME_COL])
        entry_dt = pd.to_datetime(ledger[LEDGER_ENTRY_DATETIME_COL])
        if exit_dt.dt.tz is not None:
            exit_dt = exit_dt.dt.tz_convert('UTC').dt.tz_localize(None)
        if entry_dt.dt.tz is not None:
            entry_dt = entry_dt.dt.tz_convert('UTC').dt.tz_localize(None)
        exit_ts_ns = exit_dt.astype('int64').values
        entry_ts_ns = entry_dt.astype('int64').values
        durations_days = (exit_ts_ns - entry_ts_ns) / (86400.0 * 1e9)
        duration_sums = np.bincount(group_ids, weights=durations_days, minlength=n_groups)
        avg_durations = np.divide(duration_sums, trade_counts,
                                  out=np.zeros(n_groups), where=trade_counts > 0)
    else:
        exit_ts_ns = None
        entry_ts_ns = None
        avg_durations = np.zeros(n_groups)

    # -----------------------------------------------------------------------
    # Per-group computations
    # Sort pnl by group for contiguous slice access (cache-friendly)
    # -----------------------------------------------------------------------
    sort_by_group = np.argsort(group_ids, kind='stable')
    returns_sorted = returns[sort_by_group]
    group_ids_sorted = group_ids[sort_by_group]
    group_starts = np.searchsorted(group_ids_sorted, np.arange(n_groups), side='left')
    group_ends = np.searchsorted(group_ids_sorted, np.arange(n_groups), side='right')

    if has_datetimes:
        exit_ts_sorted = exit_ts_ns[sort_by_group]
        entry_ts_sorted = entry_ts_ns[sort_by_group]

    max_drawdowns = np.zeros(n_groups)
    max_consec_losses = np.zeros(n_groups, dtype=int)
    median_returns = np.zeros(n_groups)
    skewness = np.zeros(n_groups)
    downside_std = np.zeros(n_groups)
    sharpe = np.zeros(n_groups)
    sortino = np.zeros(n_groups)
    calmar = np.zeros(n_groups)

    for i in range(n_groups):
        s, e = int(group_starts[i]), int(group_ends[i])
        if e <= s:
            continue

        g = returns_sorted[s:e]
        n = e - s
        mean = float(avg_returns[i])
        std = float(std_returns[i])

        # Median
        median_returns[i] = np.median(g)

        # Skewness: E[(x-μ)³] / σ³  (population, requires n >= 3)
        if std > 0.0 and n >= 3:
            centered = g - mean
            skewness[i] = float(np.mean(centered ** 3) / (std ** 3))

        # Downside std: sqrt(mean(min(r, 0)²))
        downside_sq_mean = float(np.mean(np.minimum(g, 0.0) ** 2))
        downside_std[i] = np.sqrt(downside_sq_mean)

        # Max drawdown — compound returns (values are percentage points)
        if n > 1:
            cum = np.cumprod(1.0 + g / 100.0)
            running_max = np.maximum.accumulate(cum)
            max_drawdowns[i] = float(np.min((cum / running_max - 1.0) * 100.0))
        elif n == 1 and g[0] < 0.0:
            max_drawdowns[i] = float(g[0])

        # Max consecutive losses — vectorised run-length encoding
        is_loss = (g < 0.0)
        padded = np.empty(n + 2, dtype=np.int8)
        padded[0] = 0
        padded[1:-1] = is_loss.astype(np.int8)
        padded[-1] = 0
        diff = np.diff(padded)
        run_starts = np.where(diff == 1)[0]
        run_ends = np.where(diff == -1)[0]
        if len(run_starts):
            max_consec_losses[i] = int(np.max(run_ends - run_starts))

        # Per-trade Sharpe: mean / std (no annualisation — comparing trade quality across regimes)
        if std > 0.0:
            sharpe[i] = mean / std

        # Per-trade Sortino: mean / downside_std
        if downside_std[i] > 0.0:
            sortino[i] = mean / float(downside_std[i])

        # Calmar: avg_return / abs(max_drawdown)
        if max_drawdowns[i] < 0.0:
            calmar[i] = mean / abs(float(max_drawdowns[i]))

    # -----------------------------------------------------------------------
    # Assemble final results dict
    # -----------------------------------------------------------------------
    results = {}
    for i, group in enumerate(unique_groups):
        pct_of_total = (float(trade_counts[i]) / total_trades * 100.0) if total_trades > 0 else 0.0
        results[str(group)] = {
            'trade_count': int(trade_counts[i]),
            'total_trades_pct': round(pct_of_total, 4),
            'win_rate_pct': round(float(win_rates[i]), 4),
            'avg_return_pct': round(float(avg_returns[i]), 4),
            'median_return_pct': round(float(median_returns[i]), 4),
            'total_return_pct': round(float(total_returns[i]), 4),
            'profit_factor': round(float(profit_factors[i]), 4),
            'return_skewness': round(float(skewness[i]), 4),
            'sharpe_ratio': round(float(sharpe[i]), 4),
            'sortino_ratio': round(float(sortino[i]), 4),
            'calmar_ratio': round(float(calmar[i]), 4),
            'avg_trade_duration_days': round(float(avg_durations[i]), 4),
            'max_drawdown_pct': round(float(max_drawdowns[i]), 4),
            'max_consecutive_losses': int(max_consec_losses[i]),
        }

    return results
