import numpy as np
import pandas as pd
from typing import Dict

def _get_empty_long_short() -> Dict[str, float]:
    return {
        'long_trades_count': 0, 'short_trades_count': 0,
        'long_trades_pct': 0.0, 'short_trades_pct': 0.0,
        'long_winning_trades': 0, 'long_losing_trades': 0,
        'short_winning_trades': 0, 'short_losing_trades': 0,
        'long_avg_duration': 0.0, 'short_avg_duration': 0.0,
        'long_win_rate': 0.0, 'short_win_rate': 0.0,
        'long_total_pnl_pct': 0.0, 'short_total_pnl_pct': 0.0,
        'long_avg_pnl_pct': 0.0, 'short_avg_pnl_pct': 0.0,
        'long_best_trade': 0.0, 'long_worst_trade': 0.0,
        'short_best_trade': 0.0, 'short_worst_trade': 0.0
    }

def _calculate_long_short_analysis(
    df_ledger: pd.DataFrame,
    directions: np.ndarray,
    actions: np.ndarray
) -> Dict[str, float]:
    """Directional trade analysis. Every row in the new ledger is a closed trade."""

    if df_ledger is None or len(df_ledger) == 0:
        return _get_empty_long_short()

    # Use account_return_pct for PnL attribution (actual account impact)
    if 'account_return_pct' not in df_ledger.columns:
        return _get_empty_long_short()

    pnl_array = df_ledger['account_return_pct'].values.astype(np.float64)

    direction_col = df_ledger['direction'].values if 'direction' in df_ledger.columns else directions
    directions_lower = np.array([str(d).lower() for d in direction_col])

    long_mask = directions_lower == 'long'
    short_mask = directions_lower == 'short'
    total_trades = len(df_ledger)

    long_pnls = pnl_array[long_mask]
    short_pnls = pnl_array[short_mask]

    long_trades_count = int(np.sum(long_mask))
    short_trades_count = int(np.sum(short_mask))

    long_trades_pct = (long_trades_count / total_trades * 100.0) if total_trades > 0 else 0.0
    short_trades_pct = (short_trades_count / total_trades * 100.0) if total_trades > 0 else 0.0

    # Durations in days
    long_avg_duration = 0.0
    short_avg_duration = 0.0
    if 'exit_datetime' in df_ledger.columns and 'entry_datetime' in df_ledger.columns:
        durations_days = (
            pd.to_datetime(df_ledger['exit_datetime']) - pd.to_datetime(df_ledger['entry_datetime'])
        ).dt.total_seconds().values / 86400.0
        long_avg_duration = float(durations_days[long_mask].mean()) if long_trades_count > 0 else 0.0
        short_avg_duration = float(durations_days[short_mask].mean()) if short_trades_count > 0 else 0.0

    # Long stats
    long_winning_trades = int(np.sum(long_pnls > 0)) if long_trades_count > 0 else 0
    long_losing_trades = int(np.sum(long_pnls < 0)) if long_trades_count > 0 else 0
    long_win_rate = (long_winning_trades / long_trades_count * 100.0) if long_trades_count > 0 else 0.0
    long_total_pnl = float(np.sum(long_pnls)) if long_trades_count > 0 else 0.0
    long_avg_pnl = float(np.mean(long_pnls)) if long_trades_count > 0 else 0.0
    long_best_trade = float(np.max(long_pnls)) if long_trades_count > 0 else 0.0
    long_worst_trade = float(np.min(long_pnls)) if long_trades_count > 0 else 0.0

    # Short stats
    short_winning_trades = int(np.sum(short_pnls > 0)) if short_trades_count > 0 else 0
    short_losing_trades = int(np.sum(short_pnls < 0)) if short_trades_count > 0 else 0
    short_win_rate = (short_winning_trades / short_trades_count * 100.0) if short_trades_count > 0 else 0.0
    short_total_pnl = float(np.sum(short_pnls)) if short_trades_count > 0 else 0.0
    short_avg_pnl = float(np.mean(short_pnls)) if short_trades_count > 0 else 0.0
    short_best_trade = float(np.max(short_pnls)) if short_trades_count > 0 else 0.0
    short_worst_trade = float(np.min(short_pnls)) if short_trades_count > 0 else 0.0

    return {
        'long_trades_count': long_trades_count,
        'short_trades_count': short_trades_count,
        'long_trades_pct': float(long_trades_pct),
        'short_trades_pct': float(short_trades_pct),
        'long_winning_trades': long_winning_trades,
        'long_losing_trades': long_losing_trades,
        'short_winning_trades': short_winning_trades,
        'short_losing_trades': short_losing_trades,
        'long_avg_duration': float(long_avg_duration),
        'short_avg_duration': float(short_avg_duration),
        'long_win_rate': float(long_win_rate),
        'short_win_rate': float(short_win_rate),
        'long_total_pnl_pct': float(long_total_pnl),
        'short_total_pnl_pct': float(short_total_pnl),
        'long_avg_pnl_pct': float(long_avg_pnl),
        'short_avg_pnl_pct': float(short_avg_pnl),
        'long_best_trade': float(long_best_trade),
        'long_worst_trade': float(long_worst_trade),
        'short_best_trade': float(short_best_trade),
        'short_worst_trade': float(short_worst_trade),
    }
