import numpy as np
import pandas as pd
from typing import Dict, Any


def get_data_directional_metrics(cache: Dict) -> Dict[str, Any]:
    """
    Directional trade metrics from cache['trades_df'] (original pf.trades).

    Uses:
      trades_df['Direction']   — 'Long' / 'Short'
      trades_df['Return']      — fractional per-trade return
      trades_df['Entry Index'] — entry timestamp
      trades_df['Exit Index']  — exit timestamp
    """
    trades_df = cache.get('trades_df', pd.DataFrame())
    if trades_df.empty or 'Direction' not in trades_df.columns:
        return _empty_directional_metrics()

    # Vectorized arrays
    direction_array = trades_df['Direction'].values

    if 'Return' in trades_df.columns:
        ret_array = trades_df['Return'].values.astype(np.float64) * 100.0  # → pct
    else:
        ret_array = np.zeros(len(trades_df))

    if 'Entry Index' in trades_df.columns and 'Exit Index' in trades_df.columns:
        durations = (
            pd.to_datetime(trades_df['Exit Index']) - pd.to_datetime(trades_df['Entry Index'])
        ).dt.total_seconds().values / 86400.0
    else:
        durations = np.zeros(len(trades_df))

    long_mask  = direction_array == 'Long'
    short_mask = direction_array == 'Short'

    long_ret  = ret_array[long_mask]
    short_ret = ret_array[short_mask]
    long_dur  = durations[long_mask]
    short_dur = durations[short_mask]

    total_trades      = len(trades_df)
    long_trades_count = int(np.sum(long_mask))
    short_trades_count= int(np.sum(short_mask))

    def _dir_stats(ret, dur, count):
        if count == 0:
            return 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        wins = int(np.sum(ret > 0))
        return (
            wins,
            count - wins,
            float(wins / count * 100),          # win_rate_pct
            float(np.mean(dur)),                 # avg_duration_days
            float(np.sum(ret)),                  # total_pnl_pct
            float(np.mean(ret)),                 # avg_pnl_pct
            float(np.max(ret)),                  # best_trade_pct
            float(np.min(ret)),                  # worst_trade_pct
        )

    lw, ll, lwr, lad, ltp, lap, lbt, lwt = _dir_stats(long_ret,  long_dur,  long_trades_count)
    sw, sl, swr, sad, stp, sap, sbt, swt = _dir_stats(short_ret, short_dur, short_trades_count)

    return {
        'long_trades_count':    long_trades_count,
        'short_trades_count':   short_trades_count,
        'long_trades_pct':      float(long_trades_count  / total_trades * 100) if total_trades > 0 else 0.0,
        'short_trades_pct':     float(short_trades_count / total_trades * 100) if total_trades > 0 else 0.0,
        'long_winning_trades':  lw,
        'long_losing_trades':   ll,
        'short_winning_trades': sw,
        'short_losing_trades':  sl,
        'long_win_rate_pct':    lwr,
        'short_win_rate_pct':   swr,
        'long_avg_duration_days':  lad,
        'short_avg_duration_days': sad,
        'long_total_pnl_pct':   ltp,
        'short_total_pnl_pct':  stp,
        'long_avg_pnl_pct':     lap,
        'short_avg_pnl_pct':    sap,
        'long_best_trade_pct':  lbt,
        'long_worst_trade_pct': lwt,
        'short_best_trade_pct': sbt,
        'short_worst_trade_pct':swt,
    }


def _empty_directional_metrics() -> Dict[str, Any]:
    return {
        'long_trades_count': 0,    'short_trades_count': 0,
        'long_trades_pct': 0.0,    'short_trades_pct': 0.0,
        'long_winning_trades': 0,  'long_losing_trades': 0,
        'short_winning_trades': 0, 'short_losing_trades': 0,
        'long_win_rate_pct': 0.0,  'short_win_rate_pct': 0.0,
        'long_avg_duration_days': 0.0, 'short_avg_duration_days': 0.0,
        'long_total_pnl_pct': 0.0, 'short_total_pnl_pct': 0.0,
        'long_avg_pnl_pct': 0.0,   'short_avg_pnl_pct': 0.0,
        'long_best_trade_pct': 0.0,'long_worst_trade_pct': 0.0,
        'short_best_trade_pct': 0.0,'short_worst_trade_pct': 0.0,
    }
