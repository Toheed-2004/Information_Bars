"""
Single source of truth for trade analysis.
Both custom/ and vectorbt_pro/ modules call calculate_trade_analysis().
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

from .utils import _max_consecutive_numpy


def _empty_trade_analysis() -> Dict[str, Any]:
    return {
        'total_trades': 0, 'win_rate_pct': 0.0, 'loss_rate_pct': 0.0,
        'best_trade_pct': 0.0, 'worst_trade_pct': 0.0,
        'winning_trades': 0, 'losing_trades': 0,
        'avg_winning_trade_pct': 0.0, 'avg_losing_trade_pct': 0.0,
        'avg_winning_trade_duration_days': 0.0, 'avg_losing_trade_duration_days': 0.0,
        'consecutive_wins': 0, 'consecutive_losses': 0,
        'max_winning_streak': 0, 'max_losing_streak': 0,
        'avg_duration_trades': 0.0, 'trade_duration_std': 0.0,
        'total_pnl_pct': 0.0, 'trade_return_std': 0.0,
        'avg_return_all_trades': 0.0, 'geometric_mean_returns': 0.0,
        'sqn': 0.0, 'edge_ratio': 0.0, 'win_loss_ratio': 0.0,
        'outlier_win_ratio': 0.0, 'outlier_loss_ratio': 0.0,
        'expectancy': 0.0, 'profit_factor': 0.0,
        'mfe_pct': 0.0, 'mae_pct': 0.0,
    }


def calculate_trade_analysis(ledger: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all trade metrics from ledger.
    Uses trade_return_pct (% return on invested capital per trade).
    """
    if ledger is None or len(ledger) == 0 or 'trade_return_pct' not in ledger.columns:
        return _empty_trade_analysis()

    returns = ledger['trade_return_pct'].values / 100.0  # decimal
    n = len(returns)

    win_mask  = returns > 0
    loss_mask = returns < 0
    n_win  = int(np.sum(win_mask))
    n_loss = int(np.sum(loss_mask))

    win_returns  = returns[win_mask]
    loss_returns = returns[loss_mask]

    win_rate  = n_win  / n * 100.0
    loss_rate = n_loss / n * 100.0

    avg_win  = float(np.mean(win_returns)  * 100) if n_win  > 0 else 0.0
    avg_loss = float(np.mean(loss_returns) * 100) if n_loss > 0 else 0.0

    # Durations
    avg_win_dur = avg_loss_dur = avg_dur = dur_std = 0.0
    if 'entry_datetime' in ledger.columns and 'exit_datetime' in ledger.columns:
        entry_dt = pd.to_datetime(ledger['entry_datetime'].values)
        exit_dt  = pd.to_datetime(ledger['exit_datetime'].values)
        durations = (exit_dt - entry_dt).total_seconds().values / 86400.0
        avg_win_dur  = float(np.mean(durations[win_mask]))  if n_win  > 0 else 0.0
        avg_loss_dur = float(np.mean(durations[loss_mask])) if n_loss > 0 else 0.0
        avg_dur  = float(np.mean(durations))
        dur_std  = float(np.std(durations, ddof=1)) if n > 1 else 0.0

    # Total PnL from balance
    if 'account_return_pct' in ledger.columns and 'balance' in ledger.columns:
        bal = ledger['balance'].values
        first_ret = float(ledger['account_return_pct'].iloc[0]) / 100.0
        initial = bal[0] / (1.0 + first_ret) if (1.0 + first_ret) != 0 else bal[0]
        total_pnl = float((bal[-1] / initial - 1) * 100)
    else:
        total_pnl = float(np.sum(returns) * 100)

    # Streaks
    max_wins   = _max_consecutive_numpy(win_mask)
    max_losses = _max_consecutive_numpy(loss_mask)

    # Outlier ratios — IQR method
    q1, q3 = np.percentile(returns, [25, 75])
    iqr = q3 - q1
    outlier_win_ratio  = float(np.sum(returns >  q3 + 1.5 * iqr) / n)
    outlier_loss_ratio = float(np.sum(returns < q1 - 1.5 * iqr) / n)

    # SQN
    std_r = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    sqn = float(np.mean(returns) / std_r * np.sqrt(n)) if std_r > 0 else 0.0

    # Edge ratio
    edge_ratio = float(avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0

    # Geometric mean
    gross = np.maximum(1 + returns, 1e-10)
    try:
        geo_mean = float((np.prod(gross) ** (1.0 / n) - 1) * 100)
    except (OverflowError, ValueError):
        geo_mean = 0.0

    # Expectancy: average return per trade (%)
    expectancy = float(np.mean(returns) * 100)

    # Profit factor: gross wins / gross losses
    gross_wins   = float(np.sum(win_returns))  if n_win  > 0 else 0.0
    gross_losses = float(np.abs(np.sum(loss_returns))) if n_loss > 0 else 0.0
    profit_factor = float(gross_wins / gross_losses) if gross_losses > 0 else 0.0

    return {
        'total_trades':                   n,
        'win_rate_pct':                   float(win_rate),
        'loss_rate_pct':                  float(loss_rate),
        'best_trade_pct':                 float(np.max(returns) * 100),
        'worst_trade_pct':                float(np.min(returns) * 100),
        'winning_trades':                 n_win,
        'losing_trades':                  n_loss,
        'avg_winning_trade_pct':          avg_win,
        'avg_losing_trade_pct':           avg_loss,
        'avg_winning_trade_duration_days': avg_win_dur,
        'avg_losing_trade_duration_days':  avg_loss_dur,
        'consecutive_wins':               max_wins,
        'consecutive_losses':             max_losses,
        'max_winning_streak':             max_wins,
        'max_losing_streak':              max_losses,
        'avg_duration_trades':            avg_dur,
        'trade_duration_std':             dur_std,
        'total_pnl_pct':                  total_pnl,
        'trade_return_std':               float(std_r * 100),
        'avg_return_all_trades':          float(np.mean(returns) * 100),
        'geometric_mean_returns':         geo_mean,
        'sqn':                            sqn,
        'edge_ratio':                     edge_ratio,
        'win_loss_ratio':                 float(n_win / n_loss) if n_loss > 0 else 0.0,
        'outlier_win_ratio':              outlier_win_ratio,
        'outlier_loss_ratio':             outlier_loss_ratio,
        'expectancy':                     expectancy,
        'profit_factor':                  profit_factor,
        'mfe_pct': float(ledger['mfe_pct'].mean()) if 'mfe_pct' in ledger.columns else 0.0,
        'mae_pct': float(ledger['mae_pct'].mean()) if 'mae_pct' in ledger.columns else 0.0,
    }
