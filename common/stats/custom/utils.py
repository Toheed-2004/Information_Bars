import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from ..shared.utils import _bh_per_trade_returns, ANN_FACTOR
from ..shared.trade import _empty_trade_analysis as _get_empty_trade_analysis
from ..shared.benchmark import _empty_benchmark_analysis as _get_empty_benchmark_analysis
from .ratios import _get_empty_risk_adjusted
from .risk import _get_empty_risk_metrics
from .drawdown import _get_empty_drawdown_analysis
from .profit_loss import _get_empty_profit_loss
from .long_short import _get_empty_long_short
from .portfolio import _get_empty_portfolio_values
from .exposure import _get_empty_exposure
from .cash_flow import _get_empty_cash_flow
from .time_series import _get_empty_time_series_analysis
from .distribution import _get_empty_distribution_analysis


def resample_minute_returns_to_daily(minute_returns: pd.Series) -> pd.Series:
    """
    Simple function to resample minute-level returns to daily returns.
    
    Compounds returns within each calendar day: (1 + r1) * (1 + r2) * ... - 1
    
    Args:
        minute_returns: pd.Series with datetime index and minute-level returns (fractional, not %)
    
    Returns:
        pd.Series with daily returns indexed by date
    """
    if minute_returns is None or minute_returns.empty:
        return pd.Series(dtype=float)
    
    # Group by calendar day and compound returns
    daily_returns = minute_returns.groupby(minute_returns.index.normalize()).apply(
        lambda x: (1 + x).prod() - 1
    )
    
    return daily_returns


def build_daily_returns_from_bars(
    df_ledger: pd.DataFrame,
    df_bars: pd.DataFrame,
    start_date=None,
) -> pd.DataFrame:
    """
    Reconstruct daily portfolio returns from ledger using VBT's exact approach.
    Optimized vectorized implementation for speed.

    Args:
        df_ledger: Closed-trade ledger
        df_bars:   1m OHLCV DataFrame with 'datetime' and 'close'
        start_date: Optional portfolio start date

    Returns:
        DataFrame with columns: date, portfolio_return, benchmark_return, balance
    """
    if df_ledger is None or df_ledger.empty or df_bars is None or df_bars.empty:
        return pd.DataFrame(columns=['date', 'portfolio_return', 'benchmark_return', 'balance'])

    # ── 1. Prepare bars ───────────────────────────────────────────────────
    bars = df_bars[['datetime', 'close']].copy()
    bars['datetime'] = pd.to_datetime(bars['datetime'])
    bars = bars.sort_values('datetime').set_index('datetime')
    close_arr = bars['close'].values.astype(np.float64)
    bar_ts    = bars.index.values.astype('int64')  # nanoseconds
    n_bars    = len(bar_ts)

    # ── 2. Initial balance ────────────────────────────────────────────────
    first_balance = float(df_ledger['balance'].iloc[0])
    first_ret     = float(df_ledger['account_return_pct'].iloc[0]) / 100.0
    initial_balance = first_balance / (1.0 + first_ret) if (1.0 + first_ret) != 0 else first_balance

    # ── 3. Vectorized trade data ──────────────────────────────────────────
    entry_ts  = pd.to_datetime(df_ledger['entry_datetime'].values).astype('int64')
    exit_ts   = pd.to_datetime(df_ledger['exit_datetime'].values).astype('int64')
    entry_px  = df_ledger['avg_entry_price'].values.astype(np.float64)
    exit_px   = df_ledger['avg_exit_price'].values.astype(np.float64)

    bal_vals  = df_ledger['balance'].values.astype(np.float64)
    acc_ret   = df_ledger['account_return_pct'].values.astype(np.float64) / 100.0
    prev_bal  = np.where((1.0 + acc_ret) != 0, bal_vals / (1.0 + acc_ret), bal_vals)
    pos_size  = df_ledger['position_size_pct'].values.astype(np.float64) / 100.0
    pos_val   = prev_bal * pos_size

    directions = df_ledger['direction'].str.lower().values if 'direction' in df_ledger.columns \
                 else np.array(['long'] * len(df_ledger))
    sign = np.where(np.isin(directions, ['long', 'buy']), 1.0, -1.0)

    entry_fee_pct = df_ledger['entry_fee_pct'].values.astype(np.float64) / 100.0 \
                    if 'entry_fee_pct' in df_ledger.columns else np.zeros(len(df_ledger))
    exit_fee_pct  = df_ledger['exit_fee_pct'].values.astype(np.float64) / 100.0 \
                    if 'exit_fee_pct' in df_ledger.columns else np.zeros(len(df_ledger))
    entry_fees = pos_val * entry_fee_pct
    exit_fees  = pos_val * exit_fee_pct

    # ── 4. Build cash change array (sparse events) ────────────────────────
    cash_changes = np.zeros(n_bars)
    cash_changes[0] = initial_balance

    # Find bar indices for entry/exit events using searchsorted
    entry_idx = np.searchsorted(bar_ts, entry_ts, side='left').clip(0, n_bars - 1)
    exit_idx  = np.searchsorted(bar_ts, exit_ts,  side='left').clip(0, n_bars - 1)

    # Apply entry fees
    np.add.at(cash_changes, entry_idx, -entry_fees)

    # Apply exit P&L and fees
    price_change = (exit_px / entry_px) - 1.0
    pnl = pos_val * sign * price_change
    np.add.at(cash_changes, exit_idx, pnl - exit_fees)

    # Cumulative cash
    cash = np.cumsum(cash_changes)

    # ── 5. Vectorized unrealized P&L ─────────────────────────────────────
    # For each bar, sum unrealized P&L of all open positions
    # Use searchsorted to find open/close bar indices per trade
    unrealized_pnl = np.zeros(n_bars)

    for i in range(len(df_ledger)):
        s = int(np.searchsorted(bar_ts, entry_ts[i], side='left'))
        e = int(np.searchsorted(bar_ts, exit_ts[i],  side='left'))  # exclusive
        if s >= e:
            continue
        price_chg = close_arr[s:e] / entry_px[i] - 1.0
        unrealized_pnl[s:e] += pos_val[i] * sign[i] * price_chg

    # ── 6. Portfolio values → minute returns → daily ──────────────────────
    portfolio_values = pd.Series(cash + unrealized_pnl, index=bars.index)
    minute_returns   = portfolio_values.pct_change().fillna(0.0)
    daily_returns    = resample_minute_returns_to_daily(minute_returns)

    daily_balances = portfolio_values.groupby(portfolio_values.index.normalize()).last()
    close_series   = bars['close']
    close_daily    = close_series.groupby(close_series.index.normalize()).last()
    bm_returns     = close_daily.pct_change().fillna(0.0)

    # ── 7. Align to full date range ───────────────────────────────────────
    range_start = pd.Timestamp(start_date).normalize() if start_date is not None \
                  else daily_returns.index.min()
    all_days = pd.date_range(start=range_start, end=daily_returns.index.max(), freq='D')

    daily_returns  = daily_returns.reindex(all_days, fill_value=0.0)
    bm_returns     = bm_returns.reindex(all_days, fill_value=0.0)
    daily_balances = daily_balances.reindex(all_days, method='ffill').fillna(initial_balance)

    return pd.DataFrame({
        'date':             all_days,
        'portfolio_return': daily_returns.values,
        'benchmark_return': bm_returns.values,
        'balance':          daily_balances.values,
    })

def _get_empty_comprehensive_stats() -> Dict[str, Dict[str, Any]]:
    """Return empty comprehensive stats structure"""
    return {
        '0': {
            'risk_adjusted': _get_empty_risk_adjusted(),
            'risk_metrics': _get_empty_risk_metrics(),
            'drawdown_analysis': _get_empty_drawdown_analysis(),
            'trade_analysis': _get_empty_trade_analysis(),
            'profit_loss': _get_empty_profit_loss(),
            'long_short': _get_empty_long_short(),
            'portfolio_values': _get_empty_portfolio_values(),
            'exposure': _get_empty_exposure(),
            'cash_flow': _get_empty_cash_flow(),
            'time_series_analysis': _get_empty_time_series_analysis(),
            'benchmark_analysis': _get_empty_benchmark_analysis(),
            'distribution_analysis': _get_empty_distribution_analysis(),
            
            # Plot data placeholders
            'cumulative_return': {},
            'drawdown_series': {},
            'drawdown_periods': [],
            'mfe_pct': {},
            'mae_pct': {},
            'pnl_distribution': {"histogram_values": [], "kde_curve": []},
            'directional_pnl': {"long": {}, "short": {}},
            'rolling_sharpe': {},
            'rolling_sortino': {},
            'heatmaps_data': {},
            'rolling_correlation': {}
        }
    }


def _adapt_ledger_format(df_ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent column types for the new ledger format.
    Expected columns: entry_datetime, exit_datetime, entry_fee_pct, exit_fee_pct,
    avg_entry_price, avg_exit_price, position_size_pct, trade_return_pct,
    account_return_pct, cum_account_return, direction, status, action, balance.
    """
    if df_ledger is None or df_ledger.empty:
        return df_ledger

    # No copy needed - modify in place
    df = df_ledger

    if 'exit_datetime' in df.columns:
        df['exit_datetime'] = pd.to_datetime(df['exit_datetime'])
    if 'entry_datetime' in df.columns:
        df['entry_datetime'] = pd.to_datetime(df['entry_datetime'])

    return df

def _create_aligned_returns_from_ledger(
    df_ledger: pd.DataFrame, 
    benchmark_series: Optional[pd.Series] = None,
    df_bars: Optional[pd.DataFrame] = None,
    start_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Build daily-resampled returns from the ledger.

    If df_bars is provided, uses mark-to-market approach with actual 1m prices
    at day boundaries (matching VBT's daily resampling). Otherwise falls back
    to trade-level resampling using account_return_pct.

    Args:
        df_ledger: Closed-trade ledger
        benchmark_series: Optional benchmark price series
        df_bars: Optional 1m OHLCV bars for MTM calculation
        start_date: Optional portfolio start date for prepending idle days

    Returns a DataFrame with columns:
        datetime, portfolio_return, benchmark_return,
        returns_pct, cumulative_returns_pct, balance
    """
    EMPTY = pd.DataFrame(columns=[
        'datetime', 'portfolio_return', 'benchmark_return',
        'returns_pct', 'cumulative_returns_pct', 'balance'
    ])

    if len(df_ledger) < 1:
        return EMPTY

    # ── MTM approach if bars provided ────────────────────────────────────
    if df_bars is not None and not df_bars.empty:
        mtm_df = build_daily_returns_from_bars(df_ledger, df_bars, start_date=start_date)
        if mtm_df.empty:
            return EMPTY
        
        # Rename 'date' to 'datetime' for consistency
        mtm_df = mtm_df.rename(columns={'date': 'datetime'})
        
        # Calculate initial balance
        first_balance = float(df_ledger['balance'].iloc[0])
        first_ret = float(df_ledger['account_return_pct'].iloc[0]) / 100.0
        initial_balance = first_balance / (1.0 + first_ret) if (1.0 + first_ret) != 0 else first_balance
        
        # Add percentage columns
        mtm_df['returns_pct'] = mtm_df['portfolio_return'] * 100
        mtm_df['cumulative_returns_pct'] = (mtm_df['balance'] / initial_balance - 1) * 100
        
        return mtm_df

    # ── Fallback: trade-level resampling ─────────────────────────────────
    # account_return_pct / 100 is the fractional account return per trade
    account_returns = df_ledger['account_return_pct'].values / 100.0

    # initial balance = balance[0] / (1 + account_return_pct[0] / 100)
    first_balance = float(df_ledger['balance'].iloc[0])
    first_ret = float(df_ledger['account_return_pct'].iloc[0]) / 100.0
    initial_balance = first_balance / (1.0 + first_ret) if (1.0 + first_ret) != 0 else first_balance

    # Build benchmark returns
    benchmark_returns_final = None
    if benchmark_series is not None:
        if isinstance(benchmark_series, pd.DataFrame):
            benchmark_series = benchmark_series.iloc[:, 0]
        if isinstance(benchmark_series, pd.Series) and not benchmark_series.empty:
            exit_times = pd.to_datetime(df_ledger['exit_datetime'].values)
            if exit_times.tz is not None:
                exit_times = exit_times.tz_localize(None)
            bench_idx = pd.to_datetime(benchmark_series.index)
            if bench_idx.tz is not None:
                bench_idx = bench_idx.tz_localize(None)
            temp_bench = pd.Series(benchmark_series.values, index=bench_idx).sort_index()
            if np.mean(np.abs(temp_bench.values)) > 2.0:
                bench_cum = temp_bench / temp_bench.iloc[0]
            else:
                bench_cum = (1 + temp_bench).cumprod()
            reindexed_cum = bench_cum.reindex(exit_times, method='ffill').fillna(1.0).values
            reindexed_cum_with_start = np.concatenate([[1.0], reindexed_cum])
            benchmark_returns_final = (reindexed_cum_with_start[1:] / reindexed_cum_with_start[:-1]) - 1

    if benchmark_returns_final is None:
        if 'avg_entry_price' in df_ledger.columns and 'avg_exit_price' in df_ledger.columns:
            benchmark_returns_final = _bh_per_trade_returns(df_ledger)
        else:
            benchmark_returns_final = np.zeros(len(account_returns))

    # Build trade-level DataFrame indexed by exit_datetime
    trade_df = pd.DataFrame({
        'datetime': pd.to_datetime(df_ledger['exit_datetime'].values),
        'portfolio_return': account_returns,
        'benchmark_return': benchmark_returns_final,
        'balance': df_ledger['balance'].values,
    }).set_index('datetime')

    # Resample to daily: compound returns within each day
    daily_df = trade_df.resample('D').agg({
        'balance': 'last',
        'benchmark_return': lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0,
        'portfolio_return': lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0,
    })

    # Keep only days that had trades (filter creates view, no copy needed)
    daily_df = daily_df[daily_df['balance'].notna()]
    if len(daily_df) == 0:
        return EMPTY

    # Sanity filter
    daily_df = daily_df[
        (daily_df['portfolio_return'] > -0.99) &
        (daily_df['portfolio_return'] < 10.0)
    ]
    if len(daily_df) == 0:
        return EMPTY

    # Fill calendar gaps with 0 returns (idle days)
    date_range = pd.date_range(start=daily_df.index[0], end=daily_df.index[-1], freq='D')
    daily_df = daily_df.reindex(date_range)
    daily_df['balance'] = daily_df['balance'].ffill()
    daily_df['portfolio_return'] = daily_df['portfolio_return'].fillna(0)
    daily_df['benchmark_return'] = daily_df['benchmark_return'].fillna(0)

    daily_df['returns_pct'] = daily_df['portfolio_return'] * 100
    daily_df['cumulative_returns_pct'] = (daily_df['balance'] / initial_balance - 1) * 100

    daily_df.reset_index(inplace=True)
    daily_df.rename(columns={'index': 'datetime'}, inplace=True)

    return daily_df[['datetime', 'portfolio_return', 'benchmark_return',
                     'returns_pct', 'cumulative_returns_pct', 'balance']]

def _norm_cdf(x: float) -> float:
    """Approximate normal CDF using error function"""
    # Simplified normal CDF approximation
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

def _round_stats_dict(stats_dict: dict, decimals: int = 5) -> dict:
    """
    Efficiently round all numeric values in a stats dictionary to specified decimal places
    Also replaces inf/-inf values with -100
    """
    if not isinstance(stats_dict, dict):
        return stats_dict

    rounded_dict = {}
    for key, value in stats_dict.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            rounded_dict[key] = _round_stats_dict(value, decimals)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            # Round numeric values (excluding booleans which are technically int in Python)
            try:
                # Check for infinity values and replace with -100
                if np.isinf(value):
                    rounded_dict[key] = -100.0
                else:
                    rounded_dict[key] = round(float(value), decimals)
            except (ValueError, TypeError):
                # Keep original value if conversion fails
                rounded_dict[key] = value
        else:
            # Keep non-numeric values as-is (strings, booleans, None, etc.)
            rounded_dict[key] = value
    return rounded_dict
