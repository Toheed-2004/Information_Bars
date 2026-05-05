"""
ULTRA-FAST Custom Statistics Calculator from Ledger Data Only
Pure NumPy implementation for maximum performance - NO VBT dependency
Calculates comprehensive nested statistics identical to VBT-based approach
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# Import refactored modules
from .utils import (
    _adapt_ledger_format,
    _create_aligned_returns_from_ledger,
    _norm_cdf,
    _round_stats_dict,
    _get_empty_comprehensive_stats
)
# from ..shared.utils import ANN_FACTOR, _auto_detect_ann_factor
from ..shared.utils import ANN_FACTOR

# from bitpredict.common.stats.custom.utils import _auto_detect_ann_factor_vbt
from ..shared.trade import calculate_trade_analysis
from .ratios import _calculate_risk_adjusted_ratios
from .drawdown import _calculate_drawdown_analysis
from .profit_loss import _calculate_profit_loss
from .long_short import _calculate_long_short_analysis
from .portfolio import _calculate_portfolio_values
from .exposure import _calculate_exposure
from .cash_flow import _calculate_cash_flow
from .time_series import _calculate_time_series_analysis
from ..shared.benchmark import calculate_benchmark_analysis
from .distribution import _calculate_distribution_analysis
from .risk import _calculate_risk_metrics

def calculate_essential_stats(df_ledger: pd.DataFrame) -> Dict[str, float]:
    """
    ULTRA-FAST: Calculate only essential trading statistics from ledger.
    Now correctly accounts for the true initial capital (balance before first trade).
    """
    if df_ledger is None or df_ledger.empty:
        return {}

    # Adapt ledger format first
    df = _adapt_ledger_format(df_ledger)

    if 'balance' not in df.columns:
        return {}

    # Recover true initial balance
    first_balance = float(df['balance'].iloc[0])
    if 'account_return_pct' in df.columns:
        first_ret = float(df['account_return_pct'].iloc[0]) / 100.0
        true_initial = first_balance / (1.0 + first_ret) if (1.0 + first_ret) != 0 else first_balance
    else:
        true_initial = first_balance

    final_balance = float(df['balance'].iloc[-1])

    # Core metrics
    total_return_pct = ((final_balance / true_initial) - 1) * 100 if true_initial != 0 else 0.0

    # Max Drawdown
    balance_array = df['balance'].values
    peak = np.maximum.accumulate(balance_array)
    drawdowns = (balance_array - peak) / peak
    max_drawdown_pct = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0.0

    # Sharpe from account_return_pct (daily after resampling is ideal, but here use trade-level)
    if 'account_return_pct' in df.columns:
        returns = df['account_return_pct'].values / 100.0
    else:
        returns = np.diff(balance_array) / balance_array[:-1]
    sharpe = 0.0
    if len(returns) > 1:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        if std_ret > 0:
            sharpe = mean_ret / std_ret * np.sqrt(ANN_FACTOR)

    # Win rate from trade_return_pct
    if 'trade_return_pct' in df.columns:
        trade_returns = df['trade_return_pct'].values
    elif 'account_return_pct' in df.columns:
        trade_returns = df['account_return_pct'].values
    else:
        trade_returns = np.zeros(len(df))
    wins = np.sum(trade_returns > 0)
    total_trades = len(trade_returns)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    # Profit Factor from account_return_pct (or trade_return_pct)
    pnl_array = trade_returns
    gains = np.sum(pnl_array[pnl_array > 0])
    losses = np.abs(np.sum(pnl_array[pnl_array < 0]))
    profit_factor = (gains / losses) if losses > 0 else (100.0 if gains > 0 else 1.0)

    stats = {
        'total_return_pct': float(total_return_pct),
        'max_drawdown_pct': float(abs(max_drawdown_pct)),
        'sharpe_ratio': float(sharpe),
        'win_rate_pct': float(win_rate),
        'profit_factor': float(profit_factor),
        'total_trades': int(total_trades),
        'final_balance': float(final_balance)
    }

    return _round_stats_dict(stats)

def calculate_comprehensive_stats(
    df_ledger: pd.DataFrame, 
    df_bars: Optional[pd.DataFrame] = None,
    initial_date: Optional[pd.Timestamp] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Comprehensive statistics from ledger. 
    
    Args:
        df_ledger: Closed-trade ledger
        df_bars: Optional 1m OHLCV bars for MTM daily returns calculation
        initial_date: Optional portfolio start date for prepending idle days
        **kwargs: Additional arguments (benchmark_returns, etc.)
    """
    if df_ledger is None or len(df_ledger) == 0:
        return _get_empty_comprehensive_stats()

    # 1. ADAPT LEDGER
    df = _adapt_ledger_format(df_ledger)


    benchmark_series = kwargs.get('benchmark_returns')

    # 2. CREATE RETURNS AND ALIGN DATA
    returns_df = _create_aligned_returns_from_ledger(
        df, 
        benchmark_series=benchmark_series,
        df_bars=df_bars,
        start_date=initial_date
    )
    
    if len(returns_df) < 1:
        return _get_empty_comprehensive_stats()
        
    # Extract arrays for high-performance calculations
    portfolio_returns = returns_df['portfolio_return'].values
    benchmark_returns = returns_df['benchmark_return'].values
    balance_array = returns_df['balance'].values
    timestamps = returns_df['datetime'].values
    
    # Compute true initial from first row using account_return_pct
    first_balance = float(df['balance'].iloc[0])
    if 'account_return_pct' in df.columns:
        first_ret = float(df['account_return_pct'].iloc[0]) / 100.0
        true_initial = first_balance / (1.0 + first_ret) if (1.0 + first_ret) != 0 else first_balance
    else:
        true_initial = first_balance

    ann_factor = ANN_FACTOR
    benchmark_return_pct = ((returns_df['benchmark_return'] + 1).prod() - 1) * 100
    
    # 4. CALCULATE METRICS IN GROUPS
    stats = {
        '0': {
            'risk_adjusted': _calculate_risk_adjusted_ratios(
                portfolio_returns, timestamps, benchmark_return_pct, 0.0, ann_factor,
                returns_df['benchmark_return'].values
            ),
            'risk_metrics': _calculate_risk_metrics(
                portfolio_returns
            ),
            'drawdown_analysis': _calculate_drawdown_analysis(
                portfolio_returns, balance_array, timestamps
            ),
            'trade_analysis': calculate_trade_analysis(df),
            'profit_loss': _calculate_profit_loss(
                df, true_initial=true_initial, returns_df=returns_df
            ),
            'long_short': _calculate_long_short_analysis(
                df, df['direction'].values if 'direction' in df.columns else np.array([]),
                df['action'].values if 'action' in df.columns else np.array([])
            ),
            'portfolio_values': _calculate_portfolio_values(
                df, balance_array, timestamps, benchmark_return_pct,
                returns_df=returns_df, true_initial=true_initial
            ),
            'exposure': _calculate_exposure(
                df, df['direction'].values if 'direction' in df.columns else np.array([]),
                df['action'].values if 'action' in df.columns else np.array([])
            ),
            'cash_flow': _calculate_cash_flow(
                balance_array, np.zeros(len(df)), df,
                returns_df=returns_df, true_initial=true_initial
            ),
            'time_series_analysis': _calculate_time_series_analysis(
                portfolio_returns, balance_array, timestamps, ann_factor
            ),
            'benchmark_analysis': calculate_benchmark_analysis(
                df, benchmark_series=kwargs.get('benchmark_returns'), ann_factor=ann_factor,
                portfolio_daily_returns=returns_df['portfolio_return'].values if returns_df is not None and not returns_df.empty else None,
                benchmark_daily_returns=returns_df['benchmark_return'].values if returns_df is not None and not returns_df.empty else None,
            ),
            'distribution_analysis': _calculate_distribution_analysis(
                portfolio_returns, balance_array, ann_factor
            )
        }
    }
    
    # Add Plot Data (matching VBT format)
    from .plots import (
        get_data_cumulative_returns_custom,
        get_data_underwater_custom,
        get_mfe_mae_custom,
        get_data_pnl_distribution_custom,
        get_data_directional_pnl_custom,
        get_data_rolling_sharpe_custom,
        get_data_rolling_correlation_custom,
        get_data_monthly_heatmap_custom,
        get_benchmark_returns_custom
    )
    
    plot_data = {}
    plot_data.update(get_data_cumulative_returns_custom(returns_df=returns_df, df_ledger=df))
    plot_data.update(get_data_underwater_custom(returns_df=returns_df, df_ledger=df))
    plot_data.update(get_mfe_mae_custom(df))
    plot_data.update(get_data_pnl_distribution_custom(df))
    plot_data.update(get_data_directional_pnl_custom(df))
    plot_data.update(get_data_rolling_sharpe_custom(returns_df=returns_df, df_ledger=df))
    monthly_heatmap = get_data_monthly_heatmap_custom(df, returns_df=returns_df)
    plot_data.update(monthly_heatmap)
    plot_data.update(get_benchmark_returns_custom(returns_df=returns_df, benchmark_data=benchmark_series))
    if 'benchmark_return' in returns_df.columns:
        benchmark_series_aligned = returns_df.set_index('datetime')['benchmark_return']
        plot_data.update(get_data_rolling_correlation_custom(df, benchmark_series_aligned))
        
    stats['0'].update(plot_data)

    # Also expose years, months, monthly_matrix at the top level for compatibility
    result = _round_stats_dict(stats)
    for key in ['years', 'months', 'monthly_matrix']:
        if key in stats['0']:
            result[key] = stats['0'][key]
    return result

def calculate_regime_stats(df_ledger: pd.DataFrame, initial_balance: float = 10000.0) -> Dict[str, Any]:
    """
    Calculate regime-specific statistics from ledger data.
    Optimized for regime analysis - recalculates balance from scratch for consistency.
    
    Args:
        df_ledger: Ledger DataFrame containing trade data
        initial_balance: Starting balance for calculations (default: 10000.0)
        
    Returns:
        Dict containing:
            - trade_count: Number of trades
            - win_rate_pct: Win rate percentage
            - avg_pnl_per_trade: Average PnL per trade
            - total_pnl: Total PnL percentage
            - profit_factor: Profit factor
            - avg_trade_duration_days: Average trade duration in days
            - pct_of_total_trades: Percentage of total trades (set to 100 for single group)
            - max_drawdown_pct: Maximum drawdown percentage
    """
    if df_ledger is None or len(df_ledger) == 0:
        return {
            'trade_count': 0,
            'win_rate_pct': 0.0,
            'avg_pnl_per_trade': 0.0,
            'total_pnl': 0.0,
            'profit_factor': 0.0,
            'avg_trade_duration_days': 0.0,
            'pct_of_total_trades': 0.0,
            'max_drawdown_pct': 0.0,
        }
    
    try:
        # 1. Adapt ledger format - no copy needed, _adapt_ledger_format returns modified df
        df = _adapt_ledger_format(df_ledger)
        
        # 2. Sort by exit datetime to ensure chronological order
        if 'exit_datetime' in df.columns:
            df = df.sort_values('exit_datetime').reset_index(drop=True)
        
        # 3. Recalculate balance from account_return_pct (new format)
        if 'account_return_pct' in df.columns:
            pnl_pct_array = df['account_return_pct'].values
        else:
            pnl_pct_array = np.zeros(len(df))

        # Recompute balance
        balance = initial_balance * np.cumprod(1 + pnl_pct_array / 100)
        df['balance'] = balance
        df['account_return_pct'] = pnl_pct_array
        
        # 4. Create aligned returns for drawdown calculation
        returns_df = _create_aligned_returns_from_ledger(df, benchmark_series=None)
        
        if len(returns_df) < 1:
            return {
                'trade_count': 0,
                'win_rate_pct': 0.0,
                'avg_pnl_per_trade': 0.0,
                'total_pnl': 0.0,
                'profit_factor': 0.0,
                'avg_trade_duration_days': 0.0,
                'pct_of_total_trades': 0.0,
                'max_drawdown_pct': 0.0,
            }
        
        # Extract arrays
        portfolio_returns = returns_df['portfolio_return'].values
        balance_array = returns_df['balance'].values
        timestamps = returns_df['datetime'].values
        
        # 5. Call specific calculation functions directly with regime_analysis flag
        trade_analysis = calculate_trade_analysis(df, regime_analysis=True)
        profit_loss = _calculate_profit_loss(df, true_initial=initial_balance, regime_analysis=True)
        drawdown_analysis = _calculate_drawdown_analysis(portfolio_returns, balance_array, timestamps, regime_analysis=True)
        
        # 6. Extract required metrics
        total_trades_group = len(df)  # Each row is a trade in regime analysis
        total_return = profit_loss.get('total_return_pct', 0.0)
        
        return {
            'trade_count': total_trades_group,
            'win_rate_pct': trade_analysis.get('win_rate_pct', 0.0),
            'avg_pnl_per_trade': total_return / total_trades_group if total_trades_group > 0 else 0.0,
            'total_pnl': total_return,
            'profit_factor': profit_loss.get('profit_factor', 0.0),
            'avg_trade_duration_days': trade_analysis.get('avg_duration_trades', 0.0),
            'pct_of_total_trades': 100.0,  # Will be recalculated by caller if needed
            'max_drawdown_pct': drawdown_analysis.get('max_drawdown_pct', 0.0),
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_essential_stats_multi(ledger_input) -> Dict[str, Any]:
    """
    Multi-strategy wrapper for calculate_essential_stats
    
    Args:
        ledger_input: Can be:
                     - Single DataFrame (single strategy)
                     - Dict of DataFrames (multi-strategy: {'strategy_name': ledger_df})
    
    Returns:
        - Single strategy: {'metric': value, ...}
        - Multi-strategy: {'strategy_name': {'metric': value, ...}, ...}
    """
    if isinstance(ledger_input, pd.DataFrame):
        return calculate_essential_stats(ledger_input)
    
    if isinstance(ledger_input, dict):
        results = {}
        for name, df in ledger_input.items():
            results[name] = calculate_essential_stats(df)
        return results
    
    return {}

def calculate_comprehensive_stats_multi(ledger_input) -> Dict[str, Any]:
    """
    Multi-strategy wrapper for calculate_comprehensive_stats
    
    Args:
        ledger_input: Can be:
                     - Single DataFrame (single strategy)  
                     - Dict of DataFrames (multi-strategy: {'strategy_name': ledger_df})
    
    Returns:
        - Single strategy: {'metric': value, ...}
        - Multi-strategy: {'strategy_name': {'metric': value, ...}, ...}
    """
    if isinstance(ledger_input, pd.DataFrame):
        return calculate_comprehensive_stats(ledger_input)
    
    if isinstance(ledger_input, dict):
        results = {}
        for name, df in ledger_input.items():
            results[name] = calculate_comprehensive_stats(df)
        return results
    
    return {}