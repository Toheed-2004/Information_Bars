import numpy as np
import pandas as pd
from typing import Dict, Any

def _get_empty_portfolio_values() -> Dict[str, Any]:
    return {
        # Core portfolio values
        'portfolio_initial_value': 0.0, 'portfolio_final_value': 0.0,
        'min_value': 0.0, 'max_value': 0.0, 'portfolio_value_mean': 0.0,
        'portfolio_value_median': 0.0, 'portfolio_value_volatility': 0.0,
        
        # Return metrics
        'total_return_pct': 0.0, 'total_return_dollar': 0.0,
        
        # Statistical metrics
        'percentile_25': 0.0, 'percentile_75': 0.0,
        'coefficient_of_variation': 0.0, 'max_drawdown_dollar': 0.0,
        
        # Time and period analysis
        'start_date': None, 'end_date': None, 'total_duration_days': 0.0,
        'total_duration_hours': 0.0, 'total_duration_minutes': 0.0,
        'total_periods': 0, 'avg_period_length_hours': 0.0,
        
        # Period return analysis
        'avg_return_per_period_pct': 0.0, 'std_return_per_period_pct': 0.0,
        'best_period_return_pct': 0.0, 'worst_period_return_pct': 0.0,
        
        # Period distribution
        'positive_periods': 0, 'negative_periods': 0, 'flat_periods': 0,
        'positive_periods_pct': 0.0,
        
        # Benchmark comparison
        'benchmark_return_pct': 0.0, 'outperformance': 0.0,
        'outperformance_ratio': 0.0
    }

def _calculate_portfolio_values(
    df_ledger: pd.DataFrame,
    balance_array: np.ndarray,
    timestamps: np.ndarray,
    benchmark_return_pct: float,
    returns_df: pd.DataFrame = None,
    true_initial: float = None
) -> Dict[str, Any]:
    """Portfolio values analysis - uses daily returns_df when available to match VBT."""
    
    if len(balance_array) == 0:
        return _get_empty_portfolio_values()
    
    # ── Use daily value array from returns_df if available ────────────────
    if returns_df is not None and not returns_df.empty and 'balance' in returns_df.columns:
        value_array = returns_df['balance'].values
        daily_returns = returns_df['portfolio_return'].values
        start_date = pd.to_datetime(returns_df['datetime'].iloc[0])
        end_date = pd.to_datetime(returns_df['datetime'].iloc[-1])
    else:
        # Fallback to ledger balance array
        value_array = balance_array
        daily_returns = np.diff(balance_array) / balance_array[:-1] if len(balance_array) > 1 else np.array([])
        if 'exit_datetime' in df_ledger.columns:
            start_date = pd.to_datetime(df_ledger['exit_datetime'].iloc[0])
            end_date = pd.to_datetime(df_ledger['exit_datetime'].iloc[-1])
        else:
            start_date = end_date = pd.Timestamp.now()

    # ── Initial value ─────────────────────────────────────────────────────
    if true_initial is not None:
        initial_value = true_initial
    elif returns_df is not None and not returns_df.empty:
        initial_value = float(value_array[0])
    else:
        if 'account_return_pct' in df_ledger.columns and len(df_ledger) > 0:
            first_ret = float(df_ledger['account_return_pct'].iloc[0]) / 100.0
            denom = 1.0 + first_ret
            initial_value = float(balance_array[0] / denom) if denom != 0 else float(balance_array[0])
        else:
            initial_value = float(balance_array[0])

    final_value = float(value_array[-1])

    # ── Value statistics ──────────────────────────────────────────────────
    min_value = float(np.min(value_array))
    max_value = float(np.max(value_array))
    mean_value = float(np.mean(value_array))
    median_value = float(np.median(value_array))
    volatility = float(np.std(value_array, ddof=1)) if len(value_array) > 1 else 0.0
    percentile_25 = float(np.percentile(value_array, 25))
    percentile_75 = float(np.percentile(value_array, 75))
    
    # coefficient_of_variation: std / mean (no * 100, matching VBT)
    coefficient_of_variation = float(np.std(value_array) / mean_value) if mean_value != 0 else 0.0
    
    # max_drawdown_dollar: max(peak - value) in dollar terms (matching VBT)
    peak = np.maximum.accumulate(value_array)
    drawdown_dollar = peak - value_array
    max_drawdown_dollar = float(np.max(drawdown_dollar))

    # ── Return metrics from daily returns ─────────────────────────────────
    total_return_pct = ((final_value / initial_value) - 1) * 100
    total_return_dollar = final_value - initial_value

    # ── Period analysis from daily returns ────────────────────────────────
    n = len(daily_returns)
    if n > 0:
        avg_return_per_period = float(np.mean(daily_returns) * 100)
        std_return_per_period = float(np.std(daily_returns, ddof=1) * 100) if n > 1 else 0.0
        best_period_return = float(np.max(daily_returns) * 100)
        worst_period_return = float(np.min(daily_returns) * 100)
        positive_periods = int(np.sum(daily_returns > 0))
        negative_periods = int(np.sum(daily_returns < 0))
        flat_periods = int(np.sum(daily_returns == 0))
        positive_periods_pct = float(positive_periods / n * 100)
        total_periods = n
    else:
        avg_return_per_period = std_return_per_period = 0.0
        best_period_return = worst_period_return = 0.0
        positive_periods = negative_periods = flat_periods = 0
        positive_periods_pct = 0.0
        total_periods = len(value_array)

    # ── Duration ──────────────────────────────────────────────────────────
    total_duration = end_date - start_date
    total_duration_days = float(total_duration.total_seconds() / 86400)
    total_duration_hours = float(total_duration.total_seconds() / 3600)
    total_duration_minutes = float(total_duration.total_seconds() / 60)
    avg_period_length_hours = total_duration_hours / total_periods if total_periods > 1 else 0.0

    # ── Benchmark comparison ──────────────────────────────────────────────
    # outperformance: strategy - benchmark (simple difference)
    outperformance = total_return_pct - benchmark_return_pct
    # outperformance_ratio: (1 + strategy) / (1 + benchmark) matching VBT
    outperformance_ratio = (
        (1 + total_return_pct / 100) / (1 + benchmark_return_pct / 100)
        if benchmark_return_pct != -100 else 0.0
    )

    return {
        'portfolio_initial_value': float(initial_value),
        'portfolio_final_value': float(final_value),
        'min_value': min_value,
        'max_value': max_value,
        'portfolio_value_mean': mean_value,
        'portfolio_value_median': median_value,
        'portfolio_value_volatility': volatility,
        'total_return_pct': float(total_return_pct),
        'total_return_dollar': float(total_return_dollar),
        'percentile_25': percentile_25,
        'percentile_75': percentile_75,
        'coefficient_of_variation': coefficient_of_variation,
        'max_drawdown_dollar': max_drawdown_dollar,
        'start_date': start_date,
        'end_date': end_date,
        'total_duration_days': total_duration_days,
        'total_duration_hours': total_duration_hours,
        'total_duration_minutes': total_duration_minutes,
        'total_periods': int(total_periods),
        'avg_period_length_hours': float(avg_period_length_hours),
        'avg_return_per_period_pct': avg_return_per_period,
        'std_return_per_period_pct': std_return_per_period,
        'best_period_return_pct': best_period_return,
        'worst_period_return_pct': worst_period_return,
        'positive_periods': positive_periods,
        'negative_periods': negative_periods,
        'flat_periods': flat_periods,
        'positive_periods_pct': positive_periods_pct,
        'benchmark_return_pct': float(benchmark_return_pct),
        'outperformance': float(outperformance),
        'outperformance_ratio': float(outperformance_ratio),
    }
