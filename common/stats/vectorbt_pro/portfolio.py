import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats as scipy_stats


def _extract_value_stats_vectorized(cache: Dict) -> Dict[str, Any]:
    """Portfolio value statistics from cached daily arrays."""
    stats: Dict[str, Any] = {}
    value_array = cache.get('value_array', np.array([]))
    cash_array  = cache.get('cash_array',  np.array([]))

    if len(value_array) > 0:
        peak            = np.maximum.accumulate(value_array)
        drawdown_dollar = peak - value_array
        stats.update({
            'portfolio_value_current':     float(value_array[-1]),
            'portfolio_value_initial':     float(value_array[0]),
            'portfolio_initial_value':     float(value_array[0]),
            'portfolio_final_value':       float(value_array[-1]),
            'portfolio_value_min':         float(np.min(value_array)),
            'portfolio_value_max':         float(np.max(value_array)),
            'portfolio_value_mean':        float(np.mean(value_array)),
            'portfolio_value_median':      float(np.median(value_array)),
            'portfolio_value_volatility':  float(np.std(value_array, ddof=1)) if len(value_array) > 1 else 0.0,
            'percentile_25':               float(np.percentile(value_array, 25)),
            'percentile_75':               float(np.percentile(value_array, 75)),
            'coefficient_of_variation':    (
                float(np.std(value_array) / np.mean(value_array))
                if np.mean(value_array) != 0 else 0.0
            ),
            'max_drawdown_dollar':         float(np.max(drawdown_dollar)),
        })

    if len(cash_array) > 0:
        stats['cash_balance_current'] = float(cash_array[-1])
        stats['cash_balance_initial'] = float(cash_array[0])

    return stats


def _extract_return_stats_vectorized(cache: Dict) -> Dict[str, Any]:
    """
    Return statistics from cached daily_returns.
    Kurtosis: excess (fisher=True, normal=0) — consistent with risk.py and distribution.py.
    """
    stats: Dict[str, Any] = {}
    returns        = cache.get('daily_returns', np.array([]))
    cumulative_arr = np.array([])
    daily_arr      = returns

    # Cumulative returns from value array
    value_array = cache.get('value_array', np.array([]))
    if len(value_array) > 1 and value_array[0] != 0:
        cumulative_arr = value_array / value_array[0] - 1

    if len(returns) > 0:
        n = len(returns)
        stats.update({
            'period_return_mean':         float(np.mean(returns)),
            'period_return_volatility':   float(np.std(returns, ddof=1)) if n > 1 else 0.0,
            'period_return_min':          float(np.min(returns)),
            'period_return_max':          float(np.max(returns)),
            'period_return_skewness':     float(scipy_stats.skew(returns, bias=False)) if n > 2 else 0.0,
            'period_return_kurtosis':     float(scipy_stats.kurtosis(returns, bias=False, fisher=True)) if n > 3 else 0.0,
            'total_periods':              int(n),
            'avg_return_per_period_pct':  float(np.mean(returns) * 100),
            'std_return_per_period_pct':  float(np.std(returns, ddof=1) * 100) if n > 1 else 0.0,
            'best_period_return_pct':     float(np.max(returns) * 100),
            'worst_period_return_pct':    float(np.min(returns) * 100),
            'positive_periods':           int(np.sum(returns > 0)),
            'negative_periods':           int(np.sum(returns < 0)),
            'flat_periods':               int(np.sum(returns == 0)),
            'positive_periods_pct':       float(np.sum(returns > 0) / n * 100),
        })

    if len(cumulative_arr) > 0:
        stats['cumulative_return_final'] = float(cumulative_arr[-1])
        stats['total_return_pct']        = float(cumulative_arr[-1] * 100)

    if len(daily_arr) > 0:
        stats['daily_return_mean']       = float(np.mean(daily_arr))
        stats['daily_return_volatility'] = float(np.std(daily_arr, ddof=1)) if len(daily_arr) > 1 else 0.0

    return stats


def _extract_essential_value_stats(cache: Dict) -> Dict[str, Any]:
    """Initial and final portfolio value only."""
    value_array = cache.get('value_array', np.array([]))
    if len(value_array) > 0:
        return {
            'initial_value': float(value_array[0]),
            'final_value':   float(value_array[-1]),
        }
    return {'initial_value': 0.0, 'final_value': 0.0}

