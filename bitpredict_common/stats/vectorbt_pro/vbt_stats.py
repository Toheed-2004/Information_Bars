"""
VectorBT Pro statistics — vectorized, daily-resampled for cross-strategy consistency.

Design:
  - All strategies resampled to daily via pf.resample('1D') before any ratio/return calc.
  - daily_pf.stats() called ONCE and stored in cache — single source of truth.
  - Original pf.trades used for trade-level records (win rate, profit factor, MFE/MAE).
  - Benchmark auto-derived from daily_pf.bm_returns (populated from close price passed to VBT).
  - bar_type and benchmark_returns params removed — both auto-handled.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from .utils import (
    _format_numeric_values_vectorized,
    _get_empty_comprehensive_stats,
    get_benchmark_to_dict,
)
from .cache_builder import _build_vbt_cache, _build_essential_vbt_cache
from .performance import (
    _extract_essential_core_stats,
)
from .plots import (
    get_data_cumulative_returns,
    get_data_monthly_heatmap,
    get_data_underwater,
    get_mfe_mae,
    get_data_pnl_distribution,
    get_data_directional_pnl,
    get_data_rolling_sharpe,
    get_data_rolling_correlation,
    get_data_monte_carlo,
)
from .portfolio import (
    _extract_value_stats_vectorized,
    _extract_return_stats_vectorized,
    _extract_essential_value_stats,
)
from .drawdown import _extract_drawdown_stats_vectorized
from .long_short import get_data_directional_metrics
from .risk import (
    _extract_risk_adjusted_stats_vectorized,
    _extract_risk_metrics_vectorized,
)
from .distribution import _extract_distribution_stats_vectorized
from ..shared.benchmark import calculate_benchmark_analysis
from ..shared.trade import calculate_trade_analysis
from ..shared.utils import ANN_FACTOR
from .profit_loss import _extract_profit_loss_from_cache
from .time_series import _extract_time_series_stats
from .exposure import _extract_exposure_stats_full
from .cash_flow import _extract_cash_flow_stats_full
# Custom ratio helpers are computed inside _extract_risk_adjusted_stats_vectorized
# via _calculate_risk_adjusted_ratios — no direct calls needed here.

try:
    import vectorbtpro as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_essential_stats_only(portfolio_obj) -> Dict[str, Dict[str, Any]]:
    """ULTRA-FAST: Essential stats only (9 core metrics)."""
    if not VBT_AVAILABLE:
        raise ImportError("VectorBT Pro is required but not installed")
    if portfolio_obj is None:
        return {}
    try:
        is_multi_column = getattr(portfolio_obj.wrapper, 'ncol', 1) > 1
        if is_multi_column:
            return {
                str(name): _calculate_essential_single_strategy_stats(portfolio_obj.select_col(name))
                for name in portfolio_obj.wrapper.columns
            }
        # Single-column always keyed as '0' for backward compat with __init__.py
        return {'0': _calculate_essential_single_strategy_stats(portfolio_obj)}
    except Exception as e:
        logger.error(f"Error in essential stats: {e}")
        return {}


def calculate_comprehensive_vbt_stats_optimized(
    portfolio_obj,
    ledger_input=None,
    calculate_monte_carlo: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Comprehensive VBT portfolio statistics.

    All strategies resampled to daily — consistent Sharpe/Sortino/Calmar
    annualisation regardless of original bar type or timeframe.

    Args:
        portfolio_obj: VBT Portfolio object (must have close price for benchmark).
        ledger_input:  Trade ledger DataFrame (always required).
        calculate_monte_carlo: Run Monte Carlo (expensive, off by default).
    """
    if not VBT_AVAILABLE:
        raise ImportError("VectorBT Pro is required but not installed")
    if portfolio_obj is None:
        logger.warning("No portfolio object provided")
        return _get_empty_comprehensive_stats()
    try:
        is_multi_column = getattr(portfolio_obj.wrapper, 'ncol', 1) > 1
        if is_multi_column:
            result = {}
            for name in portfolio_obj.wrapper.columns:
                strat_pf = portfolio_obj.select_col(name)
                strat_ledger = (
                    ledger_input.get(name)
                    if isinstance(ledger_input, dict)
                    else None
                )
                result[name] = _calculate_single_strategy_stats(
                    strat_pf, strat_ledger, calculate_monte_carlo
                )
            return result
        # Single strategy — always keyed as '0' for backward compat
        single_ledger = (
            list(ledger_input.values())[0]
            if isinstance(ledger_input, dict) and ledger_input
            else ledger_input
        )
        return {'0': _calculate_single_strategy_stats(
            portfolio_obj, single_ledger, calculate_monte_carlo
        )}
    except Exception as e:
        logger.error(f"Error in comprehensive stats: {e}")
        return _get_empty_comprehensive_stats()


# ---------------------------------------------------------------------------
# Single-strategy calculation
# ---------------------------------------------------------------------------

def _calculate_single_strategy_stats(
    pf,
    ledger_input: Optional[pd.DataFrame],
    calculate_monte_carlo: bool = False,
) -> Dict[str, Any]:
    """Comprehensive stats for a single-column portfolio."""
    try:
        stats_dict: Dict[str, Any] = {}

        # Build single cache (all expensive ops happen here, exactly once)
        cache = _build_vbt_cache(pf)

        # --- 0. TRADE ANALYSIS ----------------------------------------
        trade_analysis = calculate_trade_analysis(ledger_input)
        stats_dict['trade_analysis'] = trade_analysis

        # --- 1. RISK ADJUSTED ----------------------------------------
        # VBT-first: standard ratios come from daily_pf.stats() (authoritative).
        # Custom metrics not provided by VBT are computed once inside ra.
        ra = _extract_risk_adjusted_stats_vectorized(cache)

        risk_adjusted = {
            # --- V: direct daily_pf attrs (cache steps 12-14) ---
            'sharpe_ratio':              cache.get('sharpe_ratio',  0.0),
            'sortino_ratio':             cache.get('sortino_ratio', 0.0),
            'calmar_ratio':              cache.get('calmar_ratio',  0.0),
            'omega_ratio':               cache.get('omega_ratio',   0.0),
            'alpha':                     cache.get('alpha',             0.0),
            'beta':                      cache.get('beta',              0.0),
            'information_ratio':         cache.get('information_ratio', 0.0),
            'treynor_ratio':             cache.get('treynor_ratio',     0.0),
            'capture_ratio':             cache.get('capture_ratio',     0.0),
            'up_capture_ratio':          cache.get('up_capture_ratio',  0.0),
            'down_capture_ratio':        cache.get('down_capture_ratio',0.0),
            # --- D: custom-only metrics not provided by VBT ---
            'kelly_criterion':           ra.get('kelly_criterion',            0.0),
            'cagr':                      ra.get('cagr',                       0.0),
            'probabilistic_sharpe_ratio':ra.get('probabilistic_sharpe_ratio', 0.0),
            'adjusted_sortino':          ra.get('adjusted_sortino',           0.0),
            'serenity_index':            ra.get('serenity_index',             0.0),
            'expected_return_pct':       ra.get('expected_return_pct',        0.0),
        }
        stats_dict['risk_adjusted'] = risk_adjusted

        # --- 2. PROFIT / LOSS ----------------------------------------
        heatmap_data = get_data_monthly_heatmap(cache)
        top_heatmap  = heatmap_data.get('heatmaps_data', {})

        profit_loss = _extract_profit_loss_from_cache(cache)
        profit_loss['monthly_returns'] = top_heatmap.get('monthly_returns_series', {})
        for k in [
            'monthly_returns_mean_pct', 'monthly_returns_std_pct',
            'best_month_return_pct', 'worst_month_return_pct',
            'positive_months_pct', 'monthly_win_rate',
        ]:
            profit_loss[k] = profit_loss.get(k) or top_heatmap.get(k, 0.0)
        profit_loss['monthly_matrix'] = top_heatmap.get('monthly_matrix', [])
        stats_dict['profit_loss'] = profit_loss

        # --- 3. PORTFOLIO VALUES -------------------------------------
        portfolio_values = _extract_value_stats_vectorized(cache)
        portfolio_values.update(_extract_return_stats_vectorized(cache))

        total_duration_days = cache.get('total_duration_days', 0.0)
        total_periods       = portfolio_values.get('total_periods', 1.0)
        _init_val  = cache.get('initial_value',  portfolio_values.get('portfolio_value_initial', 0.0))
        _final_val = cache.get('final_value',    portfolio_values.get('portfolio_value_current', 0.0))
        _bm_ret    = cache.get('benchmark_return_pct', 0.0)
        _tot_ret   = cache.get('total_return_pct',     portfolio_values.get('total_return_pct', 0.0))

        portfolio_values.update({
            'start_date':              cache.get('start_date'),
            'end_date':                cache.get('end_date'),
            'total_duration_days':     total_duration_days,
            'total_duration_hours':    total_duration_days * 24.0,
            'total_duration_minutes':  total_duration_days * 1440.0,
            'avg_period_length_hours': (total_duration_days * 24.0) / total_periods if total_periods > 0 else 0.0,
            'initial_value':           _init_val,
            'final_value':             _final_val,
            'portfolio_initial_value': _init_val,
            'portfolio_final_value':   _final_val,
            'max_value':               cache.get('max_value', portfolio_values.get('portfolio_value_max', 0.0)),
            'min_value':               cache.get('min_value', portfolio_values.get('portfolio_value_min', 0.0)),
            'benchmark_return_pct':    _bm_ret,
            'total_return_pct':        _tot_ret,
            'total_return_dollar':     _final_val - _init_val,
            'outperformance':          _tot_ret - _bm_ret,
            'outperformance_ratio':    _tot_ret / _bm_ret if _bm_ret != 0 else 0.0,
        })
        stats_dict['portfolio_values'] = portfolio_values

        # --- 4. DRAWDOWN ANALYSIS ------------------------------------
        stats_dict['drawdown_analysis'] = _extract_drawdown_stats_vectorized(cache)

        # --- 5. LONG / SHORT -----------------------------------------
        stats_dict['long_short'] = get_data_directional_metrics(cache)

        # --- 6. EXPOSURE ---------------------------------------------
        stats_dict['exposure'] = _extract_exposure_stats_full(cache)

        # --- 7. CASH FLOW --------------------------------------------
        stats_dict['cash_flow'] = _extract_cash_flow_stats_full(cache)

        # --- 8. RISK METRICS -----------------------------------------
        stats_dict['risk_metrics'] = _extract_risk_metrics_vectorized(cache)

        # --- 9. TIME SERIES ANALYSIS ---------------------------------
        stats_dict['time_series_analysis'] = _extract_time_series_stats(cache)

        # --- 10. BENCHMARK ANALYSIS ----------------------------------
        stats_dict['benchmark_analysis'] = calculate_benchmark_analysis(
            portfolio_daily_returns=cache.get('daily_returns'),
            benchmark_daily_returns=cache.get('bm_returns'),
            ann_factor=ANN_FACTOR,
        )

        # --- 11. DISTRIBUTION ANALYSIS -------------------------------
        stats_dict['distribution_analysis'] = _extract_distribution_stats_vectorized(cache)

        # --- Plot data -----------------------------------------------
        stats_dict.update(get_data_cumulative_returns(cache))
        stats_dict.update(get_data_underwater(cache))
        stats_dict.update(get_mfe_mae(cache))
        stats_dict.update(get_data_pnl_distribution(cache))
        stats_dict.update(get_data_directional_pnl(cache))
        stats_dict.update(get_data_rolling_sharpe(cache))
        # stats_dict.update(heatmap_data)
        stats_dict.update(get_benchmark_to_dict(cache.get('bm_returns_series')))
        stats_dict.update(get_data_rolling_correlation(cache))

        if calculate_monte_carlo:
            mc_data = get_data_monte_carlo(cache, ledger_input)
            if mc_data:
                stats_dict.update(mc_data)

        # Canonical ordering
        _CANONICAL = [
            'risk_adjusted', 'risk_metrics', 'drawdown_analysis', 'trade_analysis',
            'profit_loss', 'long_short', 'portfolio_values', 'exposure', 'cash_flow',
            'time_series_analysis', 'benchmark_analysis', 'distribution_analysis',
            'cumulative_return', 'drawdown_series', 'drawdown_periods',
            'mfe_pct', 'mae_pct', 'pnl_distribution', 'directional_pnl',
            'rolling_sharpe', 'rolling_sortino', 'heatmaps_data',
            'benchmark_returns', 'rolling_correlation',
        ]
        stats_dict = (
            {k: stats_dict[k] for k in _CANONICAL if k in stats_dict}
            | {k: v for k, v in stats_dict.items() if k not in _CANONICAL}
        )

        return _format_numeric_values_vectorized(stats_dict)

    except Exception as e:
        logger.error(f"Error calculating single strategy stats: {e}")
        return {}


def _calculate_essential_single_strategy_stats(pf) -> Dict[str, Any]:
    """Essential stats for a single-column portfolio."""
    try:
        cache = _build_essential_vbt_cache(pf)
        stats_dict: Dict[str, Any] = {}
        stats_dict.update(_extract_essential_core_stats(cache))
        stats_dict.update(_extract_essential_value_stats(cache))
        return _format_numeric_values_vectorized(stats_dict)
    except Exception as e:
        logger.error(f"Error in essential single strategy stats: {e}")
        return {}
