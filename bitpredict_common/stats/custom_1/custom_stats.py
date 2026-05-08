"""
Custom statistics calculator - pure NumPy implementation.
Single entry point: calculate_stats() handles one or multiple strategies.
"""

import numpy as np
import pandas as pd
import time
from typing import Optional

from bitpredict.common.stats.custom_1.utils import (
    prepare_bar_arrays,
    prepare_daily_index,
    prepare_ledger_arrays,
    build_returns_all,
    build_returns_all_vectorized,
)
from bitpredict.common.stats.custom_1.ratios import calculate_risk_adjusted_ratios
from bitpredict.common.stats.custom_1.risk import risk_metrics
from bitpredict.common.stats.custom_1.drawdown import drawdown_analysis
from bitpredict.common.stats.custom_1.trade import calculate_trade_analysis
from bitpredict.common.stats.custom_1.profit_loss import calculate_profit_loss
from bitpredict.common.stats.custom_1.long_short import calculate_long_short_analysis
from bitpredict.common.stats.custom_1.portfolio import calculate_portfolio_values
from bitpredict.common.stats.custom_1.exposure import calculate_exposure_analysis
from bitpredict.common.stats.custom_1.cash_flow import calculate_cash_flow_analysis
from bitpredict.common.stats.custom_1.time_series import calculate_time_series_analysis
from bitpredict.common.stats.custom_1.benchmark import calculate_benchmark_analysis
from bitpredict.common.stats.custom_1.distribution import calculate_distribution_analysis
from bitpredict.common.stats.custom_1.plots import generate_plot_data_batched
from bitpredict.common.stats.custom_1.performance import (
    calculate_performance_batched,
)
from bitpredict.common.stats.custom_1.config import ANN_FACTOR


def calculate_stats(
    ledgers: dict,
    df_bars: pd.DataFrame,
    use_vectorized: bool = False,
) -> dict:
    """
    Calculate stats for one or multiple strategies.

    Parameters
    ----------
    ledgers : dict
        {strategy_name: df_ledger} - pass a single-key dict for one strategy.
    df_bars : pd.DataFrame
        1-minute OHLCV bars, shared across all strategies.
    use_vectorized : bool, optional
        If True, use fully vectorized implementation (faster for 100+ strategies).
        If False, use loop-based implementation (default, more memory efficient).

    Returns
    -------
    dict
        {strategy_name: {"risk_adjusted": {...}, "risk_metrics": {...}, ...}}
    """
    if df_bars is None or df_bars.empty or not ledgers:
        return {}

    t0 = time.time()
    bars = prepare_bar_arrays(df_bars)
    print(f"  prepare_bar_arrays:        {time.time()-t0:.4f}s")
    
    t0 = time.time()
    di = prepare_daily_index(bars)
    print(f"  prepare_daily_index:       {time.time()-t0:.4f}s")
    
    t0 = time.time()
    stacked = prepare_ledger_arrays(ledgers)
    print(f"  prepare_ledger_arrays:     {time.time()-t0:.4f}s")
    
    # Build returns for ALL strategies at once (batched) - TRANSPOSED layout
    t0 = time.time()
    if use_vectorized:
        batched = build_returns_all_vectorized(stacked, bars, di)
        print(f"  build_returns_vectorized:  {time.time()-t0:.4f}s")
    else:
        batched = build_returns_all(stacked, bars, di)
        print(f"  build_returns_all:         {time.time()-t0:.4f}s")
    
    # Compute ALL metrics for ALL strategies in batched mode
    t0 = time.time()
    ratios_results = calculate_risk_adjusted_ratios(
        batched.daily_returns_2d,      # (max_days, n_strats)
        batched.valid_mask_2d,         # (max_days, n_strats)
        batched.benchmark_returns_1d,  # (max_days,)
        risk_free_rate=0.0,
        ann_factor=ANN_FACTOR,
    )
    print(f"  risk_adjusted_ratios:      {time.time()-t0:.4f}s")
    
    t0 = time.time()
    risk_results = risk_metrics(batched.daily_returns_2d, batched.valid_mask_2d)
    print(f"  risk_metrics:      {time.time()-t0:.4f}s")
    
    t0 = time.time()
    dd_results = drawdown_analysis(batched.daily_returns_2d, batched.valid_mask_2d)
    print(f"  drawdown_batched:          {time.time()-t0:.4f}s")
    
    t0 = time.time()
    trade_results = calculate_trade_analysis(stacked)
    print(f"  trade_analysis:            {time.time()-t0:.4f}s")
    
    t0 = time.time()
    pl_results = calculate_profit_loss(stacked)
    print(f"  profit_loss:               {time.time()-t0:.4f}s")
    
    t0 = time.time()
    long_short_results = calculate_long_short_analysis(stacked)
    print(f"  long_short_analysis:       {time.time()-t0:.4f}s")
    
    t0 = time.time()
    portfolio_results = calculate_portfolio_values(stacked, batched)
    print(f"  portfolio_values:          {time.time()-t0:.4f}s")
    
    t0 = time.time()
    exposure_results = calculate_exposure_analysis(stacked, batched)
    print(f"  exposure_analysis:         {time.time()-t0:.4f}s")
    
    t0 = time.time()
    cash_flow_results = calculate_cash_flow_analysis(stacked, batched)
    print(f"  cash_flow_analysis:        {time.time()-t0:.4f}s")
    
    t0 = time.time()
    time_series_results = calculate_time_series_analysis(batched)
    print(f"  time_series_analysis:      {time.time()-t0:.4f}s")
    
    t0 = time.time()
    benchmark_results = calculate_benchmark_analysis(batched)
    print(f"  benchmark_analysis:        {time.time()-t0:.4f}s")
    
    t0 = time.time()
    distribution_results = calculate_distribution_analysis(batched)
    print(f"  distribution_analysis:     {time.time()-t0:.4f}s")

    # Batched performance metrics (monthly breakdown, recent perf, etc.)
    t0 = time.time()
    monthly_breakdowns, recent_perfs, monthly_stats_list, monthly_heatmaps = calculate_performance_batched(batched)
    print(f"  performance_batched:       {time.time()-t0:.4f}s")
    
    # Batched plot generation
    t0 = time.time()
    plot_data_list = generate_plot_data_batched(stacked, batched)
    print(f"  plot_generation_batched:   {time.time()-t0:.4f}s")

    # Build results dict with timing for per-strategy dict conversion
    print(f"\n  Per-strategy dict conversion:")
    t0_total = time.time()
    results = {
        stacked.names[i]: _calc_stats(
            i, ratios_results[i], risk_results[i], dd_results[i], 
            trade_results[i], pl_results[i], long_short_results[i], portfolio_results[i],
            exposure_results[i], cash_flow_results[i], time_series_results[i],
            benchmark_results[i], distribution_results[i],
            monthly_breakdowns[i], recent_perfs[i], monthly_stats_list[i], monthly_heatmaps[i],
            plot_data_list[i]
        )
        for i in range(len(stacked.names))
    }
    print(f"  All strategies converted:  {time.time()-t0_total:.4f}s")
    
    return results


def _calc_stats(idx: int, ratios_result, risk_result, dd_result, trade_result, pl_result, 
                long_short_result, portfolio_result, exposure_result, cash_flow_result, 
                time_series_result, benchmark_result, distribution_result,
                monthly_breakdown, recent_perf, monthly_stats, monthly_heatmap,
                plot_data) -> dict:
    """
    Core pipeline for one strategy.
    All metrics come from batched computation (structured arrays).
    Simply convert structured arrays to dicts and merge pre-computed performance/plot data.
    """
    # Convert structured array results to dicts
    ratios_dict = {name: float(ratios_result[name]) for name in ratios_result.dtype.names}
    risk_dict = {name: float(risk_result[name]) for name in risk_result.dtype.names}
    dd_dict = {name: float(dd_result[name]) if dd_result.dtype[name].kind == 'f' else int(dd_result[name])
               for name in dd_result.dtype.names}
    trade_dict = {name: float(trade_result[name]) if trade_result.dtype[name].kind == 'f' else int(trade_result[name])
                  for name in trade_result.dtype.names}
    pl_dict = {name: float(pl_result[name]) if pl_result.dtype[name].kind == 'f' else int(pl_result[name])
               for name in pl_result.dtype.names}
    long_short_dict = {name: float(long_short_result[name]) if long_short_result.dtype[name].kind == 'f' else int(long_short_result[name])
                       for name in long_short_result.dtype.names}
    portfolio_dict = {}
    for name in portfolio_result.dtype.names:
        if portfolio_result.dtype[name].kind == 'U':  # Unicode string
            portfolio_dict[name] = str(portfolio_result[name])
        elif portfolio_result.dtype[name].kind == 'f':
            portfolio_dict[name] = float(portfolio_result[name])
        else:
            portfolio_dict[name] = int(portfolio_result[name])
    exposure_dict = {name: float(exposure_result[name]) if exposure_result.dtype[name].kind == 'f' else int(exposure_result[name])
                     for name in exposure_result.dtype.names}
    cash_flow_dict = {name: float(cash_flow_result[name]) if cash_flow_result.dtype[name].kind == 'f' else int(cash_flow_result[name])
                      for name in cash_flow_result.dtype.names}
    time_series_dict = {name: float(time_series_result[name]) if time_series_result.dtype[name].kind in ('f', 'i') else int(time_series_result[name])
                        for name in time_series_result.dtype.names}
    benchmark_dict = {name: float(benchmark_result[name]) if benchmark_result.dtype[name].kind == 'f' else int(benchmark_result[name])
                      for name in benchmark_result.dtype.names}
    distribution_dict = {name: float(distribution_result[name]) if distribution_result.dtype[name].kind == 'f' else int(distribution_result[name])
                         for name in distribution_result.dtype.names}

    # Merge pre-computed performance metrics into profit_loss dict
    pl_dict.update(monthly_stats)
    pl_dict['monthly_returns'] = monthly_breakdown
    pl_dict.update(recent_perf)
    pl_dict['monthly_matrix'] = monthly_heatmap.get('monthly_matrix', [])

    return {
        "risk_adjusted":         ratios_dict,
        "risk_metrics":          risk_dict,
        "drawdown_analysis":     dd_dict,
        "trade_analysis":        trade_dict,
        "profit_loss":           pl_dict,
        "long_short":            long_short_dict,
        "portfolio_values":      portfolio_dict,
        "exposure":              exposure_dict,
        "cash_flow":             cash_flow_dict,
        "time_series":           time_series_dict,
        "benchmark_analysis":    benchmark_dict,
        "distribution_analysis": distribution_dict,
        **plot_data,
    }
