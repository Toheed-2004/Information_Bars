"""
Fully vectorized portfolio analysis for ALL strategies at once.
Operates on daily returns arrays with pure numpy - no pandas after input.
"""

import numpy as np
from bitpredict.common.stats.custom_1.utils import LedgerArrays, BatchedReturns, COL_ACC_RET, COL_BALANCE


def calculate_portfolio_values(
    stacked: LedgerArrays,
    batched: BatchedReturns,
) -> np.ndarray:
    """
    Batched portfolio values analysis for ALL strategies in a single pass.
    
    Parameters
    ----------
    stacked : LedgerArrays
        Pre-stacked ledger arrays with shape (n_strats, max_trades, n_cols)
    batched : BatchedReturns
        Daily returns and balances with shape (max_days, n_strats) [TRANSPOSED]
    
    Returns
    -------
    Structured array (n_strats,) with all portfolio metrics
    """
    n_strats = len(stacked.names)
    max_days = batched.daily_returns_2d.shape[0]
    
    # Extract arrays - TRANSPOSED: (max_days, n_strats)
    daily_returns = batched.daily_returns_2d  # (max_days, n_strats)
    daily_balances = batched.daily_balances_2d  # (max_days, n_strats)
    valid_mask = batched.valid_mask_2d  # (max_days, n_strats)
    benchmark_returns = batched.benchmark_returns_1d  # (max_days,)
    day_timestamps = batched.day_timestamps_ns  # (max_days,)
    n_days = batched.n_days_per_strat  # (n_strats,)
    
    # ── Initial value calculation (vectorized) ────────────────────────────
    # Use first balance and first account return to back out initial value
    first_balance = daily_balances[0, :]  # (n_strats,)
    
    # Get first trade's account return from ledger
    first_acc_ret = stacked.numeric_3d[:, 0, COL_ACC_RET] / 100.0  # (n_strats,)
    denom = 1.0 + first_acc_ret
    initial_value = np.where(denom != 0, first_balance / denom, first_balance)
    
    # ── Final value (vectorized) ──────────────────────────────────────────
    # Get last valid balance for each strategy
    last_idx = (n_days - 1).astype(np.int64)  # (n_strats,)
    final_value = daily_balances[last_idx, np.arange(n_strats)]  # (n_strats,)
    
    # ── Value statistics (vectorized with masking) ────────────────────────
    # Mask invalid days with NaN for proper statistics
    masked_balances = np.where(valid_mask, daily_balances, np.nan)  # (max_days, n_strats)
    
    min_value = np.nanmin(masked_balances, axis=0)  # (n_strats,)
    max_value = np.nanmax(masked_balances, axis=0)  # (n_strats,)
    mean_value = np.nanmean(masked_balances, axis=0)  # (n_strats,)
    median_value = np.nanmedian(masked_balances, axis=0)  # (n_strats,)
    volatility = np.nanstd(masked_balances, axis=0, ddof=1)  # (n_strats,)
    
    percentile_25 = np.nanpercentile(masked_balances, 25, axis=0)  # (n_strats,)
    percentile_75 = np.nanpercentile(masked_balances, 75, axis=0)  # (n_strats,)
    
    # Coefficient of variation: std / mean (no * 100, matching VBT)
    coefficient_of_variation = np.where(mean_value != 0, 
                                       np.nanstd(masked_balances, axis=0) / mean_value, 
                                       0.0)
    
    # ── Max drawdown in dollars (vectorized) ──────────────────────────────
    # Use 0.0 for invalid days to avoid affecting cummax
    filled_balances = np.where(valid_mask, daily_balances, 0.0)  # (max_days, n_strats)
    peak = np.maximum.accumulate(filled_balances, axis=0)  # (max_days, n_strats)
    drawdown_dollar = peak - filled_balances  # (max_days, n_strats)
    
    # Mask invalid days before taking max
    masked_dd = np.where(valid_mask, drawdown_dollar, 0.0)
    max_drawdown_dollar = np.max(masked_dd, axis=0)  # (n_strats,)
    
    # ── Return metrics (vectorized) ───────────────────────────────────────
    total_return_pct = ((final_value / initial_value) - 1) * 100  # (n_strats,)
    total_return_dollar = final_value - initial_value  # (n_strats,)
    
    # ── Period analysis from daily returns (vectorized) ───────────────────
    masked_returns = np.where(valid_mask, daily_returns, np.nan)  # (max_days, n_strats)
    
    avg_return_per_period = np.nanmean(masked_returns, axis=0) * 100  # (n_strats,)
    std_return_per_period = np.nanstd(masked_returns, axis=0, ddof=1) * 100  # (n_strats,)
    best_period_return = np.nanmax(masked_returns, axis=0) * 100  # (n_strats,)
    worst_period_return = np.nanmin(masked_returns, axis=0) * 100  # (n_strats,)
    
    # Period distribution
    positive_periods = np.sum((masked_returns > 0) & valid_mask, axis=0)  # (n_strats,)
    negative_periods = np.sum((masked_returns < 0) & valid_mask, axis=0)  # (n_strats,)
    flat_periods = np.sum((masked_returns == 0) & valid_mask, axis=0)  # (n_strats,)
    
    positive_periods_pct = np.where(n_days > 0, positive_periods / n_days * 100, 0.0)
    
    # ── Duration (vectorized) ─────────────────────────────────────────────
    # Get first and last valid timestamps for each strategy
    NS_PER_DAY = np.int64(86_400_000_000_000)
    
    # First valid day is always day 0 for each strategy
    start_timestamps = day_timestamps[0]  # scalar - same for all strategies
    
    # Last valid day varies per strategy
    end_timestamps = day_timestamps[last_idx]  # (n_strats,)
    
    # Duration in various units
    duration_ns = end_timestamps - start_timestamps  # (n_strats,)
    total_duration_days = duration_ns.astype(np.float64) / NS_PER_DAY
    total_duration_hours = duration_ns.astype(np.float64) / (NS_PER_DAY / 24)
    total_duration_minutes = duration_ns.astype(np.float64) / (NS_PER_DAY / 1440)
    
    # Average period length
    avg_period_length_hours = np.where(n_days > 1, 
                                       total_duration_hours / n_days, 
                                       0.0)
    
    # ── Benchmark comparison (vectorized) ─────────────────────────────────
    # Benchmark return: cumulative return over each strategy's valid period
    # Use only valid days for each strategy
    benchmark_return_pct = np.zeros(n_strats)
    for s in range(n_strats):
        n = int(n_days[s])
        if n > 0:
            bm_rets = benchmark_returns[:n]
            benchmark_return_pct[s] = (np.prod(1 + bm_rets) - 1) * 100
    
    # Outperformance: strategy - benchmark (simple difference)
    outperformance = total_return_pct - benchmark_return_pct
    
    # Outperformance ratio: (1 + strategy) / (1 + benchmark)
    outperformance_ratio = np.where(
        benchmark_return_pct != -100,
        (1 + total_return_pct / 100) / (1 + benchmark_return_pct / 100),
        0.0
    )
    
    # ── Additional VBT-compatible metrics ─────────────────────────────────
    # Skewness and kurtosis (numpy-based for speed)
    period_return_skewness = np.zeros(n_strats)
    period_return_kurtosis = np.zeros(n_strats)
    
    for s in range(n_strats):
        n = int(n_days[s])
        if n > 3:
            rets = masked_returns[:n, s]
            valid_rets = rets[~np.isnan(rets)]
            if len(valid_rets) > 3:
                mean_ret = np.mean(valid_rets)
                std_ret = np.std(valid_rets, ddof=1)
                if std_ret > 0:
                    # Skewness
                    period_return_skewness[s] = np.mean(((valid_rets - mean_ret) / std_ret) ** 3)
                    # Kurtosis (excess kurtosis)
                    period_return_kurtosis[s] = np.mean(((valid_rets - mean_ret) / std_ret) ** 4) - 3
    
    # Cumulative return (decimal, not pct)
    cumulative_return_final = (final_value / initial_value) - 1  # (n_strats,)
    
    # Start and end dates (formatted as "YYYY-MM-DD HH:MM:SS")
    start_dates = np.empty(n_strats, dtype='U19')
    end_dates = np.empty(n_strats, dtype='U19')
    
    # Convert nanosecond timestamps to datetime64[ns] then to string
    # start_timestamps is a scalar int64 (nanoseconds)
    start_dt = np.datetime64(int(start_timestamps), 'ns')
    start_dt_str = str(start_dt).replace('T', ' ')
    
    for s in range(n_strats):
        start_dates[s] = start_dt_str
        # end_timestamps[s] is int64 (nanoseconds)
        end_dt = np.datetime64(int(end_timestamps[s]), 'ns')
        end_dt_str = str(end_dt).replace('T', ' ')
        end_dates[s] = end_dt_str
    
    # ── Build structured array (VBT-compatible order) ─────────────────────
    dtype = [
        # VBT naming: portfolio_value_* (current, initial, min, max, mean, median, volatility)
        ('portfolio_value_current', 'f8'),
        ('portfolio_value_initial', 'f8'),
        ('portfolio_initial_value', 'f8'),  # Keep for backward compatibility
        ('portfolio_final_value', 'f8'),    # Keep for backward compatibility
        ('portfolio_value_min', 'f8'),
        ('portfolio_value_max', 'f8'),
        ('portfolio_value_mean', 'f8'),
        ('portfolio_value_median', 'f8'),
        ('portfolio_value_volatility', 'f8'),
        
        # Statistical metrics
        ('percentile_25', 'f8'),
        ('percentile_75', 'f8'),
        ('coefficient_of_variation', 'f8'),
        ('max_drawdown_dollar', 'f8'),
        
        # Cash balance (VBT naming)
        ('cash_balance_current', 'f8'),
        ('cash_balance_initial', 'f8'),
        
        # Period returns (decimal, not pct)
        ('period_return_mean', 'f8'),
        ('period_return_volatility', 'f8'),
        ('period_return_min', 'f8'),
        ('period_return_max', 'f8'),
        ('period_return_skewness', 'f8'),
        ('period_return_kurtosis', 'f8'),
        
        # Total periods
        ('total_periods', 'i8'),
        
        # Period return analysis (pct)
        ('avg_return_per_period_pct', 'f8'),
        ('std_return_per_period_pct', 'f8'),
        ('best_period_return_pct', 'f8'),
        ('worst_period_return_pct', 'f8'),
        
        # Period distribution
        ('positive_periods', 'i8'),
        ('negative_periods', 'i8'),
        ('flat_periods', 'i8'),
        ('positive_periods_pct', 'f8'),
        
        # Cumulative return (decimal)
        ('cumulative_return_final', 'f8'),
        
        # Total return (pct)
        ('total_return_pct', 'f8'),
        
        # Daily return metrics (decimal, not pct)
        ('daily_return_mean', 'f8'),
        ('daily_return_volatility', 'f8'),
        
        # Start and end dates
        ('start_date', 'U19'),
        ('end_date', 'U19'),
        
        # Time and period analysis
        ('total_duration_days', 'f8'),
        ('total_duration_hours', 'f8'),
        ('total_duration_minutes', 'f8'),
        ('avg_period_length_hours', 'f8'),
        
        # Backward compatibility
        ('initial_value', 'f8'),
        ('final_value', 'f8'),
        ('max_value', 'f8'),
        ('min_value', 'f8'),
        
        # Benchmark comparison
        ('benchmark_return_pct', 'f8'),
        ('total_return_dollar', 'f8'),
        ('outperformance', 'f8'),
        ('outperformance_ratio', 'f8'),
    ]
    
    result = np.zeros(n_strats, dtype=dtype)
    
    # VBT naming
    result['portfolio_value_current'] = final_value
    result['portfolio_value_initial'] = initial_value
    result['portfolio_initial_value'] = initial_value
    result['portfolio_final_value'] = final_value
    result['portfolio_value_min'] = min_value
    result['portfolio_value_max'] = max_value
    result['portfolio_value_mean'] = mean_value
    result['portfolio_value_median'] = median_value
    result['portfolio_value_volatility'] = volatility
    result['percentile_25'] = percentile_25
    result['percentile_75'] = percentile_75
    result['coefficient_of_variation'] = coefficient_of_variation
    result['max_drawdown_dollar'] = max_drawdown_dollar
    
    # Cash balance
    result['cash_balance_current'] = final_value
    result['cash_balance_initial'] = initial_value
    
    # Period returns (decimal)
    result['period_return_mean'] = avg_return_per_period / 100
    result['period_return_volatility'] = std_return_per_period / 100
    result['period_return_min'] = worst_period_return / 100
    result['period_return_max'] = best_period_return / 100
    result['period_return_skewness'] = period_return_skewness
    result['period_return_kurtosis'] = period_return_kurtosis
    
    # Total periods
    result['total_periods'] = n_days.astype(np.int64)
    
    # Period return analysis (pct)
    result['avg_return_per_period_pct'] = avg_return_per_period
    result['std_return_per_period_pct'] = std_return_per_period
    result['best_period_return_pct'] = best_period_return
    result['worst_period_return_pct'] = worst_period_return
    
    # Period distribution
    result['positive_periods'] = positive_periods.astype(np.int64)
    result['negative_periods'] = negative_periods.astype(np.int64)
    result['flat_periods'] = flat_periods.astype(np.int64)
    result['positive_periods_pct'] = positive_periods_pct
    
    # Cumulative return (decimal)
    result['cumulative_return_final'] = cumulative_return_final
    
    # Total return (pct)
    result['total_return_pct'] = total_return_pct
    
    # Daily return metrics (decimal)
    result['daily_return_mean'] = avg_return_per_period / 100
    result['daily_return_volatility'] = std_return_per_period / 100
    
    # Start and end dates
    result['start_date'] = start_dates
    result['end_date'] = end_dates
    
    # Time and period analysis
    result['total_duration_days'] = total_duration_days
    result['total_duration_hours'] = total_duration_hours
    result['total_duration_minutes'] = total_duration_minutes
    result['avg_period_length_hours'] = avg_period_length_hours
    
    # Backward compatibility
    result['initial_value'] = initial_value
    result['final_value'] = final_value
    result['max_value'] = max_value
    result['min_value'] = min_value
    
    # Benchmark comparison
    result['benchmark_return_pct'] = benchmark_return_pct
    result['total_return_dollar'] = total_return_dollar
    result['outperformance'] = outperformance
    result['outperformance_ratio'] = outperformance_ratio
    
    return result
