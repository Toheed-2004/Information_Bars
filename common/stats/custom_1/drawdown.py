"""
Drawdown analysis - pure NumPy, fully vectorized.
Operates on daily returns array from DailyReturns.
"""

import numpy as np

def drawdown_analysis(returns_2d: np.ndarray, valid_mask_2d: np.ndarray) -> np.ndarray:
    """
    Batch drawdown analysis for multiple strategies.
    
    Parameters
    ----------
    returns_2d : np.ndarray
        Shape (max_days, n_strategies). Daily returns in decimal form.
    valid_mask_2d : np.ndarray
        Boolean array same shape as returns_2d. True = valid observation.
    
    Returns
    -------
    np.ndarray
        Structured array of shape (n_strategies,) with drawdown metrics.
    """
    
    max_days, n_strategies = returns_2d.shape
    
    # Mask invalid returns with NaN for safe processing
    returns_masked = np.where(valid_mask_2d, returns_2d, np.nan)  # (max_days, n_strategies)
    
    # Initialize output arrays for each strategy
    max_drawdown_pct = np.zeros(n_strategies)
    max_drawdown_duration = np.zeros(n_strategies)
    avg_drawdown_pct = np.zeros(n_strategies)
    avg_drawdown_days = np.zeros(n_strategies)
    current_drawdown_pct = np.zeros(n_strategies)
    current_drawdown_days = np.zeros(n_strategies, dtype=np.int32)
    recovery_factor = np.zeros(n_strategies)
    drawdown_periods_count = np.zeros(n_strategies, dtype=np.int32)
    drawdown_total_duration = np.zeros(n_strategies)
    drawdown_volatility_pct = np.zeros(n_strategies)
    
    # Process each strategy (vectorization across strategies is limited due to variable lengths)
    for strategy_idx in range(n_strategies):
        # Extract valid days for this strategy
        valid_days_mask = valid_mask_2d[:, strategy_idx]
        if not valid_days_mask.any():
            continue
        
        # Get clean returns (no NaNs)
        strategy_returns = returns_masked[valid_days_mask, strategy_idx]
        strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
        n_obs = len(strategy_returns)
        
        # ─────────────────────────────────────────────────────────────────────────
        # Cumulative equity and drawdown series
        # ─────────────────────────────────────────────────────────────────────────
        
        equity_curve = np.cumprod(1.0 + strategy_returns)
        running_peak = np.maximum.accumulate(equity_curve)
        drawdown = equity_curve / running_peak - 1.0
        
        # Basic statistics
        max_drawdown_pct[strategy_idx] = abs(np.min(drawdown)) * 100
        current_drawdown_pct[strategy_idx] = abs(drawdown[-1]) * 100
        drawdown_volatility_pct[strategy_idx] = np.std(drawdown, ddof=1) * 100 if n_obs > 1 else 0.0
        
        # ─────────────────────────────────────────────────────────────────────────
        # Identify drawdown periods
        # ─────────────────────────────────────────────────────────────────────────
        
        in_drawdown = drawdown < -1e-10
        previous_in_drawdown = np.empty(n_obs, dtype=np.bool_)
        previous_in_drawdown[0] = False
        previous_in_drawdown[1:] = in_drawdown[:-1]
        
        period_starts = np.flatnonzero(in_drawdown & ~previous_in_drawdown)
        period_ends = np.flatnonzero(~in_drawdown & previous_in_drawdown)
        
        # Handle edge cases
        if period_starts.size and period_ends.size and period_starts[0] > period_ends[0]:
            period_starts = np.insert(period_starts, 0, 0)
        if period_starts.size and (period_ends.size == 0 or period_starts[-1] > period_ends[-1]):
            period_ends = np.append(period_ends, n_obs)
        
        # If no drawdown periods, skip duration calculations
        if period_starts.size == 0:
            current_drawdown_days[strategy_idx] = int(n_obs) if drawdown[-1] < -1e-10 else 0
            continue
        
        # ─────────────────────────────────────────────────────────────────────────
        # Duration metrics
        # ─────────────────────────────────────────────────────────────────────────
        
        period_durations = period_ends - period_starts
        max_drawdown_duration[strategy_idx] = float(period_durations.max())
        avg_drawdown_days[strategy_idx] = float(period_durations.mean())
        drawdown_total_duration[strategy_idx] = float(period_durations.sum())
        drawdown_periods_count[strategy_idx] = int(period_starts.size)
        
        # ─────────────────────────────────────────────────────────────────────────
        # Average drawdown percentage
        # ─────────────────────────────────────────────────────────────────────────
        
        peak_values = np.where(period_starts > 0, equity_curve[period_starts - 1], 1.0)
        valley_values = np.array([equity_curve[start:end].min() 
                                  for start, end in zip(period_starts, period_ends)])
        
        drawdown_percentages = np.where(
            peak_values > 0,
            (peak_values - valley_values) / peak_values * 100,
            0.0
        )
        avg_drawdown_pct[strategy_idx] = float(np.mean(drawdown_percentages))
        
        # ─────────────────────────────────────────────────────────────────────────
        # Current drawdown duration and recovery factor
        # ─────────────────────────────────────────────────────────────────────────
        
        current_drawdown_days[strategy_idx] = int(n_obs - period_starts[-1]) if drawdown[-1] < -1e-10 else 0
        
        total_return = float(equity_curve[-1] - 1.0)
        max_drawdown_abs = abs(np.min(drawdown))
        recovery_factor[strategy_idx] = abs(total_return) / max_drawdown_abs if max_drawdown_abs > 0 else 0.0
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Build structured output array
    # ─────────────────────────────────────────────────────────────────────────────
    
    output_dtype = [
        ('current_drawdown_pct', 'f8'),
        ('drawdown_volatility_pct', 'f8'),
        ('max_drawdown_pct', 'f8'),
        ('max_drawdown_duration_days', 'f8'),
        ('max_drawdown_days', 'i4'),
        ('avg_drawdown_pct', 'f8'),
        ('avg_drawdown_days', 'f8'),
        ('drawdown_periods_count', 'i4'),
        ('drawdown_duration_total', 'f8'),
        ('current_drawdown_days', 'i4'),
        ('recovery_factor', 'f8'),
    ]
    
    results = np.zeros(n_strategies, dtype=output_dtype)
    results['current_drawdown_pct'] = current_drawdown_pct
    results['drawdown_volatility_pct'] = drawdown_volatility_pct
    results['max_drawdown_pct'] = max_drawdown_pct
    results['max_drawdown_duration_days'] = max_drawdown_duration
    results['max_drawdown_days'] = max_drawdown_duration.astype(np.int32)
    results['avg_drawdown_pct'] = avg_drawdown_pct
    results['avg_drawdown_days'] = avg_drawdown_days
    results['drawdown_periods_count'] = drawdown_periods_count
    results['drawdown_duration_total'] = drawdown_total_duration
    results['current_drawdown_days'] = current_drawdown_days
    results['recovery_factor'] = recovery_factor
    
    return results