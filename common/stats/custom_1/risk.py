import numpy as np
from scipy.stats import norm as scipy_norm
from bitpredict.common.stats.custom_1.config import ANN_FACTOR, SQRT_ANN

# 95% confidence level Z-score (negative for left tail)
_Z_95 = scipy_norm.ppf(0.05)


def risk_metrics(returns_2d: np.ndarray, valid_mask_2d: np.ndarray) -> np.ndarray:
    """
    Calculate risk/performance metrics for multiple trading strategies.
    
    Parameters
    ----------
    returns_2d : np.ndarray
        Shape (max_days, n_strategies). Daily returns in decimal (e.g., 0.01 for 1%).
    valid_mask_2d : np.ndarray
        Boolean array same shape as returns_2d. True = valid observation.
    
    Returns
    -------
    np.ndarray
        Structured array with fields: tail_ratio, common_sense_ratio, skewness,
        kurtosis, sharpe_ratio_std, volatility_annualized_pct, var_95_pct,
        cvar_95_pct, ulcer_index_pct, avg_return_pct, geometric_mean_pct,
        risk_of_ruin, rolling_volatility_mean_pct, rolling_volatility_std_pct.
    
    Notes
    -----
    - Requires >=20 obs for VaR/CVaR/tail ratio
    - Requires >=30 obs for rolling volatility
    - Outputs are percentages (multiplied by 100) where applicable
    """
    
    max_days, n_strategies = returns_2d.shape
    
    # Mask invalid returns with NaN for safe statistical operations
    returns_masked = np.where(valid_mask_2d, returns_2d, np.nan)  # (max_days, n_strategies)
    
    # Count valid observations per strategy
    n_observations = valid_mask_2d.sum(axis=0).astype(np.float64)  # (n_strategies,)
    n_safe = np.maximum(n_observations, 1.0)  # Avoid division by zero
    n1_safe = np.maximum(n_observations - 1.0, 1.0)  # For degrees of freedom

    # ─────────────────────────────────────────────────────────────────────────────
    # Basic descriptive statistics
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Mean return (daily)
    mean_return = np.nansum(returns_masked, axis=0) / n_safe
    
    # Center the returns (subtract mean) for moment calculations
    returns_centered = np.where(valid_mask_2d, returns_masked - mean_return, 0.0)
    
    # Central moments
    second_moment = (returns_centered ** 2).sum(axis=0) / n1_safe  # Variance
    third_moment  = (returns_centered ** 3).sum(axis=0) / n_safe   # Skewness numerator
    fourth_moment = (returns_centered ** 4).sum(axis=0) / n_safe   # Kurtosis numerator
    
    # Volatility measures
    volatility_daily = np.sqrt(second_moment)
    volatility_annualized_pct = volatility_daily * SQRT_ANN * 100

    # ─────────────────────────────────────────────────────────────────────────────
    # Skewness and Kurtosis (bias-corrected using scipy's method)
    # Correction factors:
    # - Skewness: n * (n-1)^0.5 / (n-2)
    # - Kurtosis: (n-1) / ((n-2)(n-3)) * ((n+1) * kurtosis_raw + 6)
    # ─────────────────────────────────────────────────────────────────────────────
    
    skewness_raw = np.where(second_moment > 0, 
                           third_moment / (second_moment ** 1.5), 
                           0.0)
    kurtosis_raw = np.where(second_moment > 0, 
                           fourth_moment / (second_moment ** 2) - 3.0, 
                           0.0)
    
    skewness_corrected = np.where(
        n_observations > 2,
        skewness_raw * np.sqrt(n_observations * (n_observations - 1)) / np.maximum(n_observations - 2, 1.0),
        0.0
    )
    
    kurtosis_corrected = np.where(
        n_observations > 3,
        (n_observations - 1) / (np.maximum(n_observations - 2, 1.0) * np.maximum(n_observations - 3, 1.0)) * 
        ((n_observations + 1) * kurtosis_raw + 6),
        0.0
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Value at Risk (VaR) - 95% confidence level
    # Parametric VaR assuming normal distribution
    # ─────────────────────────────────────────────────────────────────────────────
    
    var_95 = mean_return + _Z_95 * volatility_daily

    # ─────────────────────────────────────────────────────────────────────────────
    # Conditional Value at Risk (CVaR / Expected Shortfall)
    # Fully vectorized implementation using sorted returns
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Sort returns for each strategy (NaNs naturally go to the end)
    returns_sorted = np.sort(returns_masked, axis=0)
    
    # Identify tail observations (returns <= VaR threshold)
    tail_mask = returns_sorted <= var_95[np.newaxis, :]
    tail_mask &= ~np.isnan(returns_sorted)  # Exclude NaN values
    tail_count = tail_mask.sum(axis=0).astype(np.float64)
    
    # Calculate CVaR as the mean of tail returns
    cvar_95 = np.where(
        tail_count > 0,
        np.where(tail_mask, returns_sorted, 0.0).sum(axis=0) / np.maximum(tail_count, 1.0),
        var_95  # Fallback to VaR if no tail observations
    )
    
    # Only calculate VaR/CVaR for strategies with sufficient data (>= 20 observations)
    sufficient_data_mask = n_observations >= 20
    cvar_95 = np.where(sufficient_data_mask, cvar_95, 0.0)
    var_95  = np.where(sufficient_data_mask, var_95, 0.0)

    # ─────────────────────────────────────────────────────────────────────────────
    # Tail Ratio: Ratio of upper tail (95th percentile) to lower tail (5th percentile)
    # Measures symmetry/extremity of return distribution
    # ─────────────────────────────────────────────────────────────────────────────
    
    percentile_95 = np.abs(np.nanpercentile(returns_masked, 95, axis=0))
    percentile_5  = np.abs(np.nanpercentile(returns_masked, 5,  axis=0))
    
    tail_ratio = np.where(
        percentile_5 > 0, 
        percentile_95 / percentile_5,
        np.where(percentile_95 > 0, 10.0, 1.0)  # Default values for edge cases
    )
    tail_ratio = np.where(sufficient_data_mask, tail_ratio, 0.0)

    # ─────────────────────────────────────────────────────────────────────────────
    # Profit Factor and Common Sense Ratio
    # Profit Factor: Gross gains / Gross losses
    # Common Sense Ratio: Tail Ratio * Profit Factor
    # ─────────────────────────────────────────────────────────────────────────────
    
    returns_valid = np.where(valid_mask_2d, returns_masked, 0.0)
    total_gains  = np.where(returns_valid > 0, returns_valid, 0.0).sum(axis=0)
    total_losses = np.abs(np.where(returns_valid < 0, returns_valid, 0.0).sum(axis=0))
    
    profit_factor = np.where(
        total_losses > 0, 
        total_gains / total_losses,
        np.where(total_gains > 0, 100.0, 1.0)  # High value if no losses
    )
    
    common_sense_ratio = np.where(sufficient_data_mask, tail_ratio * profit_factor, 0.0)

    # ─────────────────────────────────────────────────────────────────────────────
    # Sharpe Ratio Standard Error
    # Standard deviation of the Sharpe ratio estimate
    # ─────────────────────────────────────────────────────────────────────────────
    
    sharpe_raw = np.where(volatility_daily > 0, mean_return / volatility_daily, 0.0)
    sharpe_ratio_std = np.where(
        n_observations > 1,
        np.sqrt((1 + 0.5 * sharpe_raw ** 2) / n_safe),
        0.0
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Ulcer Index (downside risk measure based on drawdowns)
    # Measures depth and duration of price declines
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Calculate cumulative wealth and running maximum
    wealth_curve = np.cumprod(1.0 + returns_valid, axis=0)
    running_maximum = np.maximum.accumulate(wealth_curve, axis=0)
    
    # Drawdown calculation
    drawdown = np.where(valid_mask_2d, (wealth_curve - running_maximum) / running_maximum, 0.0)
    
    # Ulcer index = RMS of drawdowns (annualized)
    ulcer_index = np.sqrt((drawdown ** 2).sum(axis=0) / n_safe) * SQRT_ANN

    # ─────────────────────────────────────────────────────────────────────────────
    # Average Return (excluding zero-return days)
    # ─────────────────────────────────────────────────────────────────────────────
    
    non_zero_mask = valid_mask_2d & (returns_valid != 0)
    non_zero_count = non_zero_mask.sum(axis=0).astype(np.float64)
    
    average_return_pct = np.where(
        non_zero_count > 0,
        np.where(non_zero_mask, returns_valid, 0.0).sum(axis=0) / np.maximum(non_zero_count, 1.0) * 100,
        0.0
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Geometric Mean Return (compounded average return)
    # ─────────────────────────────────────────────────────────────────────────────
    
    log_returns = np.where(valid_mask_2d, np.log(np.maximum(1.0 + returns_masked, 1e-10)), 0.0)
    geometric_mean_pct = np.where(
        n_observations > 0,
        (np.exp(log_returns.sum(axis=0) / n_safe) - 1.0) * 100,
        0.0
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Risk of Ruin: Probability of a 100% loss at some point
    # Approximated using normal distribution with mean_return and volatility_daily
    # ─────────────────────────────────────────────────────────────────────────────
    
    risk_of_ruin = np.where(
        volatility_daily > 0,
        np.clip(scipy_norm.cdf(-1.0, mean_return, volatility_daily), 0.0, 1.0),
        np.where(mean_return >= 0, 0.0, 1.0)
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Rolling Volatility Statistics (30-day windows)
    # Uses stride tricks for efficient sliding window without loops
    # ─────────────────────────────────────────────────────────────────────────────
    
    window_size = 30
    rolling_vol_mean = np.zeros(n_strategies)
    rolling_vol_std  = np.zeros(n_strategies)
    
    if max_days >= window_size:
        from numpy.lib.stride_tricks import sliding_window_view
        
        # Create sliding windows for returns and validity masks
        # Shape: (max_days - window_size + 1, n_strategies, window_size)
        returns_windows = sliding_window_view(returns_valid, window_size, axis=0)
        valid_windows = sliding_window_view(valid_mask_2d.astype(np.float32), window_size, axis=0)
        
        # Count valid observations in each window
        valid_counts_per_window = valid_windows.sum(axis=2)  # (steps, n_strategies)
        
        # Calculate rolling volatility for each window
        rolling_volatility = np.std(returns_windows, axis=2, ddof=1) * SQRT_ANN * 100  # (steps, n_strategies)
        
        # Only consider windows with complete data (all days valid)
        complete_windows = (valid_counts_per_window >= window_size)  # (steps, n_strategies)
        rolling_volatility_masked = np.where(complete_windows, rolling_volatility, np.nan)
        
        # Calculate cross-sectional statistics across windows
        has_any_valid_windows = (~np.isnan(rolling_volatility_masked)).any(axis=0)  # (n_strategies,)
        rolling_vol_mean = np.where(
            has_any_valid_windows,
            np.nanmean(rolling_volatility_masked, axis=0),
            0.0
        )
        
        # Standard deviation of rolling volatility (measure of volatility stability)
        window_count = (~np.isnan(rolling_volatility_masked)).sum(axis=0).astype(np.float64)
        rolling_vol_std = np.where(
            window_count > 1,
            np.nanstd(rolling_volatility_masked, axis=0, ddof=1),
            0.0
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Build structured output array
    # ─────────────────────────────────────────────────────────────────────────────
    
    output_dtype = [
        ('tail_ratio', 'f8'),
        ('common_sense_ratio', 'f8'),
        ('skewness', 'f8'),
        ('kurtosis', 'f8'),
        ('sharpe_ratio_std', 'f8'),
        ('volatility_annualized_pct', 'f8'),
        ('var_95_pct', 'f8'),
        ('cvar_95_pct', 'f8'),
        ('ulcer_index_pct', 'f8'),
        ('avg_return_pct', 'f8'),
        ('geometric_mean_pct', 'f8'),
        ('risk_of_ruin', 'f8'),
        ('rolling_volatility_mean_pct', 'f8'),
        ('rolling_volatility_std_pct', 'f8'),
    ]
    
    results = np.zeros(n_strategies, dtype=output_dtype)
    results['tail_ratio']                  = tail_ratio
    results['common_sense_ratio']          = common_sense_ratio
    results['skewness']                    = skewness_corrected
    results['kurtosis']                    = kurtosis_corrected
    results['sharpe_ratio_std']            = sharpe_ratio_std
    results['volatility_annualized_pct']   = volatility_annualized_pct
    results['var_95_pct']                  = var_95 * 100
    results['cvar_95_pct']                 = cvar_95 * 100
    results['ulcer_index_pct']             = ulcer_index * 100
    results['avg_return_pct']              = average_return_pct
    results['geometric_mean_pct']          = geometric_mean_pct
    results['risk_of_ruin']                = risk_of_ruin
    results['rolling_volatility_mean_pct'] = rolling_vol_mean
    results['rolling_volatility_std_pct']  = rolling_vol_std
    
    return results