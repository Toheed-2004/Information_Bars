"""
Fully vectorized risk-adjusted ratios for ALL strategies at once.
Optimized for thousands of strategies with transposed layout: (max_days, n_strats).
"""

import numpy as np
from scipy.stats import norm as scipy_norm
from bitpredict.common.stats.custom_1.config import ANN_FACTOR, SQRT_ANN


def calculate_risk_adjusted_ratios(
    returns_2d: np.ndarray,
    valid_mask_2d: np.ndarray,
    benchmark_returns_1d: np.ndarray,
    risk_free_rate: float = 0.0,
    ann_factor: float = ANN_FACTOR,
) -> np.ndarray:
    """
    Calculate risk-adjusted performance ratios for multiple strategies.
    
    Parameters
    ----------
    returns_2d : np.ndarray
        Shape (max_days, n_strategies). Daily returns in decimal form.
    valid_mask_2d : np.ndarray
        Boolean array same shape as returns_2d. True = valid observation.
    benchmark_returns_1d : np.ndarray
        Shape (max_days,). Daily benchmark returns for comparison.
    risk_free_rate : float, default=0.0
        Daily risk-free rate (e.g., 0.0001 for 0.01%).
    ann_factor : float, default=ANN_FACTOR
        Annualization factor (e.g., 252 for daily data).
    
    Returns
    -------
    np.ndarray
        Structured array with fields: sharpe_ratio, sortino_ratio, calmar_ratio,
        omega_ratio, alpha, beta, information_ratio, treynor_ratio, capture_ratio,
        up_capture_ratio, down_capture_ratio, kelly_criterion, cagr,
        probabilistic_sharpe_ratio, adjusted_sortino, serenity_index,
        expected_return_pct.
    """
    
    max_days, n_strategies = returns_2d.shape
    sqrt_ann = np.sqrt(ann_factor)

    # Prepare returns with zeros for invalid observations
    returns_valid = np.where(valid_mask_2d, returns_2d, 0.0)           # (max_days, n_strategies)
    returns_masked = np.where(valid_mask_2d, returns_2d, np.nan)       # For percentile operations
    n_obs = valid_mask_2d.sum(axis=0).astype(np.float64)               # (n_strategies,)
    n_safe = np.maximum(n_obs, 1.0)                                    # Avoid division by zero
    n1_safe = np.maximum(n_obs - 1.0, 1.0)                             # For degrees of freedom

    # ─────────────────────────────────────────────────────────────────────────────
    # Basic statistics (excess returns over risk-free rate)
    # ─────────────────────────────────────────────────────────────────────────────
    
    mean_return = returns_valid.sum(axis=0) / n_safe                    # (n_strategies,)
    excess_returns = np.where(valid_mask_2d, returns_2d - risk_free_rate, 0.0)
    mean_excess = excess_returns.sum(axis=0) / n_safe
    
    # Center excess returns for variance calculation
    excess_centered = np.where(valid_mask_2d, excess_returns - mean_excess, 0.0)
    excess_variance = (excess_centered ** 2).sum(axis=0) / n1_safe
    excess_std = np.sqrt(excess_variance)

    # ─────────────────────────────────────────────────────────────────────────────
    # Sharpe Ratio (risk-adjusted return)
    # ─────────────────────────────────────────────────────────────────────────────
    
    sharpe_ratio = np.where(excess_std > 0, mean_excess / excess_std * sqrt_ann, 0.0)

    # ─────────────────────────────────────────────────────────────────────────────
    # Sortino Ratio (focuses on downside risk only)
    # ─────────────────────────────────────────────────────────────────────────────
    
    annualized_return = mean_excess * ann_factor
    downside_squared = np.where(valid_mask_2d & (excess_returns <= 0), excess_returns ** 2, 0.0)
    downside_risk = np.sqrt(downside_squared.sum(axis=0) / n_safe) * sqrt_ann
    sortino_ratio = np.where(downside_risk > 0, annualized_return / downside_risk, 0.0)

    # ─────────────────────────────────────────────────────────────────────────────
    # Omega Ratio (gain/loss ratio with threshold)
    # ─────────────────────────────────────────────────────────────────────────────
    
    total_gains = np.where(valid_mask_2d & (returns_2d > 0), returns_2d, 0.0).sum(axis=0)
    total_losses = np.abs(np.where(valid_mask_2d & (returns_2d <= 0), returns_2d, 0.0).sum(axis=0))
    omega_ratio = np.where(total_losses > 0, total_gains / total_losses, 1.0 + total_gains)

    # ─────────────────────────────────────────────────────────────────────────────
    # CAGR (Compound Annual Growth Rate) via log returns
    # ─────────────────────────────────────────────────────────────────────────────
    
    log_returns = np.where(valid_mask_2d, np.log(np.maximum(1.0 + returns_2d, 1e-10)), 0.0)
    cumulative_log = log_returns.sum(axis=0)                             # (n_strategies,)
    total_return = np.expm1(cumulative_log)                              # (n_strategies,)
    years = n_obs / ann_factor
    cagr = np.where(
        (years > 0) & (total_return > -1),
        np.exp(cumulative_log / np.maximum(years, 1e-10)) - 1.0,
        0.0
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Maximum Drawdown and Calmar Ratio
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Calculate cumulative equity curve
    cumulative_equity = np.cumprod(np.where(valid_mask_2d, 1.0 + returns_2d, 1.0), axis=0)  # (max_days, n_strategies)
    running_maximum = np.maximum.accumulate(cumulative_equity, axis=0)
    
    # Drawdown calculation
    drawdown = np.where(valid_mask_2d, (cumulative_equity - running_maximum) / running_maximum, 0.0)
    max_drawdown = drawdown.min(axis=0)                                  # (n_strategies,)
    calmar_ratio = np.where(max_drawdown != 0, cagr / np.abs(max_drawdown), 0.0)

    # ─────────────────────────────────────────────────────────────────────────────
    # Serenity Index (CAGR / Ulcer Index)
    # Ulcer Index = RMS of drawdowns
    # ─────────────────────────────────────────────────────────────────────────────
    
    ulcer_index = np.sqrt((drawdown ** 2).sum(axis=0) / n_safe)
    serenity_index = np.where((ulcer_index > 0) & (cagr != 0), cagr / ulcer_index, 0.0)

    # ─────────────────────────────────────────────────────────────────────────────
    # Benchmark-based metrics (fully vectorized)
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Broadcast benchmark to 2D
    benchmark_2d = benchmark_returns_1d[:, np.newaxis]                    # (max_days, 1)
    benchmark_valid = np.where(valid_mask_2d, benchmark_2d, 0.0)         # (max_days, n_strategies)
    
    # Excess returns relative to risk-free rate
    portfolio_excess = excess_returns                                     # Already masked
    benchmark_excess = np.where(valid_mask_2d, benchmark_2d - risk_free_rate, 0.0)
    
    # Means and centered values
    mean_benchmark = benchmark_excess.sum(axis=0) / n_safe
    benchmark_centered = np.where(valid_mask_2d, benchmark_excess - mean_benchmark, 0.0)
    benchmark_variance = (benchmark_centered ** 2).sum(axis=0) / n1_safe
    covariance = (excess_centered * benchmark_centered).sum(axis=0) / n1_safe
    
    # Beta (sensitivity to benchmark)
    beta = np.where(benchmark_variance > 0, covariance / benchmark_variance, 0.0)
    
    # CAGR for benchmark (reusing same pattern)
    log_benchmark = np.where(valid_mask_2d, np.log(np.maximum(1.0 + benchmark_2d, 1e-10)), 0.0)
    cagr_benchmark = np.where(years > 0, np.exp(log_benchmark.sum(axis=0) / np.maximum(years, 1e-10)) - 1.0, 0.0)
    
    # Alpha (Jensen's alpha)
    risk_free_annual = risk_free_rate * ann_factor
    alpha = cagr - (risk_free_annual + beta * (cagr_benchmark - risk_free_annual))
    
    # Treynor Ratio (excess return per unit of systematic risk)
    treynor_ratio = np.where(beta != 0, cagr / beta, 0.0)
    
    # Information Ratio (excess return over benchmark per unit of tracking error)
    active_returns = np.where(valid_mask_2d, portfolio_excess - benchmark_excess, 0.0)
    mean_active = active_returns.sum(axis=0) / n_safe
    active_centered = np.where(valid_mask_2d, active_returns - mean_active, 0.0)
    tracking_error = np.sqrt((active_centered ** 2).sum(axis=0) / n1_safe)
    information_ratio = np.where(tracking_error > 0, mean_active / tracking_error, 0.0)
    
    # Capture Ratios (up/down market performance)
    up_mask = valid_mask_2d & (benchmark_2d > 0)
    down_mask = valid_mask_2d & (benchmark_2d < 0)
    up_count = up_mask.sum(axis=0).astype(np.float64)
    down_count = down_mask.sum(axis=0).astype(np.float64)
    
    mean_return_up = np.where(up_mask, returns_2d, 0.0).sum(axis=0) / np.maximum(up_count, 1.0)
    mean_benchmark_up = np.where(up_mask, benchmark_2d, 0.0).sum(axis=0) / np.maximum(up_count, 1.0)
    mean_return_down = np.where(down_mask, returns_2d, 0.0).sum(axis=0) / np.maximum(down_count, 1.0)
    mean_benchmark_down = np.where(down_mask, benchmark_2d, 0.0).sum(axis=0) / np.maximum(down_count, 1.0)
    
    up_capture_ratio = np.where(mean_benchmark_up != 0, mean_return_up / mean_benchmark_up, 0.0)
    down_capture_ratio = np.where(mean_benchmark_down != 0, mean_return_down / mean_benchmark_down, 0.0)
    capture_ratio = np.where(down_capture_ratio != 0, up_capture_ratio / np.abs(down_capture_ratio), 0.0)

    # ─────────────────────────────────────────────────────────────────────────────
    # Skewness and Kurtosis (bias-corrected using scipy's method)
    # Reusing centered excess returns (rc)
    # ─────────────────────────────────────────────────────────────────────────────
    
    third_moment = np.where(valid_mask_2d, excess_centered ** 3, 0.0).sum(axis=0) / n_safe
    fourth_moment = np.where(valid_mask_2d, excess_centered ** 4, 0.0).sum(axis=0) / n_safe
    second_moment = excess_variance  # Already computed with ddof=1
    
    skewness_raw = np.where(excess_variance > 0, third_moment / np.maximum(excess_variance ** 1.5, 1e-30), 0.0)
    kurtosis_raw = np.where(excess_variance > 0, fourth_moment / np.maximum(excess_variance ** 2, 1e-30) - 3.0, 0.0)
    
    skewness = np.where(
        n_obs > 2,
        skewness_raw * np.sqrt(n_obs * (n_obs - 1)) / np.maximum(n_obs - 2, 1.0),
        0.0
    )
    kurtosis = np.where(
        n_obs > 3,
        (n_obs - 1) / (np.maximum(n_obs - 2, 1.0) * np.maximum(n_obs - 3, 1.0))
        * ((n_obs + 1) * kurtosis_raw + 6),
        0.0
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Kelly Criterion (optimal bet size)
    # f* = W - (1-W)/R where R = avg_win / avg_loss
    # ─────────────────────────────────────────────────────────────────────────────
    
    win_mask = valid_mask_2d & (returns_2d > 0)
    loss_mask = valid_mask_2d & (returns_2d < 0)
    win_count = win_mask.sum(axis=0).astype(np.float64)
    loss_count = loss_mask.sum(axis=0).astype(np.float64)
    
    win_probability = win_count / n_safe
    average_win = np.where(win_mask, returns_2d, 0.0).sum(axis=0) / np.maximum(win_count, 1.0)
    average_loss = np.abs(np.where(loss_mask, returns_2d, 0.0).sum(axis=0) / np.maximum(loss_count, 1.0))
    win_loss_ratio = np.where(average_loss > 0, average_win / average_loss, 1.0)
    
    kelly_criterion = np.where(
        (win_count > 0) & (loss_count > 0) & (win_loss_ratio > 0),
        win_probability - (1 - win_probability) / win_loss_ratio,
        0.0
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Probabilistic Sharpe Ratio (PSR)
    # Accounts for skewness and kurtosis in Sharpe ratio distribution
    # ─────────────────────────────────────────────────────────────────────────────
    
    psr_denominator = np.sqrt(np.maximum(
        1.0 - skewness * sharpe_ratio + (kurtosis - 1.0) / 4.0 * sharpe_ratio ** 2,
        1e-30
    ))
    psr_z_score = np.where(
        (excess_std > 0) & (psr_denominator > 0) & (n_obs > 1),
        sharpe_ratio * np.sqrt(np.maximum(n_obs - 1, 0.0)) / psr_denominator,
        0.0
    )
    probabilistic_sharpe_ratio = scipy_norm.cdf(psr_z_score)

    # ─────────────────────────────────────────────────────────────────────────────
    # Adjusted Sortino Ratio (accounts for skewness of negative returns)
    # ─────────────────────────────────────────────────────────────────────────────
    
    negative_returns = np.where(valid_mask_2d & (returns_2d < 0), returns_2d, 0.0)
    negative_count = (valid_mask_2d & (returns_2d < 0)).sum(axis=0).astype(np.float64)
    negative_count_safe = np.maximum(negative_count, 1.0)
    
    mean_negative = negative_returns.sum(axis=0) / negative_count_safe
    negative_centered = np.where(valid_mask_2d & (returns_2d < 0), negative_returns - mean_negative, 0.0)
    
    negative_variance = (negative_centered ** 2).sum(axis=0) / np.maximum(negative_count - 1, 1.0)
    negative_third_moment = (negative_centered ** 3).sum(axis=0) / negative_count_safe
    
    negative_skewness_raw = np.where(
        negative_variance > 0,
        negative_third_moment / np.maximum(negative_variance ** 1.5, 1e-30),
        0.0
    )
    
    negative_skewness = np.where(
        negative_count > 2,
        negative_skewness_raw * np.sqrt(negative_count * (negative_count - 1)) / np.maximum(negative_count - 2, 1.0),
        0.0
    )
    
    adjusted_sortino_ratio = np.where(downside_risk > 0, sortino_ratio * (1.0 + negative_skewness / 6.0), 0.0)

    # ─────────────────────────────────────────────────────────────────────────────
    # Expected Return (as percentage)
    # ─────────────────────────────────────────────────────────────────────────────
    
    expected_return_pct = mean_return * 100.0

    # ─────────────────────────────────────────────────────────────────────────────
    # Build structured output array
    # ─────────────────────────────────────────────────────────────────────────────
    
    output_dtype = [
        ('sharpe_ratio', 'f8'), 
        ('sortino_ratio', 'f8'), 
        ('calmar_ratio', 'f8'),
        ('omega_ratio', 'f8'), 
        ('alpha', 'f8'), 
        ('beta', 'f8'),
        ('information_ratio', 'f8'), 
        ('treynor_ratio', 'f8'),
        ('capture_ratio', 'f8'), 
        ('up_capture_ratio', 'f8'), 
        ('down_capture_ratio', 'f8'),
        ('kelly_criterion', 'f8'), 
        ('cagr', 'f8'),
        ('probabilistic_sharpe_ratio', 'f8'), 
        ('adjusted_sortino', 'f8'),
        ('serenity_index', 'f8'), 
        ('expected_return_pct', 'f8'),
    ]

    results = np.zeros(n_strategies, dtype=output_dtype)
    results['sharpe_ratio']              = sharpe_ratio
    results['sortino_ratio']             = sortino_ratio
    results['calmar_ratio']              = calmar_ratio
    results['omega_ratio']               = omega_ratio
    results['alpha']                     = alpha
    results['beta']                      = beta
    results['information_ratio']         = information_ratio
    results['treynor_ratio']             = treynor_ratio
    results['capture_ratio']             = capture_ratio
    results['up_capture_ratio']          = up_capture_ratio
    results['down_capture_ratio']        = down_capture_ratio
    results['kelly_criterion']           = kelly_criterion
    results['cagr']                      = cagr
    results['probabilistic_sharpe_ratio']= probabilistic_sharpe_ratio
    results['adjusted_sortino']          = adjusted_sortino_ratio
    results['serenity_index']            = serenity_index
    results['expected_return_pct']       = expected_return_pct

    return results