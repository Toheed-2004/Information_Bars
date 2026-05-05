import numpy as np
from typing import Dict, Any

def _get_empty_distribution_analysis() -> Dict[str, Any]:
    return {
        # Basic distribution moments
        'returns_mean_pct': 0.0, 'returns_std': 0.0, 'returns_variance': 0.0,
        'skewness': 0.0, 'kurtosis': 0.0,
        
        # Distribution percentiles
        'percentile_1': 0.0, 'percentile_5': 0.0, 'percentile_10': 0.0,
        'percentile_25': 0.0, 'percentile_50': 0.0, 'percentile_75': 0.0,
        'percentile_90': 0.0, 'percentile_95': 0.0, 'percentile_99': 0.0,
        
        # Risk metrics
        'var_95': 0.0, 'var_99': 0.0, 'cvar_95': 0.0, 'cvar_99': 0.0,
        
        # Tail and deviation analysis
        'tail_ratio': 0.0, 'downside_deviation': 0.0, 'upside_deviation': 0.0,
        
        # Return frequency analysis
        'positive_returns_count': 0, 'negative_returns_count': 0,
        'zero_returns_count': 0, 'positive_returns_pct': 0.0,
        'negative_returns_pct': 0.0, 'zero_returns_pct': 0.0,
        
        # Gain/Loss characteristics
        'positive_mean_pct': 0.0, 'negative_mean_pct': 0.0, 'gain_loss_ratio': 0.0,
        
        # Shape and normality
        'coefficient_of_variation': 0.0, 'jarque_bera_statistic': 0.0,
        
        # Extreme values
        'max_return_pct': 0.0, 'min_return_pct': 0.0, 'return_range_pct': 0.0,
        
        # Outlier analysis
        'outliers_upper': 0, 'outliers_lower': 0, 'total_outliers': 0,
        'outlier_pct': 0.0,
        
        # Analysis metadata
        'total_observations': 0, 'distribution_analysis_complete': 0.0
    }

def _calculate_distribution_analysis(
    portfolio_returns: np.ndarray,
    balance_array: np.ndarray,
    ann_factor: float
) -> Dict[str, Any]:
    """Calculate comprehensive return distribution characteristics"""
    
    if len(portfolio_returns) == 0:
        return _get_empty_distribution_analysis()
    
    # Basic distribution characteristics
    returns_mean = float(np.mean(portfolio_returns)) * 100
    returns_std = float(np.std(portfolio_returns, ddof=1)) if len(portfolio_returns) > 1 else 0.0
    returns_var = float(np.var(portfolio_returns, ddof=1)) if len(portfolio_returns) > 1 else 0.0
    
    # Skewness and kurtosis — use scipy for consistency with VBT/QuantStats
    from scipy import stats as scipy_stats
    if len(portfolio_returns) > 2 and returns_std > 0:
        skewness = float(scipy_stats.skew(portfolio_returns, bias=False))
        kurtosis = float(scipy_stats.kurtosis(portfolio_returns, bias=False))
    else:
        skewness = kurtosis = 0.0
    
    # Distribution percentiles for comprehensive characterization
    if len(portfolio_returns) > 0:
        percentiles = np.percentile(portfolio_returns, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        p1, p5, p10, p25, p50, p75, p90, p95, p99 = percentiles
    else:
        p1 = p5 = p10 = p25 = p50 = p75 = p90 = p95 = p99 = 0.0
    
    # Tail analysis
    tail_ratio = (p95 / abs(p5)) if p5 != 0 else 0.0
    downside_deviation = float(np.std(portfolio_returns[portfolio_returns <= 0], ddof=1)) if np.sum(portfolio_returns <= 0) > 1 else 0.0
    upside_deviation = float(np.std(portfolio_returns[portfolio_returns >= 0], ddof=1)) if np.sum(portfolio_returns >= 0) > 1 else 0.0
    
    # VaR calculations (historical method)
    var_95 = float(np.percentile(portfolio_returns, 5)) * 100  # 95% VaR (5% worst returns)
    var_99 = float(np.percentile(portfolio_returns, 1)) * 100  # 99% VaR (1% worst returns)
    
    # CVaR (Conditional VaR / Expected Shortfall)
    var_5pct_threshold = np.percentile(portfolio_returns, 5)
    var_1pct_threshold = np.percentile(portfolio_returns, 1)
    
    cvar_95 = float(np.mean(portfolio_returns[portfolio_returns <= var_5pct_threshold])) * 100 if np.any(portfolio_returns <= var_5pct_threshold) else 0.0
    cvar_99 = float(np.mean(portfolio_returns[portfolio_returns <= var_1pct_threshold])) * 100 if np.any(portfolio_returns <= var_1pct_threshold) else 0.0
    
    # Gain/Loss ratios
    positive_returns = portfolio_returns[portfolio_returns > 0]
    negative_returns = portfolio_returns[portfolio_returns < 0]
    zero_returns = portfolio_returns[portfolio_returns == 0]
    
    positive_count = len(positive_returns)
    negative_count = len(negative_returns)
    zero_count = len(zero_returns)
    
    positive_mean = float(np.mean(positive_returns)) * 100 if positive_count > 0 else 0.0
    negative_mean = float(np.mean(negative_returns)) * 100 if negative_count > 0 else 0.0
    
    gain_loss_ratio = (positive_mean / abs(negative_mean)) if negative_mean != 0 else 0.0
    
    # Distribution shape metrics
    coefficient_of_variation = (returns_std / abs(returns_mean / 100)) if returns_mean != 0 else 0.0
    
    # Jarque-Bera test statistic for normality
    n = len(portfolio_returns)
    if n > 2 and returns_std > 0:
        jb_statistic = n * (skewness**2 / 6 + kurtosis**2 / 24)
    else:
        jb_statistic = 0.0
    
    # Return distribution frequencies
    positive_pct = (positive_count / n) * 100 if n > 0 else 0.0
    negative_pct = (negative_count / n) * 100 if n > 0 else 0.0
    zero_pct = (zero_count / n) * 100 if n > 0 else 0.0
    
    # Extreme return analysis
    max_return = float(np.max(portfolio_returns)) * 100 if len(portfolio_returns) > 0 else 0.0
    min_return = float(np.min(portfolio_returns)) * 100 if len(portfolio_returns) > 0 else 0.0
    return_range = max_return - min_return
    
    # Outlier detection (using IQR method)
    iqr = p75 - p25
    outlier_threshold_upper = p75 + 1.5 * iqr
    outlier_threshold_lower = p25 - 1.5 * iqr
    
    outliers_upper = np.sum(portfolio_returns > outlier_threshold_upper)
    outliers_lower = np.sum(portfolio_returns < outlier_threshold_lower)
    total_outliers = outliers_upper + outliers_lower
    outlier_pct = (total_outliers / n) * 100 if n > 0 else 0.0
    
    return {
        # Basic distribution moments
        'returns_mean_pct': returns_mean,  # Already converted to percentage
        'returns_std': returns_std,    # Standard deviation (not percentage)
        'returns_variance': returns_var * 10000,  # Convert to basis points
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        
        # Distribution percentiles (in percentage)
        'percentile_1': float(p1) * 100,
        'percentile_5': float(p5) * 100,
        'percentile_10': float(p10) * 100,
        'percentile_25': float(p25) * 100,
        'percentile_50': float(p50) * 100,
        'percentile_75': float(p75) * 100,
        'percentile_90': float(p90) * 100,
        'percentile_95': float(p95) * 100,
        'percentile_99': float(p99) * 100,
        
        # Risk metrics
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        
        # Tail and deviation analysis
        'tail_ratio': float(tail_ratio),
        'downside_deviation': downside_deviation * 100,
        'upside_deviation': upside_deviation * 100,
        
        # Return frequency analysis
        'positive_returns_count': int(positive_count),
        'negative_returns_count': int(negative_count),
        'zero_returns_count': int(zero_count),
        'positive_returns_pct': positive_pct,
        'negative_returns_pct': negative_pct,
        'zero_returns_pct': zero_pct,
        
        # Gain/Loss characteristics
        'positive_mean_pct': positive_mean,
        'negative_mean_pct': negative_mean,
        'gain_loss_ratio': float(gain_loss_ratio),
        
        # Shape and normality
        'coefficient_of_variation': float(coefficient_of_variation),
        'jarque_bera_statistic': float(jb_statistic),
        
        # Extreme values
        'max_return_pct': max_return,
        'min_return_pct': min_return,
        'return_range_pct': return_range,
        
        # Outlier analysis
        'outliers_upper': int(outliers_upper),
        'outliers_lower': int(outliers_lower),
        'total_outliers': int(total_outliers),
        'outlier_pct': float(outlier_pct),
        
        # Analysis metadata
        'total_observations': int(n),
        'distribution_analysis_complete': 1.0
    }
