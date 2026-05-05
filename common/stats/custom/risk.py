import numpy as np
from typing import Dict, Any
from .drawdown import _calculate_ulcer_index
from ..shared.utils import _calculate_avg_return, _calculate_geometric_mean, _rolling_windows, ANN_FACTOR

def _get_empty_risk_metrics() -> Dict[str, float]:
    return {
        'tail_ratio': 0.0, 'common_sense_ratio': 0.0, 'skewness': 0.0,
        'kurtosis': 0.0, 'period_return_skewness': 0.0,
        'period_return_kurtosis': 0.0, 'sharpe_ratio_std': 0.0,
        'volatility_annualized': 0.0, 'var_95_pct': 0.0, 'cvar_95_pct': 0.0,
        'profit_factor': 0.0,
        'ulcer_index_pct': 0.0, 'avg_return_pct': 0.0, 'geometric_mean_pct': 0.0,
        'risk_of_ruin': 0.0, 'rolling_volatility_mean_pct': 0.0, 'rolling_volatility_std_pct': 0.0
    }

def _calculate_risk_of_ruin(returns: np.ndarray) -> float:
    """Calculate Risk of Ruin matching QuantStats"""
    if len(returns) == 0:
        return 0.0
    
    # Simplified Risk of Ruin calculation
    # Based on the probability of losing all capital
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0:
        return 0.0 if mean_return >= 0 else 1.0
    
    # Using normal distribution approximation for RoR
    # Probability of returns falling below -100% (total ruin)
    try:
        from scipy.stats import norm
        risk_of_ruin = norm.cdf(-1.0, mean_return, std_return)
    except ImportError:
        # Fallback calculation without scipy
        if mean_return >= 0:
            risk_of_ruin = 0.0
        else:
            # Simple approximation
            risk_of_ruin = min(1.0, abs(mean_return) / (2 * std_return))
    
    return float(np.clip(risk_of_ruin, 0.0, 1.0))

def _calculate_rolling_volatility_stats(returns: np.ndarray, window: int = 30) -> tuple:
    """Calculate rolling volatility statistics — vectorized with sliding_window_view."""
    if len(returns) < window:
        return 0.0, 0.0
    windows = _rolling_windows(returns, window)  # shape (n-window+1, window)
    # ddof=1 std across each window row
    vols = np.std(windows, axis=1, ddof=1) * np.sqrt(ANN_FACTOR)
    if len(vols) == 0:
        return 0.0, 0.0
    return float(np.mean(vols)), float(np.std(vols, ddof=1)) if len(vols) > 1 else 0.0

def _calculate_risk_metrics(
    portfolio_returns: np.ndarray
) -> Dict[str, float]:
    """Ultra-fast risk metrics using pure NumPy vectorized operations"""
    
    returns = portfolio_returns
    
    if len(returns) == 0:
        return _get_empty_risk_metrics()
    
    # Vectorized basic calculations
    n_returns = len(returns)
    valid_mask = ~np.isnan(returns)
    valid_returns = returns[valid_mask]
    
    if len(valid_returns) == 0:
        return _get_empty_risk_metrics()
    
    # Volatility (annualized for daily data)
    volatility_daily = np.std(valid_returns, ddof=1) if len(valid_returns) > 1 else 0.0
    volatility_annualized = volatility_daily * np.sqrt(ANN_FACTOR)
    
    # VaR and CVaR (95% confidence level) - QuantStats-aligned parametric approach
    if len(valid_returns) >= 20:
        # Parametric VaR: mean + z_score * volatility (matches QuantStats exactly)
        from scipy import stats as scipy_stats
        z_score_95 = scipy_stats.norm.ppf(0.05)  # -1.645 for 95% confidence
        mean_return = np.mean(valid_returns)
        var_95 = mean_return + z_score_95 * volatility_daily  # Parametric VaR
        
        # CVaR: Expected value of returns below VaR threshold
        cvar_mask = valid_returns <= var_95
        cvar_95 = np.mean(valid_returns[cvar_mask]) if np.any(cvar_mask) else var_95
    else:
        var_95 = cvar_95 = 0.0
    
    # Tail Ratio - vectorized percentile calculation
    if len(valid_returns) >= 20:
        perc_95 = np.abs(np.percentile(valid_returns, 95))
        perc_5 = np.abs(np.percentile(valid_returns, 5))
        tail_ratio = perc_95 / perc_5 if perc_5 > 0 else (10.0 if perc_95 > 0 else 1.0)  # Good tail behavior when 5th percentile is zero
    else:
        tail_ratio = 0.0
    
    # Profit Factor - vectorized calculation
    positive_mask = valid_returns > 0
    negative_mask = valid_returns < 0
    
    gains = np.sum(valid_returns[positive_mask]) if np.any(positive_mask) else 0.0
    losses = np.abs(np.sum(valid_returns[negative_mask])) if np.any(negative_mask) else 0.0
    
    profit_factor = gains / losses if losses > 0 else (100.0 if gains > 0 else 1.0)  # Excellent profit factor when no losses
    
    # Common Sense Ratio - vectorized
    common_sense_ratio = tail_ratio * profit_factor if len(valid_returns) >= 20 else 0.0
    
    # Skewness and Kurtosis - using scipy for accuracy
    from scipy import stats as scipy_stats
    if len(valid_returns) > 3:
        skewness = scipy_stats.skew(valid_returns, bias=False)  # Unbiased estimator
        kurtosis = scipy_stats.kurtosis(valid_returns, bias=False)  # Excess kurtosis
        period_return_skewness = skewness
        period_return_kurtosis = kurtosis
    else:
        skewness = kurtosis = 0.0
        period_return_skewness = period_return_kurtosis = 0.0
    
    # Sharpe Ratio Standard Deviation - vectorized
    if len(valid_returns) > 1:
        n = len(valid_returns)
        sharpe_raw = np.mean(valid_returns) / volatility_daily if volatility_daily > 0 else 0.0
        
        # Simplified but accurate formula for Sharpe ratio standard error
        sharpe_ratio_std = np.sqrt((1 + 0.5 * sharpe_raw**2) / n) if n > 0 else 0.0
    else:
        sharpe_ratio_std = 0.0
    
    mean_vol, std_vol = _calculate_rolling_volatility_stats(valid_returns)
    
    return {
        'tail_ratio': float(tail_ratio),
        'common_sense_ratio': float(common_sense_ratio),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'period_return_skewness': float(period_return_skewness),
        'period_return_kurtosis': float(period_return_kurtosis),
        'sharpe_ratio_std': float(sharpe_ratio_std),
        'volatility_annualized': float(volatility_annualized),
        'var_95_pct': float(var_95 * 100),
        'cvar_95_pct': float(cvar_95 * 100),
        'profit_factor': float(profit_factor),
        'ulcer_index_pct': float(_calculate_ulcer_index(valid_returns) * 100),
        'avg_return_pct': float(_calculate_avg_return(valid_returns) * 100),
        'geometric_mean_pct': float(_calculate_geometric_mean(valid_returns) * 100),
        'risk_of_ruin': float(_calculate_risk_of_ruin(valid_returns)),
        'rolling_volatility_mean_pct': float(mean_vol * 100),
        'rolling_volatility_std_pct': float(std_vol * 100)
    }
