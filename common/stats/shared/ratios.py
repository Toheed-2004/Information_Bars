"""
Risk-adjusted ratio calculations — single source of truth used by both
custom/ratios.py and vectorbt_pro/ratios.py.

All functions expect daily returns (decimal form, not percentage).
ann_factor should always be 365.25 (crypto convention).
"""
import numpy as np
from typing import Dict, Optional


def _get_empty_risk_adjusted() -> Dict[str, float]:
    return {
        'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0,
        'omega_ratio': 0.0, 'information_ratio': 0.0, 'treynor_ratio': 0.0,
        'alpha': 0.0, 'beta': 0.0,
        'risk_return_ratio': 0.0, 'capture_ratio': 0.0,
        'up_capture_ratio': 0.0, 'down_capture_ratio': 0.0,
        'cagr': 0.0, 'expected_return_pct': 0.0, 'adjusted_sortino': 0.0,
        'serenity_index': 0.0, 'probabilistic_sharpe_ratio': 0.0,
        'kelly_criterion': 0.0,
    }


def _calculate_cagr(returns: np.ndarray, ann_factor: float) -> float:
    """CAGR matching QuantStats. Expects daily returns, ann_factor=365.25."""
    if len(returns) == 0:
        return 0.0
    ending_value = np.prod(1 + returns)
    if ending_value <= 0:
        return -1.0
    num_years = len(returns) / ann_factor
    if num_years <= 0:
        return 0.0
    try:
        cagr = ending_value ** (1 / num_years) - 1
        return 0.0 if (np.isnan(cagr) or np.isinf(cagr)) else float(cagr)
    except (ZeroDivisionError, ValueError, OverflowError):
        return 0.0


def _calculate_expected_return(returns: np.ndarray) -> float:
    """Geometric mean of returns (QuantStats expected_return)."""
    if len(returns) == 0:
        return 0.0
    valid = returns[~np.isnan(returns)]
    if len(valid) == 0:
        return 0.0
    gross = np.maximum(1 + valid, 1e-10)
    return float(np.prod(gross) ** (1 / len(gross)) - 1)


def _calculate_adjusted_sortino(returns: np.ndarray, risk_free_rate: float,
                                 ann_factor: float) -> float:
    """Adjusted Sortino Ratio matching QuantStats."""
    if len(returns) == 0:
        return 0.0
    valid = returns[~np.isnan(returns)]
    if len(valid) == 0:
        return 0.0
    excess = valid - risk_free_rate
    mean_excess = np.mean(excess)
    neg = excess[excess < 0]
    if len(neg) == 0:
        return 10.0 if mean_excess > 0 else 0.0
    downside_dev = np.sqrt(np.mean(neg ** 2))
    if downside_dev == 0:
        return 10.0 if mean_excess > 0 else 0.0
    sqrt_ann = np.sqrt(ann_factor)
    return float((mean_excess * ann_factor) / (downside_dev * sqrt_ann))


def _calculate_serenity_index(returns: np.ndarray, ann_factor: float) -> float:
    """Serenity Index = CAGR / Ulcer Index."""
    if len(returns) == 0:
        return 0.0
    valid = returns[~np.isnan(returns)]
    if len(valid) == 0:
        return 0.0
    cumulative = np.cumprod(1 + valid)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    cagr = _calculate_cagr(valid, ann_factor)
    ulcer_index = np.sqrt(np.mean(drawdowns ** 2)) * np.sqrt(ann_factor)
    if ulcer_index == 0:
        return 50.0 if cagr > 0 else 0.0
    return float(cagr / ulcer_index)


def _calculate_probabilistic_sharpe_ratio(returns: np.ndarray,
                                           ann_factor: float) -> float:
    """Probabilistic Sharpe Ratio matching QuantStats."""
    if len(returns) == 0:
        return 0.0
    valid = returns[~np.isnan(returns)]
    if len(valid) <= 1:
        return 0.0
    mean_ret = np.mean(valid)
    std_ret = np.std(valid, ddof=1)
    if std_ret == 0:
        return 1.0 if mean_ret > 0 else 0.0
    sr = mean_ret / std_ret * np.sqrt(ann_factor)
    n = len(valid)
    sr_std = np.sqrt((1 + sr ** 2 / 2) / (n - 1))
    try:
        from scipy.stats import t
        return float(t.cdf(sr / sr_std, df=n - 1))
    except ImportError:
        return float(0.5 + 0.5 * np.sign(sr) * min(abs(sr / sr_std) / 2, 0.5))


def _calculate_kelly_criterion(returns: np.ndarray) -> float:
    """Kelly Criterion = μ/σ² (clipped to [-1, 1])."""
    if len(returns) == 0:
        return 0.0
    valid = returns[~np.isnan(returns)]
    if len(valid) == 0:
        return 0.0
    variance = np.var(valid, ddof=1)
    if variance == 0:
        return 0.0
    return float(np.clip(np.mean(valid) / variance, -1.0, 1.0))


def _calculate_risk_adjusted_ratios(
    portfolio_returns: np.ndarray,
    timestamps: np.ndarray,
    benchmark_return_pct: float,
    risk_free_rate: float,
    ann_factor: float,
    benchmark_returns: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Single source of truth for all risk-adjusted ratio calculations.

    Args:
        portfolio_returns:  Daily returns array (decimal, not percentage).
        timestamps:         Timestamps (unused here, kept for API compatibility).
        benchmark_return_pct: Benchmark total return as percentage (unused, kept for API).
        risk_free_rate:     Risk-free rate per period (default 0.0).
        ann_factor:         Annualisation factor (365.25 for crypto daily data).
        benchmark_returns:  Optional daily benchmark returns array.

    Returns:
        Dict with all risk-adjusted metrics.
    """
    if len(portfolio_returns) == 0:
        return _get_empty_risk_adjusted()

    valid_mask = ~np.isnan(portfolio_returns)
    valid_returns = portfolio_returns[valid_mask]
    if len(valid_returns) == 0:
        return _get_empty_risk_adjusted()

    excess_returns = valid_returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    sqrt_ann = np.sqrt(ann_factor)

    # Sharpe
    sharpe_ratio = float(mean_excess / std_excess * sqrt_ann) if std_excess > 0 else 0.0

    # Sortino (QuantStats: downside calc over ALL periods)
    avg_ann_return = mean_excess * ann_factor
    downside_sq = np.where(excess_returns <= 0, excess_returns ** 2, 0.0)
    downside_risk = np.sqrt(np.sum(downside_sq) / len(excess_returns)) * sqrt_ann
    sortino_ratio = float(avg_ann_return / downside_risk) if downside_risk > 0 else 0.0

    # CAGR and Calmar
    total_return = np.prod(1 + valid_returns) - 1
    periods_years = len(valid_returns) / ann_factor
    if periods_years <= 0:
        cagr_val = 0.0
    elif total_return <= -1:
        cagr_val = -1.0
    else:
        cagr_val = float((1 + total_return) ** (1 / periods_years) - 1)

    cumulative = np.cumprod(1 + valid_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    calmar_ratio = float(cagr_val / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # Omega
    pos_mask = valid_returns > 0
    gains = np.sum(valid_returns[pos_mask]) if np.any(pos_mask) else 0.0
    losses = abs(np.sum(valid_returns[~pos_mask])) if np.any(~pos_mask) else 0.0
    omega_ratio = float(gains / losses) if losses > 0 else float(1.0 + gains)

    alpha = 0.0
    beta = 0.0
    treynor_ratio = 0.0
    information_ratio = 0.0
    up_capture_ratio = 0.0
    down_capture_ratio = 0.0
    capture_ratio = 0.0

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bm_valid_mask = ~np.isnan(benchmark_returns)
        common_mask = valid_mask & bm_valid_mask
        if np.sum(common_mask) > 1:
            pf_aligned = portfolio_returns[common_mask]
            bm_aligned = benchmark_returns[common_mask]
            pf_excess = pf_aligned - risk_free_rate
            bm_excess = bm_aligned - risk_free_rate
            mean_pf_excess = np.mean(pf_excess)
            mean_bm_excess = np.mean(bm_excess)
            if len(pf_excess) > 1:
                cov_matrix = np.cov(pf_excess, bm_excess, ddof=1)
                covariance = cov_matrix[0, 1]
                bm_variance = np.var(bm_excess, ddof=1)
                beta = float(covariance / bm_variance) if bm_variance > 0 else 0.0
            # Jensen's alpha: CAGR_pf - (rf_ann + beta * (CAGR_bm - rf_ann))
            n_years = len(pf_aligned) / ann_factor
            cagr_pf = float((np.prod(1 + pf_aligned) ** (1.0 / n_years) - 1)) if n_years > 0 else 0.0
            cagr_bm = float((np.prod(1 + bm_aligned) ** (1.0 / n_years) - 1)) if n_years > 0 else 0.0
            rf_ann = risk_free_rate * ann_factor
            alpha = float(cagr_pf - (rf_ann + beta * (cagr_bm - rf_ann)))
            # Treynor = CAGR / beta (matching VBT)
            treynor_ratio = float(cagr_pf / beta) if beta != 0 else 0.0
            # Information ratio: mean(active) / std(active) per period, not annualized (matching VBT)
            active_returns_arr = pf_excess - bm_excess
            tracking_error = np.std(active_returns_arr, ddof=1)
            information_ratio = float(np.mean(active_returns_arr) / tracking_error) if tracking_error > 0 else 0.0
            up_bm = bm_aligned > 0
            down_bm = bm_aligned < 0
            if np.any(up_bm):
                up_capture_ratio = float(np.mean(pf_aligned[up_bm]) / np.mean(bm_aligned[up_bm])) if np.mean(bm_aligned[up_bm]) != 0 else 0.0
            if np.any(down_bm):
                down_capture_ratio = float(np.mean(pf_aligned[down_bm]) / np.mean(bm_aligned[down_bm])) if np.mean(bm_aligned[down_bm]) != 0 else 0.0
            capture_ratio = float(up_capture_ratio / abs(down_capture_ratio)) if down_capture_ratio != 0 else 0.0

    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'omega_ratio': omega_ratio,
        'information_ratio': information_ratio,
        'treynor_ratio': treynor_ratio,
        'alpha': alpha,
        'beta': beta,
        'risk_return_ratio': sharpe_ratio,
        'capture_ratio': capture_ratio,
        'up_capture_ratio': up_capture_ratio,
        'down_capture_ratio': down_capture_ratio,
        'cagr': cagr_val,
        'expected_return_pct': float(_calculate_expected_return(valid_returns) * 100),
        'adjusted_sortino': float(_calculate_adjusted_sortino(valid_returns, risk_free_rate, ann_factor)),
        'serenity_index': float(_calculate_serenity_index(valid_returns, ann_factor)),
        'probabilistic_sharpe_ratio': float(_calculate_probabilistic_sharpe_ratio(valid_returns, ann_factor)),
        'kelly_criterion': float(_calculate_kelly_criterion(valid_returns)),
    }
