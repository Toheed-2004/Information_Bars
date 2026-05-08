import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats as scipy_stats

from ..shared.utils import ANN_FACTOR


def _calculate_ulcer_index(returns: np.ndarray) -> float:
    """Ulcer Index = sqrt(mean(drawdown²)), annualised."""
    if len(returns) == 0:
        return 0.0
    cumulative  = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns   = (cumulative - running_max) / running_max
    return float(np.sqrt(np.mean(drawdowns ** 2)) * np.sqrt(ANN_FACTOR))


def _calculate_risk_of_ruin(returns: np.ndarray) -> float:
    """
    Risk of Ruin: probability that a single period return ≤ -100%.
    Uses normal approximation; clipped to [0, 1].
    """
    if len(returns) == 0:
        return 0.0
    mean_r = np.mean(returns)
    std_r  = np.std(returns, ddof=1)
    if std_r == 0:
        return 0.0 if mean_r >= 0 else 1.0
    try:
        ror = scipy_stats.norm.cdf(-1.0, mean_r, std_r)
    except Exception:
        ror = 0.0 if mean_r >= 0 else min(1.0, abs(mean_r) / (2 * std_r))
    return float(np.clip(ror, 0.0, 1.0))


def _calculate_geometric_mean(returns: np.ndarray) -> float:
    """Geometric mean of (1+r) series."""
    if len(returns) == 0:
        return 0.0
    gross = np.maximum(1 + returns, 1e-10)
    try:
        return float(np.prod(gross) ** (1.0 / len(gross)) - 1)
    except Exception:
        return 0.0


def _extract_risk_adjusted_stats_vectorized(cache: Dict) -> Dict[str, Any]:
    """
    Custom-only risk-adjusted metrics that daily_pf.stats() does NOT provide.

    Standard ratios (Sharpe, Sortino, Calmar, Omega, Alpha, Beta, Info, Treynor)
    come from cache['vbt_stats'] in vbt_stats.py and are NOT recomputed here (req 1+3).
    Only compute: CAGR, Expected Return, Adjusted Sortino, Serenity, PSR, Kelly,
    and capture ratios (not in VBT stats()).
    """
    from .ratios import (
        _get_empty_risk_adjusted,
        _calculate_cagr,
        _calculate_expected_return,
        _calculate_adjusted_sortino,
        _calculate_serenity_index,
        _calculate_probabilistic_sharpe_ratio,
        _calculate_kelly_criterion,
    )

    returns = cache.get('daily_returns', np.array([]))
    if len(returns) == 0:
        return _get_empty_risk_adjusted()

    valid = returns[~np.isnan(returns)]
    if len(valid) == 0:
        return _get_empty_risk_adjusted()

    # Custom-only metrics (not in daily_pf.stats())
    cagr                 = _calculate_cagr(valid, ANN_FACTOR)
    expected_return_pct  = float(_calculate_expected_return(valid) * 100)
    adjusted_sortino     = float(_calculate_adjusted_sortino(valid, 0.0, ANN_FACTOR))
    serenity_index       = float(_calculate_serenity_index(valid, ANN_FACTOR))
    psr                  = float(_calculate_probabilistic_sharpe_ratio(valid, ANN_FACTOR))
    kelly                = float(_calculate_kelly_criterion(valid))

    # capture_ratio/up_capture/down_capture come from daily_pf direct attrs in cache (V)
    return {
        'cagr':                       cagr,
        'expected_return_pct':        expected_return_pct,
        'adjusted_sortino':           adjusted_sortino,
        'serenity_index':             serenity_index,
        'probabilistic_sharpe_ratio': psr,
        'kelly_criterion':            kelly,
    }


def _extract_risk_metrics_vectorized(cache: Dict) -> Dict[str, Any]:
    """
    Comprehensive risk metrics from daily_returns (D source).
    profit_factor computed internally for common_sense_ratio only — not returned.
    VaR/CVaR: historical percentile. Kurtosis: excess (fisher=True, normal=0).
    All fraction-based return values expressed as % (_pct suffix, ×100).
    """
    returns = cache.get('daily_returns', np.array([]))

    _empty = {
        'tail_ratio': 0.0, 'common_sense_ratio': 0.0,
        'skewness': 0.0, 'kurtosis': 0.0,
        'sharpe_ratio_std': 0.0,
        'volatility_annualized_pct': 0.0,
        'var_95_pct': 0.0, 'cvar_95_pct': 0.0,
        'ulcer_index_pct': 0.0, 'avg_return_pct': 0.0,
        'geometric_mean_pct': 0.0, 'risk_of_ruin': 0.0,
        'rolling_volatility_mean_pct': 0.0, 'rolling_volatility_std_pct': 0.0,
    }
    if len(returns) == 0:
        return _empty

    n        = len(returns)
    vol      = np.std(returns, ddof=1) if n > 1 else 0.0
    mean_ret = np.mean(returns)

    skew = float(scipy_stats.skew(returns, bias=False))             if n > 3 else 0.0
    kurt = float(scipy_stats.kurtosis(returns, bias=False, fisher=True)) if n > 3 else 0.0

    # VaR / CVaR — historical percentile
    if n >= 20:
        var_95    = float(np.percentile(returns, 5))
        cvar_mask = returns <= var_95
        cvar_95   = float(np.mean(returns[cvar_mask])) if np.any(cvar_mask) else var_95
        perc_95   = abs(np.percentile(returns, 95))
        perc_5    = abs(np.percentile(returns, 5))
        tail_ratio = perc_95 / perc_5 if perc_5 > 0 else (10.0 if perc_95 > 0 else 1.0)
    else:
        var_95 = cvar_95 = tail_ratio = 0.0

    # profit_factor — internal only, for common_sense_ratio
    profit_factor = 0.0
    trades_df = cache.get('trades_df', pd.DataFrame())
    if not trades_df.empty and 'PnL' in trades_df.columns:
        pnl = trades_df['PnL'].values.astype(np.float64)
        pnl = pnl[~np.isnan(pnl)]
        win = float(np.sum(pnl[pnl > 0]))
        los = float(np.sum(pnl[pnl < 0]))
        profit_factor = win / abs(los) if los != 0 else 0.0

    # Sharpe standard error
    if n > 1 and vol > 0:
        sharpe_ratio_std = float(np.sqrt((1 + 0.5 * (mean_ret / vol) ** 2) / n))
    else:
        sharpe_ratio_std = 0.0

    # Rolling volatility (30-day window)
    window = 30
    if n >= window:
        from numpy.lib.stride_tricks import sliding_window_view
        vols     = np.std(sliding_window_view(returns, window), axis=1, ddof=1) * np.sqrt(ANN_FACTOR)
        mean_vol = float(np.mean(vols))
        std_vol  = float(np.std(vols, ddof=1))
    else:
        mean_vol = std_vol = 0.0

    nonzero        = returns[returns != 0]
    avg_return_pct = float(np.mean(nonzero) * 100) if len(nonzero) > 0 else 0.0

    return {
        'tail_ratio':                  float(tail_ratio),
        'common_sense_ratio':          float(tail_ratio * profit_factor),
        'skewness':                    skew,
        'kurtosis':                    kurt,
        'sharpe_ratio_std':            sharpe_ratio_std,
        'volatility_annualized_pct':   float(vol * np.sqrt(ANN_FACTOR) * 100),
        'var_95_pct':                  float(var_95  * 100),
        'cvar_95_pct':                 float(cvar_95 * 100),
        'ulcer_index_pct':             _calculate_ulcer_index(returns) * 100,
        'avg_return_pct':              avg_return_pct,
        'geometric_mean_pct':          _calculate_geometric_mean(returns) * 100,
        'risk_of_ruin':                _calculate_risk_of_ruin(returns),
        'rolling_volatility_mean_pct': mean_vol * 100,
        'rolling_volatility_std_pct':  std_vol  * 100,
    }
