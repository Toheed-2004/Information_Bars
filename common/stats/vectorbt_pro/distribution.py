import numpy as np
from typing import Dict, Any
from scipy import stats as scipy_stats


def _extract_distribution_stats_vectorized(cache: Dict) -> Dict[str, Any]:
    """
    Return distribution characteristics from daily_returns.

    Uses daily_returns (not raw bar returns) — consistent with all ratio calculations.
    Kurtosis: excess (fisher=True, normal=0) — consistent with risk.py.
    VaR/CVaR: historical percentile — consistent with risk.py.
    Jarque-Bera: correct formula using excess kurtosis → n/6 * (S² + K²/4).
    """
    returns = cache.get('daily_returns', np.array([]))

    _empty: Dict[str, Any] = {
        'returns_mean_pct': 0.0, 'returns_std_pct': 0.0, 'returns_variance_pct': 0.0,
        'skewness': 0.0, 'kurtosis': 0.0,
        'percentile_1': 0.0, 'percentile_5': 0.0, 'percentile_10': 0.0,
        'percentile_25': 0.0, 'percentile_50': 0.0, 'percentile_75': 0.0,
        'percentile_90': 0.0, 'percentile_95': 0.0, 'percentile_99': 0.0,
        'var_95_pct': 0.0, 'var_99_pct': 0.0, 'cvar_95_pct': 0.0, 'cvar_99_pct': 0.0,
        'tail_ratio': 0.0, 'downside_deviation_pct': 0.0, 'upside_deviation_pct': 0.0,
        'positive_returns_count': 0, 'negative_returns_count': 0,
        'zero_returns_count': 0, 'positive_returns_pct': 0.0,
        'negative_returns_pct': 0.0, 'zero_returns_pct': 0.0,
        'positive_mean_pct': 0.0, 'negative_mean_pct': 0.0, 'gain_loss_ratio': 0.0,
        'coefficient_of_variation': 0.0, 'jarque_bera_statistic': 0.0,
        'max_return_pct': 0.0, 'min_return_pct': 0.0, 'return_range_pct': 0.0,
        'outliers_upper': 0, 'outliers_lower': 0, 'total_outliers': 0,
        'outlier_pct': 0.0, 'total_observations': 0, 'distribution_analysis_complete': 0.0,
    }

    if len(returns) == 0:
        return _empty

    n       = len(returns)
    mean_r  = np.mean(returns)
    std_r   = np.std(returns, ddof=1) if n > 1 else 0.0
    var_r   = std_r ** 2

    # Skewness and excess kurtosis (fisher=True → normal=0)
    if n > 3 and std_r > 0:
        skew = float(scipy_stats.skew(returns, bias=False))
        kurt = float(scipy_stats.kurtosis(returns, bias=False, fisher=True))  # excess
    else:
        skew = kurt = 0.0

    # Jarque-Bera using excess kurtosis: JB = n/6 * (S² + K²/4)
    jb = float(n / 6.0 * (skew ** 2 + kurt ** 2 / 4.0))

    # Percentiles (single pass)
    perc = np.percentile(returns, [1, 5, 10, 25, 50, 75, 90, 95, 99])

    # VaR / CVaR — historical percentile (consistent with risk.py)
    var_95 = perc[1]   # 5th percentile
    var_99 = perc[0]   # 1st percentile
    cvar_95 = float(np.mean(returns[returns <= var_95])) if np.any(returns <= var_95) else var_95
    cvar_99 = float(np.mean(returns[returns <= var_99])) if np.any(returns <= var_99) else var_99

    # Groupings
    pos  = returns[returns > 0]
    neg  = returns[returns < 0]
    zero = returns[returns == 0]

    pos_mean = float(np.mean(pos)) if len(pos) > 0 else 0.0
    neg_mean = float(np.mean(neg)) if len(neg) > 0 else 0.0

    # Tail ratio (95th / abs(5th))
    tail_ratio = float(abs(perc[7]) / abs(perc[1])) if perc[1] != 0 else 0.0

    # Outliers via IQR
    iqr             = perc[5] - perc[3]
    outliers_upper  = int(np.sum(returns > perc[5] + 1.5 * iqr))
    outliers_lower  = int(np.sum(returns < perc[3] - 1.5 * iqr))
    total_outliers  = outliers_upper + outliers_lower

    return {
        'returns_mean_pct':       float(mean_r * 100),
        'returns_std_pct':        float(std_r * 100),
        'returns_variance_pct':   float(var_r * 10000),
        'skewness':               skew,
        'kurtosis':               kurt,
        'percentile_1':           float(perc[0] * 100),
        'percentile_5':           float(perc[1] * 100),
        'percentile_10':          float(perc[2] * 100),
        'percentile_25':          float(perc[3] * 100),
        'percentile_50':          float(perc[4] * 100),
        'percentile_75':          float(perc[5] * 100),
        'percentile_90':          float(perc[6] * 100),
        'percentile_95':          float(perc[7] * 100),
        'percentile_99':          float(perc[8] * 100),
        'var_95_pct':             float(var_95 * 100),
        'var_99_pct':             float(var_99 * 100),
        'cvar_95_pct':            float(cvar_95 * 100),
        'cvar_99_pct':            float(cvar_99 * 100),
        'tail_ratio':             tail_ratio,
        'downside_deviation_pct': float(np.std(neg, ddof=1) * 100) if len(neg) > 1 else 0.0,
        'upside_deviation_pct':   float(np.std(pos, ddof=1) * 100) if len(pos) > 1 else 0.0,
        'positive_returns_count': int(len(pos)),
        'negative_returns_count': int(len(neg)),
        'zero_returns_count':     int(len(zero)),
        'positive_returns_pct':   float(len(pos) / n * 100),
        'negative_returns_pct':   float(len(neg) / n * 100),
        'zero_returns_pct':       float(len(zero) / n * 100),
        'positive_mean_pct':      float(pos_mean * 100),
        'negative_mean_pct':      float(neg_mean * 100),
        'gain_loss_ratio':        float(abs(pos_mean / neg_mean)) if neg_mean != 0 else 0.0,
        'coefficient_of_variation': float(std_r / abs(mean_r)) if mean_r != 0 else 0.0,
        'jarque_bera_statistic':  jb,
        'max_return_pct':         float(np.max(returns) * 100),
        'min_return_pct':         float(np.min(returns) * 100),
        'return_range_pct':       float((np.max(returns) - np.min(returns)) * 100),
        'outliers_upper':         outliers_upper,
        'outliers_lower':         outliers_lower,
        'total_outliers':         total_outliers,
        'outlier_pct':            float(total_outliers / n * 100),
        'total_observations':     int(n),
        'distribution_analysis_complete': 1.0,
    }
