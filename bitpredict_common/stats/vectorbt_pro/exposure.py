"""
Exposure analysis — all from cache arrays (daily_pf gross/net/long/short exposure).
All fraction values (0-1) multiplied by 100 and named with _pct suffix.
"""
import numpy as np
from typing import Dict, Any


def _get_empty_exposure() -> Dict[str, Any]:
    return {
        'gross_exposure_current_pct': 0.0,
        'gross_exposure_max_pct':     0.0,
        'gross_exposure_avg_pct':     0.0,
        'net_exposure_current_pct':   0.0,
        'net_exposure_max_pct':       0.0,
        'net_exposure_min_pct':       0.0,
        'net_exposure_avg_pct':       0.0,
        'net_exposure_range_pct':     0.0,
        'position_coverage_pct':      0.0,
        'long_position_coverage_pct': 0.0,
        'short_position_coverage_pct':0.0,
        'long_exposure_pct':          0.0,
        'short_exposure_pct':         0.0,
        'exposure_volatility_pct':    0.0,
        'net_exposure_volatility_pct':0.0,
        'exposure_coefficient_of_variation': 0.0,
        'avg_exposure_utilization':   0.0,
        'exposure_consistency':       0.0,
        'exposure_directional_bias':  0.0,
        'exposure_p25_pct': 0.0, 'exposure_p50_pct': 0.0,
        'exposure_p75_pct': 0.0, 'exposure_p90_pct': 0.0,
        'exposure_p95_pct': 0.0,
        'total_periods':    0, 'position_periods': 0,
        'long_periods':     0, 'short_periods':    0,
        'idle_periods':     0,
    }


def _extract_exposure_stats_full(cache: Dict) -> Dict[str, Any]:
    """
    Comprehensive exposure stats from cache arrays.
    All values returned in % (fraction * 100) where applicable.
    max_gross_exposure_pct and position_coverage_pct taken directly from cache.
    """
    gross_arr = cache.get('gross_exposure_array', np.array([]))
    net_arr   = cache.get('net_exposure_array',   np.array([]))
    long_arr  = cache.get('long_exposure_array',  np.array([]))
    short_arr = cache.get('short_exposure_array', np.array([]))

    if len(gross_arr) == 0:
        return _get_empty_exposure()

    n   = len(gross_arr)
    p2c = 100.0  # fraction → pct multiplier

    # Core gross exposure
    gross_current = float(gross_arr[-1]  * p2c)
    gross_max     = float(cache.get('max_gross_exposure_pct', np.max(gross_arr) * p2c))
    gross_avg     = float(np.mean(gross_arr) * p2c)

    # Net exposure
    if len(net_arr) > 0:
        net_current = float(net_arr[-1]  * p2c)
        net_max     = float(np.max(net_arr)  * p2c)
        net_min     = float(np.min(net_arr)  * p2c)
        net_avg     = float(np.mean(net_arr) * p2c)
    else:
        net_current = net_max = net_min = net_avg = 0.0

    # Coverage periods
    active_mask = gross_arr > 0
    long_mask   = long_arr  > 0 if len(long_arr)  > 0 else np.zeros(n, dtype=bool)
    short_mask  = short_arr > 1e-10 if len(short_arr) > 0 else np.zeros(n, dtype=bool)

    position_periods = int(np.sum(active_mask))
    long_periods     = int(np.sum(long_mask))
    short_periods    = int(np.sum(short_mask))
    idle_periods     = n - position_periods

    position_coverage_pct  = float(cache.get('position_coverage_pct',
                                              position_periods / n * 100 if n > 0 else 0.0))
    long_position_cov_pct  = float(long_periods  / n * 100) if n > 0 else 0.0
    short_position_cov_pct = float(short_periods / n * 100) if n > 0 else 0.0
    long_exposure_pct      = float(long_periods  / position_periods * 100) if position_periods > 0 else 0.0
    short_exposure_pct     = float(short_periods / position_periods * 100) if position_periods > 0 else 0.0

    # Volatility / derived ratios (computed from fraction arrays, then * 100 for _pct)
    gross_std = float(np.std(gross_arr, ddof=1)) if n > 1 else 0.0
    net_std   = float(np.std(net_arr,   ddof=1)) if len(net_arr) > 1 else 0.0
    gross_avg_f = float(np.mean(gross_arr))  # fraction for ratio calcs
    gross_max_f = float(np.max(gross_arr))

    coeff_var      = gross_std / gross_avg_f                     if gross_avg_f > 0 else 0.0
    utilization    = gross_avg_f / gross_max_f                   if gross_max_f > 0 else 0.0
    consistency    = 1.0 - (gross_std / gross_avg_f)             if gross_avg_f > 0 else 0.0
    dir_bias       = abs(float(np.mean(net_arr)) if len(net_arr) > 0 else 0.0) / gross_avg_f \
                     if gross_avg_f > 0 else 0.0

    p25, p50, p75, p90, p95 = np.percentile(gross_arr, [25, 50, 75, 90, 95]) if n > 0 else (0,)*5

    return {
        'gross_exposure_current_pct':        gross_current,
        'gross_exposure_max_pct':            gross_max,
        'gross_exposure_avg_pct':            gross_avg,
        'net_exposure_current_pct':          net_current,
        'net_exposure_max_pct':              net_max,
        'net_exposure_min_pct':              net_min,
        'net_exposure_avg_pct':              net_avg,
        'net_exposure_range_pct':            net_max - net_min,
        'position_coverage_pct':             position_coverage_pct,
        'long_position_coverage_pct':        long_position_cov_pct,
        'short_position_coverage_pct':       short_position_cov_pct,
        'long_exposure_pct':                 long_exposure_pct,
        'short_exposure_pct':                short_exposure_pct,
        'exposure_volatility_pct':           gross_std * p2c,
        'net_exposure_volatility_pct':       net_std   * p2c,
        'exposure_coefficient_of_variation': float(coeff_var),
        'avg_exposure_utilization':          float(utilization),
        'exposure_consistency':              float(consistency),
        'exposure_directional_bias':         float(dir_bias),
        'exposure_p25_pct': float(p25 * p2c), 'exposure_p50_pct': float(p50 * p2c),
        'exposure_p75_pct': float(p75 * p2c), 'exposure_p90_pct': float(p90 * p2c),
        'exposure_p95_pct': float(p95 * p2c),
        'total_periods':    int(n),
        'position_periods': position_periods,
        'long_periods':     long_periods,
        'short_periods':    short_periods,
        'idle_periods':     idle_periods,
    }
