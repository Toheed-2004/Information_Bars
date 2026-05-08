import numpy as np
from typing import Dict, Any
from ..shared.utils import ANN_FACTOR

def _get_empty_drawdown_analysis() -> Dict[str, Any]:
    return {
        'max_drawdown_pct': 0.0, 'max_drawdown_duration_days': 0.0,
        'max_drawdown_days': 0, 'avg_drawdown_pct': 0.0,
        'avg_drawdown_days': 0.0, 'current_drawdown_pct': 0.0,
        'current_drawdown_days': 0, 'recovery_factor': 0.0,
        'drawdown_periods_count': 0, 'drawdown_duration_total': 0,
        'drawdown_volatility_pct': 0.0
    }

def _calculate_drawdown_analysis(
    returns: np.ndarray,
    balance_array: np.ndarray,
    timestamps: np.ndarray,
    regime_analysis: bool = False
) -> Dict[str, Any]:
    """Drawdown analysis using vectorized NumPy operations matching VBT's approach."""
    
    if len(returns) == 0:
        return _get_empty_drawdown_analysis()
    
    # Cumulative prices and drawdown series
    prices = np.cumprod(1 + returns)
    cumulative_max = np.maximum.accumulate(prices)
    drawdown_series = prices / cumulative_max - 1.0
    
    max_drawdown = np.min(drawdown_series)
    max_drawdown_pct = abs(max_drawdown) * 100

    if regime_analysis:
        return {'max_drawdown_pct': float(max_drawdown_pct)}
    
    current_drawdown_pct = abs(drawdown_series[-1]) * 100
    
    # Drawdown volatility (matching VBT: std of drawdown series)
    drawdown_volatility_pct = float(np.std(drawdown_series, ddof=1) * 100) if len(drawdown_series) > 1 else 0.0
    
    # Identify drawdown periods using transitions
    no_drawdown = drawdown_series >= -1e-10
    prev_no_dd = np.empty_like(no_drawdown)
    prev_no_dd[0] = True
    prev_no_dd[1:] = no_drawdown[:-1]
    
    starts_indices = np.flatnonzero((~no_drawdown) & prev_no_dd)
    ends_indices = np.flatnonzero(no_drawdown & (~prev_no_dd)) - 1
    
    if starts_indices.size and ends_indices.size and starts_indices[0] > ends_indices[0]:
        starts_indices = np.insert(starts_indices, 0, 0)
    if starts_indices.size and (ends_indices.size == 0 or starts_indices[-1] > ends_indices[-1]):
        ends_indices = np.append(ends_indices, len(drawdown_series) - 1)
    
    if starts_indices.size == 0:
        return {
            'max_drawdown_pct': float(max_drawdown_pct),
            'max_drawdown_duration_days': 0.0,
            'max_drawdown_days': 0,
            'avg_drawdown_pct': 0.0,
            'avg_drawdown_days': 0.0,
            'current_drawdown_pct': float(current_drawdown_pct),
            'current_drawdown_days': 0,
            'recovery_factor': 0.0,
            'drawdown_periods_count': 0,
            'drawdown_duration_total': 0,
            'drawdown_volatility_pct': float(drawdown_volatility_pct)
        }
    
    # Durations in bars
    durations = ends_indices - starts_indices + 1
    max_drawdown_duration_days = float(durations.max())
    avg_drawdown_days = float(durations.mean())
    
    # avg_drawdown_pct: per-period (start_value - valley_value) / start_value (matching VBT)
    dd_pcts = []
    for s, e in zip(starts_indices, ends_indices):
        start_price = prices[s - 1] if s > 0 else 1.0
        valley_price = prices[s:e + 1].min()
        if start_price > 0:
            dd_pcts.append((start_price - valley_price) / start_price * 100)
    avg_drawdown_pct = float(np.mean(dd_pcts)) if dd_pcts else 0.0
    
    # Current drawdown duration
    current_drawdown_days = 0
    if drawdown_series[-1] < 0:
        current_drawdown_days = int(len(drawdown_series) - starts_indices[-1])
    
    # Recovery factor: total_return / max_drawdown (matching VBT)
    total_return = float(prices[-1] - 1.0)
    recovery_factor = abs(total_return) / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0
    
    return {
        'max_drawdown_pct': float(max_drawdown_pct),
        'max_drawdown_duration_days': max_drawdown_duration_days,
        'max_drawdown_days': int(max_drawdown_duration_days),
        'avg_drawdown_pct': avg_drawdown_pct,
        'avg_drawdown_days': avg_drawdown_days,
        'current_drawdown_pct': float(current_drawdown_pct),
        'current_drawdown_days': current_drawdown_days,
        'recovery_factor': float(recovery_factor),
        'drawdown_periods_count': int(starts_indices.size),
        'drawdown_duration_total': float(durations.sum()),
        'drawdown_volatility_pct': float(drawdown_volatility_pct)
    }

def _calculate_ulcer_index(returns: np.ndarray) -> float:
    """Calculate Ulcer Index matching QuantStats"""
    if len(returns) == 0:
        return 0.0
    
    # Calculate cumulative returns for drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    
    # Ulcer Index = sqrt(mean(drawdowns^2)) * sqrt(periods_per_year)
    ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
    
    # Annualize (QuantStats uses sqrt(252) for daily data)
    ulcer_index_annualized = ulcer_index * np.sqrt(ANN_FACTOR)
    
    return ulcer_index_annualized
