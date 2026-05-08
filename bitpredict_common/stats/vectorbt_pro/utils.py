import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def _transform_ledger(ledger_df):
    """
    Transform the new single-row trade ledger format to the format expected by rolling stats.
    """
    if ledger_df is None:
        return None
    
    if 'exit_datetime' not in ledger_df.columns:
        logger.debug("Ledger missing required column: exit_datetime")
        return None

    # No copy needed - we only read/filter, not modify
    df = ledger_df

    # Filter to closed trades if status column exists
    if 'status' in df.columns:
        df = df[df['status'] == 'Closed']
        if df.empty:
            logger.debug("No closed trades found in ledger")
            return None

    df['datetime'] = pd.to_datetime(df['exit_datetime'])

    # Prefer account-level return (impact on equity); fall back to trade-level return
    if 'account_return_pct' in df.columns:
        df['return_pct'] = df['account_return_pct']
    elif 'trade_return_pct' in df.columns:
        df['return_pct'] = df['trade_return_pct']
    else:
        df['return_pct'] = 0.0

    return df[['datetime', 'return_pct']].dropna(subset=['datetime'])

def _safe_call_method(pf, attr_name, default=0.0):
    """Safely call a portfolio method"""
    try:
        if hasattr(pf, attr_name):
            result = getattr(pf, attr_name)()
            return float(result) if pd.notna(result) and not np.isinf(result) else default
    except:
        pass
    return default

def _safe_call_last(pf, attr_name, default=0.0):
    """Safely get last value from a series property or method"""
    try:
        if hasattr(pf, attr_name):
            attr = getattr(pf, attr_name)
            
            # If it's a callable (method), call it
            if callable(attr):
                series = attr()
            else:
                # It's a property, use it directly
                series = attr
                
            if hasattr(series, 'iloc') and len(series) > 0:
                return float(series.iloc[-1]) if pd.notna(series.iloc[-1]) else default
    except:
        pass
    return default

def _safe_call_first(pf, attr_name, default=0.0):
    """Safely get first value from a series property or method"""
    try:
        if hasattr(pf, attr_name):
            attr = getattr(pf, attr_name)
            
            # If it's a callable (method), call it
            if callable(attr):
                series = attr()
            else:
                # It's a property, use it directly
                series = attr
                
            if hasattr(series, 'iloc') and len(series) > 0:
                return float(series.iloc[0]) if pd.notna(series.iloc[0]) else default
    except:
        pass
    return default

def _sig_round(value: float, sig: int = 6) -> float:
    """
    Round to `sig` significant figures.
    Handles very small values correctly (e.g. 0.000123 stays 0.000123, not 0.0).
    Returns 0.0 for non-finite or zero.
    """
    import math
    if not np.isfinite(value) or value == 0.0:
        return 0.0
    d = math.ceil(math.log10(abs(value)))
    power = sig - d
    factor = 10 ** power
    return round(value * factor) / factor


def _format_numeric_values_vectorized(stats_dict: Dict[str, Any], sig: int = 6) -> Dict[str, Any]:
    """
    Recursively format all numeric values in a stats dict to `sig` significant figures.
    - Integers kept as int.
    - NaN / Inf → 0.0.
    - Very small floats preserved correctly (significant-figure rounding, not fixed decimal).
    """
    formatted: Dict[str, Any] = {}
    for key, value in stats_dict.items():
        try:
            if isinstance(value, dict):
                formatted[key] = _format_numeric_values_vectorized(value, sig)
            elif isinstance(value, list):
                formatted[key] = [
                    _sig_round(float(item), sig)
                    if isinstance(item, (float, np.floating)) and np.isfinite(item)
                    else item
                    for item in value
                ]
            elif isinstance(value, (int, np.integer)):
                formatted[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                formatted[key] = _sig_round(float(value), sig)
            else:
                formatted[key] = value
        except Exception:
            formatted[key] = value if value is not None else 0.0
    return formatted

def _get_empty_comprehensive_stats() -> Dict[str, Any]:
    """Return empty comprehensive stats dictionary"""
    return {
        "error": "No portfolio object provided or calculation failed",
        "total_stats_calculated": 0
    }

def get_benchmark_to_dict(bm_series):
    """
    Convert benchmark Series (datetime index) to {'timestamp': value} dict.
    bm_series is cache['bm_returns_series'] — auto-populated from close price.
    Values multiplied by 100 to match percentage scale of other graph metrics.
    """
    if bm_series is None or not hasattr(bm_series, 'index'):
        return {"benchmark_returns": {}}
    try:
        idx = pd.to_datetime(bm_series.index, utc=True)
        return {
            "benchmark_returns": dict(
                zip(idx.strftime("%Y-%m-%d %H:%M:%S"), (bm_series.astype(float) * 100).tolist())
            )
        }
    except Exception:
        return {"benchmark_returns": {}}
