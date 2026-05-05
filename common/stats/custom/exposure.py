import numpy as np
import pandas as pd
from typing import Dict, Any

def _get_empty_exposure() -> Dict[str, float]:
    return {
        # Core exposure metrics
        'gross_exposure_current': 0.0,
        'gross_exposure_max': 0.0,
        'gross_exposure_avg': 0.0,
        'max_gross_exposure_pct': 0.0,
        'net_exposure_current': 0.0,
        'net_exposure_max': 0.0,
        'net_exposure_min': 0.0,
        'net_exposure_avg': 0.0,
        'net_exposure_range': 0.0,
        
        # Position coverage metrics
        'position_coverage_pct': 0.0,
        'position_coverage': 0.0,
        'long_position_coverage': 0.0,
        'short_position_coverage': 0.0,
        
        # Exposure statistics
        'exposure_volatility': 0.0,
        'net_exposure_volatility': 0.0,
        'exposure_coefficient_of_variation': 0.0,
        'avg_exposure_utilization': 0.0,
        'exposure_consistency': 0.0,
        
        # Long/Short balance
        'long_exposure_pct': 0.0,
        'short_exposure_pct': 0.0,
        'exposure_directional_bias': 0.0,
        
        # Exposure distribution percentiles
        'exposure_p25': 0.0,
        'exposure_p50': 0.0,
        'exposure_p75': 0.0,
        'exposure_p90': 0.0,
        'exposure_p95': 0.0,
        
        # Time-based metrics
        'total_periods': 0,
        'position_periods': 0,
        'long_periods': 0,
        'short_periods': 0,
        'idle_periods': 0,
        
        # Missing QuantStats exposure metrics
        'market_exposure_time': 0.0,
        'active_trading_time_pct': 0.0,
        'trading_intensity': 0.0
    }

def _calculate_exposure(
    df_ledger: pd.DataFrame,
    directions: np.ndarray,
    actions: np.ndarray
) -> Dict[str, float]:
    """Optimized exposure analysis using vectorized NumPy operations"""
    
    if len(df_ledger) == 0:
        return _get_empty_exposure()
    
    # Extract position sizes (as fraction 0-1) and directions from ledger
    if 'position_size_pct' in df_ledger.columns:
        position_sizes = df_ledger['position_size_pct'].values / 100.0
    else:
        position_sizes = np.ones(len(df_ledger))
    trade_directions = df_ledger['direction'].values if 'direction' in df_ledger.columns else directions
    
    # Convert directions to numeric (1 for long, -1 for short)
    direction_numeric = np.where(pd.Series(trade_directions).str.lower() == 'long', 1, -1)
    
    # Calculate exposure values using vectorized operations
    gross_exposure = np.abs(position_sizes)  # Absolute position sizes
    net_exposure = position_sizes * direction_numeric  # Signed position sizes
    
    # Current exposure (last position in ledger)
    gross_exposure_current = float(gross_exposure[-1]) if len(gross_exposure) > 0 else 0.0
    net_exposure_current = float(net_exposure[-1]) if len(net_exposure) > 0 else 0.0
    
    # Maximum and average exposures
    gross_exposure_max = float(np.max(gross_exposure)) if len(gross_exposure) > 0 else 0.0
    gross_exposure_avg = float(np.mean(gross_exposure)) if len(gross_exposure) > 0 else 0.0
    max_gross_exposure_pct = gross_exposure_max * 100.0
    
    net_exposure_max = float(np.max(net_exposure)) if len(net_exposure) > 0 else 0.0
    net_exposure_min = float(np.min(net_exposure)) if len(net_exposure) > 0 else 0.0
    net_exposure_avg = float(np.mean(net_exposure)) if len(net_exposure) > 0 else 0.0
    
    # Position coverage analysis
    total_periods = len(df_ledger)
    
    # Count periods with positions
    long_periods = np.sum(direction_numeric > 0)
    short_periods = np.sum(direction_numeric < 0)
    position_periods = long_periods + short_periods
    
    # Coverage percentages
    position_coverage_pct = (position_periods / total_periods * 100.0) if total_periods > 0 else 0.0
    position_coverage = position_coverage_pct
    long_position_coverage = (long_periods / total_periods * 100.0) if total_periods > 0 else 0.0
    short_position_coverage = (short_periods / total_periods * 100.0) if total_periods > 0 else 0.0
    
    # Additional exposure metrics
    exposure_volatility = float(np.std(gross_exposure)) if len(gross_exposure) > 1 else 0.0
    net_exposure_volatility = float(np.std(net_exposure)) if len(net_exposure) > 1 else 0.0
    
    # Exposure efficiency metrics
    avg_exposure_utilization = gross_exposure_avg / gross_exposure_max if gross_exposure_max > 0 else 0.0
    exposure_consistency = 1.0 - (exposure_volatility / gross_exposure_avg) if gross_exposure_avg > 0 else 0.0
    
    # Long/Short exposure balance
    long_exposure_pct = (long_periods / position_periods * 100.0) if position_periods > 0 else 0.0
    short_exposure_pct = (short_periods / position_periods * 100.0) if position_periods > 0 else 0.0
    
    # Exposure concentration metrics
    exposure_percentiles = np.percentile(gross_exposure, [25, 50, 75, 90, 95]) if len(gross_exposure) > 0 else np.zeros(5)
    exposure_p25, exposure_p50, exposure_p75, exposure_p90, exposure_p95 = exposure_percentiles
    
    # Risk-adjusted exposure metrics
    exposure_coefficient_of_variation = (exposure_volatility / gross_exposure_avg) if gross_exposure_avg > 0 else 0.0
    net_exposure_range = net_exposure_max - net_exposure_min
    exposure_directional_bias = abs(net_exposure_avg) / gross_exposure_avg if gross_exposure_avg > 0 else 0.0
    
    # Missing QuantStats exposure metrics
    market_exposure_time = position_coverage_pct  # Percentage of time with market positions
    active_trading_time_pct = position_coverage_pct  # Same as market exposure for our use case
    
    # Trading intensity (position changes per period)
    # In new format, every row is a change/trade
    position_changes = len(df_ledger)
    trading_intensity = (position_changes / total_periods) * 100.0 if total_periods > 0 else 0.0
    
    return {
        # Core exposure metrics
        'gross_exposure_current': float(gross_exposure_current),
        'gross_exposure_max': float(gross_exposure_max),
        'gross_exposure_avg': float(gross_exposure_avg),
        'max_gross_exposure_pct': float(max_gross_exposure_pct),
        'net_exposure_current': float(net_exposure_current),
        'net_exposure_max': float(net_exposure_max),
        'net_exposure_min': float(net_exposure_min),
        'net_exposure_avg': float(net_exposure_avg),
        'net_exposure_range': float(net_exposure_range),
        
        # Position coverage metrics
        'position_coverage_pct': float(position_coverage_pct),
        'position_coverage': float(position_coverage),
        'long_position_coverage': float(long_position_coverage),
        'short_position_coverage': float(short_position_coverage),
        
        # Exposure statistics
        'exposure_volatility': float(exposure_volatility),
        'net_exposure_volatility': float(net_exposure_volatility),
        'exposure_coefficient_of_variation': float(exposure_coefficient_of_variation),
        'avg_exposure_utilization': float(avg_exposure_utilization),
        'exposure_consistency': float(exposure_consistency),
        
        # Long/Short balance
        'long_exposure_pct': float(long_exposure_pct),
        'short_exposure_pct': float(short_exposure_pct),
        'exposure_directional_bias': float(exposure_directional_bias),
        
        # Exposure distribution percentiles
        'exposure_p25': float(exposure_p25),
        'exposure_p50': float(exposure_p50),
        'exposure_p75': float(exposure_p75),
        'exposure_p90': float(exposure_p90),
        'exposure_p95': float(exposure_p95),
        
        # Time-based metrics
        'total_periods': int(total_periods),
        'position_periods': int(position_periods),
        'long_periods': int(long_periods),
        'short_periods': int(short_periods),
        'idle_periods': int(total_periods - position_periods),
        
        # Missing QuantStats exposure metrics
        'market_exposure_time': float(market_exposure_time),
        'active_trading_time_pct': float(active_trading_time_pct),
        'trading_intensity': float(trading_intensity)
    }
