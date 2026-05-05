"""
Technical Indicators Module

Complete indicator calculation system with 100+ indicators.
Supports TA-Lib and VectorBT libraries with unified interface.
"""

# Main calculation functions
from bitpredict.common.ta.indicators.base import (
    calculate_indicators,
    calculate_talib_indicators,
    calculate_vectorbt_indicators
)

# Registry and utilities
from bitpredict.common.ta.indicators.talib.registry import (
    get_indicators_by_category,
    create_column_name
)

# Plotting
from bitpredict.common.ta.indicators.plot import plot_indicators, IndicatorPlotter


__all__ = [
    # Main functions
    'calculate_indicators',
    'calculate_talib_indicators', 
    'calculate_vectorbt_indicators',
    
    # Registry functions
    'get_indicators_by_category',
    'create_column_name',
    
    # Plotting
    'plot_indicators',
    'IndicatorPlotter',
]