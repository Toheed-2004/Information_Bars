"""
Candlestick Patterns Module

Pattern recognition system supporting TA-Lib candlestick patterns.
Provides both functional and class-based interfaces.
"""

# Main calculation functions
from bitpredict.common.ta.patterns.base import (
    calculate_patterns,
    calculate_talib_patterns
    
)

# Utilities (optional)
try:
    from bitpredict.common.ta.patterns.plot import plot_patterns
    from bitpredict.common.ta.patterns.pattern_to_signal_converter import signals_from_patterns
except ImportError:
    plot_patterns = None
    signals_from_patterns = None

# Registry (optional)
try:
    from bitpredict.common.ta.patterns.talib.registry import PatternRegistry
except ImportError:
    PatternRegistry = None

__all__ = [
    # Main functions
    "calculate_patterns",
    "calculate_talib_patterns", 
    "calculate_vectorbt_patterns",
    
    # Utilities
    "plot_patterns",
    "signals_from_patterns",
    
    # Registry
    "PatternRegistry"
]

# Usage Examples:
# 
# Functional interface (recommended):
# from common.ta.patterns import calculate_patterns
# result, metadata = calculate_patterns(df, patterns=["CDL_DOJI", "CDL_HAMMER"])
#
# Class interface (legacy):
# from common.ta.patterns import PatternCalculator
# calc = PatternCalculator(df, ["CDL_DOJI", "CDL_HAMMER"], library="talib")
# result, config = calc.calculate()
