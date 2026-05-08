"""
Technical Analysis (TA) Module

This module provides comprehensive technical analysis capabilities including:
- Technical indicators (momentum, volatility, volume, trend)
- Candlestick pattern recognition
- Smart Money Concepts (SMC) analysis
- Utility functions for data validation and processing

Main Components:
- indicators: Technical indicator calculations (TA-Lib, VectorBT)
- patterns: Candlestick pattern recognition
- smc: Smart Money Concepts analysis
- utils: Data validation and utility functions
"""

# =============================================================================
# Technical Indicators
# =============================================================================
from bitpredict.common.ta.indicators.base import (
    # Main calculation functions
    calculate_indicators,
    calculate_talib_indicators,
    calculate_vectorbt_indicators
)

# Indicator registry and utilities
from bitpredict.common.ta.indicators.talib.registry import (
    get_indicators_by_category,
    create_column_name,
    TALIB_INDICATORS,
    INDICATOR_CATEGORIES,
    INDICATOR_COUNTS
)

# Plotting functionality
try:
    from bitpredict.common.ta.indicators.plot import (
        IndicatorPlotter,
        plot_indicators
    )
except ImportError:
    # Handle case where plotting dependencies are not available
    IndicatorPlotter = None
    plot_indicators = None

# =============================================================================
# Candlestick Patterns
# =============================================================================
from bitpredict.common.ta.patterns.base import (
    # Main calculation functions
    calculate_patterns,
    calculate_talib_patterns
)

# Pattern utilities
try:
    from bitpredict.common.ta.patterns.pattern_to_signal_converter import signals_from_patterns
    from bitpredict.common.ta.patterns.plot import plot_patterns
except ImportError:
    # Handle case where pattern utilities are not available
    signals_from_patterns = None
    plot_patterns = None

# Pattern registry
try:
    from bitpredict.common.ta.patterns.talib.registry import PatternRegistry
except ImportError:
    PatternRegistry = None

# =============================================================================
# Smart Money Concepts (SMC)
# =============================================================================
try:
    from bitpredict.common.ta.smc.base import SMCBase
    from bitpredict.common.ta.smc.core import smc
    from bitpredict.common.ta.smc.plot import (
        plot as smc_plot,
        plot_from_combined as smc_plot_from_combined,
        create_figure_with_smc
    )
except ImportError:
    # Handle case where SMC module is not available
    SMCBase = None
    smc = None
    smc_plot = None
    smc_plot_from_combined = None
    create_figure_with_smc = None


# =============================================================================
# Public API - Main Functions
# =============================================================================
__all__ = [
    
    # Technical Indicators
    "calculate_indicators",
    "calculate_talib_indicators", 
    "calculate_vectorbt_indicators",
    "get_indicators_by_category",
    "create_column_name",
    "IndicatorPlotter",
    "plot_indicators",
    
    # Candlestick Patterns
    "calculate_patterns",
    "calculate_talib_patterns",
    "calculate_vectorbt_patterns", 
    "signals_from_patterns",
    "plot_patterns",
    "PatternRegistry",
    
    # Smart Money Concepts
    "SMCBase",
    "smc",
    "smc_plot",
    "smc_plot_from_combined",
    "create_figure_with_smc",
    
    # Constants and Registries
    "TALIB_INDICATORS",
    "INDICATOR_CATEGORIES", 
    "INDICATOR_COUNTS",
]





