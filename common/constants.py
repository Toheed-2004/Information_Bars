"""
Common constants used across the BitPredict application.
"""
import os
from pathlib import Path

# =============================================================================
# Logging
# =============================================================================

# Root directory name for all application logs
LOG_DIR_NAME = "logs"


# =============================================================================
# Timezone
# =============================================================================

# Default timezone used across the system (stored in DB, used in processing)
DEFAULT_TIMEZONE = "UTC"


# =============================================================================
# Time Units
# =============================================================================

# Conversion of time units to minutes
# Used for resampling, validation, and interval calculations
TIME_UNITS = {
    "m": 1,  # minute
    "h": 60,  # hour
    "d": 1440,  # day
    "w": 10080,  # week
    "M": 43200,  # month (30 days)
}


# =============================================================================
# Supported Time Horizons
# =============================================================================

# Standard OHLCV timeframes supported by the system
# Used for resampling, storage, and validation
SUPPORTED_TIMEFRAMES = [
    "1m",   # 1 minute
    "5m",   # 5 minutes
    "15m",  # 15 minutes
    "30m",  # 30 minutes
    "1h",   # 1 hour
    "2h",   # 2 hours
    "3h",   # 3 hours
    "4h",   # 4 hours
    "6h",   # 6 hours
    "8h",   # 8 hours
    "12h",  # 12 hours
    "1d",   # 1 day (24 hours)
    "3d",   # 3 day
    "1w",   # 1 week(7 days)
    "1M"    # 1 Month
]


# =============================================================================
# OHLCV / Indicator Data
# =============================================================================

# Standard column order for OHLCV-based dataframes
OHLCV_COLUMNS = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OHLCV_COLUMN = [
    "open",
    "high",
    "low",
    "close",
    "volume"
]
# =============================================================================
# Database Naming Conventions
# =============================================================================

DATA_SCHEMA = "data_bars"
MARKET_REGIME_STATE = "market_regime_state"

# =============================================================================
# Meta Schema & Tables
# =============================================================================

# Central metadata schema
META_SCHEMA = "meta"

# Meta tables
META_TIME_BARS_TABLE = "time_bars"
META_CUSTOM_BARS_TABLE = "custom_bars"
META_TICK_TABLE = "tick"
META_BLOCKCHAIN_TABLE = "blockchain"
META_MACRO_TABLE = "macro"
META_SYMBOLS_TABLE = "symbols"

# =============================================================================
# Bars Schema
# =============================================================================

# BAR TYPES
ALLOWED_BAR_TYPES = ["volume", "volatility", "dollar", "renko", "range", "hybrid"]

# Supported pattern calculation libraries
SUPPORTED_PATTERN_LIBRARIES = ["talib"]

# Prefix for TA-Lib candlestick pattern column names
# Used to identify pattern columns in DataFrame (e.g., 'talib_cdl_doji')
PATTERN_TALIB_COL_PREFIX = "talib_"


# =============================================================================
# Strategies pipeline related contstants
# =============================================================================
STRATEGIES_SCHEMA = "strategies"
SIMULATOR_SCHEMA = "simulator"

COL_TO_DROP_PATTERNS = ['exchange', 'symbol', 'timeframe','open', 'high', 'low', 'close', 'volume', 'regime_trend', 'regime_volatility',
							'regime_momentum', 'regime_label', 'regime_confidence','trend_strength_z', 'vol_percentile', 'volatility_skew',
							'transition_pressure', 'trend_acceleration', 'adaptive_alpha', 'up_vol',
							'down_vol', 'regime_stability', 'directional_persistence', 'score_bull',
							'score_bear', 'score_range', 'score_transition', 'score_high_vol',
							'score_low_vol', 'score_accelerating']


# =============================================================================
# signals related constants
# =============================================================================
WARMUP_PERIOD_DAYS = 20



# =============================================================================
# Regime analysis related constants
# =============================================================================

LIST_EXIT_TYPES = ['SL', 'TP', 'direction_change']

#=================================================================================
# portfolio related constants
#=================================================================================
# Portfolio directory - resolved from the project root
_PROJECT_ROOT = Path(__file__).parent.parent
PORTFOLIO_DIR = os.path.join(_PROJECT_ROOT, "portfolio_objects")
if not os.path.exists(PORTFOLIO_DIR):
    os.makedirs(PORTFOLIO_DIR, exist_ok=True)