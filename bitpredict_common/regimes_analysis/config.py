# ============================================================
# Regimes Analysis — Central Configuration
# All hardcoded values live here. No magic strings anywhere else.
# ============================================================

# ------------------------------------------------------------
# Ledger column names
# ------------------------------------------------------------
LEDGER_PNL_COL = 'trade_return_pct'
LEDGER_ENTRY_DATETIME_COL = 'entry_datetime'
LEDGER_EXIT_DATETIME_COL = 'exit_datetime'
LEDGER_DIRECTION_COL = 'direction'
LEDGER_ACTION_COL = 'action'
LEDGER_STATUS_COL = 'status'
LEDGER_OPEN_STATUS = 'Open'

# ------------------------------------------------------------
# OHLCV / market data column names
# ------------------------------------------------------------
OHLCV_DATETIME_COL = 'datetime'

# ------------------------------------------------------------
# Regime column detection (used in map_regimes_to_trades)
# Columns whose name starts with any of these prefixes OR is in
# the exact set are treated as regime/score columns to map onto trades.
# ------------------------------------------------------------
REGIME_COLUMN_PREFIXES = ('score_', 'regime_')
REGIME_EXACT_COLUMNS = frozenset({
    'trend_strength_z',
    'vol_percentile',
    'volatility_skew',
    'transition_pressure',
    'trend_acceleration',
    'adaptive_alpha',
    'up_vol',
    'down_vol',
    'directional_persistence',
    'regime_stability',
})

# ------------------------------------------------------------
# Specific enriched column names (after entry_/exit_ prefix applied)
# ------------------------------------------------------------
ENTRY_PREFIX = 'entry_'
EXIT_PREFIX = 'exit_'

REGIME_LABEL_COL = 'regime_label'           # composite label e.g. BULL_LOW_VOL_STABLE
REGIME_TREND_COL = 'regime_trend'           # e.g. BULL / BEAR / RANGE
REGIME_VOLATILITY_COL = 'regime_volatility' # e.g. HIGH_VOL / LOW_VOL
REGIME_MOMENTUM_COL = 'regime_momentum'     # e.g. STABLE / UNSTABLE
REGIME_CONFIDENCE_COL = 'regime_confidence'

# Enriched names (entry side) — derived, not to be changed independently
ENTRY_REGIME_LABEL_COL = ENTRY_PREFIX + REGIME_LABEL_COL
ENTRY_REGIME_TREND_COL = ENTRY_PREFIX + REGIME_TREND_COL
ENTRY_REGIME_VOLATILITY_COL = ENTRY_PREFIX + REGIME_VOLATILITY_COL
ENTRY_REGIME_MOMENTUM_COL = ENTRY_PREFIX + REGIME_MOMENTUM_COL
ENTRY_REGIME_CONFIDENCE_COL = ENTRY_PREFIX + REGIME_CONFIDENCE_COL

EXIT_REGIME_LABEL_COL = EXIT_PREFIX + REGIME_LABEL_COL
EXIT_REGIME_TREND_COL = EXIT_PREFIX + REGIME_TREND_COL

# ------------------------------------------------------------
# Exit action categorisation (generic prefix-based)
# Everything is derived at runtime from the actual action values.
# Only the prefixes that define category membership are config.
# ------------------------------------------------------------
EXIT_ACTION_SL_PREFIX = 'SL'
EXIT_ACTION_TP_PREFIX = 'TP'
# Any action not matching either prefix is treated as "Other"

# ------------------------------------------------------------
# Continuous metrics for quartile / threshold analysis
# These are the unprefixed column names (entry_ prefix added at runtime)
# ------------------------------------------------------------
CONTINUOUS_METRICS = [
    'trend_strength_z',
    'vol_percentile',
    'regime_confidence',
    'transition_pressure',
    'regime_stability',
    'directional_persistence',
    'volatility_skew',
    'adaptive_alpha',
    'trend_acceleration',
]

SCORE_COLUMN_PREFIX = 'score_'  # used to auto-detect score columns for threshold analysis

# ------------------------------------------------------------
# Regime confidence buckets
# ------------------------------------------------------------
CONFIDENCE_BUCKET_EDGES = [0.25, 0.45]   # produces 3 buckets: <0.25, 0.25-0.45, >0.45
CONFIDENCE_BUCKET_LABELS = ['Low_(<0.25)', 'Medium_(0.25-0.45)', 'High_(>0.45)']

# ------------------------------------------------------------
# Regime fitness score
# ------------------------------------------------------------
FITNESS_WEIGHT_WIN_RATE = 0.25
FITNESS_WEIGHT_PROFIT_FACTOR = 0.30
FITNESS_WEIGHT_AVG_PNL = 0.25
FITNESS_WEIGHT_TRADE_COUNT = 0.20
FITNESS_PROFIT_FACTOR_CAP = 3.0       # profit factor clipped to [0, this] before normalising
FITNESS_AVG_PNL_CAP = 5.0             # avg_pnl clipped to [0, this] before normalising
FITNESS_MIN_TRADES_RELIABLE = 30      # fewer than this → partial reliability penalty

# ------------------------------------------------------------
# Rolling performance window
# ------------------------------------------------------------
ROLLING_WINDOW_SIZE = 20
ROLLING_MIN_TRADES = 10

# ------------------------------------------------------------
# Statistical significance — minimum trade count before
# metrics are considered meaningful (used as warning flag)
# ------------------------------------------------------------
MIN_TRADES_SIGNIFICANCE = 30
