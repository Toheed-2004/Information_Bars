"""
Bar module constants — all hardcoded values in one place.
"""

# ============================================================================
# Analysis
# ============================================================================
ANALYSIS_LOOKBACK_DAYS = 14          # Days of minute data used for analyze_market_history
MIN_HISTORICAL_DATA_MINUTES = 2000   # Minimum minutes required for full market analysis
MIN_DAILY_DATA_FOR_ANALYSIS = 3      # Minimum days required for analysis

# ============================================================================
# Bar frequency  — PRIMARY TUNING KNOBS
# ============================================================================
BAR_FREQUENCY_MULTIPLIER = 5.0       # Divisor for dollar + hybrid bars.
                                      # > 1.0 = fewer, longer bars.

# Dollar and hybrid bars use BAR_FREQUENCY_MULTIPLIER (5.0) because their base
# frequencies were tuned to produce ~12–14 bars/day at that divisor.
#
# Volume, volatility, range, and renko bars use SLOW_BAR_FREQUENCY_MULTIPLIER
# because their base frequencies (4.0–8.7) are designed to produce 4–9 bars/day
# with no extra divisor.  Applying the 5× divisor would floor all of them at the
# BARS_PER_DAY_MIN=2 floor (e.g. 4.4/5.0 = 0.88 → 2), which makes targets
# impossible to achieve organically and causes perpetual time-capping.
SLOW_BAR_FREQUENCY_MULTIPLIER = 1.0  # Divisor for volume, volatility, range, renko.

BARS_PER_DAY_MIN = 2                  # Hard floor: no bar type will target fewer bars/day than this.
BARS_PER_DAY_MAX = 100               # Hard ceiling: no bar type will target more bars/day than this.

# ============================================================================
# Frequency adjustment factor (dollar + volume bars)
#   freq_adj = FREQ_ADJ_BASE + (efficiency_metric * FREQ_ADJ_SENSITIVITY)
#   Ranges from FREQ_ADJ_BASE (low efficiency) to FREQ_ADJ_BASE + FREQ_ADJ_SENSITIVITY (high).
#   Higher base → more bars for the same BAR_FREQUENCY_MULTIPLIER.
# ============================================================================
FREQ_ADJ_BASE = 0.7
FREQ_ADJ_SENSITIVITY = 0.6

# ============================================================================
# Dollar bar — base target bars/day by asset tier (before BAR_FREQUENCY_MULTIPLIER)
# ============================================================================
# DOLLAR_TIER_BASE_BARS = {"tier1": 25.0, "tier2": 20.0, "tier3": 15.0}
DOLLAR_TIER_BASE_BARS  = {"tier1": 4.2,  "tier2": 3.4,  "tier3": 2.5}

# ============================================================================
# Volume bar — base target bars/day by asset tier (before BAR_FREQUENCY_MULTIPLIER)
# ============================================================================
# VOLUME_TIER_BASE_BARS = {"tier1": 25.0, "tier2": 20.0, "tier3": 15.0}
VOLUME_TIER_BASE_BARS  = {"tier1": 8.7,  "tier2": 6.9,  "tier3": 5.2}

# VOLUME_TIER_BASE_BARS_DEFAULT = 18.0  # Fallback when tier is unrecognised
VOLUME_TIER_BASE_BARS_DEFAULT = 5.2

# ============================================================================
# Volatility bar — base target bars/day (before BAR_FREQUENCY_MULTIPLIER)
#   target_bpd = VOLATILITY_BASE_FREQUENCY * information_multiplier
#                * activity_multiplier / BAR_FREQUENCY_MULTIPLIER
# ============================================================================
# VOLATILITY_BASE_FREQUENCY = 8        # Lower = fewer, longer volatility bars.
VOLATILITY_BASE_FREQUENCY = 4.4        # Lower = fewer, longer volatility bars.

# ============================================================================
# Range bar — base target bars/day (before BAR_FREQUENCY_MULTIPLIER)
#   Same information/activity formula as volatility bar.
# ============================================================================
RANGE_BASE_FREQUENCY = 4.4             # Lower = fewer, longer range bars.

# ============================================================================
# Renko bar — base target bars/day (before BAR_FREQUENCY_MULTIPLIER)
#   Brick size = median_daily_high_low_range / target_bars_per_day
# ============================================================================
RENKO_BASE_FREQUENCY = 6.0             # Lower = larger bricks (fewer bars).

# ============================================================================
# Hybrid bar — dollar-volume + volatility composite
#   Closes when BOTH dollar-volume AND volatility targets are hit.
#   Because AND is harder to satisfy than OR, each individual signal must
#   target a higher frequency so the combined AND rate stays at a useful level.
#   Uses SLOW_BAR_FREQUENCY_MULTIPLIER (1.0) so the base frequency is not
#   divided away.  With DV/vol correlation ~0.5–0.7 in crypto, the effective
#   AND rate is roughly HYBRID_BASE_FREQUENCY × correlation ≈ 4–6 bars/day.
# ============================================================================
HYBRID_BASE_FREQUENCY = 8.0

# ============================================================================
# Duration bounds — apply to ALL bar types
# ============================================================================
MIN_DURATION_MINUTES_ABS = 2         # Absolute minimum min_duration_minutes (enforced in base)
# Hard ceiling for all bar types (except dollar, which uses DOLLAR_MAX_DURATION_CAP).
# Must be large enough that the 3× estimated-bar-duration formula (below) is not always
# clamped for slow bar types.  For a bar targeting 3 bars/day the estimated duration is
# ~480 min; 3×480 = 1440 min = 24 h, so 1440 is a natural upper bound that keeps each
# bar within a single calendar day while still giving outlier bars room to breathe.
MAX_DURATION_MINUTES_ABS = 1440      # Absolute maximum max_duration_minutes (24 hours)

# ---- Dollar bar duration formula ----
# Same formula as volume/volatility: max(MAX_DURATION_FLOOR,
#                                        min(DOLLAR_MAX_DURATION_CAP,
#                                            int(estimated_bar_duration * DURATION_ESTIMATED_MULTIPLIER)))
DOLLAR_MAX_DURATION_CAP = 480        # Hard ceiling on dollar bar max_duration (minutes)

# min_duration = max(DOLLAR_MIN_DURATION_ABS, int(DOLLAR_MIN_DURATION_BASE / activity_factor))
DOLLAR_MIN_DURATION_ABS = 3          # Hard floor on dollar bar min_duration (minutes)
DOLLAR_MIN_DURATION_BASE = 12        # Numerator in the dollar min_duration formula (minutes)

# ---- Volume + Volatility + Range + Renko + Hybrid bar duration formula ----
# max_duration = max(MAX_DURATION_FLOOR,
#                    min(MAX_DURATION_MINUTES_ABS,
#                        int(estimated_bar_duration * DURATION_ESTIMATED_MULTIPLIER)))
#
# The multiplier defines how many times the expected bar duration an outlier bar is
# allowed to run before being force-closed.  3× is a natural "3-sigma" style threshold:
# a genuinely extreme bar can take up to 3× the typical duration before we give up.
# (The previous value of 8 always exceeded MAX_DURATION_MINUTES_ABS=480 for slow bars,
# making max_duration permanently stuck at 480 regardless of the actual bar frequency.)
DURATION_ESTIMATED_MULTIPLIER = 3    # 3× estimated bar duration = outlier cutoff
MAX_DURATION_FLOOR = 60              # Lower bound on computed max_duration (minutes)

# min_duration = max(MIN_DURATION_MINUTES_ABS, int(minutes_per_day * MIN_DURATION_FRACTION))
MIN_DURATION_FRACTION = 0.002        # Fraction of minutes/day used as min_duration

# Hybrid bar uses estimated bar duration (not minutes/day) to set min_duration.
# This prevents trivially short bars when both targets are very sensitive.
# min_duration = max(MIN_DURATION_MINUTES_ABS, int(estimated_bar_duration * HYBRID_MIN_DURATION_FRACTION))
HYBRID_MIN_DURATION_FRACTION = 0.05  # 5% of estimated bar duration (hybrid only)

# ============================================================================
# Extreme detection
# ============================================================================
EXTREME_THRESHOLD_MULTIPLIER = 3.0        # Dollar bars: extreme_threshold = target * this
VOLUME_EXTREME_THRESHOLD_MULTIPLIER = 5.0 # Volume bars: extreme_threshold = target * this

# ============================================================================
# EMA scheduling
# ============================================================================
EMA_MONITORING_INTERVAL = 50         # Self-monitoring every N bars
EMA_OPTIMIZATION_INTERVAL = 200      # Auto-optimization check every N bars
EMA_OPTIMIZATION_COOLDOWN = 400      # Min bars between optimizations
MIN_BARS_FOR_QUALITY = 50            # Min bars needed for quality analysis
QUALITY_HISTORY_LENGTH = 20          # Rolling quality history length

# ============================================================================
# Regime
# ============================================================================
REGIME_CONTEXT_BARS = 50            # Historical bars for regime context

# ============================================================================
# Alpha / parameter bounds
# ============================================================================
ALPHA_MIN_ABSOLUTE = 0.03
ALPHA_MAX_ABSOLUTE = 0.40

# ============================================================================
# Dollar bar tier thresholds
# ============================================================================
DOLLAR_TIER1_BASE = 500_000_000
DOLLAR_TIER2_BASE = 50_000_000
DOLLAR_TIER_INFLATION_BASE_YEAR = 2020
DOLLAR_TIER_INFLATION_RATE = 0.1
DOLLAR_TIER_BASELINE_MULTIPLIERS = {"tier1": 0.005, "tier2": 0.008, "tier3": 0.012}

# ============================================================================
# Optimization targets
# ============================================================================
OPTIMIZATION_CV_TARGET_LOW = 0.15
OPTIMIZATION_CV_TARGET_HIGH = 0.40
OPTIMIZATION_DURATION_CV_TARGET_LOW = 0.3
OPTIMIZATION_DURATION_CV_TARGET_HIGH = 0.6
