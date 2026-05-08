# Regime Engine Output Columns

## Layer 1 — Hard Labels

| Column | Values | Meaning |
|---|---|---|
| `regime_trend` | `BULL / BEAR / RANGE / TRANSITION / INSUFFICIENT_DATA` | Directional bias of price trend |
| `regime_volatility` | `HIGH_VOL / NORMAL_VOL / LOW_VOL / INSUFFICIENT_DATA` | Current volatility regime relative to history |
| `regime_momentum` | `ACCELERATING / STABLE / INSUFFICIENT_DATA` | Whether trend momentum is accelerating |
| `regime_label` | Combined string e.g. `BULL_HIGH_VOL_ACCELERATING` | All three dimensions joined — the main regime label |
| `regime_confidence` | `0.0 – 1.0` | How far the current readings are from regime boundaries (higher = more confident) |

## Layer 2 — Continuous Features

| Column | Meaning |
|---|---|
| `trend_strength_z` | Z-score of (fast EWMA − slow EWMA). Positive = bullish, negative = bearish, magnitude = strength |
| `vol_percentile` | Where current volatility sits in the historical ring buffer (0=lowest ever seen, 1=highest). Drives the vol label |
| `volatility_skew` | Ratio of upside vol to downside vol. >1 means more upside movement, <1 means more downside |
| `transition_pressure` | Combines vol expansion and trend acceleration. High values trigger TRANSITION regime |
| `trend_acceleration` | Rate of change of `trend_strength_z` — how fast the trend is gaining/losing strength |
| `adaptive_alpha` | The EWMA smoothing factor being used right now. Higher during high-vol (faster adaptation) |
| `up_vol` | EWMA of upward price moves (semi-deviation) |
| `down_vol` | EWMA of downward price moves (semi-deviation) |
| `regime_stability` | 0–1 score of how many consecutive bars the current `regime_label` has held (1 = very stable) |
| `directional_persistence` | EWMA of signed trend direction — measures whether trend has been consistently up or down |

## Layer 3 — Soft Scores (sigmoid, 0–1)

| Column | Meaning |
|---|---|
| `score_bull` | Probability-like score for bullish trend |
| `score_bear` | Probability-like score for bearish trend |
| `score_range` | Probability-like score for ranging/sideways |
| `score_transition` | Probability-like score for TRANSITION (high stress) |
| `score_high_vol` | Probability-like score for high volatility regime |
| `score_low_vol` | Probability-like score for low volatility regime |
| `score_accelerating` | Probability-like score for accelerating momentum |

> Soft scores are independent sigmoids — they do not sum to 1.
> They can be used directly as continuous features in a model instead of relying on hard label buckets.
