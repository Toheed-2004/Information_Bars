# Regime Analysis Module Documentation

This module provides comprehensive analysis of trading performance across different market regimes. Each analysis module examines specific aspects of how regime characteristics affect trade outcomes.

## Overview

The regime analysis pipeline processes a trading ledger (enriched with regime data at entry and exit) and produces statistical insights to help optimize strategy performance across different market conditions.

## Module Index

1. [Performance by Regime Label](#1-performance-by-regime-label)
2. [Regime Transition Matrix](#2-regime-transition-matrix)
3. [Exit Type Breakdown](#3-exit-type-breakdown)
4. [Score Threshold Analysis](#4-score-threshold-analysis)
5. [Trade Duration Analysis](#5-trade-duration-analysis)
6. [Regime Confidence Analysis](#6-regime-confidence-analysis)
7. [Transition Pressure Analysis](#7-transition-pressure-analysis)
8. [Volatility Asymmetry Analysis](#8-volatility-asymmetry-analysis)
9. [Directional Persistence Analysis](#9-directional-persistence-analysis)
10. [Continuous Metric Quartile Analysis](#10-continuous-metric-quartile-analysis)
11. [Rolling Regime Performance](#11-rolling-regime-performance)
12. [Regime Fitness Score](#12-regime-fitness-score)

---

## 1. Performance by Regime Label

**File:** `perfromance_by_regime_label.py`

### What It Calculates

Computes comprehensive performance metrics for trades grouped by:
- Full regime label (e.g., "BULL_HIGH_VOL_ACCELERATING")
- Regime trend only (BULL, BEAR, RANGE)
- Regime volatility only (HIGH_VOL, NORMAL_VOL, LOW_VOL)
- Direction × Trend combinations (e.g., "Long_in_BULL")

### Calculation Method

Uses vectorized NumPy operations for ultra-fast computation:
1. Groups trades by regime characteristics using `np.unique` and `np.bincount`
2. Aggregates metrics across all trades in each group
3. Computes cumulative returns for drawdown calculation

### Output Metrics

For each regime grouping:
- `trade_count`: Number of trades
- `win_rate_pct`: Percentage of profitable trades
- `avg_pnl_per_trade`: Average PnL per trade (%)
- `total_pnl`: Cumulative PnL (%)
- `profit_factor`: Total profits / Total losses
- `avg_trade_duration_days`: Average holding period
- `pct_of_total_trades`: Percentage of all trades
- `max_drawdown_pct`: Maximum peak-to-trough decline

### What It Means

- **High win_rate + profit_factor**: Regime is favorable for the strategy
- **High trade_count**: Strategy is active in this regime
- **Large max_drawdown**: Regime has high risk or strategy mismatch
- **Low pct_of_total_trades**: Rare regime, may need more data

**Use Case:** Identify which regimes are most/least profitable to enable regime-specific filtering or position sizing.

---

## 2. Regime Transition Matrix

**File:** `regime_transition.py`

### What It Calculates

Builds a transition matrix showing performance when trades enter one regime and exit in another.

### Calculation Method

1. Creates a 2D matrix: entry_regime × exit_regime
2. Uses flat indexing (`entry_idx * n_labels + exit_idx`) for vectorized aggregation
3. Computes three matrices: count, avg_pnl, win_rate

### Output Metrics

Three nested dictionaries (entry_regime → exit_regime):
- `count`: Number of trades with this transition
- `avg_pnl_pct`: Average PnL for this transition
- `win_rate_pct`: Win rate for this transition

### What It Means

- **Diagonal elements** (same entry/exit regime): Stable regime trades
- **Off-diagonal elements**: Regime changed during trade
- **High count off-diagonal**: Frequent regime shifts
- **Negative avg_pnl for transitions**: Regime changes hurt performance

**Use Case:** Identify if regime changes during trades are harmful. Consider shorter holding periods or tighter stops if transitions show poor performance.

---

## 3. Exit Type Breakdown

**File:** `exit_regime_breakdown.py`

### What It Calculates

Analyzes how trades exit (TP, SL, direction_change) across different regimes.

### Calculation Method

1. Maps exit actions to indices (0:SL, 1:TP, 2:direction_change)
2. Uses `np.bincount` to aggregate counts per regime
3. Computes percentages and ratios
4. Breaks down by regime × direction for SL rate

### Output Metrics

**Summary (per regime):**
- `SL_pct`: Percentage of trades hitting stop-loss
- `TP_pct`: Percentage hitting take-profit
- `pct_direction_change`: Percentage exiting on direction change
- `TP_SL_ratio`: Ratio of TP to SL exits (higher is better)

**By Direction (per regime × direction):**
- `SL_rate`: Stop-loss rate for Long/Short trades separately

### What It Means

- **High SL_pct**: Regime is choppy or strategy poorly suited
- **High TP_SL_ratio**: Strategy captures moves well in this regime
- **Different SL_rate by direction**: Asymmetric risk (e.g., longs get stopped more than shorts)

**Use Case:** Adjust stop-loss distances or avoid trading certain regimes with high SL rates.

---

## 4. Score Threshold Analysis

**File:** `score_threshold_analysis.py`

### What It Calculates

For each regime score (score_bull, score_bear, etc.), finds the threshold where average PnL becomes positive.

### Calculation Method

1. Bins scores into 10 equal-width deciles (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
2. Computes average PnL per decile
3. Finds first decile with positive avg_pnl
4. Generates filter condition (e.g., "score_bull > 0.6")

### Output Metrics

For each score:
- `decile_bins`: Bin edges [0.0, 0.1, ..., 1.0]
- `avg_pnl_per_decile`: Average PnL in each decile
- `trade_count_per_decile`: Number of trades per decile
- `filter_condition`: Suggested threshold filter

### What It Means

- **Positive avg_pnl at low scores**: Score is not predictive
- **Positive avg_pnl only at high scores**: Use threshold filter
- **"Never positive"**: Score is inversely correlated or useless

**Use Case:** Set minimum score thresholds to filter out low-quality trade signals.

---

## 5. Trade Duration Analysis

**File:** `trade_duration_analysis.py`

### What It Calculates

Analyzes how long trades are held and how duration affects performance across regimes.

### Calculation Method

1. Converts datetime to numeric timestamps
2. Computes duration in hours: `(exit_time - entry_time) / 3600`
3. Bins durations into buckets (<1h, 1-2h, 2-4h, 4-8h, 8-12h, 12-24h, >24h)
4. Aggregates performance by regime, duration bucket, and exit action

### Output Metrics

**By Regime Label:**
- `trade_count`, `avg_duration_hours`, `avg_pnl_pct`, `win_rate_pct`

**Duration Distribution:**
- Count of trades in each duration bucket per regime

**Performance by Duration Bucket:**
- `trade_count`, `avg_pnl_pct` for each regime × duration combination
- Keys formatted as: "REGIME<duration" (e.g., "BULL_HIGH_VOL<4-8h")

**By Exit Action:**
- Average duration and PnL for TP, SL, direction_change exits

**Trending Regime Long Holds:**
- Splits BULL/BEAR trades into long_hold (>median) vs short_hold
- Shows if holding longer in trends improves performance

**Low Vol Quick Trades:**
- Compares quick trades (<1h) vs normal in LOW_VOL regimes

### What It Means

- **Longer duration + higher PnL**: Regime allows trends to develop
- **Quick trades profitable**: Scalping works in this regime
- **Long holds underperform**: Mean reversion or choppy conditions
- **SL exits have short duration**: Getting stopped out quickly

**Use Case:** Optimize holding periods per regime. Use tighter time-based exits in regimes where long holds underperform.

---

## 6. Regime Confidence Analysis

**File:** `regime_confidence_analysis.py`

### What It Calculates

Analyzes trade performance based on regime confidence levels at entry.

### Calculation Method

1. Bins confidence into three buckets:
   - Low: < 0.25
   - Medium: 0.25 - 0.45
   - High: > 0.45
2. Computes performance for each bucket overall, by trend, and by full label

### Output Metrics

**Overall:**
- Performance by confidence bucket (Low, Medium, High)

**By Trend:**
- Performance by confidence × trend (e.g., "High_(>0.45)|BULL")

**By Label:**
- Performance by confidence × full regime label

Each includes: `trade_count`, `avg_pnl_pct`, `win_rate_pct`

### What It Means

- **Low confidence trades underperform**: Regime detection is uncertain
- **High confidence trades outperform**: Regime is clear and stable
- **No difference across buckets**: Confidence metric not useful

**Use Case:** Filter out trades with low regime confidence to improve overall performance.

---

## 7. Transition Pressure Analysis

**File:** `transition_pressure_analysis.py`

### What It Calculates

Analyzes how transition_pressure (a momentum metric measuring regime instability) at entry and exit affects trade outcomes.

**Transition Pressure Formula:**
```
transition_pressure = d_trend_z × vol_expansion_factor
```
Where:
- `d_trend_z` = |trend_strength_z[i] - trend_strength_z[i-1]|
- `vol_expansion_factor` = volatility_level / EWMA(volatility_level)

### Calculation Method

1. Splits trades into quartiles based on entry_transition_pressure
2. Computes performance per quartile
3. Analyzes exit_transition_pressure by exit action
4. Calculates SL rate per entry pressure quartile
5. Examines pressure levels when regimes change during trades

### Output Metrics

**Entry Quartile Performance:**
- Quartile 1-4 (Q1 = lowest pressure, Q4 = highest)
- `trade_count`, `avg_pnl_pct`, `win_rate_pct` per quartile

**Exit Pressure by Action:**
- Average exit pressure for TP, SL, direction_change
- `avg_exit_pressure`, `count`

**SL Rate by Entry Quartile:**
- Stop-loss rate for each entry pressure quartile
- `sl_rate`, `sl_rate_pct`, `trade_count`

**Regime Change Pressure:**
- Average entry/exit pressure when regime changes during trade
- `avg_entry_pressure`, `avg_exit_pressure`, `trade_count`

### What It Means

- **Q4 (high pressure) underperforms**: Entering during unstable regimes is risky
- **High exit pressure for SL**: Stops triggered during volatile transitions
- **High pressure during regime changes**: Transitions are turbulent
- **Q1 (low pressure) outperforms**: Stable regimes are safer

**Use Case:** Avoid entering trades when transition_pressure is high (Q4). Consider tighter stops or smaller positions during high-pressure periods.

---

## 8. Volatility Asymmetry Analysis

**File:** `volatility_asymmetry_analysis.py`

### What It Calculates

Analyzes how asymmetric volatility (up_vol vs down_vol) affects long and short trade performance.

**Volatility Skew:**
```
volatility_skew = up_vol / down_vol
```
- Skew < 1: Downside volatility dominates
- Skew ≈ 1: Symmetric volatility
- Skew > 1: Upside volatility dominates

### Calculation Method

1. Computes volatility_skew ratio
2. Categorizes skew into three buckets (<1, ≈1, >1)
3. Analyzes long/short performance by skew category
4. Quartile analysis for up_vol and down_vol separately
5. Special analysis: high up_vol in BEAR regimes for shorts

### Output Metrics

**Skew Performance (Long/Short):**
- Performance by skew category
- Keys: "Skew_<_1_(downside_dominant)", "Skew_≈_1_(symmetric)", "Skew_>_1_(upside_dominant)"

**Up_vol Quartile Performance:**
- Quartile_1 to Quartile_4 (all trades)

**Down_vol Quartile Performance:**
- Quartile_1 to Quartile_4 (all trades)

**Up_vol High vs Bear:**
- Short trades in BEAR regimes split by up_vol level
- Keys: "Low/Medium_up_vol", "High_up_vol"

Each includes: `trade_count`, `avg_pnl_pct`, `win_rate_pct`

### What It Means

- **Longs perform better with skew > 1**: Upside volatility favors long positions
- **Shorts perform better with skew < 1**: Downside volatility favors short positions
- **High up_vol in BEAR hurts shorts**: Volatile bounces against short positions
- **High down_vol helps shorts**: Accelerated downside moves

**Use Case:** 
- Favor long trades when skew > 1
- Favor short trades when skew < 1
- Avoid shorts in BEAR regimes with high up_vol (choppy downtrends)

---

## 9. Directional Persistence Analysis

**File:** `directional_persistence_analysis.py`

### What It Calculates

Analyzes how directional_persistence (a measure of trend consistency) at entry affects long and short trade performance.

**Directional Persistence:** Measures how consistently price moves in one direction (positive = upward persistence, negative = downward persistence).

### Calculation Method

1. Splits trades into quartiles based on entry_directional_persistence
2. Computes performance per quartile separately for:
   - Long trades
   - Short trades
   - Combined (all trades)

### Output Metrics

Three dictionaries (long, short, combined):
- Quartile 1-4 (Q1 = most negative persistence, Q4 = most positive)
- `trade_count`, `avg_pnl_pct`, `win_rate_pct` per quartile

### What It Means

- **Longs perform better in Q4**: Positive persistence favors long trades
- **Shorts perform better in Q1**: Negative persistence favors short trades
- **Q2/Q3 (neutral persistence)**: Choppy, directionless markets

**Use Case:**
- Enter longs when directional_persistence is high (Q3/Q4)
- Enter shorts when directional_persistence is low/negative (Q1/Q2)
- Avoid trading in neutral persistence regimes (Q2/Q3) if performance is poor

---

## 10. Continuous Metric Quartile Analysis

**File:** `continuous_metric_quartile.py`

### What It Calculates

Generic quartile analysis for any continuous metric at entry (trend_strength_z, vol_percentile, regime_confidence, transition_pressure, regime_stability, directional_persistence, volatility_skew, adaptive_alpha, trend_acceleration).

### Calculation Method

1. For each metric, bins trades into quartiles (Q1-Q4)
2. Uses vectorized `np.bincount` for fast aggregation
3. Computes performance per quartile

### Output Metrics

For each metric:
- Quartile_1 to Quartile_4
- `trade_count`, `avg_pnl_pct`, `win_rate_pct`

### What It Means

Identifies optimal ranges for each metric:
- **Q4 outperforms**: Higher values are better
- **Q1 outperforms**: Lower values are better
- **Q2/Q3 outperform**: Mid-range values are optimal
- **No pattern**: Metric is not predictive

**Use Case:** Set filters to trade only when metrics are in favorable quartiles.

---

## 11. Rolling Regime Performance

**File:** `rolling_regime_performance.py`

### What It Calculates

Computes rolling window performance within each regime to detect performance degradation over time.

### Calculation Method

1. Groups trades by regime label
2. Sorts by exit_datetime
3. Computes rolling average PnL and win rate using a sliding window (default: 20 trades)
4. Uses cumulative sums for O(n) efficiency

### Output Metrics

For each regime:
- Dictionary of datetime → stats
- `rolling_avg_pnl`: Average PnL over last N trades
- `rolling_win_rate_pct`: Win rate over last N trades
- `trade_count_in_window`: Number of trades in window

**Latest Performance:**
- `last_datetime`: Most recent trade
- `rolling_avg_pnl`, `rolling_win_rate_pct`, `trade_count`
- `status`: "healthy", "deteriorating", "poor_win_rate", "critical"

### What It Means

- **Declining rolling_avg_pnl**: Strategy is degrading in this regime
- **Status = "critical"**: Both PnL and win rate below thresholds
- **Status = "healthy"**: Recent performance is good

**Use Case:** Detect regime-specific strategy degradation. Pause trading in regimes with "critical" or "deteriorating" status until performance recovers.

---

## 12. Regime Fitness Score

**File:** `regime_fitness_score.py`

### What It Calculates

Computes a single fitness score (0-1) per regime combining win rate, profit factor, and trade count.

### Calculation Method

```python
win_rate_norm = win_rate_pct / 100.0
profit_factor_norm = min(max(profit_factor, 0.0), 3.0) / 3.0
trade_factor = min(trade_count / 10.0, 1.0)

fitness = (win_rate_norm * 0.3) + (profit_factor_norm * 0.4) + (trade_factor * 0.3)
```

### Output Metrics

Dictionary: regime → fitness_score (0-1, rounded to 4 decimals)

### What It Means

- **Fitness > 0.7**: Excellent regime for strategy
- **Fitness 0.5-0.7**: Good regime
- **Fitness 0.3-0.5**: Marginal regime
- **Fitness < 0.3**: Poor regime, consider avoiding

**Use Case:** Rank regimes by fitness to prioritize trading in high-fitness regimes and avoid low-fitness ones.

---

## Usage Example

```python
from bitpredict.common.regimes_analysis.main import run_regime_analysis

# Run full analysis pipeline
results = run_regime_analysis(enriched_ledger)

# Access specific modules
regime_perf = results['by_regime_label']
transition_matrix = results['transition_matrix']
confidence_analysis = results['confidence_analysis']
pressure_analysis = results['transition_pressure']

# Print results
print_regime_analysis(results)
```

## Key Insights Summary

1. **Regime Selection**: Use fitness scores and performance metrics to identify favorable regimes
2. **Entry Filters**: Apply confidence, transition pressure, and score thresholds
3. **Direction Bias**: Use volatility skew and directional persistence for long/short selection
4. **Exit Optimization**: Adjust stops and targets based on exit type breakdown and duration analysis
5. **Degradation Detection**: Monitor rolling performance to pause trading in deteriorating regimes
6. **Transition Risk**: Avoid high transition pressure periods and understand regime change impacts

---

## Notes

- All numeric outputs are rounded to 4 decimal places
- Vectorized NumPy operations ensure fast computation even on large ledgers
- Missing data is handled with `fillna` parameters (default: 0.0)
- Minimum trade counts prevent statistical noise from small samples
