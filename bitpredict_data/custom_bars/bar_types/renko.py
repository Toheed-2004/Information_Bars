"""
RenkoBar — creates bars when the absolute price displacement from the bar's open
level (renko_reference) reaches the EMA-adapted brick size.

Bar direction (bullish/bearish) is determined at close.  The renko_reference is
set to the first minute's open when each new bar starts and is persisted in
market_params["renko_reference"] so it survives state saves/restores.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from bitpredict.data.custom_bars.bar_types.base import BaseBar
from bitpredict.data.custom_bars.constants import (
    SLOW_BAR_FREQUENCY_MULTIPLIER,
    BARS_PER_DAY_MIN,
    BARS_PER_DAY_MAX,
    RENKO_BASE_FREQUENCY,
    DURATION_ESTIMATED_MULTIPLIER,
    MAX_DURATION_FLOOR,
    MAX_DURATION_MINUTES_ABS,
    MIN_DURATION_MINUTES_ABS,
    MIN_DURATION_FRACTION,
    MIN_DAILY_DATA_FOR_ANALYSIS,
    OPTIMIZATION_CV_TARGET_LOW,
    OPTIMIZATION_CV_TARGET_HIGH,
    OPTIMIZATION_DURATION_CV_TARGET_LOW,
    OPTIMIZATION_DURATION_CV_TARGET_HIGH,
)
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


class RenkoBar(BaseBar):
    """
    Adaptive Renko bars.

    accumulated_size = |current_close - renko_reference|  (displacement, NOT summed).
    Bar closes when displacement >= target_brick_size AND min_duration met, or max_duration exceeded.
    """

    TARGET_KEY = "target_brick_size"

    def __init__(self, exchange: str, symbol: str):
        super().__init__(exchange, symbol)
        self.renko_reference: Optional[float] = None  # price level at bar open

    # =========================================================================
    # analyze_market_history
    # =========================================================================

    def analyze_market_history(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse historical minute data to set initial renko brick-size parameters."""
        if not historical_data:
            return self._get_default_params()

        closes  = np.array([float(r["close"])  for r in historical_data], dtype=float)
        highs   = np.array([float(r["high"])   for r in historical_data], dtype=float)
        lows    = np.array([float(r["low"])    for r in historical_data], dtype=float)
        volumes = np.array([float(r["volume"]) for r in historical_data], dtype=float)

        # Daily high-low range — expressed as a FRACTION of the day's median close.
        # Using relative ranges makes the brick target stationary across BTC's full price
        # history ($7K → $100K).  A relative brick of 0.5% works the same whether BTC
        # is at $10K or $100K, because the displacement check is also relative.
        daily_highs:  Dict = {}
        daily_lows:   Dict = {}
        daily_closes: Dict = {}
        for i, row in enumerate(historical_data):
            ts = row.get("datetime") or row.get("timestamp")
            if hasattr(ts, "date"):
                date = ts.date()
            else:
                date = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
            h, l, c = float(highs[i]), float(lows[i]), float(closes[i])
            daily_highs[date]  = max(daily_highs.get(date, h), h)
            daily_lows[date]   = min(daily_lows.get(date, l), l)
            daily_closes.setdefault(date, []).append(c)

        if len(daily_highs) < MIN_DAILY_DATA_FOR_ANALYSIS:
            return self._get_default_params()

        dates = sorted(daily_highs.keys())
        daily_rel_ranges = np.array(
            [
                (daily_highs[d] - daily_lows[d]) / float(np.median(daily_closes[d]))
                if float(np.median(daily_closes[d])) > 0 else 0.0
                for d in dates
            ],
            dtype=float,
        )
        median_daily_range = float(np.median(daily_rel_ranges))  # relative (fraction)
        mean_daily_range   = float(np.mean(daily_rel_ranges))
        std_daily_range    = float(np.std(daily_rel_ranges))
        range_cv = std_daily_range / mean_daily_range if mean_daily_range > 0 else 0.5

        # Information metrics
        minute_returns = np.diff(closes) / closes[:-1]
        minute_returns = minute_returns[~np.isnan(minute_returns)]
        if len(minute_returns) < 100:
            return self._get_default_params()

        return_entropy = self._calculate_entropy(minute_returns.tolist())
        random_entropy = self._calculate_entropy(
            np.random.normal(0, np.std(minute_returns), len(minute_returns)).tolist()
        )
        information_ratio = (
            return_entropy / random_entropy
            if random_entropy > 0 and return_entropy > 0 else 1.0
        )

        minute_activity = volumes[1:] * np.abs(minute_returns) * closes[1:]
        if len(minute_activity) >= 1440:
            daily_activity      = np.convolve(minute_activity, np.ones(1440) / 1440, mode="valid")
            activity_percentile = float(np.mean(daily_activity <= daily_activity[-1]))
        else:
            activity_percentile = 0.5

        information_multiplier = max(0.5, min(2.0, information_ratio))
        activity_multiplier    = 0.5 + activity_percentile

        target_bars_per_day = max(
            BARS_PER_DAY_MIN,
            min(
                BARS_PER_DAY_MAX,
                RENKO_BASE_FREQUENCY * information_multiplier * activity_multiplier / SLOW_BAR_FREQUENCY_MULTIPLIER,
            ),
        )

        # Relative brick size: fraction of price the bar must travel to close.
        initial_target_brick_size = median_daily_range / target_bars_per_day

        # Alpha
        regime_stability = (
            self._calculate_regime_stability(minute_returns, closes)
            if len(minute_returns) >= 1000 else 0.5
        )
        market_noise = self._calculate_market_noise(minute_returns) if len(minute_returns) >= 50 else 0.5
        alpha_min = 0.03 + (market_noise * 0.07)
        alpha_max = 0.20 + (regime_stability * 0.25)
        normalized_cv = min(1.0, range_cv / 0.8)
        ema_alpha = alpha_min + (alpha_max - alpha_min) * normalized_cv

        # Duration bounds
        minutes_per_day        = len(historical_data) / len(daily_highs)
        estimated_bar_duration = minutes_per_day / target_bars_per_day
        min_duration_minutes   = max(MIN_DURATION_MINUTES_ABS, int(minutes_per_day * MIN_DURATION_FRACTION))
        max_duration_minutes   = max(
            MAX_DURATION_FLOOR,
            min(MAX_DURATION_MINUTES_ABS, int(estimated_bar_duration * DURATION_ESTIMATED_MULTIPLIER)),
        )

        self.renko_reference = float(closes[-1]) if len(closes) > 0 else None

        logger.debug(
            "Renko analysis: brick_size=%.4f, alpha=%.3f, bars_per_day=%.1f",
            initial_target_brick_size, ema_alpha, target_bars_per_day,
        )

        return {
            "target_brick_size":     initial_target_brick_size,
            "ema_alpha":             ema_alpha,
            "alpha_min":             alpha_min,
            "alpha_max":             alpha_max,
            "target_bars_per_day":   target_bars_per_day,
            "min_duration_minutes":  min_duration_minutes,
            "max_duration_minutes":  max_duration_minutes,
            "range_cv":              range_cv,
            "median_daily_range":    median_daily_range,
            "regime_stability":      regime_stability,
            "market_noise":          market_noise,
            "information_ratio":     information_ratio,
            "activity_percentile":   activity_percentile,
            "analysis_period_days":  len(daily_highs),
            "bars_completed":        0,
            "monitoring_counter":    0,
            "bars_since_optimization": 0,
            "target_volume_history": [],
            "optimization_events":   [],
            "renko_reference":       self.renko_reference,
        }

    # =========================================================================
    # accumulate_bar_data — override: accumulated_size = displacement, not sum
    # =========================================================================

    def accumulate_bar_data(
        self, current_bar_data: Dict[str, Any], minute_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Override: accumulated_size is the absolute price displacement from renko_reference,
        NOT the running sum.  This means 'should_create_bar' fires when the price has moved
        at least one brick away from the bar's opening level.
        """
        if not self._validate_minute_data(minute_data):
            return current_bar_data or {}

        current_close = float(minute_data["close"])

        if not current_bar_data:
            # New bar — anchor reference at the first minute's open price.
            self.renko_reference = float(minute_data["open"])
            rel_displacement = (
                abs(current_close - self.renko_reference) / self.renko_reference
                if self.renko_reference > 0 else 0.0
            )
            return {
                "datetime_start":   minute_data["datetime"],
                "datetime_end":     minute_data["datetime"],
                "open":             minute_data["open"],
                "high":             minute_data["high"],
                "low":              minute_data["low"],
                "close":            minute_data["close"],
                "volume":           minute_data["volume"],
                "accumulated_size": rel_displacement,  # relative (fraction of reference)
                "tick_count":       1,
            }

        reference = self.renko_reference
        if reference is None:
            # Fallback: use bar open if state was not restored
            reference = float(current_bar_data["open"])
            self.renko_reference = reference

        rel_displacement = (
            abs(current_close - reference) / reference if reference > 0 else 0.0
        )
        return {
            "datetime_start":   current_bar_data["datetime_start"],
            "datetime_end":     minute_data["datetime"],
            "open":             current_bar_data["open"],
            "high":             max(float(current_bar_data["high"]), float(minute_data["high"])),
            "low":              min(float(current_bar_data["low"]),  float(minute_data["low"])),
            "close":            minute_data["close"],
            "volume":           float(current_bar_data["volume"]) + float(minute_data["volume"]),
            "accumulated_size": rel_displacement,  # relative (fraction of reference)
            "tick_count":       current_bar_data.get("tick_count", 0) + 1,
        }

    def get_bar_size_value(self, minute_data: Dict[str, Any]) -> float:
        """Not called directly (accumulate_bar_data is overridden); returns open-close move as proxy."""
        if not self._validate_minute_data(minute_data):
            return 0.0
        return abs(float(minute_data["close"]) - float(minute_data["open"]))

    # =========================================================================
    # update_market_params — skip EMA for time-capped bars
    # =========================================================================

    def update_market_params(
        self, market_params: Dict[str, Any], finalized_bar: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Bidirectional EMA update of target_brick_size with an upward cap.

        See VolumeBar.update_market_params for the full rationale.

        For renko specifically:
          • Organic closes always have actual ≈ target (bar closes the moment displacement
            crosses the threshold), so the EMA is nearly stable on organic bars.
          • Time-capped bars (consolidation periods where price oscillates without making
            a sustained brick-sized move) have actual < target.  The bidirectional update
            lets the brick size gradually shrink toward the achievable displacement level,
            so renko bars can resume closing organically after prolonged consolidation.
          • Large single-candle gap-ups/downs (actual >> target) are capped at 2× target
            to prevent the brick size from jumping to flash-crash levels.
        """
        market_params = dict(market_params)
        market_params["bars_completed"] = market_params.get("bars_completed", 0) + 1

        actual_size = finalized_bar.get("bar_size", 0.0)
        target      = market_params.get(self.TARGET_KEY, 1.0)
        alpha       = market_params.get("ema_alpha", 0.15)

        ema_input = min(actual_size, target * 2.0)
        market_params[self.TARGET_KEY] = (1 - alpha) * target + alpha * ema_input

        return market_params

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def finalize_bar(self, bar_data: Dict[str, Any], market_params: Dict[str, Any]) -> Dict[str, Any]:
        start = bar_data["datetime_start"]
        end   = bar_data["datetime_end"]
        if isinstance(start, str):
            start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if isinstance(end, str):
            end = datetime.fromisoformat(end.replace("Z", "+00:00"))
        duration_minutes = int((end - start).total_seconds() / 60)

        open_val   = float(bar_data["open"])
        high_val   = float(bar_data["high"])
        low_val    = float(bar_data["low"])
        close_val  = float(bar_data["close"])
        volume_val = float(bar_data["volume"])
        price_precision = self._get_optimal_precision(close_val)
        bar_size   = float(bar_data["accumulated_size"])  # displacement from renko_reference

        direction      = "bullish" if close_val >= open_val else "bearish"
        bar_return     = round((close_val - open_val) / open_val, 6) if open_val > 0 else 0.0
        price_range    = round((high_val - low_val) / open_val, 6)   if open_val > 0 else 0.0
        close_position = (
            round((close_val - low_val) / (high_val - low_val), 6)
            if high_val != low_val else 0.5
        )

        return {
            "datetime":         bar_data["datetime_end"],
            "datetime_start":   bar_data["datetime_start"],
            "datetime_end":     bar_data["datetime_end"],
            "open":             round(open_val,  price_precision),
            "high":             round(high_val,  price_precision),
            "low":              round(low_val,   price_precision),
            "close":            round(close_val, price_precision),
            "volume":           round(volume_val, 4),
            "bar_size":         round(bar_size, 6),
            "dollar_volume":    round(volume_val * close_val, 2),
            "direction":        direction,
            "duration_minutes": duration_minutes,
            "tick_count":       bar_data.get("tick_count", 1),
            "bar_return":       bar_return,
            "price_range":      price_range,
            "close_position":   close_position,
        }

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "target_brick_size":     0.01,   # relative fraction; ~1% price displacement
            "ema_alpha":             0.20,
            "alpha_min":             0.06,
            "alpha_max":             0.32,
            "target_bars_per_day":   4,
            "min_duration_minutes":  5,
            "max_duration_minutes":  360,
            "range_cv":              0.3,
            "median_daily_range":    0.03,   # relative daily HL range (~3%)
            "regime_stability":      0.5,
            "market_noise":          0.5,
            "information_ratio":     1.0,
            "activity_percentile":   0.5,
            "analysis_period_days":  0,
            "bars_completed":        0,
            "monitoring_counter":    0,
            "bars_since_optimization": 0,
            "target_volume_history": [],
            "optimization_events":   [],
            "renko_reference":       None,
        }

    def _enforce_type_bounds(self, market_params: Dict[str, Any]) -> Dict[str, Any]:
        market_params = dict(market_params)
        tbs = market_params.get("target_brick_size", 0.01)
        # Relative bounds: floor at 0.01% (extreme precision), cap at 50% (very large brick)
        market_params["target_brick_size"] = max(1e-4, min(0.5, tbs))
        return market_params

    def _calculate_bar_quality(
        self, recent_bars: List[Dict[str, Any]], market_params: Dict[str, Any]
    ) -> Dict[str, float]:
        if len(recent_bars) < 20:
            return {"brick_consistency": 0.5, "duration_adaptability": 0.5, "overall_quality": 0.5}

        sizes     = np.array([float(b.get("bar_size", 0))         for b in recent_bars], dtype=float)
        durations = np.array([float(b.get("duration_minutes", 0)) for b in recent_bars], dtype=float)

        filtered = self._remove_outliers(sizes)
        if len(filtered) > 5 and np.mean(filtered) > 0:
            size_cv = np.std(filtered) / np.mean(filtered)
            if size_cv <= OPTIMIZATION_CV_TARGET_LOW:
                brick_consistency = 1.0
            elif size_cv <= OPTIMIZATION_CV_TARGET_HIGH:
                brick_consistency = 1.0 - (
                    (size_cv - OPTIMIZATION_CV_TARGET_LOW)
                    / (OPTIMIZATION_CV_TARGET_HIGH - OPTIMIZATION_CV_TARGET_LOW)
                ) * 0.3
            else:
                brick_consistency = max(
                    0.0, 0.7 - ((size_cv - OPTIMIZATION_CV_TARGET_HIGH) / 0.6) * 0.7
                )
        else:
            brick_consistency = 0.5

        if len(durations) > 10 and np.std(durations) > 0:
            dur_cv = np.std(durations) / np.mean(durations)
            if OPTIMIZATION_DURATION_CV_TARGET_LOW <= dur_cv <= OPTIMIZATION_DURATION_CV_TARGET_HIGH:
                duration_adaptability = 1.0
            elif dur_cv < OPTIMIZATION_DURATION_CV_TARGET_LOW:
                duration_adaptability = max(0.0, dur_cv / OPTIMIZATION_DURATION_CV_TARGET_LOW)
            else:
                duration_adaptability = max(
                    0.0, 1.0 - (dur_cv - OPTIMIZATION_DURATION_CV_TARGET_HIGH) / 0.8
                )
        else:
            duration_adaptability = 0.5

        return {
            "brick_consistency":     brick_consistency,
            "duration_adaptability": duration_adaptability,
            "overall_quality":       brick_consistency * 0.6 + duration_adaptability * 0.4,
        }

    def _apply_optimization_strategy(
        self,
        market_params: Dict[str, Any],
        quality: Dict[str, float],
        recent_bars: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        market_params  = dict(market_params)
        overall        = quality.get("overall_quality",       0.5)
        brick_c        = quality.get("brick_consistency",     0.5)
        dur_adapt      = quality.get("duration_adaptability", 0.5)
        current_target = market_params.get("target_brick_size", 0.01)

        if overall < 0.5:
            if brick_c < 0.4:
                old_bpd = market_params.get("target_bars_per_day", 20)
                new_bpd = min(BARS_PER_DAY_MAX, old_bpd * 1.1)
                market_params["target_bars_per_day"] = new_bpd
                market_params["target_brick_size"]   = current_target * (old_bpd / new_bpd)
            if dur_adapt < 0.4:
                old_min = market_params.get("min_duration_minutes", 5)
                market_params["min_duration_minutes"] = min(old_min + 1, 15)

        bars_completed = market_params.get("bars_completed", 0)
        events = list(market_params.get("optimization_events", []))
        events.append({"bar_number": bars_completed, "overall_quality": overall})
        market_params["optimization_events"] = events[-10:]
        return self._enforce_type_bounds(market_params)
