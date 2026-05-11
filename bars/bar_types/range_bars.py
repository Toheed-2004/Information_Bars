"""
RangeBar — creates bars when the accumulated sum of minute price ranges (high - low)
reaches the EMA-adapted target.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

from .base import BaseBar
from common.constants import (
    SLOW_BAR_FREQUENCY_MULTIPLIER,
    BARS_PER_DAY_MIN,
    BARS_PER_DAY_MAX,
    RANGE_BASE_FREQUENCY,
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
from common.logging import get_logger

logger = get_logger(__name__)


class RangeBar(BaseBar):
    """Adaptive range bars — close when the accumulated sum of minute (high - low) spans hits the EMA target."""

    TARGET_KEY = "target_range"

    # =========================================================================
    # analyze_market_history
    # =========================================================================

    def analyze_market_history(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse historical minute data to set initial range-bar parameters."""
        if not historical_data:
            return self._get_default_params()

        closes  = np.array([float(r["close"])  for r in historical_data], dtype=float)
        highs   = np.array([float(r["high"])   for r in historical_data], dtype=float)
        lows    = np.array([float(r["low"])    for r in historical_data], dtype=float)
        volumes = np.array([float(r["volume"]) for r in historical_data], dtype=float)

        # Vectorised minute relative ranges: (high - low) / close
        # Using relative ranges makes the metric stationary across BTC's price history
        # (7K → 100K), analogous to how volatility bars use %-based close-to-close moves.
        safe_closes   = np.where(closes > 0, closes, np.nan)
        minute_ranges = np.where(closes > 0, (highs - lows) / safe_closes, 0.0)  # shape: (N,)

        # Daily total relative range (sum of per-minute (H-L)/C)
        daily_ranges: Dict = {}
        for i, row in enumerate(historical_data):
            ts = row.get("datetime") or row.get("timestamp")
            if hasattr(ts, "date"):
                date = ts.date()
            else:
                date = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
            daily_ranges[date] = daily_ranges.get(date, 0.0) + float(minute_ranges[i])

        if len(daily_ranges) < MIN_DAILY_DATA_FOR_ANALYSIS:
            return self._get_default_params()

        daily_arr          = np.array(list(daily_ranges.values()), dtype=float)
        median_daily_range = float(np.median(daily_arr))
        mean_daily_range   = float(np.mean(daily_arr))
        std_daily_range    = float(np.std(daily_arr))
        range_cv = std_daily_range / mean_daily_range if mean_daily_range > 0 else 0.5

        # Information-based frequency multiplier
        minute_returns = np.diff(closes) / closes[:-1]
        minute_returns = minute_returns[~np.isnan(minute_returns)]
        if len(minute_returns) < 100:
            return self._get_default_params()

        return_entropy = self._calculate_entropy(minute_returns.tolist())
        # BUG-FIX 15: Fixed seed for reproducible calibration
        _rng_calib = np.random.default_rng(seed=42)
        random_entropy = self._calculate_entropy(
            _rng_calib.normal(0, np.std(minute_returns), len(minute_returns)).tolist()
        )
        information_ratio = (
            return_entropy / random_entropy
            if random_entropy > 0 and return_entropy > 0 else 1.0
        )

        minute_activity = volumes[1:] * np.abs(minute_returns) * closes[1:]
        if len(minute_activity) >= 1440:
            daily_activity     = np.convolve(minute_activity, np.ones(1440) / 1440, mode="valid")
            activity_percentile = float(np.mean(daily_activity <= daily_activity[-1]))
        else:
            activity_percentile = 0.5

        information_multiplier = max(0.5, min(2.0, information_ratio))
        activity_multiplier    = 0.5 + activity_percentile

        target_bars_per_day = max(
            BARS_PER_DAY_MIN,
            min(
                BARS_PER_DAY_MAX,
                RANGE_BASE_FREQUENCY * information_multiplier * activity_multiplier / SLOW_BAR_FREQUENCY_MULTIPLIER,
            ),
        )

        initial_target_range = median_daily_range / target_bars_per_day

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
        minutes_per_day        = len(historical_data) / len(daily_ranges)
        estimated_bar_duration = minutes_per_day / target_bars_per_day
        min_duration_minutes   = max(MIN_DURATION_MINUTES_ABS, int(minutes_per_day * MIN_DURATION_FRACTION))
        max_duration_minutes   = max(
            MAX_DURATION_FLOOR,
            min(MAX_DURATION_MINUTES_ABS, int(estimated_bar_duration * DURATION_ESTIMATED_MULTIPLIER)),
        )

        logger.debug(
            "Range analysis: target=%.4f, alpha=%.3f, bars_per_day=%.1f",
            initial_target_range, ema_alpha, target_bars_per_day,
        )

        return {
            "target_range":          initial_target_range,
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
            "analysis_period_days":  len(daily_ranges),
            "bars_completed":        0,
            "monitoring_counter":    0,
            "bars_since_optimization": 0,
            "target_volume_history": [],
            "optimization_events":   [],
        }

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def get_bar_size_value(self, minute_data: Dict[str, Any]) -> float:
        """
        Relative minute price range: (high - low) / close.

        Using a relative (percentage-based) range instead of an absolute dollar range
        makes the target stationary across different price regimes and asset price levels.
        """
        if not self._validate_minute_data(minute_data):
            return 0.0
        h = float(minute_data["high"])
        l = float(minute_data["low"])
        c = float(minute_data["close"])
        if c <= 0:
            return 0.0
        return max(0.0, (h - l) / c)

    # =========================================================================
    # update_market_params — skip EMA for time-capped bars
    # =========================================================================

    def update_market_params(
        self, market_params: Dict[str, Any], finalized_bar: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Bidirectional EMA update of target_range with an upward cap.

        See VolumeBar.update_market_params for the full rationale.  The same upward-
        ratchet / EMA-skip problem applies here: high-volatility candles (large H-L/C)
        inflate the target and the skip prevents downward correction.
        """
        market_params = dict(market_params)
        market_params["bars_completed"] = market_params.get("bars_completed", 0) + 1

        actual_size = finalized_bar.get("bar_size", 0.0)
        target      = market_params.get(self.TARGET_KEY, 1.0)
        alpha       = market_params.get("ema_alpha", 0.15)

        ema_input = min(actual_size, target * 2.0)
        market_params[self.TARGET_KEY] = (1 - alpha) * target + alpha * ema_input

        return market_params

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
        bar_size   = float(bar_data["accumulated_size"])  # total accumulated range

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
            "duration_minutes": duration_minutes,
            "tick_count":       bar_data.get("tick_count", 1),
            "bar_return":       bar_return,
            "price_range":      price_range,
            "close_position":   close_position,
        }

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "target_range":          0.003,   # relative (H-L)/C; ~0.3% per bar
            "ema_alpha":             0.20,
            "alpha_min":             0.06,
            "alpha_max":             0.32,
            "target_bars_per_day":   20,
            "min_duration_minutes":  5,
            "max_duration_minutes":  120,
            "range_cv":              0.3,
            "median_daily_range":    0.1,     # relative daily sum of (H-L)/C
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
        }

    def _enforce_type_bounds(self, market_params: Dict[str, Any]) -> Dict[str, Any]:
        market_params = dict(market_params)
        tr = market_params.get("target_range", 0.003)
        # Relative range: floor at 1e-6, cap at 0.5 (50% intra-bar relative range)
        market_params["target_range"] = max(1e-6, min(0.5, tr))
        return market_params

    def _calculate_bar_quality(
        self, recent_bars: List[Dict[str, Any]], market_params: Dict[str, Any]
    ) -> Dict[str, float]:
        if len(recent_bars) < 20:
            return {"range_consistency": 0.5, "duration_adaptability": 0.5, "overall_quality": 0.5}

        sizes     = np.array([float(b.get("bar_size", 0))         for b in recent_bars], dtype=float)
        durations = np.array([float(b.get("duration_minutes", 0)) for b in recent_bars], dtype=float)

        filtered = self._remove_outliers(sizes)
        if len(filtered) > 5 and np.mean(filtered) > 0:
            size_cv = np.std(filtered) / np.mean(filtered)
            if size_cv <= OPTIMIZATION_CV_TARGET_LOW:
                range_consistency = 1.0
            elif size_cv <= OPTIMIZATION_CV_TARGET_HIGH:
                range_consistency = 1.0 - (
                    (size_cv - OPTIMIZATION_CV_TARGET_LOW)
                    / (OPTIMIZATION_CV_TARGET_HIGH - OPTIMIZATION_CV_TARGET_LOW)
                ) * 0.3
            else:
                range_consistency = max(
                    0.0, 0.7 - ((size_cv - OPTIMIZATION_CV_TARGET_HIGH) / 0.6) * 0.7
                )
        else:
            range_consistency = 0.5

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
            "range_consistency":     range_consistency,
            "duration_adaptability": duration_adaptability,
            "overall_quality":       range_consistency * 0.6 + duration_adaptability * 0.4,
        }

    def _apply_optimization_strategy(
        self,
        market_params: Dict[str, Any],
        quality: Dict[str, float],
        recent_bars: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        market_params = dict(market_params)
        overall   = quality.get("overall_quality",     0.5)
        range_c   = quality.get("range_consistency",   0.5)
        dur_adapt = quality.get("duration_adaptability", 0.5)
        current_target = market_params.get("target_range", 0.003)

        if overall < 0.5:
            if range_c < 0.4:
                old_bpd = market_params.get("target_bars_per_day", 20)
                new_bpd = min(BARS_PER_DAY_MAX, old_bpd * 1.1)
                market_params["target_bars_per_day"] = new_bpd
                market_params["target_range"] = current_target * (old_bpd / new_bpd)
            if dur_adapt < 0.4:
                old_min = market_params.get("min_duration_minutes", 5)
                market_params["min_duration_minutes"] = min(old_min + 1, 15)

        bars_completed = market_params.get("bars_completed", 0)
        events = list(market_params.get("optimization_events", []))
        events.append({"bar_number": bars_completed, "overall_quality": overall})
        market_params["optimization_events"] = events[-10:]
        return self._enforce_type_bounds(market_params)
