"""
VolatilityBar — creates bars when accumulated close-to-close % volatility reaches the EMA target.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from bitpredict.data.custom_bars.bar_types.base import BaseBar
from bitpredict.data.custom_bars.constants import (
    SLOW_BAR_FREQUENCY_MULTIPLIER,
    BARS_PER_DAY_MIN,
    BARS_PER_DAY_MAX,
    VOLATILITY_BASE_FREQUENCY,
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


class VolatilityBar(BaseBar):
    """Adaptive volatility bars — bars close when accumulated close-to-close % volatility hits the EMA target."""

    TARGET_KEY = "target_volatility"

    def __init__(self, exchange: str, symbol: str):
        super().__init__(exchange, symbol)
        self.previous_close: Optional[float] = None

    # =========================================================================
    # analyze_market_history
    # =========================================================================

    def analyze_market_history(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse historical minute data to set initial volatility-bar parameters."""
        if not historical_data:
            return self._get_default_params()

        closes = np.array([float(r["close"]) for r in historical_data], dtype=float)
        volumes = np.array([float(r["volume"]) for r in historical_data], dtype=float)
        highs = np.array([float(r["high"]) for r in historical_data], dtype=float)
        lows = np.array([float(r["low"]) for r in historical_data], dtype=float)

        minute_returns = np.diff(closes) / closes[:-1]
        minute_returns = minute_returns[~np.isnan(minute_returns)]

        if len(minute_returns) < 100:
            return self._get_default_params()

        # Information-based bars per day
        return_entropy = self._calculate_entropy(minute_returns.tolist())
        random_returns = np.random.normal(0, np.std(minute_returns), len(minute_returns))
        random_entropy = self._calculate_entropy(random_returns.tolist())
        information_ratio = (
            return_entropy / random_entropy
            if random_entropy > 0 and return_entropy > 0
            else 1.0
        )

        minute_activity = volumes[1:] * np.abs(minute_returns) * closes[1:]
        if len(minute_activity) >= 1440:
            daily_activity = np.convolve(minute_activity, np.ones(1440) / 1440, mode="valid")
            activity_percentile = float(np.mean(daily_activity <= daily_activity[-1]))
        else:
            activity_percentile = 0.5

        information_multiplier = max(0.5, min(2.0, information_ratio))
        activity_multiplier = 0.5 + activity_percentile
        target_bars_per_day = max(
            BARS_PER_DAY_MIN,
            min(
                BARS_PER_DAY_MAX,
                VOLATILITY_BASE_FREQUENCY
                * information_multiplier
                * activity_multiplier
                / SLOW_BAR_FREQUENCY_MULTIPLIER,
            ),
        )

        # Daily close-to-close volatility
        daily_volatilities: Dict = {}
        previous_close: Optional[float] = None
        for row in historical_data:
            ts = row.get("datetime") or row.get("timestamp")
            if hasattr(ts, "date"):
                date = ts.date()
            else:
                date = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
            current_close = float(row["close"])
            if date not in daily_volatilities:
                daily_volatilities[date] = 0.0
            if previous_close is not None and previous_close > 0:
                daily_volatilities[date] += abs(current_close - previous_close) / previous_close
            previous_close = current_close

        if len(daily_volatilities) < MIN_DAILY_DATA_FOR_ANALYSIS:
            return self._get_default_params()

        daily_vols = list(daily_volatilities.values())
        daily_arr = np.array(daily_vols, dtype=float)
        median_daily_volatility = float(np.median(daily_arr))
        mean_daily_volatility = float(np.mean(daily_arr))
        std_daily_volatility = float(np.std(daily_arr))
        volatility_cv = (
            std_daily_volatility / mean_daily_volatility if mean_daily_volatility > 0 else 0.5
        )

        initial_target_volatility = median_daily_volatility / target_bars_per_day

        # Alpha
        regime_stability = (
            self._calculate_regime_stability(minute_returns, closes)
            if len(minute_returns) >= 1000
            else 0.5
        )
        market_noise = self._calculate_market_noise(minute_returns) if len(minute_returns) >= 50 else 0.5
        alpha_min = 0.03 + (market_noise * 0.07)
        alpha_max = 0.20 + (regime_stability * 0.25)
        normalized_cv = min(1.0, volatility_cv / 0.8)
        ema_alpha = alpha_min + (alpha_max - alpha_min) * normalized_cv

        # Duration bounds
        minutes_per_day = len(historical_data) / len(daily_volatilities)
        estimated_bar_duration = minutes_per_day / target_bars_per_day
        min_duration_minutes = max(MIN_DURATION_MINUTES_ABS, int(minutes_per_day * MIN_DURATION_FRACTION))
        max_duration_minutes = max(MAX_DURATION_FLOOR, min(MAX_DURATION_MINUTES_ABS, int(estimated_bar_duration * DURATION_ESTIMATED_MULTIPLIER)))

        # Sync previous_close instance var
        self.previous_close = float(closes[-1]) if len(closes) > 0 else None

        logger.debug(
            "Volatility analysis: target=%.4f, alpha=%.3f, bars_per_day=%.1f",
            initial_target_volatility,
            ema_alpha,
            target_bars_per_day,
        )

        return {
            "target_volatility": initial_target_volatility,
            "ema_alpha": ema_alpha,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "target_bars_per_day": target_bars_per_day,
            "min_duration_minutes": min_duration_minutes,
            "max_duration_minutes": max_duration_minutes,
            "volatility_cv": volatility_cv,
            "median_daily_volatility": median_daily_volatility,
            "regime_stability": regime_stability,
            "market_noise": market_noise,
            "information_ratio": information_ratio,
            "activity_percentile": activity_percentile,
            "analysis_period_days": len(daily_volatilities),
            "bars_completed": 0,
            "monitoring_counter": 0,
            "bars_since_optimization": 0,
            "target_volume_history": [],
            "optimization_events": [],
            "previous_close": self.previous_close,
        }

    # =========================================================================
    # update_market_params — skip EMA for time-capped bars
    # =========================================================================

    def update_market_params(
        self, market_params: Dict[str, Any], finalized_bar: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Bidirectional EMA update of target_volatility with an upward cap.

        See VolumeBar.update_market_params for the full rationale.  The same upward-
        ratchet / EMA-skip problem applies here: spike volatility events inflate the
        target and the skip prevents it from ever coming back down.
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

    def get_bar_size_value(self, minute_data: Dict[str, Any]) -> float:
        """Calculate close-to-close percentage volatility for this minute."""
        if not self._validate_minute_data(minute_data):
            return 0.0
        current_close = float(minute_data["close"])
        if self.previous_close is None:
            self.previous_close = current_close
            return 0.0
        volatility = (
            abs(current_close - self.previous_close) / self.previous_close
            if self.previous_close > 0
            else 0.0
        )
        self.previous_close = current_close
        return volatility

    def finalize_bar(self, bar_data: Dict[str, Any], market_params: Dict[str, Any]) -> Dict[str, Any]:
        start = bar_data["datetime_start"]
        end = bar_data["datetime_end"]
        if isinstance(start, str):
            start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if isinstance(end, str):
            end = datetime.fromisoformat(end.replace("Z", "+00:00"))
        duration_minutes = int((end - start).total_seconds() / 60)

        open_val = float(bar_data["open"])
        high_val = float(bar_data["high"])
        low_val = float(bar_data["low"])
        close_val = float(bar_data["close"])
        volume_val = float(bar_data["volume"])
        price_precision = self._get_optimal_precision(close_val)
        bar_size = float(bar_data["accumulated_size"])

        bar_return = round((close_val - open_val) / open_val, 6) if open_val > 0 else 0.0
        price_range = round((high_val - low_val) / open_val, 6) if open_val > 0 else 0.0
        close_position = (
            round((close_val - low_val) / (high_val - low_val), 6)
            if high_val != low_val else 0.5
        )

        return {
            "datetime": bar_data["datetime_end"],
            "datetime_start": bar_data["datetime_start"],
            "datetime_end": bar_data["datetime_end"],
            "open": round(open_val, price_precision),
            "high": round(high_val, price_precision),
            "low": round(low_val, price_precision),
            "close": round(close_val, price_precision),
            "volume": round(volume_val, 4),
            "bar_size": round(bar_size, 6),
            "dollar_volume": round(volume_val * close_val, 2),
            "duration_minutes": duration_minutes,
            "tick_count": bar_data.get("tick_count", 1),
            "bar_return": bar_return,
            "price_range": price_range,
            "close_position": close_position,
        }

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "target_volatility": 0.02,
            "ema_alpha": 0.20,
            "alpha_min": 0.06,
            "alpha_max": 0.32,
            "target_bars_per_day": 20,
            "min_duration_minutes": 5,
            "max_duration_minutes": 120,
            "volatility_cv": 0.3,
            "median_daily_volatility": 0.4,
            "regime_stability": 0.5,
            "market_noise": 0.5,
            "information_ratio": 1.0,
            "activity_percentile": 0.5,
            "analysis_period_days": 0,
            "bars_completed": 0,
            "monitoring_counter": 0,
            "bars_since_optimization": 0,
            "target_volume_history": [],
            "optimization_events": [],
            "previous_close": None,
        }

    def _enforce_type_bounds(self, market_params: Dict[str, Any]) -> Dict[str, Any]:
        market_params = dict(market_params)
        tv = market_params.get("target_volatility", 0.02)
        market_params["target_volatility"] = max(0.0001, min(0.20, tv))
        return market_params

    def _calculate_bar_quality(
        self, recent_bars: List[Dict[str, Any]], market_params: Dict[str, Any]
    ) -> Dict[str, float]:
        if len(recent_bars) < 20:
            return {
                "information_consistency": 0.5,
                "duration_adaptability": 0.5,
                "overall_quality": 0.5,
            }

        sizes = np.array([float(b.get("bar_size", 0)) for b in recent_bars], dtype=float)
        durations = np.array([float(b.get("duration_minutes", 0)) for b in recent_bars], dtype=float)

        filtered = self._remove_outliers(sizes)
        if len(filtered) > 5 and np.mean(filtered) > 0:
            size_cv = np.std(filtered) / np.mean(filtered)
            if size_cv <= OPTIMIZATION_CV_TARGET_LOW:
                information_consistency = 1.0
            elif size_cv <= OPTIMIZATION_CV_TARGET_HIGH:
                information_consistency = 1.0 - (
                    (size_cv - OPTIMIZATION_CV_TARGET_LOW)
                    / (OPTIMIZATION_CV_TARGET_HIGH - OPTIMIZATION_CV_TARGET_LOW)
                ) * 0.3
            else:
                information_consistency = max(
                    0.0, 0.7 - ((size_cv - OPTIMIZATION_CV_TARGET_HIGH) / 0.6) * 0.7
                )
        else:
            information_consistency = 0.5

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

        overall_quality = information_consistency * 0.6 + duration_adaptability * 0.4
        return {
            "information_consistency": information_consistency,
            "duration_adaptability": duration_adaptability,
            "overall_quality": overall_quality,
        }

    def _apply_optimization_strategy(
        self,
        market_params: Dict[str, Any],
        quality: Dict[str, float],
        recent_bars: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        market_params = dict(market_params)
        overall = quality.get("overall_quality", 0.5)
        info_consistency = quality.get("information_consistency", 0.5)
        dur_adapt = quality.get("duration_adaptability", 0.5)
        current_target = market_params.get("target_volatility", 0.02)

        if overall < 0.5:
            if info_consistency < 0.4:
                old_bpd = market_params.get("target_bars_per_day", 20)
                new_bpd = min(BARS_PER_DAY_MAX, old_bpd * 1.1)
                market_params["target_bars_per_day"] = new_bpd
                market_params["target_volatility"] = current_target * (old_bpd / new_bpd)

        bars_completed = market_params.get("bars_completed", 0)
        events = list(market_params.get("optimization_events", []))
        events.append({"bar_number": bars_completed, "overall_quality": overall})
        market_params["optimization_events"] = events[-10:]
        return self._enforce_type_bounds(market_params)
