"""
VolumeBar — creates bars when accumulated volume reaches the EMA-adapted target.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from bitpredict.data.custom_bars.bar_types.base import BaseBar
from bitpredict.data.custom_bars.constants import (
    SLOW_BAR_FREQUENCY_MULTIPLIER,
    BARS_PER_DAY_MIN,
    BARS_PER_DAY_MAX,
    FREQ_ADJ_BASE,
    FREQ_ADJ_SENSITIVITY,
    VOLUME_TIER_BASE_BARS,
    VOLUME_TIER_BASE_BARS_DEFAULT,
    VOLUME_EXTREME_THRESHOLD_MULTIPLIER,
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


class VolumeBar(BaseBar):
    """Adaptive volume bars — bars close when accumulated physical volume hits the EMA target."""

    TARGET_KEY = "target_volume"

    # =========================================================================
    # analyze_market_history
    # =========================================================================

    def analyze_market_history(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse historical minute data to set initial volume-bar parameters."""
        if not historical_data:
            logger.warning("No historical data, using defaults")
            return self._get_default_params()

        # Group by date
        daily_volumes: Dict = {}
        all_volumes: List[float] = []
        closes: List[float] = []

        for row in historical_data:
            ts = row.get("datetime") or row.get("timestamp")
            if hasattr(ts, "date"):
                date = ts.date()
            else:
                date = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
            vol = float(row["volume"])
            all_volumes.append(vol)
            closes.append(float(row["close"]))
            daily_volumes[date] = daily_volumes.get(date, 0) + vol

        if len(daily_volumes) < MIN_DAILY_DATA_FOR_ANALYSIS:
            logger.warning("Insufficient daily data, using defaults")
            return self._get_default_params()

        daily_vols = list(daily_volumes.values())
        daily_arr = np.array(daily_vols, dtype=float)
        median_daily_volume = float(np.median(daily_arr))
        mean_daily_volume = float(np.mean(daily_arr))
        std_daily_volume = float(np.std(daily_arr))
        volume_cv = std_daily_volume / mean_daily_volume if mean_daily_volume > 0 else 0.5

        asset_tier = self._classify_volume_tier(daily_vols)
        trading_activity_efficiency = self._calculate_trading_activity_efficiency(all_volumes)

        # Bars per day
        base_bpd = VOLUME_TIER_BASE_BARS.get(asset_tier, VOLUME_TIER_BASE_BARS_DEFAULT)
        activity_factor = FREQ_ADJ_BASE + (trading_activity_efficiency * FREQ_ADJ_SENSITIVITY)
        target_bars_per_day = max(
            BARS_PER_DAY_MIN,
            min(BARS_PER_DAY_MAX, base_bpd * activity_factor / SLOW_BAR_FREQUENCY_MULTIPLIER),
        )

        initial_target_volume = median_daily_volume / target_bars_per_day

        # Alpha
        closes_arr = np.array(closes, dtype=float)
        returns = np.diff(closes_arr) / closes_arr[:-1] if len(closes_arr) > 1 else np.array([])
        regime_stability = (
            self._calculate_regime_stability(returns, closes_arr) if len(returns) >= 1000 else 0.5
        )
        market_noise = self._calculate_market_noise(returns) if len(returns) >= 50 else 0.5

        alpha_min = max(0.03, 0.12 - (regime_stability * 0.08))
        alpha_max = min(0.40, 0.22 + (market_noise * 0.13))
        normalized_cv = min(1.0, volume_cv / 0.8)
        ema_alpha = alpha_min + (alpha_max - alpha_min) * normalized_cv

        # Duration bounds
        minutes_per_day = len(historical_data) / len(daily_volumes)
        estimated_bar_duration = minutes_per_day / target_bars_per_day
        min_duration_minutes = max(MIN_DURATION_MINUTES_ABS, int(minutes_per_day * MIN_DURATION_FRACTION))
        max_duration_minutes = max(MAX_DURATION_FLOOR, min(MAX_DURATION_MINUTES_ABS, int(estimated_bar_duration * DURATION_ESTIMATED_MULTIPLIER)))

        extreme_threshold = initial_target_volume * VOLUME_EXTREME_THRESHOLD_MULTIPLIER

        logger.debug(
            "Volume analysis: target=%.0f, alpha=%.3f, tier=%s, bars_per_day=%.1f",
            initial_target_volume,
            ema_alpha,
            asset_tier,
            target_bars_per_day,
        )

        return {
            "target_volume": initial_target_volume,
            "ema_alpha": ema_alpha,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "target_bars_per_day": target_bars_per_day,
            "min_duration_minutes": min_duration_minutes,
            "max_duration_minutes": max_duration_minutes,
            "asset_tier": asset_tier,
            "trading_activity_efficiency": trading_activity_efficiency,
            "volume_cv": volume_cv,
            "extreme_threshold": extreme_threshold,
            "median_daily_volume": median_daily_volume,
            "analysis_period_days": len(daily_volumes),
            "bars_completed": 0,
            "monitoring_counter": 0,
            "bars_since_optimization": 0,
            "target_volume_history": [],
            "optimization_events": [],
        }

    # =========================================================================
    # Type-specific helpers
    # =========================================================================

    def _classify_volume_tier(self, daily_volumes: List[float]) -> str:
        """Classify asset by volume tier."""
        arr = np.array(daily_volumes, dtype=float)
        median_vol = float(np.median(arr))
        std_vol = float(np.std(arr))
        if std_vol > 0 and median_vol > 0:
            cv = std_vol / median_vol
            tier1_threshold = median_vol * (1.0 + cv * 2.0)
            tier2_threshold = median_vol * (0.1 + cv * 0.5)
            if median_vol >= tier1_threshold * 0.8:
                return "tier1"
            elif median_vol >= tier2_threshold * 2.0:
                return "tier2"
        return "tier3"

    def _calculate_trading_activity_efficiency(self, volumes: List[float]) -> float:
        """Efficiency via volume entropy (0–1)."""
        non_zero = [v for v in volumes if v > 0]
        if len(non_zero) < 10:
            return 0.1
        entropy = self._calculate_entropy(non_zero)
        return float(max(0.0, min(1.0, entropy / 8.0)))

    # =========================================================================
    # update_market_params — skip EMA for time-capped bars
    # =========================================================================

    def update_market_params(
        self, market_params: Dict[str, Any], finalized_bar: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Bidirectional EMA update of target_volume with an upward cap.

        Skipping time-capped bars caused an upward ratchet: spike events (single minutes
        with 5-10× normal volume) ratchet the target up, then the skip prevents downward
        correction — leaving the target permanently inflated and almost all subsequent
        bars hitting max_duration.

        Fix: always update the EMA, but cap the input at 2× the current target.
          • Organic bars  (actual ≈ target):  EMA stays near target — stable.
          • Time-capped   (actual < target):   EMA adapts down slowly — correct.
          • Spike bars    (actual >> target):  capped at 2× target so at most doubles
                                               per event rather than jumping 5–10×.
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
        if not self._validate_minute_data(minute_data):
            return 0.0
        return max(0.0, float(minute_data["volume"]))

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
            "bar_size": round(bar_size, 4),
            "dollar_volume": round(volume_val * close_val, 2),
            "duration_minutes": duration_minutes,
            "tick_count": bar_data.get("tick_count", 1),
            "bar_return": bar_return,
            "price_range": price_range,
            "close_position": close_position,
        }

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "target_volume": 50_000,
            "ema_alpha": 0.15,
            "alpha_min": 0.05,
            "alpha_max": 0.30,
            "target_bars_per_day": 20,
            "min_duration_minutes": 5,
            "max_duration_minutes": 120,
            "asset_tier": "tier2",
            "trading_activity_efficiency": 0.5,
            "volume_cv": 0.3,
            "extreme_threshold": 250_000,
            "median_daily_volume": 1_250_000,
            "analysis_period_days": 0,
            "bars_completed": 0,
            "monitoring_counter": 0,
            "bars_since_optimization": 0,
            "target_volume_history": [],
            "optimization_events": [],
        }

    def _enforce_type_bounds(self, market_params: Dict[str, Any]) -> Dict[str, Any]:
        market_params = dict(market_params)
        tv = market_params.get("target_volume", 50_000)
        market_params["target_volume"] = max(1_000, min(50_000_000, tv))
        market_params["extreme_threshold"] = max(5_000, market_params.get("extreme_threshold", tv * 5))
        return market_params

    def _calculate_bar_quality(
        self, recent_bars: List[Dict[str, Any]], market_params: Dict[str, Any]
    ) -> Dict[str, float]:
        if len(recent_bars) < 20:
            return {"consistency": 0.5, "duration_adaptability": 0.5, "overall_quality": 0.5}

        sizes = np.array([float(b.get("bar_size", 0)) for b in recent_bars], dtype=float)
        durations = np.array([float(b.get("duration_minutes", 0)) for b in recent_bars], dtype=float)

        filtered = self._remove_outliers(sizes)
        if len(filtered) > 5 and np.mean(filtered) > 0:
            size_cv = np.std(filtered) / np.mean(filtered)
            if size_cv <= OPTIMIZATION_CV_TARGET_LOW:
                consistency = 1.0
            elif size_cv <= OPTIMIZATION_CV_TARGET_HIGH:
                consistency = 1.0 - (
                    (size_cv - OPTIMIZATION_CV_TARGET_LOW)
                    / (OPTIMIZATION_CV_TARGET_HIGH - OPTIMIZATION_CV_TARGET_LOW)
                ) * 0.3
            else:
                consistency = max(0.0, 0.7 - ((size_cv - OPTIMIZATION_CV_TARGET_HIGH) / 0.6) * 0.7)
        else:
            consistency = 0.5

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

        overall_quality = consistency * 0.6 + duration_adaptability * 0.4
        return {
            "consistency": consistency,
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
        consistency = quality.get("consistency", 0.5)
        dur_adapt = quality.get("duration_adaptability", 0.5)

        current_target = market_params.get("target_volume", 50_000)

        if overall < 0.5:
            if consistency < 0.4:
                market_params["target_volume"] = min(current_target * 1.15, current_target + 10_000)
            if dur_adapt < 0.4:
                old_min = market_params.get("min_duration_minutes", 5)
                market_params["min_duration_minutes"] = min(old_min + 1, 15)

        bars_completed = market_params.get("bars_completed", 0)
        events = list(market_params.get("optimization_events", []))
        events.append({"bar_number": bars_completed, "overall_quality": overall})
        market_params["optimization_events"] = events[-10:]
        return self._enforce_type_bounds(market_params)
