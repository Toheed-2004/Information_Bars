"""
DollarBar — creates bars when accumulated dollar volume reaches the EMA-adapted target.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

from .base import BaseBar
from common.constants import (
    BAR_FREQUENCY_MULTIPLIER,
    BARS_PER_DAY_MIN,
    BARS_PER_DAY_MAX,
    FREQ_ADJ_BASE,
    FREQ_ADJ_SENSITIVITY,
    DOLLAR_TIER1_BASE,
    DOLLAR_TIER2_BASE,
    DOLLAR_TIER_INFLATION_BASE_YEAR,
    DOLLAR_TIER_INFLATION_RATE,
    DOLLAR_TIER_BASE_BARS,
    DOLLAR_TIER_BASELINE_MULTIPLIERS,
    DOLLAR_MAX_DURATION_CAP,
    DOLLAR_MIN_DURATION_ABS,
    DOLLAR_MIN_DURATION_BASE,
    DURATION_ESTIMATED_MULTIPLIER,
    MAX_DURATION_FLOOR,
    EXTREME_THRESHOLD_MULTIPLIER,
    MIN_DAILY_DATA_FOR_ANALYSIS,
    OPTIMIZATION_CV_TARGET_LOW,
    OPTIMIZATION_CV_TARGET_HIGH,
    OPTIMIZATION_DURATION_CV_TARGET_LOW,
    OPTIMIZATION_DURATION_CV_TARGET_HIGH,
)
from common.logging import get_logger

logger = get_logger(__name__)


class DollarBar(BaseBar):
    """Adaptive dollar-volume bars — bars close when accumulated dollar volume hits the EMA target."""

    TARGET_KEY = "target_dollar_volume"

    # =========================================================================
    # analyze_market_history
    # =========================================================================

    def analyze_market_history(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse historical minute data to set initial dollar-bar parameters."""
        if not historical_data:
            logger.warning("No historical data, using defaults")
            return self._get_default_params()

        closes = np.array([float(r["close"]) for r in historical_data], dtype=float)
        volumes = np.array([float(r["volume"]) for r in historical_data], dtype=float)

        # Daily dollar volumes
        daily_volumes: Dict = {}
        for row in historical_data:
            ts = row.get("datetime") or row.get("timestamp")
            if hasattr(ts, "date"):
                date = ts.date()
            else:
                date = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
            dv = float(row["close"]) * float(row["volume"])
            daily_volumes[date] = daily_volumes.get(date, 0) + dv

        if len(daily_volumes) < MIN_DAILY_DATA_FOR_ANALYSIS:
            return self._get_default_params()

        daily_vols = list(daily_volumes.values())
        daily_arr = np.array(daily_vols, dtype=float)
        median_daily_volume = float(np.median(daily_arr))
        mean_daily_volume = float(np.mean(daily_arr))
        std_daily_volume = float(np.std(daily_arr))
        dollar_volume_cv = std_daily_volume / mean_daily_volume if mean_daily_volume > 0 else 0.3

        asset_tier = self._classify_asset_tier(daily_vols)
        baseline_liquidity = self._calculate_dynamic_baseline(daily_vols, asset_tier)

        # Market analysis
        returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([])
        information_entropy = (
            self._calculate_entropy(returns.tolist()) if len(returns) > 10 else 2.5
        )
        information_factor = max(0.5, min(2.0, information_entropy / 3.0))
        regime_stability = self._calculate_regime_stability(returns, closes) if len(returns) >= 1000 else 0.5
        market_noise = self._calculate_market_noise(returns) if len(returns) >= 50 else 0.3
        market_efficiency = self._calculate_market_efficiency(closes, volumes)

        # Bars per day
        base_bpd = DOLLAR_TIER_BASE_BARS.get(asset_tier, 18.0)
        freq_adj = FREQ_ADJ_BASE + (market_efficiency * FREQ_ADJ_SENSITIVITY)
        target_bars_per_day = max(
            BARS_PER_DAY_MIN,
            min(
                BARS_PER_DAY_MAX,
                base_bpd * information_factor * freq_adj / BAR_FREQUENCY_MULTIPLIER,
            ),
        )

        initial_target_dollar_volume = median_daily_volume / target_bars_per_day

        # Alpha
        alpha_min = max(0.05, 0.12 - (regime_stability * 0.08))
        alpha_max = min(0.35, 0.22 + (market_noise * 0.13))
        ema_alpha = max(alpha_min, min(alpha_max, 0.18 - (dollar_volume_cv * 0.25)))

        # Duration bounds — same formula as volume/volatility (frequency-based)
        minutes_per_day = len(historical_data) / len(daily_volumes)
        activity_factor = (
            float(np.mean(volumes) / np.median(volumes)) if np.median(volumes) > 0 else 1.0
        )
        min_duration_minutes = max(DOLLAR_MIN_DURATION_ABS, int(DOLLAR_MIN_DURATION_BASE / activity_factor))
        estimated_bar_duration = minutes_per_day / target_bars_per_day
        max_duration_minutes = max(MAX_DURATION_FLOOR, min(DOLLAR_MAX_DURATION_CAP, int(estimated_bar_duration * DURATION_ESTIMATED_MULTIPLIER)))

        # BUG-FIX 17: extreme_threshold must be relative to the per-bar target,
        # not the daily volume. Using median_daily_volume made the threshold
        # ~target_bars_per_day times too large (e.g. 20× instead of ~3×), 
        # effectively disabling the extreme-event early-close trigger entirely.
        extreme_threshold = initial_target_dollar_volume * EXTREME_THRESHOLD_MULTIPLIER

        logger.debug(
            "Dollar analysis: target=$%.0f, alpha=%.3f, tier=%s, bars_per_day=%.1f",
            initial_target_dollar_volume,
            ema_alpha,
            asset_tier,
            target_bars_per_day,
        )

        return {
            "target_dollar_volume": initial_target_dollar_volume,
            "ema_alpha": ema_alpha,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "target_bars_per_day": target_bars_per_day,
            "min_duration_minutes": min_duration_minutes,
            "max_duration_minutes": max_duration_minutes,
            "asset_tier": asset_tier,
            "baseline_liquidity": baseline_liquidity,
            "market_efficiency": market_efficiency,
            "regime_stability": regime_stability,
            "market_noise": market_noise,
            "extreme_threshold": extreme_threshold,
            "dollar_volume_cv": dollar_volume_cv,
            "information_entropy": information_entropy,
            "bars_completed": 0,
            "monitoring_counter": 0,
            "bars_since_optimization": 0,
            "target_volume_history": [],
            "optimization_events": [],
        }

    # =========================================================================
    # Type-specific helpers
    # =========================================================================

    def _classify_asset_tier(self, daily_volumes: List[float]) -> str:
        median_dv = float(np.median(daily_volumes))
        current_year = datetime.now().year
        inflation_factor = (
            (current_year - DOLLAR_TIER_INFLATION_BASE_YEAR) * DOLLAR_TIER_INFLATION_RATE + 1.0
        )
        if median_dv >= DOLLAR_TIER1_BASE * inflation_factor:
            return "tier1"
        elif median_dv >= DOLLAR_TIER2_BASE * inflation_factor:
            return "tier2"
        return "tier3"

    def _calculate_dynamic_baseline(self, daily_volumes: List[float], asset_tier: str) -> float:
        median_vol = float(np.median(daily_volumes))
        multiplier = DOLLAR_TIER_BASELINE_MULTIPLIERS.get(asset_tier, 0.01)
        return max(1_000_000, median_vol * multiplier)

    def _calculate_market_efficiency(self, closes: np.ndarray, volumes: np.ndarray) -> float:
        if len(closes) < 100:
            return 0.5
        returns = np.diff(closes) / closes[:-1]
        # Autocorrelation component
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            efficiency_autocorr = max(0.0, 1 - abs(autocorr) * 10)
        else:
            efficiency_autocorr = 0.5
        # Variance ratio component
        if len(returns) >= 4:
            var_1 = np.var(returns)
            var_2 = np.var(returns[::2])
            efficiency_variance = (
                max(0.0, 1 - abs((var_2 / (2 * var_1)) - 1) * 2) if var_1 > 0 else 0.5
            )
        else:
            efficiency_variance = 0.5
        return float(max(0.1, min(0.9, efficiency_autocorr * 0.5 + efficiency_variance * 0.5)))

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def should_create_bar(
        self,
        minute_data: Dict[str, Any],
        current_bar_data: Dict[str, Any],
        market_params: Dict[str, Any],
    ) -> bool:
        """Override base: adds extreme-detection trigger (half min_duration allowed)."""
        if not current_bar_data or not market_params:
            return False
        min_met, max_exceeded, duration_min = self._check_duration_constraints(
            current_bar_data, minute_data, market_params
        )
        accumulated = current_bar_data.get("accumulated_size", 0)
        target = market_params.get(self.TARGET_KEY, 1.0)
        extreme_threshold = market_params.get(
            "extreme_threshold", target * EXTREME_THRESHOLD_MULTIPLIER
        )
        half_min_duration = market_params.get("min_duration_minutes", 5) * 0.5
        extreme_detected = accumulated >= extreme_threshold and duration_min >= half_min_duration
        return (accumulated >= target and min_met) or max_exceeded or extreme_detected

    def get_bar_size_value(self, minute_data: Dict[str, Any]) -> float:
        """
        Dollar volume for one 1-minute OHLCV candle: close_price × bar_volume.

        BUG-FIX 6 (documentation / methodology clarification):
        This is a necessary approximation for minute-bar data. The true dollar
        volume for a minute would be Σ(p_i × q_i) over all individual trades,
        but 1-minute OHLCV data stores only aggregate volume, not individual trades.
        Using close × volume slightly overstates or understates the true dollar
        volume (by the difference between close and VWAP, typically < 0.05%
        per minute for BTC). The tick pipeline uses the exact formula p×q,
        so this difference is real but small in practice.

        Researchers comparing minute-bar vs tick-bar dollar volumes should note
        this approximation in their methodology section.
        """
        if not self._validate_minute_data(minute_data):
            return 0.0
        return max(0.0, float(minute_data["close"]) * float(minute_data["volume"]))

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

        vwap = round(bar_size / volume_val, price_precision) if volume_val > 0 else round(close_val, price_precision)
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
            "bar_size": round(bar_size, 2),
            "vwap": vwap,
            "duration_minutes": duration_minutes,
            "tick_count": bar_data.get("tick_count", 1),
            "bar_return": bar_return,
            "price_range": price_range,
            "close_position": close_position,
        }

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "target_dollar_volume": 25_000_000,
            "ema_alpha": 0.15,
            "alpha_min": 0.08,
            "alpha_max": 0.25,
            "target_bars_per_day": 20,
            "min_duration_minutes": 8,
            "max_duration_minutes": 120,
            "asset_tier": "tier2",
            "baseline_liquidity": 50_000_000,
            "market_efficiency": 0.5,
            "regime_stability": 0.5,
            "market_noise": 0.3,
            "extreme_threshold": 75_000_000,
            "dollar_volume_cv": 0.3,
            "information_entropy": 2.5,
            "bars_completed": 0,
            "monitoring_counter": 0,
            "bars_since_optimization": 0,
            "target_volume_history": [],
            "optimization_events": [],
        }

    def _enforce_type_bounds(self, market_params: Dict[str, Any]) -> Dict[str, Any]:
        market_params = dict(market_params)
        tv = market_params.get("target_dollar_volume", 25_000_000)
        market_params["target_dollar_volume"] = max(100_000, min(1_000_000_000, tv))
        market_params["extreme_threshold"] = (
            market_params["target_dollar_volume"] * EXTREME_THRESHOLD_MULTIPLIER
        )
        return market_params

    def _calculate_bar_quality(
        self, recent_bars: List[Dict[str, Any]], market_params: Dict[str, Any]
    ) -> Dict[str, float]:
        if len(recent_bars) < 20:
            return {"liquidity_consistency": 0.5, "duration_adaptability": 0.5, "overall_quality": 0.5}

        sizes = np.array([float(b.get("bar_size", 0)) for b in recent_bars], dtype=float)
        durations = np.array([float(b.get("duration_minutes", 0)) for b in recent_bars], dtype=float)

        filtered = self._remove_outliers(sizes)
        if len(filtered) > 5 and np.mean(filtered) > 0:
            size_cv = np.std(filtered) / np.mean(filtered)
            if size_cv <= OPTIMIZATION_CV_TARGET_LOW:
                liquidity_consistency = 1.0
            elif size_cv <= OPTIMIZATION_CV_TARGET_HIGH:
                liquidity_consistency = 1.0 - (
                    (size_cv - OPTIMIZATION_CV_TARGET_LOW)
                    / (OPTIMIZATION_CV_TARGET_HIGH - OPTIMIZATION_CV_TARGET_LOW)
                ) * 0.3
            else:
                liquidity_consistency = max(
                    0.0, 0.7 - ((size_cv - OPTIMIZATION_CV_TARGET_HIGH) / 0.6) * 0.7
                )
        else:
            liquidity_consistency = 0.5

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

        overall_quality = liquidity_consistency * 0.6 + duration_adaptability * 0.4
        return {
            "liquidity_consistency": liquidity_consistency,
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
        liquidity = quality.get("liquidity_consistency", 0.5)
        dur_adapt = quality.get("duration_adaptability", 0.5)
        current_target = market_params.get("target_dollar_volume", 25_000_000)

        if overall < 0.5:
            if liquidity < 0.4:
                old_bpd = market_params.get("target_bars_per_day", 20)
                new_bpd = min(BARS_PER_DAY_MAX, old_bpd * 1.1)
                market_params["target_bars_per_day"] = new_bpd
                market_params["target_dollar_volume"] = current_target * (old_bpd / new_bpd)
            if dur_adapt < 0.4:
                old_min = market_params.get("min_duration_minutes", 8)
                market_params["min_duration_minutes"] = min(old_min + 2, 30)

        bars_completed = market_params.get("bars_completed", 0)
        events = list(market_params.get("optimization_events", []))
        events.append({"bar_number": bars_completed, "overall_quality": overall})
        market_params["optimization_events"] = events[-10:]
        return self._enforce_type_bounds(market_params)
