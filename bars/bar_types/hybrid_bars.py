"""
HybridBar — creates bars when EITHER accumulated dollar volume OR accumulated
close-to-close volatility first reaches its respective EMA-adapted target.

Primary EMA key: "target_dollar_volume"  (bar_size = dollar volume).
Secondary EMA key: "target_volatility"   (updated in update_market_params override).

accumulated_volatility is tracked inside current_bar_data so it persists in
the saved state between runs.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseBar
from common.constants import (
    SLOW_BAR_FREQUENCY_MULTIPLIER,
    BARS_PER_DAY_MIN,
    BARS_PER_DAY_MAX,
    FREQ_ADJ_BASE,
    FREQ_ADJ_SENSITIVITY,
    HYBRID_BASE_FREQUENCY,
    DURATION_ESTIMATED_MULTIPLIER,
    MAX_DURATION_FLOOR,
    MAX_DURATION_MINUTES_ABS,
    MIN_DURATION_MINUTES_ABS,
    HYBRID_MIN_DURATION_FRACTION,
    MIN_DAILY_DATA_FOR_ANALYSIS,
    OPTIMIZATION_CV_TARGET_LOW,
    OPTIMIZATION_CV_TARGET_HIGH,
    OPTIMIZATION_DURATION_CV_TARGET_LOW,
    OPTIMIZATION_DURATION_CV_TARGET_HIGH,
)
from common.logging import get_logger

logger = get_logger(__name__)


class HybridBar(BaseBar):
    """
    Adaptive hybrid bars.

    Closes when (dollar_volume >= target_dollar_volume AND volatility >= target_volatility)
    AND min_duration met, or max_duration exceeded.

    Using AND (not OR) ensures every bar contains both meaningful volume AND meaningful
    price movement.  OR bars could close on a volume spike with near-zero price move
    (or vice versa), producing a bimodal return distribution with extreme kurtosis and
    low entropy.  AND bars are more uniform and better suited for ML training.

    Both targets adapt independently via EMA.
    """

    TARGET_KEY = "target_dollar_volume"

    def __init__(self, exchange: str, symbol: str):
        super().__init__(exchange, symbol)
        self.previous_close: Optional[float] = None  # for per-minute volatility calculation

    # =========================================================================
    # analyze_market_history
    # =========================================================================

    def analyze_market_history(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse historical minute data to set initial hybrid-bar parameters."""
        if not historical_data:
            return self._get_default_params()

        closes  = np.array([float(r["close"])  for r in historical_data], dtype=float)
        volumes = np.array([float(r["volume"]) for r in historical_data], dtype=float)

        # Daily dollar volume + daily close-to-close volatility (both vectorised)
        dollar_volumes = closes * volumes                       # per-minute dollar volume
        close_returns  = np.abs(np.diff(closes) / closes[:-1]) # per-minute |return|, shape (N-1,)

        daily_dv:  Dict = {}
        daily_vol: Dict = {}
        for i, row in enumerate(historical_data):
            ts = row.get("datetime") or row.get("timestamp")
            if hasattr(ts, "date"):
                date = ts.date()
            else:
                date = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
            daily_dv[date] = daily_dv.get(date, 0.0) + float(dollar_volumes[i])
            if i > 0:  # close_returns is offset by 1
                daily_vol[date] = daily_vol.get(date, 0.0) + float(close_returns[i - 1])
            else:
                daily_vol.setdefault(date, 0.0)

        if len(daily_dv) < MIN_DAILY_DATA_FOR_ANALYSIS:
            return self._get_default_params()

        dv_arr  = np.array(list(daily_dv.values()),  dtype=float)
        vol_arr = np.array(list(daily_vol.values()), dtype=float)

        median_daily_dv  = float(np.median(dv_arr))
        mean_daily_dv    = float(np.mean(dv_arr))
        std_daily_dv     = float(np.std(dv_arr))
        dv_cv            = std_daily_dv / mean_daily_dv if mean_daily_dv > 0 else 0.3
        median_daily_vol = float(np.median(vol_arr))

        # Information metrics
        minute_returns = np.diff(closes) / closes[:-1]
        minute_returns = minute_returns[~np.isnan(minute_returns)]
        if len(minute_returns) < 100:
            return self._get_default_params()

        return_entropy = self._calculate_entropy(minute_returns.tolist())
        random_entropy = self._calculate_entropy(
            np.random.normal(0, np.std(minute_returns), len(minute_returns)).tolist()
        )
        information_ratio  = (
            return_entropy / random_entropy
            if random_entropy > 0 and return_entropy > 0 else 1.0
        )
        information_factor = max(0.5, min(2.0, information_ratio))

        minute_activity = volumes[1:] * np.abs(minute_returns) * closes[1:]
        if len(minute_activity) >= 1440:
            daily_activity      = np.convolve(minute_activity, np.ones(1440) / 1440, mode="valid")
            activity_percentile = float(np.mean(daily_activity <= daily_activity[-1]))
        else:
            activity_percentile = 0.5

        freq_adj = FREQ_ADJ_BASE + (activity_percentile * FREQ_ADJ_SENSITIVITY)

        target_bars_per_day = max(
            BARS_PER_DAY_MIN,
            min(
                BARS_PER_DAY_MAX,
                HYBRID_BASE_FREQUENCY * information_factor * freq_adj / SLOW_BAR_FREQUENCY_MULTIPLIER,
            ),
        )

        initial_target_dv  = median_daily_dv  / target_bars_per_day
        initial_target_vol = median_daily_vol / target_bars_per_day

        # Alpha
        regime_stability = (
            self._calculate_regime_stability(minute_returns, closes)
            if len(minute_returns) >= 1000 else 0.5
        )
        market_noise = self._calculate_market_noise(minute_returns) if len(minute_returns) >= 50 else 0.5
        alpha_min = max(0.05, 0.12 - (regime_stability * 0.08))
        alpha_max = min(0.35, 0.22 + (market_noise * 0.13))
        normalized_cv = min(1.0, dv_cv / 0.8)
        ema_alpha = alpha_min + (alpha_max - alpha_min) * normalized_cv

        # Duration bounds
        minutes_per_day        = len(historical_data) / len(daily_dv)
        estimated_bar_duration = minutes_per_day / target_bars_per_day
        min_duration_minutes   = max(MIN_DURATION_MINUTES_ABS, int(estimated_bar_duration * HYBRID_MIN_DURATION_FRACTION))
        max_duration_minutes   = max(
            MAX_DURATION_FLOOR,
            min(MAX_DURATION_MINUTES_ABS, int(estimated_bar_duration * DURATION_ESTIMATED_MULTIPLIER)),
        )

        self.previous_close = float(closes[-1]) if len(closes) > 0 else None

        logger.debug(
            "Hybrid analysis: target_dv=$%.0f, target_vol=%.4f, alpha=%.3f, bars_per_day=%.1f",
            initial_target_dv, initial_target_vol, ema_alpha, target_bars_per_day,
        )

        return {
            "target_dollar_volume":    initial_target_dv,
            "target_volatility":       initial_target_vol,
            "ema_alpha":               ema_alpha,
            "alpha_min":               alpha_min,
            "alpha_max":               alpha_max,
            "target_bars_per_day":     target_bars_per_day,
            "min_duration_minutes":    min_duration_minutes,
            "max_duration_minutes":    max_duration_minutes,
            "dv_cv":                   dv_cv,
            "median_daily_dv":         median_daily_dv,
            "median_daily_volatility": median_daily_vol,
            "regime_stability":        regime_stability,
            "market_noise":            market_noise,
            "information_ratio":       information_ratio,
            "activity_percentile":     activity_percentile,
            "analysis_period_days":    len(daily_dv),
            "bars_completed":          0,
            "monitoring_counter":      0,
            "bars_since_optimization": 0,
            "target_volume_history":   [],
            "optimization_events":     [],
            "previous_close":          self.previous_close,
        }

    # =========================================================================
    # accumulate_bar_data — extend base with per-minute volatility tracking
    # =========================================================================

    def accumulate_bar_data(
        self, current_bar_data: Dict[str, Any], minute_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calls base accumulation (dollar volume via get_bar_size_value), then appends accumulated_volatility."""
        if not self._validate_minute_data(minute_data):
            return current_bar_data or {}

        current_close = float(minute_data["close"])

        # Close-to-close minute volatility
        if self.previous_close is not None and self.previous_close > 0:
            minute_vol = abs(current_close - self.previous_close) / self.previous_close
        else:
            minute_vol = 0.0
        self.previous_close = current_close

        result = super().accumulate_bar_data(current_bar_data, minute_data)

        prev_vol = current_bar_data.get("accumulated_volatility", 0.0) if current_bar_data else 0.0
        result["accumulated_volatility"] = prev_vol + minute_vol
        return result

    def get_bar_size_value(self, minute_data: Dict[str, Any]) -> float:
        """Dollar volume for the primary (dollar) EMA target."""
        if not self._validate_minute_data(minute_data):
            return 0.0
        return max(0.0, float(minute_data["close"]) * float(minute_data["volume"]))

    # =========================================================================
    # should_create_bar — fires on EITHER signal
    # =========================================================================

    def should_create_bar(
        self,
        minute_data: Dict[str, Any],
        current_bar_data: Dict[str, Any],
        market_params: Dict[str, Any],
    ) -> bool:
        """Close when dollar-volume AND volatility thresholds are both met (+ min_duration), or max exceeded."""
        if not current_bar_data or not market_params:
            return False

        min_met, max_exceeded, _ = self._check_duration_constraints(
            current_bar_data, minute_data, market_params
        )

        dv_met  = current_bar_data.get("accumulated_size",       0) >= market_params.get("target_dollar_volume", 1.0)
        vol_met = current_bar_data.get("accumulated_volatility", 0) >= market_params.get("target_volatility",    1.0)

        return ((dv_met and vol_met) and min_met) or max_exceeded

    # =========================================================================
    # update_market_params — EMA for both targets
    # =========================================================================

    def update_market_params(
        self, market_params: Dict[str, Any], finalized_bar: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Selective EMA update: only adapt the target whose threshold was actually triggered.

        Updating both EMAs unconditionally causes a feedback spiral — if the volatility
        trigger fires first, the dollar-volume EMA sees a below-target dollar-volume value
        and ratchets the target down, which makes dollar-volume bars fire sooner on the
        next candle, and so on.  We prevent this by only updating each EMA when the
        corresponding signal was what actually crossed its threshold.

        A bar is also considered time-capped if neither signal reached its threshold
        (max_duration forced closure); in that case neither EMA is updated.
        """
        market_params = dict(market_params)
        market_params["bars_completed"] = market_params.get("bars_completed", 0) + 1

        alpha      = market_params.get("ema_alpha", 0.15)
        dv_target  = market_params.get("target_dollar_volume", 1.0)
        vol_target = market_params.get("target_volatility", 0.001)
        bar_dv     = finalized_bar.get("bar_size", 0.0)
        bar_vol    = finalized_bar.get("bar_volatility", 0.0)

        dv_triggered  = bar_dv  >= dv_target  * 0.99
        vol_triggered = bar_vol >= vol_target * 0.99

        if dv_triggered:
            dv_input = min(bar_dv, dv_target * 2.0)   # cap upward spikes
            market_params["target_dollar_volume"] = (1 - alpha) * dv_target  + alpha * dv_input
        if vol_triggered:
            vol_input = min(bar_vol, vol_target * 2.0)  # cap upward spikes
            market_params["target_volatility"]    = (1 - alpha) * vol_target + alpha * vol_input

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
        bar_size        = float(bar_data["accumulated_size"])   # dollar volume
        bar_volatility  = round(float(bar_data.get("accumulated_volatility", 0.0)), 8)

        vwap           = round(bar_size / volume_val, price_precision) if volume_val > 0 else round(close_val, price_precision)
        bar_return     = round((close_val - open_val) / open_val, 6) if open_val > 0 else 0.0
        price_range    = round((high_val - low_val)  / open_val, 6) if open_val > 0 else 0.0
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
            "bar_size":         round(bar_size, 2),   # dollar volume
            "vwap":             vwap,
            "bar_volatility":   bar_volatility,
            "duration_minutes": duration_minutes,
            "tick_count":       bar_data.get("tick_count", 1),
            "bar_return":       bar_return,
            "price_range":      price_range,
            "close_position":   close_position,
        }

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "target_dollar_volume":    25_000_000,
            "target_volatility":       0.02,
            "ema_alpha":               0.15,
            "alpha_min":               0.08,
            "alpha_max":               0.25,
            "target_bars_per_day":     20,
            "min_duration_minutes":    8,
            "max_duration_minutes":    120,
            "dv_cv":                   0.3,
            "median_daily_dv":         500_000_000,
            "median_daily_volatility": 0.4,
            "regime_stability":        0.5,
            "market_noise":            0.3,
            "information_ratio":       1.0,
            "activity_percentile":     0.5,
            "analysis_period_days":    0,
            "bars_completed":          0,
            "monitoring_counter":      0,
            "bars_since_optimization": 0,
            "target_volume_history":   [],
            "optimization_events":     [],
            "previous_close":          None,
        }

    def _enforce_type_bounds(self, market_params: Dict[str, Any]) -> Dict[str, Any]:
        market_params = dict(market_params)
        tdv = market_params.get("target_dollar_volume", 25_000_000)
        market_params["target_dollar_volume"] = max(100_000, min(1_000_000_000, tdv))
        tv  = market_params.get("target_volatility", 0.02)
        market_params["target_volatility"] = max(1e-6, min(0.50, tv))
        return market_params

    def _calculate_bar_quality(
        self, recent_bars: List[Dict[str, Any]], market_params: Dict[str, Any]
    ) -> Dict[str, float]:
        if len(recent_bars) < 20:
            return {"dv_consistency": 0.5, "duration_adaptability": 0.5, "overall_quality": 0.5}

        sizes     = np.array([float(b.get("bar_size", 0))         for b in recent_bars], dtype=float)
        durations = np.array([float(b.get("duration_minutes", 0)) for b in recent_bars], dtype=float)

        filtered = self._remove_outliers(sizes)
        if len(filtered) > 5 and np.mean(filtered) > 0:
            size_cv = np.std(filtered) / np.mean(filtered)
            if size_cv <= OPTIMIZATION_CV_TARGET_LOW:
                dv_consistency = 1.0
            elif size_cv <= OPTIMIZATION_CV_TARGET_HIGH:
                dv_consistency = 1.0 - (
                    (size_cv - OPTIMIZATION_CV_TARGET_LOW)
                    / (OPTIMIZATION_CV_TARGET_HIGH - OPTIMIZATION_CV_TARGET_LOW)
                ) * 0.3
            else:
                dv_consistency = max(
                    0.0, 0.7 - ((size_cv - OPTIMIZATION_CV_TARGET_HIGH) / 0.6) * 0.7
                )
        else:
            dv_consistency = 0.5

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
            "dv_consistency":        dv_consistency,
            "duration_adaptability": duration_adaptability,
            "overall_quality":       dv_consistency * 0.6 + duration_adaptability * 0.4,
        }

    def _apply_optimization_strategy(
        self,
        market_params: Dict[str, Any],
        quality: Dict[str, float],
        recent_bars: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        market_params = dict(market_params)
        overall   = quality.get("overall_quality",       0.5)
        dv_c      = quality.get("dv_consistency",        0.5)
        dur_adapt = quality.get("duration_adaptability", 0.5)
        current_dv  = market_params.get("target_dollar_volume", 25_000_000)
        current_vol = market_params.get("target_volatility",    0.02)

        if overall < 0.5:
            if dv_c < 0.4:
                old_bpd = market_params.get("target_bars_per_day", 20)
                new_bpd = min(BARS_PER_DAY_MAX, old_bpd * 1.1)
                market_params["target_bars_per_day"]  = new_bpd
                # Scale both targets proportionally
                market_params["target_dollar_volume"] = current_dv  * (old_bpd / new_bpd)
                market_params["target_volatility"]    = current_vol * (old_bpd / new_bpd)
            if dur_adapt < 0.4:
                old_min = market_params.get("min_duration_minutes", 8)
                market_params["min_duration_minutes"] = min(old_min + 2, 30)

        bars_completed = market_params.get("bars_completed", 0)
        events = list(market_params.get("optimization_events", []))
        events.append({"bar_number": bars_completed, "overall_quality": overall})
        market_params["optimization_events"] = events[-10:]
        return self._enforce_type_bounds(market_params)
