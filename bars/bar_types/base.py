"""
BaseBar — abstract base class for all bar types.

Concrete (shared) methods live here; type-specific logic is abstract.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from common.logging import get_logger
from common.constants import (
    ALPHA_MIN_ABSOLUTE,
    ALPHA_MAX_ABSOLUTE,
    BARS_PER_DAY_MIN,
    BARS_PER_DAY_MAX,
    EMA_OPTIMIZATION_COOLDOWN,
    MIN_DURATION_MINUTES_ABS,
    MAX_DURATION_MINUTES_ABS,
    QUALITY_HISTORY_LENGTH,
)

logger = get_logger(__name__)


class BaseBar(ABC):
    """Abstract base class for all bar types."""

    # Subclass must define — e.g. 'target_volume', 'target_dollar_volume', 'target_volatility'
    TARGET_KEY: str

    # Subclass can override
    EXTREME_PRICE_MOVE_THRESHOLD: float = 0.8

    def __init__(self, exchange: str, symbol: str):
        self.exchange = exchange
        self.symbol = symbol

    # =========================================================================
    # ABSTRACT — subclass must implement
    # =========================================================================

    @abstractmethod
    def analyze_market_history(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse historical minute data and return initial market_params."""

    @abstractmethod
    def get_bar_size_value(self, minute_data: Dict[str, Any]) -> float:
        """Return the value that accumulates toward the bar threshold."""

    @abstractmethod
    def finalize_bar(self, bar_data: Dict[str, Any], market_params: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dict ready for DB insert from accumulated bar_data."""

    @abstractmethod
    def _get_default_params(self) -> Dict[str, Any]:
        """Conservative fallback market_params."""

    @abstractmethod
    def _enforce_type_bounds(self, market_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply bar-type-specific parameter bounds. Can be a no-op."""

    @abstractmethod
    def _calculate_bar_quality(
        self, recent_bars: List[Dict[str, Any]], market_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute quality metrics from recent bars."""

    @abstractmethod
    def _apply_optimization_strategy(
        self,
        market_params: Dict[str, Any],
        quality: Dict[str, float],
        recent_bars: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Adjust market_params based on quality metrics."""

    # =========================================================================
    # CONCRETE — identical across all types
    # =========================================================================

    def accumulate_bar_data(
        self, current_bar_data: Dict[str, Any], minute_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Accumulate minute data into the current bar."""
        if not current_bar_data:
            return {
                "datetime_start": minute_data["datetime"],
                "datetime_end": minute_data["datetime"],
                "open": minute_data["open"],
                "high": minute_data["high"],
                "low": minute_data["low"],
                "close": minute_data["close"],
                "volume": minute_data["volume"],
                "accumulated_size": self.get_bar_size_value(minute_data),
                "tick_count": 1,
            }
        return {
            "datetime_start": current_bar_data["datetime_start"],
            "datetime_end": minute_data["datetime"],
            "open": current_bar_data["open"],
            "high": max(current_bar_data["high"], minute_data["high"]),
            "low": min(current_bar_data["low"], minute_data["low"]),
            "close": minute_data["close"],
            "volume": current_bar_data["volume"] + minute_data["volume"],
            "accumulated_size": current_bar_data["accumulated_size"] + self.get_bar_size_value(minute_data),
            "tick_count": current_bar_data.get("tick_count", 0) + 1,
        }

    def update_market_params(
        self, market_params: Dict[str, Any], finalized_bar: Dict[str, Any]
    ) -> Dict[str, Any]:
        """EMA update of TARGET_KEY and increment bars_completed counter."""
        current_target = market_params.get(self.TARGET_KEY, 1.0)
        actual_size = finalized_bar.get("bar_size", 0)
        alpha = market_params.get("ema_alpha", 0.15)
        new_target = (1 - alpha) * current_target + alpha * actual_size
        market_params = dict(market_params)
        market_params[self.TARGET_KEY] = new_target
        market_params["bars_completed"] = market_params.get("bars_completed", 0) + 1
        return market_params

    def should_create_bar(
        self,
        minute_data: Dict[str, Any],
        current_bar_data: Dict[str, Any],
        market_params: Dict[str, Any],
    ) -> bool:
        """Base logic: threshold met + min duration OR max duration exceeded."""
        if not current_bar_data or not market_params:
            return False

        min_met, max_exceeded, _ = self._check_duration_constraints(
            current_bar_data, minute_data, market_params
        )
        accumulated = current_bar_data.get("accumulated_size", 0)
        target = market_params.get(self.TARGET_KEY, 1.0)
        return (accumulated >= target and min_met) or max_exceeded

    def _check_duration_constraints(
        self,
        current_bar_data: Dict[str, Any],
        minute_data: Dict[str, Any],
        market_params: Dict[str, Any],
    ) -> Tuple[bool, bool, float]:
        """Return (min_duration_met, max_duration_exceeded, duration_minutes)."""
        start = current_bar_data.get("datetime_start")
        end = minute_data.get("datetime")

        if start is None or end is None:
            return True, False, 0.0

        if isinstance(start, str):
            start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if isinstance(end, str):
            end = datetime.fromisoformat(end.replace("Z", "+00:00"))

        duration_min = (end - start).total_seconds() / 60
        min_dur = market_params.get("min_duration_minutes", MIN_DURATION_MINUTES_ABS)
        max_dur = market_params.get("max_duration_minutes", MAX_DURATION_MINUTES_ABS)

        return duration_min >= min_dur, duration_min >= max_dur, duration_min

    def _perform_self_monitoring(self, market_params: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt EMA alpha using TARGET_KEY history (called every N bars)."""
        market_params = dict(market_params)
        history = list(market_params.get("target_volume_history", []))
        history.append(market_params.get(self.TARGET_KEY, 0))
        if len(history) > QUALITY_HISTORY_LENGTH:
            history = history[-QUALITY_HISTORY_LENGTH:]
        market_params["target_volume_history"] = history

        if len(history) >= 5:
            targets = np.array(history, dtype=float)
            mean_t = np.mean(targets)
            if mean_t > 0:
                cv = np.std(targets) / mean_t
                alpha_min = max(market_params.get("alpha_min", ALPHA_MIN_ABSOLUTE), ALPHA_MIN_ABSOLUTE)
                alpha_max = min(market_params.get("alpha_max", ALPHA_MAX_ABSOLUTE), ALPHA_MAX_ABSOLUTE)
                normalized_cv = min(1.0, cv / 0.8)
                new_alpha = alpha_min + (alpha_max - alpha_min) * normalized_cv
                current_alpha = market_params.get("ema_alpha", 0.15)
                if abs(new_alpha - current_alpha) > 0.01:
                    market_params["ema_alpha"] = new_alpha
                    logger.debug(
                        "Alpha adapted: %.3f → %.3f (cv=%.3f)", current_alpha, new_alpha, cv
                    )
        return market_params

    def _can_optimize(self, market_params: Dict[str, Any]) -> bool:
        """Return True if optimization cooldown has passed."""
        events = market_params.get("optimization_events", [])
        if not events:
            return True
        current_bars = market_params.get("bars_completed", 0)
        last_bar = events[-1].get("bar_number", 0)
        return (current_bars - last_bar) >= EMA_OPTIMIZATION_COOLDOWN

    def _enforce_parameter_bounds(self, market_params: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp common parameters to safe ranges, then delegate to type-specific bounds."""
        market_params = dict(market_params)

        # Alpha
        alpha = market_params.get("ema_alpha", 0.15)
        alpha_min = max(market_params.get("alpha_min", ALPHA_MIN_ABSOLUTE), ALPHA_MIN_ABSOLUTE)
        alpha_max = min(market_params.get("alpha_max", ALPHA_MAX_ABSOLUTE), ALPHA_MAX_ABSOLUTE)
        market_params["ema_alpha"] = max(alpha_min, min(alpha_max, alpha))

        # Bars per day
        bpd = market_params.get("target_bars_per_day", 20)
        market_params["target_bars_per_day"] = max(BARS_PER_DAY_MIN, min(BARS_PER_DAY_MAX, bpd))

        # Duration
        min_dur = market_params.get("min_duration_minutes", MIN_DURATION_MINUTES_ABS)
        max_dur = market_params.get("max_duration_minutes", MAX_DURATION_MINUTES_ABS)
        market_params["min_duration_minutes"] = max(MIN_DURATION_MINUTES_ABS, min(60, min_dur))
        market_params["max_duration_minutes"] = max(30, min(MAX_DURATION_MINUTES_ABS, max_dur))
        if market_params["min_duration_minutes"] >= market_params["max_duration_minutes"]:
            market_params["max_duration_minutes"] = market_params["min_duration_minutes"] * 2

        return self._enforce_type_bounds(market_params)

    # -------------------------------------------------------------------------
    # Shared utility methods
    # -------------------------------------------------------------------------

    def _get_optimal_precision(self, value: float) -> int:
        """Decimal places based on value magnitude."""
        abs_val = abs(float(value))
        if abs_val >= 50_000:
            return 2
        elif abs_val >= 1_000:
            return 3
        elif abs_val >= 10:
            return 4
        elif abs_val >= 0.1:
            return 5
        elif abs_val >= 0.001:
            return 6
        return 8

    def _validate_minute_data(self, minute_data: Dict[str, Any]) -> bool:
        """OHLC sanity check + extreme move warning."""
        try:
            close = float(minute_data.get("close", 0))
            high = float(minute_data.get("high", 0))
            low = float(minute_data.get("low", 0))
            open_price = float(minute_data.get("open", 0))
            volume = float(minute_data.get("volume", 0))

            if any(p <= 0 for p in [close, high, low, open_price]):
                return False
            if high < low or high < max(open_price, close) or low > min(open_price, close):
                return False
            if volume < 0:
                return False

            # Extreme move warning (does not reject)
            prev = getattr(self, "previous_close", None)
            if prev and prev > 0:
                change = abs(close - prev) / prev
                if change > self.EXTREME_PRICE_MOVE_THRESHOLD:
                    logger.warning("Extreme price move: %.2f%%", change * 100)

            return True
        except Exception:
            return False

    def _calculate_entropy(self, data, bins: int = 50) -> float:
        """Shannon entropy of a data distribution."""
        try:
            arr = np.asarray(data, dtype=float)
            if len(arr) < 10:
                return 2.0
            hist, _ = np.histogram(arr, bins=bins, density=True)
            hist = hist[hist > 0]
            if len(hist) < 2:
                return 2.0
            bin_width = (arr.max() - arr.min()) / bins
            probs = hist * bin_width
            probs = probs / probs.sum()
            return float(-np.sum(probs * np.log2(probs)))
        except Exception:
            return 2.0

    def _calculate_regime_stability(
        self, returns, closes, window: int = 500
    ) -> float:
        """Rolling return-correlation measure of regime stability (0–1)."""
        try:
            if len(returns) < window * 2:
                return 0.5
            vol_short = np.abs(returns)
            if len(vol_short) >= window + 50:
                vol_cur = vol_short[-window:]
                vol_lag = vol_short[-window - 50:-50]
                if (
                    len(vol_cur) == len(vol_lag)
                    and len(vol_cur) > 10
                    and np.std(vol_cur) > 0
                    and np.std(vol_lag) > 0
                ):
                    corr = np.corrcoef(vol_cur, vol_lag)[0, 1]
                    if not np.isnan(corr):
                        return float(max(0.0, min(1.0, (corr + 1) / 2)))
            return 0.5
        except Exception:
            return 0.5

    def _calculate_market_noise(self, returns, window: int = 300) -> float:
        """Noise ratio of returns (0–1, higher = noisier)."""
        try:
            arr = np.asarray(returns, dtype=float)
            if len(arr) < 50:
                return 0.5
            recent = arr[-window:] if len(arr) >= window else arr
            std = np.std(recent)
            mean_abs = np.mean(np.abs(recent))
            if mean_abs > 1e-8:
                return float(max(0.0, min(1.0, (std / mean_abs - 2.0) / 18.0)))
            return 0.5
        except Exception:
            return 0.5

    def _remove_outliers(self, data_array: np.ndarray) -> np.ndarray:
        """Remove outliers via IQR method."""
        try:
            arr = np.asarray(data_array, dtype=float)
            if len(arr) < 10:
                return arr
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = q75 - q25
            if iqr == 0:
                return arr
            lo, hi = q25 - 1.5 * iqr, q75 + 1.5 * iqr
            filtered = arr[(arr >= lo) & (arr <= hi)]
            if len(filtered) < len(arr) * 0.7:
                lo, hi = q25 - 2.5 * iqr, q75 + 2.5 * iqr
                filtered = arr[(arr >= lo) & (arr <= hi)]
            return filtered if len(filtered) > 0 else arr
        except Exception:
            return data_array
