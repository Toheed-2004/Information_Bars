"""
tick_dollar.py — Dollar bars from Binance aggTrades tick data.

Class
-----
TickDollarBar   BaseBar subclass; bars close when Σ(price × qty) hits target.

Tick-level entry points (consumed by run.py)
--------------------------------------------
calibrate(bar_processor, csv_path, gather_fn)  -> market_params
process_chunk(prices, quantities, timestamps_ms, is_buyer_maker,
              bar_processor, market_params, recent, update_fn)
              -> (bars, market_params, leftover)

Key differences vs minute-bar DollarBar
-----------------------------------------
1. Dollar volume per tick = price × quantity  (exact, no approximation)
2. Duration in seconds — tick bars can close in < 1 minute
3. Tick-level OHLC — open/high/low/close are individual trade prices
4. Buy/sell imbalance — tracked per bar from is_buyer_maker
5. VWAP — exact: Σ(price×qty) / Σqty using only ticks in this bar
6. Calibration via analyze_from_dataframe() — fully vectorised, no row loops

No overflow carry-forward: the closing tick is consumed whole into the bar.
bar_size may exceed target; the next bar starts at zero accumulation.
"""

import gc
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bitpredict.data.custom_bars.bar_types.base import BaseBar
from bitpredict.data.custom_bars.constants import (
    ALPHA_MAX_ABSOLUTE,
    ALPHA_MIN_ABSOLUTE,
    ANALYSIS_LOOKBACK_DAYS,
    BARS_PER_DAY_MAX,
    BARS_PER_DAY_MIN,
    BAR_FREQUENCY_MULTIPLIER,
    DOLLAR_TIER1_BASE,
    DOLLAR_TIER2_BASE,
    DOLLAR_TIER_BASE_BARS,
    DOLLAR_TIER_BASELINE_MULTIPLIERS,
    DOLLAR_TIER_INFLATION_BASE_YEAR,
    DOLLAR_TIER_INFLATION_RATE,
    DURATION_ESTIMATED_MULTIPLIER,
    EXTREME_THRESHOLD_MULTIPLIER,
    FREQ_ADJ_BASE,
    FREQ_ADJ_SENSITIVITY,
    MIN_DAILY_DATA_FOR_ANALYSIS,
    OPTIMIZATION_CV_TARGET_HIGH,
    OPTIMIZATION_CV_TARGET_LOW,
    OPTIMIZATION_DURATION_CV_TARGET_HIGH,
    OPTIMIZATION_DURATION_CV_TARGET_LOW,
)
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)

# ── Tick-specific constants ───────────────────────────────────────────────────
TICK_MIN_DURATION_SECONDS = 10  # hard floor: never close under 10 s
TICK_MAX_DURATION_SECONDS = 28_800  # hard ceiling: 8 hours
TICK_MAX_DURATION_FLOOR_SECONDS = 300  # computed max lower bound: 5 min
MIN_TICKS_FOR_ANALYSIS = 5_000
MIN_DAILY_TICKS_FOR_ANALYSIS = 3

_MS_PER_DAY = 86_400 * 1_000  # milliseconds per day
_CUMSUM_WINDOW = 500_000  # initial search window; doubles until bar found
_MS_PER_S = 1_000  # milliseconds per second


# ═══════════════════════════════════════════════════════════════════════════════
# TickDollarBar
# ═══════════════════════════════════════════════════════════════════════════════


class TickDollarBar(BaseBar):
    """Adaptive dollar-volume bars built from individual aggTrade records."""

    TARGET_KEY = "target_dollar_volume"

    # ── Primary calibration ───────────────────────────────────────────────────

    def analyze_from_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calibrate all bar parameters from a raw tick DataFrame.

        Expected columns: timestamp_ms (int64, ms epoch), price, quantity,
        is_buyer_maker (bool, optional).
        """
        if df is None or len(df) == 0:
            logger.warning("Empty DataFrame — using defaults")
            return self._get_default_params()
        if len(df) < MIN_TICKS_FOR_ANALYSIS:
            logger.warning("Only %d ticks — using defaults", len(df))
            return self._get_default_params()

        prices = df["price"].to_numpy(dtype=np.float64)
        quantities = df["quantity"].to_numpy(dtype=np.float64)
        dv = prices * quantities

        # Daily groupby via integer day index — no datetime parsing needed
        day_idx = df["timestamp_ms"].to_numpy(dtype=np.int64) // _MS_PER_DAY
        day_s = pd.Series(day_idx)
        dv_s = pd.Series(dv)
        daily_dv = dv_s.groupby(day_s).sum()
        daily_counts = dv_s.groupby(day_s).count()

        n_days = len(daily_dv)
        if n_days < MIN_DAILY_TICKS_FOR_ANALYSIS:
            logger.warning(
                "Only %d day(s) of tick data — calibrating from available data", n_days
            )

        daily_arr = daily_dv.to_numpy(dtype=np.float64)
        ticks_per_day = float(daily_counts.mean())
        median_daily_vol = float(np.median(daily_arr))
        mad = float(np.median(np.abs(daily_arr - median_daily_vol)))
        dollar_volume_cv = mad / median_daily_vol if median_daily_vol > 0 else 0.3

        asset_tier = self._classify_asset_tier(daily_arr.tolist())
        baseline_liq = self._calculate_dynamic_baseline(daily_arr.tolist(), asset_tier)

        log_returns = np.diff(np.log(prices))
        log_returns = log_returns[log_returns != 0.0]

        information_entropy = (
            self._calculate_entropy(log_returns.tolist())
            if len(log_returns) > 10
            else 2.5
        )
        information_factor = max(0.5, min(2.0, information_entropy / 3.0))
        regime_stability = (
            self._calculate_regime_stability(log_returns, prices)
            if len(log_returns) >= 1000
            else 0.5
        )
        market_noise = (
            self._calculate_market_noise(log_returns) if len(log_returns) >= 50 else 0.3
        )
        market_efficiency = self._calculate_market_efficiency_ticks(prices, quantities)

        if "is_buyer_maker" in df.columns:
            is_sell = df["is_buyer_maker"].to_numpy(dtype=bool)
            buy_dv = float(dv[~is_sell].sum())
            sell_dv = float(dv[is_sell].sum())
            total_dv = buy_dv + sell_dv
            imbalance_baseline = (buy_dv - sell_dv) / total_dv if total_dv > 0 else 0.0
        else:
            imbalance_baseline = 0.0

        base_bpd = DOLLAR_TIER_BASE_BARS.get(asset_tier, 3.4)
        freq_adj = FREQ_ADJ_BASE + (market_efficiency * FREQ_ADJ_SENSITIVITY)
        target_bars_per_day = max(
            BARS_PER_DAY_MIN,
            min(
                BARS_PER_DAY_MAX,
                base_bpd * information_factor * freq_adj / BAR_FREQUENCY_MULTIPLIER,
            ),
        )
        initial_target_dv = median_daily_vol / target_bars_per_day

        alpha_min = max(ALPHA_MIN_ABSOLUTE, 0.12 - (regime_stability * 0.08))
        alpha_max = min(ALPHA_MAX_ABSOLUTE, 0.22 + (market_noise * 0.13))
        ema_alpha = max(alpha_min, min(alpha_max, 0.18 - (dollar_volume_cv * 0.25)))

        estimated_bar_seconds = 86_400.0 / target_bars_per_day
        max_duration_seconds = max(
            TICK_MAX_DURATION_FLOOR_SECONDS,
            min(
                TICK_MAX_DURATION_SECONDS,
                int(estimated_bar_seconds * DURATION_ESTIMATED_MULTIPLIER),
            ),
        )

        extreme_threshold = initial_target_dv * EXTREME_THRESHOLD_MULTIPLIER

        logger.debug(
            "TickDollarBar calibration: target=$%.0f  alpha=%.3f  tier=%s  "
            "bars/day=%.1f  ticks/day=%.0f",
            initial_target_dv,
            ema_alpha,
            asset_tier,
            target_bars_per_day,
            ticks_per_day,
        )

        return {
            "target_dollar_volume": initial_target_dv,
            "ema_alpha": ema_alpha,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "target_bars_per_day": target_bars_per_day,
            "min_duration_seconds": TICK_MIN_DURATION_SECONDS,
            "max_duration_seconds": max_duration_seconds,
            "asset_tier": asset_tier,
            "baseline_liquidity": baseline_liq,
            "market_efficiency": market_efficiency,
            "regime_stability": regime_stability,
            "market_noise": market_noise,
            "information_entropy": information_entropy,
            "dollar_volume_cv": dollar_volume_cv,
            "imbalance_baseline": imbalance_baseline,
            "extreme_threshold": extreme_threshold,
            "bars_completed": 0,
            "monitoring_counter": 0,
            "bars_since_optimization": 0,
            "target_volume_history": [],
            "optimization_events": [],
        }

    def analyze_market_history(
        self, historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compatibility shim — converts list-of-dicts to DataFrame and delegates."""
        if not historical_data:
            return self._get_default_params()
        has_ibm = any("is_buyer_maker" in r for r in historical_data[:10])
        rows = {
            "timestamp_ms": [
                self._to_timestamp_ms(r.get("datetime") or r.get("timestamp"))
                for r in historical_data
            ],
            "price": [float(r["price"]) for r in historical_data],
            "quantity": [float(r["quantity"]) for r in historical_data],
            "is_buyer_maker": [
                bool(r.get("is_buyer_maker", False)) for r in historical_data
            ],
        }
        df = pd.DataFrame(rows)
        if not has_ibm:
            df = df.drop(columns=["is_buyer_maker"])
        return self.analyze_from_dataframe(df)

    # ── BaseBar abstract implementations ──────────────────────────────────────

    def get_bar_size_value(self, tick_data: Dict[str, Any]) -> float:
        if not self._validate_tick_data(tick_data):
            return 0.0
        return max(0.0, float(tick_data["price"]) * float(tick_data["quantity"]))

    def accumulate_bar_data(
        self, current_bar_data: Dict[str, Any], tick_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        price = float(tick_data["price"])
        qty = float(tick_data["quantity"])
        dv = max(0.0, price * qty)
        ts = tick_data.get("datetime") or tick_data.get("timestamp")
        is_sell = bool(tick_data.get("is_buyer_maker", False))

        if not current_bar_data:
            return {
                "datetime_start": ts,
                "datetime_end": ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": qty,
                "accumulated_size": dv,
                "tick_count": 1,
                "buy_dollar_volume": 0.0 if is_sell else dv,
                "sell_dollar_volume": dv if is_sell else 0.0,
            }
        return {
            "datetime_start": current_bar_data["datetime_start"],
            "datetime_end": ts,
            "open": current_bar_data["open"],
            "high": max(current_bar_data["high"], price),
            "low": min(current_bar_data["low"], price),
            "close": price,
            "volume": current_bar_data["volume"] + qty,
            "accumulated_size": current_bar_data["accumulated_size"] + dv,
            "tick_count": current_bar_data.get("tick_count", 0) + 1,
            "buy_dollar_volume": current_bar_data.get("buy_dollar_volume", 0.0)
            + (0.0 if is_sell else dv),
            "sell_dollar_volume": current_bar_data.get("sell_dollar_volume", 0.0)
            + (dv if is_sell else 0.0),
        }

    def should_create_bar(
        self,
        tick_data: Dict[str, Any],
        current_bar_data: Dict[str, Any],
        market_params: Dict[str, Any],
    ) -> bool:
        if not current_bar_data or not market_params:
            return False
        min_met, max_exceeded, duration_sec = self._check_duration_seconds(
            current_bar_data, tick_data, market_params
        )
        accumulated = current_bar_data.get("accumulated_size", 0.0)
        target = market_params.get(self.TARGET_KEY, 1.0)
        extreme_threshold = target * EXTREME_THRESHOLD_MULTIPLIER
        half_min_sec = (
            market_params.get("min_duration_seconds", TICK_MIN_DURATION_SECONDS) * 0.5
        )
        extreme_detected = (
            accumulated >= extreme_threshold and duration_sec >= half_min_sec
        )
        return (accumulated >= target and min_met) or max_exceeded or extreme_detected

    def finalize_bar(
        self, bar_data: Dict[str, Any], market_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        start = self._parse_datetime(bar_data["datetime_start"])
        end = self._parse_datetime(bar_data["datetime_end"])
        duration_seconds = max(0.0, (end - start).total_seconds())
        open_val = float(bar_data["open"])
        high_val = float(bar_data["high"])
        low_val = float(bar_data["low"])
        close_val = float(bar_data["close"])
        volume_val = float(bar_data["volume"])  # Σ qty of ticks only
        bar_size = float(bar_data["accumulated_size"])  # Σ(p×q) for this bar
        target = market_params.get(self.TARGET_KEY, bar_size)
        price_prec = self._get_optimal_precision(close_val)

        buy_dv = float(bar_data.get("buy_dollar_volume", 0.0))
        sell_dv = float(bar_data.get("sell_dollar_volume", 0.0))

        # VWAP: tick_dollar_volume / tick_volume — excludes carry intentionally.
        # buy_dv + sell_dv = Σ(p×q) for this bar's ticks only (no carry).
        tick_dv = buy_dv + sell_dv
        vwap = (
            round(tick_dv / volume_val, price_prec)
            if volume_val > 0
            else round(close_val, price_prec)
        )

        bar_return = (
            round((close_val - open_val) / open_val, 6) if open_val > 0 else 0.0
        )
        price_range = round((high_val - low_val) / open_val, 6) if open_val > 0 else 0.0
        close_position = (
            round((close_val - low_val) / (high_val - low_val), 6)
            if high_val != low_val
            else 0.5
        )
        total_signed_dv = buy_dv + sell_dv
        buy_sell_imbalance = (
            round((buy_dv - sell_dv) / total_signed_dv, 6)
            if total_signed_dv > 0
            else 0.0
        )
        return {
            "datetime": bar_data["datetime_end"],
            "datetime_start": bar_data["datetime_start"],
            "datetime_end": bar_data["datetime_end"],
            "open": round(open_val, price_prec),
            "high": round(high_val, price_prec),
            "low": round(low_val, price_prec),
            "close": round(close_val, price_prec),
            "volume": round(volume_val, 6),
            "bar_size": round(bar_size, 2),
            "vwap": vwap,
            "duration_seconds": round(duration_seconds, 1),
            "tick_count": bar_data.get("tick_count", 1),
            "bar_return": bar_return,
            "price_range": price_range,
            "close_position": close_position,
            "buy_dollar_volume": round(buy_dv, 2),
            "sell_dollar_volume": round(sell_dv, 2),
            "buy_sell_imbalance": buy_sell_imbalance,
        }

    def update_market_params(
        self, market_params: Dict[str, Any], finalized_bar: Dict[str, Any]
    ) -> Dict[str, Any]:
        """EMA with spike cap (max 2× target per event) + carry + extreme_threshold sync."""
        market_params = dict(market_params)
        market_params["bars_completed"] = market_params.get("bars_completed", 0) + 1

        actual = finalized_bar.get("bar_size", 0.0)
        target = market_params.get(self.TARGET_KEY, 1.0)
        alpha = market_params.get("ema_alpha", 0.15)
        ema_input = min(actual, target * 2.0)
        market_params[self.TARGET_KEY] = (1 - alpha) * target + alpha * ema_input

        market_params["extreme_threshold"] = (
            market_params[self.TARGET_KEY] * EXTREME_THRESHOLD_MULTIPLIER
        )
        return market_params

    # ── Default params and bounds ─────────────────────────────────────────────

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "target_dollar_volume": 2_000_000_000,
            "ema_alpha": 0.15,
            "alpha_min": 0.08,
            "alpha_max": 0.25,
            "target_bars_per_day": 12,
            "min_duration_seconds": TICK_MIN_DURATION_SECONDS,
            "max_duration_seconds": 7_200,
            "asset_tier": "tier1",
            "baseline_liquidity": 500_000_000,
            "market_efficiency": 0.5,
            "regime_stability": 0.5,
            "market_noise": 0.3,
            "information_entropy": 2.5,
            "dollar_volume_cv": 0.3,
            "imbalance_baseline": 0.0,
            "extreme_threshold": 2_000_000_000 * EXTREME_THRESHOLD_MULTIPLIER,
            "bars_completed": 0,
            "monitoring_counter": 0,
            "bars_since_optimization": 0,
            "target_volume_history": [],
            "optimization_events": [],
        }

    def _enforce_type_bounds(self, market_params: Dict[str, Any]) -> Dict[str, Any]:
        market_params = dict(market_params)
        tv = market_params.get("target_dollar_volume", 2_000_000_000)
        market_params["target_dollar_volume"] = max(100_000, min(50_000_000_000, tv))
        market_params["extreme_threshold"] = (
            market_params["target_dollar_volume"] * EXTREME_THRESHOLD_MULTIPLIER
        )
        min_s = max(
            TICK_MIN_DURATION_SECONDS,
            market_params.get("min_duration_seconds", TICK_MIN_DURATION_SECONDS),
        )
        max_s = market_params.get("max_duration_seconds", TICK_MAX_DURATION_SECONDS)
        market_params["min_duration_seconds"] = min_s
        market_params["max_duration_seconds"] = max(
            min_s * 2, min(TICK_MAX_DURATION_SECONDS, max_s)
        )
        return market_params

    # ── Quality and optimisation ──────────────────────────────────────────────

    def _calculate_bar_quality(
        self, recent_bars: List[Dict[str, Any]], market_params: Dict[str, Any]
    ) -> Dict[str, float]:
        if len(recent_bars) < 20:
            return {
                "liquidity_consistency": 0.5,
                "duration_adaptability": 0.5,
                "imbalance_stability": 0.5,
                "overall_quality": 0.5,
            }

        sizes = np.array(
            [float(b.get("bar_size", 0)) for b in recent_bars], dtype=np.float64
        )
        durations = np.array(
            [float(b.get("duration_seconds", 0)) for b in recent_bars], dtype=np.float64
        )
        imbalances = np.array(
            [float(b.get("buy_sell_imbalance", 0)) for b in recent_bars],
            dtype=np.float64,
        )

        filtered = self._remove_outliers(sizes)
        if len(filtered) > 5 and np.mean(filtered) > 0:
            size_cv = np.std(filtered) / np.mean(filtered)
            if size_cv <= OPTIMIZATION_CV_TARGET_LOW:
                lc = 1.0
            elif size_cv <= OPTIMIZATION_CV_TARGET_HIGH:
                lc = (
                    1.0
                    - (
                        (size_cv - OPTIMIZATION_CV_TARGET_LOW)
                        / (OPTIMIZATION_CV_TARGET_HIGH - OPTIMIZATION_CV_TARGET_LOW)
                    )
                    * 0.3
                )
            else:
                lc = max(
                    0.0, 0.7 - ((size_cv - OPTIMIZATION_CV_TARGET_HIGH) / 0.6) * 0.7
                )
        else:
            lc = 0.5

        if len(durations) > 10 and np.mean(durations) > 0:
            dur_cv = np.std(durations) / np.mean(durations)
            if (
                OPTIMIZATION_DURATION_CV_TARGET_LOW
                <= dur_cv
                <= OPTIMIZATION_DURATION_CV_TARGET_HIGH
            ):
                da = 1.0
            elif dur_cv < OPTIMIZATION_DURATION_CV_TARGET_LOW:
                da = max(0.0, dur_cv / OPTIMIZATION_DURATION_CV_TARGET_LOW)
            else:
                da = max(
                    0.0, 1.0 - (dur_cv - OPTIMIZATION_DURATION_CV_TARGET_HIGH) / 0.8
                )
        else:
            da = 0.5

        imb_s = (
            max(
                0.0,
                1.0
                - float(np.mean(np.abs(imbalances))) * 2.0
                - float(np.std(imbalances)) * 0.5,
            )
            if len(imbalances) > 10
            else 0.5
        )

        return {
            "liquidity_consistency": lc,
            "duration_adaptability": da,
            "imbalance_stability": imb_s,
            "overall_quality": lc * 0.55 + da * 0.30 + imb_s * 0.15,
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
        current_target = market_params.get("target_dollar_volume", 2_000_000_000)

        if overall < 0.5:
            if liquidity < 0.4:
                old_bpd = market_params.get("target_bars_per_day", 12)
                new_bpd = min(BARS_PER_DAY_MAX, old_bpd * 1.1)
                market_params["target_bars_per_day"] = new_bpd
                market_params["target_dollar_volume"] = current_target * (
                    old_bpd / new_bpd
                )
            if dur_adapt < 0.4:
                old_min = market_params.get(
                    "min_duration_seconds", TICK_MIN_DURATION_SECONDS
                )
                market_params["min_duration_seconds"] = min(old_min + 30, 300)

        events = list(market_params.get("optimization_events", []))
        events.append(
            {
                "bar_number": market_params.get("bars_completed", 0),
                "overall_quality": overall,
            }
        )
        market_params["optimization_events"] = events[-10:]
        return self._enforce_type_bounds(market_params)

    # ── Tick-specific helpers ─────────────────────────────────────────────────

    def _check_duration_seconds(
        self,
        current_bar_data: Dict[str, Any],
        tick_data: Dict[str, Any],
        market_params: Dict[str, Any],
    ) -> Tuple[bool, bool, float]:
        start = current_bar_data.get("datetime_start")
        end = tick_data.get("datetime") or tick_data.get("timestamp")
        if start is None or end is None:
            return True, False, 0.0
        duration_sec = max(
            0.0,
            (self._parse_datetime(end) - self._parse_datetime(start)).total_seconds(),
        )
        min_s = market_params.get("min_duration_seconds", TICK_MIN_DURATION_SECONDS)
        max_s = market_params.get("max_duration_seconds", TICK_MAX_DURATION_SECONDS)
        return duration_sec >= min_s, duration_sec >= max_s, duration_sec

    def _validate_tick_data(self, tick_data: Dict[str, Any]) -> bool:
        try:
            price = float(tick_data.get("price", 0))
            qty = float(tick_data.get("quantity", 0))
            return price > 0 and qty > 0
        except Exception:
            return False

    def _calculate_market_efficiency_ticks(
        self, prices: np.ndarray, quantities: np.ndarray
    ) -> float:
        if len(prices) < 100:
            return 0.5
        p_sample = (
            prices[np.linspace(0, len(prices) - 1, 50_000, dtype=int)]
            if len(prices) > 50_000
            else prices
        )
        log_ret = np.diff(np.log(p_sample))
        log_ret = log_ret[log_ret != 0.0]
        if len(log_ret) < 4:
            return 0.5
        autocorr = float(np.corrcoef(log_ret[:-1], log_ret[1:])[0, 1])
        if np.isnan(autocorr):
            autocorr = 0.0
        eff_ac = max(0.0, 1.0 - abs(autocorr) * 10)
        var_1 = float(np.var(log_ret))
        var_2 = float(np.var(log_ret[::2]))
        eff_vr = (
            max(0.0, 1.0 - abs(var_2 / (2.0 * var_1) - 1.0) * 2.0) if var_1 > 0 else 0.5
        )
        return float(max(0.1, min(0.9, eff_ac * 0.5 + eff_vr * 0.5)))

    def _classify_asset_tier(self, daily_volumes: List[float]) -> str:
        median_dv = float(np.median(daily_volumes))
        inflation = (
            datetime.now().year - DOLLAR_TIER_INFLATION_BASE_YEAR
        ) * DOLLAR_TIER_INFLATION_RATE + 1.0
        if median_dv >= DOLLAR_TIER1_BASE * inflation:
            return "tier1"
        elif median_dv >= DOLLAR_TIER2_BASE * inflation:
            return "tier2"
        return "tier3"

    def _calculate_dynamic_baseline(
        self, daily_volumes: List[float], asset_tier: str
    ) -> float:
        return max(
            1_000_000,
            float(np.median(daily_volumes))
            * DOLLAR_TIER_BASELINE_MULTIPLIERS.get(asset_tier, 0.01),
        )

    @staticmethod
    def _to_timestamp_ms(ts: Any) -> int:
        if isinstance(ts, (int, float)):
            val = int(ts)
            return val // 1_000 if val > 9_999_999_999_999 else val
        if isinstance(ts, datetime):
            return int(ts.timestamp() * 1_000)
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000)

    @staticmethod
    def _parse_datetime(ts: Any) -> datetime:
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════════
# Tick entry points (called by run.py)
# ═══════════════════════════════════════════════════════════════════════════════


def _ms_to_dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(int(ts_ms) / 1_000.0, tz=timezone.utc)


def _find_bar_end(
    cum_dv: np.ndarray,
    dur_s: np.ndarray,
    target: float,
    min_s: float,
    max_s: float,
    extreme_thr: float,
) -> Optional[int]:
    bar_end: Optional[int] = None

    idx = np.where(cum_dv >= target)[0]
    if len(idx):
        first = idx[0]
        min_met = np.where(dur_s[first:] >= min_s)[0]
        if len(min_met):
            bar_end = first + min_met[0]

    idx = np.where(cum_dv >= extreme_thr)[0]
    if len(idx):
        first_ext = idx[0]
        if dur_s[first_ext] >= min_s * 0.5:
            if bar_end is None or first_ext < bar_end:
                bar_end = first_ext

    idx = np.where(dur_s >= max_s)[0]
    if len(idx):
        first_to = idx[0]
        if bar_end is None or first_to < bar_end:
            bar_end = first_to

    return bar_end


def _build_bar(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps_ms: np.ndarray,
    is_buyer_maker: np.ndarray,
) -> dict:
    """
    Vectorised OHLCV accumulation for one dollar bar slice.

    accumulated_size = Σ(p×q) for this bar's ticks only.
    The bar closes when accumulated_size >= target.
    The closing tick is consumed whole — no overflow carry-forward.
    """
    p = prices.astype(np.float64)
    q = quantities.astype(np.float64)
    dv = p * q
    return {
        "datetime_start": _ms_to_dt(int(timestamps_ms[0])),
        "datetime_end": _ms_to_dt(int(timestamps_ms[-1])),
        "open": float(p[0]),
        "high": float(p.max()),
        "low": float(p.min()),
        "close": float(p[-1]),
        "volume": float(q.sum()),  # Σ qty, ticks only
        "accumulated_size": float(dv.sum()),  # Σ(p×q) for this bar's ticks
        "tick_count": len(p),
        "buy_dollar_volume": float(dv[~is_buyer_maker].sum()),
        "sell_dollar_volume": float(dv[is_buyer_maker].sum()),
    }


def calibrate(
    bar_processor: TickDollarBar, csv_path: Path, gather_fn: Callable
) -> dict:
    """Calibrate from the first ANALYSIS_LOOKBACK_DAYS of tick data."""
    cal_p, cal_q, cal_ts_ms, cal_ibm = gather_fn(csv_path, ANALYSIS_LOOKBACK_DAYS)
    n_days = max(
        1, int((int(cal_ts_ms[-1]) - int(cal_ts_ms[0])) // (86_400 * 1_000)) + 1
    )
    logger.info(
        "  Calibrating dollar bars from %d ticks (%d days) ...", len(cal_ts_ms), n_days
    )

    df = pd.DataFrame(
        {
            "timestamp_ms": cal_ts_ms,
            "price": cal_p,
            "quantity": cal_q,
            "is_buyer_maker": cal_ibm,
        }
    )
    market_params = bar_processor.analyze_from_dataframe(df)
    del df, cal_p, cal_q, cal_ts_ms, cal_ibm
    gc.collect()

    logger.info(
        "  Dollar calibration done — target=$%.0f  bars/day=%.1f  tier=%s  alpha=%.3f",
        market_params["target_dollar_volume"],
        market_params["target_bars_per_day"],
        market_params["asset_tier"],
        market_params["ema_alpha"],
    )
    return market_params


def process_chunk(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps_ms: np.ndarray,
    is_buyer_maker: np.ndarray,
    bar_processor: TickDollarBar,
    market_params: dict,
    recent: deque,
    update_fn: Callable,
) -> tuple[list, dict, dict]:
    """
    Extract completed dollar bars from a tick array.

    Each bar starts with zero accumulation. When accumulated_size >= target,
    the bar closes — the closing tick is consumed whole (bar_size may exceed
    target). The next bar starts at zero. No overflow carry-forward.
    update_fn handles EMA/monitoring/optimisation via bar_processor.update_market_params.
    """
    dv_all = prices.astype(np.float64) * quantities.astype(np.float64)
    n = len(dv_all)
    pos = 0
    bars: list[dict] = []

    while pos < n:
        target = float(market_params["target_dollar_volume"])
        min_s = float(
            market_params.get("min_duration_seconds", TICK_MIN_DURATION_SECONDS)
        )
        max_s = float(
            market_params.get("max_duration_seconds", TICK_MAX_DURATION_SECONDS)
        )
        extreme_thr = target * EXTREME_THRESHOLD_MULTIPLIER


        window = min(_CUMSUM_WINDOW, n - pos)
        bar_end = None

        while bar_end is None and window <= (n - pos):
            dv_slice = dv_all[pos : pos + window]
            ts_slice = timestamps_ms[pos : pos + window]
            cum_dv = np.cumsum(dv_slice)
            dur_s = (ts_slice - ts_slice[0]) / _MS_PER_S
            bar_end = _find_bar_end(cum_dv, dur_s, target, min_s, max_s, extreme_thr)
            if bar_end is None:
                if window == n - pos:
                    break
                window = min(window * 2, n - pos)

        if bar_end is None:
            break

        end = bar_end + 1
        raw = _build_bar(
            prices[pos : pos + end],
            quantities[pos : pos + end],
            timestamps_ms[pos : pos + end],
            is_buyer_maker[pos : pos + end],
        )
        bar = bar_processor.finalize_bar(raw, market_params)
        bars.append(bar)
        recent.append(bar)
        # update_fn calls bar_processor.update_market_params — the overridden version
        # that does spike-capped EMA + carry_forward_dv + extreme_threshold sync.
        market_params = update_fn(bar_processor, market_params, bar, recent)
        pos += end

    leftover = (
        {
            "prices": prices[pos:],
            "quantities": quantities[pos:],
            "timestamps_ms": timestamps_ms[pos:],
            "is_buyer_maker": is_buyer_maker[pos:],
        }
        if pos < n
        else {}
    )
    return bars, market_params, leftover