"""
tick_volume.py — Volume bars from Binance aggTrades tick data.

Imports VolumeBar from its existing location (used by the DB pipeline too).
Adds tick-specific helpers and the two entry points consumed by run.py.

Tick entry points
-----------------
calibrate(bar_processor, csv_path, gather_fn)  -> market_params
process_chunk(prices, quantities, timestamps_ms,
              bar_processor, market_params, recent, update_fn)
              -> (bars, market_params, leftover)

What a volume bar measures
---------------------------
Accumulated BTC quantity (Σ quantity) — NOT dollar volume.
Each bar closes when Σ quantity reaches target_volume, regardless of price.
This isolates supply/demand dynamics from price level.

Key tick-level differences vs VolumeBar (minute-bar version)
-------------------------------------------------------------
1. Accumulator is Σ quantity — exact at tick level, no close-price proxy.
2. Duration in seconds — tick bars can close in seconds.
3. Extreme trigger — 5× target (VOLUME_EXTREME_THRESHOLD_MULTIPLIER).
4. VWAP — exact Σ(p×q)/Σq, only possible at tick level.
5. Calibration — ticks → synthetic 1-minute OHLCV → analyze_market_history().

No overflow carry-forward: closing tick consumed whole into bar.
bar_size may exceed target; next bar starts at zero accumulation.

EMA update design
-----------------
update_fn calls VolumeBar.update_market_params() (bidirectional EMA, 2× cap).
_sync_volume_state() keeps extreme_threshold in sync with updated target.
This avoids running the EMA twice or double-incrementing bars_completed.

Chunk-boundary handling
-----------------------
Unconsumed ticks are returned as raw leftover arrays prepended to the next chunk.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import gc
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .base import BaseBar as VolumeBar

import numpy as np

from common.logging import get_logger
from common.constants import (
    ANALYSIS_LOOKBACK_DAYS,
    VOLUME_EXTREME_THRESHOLD_MULTIPLIER,
)

logger = get_logger(__name__)

_CUMSUM_WINDOW = 500_000
_MS_PER_S = 1_000  # milliseconds per second
_MINUTE_MS = 60_000  # milliseconds per minute

TICK_MIN_DURATION_SECONDS = 10
TICK_MAX_DURATION_SECONDS = 28_800
TICK_MAX_DURATION_FLOOR_SECONDS = 300



# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ms_to_dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(int(ts_ms) / 1_000.0, tz=timezone.utc)


def _ticks_to_minute_ohlcv(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps_ms: np.ndarray,
) -> list[dict]:
    """
    Aggregate ticks into 1-minute OHLCV dicts for calibration.
    All buckets including the last are finalised (calibration only —
    a slightly incomplete last minute is acceptable for analysis).
    """
    if len(prices) == 0:
        return []
    bucket_keys = timestamps_ms // _MINUTE_MS
    change_pts = np.where(np.diff(bucket_keys))[0] + 1
    slices = np.split(np.arange(len(prices)), change_pts)
    bars = []
    for idx in slices:
        if len(idx) == 0:
            continue
        p_sl = prices[idx]
        q_sl = quantities[idx]
        bars.append(
            {
                "datetime": _ms_to_dt(int(timestamps_ms[idx[0]])),
                "open": float(p_sl[0]),
                "high": float(p_sl.max()),
                "low": float(p_sl.min()),
                "close": float(p_sl[-1]),
                "volume": float(q_sl.sum()),
            }
        )
    return bars


def _find_bar_end(
    cum_vol: np.ndarray,
    dur_s: np.ndarray,
    target: float,
    min_s: float,
    max_s: float,
    extreme_thr: float,
) -> Optional[int]:
    """
    Return relative tick index where the volume bar closes, or None.
    Three triggers — earliest wins:
      1. Normal   — cum_vol >= target  AND  min_duration_seconds met
      2. Extreme  — cum_vol >= 5× target  AND  50% of min_duration met
      3. Timeout  — elapsed seconds >= max_duration_seconds
    """
    bar_end: Optional[int] = None

    idx = np.where(cum_vol >= target)[0]
    if len(idx):
        first = idx[0]
        min_met = np.where(dur_s[first:] >= min_s)[0]
        if len(min_met):
            bar_end = first + min_met[0]

    idx = np.where(cum_vol >= extreme_thr)[0]
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
) -> dict:
    """
    Vectorised OHLCV accumulation for a single volume bar slice.

    accumulated_size = Σ quantity for this bar's ticks only.
    The closing tick is consumed whole — no overflow carry-forward.
    dollar_volume = Σ(price × quantity) for VWAP.
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
        "volume": float(q.sum()),  # Σ qty
        "accumulated_size": float(q.sum()),  # Σ qty for this bar's ticks
        "dollar_volume": float(dv.sum()),  # for VWAP — ticks only
        "tick_count": len(p),
    }


def _get_precision(value: float) -> int:
    v = abs(float(value))
    if v >= 50_000:
        return 2
    if v >= 1_000:
        return 3
    if v >= 10:
        return 4
    if v >= 0.1:
        return 5
    if v >= 0.001:
        return 6
    return 8


def _finalize_bar(raw: dict, market_params: dict) -> dict:
    """
    Convert accumulated tick data into a completed volume bar dict.

    Extensions vs minute-bar VolumeBar.finalize_bar():
      - duration_seconds   (bars close in seconds, not minutes)
      - vwap               (exact Σ(p×q)/Σq using tick-only dollar_volume)
    """
    start = raw["datetime_start"]
    end = raw["datetime_end"]
    dur_s = max(0.0, (end - start).total_seconds())

    open_val = float(raw["open"])
    high_val = float(raw["high"])
    low_val = float(raw["low"])
    close_val = float(raw["close"])
    vol_val = float(raw["volume"])  # Σ qty, ticks only
    bar_size = float(raw["accumulated_size"])  # Σ qty for this bar
    dv_val = float(raw["dollar_volume"])  # Σ(p×q), ticks only
    target = market_params.get("target_volume", bar_size)
    prec = _get_precision(close_val)

    # VWAP uses tick-only dollar_volume / tick-only volume — carry excluded
    vwap = round(dv_val / vol_val, prec) if vol_val > 0 else round(close_val, prec)

    bar_return = round((close_val - open_val) / open_val, 6) if open_val > 0 else 0.0
    price_range = round((high_val - low_val) / open_val, 6) if open_val > 0 else 0.0
    close_position = (
        round((close_val - low_val) / (high_val - low_val), 6)
        if high_val != low_val
        else 0.5
    )
    return {
        "datetime": raw["datetime_end"],
        "datetime_start": raw["datetime_start"],
        "datetime_end": raw["datetime_end"],
        "open": round(open_val, prec),
        "high": round(high_val, prec),
        "low": round(low_val, prec),
        "close": round(close_val, prec),
        "volume": round(vol_val, 6),
        "bar_size": round(bar_size, 6),
        "dollar_volume": round(dv_val, 2),
        "vwap": vwap,
        "duration_seconds": round(dur_s, 1),
        "tick_count": raw.get("tick_count", 1),
        "bar_return": bar_return,
        "price_range": price_range,
        "close_position": close_position,
    }


def _sync_volume_state(market_params: dict) -> dict:
    """
    Sync extreme_threshold after EMA update.
    Called AFTER update_fn — does NOT re-run the EMA or increment bars_completed.
    """
    market_params = dict(market_params)
    market_params["extreme_threshold"] = (
        market_params.get("target_volume", 1.0) * VOLUME_EXTREME_THRESHOLD_MULTIPLIER
    )
    return market_params


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def calibrate(
    bar_processor: VolumeBar,
    csv_path: Path,
    gather_fn: Callable,
) -> dict:
    """
    Calibrate VolumeBar from the first ANALYSIS_LOOKBACK_DAYS of tick data.
    Ticks → synthetic 1-minute OHLCV → VolumeBar.analyze_market_history().
    """
    cal_p, cal_q, cal_ts_ms, _ = gather_fn(csv_path, ANALYSIS_LOOKBACK_DAYS)
    n_days = max(
        1, int((int(cal_ts_ms[-1]) - int(cal_ts_ms[0])) // (86_400 * 1_000)) + 1
    )
    logger.info(
        "  Calibrating volume bars from %d ticks (%d days) ...", len(cal_ts_ms), n_days
    )

    minute_bars = _ticks_to_minute_ohlcv(
        cal_p.astype(np.float64), cal_q.astype(np.float64), cal_ts_ms
    )
    del cal_p, cal_q, cal_ts_ms
    gc.collect()

    market_params = bar_processor.analyze_market_history(minute_bars)
    del minute_bars
    gc.collect()

    # Add tick-specific duration bounds (minute-bar params use minutes)
    estimated_bar_seconds = 86_400.0 / market_params.get("target_bars_per_day", 6)
    market_params["min_duration_seconds"] = TICK_MIN_DURATION_SECONDS
    market_params["max_duration_seconds"] = max(
        TICK_MAX_DURATION_FLOOR_SECONDS,
        min(TICK_MAX_DURATION_SECONDS, int(estimated_bar_seconds * 3)),
    )
    logger.info(
        "  Volume calibration done — target=%.4f BTC  bars/day=%.1f  tier=%s  alpha=%.3f",
        market_params["target_volume"],
        market_params["target_bars_per_day"],
        market_params["asset_tier"],
        market_params["ema_alpha"],
    )
    return market_params


def process_chunk(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps_ms: np.ndarray,
    bar_processor: VolumeBar,
    market_params: dict,
    recent: deque,
    update_fn: Callable,
) -> tuple[list, dict, dict]:
    """
    Extract completed volume bars from a tick array.

    Each bar starts with zero accumulation. Closing tick consumed whole.
    After each bar:
      1. update_fn  → VolumeBar.update_market_params (spike-capped EMA,
                       bars_completed++, standard monitoring/optimisation)
      2. _sync_volume_state → extreme_threshold sync
         (no EMA re-run, no double bars_completed increment)
    """
    n = len(quantities)
    pos = 0
    bars: list[dict] = []

    while pos < n:
        target = float(market_params["target_volume"])
        min_s = float(
            market_params.get("min_duration_seconds", TICK_MIN_DURATION_SECONDS)
        )
        max_s = float(
            market_params.get("max_duration_seconds", TICK_MAX_DURATION_SECONDS)
        )
        extreme_thr = target * VOLUME_EXTREME_THRESHOLD_MULTIPLIER

        window = min(_CUMSUM_WINDOW, n - pos)
        bar_end = None

        while bar_end is None and window <= (n - pos):
            q_slice = quantities[pos : pos + window].astype(np.float64)
            ts_slice = timestamps_ms[pos : pos + window]
            cum_vol = np.cumsum(q_slice)
            dur_s = (ts_slice - ts_slice[0]) / _MS_PER_S
            bar_end = _find_bar_end(cum_vol, dur_s, target, min_s, max_s, extreme_thr)
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
        )
        bar = _finalize_bar(raw, market_params)
        bars.append(bar)
        recent.append(bar)

        # Step 1: EMA + monitoring + optimisation (calls VolumeBar.update_market_params)
        market_params = update_fn(bar_processor, market_params, bar, recent)
        # Step 2: sync carry and extreme_threshold only — no second EMA
        market_params = _sync_volume_state(market_params)

        pos += end

    leftover = (
        {
            "prices": prices[pos:],
            "quantities": quantities[pos:],
            "timestamps_ms": timestamps_ms[pos:],
        }
        if pos < n
        else {}
    )
    return bars, market_params, leftover