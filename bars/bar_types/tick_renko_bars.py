"""
tick_renko.py — Renko bars from Binance aggTrades tick data.

Imports RenkoBar from its existing location (used by the DB pipeline too).
Adds tick-specific helpers and the two entry points consumed by run.py.

Tick entry points
-----------------
calibrate(bar_processor, csv_path, gather_fn)  -> market_params
process_chunk(prices, quantities, timestamps_ms, is_buyer_maker,
              bar_processor, market_params, recent, update_fn)
              -> (bars, market_params, leftover)

Core mechanic — what a renko bar measures
------------------------------------------
A renko bar closes when the absolute price displacement from the bar's opening
price reaches the EMA-adapted target_brick_size (expressed as a relative
fraction of the opening price).  Direction (bullish/bearish) is determined at
close.

    accumulated_size = |current_price - renko_reference| / renko_reference

This is NOT a running sum — it is the current displacement from a fixed
reference level.  The reference is set once per bar (at the first tick) and
never changes while that bar is open.  This is identical to the minute-bar
RenkoBar, applied at tick resolution.

Key tick-level differences vs RenkoBar (minute-bar version)
-------------------------------------------------------------
1. Reference set at the first TICK's price (not first minute's open) — exact.
2. Displacement checked after every tick — no minute-bucketing needed.
   The renko condition is purely price-based, making it naturally tick-native.
3. Duration in seconds — tick bars can close in seconds.
4. VWAP — exact Σ(price×qty) / Σqty; dollar_volume exact (not close×volume proxy).
5. Carry-forward — unconsumed ticks after a bar closes are carried as raw
   leftover arrays prepended to the next chunk (same as tick_dollar/tick_volume).
6. Calibration — ticks → synthetic 1-minute OHLCV → RenkoBar.analyze_market_history().
   Same input format as the DB pipeline so calibrated parameters are comparable.

Why no minute-bucketing
------------------------
Volatility and hybrid bars bucket into minutes because their signals are defined
at the minute level (close-to-close returns).  Renko's signal is purely a price
LEVEL condition — there is no averaging or accumulation over time involved.
Every tick gives an exact, up-to-date displacement reading.  Minute-bucketing
would only delay detection of the crossing event and would not preserve
comparability with the minute-bar baseline (which also checks on every row).

renko_reference persistence
-----------------------------
bar_processor.renko_reference is set to the opening tick price of each new bar
and is persisted in market_params["renko_reference"] so it survives chunk
boundaries and process restarts — identical to how VolatilityBar persists
previous_close.

Chunk-boundary carry
---------------------
Ticks that do not complete a bar within the current chunk are returned as a
`leftover` dict containing raw numpy arrays:
    {"prices", "quantities", "timestamps_ms", "is_buyer_maker"}
run.py prepends these to the next chunk before calling process_chunk again.
This is the same pattern as tick_dollar and tick_volume.

EMA update
-----------
update_fn (from run.py / _update_and_adapt) calls
RenkoBar.update_market_params() which performs a bidirectional EMA with a 2×
upward cap — identical to the minute-bar pipeline.  No additional sync step is
needed (unlike tick_volume which has a separate _sync_volume_state).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import gc
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from common import *

import numpy as np

from common.logging import get_logger
from common.constants import ANALYSIS_LOOKBACK_DAYS
from .base import BaseBar as RenkoBar

logger = get_logger(__name__)

# ── Tick-specific constants ────────────────────────────────────────────────────

TICK_MIN_DURATION_SECONDS = 10  # absolute floor — never close under 10 s
TICK_MAX_DURATION_SECONDS = 28_800  # 8 h hard ceiling
TICK_MAX_DURATION_FLOOR_SECONDS = 300  # computed-max lower bound — 5 min

_MINUTE_MS = 60_000  # milliseconds per minute
_MS_PER_S = 1_000  # milliseconds per second
_CUMSUM_WINDOW = 500_000  # initial vectorised search window; doubles until bar found


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _ms_to_dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(int(ts_ms) / 1_000.0, tz=timezone.utc)


def _get_precision(value: float) -> int:
    """Decimal places based on price magnitude — mirrors tick_dollar/tick_volume."""
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


def _ticks_to_minute_ohlcv(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps_ms: np.ndarray,
) -> list[dict]:
    """
    Aggregate ticks into 1-minute OHLCV dicts for calibration only.

    All buckets including the last are finalised — a slightly incomplete
    final minute is acceptable for analysis (same as tick_volume calibration).
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
        p_sl = prices[idx].astype(np.float64)
        q_sl = quantities[idx].astype(np.float64)
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


# ── Vectorised bar-end search ─────────────────────────────────────────────────


def _find_bar_end(
    displacements: np.ndarray,
    dur_s: np.ndarray,
    target: float,
    min_s: float,
    max_s: float,
) -> Optional[int]:
    """
    Return the relative tick index where the renko bar closes, or None.

    Two triggers — earliest wins:
      1. Normal  — displacement >= target_brick_size  AND  min_duration_seconds met
      2. Timeout — elapsed seconds >= max_duration_seconds

    Renko has no extreme trigger (unlike dollar/volume bars) because:
      - displacement is a relative level check, not a cumulative sum
      - a 5× brick displacement would mean a flash crash / rally; time-capping
        is the correct response in that scenario, not a lower threshold
      - the bidirectional EMA in update_market_params already handles the
        subsequent recalibration of the brick size after a time-capped bar

    displacements : |price_i - renko_reference| / renko_reference  (relative)
    dur_s         : elapsed seconds from first tick (dur_s[0] == 0)
    target        : target_brick_size (relative fraction)
    min_s         : min_duration_seconds
    max_s         : max_duration_seconds
    """
    bar_end: Optional[int] = None

    # Normal trigger — threshold met, then wait for min_duration
    idx = np.where(displacements >= target)[0]
    if len(idx):
        first = idx[0]
        # min_duration check starts from the tick where threshold was crossed
        min_met = np.where(dur_s[first:] >= min_s)[0]
        if len(min_met):
            bar_end = first + min_met[0]

    # Timeout trigger — override if earlier
    idx_to = np.where(dur_s >= max_s)[0]
    if len(idx_to):
        first_to = idx_to[0]
        if bar_end is None or first_to < bar_end:
            bar_end = first_to

    return bar_end


# ── Bar accumulation ──────────────────────────────────────────────────────────


def _build_bar(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps_ms: np.ndarray,
    is_buyer_maker: np.ndarray,
    renko_reference: float,
) -> dict:
    """
    Vectorised OHLCV accumulation for one renko bar slice.

    accumulated_size = |last_price - renko_reference| / renko_reference
        — the displacement at the moment the bar closed, expressed as a
          relative fraction of the reference price.  This is NOT a running
          sum; it mirrors RenkoBar.accumulate_bar_data exactly.

    dollar_volume = Σ(price × qty) for ticks in this bar only — exact tick-
        level value, used both for VWAP and stored in the output (replacing
        the minute-bar proxy of close × volume).
    """
    p = prices.astype(np.float64)
    q = quantities.astype(np.float64)
    dv = p * q

    close_price = float(p[-1])
    rel_disp = (
        abs(close_price - renko_reference) / renko_reference
        if renko_reference > 0
        else 0.0
    )

    return {
        "datetime_start": _ms_to_dt(int(timestamps_ms[0])),
        "datetime_end": _ms_to_dt(int(timestamps_ms[-1])),
        "open": float(p[0]),
        "high": float(p.max()),
        "low": float(p.min()),
        "close": close_price,
        "volume": float(q.sum()),  # Σ qty (ticks only)
        "dollar_volume": float(dv.sum()),  # Σ(price×qty) — exact
        "accumulated_size": rel_disp,  # displacement at close
        "renko_reference": renko_reference,  # opening price anchor
        "tick_count": len(p),
        "buy_tick_count": int((~is_buyer_maker).sum()),  # buyer-aggressor ticks
        "sell_tick_count": int(is_buyer_maker.sum()),  # seller-aggressor ticks
    }


def _finalize_bar(raw: dict, market_params: dict) -> dict:
    """
    Convert accumulated tick data into a completed renko bar dict.

    Mirrors RenkoBar.finalize_bar() with the following tick-level extensions:
      - duration_seconds  (tick bars close in seconds, not minutes)
      - duration_minutes  (retained for _calculate_bar_quality compatibility)
      - vwap              (exact Σ(p×q)/Σq — not available at minute level)
      - dollar_volume     (exact Σ(p×q) — replaces close×volume proxy)
      - buy/sell tick counts (microstructure signal from is_buyer_maker)
    """
    start = raw["datetime_start"]
    end = raw["datetime_end"]
    dur_s = max(0.0, (end - start).total_seconds())
    dur_min = dur_s / 60.0

    open_val = float(raw["open"])
    high_val = float(raw["high"])
    low_val = float(raw["low"])
    close_val = float(raw["close"])
    vol_val = float(raw["volume"])  # Σ qty
    dv_val = float(raw["dollar_volume"])  # Σ(price×qty)
    bar_size = float(raw["accumulated_size"])  # relative displacement at close
    prec = _get_precision(close_val)

    direction = "bullish" if close_val >= open_val else "bearish"
    vwap = round(dv_val / vol_val, prec) if vol_val > 0 else round(close_val, prec)
    bar_return = round((close_val - open_val) / open_val, 6) if open_val > 0 else 0.0
    price_range = round((high_val - low_val) / open_val, 6) if open_val > 0 else 0.0
    close_position = (
        round((close_val - low_val) / (high_val - low_val), 6)
        if high_val != low_val
        else 0.5
    )
    buy_tick_count = int(raw.get("buy_tick_count", 0))
    sell_tick_count = int(raw.get("sell_tick_count", 0))
    total_ticks = buy_tick_count + sell_tick_count
    tick_imbalance = (
        round((buy_tick_count - sell_tick_count) / total_ticks, 6)
        if total_ticks > 0
        else 0.0
    )

    return {
        "datetime": end,
        "datetime_start": start,
        "datetime_end": end,
        "open": round(open_val, prec),
        "high": round(high_val, prec),
        "low": round(low_val, prec),
        "close": round(close_val, prec),
        "volume": round(vol_val, 6),
        "dollar_volume": round(dv_val, 2),
        "vwap": vwap,
        "bar_size": round(bar_size, 8),  # relative displacement fraction
        "direction": direction,
        "duration_seconds": round(dur_s, 1),
        "duration_minutes": round(dur_min, 4),  # for _calculate_bar_quality
        "tick_count": raw.get("tick_count", 1),
        "buy_tick_count": buy_tick_count,
        "sell_tick_count": sell_tick_count,
        "tick_imbalance": tick_imbalance,
        "bar_return": bar_return,
        "price_range": price_range,
        "close_position": close_position,
        "renko_reference": round(raw["renko_reference"], prec),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tick entry points (called by run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate(bar_processor, csv_path: Path, gather_fn: Callable) -> dict:
    """
    Tick-native calibration for renko bars.

    - target_brick_size   = median(daily H-L / median_close) / target_bars_per_day
                            uses exact tick highs/lows and individual trade prices
    - target_bars_per_day = f(log-return entropy, market efficiency)
    - ema_alpha           = f(daily range CV, regime stability, noise)
    No minute bucketing anywhere.

    renko_reference is seeded from the last calibration tick price so
    process_chunk can restore it on the first chunk — identical to before.
    """
    from common.constants import (
        ANALYSIS_LOOKBACK_DAYS, BARS_PER_DAY_MIN, BARS_PER_DAY_MAX,
        RENKO_BASE_FREQUENCY, SLOW_BAR_FREQUENCY_MULTIPLIER,
        DURATION_ESTIMATED_MULTIPLIER, ALPHA_MIN_ABSOLUTE, ALPHA_MAX_ABSOLUTE,
    )
    from common.logging import get_logger
    logger = get_logger(__name__)

    TICK_MIN_DURATION_SECONDS       = 10
    TICK_MAX_DURATION_SECONDS       = 28_800
    TICK_MAX_DURATION_FLOOR_SECONDS = 300

    cal_p, cal_q, cal_ts_ms, _ = gather_fn(csv_path, ANALYSIS_LOOKBACK_DAYS)
    n_days = max(1, int((int(cal_ts_ms[-1]) - int(cal_ts_ms[0])) // _MS_PER_DAY) + 1)
    logger.info("  Calibrating renko bars from %d ticks (%d days) ...", len(cal_ts_ms), n_days)

    prices_f64 = cal_p.astype(np.float64)
    log_ret    = _tick_log_returns(prices_f64)
    if len(log_ret) < 100:
        logger.warning("  Insufficient log-returns — using defaults")
        del cal_p, cal_q, cal_ts_ms; gc.collect()
        return bar_processor._get_default_params()

    # ── information multiplier ────────────────────────────────────────────────
    ret_entropy  = _tick_entropy(log_ret)
    rand_entropy = _tick_entropy(np.random.normal(0, np.std(log_ret), len(log_ret)))
    information_ratio      = ret_entropy / rand_entropy if rand_entropy > 0 else 1.0
    information_multiplier = max(0.5, min(2.0, information_ratio))

    # ── activity multiplier ───────────────────────────────────────────────────
    market_eff          = _tick_market_efficiency(cal_p, cal_q)
    activity_percentile = max(0.0, min(1.0, market_eff))
    activity_multiplier = 0.5 + activity_percentile

    # ── target_bars_per_day ───────────────────────────────────────────────────
    target_bpd = max(BARS_PER_DAY_MIN,
                     min(BARS_PER_DAY_MAX,
                         RENKO_BASE_FREQUENCY
                         * information_multiplier * activity_multiplier
                         / SLOW_BAR_FREQUENCY_MULTIPLIER))

    # ── tick-native daily range: (day_high - day_low) / median(day_closes) ───
    # Mirrors renko minute logic but uses exact tick prices per day.
    day_idx     = _tick_daily_split(cal_ts_ms)
    unique_days = np.unique(day_idx)
    daily_rel_ranges = []
    for d in unique_days:
        mask       = day_idx == d
        day_prices = prices_f64[mask]
        if len(day_prices) == 0:
            continue
        med_close = float(np.median(day_prices))
        if med_close > 0:
            daily_rel_ranges.append((day_prices.max() - day_prices.min()) / med_close)
    daily_rel_ranges = np.array(daily_rel_ranges, dtype=np.float64)

    median_daily_range  = float(np.median(daily_rel_ranges))
    mad_range           = float(np.median(np.abs(daily_rel_ranges - median_daily_range)))
    range_cv            = mad_range / median_daily_range if median_daily_range > 0 else 0.3

    target_brick_size = median_daily_range / max(1.0, target_bpd)

    # ── ema_alpha ─────────────────────────────────────────────────────────────
    regime_stability = _tick_regime_stability(log_ret)
    market_noise     = _tick_market_noise(log_ret)
    alpha_min, alpha_max, ema_alpha = _alpha_from_cv(range_cv, regime_stability, market_noise)
    alpha_min = max(alpha_min, ALPHA_MIN_ABSOLUTE)
    alpha_max = min(alpha_max, ALPHA_MAX_ABSOLUTE)
    ema_alpha = max(alpha_min, min(alpha_max, ema_alpha))

    min_s, max_s = _duration_seconds_from_bpd(
        target_bpd, TICK_MIN_DURATION_SECONDS,
        TICK_MAX_DURATION_FLOOR_SECONDS, TICK_MAX_DURATION_SECONDS,
        DURATION_ESTIMATED_MULTIPLIER)

    # seed renko_reference from last calibration tick — identical to before
    bar_processor.renko_reference = float(prices_f64[-1]) if len(prices_f64) > 0 else None

    del cal_p, cal_q, cal_ts_ms; gc.collect()

    logger.info(
        "  Renko calibration done — brick_size=%.6f (%.4f%%)  "
        "bars/day=%.1f  alpha=%.3f  [TICK-NATIVE]",
        target_brick_size, target_brick_size * 100, target_bpd, ema_alpha)

    return {
        "target_brick_size":      target_brick_size,
        "ema_alpha":              ema_alpha,
        "alpha_min":              alpha_min,
        "alpha_max":              alpha_max,
        "target_bars_per_day":    target_bpd,
        "min_duration_seconds":   min_s,
        "max_duration_seconds":   max_s,
        "range_cv":               range_cv,
        "median_daily_range":     median_daily_range,
        "regime_stability":       regime_stability,
        "market_noise":           market_noise,
        "information_ratio":      information_ratio,
        "market_efficiency":      market_eff,
        "renko_reference":        bar_processor.renko_reference,
        "bars_completed":         0,
        "monitoring_counter":     0,
        "bars_since_optimization":0,
        "target_volume_history":  [],
        "optimization_events":    [],
    }


def process_chunk(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps_ms: np.ndarray,
    is_buyer_maker: np.ndarray,
    bar_processor: RenkoBar,
    market_params: dict,
    recent: deque,
    update_fn: Callable,
) -> tuple[list, dict, dict]:
    """
    Extract completed renko bars from a tick array.

    Algorithm
    ----------
    For each bar iteration:
      1.  Read renko_reference (persisted from previous bar or calibration).
      2.  Compute cumulative displacement array:
              displacements[i] = |price[i] - renko_reference| / renko_reference
          This matches RenkoBar.accumulate_bar_data exactly — displacement
          is always measured from the fixed reference, NOT from price[0] of
          the current window (which would shift the reference on every chunk
          boundary if a bar straddles chunks).
      3.  Call _find_bar_end() to locate the closing tick.
      4.  Build and finalise the bar; call update_fn.
      5.  Set renko_reference = closing tick's price for the NEXT bar,
          persist into both bar_processor.renko_reference and
          market_params["renko_reference"].

    renko_reference persistence
    ----------------------------
    At the start of every chunk the current reference is restored from
    market_params["renko_reference"] into bar_processor.renko_reference.
    After every bar closes the new reference (closing price) is written
    back to both.  This guarantees the reference survives chunk boundaries
    with zero discontinuity.

    Note on renko_reference initialisation
    ----------------------------------------
    The minute-bar RenkoBar sets renko_reference = first_minute.open at the
    START of each new bar (inside accumulate_bar_data).  At tick level the
    equivalent is: renko_reference = price[pos] (the first tick of the new
    bar).  After a bar closes we set renko_reference = closing price, which
    becomes the opening anchor for the next bar — consistent with the
    minute-bar logic where the NEW bar's reference is the first minute's open
    (which equals the previous bar's close in a continuous price series).

    Carry / leftover
    -----------------
    Ticks that do not complete a bar (pos < n at loop exit) are returned as
    a leftover dict.  run.py prepends these to the next chunk.  The
    renko_reference in market_params ensures those leftover ticks continue
    measuring displacement from the correct anchor.

    update_fn
    ----------
    Calls RenkoBar.update_market_params() — bidirectional EMA with 2× cap.
    No secondary sync step needed (unlike tick_volume's _sync_volume_state).
    """
    # ── Restore renko_reference from persisted state ───────────────────────────
    ref = market_params.get("renko_reference")
    if ref is not None:
        bar_processor.renko_reference = float(ref)

    n = len(prices)
    pos = 0
    bars: list[dict] = []

    while pos < n:
        target = float(market_params["target_brick_size"])
        min_s = float(
            market_params.get("min_duration_seconds", TICK_MIN_DURATION_SECONDS)
        )
        max_s = float(
            market_params.get("max_duration_seconds", TICK_MAX_DURATION_SECONDS)
        )

        # ── Establish renko_reference for this bar ────────────────────────────
        # If we have a persisted reference from a previous bar (or calibration),
        # use it — this is the case when a bar straddles a chunk boundary.
        # If not (very first bar ever), anchor at the first tick of this bar.
        if bar_processor.renko_reference is None:
            bar_processor.renko_reference = float(prices[pos])
            market_params["renko_reference"] = bar_processor.renko_reference

        renko_ref = float(bar_processor.renko_reference)

        # ── Vectorised displacement search ────────────────────────────────────
        window = min(_CUMSUM_WINDOW, n - pos)
        bar_end = None

        while bar_end is None and window <= (n - pos):
            p_slice = prices[pos : pos + window].astype(np.float64)
            ts_slice = timestamps_ms[pos : pos + window]
            dur_s = (ts_slice - ts_slice[0]) / _MS_PER_S

            # Displacement from the FIXED renko_reference (not from p_slice[0])
            displacements = (
                np.abs(p_slice - renko_ref) / renko_ref
                if renko_ref > 0
                else np.zeros(len(p_slice))
            )

            bar_end = _find_bar_end(displacements, dur_s, target, min_s, max_s)

            if bar_end is None:
                if window == n - pos:
                    break  # exhausted chunk — carry leftover
                window = min(window * 2, n - pos)

        if bar_end is None:
            # No bar completed in remaining ticks — carry everything as leftover
            break

        end = bar_end + 1  # exclusive slice index

        raw = _build_bar(
            prices[pos : pos + end],
            quantities[pos : pos + end],
            timestamps_ms[pos : pos + end],
            is_buyer_maker[pos : pos + end],
            renko_ref,
        )
        bar = _finalize_bar(raw, market_params)
        bars.append(bar)
        recent.append(bar)

        # ── Update renko_reference for the NEXT bar ───────────────────────────
        # The next bar's reference is this bar's closing tick price.
        # This mirrors the minute-bar pipeline: after a bar closes,
        # renko_reference is implicitly reset to the next minute's open
        # (which equals the previous close in a gapless series).
        new_ref = float(prices[pos + bar_end])
        bar_processor.renko_reference = new_ref

        # ── EMA + monitoring + optimisation ───────────────────────────────────
        market_params = update_fn(bar_processor, market_params, bar, recent)

        # Persist the new reference AFTER update_fn so it is not overwritten
        market_params["renko_reference"] = new_ref

        pos += end

    # ── Leftover carry ────────────────────────────────────────────────────────
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