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


def calibrate(
    bar_processor: RenkoBar,
    csv_path: Path,
    gather_fn: Callable,
) -> dict:
    """
    Calibrate RenkoBar from the first ANALYSIS_LOOKBACK_DAYS of tick data.

    Flow: ticks → synthetic 1-minute OHLCV → RenkoBar.analyze_market_history()

    RenkoBar.analyze_market_history() expects dicts with keys:
        datetime, open, high, low, close, volume
    — exactly what _ticks_to_minute_ohlcv produces.

    After calibration the minute-based duration keys are replaced with
    tick-specific second equivalents.  The minute keys are kept in
    market_params with their calibrated values so that RenkoBar's
    optimisation methods (_calculate_bar_quality, _apply_optimization_strategy)
    which read "duration_minutes" from finalized bars continue to work.

    renko_reference is seeded from the last calibration close and persisted
    into market_params so process_chunk can restore it on the first chunk.
    """
    cal_p, cal_q, cal_ts_ms, _ = gather_fn(csv_path, ANALYSIS_LOOKBACK_DAYS)
    n_days = max(
        1, int((int(cal_ts_ms[-1]) - int(cal_ts_ms[0])) // (86_400 * 1_000)) + 1
    )
    logger.info(
        "  Calibrating renko bars from %d ticks (%d days) ...", len(cal_ts_ms), n_days
    )

    minute_bars = _ticks_to_minute_ohlcv(
        cal_p.astype(np.float64), cal_q.astype(np.float64), cal_ts_ms
    )
    del cal_p, cal_q, cal_ts_ms
    gc.collect()

    market_params = bar_processor.analyze_market_history(minute_bars)
    del minute_bars
    gc.collect()

    # ── Inject tick-specific duration bounds (seconds) ─────────────────────────
    # The minute-bar pipeline sets min/max_duration_MINUTES.  At tick resolution
    # bars close in seconds.  We add _seconds keys consumed by process_chunk
    # while leaving the _minutes keys intact for the optimisation methods.
    estimated_bar_seconds = 86_400.0 / max(
        1.0, market_params.get("target_bars_per_day", 4.0)
    )
    market_params["min_duration_seconds"] = TICK_MIN_DURATION_SECONDS
    market_params["max_duration_seconds"] = max(
        TICK_MAX_DURATION_FLOOR_SECONDS,
        min(TICK_MAX_DURATION_SECONDS, int(estimated_bar_seconds * 3)),
    )

    # Persist renko_reference so process_chunk can restore it on the first chunk
    market_params["renko_reference"] = bar_processor.renko_reference

    logger.info(
        "  Renko calibration done — "
        "brick_size=%.6f (%.4f%%)  bars/day=%.1f  alpha=%.3f",
        market_params["target_brick_size"],
        market_params["target_brick_size"] * 100,
        market_params["target_bars_per_day"],
        market_params["ema_alpha"],
    )
    return market_params


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