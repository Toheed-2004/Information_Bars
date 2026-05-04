"""
tick_range.py — Range bars from Binance aggTrades tick data.

Tick-native signal: relative price excursion  (running_high - running_low) / bar_open
---------------------------------------------------------------------------------------
Each bar closes when the span between the highest and lowest tick price seen
WITHIN the bar, expressed as a fraction of the bar's opening price, reaches
the EMA-adapted target_range:

    accumulated_size = (bar_running_high - bar_running_low) / bar_open

Why this is genuinely different from tick_volatility
------------------------------------------------------
tick_volatility accumulates Σ|log(p_i/p_{i-1})| — the TOTAL PATH LENGTH
traveled by price.  Every reversal adds to the accumulator.

tick_range accumulates (high - low) / open — the WIDTH of the price excursion.
Reversals do NOT increase the accumulator once the bar high and low are set.
Only new extremes (new bar high or new bar low) advance it.

Concrete example:
    Price sequence: 50000 → 50500 → 50000 → 50500 → 50000
    Realized vol:   4 × log(50500/50000) ≈ 4 × 0.01 = 0.04   (path = 4%)
    Range signal:   (50500 - 50000) / 50000 = 0.01             (span = 1%)

A mean-reverting bar has large realized vol but small range.
A trending bar has both large realized vol and large range.
These closing conditions select structurally different bars.

Why this is the correct tick-native analog of the minute RangeBar
------------------------------------------------------------------
The minute RangeBar accumulates Σ(minute_high - minute_low)/minute_close.
This is a sum of per-minute spans across multiple minutes.

At tick level, (high - low)/open is the natural single-bar analog.  There
is no "minute high/low" to sum — we track the bar's own running extremes
directly from every trade.  This is STRICTLY more precise: the minute
pipeline approximates the bar's true high and low from minute-bucket
boundaries; the tick pipeline tracks the exact highest and lowest trade
price within the bar.

Comparison validity
-------------------
Both the minute and tick range bars close on the same conceptual signal —
the bar has covered enough relative price territory — but computed at
different resolutions.  The minute pipeline detects this only at minute
boundaries; the tick pipeline detects it at the exact trade where the
excursion first becomes wide enough.  This is a valid comparison.

Calibration
-----------
target_bars_per_day, ema_alpha, and duration bounds: from
RangeBar.analyze_market_history() fed with 1-min OHLCV.  These
frequency/adaptation parameters do not depend on signal units.

target_range: REPLACED with the tick-native value:
    median(daily_tick_range) / target_bars_per_day
where daily_tick_range = (day_high - day_low) / day_first_price
computed from raw ticks over the calibration window.

This is in the same units as the process loop accumulator — a relative
fraction of the bar opening price — making the EMA adaptation correct.

Carry state across chunk boundaries
-------------------------------------
No previous_price needed — range is computed from within-bar prices only.

open_bar_data  dict with keys:
    ts_start_ms         int     first tick timestamp of current bar
    bar_open            float   first tick price of current bar (reference)
    bar_high            float   running maximum price seen in bar
    bar_low             float   running minimum price seen in bar
    volume              float   Σ qty
    dollar_volume       float   Σ(price×qty)
    buy_dollar_volume   float   Σ(price×qty) buyer-aggressor ticks
    sell_dollar_volume  float   Σ(price×qty) seller-aggressor ticks
    tick_count          int     ticks processed so far in bar

No leftover tick arrays — when the chunk ends mid-bar, remaining ticks are
absorbed into open_bar_data.  Same carry pattern as tick_volatility.

Vectorised search
-----------------
Uses np.maximum.accumulate and np.minimum.accumulate over the price slice,
seeded with the carry high/low.  This produces the running excursion array
in O(n) with no loop.  No log-computation, no previous_price, no numerical
guard for zero prices.

Signature
----------
calibrate(bar_processor, csv_path, gather_fn)  -> market_params
process_chunk(prices, quantities, timestamps_ms, is_buyer_maker,
              bar_processor, market_params, recent,
              open_bar_data, update_fn)
              -> (bars, market_params, open_bar_data, leftover)

No import from tick_volatility — this module is fully self-contained.
"""

import gc
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from common.logging import get_logger
from common.constants import ANALYSIS_LOOKBACK_DAYS

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_MINUTE_MS = 60_000
_MS_PER_S = 1_000

TICK_MIN_DURATION_SECONDS = 10
TICK_MAX_DURATION_SECONDS = 28_800
TICK_MAX_DURATION_FLOOR_SECONDS = 300
_CUMSUM_WINDOW = 500_000


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _ms_to_dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(int(ts_ms) / 1_000.0, tz=timezone.utc)


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


# ── Calibration helpers ───────────────────────────────────────────────────────


def _ticks_to_minute_ohlcv_for_calibration(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps_ms: np.ndarray,
) -> list[dict]:
    """
    1-minute OHLCV dicts for RangeBar.analyze_market_history() — calibration only.
    Last (possibly incomplete) bucket included; acceptable for analysis.
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


def _compute_daily_tick_range(
    prices: np.ndarray,
    timestamps_ms: np.ndarray,
) -> np.ndarray:
    """
    Compute the daily relative price excursion from raw ticks:
        daily_range = (day_high - day_low) / day_first_price

    This is the tick-native calibration input for target_range:
        target_range = median(daily_tick_range) / target_bars_per_day

    Using day_first_price (first trade of the day) as the reference mirrors
    how the bar accumulator uses bar_open as its reference.  A typical BTC
    day moves ~3–5% peak-to-trough, so a target of (3.5% / 4 bars) = 0.875%
    per bar is the expected range per bar.

    Returns an array of per-day relative ranges — one value per calendar day.
    """
    p = prices.astype(np.float64)
    ts = timestamps_ms.astype(np.int64)

    valid = p > 0
    p = p[valid]
    ts = ts[valid]
    if len(p) < 2:
        return np.array([0.0])

    day_idx = ts // (86_400 * 1_000)
    unique_days = np.unique(day_idx)

    daily_ranges = []
    for d in unique_days:
        mask = day_idx == d
        day_prices = p[mask]
        if len(day_prices) < 2:
            continue
        day_open = float(day_prices[0])
        if day_open <= 0:
            continue
        day_high = float(day_prices.max())
        day_low = float(day_prices.min())
        daily_ranges.append((day_high - day_low) / day_open)

    return np.array(daily_ranges, dtype=np.float64) if daily_ranges else np.array([0.0])


# ── Vectorised bar-end search ─────────────────────────────────────────────────


def _find_range_bar_end(
    excursion: np.ndarray,
    dur_s: np.ndarray,
    target: float,
    min_s: float,
    max_s: float,
) -> Optional[int]:
    """
    Return the relative tick index where the range bar closes, or None.

    excursion[i] = (running_high[i] - running_low[i]) / bar_open
                   from bar start through tick (pos+i).
                   Already accounts for carry high/low from previous chunks.
    dur_s[i]     = elapsed seconds from bar start through tick (pos+i).

    Two triggers — earliest wins:
      1. Normal  — excursion >= target  AND  min_duration_seconds met
      2. Timeout — dur_s    >= max_duration_seconds

    No extreme trigger: (H-L)/open is bounded by 0 and ~1 in normal markets.
    A single tick cannot create a 5× excursion spike unlike a dollar-volume
    cumsum.  Time-capping handles pathological gaps (flash crashes, halts).
    """
    bar_end: Optional[int] = None

    idx = np.where(excursion >= target)[0]
    if len(idx):
        first = idx[0]
        min_met = np.where(dur_s[first:] >= min_s)[0]
        if len(min_met):
            bar_end = first + min_met[0]

    idx_to = np.where(dur_s >= max_s)[0]
    if len(idx_to):
        first_to = idx_to[0]
        if bar_end is None or first_to < bar_end:
            bar_end = first_to

    return bar_end


# ── Bar finalisation ──────────────────────────────────────────────────────────


def _finalize_range_bar(
    ts_start_ms: int,
    ts_end_ms: int,
    bar_open: float,  # first tick price — the range reference
    bar_high: float,  # running max over all ticks in bar
    bar_low: float,  # running min over all ticks in bar
    close_price: float,  # last tick price
    volume: float,
    dollar_volume: float,
    buy_dv: float,
    sell_dv: float,
    tick_count: int,
    excursion: float,  # (bar_high - bar_low) / bar_open — the bar signal
    market_params: dict,
) -> dict:
    """
    Build the finalised range bar dict.

    bar_size = (bar_high - bar_low) / bar_open — the relative price excursion.
    This is the value RangeBar.update_market_params reads and EMAs against
    target_range.  It is in the same units as target_range.

    Note: bar_high and bar_low here are the bar's OHLCV high/low — the highest
    and lowest trade prices seen within this bar.  bar_open is the first trade
    price of this bar.  These are all tick-exact values, not minute-boundary
    approximations.

    Tick-level enhancements vs minute RangeBar:
      - duration_seconds  (tick bars close at sub-minute precision)
      - duration_minutes  (for _calculate_bar_quality compatibility)
      - vwap              (exact Σ(p×q)/Σq)
      - dollar_volume     (exact Σ(p×q), not volume×close proxy)
      - tick_count        (number of aggTrades in bar)
      - buy/sell dollar volumes and tick_imbalance
    """
    start = _ms_to_dt(ts_start_ms)
    end = _ms_to_dt(ts_end_ms)
    dur_s = max(0.0, (end - start).total_seconds())
    prec = _get_precision(close_price)

    vwap = (
        round(dollar_volume / volume, prec) if volume > 0 else round(close_price, prec)
    )
    bar_return = round((close_price - bar_open) / bar_open, 6) if bar_open > 0 else 0.0
    price_range = round((bar_high - bar_low) / bar_open, 6) if bar_open > 0 else 0.0
    close_position = (
        round((close_price - bar_low) / (bar_high - bar_low), 6)
        if bar_high != bar_low
        else 0.5
    )
    total_dv = buy_dv + sell_dv
    imbalance = round((buy_dv - sell_dv) / total_dv, 6) if total_dv > 0 else 0.0

    return {
        "datetime": end,
        "datetime_start": start,
        "datetime_end": end,
        "open": round(bar_open, prec),
        "high": round(bar_high, prec),
        "low": round(bar_low, prec),
        "close": round(close_price, prec),
        "volume": round(volume, 6),
        "dollar_volume": round(dollar_volume, 2),
        "vwap": vwap,
        # bar_size = relative excursion = the range signal
        # This is what RangeBar.update_market_params reads for EMA adaptation.
        "bar_size": round(excursion, 8),
        "duration_seconds": round(dur_s, 1),
        "duration_minutes": round(dur_s / 60.0, 4),  # _calculate_bar_quality
        "tick_count": tick_count,
        "buy_dollar_volume": round(buy_dv, 2),
        "sell_dollar_volume": round(sell_dv, 2),
        "buy_sell_imbalance": imbalance,
        "bar_return": bar_return,
        "price_range": price_range,
        "close_position": close_position,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tick entry points (called by run.py)
# ═══════════════════════════════════════════════════════════════════════════════


def calibrate(
    bar_processor: RangeBar,
    csv_path: Path,
    gather_fn: Callable,
) -> dict:
    """
    Two-step calibration:

    Step 1: ticks → 1-min OHLCV → RangeBar.analyze_market_history()
            Extracts target_bars_per_day, ema_alpha, alpha_min/max,
            min/max_duration_minutes, regime metadata.
            These frequency/adaptation parameters are independent of
            signal units and are correctly computed from minute data.

    Step 2: Compute daily tick-native range from raw calibration ticks:
                daily_range = (day_high - day_low) / day_first_price
            Replace target_range with:
                median(daily_tick_range) / target_bars_per_day
            This is the tick-native signal target in the same units as the
            process loop accumulator — (H-L)/open as a relative fraction.

    The resulting target_range is directly fed to _find_range_bar_end and
    to RangeBar.update_market_params for EMA adaptation.
    """
    cal_p, cal_q, cal_ts_ms, _ = gather_fn(csv_path, ANALYSIS_LOOKBACK_DAYS)
    n_days = max(
        1, int((int(cal_ts_ms[-1]) - int(cal_ts_ms[0])) // (86_400 * 1_000)) + 1
    )
    logger.info(
        "  Calibrating range bars from %d ticks (%d days) ...",
        len(cal_ts_ms),
        n_days,
    )

    # Step 1 — frequency/adaptation params from minute analysis
    minute_bars = _ticks_to_minute_ohlcv_for_calibration(
        cal_p.astype(np.float64), cal_q.astype(np.float64), cal_ts_ms
    )
    market_params = bar_processor.analyze_market_history(minute_bars)
    del minute_bars
    gc.collect()

    # Step 2 — replace target_range with tick-native excursion target
    daily_ranges = _compute_daily_tick_range(cal_p.astype(np.float64), cal_ts_ms)
    del cal_p, cal_q, cal_ts_ms
    gc.collect()

    target_bpd = max(1.0, market_params.get("target_bars_per_day", 4.0))
    tick_target = float(np.median(daily_ranges)) / target_bpd
    market_params["target_range"] = tick_target

    # Tick-specific duration bounds (seconds)
    estimated_bar_seconds = 86_400.0 / target_bpd
    market_params["min_duration_seconds"] = TICK_MIN_DURATION_SECONDS
    market_params["max_duration_seconds"] = max(
        TICK_MAX_DURATION_FLOOR_SECONDS,
        min(TICK_MAX_DURATION_SECONDS, int(estimated_bar_seconds * 3)),
    )

    logger.info(
        "  Range calibration done — "
        "tick target=%.6f (%.4f%% excursion)  bars/day=%.1f  alpha=%.3f",
        tick_target,
        tick_target * 100,
        target_bpd,
        market_params["ema_alpha"],
    )
    return market_params


def process_chunk(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps_ms: np.ndarray,
    is_buyer_maker: np.ndarray,
    bar_processor: RangeBar,
    market_params: dict,
    recent: deque,
    open_bar_data: dict,
    update_fn: Callable,
) -> tuple[list, dict, dict, dict]:
    """
    Build range bars from one chunk of tick arrays.

    Algorithm
    ----------
    For each bar iteration:
      1.  Restore bar_open, bar_high, bar_low from open_bar_data carry
          (non-None when the bar straddles a chunk boundary).
          bar_open is set to prices[pos] for a fresh bar.
      2.  Compute the running excursion array over the search window:
              running_high = np.maximum.accumulate(p_slice, seeded with bar_high)
              running_low  = np.minimum.accumulate(p_slice, seeded with bar_low)
              excursion    = (running_high - running_low) / bar_open
          This is O(n) with no loop.
      3.  Call _find_range_bar_end() to locate the closing tick.
      4.  Finalise and call update_fn.
      5.  Start a fresh bar at the next tick.

    Cross-chunk carry
    -----------------
    bar_open  — the first tick price of the current bar.  Fixed for the
                bar's lifetime.  If the chunk ends mid-bar, it is stored
                in open_bar_data["bar_open"] and restored next chunk.
    bar_high  — running max over all ticks seen in this bar so far.
    bar_low   — running min over all ticks seen in this bar so far.

    These seed the np.maximum/minimum.accumulate calls on the next chunk,
    so the excursion is computed correctly as if the bar never split.

    No previous_price needed — range signal uses only within-bar prices.
    No leftover tick arrays — exhausted ticks are absorbed into open_bar_data.

    Returns (bars, market_params, open_bar_data, leftover={}).
    """
    prices_f = prices.astype(np.float64)
    qty_f = quantities.astype(np.float64)
    dv_all = prices_f * qty_f

    n = len(prices_f)
    pos = 0
    bars: list[dict] = []

    # ── Restore cross-chunk bar state ─────────────────────────────────────────
    ts_bar_start: Optional[int] = open_bar_data.get("ts_start_ms")
    bar_open: Optional[float] = open_bar_data.get("bar_open")
    bar_high: Optional[float] = open_bar_data.get("bar_high")
    bar_low: Optional[float] = open_bar_data.get("bar_low")
    bar_vol: float = float(open_bar_data.get("volume", 0.0))
    bar_dv: float = float(open_bar_data.get("dollar_volume", 0.0))
    bar_buy_dv: float = float(open_bar_data.get("buy_dollar_volume", 0.0))
    bar_sell_dv: float = float(open_bar_data.get("sell_dollar_volume", 0.0))
    bar_ticks: int = int(open_bar_data.get("tick_count", 0))

    if n == 0:
        return bars, market_params, open_bar_data, {}

    while pos < n:
        target = float(market_params["target_range"])
        min_s = float(
            market_params.get("min_duration_seconds", TICK_MIN_DURATION_SECONDS)
        )
        max_s = float(
            market_params.get("max_duration_seconds", TICK_MAX_DURATION_SECONDS)
        )

        # ── Establish bar anchor at bar start ─────────────────────────────────
        # bar_open is None only for a fresh bar (no carry from previous chunk).
        if bar_open is None:
            bar_open = float(prices_f[pos])
            bar_high = bar_open
            bar_low = bar_open
            ts_bar_start = int(timestamps_ms[pos])
            # Absorb the first tick of the new bar into OHLCV accumulators.
            # It does not generate an excursion (high == low == open → 0).
            bar_vol += float(qty_f[pos])
            bar_dv += float(dv_all[pos])
            if is_buyer_maker[pos]:
                bar_sell_dv += float(dv_all[pos])
            else:
                bar_buy_dv += float(dv_all[pos])
            bar_ticks += 1
            pos += 1
            if pos >= n:
                # Only one tick left and it was the bar opener — carry forward
                open_bar_data = {
                    "ts_start_ms": ts_bar_start,
                    "bar_open": bar_open,
                    "bar_high": bar_high,
                    "bar_low": bar_low,
                    "volume": bar_vol,
                    "dollar_volume": bar_dv,
                    "buy_dollar_volume": bar_buy_dv,
                    "sell_dollar_volume": bar_sell_dv,
                    "tick_count": bar_ticks,
                }
                return bars, market_params, open_bar_data, {}

        # ── Vectorised excursion search ───────────────────────────────────────
        window = min(_CUMSUM_WINDOW, n - pos)
        bar_end = None

        while bar_end is None and window <= (n - pos):
            p_slice = prices_f[pos : pos + window]
            ts_slice = timestamps_ms[pos : pos + window]

            # Seed running max/min with the bar's carry high/low
            # so the excursion is continuous across chunk boundaries.
            # np.maximum.accumulate starts from p_slice[0], but we need to
            # initialise from bar_high/bar_low carried from previous ticks.
            # Prepend the carry values as a single-element seed, then slice off.
            p_seeded_high = np.concatenate([[bar_high], p_slice])
            p_seeded_low = np.concatenate([[bar_low], p_slice])

            running_high = np.maximum.accumulate(p_seeded_high)[1:]  # shape: (window,)
            running_low = np.minimum.accumulate(p_seeded_low)[1:]  # shape: (window,)

            # Excursion relative to the bar's opening price — stationary signal
            if bar_open > 0:
                excursion = (running_high - running_low) / bar_open
            else:
                excursion = np.zeros(len(p_slice))

            dur_s = (ts_slice - ts_bar_start) / _MS_PER_S

            bar_end = _find_range_bar_end(excursion, dur_s, target, min_s, max_s)

            if bar_end is None:
                if window == n - pos:
                    break
                window = min(window * 2, n - pos)

        if bar_end is None:
            # Chunk exhausted — accumulate remaining ticks into open_bar_data
            p_rem = prices_f[pos:]
            q_rem = qty_f[pos:]
            dv_rem = dv_all[pos:]
            ibm_rem = is_buyer_maker[pos:]

            if len(p_rem) > 0:
                bar_high = max(bar_high, float(p_rem.max()))
                bar_low = min(bar_low, float(p_rem.min()))
                bar_vol += float(q_rem.sum())
                bar_dv += float(dv_rem.sum())
                bar_buy_dv += float(dv_rem[~ibm_rem].sum())
                bar_sell_dv += float(dv_rem[ibm_rem].sum())
                bar_ticks += len(p_rem)

            open_bar_data = {
                "ts_start_ms": ts_bar_start,
                "bar_open": bar_open,
                "bar_high": bar_high,
                "bar_low": bar_low,
                "volume": bar_vol,
                "dollar_volume": bar_dv,
                "buy_dollar_volume": bar_buy_dv,
                "sell_dollar_volume": bar_sell_dv,
                "tick_count": bar_ticks,
            }
            return bars, market_params, open_bar_data, {}

        # ── Bar completed at bar_end ──────────────────────────────────────────
        end = bar_end + 1  # exclusive
        p_bar = prices_f[pos : pos + end]
        q_bar = qty_f[pos : pos + end]
        dv_bar = dv_all[pos : pos + end]
        ibm_bar = is_buyer_maker[pos : pos + end]
        ts_bar = timestamps_ms[pos : pos + end]

        # Final bar metrics — merge carry with this chunk's slice
        final_high = max(bar_high, float(p_bar.max()))
        final_low = min(bar_low, float(p_bar.min()))
        final_vol = bar_vol + float(q_bar.sum())
        final_dv = bar_dv + float(dv_bar.sum())
        final_buy = bar_buy_dv + float(dv_bar[~ibm_bar].sum())
        final_sell = bar_sell_dv + float(dv_bar[ibm_bar].sum())
        final_ticks = bar_ticks + len(p_bar)
        # Excursion at the exact closing tick — the bar_size value the EMA reads
        final_excursion = (final_high - final_low) / bar_open if bar_open > 0 else 0.0

        bar = _finalize_range_bar(
            ts_start_ms=ts_bar_start,
            ts_end_ms=int(ts_bar[-1]),
            bar_open=bar_open,
            bar_high=final_high,
            bar_low=final_low,
            close_price=float(p_bar[-1]),
            volume=final_vol,
            dollar_volume=final_dv,
            buy_dv=final_buy,
            sell_dv=final_sell,
            tick_count=final_ticks,
            excursion=final_excursion,
            market_params=market_params,
        )
        bars.append(bar)
        recent.append(bar)

        market_params = update_fn(bar_processor, market_params, bar, recent)

        # Reset all bar carry state for the next bar
        pos += end
        bar_open = None
        bar_high = None
        bar_low = None
        ts_bar_start = None
        bar_vol = 0.0
        bar_dv = 0.0
        bar_buy_dv = 0.0
        bar_sell_dv = 0.0
        bar_ticks = 0

    # Chunk fully consumed, no partial bar
    open_bar_data = {}
    return bars, market_params, open_bar_data, {}