"""
tick_volatility.py — Volatility bars from Binance aggTrades tick data.

Tick-native signal: realized volatility  Σ |log(p_i / p_{i-1})|
-----------------------------------------------------------------
Each bar closes when the sum of absolute log-returns across every consecutive
tick pair WITHIN the bar reaches the EMA-adapted target_volatility.

Why not minute-bucketed close-to-close returns
-----------------------------------------------
Bucketing ticks to 1-minute windows and using the bucket close as the
"close price" reconstructs the identical numbers Binance stores as 1-minute
OHLCV bars.  The comparison between the tick and minute pipelines would then
measure nothing: same signal, same values, bars closing at the same minute
boundaries.

Realized volatility is the tick-native equivalent.  It sees every intra-
minute price move — reversals, momentum bursts, microstructure noise — that
the minute pipeline cannot access.  A minute with three 0.05% moves that
cancel out has close-to-close return ≈ 0 but realized vol = 0.15%.  The tick
pipeline treats these differently; the minute pipeline cannot.

Calibration
-----------
target_bars_per_day, ema_alpha, and duration bounds are extracted from
VolatilityBar.analyze_market_history() fed with synthetic 1-minute OHLCV
(built from calibration ticks).  These frequency/adaptation parameters are
independent of signal units and are correctly computed from minute data.

target_volatility is then REPLACED with the tick-native value:
    median(daily_realized_vol) / target_bars_per_day
where daily_realized_vol = Σ|log(p_i/p_{i-1})| over all ticks in a day.

Carry state across chunk boundaries
-------------------------------------
market_params["previous_price"]
    Price of the last tick processed.  Needed to compute the log-return for
    the first tick of the next chunk.  None before the very first tick.

open_bar_data   dict with keys:
    carry_rv            float   realized vol accumulated before this chunk
    ts_start_ms         int     first tick timestamp of the current bar
    open                float   bar opening price
    high                float   running high
    low                 float   running low
    volume              float   Σ qty
    dollar_volume       float   Σ(price×qty)
    buy_dollar_volume   float   Σ(price×qty) for buyer-aggressor ticks
    sell_dollar_volume  float   Σ(price×qty) for seller-aggressor ticks
    tick_count          int     ticks processed so far

leftover        dict with raw tick arrays not yet consumed.
                run.py prepends these to the next chunk.

No partial_minute carry — there is no minute bucketing.

Signature change vs old tick_volatility
-----------------------------------------
Old: process_chunk(prices, quantities, timestamps_ms,
                   bar_processor, market_params, recent,
                   open_bar_data, partial_minute, update_fn)
     -> (bars, market_params, open_bar_data, partial_minute)

New: process_chunk(prices, quantities, timestamps_ms, is_buyer_maker,
                   bar_processor, market_params, recent,
                   open_bar_data, update_fn)
     -> (bars, market_params, open_bar_data, leftover)

run.py is updated to match.
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
# BUG-FIX 3: import actual VolatilityBar (not abstract BaseBar).
# BUG-FIX 12: _MS_PER_DAY used in calibrate() n_days calculation.
from .volatility_bars import VolatilityBar
_MS_PER_DAY = 86_400 * 1_000  # milliseconds per day
from common import *

logger = get_logger(__name__)

_MINUTE_MS = 60_000
_MS_PER_S = 1_000

TICK_MIN_DURATION_SECONDS = 10
TICK_MAX_DURATION_SECONDS = 28_800
TICK_MAX_DURATION_FLOOR_SECONDS = 300
_CUMSUM_WINDOW = 500_000


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (also imported by tick_hybrid.py)
# ─────────────────────────────────────────────────────────────────────────────


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


def _ticks_to_minute_ohlcv_for_calibration(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps_ms: np.ndarray,
) -> list[dict]:
    """
    1-minute OHLCV dicts for analyze_market_history() — calibration only.
    The last (possibly incomplete) bucket is included; acceptable for analysis.
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


def _compute_daily_realized_vol(
    prices: np.ndarray,
    timestamps_ms: np.ndarray,
) -> np.ndarray:
    """
    Daily Σ|log(p_i/p_{i-1})| over all ticks.
    Returns an array of per-day totals for calibration.
    """
    p = prices.astype(np.float64)
    valid = p > 0
    p = p[valid]
    ts = timestamps_ms[valid]
    if len(p) < 2:
        return np.array([0.0])
    log_rets = np.abs(np.diff(np.log(p)))
    day_idx = ts[1:] // (86_400 * 1_000)
    unique_days = np.unique(day_idx)
    return np.array(
        [float(log_rets[day_idx == d].sum()) for d in unique_days],
        dtype=np.float64,
    )


def _find_rv_bar_end(
    cum_rv: np.ndarray,
    dur_s: np.ndarray,
    target: float,
    min_s: float,
    max_s: float,
) -> Optional[int]:
    """
    Earliest tick where the realized-volatility bar closes.
    cum_rv[i] = total realized vol from bar start through tick (pos+i),
                already including carry_rv from previous chunks.
    """
    bar_end: Optional[int] = None

    idx = np.where(cum_rv >= target)[0]
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


def _make_rv_bar(
    ts_start_ms: int,
    ts_end_ms: int,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    volume: float,
    dollar_volume: float,
    buy_dv: float,
    sell_dv: float,
    tick_count: int,
    realized_vol: float,
) -> dict:
    """Finalise a realized-volatility bar into the output dict."""
    start = _ms_to_dt(ts_start_ms)
    end = _ms_to_dt(ts_end_ms)
    dur_s = max(0.0, (end - start).total_seconds())
    prec = _get_precision(close_price)

    vwap = (
        round(dollar_volume / volume, prec) if volume > 0 else round(close_price, prec)
    )
    bar_return = (
        round((close_price - open_price) / open_price, 6) if open_price > 0 else 0.0
    )
    price_range = (
        round((high_price - low_price) / open_price, 6) if open_price > 0 else 0.0
    )
    close_position = (
        round((close_price - low_price) / (high_price - low_price), 6)
        if high_price != low_price
        else 0.5
    )
    total_dv = buy_dv + sell_dv
    imbalance = round((buy_dv - sell_dv) / total_dv, 6) if total_dv > 0 else 0.0

    return {
        "datetime": end,
        "datetime_start": start,
        "datetime_end": end,
        "open": round(open_price, prec),
        "high": round(high_price, prec),
        "low": round(low_price, prec),
        "close": round(close_price, prec),
        "volume": round(volume, 6),
        "dollar_volume": round(dollar_volume, 2),
        "vwap": vwap,
        "bar_size": round(realized_vol, 8),  # Σ|log-return|
        "bar_volatility": round(realized_vol, 8),  # alias for HybridBar EMA
        "duration_seconds": round(dur_s, 1),
        "duration_minutes": round(dur_s / 60.0, 4),  # for _calculate_bar_quality
        "tick_count": tick_count,
        "buy_dollar_volume": round(buy_dv, 2),
        "sell_dollar_volume": round(sell_dv, 2),
        "buy_sell_imbalance": imbalance,
        "bar_return": bar_return,
        "price_range": price_range,
        "close_position": close_position,
    }


def _rv_process_chunk_inner(
    prices_f: np.ndarray,  # float64
    qty_f: np.ndarray,  # float64
    dv_all: np.ndarray,  # float64  price * qty
    timestamps_ms: np.ndarray,  # int64
    is_buyer_maker: np.ndarray,  # bool
    bar_processor,
    market_params: dict,
    recent: deque,
    update_fn: Callable,
    target_key: str,  # "target_volatility" or "target_range"
    open_bar_data: dict,
) -> tuple[list, dict, dict]:
    """
    Core realized-volatility tick loop shared by tick_volatility and tick_range.

    Returns (bars, market_params, open_bar_data_or_empty).
    Leftover is handled by the caller — this function consumes the full array.
    The caller is responsible for prepending leftover ticks before calling.
    """
    n = len(prices_f)
    pos = 0
    bars: list[dict] = []

    # ── Restore cross-chunk bar state ─────────────────────────────────────────
    prev_price: Optional[float] = market_params.get("previous_price")
    carry_rv: float = float(open_bar_data.get("carry_rv", 0.0))
    ts_bar_start: Optional[int] = open_bar_data.get("ts_start_ms")
    bar_open: Optional[float] = open_bar_data.get("open")
    bar_high: Optional[float] = open_bar_data.get("high")
    bar_low: Optional[float] = open_bar_data.get("low")
    bar_vol: float = float(open_bar_data.get("volume", 0.0))
    bar_dv: float = float(open_bar_data.get("dollar_volume", 0.0))
    bar_buy_dv: float = float(open_bar_data.get("buy_dollar_volume", 0.0))
    bar_sell_dv: float = float(open_bar_data.get("sell_dollar_volume", 0.0))
    bar_ticks: int = int(open_bar_data.get("tick_count", 0))

    # ── Seed previous_price from first tick if never set ─────────────────────
    if prev_price is None:
        if n == 0:
            return bars, market_params, {}
        prev_price = float(prices_f[0])
        # Absorb first tick into open bar; it has no log-return partner yet.
        if ts_bar_start is None:
            ts_bar_start = int(timestamps_ms[0])
            bar_open = bar_high = bar_low = prev_price
        else:
            bar_high = max(bar_high, prev_price)
            bar_low = min(bar_low, prev_price)
        bar_vol += float(qty_f[0])
        bar_dv += float(dv_all[0])
        if is_buyer_maker[0]:
            bar_sell_dv += float(dv_all[0])
        else:
            bar_buy_dv += float(dv_all[0])
        bar_ticks += 1
        pos = 1
        market_params["previous_price"] = prev_price

    while pos < n:
        target = float(market_params[target_key])
        min_s = float(
            market_params.get("min_duration_seconds", TICK_MIN_DURATION_SECONDS)
        )
        max_s = float(
            market_params.get("max_duration_seconds", TICK_MAX_DURATION_SECONDS)
        )

        if ts_bar_start is None:
            ts_bar_start = int(timestamps_ms[pos])

        # ── Vectorised search over expanding window ───────────────────────────
        window = min(_CUMSUM_WINDOW, n - pos)
        bar_end = None

        while bar_end is None and window <= (n - pos):
            p_slice = prices_f[pos : pos + window]
            ts_slice = timestamps_ms[pos : pos + window]

            p_ext = np.where(
                np.concatenate([[prev_price], p_slice]) > 0,
                np.concatenate([[prev_price], p_slice]),
                1e-10,
            )
            log_rets = np.abs(np.diff(np.log(p_ext)))
            cum_rv = np.cumsum(log_rets) + carry_rv
            dur_s = (ts_slice - ts_bar_start) / _MS_PER_S

            bar_end = _find_rv_bar_end(cum_rv, dur_s, target, min_s, max_s)

            if bar_end is None:
                if window == n - pos:
                    break
                window = min(window * 2, n - pos)

        if bar_end is None:
            # Chunk exhausted — accumulate remaining into open_bar_data
            p_rem = prices_f[pos:]
            q_rem = qty_f[pos:]
            dv_rem = dv_all[pos:]
            ibm_rem = is_buyer_maker[pos:]

            if len(p_rem) > 0:
                p_ext_rem = np.where(
                    np.concatenate([[prev_price], p_rem]) > 0,
                    np.concatenate([[prev_price], p_rem]),
                    1e-10,
                )
                carry_rv += float(np.abs(np.diff(np.log(p_ext_rem))).sum())

                if bar_open is None:
                    bar_open = float(p_rem[0])
                    bar_high = float(p_rem.max())
                    bar_low = float(p_rem.min())
                else:
                    bar_high = max(bar_high, float(p_rem.max()))
                    bar_low = min(bar_low, float(p_rem.min()))

                bar_vol += float(q_rem.sum())
                bar_dv += float(dv_rem.sum())
                bar_buy_dv += float(dv_rem[~ibm_rem].sum())
                bar_sell_dv += float(dv_rem[ibm_rem].sum())
                bar_ticks += len(p_rem)
                prev_price = float(p_rem[-1])

            market_params["previous_price"] = prev_price
            return (
                bars,
                market_params,
                {
                    "carry_rv": carry_rv,
                    "ts_start_ms": ts_bar_start,
                    "open": bar_open,
                    "high": bar_high,
                    "low": bar_low,
                    "volume": bar_vol,
                    "dollar_volume": bar_dv,
                    "buy_dollar_volume": bar_buy_dv,
                    "sell_dollar_volume": bar_sell_dv,
                    "tick_count": bar_ticks,
                },
            )

        # ── Bar completed ─────────────────────────────────────────────────────
        end = bar_end + 1
        p_bar = prices_f[pos : pos + end]
        q_bar = qty_f[pos : pos + end]
        dv_bar = dv_all[pos : pos + end]
        ibm_bar = is_buyer_maker[pos : pos + end]
        ts_bar = timestamps_ms[pos : pos + end]

        final_open = bar_open if bar_open is not None else float(p_bar[0])
        final_high = (
            max(bar_high, float(p_bar.max()))
            if bar_high is not None
            else float(p_bar.max())
        )
        final_low = (
            min(bar_low, float(p_bar.min()))
            if bar_low is not None
            else float(p_bar.min())
        )
        final_vol = bar_vol + float(q_bar.sum())
        final_dv = bar_dv + float(dv_bar.sum())
        final_buy = bar_buy_dv + float(dv_bar[~ibm_bar].sum())
        final_sell = bar_sell_dv + float(dv_bar[ibm_bar].sum())
        final_ticks = bar_ticks + len(p_bar)
        final_rv = float(cum_rv[bar_end])  # includes carry_rv

        bar = _make_rv_bar(
            ts_start_ms=ts_bar_start,
            ts_end_ms=int(ts_bar[-1]),
            open_price=final_open,
            high_price=final_high,
            low_price=final_low,
            close_price=float(p_bar[-1]),
            volume=final_vol,
            dollar_volume=final_dv,
            buy_dv=final_buy,
            sell_dv=final_sell,
            tick_count=final_ticks,
            realized_vol=final_rv,
        )
        bars.append(bar)
        recent.append(bar)

        market_params = update_fn(bar_processor, market_params, bar, recent)

        prev_price = float(p_bar[-1])
        market_params["previous_price"] = prev_price

        # Reset bar carry state for next bar
        carry_rv = 0.0
        ts_bar_start = None
        bar_open = bar_high = bar_low = None
        bar_vol = bar_dv = bar_buy_dv = bar_sell_dv = 0.0
        bar_ticks = 0

        pos += end

    # Chunk fully consumed with no partial bar remaining
    market_params["previous_price"] = prev_price
    return bars, market_params, {}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def calibrate(bar_processor, csv_path: Path, gather_fn: Callable) -> dict:
    """
    Tick-native calibration for volatility bars.

    - target_volatility   = median(daily_realized_vol) / target_bars_per_day
                            computed from raw ticks (already done before, kept)
    - target_bars_per_day = f(log-return entropy, market efficiency)
    - ema_alpha           = f(daily realized-vol CV, regime stability, noise)
    No minute bucketing anywhere.
    """
    from common.constants import (
        ANALYSIS_LOOKBACK_DAYS, BARS_PER_DAY_MIN, BARS_PER_DAY_MAX,
        VOLATILITY_BASE_FREQUENCY, SLOW_BAR_FREQUENCY_MULTIPLIER,
        DURATION_ESTIMATED_MULTIPLIER, ALPHA_MIN_ABSOLUTE, ALPHA_MAX_ABSOLUTE,
    )
    from common.logging import get_logger
    logger = get_logger(__name__)

    TICK_MIN_DURATION_SECONDS       = 10
    TICK_MAX_DURATION_SECONDS       = 28_800
    TICK_MAX_DURATION_FLOOR_SECONDS = 300

    cal_p, cal_q, cal_ts_ms, _ = gather_fn(csv_path, ANALYSIS_LOOKBACK_DAYS)
    n_days = max(1, int((int(cal_ts_ms[-1]) - int(cal_ts_ms[0])) // _MS_PER_DAY) + 1)
    logger.info("  Calibrating volatility bars from %d ticks (%d days) ...", len(cal_ts_ms), n_days)

    log_ret = _tick_log_returns(cal_p.astype(np.float64))
    if len(log_ret) < 100:
        logger.warning("  Insufficient log-returns — using defaults")
        del cal_p, cal_q, cal_ts_ms
        return bar_processor._get_default_params()

    # ── information multiplier ────────────────────────────────────────────────
    ret_entropy  = _tick_entropy(log_ret)
    # BUG-FIX 15: fixed seed for reproducible tick calibration
    _rng_calib = np.random.default_rng(seed=42)
    rand_entropy = _tick_entropy(_rng_calib.normal(0, np.std(log_ret), len(log_ret)))
    information_ratio      = ret_entropy / rand_entropy if rand_entropy > 0 else 1.0
    information_multiplier = max(0.5, min(2.0, information_ratio))

    # ── activity multiplier ───────────────────────────────────────────────────
    market_eff          = _tick_market_efficiency(cal_p, cal_q)
    activity_percentile = max(0.0, min(1.0, market_eff))
    activity_multiplier = 0.5 + activity_percentile

    # ── target_bars_per_day ───────────────────────────────────────────────────
    target_bpd = max(BARS_PER_DAY_MIN,
                     min(BARS_PER_DAY_MAX,
                         VOLATILITY_BASE_FREQUENCY
                         * information_multiplier * activity_multiplier
                         / SLOW_BAR_FREQUENCY_MULTIPLIER))

    # ── tick-native daily realized vol ────────────────────────────────────────
    # Σ|log(p_i/p_{i-1})| per day — same as _compute_daily_realized_vol
    day_idx    = _tick_daily_split(cal_ts_ms)
    prices_f64 = cal_p.astype(np.float64)
    # log-return per tick aligned to the tick it lands on (index 1..N)
    abs_lr_full = np.abs(np.diff(np.log(np.where(prices_f64 > 0, prices_f64, np.nan))))
    abs_lr_full = np.nan_to_num(abs_lr_full, nan=0.0)
    daily_rv, _ = _tick_daily_metric(abs_lr_full, day_idx[1:])

    median_daily_rv = float(np.median(daily_rv))
    mad_rv          = float(np.median(np.abs(daily_rv - median_daily_rv)))
    vol_cv          = mad_rv / median_daily_rv if median_daily_rv > 0 else 0.3

    target_volatility = median_daily_rv / max(1.0, target_bpd)

    # ── ema_alpha ─────────────────────────────────────────────────────────────
    regime_stability = _tick_regime_stability(log_ret)
    market_noise     = _tick_market_noise(log_ret)
    alpha_min, alpha_max, ema_alpha = _alpha_from_cv(vol_cv, regime_stability, market_noise)
    alpha_min = max(alpha_min, ALPHA_MIN_ABSOLUTE)
    alpha_max = min(alpha_max, ALPHA_MAX_ABSOLUTE)
    ema_alpha = max(alpha_min, min(alpha_max, ema_alpha))

    min_s, max_s = _duration_seconds_from_bpd(
        target_bpd, TICK_MIN_DURATION_SECONDS,
        TICK_MAX_DURATION_FLOOR_SECONDS, TICK_MAX_DURATION_SECONDS,
        DURATION_ESTIMATED_MULTIPLIER)

    del cal_p, cal_q, cal_ts_ms

    logger.info(
        "  Volatility calibration done — tick_target=%.6f (%.4f%%)  "
        "bars/day=%.1f  alpha=%.3f  [TICK-NATIVE]",
        target_volatility, target_volatility * 100, target_bpd, ema_alpha)

    return {
        "target_volatility":      target_volatility,
        "ema_alpha":              ema_alpha,
        "alpha_min":              alpha_min,
        "alpha_max":              alpha_max,
        "target_bars_per_day":    target_bpd,
        "min_duration_seconds":   min_s,
        "max_duration_seconds":   max_s,
        "volatility_cv":          vol_cv,
        "median_daily_volatility":median_daily_rv,
        "regime_stability":       regime_stability,
        "market_noise":           market_noise,
        "information_ratio":      information_ratio,
        "market_efficiency":      market_eff,
        "previous_price":         None,
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
    bar_processor: VolatilityBar,
    market_params: dict,
    recent: deque,
    open_bar_data: dict,
    update_fn: Callable,
) -> tuple[list, dict, dict, dict]:
    """
    Build realized-volatility bars from one chunk.
    Returns (bars, market_params, open_bar_data, leftover).
    leftover is {} when the chunk was fully consumed with no partial bar.
    """
    prices_f = prices.astype(np.float64)
    qty_f = quantities.astype(np.float64)
    dv_all = prices_f * qty_f

    bars, market_params, open_bar_data = _rv_process_chunk_inner(
        prices_f,
        qty_f,
        dv_all,
        timestamps_ms,
        is_buyer_maker,
        bar_processor,
        market_params,
        recent,
        update_fn,
        target_key="target_volatility",
        open_bar_data=open_bar_data,
    )

    # open_bar_data is non-empty only when chunk was exhausted mid-bar.
    # In that case there are no leftover ticks — they were all absorbed.
    # leftover is only non-empty when a bar completed and ticks remain after it,
    # which the inner loop handles by continuing; on normal exit pos == n.
    leftover: dict = {}
    return bars, market_params, open_bar_data, leftover

# ── BUG-FIX 1 (companion): TickVolatilityBar class for registry ──────────────
class TickVolatilityBar(VolatilityBar):
    """
    Tick-level volatility bar processor.
    Inherits VolatilityBar for EMA and quality-assessment methods.
    The tick-native calibrate() and process_chunk() entry points use
    realized volatility Σ|log(p_i/p_{i-1})| instead of minute close-to-close.
    """
    pass