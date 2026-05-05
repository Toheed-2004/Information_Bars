"""
tick_hybrid.py — Hybrid bars from Binance aggTrades tick data.

Closes when BOTH accumulated dollar-volume AND accumulated realized volatility
first reach their respective EMA-adapted targets — identical AND logic to the
minute-bar HybridBar, with both signals now computed at tick resolution.

Tick-native signals
--------------------
Dollar-volume : Σ(price × qty)       — exact, computed tick-by-tick.
Realized vol  : Σ|log(p_i/p_{i-1})| — tick-native, captures intra-minute
                price path invisible to the minute pipeline.

Why the original minute-bucketed volatility was wrong
------------------------------------------------------
The previous tick_hybrid bucketed ticks into 1-minute windows and used the
bucket close for the volatility component.  Binance's 1-minute close IS the
last trade price of that minute, so the bucketed signal reconstructed the
identical numbers that the minute pipeline receives — making the comparison
vacuous.

Tick realized volatility sees every intra-minute move.  A quiet minute and a
volatile-but-mean-reverting minute look identical to the minute pipeline; the
tick pipeline correctly distinguishes them.

Architecture
-------------
Dollar-volume : pure tick cumsum, identical to tick_dollar.
Realized vol  : pure tick cumsum Σ|log-return|, identical to tick_volatility.
Both accumulators run simultaneously in a single tick loop.

The bar closes when:
    cum_dv  >= target_dollar_volume  AND
    cum_rv  >= target_volatility     AND
    min_duration_seconds met
OR:
    max_duration_seconds exceeded (time-cap)

Selective EMA update (from HybridBar.update_market_params)
------------------------------------------------------------
Only the EMA whose threshold was actually triggered updates after each bar.
A time-capped bar (neither threshold reached) updates neither EMA.
This anti-feedback-spiral logic is preserved unchanged — HybridBar's
update_market_params reads "bar_size" (dollar vol) and "bar_volatility"
(realized vol) from the finalized bar dict.

Calibration
-----------
target_bars_per_day, ema_alpha, duration bounds:
    from HybridBar.analyze_market_history() fed with 1-min OHLCV.
target_dollar_volume:
    from analyze_market_history() — uses median daily Σ(close×vol) / bpd.
    This is a reasonable proxy; exact tick Σ(p×q) is very close.
target_volatility:
    REPLACED with median(daily_realized_vol) / target_bars_per_day
    using the same _compute_daily_realized_vol() as tick_volatility.

Carry state across chunk boundaries
-------------------------------------
market_params["previous_price"]   — for log-return chain
                                     (same carry pattern as tick_dollar)
open_bar_data   — partial bar (carry_rv, carry_dv, OHLCV, ts_start_ms, ...)
leftover        — unconsumed tick arrays; prepended by run.py

No partial_minute carry — no minute bucketing.

Signature
----------
calibrate(bar_processor, csv_path, gather_fn)  -> market_params
process_chunk(prices, quantities, timestamps_ms, is_buyer_maker,
              bar_processor, market_params, recent,
              open_bar_data, update_fn)
              -> (bars, market_params, open_bar_data, leftover)
"""

import gc
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from bitpredict.common.logging import get_logger
from bitpredict.data.custom_bars.constants import (
    ANALYSIS_LOOKBACK_DAYS,
    EXTREME_THRESHOLD_MULTIPLIER,
)
from bitpredict.data.custom_bars.bar_types.hybrid import HybridBar

from tick_volatility import (
    _ms_to_dt,
    _get_precision,
    _ticks_to_minute_ohlcv_for_calibration,
    _compute_daily_realized_vol,
    TICK_MIN_DURATION_SECONDS,
    TICK_MAX_DURATION_SECONDS,
    TICK_MAX_DURATION_FLOOR_SECONDS,
    _CUMSUM_WINDOW,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Bar finalisation
# ─────────────────────────────────────────────────────────────────────────────


def _finalize_hybrid_bar(
    ts_start_ms: int,
    ts_end_ms: int,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    volume: float,
    dollar_volume: float,  # Σ(p×q) ticks only — no carry
    buy_dv: float,
    sell_dv: float,
    tick_count: int,
    bar_size_dv: float,  # dollar vol including carry — for EMA
    realized_vol: float,  # Σ|log-return|
) -> dict:
    """
    Finalise a hybrid bar.

    bar_size      = dollar volume including carry (what HybridBar EMA reads)
    bar_volatility = realized vol (what HybridBar EMA reads for vol target)
    vwap          = dollar_volume (ticks only, no carry) / volume
    """
    start = _ms_to_dt(ts_start_ms)
    end = _ms_to_dt(ts_end_ms)
    dur_s = max(0.0, (end - start).total_seconds())
    prec = _get_precision(close_price)

    # VWAP uses tick-only dollar_volume (excludes carry) — correct
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
        "dollar_volume": round(dollar_volume, 2),  # ticks only
        "vwap": vwap,
        "bar_size": round(bar_size_dv, 2),  # dv + carry — HybridBar EMA input
        "bar_volatility": round(realized_vol, 8),  # HybridBar vol EMA input
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


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def calibrate(
    bar_processor: HybridBar,
    csv_path: Path,
    gather_fn: Callable,
) -> dict:
    """
    Two-step calibration:
      1. ticks → 1-min OHLCV → HybridBar.analyze_market_history() for
         target_bars_per_day, target_dollar_volume, ema_alpha, duration bounds.
      2. Replace target_volatility with tick-native realized vol target.
    """
    cal_p, cal_q, cal_ts_ms, _ = gather_fn(csv_path, ANALYSIS_LOOKBACK_DAYS)
    n_days = max(
        1, int((int(cal_ts_ms[-1]) - int(cal_ts_ms[0])) // (86_400 * 1_000)) + 1
    )
    logger.info(
        "  Calibrating hybrid bars from %d ticks (%d days) ...", len(cal_ts_ms), n_days
    )

    minute_bars = _ticks_to_minute_ohlcv_for_calibration(
        cal_p.astype(np.float64), cal_q.astype(np.float64), cal_ts_ms
    )
    market_params = bar_processor.analyze_market_history(minute_bars)
    del minute_bars
    gc.collect()

    # Replace minute-based target_volatility with tick-native realized vol
    daily_rv = _compute_daily_realized_vol(cal_p.astype(np.float64), cal_ts_ms)
    del cal_p, cal_q, cal_ts_ms
    gc.collect()

    target_bpd = max(1.0, market_params.get("target_bars_per_day", 8.0))
    tick_vol_target = float(np.median(daily_rv)) / target_bpd
    market_params["target_volatility"] = tick_vol_target

    estimated_bar_seconds = 86_400.0 / target_bpd
    market_params["min_duration_seconds"] = TICK_MIN_DURATION_SECONDS
    market_params["max_duration_seconds"] = max(
        TICK_MAX_DURATION_FLOOR_SECONDS,
        min(TICK_MAX_DURATION_SECONDS, int(estimated_bar_seconds * 3)),
    )
    market_params["previous_price"] = None
    logger.info(
        "  Hybrid calibration done — "
        "target_dv=$%.0f  tick_vol=%.6f (%.4f%%)  bars/day=%.1f  alpha=%.3f",
        market_params["target_dollar_volume"],
        tick_vol_target,
        tick_vol_target * 100,
        target_bpd,
        market_params["ema_alpha"],
    )
    return market_params


def process_chunk(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps_ms: np.ndarray,
    is_buyer_maker: np.ndarray,
    bar_processor: HybridBar,
    market_params: dict,
    recent: deque,
    open_bar_data: dict,
    update_fn: Callable,
) -> tuple[list, dict, dict, dict]:
    """
    Build hybrid bars from one chunk of tick arrays.

    Both signals accumulate simultaneously in a single tick loop:
      cum_dv[i]  = Σ(p×q) from bar start through tick (pos+i)
      cum_rv[i]  = Σ|log-return| from bar start through tick (pos+i)

    Closing condition (earliest wins):
      1. AND trigger — cum_dv >= target_dv AND cum_rv >= target_vol AND min_dur met
      2. Timeout     — elapsed seconds >= max_duration_seconds

    Returns (bars, market_params, open_bar_data, leftover).
    """
    prices_f = prices.astype(np.float64)
    qty_f = quantities.astype(np.float64)
    dv_all = prices_f * qty_f

    n = len(prices_f)
    pos = 0
    bars: list[dict] = []

    # ── Restore cross-chunk state ─────────────────────────────────────────────
    prev_price: Optional[float] = market_params.get("previous_price")
    carry_rv: float = float(open_bar_data.get("carry_rv", 0.0))
    carry_dv_bar: float = float(open_bar_data.get("carry_dv", 0.0))
    ts_bar_start: Optional[int] = open_bar_data.get("ts_start_ms")
    bar_open: Optional[float] = open_bar_data.get("open")
    bar_high: Optional[float] = open_bar_data.get("high")
    bar_low: Optional[float] = open_bar_data.get("low")
    bar_vol: float = float(open_bar_data.get("volume", 0.0))
    bar_tick_dv: float = float(open_bar_data.get("dollar_volume", 0.0))  # ticks only
    bar_buy_dv: float = float(open_bar_data.get("buy_dollar_volume", 0.0))
    bar_sell_dv: float = float(open_bar_data.get("sell_dollar_volume", 0.0))
    bar_ticks: int = int(open_bar_data.get("tick_count", 0))

    # ── Seed previous_price ───────────────────────────────────────────────────
    if prev_price is None:
        if n == 0:
            return bars, market_params, {}, {}
        prev_price = float(prices_f[0])
        if ts_bar_start is None:
            ts_bar_start = int(timestamps_ms[0])
            bar_open = bar_high = bar_low = prev_price
        else:
            bar_high = max(bar_high, prev_price)
            bar_low = min(bar_low, prev_price)
        bar_vol += float(qty_f[0])
        bar_tick_dv += float(dv_all[0])
        if is_buyer_maker[0]:
            bar_sell_dv += float(dv_all[0])
        else:
            bar_buy_dv += float(dv_all[0])
        bar_ticks += 1
        pos = 1
        market_params["previous_price"] = prev_price

    while pos < n:
        target_dv = float(market_params["target_dollar_volume"])
        target_vol = float(market_params["target_volatility"])
        min_s = float(
            market_params.get("min_duration_seconds", TICK_MIN_DURATION_SECONDS)
        )
        max_s = float(
            market_params.get("max_duration_seconds", TICK_MAX_DURATION_SECONDS)
        )
        extreme_thr = target_dv * EXTREME_THRESHOLD_MULTIPLIER

        if ts_bar_start is None:
            ts_bar_start = int(timestamps_ms[pos])

        # ── Vectorised search ─────────────────────────────────────────────────
        window = min(_CUMSUM_WINDOW, n - pos)
        bar_end = None

        while bar_end is None and window <= (n - pos):
            p_slice = prices_f[pos : pos + window]
            ts_slice = timestamps_ms[pos : pos + window]
            q_slice = qty_f[pos : pos + window]
            dv_slice = dv_all[pos : pos + window]

            # Dollar-volume cumsum
            cum_dv = np.cumsum(dv_slice) + carry_dv_bar

            # Realized vol cumsum
            p_ext = np.where(
                np.concatenate([[prev_price], p_slice]) > 0,
                np.concatenate([[prev_price], p_slice]),
                1e-10,
            )
            log_rets = np.abs(np.diff(np.log(p_ext)))
            cum_rv = np.cumsum(log_rets) + carry_rv

            dur_s = (ts_slice - ts_bar_start) / 1_000.0

            # AND trigger: both thresholds met + min_duration
            dv_met_idx = np.where(cum_dv >= target_dv)[0]
            vol_met_idx = np.where(cum_rv >= target_vol)[0]

            if len(dv_met_idx) and len(vol_met_idx):
                # Both thresholds crossed — find the later crossing
                both_met = max(dv_met_idx[0], vol_met_idx[0])
                min_met = np.where(dur_s[both_met:] >= min_s)[0]
                if len(min_met):
                    bar_end = both_met + min_met[0]

            # Extreme dollar-volume trigger (single-signal override)
            ext_idx = np.where(cum_dv >= extreme_thr)[0]
            if len(ext_idx):
                first_ext = ext_idx[0]
                if dur_s[first_ext] >= min_s * 0.5:
                    if bar_end is None or first_ext < bar_end:
                        bar_end = first_ext

            # Timeout trigger
            to_idx = np.where(dur_s >= max_s)[0]
            if len(to_idx):
                first_to = to_idx[0]
                if bar_end is None or first_to < bar_end:
                    bar_end = first_to

            if bar_end is None:
                if window == n - pos:
                    break
                window = min(window * 2, n - pos)

        if bar_end is None:
            # Accumulate remaining ticks into open_bar_data
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
                carry_dv_bar += float(dv_rem.sum())

                if bar_open is None:
                    bar_open = float(p_rem[0])
                    bar_high = float(p_rem.max())
                    bar_low = float(p_rem.min())
                else:
                    bar_high = max(bar_high, float(p_rem.max()))
                    bar_low = min(bar_low, float(p_rem.min()))

                bar_vol += float(q_rem.sum())
                bar_tick_dv += float(dv_rem.sum())
                bar_buy_dv += float(dv_rem[~ibm_rem].sum())
                bar_sell_dv += float(dv_rem[ibm_rem].sum())
                bar_ticks += len(p_rem)
                prev_price = float(p_rem[-1])

            market_params["previous_price"] = prev_price
            open_bar_data = {
                "carry_rv": carry_rv,
                "carry_dv": carry_dv_bar,
                "ts_start_ms": ts_bar_start,
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "volume": bar_vol,
                "dollar_volume": bar_tick_dv,
                "buy_dollar_volume": bar_buy_dv,
                "sell_dollar_volume": bar_sell_dv,
                "tick_count": bar_ticks,
            }
            return bars, market_params, open_bar_data, {}

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
        final_tick_dv = bar_tick_dv + float(dv_bar.sum())  # ticks only (no carry)
        final_buy = bar_buy_dv + float(dv_bar[~ibm_bar].sum())
        final_sell = bar_sell_dv + float(dv_bar[ibm_bar].sum())
        final_ticks = bar_ticks + len(p_bar)
        final_dv_with_carry = float(cum_dv[bar_end])  # includes carry_dv_bar
        final_rv = float(cum_rv[bar_end])  # includes carry_rv

        bar = _finalize_hybrid_bar(
            ts_start_ms=ts_bar_start,
            ts_end_ms=int(ts_bar[-1]),
            open_price=final_open,
            high_price=final_high,
            low_price=final_low,
            close_price=float(p_bar[-1]),
            volume=final_vol,
            dollar_volume=final_tick_dv,
            buy_dv=final_buy,
            sell_dv=final_sell,
            tick_count=final_ticks,
            bar_size_dv=final_dv_with_carry,
            realized_vol=final_rv,
        )
        bars.append(bar)
        recent.append(bar)

        market_params = update_fn(bar_processor, market_params, bar, recent)

        prev_price = float(p_bar[-1])
        market_params["previous_price"] = prev_price
        # Reset bar carry state
        carry_rv = 0.0
        carry_dv_bar = 0.0
        ts_bar_start = None
        bar_open = bar_high = bar_low = None
        bar_vol = bar_tick_dv = bar_buy_dv = bar_sell_dv = 0.0
        bar_ticks = 0

        pos += end

    # Chunk fully consumed
    market_params["previous_price"] = prev_price
    open_bar_data = {}
    return bars, market_params, open_bar_data, {}