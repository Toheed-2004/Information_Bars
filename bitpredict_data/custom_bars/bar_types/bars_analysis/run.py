"""
run.py — Build custom bars from a single merged Binance aggTrades CSV.

Usage
-----
    python run.py <bar_type> <merged_csv> [output_csv]

    bar_type    'dollar' | 'volume' | 'renko' | 'volatility' | 'range' | 'hybrid'
    merged_csv  single merged yearly aggTrades CSV
    output_csv  destination (default: tick_<bar_type>_bars.csv)

Binance aggTrades column layout (no header, millisecond timestamps):
    0  agg_trade_id
    1  price
    2  quantity
    3  first_trade_id
    4  last_trade_id
    5  timestamp_ms    millisecond Unix epoch
    6  is_buyer_maker  TRUE = seller aggressor
    7  is_best_match   ignored

Memory model
------------
Streamed in CHUNK_SIZE-row chunks (~40 MB each) — file is never fully loaded.
Calibration runs in a separate read pass BEFORE the main loop.

Per-type carry state across chunk boundaries
--------------------------------------------
    dollar      leftover tick arrays (unconsumed ticks at chunk end)
    volume      leftover tick arrays (unconsumed ticks at chunk end)
    renko       leftover tick arrays
                renko_reference persisted in market_params["renko_reference"]
    volatility  open_bar_data (partial bar) — no leftover
    range       open_bar_data (partial bar) — no leftover
    hybrid      open_bar_data (partial bar) — no leftover

Tick-native signals (volatility / range / hybrid)
--------------------------------------------------
All three use Σ|log(p_i/p_{i-1})| accumulated tick-by-tick — pure tick loops
with no minute bucketing.  previous_price is persisted in market_params.
open_bar_data carries the partial bar across chunks; when the chunk ends
mid-bar there are no leftover ticks (they are absorbed into open_bar_data).
"""

import gc
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from bitpredict.common.logging import get_logger, setup_logging
from bitpredict.data.custom_bars.constants import (
    EMA_MONITORING_INTERVAL,
    EMA_OPTIMIZATION_INTERVAL,
    MIN_BARS_FOR_QUALITY,
)
from bitpredict.data.custom_bars.bar_types.bars_analysis.tick_dollar import (
    TickDollarBar,
)
from bitpredict.data.custom_bars.bar_types.volatility import VolatilityBar
from bitpredict.data.custom_bars.bar_types.volume import VolumeBar
from bitpredict.data.custom_bars.bar_types.hybrid import HybridBar
from bitpredict.data.custom_bars.bar_types.renko import RenkoBar
from bitpredict.data.custom_bars.bar_types.range_bar import RangeBar

import tick_dollar as dollar_mod
import tick_volatility as vol_mod
import tick_volume as volume_mod
import tick_hybrid as hybrid_mod
import tick_renko as renko_mod
import tick_range as range_mod

setup_logging("data.bars")
logger = get_logger(__name__)

_BINANCE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp_ms",
    "is_buyer_maker",
    "is_best_match",
]

CHUNK_SIZE = 1_000_000  # ~40 MB for the four essential columns


# ── I/O helpers ───────────────────────────────────────────────────────────────


def _iter_csv_chunks(path: str) -> Iterator[tuple]:
    """
    Yield (prices_f32, quantities_f32, timestamps_ms_i64, is_buyer_maker_bool)
    one CHUNK_SIZE rows at a time.
    Timestamp resolution auto-detected: 16-digit → microseconds → //1000.
    """
    ms_divisor: Optional[int] = None

    for chunk in pd.read_csv(
        path,
        header=None,
        names=_BINANCE_COLUMNS,
        dtype={
            "agg_trade_id": "int64",
            "price": "float32",
            "quantity": "float32",
            "first_trade_id": "int64",
            "last_trade_id": "int64",
            "timestamp_ms": "int64",
            "is_buyer_maker": "str",
            "is_best_match": "str",
        },
        chunksize=CHUNK_SIZE,
    ):
        chunk["is_buyer_maker"] = chunk["is_buyer_maker"].str.upper() == "TRUE"
        chunk = chunk[(chunk["price"] > 0) & (chunk["quantity"] > 0)]
        if chunk.empty:
            continue

        if ms_divisor is None:
            sample = int(chunk["timestamp_ms"].iloc[0])
            ms_divisor = 1_000 if sample > 9_999_999_999_999 else 1

        ts = chunk["timestamp_ms"].to_numpy(dtype=np.int64)
        if ms_divisor != 1:
            ts = ts // ms_divisor

        yield (
            chunk["price"].to_numpy(dtype=np.float32),
            chunk["quantity"].to_numpy(dtype=np.float32),
            ts,
            chunk["is_buyer_maker"].to_numpy(dtype=bool),
        )
        del chunk
        gc.collect()


def _gather_calibration_data(
    csv_path: Path, lookback_days: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Independent read pass — collect up to lookback_days of tick data.
    Called BEFORE the main chunk loop; main iterator starts from byte 0.
    Returns (prices_f64, quantities_f64, timestamps_ms_i64, is_buyer_maker_bool).
    """
    lookback_ms = lookback_days * 86_400 * 1_000
    cal_p, cal_q, cal_ts, cal_ibm = [], [], [], []
    cutoff_ms: Optional[int] = None

    for prices, quantities, timestamps, ibm in _iter_csv_chunks(str(csv_path)):
        if cutoff_ms is None:
            cutoff_ms = int(timestamps[0]) + lookback_ms

        beyond = timestamps > cutoff_ms
        if beyond.any():
            within = ~beyond
            cal_p.append(prices[within].astype(np.float64))
            cal_q.append(quantities[within].astype(np.float64))
            cal_ts.append(timestamps[within])
            cal_ibm.append(ibm[within])
            del prices, quantities, timestamps, ibm
            gc.collect()
            break

        cal_p.append(prices.astype(np.float64))
        cal_q.append(quantities.astype(np.float64))
        cal_ts.append(timestamps)
        cal_ibm.append(ibm)
        del prices, quantities, timestamps, ibm
        gc.collect()

    return (
        np.concatenate(cal_p),
        np.concatenate(cal_q),
        np.concatenate(cal_ts),
        np.concatenate(cal_ibm),
    )


def _write_bars(bars: list, output_csv: str, write_header: bool) -> bool:
    """Append finalised bars to output CSV. Returns True if anything was written."""
    if not bars:
        return False
    df = pd.DataFrame(bars)
    for col in ("datetime", "datetime_start", "datetime_end"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    df.to_csv(
        output_csv, mode="w" if write_header else "a", header=write_header, index=False
    )
    del df
    return True


# ── Post-bar EMA / monitoring / optimisation ──────────────────────────────────


def _update_and_adapt(
    bar_processor, market_params: dict, bar: dict, recent: deque
) -> dict:
    """
    Called after every bar closes:
      1. bar_processor.update_market_params()  — type-specific EMA
      2. Self-monitoring every EMA_MONITORING_INTERVAL bars
      3. Auto-optimisation every EMA_OPTIMIZATION_INTERVAL bars
      4. Parameter bounds enforcement
    """
    market_params = bar_processor.update_market_params(market_params, bar)
    market_params["monitoring_counter"] = market_params.get("monitoring_counter", 0) + 1
    market_params["bars_since_optimization"] = (
        market_params.get("bars_since_optimization", 0) + 1
    )

    if market_params["monitoring_counter"] >= EMA_MONITORING_INTERVAL:
        market_params = bar_processor._perform_self_monitoring(market_params)
        market_params["monitoring_counter"] = 0

    if market_params["bars_since_optimization"] >= EMA_OPTIMIZATION_INTERVAL:
        if (
            bar_processor._can_optimize(market_params)
            and len(recent) >= MIN_BARS_FOR_QUALITY
        ):
            quality = bar_processor._calculate_bar_quality(list(recent), market_params)
            market_params = bar_processor._apply_optimization_strategy(
                market_params, quality, list(recent)
            )
        market_params["bars_since_optimization"] = 0

    return bar_processor._enforce_parameter_bounds(market_params)


# ── Main ──────────────────────────────────────────────────────────────────────


def run(bar_type: str, csv_path: str, output_csv: str) -> int:
    """
    Calibrate then stream the merged CSV, writing bars to output_csv.
    Returns total bars created.
    """
    p = Path(csv_path)
    if not p.exists() or p.is_dir():
        logger.error("Expected a single merged CSV file, got: %s", csv_path)
        return 0

    if bar_type == "dollar":
        bar_processor = TickDollarBar("binance", "btc")
        mod = dollar_mod
    elif bar_type == "volume":
        bar_processor = VolumeBar("binance", "btc")
        mod = volume_mod
    elif bar_type == "renko":
        bar_processor = RenkoBar("binance", "btc")
        mod = renko_mod
    elif bar_type == "volatility":
        bar_processor = VolatilityBar("binance", "btc")
        mod = vol_mod
    elif bar_type == "range":
        bar_processor = RangeBar("binance", "btc")
        mod = range_mod
    elif bar_type == "hybrid":
        bar_processor = HybridBar("binance", "btc")
        mod = hybrid_mod
    else:
        logger.error(
            "Unknown bar type '%s'. Use: dollar | volume | renko | volatility | range | hybrid",
            bar_type,
        )
        return 0

    recent = deque(maxlen=MIN_BARS_FOR_QUALITY)
    total_bars = 0
    write_header = True
    t0 = time.time()

    # ── Step 1: Calibrate ─────────────────────────────────────────────────────
    logger.info("Calibrating %s bars from %s ...", bar_type, p.name)
    market_params = mod.calibrate(bar_processor, p, _gather_calibration_data)

    # ── Step 2: Per-type carry state ──────────────────────────────────────────
    # dollar / volume / renko: leftover tick arrays
    leftover: dict = {}
    # volatility / range / hybrid: partial bar accumulated mid-chunk
    open_bar_data: dict = {}
    # renko_reference persisted in market_params by tick_renko.calibrate

    # ── Step 3: Stream and build bars ─────────────────────────────────────────
    logger.info("Processing %s → %s", p.name, output_csv)

    for chunk_idx, (prices, quantities, timestamps_ms, is_buyer_maker) in enumerate(
        _iter_csv_chunks(str(p))
    ):
        if bar_type == "dollar":
            if leftover:
                prices = np.concatenate([leftover["prices"], prices])
                quantities = np.concatenate([leftover["quantities"], quantities])
                timestamps_ms = np.concatenate(
                    [leftover["timestamps_ms"], timestamps_ms]
                )
                is_buyer_maker = np.concatenate(
                    [leftover["is_buyer_maker"], is_buyer_maker]
                )
                leftover = {}

            bars, market_params, leftover = mod.process_chunk(
                prices,
                quantities,
                timestamps_ms,
                is_buyer_maker,
                bar_processor,
                market_params,
                recent,
                _update_and_adapt,
            )

        elif bar_type == "volume":
            if leftover:
                prices = np.concatenate([leftover["prices"], prices])
                quantities = np.concatenate([leftover["quantities"], quantities])
                timestamps_ms = np.concatenate(
                    [leftover["timestamps_ms"], timestamps_ms]
                )
                leftover = {}

            bars, market_params, leftover = mod.process_chunk(
                prices,
                quantities,
                timestamps_ms,
                bar_processor,
                market_params,
                recent,
                _update_and_adapt,
            )

        elif bar_type == "renko":
            if leftover:
                prices = np.concatenate([leftover["prices"], prices])
                quantities = np.concatenate([leftover["quantities"], quantities])
                timestamps_ms = np.concatenate(
                    [leftover["timestamps_ms"], timestamps_ms]
                )
                is_buyer_maker = np.concatenate(
                    [leftover["is_buyer_maker"], is_buyer_maker]
                )
                leftover = {}

            bars, market_params, leftover = mod.process_chunk(
                prices,
                quantities,
                timestamps_ms,
                is_buyer_maker,
                bar_processor,
                market_params,
                recent,
                _update_and_adapt,
            )

        elif bar_type == "volatility":
            # No leftover: chunk-end ticks are absorbed into open_bar_data.
            # previous_price in market_params maintains the log-return chain.
            bars, market_params, open_bar_data, _ = mod.process_chunk(
                prices,
                quantities,
                timestamps_ms,
                is_buyer_maker,
                bar_processor,
                market_params,
                recent,
                open_bar_data,
                _update_and_adapt,
            )

        elif bar_type == "range":
            bars, market_params, open_bar_data, _ = mod.process_chunk(
                prices,
                quantities,
                timestamps_ms,
                is_buyer_maker,
                bar_processor,
                market_params,
                recent,
                open_bar_data,
                _update_and_adapt,
            )

        else:  # hybrid
            bars, market_params, open_bar_data, _ = mod.process_chunk(
                prices,
                quantities,
                timestamps_ms,
                is_buyer_maker,
                bar_processor,
                market_params,
                recent,
                open_bar_data,
                _update_and_adapt,
            )

        del prices, quantities, timestamps_ms, is_buyer_maker
        gc.collect()

        if bars:
            if _write_bars(bars, output_csv, write_header):
                write_header = False
            total_bars += len(bars)
        del bars

        if (chunk_idx + 1) % 10 == 0:
            logger.info(
                "  chunk %d — %d bars so far (%.1fs elapsed)",
                chunk_idx + 1,
                total_bars,
                time.time() - t0,
            )

    logger.info(
        "═══ DONE: %d %s bars → %s in %.1fs ═══",
        total_bars,
        bar_type,
        output_csv,
        time.time() - t0,
    )
    return total_bars


if __name__ == "__main__":

    _csv_path = r"D:\bitpredict\data\custom_bars\bar_types\bars_analysis\data\merged_tick_data.csv"

    if len(sys.argv) < 3:

        for _bar_type in [ "dollar","hybrid","range","renko","volatility",""]:

            # _bar_type = "volatility"

            _output_csv = f"tick_{_bar_type}_bars.csv"
            run(_bar_type, _csv_path, _output_csv)

    else:
        _bar_type = sys.argv[1].lower()
        _csv_path = sys.argv[2]
        _output_csv = sys.argv[3] if len(sys.argv) > 3 else f"tick_{_bar_type}_bars.csv"

        run(_bar_type, _csv_path, _output_csv)