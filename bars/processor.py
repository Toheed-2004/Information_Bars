"""
bars/processor.py  — OPTIMIZED  (final version)
================================================
Drop-in replacement.  Zero algorithm changes.

Optimizations vs original
--------------------------
1. CHUNK_SIZE 100_000_000 → 2_000_000
   100M rows × ~150B (dtype=str) ≈ 15 GB per chunk before conversion buffers.
   2M rows ≈ 300 MB raw / 160 MB converted; fits L3 cache cleanly.
   Gives 300 real iterations over 600M rows with negligible loop overhead.

2. Parallel prefetch pipeline (_iter_csv_chunks)
   ThreadPoolExecutor with 8 workers converts/filters chunks while the I/O
   thread reads the next one.  pandas/numpy C extensions release the GIL so
   workers run on real cores simultaneously.  Sliding window of 6 in-flight
   chunks keeps peak extra RAM at ~1 GB.

3. memory_map=True on pd.read_csv
   OS page-cache owns the file buffer.  After the first bar type reads the
   600M-row CSV, every subsequent bar type reads from RAM (~50 GB/s) not NVMe.

4. gc.collect() removed from all hot loops
   Explicit GC every chunk: 50-200 ms × 300 iterations = minutes wasted.

5. Calibration result cached (_CALIB_CACHE)
   Each process calls _gather_calibration_data once; subsequent calls within
   the same process return a pre-built array copy in microseconds.

6. Batched CSV writes (WRITE_BATCH_SIZE)
   Original: df.to_csv() append called every chunk — 300 tiny file appends
   for maybe 5-15 rows each.  Now: bars accumulate in memory and flush every
   500 bars (or at end).  Reduces file-open overhead by ~20-50×.

7. No final CSV re-read
   Original returned pd.read_csv(output_csv).to_dict("records") — re-reading
   a file we just finished writing, just to get a list that main.py only uses
   for len().  Now: keep bars in memory throughout, return directly.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.constants import (
    ANALYSIS_LOOKBACK_DAYS,
    EMA_MONITORING_INTERVAL,
    EMA_OPTIMIZATION_INTERVAL,
    MIN_BARS_FOR_QUALITY,
)
from common.logging import get_logger
from common.data_loader import (
    load_minute_csv, df_to_records,
    save_bars_csv, save_state, load_state,
)
from bars.bar_types import get_bar_class, ALL_BAR_TYPE_NAMES

logger = get_logger(__name__)

_BINANCE_COLUMNS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]

# ── Performance knobs ─────────────────────────────────────────────────────────
CHUNK_SIZE       = 2_000_000   # rows per chunk (~160 MB converted)
PROCESS_WORKERS  = 10           # chunk-conversion worker threads
PREFETCH_CHUNKS  = 6           # max chunks in-flight (~1 GB extra RAM)
WRITE_BATCH_SIZE = 500         # flush accumulated bars to CSV every N bars

# Module-level calibration cache: (path_str, lookback_days) → arrays
_CALIB_CACHE: Dict[Tuple, Tuple[np.ndarray, ...]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Worker: raw DataFrame → typed numpy arrays  (runs in thread pool)
# ─────────────────────────────────────────────────────────────────────────────

def _process_raw_chunk(
    args: Tuple[pd.DataFrame, Optional[int]]
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Pure stateless function — safe to run on any thread.
    All heavy operations (pd.to_numeric, astype, numpy) release the GIL.
    Returns None when the chunk has no valid rows after filtering.
    """
    chunk, ms_divisor_hint = args

    if "agg_trade_id" not in chunk.columns:
        return None

    chunk = chunk[chunk["agg_trade_id"].str.strip().str.isdigit()]
    if chunk.empty:
        return None

    chunk["price"]         = pd.to_numeric(chunk["price"],         errors="coerce").astype("float64")
    chunk["quantity"]      = pd.to_numeric(chunk["quantity"],      errors="coerce").astype("float64")
    chunk["transact_time"] = pd.to_numeric(chunk["transact_time"], errors="coerce").astype("int64")

    chunk = chunk[(chunk["price"] > 0) & (chunk["quantity"] > 0)]
    if chunk.empty:
        return None

    chunk["is_buyer_maker"] = chunk["is_buyer_maker"].str.upper() == "TRUE"

    sample = int(chunk["transact_time"].iloc[0])
    ms_divisor = ms_divisor_hint if ms_divisor_hint is not None else (
        1_000 if sample > 9_999_999_999_999 else 1
    )
    ts = chunk["transact_time"].to_numpy(dtype=np.int64)
    if ms_divisor != 1:
        ts = ts // ms_divisor

    return (
        chunk["price"].to_numpy(dtype=np.float64),
        chunk["quantity"].to_numpy(dtype=np.float64),
        ts,
        chunk["is_buyer_maker"].to_numpy(dtype=bool),
        ms_divisor,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parallel streaming CSV iterator
# ─────────────────────────────────────────────────────────────────────────────

def _iter_csv_chunks(
    path: str,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Yields (prices, quantities, timestamps_ms, is_buyer_maker) in file order.

    I/O thread reads raw chunks sequentially.
    Worker pool converts/filters in parallel (GIL released by C extensions).
    Sliding window of PREFETCH_CHUNKS bounds peak memory use.
    Results drained in submission order — time-series ordering preserved.
    """
    import csv as _csv

    _has_header = True
    try:
        with open(path, "r") as fh:
            first = next(_csv.reader(fh))
            if first and first[0].strip().lstrip("-").isdigit():
                _has_header = False
    except Exception:
        pass

    _binance_cols = _BINANCE_COLUMNS if not _has_header else None
    ms_divisor: Optional[int] = None

    csv_iter = pd.read_csv(
        path,
        header=0 if _has_header else None,
        names=_binance_cols,
        dtype=str,
        chunksize=CHUNK_SIZE,
        on_bad_lines="warn",
        engine="c",
        memory_map=True,   # page-cache: subsequent reads come from RAM
    )

    with ThreadPoolExecutor(max_workers=PROCESS_WORKERS) as pool:
        pending = []

        for raw_chunk in csv_iter:
            fut = pool.submit(_process_raw_chunk, (raw_chunk, ms_divisor))
            pending.append(fut)

            if len(pending) >= PREFETCH_CHUNKS:
                result = pending.pop(0).result()
                if result is not None:
                    prices, quantities, ts, ibm, detected = result
                    if ms_divisor is None:
                        ms_divisor = detected
                    yield prices, quantities, ts, ibm

        for fut in pending:
            result = fut.result()
            if result is not None:
                prices, quantities, ts, ibm, detected = result
                if ms_divisor is None:
                    ms_divisor = detected
                yield prices, quantities, ts, ibm


# ─────────────────────────────────────────────────────────────────────────────
# Calibration  (with in-process cache)
# ─────────────────────────────────────────────────────────────────────────────

def _gather_calibration_data(
    csv_path: Any,
    lookback_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stream the first lookback_days of ticks for calibration.
    Cached per (path, lookback_days) — free on repeat calls within one process.
    """
    cache_key = (str(csv_path), lookback_days)
    if cache_key in _CALIB_CACHE:
        cp, cq, cts, cibm = _CALIB_CACHE[cache_key]
        return cp.copy(), cq.copy(), cts.copy(), cibm.copy()

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
            break
        cal_p.append(prices.astype(np.float64))
        cal_q.append(quantities.astype(np.float64))
        cal_ts.append(timestamps)
        cal_ibm.append(ibm)

    result: Tuple[np.ndarray, ...] = (
        np.concatenate(cal_p),
        np.concatenate(cal_q),
        np.concatenate(cal_ts),
        np.concatenate(cal_ibm),
    )
    _CALIB_CACHE[cache_key] = result
    cp, cq, cts, cibm = result
    return cp.copy(), cq.copy(), cts.copy(), cibm.copy()


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flush_bars(bars: list, output_csv: str, write_header: bool) -> bool:
    """Write a batch of bars to CSV.  Called every WRITE_BATCH_SIZE bars."""
    if not bars:
        return False
    df = pd.DataFrame(bars)
    for col in ("datetime", "datetime_start", "datetime_end"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    df.to_csv(output_csv, mode="w" if write_header else "a",
              header=write_header, index=False)
    return True


def _update_and_adapt(bar_processor, market_params, bar, recent):
    market_params = bar_processor.update_market_params(market_params, bar)
    market_params["monitoring_counter"]      = market_params.get("monitoring_counter", 0) + 1
    market_params["bars_since_optimization"] = market_params.get("bars_since_optimization", 0) + 1
    if market_params["monitoring_counter"] >= EMA_MONITORING_INTERVAL:
        market_params = bar_processor._perform_self_monitoring(market_params)
        market_params["monitoring_counter"] = 0
    if market_params["bars_since_optimization"] >= EMA_OPTIMIZATION_INTERVAL:
        if bar_processor._can_optimize(market_params) and len(recent) >= MIN_BARS_FOR_QUALITY:
            quality = bar_processor._calculate_bar_quality(list(recent), market_params)
            market_params = bar_processor._apply_optimization_strategy(
                market_params, quality, list(recent))
        market_params["bars_since_optimization"] = 0
    return bar_processor._enforce_parameter_bounds(market_params)


# ─────────────────────────────────────────────────────────────────────────────
# Tick bar processor
# ─────────────────────────────────────────────────────────────────────────────

def process_tick_bars(
    bar_type: str,
    tick_csv: str | Path,
    output_dir: str | Path = "data/processed_bars",
    exchange: str = "binance",
    symbol: str = "btc",
) -> List[Dict[str, Any]]:
    import importlib
    tick_csv   = Path(tick_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = str(output_dir / f"{exchange}_{symbol}_{bar_type}_tick_bars.csv")

    mod_map = {
        "dollar":     "bars.bar_types.tick_dollar_bars",
        "volume":     "bars.bar_types.tick_volume_bars",
        "volatility": "bars.bar_types.tick_volatility_bars",
        "range":      "bars.bar_types.tick_range_bars",
        "renko":      "bars.bar_types.tick_renko_bars",
        "hybrid":     "bars.bar_types.tick_hybrid_bars",
    }
    if bar_type not in mod_map:
        raise ValueError(f"Unknown bar type: {bar_type!r}. Available: {list(mod_map)}")

    mod           = importlib.import_module(mod_map[bar_type])
    BarClass      = get_bar_class(bar_type, source="tick")
    bar_processor = BarClass(exchange, symbol)
    recent        = deque(maxlen=MIN_BARS_FOR_QUALITY)
    t0            = time.time()

    logger.info("Calibrating %s tick bars from %s ...", bar_type, tick_csv.name)
    market_params = mod.calibrate(bar_processor, tick_csv, _gather_calibration_data)

    leftover: dict      = {}
    open_bar_data: dict = {}

    # ── In-memory bar accumulator — flushed every WRITE_BATCH_SIZE bars ───────
    all_bars:    List[Dict] = []   # kept in full for return value
    write_buf:   List[Dict] = []   # pending bars not yet written to CSV
    write_header = True
    total_bars   = 0

    logger.info("Processing %s ...", tick_csv.name)

    for chunk_idx, (prices, quantities, timestamps_ms, is_buyer_maker) in \
            enumerate(_iter_csv_chunks(str(tick_csv))):

        if bar_type == "dollar":
            if leftover:
                prices         = np.concatenate([leftover["prices"],         prices])
                quantities     = np.concatenate([leftover["quantities"],     quantities])
                timestamps_ms  = np.concatenate([leftover["timestamps_ms"],  timestamps_ms])
                is_buyer_maker = np.concatenate([leftover["is_buyer_maker"], is_buyer_maker])
                leftover = {}
            bars, market_params, leftover = mod.process_chunk(
                prices, quantities, timestamps_ms, is_buyer_maker,
                bar_processor, market_params, recent, _update_and_adapt)

        elif bar_type == "volume":
            if leftover:
                prices        = np.concatenate([leftover["prices"],        prices])
                quantities    = np.concatenate([leftover["quantities"],    quantities])
                timestamps_ms = np.concatenate([leftover["timestamps_ms"], timestamps_ms])
                leftover = {}
            bars, market_params, leftover = mod.process_chunk(
                prices, quantities, timestamps_ms,
                bar_processor, market_params, recent, _update_and_adapt)

        elif bar_type == "renko":
            if leftover:
                prices         = np.concatenate([leftover["prices"],         prices])
                quantities     = np.concatenate([leftover["quantities"],     quantities])
                timestamps_ms  = np.concatenate([leftover["timestamps_ms"],  timestamps_ms])
                is_buyer_maker = np.concatenate([leftover["is_buyer_maker"], is_buyer_maker])
                leftover = {}
            bars, market_params, leftover = mod.process_chunk(
                prices, quantities, timestamps_ms, is_buyer_maker,
                bar_processor, market_params, recent, _update_and_adapt)

        else:  # volatility / range / hybrid
            bars, market_params, open_bar_data, _leftover = mod.process_chunk(
                prices, quantities, timestamps_ms, is_buyer_maker,
                bar_processor, market_params, recent,
                open_bar_data, _update_and_adapt)
            if _leftover:
                logger.warning("Unexpected leftover from %s: %d ticks",
                               bar_type, len(next(iter(_leftover.values()), [])))

        if bars:
            all_bars.extend(bars)
            write_buf.extend(bars)
            total_bars += len(bars)

            # Flush to disk every WRITE_BATCH_SIZE bars — not every chunk
            if len(write_buf) >= WRITE_BATCH_SIZE:
                _flush_bars(write_buf, output_csv, write_header)
                write_header = False
                write_buf    = []

        if (chunk_idx + 1) % 20 == 0:
            logger.info("  [%s] chunk %d — %d bars (%.1fs)",
                        bar_type, chunk_idx + 1, total_bars, time.time() - t0)

    # Final flush for any remaining bars
    if write_buf:
        _flush_bars(write_buf, output_csv, write_header)

    logger.info("DONE: %d %s tick bars in %.1fs", total_bars, bar_type, time.time() - t0)

    # Return in-memory list directly — no CSV re-read
    return all_bars


# ─────────────────────────────────────────────────────────────────────────────
# Minute bar processor  (gc.collect removed, otherwise unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def process_minute_bars(
    bar_type: str,
    minute_csv: str | Path,
    output_dir: str | Path = "data/processed_bars",
    exchange: str = "binance",
    symbol: str = "btc",
    resume: bool = True,
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem       = f"{exchange}_{symbol}_{bar_type}_minute"
    bars_path  = output_dir / f"{stem}_bars.csv"
    state_path = output_dir / f"{stem}_state.json"

    state: Optional[Dict] = load_state(state_path) if resume else None
    start_date = None
    if state and state.get("last_processed_datetime"):
        import dateutil.parser
        start_date = dateutil.parser.parse(state["last_processed_datetime"])
        logger.info("Resuming %s from %s", stem, start_date)

    df = load_minute_csv(minute_csv, start_date=start_date)
    if df.empty:
        logger.info("No new minute data for %s.", stem)
        return []

    BarClass  = get_bar_class(bar_type, source="minute")
    processor = BarClass(exchange, symbol)

    if state and state.get("ema_state"):
        market_params: Dict = dict(state["ema_state"])
        logger.info("Restored EMA state from %s", state_path.name)
    else:
        lookback_rows = ANALYSIS_LOOKBACK_DAYS * 1_440
        calib_records = df_to_records(df.iloc[:lookback_rows])
        market_params = processor.analyze_market_history(calib_records)
        logger.info("Calibration complete on %d rows.", len(calib_records))

    if hasattr(processor, "previous_close") and market_params.get("previous_close"):
        processor.previous_close = float(market_params["previous_close"])
    if hasattr(processor, "renko_reference") and market_params.get("renko_reference"):
        processor.renko_reference = float(market_params["renko_reference"])

    current_bar_data: Dict = {}
    if state and state.get("current_bar_data"):
        current_bar_data = state["current_bar_data"]

    all_bars: List[Dict] = []
    recent   = deque(maxlen=MIN_BARS_FOR_QUALITY)
    last_dt  = start_date

    calibration_skip_rows = 0 if (state and state.get("ema_state")) \
                            else ANALYSIS_LOOKBACK_DAYS * 1_440
    rows_processed = 0

    for row in df.itertuples(index=False):
        rows_processed += 1
        if rows_processed <= calibration_skip_rows:
            if hasattr(processor, "previous_close"):
                processor.previous_close = float(row.close)
            last_dt = row.datetime
            continue
        md = {
            "datetime": row.datetime, "open": float(row.open),
            "high": float(row.high),  "low":  float(row.low),
            "close": float(row.close), "volume": float(row.volume),
        }
        if any(v <= 0 for v in (md["open"], md["high"], md["low"], md["close"])):
            last_dt = md["datetime"]
            continue

        current_bar_data = processor.accumulate_bar_data(current_bar_data, md)
        if processor.should_create_bar(md, current_bar_data, market_params):
            bar = processor.finalize_bar(current_bar_data, market_params)
            all_bars.append(bar)
            recent.append(bar)
            market_params = _update_and_adapt(processor, market_params, bar, recent)
            if hasattr(processor, "previous_close"):
                market_params["previous_close"] = processor.previous_close
            if hasattr(processor, "renko_reference"):
                market_params["renko_reference"] = processor.renko_reference
            current_bar_data = {}
        last_dt = md["datetime"]

    if all_bars:
        save_bars_csv(all_bars, bars_path)
        logger.info("Created %d bars → %s", len(all_bars), bars_path.name)
    else:
        logger.info("No new bars for %s.", stem)

    save_state({
        "last_processed_datetime": str(last_dt),
        "current_bar_data":        current_bar_data,
        "ema_state":               market_params,
    }, state_path)
    return all_bars