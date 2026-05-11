"""
bars/processor.py
-----------------
Bar processing engine — mirrors run.py (tick) and utils.py (minute) exactly.

Minute pipeline  →  process_minute_bars()   mirrors bitpredict utils.py
Tick pipeline    →  process_tick_bars()      mirrors bitpredict run.py

Key design points
-----------------
- Tick calibration uses the first ANALYSIS_LOOKBACK_DAYS of TICK data directly
  via _gather_calibration_data() — no minute CSV used anywhere.
- Chunk-based streaming for tick data — never loads the full CSV into memory.
- Per-type carry state (leftover ticks / open_bar_data) preserved across chunks
  exactly as run.py does it.
"""
from __future__ import annotations

import gc
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

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
CHUNK_SIZE = 1_000_000*5


def _iter_csv_chunks(path: str) -> Iterator[tuple]:
    ms_divisor: Optional[int] = None
    # BUG-FIX 13: Detect whether the CSV has a header row before filtering.
    # Binance raw aggTrade CSVs may lack a header (columns are positional).
    # If the first cell of the first row is not a valid integer agg_trade_id,
    # we re-read with explicit column names to avoid silently dropping all data.
    import csv as _csv
    _has_header = True
    try:
        with open(path, "r") as _fh:
            _first = next(_csv.reader(_fh))
            if _first and _first[0].strip().lstrip("-").isdigit():
                _has_header = False  # first row is data, not a header
    except Exception:
        pass

    _binance_cols = _BINANCE_COLUMNS if not _has_header else None
    for chunk in pd.read_csv(
        path,
        header=0 if _has_header else None,
        names=_binance_cols,
        dtype=str,
        chunksize=CHUNK_SIZE,
    ):
        # Guard: if agg_trade_id column is missing fall back gracefully
        if "agg_trade_id" not in chunk.columns:
            continue
        chunk = chunk[chunk["agg_trade_id"].str.strip().str.isdigit()]
        if chunk.empty:
            continue
        # BUG-FIX 5: use float64 (not float32) to prevent ~0.001 USD precision loss
        # per tick in dollar-volume accumulation. Over millions of ticks this causes
        # a measurable systematic bias in target calibration and bar boundaries.
        chunk["price"]        = pd.to_numeric(chunk["price"],        errors="coerce").astype("float64")
        chunk["quantity"]     = pd.to_numeric(chunk["quantity"],     errors="coerce").astype("float64")
        chunk["transact_time"]= pd.to_numeric(chunk["transact_time"],errors="coerce").astype("int64")
        chunk = chunk[(chunk["price"] > 0) & (chunk["quantity"] > 0)]
        if chunk.empty:
            continue
        chunk["is_buyer_maker"] = chunk["is_buyer_maker"].str.upper() == "TRUE"
        if ms_divisor is None:
            sample = int(chunk["transact_time"].iloc[0])
            ms_divisor = 1_000 if sample > 9_999_999_999_999 else 1
        ts = chunk["transact_time"].to_numpy(dtype=np.int64)
        if ms_divisor != 1:
            ts = ts // ms_divisor
        yield (
            chunk["price"].to_numpy(dtype=np.float64),    # BUG-FIX 5
            chunk["quantity"].to_numpy(dtype=np.float64), # BUG-FIX 5
            ts,
            chunk["is_buyer_maker"].to_numpy(dtype=bool),
        )
        del chunk; gc.collect()


def _gather_calibration_data(csv_path, lookback_days):
    lookback_ms = lookback_days * 86_400 * 1_000
    cal_p, cal_q, cal_ts, cal_ibm = [], [], [], []
    cutoff_ms = None
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
            del prices, quantities, timestamps, ibm; gc.collect(); break
        cal_p.append(prices.astype(np.float64))
        cal_q.append(quantities.astype(np.float64))
        cal_ts.append(timestamps); cal_ibm.append(ibm)
        del prices, quantities, timestamps, ibm; gc.collect()
    return (np.concatenate(cal_p), np.concatenate(cal_q),
            np.concatenate(cal_ts), np.concatenate(cal_ibm))


def _write_bars(bars, output_csv, write_header):
    if not bars: return False
    df = pd.DataFrame(bars)
    for col in ("datetime","datetime_start","datetime_end"):
        if col in df.columns: df[col] = df[col].astype(str)
    df.to_csv(output_csv, mode="w" if write_header else "a",
              header=write_header, index=False)
    del df; return True


def _update_and_adapt(bar_processor, market_params, bar, recent):
    market_params = bar_processor.update_market_params(market_params, bar)
    market_params["monitoring_counter"] = market_params.get("monitoring_counter",0)+1
    market_params["bars_since_optimization"] = market_params.get("bars_since_optimization",0)+1
    if market_params["monitoring_counter"] >= EMA_MONITORING_INTERVAL:
        market_params = bar_processor._perform_self_monitoring(market_params)
        market_params["monitoring_counter"] = 0
    if market_params["bars_since_optimization"] >= EMA_OPTIMIZATION_INTERVAL:
        if bar_processor._can_optimize(market_params) and len(recent)>=MIN_BARS_FOR_QUALITY:
            quality = bar_processor._calculate_bar_quality(list(recent), market_params)
            market_params = bar_processor._apply_optimization_strategy(
                market_params, quality, list(recent))
        market_params["bars_since_optimization"] = 0
    return bar_processor._enforce_parameter_bounds(market_params)


def process_tick_bars(
    bar_type: str,
    tick_csv: str | Path,
    output_dir: str | Path = "data/processed_bars",
    exchange: str = "binance",
    symbol: str = "btc",
) -> List[Dict[str, Any]]:
    """
    Generate tick-level bars from a Binance aggTrades CSV.
    Calibration uses first ANALYSIS_LOOKBACK_DAYS of TICK data — no minute CSV.
    Mirrors run.py exactly.
    """
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
    total_bars    = 0
    write_header  = True
    t0            = time.time()

    logger.info("Calibrating %s tick bars from %s ...", bar_type, tick_csv.name)
    market_params = mod.calibrate(bar_processor, tick_csv, _gather_calibration_data)

    leftover: dict      = {}
    open_bar_data: dict = {}

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
            # BUG-FIX 9: The 4th return value (leftover) is always {} for
            # volatility/range/hybrid — exhausted ticks are absorbed into
            # open_bar_data carry rather than returned as raw arrays.
            # We name it explicitly instead of using _ to make this contract
            # visible and catch any future implementation that changes it.
            bars, market_params, open_bar_data, _leftover = mod.process_chunk(
                prices, quantities, timestamps_ms, is_buyer_maker,
                bar_processor, market_params, recent,
                open_bar_data, _update_and_adapt)
            if _leftover:  # should always be empty — log if not
                logger.warning("Unexpected leftover from %s process_chunk: %d ticks",
                               bar_type, len(next(iter(_leftover.values()), [])))

        del prices, quantities, timestamps_ms, is_buyer_maker; gc.collect()

        if bars:
            if _write_bars(bars, output_csv, write_header):
                write_header = False
            total_bars += len(bars)
        del bars

        if (chunk_idx + 1) % 10 == 0:
            logger.info("  chunk %d — %d bars so far (%.1fs)",
                        chunk_idx+1, total_bars, time.time()-t0)

    logger.info("DONE: %d %s tick bars in %.1fs", total_bars, bar_type, time.time()-t0)
    return pd.read_csv(output_csv).to_dict("records") if total_bars else []


def process_minute_bars(
    bar_type: str,
    minute_csv: str | Path,
    output_dir: str | Path = "data/processed_bars",
    exchange: str = "binance",
    symbol: str = "btc",
    resume: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate minute-level bars from a 1-minute OHLCV CSV.
    Mirrors bitpredict utils.py process_bar_data() exactly.
    """
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
        logger.info("No new minute data for %s.", stem); return []

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

    if hasattr(processor,"previous_close") and market_params.get("previous_close"):
        processor.previous_close = float(market_params["previous_close"])
    if hasattr(processor,"renko_reference") and market_params.get("renko_reference"):
        processor.renko_reference = float(market_params["renko_reference"])

    current_bar_data: Dict = {}
    if state and state.get("current_bar_data"):
        current_bar_data = state["current_bar_data"]

    all_bars: List[Dict] = []
    recent   = deque(maxlen=MIN_BARS_FOR_QUALITY)
    last_dt  = start_date

    # BUG-FIX 16: Determine how many rows were used for calibration so we can skip
    # them in the production loop. Without this skip, calibration data is processed
    # twice — once for parameter estimation and once as production bars — introducing
    # look-ahead bias (the EMA target was estimated from the very data it then uses).
    # On resume (state loaded) no calibration was done this run, so skip_rows = 0.
    if state and state.get("ema_state"):
        calibration_skip_rows = 0  # resumed — no calibration done this run
    else:
        calibration_skip_rows = ANALYSIS_LOOKBACK_DAYS * 1_440
    rows_processed = 0

    for row in df.itertuples(index=False):
        rows_processed += 1
        if rows_processed <= calibration_skip_rows:
            # Skip rows that were used only for calibration.
            # Still update previous_close / renko_reference state so
            # the first production bar starts with correct context.
            if hasattr(processor, "previous_close"):
                processor.previous_close = float(row.close)
            last_dt = row.datetime
            continue
        md = {"datetime":row.datetime,"open":float(row.open),"high":float(row.high),
              "low":float(row.low),"close":float(row.close),"volume":float(row.volume)}
        if any(v<=0 for v in (md["open"],md["high"],md["low"],md["close"])):
            last_dt = md["datetime"]; continue

        current_bar_data = processor.accumulate_bar_data(current_bar_data, md)

        if processor.should_create_bar(md, current_bar_data, market_params):
            bar = processor.finalize_bar(current_bar_data, market_params)
            all_bars.append(bar); recent.append(bar)
            market_params = _update_and_adapt(processor, market_params, bar, recent)
            if hasattr(processor,"previous_close"):
                market_params["previous_close"] = processor.previous_close
            if hasattr(processor,"renko_reference"):
                market_params["renko_reference"] = processor.renko_reference
            current_bar_data = {}
        last_dt = md["datetime"]

    if all_bars:
        save_bars_csv(all_bars, bars_path)
        logger.info("Created %d bars → %s", len(all_bars), bars_path.name)
    else:
        logger.info("No new bars for %s.", stem)

    save_state({"last_processed_datetime":str(last_dt),
                "current_bar_data":current_bar_data,
                "ema_state":market_params}, state_path)
    return all_bars
