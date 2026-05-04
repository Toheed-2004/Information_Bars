"""
bars/processor.py
-----------------
Core bar-processing engine — CSV-based, no database required.

process_minute_bars()  orchestrates the minute-data loop.
process_tick_bars()    orchestrates the tick-data loop.

Both functions:
  1. Load data from CSV.
  2. Calibrate the bar processor from the first ANALYSIS_LOOKBACK_DAYS of data.
  3. Run the main accumulation loop.
  4. Save the resulting bars to data/processed_bars/.
  5. Persist EMA state to data/processed_bars/<name>_state.json.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add repo root to path so common/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.constants import ANALYSIS_LOOKBACK_DAYS, EMA_MONITORING_INTERVAL, \
    EMA_OPTIMIZATION_INTERVAL, MIN_BARS_FOR_QUALITY
from common.logging import get_logger
from common.data_loader import (
    load_minute_csv, load_tick_csv, df_to_records,
    save_bars_csv, save_state, load_state,
)
from bars.bar_types import get_bar_class, ALL_BAR_TYPE_NAMES

logger = get_logger(__name__)


# ── Minute pipeline ───────────────────────────────────────────────────────────

def process_minute_bars(
    bar_type: str,
    minute_csv: str | Path,
    output_dir: str | Path = "data/processed_bars",
    exchange: str = "binance",
    symbol: str = "btc",
    resume: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate minute-level information bars from a CSV of 1-minute OHLCV data.

    Args:
        bar_type:   One of dollar/volume/volatility/range/renko/hybrid.
        minute_csv: Path to the 1-minute OHLCV CSV file.
        output_dir: Directory where bars CSV and state JSON are saved.
        exchange:   Exchange label (used for naming output files).
        symbol:     Symbol label (used for naming output files).
        resume:     If True and a state file exists, resume from last position.

    Returns:
        List of bar dicts (also saved to output_dir).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem        = f"{exchange}_{symbol}_{bar_type}_minute"
    bars_path   = output_dir / f"{stem}_bars.csv"
    state_path  = output_dir / f"{stem}_state.json"

    # Load state if resuming
    state: Optional[Dict] = load_state(state_path) if resume else None
    start_date  = None
    if state and state.get("last_processed_datetime"):
        import dateutil.parser
        start_date = dateutil.parser.parse(state["last_processed_datetime"])
        logger.info("Resuming %s from %s", stem, start_date)

    # Load minute data
    df = load_minute_csv(minute_csv, start_date=start_date)
    if df.empty:
        logger.info("No new minute data for %s — nothing to do.", stem)
        return []

    # Instantiate bar processor
    BarClass   = get_bar_class(bar_type, source="minute")
    processor  = BarClass(exchange, symbol)

    # Calibrate
    if state and state.get("ema_state"):
        market_params: Dict = dict(state["ema_state"])
        logger.info("Restored EMA state from %s", state_path.name)
    else:
        lookback_rows = ANALYSIS_LOOKBACK_DAYS * 1_440
        calib_records = df_to_records(df.iloc[:lookback_rows])
        market_params = processor.analyze_market_history(calib_records)
        logger.info("Calibration complete on %d rows.", len(calib_records))

    # Restore instance-variable state
    if hasattr(processor, "previous_close") and market_params.get("previous_close"):
        processor.previous_close = float(market_params["previous_close"])
    if hasattr(processor, "renko_reference") and market_params.get("renko_reference"):
        processor.renko_reference = float(market_params["renko_reference"])

    # Restore open bar
    current_bar_data: Dict = {}
    if state and state.get("current_bar_data"):
        current_bar_data = state["current_bar_data"]

    # ── Main loop ─────────────────────────────────────────────────────────────
    all_bars: List[Dict] = []
    recent   = deque(maxlen=MIN_BARS_FOR_QUALITY)
    last_dt  = start_date

    for row in df.itertuples(index=False):
        md = {
            "datetime": row.datetime,
            "open":  float(row.open),
            "high":  float(row.high),
            "low":   float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        if any(v <= 0 for v in (md["open"], md["high"], md["low"], md["close"])):
            last_dt = md["datetime"]
            continue

        current_bar_data = processor.accumulate_bar_data(current_bar_data, md)

        if processor.should_create_bar(md, current_bar_data, market_params):
            bar = processor.finalize_bar(current_bar_data, market_params)
            all_bars.append(bar)
            recent.append(bar)

            market_params = processor.update_market_params(market_params, bar)
            market_params["monitoring_counter"] = \
                market_params.get("monitoring_counter", 0) + 1
            market_params["bars_since_optimization"] = \
                market_params.get("bars_since_optimization", 0) + 1

            if hasattr(processor, "previous_close"):
                market_params["previous_close"] = processor.previous_close
            if hasattr(processor, "renko_reference"):
                market_params["renko_reference"] = processor.renko_reference

            if market_params["monitoring_counter"] >= EMA_MONITORING_INTERVAL:
                market_params = processor._perform_self_monitoring(market_params)
                market_params["monitoring_counter"] = 0

            if (market_params["bars_since_optimization"] >= EMA_OPTIMIZATION_INTERVAL
                    and processor._can_optimize(market_params)
                    and len(recent) >= MIN_BARS_FOR_QUALITY):
                quality = processor._calculate_bar_quality(list(recent), market_params)
                market_params = processor._apply_optimization_strategy(
                    market_params, quality, list(recent))
                market_params["bars_since_optimization"] = 0

            market_params = processor._enforce_parameter_bounds(market_params)
            current_bar_data = {}

        last_dt = md["datetime"]

    # ── Persist ───────────────────────────────────────────────────────────────
    if all_bars:
        save_bars_csv(all_bars, bars_path)
        logger.info("Created %d bars → %s", len(all_bars), bars_path.name)
    else:
        logger.info("No new bars created for %s.", stem)

    save_state({
        "last_processed_datetime": str(last_dt),
        "current_bar_data": current_bar_data,
        "ema_state": market_params,
    }, state_path)

    return all_bars


# ── Tick pipeline ─────────────────────────────────────────────────────────────

def process_tick_bars(
    bar_type: str,
    tick_csv: str | Path,
    minute_csv: str | Path,
    output_dir: str | Path = "data/processed_bars",
    exchange: str = "binance",
    symbol: str = "btc",
    resume: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate tick-level information bars from a Binance aggTrades CSV.

    Uses minute OHLCV data for initial calibration (shared calibration
    framework), then replaces the threshold with a tick-native equivalent
    computed from the tick data.

    Args:
        bar_type:   One of dollar/volume/volatility/range/renko/hybrid.
        tick_csv:   Path to the aggTrades CSV (price, qty, timestamp_ms, …).
        minute_csv: Path to 1-minute OHLCV CSV used for calibration.
        output_dir: Where to write the output bars CSV and state JSON.
        exchange:   Exchange label for file naming.
        symbol:     Symbol label for file naming.
        resume:     If True, resume from last saved state.

    Returns:
        List of bar dicts (also saved to output_dir).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem       = f"{exchange}_{symbol}_{bar_type}_tick"
    bars_path  = output_dir / f"{stem}_bars.csv"
    state_path = output_dir / f"{stem}_state.json"

    state: Optional[Dict] = load_state(state_path) if resume else None

    # Load tick data
    tick_df = load_tick_csv(tick_csv)
    if tick_df.empty:
        logger.info("No tick data found in %s — nothing to do.", tick_csv)
        return []

    # Instantiate bar processor
    BarClass  = get_bar_class(bar_type, source="tick")
    processor = BarClass(exchange, symbol)

    # Calibrate using minute data (shared framework)
    if state and state.get("ema_state"):
        market_params: Dict = dict(state["ema_state"])
        logger.info("Restored EMA state from %s", state_path.name)
    else:
        min_df        = load_minute_csv(minute_csv)
        lookback_rows = ANALYSIS_LOOKBACK_DAYS * 1_440
        calib_records = df_to_records(min_df.iloc[:lookback_rows])

        # Minute-based calibration provides the framework
        minute_params = processor.analyze_market_history(calib_records)

        # Replace threshold with tick-native equivalent
        # (delegated to calibrate() if the tick class provides it)
        if hasattr(processor, "calibrate"):
            market_params = processor.calibrate(
                minute_params,
                tick_df,
            )
        else:
            market_params = minute_params
        logger.info("Tick calibration complete.")

    # Restore open bar accumulator
    current_bar_data: Dict = {}
    if state and state.get("current_bar_data"):
        current_bar_data = state["current_bar_data"]

    # ── Tick loop (delegated to process_chunk if available) ───────────────────
    all_bars: List[Dict] = []

    if hasattr(processor, "process_chunk"):
        # Tick bar classes expose a vectorised process_chunk() function
        prices        = tick_df["price"].to_numpy(dtype=np.float64)
        quantities    = tick_df["qty"].to_numpy(dtype=np.float64)
        timestamps_ms = tick_df["timestamp_ms"].to_numpy(dtype=np.int64)
        is_buyer_maker = (
            tick_df["is_buyer_maker"].to_numpy(dtype=bool)
            if "is_buyer_maker" in tick_df.columns
            else np.zeros(len(prices), dtype=bool)
        )
        recent: deque = deque(maxlen=MIN_BARS_FOR_QUALITY)

        def _update(mp: Dict, bar: Dict) -> Dict:
            mp = processor.update_market_params(mp, bar)
            recent.append(bar)
            mp["monitoring_counter"] = mp.get("monitoring_counter", 0) + 1
            if mp["monitoring_counter"] >= EMA_MONITORING_INTERVAL:
                mp = processor._perform_self_monitoring(mp)
                mp["monitoring_counter"] = 0
            return mp

        bars, market_params, _ = processor.process_chunk(
            prices, quantities, timestamps_ms, is_buyer_maker,
            processor, market_params, recent, _update,
        )
        all_bars.extend(bars)
    else:
        logger.warning(
            "TickBar class for %s has no process_chunk(); falling back to row loop.",
            bar_type,
        )
        for row in tick_df.itertuples(index=False):
            td = {
                "price":          float(row.price),
                "qty":            float(row.qty),
                "timestamp_ms":   int(row.timestamp_ms),
                "is_buyer_maker": bool(getattr(row, "is_buyer_maker", False)),
            }
            current_bar_data = processor.accumulate_bar_data(current_bar_data, td)
            if processor.should_create_bar(td, current_bar_data, market_params):
                bar = processor.finalize_bar(current_bar_data, market_params)
                all_bars.append(bar)
                market_params    = processor.update_market_params(market_params, bar)
                current_bar_data = {}

    # ── Persist ───────────────────────────────────────────────────────────────
    if all_bars:
        save_bars_csv(all_bars, bars_path)
        logger.info("Created %d tick bars → %s", len(all_bars), bars_path.name)

    save_state({
        "last_processed_datetime": str(tick_df["timestamp_ms"].iloc[-1]),
        "current_bar_data": current_bar_data,
        "ema_state": market_params,
    }, state_path)

    return all_bars
