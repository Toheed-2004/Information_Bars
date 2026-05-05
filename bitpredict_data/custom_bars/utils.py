"""
bars/utils.py — Core bar-processing logic.

process_bar_data(exchange, symbol, bar_type, mode) owns:
  - DB setup, state load/save
  - The main minute-data loop
  - EMA update scheduling (monitoring + optimization)
  - Batch DB insert
  - Regime calculation
"""
import pandas as pd
from collections import deque
from typing import Optional

from bitpredict.common.logging import get_logger
from bitpredict.common.db.models.data import ensure_bars_table, ensure_state_table
from bitpredict.common.db.services.data import read_ohlcv
from bitpredict.data.custom_bars.bar_types import get_bar_class
from bitpredict.data.custom_bars.db import (
    get_bar_state,
    update_bar_state,
    batch_insert_bars,
)
from bitpredict.data.custom_bars.regime import calculate_and_update_regimes
from bitpredict.data.custom_bars.constants import (
    ANALYSIS_LOOKBACK_DAYS,
    EMA_MONITORING_INTERVAL,
    EMA_OPTIMIZATION_INTERVAL,
    MIN_BARS_FOR_QUALITY,
)

logger = get_logger(__name__)


def process_bar_data(exchange: str, symbol: str, bar_type: str, mode: str) -> int:
    """
    Process minute OHLCV data and create bars for the given exchange/symbol/bar_type.

    Args:
        exchange:  Exchange identifier (e.g. 'okx', 'binance').
        symbol:    Symbol identifier (e.g. 'btc', 'eth').
        bar_type:  Bar type ('volume', 'dollar', 'volatility').
        mode:      'init' (no prior state) or 'update' (existing state).

    Returns:
        Number of new bars created.
    """
    # ------------------------------------------------------------------
    # 1. Ensure DB tables exist
    # ------------------------------------------------------------------
    try:
        bars_table = ensure_bars_table(bar_type)
    except Exception as e:
        logger.error("Failed to create bars table for %s: %s", bar_type, e)
        return 0

    try:
        ensure_state_table()
    except Exception as e:
        logger.error("Failed to create state table: %s", e)
        return 0

    # ------------------------------------------------------------------
    # 2. Load bar class + existing state
    # ------------------------------------------------------------------
    try:
        bar_class = get_bar_class(bar_type)
        bar_processor = bar_class(exchange, symbol)
    except ValueError as e:
        logger.error("Unknown bar type: %s", e)
        return 0

    state = get_bar_state(exchange, symbol, bar_type)

    # ------------------------------------------------------------------
    # 3. Determine start date and fetch minute data
    # ------------------------------------------------------------------
    if mode == "update" and state and state.get("last_processed_datetime"):
        start_date = state["last_processed_datetime"]
        logger.info("Update mode: resuming from %s", start_date)
        df = read_ohlcv(exchange=exchange, symbol=symbol, timeframe="1m", start_date=start_date)
    else:
        start_date = None
        logger.info("Init mode: processing full history")
        df = read_ohlcv(exchange=exchange, symbol=symbol, timeframe="1m")
    if df.empty:
        logger.info("No minute data to process for %s_%s", exchange, symbol)
        return 0

    # Normalise column naming — ensure a 'datetime' column exists
    if "datetime" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "datetime"})
        elif df.index.name in ("datetime", "timestamp"):
            df = df.reset_index()
            if "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    if df.empty:
        logger.info("No new minute data after filtering for %s_%s", exchange, symbol)
        return 0

    logger.info("Processing %d minute rows for %s_%s_%s", len(df), exchange, symbol, bar_type)

    # ------------------------------------------------------------------
    # 4. Initialise market_params
    # ------------------------------------------------------------------
    market_params: dict

    if mode == "init" or not (state and state.get("ema_state")):
        # Full market analysis on the first ANALYSIS_LOOKBACK_DAYS * 1440 rows
        lookback_rows = ANALYSIS_LOOKBACK_DAYS * 1440
        analysis_rows = df.iloc[:lookback_rows]
        formatted = _df_to_list(analysis_rows)
        market_params = bar_processor.analyze_market_history(formatted)
        logger.info(
            "Market analysis complete for %s_%s_%s (%d rows analysed)",
            exchange, symbol, bar_type, len(formatted),
        )
    else:
        market_params = dict(state["ema_state"])
        logger.debug("Restored market_params from state for %s_%s_%s", exchange, symbol, bar_type)

    # Restore instance-variable state from market_params (VolatilityBar, HybridBar, RenkoBar)
    if hasattr(bar_processor, "previous_close") and market_params.get("previous_close") is not None:
        bar_processor.previous_close = float(market_params["previous_close"])
    if hasattr(bar_processor, "renko_reference") and market_params.get("renko_reference") is not None:
        bar_processor.renko_reference = float(market_params["renko_reference"])

    # ------------------------------------------------------------------
    # 5. Restore current_bar_data
    # ------------------------------------------------------------------
    current_bar_data: dict = {}
    if state and state.get("current_bar_data"):
        raw = state["current_bar_data"]
        # Backward-compat: rename old keys if present
        if "timestamp_start" in raw and "datetime_start" not in raw:
            raw["datetime_start"] = raw.pop("timestamp_start")
        if "timestamp_end" in raw and "datetime_end" not in raw:
            raw["datetime_end"] = raw.pop("timestamp_end")
        if "accumulated_volume" in raw and "accumulated_size" not in raw:
            raw["accumulated_size"] = raw.pop("accumulated_volume")
        if "accumulated_dollar_volume" in raw and "accumulated_size" not in raw:
            raw["accumulated_size"] = raw.pop("accumulated_dollar_volume")
        if "accumulated_volatility" in raw and "accumulated_size" not in raw:
            raw["accumulated_size"] = raw.pop("accumulated_volatility")
        if "minute_count" in raw and "tick_count" not in raw:
            raw["tick_count"] = raw.pop("minute_count")
        current_bar_data = raw

    # ------------------------------------------------------------------
    # 6. Main processing loop
    # ------------------------------------------------------------------
    all_bars = []                       # Collect for batch insert
    recent_bars_buffer = deque(maxlen=MIN_BARS_FOR_QUALITY)  # For quality analysis

    last_processed_datetime = start_date
    bars_created = 0

    for row in df.itertuples(index=False):
        minute_data = {
            "datetime": row.datetime,
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }

        # Skip rows that can't pass basic validation
        if any(v <= 0 for v in [minute_data["open"], minute_data["high"],
                                 minute_data["low"], minute_data["close"]]):
            last_processed_datetime = minute_data["datetime"]
            continue

        # Accumulate into current bar
        current_bar_data = bar_processor.accumulate_bar_data(current_bar_data, minute_data)

        # Check if a bar should close
        if bar_processor.should_create_bar(minute_data, current_bar_data, market_params):
            finalized_bar = bar_processor.finalize_bar(current_bar_data, market_params)

            all_bars.append(finalized_bar)
            recent_bars_buffer.append(finalized_bar)
            bars_created += 1

            # EMA update
            market_params = bar_processor.update_market_params(market_params, finalized_bar)
            market_params["monitoring_counter"] = market_params.get("monitoring_counter", 0) + 1
            market_params["bars_since_optimization"] = (
                market_params.get("bars_since_optimization", 0) + 1
            )

            # Sync instance-variable state to market_params for persistence
            if hasattr(bar_processor, "previous_close"):
                market_params["previous_close"] = bar_processor.previous_close
            if hasattr(bar_processor, "renko_reference"):
                market_params["renko_reference"] = bar_processor.renko_reference

            # Self-monitoring
            if market_params["monitoring_counter"] >= EMA_MONITORING_INTERVAL:
                market_params = bar_processor._perform_self_monitoring(market_params)
                market_params["monitoring_counter"] = 0

            # Auto-optimization
            if market_params["bars_since_optimization"] >= EMA_OPTIMIZATION_INTERVAL:
                if bar_processor._can_optimize(market_params) and len(recent_bars_buffer) >= MIN_BARS_FOR_QUALITY:
                    quality = bar_processor._calculate_bar_quality(
                        list(recent_bars_buffer), market_params
                    )
                    market_params = bar_processor._apply_optimization_strategy(
                        market_params, quality, list(recent_bars_buffer)
                    )
                market_params["bars_since_optimization"] = 0

            # Apply parameter bounds after any updates
            market_params = bar_processor._enforce_parameter_bounds(market_params)

            current_bar_data = {}

        last_processed_datetime = minute_data["datetime"]

    # ------------------------------------------------------------------
    # 7. Batch insert all bars
    # ------------------------------------------------------------------
    if all_bars:
        try:
            inserted = batch_insert_bars(bars_table, all_bars, bar_type, exchange, symbol)
            logger.info(
                "Inserted %d/%d bars for %s_%s_%s",
                inserted, len(all_bars), exchange, symbol, bar_type,
            )
        except Exception as e:
            logger.error("Batch insert failed for %s_%s_%s: %s", exchange, symbol, bar_type, e)

    # ------------------------------------------------------------------
    # 8. Save state
    # ------------------------------------------------------------------
    current_bar_datetime = None
    if current_bar_data:
        current_bar_datetime = current_bar_data.get("datetime_start")

    try:
        update_bar_state(
            exchange,
            symbol,
            bar_type,
            last_processed_datetime=last_processed_datetime,
            current_bar_datetime=current_bar_datetime,
            current_bar_data=current_bar_data,
            ema_state=market_params,
        )
    except Exception as e:
        logger.error("Failed to save state for %s_%s_%s: %s", exchange, symbol, bar_type, e)

    logger.info(
        "Done: %d bars created for %s_%s_%s", bars_created, exchange, symbol, bar_type
    )

    # ------------------------------------------------------------------
    # 9. Regime calculation
    # ------------------------------------------------------------------
    if all_bars:
        try:
            new_bars_df = _bars_to_dataframe(all_bars)
            calculate_and_update_regimes(bars_table, exchange, symbol, bar_type, new_bars_df)
        except Exception as e:
            logger.error(
                "Regime calculation failed for %s_%s_%s: %s", exchange, symbol, bar_type, e
            )

    return bars_created


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_list(df: pd.DataFrame) -> list:
    """Convert a minute DataFrame to a list of dicts for analyze_market_history."""
    records = []
    for row in df.itertuples(index=False):
        records.append({
            "datetime": row.datetime,
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        })
    return records


def _bars_to_dataframe(bars: list) -> pd.DataFrame:
    """Convert finalized bar dicts to a DataFrame with datetime index."""
    df = pd.DataFrame(bars)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    return df