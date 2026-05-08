"""
Single entry point for the regime engine.

    from common.market_regimes import calculate_regimes

    # Batch (first time, no saved state) — pass full history DataFrame
    df_out = calculate_regimes(df, exchange="binance", symbol="btc", bar_type="time", bar_timeframe="1h")

    # Incremental (state exists in DB) — pass only new rows
    df_out = calculate_regimes(df_new, exchange="binance", symbol="btc", bar_type="time", bar_timeframe="1h")

Mode is auto-detected:
  - If a saved state exists in DB (or engine already in memory registry) → incremental
  - Otherwise → batch (full recalculation)

Incremental safety net:
  - Rows at or before last_bar_id are silently dropped before processing.
  - Callers should pass only new rows, but this handles accidental overlap.

Everything else is handled internally:
  - DB engine obtained via get_engine()
  - DB table created on first use
  - symbol_id resolved from meta.symbols(exchange, symbol)
  - State loaded from DB on first incremental call per process lifetime
  - State saved to DB after batch, and after last row of each incremental call
  - Engines kept in memory between incremental calls (no repeated DB reads per bar)
  - Config auto-derived from bar_type + bar_timeframe when not explicitly supplied
"""
from __future__ import annotations

from typing import Optional, Dict, Tuple

import pandas as pd

from bitpredict.common.market_regimes.engine import RegimeEngine
from bitpredict.common.market_regimes.config import RegimeConfig
from bitpredict.common.db.models.market_regime import ensure_regime_state_table
from bitpredict.common.db.services.market_regime import (
    get_symbol_id, load_state, save_state, load_last_bar_id,
)
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)

_Key = Tuple[int, str, Optional[str]]   # (symbol_id, bar_type, bar_timeframe)

# In-memory engine registry — avoids DB round-trips on every incremental call
_registry: Dict[_Key, RegimeEngine] = {}

# Ensure table is only created once per process
_table_ensured: bool = False

# ---------------------------------------------------------------------------
# Timeframe → hours lookup (case-insensitive via .lower() at lookup time)
# ---------------------------------------------------------------------------
_TIMEFRAME_HOURS = {
    "1m": 1 / 60,  "3m": 3 / 60,  "5m": 5 / 60, "10m": 10 / 60, "15m": 15 / 60,
    "30m": 0.5,    "45m": 0.75,
    "1h": 1.0,  "2h": 2.0,  "3h": 3.0,  "4h": 4.0,  "6h": 6.0,  "8h": 8.0,  "12h": 12.0,
    "1d": 24.0, "2d": 48.0, "3d": 72.0,
    "1w": 168.0,
}


def calculate_regimes(
    data: pd.DataFrame,
    exchange: str,
    symbol: str,
    bar_type: str,
    bar_timeframe: Optional[str] = None,
    config: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    """
    Run the regime engine on a DataFrame.

    Mode is auto-detected:
      - Incremental: a saved state exists in DB for this (symbol, bar_type, bar_timeframe),
        or the engine is already loaded in the in-memory registry from a previous call.
      - Batch: no saved state exists anywhere — full recalculation from scratch.

    Parameters
    ----------
    data         : pd.DataFrame with at least a 'close' column.
                   'datetime' or 'timestamp' column used for gap detection and
                   last_bar_id tracking if present.
                   For time bars, 'datetime' or 'timestamp' is expected.
                   For non-time bars (dollar, volume, etc.), bar_timeframe should be None.
    exchange     : e.g. "binance" — must exist in meta.symbols
    symbol       : e.g. "btc"    — must exist in meta.symbols
    bar_type     : "time" | "volume" | "dollar" | "renko" | "range" | "volatility" | "hybrid"
    bar_timeframe: e.g. "1h", "4h", "1d" — None for non-time bar types
    config       : Optional RegimeConfig override. Auto-derived from bar_type/bar_timeframe
                   if not provided.

    Returns
    -------
    pd.DataFrame with all three output layers merged with input columns.
    In incremental mode, only newly processed rows are returned (rows at or before
    last_bar_id are silently dropped by the safety net).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pd.DataFrame, got {type(data)}")

    logger.info(f"calculate_regimes called for {exchange}:{symbol} {bar_type}/{bar_timeframe} with {len(data)} rows")

    _ensure_table()
    symbol_id = _resolve_symbol_id(exchange, symbol)  # None if DB unavailable
    key = (symbol_id, bar_type, bar_timeframe)

    if _detect_incremental(key, symbol_id, bar_type, bar_timeframe, config):
        logger.info(f"Running in INCREMENTAL mode for {exchange}:{symbol} {bar_type}/{bar_timeframe}")
        return _run_incremental_df(data, key, config)
    else:
        logger.info(f"Running in BATCH mode for {exchange}:{symbol} {bar_type}/{bar_timeframe}")
        return _run_batch(data, key, config)


def clear_registry():
    """Drop all cached in-memory engines. Useful in tests or when switching configs."""
    global _table_ensured
    _registry.clear()
    _table_ensured = False


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def _detect_incremental(
    key: _Key,
    symbol_id: Optional[int],
    bar_type: str,
    bar_timeframe: Optional[str],
    config: Optional[RegimeConfig],
) -> bool:
    """
    Return True if incremental mode should be used.

    Side effect: on the first call for a given key (not yet in registry), if a DB
    state is found, the engine is created with that state and added to _registry.
    This ensures state is loaded from DB exactly once per process lifetime.
    _run_incremental_df relies on the engine already being present in _registry.
    """
    # Fast path: engine already loaded in memory from a previous call this process
    if key in _registry:
        return True

    # No DB available (symbol could not be resolved) → batch only
    if symbol_id is None:
        return False

    # Check DB for an existing state row
    try:
        state = load_state(symbol_id, bar_type, bar_timeframe)
    except Exception:
        return False

    if state is None:
        return False

    # State found — create engine, restore state, register for future calls
    engine = RegimeEngine(config or _make_config(bar_type, bar_timeframe))
    engine._state = state
    _registry[key] = engine
    return True


# ---------------------------------------------------------------------------
# Config auto-derivation
# ---------------------------------------------------------------------------

def _make_config(bar_type: str, bar_timeframe: Optional[str]) -> RegimeConfig:
    """
    Build a RegimeConfig with parameters auto-derived from bar_type and bar_timeframe.

    For time bars:
      - ring_buffer_size : targets ~4 weeks (672 h) of lookback, clamped [200, 5000]
      - min_duration_bars: targets ~3 h minimum regime hold, clamped [2, 20]
      - gap_multiplier   : 5.0 (time-bar intervals are regular)

    For non-time bars (volume, dollar, renko, range, volatility, hybrid):
      - ring_buffer_size : 500 (bar-count based, sensible default)
      - min_duration_bars: 3
      - gap_multiplier   : 15.0 (intervals naturally vary, higher threshold needed)
    """
    cfg = RegimeConfig()

    if bar_type == "time" and bar_timeframe is not None:
        hours = _TIMEFRAME_HOURS.get(bar_timeframe.lower())
        if hours is not None:
            cfg.ring_buffer_size = int(max(200, min(5000, round(672.0 / hours))))
            cfg.min_duration_bars = int(max(2, min(20, round(3.0 / hours))))
            # gap_multiplier stays at default 5.0 for time bars
    else:
        # Non-time bars: intervals vary by construction, need a looser gap threshold
        cfg.gap_multiplier = 15.0

    return cfg


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _run_batch(df: pd.DataFrame, key: _Key, config: Optional[RegimeConfig]) -> pd.DataFrame:
    symbol_id, bar_type, bar_timeframe = key

    logger.debug(f"_run_batch: Starting batch calculation for symbol_id={symbol_id} {bar_type}/{bar_timeframe}")
    engine = RegimeEngine(config or _make_config(bar_type, bar_timeframe))
    engine.reset()

    df_out = engine.calculate_batch(df)
    logger.info(f"_run_batch: Completed batch calculation, processed {len(df_out)} rows")

    _registry[key] = engine
    if symbol_id is not None:
        _save_state(engine, symbol_id, bar_type, bar_timeframe,
                    last_bar_id=_extract_last_bar_id(df_out))
        logger.debug(f"_run_batch: Saved state for symbol_id={symbol_id}")

    return df_out


# ---------------------------------------------------------------------------
# Incremental (DataFrame — one or more new rows)
# ---------------------------------------------------------------------------

def _run_incremental_df(
    df: pd.DataFrame,
    key: _Key,
    config: Optional[RegimeConfig],
) -> pd.DataFrame:
    symbol_id, bar_type, bar_timeframe = key
    engine = _registry[key]  # guaranteed by _detect_incremental

    logger.debug(f"_run_incremental_df: Starting incremental update for symbol_id={symbol_id} {bar_type}/{bar_timeframe} with {len(df)} rows")

    # Safety net: drop rows that were already processed in a previous call
    df = _drop_processed_rows(df, symbol_id, bar_type, bar_timeframe)
    logger.debug(f"_run_incremental_df: After safety net, {len(df)} rows remain to process")

    if df.empty:
        logger.debug(f"_run_incremental_df: No new rows to process after safety net")
        return df

    # Process each row sequentially via the incremental engine update
    results = []
    for _, row in df.iterrows():
        result = engine.update(row.to_dict())
        results.append(result)

    logger.debug(f"_run_incremental_df: Processed {len(results)} rows via incremental update")

    # Merge regime output columns into a copy of the (filtered) input DataFrame
    result_df = pd.DataFrame(results, index=df.index)
    out = df.copy()
    for col in result_df.columns:
        out[col] = result_df[col]

    # Save state once, after the last row
    if symbol_id is not None:
        _save_state(engine, symbol_id, bar_type, bar_timeframe,
                    last_bar_id=_extract_last_bar_id(out))
        logger.debug(f"_run_incremental_df: Saved state for symbol_id={symbol_id}")

    logger.info(f"_run_incremental_df: Completed incremental update, returned {len(out)} rows")
    return out


# ---------------------------------------------------------------------------
# Safety net: drop already-processed rows
# ---------------------------------------------------------------------------

def _drop_processed_rows(
    df: pd.DataFrame,
    symbol_id: Optional[int],
    bar_type: str,
    bar_timeframe: Optional[str],
) -> pd.DataFrame:
    """
    Drop rows at or before last_bar_id stored in DB.
    Returns df unchanged if last_bar_id is unavailable, unparseable,
    or no datetime/timestamp column is found.
    """
    if symbol_id is None:
        return df

    try:
        last_bar_id = load_last_bar_id(symbol_id, bar_type, bar_timeframe)
    except Exception:
        return df

    if not last_bar_id:
        return df

    try:
        # 1. Ensure last_ts is a proper UTC-aware Timestamp
        # If last_bar_id is already a string or date, this makes it a Timestamp object
        last_ts = pd.to_datetime(last_bar_id, utc=True)
    except Exception:
        return df

    if "datetime" in df.columns:
        try:
            # 2. Ensure the column is actually datetime objects (UTC aware)
            col_dt = pd.to_datetime(df["datetime"], utc=True)
            
            # 3. Compare the objects directly. Pandas handles the ns precision internally.
            mask = col_dt > last_ts
            
            dropped_count = (~mask).sum()
            if dropped_count > 0:
                logger.debug(f"Dropped {dropped_count} rows with datetime <= {last_bar_id}")
                
            return df[mask]
        except Exception as e:
            logger.error(f"Error in mask comparison: {e}")
            return df

    if "timestamp" in df.columns:
        try:
            # Convert last_ts to milliseconds for timestamp column comparison
            last_ts_ms = int(last_ts.timestamp() * 1000)
            
            # Compare timestamp values
            mask = df["timestamp"].astype(int) > last_ts_ms
            
            dropped_count = (~mask).sum()
            if dropped_count > 0:
                logger.debug(f"Dropped {dropped_count} rows with timestamp <= {last_ts_ms}")
                
            return df[mask]
        except Exception as e:
            logger.error(f"Error in timestamp comparison: {e}")
            return df

    return df


# ---------------------------------------------------------------------------
# last_bar_id: always stored as ISO datetime string
# ---------------------------------------------------------------------------

def _extract_last_bar_id(df: pd.DataFrame) -> Optional[str]:
    """
    Return the last bar's datetime as an ISO format string.
    Tries 'datetime' column first, then 'timestamp' (epoch seconds).
    Returns None if neither column is present or df is empty.
    """
    if df.empty:
        return None

    if "datetime" in df.columns:
        try:
            return pd.Timestamp(df["datetime"].iloc[-1]).isoformat()
        except Exception:
            return str(df["datetime"].iloc[-1])

    if "timestamp" in df.columns:
        try:
            return pd.Timestamp(float(df["timestamp"].iloc[-1]), unit="s").isoformat()
        except Exception:
            return str(df["timestamp"].iloc[-1])

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_table():
    global _table_ensured
    if not _table_ensured:
        try:
            ensure_regime_state_table()
        except Exception:
            pass
        _table_ensured = True


def _resolve_symbol_id(exchange: str, symbol: str) -> Optional[int]:
    try:
        return get_symbol_id(exchange, symbol)
    except Exception:
        return None


def _save_state(
    engine: RegimeEngine,
    symbol_id: int,
    bar_type: str,
    bar_timeframe: Optional[str],
    last_bar_id: Optional[str] = None,
):
    try:
        save_state(symbol_id, bar_type, engine.state, bar_timeframe, last_bar_id)
    except Exception:
        pass
