from bitpredict.common.db.config import get_engine
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from bitpredict.common.logging import get_logger
from bitpredict.common.constants import DATA_SCHEMA, MARKET_REGIME_STATE, META_SYMBOLS_TABLE, META_SCHEMA
import json
from datetime import datetime, timezone
from typing import Optional
from bitpredict.common.market_regimes import RegimeState

logger = get_logger(__name__)

_FULL_TABLE = f"{DATA_SCHEMA}.{MARKET_REGIME_STATE}"
_META_SYMBOLS = f"{META_SCHEMA}.{META_SYMBOLS_TABLE}"


def _tf(bar_timeframe: Optional[str]) -> str:
    """Normalize bar_timeframe: None → '' so it's safe in PRIMARY KEY."""
    return bar_timeframe or ""

# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------

def get_symbol_id(exchange: str, symbol: str) -> int:
    """
    Resolve (exchange, symbol) to the integer id in meta.symbols.
    Raises ValueError if the row does not exist.
    """
    engine = get_engine()
    sql = text(f"""
        SELECT id FROM {_META_SYMBOLS}
        WHERE exchange = :exchange AND symbol = :symbol
        LIMIT 1
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql, {"exchange": exchange, "symbol": symbol}).fetchone()
    except SQLAlchemyError as e:
        logger.error(f"Failed to look up symbol ({exchange}, {symbol}): {e}")
        raise

    if row is None:
        raise ValueError(
            f"Symbol ({exchange}, {symbol}) not found in {_META_SYMBOLS}. "
            "Ensure the symbol is registered there before using the regime engine."
        )
    return int(row[0])


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_state(
    symbol_id: int,
    bar_type: str,
    bar_timeframe: Optional[str] = None,
) -> Optional[RegimeState]:
    """
    Load RegimeState for the given key.
    Returns None if no row exists (caller should call engine.reset()).
    """
    engine = get_engine()
    sql = text(f"""
        SELECT state_json
        FROM {_FULL_TABLE}
        WHERE symbol_id    = :symbol_id
          AND bar_type     = :bar_type
          AND bar_timeframe = :bar_timeframe
        LIMIT 1
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql, {
                "symbol_id": symbol_id,
                "bar_type": bar_type,
                "bar_timeframe": _tf(bar_timeframe),
            }).fetchone()
    except SQLAlchemyError as e:
        logger.error(f"Failed to load state for symbol_id={symbol_id} {bar_type}/{bar_timeframe}: {e}")
        raise

    if row is None:
        logger.debug(f"No state found for symbol_id={symbol_id} {bar_type}/{bar_timeframe}")
        return None

    state_dict = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    state = RegimeState.from_dict(state_dict)
    
    # Validate interval tracking state: if ewma_interval is 0.0, reset initialization flag
    if state.ewma_interval == 0.0 and state.interval_ewma_initialized:
        logger.warning(f"LOAD_STATE: Invalid state detected - ewma_interval=0.0 but interval_ewma_initialized=True. "
                      f"Resetting interval_ewma_initialized to False for symbol_id={symbol_id}")
        state.interval_ewma_initialized = False
    
    logger.info(f"LOAD_STATE: symbol_id={symbol_id} {bar_type}/{bar_timeframe} | "
               f"bars_seen={state.bars_seen}, warmup_complete={state.warmup_complete}, "
               f"ewma_interval={state.ewma_interval}, interval_ewma_initialized={state.interval_ewma_initialized}")
    return state


def load_last_bar_id(
    symbol_id: int,
    bar_type: str,
    bar_timeframe: Optional[str] = None,
) -> Optional[str]:
    """Return last_bar_id only — for deduplication without loading full state."""
    engine = get_engine()
    sql = text(f"""
        SELECT last_bar_id FROM {_FULL_TABLE}
        WHERE symbol_id    = :symbol_id
          AND bar_type     = :bar_type
          AND bar_timeframe = :bar_timeframe
        LIMIT 1
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql, {
                "symbol_id": symbol_id,
                "bar_type": bar_type,
                "bar_timeframe": _tf(bar_timeframe),
            }).fetchone()
        return row[0] if row else None
    except SQLAlchemyError as e:
        logger.error(f"Failed to load last_bar_id for symbol_id={symbol_id}: {e}")
        raise


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_state(
    symbol_id: int,
    bar_type: str,
    state: RegimeState,
    bar_timeframe: Optional[str] = None,
    last_bar_id: Optional[str] = None,
) -> None:
    """Upsert RegimeState for the given key."""
    engine = get_engine()
    state_dict = state.to_dict()
    state_dict["ring_buffer"] = [float(v) for v in state_dict["ring_buffer"]]
    state_json = json.dumps(state_dict)
    now = datetime.now(timezone.utc)

    logger.info(f"SAVE_STATE: symbol_id={symbol_id} {bar_type}/{bar_timeframe} | "
               f"bars_seen={state.bars_seen}, warmup_complete={state.warmup_complete}, "
               f"last_bar_id={last_bar_id}, ewma_interval={state.ewma_interval}, "
               f"interval_ewma_initialized={state.interval_ewma_initialized}")

    sql = text(f"""
        INSERT INTO {_FULL_TABLE}
            (symbol_id, bar_type, bar_timeframe, state_json, warmup_complete, last_bar_id, updated_at)
        VALUES
            (:symbol_id, :bar_type, :bar_timeframe, :state_json,
             :warmup_complete, :last_bar_id, :updated_at)
        ON CONFLICT (symbol_id, bar_type, bar_timeframe)
        DO UPDATE SET
            state_json      = EXCLUDED.state_json,
            warmup_complete = EXCLUDED.warmup_complete,
            last_bar_id     = EXCLUDED.last_bar_id,
            updated_at      = EXCLUDED.updated_at
    """)
    try:
        with engine.begin() as conn:
            conn.execute(sql, {
                "symbol_id": symbol_id,
                "bar_type": bar_type,
                "bar_timeframe": _tf(bar_timeframe),
                "state_json": state_json,
                "warmup_complete": state.warmup_complete,
                "last_bar_id": last_bar_id,
                "updated_at": now,
            })
        logger.debug(f"save_state: Successfully saved state for symbol_id={symbol_id} {bar_type}/{bar_timeframe}")
    except SQLAlchemyError as e:
        logger.error(f"Failed to save state for symbol_id={symbol_id}: {e}")
        raise


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete_state(
    symbol_id: int,
    bar_type: str,
    bar_timeframe: Optional[str] = None,
) -> bool:
    """Delete state row. Returns True if a row was deleted."""
    engine = get_engine()
    sql = text(f"""
        DELETE FROM {_FULL_TABLE}
        WHERE symbol_id    = :symbol_id
          AND bar_type     = :bar_type
          AND bar_timeframe = :bar_timeframe
    """)
    try:
        with engine.begin() as conn:
            result = conn.execute(sql, {
                "symbol_id": symbol_id,
                "bar_type": bar_type,
                "bar_timeframe": _tf(bar_timeframe),
            })
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Deleted state for symbol_id={symbol_id} {bar_type}/{bar_timeframe}")
        return deleted
    except SQLAlchemyError as e:
        logger.error(f"Failed to delete state for symbol_id={symbol_id}: {e}")
        raise


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

def list_state_keys() -> list:
    """Return all tracked (symbol_id, bar_type, bar_timeframe) tuples.
    bar_timeframe is returned as None when stored as ''."""
    engine = get_engine()
    sql = text(f"""
        SELECT symbol_id, bar_type, bar_timeframe
        FROM {_FULL_TABLE}
        ORDER BY symbol_id, bar_type, bar_timeframe
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [(r[0], r[1], r[2] or None) for r in rows]
    except SQLAlchemyError as e:
        logger.error(f"Failed to list state keys: {e}")
        raise
