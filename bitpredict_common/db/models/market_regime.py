from bitpredict.common.db.config import get_engine
from bitpredict.common.db.utils import ensure_schema
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from bitpredict.common.logging import get_logger
from bitpredict.common.constants import DATA_SCHEMA, MARKET_REGIME_STATE, META_SYMBOLS_TABLE, META_SCHEMA


logger = get_logger(__name__)


_FULL_TABLE = f"{DATA_SCHEMA}.{MARKET_REGIME_STATE}"
_META_SYMBOLS = f"{META_SCHEMA}.{META_SYMBOLS_TABLE}"
# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

def ensure_regime_state_table() -> None:
    """
    Create market_data.market_regime_state if it does not exist.
    Idempotent — safe to call on every startup.
    bar_timeframe is NOT NULL DEFAULT '' — empty string used when timeframe is absent.
    """
    engine = get_engine()
    ensure_schema(engine, DATA_SCHEMA)

    ddl = f"""
        CREATE TABLE IF NOT EXISTS {_FULL_TABLE} (
            symbol_id       INTEGER     NOT NULL
                                        REFERENCES {_META_SYMBOLS}(id) ON DELETE CASCADE,
            bar_type        TEXT        NOT NULL,
            bar_timeframe   TEXT        NOT NULL DEFAULT '',
            state_json      JSONB       NOT NULL,
            warmup_complete BOOLEAN     NOT NULL DEFAULT FALSE,
            last_bar_id     TEXT,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (symbol_id, bar_type, bar_timeframe)
        )
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.debug(f"Ensured table {_FULL_TABLE}")
    except SQLAlchemyError as e:
        logger.error(f"Failed to ensure table {_FULL_TABLE}: {e}")
        raise