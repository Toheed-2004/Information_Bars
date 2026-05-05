from sqlalchemy import text, Engine
from sqlalchemy.exc import SQLAlchemyError
from bitpredict.common.db.utils import ensure_schema, hypertable_exists, create_hypertable, ensure_table
import logging

logger = logging.getLogger(__name__)

#======================================================================
# signals
#=======================================================================

def ensure_signal_table(
    engine: Engine,
    schema_name: str = "strategies",
    table_name: str = "signals",
    is_timeseries: bool = True
) -> bool:
    """
    Ensure signals table exists under the given schema.
    Creates table and hypertable if needed.
    Includes created_at and updated_at columns with auto-update trigger.
    """
    if ensure_table(engine, schema_name, table_name):
        return
    try:
        ensure_schema(engine, schema_name)
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Failed to ensure schema '{schema_name}' exists") from exc

    try:
        with engine.begin() as conn:
            # Create the signals table with created_at and updated_at columns
            conn.execute(text(f"""
                CREATE TABLE {schema_name}.{table_name} (
                    strategy_id   BIGINT      NOT NULL,
                    datetime      TIMESTAMPTZ NOT NULL,
                    components    JSONB,
                    signals       INTEGER,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (strategy_id, datetime)
                )
            """))
            logger.debug(f"Successfully created/verified table {schema_name}.{table_name}")
            
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_datetime 
                ON {schema_name}.{table_name} (datetime DESC)
            """))
            
            # Index on signals for filtering by signal value
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_signals 
                ON {schema_name}.{table_name} (signals)
                WHERE signals IS NOT NULL
            """))
            
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e):
            logger.debug(f"Race condition creating {schema_name}.{table_name}, table already exists")
        else:
            logger.error(f"Failed to create table {schema_name}.{table_name}: {e}")
            raise

    # Optional hypertable creation
    if is_timeseries:
        if hypertable_exists(engine, schema_name, table_name):
            logger.debug(f"Hypertable for {schema_name}.{table_name} already exists")
            return True

        try:
            create_hypertable(
                engine=engine,
                schema_name=schema_name,
                table_name=table_name,
                time_column="datetime",
                compress=True,
                compress_segmentby="strategy_id"
            )
            logger.debug(f"Successfully created hypertable for {schema_name}.{table_name}")
        except Exception as e:
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in ["already a hypertable", "already exists", "duplicate key value"]):
                if hypertable_exists(engine, schema_name, table_name):
                    logger.debug(f"Race condition creating hypertable for {schema_name}.{table_name}, exists now")
                    return True
                else:
                    logger.warning(f"Could not create hypertable {schema_name}.{table_name}: {e}")
            else:
                logger.warning(f"Unexpected error creating hypertable {schema_name}.{table_name}: {e}")

    return True