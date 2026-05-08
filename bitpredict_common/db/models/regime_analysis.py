from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine
from sqlalchemy import text, inspect

from bitpredict.common.db.models.simulator import ensure_simulator_analytics_table
from bitpredict.common.db.models.strategies import ensure_strategies_training_table
from bitpredict.common.db.exceptions import SchemaError, TableError
from bitpredict.common.db.utils import ensure_schema, ensure_table
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)

def ensure_regime_analysis_column_in_analytics_table(
    engine: Engine,
    schema_name: str = "simulator",
    table_name: str = "analytics"
) -> None:
    """
    Ensure the simulator.analytics table exists and has a 'regime_analysis' column.
    First creates the analytics table with all its columns, then adds regime_analysis column.
    Idempotent - safe to call on every startup.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine for database connection
    schema_name : str, default "simulator"
        Schema name where the table exists
    table_name : str, default "analytics"
        Table name to add the column to
    """
    # First ensure the analytics table exists with all its standard columns
    if table_name == 'analytics':
        ensure_simulator_analytics_table(engine)
    else:
        ensure_strategies_training_table(engine)
    
    full_table_name = f"{schema_name}.{table_name}"

    with engine.connect() as conn:
        if engine.dialect.name == 'postgresql':
            # Add regime_analysis column if it doesn't exist
            conn.execute(text(f"""
                ALTER TABLE {full_table_name}
                ADD COLUMN IF NOT EXISTS regime_analysis JSONB
            """))

        elif engine.dialect.name == 'sqlite':
            # SQLite does not support schemas or IF NOT EXISTS on ALTER TABLE
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    strategy_id TEXT PRIMARY KEY,
                    regime_analysis TEXT
                )
            """))

            # Inspect to check if column exists before ALTER
            inspector = inspect(engine)
            try:
                existing_columns = [
                    col['name']
                    for col in inspector.get_columns(table_name)
                ]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to inspect {table_name} after CREATE TABLE IF NOT EXISTS. "
                    f"Original error: {e}"
                )

            if 'regime_analysis' not in existing_columns:
                conn.execute(text(
                    f"ALTER TABLE {table_name} ADD COLUMN regime_analysis TEXT"
                ))

        else:
            # Generic fallback
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {full_table_name} (
                    strategy_id VARCHAR(36) PRIMARY KEY,
                    regime_analysis TEXT
                )
            """))

            inspector = inspect(engine)
            try:
                existing_columns = [
                    col['name']
                    for col in inspector.get_columns(table_name, schema=schema_name)
                ]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to inspect {full_table_name} after CREATE TABLE IF NOT EXISTS. "
                    f"Original error: {e}"
                )

            if 'regime_analysis' not in existing_columns:
                conn.execute(text(
                    f"ALTER TABLE {full_table_name} ADD COLUMN regime_analysis TEXT"
                ))

        conn.commit()

    logger.info(f"'regime_analysis' column verified/created in {full_table_name}")
  