"""Database models for bars data.

This module provides DDL (Data Definition Language) functions for creating
and managing bar-related tables:
- Table schemas for different bar types (volume, volatility, dollar, tick, time)
- State table for tracking bar processing state
- TimescaleDB hypertable conversion support

All functions handle PostgreSQL-specific features and error management.
"""

from sqlalchemy.engine import Engine
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from bitpredict.common.logging import get_logger
from bitpredict.common.db.config import get_engine
from bitpredict.common.db.utils import ensure_schema
from bitpredict.common.db.exceptions import SchemaError, TableError
from bitpredict.common.constants import DATA_SCHEMA
from bitpredict.common.db.utils import create_hypertable, hypertable_exists

logger = get_logger(__name__)



# ============================================================================
# TABLE CREATION - OHLCV
# ============================================================================

def ensure_ohlcv_table(engine: Engine, schema_name: str, table_name: str, is_timeseries: bool = True) -> bool:
    """
    Ensure OHLCV table exists and create hypertable if needed.
    Returns True if table exists or was created.
    """
    
    try:
        with engine.begin() as conn:
            columns_sql = """
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                timestamp BIGINT NOT NULL,
                datetime TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                regime_trend TEXT,
                regime_volatility TEXT,
                regime_momentum TEXT,
                regime_label TEXT,
                regime_confidence DOUBLE PRECISION,
                trend_strength_z DOUBLE PRECISION,
                vol_percentile DOUBLE PRECISION,
                volatility_skew DOUBLE PRECISION,
                transition_pressure DOUBLE PRECISION,
                trend_acceleration DOUBLE PRECISION,
                adaptive_alpha DOUBLE PRECISION,
                up_vol DOUBLE PRECISION,
                down_vol DOUBLE PRECISION,
                regime_stability DOUBLE PRECISION,
                directional_persistence DOUBLE PRECISION,
                score_bull DOUBLE PRECISION,
                score_bear DOUBLE PRECISION,
                score_range DOUBLE PRECISION,
                score_transition DOUBLE PRECISION,
                score_high_vol DOUBLE PRECISION,
                score_low_vol DOUBLE PRECISION,
                score_accelerating DOUBLE PRECISION,
                PRIMARY KEY (exchange, symbol, timeframe, datetime)
                """

            conn.execute(
                text(f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} ({columns_sql})")
            )
            logger.info(f"Successfully created/verified table {schema_name}.{table_name}")
            
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e) and "pg_type_typname_nsp_index" in str(e):
            logger.debug(f"Race condition creating {schema_name}.{table_name}, but table should exist")
            # Table was created by another process, this is fine
            pass
        else:
            # Re-raise other errors
            logger.error(f"Failed to create table {schema_name}.{table_name}: {e}")
            raise

    # Handle hypertable creation if needed
    if is_timeseries:
        # Check if hypertable already exists
        if hypertable_exists(engine, schema_name, table_name):
            logger.info(f"Hypertable for {schema_name}.{table_name} already exists")
            return True
        
        # Try to create hypertable
        try:
            create_hypertable(
                engine=engine,
                schema_name=schema_name,
                table_name=table_name,
                time_column="datetime",
                compress=True,
                compress_segmentby="exchange, symbol, timeframe"
            )
            logger.info(f"Successfully created hypertable for {schema_name}.{table_name}")
            
        except Exception as e:
            # Check if it's a race condition (hypertable was created by another process)
            error_msg = str(e).lower()
            
            # Common error patterns for hypertable race conditions
            if any(pattern in error_msg for pattern in [
                "already a hypertable",
                "hypertable",
                "already exists",
                "duplicate key value"
            ]):
                # Verify if hypertable now exists
                if hypertable_exists(engine, schema_name, table_name):
                    logger.debug(f"Race condition creating hypertable for {schema_name}.{table_name}, but it exists now")
                    return True
                else:
                    logger.warning(
                        "Could not create hypertable for %s.%s: %s",
                        schema_name,
                        table_name,
                        e
                    )
            else:
                # Different error, log warning but don't fail
                logger.warning(
                    "Unexpected error creating hypertable for %s.%s: %s",
                    schema_name,
                    table_name,
                    e
                )

    return True


# ============================================================================
# TABLE CREATION - BARS
# ============================================================================

def ensure_bars_table(bar_type: str) -> str:
    """Create table for storing bars if it doesn't exist.
    
    Routes to specific table creation function based on bar type.
    Ensures schema exists before attempting table creation.
    
    Args:
        bar_type: Type of bars (volume, volatility, dollar, tick, time).
    
    Returns:
        str: Fully qualified table name (schema.bar_type).
        
    Raises:
        SchemaError: If schema creation fails.
        TableError: If table creation fails.
        ValueError: If bar_type is unknown.
    """
    
    if not isinstance(bar_type, str) or not bar_type.strip():
        raise ValueError("bar_type must be a non-empty string")
    
    engine = get_engine()
    
    # Ensure schema exists
    try:
        ensure_schema(engine, DATA_SCHEMA)
    except Exception as exc:
        logger.error("Failed to ensure schema '%s' exists", DATA_SCHEMA)
        raise SchemaError(f"Failed to create or verify schema {DATA_SCHEMA}") from exc
    
    # Normalize names
    bar_type = bar_type.strip().lower()
    table_name = f"{DATA_SCHEMA}.{bar_type}"
    
    # Route to appropriate table creation function
    if bar_type == 'volume':
        return _create_volume_table(engine, table_name)
    elif bar_type == 'volatility':
        return _create_volatility_table(engine, table_name)
    elif bar_type == 'dollar':
        return _create_dollar_table(engine, table_name)
    elif bar_type == 'range':
        return _create_range_table(engine, table_name)
    elif bar_type == 'renko':
        return _create_renko_table(engine, table_name)
    elif bar_type == 'hybrid':
        return _create_hybrid_table(engine, table_name)
    elif bar_type == 'tick':
        return _create_tick_table(engine, table_name)
    elif bar_type == 'time':
        return _create_time_table(engine, table_name)
    else:
        raise ValueError(f"Unknown bar type: {bar_type}")


def _create_volume_table(engine: Engine, table_name: str) -> str:
    """Create TimescaleDB hypertable for volume bars."""
    schema_name = table_name.split('.')[0]
    table_only = table_name.split('.')[-1]
        
    try:
        with engine.begin() as conn:
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                datetime TIMESTAMPTZ NOT NULL,
                datetime_start TIMESTAMPTZ,
                datetime_end TIMESTAMPTZ,
                open DECIMAL(20,8),
                high DECIMAL(20,8),
                low DECIMAL(20,8),
                close DECIMAL(20,8),
                volume DECIMAL(20,8) DEFAULT 0,
                bar_size DECIMAL(20,8) DEFAULT 0,
                dollar_volume DECIMAL(40,20) DEFAULT 0,
                duration_minutes INTEGER DEFAULT 0,
                tick_count INTEGER DEFAULT 0,
                bar_return DECIMAL(10,6) DEFAULT 0,
                price_range DECIMAL(10,6) DEFAULT 0,
                close_position DECIMAL(10,6) DEFAULT 0,
                regime_trend TEXT,
                regime_volatility TEXT,
                regime_momentum TEXT,
                regime_label TEXT,
                regime_confidence DOUBLE PRECISION,
                trend_strength_z DOUBLE PRECISION,
                vol_percentile DOUBLE PRECISION,
                volatility_skew DOUBLE PRECISION,
                transition_pressure DOUBLE PRECISION,
                trend_acceleration DOUBLE PRECISION,
                adaptive_alpha DOUBLE PRECISION,
                up_vol DOUBLE PRECISION,
                down_vol DOUBLE PRECISION,
                regime_stability DOUBLE PRECISION,
                directional_persistence DOUBLE PRECISION,
                score_bull DOUBLE PRECISION,
                score_bear DOUBLE PRECISION,
                score_range DOUBLE PRECISION,
                score_transition DOUBLE PRECISION,
                score_high_vol DOUBLE PRECISION,
                score_low_vol DOUBLE PRECISION,
                score_accelerating DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (datetime, exchange, symbol)
            )
            """
            conn.execute(text(create_query))
            logger.info("Created base table %s", table_name)
            
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e) and "pg_type_typname_nsp_index" in str(e):
            logger.debug(f"Race condition creating {table_name}, but table should exist")
            # Table was created by another process, this is fine
        else:
            logger.exception("Failed to create volume table %s", table_name)
            raise TableError(f"Failed to create table {table_name}") from e

    # Convert to hypertable after table creation
    if not hypertable_exists(engine, schema_name, table_only):
        try:
            create_hypertable(engine=engine, schema_name=schema_name,
                              table_name=table_only, time_column="datetime", 
                              compress=True, compress_segmentby="exchange, symbol")
            logger.info("Converted %s to hypertable", table_name)
        except Exception as e:
            # Check if another process created it
            if hypertable_exists(engine, schema_name, table_only):
                logger.info(f"Hypertable for {table_name} was created by another process")
            else:
                logger.warning("Could not create hypertable for %s: %s", table_name, e)

    logger.info("Created volume table %s", table_name)
    return table_name


def _create_volatility_table(engine: Engine, table_name: str) -> str:
    """Create TimescaleDB hypertable for volatility bars."""
    schema_name = table_name.split('.')[0]
    table_only = table_name.split('.')[-1]
    
        
    try:
        with engine.begin() as conn:
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                datetime TIMESTAMPTZ NOT NULL,
                datetime_start TIMESTAMPTZ,
                datetime_end TIMESTAMPTZ,
                open DECIMAL(20,8),
                high DECIMAL(20,8),
                low DECIMAL(20,8),
                close DECIMAL(20,8),
                volume DECIMAL(20,8) DEFAULT 0,
                bar_size DECIMAL(20,8) DEFAULT 0,
                dollar_volume DECIMAL(40,20) DEFAULT 0,
                duration_minutes INTEGER DEFAULT 0,
                tick_count INTEGER DEFAULT 0,
                bar_return DECIMAL(10,6) DEFAULT 0,
                price_range DECIMAL(10,6) DEFAULT 0,
                close_position DECIMAL(10,6) DEFAULT 0,
                regime_trend TEXT,
                regime_volatility TEXT,
                regime_momentum TEXT,
                regime_label TEXT,
                regime_confidence DOUBLE PRECISION,
                trend_strength_z DOUBLE PRECISION,
                vol_percentile DOUBLE PRECISION,
                volatility_skew DOUBLE PRECISION,
                transition_pressure DOUBLE PRECISION,
                trend_acceleration DOUBLE PRECISION,
                adaptive_alpha DOUBLE PRECISION,
                up_vol DOUBLE PRECISION,
                down_vol DOUBLE PRECISION,
                regime_stability DOUBLE PRECISION,
                directional_persistence DOUBLE PRECISION,
                score_bull DOUBLE PRECISION,
                score_bear DOUBLE PRECISION,
                score_range DOUBLE PRECISION,
                score_transition DOUBLE PRECISION,
                score_high_vol DOUBLE PRECISION,
                score_low_vol DOUBLE PRECISION,
                score_accelerating DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (datetime, exchange, symbol)
            )
            """
            conn.execute(text(create_query))
            logger.info("Created base table %s", table_name)
            
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e) and "pg_type_typname_nsp_index" in str(e):
            logger.debug(f"Race condition creating {table_name}, but table should exist")
            # Table was created by another process, this is fine
        else:
            logger.exception("Failed to create volatility table %s", table_name)
            raise TableError(f"Failed to create table {table_name}") from e

    # Convert to hypertable after table creation
    if not hypertable_exists(engine, schema_name, table_only):
        try:
            create_hypertable(engine=engine, schema_name=schema_name,
                              table_name=table_only, time_column="datetime", 
                              compress=True, compress_segmentby="exchange, symbol")
            logger.info("Converted %s to hypertable", table_name)
        except Exception as e:
            if hypertable_exists(engine, schema_name, table_only):
                logger.info(f"Hypertable for {table_name} was created by another process")
            else:
                logger.warning("Could not create hypertable for %s: %s", table_name, e)

    logger.info("Created volatility table %s", table_name)
    return table_name


def _create_dollar_table(engine: Engine, table_name: str) -> str:
    """Create TimescaleDB hypertable for dollar bars."""
    schema_name = table_name.split('.')[0]
    table_only = table_name.split('.')[-1]
        
    try:
        with engine.begin() as conn:
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                datetime TIMESTAMPTZ NOT NULL,
                datetime_start TIMESTAMPTZ,
                datetime_end TIMESTAMPTZ,
                open DECIMAL(20,8),
                high DECIMAL(20,8),
                low DECIMAL(20,8),
                close DECIMAL(20,8),
                volume DECIMAL(20,8) DEFAULT 0,
                bar_size DECIMAL(40,20) DEFAULT 0,
                vwap DECIMAL(20,8) DEFAULT 0,
                duration_minutes INTEGER DEFAULT 0,
                tick_count INTEGER DEFAULT 0,
                bar_return DECIMAL(10,6) DEFAULT 0,
                price_range DECIMAL(10,6) DEFAULT 0,
                close_position DECIMAL(10,6) DEFAULT 0,
                regime_trend TEXT,
                regime_volatility TEXT,
                regime_momentum TEXT,
                regime_label TEXT,
                regime_confidence DOUBLE PRECISION,
                trend_strength_z DOUBLE PRECISION,
                vol_percentile DOUBLE PRECISION,
                volatility_skew DOUBLE PRECISION,
                transition_pressure DOUBLE PRECISION,
                trend_acceleration DOUBLE PRECISION,
                adaptive_alpha DOUBLE PRECISION,
                up_vol DOUBLE PRECISION,
                down_vol DOUBLE PRECISION,
                regime_stability DOUBLE PRECISION,
                directional_persistence DOUBLE PRECISION,
                score_bull DOUBLE PRECISION,
                score_bear DOUBLE PRECISION,
                score_range DOUBLE PRECISION,
                score_transition DOUBLE PRECISION,
                score_high_vol DOUBLE PRECISION,
                score_low_vol DOUBLE PRECISION,
                score_accelerating DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (datetime, exchange, symbol)
            )
            """
            conn.execute(text(create_query))
            logger.info("Created dollar bars table %s", table_name)
            
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e) and "pg_type_typname_nsp_index" in str(e):
            logger.debug(f"Race condition creating {table_name}, but table should exist")
            # Table was created by another process, this is fine
        else:
            logger.exception("Failed to create dollar table %s", table_name)
            raise TableError(f"Failed to create table {table_name}") from e

    # Convert to hypertable after table creation
    if not hypertable_exists(engine, schema_name, table_only):
        try:
            create_hypertable(engine=engine, schema_name=schema_name,
                              table_name=table_only, time_column="datetime", 
                              compress=True, compress_segmentby="exchange, symbol")
            logger.info("Created TimescaleDB hypertable for %s", table_name)
        except Exception as e:
            if hypertable_exists(engine, schema_name, table_only):
                logger.info(f"Hypertable for {table_name} was created by another process")
            else:
                logger.warning("Could not create TimescaleDB hypertable for %s: %s", table_name, e)

    logger.info("Created dollar table %s", table_name)
    return table_name


def _create_range_table(engine: Engine, table_name: str) -> str:
    """Create TimescaleDB hypertable for range bars."""
    schema_name = table_name.split('.')[0]
    table_only = table_name.split('.')[-1]
        
    try:
        with engine.begin() as conn:
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                datetime TIMESTAMPTZ NOT NULL,
                datetime_start TIMESTAMPTZ,
                datetime_end TIMESTAMPTZ,
                open DECIMAL(20,8),
                high DECIMAL(20,8),
                low DECIMAL(20,8),
                close DECIMAL(20,8),
                volume DECIMAL(20,8) DEFAULT 0,
                bar_size DECIMAL(20,8) DEFAULT 0,
                dollar_volume DECIMAL(40,20) DEFAULT 0,
                duration_minutes INTEGER DEFAULT 0,
                tick_count INTEGER DEFAULT 0,
                bar_return DECIMAL(10,6) DEFAULT 0,
                price_range DECIMAL(10,6) DEFAULT 0,
                close_position DECIMAL(10,6) DEFAULT 0,
                regime_trend TEXT,
                regime_volatility TEXT,
                regime_momentum TEXT,
                regime_label TEXT,
                regime_confidence DOUBLE PRECISION,
                trend_strength_z DOUBLE PRECISION,
                vol_percentile DOUBLE PRECISION,
                volatility_skew DOUBLE PRECISION,
                transition_pressure DOUBLE PRECISION,
                trend_acceleration DOUBLE PRECISION,
                adaptive_alpha DOUBLE PRECISION,
                up_vol DOUBLE PRECISION,
                down_vol DOUBLE PRECISION,
                regime_stability DOUBLE PRECISION,
                directional_persistence DOUBLE PRECISION,
                score_bull DOUBLE PRECISION,
                score_bear DOUBLE PRECISION,
                score_range DOUBLE PRECISION,
                score_transition DOUBLE PRECISION,
                score_high_vol DOUBLE PRECISION,
                score_low_vol DOUBLE PRECISION,
                score_accelerating DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (datetime, exchange, symbol)
            )
            """
            conn.execute(text(create_query))
            logger.info("Created base table %s", table_name)
            
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e) and "pg_type_typname_nsp_index" in str(e):
            logger.debug(f"Race condition creating {table_name}, but table should exist")
            # Table was created by another process, this is fine
        else:
            logger.exception("Failed to create range table %s", table_name)
            raise TableError(f"Failed to create table {table_name}") from e

    # Convert to hypertable after table creation
    if not hypertable_exists(engine, schema_name, table_only):
        try:
            create_hypertable(engine=engine, schema_name=schema_name,
                              table_name=table_only, time_column="datetime", 
                              compress=True, compress_segmentby="exchange, symbol")
            logger.info("Converted %s to hypertable", table_name)
        except Exception as e:
            if hypertable_exists(engine, schema_name, table_only):
                logger.info(f"Hypertable for {table_name} was created by another process")
            else:
                logger.warning("Could not create hypertable for %s: %s", table_name, e)

    logger.info("Created range table %s", table_name)
    return table_name


def _create_renko_table(engine: Engine, table_name: str) -> str:
    """Create TimescaleDB hypertable for renko bars."""
    schema_name = table_name.split('.')[0]
    table_only = table_name.split('.')[-1]
    

        
    try:
        with engine.begin() as conn:
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                datetime TIMESTAMPTZ NOT NULL,
                datetime_start TIMESTAMPTZ,
                datetime_end TIMESTAMPTZ,
                open DECIMAL(20,8),
                high DECIMAL(20,8),
                low DECIMAL(20,8),
                close DECIMAL(20,8),
                volume DECIMAL(20,8) DEFAULT 0,
                bar_size DECIMAL(20,8) DEFAULT 0,
                dollar_volume DECIMAL(40,20) DEFAULT 0,
                direction TEXT,
                duration_minutes INTEGER DEFAULT 0,
                tick_count INTEGER DEFAULT 0,
                bar_return DECIMAL(10,6) DEFAULT 0,
                price_range DECIMAL(10,6) DEFAULT 0,
                close_position DECIMAL(10,6) DEFAULT 0,
                regime_trend TEXT,
                regime_volatility TEXT,
                regime_momentum TEXT,
                regime_label TEXT,
                regime_confidence DOUBLE PRECISION,
                trend_strength_z DOUBLE PRECISION,
                vol_percentile DOUBLE PRECISION,
                volatility_skew DOUBLE PRECISION,
                transition_pressure DOUBLE PRECISION,
                trend_acceleration DOUBLE PRECISION,
                adaptive_alpha DOUBLE PRECISION,
                up_vol DOUBLE PRECISION,
                down_vol DOUBLE PRECISION,
                regime_stability DOUBLE PRECISION,
                directional_persistence DOUBLE PRECISION,
                score_bull DOUBLE PRECISION,
                score_bear DOUBLE PRECISION,
                score_range DOUBLE PRECISION,
                score_transition DOUBLE PRECISION,
                score_high_vol DOUBLE PRECISION,
                score_low_vol DOUBLE PRECISION,
                score_accelerating DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (datetime, exchange, symbol)
            )
            """
            conn.execute(text(create_query))
            logger.info("Created base table %s", table_name)
            
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e) and "pg_type_typname_nsp_index" in str(e):
            logger.debug(f"Race condition creating {table_name}, but table should exist")
            # Table was created by another process, this is fine
        else:
            logger.exception("Failed to create renko table %s", table_name)
            raise TableError(f"Failed to create table {table_name}") from e

    # Convert to hypertable after table creation
    if not hypertable_exists(engine, schema_name, table_only):
        try:
            create_hypertable(engine=engine, schema_name=schema_name,
                              table_name=table_only, time_column="datetime", 
                              compress=True, compress_segmentby="exchange, symbol")
            logger.info("Converted %s to hypertable", table_name)
        except Exception as e:
            if hypertable_exists(engine, schema_name, table_only):
                logger.info(f"Hypertable for {table_name} was created by another process")
            else:
                logger.warning("Could not create hypertable for %s: %s", table_name, e)

    logger.info("Created renko table %s", table_name)
    return table_name


def _create_hybrid_table(engine: Engine, table_name: str) -> str:
    """Create TimescaleDB hypertable for hybrid bars."""
    schema_name = table_name.split('.')[0]
    table_only = table_name.split('.')[-1]
    
    try:
        with engine.begin() as conn:
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                datetime TIMESTAMPTZ NOT NULL,
                datetime_start TIMESTAMPTZ,
                datetime_end TIMESTAMPTZ,
                open DECIMAL(20,8),
                high DECIMAL(20,8),
                low DECIMAL(20,8),
                close DECIMAL(20,8),
                volume DECIMAL(20,8) DEFAULT 0,
                bar_size DECIMAL(40,20) DEFAULT 0,
                vwap DECIMAL(20,8) DEFAULT 0,
                bar_volatility DECIMAL(10,8) DEFAULT 0,
                duration_minutes INTEGER DEFAULT 0,
                tick_count INTEGER DEFAULT 0,
                bar_return DECIMAL(10,6) DEFAULT 0,
                price_range DECIMAL(10,6) DEFAULT 0,
                close_position DECIMAL(10,6) DEFAULT 0,
                regime_trend TEXT,
                regime_volatility TEXT,
                regime_momentum TEXT,
                regime_label TEXT,
                regime_confidence DOUBLE PRECISION,
                trend_strength_z DOUBLE PRECISION,
                vol_percentile DOUBLE PRECISION,
                volatility_skew DOUBLE PRECISION,
                transition_pressure DOUBLE PRECISION,
                trend_acceleration DOUBLE PRECISION,
                adaptive_alpha DOUBLE PRECISION,
                up_vol DOUBLE PRECISION,
                down_vol DOUBLE PRECISION,
                regime_stability DOUBLE PRECISION,
                directional_persistence DOUBLE PRECISION,
                score_bull DOUBLE PRECISION,
                score_bear DOUBLE PRECISION,
                score_range DOUBLE PRECISION,
                score_transition DOUBLE PRECISION,
                score_high_vol DOUBLE PRECISION,
                score_low_vol DOUBLE PRECISION,
                score_accelerating DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (datetime, exchange, symbol)
            )
            """
            conn.execute(text(create_query))
            logger.info("Created base table %s", table_name)
            
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e) and "pg_type_typname_nsp_index" in str(e):
            logger.debug(f"Race condition creating {table_name}, but table should exist")
            # Table was created by another process, this is fine
        else:
            logger.exception("Failed to create hybrid table %s", table_name)
            raise TableError(f"Failed to create table {table_name}") from e

    # Convert to hypertable after table creation
    if not hypertable_exists(engine, schema_name, table_only):
        try:
            create_hypertable(engine=engine, schema_name=schema_name,
                              table_name=table_only, time_column="datetime", 
                              compress=True, compress_segmentby="exchange, symbol")
            logger.info("Converted %s to hypertable", table_name)
        except Exception as e:
            if hypertable_exists(engine, schema_name, table_only):
                logger.info(f"Hypertable for {table_name} was created by another process")
            else:
                logger.warning("Could not create hypertable for %s: %s", table_name, e)

    logger.info("Created hybrid table %s", table_name)
    return table_name


def _create_tick_table(engine: Engine, table_name: str) -> str:
    """Create TimescaleDB hypertable for tick bars."""
    return _create_standard_bar_table(engine, table_name, "tick")


def _create_time_table(engine: Engine, table_name: str) -> str:
    """Create TimescaleDB hypertable for time bars."""
    return _create_standard_bar_table(engine, table_name, "time")


def _create_standard_bar_table(engine: Engine, table_name: str, bar_type: str) -> str:
    """Create standard bar table with TimescaleDB hypertable."""
    schema_name = table_name.split('.')[0]
    table_only = table_name.split('.')[-1]
        
    try:
        with engine.begin() as conn:
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                timestamp TIMESTAMPTZ PRIMARY KEY,
                open DECIMAL(20,8) NOT NULL,
                high DECIMAL(20,8) NOT NULL,
                low DECIMAL(20,8) NOT NULL,
                close DECIMAL(20,8) NOT NULL,
                volume DECIMAL(20,8) NOT NULL,
                bar_size DECIMAL(40,20) NOT NULL,
                tick_count INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
            conn.execute(text(create_query))
            logger.info("Created base table %s", table_name)
            
            # Create index
            index_query = f"""
            CREATE INDEX IF NOT EXISTS idx_{table_only}_timestamp 
            ON {table_name} (timestamp DESC)
            """
            conn.execute(text(index_query))
            
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e) and "pg_type_typname_nsp_index" in str(e):
            logger.debug(f"Race condition creating {table_name}, but table should exist")
            # Table was created by another process, this is fine
        else:
            logger.exception("Failed to create %s table %s", bar_type, table_name)
            raise TableError(f"Failed to create table {table_name}") from e

    # Convert to hypertable after table creation
    if not hypertable_exists(engine, schema_name, table_only):
        try:
            create_hypertable(engine=engine, schema_name=schema_name,
                              table_name=table_only, time_column="timestamp", 
                              compress=True, compress_segmentby="exchange, symbol")
            logger.info("Converted %s to hypertable", table_name)
        except Exception as e:
            if hypertable_exists(engine, schema_name, table_only):
                logger.info(f"Hypertable for {table_name} was created by another process")
            else:
                logger.warning("Could not create hypertable for %s: %s", table_name, e)
    
    logger.info("Created %s bar table %s", bar_type, table_name)
    return table_name


# ============================================================================
# TABLE CREATION - BAR QUALITY STATS
# ============================================================================

def ensure_bar_stats_table() -> str:
    """Create bar quality stats table if it doesn't exist."""
    engine = get_engine()
    
    try:
        ensure_schema(engine, DATA_SCHEMA)
    except Exception as exc:
        raise SchemaError(f"Failed to create or verify schema {DATA_SCHEMA}") from exc

    table_name = f"{DATA_SCHEMA}.bars_quality_stats"    

    try:
        with engine.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    exchange                    VARCHAR(50)  NOT NULL,
                    symbol                      VARCHAR(50)  NOT NULL,
                    bar_type                    VARCHAR(50)  NOT NULL,
                    total_bars                  INTEGER,
                    date_range_start            TIMESTAMPTZ,
                    date_range_end              TIMESTAMPTZ,
                    calendar_days               INTEGER,
                    mean_bars_per_day           DOUBLE PRECISION,
                    std_bars_per_day            DOUBLE PRECISION,
                    bar_size_mean               DOUBLE PRECISION,
                    bar_size_std                DOUBLE PRECISION,
                    bar_size_cv                 DOUBLE PRECISION,
                    bar_size_p5                 DOUBLE PRECISION,
                    bar_size_p25                DOUBLE PRECISION,
                    bar_size_p50                DOUBLE PRECISION,
                    bar_size_p75                DOUBLE PRECISION,
                    bar_size_p95                DOUBLE PRECISION,
                    duration_mean               DOUBLE PRECISION,
                    duration_std                DOUBLE PRECISION,
                    duration_cv                 DOUBLE PRECISION,
                    duration_p95                DOUBLE PRECISION,
                    tick_count_mean             DOUBLE PRECISION,
                    return_mean                 DOUBLE PRECISION,
                    return_std                  DOUBLE PRECISION,
                    return_skew                 DOUBLE PRECISION,
                    return_kurtosis             DOUBLE PRECISION,
                    return_entropy              DOUBLE PRECISION,
                    return_autocorr_lag1        DOUBLE PRECISION,
                    abs_return_autocorr_lag1    DOUBLE PRECISION,
                    variance_ratio_lag2         DOUBLE PRECISION,
                    variance_ratio_lag5         DOUBLE PRECISION,
                    rolling_vol_cv              DOUBLE PRECISION,
                    eff_sample_size             INTEGER,
                    pct_valid_bars              DOUBLE PRECISION,
                    close_position_mean         DOUBLE PRECISION,
                    close_position_std          DOUBLE PRECISION,
                    price_range_mean            DOUBLE PRECISION,
                    sampling_score              DOUBLE PRECISION,
                    ml_score                    DOUBLE PRECISION,
                    integrity_score             DOUBLE PRECISION,
                    quality_score               DOUBLE PRECISION,
                    bars_used                   INTEGER,
                    computed_at                 TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (exchange, symbol, bar_type)
                )
            """))
        logger.info("Created bar quality stats table: %s", table_name)
        return table_name

    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e) and "pg_type_typname_nsp_index" in str(e):
            logger.debug(f"Race condition creating {table_name}, but table should exist")
            return table_name
        else:
            logger.exception("Failed to create bar stats table %s", table_name)
            raise TableError(f"Failed to create table {table_name}") from e

# ============================================================================
# TABLE CREATION - STATE
# ============================================================================

def ensure_state_table() -> str:
    """Create state table for tracking bar processing state."""
    engine = get_engine()
    
    # Ensure schema exists
    try:
        ensure_schema(engine, DATA_SCHEMA)
    except Exception as exc:
        logger.error("Failed to ensure schema '%s' exists", DATA_SCHEMA)
        raise SchemaError(f"Failed to create or verify schema {DATA_SCHEMA}") from exc
    
    table_name = f"{DATA_SCHEMA}.bars_state"
    table_only = "bars_state"
    
    try:
        with engine.begin() as conn:
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id BIGSERIAL PRIMARY KEY,
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                bar_type VARCHAR(50) NOT NULL,
                last_processed_datetime TIMESTAMPTZ,
                current_bar_datetime TIMESTAMPTZ,
                current_bar_data JSONB,
                ema_state JSONB,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(exchange, symbol, bar_type)
            )
            """
            conn.execute(text(create_query))

            # Create index if not exists
            index_query = f"""
            CREATE INDEX IF NOT EXISTS idx_bars_state_lookup
            ON {table_name} (exchange, symbol, bar_type)
            """
            conn.execute(text(index_query))

            logger.info("Created state table %s", table_name)
            return table_name

    except Exception as e:
        error_msg = str(e)
        
        # Check for race condition errors (both table type and sequence)
        if ("duplicate key value violates unique constraint" in error_msg and 
            ("pg_type_typname_nsp_index" in error_msg or "pg_class_relname_nsp_index" in error_msg)):
            
            logger.debug(f"Race condition creating {table_name}, checking if table exists...")
            
        else:
            # Different error, re-raise
            logger.exception("Failed to create state table %s", table_name)
            raise TableError(f"Failed to create table {table_name}") from e