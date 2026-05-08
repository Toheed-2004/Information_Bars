from sqlalchemy.engine import Engine
from bitpredict.common.db.utils import ensure_schema, ensure_table
from bitpredict.common.constants import *
from bitpredict.common.db.exceptions import SchemaError, TableError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DATABASE SETUP - MASTER SYMBOLS TABLE
# ---------------------------------------------------------------------------


def ensure_symbols_table(engine: Engine) -> None:
    """Ensure the meta.symbols table exists with proper schema and indexes.

    Creates the master symbols table that serves as the central registry for all
    exchange-symbol pairs. Other tables (ohlcv, bars) reference this table via
    foreign keys to maintain data consistency.

    The table schema includes:
        - id: Auto-incrementing primary key
        - exchange: Text field for exchange name (NOT NULL)
        - symbol: Text field for symbol/asset name (NOT NULL)
        - enabled: Boolean flag for symbol status (NOT NULL, default true)
        - created_at: Timestamp with timezone (defaults to now())
        - updated_at: Timestamp with timezone (updates on change)
        - UNIQUE constraint on (exchange, symbol)

    Indexes created:
        - idx_symbols_enabled: On enabled column
        - idx_symbols_exchange: On exchange column

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.

    Raises:
        SchemaError: If schema creation fails.
        TableError: If table creation fails.
        TypeError: If engine is None.

    Example:
        >>> engine = get_engine()
        >>> ensure_symbols_table(engine)
        # Creates meta.symbols table if not exists

    Notes:
        - This is the master table that ohlcv and bars tables reference
        - Enabling/disabling a symbol here affects all dependent tables
        - Updated_at trigger auto-updates on row changes
    """

    if engine is None:
        raise TypeError("Database engine is None; cannot create table")

    logger.debug("Ensuring schema '%s' exists", META_SCHEMA)

    try:
        ensure_schema(engine, META_SCHEMA)
    except Exception as exc:
        logger.error("Failed to ensure schema '%s' exists", META_SCHEMA)
        raise SchemaError(f"Failed to create or verify schema {META_SCHEMA}") from exc

    # Check if table already exists
    try:
        if ensure_table(engine, META_SCHEMA, "symbols"):
            logger.debug("Table %s.symbols already exists", META_SCHEMA)
            return
    except Exception as exc:
        logger.error("Failed to check if table %s.symbols exists", META_SCHEMA)
        raise TableError(
            f"Failed to verify table existence for {META_SCHEMA}.symbols"
        ) from exc

    logger.info("Creating table %s.symbols with indexes", META_SCHEMA)

    ddl = f"""
    CREATE TABLE {META_SCHEMA}.symbols (
        id SERIAL PRIMARY KEY,
        exchange TEXT NOT NULL,
        symbol TEXT NOT NULL,
        enabled BOOLEAN NOT NULL DEFAULT true,
        created_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        UNIQUE (exchange, symbol)
    );

    CREATE INDEX idx_symbols_enabled 
        ON {META_SCHEMA}.symbols (enabled);
    
    CREATE INDEX idx_symbols_exchange 
        ON {META_SCHEMA}.symbols (exchange);
        
    -- Create trigger for updated_at
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = now();
        RETURN NEW;
    END;
    $$ language 'plpgsql';

    CREATE TRIGGER update_symbols_updated_at 
        BEFORE UPDATE ON {META_SCHEMA}.symbols
        FOR EACH ROW 
        EXECUTE FUNCTION update_updated_at_column();
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info(
            "Table %s.symbols created successfully with 2 indexes and updated_at trigger",
            META_SCHEMA,
        )
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.symbols", META_SCHEMA)
        raise TableError(f"Failed to create table {META_SCHEMA}.symbols") from exc


# ---------------------------------------------------------------------------
# DATABASE SETUP - OHLCV
# ---------------------------------------------------------------------------


def ensure_time_bars_table(engine: Engine) -> None:
    """Ensure the meta.time_bars table exists with proper schema and indexes.

    Creates the time_bars table that references the symbols table via foreign key.
    Each row represents time bars configuration for a specific symbol.

    The table schema includes:
        - id: Auto-incrementing primary key
        - symbol_id: Foreign key to symbols.id (NOT NULL, UNIQUE)
        - timeframes: JSONB field for available timeframes (NOT NULL)
        - created_at: Timestamp with timezone (defaults to now())
        - updated_at: Timestamp with timezone (updates on change)

    Foreign Key:
        - symbol_id references symbols(id) ON DELETE CASCADE
        - This ensures automatic cleanup when a symbol is deleted

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.

    Raises:
        SchemaError: If schema creation fails.
        TableError: If table creation fails.
        TypeError: If engine is None.

    Example:
        >>> engine = get_engine()
        >>> ensure_ohlcv_table(engine)
        # Creates meta.ohlcv table if not exists

    Notes:
        - Must be created AFTER symbols table
        - symbol_id is UNIQUE, enforcing one OHLCV config per symbol
        - CASCADE delete removes ohlcv entries when symbol is deleted
        - Exchange/symbol info is retrieved via JOIN with symbols
    """

    if engine is None:
        raise TypeError("Database engine is None; cannot create table")

    logger.debug("Ensuring schema '%s' exists", META_SCHEMA)

    try:
        ensure_schema(engine, META_SCHEMA)
    except Exception as exc:
        logger.error("Failed to ensure schema '%s' exists", META_SCHEMA)
        raise SchemaError(f"Failed to create or verify schema {META_SCHEMA}") from exc

    # Check if table already exists
    try:
        if ensure_table(engine, META_SCHEMA, META_TIME_BARS_TABLE):
            logger.debug("Table %s.%s already exists", META_SCHEMA, META_TIME_BARS_TABLE)
            return
    except Exception as exc:
        logger.error("Failed to check if table %s.%s exists", META_SCHEMA, META_TIME_BARS_TABLE)
        raise TableError(
            f"Failed to verify table existence for {META_SCHEMA}.{META_TIME_BARS_TABLE}"
        ) from exc

    logger.info("Creating table %s.%s with foreign key", META_SCHEMA, META_TIME_BARS_TABLE)

    ddl = f"""
    CREATE TABLE {META_SCHEMA}.{META_TIME_BARS_TABLE} (
        id SERIAL PRIMARY KEY,
        symbol_id INTEGER NOT NULL UNIQUE,
        timeframes JSONB NOT NULL,
        created_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        CONSTRAINT fk_ohlcv_symbol 
            FOREIGN KEY (symbol_id) 
            REFERENCES {META_SCHEMA}.symbols(id) 
            ON DELETE CASCADE
    );

    CREATE INDEX idx_time_bars_symbol_id 
        ON {META_SCHEMA}.{META_TIME_BARS_TABLE} (symbol_id);
        
    -- Create trigger for updated_at
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = now();
        RETURN NEW;
    END;
    $$ language 'plpgsql';

    CREATE TRIGGER update_time_bars_updated_at 
        BEFORE UPDATE ON {META_SCHEMA}.{META_TIME_BARS_TABLE}
        FOR EACH ROW 
        EXECUTE FUNCTION update_updated_at_column();
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info(
            "Table %s.%s created successfully with foreign key and updated_at trigger",
            META_SCHEMA,
            META_TIME_BARS_TABLE,
        )
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.%s", META_SCHEMA, META_TIME_BARS_TABLE)
        raise TableError(f"Failed to create table {META_SCHEMA}.{META_TIME_BARS_TABLE}") from exc


# ---------------------------------------------------------------------------
# DATABASE SETUP - BARS
# ---------------------------------------------------------------------------


def ensure_custom_bars_table(engine: Engine) -> None:
    """Ensure the meta.bars table exists with proper schema and indexes.

    Creates the bars table that references the symbols table via foreign key.
    Each row represents bar configuration for a specific symbol.

    The table schema includes:
        - id: Auto-incrementing primary key
        - symbol_id: Foreign key to symbols.id (NOT NULL, UNIQUE)
        - bars: JSONB field for bar types and their enabled status (NOT NULL)
        - created_at: Timestamp with timezone (defaults to now())
        - updated_at: Timestamp with timezone (updates on change)

    Foreign Key:
        - symbol_id references symbols(id) ON DELETE CASCADE
        - This ensures automatic cleanup when a symbol is deleted

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.

    Raises:
        SchemaError: If schema creation fails.
        TableError: If table creation fails.
        TypeError: If engine is None.

    Example:
        >>> engine = get_engine()
        >>> ensure_custom_bars_table(engine)
        # Creates meta.bars table if not exists

    Notes:
        - Must be created AFTER symbols table
        - symbol_id is UNIQUE, enforcing one bars config per symbol
        - CASCADE delete removes bars entries when symbol is deleted
        - Exchange/symbol info is retrieved via JOIN with symbols
    """

    if engine is None:
        raise TypeError("Database engine is None; cannot create table")

    logger.debug("Ensuring schema '%s' exists", META_SCHEMA)

    try:
        ensure_schema(engine, META_SCHEMA)
    except Exception as exc:
        logger.error("Failed to ensure schema '%s' exists", META_SCHEMA)
        raise SchemaError(f"Failed to create or verify schema {META_SCHEMA}") from exc

    # Check if table already exists
    try:
        if ensure_table(engine, META_SCHEMA, META_CUSTOM_BARS_TABLE):
            logger.debug("Table %s.%s already exists", META_SCHEMA, META_CUSTOM_BARS_TABLE)
            return
    except Exception as exc:
        logger.error("Failed to check if table %s.%s exists", META_SCHEMA, META_CUSTOM_BARS_TABLE)
        raise TableError(
            f"Failed to verify table existence for {META_SCHEMA}.{META_CUSTOM_BARS_TABLE}"
        ) from exc

    logger.info("Creating table %s.%s with foreign key", META_SCHEMA, META_CUSTOM_BARS_TABLE)

    ddl = f"""
    CREATE TABLE {META_SCHEMA}.{META_CUSTOM_BARS_TABLE} (
        id SERIAL PRIMARY KEY,
        symbol_id INTEGER NOT NULL UNIQUE,
        bars JSONB NOT NULL,
        created_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        CONSTRAINT fk_bars_symbol 
            FOREIGN KEY (symbol_id) 
            REFERENCES {META_SCHEMA}.symbols(id) 
            ON DELETE CASCADE
    );
    
    CREATE INDEX idx_custom_bars_symbol_id 
        ON {META_SCHEMA}.{META_CUSTOM_BARS_TABLE} (symbol_id);
        
    -- Create trigger for updated_at
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = now();
        RETURN NEW;
    END;
    $$ language 'plpgsql';

    CREATE TRIGGER update_bars_updated_at 
        BEFORE UPDATE ON {META_SCHEMA}.{META_CUSTOM_BARS_TABLE}
        FOR EACH ROW 
        EXECUTE FUNCTION update_updated_at_column();
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info(
            "Table %s.%s created successfully with foreign key and updated_at trigger",
            META_SCHEMA,
            META_CUSTOM_BARS_TABLE,
        )
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.%s", META_SCHEMA, META_CUSTOM_BARS_TABLE)
        raise TableError(f"Failed to create table {META_SCHEMA}.{META_CUSTOM_BARS_TABLE}") from exc


# ---------------------------------------------------------------------------
# DATABASE SETUP - TICK META
# ---------------------------------------------------------------------------


def ensure_tick_table(engine: Engine) -> None:
    """Ensure the meta.data_tick table exists with proper schema and indexes.

    Creates the data_tick table for storing tick data collection configurations.
    The table stores settings per (exchange, market_type) combination.

    The table schema includes:
        - id: Auto-incrementing primary key
        - exchange: Text field for exchange name (NOT NULL)
        - market_type: Text field for market type (NOT NULL)
        - enabled: Boolean flag for market status (NOT NULL)
        - symbols: JSONB field for list of symbols to collect (NOT NULL)
        - streams: JSONB field for stream types and their enabled status (NOT NULL)
        - depth: Integer for orderbook depth level (NOT NULL, default 50)
        - created_at: Timestamp with timezone (defaults to now())
        - updated_at: Timestamp with timezone (updates on change)
        - UNIQUE constraint on (exchange, market_type)

    Indexes created:
        - idx_data_tick_enabled: On enabled column
        - idx_data_tick_exchange: On exchange column

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.

    Raises:
        SchemaError: If schema creation fails.
        TableError: If table creation fails.
        TypeError: If engine is None.

    Example:
        >>> engine = get_engine()
        >>> ensure_tick_table(engine)
        # Creates meta.data_tick table if not exists

    Notes:
        - Reuses meta schema from symbols table
        - JSONB columns for flexible configuration storage
        - Updated_at trigger auto-updates on row changes
    """
        
    if engine is None:
        raise TypeError("Database engine is None; cannot create table")

    logger.debug("Ensuring schema '%s' exists", META_SCHEMA)

    try:
        ensure_schema(engine, META_SCHEMA)
    except Exception as exc:
        logger.error("Failed to ensure schema '%s' exists", META_SCHEMA)
        raise SchemaError(f"Failed to create or verify schema {META_SCHEMA}") from exc

    # Check if table already exists
    try:
        if ensure_table(engine, META_SCHEMA, META_TICK_TABLE):
            logger.debug("Table %s.%s already exists", META_SCHEMA, META_TICK_TABLE)
            return
    except Exception as exc:
        logger.error("Failed to check if table %s.%s exists", META_SCHEMA, META_TICK_TABLE)
        raise TableError(
            f"Failed to verify table existence for {META_SCHEMA}.{META_TICK_TABLE}"
        ) from exc

    logger.info("Creating table %s.%s with indexes", META_SCHEMA, META_TICK_TABLE)

    # DDL statement for table and indexes
    ddl = f"""
    CREATE TABLE {META_SCHEMA}.{META_TICK_TABLE} (
        id SERIAL PRIMARY KEY,
        exchange TEXT NOT NULL,
        market_type TEXT NOT NULL,
        enabled BOOLEAN NOT NULL DEFAULT true,
        symbols JSONB NOT NULL,
        streams JSONB NOT NULL,
        depth INTEGER NOT NULL DEFAULT 50,
        created_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        UNIQUE (exchange, market_type)
    );

    CREATE INDEX idx_data_tick_enabled 
        ON {META_SCHEMA}.{META_TICK_TABLE} (enabled);
    
    CREATE INDEX idx_data_tick_exchange 
        ON {META_SCHEMA}.{META_TICK_TABLE} (exchange);
        
    -- Create trigger for updated_at
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = now();
        RETURN NEW;
    END;
    $$ language 'plpgsql';

    CREATE TRIGGER update_data_tick_updated_at 
        BEFORE UPDATE ON {META_SCHEMA}.{META_TICK_TABLE}
        FOR EACH ROW 
        EXECUTE FUNCTION update_updated_at_column();
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info(
            "Table %s.%s created successfully with 2 indexes and updated_at trigger",
            META_SCHEMA,
            META_TICK_TABLE,
        )
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.%s", META_SCHEMA, META_TICK_TABLE)
        raise TableError(f"Failed to create table {META_SCHEMA}.{META_TICK_TABLE}") from exc


# ---------------------------------------------------------------------------
# DATABASE SETUP - BLOCKCHAIN
# ---------------------------------------------------------------------------

def ensure_blockchain_table(engine: Engine) -> None:
    """Ensure the meta.blockchain table exists with proper schema and indexes.

    Creates the blockchain table for storing blockchain chart metadata
    and data collection settings.

    The table schema includes:
        - id: Auto-incrementing primary key
        - chart_name: Text field for chart identifier (NOT NULL)
        - category: Text field for chart category (NOT NULL)
        - enabled: Boolean flag for data collection status (NOT NULL)
        - start_date: Text field for initial fetch date (NOT NULL)
        - end_date: Text field for final fetch date (NOT NULL)
        - created_at: Timestamp with timezone (defaults to now())
        - updated_at: Timestamp with timezone (updates on change)
        - UNIQUE constraint on chart_name

    Indexes created:
        - idx_blockchain_enabled: On enabled column
        - idx_blockchain_category: On category column

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.

    Raises:
        SchemaError: If schema creation fails.
        TableError: If table creation fails.
        TypeError: If engine is None.

    Example:
        >>> engine = get_engine()
        >>> ensure_blockchain_table(engine)
        # Creates meta.blockchain table if not exists

    Notes:
        - Reuses meta schema from symbols table
        - Updated_at trigger auto-updates on row changes
    """

    if engine is None:
        raise TypeError("Database engine is None")

    logger.debug("Ensuring schema '%s' exists", META_SCHEMA)

    try:
        ensure_schema(engine, META_SCHEMA)
    except Exception as exc:
        logger.exception("Failed to ensure schema '%s' exists", META_SCHEMA)
        raise

    # Check if table already exists
    try:
        if ensure_table(engine, META_SCHEMA, META_BLOCKCHAIN_TABLE):
            logger.debug("Table %s.%s already exists", META_SCHEMA, META_BLOCKCHAIN_TABLE)
            return
    except Exception as exc:
        logger.exception(
            "Error checking if table %s.%s exists",
            META_SCHEMA,
            META_BLOCKCHAIN_TABLE,
        )
        raise

    logger.info("Creating table %s.%s with indexes", META_SCHEMA, META_BLOCKCHAIN_TABLE)

    ddl = f"""
    CREATE TABLE {META_SCHEMA}.{META_BLOCKCHAIN_TABLE} (
        id SERIAL PRIMARY KEY,
        chart_name TEXT NOT NULL UNIQUE,
        category TEXT NOT NULL,
        enabled BOOLEAN NOT NULL DEFAULT true,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        created_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT now()
    );
    
    CREATE INDEX idx_blockchain_enabled 
        ON {META_SCHEMA}.{META_BLOCKCHAIN_TABLE} (enabled);
    
    CREATE INDEX idx_blockchain_category 
        ON {META_SCHEMA}.{META_BLOCKCHAIN_TABLE} (category);
        
    -- Create trigger for updated_at
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = now();
        RETURN NEW;
    END;
    $$ language 'plpgsql';

    CREATE TRIGGER update_blockchain_updated_at 
        BEFORE UPDATE ON {META_SCHEMA}.{META_BLOCKCHAIN_TABLE}
        FOR EACH ROW 
        EXECUTE FUNCTION update_updated_at_column();
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info(
            "Table %s.%s created successfully with 2 indexes and updated_at trigger",
            META_SCHEMA,
            META_BLOCKCHAIN_TABLE,
        )
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.%s", META_SCHEMA, META_BLOCKCHAIN_TABLE)
        raise TableError(f"Failed to create table {META_SCHEMA}.{META_BLOCKCHAIN_TABLE}") from exc


# ---------------------------------------------------------------------------
# DATABASE SETUP - MACRO
# ---------------------------------------------------------------------------

def ensure_macro_table(engine: Engine) -> None:
    """Ensure the meta.macro table exists with proper schema and indexes.

    Creates the macro table for storing macro economic indicator metadata
    and FRED API integration settings.

    The table schema includes:
        - id: Auto-incrementing primary key
        - indicator_key: Text field for indicator identifier (NOT NULL)
        - fred_series: Text field for FRED series ID (NOT NULL)
        - frequency: Text field for data frequency (NOT NULL)
        - enabled: Boolean flag for data collection status (NOT NULL)
        - created_at: Timestamp with timezone (defaults to now())
        - updated_at: Timestamp with timezone (updates on change)
        - UNIQUE constraint on indicator_key

    Indexes created:
        - idx_macro_enabled: On enabled column
        - idx_macro_frequency: On frequency column

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.

    Raises:
        SchemaError: If schema creation fails.
        TableError: If table creation fails.
        TypeError: If engine is None.

    Example:
        >>> engine = get_engine()
        >>> ensure_macro_table(engine)
        # Creates meta.macro table if not exists

    Notes:
        - Reuses meta schema from symbols table
        - Updated_at trigger auto-updates on row changes
        - fred_series stores FRED API series IDs (e.g., 'UNRATE', 'GDP')
    """
        
    if engine is None:
        raise TypeError("Database engine is None")

    logger.debug("Ensuring schema '%s' exists", META_SCHEMA)

    try:
        ensure_schema(engine, META_SCHEMA)
    except Exception as exc:
        logger.exception("Failed to ensure schema '%s' exists", META_SCHEMA)
        raise

    # Check if table already exists
    try:
        if ensure_table(engine, META_SCHEMA, META_MACRO_TABLE):
            logger.debug("Table %s.%s already exists", META_SCHEMA, META_MACRO_TABLE)
            return
    except Exception as exc:
        logger.exception(
            "Error checking if table %s.%s exists",
            META_SCHEMA,
            META_MACRO_TABLE,
        )
        raise

    logger.info("Creating table %s.%s with indexes", META_SCHEMA, META_MACRO_TABLE)

    ddl = f"""
    CREATE TABLE {META_SCHEMA}.{META_MACRO_TABLE} (
        id SERIAL PRIMARY KEY,
        indicator_key TEXT NOT NULL UNIQUE,
        fred_series TEXT NOT NULL,
        frequency TEXT NOT NULL,
        enabled BOOLEAN NOT NULL DEFAULT true,
        created_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT now()
    );
    
    CREATE INDEX idx_macro_enabled 
        ON {META_SCHEMA}.{META_MACRO_TABLE} (enabled);
    
    CREATE INDEX idx_macro_frequency 
        ON {META_SCHEMA}.{META_MACRO_TABLE} (frequency);
        
    -- Create trigger for updated_at
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = now();
        RETURN NEW;
    END;
    $$ language 'plpgsql';

    CREATE TRIGGER update_macro_updated_at 
        BEFORE UPDATE ON {META_SCHEMA}.{META_MACRO_TABLE}
        FOR EACH ROW 
        EXECUTE FUNCTION update_updated_at_column();
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info(
            "Table %s.%s created successfully with 2 indexes and updated_at trigger",
            META_SCHEMA,
            META_MACRO_TABLE,
        )
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.%s", META_SCHEMA, META_MACRO_TABLE)
        raise TableError(f"Failed to create table {META_SCHEMA}.{META_MACRO_TABLE}") from exc


