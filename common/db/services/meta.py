"""
Database service layer for meta-related operations.

This module provides database access functions for:
- meta.symbols (OHLCV) table operations
- meta.data_tick table operations
- meta.bars table operations
- meta.blockchain table operations

All functions handle parameterized queries, error handling, logging,
and transaction management.
"""

import json
from typing import Dict, List, Any
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from bitpredict.common.logging import get_logger
from bitpredict.common.db.utils import read_df
from bitpredict.common.db.config import get_engine
from bitpredict.data.meta.utils import build_custom_bars_records, build_blockchain_records, build_tick_records, build_macro_records, build_time_bars_records, build_symbols_records
from bitpredict.common.constants import *
from bitpredict.common.db.models import ensure_custom_bars_table,ensure_blockchain_table,ensure_macro_table,ensure_time_bars_table, ensure_tick_table, ensure_symbols_table
from bitpredict.common.db.exceptions import DataSaveError, SchemaError, TableError
from bitpredict.common.constants import *


logger = get_logger(__name__)


# ============================================================================
# SYMBOLS TABLE OPERATIONS
# ============================================================================


def insert_symbols(config: Dict[str, Any]) -> None:
    """Insert or update symbols_master records from configuration.

    This function orchestrates the complete workflow for loading symbols into
    the meta.symbols table:
    1. Ensure the symbols_master table exists
    2. Parse and validate the symbols.yaml configuration
    3. Build flat records for database insertion
    4. Perform upsert operation (INSERT ... ON CONFLICT DO UPDATE)

    The symbols_master table is the master registry for all exchange-symbol
    pairs. Other tables (ohlcv, bars) reference this table via foreign keys.

    Args:
        config: Parsed symbols.yaml configuration dictionary containing
            exchange and symbol definitions.

    Raises:
        ValueError: If configuration is invalid or missing required fields.
        SQLAlchemyError: If database operations fail.
        IntegrityError: If unique constraint violations occur.

    Example:
        >>> from bitpredict.common.utils.file_system import read_yaml_config
        >>> config = read_yaml_config('symbols.yaml')
        >>> insert_symbols(config)
        INFO: Inserted/updated 25 records into meta.symbols

    Notes:
        - Uses UPSERT to handle both new and existing symbols
        - On conflict, updates enabled status and updated_at timestamp
        - Disabled symbols are inserted but marked with enabled=False
        - This must be run BEFORE insert_ohlcv and insert_custom_bars
        - All operations are performed in a single transaction
    """
    logger.info("Starting symbols_master insertion process")

    # Get database engine
    engine = get_engine()

    # Ensure symbols_master table exists
    logger.debug("Ensuring meta.symbols table exists")
    ensure_symbols_table(engine)

    # Build records from configuration
    logger.debug("Building records from symbols configuration")
    records = build_symbols_records(config)

    if not records:
        logger.warning("No records to insert into meta.symbols")
        return

    # Prepare UPSERT statement
    # On conflict (exchange, symbol), update enabled and updated_at
    upsert_sql = """
        INSERT INTO meta.symbols (exchange, symbol, enabled)
        VALUES (:exchange, :symbol, :enabled)
        ON CONFLICT (exchange, symbol)
        DO UPDATE SET
            enabled = EXCLUDED.enabled,
            updated_at = now()
    """

    try:
        with engine.begin() as conn:
            # Execute batch insert with UPSERT
            result = conn.execute(text(upsert_sql), records)
            
            # Log success
            logger.info(
                "Inserted/updated %d records into meta.symbols",
                len(records)
            )

    except SQLAlchemyError as exc:
        logger.exception("Database error during symbols_master insertion")
        raise

    logger.info("Symbols_master insertion completed successfully")


def get_symbols(exchange: str = None) -> List[Dict]:
    """
    Fetch all enabled symbols from symbols_master table.
    
    This is a new helper function that provides direct access to the
    symbols_master table without needing OHLCV or bars configuration.
    
    Args:
        exchange (str, optional): Filter by exchange. If None, returns all exchanges.
    
    Returns:
        List[Dict]: List of dictionaries containing:
            - id (int): Symbol ID (for foreign key references)
            - exchange (str): Exchange name
            - symbol (str): Symbol name
            - enabled (bool): Whether symbol is enabled
    
    Example:
        >>> symbols = get_enabled_symbols('binance')
        >>> symbols[0]
        {
            'id': 1,
            'exchange': 'binance',
            'symbol': 'btc',
            'enabled': True
        }
    """
    
    engine = get_engine()
    
    try:
        if exchange:
            query = """
                SELECT id, exchange, symbol, enabled
                FROM meta.symbols
                WHERE exchange = :exchange
                AND enabled = true
                ORDER BY exchange, symbol
            """
            df = read_df(engine=engine, query=query, params={"exchange": exchange.lower()})
        else:
            query = """
                SELECT id, exchange, symbol, enabled
                FROM meta.symbols
                WHERE enabled = true
                ORDER BY exchange, symbol
            """
            df = read_df(engine=engine, query=query)
        
        if df.empty:
            logger.warning("No enabled symbols found" + (f" for exchange: {exchange}" if exchange else ""))
            return []
        
        return df.to_dict('records')
        
    except Exception as exc:
        logger.error("Error fetching enabled symbols: %s", exc)
        return []


# ============================================================================
# OHLCV TABLE OPERATIONS
# ============================================================================


def insert_time_bars(ohlcv_config: dict) -> None:
    """Load OHLCV configuration and upsert records into the meta.ohlcv table.

    REFACTORED: This function now works with the symbols_master foreign key pattern.
    It looks up symbol_id from symbols_master for each (exchange, symbol) pair
    and only processes symbols that exist and are enabled in symbols_master.

    This function orchestrates:
    1. Validates OHLCV configuration
    2. Transforms configuration into database records
    3. Ensures database schema and table exist
    4. Looks up symbol_id from symbols_master for each symbol
    5. Upserts records using symbol_id foreign key
    6. Updates the updated_at timestamp for modified records

    Args:
        ohlcv_config: Dictionary containing OHLCV configuration with structure:
            {
                'exchanges': {
                    'exchange_name': {
                        'symbols': {
                            'symbol_name': {
                                'timeframes': dict
                            }
                        }
                    }
                }
            }

    Raises:
        TypeError: If ohlcv_config is None or not a dictionary.
        ValueError: If configuration is invalid or missing required fields.
        SchemaError: If database schema creation fails.
        TableError: If table creation or upsert fails.
        DataSaveError: If database operations fail.

    Example:
        >>> ohlcv_config = {
        ...     'exchanges': {
        ...         'binance': {
        ...             'symbols': {
        ...                 'btc': {
        ...                     'timeframes': {'1m': True, '1h': True}
        ...                 }
        ...             }
        ...         }
        ...     }
        ... }
        >>> insert_time_bars(ohlcv_config)
        # INFO: Successfully upserted 1 records into meta.time_bars

    Notes:
        - Requires symbols_master table to be populated first
        - Only processes symbols that exist and are enabled in symbols_master
        - Skips symbols not found in symbols_master with warning
        - Uses UPSERT pattern for intelligent insert/update handling
        - All database operations are transactional
    """

    engine = get_engine()
        
    # Validate OHLCV configuration is not None
    if ohlcv_config is None:
        raise ValueError("OHLCV configuration is None; expected a dictionary")

    # Transform configuration into database records
    logger.debug("Building database records from OHLCV configuration")
    try:
        records = build_time_bars_records(ohlcv_config)
    except (ValueError, TypeError) as exc:
        logger.error("Failed to build records from configuration: %s", exc)
        raise

    # Early exit if no records to process
    if not records:
        logger.warning("No records generated from OHLCV configuration; nothing to process")
        return

    # Ensure database schema and table are ready
    logger.debug("Ensuring database schema and table exist")
    try:
        ensure_time_bars_table(engine)
    except (SchemaError, TableError) as exc:
        logger.error("Failed to prepare database: %s", exc)
        raise

    # Resolve symbol_id for each record by looking up symbols_master
    logger.debug("Resolving symbol_ids from symbols_master table")
    records_with_symbol_id = []
    
    try:
        with engine.begin() as conn:
            for record in records:
                exchange = record["exchange"]
                symbol = record["symbol"]
                
                # Lookup symbol_id from symbols_master
                result = conn.execute(
                    text("""
                        SELECT id, enabled 
                        FROM meta.symbols 
                        WHERE exchange = :exchange AND symbol = :symbol
                    """),
                    {"exchange": exchange, "symbol": symbol}
                ).fetchone()
                
                if result is None:
                    logger.warning(
                        "Symbol %s:%s not found in symbols_master table; skipping. "
                        "Ensure symbols.yaml is loaded first.",
                        exchange,
                        symbol,
                    )
                    continue
                
                symbol_id, enabled = result
                
                if not enabled:
                    logger.debug(
                        "Symbol %s:%s is disabled in symbols_master; skipping OHLCV config",
                        exchange,
                        symbol,
                    )
                    continue
                
                # Add symbol_id to record and remove exchange/symbol
                records_with_symbol_id.append({
                    "symbol_id": symbol_id,
                    "timeframes": record["timeframes"],
                })
                
    except SQLAlchemyError as exc:
        logger.exception("Failed to lookup symbol_ids from symbols_master")
        raise DataSaveError("Failed to resolve symbol_ids from symbols_master") from exc

    # Early exit if no valid symbols found
    if not records_with_symbol_id:
        logger.warning(
            "No valid symbols found in symbols_master; nothing to insert. "
            "Ensure symbols.yaml is loaded and symbols are enabled."
        )
        return

    logger.info(
        "Resolved %d symbols from symbols_master (%d skipped)",
        len(records_with_symbol_id),
        len(records) - len(records_with_symbol_id),
    )

    # Perform upsert operation
    logger.debug("Upserting records into %s.%s", META_SCHEMA, META_TIME_BARS_TABLE)
    try:
        with engine.begin() as conn:
            # Build upsert query using PostgreSQL ON CONFLICT clause
            # Conflict on symbol_id (UNIQUE constraint)
            upsert_query = f"""
                INSERT INTO {META_SCHEMA}.{META_TIME_BARS_TABLE} 
                (symbol_id, timeframes)
                VALUES (:symbol_id, :timeframes)
                ON CONFLICT (symbol_id) 
                DO UPDATE SET
                    timeframes = EXCLUDED.timeframes,
                    updated_at = now()
                WHERE {META_SCHEMA}.{META_TIME_BARS_TABLE}.timeframes != EXCLUDED.timeframes
            """
            
            # Execute upsert for each record
            for record in records_with_symbol_id:
                conn.execute(
                    text(upsert_query),
                    {
                        "symbol_id": record["symbol_id"],
                        "timeframes": record["timeframes"],
                    }
                )
        
        logger.debug("Upsert operation successful")
        

    except SQLAlchemyError as exc:
        logger.exception("Failed to upsert records into %s.%s", META_SCHEMA, META_TIME_BARS_TABLE)
        raise TableError(
            f"Failed to upsert records into {META_SCHEMA}.{META_TIME_BARS_TABLE}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during upsert operation: %s", exc)
        raise DataSaveError(
            f"Failed to upsert records into {META_SCHEMA}.{META_TIME_BARS_TABLE}"
        ) from exc

    logger.info(
        "Successfully upserted %d records into %s.%s",
        len(records_with_symbol_id),
        META_SCHEMA,
        META_TIME_BARS_TABLE,
    )

def get_time_bars_meta(exchange: str) -> List[Dict]:
    """
    Load symbols metadata for a given exchange from the meta.time_bars table
    by joining with symbols_master. Only returns enabled symbols.

    Args:
        exchange (str): Exchange name, e.g., "binance"

    Returns:
        list[dict]: Each dict contains:
            - symbol (str)
            - timeframes (dict)
            - allowed (bool)
    """
    engine = get_engine()

    try:
        query = f"""
            SELECT 
                sm.exchange,
                sm.symbol,
                sm.enabled,
                o.timeframes
            FROM {META_SCHEMA}.{META_TIME_BARS_TABLE} o
            JOIN {META_SCHEMA}.symbols sm ON sm.id = o.symbol_id
            WHERE sm.exchange = :exchange
            AND sm.enabled = true
        """

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"exchange": exchange.lower()})

    except Exception as exc:
        logger.error("Failed to fetch OHLCV metadata for exchange %s: %s", exchange, exc)
        return []

    if df.empty:
        logger.warning("No symbols found for exchange: %s", exchange)
        return []

    symbols_meta: List[Dict] = []

    for _, row in df.iterrows():
        timeframe = row.get("timeframes") or {}
        symbols_meta.append({
            "symbol": row["symbol"],
            "timeframe": timeframe,
            "allowed": bool(row["enabled"]),
        })

    return symbols_meta


# ============================================================================
# TICK TABLE OPERATIONS
# ============================================================================


def insert_tick(data_tick_config: dict) -> None:
    """Load data_tick configuration and upsert records into the meta.data_tick table.

    This function orchestrates the tick data collection configuration loading:
    1. Validates data_tick configuration
    2. Transforms configuration into database records
    3. Ensures database schema and table exist
    4. Upserts (insert or update) records in the database
    5. Updates the updated_at timestamp for modified records
    6. Preserves historical data while updating metadata

    The function uses PostgreSQL UPSERT (INSERT ... ON CONFLICT ... DO UPDATE)
    to intelligently handle both new and existing markets, ensuring records
    are created or updated without data loss.

    Args:
        data_tick_config: Dictionary containing data_tick configuration with structure:
            {
                'exchanges': {
                    'exchange_name': {
                        'market_type': {
                            'enabled': bool,
                            'symbols': list,
                            'streams': list,
                            'depth': int
                        }
                    }
                }
            }

    Raises:
        TypeError: If data_tick_config is None or not a dictionary.
        ValueError: If configuration is invalid or missing required fields.
        SchemaError: If database schema creation fails.
        TableError: If table creation or upsert fails.
        DataSaveError: If database operations fail.

    Example:
        >>> data_tick_config = {
        ...     'exchanges': {
        ...         'binance': {
        ...             'spot': {
        ...                 'enabled': True,
        ...                 'symbols': ['BTCUSDT'],
        ...                 'streams': ['trade'],
        ...                 'depth': None
        ...             }
        ...         }
        ...     }
        ... }
        >>> insert_tick(data_tick_config)
        # INFO: Successfully upserted 1 records into meta.data_tick

    Notes:
        - Uses UPSERT pattern for intelligent insert/update handling
        - Only updates changed records, preserves created_at timestamp
        - Updates updated_at timestamp for all touched records
        - All database operations are transactional
        - Detailed logging at each step for troubleshooting
        - Validates inputs before attempting any operations
    """

        
    engine = get_engine()
    
    # Validate data_tick configuration is not None
    if data_tick_config is None:
        raise ValueError("data_tick configuration is None; expected a dictionary")

    # Transform configuration into database records
    logger.debug("Building database records from data_tick configuration")
    try:
        records = build_tick_records(data_tick_config)
    except (ValueError, TypeError) as exc:
        logger.error("Failed to build records from data_tick configuration: %s", exc)
        raise

    # Early exit if no records to process
    if not records:
        logger.warning("No records generated from data_tick.yaml; nothing to process")
        return

    # Ensure database schema and table are ready
    logger.debug("Ensuring database schema and data_tick table exist")
    try:
        ensure_tick_table(engine)
    except (SchemaError, TableError) as exc:
        logger.error("Failed to prepare database: %s", exc)
        raise

    # Convert to DataFrame for bulk operations
    try:
        df = pd.DataFrame.from_records(records)
    except Exception as exc:
        logger.error("Failed to create DataFrame from records: %s", exc)
        raise ValueError("Failed to convert records to DataFrame") from exc

    # Validate DataFrame is not empty
    if df.empty:
        logger.warning("DataFrame is empty after conversion; nothing to process")
        return

    # Log statistics before upsert
    enabled_count = int(df["enabled"].sum())
    disabled_count = int((~df["enabled"]).sum())
    logger.info(
        "Preparing to upsert %d data_tick records (%d enabled, %d disabled)",
        len(df),
        enabled_count,
        disabled_count,
    )

    # Perform upsert operation (insert or update based on conflict)
    logger.debug("Upserting records into %s.%s", META_SCHEMA, META_TICK_TABLE)
    try:
        with engine.begin() as conn:
            # Build upsert query using PostgreSQL ON CONFLICT clause
            # Conflict on (exchange, market_type) - the natural unique key
            upsert_query = f"""
                INSERT INTO {META_SCHEMA}.{META_TICK_TABLE}
                (exchange, market_type, enabled, symbols, streams, depth)
                VALUES (:exchange, :market_type, :enabled, :symbols, :streams, :depth)
                ON CONFLICT (exchange, market_type)
                DO UPDATE SET
                    enabled = EXCLUDED.enabled,
                    symbols = EXCLUDED.symbols,
                    streams = EXCLUDED.streams,
                    depth = EXCLUDED.depth,
                    updated_at = CURRENT_TIMESTAMP
                WHERE {META_SCHEMA}.{META_TICK_TABLE}.enabled != EXCLUDED.enabled
                   OR {META_SCHEMA}.{META_TICK_TABLE}.symbols != EXCLUDED.symbols
                   OR {META_SCHEMA}.{META_TICK_TABLE}.streams != EXCLUDED.streams
                   OR {META_SCHEMA}.{META_TICK_TABLE}.depth != EXCLUDED.depth
            """

            # Execute upsert for each record
            for record in records:
                conn.execute(
                    text(upsert_query),
                    {
                        "exchange": record["exchange"],
                        "market_type": record["market_type"],
                        "enabled": record["enabled"],
                        "symbols": record["symbols"],
                        "streams": record["streams"],
                        "depth": record["depth"],
                    }
                )

        logger.debug("Upsert operation successful")

    except SQLAlchemyError as exc:
        logger.exception("Failed to upsert records into %s.%s", META_SCHEMA, META_TICK_TABLE)
        raise TableError(
            f"Failed to upsert records into {META_SCHEMA}.{META_TICK_TABLE}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during upsert operation: %s", exc)
        raise DataSaveError(
            f"Failed to upsert records into {META_SCHEMA}.{META_TICK_TABLE}"
        ) from exc

    logger.info(
        "Successfully upserted %d records into %s.%s",
        len(df),
        META_SCHEMA,
        META_TICK_TABLE,
    )


def get_tick_meta(
    schema: str = META_SCHEMA,
    table: str = META_TICK_TABLE,
    exchange: str = "binance", 
) -> List[Dict[str, Any]]:
    """Retrieve all enabled tick data configurations.

    Queries the meta.data_tick table for all enabled configurations.
    Useful for starting multiple data collectors at once.

    Args:
        engine: SQLAlchemy database engine.
        schema: Database schema name. Defaults to "meta".
        table: Table name. Defaults to "data_tick".

    Returns:
        List of configuration dictionaries for all enabled markets.
        Empty list if no enabled configurations found.

    Raises:
        TypeError: If engine is None.
        ValueError: If parameters are invalid.
        RuntimeError: If query fails.

    Example:
        >>> engine = get_engine()
        >>> configs = get_tick_meta(engine)
        >>> for config in configs:
        ...     print(f"Starting {config['exchange']}:{config['market_type']}")
        ...     # Start data collector with config

    Notes:
        - Only returns enabled=True configurations
        - Deserializes all JSON fields
        - Orders results by exchange, then market_type
        - Returns empty list if no enabled configs found
    """

    if not isinstance(schema, str) or not schema.strip():
        raise ValueError("schema must be a non-empty string")

    if not isinstance(table, str) or not table.strip():
        raise ValueError("table must be a non-empty string")


        
    engine = get_engine()

    query = f"""
        SELECT 
            exchange,
            market_type,
            enabled,
            symbols,
            streams,
            depth
        FROM {schema}.{table}
        WHERE enabled = true
        {'AND exchange = :exchange' if exchange else ''}
        ORDER BY exchange, market_type
    """

    try:
        with engine.begin() as conn:
            params = {"exchange": exchange} if exchange else {}
            result = conn.execute(text(query), params)
            rows = result.fetchall()

        if not rows:
            return []

        configs = []
        for row in rows:
            config = {
                "exchange": row[0],
                "market_type": row[1],
                "enabled": row[2],
                "symbols": json.loads(row[3])["symbols"] if isinstance(row[3], str) else row[3].get("symbols", []),
                "streams": json.loads(row[4]) if isinstance(row[4], str) else row[4],
                "depth": row[5],
            }
            configs.append(config)



        return configs

    except SQLAlchemyError as exc:
        raise RuntimeError("Failed to retrieve enabled data_tick configurations") from exc
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RuntimeError("Failed to parse data_tick configurations") from exc


# ============================================================================
# BAR TABLE OPERATIONS
# ============================================================================


def insert_custom_bars(bars_config: dict) -> None:
    """Load bars configuration and upsert records into the meta.bars table.

    REFACTORED: This function now works with the symbols_master foreign key pattern.
    It looks up symbol_id from symbols_master for each (exchange, symbol) pair
    and only processes symbols that exist and are enabled in symbols_master.

    This function orchestrates:
    1. Validates bars configuration
    2. Transforms configuration into database records
    3. Ensures database schema and table exist
    4. Looks up symbol_id from symbols_master for each symbol
    5. Upserts records using symbol_id foreign key
    6. Updates the updated_at timestamp for modified records

    Args:
        bars_config: Dictionary containing bars configuration with structure:
            {
                'exchanges': {
                    'exchange_name': {
                        'symbols': {
                            'symbol_name': {
                                'bars': {
                                    'bar_type': bool,
                                }
                            }
                        }
                    }
                }
            }

    Raises:
        TypeError: If bars_config is None or not a dictionary.
        ValueError: If configuration is invalid or missing required fields.
        SchemaError: If database schema creation fails.
        TableError: If table creation or upsert fails.
        DataSaveError: If database operations fail.

    Example:
        >>> bars_config = {
        ...     'exchanges': {
        ...         'binance': {
        ...             'symbols': {
        ...                 'btc': {
        ...                     'bars': {'dollar': True, 'volume': True}
        ...                 }
        ...             }
        ...         }
        ...     }
        ... }
        >>> insert_custom_bars(bars_config)
        # INFO: Successfully upserted 1 records into meta.bars

    Notes:
        - Requires symbols_master table to be populated first
        - Only processes symbols that exist and are enabled in symbols_master
        - Skips symbols not found in symbols_master with warning
        - Uses UPSERT pattern for intelligent insert/update handling
        - All database operations are transactional
    """

    engine = get_engine()
    
    # Validate bars configuration is not None
    if bars_config is None:
        raise ValueError("bars configuration is None; expected a dictionary")

    # Transform configuration into database records
    logger.debug("Building database records from bars configuration")
    try:
        records = build_custom_bars_records(bars_config)
    except (ValueError, TypeError) as exc:
        logger.error("Failed to build records from bars configuration: %s", exc)
        raise

    # Early exit if no records to process
    if not records:
        logger.warning("No records generated from bars configuration; nothing to process")
        return

    # Ensure database schema and table are ready
    logger.debug("Ensuring database schema and bars table exist")
    try:
        ensure_custom_bars_table(engine)
    except (SchemaError, TableError) as exc:
        logger.error("Failed to prepare database: %s", exc)
        raise

    # Resolve symbol_id for each record by looking up symbols_master
    logger.debug("Resolving symbol_ids from symbols_master table")
    records_with_symbol_id = []
    
    try:
        with engine.begin() as conn:
            for record in records:
                exchange = record["exchange"]
                symbol = record["symbol"]
                
                # Lookup symbol_id from symbols_master
                result = conn.execute(
                    text("""
                        SELECT id, enabled 
                        FROM meta.symbols 
                        WHERE exchange = :exchange AND symbol = :symbol
                    """),
                    {"exchange": exchange, "symbol": symbol}
                ).fetchone()
                
                if result is None:
                    logger.warning(
                        "Symbol %s:%s not found in symbols_master table; skipping. "
                        "Ensure symbols.yaml is loaded first.",
                        exchange,
                        symbol,
                    )
                    continue
                
                symbol_id, enabled = result
                
                if not enabled:
                    logger.debug(
                        "Symbol %s:%s is disabled in symbols_master; skipping bars config",
                        exchange,
                        symbol,
                    )
                    continue
                
                # Add symbol_id to record and remove exchange/symbol
                records_with_symbol_id.append({
                    "symbol_id": symbol_id,
                    "bars": record["bars"],
                })
                
    except SQLAlchemyError as exc:
        logger.exception("Failed to lookup symbol_ids from symbols_master")
        raise DataSaveError("Failed to resolve symbol_ids from symbols_master") from exc

    # Early exit if no valid symbols found
    if not records_with_symbol_id:
        logger.warning(
            "No valid symbols found in symbols_master; nothing to insert. "
            "Ensure symbols.yaml is loaded and symbols are enabled."
        )
        return

    logger.info(
        "Resolved %d symbols from symbols_master (%d skipped)",
        len(records_with_symbol_id),
        len(records) - len(records_with_symbol_id),
    )

    # Perform upsert operation
    logger.debug("Upserting records into %s.%s", META_SCHEMA, META_CUSTOM_BARS_TABLE)
    try:
        with engine.begin() as conn:
            # Build upsert query using PostgreSQL ON CONFLICT clause
            # Conflict on symbol_id (UNIQUE constraint)
            upsert_query = f"""
                INSERT INTO {META_SCHEMA}.{META_CUSTOM_BARS_TABLE}
                (symbol_id, bars)
                VALUES (:symbol_id, :bars)
                ON CONFLICT (symbol_id)
                DO UPDATE SET
                    bars = EXCLUDED.bars,
                    updated_at = now()
                WHERE {META_SCHEMA}.{META_CUSTOM_BARS_TABLE}.bars != EXCLUDED.bars
            """

            # Execute upsert for each record
            for record in records_with_symbol_id:
                conn.execute(
                    text(upsert_query),
                    {
                        "symbol_id": record["symbol_id"],
                        "bars": record["bars"],
                    }
                )

        logger.debug("Upsert operation successful")


    except SQLAlchemyError as exc:
        logger.exception("Failed to upsert records into %s.%s", META_SCHEMA, META_CUSTOM_BARS_TABLE)
        raise TableError(
            f"Failed to upsert records into {META_SCHEMA}.{META_CUSTOM_BARS_TABLE}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during upsert operation: %s", exc)
        raise DataSaveError(
            f"Failed to upsert records into {META_SCHEMA}.{META_CUSTOM_BARS_TABLE}"
        ) from exc

    logger.info(
        "Successfully upserted %d records into %s.%s",
        len(records_with_symbol_id),
        META_SCHEMA,
        META_CUSTOM_BARS_TABLE,
    )


def get_custom_bar_meta(schema: str = META_SCHEMA, table_name: str = META_CUSTOM_BARS_TABLE) -> Dict[str, Dict[str, List[str]]]:
    """
    Fetch configurations from the 'meta.custom_bars' table using direct SQL read.
    
    Returns all bar types configured for each exchange-symbol pair.
    
    Args:
        schema (str): Database schema name (default: META_SCHEMA)
        table_name (str): Table name (default: META_CUSTOM_BARS_TABLE)
    
    Returns:
        Dict mapping exchange -> symbol -> list of bar types
        
    Example:
        >>> config = get_custom_bar_meta()
        >>> config['binance']['btc']
        ['dollar', 'volume', 'volatility']
    """
    try:
        engine = get_engine()
        
        query = f"""
            SELECT 
                sm.exchange,
                sm.symbol,
                sm.enabled,
                b.bars
            FROM {schema}.{table_name} b
            JOIN {schema}.symbols sm ON sm.id = b.symbol_id
            WHERE sm.enabled = true
        """

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        if df.empty:
            return {}

        config_map: Dict[str, Dict[str, List[str]]] = {}

        for _, row in df.iterrows():
            exch = row['exchange']
            sym = row['symbol']

            # Extract bar types from 'bars' JSON/dict
            bars_dict = row['bars'] if isinstance(row['bars'], dict) else {}
            # bar_types = list(bars_dict.keys())
            # Only include bar types where the value is True
            bar_types = [bar_type for bar_type, enabled in bars_dict.items() if enabled]

            if bar_types:
                if exch not in config_map:
                    config_map[exch] = {}
                config_map[exch][sym] = bar_types

        return config_map

    except Exception as exc:
        logger.error("Error fetching bar config: %s", exc)
        return {}


# ============================================================================
# BLOCKCHAIN TABLE OPERATIONS
# ============================================================================


def insert_blockchain(blockchain_config: dict) -> None:
    """Load blockchain configuration and upsert records into the meta.blockchain table.

    This function orchestrates the blockchain data configuration loading:
    1. Validates blockchain configuration
    2. Transforms configuration into database records
    3. Ensures database schema and table exist
    4. Upserts (insert or update) records in the database
    5. Updates the updated_at timestamp for modified records
    6. Preserves historical data while updating metadata

    The function uses PostgreSQL UPSERT (INSERT ... ON CONFLICT ... DO UPDATE)
    to intelligently handle both new and existing blockchain charts, ensuring
    records are created or updated without data loss.

    Args:
        blockchain_config: Dictionary containing blockchain chart configuration with structure:
            {
                'blockchain': {
                    'category_name': [
                        {
                            'chart_name': str,
                            'enabled': bool,
                            'start_date': date,
                            'end_date': date
                        }
                    ]
                }
            }

    Raises:
        TypeError: If blockchain_config is None or not a dictionary.
        ValueError: If configuration is invalid or missing required fields.
        SchemaError: If database schema creation fails.
        TableError: If table creation or upsert fails.
        DataSaveError: If database operations fail.

    Example:
        >>> blockchain_config = {
        ...     'blockchain': {
        ...         'addresses': [
        ...             {
        ...                 'chart_name': 'bitcoin_addresses',
        ...                 'enabled': True,
        ...                 'start_date': '2015-01-01',
        ...                 'end_date': None
        ...             }
        ...         ]
        ...     }
        ... }
        >>> insert_blockchain(blockchain_config)
        # INFO: Successfully upserted 1 records into meta.blockchain

    Notes:
        - Uses UPSERT pattern for intelligent insert/update handling
        - Only updates changed records, preserves created_at timestamp
        - Updates updated_at timestamp for all touched records
        - All database operations are transactional
        - Detailed logging at each step for troubleshooting
        - Validates inputs before attempting any operations
    """
        
    engine = get_engine()
    
    # Validate blockchain configuration is not None
    if blockchain_config is None:
        raise ValueError("blockchain configuration is None; expected a dictionary")

    # Transform configuration into database records
    logger.debug("Building database records from blockchain configuration")
    try:
        records = build_blockchain_records(blockchain_config)
    except (ValueError, TypeError) as exc:
        logger.error("Failed to build records from blockchain configuration: %s", exc)
        raise

    # Early exit if no records to process
    if not records:
        logger.warning("No records generated from blockchain configuration; nothing to process")
        return

    # Ensure database schema and table are ready
    logger.debug("Ensuring database schema and blockchain table exist")
    try:
        ensure_blockchain_table(engine)
    except (SchemaError, TableError) as exc:
        logger.error("Failed to prepare database: %s", exc)
        raise

    # Convert to DataFrame for bulk operations
    try:
        df = pd.DataFrame.from_records(records)
    except Exception as exc:
        logger.error("Failed to create DataFrame from records: %s", exc)
        raise ValueError("Failed to convert records to DataFrame") from exc

    if df.empty:
        logger.warning("DataFrame is empty after conversion; nothing to insert")
        return

    logger.info(
        "Preparing to insert %d blockchain chart records",
        len(df),
    )

    # Perform upsert operation (insert or update based on conflict)
    logger.debug("Upserting records into %s.%s", META_SCHEMA, META_BLOCKCHAIN_TABLE)
    try:
        with engine.begin() as conn:
            # Build upsert query using PostgreSQL ON CONFLICT clause
            # Conflict on chart_name - the natural unique key
            upsert_query = f"""
                INSERT INTO {META_SCHEMA}.{META_BLOCKCHAIN_TABLE}
                (chart_name, category, enabled, start_date, end_date)
                VALUES (:chart_name, :category, :enabled, :start_date, :end_date)
                ON CONFLICT (chart_name)
                DO UPDATE SET
                    category = EXCLUDED.category,
                    enabled = EXCLUDED.enabled,
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date,
                    updated_at = CURRENT_TIMESTAMP
                WHERE {META_SCHEMA}.{META_BLOCKCHAIN_TABLE}.enabled != EXCLUDED.enabled
                   OR {META_SCHEMA}.{META_BLOCKCHAIN_TABLE}.category != EXCLUDED.category
                   OR {META_SCHEMA}.{META_BLOCKCHAIN_TABLE}.start_date != EXCLUDED.start_date
                   OR {META_SCHEMA}.{META_BLOCKCHAIN_TABLE}.end_date != EXCLUDED.end_date
            """

            # Execute upsert for each record
            for record in records:
                conn.execute(
                    text(upsert_query),
                    {
                        "chart_name": record["chart_name"],
                        "category": record["category"],
                        "enabled": record["enabled"],
                        "start_date": record["start_date"],
                        "end_date": record["end_date"],
                    }
                )

        logger.debug("Upsert operation successful")

    except SQLAlchemyError as exc:
        logger.exception("Failed to upsert records into %s.%s", META_SCHEMA, META_BLOCKCHAIN_TABLE)
        raise TableError(
            f"Failed to upsert records into {META_SCHEMA}.{META_BLOCKCHAIN_TABLE}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during upsert operation: %s", exc)
        raise DataSaveError(
            f"Failed to upsert records into {META_SCHEMA}.{META_BLOCKCHAIN_TABLE}"
        ) from exc

    logger.info(
        "Successfully upserted %d blockchain chart records into %s.%s",
        len(df),
        META_SCHEMA,
        META_BLOCKCHAIN_TABLE,
    )



def get_blockchain_meta(schema=META_SCHEMA, table_name=META_BLOCKCHAIN_TABLE) -> Dict[str, List[Dict]]:
    """Retrieve blockchain chart configuration organized by category with full metadata.

    Loads all enabled blockchain charts from the meta.blockchain table
    and organizes them by category for use by the blockchain data fetcher.
    Includes per-chart metadata like start_date and end_date.

    Args:
        schema: Database schema name. Defaults to "meta".
        table: Table name. Defaults to "blockchain".

    Returns:
        Dictionary with category names as keys and lists of chart metadata dicts as values.
        Each chart metadata dict contains:
            - chart_name: str
            - start_date: str (YYYY-MM-DD format)
            - end_date: str (YYYY-MM-DD or 'now')
            - enabled: bool
            - id: int
        Example:
        {
            'currency_stats': [
                {'chart_name': 'market-price', 'start_date': '2010-01-01', 'end_date': 'now', 'enabled': True, 'id': 1},
                {'chart_name': 'trade-volume', 'start_date': '2010-01-01', 'end_date': 'now', 'enabled': True, 'id': 2}
            ],
            'mining_information': [
                {'chart_name': 'hash-rate', 'start_date': '2010-01-01', 'end_date': 'now', 'enabled': True, 'id': 3}
            ]
        }
        Returns empty dict if no records found or all are disabled.

    Example:
        >>> blockchain_meta = get_blockchain_meta()
        >>> for category, charts in blockchain_meta.items():
        ...     for chart in charts:
        ...         print(f"{chart['chart_name']} from {chart['start_date']} to {chart['end_date']}")

    Notes:
        - Uses get_engine() internally for database connection
        - Only returns enabled=True records
        - Charts are sorted by chart_name within each category
        - Preserves all relevant metadata for each chart
    """

    try:
        engine = get_engine()
        
        df = read_df(
            engine=engine,
            schema=schema,
            table_name=table_name,
            method='sql'
        )
        
        # Filter for enabled charts only
        df = df[df["enabled"] == True]
        
        if df.empty:
            logger.warning("No enabled blockchain charts found in %s.%s", schema, table_name)
            return {}
        
        # Organize charts by category with full metadata
        blockchain_meta: Dict[str, List[Dict]] = {}
        
        for _, row in df.iterrows():
            category = row.get("category")
            
            if not category:
                continue
            
            # Create chart metadata dict with all relevant fields
            chart_meta = {
                "chart_name": row.get("chart_name"),
                "start_date": row.get("start_date", "2010-01-01"),  # Fallback if NULL
                "end_date": row.get("end_date", "now"),  # Fallback if NULL
                "enabled": row.get("enabled", True),
                "id": row.get("id")
            }
            
            # Only add if chart_name exists
            if not chart_meta["chart_name"]:
                continue
                
            if category not in blockchain_meta:
                blockchain_meta[category] = []
            
            blockchain_meta[category].append(chart_meta)
        
        # Sort charts by chart_name within each category
        for category in blockchain_meta:
            blockchain_meta[category].sort(key=lambda x: x["chart_name"])
        
        # Log summary
        total_charts = sum(len(charts) for charts in blockchain_meta.values())
        logger.info(
            "Retrieved %d blockchain charts in %d categories from %s.%s",
            total_charts,
            len(blockchain_meta),
            schema,
            table_name,
        )
        
        return blockchain_meta
        
    except Exception as e:
        logger.error(f"Error retrieving blockchain meta: {e}")
        return {}


# ============================================================================
# MACRO TABLE OPERATIONS
# ============================================================================


def insert_macro(macro_config: dict) -> None:
    """Load macro economic indicator configuration and upsert records into the meta.macro table.

    This function orchestrates the macro economic indicator configuration loading:
    1. Validates macro configuration
    2. Transforms configuration into database records
    3. Ensures database schema and table exist
    4. Upserts (insert or update) records in the database
    5. Updates the updated_at timestamp for modified records
    6. Preserves historical data while updating metadata

    The function uses PostgreSQL UPSERT (INSERT ... ON CONFLICT ... DO UPDATE)
    to intelligently handle both new and existing indicators, ensuring records
    are created or updated without data loss.

    Args:
        macro_config: Dictionary containing macro indicator configuration with structure:
            {
                'economic_indicators': {
                    'category_name': [
                        {
                            'indicator_key': str,
                            'fred_series': str,
                            'frequency': str,
                            'enabled': bool
                        }
                    ]
                }
            }

    Raises:
        TypeError: If macro_config is None or not a dictionary.
        ValueError: If configuration is invalid or missing required fields.
        SchemaError: If database schema creation fails.
        TableError: If table creation or upsert fails.
        DataSaveError: If database operations fail.

    Example:
        >>> macro_config = {
        ...     'economic_indicators': {
        ...         'employment': [
        ...             {
        ...                 'indicator_key': 'unemployment_rate',
        ...                 'fred_series': 'UNRATE',
        ...                 'frequency': 'M',
        ...                 'enabled': True
        ...             }
        ...         ]
        ...     }
        ... }
        >>> insert_macro(macro_config)
        # INFO: Successfully upserted 1 records into meta.macro

    Notes:
        - Uses UPSERT pattern for intelligent insert/update handling
        - Only updates changed records, preserves created_at timestamp
        - Updates updated_at timestamp for all touched records
        - All database operations are transactional
        - Detailed logging at each step for troubleshooting
        - Validates inputs before attempting any operations
    """
        
    engine = get_engine()
    
    # Validate macro configuration is not None
    if macro_config is None:
        raise ValueError("macro configuration is None; expected a dictionary")

    # Transform configuration into database records
    logger.debug("Building database records from macro configuration")
    try:
        records = build_macro_records(macro_config)
    except (ValueError, TypeError) as exc:
        logger.error("Failed to build records from macro configuration: %s", exc)
        raise

    # Early exit if no records to process
    if not records:
        logger.warning("No records generated from macro configuration; nothing to process")
        return

    # Ensure database schema and table are ready
    logger.debug("Ensuring database schema and macro table exist")
    try:
        ensure_macro_table(engine)
    except (SchemaError, TableError) as exc:
        logger.error("Failed to prepare database: %s", exc)
        raise

    # Convert to DataFrame for bulk operations
    try:
        df = pd.DataFrame.from_records(records)
    except Exception as exc:
        logger.error("Failed to create DataFrame from records: %s", exc)
        raise ValueError("Failed to convert records to DataFrame") from exc

    if df.empty:
        logger.warning("DataFrame is empty after conversion; nothing to insert")
        return

    logger.info(
        "Preparing to insert %d macro indicator records",
        len(df),
    )

    # Perform upsert operation (insert or update based on conflict)
    logger.debug("Upserting records into %s.%s", META_SCHEMA, META_MACRO_TABLE)
    try:
        with engine.begin() as conn:
            # Build upsert query using PostgreSQL ON CONFLICT clause
            # Conflict on indicator_key - the natural unique key
            upsert_query = f"""
                INSERT INTO {META_SCHEMA}.{META_MACRO_TABLE}
                (indicator_key, fred_series, frequency, enabled)
                VALUES (:indicator_key, :fred_series, :frequency, :enabled)
                ON CONFLICT (indicator_key)
                DO UPDATE SET
                    fred_series = EXCLUDED.fred_series,
                    frequency = EXCLUDED.frequency,
                    enabled = EXCLUDED.enabled,
                    updated_at = CURRENT_TIMESTAMP
                WHERE {META_SCHEMA}.{META_MACRO_TABLE}.enabled != EXCLUDED.enabled
                   OR {META_SCHEMA}.{META_MACRO_TABLE}.fred_series != EXCLUDED.fred_series
                   OR {META_SCHEMA}.{META_MACRO_TABLE}.frequency != EXCLUDED.frequency
            """

            # Execute upsert for each record
            for record in records:
                conn.execute(
                    text(upsert_query),
                    {
                        "indicator_key": record["indicator_key"],
                        "fred_series": record["fred_series"],
                        "frequency": record["frequency"],
                        "enabled": record["enabled"],
                    }
                )

        logger.debug("Upsert operation successful")

    except SQLAlchemyError as exc:
        logger.exception("Failed to upsert records into %s.%s", META_SCHEMA, META_MACRO_TABLE)
        raise TableError(
            f"Failed to upsert records into {META_SCHEMA}.{META_MACRO_TABLE}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during upsert operation: %s", exc)
        raise DataSaveError(
            f"Failed to upsert records into {META_SCHEMA}.{META_MACRO_TABLE}"
        ) from exc

    logger.info(
        "Successfully upserted %d macro indicator records into %s.%s",
        len(df),
        META_SCHEMA,
        META_MACRO_TABLE,
    )



def get_macro_meta(
    schema: str = META_SCHEMA,
    table: str = META_MACRO_TABLE
) -> Dict[str, Any]:
    """Retrieve macro economic indicator configuration organized by frequency.

    Loads all enabled macro economic indicators from the meta.macro table
    and organizes them by frequency for use by the macro data fetcher.
    Uses read_df utility for consistent database access patterns.

    Args:
        schema: Database schema name. Defaults to "meta".
        table: Table name. Defaults to "macro".

    Returns:
        Dictionary with two keys:
        - 'economic_indicators': Dict mapping indicator_key to config:
          {
              'unemployment_rate': {
                  'fred_series': 'UNRATE',
                  'frequency': 'monthly'
              },
              'gdp': {
                  'fred_series': 'GDP',
                  'frequency': 'quarterly'
              }
          }
        - 'indicators_by_frequency': Dict organizing indicators by frequency:
          {
              'monthly': [...],
              'quarterly': [...]
          }
        Returns empty dict if no records found or all are disabled.

    Example:
        >>> macro_meta = get_macro_meta()
        >>> indicators_config = macro_meta['economic_indicators']
        >>> for indicator_key, config in indicators_config.items():
        ...     print(f"{indicator_key}: {config['fred_series']}")

    Notes:
        - Uses get_engine() internally for database connection
        - Only returns enabled=True records
        - Returns both flat dict and frequency-organized dict for flexibility
        - Automatically filters by enabled status
    """
    try:
        engine = get_engine()        
        df = read_df(
            engine=engine,
            table_name=table,
            schema=schema,
            method='sql'
        )
        
        # Filter for enabled indicators only
        df = df[df["enabled"].astype(bool)]
        
        if df.empty:
            logger.warning("No enabled macro indicators found in %s.%s", schema, table)
            return {}
        
        # Build economic_indicators dict for backward compatibility
        economic_indicators: Dict[str, Any] = {}
        indicators_by_frequency: Dict[str, List[Dict[str, str]]] = {}
        
        for _, row in df.iterrows():
            indicator_key = row.get("indicator_key")
            fred_series = row.get("fred_series")
            frequency = row.get("frequency")
            
            if not indicator_key or not fred_series or not frequency:
                continue
            
            # Build economic_indicators dict
            economic_indicators[indicator_key] = {
                'fred_series': fred_series,
                'frequency': frequency
            }
            
            # Build indicators_by_frequency dict
            if frequency not in indicators_by_frequency:
                indicators_by_frequency[frequency] = []
            
            indicators_by_frequency[frequency].append({
                'indicator_key': indicator_key,
                'fred_series': fred_series
            })
        
        result = {
            'economic_indicators': economic_indicators,
            'indicators_by_frequency': indicators_by_frequency
        }
        
        logger.info(
            "Retrieved %d macro indicators in %d frequencies from %s.%s",
            len(economic_indicators),
            len(indicators_by_frequency),
            schema,
            table,
        )
        return result
    
    except Exception as exc:
        logger.error(
            "Failed to retrieve macro metadata from %s.%s: %s",
            schema,
            table,
            str(exc),
        )
        return {}
