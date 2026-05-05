"""
Functional service layer for database operations.

This module provides high-level functions for common database operations
including OHLCV data, tick data, and bar data management. All functions
are standalone and easy to use.

Key Features:
- OHLCV data insert/read operations
- Tick-by-tick trade data operations
- Specialized bar data storage (volume, volatility, dollar bars)
- Automatic table schema detection and handling
"""

from typing import Optional, List, Dict, Any, Union, Final
import pandas as pd
import json
from datetime import datetime
from common.db.utils import ensure_schema, insert_df, read_df, insert_dict, ensure_table
from common.db.config import get_engine
from common.utils.time import datetime_to_timestamp
from common.db.models.data import ensure_state_table, ensure_ohlcv_table
from sqlalchemy import text
from common.utils.json_encoder import RobustJSONEncoder
from sqlalchemy.exc import SQLAlchemyError
from common.constants import DATA_SCHEMA
from common.db.exceptions import QueryError
from common.logging import get_logger


logger = get_logger(__name__)

# Get engine once at module load time
# =============================================================================
# OHLCV OPERATIONS
# =============================================================================


def insert_ohlcv(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
    if_exists: str = "append",
    index: bool = False,
    method: Optional[str] = "copy",
    chunksize: int = 10_000,
) -> None:
    """
    Insert OHLCV (Open, High, Low, Close, Volume) data into database.

    Automatically creates a TimescaleDB hypertable for efficient time-series
    storage and querying. Uses the fastest available insertion method by default.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data. Must contain columns:
        - timestamp: DateTime or unix timestamp
        - open: Opening price
        - high: Highest price
        - low: Lowest price
        - close: Closing price
        - volume: Trading volume
    table_name : str
        Target table name (e.g., 'btc_1m', 'eth_5m')
    schema : str
        Target schema name (e.g., 'data_binance', 'data_coinbase')
    if_exists : str, default "append"
        Behavior when table exists. One of:
        - "append": Add to existing table
        - "replace": Drop and recreate table
        - "fail": Raise error if exists
    index : bool, default False
        Whether to write DataFrame index as a column.
    method : Optional[str], default "copy"
        Write method. One of: "copy", "sql", "executemany"
    chunksize : int, default 10_000
        Rows per batch for chunked operations.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'timestamp': ['2024-01-01', '2024-01-02'],
    ...     'open': [100.0, 101.0],
    ...     'high': [102.0, 103.0],
    ...     'low': [99.0, 100.5],
    ...     'close': [101.0, 102.5],
    ...     'volume': [1000, 1500]
    ... })
    >>> insert_ohlcv(df, 'btc_1m', 'data_binance')

    >>> # Replace existing data
    >>> insert_ohlcv(df, 'eth_5m', 'data_coinbase', if_exists='replace')
    """

    df["exchange"] = exchange
    df["symbol"] = symbol
    df["timeframe"] = timeframe

    # Optional: reorder columns to have exchange, symbol, timeframe first
    cols_order = ["exchange", "symbol", "timeframe"] + [
        c for c in df.columns if c not in ["exchange", "symbol", "timeframe"]
    ]
    df = df[cols_order]

    engine = get_engine()
    schema_name = DATA_SCHEMA
    table_name = "time"

    insert_df(
        df=df,
        engine=engine,
        schema_name=schema_name,
        table_name=table_name,
        if_exists=if_exists,
        index=index,
        method=method,
        chunksize=chunksize,
        is_timeseries=True,
    )


def read_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
    bar_type: str = 'time',
    start_date: Optional[Union[str, int]] = None,
    end_date: Optional[Union[str, int]] = None,
    columns: Optional[list] = None,
    limit: Optional[int] = None,
    method: str = "copy",
    return_timestamp: bool = False,  
) -> pd.DataFrame:
    """
    Read OHLCV data from database with optional filtering.

    Parameters
    ----------
    table_name : str
        Source table name (e.g., 'btc_1m', 'eth_5m')
    schema : str
        Source schema name (e.g., 'data_binance', 'data_coinbase')
    start_date : Optional[str], default None
        Start date filter in ISO format (e.g., '2024-01-01')
        If None, reads from beginning.
    end_date : Optional[str], default None
        End date filter in ISO format (e.g., '2024-01-31')
        If None, reads to end.
    columns : Optional[list], default None
        Specific columns to read. If None, reads all columns.
    limit : Optional[int], default None
        Maximum number of rows to return.
    method : str, default "sql"
        Read method. One of: "sql", "copy", "connectorx"

    Returns
    -------
    pd.DataFrame
        DataFrame containing OHLCV data with columns:
        - timestamp: DateTime
        - open, high, low, close: Price data
        - volume: Trading volume

    Examples
    --------
    >>> # Read all data
    >>> df = read_ohlcv('btc_1m', 'data_binance')

    >>> # Read specific date range
    >>> df = read_ohlcv(
    ...     'eth_5m',
    ...     'data_coinbase',
    ...     start_date='2024-01-01',
    ...     end_date='2024-01-31'
    ... )

    >>> # Read specific columns with limit
    >>> df = read_ohlcv(
    ...     'btc_1h',
    ...     'data_binance',
    ...     columns=['timestamp', 'close', 'volume'],
    ...     limit=1000
    ... )
    """

    if bar_type != 'time':
        df = read_bar(exchange=exchange, symbol=symbol, bar_type=bar_type, start_date=start_date, end_date=end_date, columns=columns, limit=limit, method=method)
        return df
    
    engine = get_engine()

    schema = DATA_SCHEMA
    table_name = bar_type

    # Build WHERE clause for date filtering
    where_parts = [
        f"exchange = '{exchange}'",
        f"symbol = '{symbol}'",
        f"timeframe = '{timeframe}'"
    ]
    if start_date is not None:
        if isinstance(start_date, int):
            start_ts = start_date  # already in unix ms
        else:
            start_ts = datetime_to_timestamp(start_date)
        where_parts.append(f"timestamp >= {start_ts}")

    if end_date is not None:
        if isinstance(end_date, int):
            end_ts = end_date  # already in unix ms
        else:
            end_ts = datetime_to_timestamp(end_date)
        where_parts.append(f"timestamp <= {end_ts}")

    where_clause = " AND ".join(where_parts) 

    # Read data
    df = read_df(
        engine=engine,
        schema=schema,
        table_name=table_name,
        columns=columns,
        where=where_clause,
        method=method,
        order_by="timestamp",
    )

    # Apply limit if specified
    if limit and not df.empty:
        df = df.head(limit)

    # Drop timestamp column if not requested
    if not return_timestamp and "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    return df


def read_bar(
    exchange: str,
    symbol: str,
    bar_type: str,
    start_date: Optional[Union[str, int]] = None,
    end_date: Optional[Union[str, int]] = None,
    columns: Optional[list] = None,
    limit: Optional[int] = None,
    method: str = "copy"
) -> pd.DataFrame:
    """
    Read bar data (volume, dollar, volatility, or time bars) from database.
    """
    
    engine = get_engine()
    schema = DATA_SCHEMA
    table_name = bar_type
    
    try:
        
        # Verify table exists
        if not ensure_table(engine, schema, table_name):
            logger.warning(f"Table {schema}.{table_name} does not exist for {exchange}/{symbol}")
            return pd.DataFrame()

        # Build WHERE clause
        where_parts = [f"exchange = '{exchange}'", f"symbol = '{symbol}'"]

        if start_date is not None:
            # Add single quotes around the value
            where_parts.append(f"datetime >= '{start_date}'") 
        if end_date is not None:
            # Add single quotes around the value
            where_parts.append(f"datetime <= '{end_date}'")

        where_clause = " AND ".join(where_parts) if where_parts else None
        
        # Read data
        df = read_df(
            engine=engine,
            schema=schema,
            table_name=table_name,
            columns=columns,
            where=where_clause,
            method=method,
            order_by="datetime",
        )
        
        # Apply limit
        if limit and not df.empty:
            df = df.head(limit)
        
        logger.debug(f"Read {len(df)} rows from {schema}.{table_name} for {exchange}/{symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error reading bar data for {exchange}/{symbol}/{bar_type}: {e}")
        return pd.DataFrame()

def get_ohlcv_last_timestamp(exchange, symbol, timeframe='1m'):
    """
    Retrieve the latest timestamp for a specific exchange, symbol, and timeframe
    from the unified time_bar table.

    Returns datetime or None if no data exists.
    """
    schema_name = DATA_SCHEMA
    table_name = "time"

    try:
        engine = get_engine()
        ensure_schema(engine, schema_name)
        if not ensure_ohlcv_table(engine, schema_name, table_name):
            return None

        query = text(
            f"""
            SELECT MAX(timestamp) AS last_ts
            FROM {schema_name}.{table_name}
            WHERE exchange = :exchange
              AND symbol = :symbol
              AND timeframe = :timeframe
            """
        )

        with engine.connect() as conn:
            result = conn.execute(
                query,
                {"exchange": exchange, "symbol": symbol, "timeframe": timeframe}
            ).scalar()

        return result  # Will be None if no rows exist

    except SQLAlchemyError as e:
        logger.error("Database error while retrieving OHLCV timestamp: %s", e)
        return None

def get_ohlcv_last_datetime(exchange, symbol, timeframe='1m'):
    """
    Retrieve the latest datetime for a specific exchange, symbol, and timeframe
    from the unified time_bar table.

    Returns datetime or None if no data exists.
    """
    schema_name = DATA_SCHEMA
    table_name = "time"

    try:
        engine = get_engine()
        ensure_schema(engine, schema_name)
        if not ensure_ohlcv_table(engine, schema_name, table_name):
            return None

        query = text(
            f"""
            SELECT MAX(datetime) AS last_datetime
            FROM {schema_name}.{table_name}
            WHERE exchange = :exchange
              AND symbol = :symbol
              AND timeframe = :timeframe
            """
        )

        with engine.connect() as conn:
            result = conn.execute(
                query,
                {"exchange": exchange, "symbol": symbol, "timeframe": timeframe}
            ).scalar()

        return result  # Will be None if no rows exist

    except SQLAlchemyError as e:
        logger.error("Database error while retrieving OHLCV timestamp for %s/%s/%s: %s",exchange, symbol, timeframe, e)
        return None

# =============================================================================
# BAR QUALITY STATS OPERATIONS
# =============================================================================

_BAR_STATS_COLUMNS: Final = (
    "exchange", "symbol", "bar_type",
    # coverage
    "total_bars", "date_range_start", "date_range_end", "calendar_days",
    "mean_bars_per_day", "std_bars_per_day",
    # sampling
    "bar_size_mean", "bar_size_std", "bar_size_cv",
    "bar_size_p5", "bar_size_p25", "bar_size_p50", "bar_size_p75", "bar_size_p95",
    "duration_mean", "duration_std", "duration_cv", "duration_p95",
    "tick_count_mean",
    # return distribution
    "return_mean", "return_std", "return_skew", "return_kurtosis", "return_entropy",
    # ML / IID quality
    "return_autocorr_lag1", "abs_return_autocorr_lag1",
    "variance_ratio_lag2", "variance_ratio_lag5",
    "rolling_vol_cv", "eff_sample_size",
    # OHLC integrity
    "pct_valid_bars", "close_position_mean", "close_position_std", "price_range_mean",
    # scores
    "sampling_score", "ml_score", "integrity_score", "quality_score",
    # metadata
    "bars_used", "computed_at",
)


def upsert_bar_stats(stats: Dict[str, Any]) -> None:
    """Upsert one row of bar quality stats into data_bars.bars_quality_stats.

    Parameters
    ----------
    stats : dict
        Keys must match the columns in _BAR_STATS_COLUMNS.
        Missing keys are inserted as NULL.
    """
    from common.db.models.data import ensure_bar_stats_table

    engine = get_engine()
    ensure_bar_stats_table()

    table = f"{DATA_SCHEMA}.bars_quality_stats"
    col_list = ", ".join(_BAR_STATS_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in _BAR_STATS_COLUMNS)
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in _BAR_STATS_COLUMNS
        if c not in ("exchange", "symbol", "bar_type")
    )

    sql = f"""
        INSERT INTO {table} ({col_list})
        VALUES ({placeholders})
        ON CONFLICT (exchange, symbol, bar_type)
        DO UPDATE SET {updates}
    """

    # Fill missing keys with None so SQLAlchemy binds them as NULL
    row = {c: stats.get(c) for c in _BAR_STATS_COLUMNS}

    try:
        with engine.begin() as conn:
            conn.execute(text(sql), row)
        logger.info(
            "Upserted bar stats: %s/%s/%s (quality=%.1f)",
            stats.get("exchange"), stats.get("symbol"), stats.get("bar_type"),
            stats.get("quality_score") or 0.0,
        )
    except SQLAlchemyError as exc:
        logger.exception(
            "Failed to upsert bar stats for %s/%s/%s",
            stats.get("exchange"), stats.get("symbol"), stats.get("bar_type"),
        )
        raise


# =============================================================================
# TICK DATA OPERATIONS
# =============================================================================


def insert_tick_data(
    data: List[Dict],
    schema: str,
    table_name: str,
) -> None:
    """
    Insert tick-by-tick trade data into database.

    Stores individual trades with high precision timestamps. Automatically
    creates TimescaleDB hypertable for efficient storage of high-frequency data.

    Parameters
    ----------
    data : List[Dict]
        List of tick dictionaries. Each dict should contain:
        - timestamp: Trade timestamp
        - price: Trade price
        - volume: Trade volume/size
        - side: 'buy' or 'sell' (optional)
        - trade_id: Unique trade identifier (optional)
    table_name : str
        Target table name (e.g., 'btc_trades', 'eth_trades')
    schema : str
        Target schema name (e.g., 'data_tick_binance')

    Examples
    --------
    >>> tick_data = [
    ...     {
    ...         'timestamp': '2024-01-01 10:00:00',
    ...         'price': 45000.0,
    ...         'volume': 0.5,
    ...         'side': 'buy'
    ...     },
    ...     {
    ...         'timestamp': '2024-01-01 10:00:01',
    ...         'price': 45001.0,
    ...         'volume': 0.3,
    ...         'side': 'sell'
    ...     }
    ... ]
    >>> insert_tick_data(tick_data, 'btc_trades', 'data_tick_binance')
    """

    engine = get_engine()

    # Insert using dict method for structured data
    insert_dict(
        engine=engine,
        data=data,
        table_name=table_name,
        schema_name=schema,
        is_timeseries=True,
    )


def read_tick_data(
    schema: str,
    table_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    method: str = "sql",
) -> pd.DataFrame:
    """
    Read tick-by-tick trade data from database.

    Parameters
    ----------
    table_name : str
        Source table name (e.g., 'btc_trades', 'eth_trades')
    schema : str
        Source schema name (e.g., 'data_tick_binance')
    start_date : Optional[str], default None
        Start date filter in ISO format (e.g., '2024-01-01')
    end_date : Optional[str], default None
        End date filter in ISO format (e.g., '2024-01-31')
    limit : Optional[int], default None
        Maximum number of rows to return.
    method : str, default "sql"
        Read method. One of: "sql", "copy", "connectorx"

    Returns
    -------
    pd.DataFrame
        DataFrame containing tick data with timestamp, price, volume, etc.

    Examples
    --------
    >>> # Read all tick data
    >>> df = read_tick_data('btc_trades', 'data_tick_binance')

    >>> # Read specific time range
    >>> df = read_tick_data(
    ...     'eth_trades',
    ...     'data_tick_coinbase',
    ...     start_date='2024-01-01 10:00:00',
    ...     end_date='2024-01-01 11:00:00',
    ...     limit=10000
    ... )
    """

    engine = get_engine()

    # Build WHERE clause for date filtering
    where_parts = []
    if start_date:
        where_parts.append(f"timestamp >= '{start_date}'")
    if end_date:
        where_parts.append(f"timestamp <= '{end_date}'")

    where_clause = " AND ".join(where_parts) if where_parts else None

    # Read data
    df = read_df(
        engine=engine,
        schema=schema,
        table_name=table_name,
        where=where_clause,
        method=method,
        order_by="timestamp",
    )

    # Apply limit if specified
    if limit and not df.empty:
        return df.head(limit)

    return df


# =============================================================================
# BAR DATA OPERATIONS
# =============================================================================

def insert_volume_bar(cursor, table_name: str, bar_data: Dict, exchange: str, symbol: str) -> None:
    """Store volume bar."""
    query = f"""
    INSERT INTO {table_name}
    (exchange, symbol, datetime, datetime_start, datetime_end, open, high, low, close, volume,
     bar_size, dollar_volume, duration_minutes, tick_count, bar_return, price_range, close_position)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (exchange, symbol, datetime) DO NOTHING
    """
    cursor.execute(
        query,
        (
            exchange,
            symbol,
            bar_data["datetime"],
            bar_data.get("datetime_start"),
            bar_data.get("datetime_end"),
            bar_data["open"],
            bar_data["high"],
            bar_data["low"],
            bar_data["close"],
            bar_data["volume"],
            bar_data.get("bar_size", 0),
            bar_data.get("dollar_volume", 0),
            bar_data.get("duration_minutes", 0),
            bar_data.get("tick_count", 0),
            bar_data.get("bar_return", 0),
            bar_data.get("price_range", 0),
            bar_data.get("close_position", 0),
        ),
    )


def insert_volatility_bar(cursor, table_name: str, bar_data: Dict, exchange: str, symbol: str) -> None:
    """Store volatility bar."""
    query = f"""
    INSERT INTO {table_name}
    (exchange, symbol, datetime, datetime_start, datetime_end, open, high, low, close, volume,
        bar_size, dollar_volume, duration_minutes, tick_count,
        bar_return, price_range, close_position)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (datetime, exchange, symbol) DO NOTHING
    """
    cursor.execute(
        query,
        (
            exchange,
            symbol,
            bar_data["datetime"],
            bar_data.get("datetime_start"),
            bar_data.get("datetime_end"),
            bar_data["open"],
            bar_data["high"],
            bar_data["low"],
            bar_data["close"],
            bar_data["volume"],
            bar_data.get("bar_size", 0),
            bar_data.get("dollar_volume", 0),
            bar_data.get("duration_minutes", 0),
            bar_data.get("tick_count", 0),
            bar_data.get("bar_return", 0),
            bar_data.get("price_range", 0),
            bar_data.get("close_position", 0),
        ),
    )


def insert_dollar_bar(cursor, table_name: str, bar_data: Dict, exchange: str, symbol: str) -> None:
    """Store dollar bar."""
    query = f"""
    INSERT INTO {table_name}
    (exchange, symbol, datetime, datetime_start, datetime_end, open, high, low, close, volume,
        bar_size, vwap, duration_minutes, tick_count,
        bar_return, price_range, close_position)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (datetime, exchange, symbol) DO NOTHING
    """
    cursor.execute(
        query,
        (
            exchange,
            symbol,
            bar_data["datetime"],
            bar_data.get("datetime_start"),
            bar_data.get("datetime_end"),
            bar_data["open"],
            bar_data["high"],
            bar_data["low"],
            bar_data["close"],
            bar_data["volume"],
            bar_data.get("bar_size", 0),
            bar_data.get("vwap", bar_data.get("close", 0)),
            bar_data.get("duration_minutes", 0),
            bar_data.get("tick_count", 0),
            bar_data.get("bar_return", 0),
            bar_data.get("price_range", 0),
            bar_data.get("close_position", 0),
        ),
    )


def insert_range_bar(cursor, table_name: str, bar_data: Dict, exchange: str, symbol: str) -> None:
    """Store range bar (same schema as volume bar)."""
    query = f"""
    INSERT INTO {table_name}
    (exchange, symbol, datetime, datetime_start, datetime_end, open, high, low, close, volume,
        bar_size, dollar_volume, duration_minutes, tick_count,
        bar_return, price_range, close_position)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (datetime, exchange, symbol) DO NOTHING
    """
    cursor.execute(
        query,
        (
            exchange,
            symbol,
            bar_data["datetime"],
            bar_data.get("datetime_start"),
            bar_data.get("datetime_end"),
            bar_data["open"],
            bar_data["high"],
            bar_data["low"],
            bar_data["close"],
            bar_data["volume"],
            bar_data.get("bar_size", 0),
            bar_data.get("dollar_volume", 0),
            bar_data.get("duration_minutes", 0),
            bar_data.get("tick_count", 0),
            bar_data.get("bar_return", 0),
            bar_data.get("price_range", 0),
            bar_data.get("close_position", 0),
        ),
    )


def insert_renko_bar(cursor, table_name: str, bar_data: Dict, exchange: str, symbol: str) -> None:
    """Store renko bar (includes direction column)."""
    query = f"""
    INSERT INTO {table_name}
    (exchange, symbol, datetime, datetime_start, datetime_end, open, high, low, close, volume,
        bar_size, dollar_volume, direction, duration_minutes, tick_count,
        bar_return, price_range, close_position)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (datetime, exchange, symbol) DO NOTHING
    """
    cursor.execute(
        query,
        (
            exchange,
            symbol,
            bar_data["datetime"],
            bar_data.get("datetime_start"),
            bar_data.get("datetime_end"),
            bar_data["open"],
            bar_data["high"],
            bar_data["low"],
            bar_data["close"],
            bar_data["volume"],
            bar_data.get("bar_size", 0),
            bar_data.get("dollar_volume", 0),
            bar_data.get("direction", "bullish"),
            bar_data.get("duration_minutes", 0),
            bar_data.get("tick_count", 0),
            bar_data.get("bar_return", 0),
            bar_data.get("price_range", 0),
            bar_data.get("close_position", 0),
        ),
    )


def insert_hybrid_bar(cursor, table_name: str, bar_data: Dict, exchange: str, symbol: str) -> None:
    """Store hybrid bar (bar_size = dollar volume, plus vwap and bar_volatility)."""
    query = f"""
    INSERT INTO {table_name}
    (exchange, symbol, datetime, datetime_start, datetime_end, open, high, low, close, volume,
        bar_size, vwap, bar_volatility, duration_minutes, tick_count,
        bar_return, price_range, close_position)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (datetime, exchange, symbol) DO NOTHING
    """
    cursor.execute(
        query,
        (
            exchange,
            symbol,
            bar_data["datetime"],
            bar_data.get("datetime_start"),
            bar_data.get("datetime_end"),
            bar_data["open"],
            bar_data["high"],
            bar_data["low"],
            bar_data["close"],
            bar_data["volume"],
            bar_data.get("bar_size", 0),
            bar_data.get("vwap", bar_data.get("close", 0)),
            bar_data.get("bar_volatility", 0),
            bar_data.get("duration_minutes", 0),
            bar_data.get("tick_count", 0),
            bar_data.get("bar_return", 0),
            bar_data.get("price_range", 0),
            bar_data.get("close_position", 0),
        ),
    )


def insert_standard_bar(cursor, table_name: str, bar_data: Dict, exchange: str, symbol: str) -> bool:
    """
    Store standard time-based bar with basic OHLCV schema.

    Standard bars are fixed-time interval bars (e.g., 1-minute, 5-minute)
    with basic OHLCV data and optional metadata.

    Parameters
    ----------
    connection : psycopg2.connection
        Active database connection.
    cursor : psycopg2.cursor
        Active cursor for query execution.
    table_name : str
        Fully qualified table name (schema.table)
    bar_data : Dict
        Bar data with fields:
        - timestamp: Bar timestamp (required)
        - open, high, low, close: OHLC prices (required)
        - volume: Trading volume (required)
        - bar_size: Size metric (optional)
        - tick_count: Number of ticks (optional)

    Returns
    -------
    bool
        True if insert successful, raises exception otherwise.

    Notes
    -----
    - Uses ON CONFLICT (timestamp) DO NOTHING for idempotent inserts
    - Minimal schema for simple time-based bars
    - Tick count defaults to 0 if not provided
    """

    query = f"""
    INSERT INTO {table_name}
    (exchange, symbol, timestamp, open, high, low, close, volume, bar_size, tick_count)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (timestamp, exchange, symbol) DO NOTHING
    """

    cursor.execute(
        query,
        (
            exchange,
            symbol,
            bar_data["timestamp"],
            bar_data["open"],
            bar_data["high"],
            bar_data["low"],
            bar_data["close"],
            bar_data["volume"],
            bar_data["bar_size"],
            bar_data.get("tick_count", 0),
        ),
    )


# ============================================================================
# STATE MANAGEMENT
# ============================================================================


def get_bar_state(
    exchange: str, symbol: str, bar_type: str
) -> Optional[Dict[str, Any]]:
    """Retrieve current bar processing state.

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.
        exchange: Exchange name (e.g., 'binance', 'bybit').
        symbol: Symbol name (e.g., 'BTC', 'ETH').
        bar_type: Type of bars (volume, volatility, dollar, tick, time).

    Returns:
        Dict containing state data if exists:
            - last_processed_timestamp: Last timestamp processed
            - current_bar_start: Start time of current bar being built
            - current_bar_data: Accumulated data for current bar
            - ema_state: EMA calculation state
            - config_params: Bar configuration parameters
        Or None if no state exists yet.

    Raises:
        QueryError: If database query fails.

    Example:
        >>> engine = get_engine()
        >>> state = get_bar_state(engine, 'binance', 'BTC', 'dollar')
        >>> if state:
        ...     print(f"Last processed: {state['last_processed_timestamp']}")
    """
    engine = get_engine()

    try:
        ensure_state_table()

        with engine.connect() as conn:
            query = text(
                f"""
                SELECT
                    last_processed_datetime,
                    current_bar_datetime,
                    current_bar_data,
                    ema_state
                FROM {DATA_SCHEMA}.bars_state
                WHERE exchange = :exchange
                  AND symbol = :symbol
                  AND bar_type = :bar_type
            """
            )

            result = conn.execute(
                query,
                {
                    "exchange": exchange.strip().lower(),
                    "symbol": symbol.strip().lower(),
                    "bar_type": bar_type.strip().lower(),
                },
            ).fetchone()

            if result is None:
                logger.info(
                    "No state found for %s/%s/%s, will use default",
                    exchange,
                    symbol,
                    bar_type,
                )
                return None

            # Parse JSONB fields
            current_bar_data = result[2] if result[2] else {}
            ema_state = result[3] if result[3] else {}

            state = {
                "last_processed_datetime": result[0],
                "current_bar_datetime": result[1],
                "current_bar_data": current_bar_data,
                "ema_state": ema_state,
            }

            logger.debug(
                "Retrieved state for %s/%s/%s: last_datetime=%s",
                exchange,
                symbol,
                bar_type,
                result[0],
            )
            return state

    except SQLAlchemyError as exc:
        logger.exception("Failed to retrieve bar state")
        raise QueryError(
            f"Failed to get bar state for {exchange}/{symbol}/{bar_type}"
        ) from exc


def update_bar_state(
    exchange: str,
    symbol: str,
    bar_type: str,
    last_processed_datetime: Optional[datetime] = None,
    current_bar_datetime: Optional[datetime] = None,
    current_bar_data: Optional[Dict[str, Any]] = None,
    ema_state: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update bar processing state.

    Uses UPSERT pattern to handle both new and existing states.
    Only updates provided fields, leaving others unchanged.

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.
        exchange: Exchange name (e.g., 'binance', 'bybit').
        symbol: Symbol name (e.g., 'BTC', 'ETH').
        bar_type: Type of bars (volume, volatility, dollar, tick, time).
        last_processed_timestamp: Last timestamp successfully processed.
        current_bar_start: Start time of bar currently being accumulated.
        current_bar_data: Accumulated data for current bar.
        ema_state: EMA calculation state.

    Returns:
        bool: True if update successful, False otherwise.

    Raises:
        QueryError: If database update fails.

    Example:
        >>> engine = get_engine()
        >>> success = update_bar_state(
        ...     engine, 'binance', 'BTC', 'dollar',
        ...     last_processed_timestamp=datetime.now(),
        ...     current_bar_data={'volume': 100, 'high': 30000}
        ... )
    """
    engine = get_engine()

    try:
        ensure_state_table()

        # Normalize inputs
        exchange = exchange.strip().lower()
        symbol = symbol.strip().lower()
        bar_type = bar_type.strip().lower()

        # Serialize JSONB fields using RobustJSONEncoder for pandas/numpy types
        current_bar_data_json = json.dumps(
            current_bar_data or {}, cls=RobustJSONEncoder
        )
        ema_state_json = json.dumps(ema_state or {}, cls=RobustJSONEncoder)

        # Use raw psycopg2 connection to avoid SQLAlchemy parameter style issues
        raw_conn = engine.raw_connection()
        try:
            cursor = raw_conn.cursor()
            try:
                query = f"""
                    INSERT INTO {DATA_SCHEMA}.bars_state
                    (exchange, symbol, bar_type, last_processed_datetime,
                     current_bar_datetime, current_bar_data, ema_state)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (exchange, symbol, bar_type)
                    DO UPDATE SET
                        last_processed_datetime = COALESCE(EXCLUDED.last_processed_datetime,
                                                           bars_state.last_processed_datetime),
                        current_bar_datetime = COALESCE(EXCLUDED.current_bar_datetime,
                                                        bars_state.current_bar_datetime),
                        current_bar_data = COALESCE(EXCLUDED.current_bar_data,
                                                    bars_state.current_bar_data),
                        ema_state = COALESCE(EXCLUDED.ema_state, bars_state.ema_state),
                        updated_at = CURRENT_TIMESTAMP
                """

                cursor.execute(
                    query,
                    (
                        exchange,
                        symbol,
                        bar_type,
                        last_processed_datetime,
                        current_bar_datetime,
                        current_bar_data_json,
                        ema_state_json,
                    ),
                )
                raw_conn.commit()

                logger.debug(
                    "Updated state for %s/%s/%s: last_datetime=%s, bar_start=%s",
                    exchange,
                    symbol,
                    bar_type,
                    last_processed_datetime,
                    current_bar_datetime,
                )
                return True
            finally:
                cursor.close()
        finally:
            raw_conn.close()

    except Exception as exc:
        logger.exception("Failed to update bar state")
        raise QueryError(
            f"Failed to update bar state for {exchange}/{symbol}/{bar_type}"
        ) from exc


# ============================================================================
# RECENT BARS RETRIEVAL
# ============================================================================


def get_recent_bars(table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent bars from the database for EMA initialization.

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.
        table_name: Name of the bars table.
        limit: Maximum number of bars to retrieve.

    Returns:
        List of bar dictionaries with timestamp_start, open, high, low, close, volume.

    Raises:
        QueryError: If database query fails.

    Example:
        >>> engine = get_engine()
        >>> bars = get_recent_bars(engine, 'data_bars.binance_BTC_dollar', limit=100)
        >>> print(f"Retrieved {len(bars)} recent bars")
    """
    engine = get_engine()

    if not table_name or "." not in table_name:
        raise ValueError("table_name must be fully qualified (schema.table)")

    try:
        with engine.connect() as conn:
            # Check if it's a volume table by checking table name
            if "volume" in table_name:
                query = text(
                    f"""
                SELECT timestamp, open, high, low, close, volume, 
                       cum_ticks, cum_dollar_value
                FROM {table_name}
                WHERE bar_status = 'COMPLETED'
                ORDER BY timestamp DESC
                LIMIT :limit
                """
                )
                rows = conn.execute(query, {"limit": limit}).fetchall()

                bars = []
                for row in rows:
                    bars.append(
                        {
                            "timestamp_start": row[0],
                            "timestamp_end": row[0],
                            "open": float(row[1]),
                            "high": float(row[2]),
                            "low": float(row[3]),
                            "close": float(row[4]),
                            "volume": float(row[5]),
                            "cum_ticks": int(row[6]) if row[6] is not None else 0,
                            "dollar_volume": float(row[7]) if row[7] is not None else 0,
                            "bar_size": float(row[7]) if row[7] is not None else 0,
                            "tick_count": int(row[6]) if row[6] is not None else 0,
                        }
                    )
            else:
                # Standard bar table
                query = text(
                    f"""
                SELECT timestamp, open, high, low, close, volume, bar_size, tick_count
                FROM {table_name}
                ORDER BY timestamp DESC
                LIMIT :limit
                """
                )
                rows = conn.execute(query, {"limit": limit}).fetchall()

                bars = []
                for row in rows:
                    bars.append(
                        {
                            "timestamp_start": row[0],
                            "timestamp_end": row[0],
                            "open": float(row[1]),
                            "high": float(row[2]),
                            "low": float(row[3]),
                            "close": float(row[4]),
                            "volume": float(row[5]),
                            "bar_size": float(row[6]),
                            "tick_count": int(row[7]) if row[7] is not None else 0,
                        }
                    )

            # Reverse to get chronological order
            logger.debug("Retrieved %d recent bars from %s", len(bars), table_name)
            return list(reversed(bars))

    except SQLAlchemyError as exc:
        logger.exception("Failed to get recent bars from %s", table_name)
        raise QueryError(f"Failed to get recent bars from {table_name}") from exc


# ============================================================================
# REGIME CALCULATION
# ============================================================================


def get_historical_bars_with_regime(
    table_name: str, limit: int, offset: int = 0
) -> pd.DataFrame:
    """Get historical bars that already have regime data.

    Retrieves bars with completed regime calculations for use as context
    in incremental regime updates.

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.
        table_name: Name of the bars table.
        limit: Number of bars to retrieve.
        offset: Number of bars to skip from most recent.

    Returns:
        DataFrame with columns: timestamp (index), open, high, low, close, volume.
        Empty DataFrame if no bars with regimes found.

    Raises:
        QueryError: If database query fails.
    """
    engine = get_engine()

    if not table_name or "." not in table_name:
        raise ValueError("table_name must be fully qualified (schema.table)")

    try:
        with engine.connect() as conn:
            # Try to get bars that already have regime data
            query = text(
                f"""
            SELECT datetime, open, high, low, close, volume
            FROM {table_name}
            WHERE primary_regime IS NOT NULL AND regime_confidence IS NOT NULL
            ORDER BY datetime DESC
            LIMIT :limit OFFSET :offset
            """
            )

            rows = conn.execute(query, {"limit": limit, "offset": offset}).fetchall()

            if not rows:
                logger.info(
                    "No bars with regime data found, fetching any available bars for context"
                )
                query_any = text(
                    f"""
                SELECT datetime, open, high, low, close, volume
                FROM {table_name}
                ORDER BY datetime DESC
                LIMIT :limit OFFSET :offset
                """
                )
                rows = conn.execute(
                    query_any, {"limit": limit, "offset": offset}
                ).fetchall()

                if not rows:
                    logger.info(f"No historical bars found in {table_name}")
                    return pd.DataFrame(
                        columns=["open", "high", "low", "close", "volume"]
                    )

            # Create DataFrame with datetime index
            df = pd.DataFrame(
                rows, columns=["datetime", "open", "high", "low", "close", "volume"]
            )
            df.set_index("datetime", inplace=True)

            # Reverse to chronological order
            df = df.iloc[::-1]

            # Convert to float64
            df = df.astype("float64")

            logger.info(f"Retrieved {len(df)} historical bars from {table_name}")
            return df

    except SQLAlchemyError as exc:
        logger.exception("Failed to get historical bars from %s", table_name)
        raise QueryError(f"Failed to get historical bars from {table_name}") from exc


def get_bars_without_regime(table_name: str) -> pd.DataFrame:
    """Get bars that don't have regime information.

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.
        table_name: Name of the bars table.

    Returns:
        DataFrame with columns: timestamp (index), open, high, low, close, volume.
        Empty DataFrame if all bars have regimes.

    Raises:
        QueryError: If database query fails.
    """
    engine = get_engine()

    if not table_name or "." not in table_name:
        raise ValueError("table_name must be fully qualified (schema.table)")

    try:
        with engine.connect() as conn:
            query = text(
                f"""
            SELECT datetime, open, high, low, close, volume
            FROM {table_name}
            WHERE primary_regime IS NULL OR regime_confidence IS NULL
            ORDER BY datetime ASC
            """
            )

            rows = conn.execute(query).fetchall()

            if not rows:
                logger.info(f"No bars missing regime data in {table_name}")
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

            # Create DataFrame with datetime index
            df = pd.DataFrame(
                rows, columns=["datetime", "open", "high", "low", "close", "volume"]
            )
            df.set_index("datetime", inplace=True)

            # Convert to float64
            df = df.astype("float64")

            logger.info(f"Found {len(df)} bars without regime data in {table_name}")
            return df

    except SQLAlchemyError as exc:
        logger.exception("Failed to get bars without regime from %s", table_name)
        raise QueryError(
            f"Failed to get bars without regime from {table_name}"
        ) from exc

def update_bars_with_regime(table_name: str, bars_df: pd.DataFrame) -> int:
    """
    Bulk update OHLCV bars with market regime metadata using a temporary table.

    This implementation is optimized for PostgreSQL and large datasets.
    It uses a TEMP TABLE + single UPDATE ... FROM statement, which is
    significantly faster than row-wise or batched updates.

    Args:
        table_name: Name of the target bars table.
        bars_df: DataFrame indexed by timestamp and containing regime columns.

    Returns:
        Number of rows updated in the target table.

    Raises:
        QueryError: If the database update fails.
    """
    if bars_df.empty:
        logger.info("No bars to update with regime data")
        return 0

    engine = get_engine()

    # Columns expected in the temp table and DataFrame
    temp_columns: Final[list[str]] = [
        "datetime",
        "regime_trend",
        "regime_volatility",
        "regime_momentum",
        "regime_label",
        "regime_confidence",
        "trend_strength_z",
        "vol_percentile",
        "volatility_skew",
        "transition_pressure",
        "trend_acceleration",
        "adaptive_alpha",
        "up_vol",
        "down_vol",
        "regime_stability",
        "directional_persistence",
        "score_bull",
        "score_bear",
        "score_range",
        "score_transition",
        "score_high_vol",
        "score_low_vol",
        "score_accelerating",
    ]

    try:
        logger.info(
            "Starting temp-table bulk update for %d bars", len(bars_df)
        )

        # Prepare DataFrame for COPY
        update_df = bars_df.copy()
        update_df = update_df.reset_index(names="datetime")
        update_df = update_df[temp_columns]

        with engine.begin() as conn:
            # 1. Create temp table (ON COMMIT DROP cleans it automatically)
            conn.execute(
                text(
                    """
                    CREATE TEMP TABLE tmp_bar_regime (
                        datetime TIMESTAMPTZ PRIMARY KEY,
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
                        score_accelerating DOUBLE PRECISION
                    ) ON COMMIT DROP
                    """
                )
            )

            # 2. Bulk insert via COPY-equivalent (fastest path in SQLAlchemy)
            update_df.to_sql(
                name="tmp_bar_regime",
                con=conn,
                if_exists="append",
                index=False,
                method="multi",
            )

            # 3. Single set-based UPDATE
            result = conn.execute(
                text(
                    f"""
                    UPDATE {table_name} AS b
                    SET
                        regime_trend = t.regime_trend,
                        regime_volatility = t.regime_volatility,
                        regime_momentum = t.regime_momentum,
                        regime_label = t.regime_label,
                        regime_confidence = t.regime_confidence,
                        trend_strength_z = t.trend_strength_z,
                        vol_percentile = t.vol_percentile,
                        volatility_skew = t.volatility_skew,
                        transition_pressure = t.transition_pressure,
                        trend_acceleration = t.trend_acceleration,
                        adaptive_alpha = t.adaptive_alpha,
                        up_vol = t.up_vol,
                        down_vol = t.down_vol,
                        regime_stability = t.regime_stability,
                        directional_persistence = t.directional_persistence,
                        score_bull = t.score_bull,
                        score_bear = t.score_bear,
                        score_range = t.score_range,
                        score_transition = t.score_transition,
                        score_high_vol = t.score_high_vol,
                        score_low_vol = t.score_low_vol,
                        score_accelerating = t.score_accelerating
                    FROM tmp_bar_regime AS t
                    WHERE b.datetime = t.datetime
                    """
                )
            )

            updated_rows = result.rowcount or 0

        logger.info(
            "Regime update complete: %d rows updated in %s",
            updated_rows,
            table_name,
        )

        return updated_rows

    except SQLAlchemyError as exc:
        logger.exception(
            "Failed to update bars with regime data in %s", table_name
        )
        raise QueryError(
            f"Failed to update bars with regime data in {table_name}"
        ) from exc
    

def get_total_bar_count(table_name: str) -> int:
    """Get total number of bars in table.

    Args:
        engine: SQLAlchemy engine connected to PostgreSQL database.
        table_name: Name of the bars table.

    Returns:
        Total count of bars in table.

    Raises:
        QueryError: If database query fails.
    """
    engine = get_engine()

    try:
        with engine.connect() as conn:
            query = text(f"SELECT COUNT(*) FROM {table_name}")
            result = conn.execute(query)
            count = result.scalar()
            return count or 0

    except SQLAlchemyError as exc:
        logger.exception("Failed to get bar count from %s", table_name)
        raise QueryError(f"Failed to get bar count from {table_name}") from exc


def get_recent_bars_for_regime(table_name: str, limit: int) -> pd.DataFrame:
    """Get recent bars with datetime index for regime context (OHLCV only).

    Args:
        table_name: Fully qualified table name.
        limit: Number of bars to retrieve.

    Returns:
        DataFrame with datetime index and OHLCV columns.
    """
    engine = get_engine()
    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT datetime, open, high, low, close, volume
                FROM {table_name}
                ORDER BY datetime DESC
                LIMIT :limit
            """)
            rows = conn.execute(query, {"limit": limit}).fetchall()
            if not rows:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
            df.set_index("datetime", inplace=True)
            df = df.iloc[::-1].astype("float64")
            return df
    except SQLAlchemyError as exc:
        logger.exception("Failed to get recent bars for regime from %s", table_name)
        raise QueryError(f"Failed to get recent bars for regime from {table_name}") from exc


def batch_insert_bars(table_name: str, bars: List[Dict], bar_type: str, exchange: str, symbol: str) -> int:
    """Insert all bars in a single transaction using executemany.

    Args:
        table_name: Fully qualified table name.
        bars: List of finalized bar dicts.
        bar_type: 'volume', 'volatility', or 'dollar'.
        exchange: Exchange identifier.
        symbol: Symbol identifier.

    Returns:
        Number of bars inserted.
    """
    if not bars:
        return 0

    engine = get_engine()
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        inserted = 0
        for bar in bars:
            try:
                if bar_type == "volume":
                    insert_volume_bar(cursor, table_name, bar, exchange, symbol)
                elif bar_type == "volatility":
                    insert_volatility_bar(cursor, table_name, bar, exchange, symbol)
                elif bar_type == "dollar":
                    insert_dollar_bar(cursor, table_name, bar, exchange, symbol)
                elif bar_type == "range":
                    insert_range_bar(cursor, table_name, bar, exchange, symbol)
                elif bar_type == "renko":
                    insert_renko_bar(cursor, table_name, bar, exchange, symbol)
                elif bar_type == "hybrid":
                    insert_hybrid_bar(cursor, table_name, bar, exchange, symbol)
                else:
                    insert_standard_bar(cursor, table_name, bar, exchange, symbol)
                inserted += 1
            except Exception as e:
                logger.warning("Skipping bar insert error: %s", e)
        raw_conn.commit()
        logger.info("Batch inserted %d/%d bars into %s", inserted, len(bars), table_name)
        return inserted
    except Exception as exc:
        raw_conn.rollback()
        logger.error("Batch insert failed for %s: %s", table_name, exc)
        raise
    finally:
        cursor.close()
        raw_conn.close()


if __name__=="__main__":
    # bar_types=["volume", "volatility", "dollar", "range", "renko", "hybrid"]
    # for bar in bar_types:

    #     bars=read_bar("binance", "btc", bar,start_date="2024-01-01", end_date="2024-12-31")
    #     bars.to_csv(f"1minute_{bar}_bars.csv")
    raw=read_ohlcv("binance", "btc", start_date="2024-01-01", end_date="2024-12-31",timeframe="1m",columns=["datetime", "open", "high", "low", "close", "volume"])  
    raw.to_csv("1minute_ohlcv_bars.csv")  