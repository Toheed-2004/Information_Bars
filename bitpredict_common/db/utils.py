from datetime import datetime, timezone
from typing import Iterable, Optional, List, Dict, Optional, Any, Literal, Union
import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from io import StringIO
import connectorx as cx
import psycopg2
import json
from .exceptions import *
from bitpredict.common.logging import get_logger
import numpy as np
import math
from bitpredict.common.db.config import get_engine
from bitpredict.common.constants import STRATEGIES_SCHEMA, SIMULATOR_SCHEMA


logger = get_logger(__name__)
"""
Unified Database Read/Write Operations

This module provides simplified, unified interfaces for reading and writing
DataFrames to PostgreSQL. Each function supports multiple methods and 
automatically selects the fastest method by default.

Key Features:
- Single function for reading: read_df()
- Single function for writing: insert_df()
- Automatic method selection (defaults to fastest)
- Consistent API across all methods
- Proper error handling and validation

Performance Guide:
- Small tables (<10K rows): Any method works fine
- Medium tables (10K-1M rows): Use "copy" method
- Large tables (>1M rows): Use "connectorx" for reads, "copy" for writes
"""


# =============================================================================
# READ OPERATIONS
# =============================================================================


def read_df(
    engine: Engine,
    schema: str,
    table_name: str,
    columns: Optional[Iterable[str]] = None,
    where: Optional[str] = None,
    method: Optional[str] = "copy",
    conn_str: Optional[str] = None,
    order_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Unified interface for reading DataFrames from PostgreSQL.

    Supports multiple read methods and automatically selects the fastest one
    by default. This function replaces all individual read functions.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy Engine connected to PostgreSQL.
    table_name : str
        Table name to read from.
    schema : str
        Schema where the table resides.
    columns : Optional[Iterable[str]], default None
        Specific columns to read. If None, reads all columns.
    where : Optional[str], default None
        WHERE clause for filtering (without the 'WHERE' keyword).
        Example: "timestamp > '2024-01-01'"
    method : Optional[str], default "sql"
        Read method to use. One of:
        - "sql": Standard pandas.read_sql() - slowest, most compatible
        - "copy": PostgreSQL COPY command - very fast for large tables
        - "connectorx": ConnectorX library - fastest for large datasets
    conn_str : Optional[str], default None
        PostgreSQL connection string. Required only for "connectorx" method.
        Format: "postgresql://user:password@host:port/database"

    Returns
    -------
    pd.DataFrame
        DataFrame containing the query result.

    Raises
    ------
    DataLoadError
        If validation fails or read operation fails.
    ValueError
        If invalid method specified or missing required parameters.

    Examples
    --------
    # Basic usage - uses SQL method by default
    >>> df = read_df(engine, "trades", "data_tick")

    # Read specific columns with filter
    >>> df = read_df(
    ...     engine,
    ...     "trades",
    ...     "data_tick",
    ...     columns=["timestamp", "price", "volume"],
    ...     where="timestamp > '2024-01-01'"
    ... )

    # Use COPY for faster reads of large tables
    >>> df = read_df(engine, "trades", "data_tick", method="copy")

    # Use connectorx for maximum speed on very large tables
    >>> df = read_df(
    ...     engine,
    ...     "trades",
    ...     "data_tick",
    ...     method="connectorx",
    ...     conn_str="postgresql://user:pass@localhost:5432/mydb"
    ... )

    Performance Guide
    -----------------
    - **Small tables (<10K rows)**: Any method works fine, use "sql"
    - **Medium tables (10K-1M rows)**: Use "copy"
    - **Large tables (>1M rows)**: Use "connectorx" for maximum speed
    - **With WHERE clause**: Use "sql" (COPY doesn't support WHERE)

    Notes
    -----
    - Default method is "sql" for maximum compatibility
    - ConnectorX requires separate installation: pip install connectorx
    - COPY method doesn't support WHERE clauses - falls back to SQL
    - SQL method is slowest but most flexible
    """

    # Input Validation
    if not table_name or not schema:
        raise ValueError("table_name and schema must be non-empty strings")
    
    if not ensure_table(engine, schema, table_name):
        logger.debug(f"Table {schema}.{table_name} does not exist, returning empty DataFrame")
        return pd.DataFrame()

    if method and method not in ("sql", "copy", "connectorx"):
        raise ValueError(
            f"Invalid method '{method}'. Must be one of: sql, copy, connectorx"
        )

    # ConnectorX Method
    if method == "connectorx":
        if not conn_str:
            # render_as_string(hide_password=False) gives the full URL including password
            raw_url = engine.url.render_as_string(hide_password=False)
            conn_str = raw_url.replace("postgresql+psycopg2://", "postgresql://") \
                               .replace("postgresql+psycopg://", "postgresql://")

        # ConnectorX requires manual query construction
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {schema}.{table_name}"
        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"

        return read_df_connectorx(conn_str, query)

    # COPY Method - fall back to SQL if WHERE clause provided
    elif method == "copy":
        return read_df_copy(engine, table_name, schema, columns, where, order_by)
    else:
    # SQL Method (default fallback)
        return read_df_sql(engine, table_name, schema, columns, where, order_by)


def read_df_sql(
    engine: Engine,
    table_name: str,
    schema: str,
    columns: Optional[Iterable[str]] = None,
    where: Optional[str] = None,
    order_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load data from PostgreSQL into a pandas DataFrame using standard SQL.

    This is the most compatible method but slowest for large datasets.
    Supports all SQL features including WHERE clauses, joins, etc.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy Engine connected to PostgreSQL.
    table_name : str
        Table name to read from.
    schema : str
        Schema where the table resides.
    columns : Optional[Iterable[str]], default None
        Specific columns to read. If None, reads all columns.
    where : Optional[str], default None
        WHERE clause for filtering (without the 'WHERE' keyword).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the query result.

    Raises
    ------
    DataLoadError
        If validation fails or read operation fails.
    """

    # Validate required parameters
    if not table_name or not schema:
        raise DataLoadError("Table name and schema must be provided")

    # Prepare SELECT statement
    cols = ", ".join(columns) if columns else "*"
    query = f"SELECT {cols} FROM {schema}.{table_name}"

    if where:
        query += f" WHERE {where}"
    if order_by:
        query += f" ORDER BY {order_by}"

    # Execute query and load into pandas DataFrame
    try:
        return pd.read_sql(query, engine)
    except SQLAlchemyError as exc:
        raise DataLoadError(f"Failed to load data from {schema}.{table_name}") from exc


def read_df_connectorx(conn_str: str, query: str) -> pd.DataFrame:
    """
    Read a DataFrame from PostgreSQL using ConnectorX.

    ConnectorX is extremely fast for large datasets (5-10x faster than pandas)
    and directly returns a pandas DataFrame with optimal memory usage.

    Parameters
    ----------
    conn_str : str
        PostgreSQL connection string in the form:
        "postgresql://user:password@host:port/database"
    query : str
        SQL query to execute.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the query result.

    Raises
    ------
    DataReadError
        If the query execution fails.

    Notes
    -----
    - Requires connectorx package: pip install connectorx
    - Best for reading large tables (>1M rows)
    - Does not support all PostgreSQL data types
    """

    try:
        df = cx.read_sql(conn_str, query)
        return df
    except Exception as exc:
        raise DataReadError(f"Failed to read data via connectorx: {exc}") from exc


def read_df_copy(
    engine: Engine,
    table_name: str,
    schema: str = "public",
    columns: Optional[List[str]] = None,
    where: Optional[str] = None,
    order_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read a PostgreSQL table into a DataFrame using COPY TO STDOUT.

    This method is very fast for large tables (3-5x faster than pandas.read_sql).
    It uses PostgreSQL's native COPY command to stream data efficiently.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy Engine connected to PostgreSQL.
    table_name : str
        Table name to read.
    schema : str, default 'public'
        Schema where the table resides.
    columns : Optional[List[str]], default None
        List of columns to read. If None, reads all columns.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the table data.

    Raises
    ------
    DataReadError
        If the COPY operation or parsing fails.

    Notes
    -----
    - Does not support WHERE clauses - use read_df_sql() instead
    - Best for reading entire tables or specific columns
    - Faster than pandas.read_sql() for tables >10K rows
    """

    qualified_table = f"{schema}.{table_name}"

    # Build COPY command
    # Use SELECT query wrapped in COPY to support TimescaleDB hypertables
    # Direct COPY table_name doesn't work with hypertables
    cols_sql = "*" if columns is None else ", ".join(columns)

    where_sql = f"WHERE {where}" if where else ""
    order_sql = f"ORDER BY {order_by}" if order_by else ""

    copy_sql = f"COPY (SELECT {cols_sql} FROM {qualified_table} {where_sql} {order_sql}) TO STDOUT WITH CSV HEADER"

    try:
        raw_conn = engine.raw_connection()
        cursor = raw_conn.cursor()
        cursor.execute(f"SET search_path TO {schema}")

        # Stream data to StringIO buffer
        output = StringIO()
        cursor.copy_expert(copy_sql, output)
        output.seek(0)

        # Parse CSV into DataFrame
        df = pd.read_csv(output)
        return df

    except (psycopg2.Error, SQLAlchemyError, IOError) as exc:
        raise DataReadError(
            f"Failed to read table {qualified_table} using COPY: {exc}"
        ) from exc
    finally:
        try:
            cursor.close()
            raw_conn.close()
        except Exception:
            pass


# =============================================================================
# WRITE OPERATIONS
# =============================================================================


def insert_df(
    df: pd.DataFrame,
    engine: Engine,
    schema_name: str,
    table_name: str,
    if_exists: str = "append",
    index: bool = False,
    method: Optional[str] = "copy",
    chunksize: int = 10_000,
    is_timeseries: bool = False,
) -> None:
    """
    Unified interface for writing DataFrames to PostgreSQL.

    Supports multiple write methods and automatically selects the fastest one
    by default. This function replaces all individual write functions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write. If empty or None, function returns silently.
    engine : Engine
        SQLAlchemy Engine connected to PostgreSQL.
    table_name : str
        Target table name.
    schema_name : str
        Target schema name.
    if_exists : str, default "append"
        Behavior when table exists. One of:
        - "fail": Raise error if table exists
        - "replace": Drop table and recreate
        - "append": Insert into existing table (default)
    index : bool, default False
        Whether to write DataFrame index as a column.
    method : Optional[str], default "copy"
        Write method to use. One of:
        - "sql": Standard pandas.to_sql() - slowest, most compatible
        - "copy": PostgreSQL COPY command - 5-10x faster than SQL
        - "executemany": Batch inserts - faster than SQL, slower than COPY
    chunksize : int, default 10_000
        Number of rows per batch. Used by "sql" and "executemany" methods.
        Ignored by "copy" method.
    is_timeseries : bool, default False
        If True, converts table to TimescaleDB hypertable using 'timestamp' column.
        Requires TimescaleDB extension enabled.

    Raises
    ------
    DataSaveError
        If validation fails or write operation fails.
    ValueError
        If invalid method or if_exists value specified.

    Examples
    --------
    # Basic usage - uses COPY method by default (fastest)
    >>> insert_df(df, engine, "trades", "data_tick")

    # Replace existing table
    >>> insert_df(df, engine, "trades", "data_tick", if_exists="replace")

    # Create TimescaleDB hypertable
    >>> insert_df(
    ...     df,
    ...     engine,
    ...     "trades",
    ...     "data_tick",
    ...     is_timeseries=True
    ... )

    # Force specific method for compatibility
    >>> insert_df(df, engine, "trades", "data_tick", method="sql")

    # Write with index column
    >>> insert_df(df, engine, "trades", "data_tick", index=True)

    Performance Comparison
    ----------------------
    For 100K rows with 10 columns:
    - **copy**: ~0.5 seconds (FASTEST - default)
    - **executemany**: ~2-3 seconds (batch inserts)
    - **sql**: ~8-10 seconds (slowest, but most compatible)

    Method Selection Guide
    ----------------------
    - **copy** (default): Best for most use cases, 5-10x faster than SQL
    - **sql**: Use for maximum compatibility or small datasets (<1K rows)
    - **executemany**: Middle ground, use if COPY has issues

    Notes
    -----
    - Default method is "copy" for maximum performance
    - All methods support schema creation, table creation, and replacement
    - TimescaleDB conversion happens after data insert
    - Empty DataFrames are silently ignored
    """

    # Early exit for empty DataFrames
    if df is None or df.empty:
        return

    # Input Validation
    if not table_name or not schema_name:
        raise ValueError("table_name and schema must be non-empty strings")

    if if_exists not in ("fail", "replace", "append"):
        raise ValueError(
            f"Invalid if_exists '{if_exists}'. Must be one of: fail, replace, append"
        )

    if method and method not in ("sql", "copy", "executemany"):
        raise ValueError(
            f"Invalid method '{method}'. Must be one of: sql, copy, executemany"
        )

    # Automatic Method Selection - default to COPY for speed
    if method is None:
        method = "copy"

    # Route to appropriate function
    if method == "copy":
        insert_df_copy(
            df=df,
            engine=engine,
            table_name=table_name,
            schema_name=schema_name,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
            is_timeseries=is_timeseries,
        )
    elif method == "executemany":
        insert_df_executemany(
            df=df,
            engine=engine,
            table_name=table_name,
            schema_name=schema_name,
            if_exists=if_exists,
            index=index,
            chunk_size=chunksize,
            is_timeseries=is_timeseries,
        )
    elif method == "sql": 
        insert_df_sql(
            df=df,
            engine=engine,
            table_name=table_name,
            schema_name=schema_name,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
            is_timeseries=is_timeseries,
        )


def insert_df_sql(
    df: pd.DataFrame,
    engine: Engine,
    schema_name: str,
    table_name: str,
    if_exists: str = "append",
    index: bool = False,
    chunksize: int = 10_000,
    is_timeseries: bool = False,
) -> None:
    """
    Persist a pandas DataFrame into PostgreSQL using pandas.to_sql().

    This is the simplest and most compatible method for inserting data.
    It is reliable but slower for large datasets compared to COPY or executemany.
    Handles schema management and table creation automatically.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist. If empty or None, the function returns silently.
    engine : Engine
        SQLAlchemy Engine connected to PostgreSQL.
    table_name : str
        Target table name.
    schema : str
        Target schema name.
    if_exists : str, default "append"
        Behavior when the table already exists. One of {"fail", "replace", "append"}.
    index : bool, default False
        Whether to write the DataFrame index as a column.
    chunksize : int, default 10_000
        Number of rows to write at a time. Useful for large datasets.
    is_timeseries : bool, default False
        If True, converts table to TimescaleDB hypertable.

    Raises
    ------
    DataSaveError
        If validation fails or database operations fail.

    Notes
    -----
    - Slowest method but most compatible
    - Automatically creates table structure
    - Best for small datasets (<10K rows) or when compatibility is critical
    """

    if df is None or df.empty:
        return

    if not table_name or not schema_name:
        raise DataSaveError("Table name and schema must be provided")

    # Ensure schema exists
    ensure_schema(engine, schema_name)

    # Prepare DataFrame
    df_to_write = df.copy()
    if index:
        df_to_write = df_to_write.reset_index()

    qualified_table = f"{schema_name}.{table_name}"

    try:
        # Write using pandas.to_sql with multi-row inserts
        df_to_write.to_sql(
            name=table_name,
            con=engine,
            schema=schema_name,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
            method="multi",  # Use multi-row INSERT for efficiency
        )

        # Convert to hypertable if requested
        if is_timeseries:
            create_hypertable(
                engine=engine,
                schema_name=schema_name,
                table_name=table_name,
                time_column="datetime",
            )

    except (IntegrityError, OperationalError, SQLAlchemyError) as exc:
        raise DataSaveError(
            f"Failed to insert data to {qualified_table} using to_sql: {exc}"
        ) from exc


def insert_df_executemany(
    df: pd.DataFrame,
    engine: Engine,
    schema_name: str,
    table_name: str,
    if_exists: str = "append",

    conflict_columns: list[str] = ['datetime','exchange' ,'symbol', 'timeframe'],
    index: bool = False,
    chunk_size: int = 1000,
    is_timeseries: bool = False,
) -> None:
    """
    Upsert a pandas DataFrame into PostgreSQL using cursor.executemany() with ON CONFLICT DO UPDATE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to upsert. Must include all columns to insert/update.
    engine : Engine
        SQLAlchemy Engine connected to PostgreSQL.
    schema_name : str
        Target schema name.
    table_name : str
        Target table name.
    conflict_columns : list[str]
        Columns used to detect conflict (typically primary keys like ['datetime', 'symbol']).
    chunk_size : int, default 1000
        Number of rows per batch.
    is_timeseries : bool, default False
        If True, converts table to TimescaleDB hypertable.

    Raises
    ------
    DataSaveError
        If validation fails or database operations fail.
    """
    if df is None or df.empty:
        return

    if not table_name or not schema_name:
        raise DataSaveError("Table name and schema must be provided")

    if not conflict_columns:
        raise DataSaveError("At least one conflict column must be specified for upsert")

    # Ensure schema exists
    ensure_schema(engine, schema_name)

    df_to_write = df.copy()
    qualified_table = f"{schema_name}.{table_name}"
    columns = list(df_to_write.columns)

    try:
        # Check if table exists, create if not
        table_exists = ensure_table(engine, schema_name, table_name)

        # Handle table existence based on if_exists parameter
        if table_exists:
            if if_exists == "fail":
                raise DataSaveError(
                    f"Table {qualified_table} already exists and if_exists='fail'"
                )
            elif if_exists == "replace":
                with engine.begin() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {qualified_table}"))
                table_exists = False
        if not table_exists:
            with engine.begin() as conn:
                df_to_write.head(0).to_sql(
                    name=table_name,
                    con=conn,
                    schema=schema_name,
                    if_exists="replace",
                    index=False,
                )
            if is_timeseries:
                create_hypertable(
                    engine=engine,
                    schema_name=schema_name,
                    table_name=table_name,
                    time_column="datetime",
                )

        # Prepare parameterized INSERT ... ON CONFLICT ... DO UPDATE
        col_str = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))

        # Generate update clause, exclude conflict columns
        update_cols = [col for col in columns if col not in conflict_columns]
        update_str = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_cols)

        upsert_sql = (
            f"INSERT INTO {qualified_table} ({col_str}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(conflict_columns)}) DO UPDATE "
            f"SET {update_str}"
        )

        # Execute in chunks
        raw_conn = engine.raw_connection()
        try:
            cursor = raw_conn.cursor()
            for start in range(0, len(df_to_write), chunk_size):
                chunk = df_to_write.iloc[start : start + chunk_size]
                records = [tuple(row) for row in chunk.values]
                cursor.executemany(upsert_sql, records)
            raw_conn.commit()
        except Exception as e:
            raw_conn.rollback()
            raise DataSaveError(f"Failed to upsert data to {qualified_table}: {e}") from e
        finally:
            raw_conn.close()

    except (IntegrityError, OperationalError, SQLAlchemyError) as exc:
        raise DataSaveError(f"Failed to upsert data to {qualified_table}: {exc}") from exc

def insert_df_copy(
    df: pd.DataFrame,
    engine: Engine,
    schema_name: str,
    table_name: str,
    if_exists: str = "append",
    index: bool = False,
    chunksize: int = 50_000,
    is_timeseries: bool = False,
) -> None:
    """
    Insert a pandas DataFrame into PostgreSQL using COPY FROM STDIN.

    This is the fastest method for inserting large amounts of data (5-10x faster
    than to_sql). Uses PostgreSQL's native COPY command for optimal performance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist. If empty or None, the function returns silently.
    engine : Engine
        SQLAlchemy Engine connected to PostgreSQL.
    table_name : str
        Target table name.
    schema : str
        Target schema name.
    if_exists : str, default "append"
        Behavior when the table already exists. One of {"fail", "replace", "append"}.
    index : bool, default False
        Whether to write the DataFrame index as a column.
    chunksize : int, default 10_000
        Not used by COPY method (included for API consistency).
    is_timeseries : bool, default False
        If True, converts table to TimescaleDB hypertable.

    Raises
    ------
    DataSaveError
        If validation fails or database operations fail.

    Notes
    -----
    - Fastest method for bulk inserts (5-10x faster than to_sql)
    - Best for large datasets (>10K rows)
    - Requires table to exist or will create it
    - Sets search_path to target schema only for proper resolution
    """

    if df is None or df.empty:
        return

    if not table_name or not schema_name:
        raise DataSaveError("Table name and schema must be provided")

    # Ensure schema exists
    ensure_schema(engine, schema_name)

    # Prepare DataFrame
    df_to_write = df.copy()
    if index:
        df_to_write = df_to_write.reset_index()

    try:
        qualified_table = f"{schema_name}.{table_name}"
        table_exists = ensure_table(engine, schema_name, table_name)

        # Handle table existence based on if_exists parameter
        if table_exists:
            if if_exists == "fail":
                raise DataSaveError(
                    f"Table {qualified_table} already exists and if_exists='fail'"
                )
            elif if_exists == "replace":
                with engine.begin() as conn:
                    conn.execute(
                        text(f"DROP TABLE IF EXISTS {qualified_table} CASCADE")
                    )
                table_exists = False

        # Create table structure if it doesn't exist
        if not table_exists:
            with engine.begin() as conn:
                df_to_write.head(0).to_sql(
                    name=table_name,
                    con=conn,
                    schema=schema_name,
                    if_exists="replace",
                    index=False,
                )

            # Convert to hypertable if requested
            if is_timeseries:
                create_hypertable(
                    engine=engine,
                    schema_name=schema_name,
                    table_name=table_name,
                    time_column="datetime",
                    compress=True,

                )

        # Execute COPY command
        raw_conn = None
        cursor = None
        try:
            raw_conn = engine.raw_connection()
            cursor = raw_conn.cursor()

            # Set search_path to target schema only (not public)
            cursor.execute(f"SET search_path TO {schema_name}")

            for start in range(0, len(df), chunksize):
                chunk = df.iloc[start : start + chunksize]
                if index:
                    chunk = chunk.reset_index(drop=False)
                # Stream to COPY
                output = StringIO()
                chunk.to_csv(
                    output, sep="\t", header=False, index=False, na_rep="\\N", quoting=3
                )
                output.seek(0)
                cursor.copy_from(
                    output,
                    table_name,
                    sep="\t",
                    null="\\N",
                    columns=list(chunk.columns),
                )

            raw_conn.commit()

        except Exception as e:
            if raw_conn:
                raw_conn.rollback()
            raise DataSaveError(
                f"Failed to copy data to {qualified_table}: {str(e)}"
            ) from e
        finally:
            if cursor:
                cursor.close()
            if raw_conn:
                raw_conn.close()

    except IntegrityError as exc:
        raise DataSaveError(
            f"Integrity constraint violated while saving to {qualified_table}"
        ) from exc
    except OperationalError as exc:
        raise DataSaveError(
            f"Operational error while saving to {qualified_table}"
        ) from exc
    except DataSaveError:
        raise
    except SQLAlchemyError as exc:
        raise DataSaveError(
            f"Unexpected error while saving to {qualified_table}"
        ) from exc


# =============================================================================
# DICTIONARY INSERT OPERATION
# =============================================================================


def insert_dict(
    data: List[Dict],
    engine: Engine,
    schema_name: str,
    table_name: str,
    is_timeseries: bool = False,
    time_col: str = "timestamp",
) -> None:
    """
    Fast bulk insert of dictionary data using PostgreSQL COPY command.

    Uses a single raw connection for both table creation and COPY.
    Automatically detects column types and creates table if needed.
    Handles JSONB columns for nested data structures.

    Parameters
    ----------
    data : List[Dict]
        List of dictionaries to insert. All dicts should have same keys.
    engine : Engine
        SQLAlchemy Engine connected to PostgreSQL.
    table_name : str
        Target table name.
    schema : str
        Target schema name.
    is_timeseries : bool, default False
        If True, converts table to TimescaleDB hypertable.
    time_col : str, default "timestamp"
        Timestamp column name for hypertable partitioning.

    Raises
    ------
    Exception
        If COPY operation fails or table creation fails.

    Notes
    -----
    - Automatically creates table based on first dictionary
    - Detects JSONB columns for nested data (lists/dicts)
    - Adds inserted_at column automatically
    - Very fast for bulk inserts of structured data
    - Best for inserting API responses or JSON data

    Examples
    --------
    >>> data = [
    ...     {"timestamp": "2024-01-01", "price": 100.5, "volume": 1000},
    ...     {"timestamp": "2024-01-02", "price": 101.2, "volume": 1500}
    ... ]
    >>> insert_dict(data, engine, "trades", "data_tick")
    """

    if not data:
        return

    qualified_table = f"{schema_name}.{table_name}"
    raw_conn = engine.raw_connection()

    try:
        cursor = raw_conn.cursor()

        # Set search_path for this connection
        cursor.execute(f"SET search_path TO {schema_name}, public")

        # Ensure schema exists
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

        # Check if table exists
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            )
        """,
            (schema_name, table_name),
        )

        table_exists_flag = cursor.fetchone()[0]

        # Create table if it doesn't exist
        if not table_exists_flag:
            # Build CREATE TABLE statement from sample data
            columns_sql = []

            for key, value in data[0].items():
                if value is None:
                    col_type = "VARCHAR(255)"
                elif isinstance(value, bool):
                    col_type = "BOOLEAN"
                elif isinstance(value, int):
                    col_type = "BIGINT"
                elif isinstance(value, float):
                    col_type = "DOUBLE PRECISION"
                elif isinstance(value, str):
                    col_type = "VARCHAR(255)"
                elif isinstance(value, (list, dict)):
                    col_type = "JSONB"
                else:
                    col_type = "VARCHAR(255)"

                columns_sql.append(f'"{key}" {col_type}')

            # Add inserted_at column
            columns_sql.append('"inserted_at" TIMESTAMP DEFAULT NOW()')

            # Execute CREATE TABLE
            create_sql = f"""
                CREATE TABLE {qualified_table} (
                    {', '.join(columns_sql)}
                )
            """
            cursor.execute(create_sql)

            # Convert to hypertable if needed
            if is_timeseries:
                create_hypertable(
                    schema_name=schema_name,
                    table_name=table_name,
                    time_column=time_col,
                    cursor=cursor,
                )
            raw_conn.commit()

        # Prepare data for COPY
        columns = list(data[0].keys())
        jsonb_columns = set()

        # Detect which columns contain list/dict data
        for col in columns:
            if isinstance(data[0].get(col), (list, dict)):
                jsonb_columns.add(col)

        # Build TSV data manually
        lines = []
        for record in data:
            row_parts = []
            for col in columns:
                value = record.get(col)
                if value is None:
                    row_parts.append(r"\N")
                elif col in jsonb_columns:
                    # Dump JSON directly for JSONB columns
                    row_parts.append(json.dumps(value))
                else:
                    # Escape special characters for text columns
                    str_val = (
                        str(value)
                        .replace("\\", "\\\\")
                        .replace("\t", "\\t")
                        .replace("\n", "\\n")
                        .replace("\r", "\\r")
                    )
                    row_parts.append(str_val)
            lines.append("\t".join(row_parts))

        # Join all lines
        tsv_data = "\n".join(lines)
        output = StringIO(tsv_data)

        # Execute COPY command
        copy_sql = f"""
            COPY {qualified_table} ({', '.join(f'"{col}"' for col in columns)})
            FROM STDIN WITH (FORMAT text, DELIMITER E'\\t', NULL '\\N')
        """

        cursor.copy_expert(copy_sql, output)
        raw_conn.commit()

    except Exception as e:
        raw_conn.rollback()
        raise
    finally:
        raw_conn.close()





# =============================================================================
# HYPERTABLE MANAGEMENT
# =============================================================================

def create_hypertable(
    *,
    schema_name: str,
    table_name: str,
    time_column: str = "timestamp",
    engine: Engine | None = None,
    raw_conn=None,
    cursor=None,
    compress: bool = False,
    compress_segmentby: str | None = None,  # e.g., "exchange,symbol,timeframe"
    index_time_desc: bool = True,
) -> None:
    """
    Create a TimescaleDB hypertable with optional compression and segmentation.

    Parameters
    ----------
    schema_name : str
        Schema name of the table.
    table_name : str
        Table name (without schema).
    time_column : str
        Name of the timestamp column.
    engine : Engine, optional
        SQLAlchemy engine.
    raw_conn : optional
        psycopg2 raw connection.
    cursor : optional
        psycopg2 cursor.
    compress : bool
        If True, enable compression.
    compress_segmentby : str, optional
        Column(s) for segmenting compressed chunks (comma-separated).
    index_time_desc : bool
        If True, add index on time column DESC.
    """
    qualified_table = f"{schema_name}.{table_name}"

    try:
        def _create_internal(cur):
            # Set search path
            cur.execute(f"SET search_path TO {schema_name}, public")

            # Create hypertable
            cur.execute(
                "SELECT create_hypertable(%s, %s, if_not_exists => TRUE)",
                (qualified_table, time_column),
            )

            # Optional index
            if index_time_desc:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{time_column}_desc "
                    f"ON {qualified_table} ({time_column} DESC)"
                )

            # Optional compression
            if compress:
                if compress_segmentby:
                    cur.execute(
                        f"ALTER TABLE {qualified_table} SET ("
                        f"timescaledb.compress, "
                        f"timescaledb.compress_segmentby = '{compress_segmentby}', "
                        f"timescaledb.compress_orderby = '{time_column} DESC')"
                    )
                else:
                    cur.execute(
                        f"ALTER TABLE {qualified_table} SET ("
                        f"timescaledb.compress, "
                        f"timescaledb.compress_orderby = '{time_column} DESC')"
                    )
                cur.execute(
                    f"SELECT add_compression_policy('{qualified_table}', INTERVAL '30 days')"
                )                
            # Reset search path
            cur.execute(f"SET search_path TO {schema_name}")

        # Cursor case
        if cursor is not None:
            _create_internal(cursor)
            return

        # Raw connection case
        if raw_conn is not None:
            cur = raw_conn.cursor()
            try:
                _create_internal(cur)
                raw_conn.commit()
            finally:
                cur.close()
            return

        # Engine case (maintain old behavior)
        if engine is not None:
            with engine.begin() as conn:
                # Timescale functions live in public
                conn.execute(text(f"SET search_path TO {schema_name}, public"))

                # Create hypertable using parameterized SQL
                conn.execute(
                    text(
                        """
                        SELECT create_hypertable(
                            :table,
                            :time_col,
                            if_not_exists => TRUE
                        )
                        """
                    ),
                    {"table": qualified_table, "time_col": time_column},
                )

                # Optional index
                if index_time_desc:
                    conn.execute(
                        text(
                            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{time_column}_desc "
                            f"ON {qualified_table} ({time_column} DESC)"
                        )
                    )

                # Optional compression
                if compress:
                    if compress_segmentby:
                        conn.execute(
                            text(
                                f"ALTER TABLE {qualified_table} SET ("
                                f"timescaledb.compress, "
                                f"timescaledb.compress_segmentby = '{compress_segmentby}', "
                                # f"timescaledb.compress_orderby = '{time_column} ASC')"
                                f"timescaledb.compress_orderby = '{time_column} DESC')"
                            )
                        )
                    else:
                        conn.execute(
                            text(
                                f"ALTER TABLE {qualified_table} SET ("
                                f"timescaledb.compress, "
                                # f"timescaledb.compress_orderby = '{time_column} ASC')"
                                f"timescaledb.compress_orderby = '{time_column} DESC')"
                            )
                        )
                    conn.execute(
                        text(f"SELECT add_compression_policy('{qualified_table}', INTERVAL '30 days')")
                    )           
                # Reset search path
                conn.execute(text(f"SET search_path TO {schema_name}"))

            return

        raise ValueError("Must provide engine, raw_conn, or cursor")

    except SQLAlchemyError as exc:
        raise DataSaveError(f"Failed to create hypertable {qualified_table}") from exc
    
def hypertable_exists(engine: Engine, schema_name: str, table_name: str) -> bool:
    """
    Check if a table is a TimescaleDB hypertable.
    
    Args:
        engine: SQLAlchemy engine
        schema_name: Database schema name
        table_name: Table name
    
    Returns:
        bool: True if table is a hypertable, False otherwise
    """
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = :schema
                    AND hypertable_name = :table
                )
            """),
            {"schema": schema_name, "table": table_name}
        )
        return result.scalar()
    
# =============================================================================
# UTILITY MANAGEMENT
# =============================================================================

def get_last_timestamp(engine, schema_name, table_name):
    """
    Retrieve the latest timestamp from the exchange's symbol table.

    Args:
        engine: SQLAlchemy engine instance
        schema_name: Name of the database schema
        table_name: Name of the table to query

    Returns:
        Latest timestamp from the table as datetime, or None if table doesn't exist or is empty
    """
    try:
        ensure_schema(engine, schema_name)

        if not ensure_table(engine, schema_name, table_name):
            return None

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    f"SELECT MAX(timestamp) AS last_date " f"FROM {schema_name}.{table_name}"
                )
            )
            last_date = result.scalar()
            return last_date

    except (SchemaError, TableError) as e:
        raise

    except SQLAlchemyError as e:
        return None


# =============================================================================
# SCHEMA METHODS
# =============================================================================


def ensure_schema(engine: Engine, schema_name: str) -> None:
    if not schema_name:
        raise SchemaError("Schema name cannot be empty")

    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
    except IntegrityError:
        # This catches the "UniqueViolation" race condition.
        # If we are here, another process created the schema 
        # between our check and our execution. We can safely ignore this.
        pass
    except SQLAlchemyError as exc:
        # Still raise for other serious DB issues (connection lost, permissions, etc.)
        raise SchemaError(f"Failed to ensure schema '{schema_name}' exists") from exc


def ensure_schema_and_table(
    engine: Engine,
    schema: str,
    table: str,
    create_if_missing: bool = False,
    create_table_sql: str | None = None,
) -> bool:
    """
    Check whether a PostgreSQL schema and table exist, and optionally create them.
    """

    # ------------------------------------------------------------------------
    # Initialize existence flags
    # ------------------------------------------------------------------------
    schema_exists = False
    table_exists = False

    # ------------------------------------------------------------------------
    # Begin a transaction context; all operations inside are atomic
    # ------------------------------------------------------------------------
    with engine.begin() as conn:

        # --------------------------------------------------------------------
        # Check if the schema exists using information_schema.schemata
        # `SELECT 1` returns a single row if schema exists
        # `.scalar()` retrieves the first column of the first row or None
        # --------------------------------------------------------------------
        schema_check_sql = text(
            """
            SELECT 1
            FROM information_schema.schemata
            WHERE schema_name = :schema
        """
        )
        schema_exists = (
            conn.execute(schema_check_sql, {"schema": schema}).scalar() is not None
        )

        # --------------------------------------------------------------------
        # Optionally create schema if it does not exist
        # `CREATE SCHEMA IF NOT EXISTS` is idempotent
        # --------------------------------------------------------------------
        if not schema_exists and create_if_missing:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            schema_exists = True

        # --------------------------------------------------------------------
        # Check if table exists within the schema
        # Only proceed if schema exists
        # Uses information_schema.tables for safe and portable check
        # --------------------------------------------------------------------
        if schema_exists:
            table_check_sql = text(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = :schema
                  AND table_name = :table
            """
            )
            table_exists = (
                conn.execute(
                    table_check_sql,
                    {"schema": schema, "table": table},
                ).scalar()
                is not None
            )

        # --------------------------------------------------------------------
        # Optionally create table if it does not exist
        # Requires `create_table_sql` parameter to provide full CREATE TABLE statement
        # --------------------------------------------------------------------
        if schema_exists and not table_exists and create_if_missing:
            if not create_table_sql:
                raise ValueError(
                    "create_table_sql must be provided when table creation is enabled"
                )

            # Execute provided SQL to create table
            conn.execute(text(create_table_sql))
            table_exists = True

    # Return True only if both schema and table exist (or were successfully created)
    return schema_exists and table_exists


# =============================================================================
# TABLE METHODS
# =============================================================================


def ensure_table(engine: Engine, schema_name: str, table_name: str, ) -> bool:
    """
    Check whether a table exists in a schema.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy Engine connected to PostgreSQL.
    table_name : str
        Table name to check.
    schema : str
        Schema name.

    Returns
    -------
    bool
        True if the table exists, False otherwise.

    Raises
    ------
    TableError
        If inspection fails.
    """
    try:
        inspector = inspect(engine)
        return table_name in inspector.get_table_names(schema=schema_name)
    except SQLAlchemyError as exc:
        raise TableError(f"Failed to check existence of {schema_name}.{table_name}") from exc


# =============================================================================
# SIMULATION METHODS
# =============================================================================

def clean_for_json(obj):
    """
    Recursively replace NaN, inf, -inf with None
    so PostgreSQL JSONB accepts it.
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    
    elif isinstance(obj, (int, bool, type(None), str)):
        return obj
    
    else:
        # For other types, try to convert to string
        try:
            return str(obj)
        except:
            return None


def downsample_to_avg(data, max_points=50):
    n = len(data)

    if n <= max_points:
        return data

    # Split data into equal chunks and take mean of each chunk
    chunks = np.array_split(np.array(data), max_points)
    return [float(chunk.mean()) for chunk in chunks]


def sanitize_fields(data: Dict[str, Any], numeric_fields: Iterable[str]) -> Dict[str, Any]:

    d = data
    clip_min = -999999.9999
    clip_max = 999999.9999

    for f in numeric_fields:
        v = d.get(f)

        if v is None:
            continue

        try:
            v = float(v)

            if v != v or v == float("inf") or v == float("-inf"):
                d[f] = None
            else:
                if v > clip_max:
                    d[f] = clip_max
                elif v < clip_min:
                    d[f] = clip_min
                else:
                    d[f] = v

        except Exception:
            d[f] = None

    return d

def update_strategy_metadata(strategy_id: str, updates: Dict[str, Any]) -> None:
    """
    Perform a partial update on strategy metadata.
    """
    if not updates:
        return

    engine = get_engine()
    
    # Filter for valid columns
    valid_columns = {
        'strategy_type', 'exchange', 'symbol',
        'bar_type', 'timeframe', 'status', 'simulator', 'live', 'access_level', 'tags',
        'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct',
        'sortino_ratio', 'calmar_ratio', 'profit_factor', 'sparkline_data', 'version',
        'description', 'demo'
    }
    
    row_data = {col: updates[col] for col in updates if col in valid_columns}
    if not row_data:
        return

    set_parts = [f"{col} = :{col}" for col in row_data.keys()]
    set_clause = ", ".join(set_parts)
    
    query = text(f"""
        UPDATE {STRATEGIES_SCHEMA}.metadata 
        SET {set_clause}
        WHERE id = :id
    """)
    
    params = {**row_data, "id": strategy_id}

    try:
        with engine.begin() as conn:
            conn.execute(query, params)
    except SQLAlchemyError as exc:
        logger.exception(f"Failed to update strategy metadata for {strategy_id}")
        raise QueryError("Failed to update strategy metadata") from exc


def filter_all_timeseries(stats, first_trade_time, keys):
    first_date = pd.to_datetime(first_trade_time, utc=True).date()

    out = {}
    for key in keys:
        series = stats.get(key, {})
        if not series:
            out[key] = {}
            continue

        # Vectorized: parse all timestamps at once instead of one-by-one
        ts_index = pd.to_datetime(list(series.keys()), utc=True, errors="coerce")
        values   = list(series.values())

        mask = (ts_index.date >= first_date) & ts_index.notna()
        out[key] = {
            ts.isoformat(): v
            for ts, v, keep in zip(ts_index, values, mask)
            if keep
        }

    return out

def parse_timeseries_rows(
    id_value: str,
    series_dict: Dict[str, Any],
    metrics: set,
    id_column: str = "id",
    return_type: Literal["records", "df"] = "records",
) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """
    Parse {metric: {timestamp: value}} into DB row dicts or a DataFrame.

    Args:
        id_value:     Value for the ID column.
        series_dict:  {metric: {ts: val}} or {metric: [(ts, val), ...]}
        metrics:      Set of metric names to extract.
        id_column:    Name of the ID column.
        return_type:  'records' → List[Dict]  |  'df' → pd.DataFrame

    Vectorized via pandas: one pd.to_datetime call per metric (no per-row
    Timestamp parsing), concat aligns timestamps across metrics, to_dict
    produces the final records in one shot.
    """
    frames = {}
    for metric in metrics:
        raw = series_dict.get(metric)
        if not raw:
            continue
        try:
            if isinstance(raw, dict):
                s = pd.Series(raw, dtype=float)
            elif isinstance(raw, list):
                if not raw:
                    continue
                ts_arr, val_arr = zip(*raw)
                s = pd.Series(list(val_arr), index=list(ts_arr), dtype=float)
            else:
                logger.warning("parse_timeseries_rows: unexpected type for '%s': %s", metric, type(raw))
                continue
            s.index = pd.to_datetime(s.index, utc=True)
            frames[metric] = s.dropna()
        except Exception:
            logger.warning("parse_timeseries_rows: failed to parse metric '%s'", metric)

    if not frames:
        return [] if return_type == "records" else pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df[id_column] = id_value
    df["datetime"]    = df.index.to_pydatetime()
    df = df.dropna(subset=list(frames.keys()), how="all").reset_index(drop=True)

    if return_type == "df":
        return df
    return df.to_dict(orient="records")