
from typing import List, Dict, Any, Optional

import pandas as pd
import psycopg2.extras
from psycopg2.extras import Json as PgJson
from io import StringIO
from bitpredict.common.db.config import get_engine
from datetime import datetime
from typing import Optional, Union, List, Dict, Any
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from bitpredict.common.db.utils import  clean_for_json
from bitpredict.common.constants import SIMULATOR_SCHEMA
from bitpredict.common.db.exceptions import QueryError
from bitpredict.common.logging import get_logger
from bitpredict.common.db.utils import read_df

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared metric sets (mirrors simulator.py)
# ---------------------------------------------------------------------------
TIMESERIES_SCALAR_METRICS = {
    "daily_cumulative_return",
    "drawdown_pct",
    "benchmark_return_pct",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_correlation",
}

TIMESERIES_TRADE_METRICS = {
    "mfe_pct",
    "mae_pct",
    "long_return_pct",
    "short_return_pct",
}

ANALYTICS_JSONB_COLS = {
    "risk_adjusted",
    "risk_metrics",
    "drawdown_analysis",
    "trade_analysis",
    "profit_loss",
    "long_short",
    "portfolio_values",
    "exposure",
    "cash_flow",
    "time_series_analysis",
    "benchmark_analysis",
    "distribution_analysis",
    "drawdown_periods",
    "directional_metrics",
    "regimes_analysis",
}

ANALYTICS_SCALAR_COLS = {"portfolio_state_path"}

BACKTEST_REQUEST_JSONB_COLS = {"backtest_config", "strategy_config"}

# ============================================================================
# LEDGER FUNCTIONS
# ============================================================================

def upsert_ledgers(
    ledger: pd.DataFrame, 
    schema_name: str = SIMULATOR_SCHEMA, 
    engine=None, 
    skip_conflict_handling: bool = False
) -> None:
    """
    Bulk upsert trades for multiple strategies into ledgers table.

    Input format:
        pd.DataFrame with all trades concatenated.
        Must include 'id' column (strategy_id or request_id).
        Optional 'trial_id' boolean column.
        Column order does not matter.

    Example:
        id | trial_id | entry_datetime      | avg_entry_price | status | ...
        1  | NULL     | 2024-01-01 09:00:00 | 45000.0         | Closed | ...
        2  | true     | 2024-01-01 10:00:00 | 44500.0         | Open   | ...

    Strategy:
        COPY into a temp table → INSERT ... ON CONFLICT DO UPDATE
        WHERE status = 'Open' (same logic as single-strategy upsert).
        One COPY + one INSERT for all strategies combined.
        
    Args:
        skip_conflict_handling: If True, uses plain INSERT (faster, but errors on duplicate keys)
    """
    if ledger is None or ledger.empty:
        return

    if engine is None:
        engine = get_engine()

    # Ensure trial_id column exists with default NULL
    if "trial_id" not in ledger.columns:
        ledger["trial_id"] = None
    
    columns = list(ledger.columns)
    col_list = ", ".join(columns)

    # Build typed SELECT with explicit casts from TEXT staging
    ledger_col_types = {
        "id": "BIGINT",
        "trial_id": "BOOLEAN",
        "entry_datetime": "TIMESTAMPTZ",
        "entry_fee_pct": "DOUBLE PRECISION",
        "avg_entry_price": "DOUBLE PRECISION",
        "exit_datetime": "TIMESTAMPTZ",
        "exit_fee_pct": "DOUBLE PRECISION",
        "avg_exit_price": "DOUBLE PRECISION",
        "position_size_pct": "DOUBLE PRECISION",
        "trade_return_pct": "DOUBLE PRECISION",
        "account_return_pct": "DOUBLE PRECISION",
        "cum_account_return": "DOUBLE PRECISION",
        "direction": "TEXT",
        "status": "TEXT",
        "action": "TEXT",
        "balance": "DOUBLE PRECISION",
        "updated_at": "TIMESTAMPTZ",
    }
    cast_select = ", ".join(
        f'"{c}"::{ledger_col_types.get(c, "TEXT")}' for c in columns
    )

    # Choose between fast INSERT or slower UPSERT based on flag
    if skip_conflict_handling:
        # Fast path: plain INSERT without conflict handling
        insert_sql = f"""
            INSERT INTO {schema_name}.ledgers ({col_list})
            SELECT {cast_select} FROM _bulk_ledger_{schema_name}_staging
        """
        final_sql = insert_sql
    else:
        # Slow path: INSERT with ON CONFLICT DO UPDATE
        updates = ", ".join(
            f"{c} = EXCLUDED.{c}"
            for c in columns
            if c not in ("id", "entry_datetime", "updated_at")
        ) + ", updated_at = NOW()"

        upsert_sql = f"""
            INSERT INTO {schema_name}.ledgers ({col_list})
            SELECT {cast_select} FROM _bulk_ledger_{schema_name}_staging
            ON CONFLICT (id, entry_datetime)
            DO UPDATE SET {updates}
            WHERE {schema_name}.ledgers.status = 'Open'
        """
        final_sql = upsert_sql

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            # Temp table — no indexes, no FK, no triggers = fastest possible COPY target
            cur.execute(f"""
                CREATE TEMP TABLE _bulk_ledger_{schema_name}_staging (
                    {', '.join(f'"{c}" TEXT' for c in columns)}
                ) ON COMMIT DROP
            """)

            buf = StringIO()
            ledger.to_csv(buf, index=False, header=False, na_rep="\\N")
            buf.seek(0)
            cur.copy_expert(
                f"COPY _bulk_ledger_{schema_name}_staging ({col_list}) FROM STDIN WITH CSV NULL '\\N'",
                buf,
            )

            # Cast from TEXT staging → typed real table in one shot
            cur.execute(final_sql)

        raw_conn.commit()
        logger.info("Bulk upserted %d ledger rows", len(ledger))
    except Exception as exc:
        raw_conn.rollback()
        logger.exception("Failed to bulk upsert ledgers")
        raise QueryError("Failed to bulk upsert ledgers") from exc
    finally:
        raw_conn.close()

def read_ledger(
    id_value: Union[str, int],
    schema_name: str = "simulator",
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    trial_id: Optional[bool] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read ledger entries for a specific id.

    Args:
        id_value: The id (strategy_id or request_id)
        schema_name: Target schema ('simulator' or 'backtest')
        start_date: Filter by entry_datetime >= start_date
        end_date: Filter by entry_datetime <= end_date
        trial_id: Filter by trial_id (True/False)
        columns: Specific columns to return
        limit: Max rows to return
    """
    engine = get_engine()


    where_parts = [f"id = '{id_value}'"]

    if start_date:
        where_parts.append(f"entry_datetime >= '{start_date}'")

    if end_date:
        where_parts.append(f"entry_datetime <= '{end_date}'")

    if trial_id is not None:
        where_parts.append(f"trial_id = {str(trial_id).lower()}")

    where_clause = " AND ".join(where_parts)

    # Read using your existing read_df
    df = read_df(
        engine=engine,
        schema=schema_name,
        table_name="ledgers",
        columns=columns,
        where=where_clause,
        method="copy"  # copy, sql and connectrox
    )

    # Sort by entry_datetime after reading
    if not df.empty:
        df = df.sort_values("entry_datetime")

    if limit and not df.empty:
        df = df.head(limit)

    return df

def get_last_trade(
    id_value: Union[str, int],
    schema_name: str = "simulator",
    trial_id: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the most recent trade for a specific id.

    Args:
        id_value: The id (strategy_id or request_id)
        schema_name: Target schema ('simulator' or 'backtest')
        trial_id: Filter by trial_id (True/False)
    """
    engine = get_engine()

    sql = f"""
        SELECT *
        FROM {schema_name}.ledgers
        WHERE id = :id_value
    """
    
    params = {"id_value": id_value}
    
    if trial_id is not None:
        sql += " AND trial_id = :trial_id"
        params["trial_id"] = trial_id
    
    sql += " ORDER BY entry_datetime DESC LIMIT 1"

    with engine.connect() as conn:
        result = conn.execute(text(sql), params).fetchone()

    if result is None:
        return None

    return dict(result._mapping)

def get_open_trades(
    id_value: Union[str, int],
    schema_name: str = "simulator",
    trial_id: Optional[bool] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get all open trades (exit_datetime IS NULL) for a specific id.

    Args:
        id_value: The id (strategy_id or request_id)
        schema_name: Target schema ('simulator' or 'backtest')
        trial_id: Filter by trial_id (True/False)
        columns: Specific columns to return
    """
    engine = get_engine()

    where_parts = [
        f"id = '{id_value}'",
        "exit_datetime IS NULL"
    ]
    
    if trial_id is not None:
        where_parts.append(f"trial_id = {str(trial_id).lower()}")

    where_clause = " AND ".join(where_parts)

    df = read_df(
        engine=engine,
        schema=schema_name,
        table_name="ledgers",
        columns=columns,
        where=where_clause,
        method="sql"
    )

    if not df.empty:
        df = df.sort_values("entry_datetime")

    return df

def get_all_open_trades(
    schema_name: str = "simulator",
    trial_id: Optional[bool] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get all open trades (exit_datetime IS NULL) across all ids.

    Args:
        schema_name: Target schema ('simulator' or 'backtest')
        trial_id: Filter by trial_id (True/False)
        columns: Specific columns to return
    """
    engine = get_engine()

    where_parts = ["exit_datetime IS NULL"]
    
    if trial_id is not None:
        where_parts.append(f"trial_id = {str(trial_id).lower()}")

    where_clause = " AND ".join(where_parts)

    df = read_df(
        engine=engine,
        schema=schema_name,
        table_name="ledgers",
        columns=columns,
        where=where_clause,
        method="sql"
    )

    if not df.empty:
        df = df.sort_values(["id", "entry_datetime"])

    return df

def upsert_graphs(
    df: pd.DataFrame, 
    schema_name: str = SIMULATOR_SCHEMA, 
    engine=None
) -> None:
    """
    Bulk upsert per-trade graph rows into graphs table.
    Uses COPY with staging table for maximum performance.

    Input format:
        pd.DataFrame must include 'id' column and 'datetime' column.
        Optional 'trial_id' boolean column.

    Example:
        id | trial_id | datetime            | mfe_pct | mae_pct | ...
        1  | NULL     | 2024-01-01 09:00:00 | 2.5     | -1.2    | ...
        2  | true     | 2024-01-01 10:00:00 | 1.8     | -0.5    | ...
    """
    if df is None or df.empty:
        return

    if engine is None:
        engine = get_engine()

    cols = list(TIMESERIES_TRADE_METRICS)
    insert_cols = ["id", "datetime"] + cols
    
    # Ensure trial_id column exists
    if "trial_id" not in df.columns:
        df["trial_id"] = None
    insert_cols.append("trial_id")

    for c in cols:
        if c not in df.columns:
            df[c] = None

    # Build column list with proper quoting for reserved words (datetime)
    col_list = ', '.join(f'"{c}"' if c == "datetime" else c for c in insert_cols)
    
    set_clauses = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in cols + ["trial_id"]
    ) + ", updated_at = now()"

    # Cast from staging table with proper types
    cast_parts = []
    for col in insert_cols:
        if col == "id":
            cast_parts.append(f'"{col}"::BIGINT')
        elif col == "datetime":
            cast_parts.append(f'"{col}"::TIMESTAMPTZ')
        elif col == "trial_id":
            cast_parts.append(f'"{col}"::BOOLEAN')
        elif col in TIMESERIES_TRADE_METRICS:
            cast_parts.append(f'"{col}"::DOUBLE PRECISION')
        else:
            cast_parts.append(f'"{col}"::TEXT')
    cast_select = ", ".join(cast_parts)

    upsert_sql = f"""
        INSERT INTO {schema_name}.graphs ({col_list})
        SELECT {cast_select} FROM _graphs_{schema_name}_staging
        ON CONFLICT (id, datetime)
        DO UPDATE SET {set_clauses}
    """

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            # Create staging table as TEXT columns for fastest COPY
            staging_cols = ', '.join(f'"{c}" TEXT' for c in insert_cols)
            cur.execute(f"""
                CREATE TEMP TABLE _graphs_{schema_name}_staging (
                    {staging_cols}
                ) ON COMMIT DROP
            """)

            # COPY data to staging table
            buf = StringIO()
            df[insert_cols].to_csv(buf, index=False, header=False, na_rep='\\N')
            buf.seek(0)
            cur.copy_expert(
                f"COPY _graphs_{schema_name}_staging ({col_list}) FROM STDIN WITH CSV NULL '\\N'",
                buf
            )

            # Insert from staging to real table with casting
            cur.execute(upsert_sql)

        raw_conn.commit()
        logger.info("Bulk upserted %d graph rows", len(df))
    except Exception as exc:
        raw_conn.rollback()
        logger.exception("Failed to bulk upsert graphs")
        raise QueryError("Failed to bulk upsert graphs") from exc
    finally:
        raw_conn.close()

def read_graphs(
    id_value: Union[str, int],
    schema_name: str = "simulator",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    trial_id: Optional[bool] = None,
    cols: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Read graph data for a specific id.

    Args:
        id_value: The id (strategy_id or request_id)
        schema_name: Target schema ('simulator' or 'backtest')
        start_date: Filter by datetime >= start_date
        end_date: Filter by datetime <= end_date
        trial_id: Filter by trial_id (True/False)
        cols: Specific columns to return (must be in TIMESERIES_TRADE_METRICS)

    Returns:
        Dict with datetime as key and column values as dict
    """
    engine = get_engine()

    requested_columns = cols if cols else list(TIMESERIES_TRADE_METRICS)
    invalid = [c for c in requested_columns if c not in TIMESERIES_TRADE_METRICS]
    if invalid:
        raise ValueError(f"Invalid columns: {invalid}")

    select_clause = ", ".join(["datetime"] + requested_columns)

    # Build query with optional trial_id filter
    query_str = f"""
        SELECT {select_clause}
        FROM {schema_name}.graphs
        WHERE id = :id_value
        AND (:start_date IS NULL OR datetime >= :start_date)
        AND (:end_date IS NULL OR datetime <= :end_date)
    """
    
    params = {
        "id_value": id_value,
        "start_date": start_date,
        "end_date": end_date
    }
    
    if trial_id is not None:
        query_str += " AND trial_id = :trial_id"
        params["trial_id"] = trial_id
    
    query_str += " ORDER BY datetime"
    
    query = text(query_str)

    try:
        with engine.connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return {
            row.datetime.isoformat(): {
                col: getattr(row, col)
                for col in requested_columns
                if getattr(row, col) is not None
            }
            for row in rows
        }
    except SQLAlchemyError as exc:
        logger.exception("Failed to read graphs")
        raise QueryError("Failed to read graphs") from exc

# ============================================================================
# GRAPHS DAILY UPSERT
# ============================================================================

def upsert_graphs_daily(
    df: pd.DataFrame, 
    schema_name: str = SIMULATOR_SCHEMA, 
    engine=None
) -> None:
    """
    Bulk upsert daily scalar graph rows into graphs_daily table.
    
    Input format:
        pd.DataFrame must include 'id' column and 'datetime' column.
        Optional 'trial_id' boolean column.
    """
    if df is None or df.empty:
        return

    if engine is None:
        engine = get_engine()

    cols = list(TIMESERIES_SCALAR_METRICS)
    insert_cols = ["id", "datetime"] + cols
    
    # Ensure trial_id column exists
    if "trial_id" not in df.columns:
        df["trial_id"] = None
    insert_cols.append("trial_id")

    for c in cols:
        if c not in df.columns:
            df[c] = None

    set_clauses = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in cols + ["trial_id"]
    ) + ", updated_at = now()"

    sql = f"""
        INSERT INTO {schema_name}.graphs_daily ({', '.join(insert_cols)})
        VALUES %s
        ON CONFLICT (id, datetime)
        DO UPDATE SET {set_clauses}
    """

    rows = df[insert_cols].itertuples(index=False, name=None)

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows, page_size=2000)
        raw_conn.commit()
        logger.info("Bulk upserted %d daily graph rows", len(df))
    except Exception as exc:
        raw_conn.rollback()
        logger.exception("Failed to bulk upsert graphs daily")
        raise QueryError("Failed to bulk upsert graphs daily") from exc
    finally:
        raw_conn.close()

def read_graphs_daily(
    id_value: Union[str, int],
    schema_name: str = "simulator",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    trial_id: Optional[bool] = None,
    cols: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Read daily graph data for a specific id.

    Args:
        id_value: The id (strategy_id or request_id)
        schema_name: Target schema ('simulator' or 'backtest')
        start_date: Filter by datetime >= start_date
        end_date: Filter by datetime <= end_date
        trial_id: Filter by trial_id (True/False)
        cols: Specific columns to return (must be in TIMESERIES_SCALAR_METRICS)

    Returns:
        Dict with datetime as key and column values as dict
    """
    engine = get_engine()

    requested_columns = cols if cols else list(TIMESERIES_SCALAR_METRICS)
    invalid = [c for c in requested_columns if c not in TIMESERIES_SCALAR_METRICS]
    if invalid:
        raise ValueError(f"Invalid columns: {invalid}")

    select_clause = ", ".join(["datetime"] + requested_columns)

    # Build query with optional trial_id filter
    query_str = f"""
        SELECT {select_clause}
        FROM {schema_name}.graphs_daily
        WHERE id = :id_value
        AND (:start_date IS NULL OR datetime >= :start_date)
        AND (:end_date IS NULL OR datetime <= :end_date)
    """
    
    params = {
        "id_value": id_value,
        "start_date": start_date,
        "end_date": end_date
    }
    
    if trial_id is not None:
        query_str += " AND trial_id = :trial_id"
        params["trial_id"] = trial_id
    
    query_str += " ORDER BY datetime"
    
    query = text(query_str)

    try:
        with engine.connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return {
            row.datetime.isoformat(): {
                col: getattr(row, col)
                for col in requested_columns
                if getattr(row, col) is not None
            }
            for row in rows
        }
    except SQLAlchemyError as exc:
        logger.exception("Failed to read graphs daily")
        raise QueryError("Failed to read graphs daily") from exc
    
# ============================================================================
# ANALYTICS UPSERT
# ============================================================================

def upsert_analytics(
    rows: List[Dict[str, Any]], 
    schema_name: str = SIMULATOR_SCHEMA, 
    engine=None
) -> None:
    """
    Bulk upsert analytics into analytics table.

    Input format:
        List of dicts, one per id. Each dict must have 'id'.
        Optional 'trial_id' field.
        JSONB columns are plain Python dicts — serialization handled here.

    Example:
        [
            {
                "id": 1,
                "trial_id": None,
                "risk_adjusted": {"sharpe_ratio": 1.2, ...},
                "trade_analysis": {"win_rate_pct": 55.0, ...},
                "portfolio_state_path": "/path/to/1.pkl",
            },
        ]
    """
    if not rows:
        return

    if engine is None:
        engine = get_engine()

    # Build normalized row list — serialize JSONB, skip unknown keys
    normalized = []

    for raw in rows:
        row: Dict[str, Any] = {"id": raw["id"]}
        
        # Add trial_id if present
        if "trial_id" in raw:
            row["trial_id"] = raw["trial_id"]
        
        for k, v in raw.items():
            if k in ("id", "trial_id"):
                continue
            if k in ANALYTICS_JSONB_COLS:
                row[k] = PgJson(clean_for_json(v))
            elif k in ANALYTICS_SCALAR_COLS:
                row[k] = v
        normalized.append(row)

    if not normalized:
        return

    # Union of all keys present across all rows
    col_names = ["id"] + sorted(
        {k for r in normalized for k in r if k != "id"}
    )

    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in col_names if c != "id"
    ) + ", updated_at = now()"

    sql = f"""
        INSERT INTO {schema_name}.analytics ({', '.join(col_names)})
        VALUES %s
        ON CONFLICT (id)
        DO UPDATE SET {updates}
    """

    for row in normalized:
        for c in col_names:
            row.setdefault(c, None)

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            tuples = [tuple(row[c] for c in col_names) for row in normalized]
            psycopg2.extras.execute_values(cur, sql, tuples, page_size=100)
        raw_conn.commit()
        logger.info("Bulk upserted analytics for %d records", len(normalized))
    except Exception as exc:
        raw_conn.rollback()
        logger.exception("Failed to bulk upsert analytics")
        raise QueryError("Failed to bulk upsert analytics") from exc
    finally:
        raw_conn.close()

def read_analytics(
    id_value: Union[str, int],
    schema_name: str = "simulator",
    trial_id: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Read analytics for a specific id.

    Args:
        id_value: The id (strategy_id or request_id)
        schema_name: Target schema ('simulator' or 'backtest')
        trial_id: Filter by trial_id (True/False)
    """
    engine = get_engine()

    
    sql = f"""
        SELECT *
        FROM {schema_name}.analytics
        WHERE id = :id_value
    """
    
    params = {"id_value": id_value}
    
    if trial_id is not None:
        sql += " AND trial_id = :trial_id"
        params["trial_id"] = trial_id
    
    sql += " LIMIT 1"
    
    with engine.connect() as conn:
        result = conn.execute(text(sql), params).fetchone()
    
    if result is None:
        return None
    
    return dict(result._mapping)

# ============================================================================
# REQUESTS UPSERT / READ
# ============================================================================

def upsert_requests(
    rows: List[Dict[str, Any]],
    schema_name: str = "backtest",
) -> None:
    """
    Bulk upsert rows into requests table.

    Input format:
        List of dicts, one per request. Each dict must have 'id' for updates,
        or omit 'id' to let the DB auto-assign it on insert.
        Optional 'trial_id' boolean field.
        JSONB columns (backtest_config, strategy_config) are plain Python dicts.
    """
    if not rows:
        return

    engine = get_engine()

    normalized = []
    for raw in rows:
        row: Dict[str, Any] = {}
        for k, v in raw.items():
            if k in BACKTEST_REQUEST_JSONB_COLS:
                row[k] = PgJson(clean_for_json(v)) if v is not None else None
            else:
                row[k] = v
        # Ensure trial_id exists
        if "trial_id" not in row:
            row["trial_id"] = None
        normalized.append(row)

    if not normalized:
        return

    col_names = list({k for r in normalized for k in r})
    # Stable ordering: id first if present
    if "id" in col_names:
        col_names = ["id"] + sorted(c for c in col_names if c != "id")
    else:
        col_names = sorted(col_names)

    for row in normalized:
        for c in col_names:
            row.setdefault(c, None)

    update_cols = [c for c in col_names if c not in ("id", "created_at")]
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols) + ", updated_at = now()"

    sql = f"""
        INSERT INTO {schema_name}.requests ({', '.join(col_names)})
        VALUES %s
        ON CONFLICT (id)
        DO UPDATE SET {updates}
    """

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            tuples = [tuple(row[c] for c in col_names) for row in normalized]
            psycopg2.extras.execute_values(cur, sql, tuples, page_size=500)
        raw_conn.commit()
        logger.info("Bulk upserted %d request rows", len(normalized))
    except Exception as exc:
        raw_conn.rollback()
        logger.exception("Failed to bulk upsert requests")
        raise QueryError("Failed to bulk upsert requests") from exc
    finally:
        raw_conn.close()

def read_requests(
    request_ids: Optional[List[int]] = None,
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    strategy_id: Optional[int] = None,
    trial_id: Optional[bool] = None,
    limit: Optional[int] = None,
    schema_name: str = "backtest",
) -> pd.DataFrame:
    """
    Read rows from requests table with optional filters.

    Args:
        request_ids:  Filter by specific request IDs.
        user_id:      Filter by user_id.
        status:       Filter by status (e.g. 'pending', 'running', 'done').
        strategy_id:  Filter by strategy_id.
        trial_id:     Filter by trial_id (True/False).
        limit:        Max rows to return.
        schema_name:  Target schema (default 'backtest').

    Returns:
        pd.DataFrame with matching rows, ordered by created_at DESC.
    """

    engine = get_engine()

    where_parts = []
    if request_ids:
        ids_str = ", ".join(str(i) for i in request_ids)
        where_parts.append(f"id IN ({ids_str})")
    if user_id is not None:
        where_parts.append(f"user_id = '{user_id}'")
    if status is not None:
        where_parts.append(f"status = '{status}'")
    if strategy_id is not None:
        where_parts.append(f"strategy_id = {strategy_id}")
    if trial_id is not None:
        where_parts.append(f"trial_id = {str(trial_id).lower()}")

    where = " AND ".join(where_parts) if where_parts else None
    order_by = "created_at DESC"
    if limit:
        order_by += f" LIMIT {limit}"

    try:
        df = read_df(
            engine=engine,
            schema=schema_name,
            table_name="requests",
            where=where,
            order_by=order_by,
            method="sql",
        )
        return df
    except Exception as exc:
        logger.exception("Failed to read requests")
        raise QueryError("Failed to read requests") from exc