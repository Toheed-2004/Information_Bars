# bitpredict/common/db/tables/backtest_ops.py
"""
Backtest-specific wrapper functions for table operations.
All functions delegate to core implementations with schema_name="backtest".
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
from datetime import datetime
from bitpredict.common.db.services.shared import (
    upsert_ledgers,
    read_ledger ,
    get_last_trade,
    get_open_trades, 
    get_all_open_trades, 
    upsert_graphs ,
    read_graphs ,
    upsert_graphs_daily,
    read_graphs_daily,
    upsert_analytics ,
    read_analytics ,
    upsert_requests ,
    read_requests ,
)


# ============================================================================
# LEDGER OPERATIONS
# ============================================================================

def upsert_backtest_ledgers(
    ledger: pd.DataFrame,
    engine=None,
    skip_conflict_handling: bool = False
) -> None:
    """
    Bulk upsert ledger entries for backtest schema.
    
    Args:
        ledger: DataFrame with 'id' column (request_id) and optional 'trial_id'
        engine: Database engine (optional)
        skip_conflict_handling: If True, uses plain INSERT (faster)
    """
    return upsert_ledgers(
        ledger=ledger,
        schema_name="backtest",
        engine=engine,
        skip_conflict_handling=skip_conflict_handling
    )


def read_backtest_ledger(
    id_value: Union[str, int],
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    trial_id: Optional[bool] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read ledger entries for a specific request id.
    """
    return read_ledger(
        id_value=id_value,
        schema_name="backtest",
        start_date=start_date,
        end_date=end_date,
        trial_id=trial_id,
        columns=columns,
        limit=limit
    )


def get_backtest_last_trade(
    id_value: Union[str, int],
    trial_id: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the most recent trade for a specific request id.
    """
    return get_last_trade(
        id_value=id_value,
        schema_name="backtest",
        trial_id=trial_id
    )


def get_backtest_open_trades(
    id_value: Union[str, int],
    trial_id: Optional[bool] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get all open trades for a specific request id.
    """
    return get_open_trades(
        id_value=id_value,
        schema_name="backtest",
        trial_id=trial_id,
        columns=columns
    )


def get_backtest_all_open_trades(
    trial_id: Optional[bool] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get all open trades across all request ids.
    """
    return get_all_open_trades(
        schema_name="backtest",
        trial_id=trial_id,
        columns=columns
    )


# ============================================================================
# GRAPHS OPERATIONS
# ============================================================================

def upsert_backtest_graphs(
    df: pd.DataFrame,
    engine=None
) -> None:
    """
    Bulk upsert graph entries for backtest schema.
    
    Args:
        df: DataFrame with 'id' column (request_id) and optional 'trial_id'
        engine: Database engine (optional)
    """
    return upsert_graphs(
        df=df,
        schema_name="backtest",
        engine=engine
    )


def read_backtest_graphs(
    id_value: Union[str, int],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    trial_id: Optional[bool] = None,
    cols: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Read graph data for a specific request id.
    """
    return read_graphs(
        id_value=id_value,
        schema_name="backtest",
        start_date=start_date,
        end_date=end_date,
        trial_id=trial_id,
        cols=cols
    )


def upsert_backtest_graphs_daily(
    df: pd.DataFrame,
    engine=None
) -> None:
    """
    Bulk upsert daily graph entries for backtest schema.
    
    Args:
        df: DataFrame with 'id' column (request_id) and optional 'trial_id'
        engine: Database engine (optional)
    """
    return upsert_graphs_daily(
        df=df,
        schema_name="backtest",
        engine=engine
    )


def read_backtest_graphs_daily(
    id_value: Union[str, int],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    trial_id: Optional[bool] = None,
    cols: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Read daily graph data for a specific request id.
    """
    return read_graphs_daily(
        id_value=id_value,
        schema_name="backtest",
        start_date=start_date,
        end_date=end_date,
        trial_id=trial_id,
        cols=cols
    )


# ============================================================================
# ANALYTICS OPERATIONS
# ============================================================================

def upsert_backtest_analytics(
    rows: List[Dict[str, Any]],
    engine=None
) -> None:
    """
    Bulk upsert analytics for backtest schema.
    
    Args:
        rows: List of dicts with 'id' and optional 'trial_id'
        engine: Database engine (optional)
    """
    return upsert_analytics(
        rows=rows,
        schema_name="backtest",
        engine=engine
    )


def read_backtest_analytics(
    id_value: Union[str, int],
    trial_id: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Read analytics for a specific request id.
    """
    return read_analytics(
        id_value=id_value,
        schema_name="backtest",
        trial_id=trial_id
    )


# ============================================================================
# REQUESTS OPERATIONS (Backtest specific)
# ============================================================================

def upsert_backtest_requests(
    rows: List[Dict[str, Any]],
) -> None:
    """
    Bulk upsert requests for backtest schema.
    
    Args:
        rows: List of dicts with 'id' (optional for auto-assign) and 'trial_id'
    """
    return upsert_requests(
        rows=rows,
        schema_name="backtest"
    )


def read_backtest_requests(
    request_ids: Optional[List[int]] = None,
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    strategy_id: Optional[int] = None,
    trial_id: Optional[bool] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read requests with optional filters.
    """
    return read_requests(
        request_ids=request_ids,
        user_id=user_id,
        status=status,
        strategy_id=strategy_id,
        trial_id=trial_id,
        limit=limit,
        schema_name="backtest"
    )