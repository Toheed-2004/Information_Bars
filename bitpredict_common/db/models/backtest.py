# bitpredict/common/db/tables/backtest_wrappers.py
from sqlalchemy import Engine
from bitpredict.common.db.models.shared import (
    ensure_ledger_table,
    ensure_analytics_table,
    ensure_graphs_table,
    ensure_graphs_daily_table,
    ensure_requests_table
)

# ============================================================================
# BACKTEST LEDGER TABLE
# ============================================================================

def ensure_backtest_ledger_table(
    engine: Engine,
    schema_name: str = "backtest",
    table_name: str = "ledgers",
    is_timeseries: bool = True
) -> bool:
    """
    Wrapper for backtest ledger table creation.
    """
    return ensure_ledger_table(
        engine=engine,
        schema_name=schema_name,
        table_name=table_name,
        is_timeseries=is_timeseries
    )


# ============================================================================
# BACKTEST ANALYTICS TABLE
# ============================================================================

def ensure_backtest_analytics_table(
    engine: Engine, 
    schema_name: str = "backtest",
    table_name: str = "analytics"
) -> None:
    """
    Wrapper for backtest analytics table creation.
    """
    ensure_analytics_table(
        engine=engine,
        schema_name=schema_name,
        table_name=table_name
    )


# ============================================================================
# BACKTEST GRAPHS TABLES
# ============================================================================

def ensure_backtest_graphs_table(
    engine: Engine, 
    schema_name: str = "backtest",
    table_name: str = "graphs", 
    is_timeseries: bool = True
) -> None:
    """
    Wrapper for backtest graphs table creation.
    """
    ensure_graphs_table(
        engine=engine,
        schema_name=schema_name,
        table_name=table_name,
        is_timeseries=is_timeseries
    )


def ensure_backtest_graphs_daily_table(
    engine: Engine, 
    schema_name: str = "backtest",
    table_name: str = "graphs_daily", 
    is_timeseries: bool = True
) -> None:
    """
    Wrapper for backtest graphs_daily table creation.
    """
    ensure_graphs_daily_table(
        engine=engine,
        schema_name=schema_name,
        table_name=table_name,
        is_timeseries=is_timeseries
    )


# ============================================================================
# BACKTEST REQUESTS TABLE
# ============================================================================

def ensure_backtest_requests_table(
    engine: Engine,
    schema_name: str = "backtest",
) -> None:
    """
    Wrapper for backtest requests table creation.
    """
    ensure_requests_table(
        engine=engine,
        schema_name=schema_name,
    )