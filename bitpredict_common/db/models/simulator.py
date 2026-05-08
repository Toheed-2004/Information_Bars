from sqlalchemy import Engine
from bitpredict.common.constants import SIMULATOR_SCHEMA
from bitpredict.common.db.models.shared import (
    ensure_ledger_table, 
    ensure_analytics_table, 
    ensure_graphs_table, 
    ensure_graphs_daily_table
)

# ============================================================================
# LEDGER TABLE
# ============================================================================

def ensure_simulator_ledger_table(
    engine: Engine,
    schema_name: str = "simulator",
    table_name: str = "ledgers",
    is_timeseries: bool = True
) -> bool:
    """
    Wrapper for simulator ledger table creation.
    """
    return ensure_ledger_table(
        engine=engine,
        schema_name=schema_name,
        table_name=table_name,
        is_timeseries=is_timeseries
    )


# ============================================================================
# SIMULATOR ANALYTICS TABLE
# ============================================================================

def ensure_simulator_analytics_table(
    engine: Engine, 
    schema_name: str = SIMULATOR_SCHEMA,
    table_name: str = "analytics"
) -> None:
    """
    Wrapper for simulator analytics table creation.
    """
    ensure_analytics_table(
        engine=engine,
        schema_name=schema_name,
        table_name=table_name
    )


# ============================================================================
# SIMULATOR GRAPHS TABLES
# ============================================================================

def ensure_simulator_graphs_table(
    engine: Engine, 
    schema_name: str = SIMULATOR_SCHEMA,
    table_name: str = "graphs", 
    is_timeseries: bool = True
) -> None:
    """
    Wrapper for simulator graphs table creation.
    """
    ensure_graphs_table(
        engine=engine,
        schema_name=schema_name,
        table_name=table_name,
        is_timeseries=is_timeseries
    )


def ensure_simulator_graphs_daily_table(
    engine: Engine, 
    schema_name: str = SIMULATOR_SCHEMA,
    table_name: str = "graphs_daily", 
    is_timeseries: bool = True
) -> None:
    """
    Wrapper for simulator graphs_daily table creation.
    """
    ensure_graphs_daily_table(
        engine=engine,
        schema_name=schema_name,
        table_name=table_name,
        is_timeseries=is_timeseries
    )