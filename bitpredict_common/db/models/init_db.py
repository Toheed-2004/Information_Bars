"""
Master script to create all database tables.

This script automatically imports and executes all table creation functions
from the models package to set up the complete database schema.
"""

import sys
from typing import List, Tuple

from sqlalchemy.engine import Engine

from bitpredict.common.logging import get_logger
from bitpredict.common.db.config import get_engine, check_connection
from bitpredict.common.constants import DATA_SCHEMA, ALLOWED_BAR_TYPES
from bitpredict.common.db.models import (
    # Bar table operations
    ensure_bars_table,
    ensure_state_table,
    
    # Metadata table operations
    ensure_symbols_table,
    ensure_time_bars_table,
    ensure_tick_table,
    ensure_custom_bars_table,
    ensure_blockchain_table,
    ensure_macro_table,
    
    # Strategies table operations
    ensure_strategies_metadata_table,
    ensure_strategies_configs_table,
    ensure_strategies_training_table,

    # Simulator
    ensure_simulator_ledger_table,
    ensure_signal_table,
    ensure_simulator_analytics_table,
    ensure_simulator_graphs_table,
    ensure_simulator_graphs_daily_table,

    # Backtest
    ensure_backtest_ledger_table,
    ensure_backtest_analytics_table,
    ensure_backtest_graphs_table,
    ensure_backtest_graphs_daily_table,
    ensure_backtest_requests_table,
    # Market Regime
    ensure_regime_state_table,

    # Regime Analysis
    ensure_regime_analysis_column_in_analytics_table
)
from bitpredict.common.db.models.data import ensure_bar_stats_table, ensure_ohlcv_table
from bitpredict.common.db.utils import ensure_schema

logger = get_logger(__name__)


def create_metadata_tables(engine: Engine) -> Tuple[bool, List[str]]:
    """
    Create all metadata tables.
    
    Parameters
    ----------
    engine : Engine
        SQLAlchemy database engine
    
    Returns
    -------
    Tuple[bool, List[str]]
        Success status and list of created tables
    """
    created_tables = []
    
    try:
        ensure_schema(engine, "metadata")
        logger.info("Creating metadata tables...")
        
        ensure_symbols_table(engine)
        created_tables.append("symbols")
        logger.info("  ✓ Symbols table created")
        
        ensure_time_bars_table(engine)
        created_tables.append("time_bars")
        logger.info("  ✓ Time bars table created")
        
        ensure_tick_table(engine)
        created_tables.append("tick")
        logger.info("  ✓ Tick table created")
        
        ensure_custom_bars_table(engine)
        created_tables.append("custom_bars")
        logger.info("  ✓ Custom bars table created")
        
        ensure_blockchain_table(engine)
        created_tables.append("blockchain")
        logger.info("  ✓ Blockchain table created")
        
        ensure_macro_table(engine)
        created_tables.append("macro")
        logger.info("  ✓ Macro table created")
        
        return True, created_tables
        
    except Exception as e:
        logger.error(f"  ✗ Failed to create metadata tables: {e}")
        return False, created_tables


def create_data_tables(engine: Engine) -> Tuple[bool, List[str]]:
    """
    Create all data tables (bars, state, stats).
    
    Parameters
    ----------
    engine : Engine
        SQLAlchemy database engine
    
    Returns
    -------
    Tuple[bool, List[str]]
        Success status and list of created tables
    """
    created_tables = []
    
    try:
        logger.info("Creating data tables...")
        
        # Create schema
        ensure_schema(engine, DATA_SCHEMA)
        created_tables.append(f"schema:{DATA_SCHEMA}")
        logger.info(f"  ✓ Schema '{DATA_SCHEMA}' ensured")
        
        # Create OHLCV table
        ensure_ohlcv_table(engine, schema_name=DATA_SCHEMA, table_name="time")
        created_tables.append("ohlcv_time")
        logger.info("  ✓ OHLCV time table created")
        
        # Create bars tables for each bar type
        for bar_type in ALLOWED_BAR_TYPES:
            try:
                ensure_bars_table(bar_type)
                created_tables.append(f"bars_{bar_type}")
                logger.info(f"  ✓ {bar_type.capitalize()} bars table created")
            except ValueError as e:
                logger.warning(f"  ⚠ Skipped {bar_type} bars table: {e}")
        
        # Create state table
        ensure_state_table()
        created_tables.append("state")
        logger.info("  ✓ State table created")
        
        # Create bar stats table
        ensure_bar_stats_table()
        created_tables.append("bar_stats")
        logger.info("  ✓ Bar stats table created")
        
        return True, created_tables
        
    except Exception as e:
        logger.error(f"  ✗ Failed to create data tables: {e}")
        return False, created_tables


def create_strategies_tables(engine: Engine) -> Tuple[bool, List[str]]:
    """
    Create all strategies-related tables.
    
    Parameters
    ----------
    engine : Engine
        SQLAlchemy database engine
    
    Returns
    -------
    Tuple[bool, List[str]]
        Success status and list of created tables
    """
    created_tables = []
    
    try:
        ensure_schema(engine, "strategies")
        logger.info("Creating strategies tables...")
        
        ensure_strategies_metadata_table(engine)
        created_tables.append("strategies_metadata")
        logger.info("  ✓ Strategies metadata table created")
        
        ensure_strategies_configs_table(engine)
        created_tables.append("strategies_configs")
        logger.info("  ✓ Strategies configs table created")
        
        ensure_strategies_training_table(engine)
        created_tables.append("strategies_training")
        logger.info("  ✓ Strategies training table created")
        
        return True, created_tables
        
    except Exception as e:
        logger.error(f"  ✗ Failed to create strategies tables: {e}")
        return False, created_tables


def create_simulator_tables(engine: Engine) -> Tuple[bool, List[str]]:
    """
    Create all simulator-related tables.
    
    Parameters
    ----------
    engine : Engine
        SQLAlchemy database engine
    
    Returns
    -------
    Tuple[bool, List[str]]
        Success status and list of created tables
    """
    created_tables = []
    
    try:
        ensure_schema(engine, "simulator")
        logger.info("Creating simulator tables...")
        
        ensure_simulator_ledger_table(engine)
        created_tables.append("ledger")
        logger.info("  ✓ Ledger table created")
        
        ensure_simulator_analytics_table(engine)
        created_tables.append("simulator_analytics")
        logger.info("  ✓ Simulator analytics table created")
        
        ensure_simulator_graphs_table(engine)
        created_tables.append("simulator_graphs")
        logger.info("  ✓ Simulator graphs table created")
        
        ensure_simulator_graphs_daily_table(engine)
        created_tables.append("simulator_graphs_daily")
        logger.info("  ✓ Simulator graphs daily table created")
        
        return True, created_tables
        
    except Exception as e:
        logger.error(f"  ✗ Failed to create simulator tables: {e}")
        return False, created_tables
    
def create_backtest_tables(engine: Engine) -> Tuple[bool, List[str]]:
    """
    Create all backtest-related tables.
    
    Parameters
    ----------
    engine : Engine
        SQLAlchemy database engine
    
    Returns
    -------
    Tuple[bool, List[str]]
        Success status and list of created tables
    """
    created_tables = []
    
    try:
        ensure_schema(engine, "backtest")
        logger.info("Creating backtest tables...")

        ensure_backtest_requests_table(engine)
        created_tables.append("backtest_requests")
        logger.info("  ✓ Backtest requests table created")
        
        ensure_backtest_ledger_table(engine)
        created_tables.append("ledger")
        logger.info("  ✓ Ledger table created")
        
        ensure_backtest_analytics_table(engine)
        created_tables.append("backtest_analytics")
        logger.info("  ✓ Backtest analytics table created")
        
        ensure_backtest_graphs_table(engine)
        created_tables.append("backtest_graphs")
        logger.info("  ✓ Backtest graphs table created")
        
        ensure_backtest_graphs_daily_table(engine)
        created_tables.append("backtest_graphs_daily")
        logger.info("  ✓ Backtest graphs daily table created")

        return True, created_tables
        
    except Exception as e:
        logger.error(f"  ✗ Failed to create backtest tables: {e}")
        return False, created_tables



def create_signals_tables(engine: Engine) -> Tuple[bool, List[str]]:
    """
    Create all signals-related tables.
    
    Parameters
    ----------
    engine : Engine
        SQLAlchemy database engine
    
    Returns
    -------
    Tuple[bool, List[str]]
        Success status and list of created tables
    """
    created_tables = []
    
    try:
        logger.info("Creating signals tables...")
        
        ensure_signal_table(engine)
        created_tables.append("signals")
        logger.info("  ✓ Signals table created")
        
        return True, created_tables
        
    except Exception as e:
        logger.error(f"  ✗ Failed to create signals tables: {e}")
        return False, created_tables


def create_market_regime_tables(engine: Engine) -> Tuple[bool, List[str]]:
    """
    Create all market regime tracking tables.
    
    Parameters
    ----------
    engine : Engine
        SQLAlchemy database engine
    
    Returns
    -------
    Tuple[bool, List[str]]
        Success status and list of created tables
    """
    created_tables = []
    
    try:
        logger.info("Creating market regime tables...")
        
        ensure_regime_state_table()
        created_tables.append("regime_state")
        logger.info("  ✓ Market regime state table created")
        
        return True, created_tables
        
    except Exception as e:
        logger.error(f"  ✗ Failed to create market regime tables: {e}")
        return False, created_tables


def create_regime_analysis_tables(engine: Engine) -> Tuple[bool, List[str]]:
    """
    Create regime analysis columns in existing tables.
    
    Parameters
    ----------
    engine : Engine
        SQLAlchemy database engine
    
    Returns
    -------
    Tuple[bool, List[str]]
        Success status and list of created modifications
    """
    created_modifications = []
    
    try:
        logger.info("Creating regime analysis columns...")
        
        ensure_regime_analysis_column_in_analytics_table(engine)
        created_modifications.append("regime_analysis_column_in_analytics")
        logger.info("  ✓ Regime analysis column added to analytics table")
        
        return True, created_modifications
        
    except Exception as e:
        logger.error(f"  ✗ Failed to create regime analysis columns: {e}")
        return False, created_modifications


def create_all_tables() -> bool:
    """
    Create all database tables by calling all category-specific functions.
    
    Functions are called in a logical dependency order:
    1. Metadata tables (symbols, time bars, etc.) - these are referenced by other tables
    2. Data tables (bars, state) - core data storage
    3. Strategies tables - user strategies
    4. Simulator tables - simulation results
    5. Signals tables - signal data
    6. Market regime tables - regime state tracking
    7. Regime analysis tables - analysis results
    
    Returns
    -------
    bool
        True if all tables were created successfully, False otherwise.
    """
    engine = get_engine()
    
    # Test connection before attempting to create tables
    if not check_connection(engine):
        logger.error("Failed to connect to database")
        return False
    
    logger.info("=" * 70)
    logger.info("Starting database table creation...")
    
    # Track overall success and created components
    all_success = True
    all_created = []
    
    # Define category functions in order
    category_functions = [
        ("Metadata Tables", create_metadata_tables),
        ("Data Tables", create_data_tables),
        ("Strategies Tables", create_strategies_tables),
        ("Simulator Tables", create_simulator_tables),
        ("Backtest Tables", create_backtest_tables),
        ("Signals Tables", create_signals_tables),
        ("Market Regime Tables", create_market_regime_tables),
        ("Regime Analysis", create_regime_analysis_tables),
    ]
    
    # Execute each category
    for category_name, category_func in category_functions:
        logger.info(f"\n[{category_name}]")
        success, created = category_func(engine)
        
        if success:
            all_created.extend(created)
            logger.info(f"✓ {category_name} completed: {len(created)} components created")
        else:
            all_success = False
            logger.error(f"✗ {category_name} failed")
            # Continue with other categories even if one fails
    
    # Final summary
    logger.info("\n" + "=" * 70)
    if all_success:
        logger.info(f"✓ ALL DATABASE TABLES CREATED SUCCESSFULLY!")
        logger.info(f"  Total components created: {len(all_created)}")
        logger.info(f"  Components: {', '.join(all_created[:10])}{'...' if len(all_created) > 10 else ''}")
    else:
        logger.error(f"✗ DATABASE CREATION COMPLETED WITH ERRORS")
        logger.info(f"  Successfully created: {len(all_created)} components")
        logger.info(f"  Some categories failed - check logs above for details")
    
    logger.info("=" * 70)
    
    return all_success


if __name__ == "__main__":
    """Execute when script is run directly."""
    success = create_all_tables()
    sys.exit(0 if success else 1)