"""
Master script to create all database tables.

This script automatically imports and executes all table creation functions
from the models package to set up the complete database schema.
"""

import sys
from pathlib import Path

from bitpredict.common.logging import get_logger
from bitpredict.common.db.config import get_engine, check_connection
from bitpredict.common.constants import DATA_SCHEMA, ALLOWED_BAR_TYPES
# Import all table creation functions from the models package
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
    ensure_ledger_table,
    ensure_signal_table,
    ensure_simulator_analytics_table,
    ensure_simulator_graphs_table,
    ensure_simulator_graphs_daily_table,

    # Market Regime
    ensure_regime_state_table,

    # Regime Analysis
    ensure_regime_analysis_column_in_analytics_table
)
from bitpredict.common.db.models.data import ensure_bar_stats_table, ensure_ohlcv_table
from bitpredict.common.db.utils import ensure_schema

logger = get_logger(__name__)


def create_all_tables():
    """
    Create all database tables by calling all ensure_*_table functions.
    
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
    logger.info("=" * 70)
    
    # List of bar types to create tables for
    
    try:

        ensure_schema(engine, DATA_SCHEMA)

        # # 1. Metadata tables (dependencies for other tables)
        # logger.info("\n[1/7] Creating metadata tables...")
        # ensure_symbols_table(engine)
        # logger.info("✓ Symbols table created")
        
        # ensure_time_bars_table(engine)
        # logger.info("✓ Time bars table created")
        
        # ensure_tick_table(engine)
        # logger.info("✓ Tick table created")
        
        # ensure_custom_bars_table(engine)
        # logger.info("✓ Custom bars table created")
        
        # ensure_blockchain_table(engine)
        # logger.info("✓ Blockchain table created")
        
        # ensure_macro_table(engine)
        # logger.info("✓ Macro table created")
        
        # 2. Data tables (bars and state tracking)

        logger.info("\n[2/7] Creating data tables...")

        ensure_ohlcv_table(engine, schema_name=DATA_SCHEMA, table_name="time")
       
        # Create bars tables for each bar type
        for bar_type in ALLOWED_BAR_TYPES:
            try:
                ensure_bars_table(bar_type)
                logger.info(f"✓ {bar_type.capitalize()} bars table created")
            except ValueError as e:
                logger.warning(f"⚠ Skipped {bar_type} bars table: {e}")
        
        ensure_state_table()
        logger.info("✓ State table created")

        ensure_bar_stats_table()
        logger.info("✓ Bar stats table created")
        
        # # 3. Strategies tables
        # logger.info("\n[3/7] Creating strategies tables...")
        # ensure_strategies_metadata_table(engine)
        # logger.info("✓ Strategies metadata table created")
        
        # ensure_strategies_configs_table(engine)
        # logger.info("✓ Strategies configs table created")
        
        # ensure_strategies_training_table(engine)
        # logger.info("✓ Strategies training table created")
        
        # # 4. Simulator tables
        # logger.info("\n[4/7] Creating simulator tables...")
        # ensure_ledger_table(engine)
        # logger.info("✓ Ledger table created")
        
        # ensure_simulator_analytics_table(engine)
        # logger.info("✓ Simulator analytics table created")
        
        # ensure_simulator_graphs_table(engine)
        # logger.info("✓ Simulator graphs table created")
        
        # ensure_simulator_graphs_daily_table(engine)
        # logger.info("✓ Simulator graphs daily table created")
        
        # # 5. Signals tables
        # logger.info("\n[5/7] Creating signals tables...")
        # ensure_signal_table(engine)
        # logger.info("✓ Signals table created")
        
        # 6. Market regime tables
        logger.info("\n[6/7] Creating market regime tables...")
        ensure_regime_state_table()
        logger.info("✓ Market regime state table created")
        
        # # 7. Regime analysis tables (adds column to existing tables)
        # logger.info("\n[7/7] Creating regime analysis columns...")
        # ensure_regime_analysis_column_in_analytics_table(engine)
        # logger.info("✓ Regime analysis column added to analytics table")
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ All database tables created successfully!")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Error creating tables: {e}", exc_info=True)
        logger.error("=" * 70)
        return False


if __name__ == "__main__":
    """Execute when script is run directly."""
    success = create_all_tables()
    sys.exit(0 if success else 1)

