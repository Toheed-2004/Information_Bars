"""
Database Models Package

This package provides DDL (Data Definition Language) functions for creating
and managing database table schemas. It includes:

- Bar table creation (volume, volatility, dollar, tick, time bars)
- State table management for bar processing
- Metadata table operations
- Strategies table operations (metadata, analytics, configs, time_series)
- TimescaleDB hypertable support

All commonly used table creation functions are re-exported here for convenient imports.

Usage:
    Instead of:
        from bitpredict.common.db.models.data import ensure_bars_table
        from bitpredict.common.db.models.meta import ensure_meta_tables
        from bitpredict.common.db.models.strategies import ensure_strategies_metadata_table
    
    You can now do:
        from bitpredict.common.db.models import (
            ensure_bars_table, 
            ensure_meta_tables,
            ensure_strategies_metadata_table
        )
"""

# ============================================================================
# BAR TABLE OPERATIONS - from data.py
# ============================================================================
from bitpredict.common.db.models.data import (
    ensure_bars_table,
    ensure_state_table,
)

# ============================================================================
# METADATA TABLE OPERATIONS - from meta.py
# ============================================================================
from bitpredict.common.db.models.meta import (
    ensure_symbols_table,
    ensure_time_bars_table,
    ensure_tick_table,
    ensure_custom_bars_table,
    ensure_blockchain_table,
    ensure_macro_table,
)

# ============================================================================
# STRATEGIES TABLE OPERATIONS - from strategies.py
# ============================================================================
from bitpredict.common.db.models.strategies import (
    ensure_strategies_metadata_table,
    ensure_strategies_configs_table,
    ensure_strategies_training_table
)

#=============================================================================
# Simulator
#=============================================================================
from bitpredict.common.db.models.simulator import (
    ensure_simulator_analytics_table,
    ensure_simulator_graphs_table,
    ensure_simulator_graphs_daily_table,
    ensure_simulator_ledger_table
)

#=============================================================================
# Backtest
#=============================================================================
from bitpredict.common.db.models.backtest import (
    ensure_backtest_ledger_table,
    ensure_backtest_analytics_table,
    ensure_backtest_graphs_table,
    ensure_backtest_graphs_daily_table,
    ensure_backtest_requests_table
)

#=============================================================================
# Signals
#=============================================================================

from bitpredict.common.db.models.signals import ensure_signal_table

#=============================================================================
# Market Regime
#=============================================================================
from bitpredict.common.db.models.market_regime import ensure_regime_state_table

#=============================================================================
# Regime Analysis
#=============================================================================
from bitpredict.common.db.models.regime_analysis import ensure_regime_analysis_column_in_analytics_table

# ============================================================================
# DEFINE PUBLIC API
# ============================================================================
__all__ = [
    # Bar table operations
    "ensure_bars_table",
    "ensure_state_table",
    
    # Metadata table operations
    "ensure_symbols_table",
    "ensure_time_bars_table",
    "ensure_tick_table",
    "ensure_custom_bars_table",
    "ensure_blockchain_table",
    "ensure_macro_table",
    
    # Strategies table operations
    "ensure_strategies_metadata_table",
    "ensure_strategies_configs_table",
    "ensure_strategies_training_table",

    # Simulator
    "ensure_signal_table",
    "ensure_simulator_analytics_table",
    "ensure_simulator_graphs_table",
    "ensure_simulator_graphs_daily_table",
    "ensure_simulator_ledger_table",

    # Backtest
    "ensure_backtest_ledger_table",
    "ensure_backtest_analytics_table",
    "ensure_backtest_graphs_table",
    "ensure_backtest_graphs_daily_table",
    "ensure_backtest_requests_table",

    # Market Regime
    "ensure_regime_state_table",

    # Regime Analysis
    "ensure_regime_analysis_column_in_analytics_table"
    
]