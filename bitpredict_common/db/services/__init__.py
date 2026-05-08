"""
Database Services Package

This package provides high-level database operations for different data types:
- OHLCV data management
- Bar data operations (volume, volatility, dollar, tick, time)
- State management for bar processing
- Market regime calculations
- Blockchain data operations

All commonly used functions are re-exported here for convenient imports.

Usage:
    Instead of:
        from bitpredict.common.db.services.bars import get_recent_bars
        from bitpredict.common.db.services.data import update_bar_state
    
    You can now do:
        from bitpredict.common.db.services import get_recent_bars, update_bar_state
"""

# ============================================================================
# DATA OPERATIONS - from data.py
# ============================================================================
from bitpredict.common.db.services.data import (
    insert_ohlcv,
    read_ohlcv,
    get_ohlcv_last_timestamp,
    insert_tick_data,
    read_tick_data,
    insert_volume_bar,
    insert_volatility_bar,
    insert_dollar_bar,
    insert_standard_bar,
    get_bar_state,
    update_bar_state,
    get_recent_bars,
    get_historical_bars_with_regime,
    get_bars_without_regime,
    update_bars_with_regime,
    get_total_bar_count,
    get_recent_bars_for_regime,
    batch_insert_bars,
    get_ohlcv_last_datetime
)

# ============================================================================
# METADATA OPERATIONS - from meta.py
# ============================================================================
from bitpredict.common.db.services.meta import (
    get_time_bars_meta,
    get_blockchain_meta,
    get_macro_meta,
    get_custom_bar_meta,
    get_tick_meta,
)
# ============================================================================
# Strategies OPERATIONS - from strategies.py
# ============================================================================
from bitpredict.common.db.services.strategies import (
    upsert_strategy_metadata,
    read_strategy_metadata,
    get_strategy_metadata_by_id,
    upsert_strategy_config,
    read_strategy_config,
    save_strategy,
    
)

#=============================================================================
# Simulator
#=============================================================================
from bitpredict.common.db.services.simulator import(
    read_simulator_ledger,
    upsert_simulator_ledgers,
    get_simulator_last_trade,
    get_simulator_open_trades,
    get_simulator_all_open_trades,
    get_portfolio_object,
    read_simulator_graphs,
    upsert_simulator_graphs,
    upsert_simulator_graphs_daily,
    read_simulator_graphs_daily,
    save_portfolio_object,
    read_simulator_analytics,
    upsert_simulator_analytics,
    update_strategies
)


from bitpredict.common.db.services.backtest import(
    read_backtest_ledger,
    upsert_backtest_ledgers,
    get_backtest_last_trade,
    get_backtest_open_trades,
    get_backtest_all_open_trades,
    read_backtest_graphs,
    upsert_backtest_graphs,
    upsert_backtest_graphs_daily,
    read_backtest_graphs_daily,
    read_backtest_analytics,
    upsert_backtest_analytics,
)


# ============================================================================
# SIGNALS
# ============================================================================
from bitpredict.common.db.services.signals import(
    read_signals,
    get_last_signal,
    upsert_signals
)

# ============================================================================
# Regime_analysis
# ============================================================================
from bitpredict.common.db.services.regime_analysis import upsert_into_regime_analysis_col

# ============================================================================
# DEFINE PUBLIC API
# ============================================================================
__all__ = [
    # Data operations    
    "get_bar_state",
    "update_bar_state",
    "get_recent_bars",
    "get_historical_bars_with_regime",
    "get_bars_without_regime",
    "update_bars_with_regime",
    "get_total_bar_count",
    "get_recent_bars_for_regime",
    "batch_insert_custom_bars",
    "insert_ohlcv",
    "read_ohlcv",
    "get_ohlcv_last_timestamp",
    "insert_tick_data",
    "read_tick_data",
    "insert_volume_bar",
    "insert_volatility_bar",
    "insert_dollar_bar",
    "insert_standard_bar",
    "get_ohlcv_last_datetime"
    
    # Metadata operations
    "get_time_bars_meta",
    "get_blockchain_meta",
    "get_macro_meta",
    "get_tick_meta",
    "get_custom_bar_meta"

    # Strategies operations
    "read_simulator_ledger",
    "upsert_simulator_ledgers",
    "get_simulator_last_trade",
    "get_simulator_open_trades",
    "get_simulator_all_open_trades",
    "get_portfolio_object",
    "read_simulator_graphs",
    "upsert_simulator_graphs",
    "upsert_simulator_graphs_daily",
    "read_simulator_graphs_daily",
    "save_portfolio_object",
    "read_simulator_analytics",
    "upsert_simulator_analytics",
    "update_strategies",


    # Backtest
    "read_backtest_ledger",
    "upsert_backtest_ledgers",
    "get_backtest_last_trade",
    "get_backtest_open_trades",
    "get_backtest_all_open_trades",
    "read_backtest_graphs",
    "upsert_backtest_graphs",
    "upsert_backtest_graphs_daily",
    "read_backtest_graphs_daily",
    "read_backtest_analytics",
    "upsert_backtest_analytics",

    # Signals
    "get_last_trade",
    "get_open_trades",
    "get_all_open_trades",

    # Regime Analysis
    "upsert_into_regime_analysis_col"

]
