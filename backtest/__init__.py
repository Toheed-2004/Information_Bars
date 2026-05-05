"""
Initialization file for the Backtest package.

Provides a unified interface to initialize and run different backtest 
implementations with a standard signature: (df_1m, df_signals, config).
"""

import sys
import os

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest.custom.custom_backtest import Backtest
from backtest.vectorbt_pro.vbt_backtest import VBTBacktestOptimized

def run_backtest(df_ohlcv, df_signals, config, type='vectorbtpro', config_data=None, df_bars=None, pf_object=None):
    """
    Unified entry point to run any backtest implementation.

    Args:
        df_ohlcv (pd.DataFrame): 1-minute OHLCV data.
        df_signals (pd.DataFrame): Signal data.
        config (dict): Parameters for the backtest.
        type (str): 'custom' or 'vectorbtpro'.
        config_data (dict): Optional bar metadata (exchange, symbol, timeframe, bar_type).
        df_bars (pd.DataFrame): Optional pre-loaded bar data for ATR calculation.
            If provided, avoids a DB fetch during ATR-based stop computation.
    """
    if type == 'custom':

        bt = Backtest(
            df_ohlcv=df_ohlcv,
            df_signals=df_signals,
            config = config
        )
        return bt.run()

    elif type == 'vectorbtpro':
        # VBT implementation
        bt = VBTBacktestOptimized(df_signals, df_ohlcv, config, config_data, df_bars, pf_object=pf_object)
        return bt.run()
    
    else:
        raise ValueError(f"Unknown backtest type: {type}")

__all__ = ['Backtest', 'VBTBacktestOptimized', 'run_backtest']
