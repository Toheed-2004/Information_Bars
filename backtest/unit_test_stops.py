import pandas as pd
import numpy as np
import vectorbtpro as vbt
from bitpredict.backtest.vectorbtpro.vbt_backtest import VBTBacktestOptimized
from bitpredict.common.db.services.data import read_ohlcv
from bitpredict.signals import SignalGenerator
from bitpredict.common.ta.indicators import calculate_indicators


def run_test_case(name, risk_mgmt_config, df_ohlcv, df_signals):
    print(f"\n{'='*20} Testing: {name} {'='*20}")
    
    params = {
        'starting_balance': 10000,
        'take_profit': 2.0,
        'stop_loss': 1.0,
        'transaction_fee': 0.1,
        'leverage': 1.0,
        'position_size': 0.5,
        'create_ledger': True,
        'risk_mgmt': risk_mgmt_config
    }
    
    try:
        engine = VBTBacktestOptimized(df_signals, df_ohlcv, params)
        pf, ledger = engine.run()
        
        if isinstance(ledger, dict):
            # Multi-strategy case
            for s_name, s_ledger in ledger.items():
                print(f"\nLedger for {s_name}:")
                print(s_ledger.head(10))
        else:
            print("\nLedger:")
            print(ledger.head(10))
            if len(ledger) > 0:
                print("\nLast 5 trades summary:")
                print(ledger.tail(5))
                
        print(f"\nTotal Trades: {len(ledger) if not isinstance(ledger, dict) else sum(len(l) for l in ledger.values())}")
        print(f"Final Value: {pf.value.iloc[-1]}")
        
    except Exception as e:
        print(f"Error in {name}: {e}")

if __name__ == "__main__":
    df_ohlcv = read_ohlcv(exchange="bybit", symbol="eth", timeframe="1h", start_date="2025-01-01")
    minutes_data = read_ohlcv(exchange="bybit", symbol="eth", timeframe="1m", start_date="2025-01-01")
    indicator_df, metadata = calculate_indicators(
        data=df_ohlcv,
        indicators=["RSI"],
        # library='vectorbtpro'
    )
    signal_gen = SignalGenerator()
    signal_params_batch = {
        "RSI": {
            "upper_threshold": 60,
            "lower_threshold": 40,
            "exit_threshold": 50,
            "use_crossover": True
        },
    }
    df_signals, info = signal_gen.generate_signals(
        indicators_data=indicator_df, 
        signal_params_batch=signal_params_batch
    )
    df_signals = df_signals.reset_index()
    df_signals = df_signals[['datetime', 'rsi_signals']]    
    
    # 1. Default (Static SL/TP)
    run_test_case("Static SL/TP (Default)", {
        'trailing_stop': {'enabled': False},
    }, minutes_data, df_signals)
    
    # 2. Trailing Stop
    run_test_case("Trailing Stop", {
        'trailing_stop': {'enabled': True, 'activation_pct': 0.5, 'trail_pct': 0.3}
    }, minutes_data, df_signals)
    
    # 3. Time Stop
    run_test_case("Time Stop (100 minutes)", {
        'time_stop': {'enabled': True, 'max_duration': 100}
    }, minutes_data, df_signals)
    
    # 4. ATR Stop
    run_test_case("ATR Stop (2.0x ATR)", {
        'atr_stop': {'enabled': True, 'period': 14, 'multiplier': 2.0}
    }, minutes_data, df_signals)
    
    # 5. Chandelier Stop
    run_test_case("Chandelier Stop (3.0x ATR)", {
        'chandelier_stop': {'enabled': True, 'period': 22, 'multiplier': 3.0}
    }, minutes_data, df_signals)