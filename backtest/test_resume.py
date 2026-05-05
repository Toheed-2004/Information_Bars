"""
VBTBacktestOptimized Test Script

Loads configuration from config.yml in the same directory.
Run: python backtest/test.py

Bar data is used for signal generation (any bar type — time, dollar, volume, etc.).
Minute data (1m) is used for accurate intra-bar stop simulation.
If minute data starts early enough it also provides ATR history without a DB re-fetch.
"""

import warnings
import os
import numpy as np
import pandas as pd
import time
import yaml
from common.db.services.data import read_ohlcv
from backtest import run_backtest

warnings.filterwarnings("ignore")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "vbt_backtest.csv")

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

class TestDataGenerator:
    @staticmethod
    def generate_signals(df_bars: pd.DataFrame, frequency: int = 1) -> pd.DataFrame:
        """Generate SMA crossover signals from bar data.

        Uses bar timestamps directly — works for any bar type (regular time
        intervals or irregular dollar/volume bars).
        frequency=1 uses every bar; frequency=N subsamples every N bars.
        """
        df = df_bars.copy()
        df['sma_fast'] = df['close'].rolling(20).mean()
        df['sma_slow'] = df['close'].rolling(50).mean()

        signals_all = np.zeros(len(df))
        signals_all[df['sma_fast'] > df['sma_slow']] = 1
        signals_all[df['sma_fast'] < df['sma_slow']] = -1

        indices = np.arange(0, len(df), frequency)

        return pd.DataFrame({
            'datetime': df['datetime'].iloc[indices].values,
            'signal': signals_all[indices]
        }).dropna()

def run_unit_test():
    cfg = load_config(CONFIG_PATH)
    backtest_cfg = cfg['backtest']
    bar_cfg = cfg['bar_data']
    min_cfg = cfg['minute_data']
    test_cfg = cfg['test']

    print("\n" + "=" * 60)
    print(" TESTING VECTORBTPRO BACKTEST ".center(60, "="))
    print("=" * 60)

    # Load bar data — used for signal generation (any bar type)
    t0 = time.time()
    df_bars = read_ohlcv(
        exchange=bar_cfg['exchange'],
        symbol=bar_cfg['symbol'],
        bar_type=bar_cfg['bar_type'],
        timeframe=bar_cfg['timeframe'],
        start_date=bar_cfg['start_date']
    )
    df_bars["datetime"] = pd.to_datetime(df_bars["datetime"]).dt.tz_localize(None)
    print(f"Bar data loaded:   {time.time() - t0:.2f}s  ({len(df_bars):,} rows @ {bar_cfg['timeframe']})")

    # Load 1m minute data — used for accurate stop simulation in VBT
    # Starting earlier than bar data provides ATR history so the smart check
    # in calculate_atr_for_trailing_stop avoids a DB re-fetch.
    t0 = time.time()
    df_1m = read_ohlcv(
        exchange=min_cfg['exchange'],
        symbol=min_cfg['symbol'],
        timeframe='1m',
        start_date=min_cfg['start_date']
    )
    df_1m["datetime"] = pd.to_datetime(df_1m["datetime"]).dt.tz_localize(None)
    print(f"Minute data loaded:{time.time() - t0:.2f}s  ({len(df_1m):,} rows @ 1m)")

    # Generate signals from bar data (bar timestamps used as-is)
    t0 = time.time()
    # df_bars = df_bars[df_bars["datetime"] >= "2026-01-01"]
    df_signals = TestDataGenerator.generate_signals(df_bars[df_bars["datetime"] >= "2026-01-01"], frequency=test_cfg['signal_frequency'])
    # df_signals = TestDataGenerator.generate_signals(df_bars, frequency=test_cfg['signal_frequency'])
    df_signals["datetime"] = pd.to_datetime(df_signals["datetime"]).dt.tz_localize(None)
    df_1m = df_1m[df_1m["datetime"] >= df_signals["datetime"].iloc[0]]
    print(f"Signals generated: {time.time() - t0:.2f}s  ({len(df_signals)} signals)")
    print(df_signals)
    print(df_1m)

    ######################################################################
    print("\nComplete backtest...")
    t0 = time.time()
    pf_complete, ledger_complete = run_backtest(df_1m, df_signals, backtest_cfg, type='vectorbtpro', config_data=bar_cfg, df_bars=df_bars)
    print(f"Backtest done:     {time.time() - t0:.2f}s")
    print(f"Total return:      {pf_complete.total_return * 100:.2f}%")
    print(ledger_complete)


    ######################################################################
    print("\nPartial backtest 1...")
    # df_signals_tmp = df_signals[df_signals["datetime"] <= "2026-02-01 00:00:00"]
    # df_1m_tmp = df_1m[df_1m["datetime"] <= "2026-02-01 07:00:00"]
    df_signals_tmp = df_signals[df_signals["datetime"] <= "2026-01-31 20:00:00"]
    df_1m_tmp = df_1m[df_1m["datetime"] <= "2026-01-31 20:00:00"]
    ledger_complete.to_csv(os.path.join(os.path.dirname(__file__), "vbt_backtest.csv"), index=False)
    pf, ledger = run_backtest(df_1m_tmp, df_signals_tmp, backtest_cfg, type='vectorbtpro', config_data=bar_cfg, df_bars=df_bars)
    print(df_signals_tmp)
    print(df_1m_tmp)
    print(ledger)
    ledger.to_csv(os.path.join(os.path.dirname(__file__), "vbt_backtest1.csv"), index=False)


    ######################################################################
    print("\nPartial backtest 2...")
    last_timestamp_minute = last_timestamp_signal = pf.wrapper.index[-1]
    trades = pf.trades.records
    if len(trades) > 0 and trades.iloc[-1]['status'] == 0:
        last_closed_idx = trades.iloc[-1]['entry_idx']
        last_timestamp_minute = pf.wrapper.index[int(last_closed_idx)]

    df_signals_tmp2 = df_signals[df_signals["datetime"] > last_timestamp_signal]
    df_1m_tmp2 = df_1m[df_1m["datetime"] >= last_timestamp_minute]
    print(df_signals_tmp2)
    print(df_1m_tmp2)

    pf1, ledger = run_backtest(df_1m_tmp2, df_signals_tmp2, backtest_cfg, type='vectorbtpro', config_data=bar_cfg, df_bars=df_bars, pf_object=pf)
    print(ledger)
    ledger.to_csv(os.path.join(os.path.dirname(__file__), "vbt_backtest2.csv"), index=False)

    from common.stats import calculate_essential_stats, calculate_comprehensive_stats

    comprehensive_stats_vbt = calculate_comprehensive_stats(
        data=pf_complete,
        ledger_input=ledger_complete,
        bar_type='dollar'
    )
    comprehensive_stats_custom = calculate_comprehensive_stats(
        data=pf1,
        ledger_input=ledger,
        bar_type='dollar'
    )

# #     trade_analysis
# #   risk_adjusted
# #   risk_metrics
# #   drawdown_analysis
# #   profit_loss
# #   long_short
# #   portfolio_values
# #   exposure
# #   cash_flow
# #   time_series_analysis
# #   benchmark_analysis
# #   distribution_analysis
    keys_to_compare = ['risk_adjusted']
    compare_dicts(comprehensive_stats_vbt, comprehensive_stats_custom, keys_to_compare)


def compare_dicts(dict1, dict2, keys):
    """Compare specified top-level keys from two dictionaries"""
    for key in keys:
        if key in dict1 and key in dict2:
            print(f"\n--- {key} ---")
            for sub_key in dict1[key]:
                v1 = dict1[key].get(sub_key)
                v2 = dict2[key].get(sub_key)
                
                if v1 == v2:
                    status = "SAME"
                    diff = 0
                else:
                    status = "DIFF"
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        diff = abs(v1 - v2)
                    else:
                        diff = "N/A"
                print(f"{sub_key:30} | {str(v1):>12} | {str(v2):>12} | {status:6} | Diff: {diff}")
        else:
            print(f"Key '{key}' missing in one of the dicts")


if __name__ == "__main__":
    run_unit_test()
