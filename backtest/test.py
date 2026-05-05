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
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "vbt_backtest_resume.csv")
BARS_PATH = os.path.join(os.path.dirname(__file__), "dollar_bars.csv")

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
    print (df_bars)

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
    print(f"Signals generated: {time.time() - t0:.2f}s  ({len(df_signals)} signals)")
    print(df_signals)
    # Run backtest — df_1m for VBT simulation, signals from bar data
    print("\nRunning backtest...")
    t0 = time.time()
    pf, ledger = run_backtest(df_1m, df_signals, backtest_cfg, type='vectorbtpro', config_data=bar_cfg, df_bars=df_bars)
    print(f"Backtest done:     {time.time() - t0:.2f}s")
    print(f"Total return:      {pf.total_return * 100:.2f}%")

    if ledger is not None and not ledger.empty:
        ledger.to_csv(LEDGER_PATH, index=False)
        df_bars.to_csv(BARS_PATH, index=False)
        # print(f"Ledger saved:      {filename}  ({len(ledger)} trades)")
        # print(f"\n{ledger.head(10).to_string()}")


if __name__ == "__main__":
    run_unit_test()
