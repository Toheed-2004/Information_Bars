import warnings
import os
import numpy as np
import pandas as pd
import time
import yaml
import json
from datetime import datetime, date

from bitpredict.common.db.services.data import read_ohlcv
from bitpredict.backtest import run_backtest
from bitpredict.common.stats.vectorbt_pro.vbt_stats import calculate_comprehensive_vbt_stats_optimized
from bitpredict.common.stats.vectorbt_pro.cache_builder import _build_vbt_cache
from bitpredict.common.stats.custom.custom_stats import calculate_comprehensive_stats
warnings.filterwarnings("ignore")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")
BASE_DIR = os.path.dirname(__file__)


def make_json_serializable(obj):
    # numpy
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pandas
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DatetimeIndex):
        return obj.astype(str).tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    # python datetime
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # dict recursion
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    # list recursion (IMPORTANT — you were missing this)
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    return obj
# ------------------------------------------------------------
# Config Loader
# ------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------
# Signal Generator
# ------------------------------------------------------------
class TestDataGenerator:
    @staticmethod
    def generate_signals(df_bars: pd.DataFrame, frequency: int = 1) -> pd.DataFrame:
        df = df_bars.copy()

        df['sma_fast'] = df['close'].rolling(20).mean()
        df['sma_slow'] = df['close'].rolling(50).mean()

        signals = np.zeros(len(df))
        signals[df['sma_fast'] > df['sma_slow']] = 1
        signals[df['sma_fast'] < df['sma_slow']] = -1

        idx = np.arange(0, len(df), frequency)

        return pd.DataFrame({
            'datetime': df['datetime'].iloc[idx].values,
            'signal': signals[idx]
        }).dropna()


# ------------------------------------------------------------
# Custom Stats (YOUR IMPLEMENTATIONS)
# ------------------------------------------------------------


# ------------------------------------------------------------
# Manual Comparison Printer
# ------------------------------------------------------------
def print_compare(name, mine, vbt):
    diff = mine - vbt
    print(f"\n{name}")
    print(f"VBT   : {vbt}")
    print(f"MINE  : {mine}")
    print(f"DIFF  : {diff}")


# ------------------------------------------------------------
# Main Test Runner
# ------------------------------------------------------------
def run_unit_test():
    cfg = load_config(CONFIG_PATH)

    backtest_cfg = cfg['backtest']
    bar_cfg = cfg['bar_data']
    min_cfg = cfg['minute_data']
    test_cfg = cfg['test']

    print("\n" + "=" * 60)
    print(" STATS CALCULATION ".center(60, "="))
    print("=" * 60)

    # ---------------- Load Data ----------------
    print("\nLoading data...")
    df_bars = read_ohlcv(
        exchange=bar_cfg['exchange'],
        symbol=bar_cfg['symbol'],
        bar_type=bar_cfg['bar_type'],
        timeframe=bar_cfg['timeframe'],
        start_date=bar_cfg['start_date']
    )
    df_bars["datetime"] = pd.to_datetime(df_bars["datetime"]).dt.tz_localize(None)

    df_1m = read_ohlcv(
        exchange=min_cfg['exchange'],
        symbol=min_cfg['symbol'],
        timeframe='1m',
        start_date=min_cfg['start_date']
    )
    df_1m["datetime"] = pd.to_datetime(df_1m["datetime"]).dt.tz_localize(None)

    # ---------------- Load Signals ----------------
    df_signals = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_signals11.csv"))
    df_signals["datetime"] = pd.to_datetime(df_signals["datetime"]).dt.tz_localize(None)

    # ---------------- Run Backtest ----------------
    print("Running backtest...")
    pf, ledger = run_backtest(
        df_1m,
        df_signals,
        backtest_cfg,
        type='vectorbtpro',
        config_data=bar_cfg,
        df_bars=df_bars
    )
    print(f"Total return: {pf.total_return * 100:.2f}%")

    # ---------------- Calculate VBT Stats ----------------
    print("\nCalculating VBT stats...")
    stats_vbt = calculate_comprehensive_vbt_stats_optimized(pf, ledger_input=ledger)

    STATS_PATH = os.path.join(BASE_DIR, "vbt_stats.json")
    with open(STATS_PATH, "w") as f:
        json.dump(make_json_serializable(stats_vbt), f, indent=2)
    print(f"VBT stats saved to: {STATS_PATH}")

    # ---------------- Calculate Custom Stats ----------------
    print("\nCalculating custom stats...")
    initial_date = pd.to_datetime(bar_cfg['start_date'])

    stats_custom = calculate_comprehensive_stats(
        ledger,
        df_bars=df_1m,
        initial_date=initial_date
    )

    CUSTOM_STATS_PATH = os.path.join(BASE_DIR, "custom_stats.json")
    with open(CUSTOM_STATS_PATH, "w") as f:
        json.dump(make_json_serializable(stats_custom), f, indent=2)
    print(f"Custom stats saved to: {CUSTOM_STATS_PATH}")

    # ---------------- Compare Key Metrics ----------------
    print("\n" + "=" * 60)
    print(" COMPARISON ".center(60, "="))
    print("=" * 60)

    vbt_sharpe    = stats_vbt['0']['risk_adjusted']['sharpe_ratio']
    custom_sharpe = stats_custom['0']['risk_adjusted']['sharpe_ratio']

    print(f"Sharpe Ratio:")
    print(f"  VBT    : {vbt_sharpe:+.5f}")
    print(f"  Custom : {custom_sharpe:+.5f}  diff: {abs(vbt_sharpe - custom_sharpe):.5f}")
    print("=" * 60)

# ------------------------------------------------------------
if __name__ == "__main__":
    run_unit_test()