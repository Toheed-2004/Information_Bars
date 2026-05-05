"""
Multi-strategy stats test using read_signals for strategies 1-15.
Run: python common/stats/test_custom_1.py
"""

import warnings
import os
import time
import numpy as np
import pandas as pd
import yaml
import json
from datetime import datetime, date

warnings.filterwarnings("ignore")

from bitpredict.common.db.services import read_ohlcv, read_signals
from bitpredict.backtest import run_backtest
from bitpredict.common.stats.custom_1.custom_stats import calculate_stats
from bitpredict.common.stats.vectorbt_pro.vbt_stats import calculate_comprehensive_vbt_stats_optimized


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")
BASE_DIR    = os.path.dirname(__file__)

STRATEGY_IDS = [str(i) for i in range(1, 150)]  # strategies 1 to 2


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_serializable(obj):
    if isinstance(obj, (np.integer,)):    return int(obj)
    if isinstance(obj, (np.floating,)):   return float(obj)
    if isinstance(obj, np.ndarray):       return obj.tolist()
    if isinstance(obj, pd.Timestamp):     return obj.isoformat()
    if isinstance(obj, (datetime, date)): return obj.isoformat()
    if isinstance(obj, dict):             return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):             return [make_serializable(v) for v in obj]
    return obj


def run_test():
    cfg          = load_config(CONFIG_PATH)
    backtest_cfg = cfg["backtest"]
    bar_cfg      = cfg["bar_data"]
    min_cfg      = cfg["minute_data"]

    # ── Load bar data ─────────────────────────────────────────────────────
    print("Loading bar data...")
    df_bars = read_ohlcv(
        exchange=bar_cfg["exchange"], symbol=bar_cfg["symbol"],
        bar_type=bar_cfg["bar_type"], timeframe=bar_cfg["timeframe"],
        start_date=bar_cfg["start_date"],
    )
    df_bars["datetime"] = pd.to_datetime(df_bars["datetime"]).dt.tz_localize(None)

    df_1m = read_ohlcv(
        exchange=min_cfg["exchange"], symbol=min_cfg["symbol"],
        timeframe="1m", start_date=min_cfg["start_date"],
    )
    df_1m["datetime"] = pd.to_datetime(df_1m["datetime"]).dt.tz_localize(None)

    # ── Load signals and run backtests for all strategies ─────────────────
    print(f"\nRunning backtests for {len(STRATEGY_IDS)} strategies...")
    ledgers = {}
    portfolios = {}
    t0 = time.time()

    for strategy_id in STRATEGY_IDS:
        signals = read_signals(
            strategy_id=strategy_id,
            start_date=bar_cfg["start_date"],
            columns=["datetime", "signals"]
        )
        if signals is None or signals.empty:
            print(f"  Strategy {strategy_id}: no signals, skipping")
            continue

        signals["datetime"] = pd.to_datetime(signals["datetime"]).dt.tz_localize(None)

        try:
            portfolio, ledger = run_backtest(
                df_1m, signals, backtest_cfg,
                type="vectorbtpro", config_data=bar_cfg, df_bars=df_bars,
            )
            if ledger is not None and len(ledger) > 0:
                strategy_name = f"strategy_{strategy_id}"
                ledgers[strategy_name] = ledger
                portfolios[strategy_name] = portfolio
                print(f"  Strategy {strategy_id}: {len(ledger)} trades")
            else:
                print(f"  Strategy {strategy_id}: empty ledger, skipping")
        except Exception as e:
            print(f"  Strategy {strategy_id}: backtest failed - {e}")

    print(f"Backtests done: {time.time()-t0:.2f}s  |  {len(ledgers)} strategies with trades")

    if not ledgers:
        print("No valid ledgers, exiting.")
        return

    # ── Custom_1 stats (with internal timing) ────────────────────────────
    print("\n--- Custom_1 stats (loop-based) ---")
    t0 = time.time()
    results_loop = calculate_stats(ledgers, df_1m, use_vectorized=False)
    print(f"Total time: {time.time()-t0:.4f}s\n")

    print("--- Custom_1 stats (vectorized) ---")
    t0 = time.time()
    results_vec = calculate_stats(ledgers, df_1m, use_vectorized=True)
    print(f"Total time: {time.time()-t0:.4f}s\n")

    # Save custom_1 stats
    out_path = os.path.join(BASE_DIR, "custom_1_stats.json")
    with open(out_path, "w") as f:
        json.dump(make_serializable(results_loop), f, indent=2)
    print(f"Custom_1 stats saved to: {out_path}")
    
    # ── VBT stats for all strategies ──────────────────────────────────────
    print("\n--- VBT stats for all strategies ---")
    vbt_results = {}
    t0_vbt_total = time.time()
    
    for strategy_name in ledgers.keys():
        t0 = time.time()
        try:
            vbt_stats = calculate_comprehensive_vbt_stats_optimized(
                portfolios[strategy_name],
                ledger_input=ledgers[strategy_name],
            )
            vbt_results[strategy_name] = vbt_stats
            print(f"  {strategy_name}: {time.time()-t0:.4f}s")
        except Exception as e:
            print(f"  {strategy_name}: FAILED - {e}")
    
    print(f"VBT stats total: {time.time()-t0_vbt_total:.2f}s  |  {len(vbt_results)} strategies")
    
    # Save VBT stats
    vbt_out_path = os.path.join(BASE_DIR, "vbt_stats_all.json")
    vbt_output = {"stats": vbt_results}
    
    with open(vbt_out_path, "w") as f:
        json.dump(make_serializable(vbt_output), f, indent=2)
    print(f"VBT stats saved to: {vbt_out_path}")


if __name__ == "__main__":
    run_test()
