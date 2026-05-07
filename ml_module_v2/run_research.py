"""
run_research.py
---------------
Entry point for the information bars ML research pipeline.

Usage
-----
    # Run all bar types (ML + signals only)
    python run_research.py --no-backtest

    # Run specific bar types
    python run_research.py --bar-types dollar_1m dollar_tick

    # Full run including VBT backtest
    python run_research.py

    # Override config
    python run_research.py --config ml/ml_module/config/ml_config.yaml
"""
from __future__ import annotations
import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)
import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from ml_module.pipeline import Pipeline
from ml_module.utils.helpers import load_config, get_logger, save_json
from ml_module.bridge.metrics import compute_backtest_metrics

logger = get_logger("run_research")

CONFIG_PATH  = Path(__file__).parent / "ml_module" / "config" / "ml_config.yaml"
OUTPUT_DIR   = Path(__file__).parent / "outputs"
OHLCV_1M     = Path(__file__).parent.parent / "data" / "raw_data" / "1minute_ohlcv_bars.csv"


# ---------------------------------------------------------------------------
# VBT backtest
# ---------------------------------------------------------------------------

def run_vbt_backtest(sig_path: Path, ohlcv_path: Path, bar_type: str,
                     output_dir: Path, asset: str):
    """Run VBT backtest on exported signal CSV."""
    try:
        import vectorbtpro as vbt
        import pandas as pd
        from ml_module.bridge.metrics import compute_backtest_metrics
    except ImportError:
        logger.warning("vectorbtpro not installed — skipping backtest.")
        return None, None

    sig_df = pd.read_csv(sig_path)
    sig_df["datetime"] = pd.to_datetime(sig_df["datetime"], utc=True, errors="coerce")
    sig_df = sig_df.dropna(subset=["datetime"])

    ohlcv = pd.read_csv(ohlcv_path)
    ohlcv["datetime"] = pd.to_datetime(ohlcv["datetime"], utc=True, errors="coerce")
    ohlcv = ohlcv.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    logger.info("[%s] Running VBT backtest (%d signals)…", bar_type, len(sig_df))

    sig_series = sig_df.set_index("datetime")["signals"]

    entries  = sig_series.reindex(ohlcv.index, fill_value=0) ==  1
    exits    = sig_series.reindex(ohlcv.index, fill_value=0) == -1

    pf = vbt.Portfolio.from_signals(
        ohlcv["close"],
        entries  = entries,
        exits    = exits,
        short_entries = sig_series.reindex(ohlcv.index, fill_value=0) == -1,
        short_exits   = sig_series.reindex(ohlcv.index, fill_value=0) ==  1,
        fees     = 0.0006,
        slippage = 0.0002,
        init_cash = 10_000,
        freq      = "1min",
    )

    try:
        trades   = pf.trades.records_readable
        ledger   = _vbt_to_ledger(trades, init_cash=10_000)
        metrics  = compute_backtest_metrics(ledger)
    except Exception as e:
        logger.warning("Metrics computation failed: %s", e)
        metrics = {}
        ledger  = None

    n_trades = metrics.get("n_trades", 0)
    ret      = metrics.get("total_return_pct", float("nan"))
    sharpe   = metrics.get("sharpe", float("nan"))
    logger.info("[%s] Backtest done — %d trades, return=%.2f%%, sharpe=%.3f",
                bar_type, n_trades, ret or 0, sharpe or 0)

    save_json(metrics, output_dir / f"backtest_metrics_{asset}_{bar_type}.json")
    return pf, ledger


def _vbt_to_ledger(trades, init_cash: float = 10_000) -> "pd.DataFrame":
    import pandas as pd
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    t = trades.copy()
    t.columns = [c.lower().replace(" ","_") for c in t.columns]
    ret_col  = next((c for c in t.columns if "return" in c), None)
    en_col   = next((c for c in t.columns if "entry" in c and "time" in c), None)
    ex_col   = next((c for c in t.columns if "exit"  in c and "time" in c), None)
    if ret_col is None:
        return pd.DataFrame()
    t["account_return_pct"] = t[ret_col] * 100
    t["cum_account_return"] = (1 + t[ret_col]).cumprod().sub(1).mul(100)
    t["balance"]            = init_cash * (1 + t[ret_col]).cumprod()
    if en_col: t["entry_datetime"] = t[en_col]
    if ex_col: t["exit_datetime"]  = t[ex_col]
    return t


# ---------------------------------------------------------------------------
# Research summary
# ---------------------------------------------------------------------------

def print_summary(results: dict, backtest_metrics: dict):
    """Print final research table."""
    header = f"{'Bar Type':<20} {'Bars':>6} {'Feats':>6} {'Folds':>6} {'Acc':>7} {'MCC':>7} {'Return%':>9} {'Sharpe':>8} {'Trades':>7} {'WinRate%':>9}"
    print("\n" + "="*90)
    print("  RESEARCH SUMMARY")
    print("="*90)
    print(header)
    print("-"*90)

    for bar_type, res in results.items():
        if "error" in res:
            print(f"  {bar_type:<20} ERROR: {res['error']}")
            continue
        wf = res.get("wf_summary", {})
        bm = backtest_metrics.get(bar_type, {})
        print(
            f"  {bar_type:<20}"
            f" {res.get('n_bars',0):>6}"
            f" {res.get('n_features',0):>6}"
            f" {wf.get('n_folds',0):>6}"
            f" {wf.get('accuracy_mean',0):>7.3f}"
            f" {wf.get('mcc_mean',0):>+7.3f}"
            f" {bm.get('total_return_pct') or 0:>9.2f}"
            f" {bm.get('sharpe') or 0:>8.3f}"
            f" {bm.get('n_trades') or 0:>7}"
            f" {bm.get('win_rate_pct') or 0:>9.2f}"
        )
    print("="*90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Information Bars ML Research Pipeline")
    p.add_argument("--bar-types", nargs="*", default=None,
                   help="Bar types to run. Default: all in config.")
    p.add_argument("--no-backtest", action="store_true",
                   help="Skip VBT backtest (ML + signals only).")
    p.add_argument("--config", default=str(CONFIG_PATH),
                   help=f"Config YAML path. Default: {CONFIG_PATH}")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help=f"Output directory. Default: {OUTPUT_DIR}")
    p.add_argument("--ohlcv-1m", default=str(OHLCV_1M),
                   help="Path to 1-minute OHLCV CSV for backtesting.")
    return p.parse_args()


def main():
    args    = parse_args()
    cfg     = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    asset   = cfg["data"].get("asset", "btc")

    # Filter bar types if specified
    if args.bar_types:
        cfg["data"]["bar_files"] = {
            k: v for k, v in cfg["data"]["bar_files"].items()
            if k in args.bar_types
        }
        missing = [t for t in args.bar_types
                   if t not in cfg["data"]["bar_files"]]
        if missing:
            logger.warning("Unknown bar types: %s", missing)

    cfg["output_dir"] = str(out_dir)

    # ── STEP 1: ML Pipeline ───────────────────────────────────────────
    logger.info("="*60)
    logger.info("  STEP 1 / 2 — ML Pipeline  [%s]", asset)
    logger.info("="*60)

    pipe    = Pipeline(cfg)
    results = pipe.run_all()

    # ── STEP 2: Backtest ──────────────────────────────────────────────
    backtest_metrics: dict = {}

    if not args.no_backtest:
        logger.info("="*60)
        logger.info("  STEP 2 / 2 — Backtest  [%s]", asset)
        logger.info("="*60)

        ohlcv_path = Path(args.ohlcv_1m)
        if not ohlcv_path.exists():
            logger.error("OHLCV file not found: %s  — skipping backtest.", ohlcv_path)
        else:
            for bar_type in results:
                if "error" in results[bar_type]:
                    continue
                sig_path = out_dir / f"signals_{asset}_{bar_type}.csv"
                if not sig_path.exists():
                    logger.warning("[%s] Signal file missing — skipping backtest.", bar_type)
                    continue
                try:
                    _, ledger = run_vbt_backtest(
                        sig_path, ohlcv_path, bar_type, out_dir, asset
                    )
                    bm_path = out_dir / f"backtest_metrics_{asset}_{bar_type}.json"
                    if bm_path.exists():
                        import json
                        backtest_metrics[bar_type] = json.load(open(bm_path))
                except Exception as e:
                    logger.error("[%s] Backtest failed: %s", bar_type, e)

    # ── Summary ───────────────────────────────────────────────────────
    print_summary(results, backtest_metrics)


if __name__ == "__main__":
    main()
