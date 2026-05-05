"""
ml/run_research.py
------------------
Single entry-point that chains the full research pipeline:

    ML Pipeline  →  Signal CSV  →  VBT Backtest  →  Metrics Report

Run from the project root (Information_Bars_Research/):

    python ml/run_research.py
    python ml/run_research.py --bar-types dollar volume
    python ml/run_research.py --bar-types dollar --no-backtest

Outputs (all written to ml/outputs/):
    signals_{bar_type}.csv              ← VBT-compatible signal file
    wf_predictions_{bar_type}.csv       ← walk-forward OOS predictions
    diagnostics_{bar_type}.json         ← ML metrics per bar type
    backtest_metrics_{bar_type}.json    ← Sharpe, Sortino, drawdown, trades…
    equity_curve_{bar_type}.csv         ← full equity time series
    research_report.csv                 ← one row per bar type (paper table)
    pipeline_summary.json               ← ML summary across all bar types

Structure
---------
Information_Bars_Research/          ← project root  (run from here)
├── data/
│   ├── processed_bars/             ← bar CSVs (input to ML pipeline)
│   └── raw_data/
│       └── 1minute_ohlcv_bars.csv  ← 1m data (input to VBT backtest)
├── ml/
│   ├── run_research.py             ← THIS FILE
│   ├── ml_module/
│   │   ├── config/ml_config.yaml
│   │   ├── pipeline.py
│   │   ├── backtest_bridge/
│   │   │   ├── signal_exporter.py
│   │   │   └── backtest_reporter.py
│   │   └── ...
│   └── outputs/                    ← all outputs written here
└── ...
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # Information_Bars_Research/
ML_DIR = ROOT / "ml"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ML_DIR))

# Register bitpredict as an alias for the project root so the backtest
# module resolves all its  "from bitpredict.X import ..."  without changes.
from ml_module.pipeline import MLPipeline
from ml_module.backtest_bridge.backtest_reporter import BacktestReporter
from ml_module.utils.helpers import get_logger, load_config

logger = get_logger("run_research")

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIG_PATH = ML_DIR / "ml_module" / "config" / "ml_config.yaml"
OUTPUT_DIR = ML_DIR / "outputs"
OHLCV_1M_PATH = ROOT / "data" / "raw_data" / "1minute_ohlcv_bars.csv"
BAR_DATA_DIR = ROOT / "data" / "processed_bars"

                                                                                         
# =============================================================================
# VBT Backtest runner (standalone CSV-based, no DB)
# =============================================================================


def run_vbt_backtest(
    df_signals: pd.DataFrame,
    ohlcv_1m_path: Path,
    backtest_cfg: dict,
    bar_type: str,
):
    """
    Run VBTBacktestOptimized using the local 1-minute OHLCV CSV.

    Parameters
    ----------
    df_signals    : Signal DataFrame with columns [datetime, signals].
    ohlcv_1m_path : Path to the 1-minute OHLCV CSV file.
    backtest_cfg  : Backtest parameter dict (from ml_config.yaml backtest section
                    or a separately supplied dict).
    bar_type      : Used only for logging.

    Returns
    -------
    (pf, ledger) — VBT Portfolio object and trade ledger DataFrame.
    Returns (None, None) if VBT is not installed or loading fails.
    """
    # Load 1-minute OHLCV
    logger.info("[%s] Loading 1-minute OHLCV from %s …", bar_type, ohlcv_1m_path.name)
    if not ohlcv_1m_path.exists():
        logger.error("1m OHLCV file not found: %s", ohlcv_1m_path)
        return None, None

    df_1m = _load_ohlcv_1m(ohlcv_1m_path)
    if df_1m is None or df_1m.empty:
        return None, None

    # Check VBT is importable before attempting backtest
    try:
        import vectorbtpro  # noqa
    except ImportError:
        logger.warning(
            "VBT Pro not installed — skipping backtest for %s. "
            "Install vectorbtpro to enable full backtest.",
            bar_type,
        )
        return None, None

    try:
        # Use the project backtest module directly (bitpredict shim is already active)
        from backtest import run_backtest

        logger.info(
            "[%s] Running VBT backtest (%d signals) …", bar_type, len(df_signals)
        )
        pf, ledger = run_backtest(
            df_ohlcv=df_1m,
            df_signals=df_signals,
            config=backtest_cfg,
            type="vectorbtpro",
        )
        logger.info(
            "[%s] Backtest done — %d trades, total return %.2f%%",
            bar_type,
            len(ledger) if ledger is not None else 0,
            float(pf.total_return * 100) if pf is not None else 0,
        )
        return pf, ledger
    except Exception as e:
        logger.error("[%s] VBT backtest failed: %s", bar_type, e, exc_info=True)
        return None, None


def _load_ohlcv_1m(path: Path) -> pd.DataFrame | None:
    """
    Load the 1-minute OHLCV CSV into the format VBTBacktestOptimized expects:
      columns: datetime, open, high, low, close, volume
      datetime: timezone-aware UTC
    """
    try:
        df = pd.read_csv(path, low_memory=False)

        # Find and normalise datetime column
        dt_candidates = [
            c
            for c in df.columns
            if c.lower() in ("datetime", "timestamp", "time", "date")
        ]
        if not dt_candidates:
            logger.error("No datetime column found in %s", path.name)
            return None
        dt_col = dt_candidates[0]
        if dt_col != "datetime":
            df.rename(columns={dt_col: "datetime"}, inplace=True)

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df.dropna(subset=["datetime"], inplace=True)

        # Drop spurious index column if present
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("1m OHLCV missing columns: %s", missing)
            return None

        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(
            "1m OHLCV loaded: %d rows (%s → %s)",
            len(df),
            str(df["datetime"].iloc[0])[:10],
            str(df["datetime"].iloc[-1])[:10],
        )
        return df

    except Exception as e:
        logger.error("Failed to load 1m OHLCV: %s", e)
        return None


# =============================================================================
# Default backtest config (used when no separate backtest YAML is supplied)
# =============================================================================

DEFAULT_BACKTEST_CFG = {
    "starting_balance": 10_000,
    "position_size": 1.0,  # 100% of balance per trade
    "leverage": 1.0,
    "transaction_fee": 0.05,  # 0.05%
    "slippage": 0.0,
    "create_ledger": True,
    "direction": "both",  # long_only | short_only | both
    "zero_signal_mode": "close_position",
    "risk_management": {
        "static": {
            "enabled": True,
            "take_profit": 2.0,  # %
            "stop_loss": 1.0,  # %
        },
        "time_stop": {
            "enabled": False,
            "max_duration": "12h",
        },
        "atr_stop": {"enabled": False},
        "chandelier_stop": {"enabled": False},
        "trailing_stop": {"enabled": False},
    },
}


# =============================================================================
# Main pipeline runner
# =============================================================================


def run_research(
    bar_types: list[str] | None = None,
    run_backtest: bool = True,
    backtest_cfg: dict | None = None,
    config_path: Path = CONFIG_PATH,
    output_dir: Path = OUTPUT_DIR,
    ohlcv_1m_path: Path = OHLCV_1M_PATH,
    bar_data_dir: Path = BAR_DATA_DIR,
) -> dict:
    """
    Full research pipeline: ML → signals → backtest → metrics.

    Parameters
    ----------
    bar_types     : List of bar type names to run. None = all in config.
    run_backtest  : If False, skip VBT backtest (ML + signals only).
    backtest_cfg  : Override backtest parameters. None = DEFAULT_BACKTEST_CFG.
    config_path   : Path to ml_config.yaml.
    output_dir    : Where all output files are written.
    ohlcv_1m_path : Path to the 1-minute OHLCV CSV for VBT.
    bar_data_dir  : Directory containing bar CSV files.

    Returns
    -------
    Dict mapping bar_type → {ml_results, backtest_metrics}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(config_path)

    # Override paths to be relative to project root
    cfg["data"]["bar_data_dir"] = str(bar_data_dir)
    cfg["backtest"]["output_dir"] = str(output_dir)

    # Filter bar types if requested
    if bar_types:
        cfg["data"]["bar_files"] = {
            k: v for k, v in cfg["data"]["bar_files"].items() if k in bar_types
        }
        if not cfg["data"]["bar_files"]:
            logger.error(
                "None of the requested bar types found in config: %s", bar_types
            )
            return {}

    bt_cfg = backtest_cfg or DEFAULT_BACKTEST_CFG

    # ── Run ML pipeline ───────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  STEP 1 / 2 — ML Pipeline")
    logger.info("=" * 65)

    pipeline = MLPipeline(cfg)
    ml_results = pipeline.run_all()

    # ── For each bar type: backtest + report ──────────────────────────────────
    reporter = BacktestReporter(output_dir=output_dir)
    all_results = {}

    logger.info("=" * 65)
    logger.info("  STEP 2 / 2 — Backtest + Metrics")
    logger.info("=" * 65)

    for bar_type, ml_result in ml_results.items():

        result_entry = {"ml": ml_result, "backtest_metrics": None}

        if "error" in ml_result:
            logger.warning("[%s] ML failed — skipping backtest.", bar_type)
            all_results[bar_type] = result_entry
            continue

        # Load the signal CSV just exported by the ML pipeline
        sig_path = output_dir / f"signals_unknown_{bar_type}.csv"
        if not sig_path.exists():
            # Also try without 'unknown_'
            sig_path = output_dir / f"signals_{bar_type}.csv"

        if not sig_path.exists():
            logger.warning(
                "[%s] Signal CSV not found at %s — skipping backtest.",
                bar_type,
                sig_path,
            )
            all_results[bar_type] = result_entry
            continue

        df_signals = pd.read_csv(sig_path)
        df_signals["datetime"] = pd.to_datetime(
            df_signals["datetime"], utc=True, errors="coerce"
        )

        if not run_backtest:
            logger.info("[%s] Backtest skipped (--no-backtest).", bar_type)
            all_results[bar_type] = result_entry
            continue

        # ── VBT backtest ─────────────────────────────────────────────────────
        pf, ledger = run_vbt_backtest(df_signals, ohlcv_1m_path, bt_cfg, bar_type)

        if pf is None:
            logger.warning(
                "[%s] Backtest returned no portfolio — metrics skipped.", bar_type
            )
            all_results[bar_type] = result_entry
            continue

        # ── Compute and save all metrics ──────────────────────────────────────
        starting_balance = bt_cfg.get("starting_balance", 10_000)
        bt_metrics = reporter.compute_and_save(
            pf=pf,
            ledger=ledger,
            bar_type=bar_type,
            df_signals=df_signals,
            starting_balance=starting_balance,
        )

        # Save ledger CSV alongside the other outputs
        if ledger is not None and not ledger.empty:
            ledger_path = output_dir / f"ledger_{bar_type}.csv"
            ledger.to_csv(ledger_path, index=False)
            logger.info(
                "[%s] Ledger → %s  (%d trades)", bar_type, ledger_path.name, len(ledger)
            )

        result_entry["backtest_metrics"] = bt_metrics
        all_results[bar_type] = result_entry

    # ── Print final summary table ─────────────────────────────────────────────
    _print_summary(all_results)

    return all_results


# =============================================================================
# Summary printer
# =============================================================================


def _print_summary(all_results: dict) -> None:
    print("\n" + "=" * 85)
    print("  RESEARCH SUMMARY")
    print("=" * 85)
    print(
        f"  {'Bar Type':<12} {'Bars':>6} {'WF-Acc':>8} {'Return%':>8} "
        f"{'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'Trades':>7} {'WinRate%':>9}"
    )
    print("  " + "-" * 83)

    for bar_type, res in all_results.items():
        ml = res.get("ml", {})
        bm = res.get("backtest_metrics") or {}
        wf = ml.get("wf_summary", {})

        def _f(v, fmt=".3f"):
            return f"{v:{fmt}}" if v is not None and v == v else "  N/A  "

        print(
            f"  {bar_type:<12} "
            f"{ml.get('n_bars', 0):>6} "
            f"{wf.get('accuracy_mean', 0):>8.3f} "
            f"{_f(bm.get('total_return_pct'), '>8.2f'):>8} "
            f"{_f(bm.get('sharpe_ratio'),     '>7.3f'):>7} "
            f"{_f(bm.get('sortino_ratio'),    '>8.3f'):>8} "
            f"{_f(bm.get('max_drawdown_pct'), '>8.2f'):>8} "
            f"{bm.get('total_trades', 0):>7} "
            f"{_f(bm.get('win_rate_pct'),     '>9.2f'):>9}"
        )

    report_path = OUTPUT_DIR / "research_report.csv"
    if report_path.exists():
        print(f"\n  Full report → {report_path}")
    print("=" * 85 + "\n")


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full research pipeline: ML → signals → backtest → metrics"
    )
    p.add_argument(
        "--bar-types",
        nargs="*",
        default=None,
        metavar="BAR_TYPE",
        help="Bar types to run (e.g. dollar volume). Default: all in config.",
    )
    p.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip VBT backtest — run ML pipeline and signal export only.",
    )
    p.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help=f"Path to ml_config.yaml. Default: {CONFIG_PATH}",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Output directory. Default: {OUTPUT_DIR}",
    )
    p.add_argument(
        "--ohlcv-1m",
        default=str(OHLCV_1M_PATH),
        help=f"Path to 1-minute OHLCV CSV. Default: {OHLCV_1M_PATH}",
    )
    p.add_argument(
        "--tp",
        type=float,
        default=2.0,
        help="Static take-profit %% (default 2.0)",
    )
    p.add_argument(
        "--sl",
        type=float,
        default=1.0,
        help="Static stop-loss %% (default 1.0)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Build backtest config from CLI args
    bt_cfg = DEFAULT_BACKTEST_CFG.copy()
    bt_cfg["risk_management"]["static"]["take_profit"] = args.tp
    bt_cfg["risk_management"]["static"]["stop_loss"] = args.sl

    run_research(
        bar_types=args.bar_types,
        run_backtest=not args.no_backtest,
        backtest_cfg=bt_cfg,
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        ohlcv_1m_path=Path(args.ohlcv_1m),
    )
