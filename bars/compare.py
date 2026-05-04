"""
bars/compare.py
---------------
Compare minute-level vs tick-level information bars.

Wraps the analysis logic from compare_bars.py into a clean CLI entry point.
Run this after generating bars with bars/main.py.

Usage
-----
    python bars/compare.py \\
        --types dollar volatility \\
        --time-bars \\
        --minute-csv data/raw_data/1minute_btcusdt_2024.csv \\
        --tick-csv   data/raw_data/btcusdt_aggTrades_2024.csv \\
        --output-dir data/comparison_results
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logging import setup_logging, get_logger
from bars.cli_control import build_compare_parser
from bars.bar_types import ALL_BAR_TYPE_NAMES

logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    parser = build_compare_parser()
    args   = parser.parse_args()

    bar_types  = ALL_BAR_TYPE_NAMES if args.types == ["all"] else args.types
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate compare_bars.py — shipped in the repo root or bars_analysis/
    compare_script = Path(__file__).parent / "compare_bars.py"
    if not compare_script.exists():
        compare_script = (
            Path(__file__).resolve().parents[1]
            / "data" / "raw_data" / "compare_bars.py"
        )

    if compare_script.exists():
        # Run compare_bars.py programmatically via its _run_one function
        import importlib.util
        spec = importlib.util.spec_from_file_location("compare_bars", compare_script)
        cb   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cb)

        for bar_type in bar_types:
            logger.info("Comparing: %s", bar_type)
            t0 = time.time()
            try:
                out = output_dir / bar_type
                out.mkdir(exist_ok=True)
                cb._run_one(
                    bar_type,
                    data_dir      = Path(args.minute_csv).parent,
                    out_dir       = output_dir,
                    with_time_bars = args.time_bars,
                    exchange      = args.exchange,
                    symbol        = args.symbol,
                )
                logger.info("Done %-12s in %.1fs", bar_type, time.time() - t0)
            except Exception:
                logger.exception("Error comparing %s", bar_type)
    else:
        # Fallback: load bar CSVs directly and compute basic stats
        from common.data_loader import load_bars_csv
        import pandas as pd

        for bar_type in bar_types:
            logger.info("Basic stats for: %s", bar_type)
            proc = Path("data/processed_bars")
            min_path  = proc / f"{args.exchange}_{args.symbol}_{bar_type}_minute_bars.csv"
            tick_path = proc / f"{args.exchange}_{args.symbol}_{bar_type}_tick_bars.csv"

            results = {}
            for label, path in [("minute", min_path), ("tick", tick_path)]:
                if path.exists():
                    df = load_bars_csv(path)
                    if "bar_return" in df.columns:
                        r = df["bar_return"].dropna()
                        results[label] = {
                            "n_bars":   len(df),
                            "mean_ret": float(r.mean()),
                            "std_ret":  float(r.std()),
                            "kurtosis": float(r.kurtosis()),
                        }
                        logger.info(
                            "  %-6s | bars=%d | kurt=%.3f",
                            label, len(df), results[label]["kurtosis"],
                        )
                else:
                    logger.warning("  %s bars not found at %s", label, path)

            if results:
                import json
                out_file = output_dir / f"{bar_type}_stats.json"
                with open(out_file, "w") as fh:
                    json.dump(results, fh, indent=2)
                logger.info("  Stats written → %s", out_file)


if __name__ == "__main__":
    main()
