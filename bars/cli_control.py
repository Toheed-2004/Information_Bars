"""
bars/cli_control.py
-------------------
CLI argument definitions for the information bar pipeline.
"""
import argparse
from bars.bar_types import ALL_BAR_TYPE_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python bars/main.py",
        description=(
            "Generate information bars from minute OHLCV or raw tick data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # All minute bar types (if --types omitted, all six are processed)
  python bars/main.py --source minute \
      --minute-csv data/raw_data/1minute_ohlcv.csv

  # All tick bar types — no minute CSV needed
  python bars/main.py --source tick \
      --tick-csv data/raw_data/merged_tickdata.csv

  # Specific bar types
  python bars/main.py --source minute --types dollar volatility \
      --minute-csv data/raw_data/1minute_ohlcv.csv

  # Custom output directory
  python bars/main.py --source tick --types dollar \
      --tick-csv data/raw_data/merged_tickdata.csv \
      --output-dir data/processed_bars/run_01
""",
    )

    parser.add_argument(
        "--source", choices=["minute", "tick"], default="minute",
        help="Data resolution. 'minute' = 1-min OHLCV CSV. 'tick' = aggTrades CSV. (default: minute)",
    )
    parser.add_argument(
        "--types", nargs="+", default=None, metavar="TYPE",
        help=(
            f"Bar type(s) to generate. Choices: {', '.join(ALL_BAR_TYPE_NAMES)}, all. "
            "If omitted, all six types are generated."
        ),
    )
    parser.add_argument(
        "--minute-csv", default=None, metavar="PATH",
        help="Path to 1-minute OHLCV CSV. Required for --source minute.",
    )
    parser.add_argument(
        "--tick-csv", default=None, metavar="PATH",
        help=(
            "Path to merged aggTrades CSV (Binance format, no header). "
            "Required for --source tick. Calibration also uses tick data."
        ),
    )
    parser.add_argument(
        "--output-dir", default="data/processed_bars", metavar="DIR",
        help="Output directory for bar CSVs and state files. (default: data/processed_bars)",
    )
    parser.add_argument(
        "--exchange", default="binance",
        help="Exchange label for output file names. (default: binance)",
    )
    parser.add_argument(
        "--symbol", default="btc",
        help="Symbol label for output file names. (default: btc)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", default=False,
        help="Ignore saved state and start from scratch. (default: resume if state exists)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. (default: INFO)",
    )
    return parser
