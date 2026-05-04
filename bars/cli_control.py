"""
bars/cli_control.py
-------------------
Command-line interface definitions for the information bar pipeline.

All argument parsing lives here so it can be reused by other scripts
(e.g. comparison or analysis tools).
"""
import argparse
from bars.bar_types import ALL_BAR_TYPE_NAMES


def build_parser() -> argparse.ArgumentParser:
    """Return the fully configured ArgumentParser for bars/main.py."""

    parser = argparse.ArgumentParser(
        prog="python bars/main.py",
        description=(
            "Generate information bars (dollar, volume, volatility, range, "
            "renko, hybrid) from minute OHLCV or raw tick data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # All minute-level bar types
  python bars/main.py --source minute --types all \\
      --minute-csv data/raw_data/1minute_btcusdt_2024.csv

  # Tick-level dollar bars only
  python bars/main.py --source tick --types dollar \\
      --tick-csv   data/raw_data/btcusdt_aggTrades_2024.csv \\
      --minute-csv data/raw_data/1minute_btcusdt_2024.csv

  # Multiple types, custom output directory
  python bars/main.py --source minute --types dollar volatility hybrid \\
      --minute-csv data/raw_data/1minute_btcusdt_2024.csv \\
      --output-dir data/processed_bars/run_01
""",
    )

    # ── Data source ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--source",
        choices=["minute", "tick"],
        default="minute",
        help=(
            "Data resolution to use.  "
            "'minute' reads pre-aggregated 1-min OHLCV CSVs; "
            "'tick' reads raw aggTrade CSVs.  "
            "(default: minute)"
        ),
    )

    # ── Bar types ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--types",
        nargs="+",
        default=["all"],
        metavar="TYPE",
        help=(
            "Bar type(s) to generate.  "
            f"Choices: {', '.join(ALL_BAR_TYPE_NAMES)}, all.  "
            "Multiple types can be listed: --types dollar volatility hybrid.  "
            "(default: all)"
        ),
    )

    # ── Input files ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--minute-csv",
        default=None,
        metavar="PATH",
        help=(
            "Path to the 1-minute OHLCV CSV file.  "
            "Required for source=minute; also required for tick calibration."
        ),
    )
    parser.add_argument(
        "--tick-csv",
        default=None,
        metavar="PATH",
        help=(
            "Path to the raw aggTrade (tick) CSV file.  "
            "Required when --source tick."
        ),
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        default="data/processed_bars",
        metavar="DIR",
        help="Directory where bar CSVs and state files are written.  (default: data/processed_bars)",
    )

    # ── Symbol metadata ───────────────────────────────────────────────────────
    parser.add_argument(
        "--exchange",
        default="binance",
        help="Exchange label used in output file names.  (default: binance)",
    )
    parser.add_argument(
        "--symbol",
        default="btc",
        help="Symbol label used in output file names.  (default: btc)",
    )

    # ── Behaviour ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help=(
            "Always start from scratch — ignore any saved state file.  "
            "By default the pipeline resumes from the last processed timestamp."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.  (default: INFO)",
    )

    return parser


def build_compare_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the comparison / analysis script."""

    parser = argparse.ArgumentParser(
        prog="python bars/compare.py",
        description=(
            "Compare minute-level vs tick-level information bars and "
            "optionally against a time-bar baseline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Compare dollar bars (minute vs tick) with time-bar baseline
  python bars/compare.py --types dollar --time-bars \\
      --minute-csv data/raw_data/1minute_btcusdt_2024.csv \\
      --tick-csv   data/raw_data/btcusdt_aggTrades_2024.csv

  # Compare all bar types, no time bars
  python bars/compare.py --types all \\
      --minute-csv data/raw_data/1minute_btcusdt_2024.csv \\
      --tick-csv   data/raw_data/btcusdt_aggTrades_2024.csv
""",
    )

    parser.add_argument("--types", nargs="+", default=["all"], metavar="TYPE",
                        help=f"Bar type(s).  Choices: {', '.join(ALL_BAR_TYPE_NAMES)}, all.")
    parser.add_argument("--minute-csv", required=True, metavar="PATH")
    parser.add_argument("--tick-csv",   required=True, metavar="PATH")
    parser.add_argument("--time-bars",  action="store_true",
                        help="Include a time-bar baseline in the comparison.")
    parser.add_argument("--output-dir", default="data/comparison_results",
                        metavar="DIR")
    parser.add_argument("--exchange",   default="binance")
    parser.add_argument("--symbol",     default="btc")

    return parser
