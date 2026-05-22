"""
bars/main.py
------------
Entry point for information bar generation.

Usage
-----
    # All minute-level bar types
    python bars/main.py --source minute --types all \
        --minute-csv data/raw_data/1minute_ohlcv.csv

    # All tick-level bar types
    python bars/main.py --source tick --types all \
        --tick-csv data/raw_data/merged_tickdata.csv

    # Specific types
    python bars/main.py --source minute --types dollar volatility \
        --minute-csv data/raw_data/1minute_ohlcv.csv

    # If --types not provided, all six bar types are processed
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logging import setup_logging, get_logger
from bars.cli_control import build_parser
from bars.processor import process_minute_bars, process_tick_bars
from bars.bar_types import ALL_BAR_TYPE_NAMES

logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    parser = build_parser()
    args   = parser.parse_args()

    bar_types  = ALL_BAR_TYPE_NAMES if not args.types or args.types == ["all"]                  else args.types
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("Information Bar Generator")
    logger.info("  source   : %s", args.source)
    logger.info("  bar types: %s", bar_types)
    logger.info("  output   : %s", output_dir)
    logger.info("=" * 60)

    total_bars = 0
    t0 = time.time()

    for bar_type in bar_types:
        logger.info("Processing: %s (%s)", bar_type, args.source)
        t1 = time.time()
        try:
            if args.source == "minute":
                if not args.minute_csv:
                    logger.error("--minute-csv is required for --source minute")
                    sys.exit(1)
                bars = process_minute_bars(
                    bar_type   = bar_type,
                    minute_csv = args.minute_csv,
                    output_dir = output_dir,
                    exchange   = args.exchange,
                    symbol     = args.symbol,
                    resume     = not args.no_resume,
                )
            else:  # tick
                if not args.tick_csv:
                    logger.error("--tick-csv is required for --source tick")
                    sys.exit(1)
                bars = process_tick_bars(
                    bar_type   = bar_type,
                    tick_csv   = args.tick_csv,
                    output_dir = output_dir,
                    exchange   = args.exchange,
                    symbol     = args.symbol,
                )

            logger.info("Done %-12s : %d bars in %.1fs",
                        bar_type, len(bars), time.time() - t1)
            total_bars += len(bars)

        except Exception:
            logger.exception("Error processing %s — skipping.", bar_type)

    logger.info("=" * 60)
    logger.info("COMPLETE: %d bars total in %.1fs",
                total_bars, time.time() - t0)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
