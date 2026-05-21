"""
bars/main.py  — OPTIMIZED
==========================
Drop-in replacement for the original main.py.

Core change: all bar types run in PARALLEL via ProcessPoolExecutor.

Original (sequential):
    dollar → volume → volatility → range → renko → hybrid
    wall time = sum of all 6 individual times

Optimized (parallel):
    dollar ─┐
    volume  ─┤
    volatility─┤  all launched simultaneously
    range   ─┤
    renko   ─┤
    hybrid  ─┘
    wall time ≈ max(individual times)  ≈ 1/4 to 1/5 of original

Why ProcessPoolExecutor and NOT ThreadPoolExecutor
---------------------------------------------------
The bar algorithm inner loop (process_chunk while-loop) is pure Python that
HOLDS the GIL.  Threads cannot parallelize GIL-bound code — they interleave
on one core.  Separate processes each get their own Python interpreter and
GIL — genuine parallelism across cores.

Memory usage under multiprocessing
------------------------------------
Each process reads the CSV independently.  memory_map=True (in processor.py)
tells the OS to memory-map the file.  The kernel shares physical file pages
across all processes — so 6 processes reading the same 20 GB file costs ~1×
the file size in RAM, not 6×.  Your 64 GB is more than sufficient.

Default parallelism: 4 workers (safe for a cold file not yet in page-cache).
After the first complete run the entire file is in the OS page-cache; you can
safely raise --parallel to 6 for maximum speed on subsequent runs.

Usage
-----
    # Default: 4 parallel bar types
    python bars/main.py --source tick --types all \
        --tick-csv data/raw_data/BTCUSDT-aggTrades-ALL.csv

    # Max parallelism (use after file is in page-cache)
    python bars/main.py --source tick --types all \
        --tick-csv data/raw_data/BTCUSDT-aggTrades-ALL.csv --parallel 6

    # Specific bar types in parallel
    python bars/main.py --source tick --types dollar volatility range \
        --tick-csv data/raw_data/BTCUSDT-aggTrades-ALL.csv --parallel 3
"""
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logging import setup_logging, get_logger
from bars.cli_control import build_parser
from bars.processor import process_minute_bars, process_tick_bars
from bars.bar_types import ALL_BAR_TYPE_NAMES
from typing import Dict

logger = get_logger(__name__)

DEFAULT_PARALLEL = 6   # safe default — works cold


# ─────────────────────────────────────────────────────────────────────────────
# Worker functions — must be module-level for multiprocessing pickling
# ─────────────────────────────────────────────────────────────────────────────

def _tick_worker(args: tuple) -> tuple:
    """Runs in a separate process — one bar type."""
    bar_type, tick_csv, output_dir, exchange, symbol = args
    setup_logging()
    t0 = time.time()
    try:
        bars = process_tick_bars(
            bar_type=bar_type, tick_csv=tick_csv,
            output_dir=output_dir, exchange=exchange, symbol=symbol,
        )
        return bar_type, len(bars), time.time() - t0, None
    except Exception:
        import traceback
        return bar_type, 0, time.time() - t0, traceback.format_exc()


def _minute_worker(args: tuple) -> tuple:
    """Runs in a separate process — one bar type."""
    bar_type, minute_csv, output_dir, exchange, symbol, no_resume = args
    setup_logging()
    t0 = time.time()
    try:
        bars = process_minute_bars(
            bar_type=bar_type, minute_csv=minute_csv,
            output_dir=output_dir, exchange=exchange,
            symbol=symbol, resume=not no_resume,
        )
        return bar_type, len(bars), time.time() - t0, None
    except Exception:
        import traceback
        return bar_type, 0, time.time() - t0, traceback.format_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    multiprocessing.freeze_support()   # required on Windows

    setup_logging()
    parser = build_parser()
    parser.add_argument(
        "--parallel", type=int, default=DEFAULT_PARALLEL, metavar="N",
        help=(
            f"Bar types to process simultaneously (default: {DEFAULT_PARALLEL}). "
            "Raise to 6 after the first run when the CSV is in the OS page-cache."
        ),
    )
    args = parser.parse_args()

    bar_types = (
        ALL_BAR_TYPE_NAMES
        if not args.types or args.types == ["all"]
        else args.types
    )
    output_dir = Path(args.output_dir)
    n_workers  = min(args.parallel, len(bar_types))

    logger.info("=" * 60)
    logger.info("Information Bar Generator  [PARALLEL]")
    logger.info("  source    : %s", args.source)
    logger.info("  bar types : %s", bar_types)
    logger.info("  output    : %s", output_dir)
    logger.info("  workers   : %d  (processing %d bar types)", n_workers, len(bar_types))
    logger.info("=" * 60)

    t0 = time.time()

    if args.source == "tick":
        if not args.tick_csv:
            logger.error("--tick-csv is required for --source tick")
            sys.exit(1)
        worker_args = [
            (bt, args.tick_csv, str(output_dir), args.exchange, args.symbol)
            for bt in bar_types
        ]
        worker_fn = _tick_worker
    else:
        if not args.minute_csv:
            logger.error("--minute-csv is required for --source minute")
            sys.exit(1)
        worker_args = [
            (bt, args.minute_csv, str(output_dir), args.exchange, args.symbol,
             getattr(args, "no_resume", False))
            for bt in bar_types
        ]
        worker_fn = _minute_worker

    total_bars = 0
    results: Dict[str, tuple] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_map = {pool.submit(worker_fn, wa): wa[0] for wa in worker_args}

        for future in as_completed(future_map):
            bt_name = future_map[future]
            try:
                bt, n_bars, elapsed, tb = future.result()
                if tb:
                    logger.error("FAILED %s:\n%s", bt, tb)
                else:
                    logger.info("  DONE %-12s : %6d bars in %5.1fs", bt, n_bars, elapsed)
                    total_bars      += n_bars
                    results[bt]      = (n_bars, elapsed)
            except Exception:
                logger.exception("Unexpected error for bar type %s", bt_name)

    wall = time.time() - t0

    logger.info("=" * 60)
    logger.info("COMPLETE: %d bars total", total_bars)
    logger.info("  Wall time  : %.1fs  (%.1f min)", wall, wall / 60)
    if results:
        seq_est = sum(v[1] for v in results.values())
        logger.info("  Sequential : ~%.1fs estimated", seq_est)
        logger.info("  Speedup    : %.1f×", seq_est / max(wall, 1))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()