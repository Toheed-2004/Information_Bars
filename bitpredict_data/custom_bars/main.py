"""
bars/main.py — Single entry point for bars: init, update, stats.

Usage:
    python main.py init     # initialise new symbols
    python main.py update   # update existing symbols
    python main.py stats    # compute bar quality statistics
"""
import sys
import time
from dotenv import load_dotenv

from bitpredict.common.logging import get_logger, setup_logging, set_run_mode
from bitpredict.common.db.services.meta import get_custom_bar_meta
from bitpredict.data.custom_bars.db import get_bar_state, read_bar, upsert_bar_stats
from bitpredict.data.custom_bars.utils import process_bar_data
from bitpredict.data.custom_bars.stats import compute_bar_stats  # keep only this function in stats.py

load_dotenv()
# setup_logging("data.bars")
# logger = get_logger(__name__)


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "update"
    # mode = "init"

    # Set run_mode in global logging context
    set_run_mode(mode)

    # Now initialize logging
    setup_logging("data.bars")

    logger = get_logger(__name__)

    if mode not in ("init", "update", "stats"):
        logger.error("Unknown mode '%s'. Use 'init', 'update', or 'stats'.", mode)
        sys.exit(1)

    runtime_config = get_custom_bar_meta()
    if not runtime_config:
        logger.info("No enabled bar configurations found. Exiting...")
        sys.exit(1)

    total_bars = 0
    total_computed = 0
    total_skipped = 0
    t0 = time.time()

    for exchange, symbol_map in runtime_config.items():
        for symbol, bar_types in symbol_map.items():
            for bar_type in bar_types:
                state = get_bar_state(exchange, symbol, bar_type)
                
                if mode == "init":
                    if state and state.get("last_processed_datetime"):
                        logger.info(
                            "Skipping %s_%s_%s — already initialised (use update mode)",
                            exchange, symbol, bar_type,
                        )
                        continue
                    logger.info("Initialising %s_%s_%s …", exchange, symbol, bar_type)
                    try:
                        t1 = time.time()
                        bars = process_bar_data(exchange, symbol, bar_type, mode="init")
                        elapsed = time.time() - t1
                        logger.info(
                            "Created %d bars for %s_%s_%s in %.2fs",
                            bars, exchange, symbol, bar_type, elapsed,
                        )
                        total_bars += bars
                    except Exception as e:
                        logger.error("Error initialising %s_%s_%s: %s", exchange, symbol, bar_type, e)

                elif mode == "update":
                    if not state or not state.get("last_processed_datetime"):
                        logger.info(
                            "Skipping %s_%s_%s — not yet initialised (run init mode first)",
                            exchange, symbol, bar_type,
                        )
                        continue
                    logger.info("Updating %s_%s_%s …", exchange, symbol, bar_type)
                    try:
                        t1 = time.time()
                        bars = process_bar_data(exchange, symbol, bar_type, mode="update")
                        elapsed = time.time() - t1
                        logger.info(
                            "Created %d bars for %s_%s_%s in %.2fs",
                            bars, exchange, symbol, bar_type, elapsed,
                        )
                        total_bars += bars
                    except Exception as e:
                        logger.error("Error updating %s_%s_%s: %s", exchange, symbol, bar_type, e)

                elif mode == "stats":
                    logger.info("Computing stats: %s/%s/%s …", exchange, symbol, bar_type)
                    try:
                        t1 = time.time()
                        df = read_bar(exchange, symbol, bar_type)
                        if df.empty:
                            logger.warning(
                                "No bars found for %s/%s/%s — skipping",
                                exchange, symbol, bar_type,
                            )
                            total_skipped += 1
                            continue
                        stats = compute_bar_stats(df, exchange, symbol, bar_type)
                        if not stats:
                            total_skipped += 1
                            continue
                        upsert_bar_stats(stats)
                        elapsed = time.time() - t1
                        logger.info(
                            "Done %s/%s/%s — %d bars | quality=%.1f | autocorr=%.3f | bar_cv=%.3f (%.2fs)",
                            exchange, symbol, bar_type,
                            stats["total_bars"],
                            stats.get("quality_score") or 0.0,
                            stats.get("return_autocorr_lag1") or 0.0,
                            stats.get("bar_size_cv") or 0.0,
                            elapsed,
                        )
                        total_computed += 1
                    except Exception:
                        logger.exception("Error computing stats for %s/%s/%s", exchange, symbol, bar_type)
                        total_skipped += 1

    elapsed_total = time.time() - t0
    if mode in ("init", "update"):
        logger.info("=== %s COMPLETE: %d bars processed in %.2fs ===", mode.upper(), total_bars, elapsed_total)
    else:
        logger.info("=== STATS COMPLETE: %d computed, %d skipped in %.2fs ===", total_computed, total_skipped, elapsed_total)


if __name__ == "__main__":
    main()