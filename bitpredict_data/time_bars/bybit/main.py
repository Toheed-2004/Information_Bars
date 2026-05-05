import sys
import warnings
from bitpredict.data.time_bars.bybit.fetcher import BybitFetcher
from bitpredict.common.logging import get_logger, setup_logging, set_run_mode

from bitpredict.common.db.services.meta import get_time_bars_meta

from dotenv import load_dotenv

# Silence warnings globally
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()


def run(symbol: str, exchange: str, run_mode: str, timeframes: dict) -> None:
    """
    Process a single symbol using the refactored Base with BybitFetcher.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        exchange: Exchange name (e.g., "bybit")
        run_mode: "init" or "update"
        timeframes: Dictionary of timeframes to resample
    """    
    # Initialize Base with the fetcher
    data_handler = BybitFetcher(
        exchange=exchange,
        symbol=symbol
    )
    
    # Run the data processing pipeline
    data_handler.run(run_mode=run_mode, timeframes=timeframes)


def main():
    """
    Main entry point for the Bybit data downloader.
    """
    # Determine run mode from CLI argument or default to "update"
    run_mode = sys.argv[1].lower() if len(sys.argv) > 1 else "update"
    # run_mode = "resample"
    
    # Module-level logger
    set_run_mode(run_mode)
    setup_logging("data.ochlv.bybit")
    logger = get_logger(__name__)

    logger.info("Starting data downloader in mode: %s", run_mode)

    # Define exchange
    exchange = "bybit"

    # Fetch symbols and their meta from DB
    symbols_meta = get_time_bars_meta(exchange)

    # Iterate over symbols
    for meta in symbols_meta:
        symbol = meta["symbol"]

        if not meta["allowed"]:
            logger.info("Symbol %s disabled in meta table, skipping", symbol)
            continue

        # Process symbol using the new structure
        run(symbol, exchange, run_mode, meta["timeframe"])

    logger.info("Data downloader finished.")


if __name__ == "__main__":
    main()