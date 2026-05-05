import sys
import warnings
from bitpredict.common.logging import get_logger, setup_logging, set_run_mode
from bitpredict.data.time_bars.okx.fetcher import OKXFetcher
from bitpredict.common.db.services.meta import get_time_bars_meta

from dotenv import load_dotenv

# Silence warnings globally
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()


def run(symbol: str, exchange: str, run_mode: str, timeframes: dict) -> None:
    """
    Process a single symbol using the refactored Base with OKXFetcher.
    
    Args:
        symbol: Trading symbol (e.g., "BTC-USDT")
        exchange: Exchange name (e.g., "okx")
        run_mode: "init" or "update"
        timeframes: Dictionary of timeframes to resample
    """

    
    # Initialize Base with the fetcher
    data_handler = OKXFetcher(
        exchange=exchange,
        symbol=symbol
    )
    
    # Run the data processing pipeline
    data_handler.run(run_mode=run_mode, timeframes=timeframes)


def main():
    """
    Main entry point for the OKX data downloader.

    Steps:
        1. Determines the run mode ("init" or "update") from CLI arguments.
        2. Loads allowed symbols and their active timeframes from meta DB.
        3. Iterates through each symbol:
           - Skips symbols that are disabled in meta.
           - Processes OHLC data using run().
        4. Logs completion message.

    CLI Usage:
        python main.py [run_mode]
        - run_mode: "init" or "update" (default: "update")
    """
    # Determine run mode from CLI argument or default to "update"
    run_mode = sys.argv[1].lower() if len(sys.argv) > 1 else "update"
    # run_mode = "init"

    # Module-level logger
    set_run_mode(run_mode)
    setup_logging("data.ochlv.okx")
    logger = get_logger(__name__)
    
    logger.info("Starting data downloader in mode: %s", run_mode)

    # Define exchange
    exchange = "okx"

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