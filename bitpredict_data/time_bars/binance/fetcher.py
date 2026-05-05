import time
import pandas as pd
from datetime import datetime
from binance.client import Client
import warnings
from bitpredict.common.logging import get_logger
from bitpredict.data.time_bars.base import Base
from typing import Union
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

warnings.filterwarnings("ignore")

# Environment variables
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    raise RuntimeError("BINANCE_API_KEY or BINANCE_API_SECRET is missing")

logger = get_logger(__name__)


class BinanceFetcher(Base):
    """
    Binance-specific implementation of BaseFetcher.
    
    Fetches historical OHLCV data from Binance Futures using the Binance API.
    Handles automatic pagination, retries, and converts data to a standard DataFrame format.
    """

    def __init__(self, exchange, symbol, **kwargs):
        """
        Initializes Binance API client using keys from environment variables.

        Args:
            exchange (str): Exchange name (e.g., 'binance')
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            kwargs: Additional arguments passed to Base class
        """
        super().__init__(exchange=exchange, symbol=symbol, **kwargs)

        logger.debug("Initializing Binance client for symbol: %s", symbol)
        self.exchange_client = Client(api_key=API_KEY, api_secret=API_SECRET)
        logger.info("Binance client initialized successfully for %s", symbol)

    def fetch_ohlc(
        self,
        symbol: str,
        start_datetime: Union[datetime, int, None],
        end_datetime: Union[datetime, int]
    ) -> pd.DataFrame:
        """
        Fetch OHLC (Open, High, Low, Close, Volume) data from Binance Futures.

        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')
            start_datetime (datetime|int|None): Start time for fetching data
            end_datetime (datetime|int): End time for fetching data
            time_horizon: Time interval in minutes (currently 1m fixed)
            retries: Number of retry attempts for API calls
            retry_delay: Delay between retries in seconds

        Returns:
            pd.DataFrame: DataFrame containing ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        Notes:
            - Fetches 1-minute candles only.
            - Automatically handles pagination (max 1000 candles per request).
            - Retries failed requests up to self.retries times with self.retry_delay.
            - Drops the last incomplete row if exists.
        """
        mapped_symbol = f"{symbol.upper()}USDT"
        logger.debug("Mapped symbol for Binance API: %s", mapped_symbol)

        # Convert start timestamp to milliseconds
    # ---------------------------------------------------
    # Normalize start and end timestamps to milliseconds
    # ---------------------------------------------------
        if start_datetime is None:
            # Safe earliest futures USDT-M history
            start_ms = int(datetime(2024, 1, 1).timestamp() * 1000)
            logger.info("Start timestamp not provided. Using safe default: %s", start_ms)
        elif isinstance(start_datetime, datetime):
            # Ensure datetime is timezone-aware before converting
            if start_datetime.tzinfo is None:
                start_datetime = start_datetime.replace(tzinfo=self.timezone)
            start_ms = int(start_datetime.timestamp() * 1000)
        else:
            start_ms = int(start_datetime)

        if isinstance(end_datetime, datetime):
            # Ensure datetime is timezone-aware before converting
            if end_datetime.tzinfo is None:
                end_datetime = end_datetime.replace(tzinfo=self.timezone)
            end_ms = int(end_datetime.timestamp() * 1000)
        else:
            end_ms = int(end_datetime)
        # end_ms = int(datetime(2026, 1, 19).timestamp() * 1000)
        logger.debug("Fetching futures data for %s from %s to %s", mapped_symbol, start_ms, end_ms)

        # ---------------------------------------------------
        # Fetch loop with retries
        # ---------------------------------------------------
        all_klines = []
        current_start = start_ms
        retries_left = self.retries

        while current_start < end_ms:
            try:
                klines = self.exchange_client.futures_klines(
                    symbol=mapped_symbol,
                    interval="1m",
                    startTime=current_start,
                    endTime=end_ms,
                    limit=1000,
                )

                if not klines:
                    logger.info("No more klines returned for %s at %s", mapped_symbol, current_start)
                    break

                all_klines.extend(klines)
                current_start = klines[-1][0] + 1  # Move to next timestamp
                retries_left = self.retries  # Reset retries after successful fetch

            except Exception as e:
                retries_left -= 1
                logger.warning("Attempt failed for %s: %s (%d retries left)", mapped_symbol, str(e), retries_left)
                if retries_left <= 0:
                    logger.error("Max retries exceeded for %s. Exiting fetch.", mapped_symbol)
                    raise
                time.sleep(self.retry_delay)

        if not all_klines:
            logger.warning("No data fetched for symbol: %s", mapped_symbol)
            return pd.DataFrame()

        # ---------------------------------------------------
        # Convert raw klines to standardized DataFrame
        # ---------------------------------------------------
        df = pd.DataFrame(
            all_klines,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ],
        )

        df = df[["open_time", "open", "high", "low", "close", "volume"]]
        df = df.astype({
            "open_time": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })
        df.rename(columns={"open_time": "timestamp", "open": "open", "high": "high",
                        "low": "low", "close": "close", "volume": "volume"}, inplace=True)

        # Drop last incomplete row if exists
        if not df.empty:
            df = df.iloc[:-2]

        logger.info("Fetched %d rows for %s", len(df), mapped_symbol)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]