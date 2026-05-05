import warnings
import time
from datetime import datetime
import requests
import pandas as pd
from dotenv import load_dotenv
from bitpredict.common.logging import get_logger
from bitpredict.data.time_bars.base import Base
from typing import Union
from bitpredict.data.time_bars.kraken.utils import map_symbol

# Load environment variables from .env
load_dotenv()

# Silence warnings globally
warnings.filterwarnings("ignore")

logger = get_logger(__name__)


class KrakenFetcher(Base):
    """
    Kraken-specific implementation of BaseFetcher.
    
    Fetches OHLC data from Kraken Futures API.
    Handles chunked requests, retries, and converts raw data into
    standardized OHLCV DataFrame format.
    """

    def __init__(self, exchange, symbol, **kwargs):
        """
        Initialize KrakenFetcher instance.

        Args:
            exchange (str): Exchange name (e.g., 'kraken')
            symbol (str): Trading symbol (e.g., 'PF_BTCUSD')
            kwargs: Additional arguments passed to Base class
        """
        super().__init__(exchange=exchange, symbol=symbol, **kwargs)
        logger.debug("KrakenFetcher initialized for symbol: %s", symbol)

    def get_oldest_available_timestamp(self, symbol: str) -> datetime | None:
        """
        Retrieve the oldest available candle timestamp for the given symbol from Kraken Futures API.

        Args:
            symbol (str): Trading symbol (e.g., "PF_BTCUSD")
            
        Returns:
            datetime | None: Oldest available timestamp in UTC, or None if unavailable
        """
        url = f"https://futures.kraken.com/api/charts/v1/trade/{map_symbol(symbol)}/1m"
        params = {"from": 0, "to": int(time.time())}

        try:
            response = requests.get(url, headers={"Accept": "application/json"}, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if "candles" in data and data["candles"]:
                    oldest_ts = int(data["candles"][0]['time'] / 1000)  # Convert ms to seconds
                    oldest_dt = datetime.utcfromtimestamp(oldest_ts).replace(tzinfo=None)
                    logger.info("Oldest available timestamp for %s: %s", symbol, oldest_dt)
                    return oldest_dt
        except Exception as exc:
            logger.error("Error fetching oldest timestamp for %s: %s", symbol, exc)

        return None

    def fetch_candles(
        self, 
        symbol: str,
        start_datetime: datetime, 
        end_datetime: datetime
    ) -> list:
        """
        Fetch raw candle data from Kraken Futures API in chunks.

        Args:
            symbol (str): Trading symbol
            start_datetime (datetime): Start UTC datetime for fetching
            end_datetime (datetime): End UTC datetime for fetching

        Returns:
            list: List of raw candle dictionaries from Kraken API
        """
        url = f"https://futures.kraken.com/api/charts/v1/trade/{map_symbol(symbol)}/1m"
        candles_list = []
        total_requests = 0

        current_ts = int(start_datetime.timestamp())
        end_ts = int(end_datetime.timestamp())

        logger.debug("Fetching candles for %s from %s to %s", symbol, start_datetime, end_datetime)

        while current_ts < end_ts:
            # Chunk request parameters (max 2000 minutes per request)
            params = {
                "from": current_ts,
                "to": min(current_ts + (2000 * 60), end_ts)
            }

            try:
                response = requests.get(url, headers={"Accept": "application/json"}, params=params, timeout=30)
                total_requests += 1

                if response.status_code == 200:
                    data = response.json()
                    if "candles" in data and data["candles"]:
                        last_ts = int(data["candles"][-1]["time"] / 1000)
                        candles_list.extend(data["candles"])
                        current_ts = last_ts + 60  # Next minute after last candle

                        logger.info(
                            "Fetched %d candles for %s (total requests: %d)",
                            len(data["candles"]),
                            symbol,
                            total_requests
                        )
                    else:
                        # No candles returned, skip forward by chunk
                        current_ts += (2000 * 60)
                else:
                    logger.warning("HTTP %d for %s, retrying in 5s", response.status_code, symbol)
                    time.sleep(5)

                time.sleep(0.5)  # Avoid hitting API too quickly

            except Exception as exc:
                logger.error("Error fetching candles for %s: %s", symbol, exc)
                time.sleep(5)

        logger.debug("Completed fetching candles for %s. Total candles fetched: %d", symbol, len(candles_list))
        return candles_list

    def fetch_ohlc(
        self,
        symbol: str,
        start_datetime: Union[datetime, int, None],
        end_datetime: Union[datetime, int]
    ) -> pd.DataFrame:
        """
        Fetch OHLC data from Kraken Futures and return as DataFrame.

        Args:
            symbol (str): Trading symbol (e.g., 'PF_BTCUSD')
            start_datetime (datetime|int|None): Start time for fetching
            end_datetime (datetime|int): End time for fetching

        Returns:
            pd.DataFrame: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        # Convert start timestamp to datetime
        if start_datetime is None:
            start_dt = self.get_oldest_available_timestamp(symbol)
            if start_dt is None:
                start_dt = datetime(2015, 1, 1)
                logger.warning("Using default start timestamp for %s: %s", symbol, start_dt)
        elif isinstance(start_datetime, int):
            start_dt = datetime.utcfromtimestamp(start_datetime / 1000).replace(tzinfo=None)
        else:
            start_dt = start_datetime.replace(tzinfo=None) if start_datetime.tzinfo else start_datetime

        # Convert end timestamp to datetime
        if isinstance(end_datetime, int):
            end_dt = datetime.utcfromtimestamp(end_datetime / 1000).replace(tzinfo=None)
        else:
            end_dt = end_datetime.replace(tzinfo=None) if end_datetime.tzinfo else end_datetime

        logger.info("Fetching OHLC data for %s from %s to %s", symbol, start_dt, end_dt)

        # Fetch raw candles
        candles = self.fetch_candles(symbol, start_dt, end_dt)

        if not candles:
            logger.warning("No data fetched for %s", symbol)
            return pd.DataFrame()

        # Convert raw data to DataFrame
        df = pd.DataFrame(candles)

        # Map Kraken API time to standard timestamp column
        if 'time' in df.columns:
            df['timestamp'] = df['time'].astype('int64')

        # Keep only required OHLCV columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]

        # Ensure numeric dtypes
        df = df.astype({
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })

        # Drop last incomplete candle if exists
        if len(df) > 0:
            df = df[:-2]

        # Remove duplicates, sort by timestamp, and reset index
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        logger.info("Fetched %d rows of OHLCV data for %s", len(df), symbol)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
