import time
import pandas as pd
import requests
from datetime import datetime, timezone
from bitpredict.common.logging import get_logger
from bitpredict.data.time_bars.base import Base
from typing import Union

logger = get_logger(__name__)


class OKXFetcher(Base):
    """
    OKX-specific implementation of BaseFetcher.
    
    Fetches historical OHLCV data from OKX public API.
    """

    def __init__(self, exchange, symbol, **kwargs):
        """
        Initializes OKX client. No API keys needed for public market data.
        """
        super().__init__(exchange=exchange, symbol=symbol, **kwargs)

        self.api_base_url = "https://www.okx.com/api/v5/market/history-candles"

    def fetch_ohlc(
        self,
        symbol: str,
        start_datetime: Union[datetime, int, None],
        end_datetime: Union[datetime, int]
    ) -> pd.DataFrame:
        """
        Fetch OHLC data from OKX.

        Args:
            symbol: Trading pair (e.g., 'BTC-USDT')
            start_datetime: Start time (datetime, int ms, or None)
            end_datetime: End time (datetime or int ms)
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        # Convert timestamps to milliseconds
        if start_datetime is None:
            start_ms = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        elif isinstance(start_datetime, datetime):
            if start_datetime.tzinfo is None:
                start_datetime = start_datetime.replace(tzinfo=self.timezone)
            start_ms = int(start_datetime.timestamp() * 1000)
        else:
            start_ms = int(start_datetime)

        if isinstance(end_datetime, datetime):
            if end_datetime.tzinfo is None:
                end_datetime = end_datetime.replace(tzinfo=self.timezone)
            start_ms = int(end_datetime.timestamp() * 1000)
        else:
            end_ms = int(end_datetime)

        mapped_symbol = f"{symbol.upper()}-USDT-SWAP"
        all_candles = []
        retries_left = self.retries
        current_ms = start_ms

        while current_ms < end_ms:
            try:
                params = {
                    "instId": mapped_symbol,
                    "bar": "1m",
                    "limit": 100,
                    "after": str(current_ms),
                }

                response = requests.get(self.api_base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json().get("data", [])

                if not data:
                    logger.info("No more data returned by OKX API for %s", symbol)
                    break

                for candle in data:
                    ts_ms = int(candle[0])
                    if ts_ms < start_ms or ts_ms >= end_ms:
                        continue
                    
                    all_candles.append({
                        "timestamp": ts_ms,
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5])
                    })

                current_ms = int(data[-1][0]) + 1
                time.sleep(0.1)

            except Exception as e:
                retries_left -= 1
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    self.retries - retries_left,
                    self.retries,
                    symbol,
                    str(e),
                )
                if retries_left <= 0:
                    logger.error("Max retries reached for %s. Exiting fetch.", symbol)
                    raise
                time.sleep(self.retry_delay)

        if not all_candles:
            logger.warning("No data fetched for symbol: %s", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        
        df = df.astype({
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })
        
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info("Fetched %d rows of data for symbol: %s", len(df), symbol)
        
        return df[["timestamp", "open", "high", "low", "close", "volume"]]