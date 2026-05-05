from bitpredict.data.time_bars.base import Base
from pybit.unified_trading import HTTP
from datetime import datetime, timezone
from typing import Union
import pandas as pd
import time as time_module
import os
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")


class BybitFetcher(Base):
    """
    Bybit-specific OHLCV data fetcher.

    Responsibilities:
    - Fetch historical OHLCV candles from Bybit
    - Handle pagination / chunking limits
    - Retry on transient API failures
    - Return normalized pandas DataFrame

    This fetcher always returns **1-minute candles**.
    Higher timeframes must be built via resampling.
    """

    def __init__(self, exchange: str, symbol: str, **kwargs):
        """
        Initialize BybitFetcher.

        Args:
            exchange: Exchange name (e.g. "bybit")
            symbol: Trading symbol (e.g. "BTCUSDT")
            **kwargs: Passed to Base (retries, retry_delay, etc.)
        """
        super().__init__(exchange=exchange, symbol=symbol, **kwargs)

        logger.info(
            "Initializing BybitFetcher | exchange=%s symbol=%s",
            exchange,
            symbol,
        )

        # Create Bybit HTTP client (Unified Trading API)
        self.client = HTTP(
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=False
        )

    def fetch_ohlc(
        self,
        symbol: str,
        start_datetime: Union[datetime, int, None],
        end_datetime: Union[datetime, int],
    ) -> pd.DataFrame:
        """
        Fetch 1m OHLCV candles from Bybit.

        Unified logic:
        - Always fetch BACKWARD from end_datetime
        - If start_datetime is provided → stop once reached
        - If start_datetime is None → stop when history ends
        """

        logger.info(
            "Fetching OHLCV | symbol=%s start=%s end=%s",
            symbol,
            start_datetime,
            end_datetime,
        )

        # --------------------------------------------------
        # Normalize timestamps
        # --------------------------------------------------
        if isinstance(end_datetime, datetime):
            # Ensure datetime is timezone-aware before converting
            if end_datetime.tzinfo is None:
                end_datetime = end_datetime.replace(tzinfo=self.timezone)
            end_ms = int(end_datetime.timestamp() * 1000)
        else:
            end_ms = int(end_datetime)

        if start_datetime is None:
            start_ms = None
            logger.debug("Start timestamp is None → auto-discover history")
        elif isinstance(start_datetime, datetime):
            if start_datetime.tzinfo is None:
                start_datetime = start_datetime.replace(tzinfo=self.timezone)
            start_ms = int(start_datetime.timestamp() * 1000)
        else:
            start_ms = int(start_datetime)

        mapped_symbol = f"{symbol.upper()}USDT"

        WINDOW_MS = 60_000_000   # ~1000 minutes (Bybit hard limit)
        MAX_EMPTY_WINDOWS = 3

        all_data: list = []
        request_count = 0
        empty_windows = 0

        current_end = end_ms

        # --------------------------------------------------
        # SINGLE BACKWARD FETCH LOOP
        # --------------------------------------------------
        while current_end > 0:
            # Respect explicit start boundary
            if start_ms is not None and current_end <= start_ms:
                break

            current_start = max(
                start_ms if start_ms is not None else 0,
                current_end - WINDOW_MS,
            )

            logger.debug(
                "Request chunk | start_ms=%d end_ms=%d",
                current_start,
                current_end,
            )

            for attempt in range(1, self.retries + 1):
                try:
                    request_count += 1

                    resp = self.client.get_kline(
                        category="linear",
                        symbol=mapped_symbol,
                        interval=1,
                        start=current_start,
                        end=current_end,
                        limit=1000,
                    )

                    if resp["retCode"] != 0:
                        raise RuntimeError(resp["retMsg"])

                    candles = resp["result"]["list"]

                    # Empty window
                    if not candles:
                        empty_windows += 1
                        logger.debug(
                            "Empty window %d/%d | %s",
                            empty_windows,
                            MAX_EMPTY_WINDOWS,
                            symbol,
                        )

                        # Stop only if auto-discovering history
                        if start_ms is None and empty_windows >= MAX_EMPTY_WINDOWS:
                            logger.info(
                                "No more historical data | stopping for %s",
                                symbol,
                            )
                            current_end = 0
                            break

                        current_end = current_start
                        break

                    # Got data
                    empty_windows = 0
                    all_data.extend(candles)
                    current_end = current_start
                    break

                except Exception as e:
                    logger.warning(
                        "Retry %d/%d failed | symbol=%s error=%s",
                        attempt,
                        self.retries,
                        symbol,
                        str(e),
                    )

                    if attempt == self.retries:
                        logger.exception(
                            "Max retries exceeded | symbol=%s start_ms=%d",
                            symbol,
                            current_start,
                        )
                        raise

                    time_module.sleep(self.retry_delay)

        # --------------------------------------------------
        # FINALIZE
        # --------------------------------------------------
        if not all_data:
            logger.warning("No OHLCV data fetched | symbol=%s", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(
            all_data,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )

        df = (
            df.astype({
                "timestamp": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            })
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp", ignore_index=True)
        )

        # Drop last incomplete row if exists
        if not df.empty:
            df = df.iloc[:-2]

        logger.info(
            "Finished fetching OHLCV | symbol=%s rows=%d requests=%d",
            symbol,
            len(df),
            request_count,
        )

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

