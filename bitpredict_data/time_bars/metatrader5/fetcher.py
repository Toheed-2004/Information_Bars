import time
import pandas as pd
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5
import warnings
from bitpredict.common.logging import get_logger
from bitpredict.data.time_bars.base import Base
from typing import Union
from dotenv import load_dotenv
from bitpredict.common.utils.time import datetime_to_timestamp
import os

# Load environment variables
load_dotenv()

warnings.filterwarnings("ignore")

# Environment variables
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
MT5_SERVER = os.getenv("MTRADER_SERVER", "").replace(" ", "")
MT5_PASSWORD = os.getenv("MTRADER_PASSWORD", "").replace(" ", "")
MT5_LOGIN = int(os.getenv("MTRADER_ACCOUNT", "0"))

if not MT5_SERVER or not MT5_PASSWORD or MT5_LOGIN == 0:
    raise RuntimeError("MTRADER_SERVER, MTRADER_PASSWORD, or MTRADER_ACCOUNT is missing")

logger = get_logger(__name__)


class MetatraderFetcher(Base):
    """
    MetaTrader5-specific implementation of Base.
    
    Fetches historical OHLCV data from MetaTrader5 terminal.
    """

    def __init__(self, exchange, symbol, **kwargs):
        """
        Initializes MetaTrader5 client using credentials from environment variables.
        """
        super().__init__(exchange=exchange, symbol=symbol, **kwargs)
        
        self.broker_utc_offset = 2  # Adjust based on your broker's timezone
        self.exchange_client = None
        self._initialize_mt5()

    def _initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            logger.info("Basic MT5 initialization failed, trying with credentials...")
            if not mt5.initialize(
                path=MT5_PATH,
                login=MT5_LOGIN,
                server=MT5_SERVER,
                password=MT5_PASSWORD
            ):
                error = mt5.last_error()
                logger.error("MT5 initialization failed: %s", error)
                raise RuntimeError(f"Failed to initialize MetaTrader5: {error}")
        
        logger.info("MT5 initialized successfully")
        self.exchange_client = mt5

    def _ensure_mt5_connected(self):
        """Ensure MT5 is connected, reinitialize if needed"""
        if not mt5.initialize():
            logger.warning("MT5 connection lost, reinitializing...")
            self._initialize_mt5()

    def fetch_ohlc(
        self,
        symbol: str,
        start_datetime: Union[datetime, int, None],
        end_datetime: Union[datetime, int]
    ) -> pd.DataFrame:
        """
        Fetch OHLC data from MetaTrader5.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'PFE')
            start_datetime: Start time (datetime, int ms, or None)
            end_datetime: End time (datetime or int ms)
            timeframe: Time interval in minutes (currently fixed to 1)
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """

        symbol = f"{symbol.upper()}USD"
        # Convert timestamps to datetime objects
        if start_datetime is None:
            # Get earliest available data
            try:
                self._ensure_mt5_connected()
                # Try to get symbol info to determine available data range
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.error("Symbol %s not found", symbol)
                    return pd.DataFrame()
                
                if not symbol_info.visible:
                    mt5.symbol_select(symbol, True)

                # Fetch earliest available bar
                start_guess = datetime(2000, 1, 1, tzinfo=timezone.utc)
                data = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, start_guess, 1)
                if data is None or len(data) == 0:
                    logger.warning("No historical data available for symbol %s", symbol)
                    return pd.DataFrame()

                start_dt = datetime.fromtimestamp(data[0]['time'], tz=timezone.utc)
                logger.info("Fetching from oldest available data: %s", start_dt)
            except Exception:
                logger.exception("Error determining earliest start date for %s", symbol)
                return pd.DataFrame()
        elif isinstance(start_datetime, int):
            start_dt = datetime.fromtimestamp(start_datetime / 1000, tz=timezone.utc)
        else:
            start_dt = start_datetime
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)

        if isinstance(end_datetime, int):
            end_dt = datetime.fromtimestamp(end_datetime / 1000, tz=timezone.utc)
        else:
            end_dt = end_datetime
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)

        # Adjust for broker UTC offset
        start_dt = start_dt + timedelta(hours=self.broker_utc_offset)
        end_dt = end_dt + timedelta(hours=self.broker_utc_offset)

        retries_left = self.retries
        df = None

        while retries_left > 0:
            try:
                self._ensure_mt5_connected()
                
                # Check if symbol is visible/available
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.error("Symbol %s not found", symbol)
                    return pd.DataFrame()
                
                if not symbol_info.visible:
                    logger.info("Symbol %s not visible, attempting to select it", symbol)
                    if not mt5.symbol_select(symbol, True):
                        logger.error("Failed to select symbol %s", symbol)
                        return pd.DataFrame()

                # Fetch data
                data = mt5.copy_rates_range(
                    symbol,
                    mt5.TIMEFRAME_M1,
                    start_dt,
                    end_dt
                )
                
                if data is None or len(data) == 0:
                    logger.warning("No data received for symbol: %s", symbol)
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(data)
                # Drop all data before first non-zero spread
                if "spread" in df.columns:
                    non_zero_spread_idx = df.index[df["spread"] > 0]
                    if len(non_zero_spread_idx) == 0:
                        logger.warning("All spread values are zero for symbol: %s", symbol)
                        return pd.DataFrame()
                    df = df.loc[non_zero_spread_idx[0]:].reset_index(drop=True)

                df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
                
                # Adjust back from broker timezone
                df["datetime"] = df["datetime"] - pd.Timedelta(hours=self.broker_utc_offset)
                
                # Convert datetime to millisecond timestamp
                df = datetime_to_timestamp(df, column="datetime")
                df.rename(columns={"datetime": "timestamp"}, inplace=True)
                
                df.rename(columns={"tick_volume": "volume"}, inplace=True)
                
                df = df.astype({
                    "timestamp": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                })
                
                # Drop last incomplete row if exists
                if not df.empty:
                    df = df.iloc[:-1]

                logger.info("Fetched %d rows of data for symbol: %s", len(df), symbol)
                break
                
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
                    logger.error("Maximum retries reached for %s. Exiting fetch.", symbol)
                    raise
                time.sleep(self.retry_delay)

        if df is None or df.empty:
            logger.warning("No data fetched for symbol: %s", symbol)
            return pd.DataFrame()

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def __del__(self):
        """Cleanup: shutdown MT5 connection when object is destroyed"""
        try:
            if self.exchange_client is not None:
                mt5.shutdown()
                logger.info("MT5 connection closed")
        except Exception:
            pass