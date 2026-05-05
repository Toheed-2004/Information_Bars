from bitpredict.data.time_bars.base import Base
from datetime import datetime, timezone
from typing import Union
import pandas as pd
import time as time_module
import requests
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


class HyperliquidFetcher(Base):
    """Hyperliquid-specific implementation of BaseFetcher."""

    def __init__(self, exchange, symbol, **kwargs):

        super().__init__(exchange=exchange, symbol=symbol, **kwargs)

        self.api_endpoint = "https://api.hyperliquid.xyz"
        
    def _make_public_request(self, endpoint: str, payload: dict = None) -> dict:
        """Make a public request to Hyperliquid API."""
        url = f"{self.api_endpoint}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
        }
        
        try:
            response = requests.post(
                url, 
                headers=headers,
                json=payload,  # Send payload directly as JSON
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text}")
            raise

    def fetch_ohlc(
        self,
        symbol: str,
        start_datetime: Union[datetime, int, None],
        end_datetime: Union[datetime, int]
    ) -> pd.DataFrame:
        """
        Fetch OHLC data from Hyperliquid (always fetches 1-minute candles).
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH', 'ARB')
            start_datetime: Start time (datetime, int ms, or None)
            end_datetime: End time (datetime or int ms)
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            DataFrame with OHLC data
        """
        symbol = f"{symbol.upper()}"
        # Convert timestamps to milliseconds
        if start_datetime is None:
            # Start from early 2026 when Hyperliquid launched
            start_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        elif isinstance(start_datetime, datetime):
            if start_datetime.tzinfo is None:
                start_datetime = start_datetime.replace(tzinfo=self.timezone)
            start_ms = int(start_datetime.timestamp() * 1000)
        else:
            start_ms = int(start_datetime)

        if isinstance(end_datetime, datetime):
            if end_datetime.tzinfo is None:
                end_datetime = end_datetime.replace(tzinfo=self.timezone)
            end_ms = int(end_datetime.timestamp() * 1000)
        else:
            end_ms = int(end_datetime)

        all_data = []
        current_start = start_ms
        
        # 1000 candles per request (1m each = 1000 minutes)
        while current_start < end_ms:
            chunk_end = min(current_start + (1000 * 60 * 1000), end_ms)  # 1000 minutes max per request

            for attempt in range(self.retries):
                try:
                    # Hyperliquid candles API
                    endpoint = "/info"
                    
                    payload = {
                        "type": "candleSnapshot",  # Note: singular, not plural
                        "req": {
                            "coin": symbol,  # Use symbol as-is (BTC, ETH, etc.)
                            "interval": "1m",
                            "startTime": current_start,
                            "endTime": chunk_end
                        }
                    }
                    
                    logger.info(f"Fetching {symbol} data from {current_start} to {chunk_end}")
                    
                    response = self._make_public_request(endpoint, payload)
                    
                    # Check if response is a list (successful) or dict with error
                    if isinstance(response, dict) and "error" in response:
                        logger.error(f"Hyperliquid API error: {response}")
                        raise RuntimeError(f"Failed to fetch candles: {response}")
                    
                    # Response should be a list of candles
                    candles = response if isinstance(response, list) else []
                    
                    if not candles:
                        # No more data available
                        logger.info(f"No data returned for {symbol} in range {current_start}-{chunk_end}")
                        current_start = chunk_end
                        break
                    
                    # Process candles
                    for candle in candles:
                        all_data.append([
                            int(candle["t"]),  # timestamp
                            float(candle["o"]),  # open
                            float(candle["h"]),  # high
                            float(candle["l"]),  # low
                            float(candle["c"]),  # close
                            float(candle["v"])   # volume
                        ])
                    
                    logger.info(f"Fetched {len(candles)} candles for {symbol}")
                    
                    # Update current_start to chunk_end
                    current_start = chunk_end
                    
                    # Small delay to avoid rate limiting
                    time_module.sleep(0.2)
                    break
                    
                except Exception as e:
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt + 1,
                        self.retries,
                        symbol,
                        str(e),
                    )
                    if attempt == self.retries - 1:
                        raise
                    time_module.sleep(self.retry_delay)

        if not all_data:
            logger.warning(f"No data found for symbol {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        # Sort by timestamp (ascending)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Ensure proper data types
        df = df.astype({
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_available_symbols(self) -> list:
        """
        Get list of available trading symbols on Hyperliquid.
        """
        try:
            endpoint = "/info"
            payload = {
                "type": "meta"  # Changed from metaAndAssetCtxs
            }
            
            response = self._make_public_request(endpoint, payload)
            
            if "universe" not in response:
                return []
            
            symbols = []
            for coin_info in response["universe"]:
                symbols.append(coin_info["name"])
            
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to fetch available symbols: {e}")
            return []