"""
Hyperliquid WebSocket Stream Manager Module.

This module provides functionality to manage multiple Hyperliquid WebSocket
streams for cryptocurrency market data collection and database storage
orchestration.
"""

import json
import time
import logging
import threading
from datetime import datetime
from typing import Callable, List

from websocket import WebSocketApp
from bitpredict.common.db.services import insert_tick_data
from dotenv import load_dotenv
from bitpredict.common.logging import get_logger

load_dotenv()
# Configure module logger

logger = get_logger(__name__)


class HyperliquidStreamManager:
    """
    Manages multiple Hyperliquid WebSocket streams and database orchestration.

    This class handles WebSocket connections to Hyperliquid for various market
    data streams. It buffers incoming data for database persistence.

    Attributes:
        config (dict): Configuration dictionary.
        engine: SQLAlchemy database engine.
        symbols (list): List of symbols from config.
        market_type (str): Market type from config ('perpetual' or 'spot').
        streams (list): List of stream types to subscribe to.
        buffer_size (int): Number of records to buffer before database write.
        network (str): Network type - 'mainnet' or 'testnet'.
        base_url (str): Base WebSocket URL for the network.
        candle_interval (str): Candle interval (e.g., '1m', '5m', '1h').
        last_flush_times (dict): Tracks last flush time for each stream.
        active_websockets (List[WebSocketApp]): List of active WebSocket connections.
        threads (List[threading.Thread]): List of running threads.
        is_running (bool): Flag to control stream execution.
    """

    def __init__(self, config: dict):
        """
        Initialize the HyperliquidStreamManager.

        Args:
            config (dict): Configuration dictionary.
            db_engine: SQLAlchemy database engine for data persistence.
        """
        self.config = config        
        # Get symbols and market type from config
        raw_symbols = config['collection']['symbols']
        self.symbols = [s.upper() for s in raw_symbols]
        self.market_type = config['collection']['market_type']
        
        self.streams = config['collection']['stream_types']
        self.buffer_size = config['settings']['buffer_size']
        self.network = "mainnet"
        self.base_url = "wss://api.hyperliquid.xyz/ws"
        self.candle_interval = "1m"
        
        # Track last flush time for each stream
        self.last_flush_times = {}
        self.active_websockets: List[WebSocketApp] = []
        self.threads: List[threading.Thread] = []
        self.is_running = True

         
    # --- Database Pathing Methods ---

    def _get_schema_path(self) -> str:
        """
        Construct the database schema path.

        Returns:
            str: The schema path in format: 'base_schema_exchange'.
        """
        db_conf = self.config['database']
        return f"{db_conf['base_schema']}_{db_conf['exchange']}"

    def _get_table_name(self, stream_type: str, market_type: str = None, 
                       symbol: str = None) -> str:
        """
        Generate the database table name for a given stream, market type, and symbol.

        Table names follow the pattern:
        - allMids: 'allmids_perpetual' (only for perpetual)
        - Candles: 'candle_{interval}_{market_type}_{symbol}'
        - Others: '{stream_type}_{market_type}_{symbol}'

        Args:
            stream_type (str): Type of stream (e.g., 'l2Book', 'trades').
            market_type (str): Market type ('perpetual' or 'spot').
            symbol (str, optional): Trading symbol.

        Returns:
            str: The constructed table name in lowercase.
        """
        # AllMids is global and doesn't need a symbol (perpetual only)
        if stream_type == 'allMids':
            return f'allmids_{market_type}'.lower()
        
        # Sanitize symbol for table name (replace special chars)
        safe_symbol = symbol.replace('/', '_').replace('@', 'spot_').replace('-', '_') if symbol else ''
        
        # Candle includes interval in table name
        if stream_type == 'candle' and symbol and market_type:
            return f"candle_{self.candle_interval}_{market_type}_{safe_symbol}".lower()
        
        # Standard format for other streams
        if symbol and market_type:
            return f"{stream_type}_{market_type}_{safe_symbol}".lower()
        
        return stream_type.lower()

    # --- Data Parser Methods ---

    def _get_parser(self, stream_type: str) -> Callable:
        """
        Retrieve the appropriate parser function for a stream type.

        Args:
            stream_type (str): Type of stream to parse.

        Returns:
            Callable: Parser function that accepts a dict and returns a dict.
        """
        parsers = {
            'allMids': self._parse_allmids,
            'l2Book': self._parse_l2book,
            'trades': self._parse_trades,
            'candle': self._parse_candle
        }
        return parsers.get(stream_type, lambda x: x)

    def _parse_allmids(self, data: dict) -> dict:
        """Parse allMids stream data."""
        mids_data = data.get('data', {}).get('mids', {})
        return {
            'timestamp': int(datetime.utcnow().timestamp() * 1000),
            'mids': mids_data
        }

    def _parse_l2book(self, data: dict) -> dict:
        """Parse l2Book (Level 2 orderbook) stream data."""
        book_data = data.get('data', {})
        levels = book_data.get('levels', [[], []])
        
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []

        return {
            'timestamp': book_data.get('time', datetime.utcnow().timestamp() * 1000),
            "bids": bids,  
            "asks": asks,             
            'best_bid_price': float(bids[0]['px']) if bids else None,
            'best_bid_size': float(bids[0]['sz']) if bids else None,
            'best_bid_orders': bids[0].get('n', 0) if bids else 0,
            'best_ask_price': float(asks[0]['px']) if asks else None,
            'best_ask_size': float(asks[0]['sz']) if asks else None,
            'best_ask_orders': asks[0].get('n', 0) if asks else 0,
        }

    def _parse_trades(self, data: dict) -> dict:
        """Parse trades stream data."""
        trades_list = data.get('data', [])
        if not trades_list:
            return {}
        
        trade = trades_list[0]
        return {
            'timestamp': trade.get('time', datetime.utcnow().timestamp() * 1000),
            'side': trade.get('side'),
            'price': float(trade.get('px', 0)),
            'size': float(trade.get('sz', 0)),
        }

    def _parse_candle(self, data: dict) -> dict:
        """Parse candle (OHLC) stream data."""
        candle_data = data.get('data', {})
        return {
            'timestamp': candle_data.get('t', datetime.utcnow().timestamp() * 1000),
            'open': float(candle_data.get('o', 0)),
            'high': float(candle_data.get('h', 0)),
            'low': float(candle_data.get('l', 0)),
            'close': float(candle_data.get('c', 0)),
            'volume': float(candle_data.get('v', 0)),
            'num_trades': candle_data.get('n', 0),
        }

    # --- WebSocket Connection Logic ---

    def _run_stream(self, symbol: str = None, stream_type: str = None, 
                   market_type: str = None):
        """
        Run a single WebSocket stream.

        Args:
            symbol (str, optional): Trading symbol.
            stream_type (str): Type of stream to subscribe to.
            market_type (str): Market type ('perpetual' or 'spot').
        """
        buffer = []
        parser = self._get_parser(stream_type)
        table = self._get_table_name(stream_type, market_type, symbol)
        schema = self._get_schema_path()

        def on_message(ws, message):
            """Handle incoming WebSocket messages."""
            try:
                data = json.loads(message)
                # Handle subscription response
                if data.get('channel') == 'subscriptionResponse':
                    logger.info(
                        f"Subscription confirmed for {table}: "
                        f"{data.get('data', {})}"
                    )
                    return
                
                # Handle pong messages
                if data.get('channel') == 'pong':
                    logger.debug(f"Pong received for {table}")
                    return
                
                # Process data messages
                if data.get('channel') == stream_type:
                    if stream_type == 'trades' and isinstance(data.get('data'), list):
                        for trade in data.get('data', []):
                            parsed_data = parser({'data': [trade], 'channel': stream_type})
                            if parsed_data:
                                buffer.append(parsed_data)
                    else:
                        parsed_data = parser(data)
                        if parsed_data:
                            buffer.append(parsed_data)
                else:
                    return

                # Write to database when buffer is full
                if len(buffer) >= self.buffer_size:
                    success = False
                    retries = 2

                    for i in range(retries):
                        try:
                            insert_tick_data(
                                data=buffer,
                                schema=schema,
                                table_name=table,
                            )
                            success = True
                            logger.debug(
                                f"Successfully saved {len(buffer)} records to {table}"
                            )
                            break
                        except Exception as db_err:
                            logger.warning(
                                f"Retry {i+1}/{retries} for {table} failed: {db_err}"
                            )
                            time.sleep(2 ** i)

                    if success:
                        buffer.clear()
                    else:
                        logger.error(
                            f"CRITICAL: Failed to save data for {table} "
                            f"after {retries} retries. Buffer kept."
                        )
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {table}: {e}")
            except Exception as e:
                logger.error(f"Error in {table}: {e}", exc_info=True)

        def on_open(ws):
            """Handle WebSocket connection open event."""
            try:
                subscription = {
                    "method": "subscribe",
                    "subscription": {"type": stream_type}
                }
                
                if stream_type in ['l2Book', 'trades']:
                    subscription["subscription"]["coin"] = symbol
                elif stream_type == 'candle':
                    subscription["subscription"]["coin"] = symbol
                    subscription["subscription"]["interval"] = self.candle_interval
                
                ws.send(json.dumps(subscription))
                logger.info(
                    f"✓ Connection Opened: {table} "
                    f"(market: {market_type}, symbol: {symbol})"
                )
            except Exception as e:
                logger.error(f"Error subscribing to {table}: {e}")

        def on_error(ws, error):
            """Handle WebSocket errors."""
            logger.error(f"WebSocket error in {table}: {error}")

        def on_close(ws, close_status_code, close_msg):
            """Handle WebSocket connection close event."""
            logger.warning(
                f"Connection closed for {table}: "
                f"code={close_status_code}, msg={close_msg}"
            )

        # Create WebSocket connection
        ws = WebSocketApp(
            self.base_url,
            on_message=on_message,
            on_open=on_open,
            on_error=on_error,
            on_close=on_close
        )
        self.active_websockets.append(ws)

        # Keep connection alive with automatic reconnection
        while self.is_running:
            ws.run_forever(ping_interval=20, ping_timeout=10)
            if self.is_running:
                logger.warning(f"Reconnecting {table}...")
                time.sleep(5)

    def start_all(self):
        """Start all configured WebSocket streams."""
        connection_delay = self.config['settings'].get('connection_delay', 0.5)

        for stream in self.streams:
            # Skip allMids for spot markets
            if stream == 'allMids':
                if self.market_type == 'spot':
                    logger.info("Skipping allMids stream for spot market")
                    continue
                
                # allMids is global - start once per market type
                t = threading.Thread(
                    target=self._run_stream,
                    kwargs={
                        'stream_type': stream,
                        'market_type': self.market_type
                    },
                    name=f"{self.market_type}-allMids",
                    daemon=True
                )
                self.threads.append(t)
                t.start()
                time.sleep(connection_delay)
            else:
                # Start stream for each symbol
                for symbol in self.symbols:
                    t = threading.Thread(
                        target=self._run_stream,
                        kwargs={
                            'symbol': symbol,
                            'stream_type': stream,
                            'market_type': self.market_type
                        },
                        name=f"{self.market_type}-{symbol}-{stream}",
                        daemon=True
                    )
                    self.threads.append(t)
                    t.start()
                    time.sleep(connection_delay)

        logger.info(
            f"Started {len(self.threads)} streams for Hyperliquid "
            f"{self.network} ({self.market_type} market)"
        )

    def stop_all(self):
        """Gracefully shut down all WebSocket connections and threads."""
        if not self.is_running:
            return
            
        self.is_running = False
        logger.info(f"Shutting down {len(self.active_websockets)} streams...")

        for ws in self.active_websockets:
            try:
                ws.close()
            except:
                pass

        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2.0)

        logger.info("HyperLiquid shutdown complete.")