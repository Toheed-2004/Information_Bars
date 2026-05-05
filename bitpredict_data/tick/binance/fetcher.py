"""
Binance WebSocket Stream Manager Module.

This module provides functionality to manage multiple Binance WebSocket streams
for cryptocurrency market data collection and database storage orchestration.
"""

import json
import time
import threading
from typing import Callable, List
from websocket import WebSocketApp
from bitpredict.common.db.services import insert_tick_data
from dotenv import load_dotenv
from bitpredict.common.logging import get_logger

load_dotenv()

logger = get_logger(__name__)


class BinanceStreamManager:
    """
    Manages multiple Binance WebSocket streams and database orchestration.

    This class handles WebSocket connections to Binance for various market data
    streams including mark price, order book, klines, trades, tickers, and
    liquidations. It buffers incoming data and persists it to a database.

    Attributes:
        config (dict): Configuration dictionary containing collection settings,
                      database settings, and URLs.
        engine: SQLAlchemy database engine for data persistence.
        symbols (list): List of trading symbols to monitor.
        streams (list): List of stream types to subscribe to.
        buffer_size (int): Number of records to buffer before database write.
        market_type (str): Market type - 'futures' or 'spot'.
        base_url (str): Base WebSocket URL for the market type.
        depth (int): Order book depth level (default: 5).
        last_flush_times (dict): Tracks last flush time for each stream.
        active_websockets (List[WebSocketApp]): List of active WebSocket
                                                 connections.
        threads (List[threading.Thread]): List of running threads.
        is_running (bool): Flag to control stream execution.
    """

    def __init__(self, config: dict):
        """
        Initialize the BinanceStreamManager.

        Args:
            config (dict): Configuration dictionary with the following structure:
                          - collection: symbols, stream_types, market_type, depth
                          - settings: buffer_size, connection_delay
                          - database: base_schema, exchange
                          - urls: futures, spot
            db_engine: SQLAlchemy database engine for data persistence.
        """
        self.config = config
        raw_symbols = config['collection']['symbols']
        self.symbols = [self.map_symbol(s) for s in raw_symbols]
        self.streams = config['collection']['stream_types']
        self.buffer_size = config['settings']['buffer_size']
        self.market_type = config['collection'].get(
            'market_type', 'futures'
        ).lower()
        self.depth = config['collection'].get('depth', 5)
        self.data_buffers = {}
        self.active_websockets: List[WebSocketApp] = []
        self.threads: List[threading.Thread] = []
        self.is_running = True
        if self.market_type == 'spot':
            self.base_url = "wss://stream.binance.com:9443/ws"
        else:
            self.base_url = "wss://fstream.binance.com/ws"

    def map_symbol(self, symbol: str) -> str:
        """
        Standardizes symbol to uppercase and ensures 'USDT' suffix.
        Example: 'sol' -> 'SOLUSDT', 'BTC-USDT' -> 'BTCUSDT'
        """
        # 1. Convert to uppercase
        clean_symbol = symbol.upper()
        
        # 2. Remove any delimiters (dashes/underscores) common in config files
        clean_symbol = clean_symbol.replace("-", "").replace("_", "")
        
        # 3. Append USDT if not already present
        if not clean_symbol.endswith("USDT"):
            clean_symbol = f"{clean_symbol}USDT"
            
        return clean_symbol
    
    def _get_schema_path(self) -> str:
        """
        Construct the database schema path.

        Combines the base schema name with the exchange name to create
        a fully qualified schema path.

        Returns:
            str: The schema path in format: 'base_schema_exchange'.
        """
        db_conf = self.config['database']
        return f"{db_conf['base_schema']}_{db_conf['exchange']}"

    def _get_table_name(self, stream_type: str, symbol: str, market_type: str) -> str:
        """
        Generate the database table name for a given stream and symbol.

        Table names follow the pattern:
        - Order book: 'order_book_{depth}_{symbol}'
        - Others: '{stream_type}_{symbol}'

        Args:
            stream_type (str): Type of stream (e.g., 'order_book', 'trades').
            symbol (str): Trading symbol (e.g., 'BTCUSDT').

        Returns:
            str: The constructed table name in lowercase.
        """
        cleaned_symbol = symbol.replace('USDT', '').lower()
        if stream_type == 'order_book':
            return f"orderbook_{market_type}_{self.depth}_{cleaned_symbol}"
        if stream_type == 'book_ticker':
            return f"bookticker_{market_type}_{self.depth}_{cleaned_symbol}"
        return f"{stream_type}_{market_type}_{cleaned_symbol}"

    # --- Data Parser Methods ---

    def _get_parser(self, stream_type: str) -> Callable:
        """
        Retrieve the appropriate parser function for a stream type.

        Args:
            stream_type (str): Type of stream to parse.

        Returns:
            Callable: Parser function that accepts a dict and returns a dict.
                     Returns identity function if stream type not found.
        """
        parsers = {
            'mark_price': self._parse_mark_price,
            'order_book': self._parse_order_book,
            'kline': self._parse_kline,
            'trades': self._parse_trades,
            'book_ticker': self._parse_book_ticker,
            'ticker': self._parse_ticker,
            'liquidations': self._parse_liquidations
        }
        return parsers.get(stream_type, lambda x: x)

    def _parse_mark_price(self, data: dict) -> dict:
        """
        Parse mark price stream data.

        Args:
            data (dict): Raw WebSocket message containing mark price data.
                        Expected keys: 'E' (event time), 'p' (mark price),
                        'r' (funding rate).

        Returns:
            dict: Parsed data with timestamp, mark_price, and funding_rate.
        """
        return {
            'timestamp': data['E'],
            'mark_price': float(data['p']),
            'funding_rate': float(data['r']),
            
        }

    def _parse_order_book(self, data: dict) -> dict:
        """
        Unified parser for Binance Spot & Futures partial depth streams.
        """

        # --- Detect payload type ---
        if self.market_type == "spot":
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            timestamp = int(time.time() * 1000)  # Spot has no event time
        else:
            bids = data.get("b", [])
            asks = data.get("a", [])
            timestamp = data.get("E")

        return {
            "timestamp": timestamp,
            "bids": bids,
            "asks": asks,
            "best_bid_price": float(bids[0][0]) if bids else None,
            "best_bid_qty": float(bids[0][1]) if bids else None,
            "best_ask_price": float(asks[0][0]) if asks else None,
            "best_ask_qty": float(asks[0][1]) if asks else None,
            "config_depth": self.depth,
        }


    def _parse_kline(self, data: dict) -> dict:
        """
        Parse kline (candlestick) stream data.

        Args:
            data (dict): Raw WebSocket message containing kline data.
                        Expected key: 'k' (kline data dict).

        Returns:
            dict: Parsed data with OHLCV (Open, High, Low, Close, Volume)
                 information, close status, and number of trades.
        """
        k = data['k']
        return {
            'timestamp': k['t'],
            'open': float(k['o']),
            'high': float(k['h']),
            'low': float(k['l']),
            'close': float(k['c']),
            'volume': float(k['v']),
            'closed': k['x'],
            'no_of_trades': float(k['n']),
        }

    def _parse_trades(self, data: dict) -> dict:
        """
        Parse trade stream data.

        Args:
            data (dict): Raw WebSocket message containing trade data.
                        Expected keys: 'T' (trade time), 'p' (price),
                        'q' (quantity), 'm' (is buyer market maker).

        Returns:
            dict: Parsed data with timestamp, price, quantity, and
                 market maker flag.
        """
        return {
            'timestamp': data['T'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'market_maker': data['m'],
            

        }

    def _parse_book_ticker(self, data: dict) -> dict:
        """
        Unified parser for Binance Spot & Futures bookTicker streams.
        """

        # Futures has T/E, Spot has none
        timestamp = (
            data.get("T")
            or data.get("E")
            or int(time.time() * 1000)
        )

        return {
            "timestamp": timestamp,
            "bid_price": float(data["b"]),
            "bid_qty": float(data["B"]),
            "ask_price": float(data["a"]),
            "ask_qty": float(data["A"]),
        }


    def _parse_ticker(self, data: dict) -> dict:
        """
        Parse 24hr ticker stream data.

        Args:
            data (dict): Raw WebSocket message containing 24hr ticker data.

        Returns:
            dict: Parsed data with price changes, volume, and high/low prices.
        """
    
        return {
            'timestamp': data['E'],
            'price_change': float(data['p']),
            'price_change_percent': float(data['P']),
            'weighted_average_price': float(data['w']),
            'last_price': float(data['c']),
            'volume': float(data['v']),
            'high': float(data['h']),
            'low': float(data['l']),
            

        }

    def _parse_liquidations(self, data: dict) -> dict:
        """
        Parse liquidation order snapshot data.

        Captures details of liquidation orders including side, type, price,
        and execution status.

        Args:
            data (dict): Raw WebSocket message containing liquidation data.
                        Expected keys: 'E' (event time), 'o' (order details).

        Returns:
            dict: Parsed data with comprehensive liquidation order information.
        """
        o = data['o']
        return {
            'timestamp': data['E'],
            'side': o['S'],
            'order_type': o['o'],
            'time_in_force': o['f'],
            'orig_qty': float(o['q']),
            'price': float(o['p']),
            'avg_price': float(o['ap']),
            'order_status': o['X'],
            'last_filled_qty': float(o['l']),
            'acc_filled_qty': float(o['z']),
            'event_time': data['E'],
            'trade_time': o['T'],
            

        }

    # --- WebSocket Connection Logic ---

    def _run_stream(self, symbol: str, stream_type: str):
        """
        Run a single WebSocket stream for a symbol and stream type.

        This method establishes a WebSocket connection, buffers incoming data,
        and periodically writes to the database when buffer is full.

        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT').
            stream_type (str): Type of stream to subscribe to.
        """
        # Initialize buffer and get appropriate parser
        buffer = []
        parser = self._get_parser(stream_type)
        table = self._get_table_name(stream_type, symbol, self.market_type)
        schema = self._get_schema_path()

        buffer_key = (symbol, stream_type)
        self.data_buffers[buffer_key] = []
        
        # Map stream types to their WebSocket endpoint suffixes
        url_map = {
            'mark_price': f"{symbol.lower()}@markPrice@1s",
            'order_book': f"{symbol.lower()}@depth{self.depth}@100ms",
            'kline': f"{symbol.lower()}@kline_1m",
            'trades': (f"{symbol.lower()}@aggTrade"
                      if self.market_type == 'futures'
                      else f"{symbol.lower()}@trade"),
            'book_ticker': f"{symbol.lower()}@bookTicker",
            'ticker': f"{symbol.lower()}@ticker",
            'liquidations': f"{symbol.lower()}@forceOrder",
        }

        def on_message(ws, message):
            """
            Handle incoming WebSocket messages.

            Parses messages, buffers data, and writes to database when
            buffer reaches configured size.

            Args:
                ws: WebSocket instance.
                message (str): Raw JSON message from WebSocket.
            """
            try:
                # Parse incoming JSON message
                data = json.loads(message)
                buffer.append(parser(data))

                # Check if buffer has reached the configured size
                if len(buffer) >= self.buffer_size:
                    # Convert buffer to DataFrame for batch processing
                    # df = pd.DataFrame(buffer)
                    success = False
                    retries = 2

                    # Retry logic for database operations
                    for i in range(retries):


                        try:
                            # Attempt to save data to database
                            insert_tick_data(
                                data=buffer,
                                schema=schema,
                                table_name=table,
                            )
                            success = True
                            break  # Exit retry loop on success
                        except Exception as db_err:
                            logger.warning(
                                f"Retry {i+1}/{retries} for {table} "
                                f"failed: {db_err}"
                            )
                            # Exponential backoff between retries
                            time.sleep(2 ** i)

                    if success:
                        buffer.clear()
                    else:
                        logger.error(
                            f"CRITICAL: Failed to save data for {table} "
                            f"after {retries} retries. Buffer kept."
                        )
            except Exception as e:
                logger.error(f"Error in {table}: {e}")

        def on_open(ws):
            """
            Handle WebSocket connection open event.

            Args:
                ws: WebSocket instance.
            """
            logger.info(f"✓ Connection Opened: {table}")

        # Construct WebSocket URL and create connection
        ws_url = f"{self.base_url}/{url_map[stream_type]}"
        ws = WebSocketApp(ws_url, on_message=on_message, on_open=on_open)
        self.active_websockets.append(ws)

        # Keep connection alive with automatic reconnection
        while self.is_running:
            ws.run_forever(ping_interval=20, ping_timeout=10)
            if self.is_running:
                logger.warning(f"Reconnecting {table}...")
                time.sleep(5)

    def start_all(self):
        """
        Start all configured WebSocket streams.

        Spawns a separate daemon thread for each symbol/stream combination.
        Includes a configurable delay between connection attempts to avoid
        overwhelming the WebSocket server.
        """
        connection_delay = self.config['settings'].get(
            'connection_delay', 0.5
        )

        # Iterate through all symbol and stream combinations
        for symbol in self.symbols:
            for stream in self.streams:
                # Skip mark_price stream for spot markets (not available)
                if self.market_type == 'spot' and stream in ("mark_price", "liquidations"):
                    logger.warning(f"Skipping {stream} for Spot market.")
                    continue

                # Create and start a daemon thread for each stream
                t = threading.Thread(
                    target=self._run_stream,
                    args=(symbol, stream),
                    name=f"{symbol}-{stream}",
                    daemon=True
                )
                self.threads.append(t)
                t.start()

                # Delay between connections to prevent rate limiting
                time.sleep(connection_delay)


    def stop_all(self):
        """Gracefully shut down all WebSocket connections and threads."""
        if not self.is_running:
            return
            
        self.is_running = False
        logger.info(f"Shutting down {len(self.active_websockets)} streams...")

        # 1. Close all Sockets (This breaks the blocking run_forever loop)
        for ws in self.active_websockets:
            try:
                ws.close()
            except:
                pass

        # 2. Join Threads (Wait for them to finish saving their last buffers)
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2.0)

        logger.info("Binance shutdown complete.")