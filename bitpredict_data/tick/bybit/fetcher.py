"""
Bybit WebSocket Stream Manager Module.

This module provides functionality to manage multiple Bybit WebSocket streams
for cryptocurrency market data collection and database storage orchestration.
Supports multiple stream types including orderbook, RPI orderbook, trade,
ticker, kline, and liquidations.
"""

import json
import time
import logging
import threading
from typing import Callable, List, Dict, Any, Optional
from websocket import WebSocketApp
from bitpredict.common.db.services import insert_tick_data

# Configure module logger
logger = logging.getLogger(__name__)


class BybitStreamManager:
    """
    Manages multiple Bybit WebSocket streams and database orchestration.

    This class handles WebSocket connections to Bybit for various market data
    streams including orderbook (regular and RPI), trade, ticker, kline, and
    liquidations. It buffers incoming data and persists it to a database.

    Attributes:
        config (dict): Configuration dictionary containing collection settings,
                      database settings, and URLs.
        engine: SQLAlchemy database engine for data persistence.
        symbols (list): List of trading symbols to monitor.
        streams (list): List of stream types to subscribe to.
        buffer_size (int): Number of records to buffer before database write.
        market_type (str): Market type - 'linear', 'inverse', or 'spot'.
        base_url (str): Base WebSocket URL for the market type.
        depth (int): Order book depth level (default: 50).
        last_flush_times (dict): Tracks last flush time for each stream.
        active_websockets (List[WebSocketApp]): List of active WebSocket
                                                 connections.
        threads (List[threading.Thread]): List of running threads.
        is_running (bool): Flag to control stream execution.
    """

    def __init__(self, config: dict):
        """
        Initialize the BybitStreamManager.

        Args:
            config (dict): Configuration dictionary with the following structure:
                          - collection: symbols, stream_types, market_type, depth
                          - settings: buffer_size, connection_delay
                          - database: base_schema, exchange
                          - urls: linear, inverse, spot
            db_engine: SQLAlchemy database engine for data persistence.
        """
        self.config = config
        raw_symbols = config['collection']['symbols']
        self.symbols = [self.map_symbol(s) for s in raw_symbols]        
        self.streams = config['collection']['stream_types']
        self.buffer_size = config['settings']['buffer_size']
        self.market_type = config['collection'].get(
            'market_type', 'linear'
        ).lower()
        self.depth = config['collection'].get('depth', 50)
        self.ticker_states: Dict[str, Dict[str, Any]] = {}
        self.active_websockets: List[WebSocketApp] = []
        self.threads: List[threading.Thread] = []
        self.is_running = True

        if self.market_type == 'linear':
            self.base_url = "wss://stream.bybit.com/v5/public/linear"
        elif self.market_type == 'inverse':
            self.base_url = "wss://stream.bybit.com/v5/public/inverse"
        else:
            self.base_url = "wss://stream.bybit.com/v5/public/spot"

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

    # --- Database Pathing Methods ---

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
        - Order book: 'orderbook_{depth}_{symbol}'
        - RPI Order book: 'rpi_orderbook_{depth}_{symbol}'
        - Others: '{stream_type}_{symbol}'

        Args:
            stream_type (str): Type of stream (e.g., 'orderbook', 'trade').
            symbol (str): Trading symbol (e.g., 'BTCUSDT').

        Returns:
            str: The constructed table name in lowercase.
        """
        cleaned_symbol = symbol.replace('USDT', '').lower()

        if stream_type == 'orderbook':
            return f"{stream_type}_{market_type}_{self.depth}_{cleaned_symbol}"
        elif stream_type == 'rpi_orderbook':
            return f"rpiorderbook_{market_type}_{self.depth}_{cleaned_symbol}"
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
            'orderbook': self._parse_orderbook,
            'rpi_orderbook': self._parse_rpi_orderbook,
            'trade': self._parse_trade,
            'ticker': self._parse_ticker,
            'kline': self._parse_kline,
            'liquidation': self._parse_liquidation
        }
        return parsers.get(stream_type, lambda x: x)

    def _parse_orderbook(self, data: dict) -> dict:
        """
        Parse public orderbook stream data.

        Extracts the best bid and ask prices with their quantities from
        the order book snapshot or delta update.

        Args:
            data (dict): Raw WebSocket message containing orderbook data.
                        Expected keys: 'ts' (timestamp), 'b' (bids),
                        'a' (asks).

        Returns:
            dict: Parsed data with timestamp, best bid/ask prices and
                 quantities, update type, and configured depth.
        """
        # Extract data from the nested structure
        orderbook_data = data.get('data', {})
        
        # Get bids and asks arrays
        bids = orderbook_data.get('b', [])
        asks = orderbook_data.get('a', [])

        return {
            'timestamp': data.get('ts'),
            'update_type': data.get('type', 'snapshot'),
            "bids": bids,
            "asks": asks,
            'best_bid_price': float(bids[0][0]) if bids else None,
            'best_bid_qty': float(bids[0][1]) if bids else None,
            'best_ask_price': float(asks[0][0]) if asks else None,
            'best_ask_qty': float(asks[0][1]) if asks else None,
        }

    def _parse_rpi_orderbook(self, data: dict) -> dict:
        """
        Parse RPI (Risk Protection Index) orderbook stream data.

        The RPI orderbook provides order book data with additional risk
        metrics and protection information.

        Args:
            data (dict): Raw WebSocket message containing RPI orderbook data.
                        Expected keys: 'ts' (timestamp), 'data' (orderbook).

        Returns:
            dict: Parsed data with timestamp, best bid/ask with RPI info,
                 and configured depth.
        """
        # Extract data from the nested structure

        orderbook_data = data.get('data', {})
        
        # Get bids and asks arrays
        bids = orderbook_data.get('b', [])
        asks = orderbook_data.get('a', [])

        return {
            'timestamp': data.get('ts'),
            'update_type': data.get('type', 'snapshot'),
            "bids": bids,
            "asks": asks,
            'best_bid_price': float(bids[0][0]) if bids else None,
            'best_bid_qty': float(bids[0][1]) if bids else None,
            'best_ask_price': float(asks[0][0]) if asks else None,
            'best_ask_qty': float(asks[0][1]) if asks else None,
            'sequence': orderbook_data.get('seq'),
            

        }

    def _parse_trade(self, data: dict) -> dict:
        """
        Parse public trade stream data.

        Captures executed trade information including price, quantity,
        side, and execution timestamp.

        Args:
            data (dict): Raw WebSocket message containing trade data.
                        Expected keys: 'ts' (timestamp), 'data' (trade list).

        Returns:
            dict: Parsed data with trade execution details.
        """

     
        # Bybit sends trades as a list in data
        trade_data = data.get('data', [{}])[0]
        
        return {
            'timestamp': data.get('ts'),
            'price': float(trade_data.get('p', 0)),
            'quantity': float(trade_data.get('v', 0)),
            'side': trade_data.get('S'),
            'size': float(trade_data.get('v') or 0),
            'tick_direction': trade_data.get('L'),  # This is your 'L' field
            'trade_time': trade_data.get('T'),
            'block_trade': trade_data.get('BT', False),
            

        }


    def _parse_ticker(self, data: dict, symbol: str) -> Optional[dict]:
        """Parse ticker with state management and forward fill."""
        ticker_data = data.get('data', {})
        msg_type = data.get('type')
        
        # Initialize state for this symbol if needed
        if symbol not in self.ticker_states:
            self.ticker_states[symbol] = {}
        
        state = self.ticker_states[symbol]
        
        # Always update timestamp and type
        state['timestamp'] = data.get('ts')
        state['type'] = msg_type
        
        # Update only non-null fields (forward fill)
        if ticker_data.get('lastPrice') is not None:
            state['last_price'] = ticker_data.get('lastPrice')
        if ticker_data.get('prevPrice24h') is not None:
            state['prev_price_24h'] = ticker_data.get('prevPrice24h')
        if ticker_data.get('price24hPcnt') is not None:
            state['price_24h_pcnt'] = ticker_data.get('price24hPcnt')
        if ticker_data.get('highPrice24h') is not None:
            state['high_price_24h'] = ticker_data.get('highPrice24h')
        if ticker_data.get('lowPrice24h') is not None:
            state['low_price_24h'] = ticker_data.get('lowPrice24h')
        if ticker_data.get('volume24h') is not None:
            state['volume_24h'] = ticker_data.get('volume24h')
        if ticker_data.get('turnover24h') is not None:
            state['turnover_24h'] = ticker_data.get('turnover24h')
        if ticker_data.get('bid1Price') is not None:
            state['bid1_price'] = ticker_data.get('bid1Price')
        if ticker_data.get('bid1Size') is not None:
            state['bid1_size'] = ticker_data.get('bid1Size')
        if ticker_data.get('ask1Price') is not None:
            state['ask1_price'] = ticker_data.get('ask1Price')
        if ticker_data.get('ask1Size') is not None:
            state['ask1_size'] = ticker_data.get('ask1Size')
        
        # Only return complete records (after first snapshot)
        if state.get('last_price') is not None:
            return dict(state)  # Return a copy
        
        return None  # Skip until we get first snapshot        
    
    
    def _parse_kline(self, data: dict) -> dict:
        """
        Parse kline (candlestick) stream data.

        Provides OHLCV (Open, High, Low, Close, Volume) data for
        specified time intervals.

        Args:
            data (dict): Raw WebSocket message containing kline data.
                        Expected keys: 'ts' (timestamp), 'data' (kline list).

        Returns:
            dict: Parsed data with OHLCV information and confirmation status.
        """
        kline_data = data.get('data', [{}])[0]
        
        return {
            'timestamp': data.get('ts'),
            'start_time': kline_data.get('start'),
            'end_time': kline_data.get('end'),
            'open': float(kline_data.get('open', 0)),
            'high': float(kline_data.get('high', 0)),
            'low': float(kline_data.get('low', 0)),
            'close': float(kline_data.get('close', 0)),
            'volume': float(kline_data.get('volume', 0)),
            'turnover': float(kline_data.get('turnover', 0)),
            'confirm': kline_data.get('confirm', False),
            

        }

    def _parse_liquidation(self, data: dict) -> dict:
        """
        Parse liquidation order stream data.

        Captures details of liquidation events including order information,
        execution details, and liquidation-specific metadata.

        Args:
            data (dict): Raw WebSocket message containing liquidation data.
                        Expected keys: 'ts' (timestamp),
                        'data' (liquidation info).

        Returns:
            dict: Parsed data with comprehensive liquidation information.
        """
        liq_data = data.get('data', [{}])[0]  # Get first item from list
        
        return {
            'timestamp': data.get('ts'),  # 'ts' not 'T'
            'side': liq_data.get('S'),
            'price': float(liq_data.get('p', 0)),
            'size': float(liq_data.get('v', 0)),  # lowercase 'v' not 'V'
            'update_time': liq_data.get('T')      # Updated timestamp

            
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

        # Map stream types to their WebSocket topics
        # Bybit uses different topic formats based on stream type
        topic_map = {
            'orderbook': f"orderbook.{self.depth}.{symbol}",
            'rpi_orderbook': f"orderbook.rpi.{symbol}",
            'trade': f"publicTrade.{symbol}",
            'ticker': f"tickers.{symbol}",
            'kline': f"kline.1.{symbol}",  # 1 minute kline
            'liquidation': f"allLiquidation.{symbol}"
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
                
                # Skip ping/pong and subscription confirmation messages
                if data.get('op') in ['ping', 'pong', 'subscribe']:
                    return
                
                if stream_type == 'ticker':
                    parsed = parser(data, symbol)
                    if parsed:
                        buffer.append(parsed)
                else:
                    buffer.append(parser(data))

                

                # Check if buffer has reached the configured size
                if len(buffer) >= self.buffer_size:
                    # Convert buffer to DataFrame for batch processing
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

            Sends subscription message to Bybit WebSocket server.

            Args:
                ws: WebSocket instance.
            """
            # Subscribe to the specific topic
            subscribe_message = {
                "op": "subscribe",
                "args": [topic_map[stream_type]]
            }
            ws.send(json.dumps(subscribe_message))
            logger.info(f"✓ Connection Opened and Subscribed: {table}")

        def on_error(ws, error):
            """
            Handle WebSocket errors.

            Args:
                ws: WebSocket instance.
                error: Error object or message.
            """
            logger.error(f"WebSocket error in {table}: {error}")

        def on_close(ws, close_status_code, close_msg):
            """
            Handle WebSocket connection close event.

            Args:
                ws: WebSocket instance.
                close_status_code: Status code for connection closure.
                close_msg: Close message from server.
            """
            logger.warning(
                f"Connection closed for {table}: "
                f"code={close_status_code}, msg={close_msg}"
            )

        # Create WebSocket connection with all handlers
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
                if self.market_type == 'spot' and stream == 'liquidation':
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

        logger.info(
            f"Started {len(self.threads)} streams for Bybit "
            f"{self.market_type} market"
        )

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

        logger.info("Bybit shutdown complete.")