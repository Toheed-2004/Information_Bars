"""
Base WebSocket Stream Manager Module.

This module provides a base class for managing WebSocket streams across different
cryptocurrency exchanges. It handles common functionality like connection management,
buffering, database operations, and threading.
"""

import json
import time
import threading
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Optional
from websocket import WebSocketApp
from bitpredict.common.db.services import insert_tick_data
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


class BaseStreamManager(ABC):
    """
    Abstract base class for managing WebSocket streams and database orchestration.

    This class provides common functionality for WebSocket connections including:
    - Connection lifecycle management
    - Data buffering and database persistence
    - Thread management
    - Automatic reconnection logic
    - Retry mechanisms for database operations

    Subclasses must implement exchange-specific methods for:
    - Symbol mapping
    - Table naming
    - Data parsing
    - WebSocket URL construction
    - Subscription message creation
    """

    def __init__(self, config: dict):
        """
        Initialize the BaseStreamManager.

        Args:
            config (dict): Configuration dictionary with structure:
                          - collection: symbols, stream_types, market_type, depth
                          - settings: buffer_size, connection_delay
                          - database: base_schema, exchange
        """
        self.config = config
        
        # Map symbols using exchange-specific mapping
        raw_symbols = config['collection']['symbols']
        self.symbols = [self._map_symbol(s) for s in raw_symbols]
        
        # Stream configuration
        self.streams = config['collection']['stream_types']
        self.buffer_size = config['settings']['buffer_size']
        self.market_type = config['collection'].get('market_type', 'spot').lower()
        self.depth = config['collection'].get('depth', 10)
        
        # Connection management
        self.active_websockets: List[WebSocketApp] = []
        self.threads: List[threading.Thread] = []
        self.is_running = True
        
        # Data buffers for each symbol/stream combination
        self.data_buffers: Dict[tuple, List] = {}

    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def _map_symbol(self, symbol: str) -> str:
        """
        Map generic symbol to exchange-specific format.

        Args:
            symbol (str): Generic symbol (e.g., "BTC", "ETH")

        Returns:
            str: Exchange-specific symbol format
        """
        pass

    @abstractmethod
    def _get_table_name(self, stream_type: str, symbol: str) -> str:
        """
        Generate database table name for a stream and symbol.

        Args:
            stream_type (str): Type of stream (e.g., 'trades', 'order_book')
            symbol (str): Trading symbol (exchange-specific format)

        Returns:
            str: Table name for database storage
        """
        pass

    @abstractmethod
    def _get_parser(self, stream_type: str) -> Callable:
        """
        Get the appropriate parser function for a stream type.

        Args:
            stream_type (str): Type of stream to parse

        Returns:
            Callable: Parser function that accepts a dict and returns
                     a dict or list of dicts
        """
        pass

    @abstractmethod
    def _get_websocket_url(self, stream_type: str) -> str:
        """
        Get the WebSocket URL for a specific stream type.

        Args:
            stream_type (str): Type of stream

        Returns:
            str: WebSocket URL endpoint
        """
        pass

    @abstractmethod
    def _create_subscription_message(
        self, 
        symbol: str, 
        stream_type: str
    ) -> dict:
        """
        Create exchange-specific subscription message.

        Args:
            symbol (str): Trading symbol (exchange-specific format)
            stream_type (str): Type of stream to subscribe to

        Returns:
            dict: Subscription message to send via WebSocket
        """
        pass

    @abstractmethod
    def _should_skip_message(self, data: dict, stream_type: str) -> bool:
        """
        Determine if a message should be skipped (e.g., heartbeats, status).

        Args:
            data (dict): Parsed JSON message from WebSocket
            stream_type (str): Type of stream

        Returns:
            bool: True if message should be skipped, False otherwise
        """
        pass

    # ========================================================================
    # CONCRETE METHODS - Common functionality
    # ========================================================================

    def _get_schema_path(self) -> str:
        """
        Construct the database schema path.

        Combines the base schema name with the exchange name to create
        a fully qualified schema path.

        Returns:
            str: The schema path in format: 'base_schema_exchange'
        """
        db_conf = self.config['database']
        return f"{db_conf['base_schema']}_{db_conf['exchange']}"

    def _flush_buffer_to_db(
        self,
        buffer: List[dict],
        schema: str,
        table_name: str,
        retries: int = 2
    ) -> bool:
        """
        Flush buffered data to database with retry logic.

        Args:
            buffer (List[dict]): List of records to insert
            schema (str): Database schema name
            table_name (str): Table name
            retries (int): Number of retry attempts (default: 2)

        Returns:
            bool: True if successful, False otherwise
        """
        success = False

        for i in range(retries):
            try:
                insert_tick_data(
                    data=buffer,
                    schema=schema,
                    table_name=table_name,
                )
                success = True
                logger.debug(f"Saved {len(buffer)} records to {table_name}")
                break
            except Exception as db_err:
                logger.warning(
                    f"Retry {i+1}/{retries} for {table_name} failed: {db_err}"
                )
                # Exponential backoff
                time.sleep(2 ** i)

        if not success:
            logger.error(
                f"CRITICAL: Failed to save data for {table_name} "
                f"after {retries} retries. Buffer kept."
            )

        return success

    def _create_on_message_handler(
        self,
        buffer: List[dict],
        parser: Callable,
        table_name: str,
        schema: str,
        stream_type: str
    ) -> Callable:
        """
        Create the on_message callback for WebSocket.

        Args:
            buffer (List[dict]): Buffer to store parsed data
            parser (Callable): Parser function for the stream type
            table_name (str): Database table name
            schema (str): Database schema name
            stream_type (str): Type of stream

        Returns:
            Callable: on_message handler function
        """
        def on_message(ws, message):
            try:
                data = json.loads(message)

                # Skip heartbeats, status messages, etc.
                if self._should_skip_message(data, stream_type):
                    return

                # Parse data
                parsed_data = parser(data)

                # Handle both single records and lists
                if isinstance(parsed_data, list):
                    buffer.extend(parsed_data)
                elif parsed_data:
                    buffer.append(parsed_data)

                # Flush to database when buffer is full
                if len(buffer) >= self.buffer_size:
                    success = self._flush_buffer_to_db(
                        buffer=buffer,
                        schema=schema,
                        table_name=table_name
                    )

                    if success:
                        buffer.clear()

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {table_name}: {e}")
            except Exception as e:
                logger.error(f"Error in {table_name}: {e}", exc_info=True)

        return on_message

    def _create_on_open_handler(
        self,
        symbol: str,
        stream_type: str,
        table_name: str
    ) -> Callable:
        """
        Create the on_open callback for WebSocket.

        Args:
            symbol (str): Trading symbol
            stream_type (str): Type of stream
            table_name (str): Table name for logging

        Returns:
            Callable: on_open handler function
        """
        def on_open(ws):
            try:
                # Create exchange-specific subscription message
                subscribe_message = self._create_subscription_message(
                    symbol, stream_type
                )
                ws.send(json.dumps(subscribe_message))
                logger.info(
                    f"✓ Connected: {table_name} ({self.market_type.upper()})"
                )
            except Exception as e:
                logger.error(f"Error subscribing to {table_name}: {e}")

        return on_open

    def _create_on_error_handler(self, table_name: str) -> Callable:
        """
        Create the on_error callback for WebSocket.

        Args:
            table_name (str): Table name for logging

        Returns:
            Callable: on_error handler function
        """
        def on_error(ws, error):
            logger.error(f"WebSocket error in {table_name}: {error}")

        return on_error

    def _create_on_close_handler(self, table_name: str) -> Callable:
        """
        Create the on_close callback for WebSocket.

        Args:
            table_name (str): Table name for logging

        Returns:
            Callable: on_close handler function
        """
        def on_close(ws, close_status_code, close_msg):
            logger.warning(
                f"Connection closed for {table_name}: {close_status_code}"
            )

        return on_close

    def _run_stream(self, symbol: str, stream_type: str):
        """
        Run a single WebSocket stream for a symbol and stream type.

        This method establishes a WebSocket connection, buffers incoming data,
        and periodically writes to the database when buffer is full.

        Args:
            symbol (str): Trading symbol (exchange-specific format)
            stream_type (str): Type of stream to subscribe to
        """
        # Initialize buffer and get configurations
        buffer = []
        parser = self._get_parser(stream_type)
        table_name = self._get_table_name(stream_type, symbol)
        schema = self._get_schema_path()
        ws_url = self._get_websocket_url(stream_type)

        # Track buffer for this stream
        buffer_key = (symbol, stream_type)
        self.data_buffers[buffer_key] = buffer

        # Create WebSocket handlers
        on_message = self._create_on_message_handler(
            buffer, parser, table_name, schema, stream_type
        )
        on_open = self._create_on_open_handler(symbol, stream_type, table_name)
        on_error = self._create_on_error_handler(table_name)
        on_close = self._create_on_close_handler(table_name)

        # Create WebSocket connection
        ws = WebSocketApp(
            ws_url,
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
                logger.warning(f"Reconnecting {table_name}...")
                time.sleep(5)

    def start_all(self):
        """
        Start all configured WebSocket streams.

        Spawns a separate daemon thread for each symbol/stream combination.
        Includes a configurable delay between connection attempts to avoid
        overwhelming the WebSocket server.
        """
        connection_delay = self.config['settings'].get('connection_delay', 0.5)

        # Iterate through all symbol and stream combinations
        for symbol in self.symbols:
            for stream in self.streams:
                # Check if stream should be skipped for this market type
                if self._should_skip_stream(stream):
                    logger.warning(
                        f"Skipping {stream} for {self.market_type} market."
                    )
                    continue

                # Create and start a daemon thread for each stream
                t = threading.Thread(
                    target=self._run_stream,
                    args=(symbol, stream),
                    name=f"{self.market_type}-{symbol}-{stream}",
                    daemon=True
                )
                self.threads.append(t)
                t.start()

                # Delay between connections to prevent rate limiting
                time.sleep(connection_delay)

        logger.info(
            f"Started {len(self.threads)} {self.market_type.upper()} streams"
        )

    def stop_all(self):
        """
        Gracefully shut down all WebSocket connections and threads.

        Closes all active WebSocket connections and waits for threads to
        complete their current operations before terminating.
        """
        if not self.is_running:
            return

        self.is_running = False
        logger.info(
            f"Shutting down {len(self.active_websockets)} "
            f"{self.market_type.upper()} streams..."
        )

        # Close all WebSocket connections
        for ws in self.active_websockets:
            try:
                ws.close()
            except Exception:
                pass

        # Wait for threads to finish
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2.0)

        exchange = self.config['database']['exchange']
        logger.info(
            f"{exchange.capitalize()} {self.market_type.upper()} shutdown complete."
        )

    def _should_skip_stream(self, stream_type: str) -> bool:
        """
        Determine if a stream should be skipped based on market type.

        Default implementation - can be overridden by subclasses.

        Args:
            stream_type (str): Type of stream

        Returns:
            bool: True if stream should be skipped, False otherwise
        """
        return False
