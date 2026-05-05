"""
Kraken WebSocket Stream Manager Module.

This module provides a unified stream manager that handles both Spot (V2 API)
and Futures (V1 API) cryptocurrency market data collection with automatic
symbol mapping and database storage orchestration.
"""

import json
import time
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


class KrakenStreamManager:
    """
    Unified manager for Kraken WebSocket streams (Spot V2 and Futures V1).

    Handles WebSocket connections for both spot and futures market data streams
    with automatic symbol mapping, buffers incoming data, and persists to database.
    """

    def __init__(self, config: dict):
        """
        Initialize the KrakenStreamManager.

        Args:
            config (dict): Configuration dictionary
            db_engine: SQLAlchemy database engine
            market_type (str): Market type - 'spot' or 'futures'
        """
        self.config = config
        self.market_type = config['collection'].get(
            'market_type', 'spot'
        ).lower()        
        # Map symbols based on market type
        raw_symbols = config['collection']['symbols']
        self.symbols = [self._map_symbol(s) for s in raw_symbols]
        
        self.streams = config['collection']['stream_types']
        self.buffer_size = config['settings']['buffer_size']
        
        # Set WebSocket URLs based on market type
        if self.market_type == 'futures':
            self.base_url = "wss://futures.kraken.com/ws/v1"
            self.level3_url = None
        else:  # spot
            self.base_url = "wss://ws.kraken.com/v2"
            self.level3_url = "wss://ws-l3.kraken.com/v2"
        
        self.depth = config['collection'].get('depth', 10)
        self.ohlc_interval = 1
        self.active_websockets: List[WebSocketApp] = []
        self.threads: List[threading.Thread] = []
        self.is_running = True

    def _map_symbol(self, symbol: str) -> str:
        """
        Map generic symbol to market-specific format.
        
        Args:
            symbol (str): Generic symbol (e.g., "BTC", "ETH")
            
        Returns:
            str: Market-specific symbol format
                - Spot: "BTC/USD", "ETH/USD"
                - Futures: "PI_XBTUSD", "PI_ETHUSD" (perpetuals only)
        """
        symbol = symbol.upper()
        
        if self.market_type == 'spot':
            # Spot format: BASE/QUOTE
            return f"{symbol}/USD"
        
        elif self.market_type == 'futures':
            # Futures perpetual format: PI_{BASE}USD
            # Special case: BTC -> XBT for Kraken futures
            if symbol == 'BTC':
                return 'PI_XBTUSD'
            else:
                return f'PI_{symbol}USD'
        
        return symbol

    def _get_schema_path(self) -> str:
        """Construct the database schema path."""
        db_conf = self.config['database']
        return f"{db_conf['base_schema']}_{db_conf['exchange']}"

    def _get_table_name(self, stream_type: str, symbol: str) -> str:
        """
        Generate database table name for a stream and symbol.
        
        Args:
            stream_type (str): Type of stream
            symbol (str): Mapped symbol (market-specific format)
            
        Returns:
            str: Table name with market type included
        """
        # Clean symbol for table name
        if self.market_type == 'futures':
            clean_symbol = symbol.replace('PI_', '').replace('PF_', '').replace('USD', '').lower()
        else:
            clean_symbol = symbol.replace('/USD', '').lower()
        
        if stream_type == 'book':
            return f"book_{self.market_type}_{self.depth}_{clean_symbol}"
        elif stream_type == 'level3' and self.market_type == 'spot':
            return f"level3_{self.market_type}_{clean_symbol}"
        elif stream_type == 'ohlc' and self.market_type == 'spot':
            return f"ohlc_{self.market_type}_{self.ohlc_interval}m_{clean_symbol}"
        else:
            return f"{stream_type}_{self.market_type}_{clean_symbol}"
    
    def iso_to_unix_ms(self, ts: str) -> int:
        """Convert ISO-8601 UTC timestamp to Unix milliseconds."""
        return int(
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
            .timestamp() * 1000
        )

    def _get_parser(self, stream_type: str) -> Callable:
        """
        Get parser based on market type and stream.
        
        Args:
            stream_type (str): Type of stream to parse
            
        Returns:
            Callable: Parser function for the stream type
        """
        if self.market_type == 'spot':
            parsers = {
                'ticker': self._parse_spot_ticker,
                'book': self._parse_spot_book,
                'level3': self._parse_spot_level3,
                'trade': self._parse_spot_trade,
                'ohlc': self._parse_spot_ohlc
            }
        else:  # futures
            parsers = {
                'ticker': self._parse_futures_ticker,
                'ticker_lite': self._parse_futures_ticker_lite,
                'book': self._parse_futures_book,
                'trade': self._parse_futures_trade
            }
        
        return parsers.get(stream_type, lambda x: x)

    # ========================================================================
    # SPOT PARSERS (V2 API)
    # ========================================================================

    def _parse_spot_ticker(self, data: dict) -> dict:
        """Parse spot ticker (Level 1) stream data."""
        ticker_data = data.get('data', [{}])[0]
        
        return {
            'timestamp': self.iso_to_unix_ms(ticker_data.get('timestamp', 0)),
            'ask': float(ticker_data.get('ask', 0)),
            'ask_qty': float(ticker_data.get('ask_qty', 0)),
            'bid': float(ticker_data.get('bid', 0)),
            'bid_qty': float(ticker_data.get('bid_qty', 0)),
            'change': float(ticker_data.get('change', 0)),
            'change_pct': float(ticker_data.get('change_pct', 0)),
            'high': float(ticker_data.get('high', 0)),
            'last': float(ticker_data.get('last', 0)),
            'low': float(ticker_data.get('low', 0)),
            'volume': float(ticker_data.get('volume', 0)),
            'vwap': float(ticker_data.get('vwap', 0)),
        }

    def _parse_spot_book(self, data: dict) -> dict:
        """Parse spot book (Level 2) orderbook stream data."""
        book_data = data.get('data', [{}])[0]
        bids = book_data.get('bids', [])
        asks = book_data.get('asks', [])

        return {
            'timestamp': self.iso_to_unix_ms(book_data.get('timestamp', 0)),
            'update_type': data.get('type', 'update'),
            'bids': bids,
            'asks': asks,
            'best_bid_price': float(bids[0]['price']) if bids else None,
            'best_bid_qty': float(bids[0]['qty']) if bids else None,
            'best_ask_price': float(asks[0]['price']) if asks else None,
            'best_ask_qty': float(asks[0]['qty']) if asks else None,
        }

    def _parse_spot_level3(self, data: dict) -> dict:
        """Parse spot level3 (individual orders) orderbook stream data."""
        book_data = data.get('data', [{}])[0]
        bids = book_data.get('bids', [])
        asks = book_data.get('asks', [])

        return {
            'timestamp': self.iso_to_unix_ms(book_data.get('timestamp', 0)),
            'update_type': data.get('type', 'update'),
            'bids': bids,
            'asks': asks,
            'best_bid_price': float(bids[0]['price']) if bids else None,
            'best_bid_qty': float(bids[0]['qty']) if bids else None,
            'best_ask_price': float(asks[0]['price']) if asks else None,
            'best_ask_qty': float(asks[0]['qty']) if asks else None,
            'checksum': book_data.get('checksum'),
        }

    def _parse_spot_trade(self, data: dict) -> List[dict]:
        """Parse spot trade stream data. Returns list of trades."""
        trades_list = []
        trades_data = data.get('data', [])
        
        for trade in trades_data:
            parsed_trade = {
                'timestamp': self.iso_to_unix_ms(trade.get('timestamp', 0)),
                'side': trade.get('side'),
                'price': float(trade.get('price', 0)),
                'qty': float(trade.get('qty', 0)),
                'ord_type': trade.get('ord_type'),
            }
            trades_list.append(parsed_trade)
        
        return trades_list

    def _parse_spot_ohlc(self, data: dict) -> dict:
        """Parse spot OHLC (candles) stream data."""
        ohlc_data = data.get('data', [{}])[0]
        
        return {
            'timestamp': self.iso_to_unix_ms(ohlc_data.get('timestamp', 0)),
            'open': float(ohlc_data.get('open', 0)),
            'high': float(ohlc_data.get('high', 0)),
            'low': float(ohlc_data.get('low', 0)),
            'close': float(ohlc_data.get('close', 0)),
            'trades': ohlc_data.get('trades', 0),
            'volume': float(ohlc_data.get('volume', 0)),
            'vwap': float(ohlc_data.get('vwap', 0)),
        }

    # ========================================================================
    # FUTURES PARSERS (V1 API)
    # ========================================================================

    def _parse_futures_ticker_lite(self, data: dict) -> dict:
        """Parse futures ticker_lite stream data (V1 format)."""
        return {
            'timestamp': int(time.time() * 1000),
            'bid': float(data.get('bid', 0)),
            'ask': float(data.get('ask', 0)),
            'change': float(data.get('change', 0)),
            'premium': float(data.get('premium', 0)),
            'volume': float(data.get('volume', 0)),
        }

    def _parse_futures_ticker(self, data: dict) -> dict:
        """Parse futures ticker stream data (full ticker)."""
        return {
            'timestamp': int(time.time() * 1000),
            'bid': float(data.get('bid', 0)),
            'ask': float(data.get('ask', 0)),
            'bid_size': float(data.get('bid_size', 0)),
            'ask_size': float(data.get('ask_size', 0)),
            'volume': float(data.get('volume', 0)),
            'mark_price': float(data.get('markPrice', 0)),
            'index_price': float(data.get('index', 0)),
            'last': float(data.get('last', 0)),
            'change': float(data.get('change', 0)),
            'funding_rate': float(data.get('funding_rate', 0)),
            'funding_rate_prediction': float(data.get('funding_rate_prediction', 0)),
            'open_interest': float(data.get('openInterest', 0)),
            'tag': data.get('tag'),
        }

    def _parse_futures_book(self, data: dict) -> dict:
        """Parse futures book stream data (incremental updates)."""
        side = data.get('side')
        price = float(data.get('price', 0))
        qty = float(data.get('qty', 0))
        
        return {
            'timestamp': data.get('timestamp', int(time.time() * 1000)),
            'seq': data.get('seq'),
            'side': side,
            'price': price,
            'qty': qty,
        }

    def _parse_futures_trade(self, data: dict) -> List[dict]:
        """
        Parse futures trade stream data.
        
        Handles both:
        - trade_snapshot: Initial snapshot with array of trades
        - trade: Individual trade updates
        
        Returns:
            List[dict]: List of parsed trades (may contain 1 or many)
        """
        trades_list = []
        
        # Handle trade snapshot (initial subscription response)
        if data.get('feed') == 'trade_snapshot':
            for trade in data.get('trades', []):
                parsed_trade = {
                    'timestamp': trade.get('time', int(time.time() * 1000)),
                    'side': trade.get('side'),
                    'type': trade.get('type'),
                    'price': float(trade.get('price', 0)),
                    'qty': float(trade.get('qty', 0)),
                    'seq': trade.get('seq'),
                }
                trades_list.append(parsed_trade)
        
        # Handle individual trade updates
        elif data.get('feed') == 'trade':
            parsed_trade = {
                'timestamp': data.get('time', int(time.time() * 1000)),
                'side': data.get('side'),
                'type': data.get('type'),
                'price': float(data.get('price', 0)),
                'qty': float(data.get('qty', 0)),
                'seq': data.get('seq'),
            }
            trades_list.append(parsed_trade)
        
        return trades_list

    # ========================================================================
    # WEBSOCKET STREAM LOGIC
    # ========================================================================

    def _run_stream(self, symbol: str, stream_type: str):
        """Run a single WebSocket stream for a symbol and stream type."""
        buffer = []
        parser = self._get_parser(stream_type)
        table = self._get_table_name(stream_type, symbol)
        schema = self._get_schema_path()
        
        # Select appropriate WebSocket URL
        if self.market_type == 'futures':
            ws_url = self.base_url
        elif stream_type == 'level3':
            ws_url = self.level3_url
        else:
            ws_url = self.base_url

        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Handle message based on market type
                if self.market_type == 'spot':
                    # V2 Spot handling
                    if 'method' in data and data.get('method') == 'subscribe':
                        if data.get('success'):
                            logger.info(f"✓ Subscribed to {table}")
                        else:
                            logger.error(f"✗ Subscription failed for {table}: {data.get('error')}")
                        return
                    
                    # Handle heartbeat and status
                    if data.get('channel') in ['heartbeat', 'status']:
                        return
                    
                else:  # futures
                    # V1 Futures handling
                    if data.get('event') == 'subscribed':
                        logger.info(f"✓ Subscribed to {table}: {data.get('feed')}")
                        return
                    
                    if data.get('event') == 'info':
                        logger.info(f"Info for {table}: {data.get('message')}")
                        return
                    
                    if data.get('event') == 'error':
                        logger.error(f"Error for {table}: {data.get('message')}")
                        return
                    
                    # Handle heartbeat
                    if data.get('feed') == 'heartbeat':
                        return
                
                # Parse data
                parsed_data = parser(data)
                if isinstance(parsed_data, list):
                    buffer.extend(parsed_data)
                elif parsed_data:
                    buffer.append(parsed_data)

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
                            logger.debug(f"Saved {len(buffer)} records to {table}")
                            break
                        except Exception as db_err:
                            logger.warning(f"Retry {i+1}/{retries} for {table}: {db_err}")
                            time.sleep(2 ** i)

                    if success:
                        buffer.clear()
                    else:
                        logger.error(f"CRITICAL: Failed to save {table} after {retries} retries")
                        
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {table}: {e}")
            except Exception as e:
                logger.error(f"Error in {table}: {e}", exc_info=True)

        def on_open(ws):
            try:
                if self.market_type == 'spot':
                    # V2 Spot subscription format
                    params = {
                        "channel": stream_type,
                        "symbol": [symbol]
                    }
                    
                    if stream_type == 'book':
                        params["depth"] = self.depth
                        params["snapshot"] = True
                    elif stream_type == 'ohlc':
                        params["interval"] = self.ohlc_interval
                        params["snapshot"] = True
                    elif stream_type in ['trade', 'ticker']:
                        params["snapshot"] = True
                    
                    subscribe_message = {
                        "method": "subscribe",
                        "params": params
                    }
                else:  # futures
                    # V1 Futures subscription format
                    feed_map = {
                        'ticker': 'ticker',
                        'ticker_lite': 'ticker_lite',
                        'book': 'book',
                        'trade': 'trade'
                    }
                    
                    subscribe_message = {
                        "event": "subscribe",
                        "feed": feed_map.get(stream_type, stream_type),
                        "product_ids": [symbol]
                    }
                
                ws.send(json.dumps(subscribe_message))
                logger.info(f"✓ Connected: {table} ({self.market_type.upper()})")
            except Exception as e:
                logger.error(f"Error subscribing to {table}: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error in {table}: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"Connection closed for {table}: {close_status_code}")

        # Reconnection settings
        retry_delay = 5
        max_retry_delay = 60
        backoff_factor = 2

        while self.is_running:
            try:
                ws = WebSocketApp(
                    ws_url,
                    on_message=on_message,
                    on_open=on_open,
                    on_error=on_error,
                    on_close=on_close
                )
                self.active_websockets.append(ws)
                
                # Increased ping interval and timeout for better stability
                # Using 30/15 or 60/30 is safer for busy feeds
                ws.run_forever(ping_interval=30, ping_timeout=15)
                
                # If we get here, connection closed. Remove from active list.
                if ws in self.active_websockets:
                    self.active_websockets.remove(ws)
                
            except Exception as conn_err:
                logger.error(f"Error in connection loop for {table}: {conn_err}")
            
            if self.is_running:
                # Exponential backoff with jitter to avoid thundering herd (Cloudflare 503s)
                import random
                jitter = random.uniform(0.5, 1.5)
                current_sleep = min(retry_delay * jitter, max_retry_delay)
                
                logger.warning(f"Reconnecting {table} in {current_sleep:.1f}s...")
                time.sleep(current_sleep)
                
                # Increase delay for next time if we failed quickly
                retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
            else:
                break

    def start_all(self):
        """Start all configured WebSocket streams."""
        base_delay = self.config['settings'].get('connection_delay', 1.0)

        for symbol in self.symbols:
            for stream in self.streams:
                # Add jitter to connection timing to avoid 503/Cloudflare blocks
                import random
                jitter = random.uniform(0.1, 1.5)
                delay = base_delay * jitter
                
                t = threading.Thread(
                    target=self._run_stream,
                    args=(symbol, stream),
                    name=f"{self.market_type}-{symbol}-{stream}",
                    daemon=True
                )
                self.threads.append(t)
                t.start()
                
                logger.debug(f"Scheduled stream {symbol}-{stream} (delay: {delay:.2f}s)")
                time.sleep(delay)

        logger.info(
            f"Started {len(self.threads)} {self.market_type.upper()} streams for Kraken"
        )

    def stop_all(self):
        """Gracefully shut down all WebSocket connections."""
        if not self.is_running:
            return
            
        self.is_running = False
        logger.info(f"Shutting down {len(self.active_websockets)} {self.market_type.upper()} streams...")

        for ws in self.active_websockets:
            try:
                ws.close()
            except:
                pass

        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2.0)

        logger.info(f"Kraken {self.market_type.upper()} shutdown complete.")