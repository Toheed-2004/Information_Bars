"""Utility helpers for meta module.

This module provides all business logic for meta.symbols and meta.data_tick 
management including:
- Configuration parsing and validation
- Database schema and table management
- Data transformation and loading
- CRUD operations for symbols and data_tick

Module Organization:
    1. Data Transformation Functions (build_ohlcv_records, build_tick_records)
    2. Validation Helpers (validate_timeframes)
    3. Serialization Helpers (serialize_dict)
    4. Database Helper Functions (symbol_exists, insert_symbol, etc.)
    5. Tick Meta Query Functions (get_data_tick_config, get_enabled_exchanges)
"""

import json
from typing import Any, Dict, List, Optional

from bitpredict.common.logging import get_logger
from bitpredict.common.constants import *

logger = get_logger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Valid time horizon suffixes for bar notation
# m=minutes, h=hours, d=days, w=weeks, M=months
VALID_HORIZONS = {"m", "h", "d", "w", "M"}

# ---------------------------------------------------------------------------
# TRANSFORMATION LOGIC - SYMBOLS MASTER
# ---------------------------------------------------------------------------


def build_symbols_records(symbols_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build flat database records from parsed symbols.yaml configuration.

    Iterates over all exchanges and their symbols defined in the YAML file,
    validates the configuration structure, and transforms nested dictionaries
    into flat row-level records suitable for database insertion into the
    symbols_master table.

    Each (exchange, symbol) pair produces exactly one record. Disabled symbols
    are included in the output but marked with enabled=False.

    This function builds records for the MASTER symbols table. Other tables
    (ohlcv, bars) will reference these symbols via foreign keys.

    Args:
        symbols_config: Parsed symbols.yaml configuration containing exchanges
            and symbols. Expected structure:
            {
                'exchanges': {
                    'exchange_name': {
                        'symbols': {
                            'symbol_name': {
                                'enabled': bool
                            }
                        }
                    }
                }
            }

    Returns:
        List of dictionaries representing database rows. Each dictionary
        contains keys: exchange, symbol, enabled.

    Raises:
        ValueError: If any of the following conditions are met:
            - symbols_config is None or not a dictionary
            - 'exchanges' key is missing from config
            - Required 'enabled' flag is missing for any symbol
            - Exchange name or symbol name is empty/whitespace
            - Duplicate (exchange, symbol) pairs found

    Example:
        >>> symbols_config = {
        ...     'exchanges': {
        ...         'binance': {
        ...             'symbols': {
        ...                 'btc': {'enabled': True},
        ...                 'eth': {'enabled': True}
        ...             }
        ...         }
        ...     }
        ... }
        >>> records = build_symbols_records(symbols_config)
        >>> len(records)
        2
        >>> records[0]['symbol']
        'btc'

    Notes:
        - Logs warnings for exchanges with no symbols defined
        - Logs debug messages for disabled symbols
        - All validation errors include exchange:symbol context
        - Checks for duplicate (exchange, symbol) pairs
        - Validates exchange and symbol names are non-empty
    """
    # Edge case: Validate symbols_config is not None and is a dictionary
    if symbols_config is None:
        raise ValueError("symbols_config is None; expected a dictionary")

    if not isinstance(symbols_config, dict):
        raise ValueError(
            f"symbols_config must be a dictionary, got {type(symbols_config).__name__}"
        )

    records: List[Dict[str, Any]] = []
    seen_pairs = set()  # Track (exchange, symbol) pairs for duplicates

    # Extract exchanges configuration
    exchanges_cfg = symbols_config.get("exchanges", {})
    if not exchanges_cfg:
        raise ValueError("symbols.yaml contains no 'exchanges' definitions")

    # Edge case: Validate exchanges_cfg is a dictionary
    if not isinstance(exchanges_cfg, dict):
        raise ValueError(
            f"'exchanges' must be a dictionary, got {type(exchanges_cfg).__name__}"
        )

    logger.debug("Processing %d exchanges from symbols configuration", len(exchanges_cfg))

    # Iterate through each exchange
    for exchange_name, exchange_cfg in exchanges_cfg.items():
        # Edge case: Validate exchange name is non-empty string
        if not isinstance(exchange_name, str) or not exchange_name.strip():
            raise ValueError(
                f"Exchange name must be a non-empty string, got: '{exchange_name}'"
            )

        exchange_name = exchange_name.strip()

        # Edge case: Validate exchange_cfg is a dictionary
        if not isinstance(exchange_cfg, dict):
            logger.warning(
                "Exchange '%s' configuration is not a dictionary; skipping",
                exchange_name,
            )
            continue

        symbols = exchange_cfg.get("symbols", {})

        # Edge case: Warn if exchange has no symbols defined
        if not symbols:
            logger.warning(
                "No symbols defined for exchange '%s'; skipping", exchange_name
            )
            continue

        # Edge case: Validate symbols is a dictionary
        if not isinstance(symbols, dict):
            logger.warning(
                "Symbols for exchange '%s' is not a dictionary; skipping", exchange_name
            )
            continue

        logger.debug(
            "Processing %d symbols for exchange '%s'", len(symbols), exchange_name
        )

        # Process each symbol for this exchange
        for symbol_name, symbol_cfg in symbols.items():
            # Edge case: Validate symbol name is non-empty string
            if not isinstance(symbol_name, str) or not symbol_name.strip():
                logger.error(
                    "Symbol name must be a non-empty string for exchange '%s', "
                    "got: '%s'; skipping",
                    exchange_name,
                    symbol_name,
                )
                continue

            symbol_name = symbol_name.strip()

            # Edge case: Check for duplicate (exchange, symbol) pairs
            pair_key = (exchange_name, symbol_name)
            if pair_key in seen_pairs:
                raise ValueError(
                    f"Duplicate symbol found: {exchange_name}:{symbol_name}. "
                    "Each (exchange, symbol) pair must be unique."
                )
            seen_pairs.add(pair_key)

            logger.debug(
                "Processing symbol: exchange=%s, symbol=%s",
                exchange_name,
                symbol_name,
            )

            # Edge case: Validate symbol_cfg is a dictionary
            if not isinstance(symbol_cfg, dict):
                logger.error(
                    "Symbol configuration for %s:%s is not a dictionary; skipping",
                    exchange_name,
                    symbol_name,
                )
                continue

            # Validate required 'enabled' field
            if "enabled" not in symbol_cfg:
                raise ValueError(
                    f"Missing 'enabled' flag for symbol {exchange_name}:{symbol_name}"
                )

            # Edge case: Handle non-boolean enabled values
            enabled_value = symbol_cfg["enabled"]
            if not isinstance(enabled_value, bool):
                logger.warning(
                    "Non-boolean 'enabled' value for %s:%s (got %s); "
                    "converting to boolean",
                    exchange_name,
                    symbol_name,
                    type(enabled_value).__name__,
                )
            enabled = bool(enabled_value)

            # Log disabled symbols for visibility
            if not enabled:
                logger.debug(
                    "Symbol %s:%s is disabled; will be inserted but marked disabled",
                    exchange_name,
                    symbol_name,
                )

            # Build record
            record = {
                "exchange": exchange_name,
                "symbol": symbol_name,
                "enabled": enabled,
            }

            records.append(record)

    # Edge case: Log warning if no records were generated
    if not records:
        logger.warning(
            "No valid records generated from symbols configuration. "
            "Check for empty exchanges or invalid symbol definitions."
        )

    logger.info(
        "Built %d meta.symbols records from configuration",
        len(records),
    )
    return records


# ---------------------------------------------------------------------------
# TRANSFORMATION LOGIC - OHLCV (REFACTORED)
# ---------------------------------------------------------------------------


def build_time_bars_records(ohlcv_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build flat database records from parsed ohlcv.yaml configuration.

    REFACTORED: This function now builds records with (exchange, symbol, timeframes)
    tuples. The insert function will handle the symbol_id lookup from symbols_master.

    Each (exchange, symbol) pair produces exactly one record containing
    timeframes configuration.

    Args:
        ohlcv_config: Parsed ohlcv.yaml configuration containing exchanges,
            symbols, and their timeframes. Expected structure:
            {
                'exchanges': {
                    'exchange_name': {
                        'symbols': {
                            'symbol_name': {
                                'timeframes': dict
                            }
                        }
                    }
                }
            }

    Returns:
        List of dictionaries representing database rows. Each dictionary
        contains keys: exchange, symbol, timeframes.
        The timeframes values are serialized to JSON format.

    Raises:
        ValueError: If any of the following conditions are met:
            - ohlcv_config is None or not a dictionary
            - 'exchanges' key is missing from config
            - Required 'timeframes' field is missing or empty
            - Exchange name or symbol name is empty/whitespace
            - Duplicate (exchange, symbol) pairs found
            - timeframes validation fails

    Example:
        >>> ohlcv_config = {
        ...     'exchanges': {
        ...         'binance': {
        ...             'symbols': {
        ...                 'btc': {
        ...                     'timeframes': {'1m': True, '1h': True}
        ...                 }
        ...             }
        ...         }
        ...     }
        ... }
        >>> records = build_ohlcv_records(ohlcv_config)
        >>> len(records)
        1
        >>> records[0].keys()
        dict_keys(['exchange', 'symbol', 'timeframes'])

    Notes:
        - NO database lookups in this function
        - The insert function will resolve symbol_id from symbols_master
        - Logs warnings for exchanges with no symbols defined
        - Calls validate_timeframes() for additional validation
        - Calls serialize_dict() to convert dictionaries to JSON format
        - All validation errors include exchange:symbol context
    """
    # Edge case: Validate ohlcv_config is not None and is a dictionary
    if ohlcv_config is None:
        raise ValueError("ohlcv_config is None; expected a dictionary")

    if not isinstance(ohlcv_config, dict):
        raise ValueError(
            f"ohlcv_config must be a dictionary, got {type(ohlcv_config).__name__}"
        )

    records: List[Dict[str, Any]] = []
    seen_pairs = set()  # Track (exchange, symbol) pairs for duplicates

    # Extract exchanges configuration
    exchanges_cfg = ohlcv_config.get("exchanges", {})
    if not exchanges_cfg:
        raise ValueError("time_bars.yaml contains no 'exchanges' definitions")

    # Edge case: Validate exchanges_cfg is a dictionary
    if not isinstance(exchanges_cfg, dict):
        raise ValueError(
            f"'exchanges' must be a dictionary, got {type(exchanges_cfg).__name__}"
        )

    logger.debug("Processing %d exchanges from time_bars configuration", len(exchanges_cfg))

    # Iterate through each exchange
    for exchange_name, exchange_cfg in exchanges_cfg.items():
        # Edge case: Validate exchange name is non-empty string
        if not isinstance(exchange_name, str) or not exchange_name.strip():
            raise ValueError(
                f"Exchange name must be a non-empty string, got: '{exchange_name}'"
            )

        exchange_name = exchange_name.strip()

        # Edge case: Validate exchange_cfg is a dictionary
        if not isinstance(exchange_cfg, dict):
            logger.warning(
                "Exchange '%s' configuration is not a dictionary; skipping",
                exchange_name,
            )
            continue

        symbols = exchange_cfg.get("symbols", {})

        # Edge case: Warn if exchange has no symbols defined
        if not symbols:
            logger.warning(
                "No symbols defined for exchange '%s'; skipping", exchange_name
            )
            continue

        # Edge case: Validate symbols is a dictionary
        if not isinstance(symbols, dict):
            logger.warning(
                "Symbols for exchange '%s' is not a dictionary; skipping", exchange_name
            )
            continue

        logger.debug(
            "Processing %d symbols for exchange '%s'", len(symbols), exchange_name
        )

        # Process each symbol for this exchange
        for symbol_name, symbol_cfg in symbols.items():
            # Edge case: Validate symbol name is non-empty string
            if not isinstance(symbol_name, str) or not symbol_name.strip():
                logger.error(
                    "Symbol name must be a non-empty string for exchange '%s', "
                    "got: '%s'; skipping",
                    exchange_name,
                    symbol_name,
                )
                continue

            symbol_name = symbol_name.strip()

            # Edge case: Check for duplicate (exchange, symbol) pairs
            pair_key = (exchange_name, symbol_name)
            if pair_key in seen_pairs:
                raise ValueError(
                    f"Duplicate symbol found: {exchange_name}:{symbol_name}. "
                    "Each (exchange, symbol) pair must be unique."
                )
            seen_pairs.add(pair_key)

            logger.debug(
                "Processing OHLCV config: exchange=%s, symbol=%s",
                exchange_name,
                symbol_name,
            )

            # Edge case: Validate symbol_cfg is a dictionary
            if not isinstance(symbol_cfg, dict):
                logger.error(
                    "Symbol configuration for %s:%s is not a dictionary; skipping",
                    exchange_name,
                    symbol_name,
                )
                continue

            # Extract and validate timeframes
            timeframes = symbol_cfg.get("timeframes")
            if not timeframes:
                raise ValueError(
                    f"Missing 'timeframes' for symbol "
                    f"{exchange_name}:{symbol_name}"
                )

            # Delegate validation to utility function
            validate_timeframes(timeframes, exchange_name, symbol_name)

            # Edge case: Catch serialization errors for timeframes
            try:
                timeframes_json = serialize_dict(timeframes)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Failed to serialize 'timeframes' for "
                    f"{exchange_name}:{symbol_name}: {exc}"
                ) from exc

            record = {
                "exchange": exchange_name,
                "symbol": symbol_name,
                "timeframes": timeframes_json,
            }

            records.append(record)

    # Edge case: Log warning if no records were generated
    if not records:
        logger.warning(
            "No valid records generated from ohlcv configuration. "
            "Check for empty exchanges or invalid symbol definitions."
        )

    logger.info(
        "Built %d meta.ohlcv records from configuration",
        len(records),
    )
    return records


# ---------------------------------------------------------------------------
# TRANSFORMATION LOGIC - BARS (REFACTORED)
# ---------------------------------------------------------------------------


def build_custom_bars_records(bars_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build flat database records from parsed bars configuration.

    REFACTORED: This function now builds records with (exchange, symbol, bars)
    tuples. The insert function will handle the symbol_id lookup from symbols_master.

    Each (exchange, symbol) pair produces exactly one record containing
    bars configuration.

    Args:
        bars_config: Parsed bars configuration containing exchanges,
            symbols, and bars settings. Expected structure:
            {
                'exchanges': {
                    'exchange_name': {
                        'symbols': {
                            'symbol_name': {
                                'bars': {
                                    'bar_type': bool,
                                }
                            }
                        }
                    }
                }
            }

    Returns:
        List of dictionaries representing database rows. Each dictionary
        contains keys: exchange, symbol, bars.

    Raises:
        ValueError: If configuration structure is invalid or missing
            required fields.
        TypeError: If field values are not JSON-serializable.

    Example:
        >>> bars_config = {
        ...     'exchanges': {
        ...         'bybit': {
        ...             'symbols': {
        ...                 'btc': {
        ...                     'bars': {'dollar': True, 'volume': True}
        ...                 }
        ...             }
        ...         }
        ...     }
        ... }
        >>> records = build_custom_bars_records(bars_config)
        >>> len(records)
        1
        >>> records[0].keys()
        dict_keys(['exchange', 'symbol', 'bars'])

    Notes:
        - NO database lookups in this function
        - The insert function will resolve symbol_id from symbols_master
        - Logs warnings for exchanges with no symbols defined
        - Validates all required fields are present
        - Serializes bars to JSON format
    """
    # Validate bars_config is not None and is a dictionary
    if bars_config is None:
        raise ValueError("custom_bars_config is None; expected a dictionary")

    if not isinstance(bars_config, dict):
        raise ValueError(
            f"custom_bars_config must be a dictionary, got {type(bars_config).__name__}"
        )

    records: List[Dict[str, Any]] = []
    seen_pairs = set()  # Track (exchange, symbol) pairs for duplicates

    # Extract exchanges configuration
    exchanges_cfg = bars_config.get("exchanges", {})
    if not exchanges_cfg:
        raise ValueError("custom_bars_config contains no 'exchanges' definitions")

    if not isinstance(exchanges_cfg, dict):
        raise ValueError(
            f"'exchanges' must be a dictionary, got {type(exchanges_cfg).__name__}"
        )

    logger.debug("Processing %d exchanges from bars configuration", len(exchanges_cfg))

    # Iterate through each exchange
    for exchange_name, exchange_cfg in exchanges_cfg.items():
        # Validate exchange name
        if not isinstance(exchange_name, str) or not exchange_name.strip():
            raise ValueError(
                f"Exchange name must be a non-empty string, got: '{exchange_name}'"
            )

        exchange_name = exchange_name.strip()

        if not isinstance(exchange_cfg, dict):
            logger.warning(
                "Exchange '%s' configuration is not a dictionary; skipping",
                exchange_name,
            )
            continue

        symbols = exchange_cfg.get("symbols", {})

        if not symbols:
            logger.warning(
                "No symbols defined for custom_bars_config exchange '%s'; skipping", exchange_name
            )
            continue

        if not isinstance(symbols, dict):
            logger.warning(
                "Symbols for exchange '%s' is not a dictionary; skipping", exchange_name
            )
            continue

        logger.debug(
            "Processing %d symbols for exchange '%s'", len(symbols), exchange_name
        )

        # Process each symbol for this exchange
        for symbol_name, symbol_cfg in symbols.items():
            # Validate symbol_name
            if not isinstance(symbol_name, str) or not symbol_name.strip():
                logger.error(
                    "Symbol name must be a non-empty string for exchange '%s', "
                    "got: '%s'; skipping",
                    exchange_name,
                    symbol_name,
                )
                continue

            symbol_name = symbol_name.strip()

            # Check for duplicate (exchange, symbol) pairs
            pair_key = (exchange_name, symbol_name)
            if pair_key in seen_pairs:
                raise ValueError(
                    f"Duplicate symbol found: {exchange_name}:{symbol_name}. "
                    "Each (exchange, symbol) pair must be unique."
                )
            seen_pairs.add(pair_key)

            logger.debug(
                "Processing bars configuration: exchange=%s, symbol=%s",
                exchange_name,
                symbol_name,
            )

            # Validate symbol_cfg is a dictionary
            if not isinstance(symbol_cfg, dict):
                logger.error(
                    "Symbol configuration for %s:%s is not a dictionary; skipping",
                    exchange_name,
                    symbol_name,
                )
                continue

            # Extract and validate bars (dict)
            bars = symbol_cfg.get("bars")
            if not bars:
                raise ValueError(
                    f"Missing 'bars' for symbol {exchange_name}:{symbol_name}"
                )

            if not isinstance(bars, dict):
                raise ValueError(
                    f"'bars' must be a dictionary for symbol {exchange_name}:{symbol_name}"
                )

            # Validate bars dictionary is not empty
            if not bars:
                raise ValueError(
                    f"'bars' dictionary is empty for symbol {exchange_name}:{symbol_name}"
                )

            # Validate each bar type has a boolean value
            for bar_type, bar_enabled in bars.items():
                if not isinstance(bar_enabled, bool):
                    raise ValueError(
                        f"Bar '{bar_type}' must have boolean value (True/False) "
                        f"for symbol {exchange_name}:{symbol_name}"
                    )

            # Build record with serialized JSON fields
            try:
                bars_json = serialize_dict(bars)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Failed to serialize 'bars' for {exchange_name}:{symbol_name}: {exc}"
                ) from exc

            record = {
                "exchange": exchange_name,
                "symbol": symbol_name,
                "bars": bars_json,
            }

            records.append(record)

    # Log warning if no records were generated
    if not records:
        logger.warning(
            "No valid records generated from bars configuration. "
            "Check for empty exchanges or invalid symbol definitions."
        )

    logger.info(
        "Built %d meta.bars records from configuration",
        len(records),
    )
    return records


# ---------------------------------------------------------------------------
# TRANSFORMATION LOGIC - TICK META
# ---------------------------------------------------------------------------


def build_tick_records(data_tick: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build flat database records from parsed data_tick.yaml configuration.

    Iterates over all exchanges and their market configurations, validates
    the structure, and transforms nested dictionaries into flat row-level
    records suitable for database insertion.

    Each (exchange, market_type) pair produces exactly one record. Disabled
    markets are included in the output but marked with enabled=False.

    Args:
        data_tick: Parsed data_tick.yaml configuration containing exchanges,
            markets, symbols, and stream settings. Expected structure:
            {
                'exchanges': {
                    'exchange_name': {
                        'markets': {
                            'market_type': {
                                'enabled': bool,
                                'symbols': list,
                                'streams': dict,
                                'depth': int,
                                'buffer_size': int,
                                'connection_delay': float
                            }
                        }
                    }
                }
            }

    Returns:
        List of dictionaries representing database rows. Each dictionary
        contains keys: exchange, market_type, enabled, symbols, streams,
        depth, buffer_size, connection_delay.

    Raises:
        ValueError: If configuration structure is invalid or missing
            required fields.
        TypeError: If field values are not JSON-serializable.

    Example:
        >>> data_tick = {
        ...     'exchanges': {
        ...         'bybit': {
        ...             'markets': {
        ...                 'linear': {
        ...                     'enabled': True,
        ...                     'symbols': ['BTCUSDT', 'ETHUSDT'],
        ...                     'streams': {'orderbook': True, 'trade': True},
        ...                     'depth': 50,
        ...                     'buffer_size': 1,
        ...                     'connection_delay': 1.0
        ...                 }
        ...             }
        ...         }
        ...     }
        ... }
        >>> records = build_tick_records(data_tick)
        >>> len(records)
        1

    Notes:
        - Logs warnings for exchanges with no markets defined
        - Validates all required fields are present
        - Serializes symbols and streams to JSON format
    """
    # Validate data_tick is not None and is a dictionary
    if data_tick is None:
        raise ValueError("data_tick configuration is None; expected a dictionary")

    if not isinstance(data_tick, dict):
        raise ValueError(
            f"data_tick must be a dictionary, got {type(data_tick).__name__}"
        )

    records: List[Dict[str, Any]] = []
    seen_pairs = set()  # Track (exchange, market_type) pairs for duplicates

    # Extract exchanges configuration
    exchanges_cfg = data_tick.get("exchanges", {})
    if not exchanges_cfg:
        raise ValueError("data_tick.yaml contains no 'exchanges' definitions")

    if not isinstance(exchanges_cfg, dict):
        raise ValueError(
            f"'exchanges' must be a dictionary, got {type(exchanges_cfg).__name__}"
        )

    logger.debug("Processing %d exchanges from data_tick configuration", len(exchanges_cfg))

    # Iterate through each exchange
    for exchange_name, exchange_cfg in exchanges_cfg.items():
        # Validate exchange name
        if not isinstance(exchange_name, str) or not exchange_name.strip():
            raise ValueError(
                f"Exchange name must be a non-empty string, got: '{exchange_name}'"
            )

        exchange_name = exchange_name.strip()

        if not isinstance(exchange_cfg, dict):
            logger.warning(
                "Exchange '%s' configuration is not a dictionary; skipping",
                exchange_name,
            )
            continue

        markets = exchange_cfg.get("markets", {})

        if not markets:
            logger.warning(
                "No markets defined for exchange '%s'; skipping", exchange_name
            )
            continue

        if not isinstance(markets, dict):
            logger.warning(
                "Markets for exchange '%s' is not a dictionary; skipping", exchange_name
            )
            continue

        logger.debug(
            "Processing %d markets for exchange '%s'", len(markets), exchange_name
        )

        # Process each market for this exchange
        for market_type, market_cfg in markets.items():
            # Validate market_type
            if not isinstance(market_type, str) or not market_type.strip():
                logger.error(
                    "Market type must be a non-empty string for exchange '%s', "
                    "got: '%s'; skipping",
                    exchange_name,
                    market_type,
                )
                continue

            market_type = market_type.strip()

            # Check for duplicate (exchange, market_type) pairs
            pair_key = (exchange_name, market_type)
            if pair_key in seen_pairs:
                raise ValueError(
                    f"Duplicate market found: {exchange_name}:{market_type}. "
                    "Each (exchange, market_type) pair must be unique."
                )
            seen_pairs.add(pair_key)

            logger.debug(
                "Processing market metadata: exchange=%s, market_type=%s",
                exchange_name,
                market_type,
            )

            # Validate market_cfg is a dictionary
            if not isinstance(market_cfg, dict):
                logger.error(
                    "Market configuration for %s:%s is not a dictionary; skipping",
                    exchange_name,
                    market_type,
                )
                continue

            # Validate required 'enabled' field
            if "enabled" not in market_cfg:
                raise ValueError(
                    f"Missing 'enabled' flag for market {exchange_name}:{market_type}"
                )

            enabled = bool(market_cfg["enabled"])

            if not enabled:
                logger.debug(
                    "Market %s:%s is disabled; will be inserted but marked disabled",
                    exchange_name,
                    market_type,
                )

            # Extract and validate symbols (list)
            symbols = market_cfg.get("symbols")
            if not symbols:
                raise ValueError(
                    f"Missing 'symbols' for market {exchange_name}:{market_type}"
                )

            if not isinstance(symbols, list):
                raise ValueError(
                    f"'symbols' must be a list for market {exchange_name}:{market_type}"
                )

            # Validate symbols list is not empty
            if not symbols:
                raise ValueError(
                    f"'symbols' list is empty for market {exchange_name}:{market_type}"
                )

            # Extract and validate streams (dict)
            streams = market_cfg.get("streams")
            if not streams:
                raise ValueError(
                    f"Missing 'streams' for market {exchange_name}:{market_type}"
                )

            if not isinstance(streams, dict):
                raise ValueError(
                    f"'streams' must be a dictionary for market {exchange_name}:{market_type}"
                )


            # Extract and validate depth
            depth = market_cfg.get("depth", 50)
            if not isinstance(depth, int) or depth <= 0:
                raise ValueError(
                    f"'depth' must be a positive integer for market {exchange_name}:{market_type}"
                )


            # Build record with serialized JSON fields
            try:
                symbols_json = serialize_dict({"symbols": symbols})
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Failed to serialize 'symbols' for {exchange_name}:{market_type}: {exc}"
                ) from exc

            try:
                streams_json = serialize_dict(streams)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Failed to serialize 'streams' for {exchange_name}:{market_type}: {exc}"
                ) from exc

            record = {
                "exchange": exchange_name,
                "market_type": market_type,
                "enabled": enabled,
                "symbols": symbols_json,
                "streams": streams_json,
                "depth": depth,
            }

            records.append(record)

    # Log warning if no records were generated
    if not records:
        logger.warning(
            "No valid records generated from data_tick configuration. "
            "Check for empty exchanges or invalid market definitions."
        )

    logger.info(
        "Built %d meta.data_tick records from configuration",
        len(records),
    )
    return records



# ---------------------------------------------------------------------------
# TRANSFORMATION LOGIC - BLOCKCHAIN
# ---------------------------------------------------------------------------

def build_blockchain_records(blockchain_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build flat database records from blockchain configuration.

    Transforms the blockchain.yaml configuration into database records
    for storing blockchain chart metadata.

    Args:
        blockchain_config: Parsed blockchain configuration containing:
            {
                'blockchain_charts': {
                    'category_name': ['chart1', 'chart2', ...]
                },
                'settings': {
                    'start_date': 'YYYY-MM-DD',
                    'end_date': 'now' or 'YYYY-MM-DD',
                    'schema_name': 'data_blockchain'
                }
            }

    Returns:
        List of dictionaries representing database rows. Each dictionary
        contains keys: chart_name, category, enabled, start_date, end_date, schema_name.

    Raises:
        ValueError: If configuration structure is invalid or missing required fields.

    Example:
        >>> blockchain_config = {
        ...     'blockchain_charts': {
        ...         'currency_stats': ['trade-volume', 'market-price'],
        ...         'mining_information': ['hash-rate', 'difficulty']
        ...     },
        ...     'settings': {
        ...         'start_date': '2020-01-01',
        ...         'end_date': 'now',
        ...         'schema_name': 'data_blockchain'
        ...     }
        ... }
        >>> records = build_blockchain_records(blockchain_config)
        >>> len(records)
        4

    Notes:
        - Logs warnings for empty categories
        - Validates all required fields are present
        - Stores all charts as enabled by default
    """
    if blockchain_config is None:
        raise ValueError("blockchain_config cannot be None")

    if not isinstance(blockchain_config, dict):
        raise ValueError("blockchain_config must be a dictionary")

    records: List[Dict[str, Any]] = []

    # Extract charts and settings
    charts_cfg = blockchain_config.get("blockchain_charts", {})
    settings_cfg = blockchain_config.get("settings", {})

    if not charts_cfg:
        raise ValueError("blockchain_charts configuration is missing or empty")

    if not settings_cfg:
        raise ValueError("settings configuration is missing")

    # Extract settings
    start_date = settings_cfg.get("start_date", "2020-01-01")
    end_date = settings_cfg.get("end_date", "now")

    logger.debug("Processing blockchain configuration with %d categories", len(charts_cfg))

    # Iterate through each category and its charts
    for category_name, charts_list in charts_cfg.items():
        if not category_name or not category_name.strip():
            logger.warning("Found empty category name, skipping")
            continue

        if not isinstance(charts_list, list):
            logger.warning("Category %s does not have a list of charts, skipping", category_name)
            continue

        if not charts_list:
            logger.warning("Category %s has no charts defined", category_name)
            continue

        # Create a record for each chart
        for chart_item in charts_list:
            # Handle both old format (string) and new format (dict with name and enabled)
            if isinstance(chart_item, str):
                chart_name = chart_item
                chart_enabled = True
            elif isinstance(chart_item, dict):
                chart_name = chart_item.get("name")
                chart_enabled = chart_item.get("enabled", True)
            else:
                logger.warning("Invalid chart item type in category %s, skipping", category_name)
                continue

            if not chart_name or not str(chart_name).strip():
                logger.warning("Found empty chart name in category %s, skipping", category_name)
                continue

            record = {
                "chart_name": str(chart_name).strip(),
                "category": category_name.strip(),
                "enabled": chart_enabled,
                "start_date": start_date,
                "end_date": end_date,
            }
            records.append(record)

    if not records:
        logger.warning("No valid blockchain chart records generated from configuration")

    logger.info(
        "Built %d blockchain chart records from configuration",
        len(records),
    )
    return records


# ---------------------------------------------------------------------------
# TRANSFORMATION LOGIC - MACRO
# ---------------------------------------------------------------------------

def build_macro_records(macro_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build flat database records from macro economic indicator configuration.

    Transforms the macro.yaml configuration into database records
    for storing macro economic indicator metadata.

    Args:
        macro_config: Parsed macro configuration containing:
            {
                'economic_indicators': {
                    'indicator_key': {
                        'name': 'FRED_series_id',
                        'frequency': 'daily|weekly|monthly|quarterly|annual'
                    },
                    ...
                }
            }

    Returns:
        List of dictionaries representing database rows. Each dictionary
        contains keys: indicator_key, fred_series, frequency, enabled.

    Raises:
        ValueError: If configuration structure is invalid or missing required fields.

    Example:
        >>> macro_config = {
        ...     'economic_indicators': {
        ...         'unemployment_rate': {
        ...             'name': 'UNRATE',
        ...             'frequency': 'monthly'
        ...         },
        ...         'gdp': {
        ...             'name': 'GDP',
        ...             'frequency': 'quarterly'
        ...         }
        ...     }
        ... }
        >>> records = build_macro_records(macro_config)
        >>> len(records)
        2

    Notes:
        - Logs warnings for invalid entries
        - Validates all required fields are present
        - Stores all indicators as enabled by default
    """
    if macro_config is None:
        raise ValueError("macro_config cannot be None")

    if not isinstance(macro_config, dict):
        raise ValueError("macro_config must be a dictionary")

    records: List[Dict[str, Any]] = []

    # Extract indicators configuration
    indicators_cfg = macro_config.get("economic_indicators", {})

    if not indicators_cfg:
        raise ValueError("economic_indicators configuration is missing or empty")

    logger.debug("Processing macro configuration with %d indicators", len(indicators_cfg))

    # Iterate through each indicator
    for indicator_key, indicator_config in indicators_cfg.items():
        if not indicator_key or not indicator_key.strip():
            logger.warning("Found empty indicator key, skipping")
            continue

        if not isinstance(indicator_config, dict):
            logger.warning("Indicator %s does not have a valid configuration, skipping", indicator_key)
            continue

        # Extract required fields - try both 'name' and 'fred_series' for compatibility
        fred_series = indicator_config.get("fred_series") or indicator_config.get("name")
        frequency = indicator_config.get("frequency")
        indicator_enabled = indicator_config.get("enabled", True)

        if not fred_series or not str(fred_series).strip():
            logger.warning("Indicator %s is missing 'fred_series' field, skipping", indicator_key)
            continue

        if not frequency or not str(frequency).strip():
            logger.warning("Indicator %s is missing 'frequency' field, skipping", indicator_key)
            continue

        record = {
            "indicator_key": indicator_key.strip(),
            "fred_series": str(fred_series).strip(),
            "frequency": str(frequency).strip(),
            "enabled": indicator_enabled,
        }
        records.append(record)

    if not records:
        logger.warning("No valid macro indicator records generated from configuration")

    logger.info(
        "Built %d macro indicator records from configuration",
        len(records),
    )
    return records


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def validate_timeframes(
    timeframes: Dict[str, Any],
    exchange: str,
    symbol: str,
) -> None:
    """Validate the structure and contents of a timeframes dictionary.

    Ensures that time horizons follow the expected format where keys are
    bar notation strings (e.g., "1m", "4h", "1d") and values are booleans
    indicating whether the timeframe is enabled.

    Expected Format:
        {
            "1m": true,   # 1-minute bars enabled
            "15m": true,  # 15-minute bars enabled
            "1h": true,   # 1-hour bars enabled
            "4h": false,  # 4-hour bars disabled
            "1d": true    # 1-day bars enabled
        }

    Validation Rules:
        - timeframes must be a non-empty dictionary
        - Keys must be strings representing concrete bars (e.g., 1m, 4h, 1d)
        - Keys must end with valid suffixes (m, h, d, w, M)
        - Keys must have at least one digit before the suffix
        - Values must be booleans (enabled/disabled flags)

    Args:
        timeframes: Time horizon configuration dictionary to validate.
            Keys are bar notation strings, values are boolean enable flags.
        exchange: Exchange name (used for error context in exceptions).
        symbol: Symbol name (used for error context in exceptions).

    Raises:
        ValueError: If any of the following conditions are met:
            - timeframes is not a dictionary
            - timeframes is empty
            - exchange or symbol are empty/whitespace strings
            - Any key is not a string
            - Any key is empty or only whitespace
            - Any key has an invalid suffix (not in m, h, d, w, M)
            - Any key has no numeric component before suffix
            - Any value is not a boolean

    Example:
        >>> horizons = {"1m": True, "1h": True, "1d": False}
        >>> validate_timeframes(horizons, "binance", "BTC")
        # Passes validation

        >>> invalid = {"1x": True}  # Invalid suffix
        >>> validate_timeframes(invalid, "binance", "BTC")
        # Raises ValueError: Invalid bar suffix '1x'

        >>> invalid = {"m": True}  # No numeric component
        >>> validate_timeframes(invalid, "binance", "BTC")
        # Raises ValueError: Bar 'm' has no numeric component

    Notes:
        - Logs validation activity at debug level
        - Error messages include exchange:symbol context for debugging
        - Validates numeric component exists but doesn't validate range
        - Focuses on structural validation, not semantic validation
        - Validates exchange and symbol names for completeness
    """
    # Edge case: Validate exchange and symbol names
    if not isinstance(exchange, str) or not exchange.strip():
        raise ValueError("exchange must be a non-empty string")

    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    # Ensure timeframes is a non-empty dictionary
    if not isinstance(timeframes, dict):
        raise ValueError(
            f"timeframes must be a dictionary for {exchange}:{symbol}, "
            f"got {type(timeframes).__name__}"
        )

    if not timeframes:
        raise ValueError(f"timeframes cannot be empty for {exchange}:{symbol}")

    logger.debug(
        "Validating %d time horizons for %s:%s",
        len(timeframes),
        exchange,
        symbol,
    )

    # Validate each bar and its enabled flag
    for bar, enabled in timeframes.items():
        # Validate bar key is a string
        if not isinstance(bar, str):
            raise ValueError(
                f"Invalid bar key type '{type(bar).__name__}' "
                f"(expected str) for {exchange}:{symbol}"
            )

        # Edge case: Validate bar is not empty or whitespace
        if not bar or not bar.strip():
            raise ValueError(
                f"Bar notation cannot be empty or whitespace for {exchange}:{symbol}"
            )

        bar = bar.strip()

        # Edge case: Validate bar has at least 2 characters (number + suffix)
        if len(bar) < 2:
            raise ValueError(
                f"Invalid bar notation '{bar}' for {exchange}:{symbol}. "
                "Bar must have format: <number><suffix> (e.g., '1m', '4h', '1d')"
            )

        # Validate bar notation suffix
        if bar[-1] not in VALID_HORIZONS:
            raise ValueError(
                f"Invalid bar suffix '{bar}' for {exchange}:{symbol}. "
                f"Valid suffixes: {', '.join(sorted(VALID_HORIZONS))}"
            )

        # Edge case: Validate bar has numeric component before suffix
        numeric_part = bar[:-1]
        if not numeric_part or not numeric_part.isdigit():
            raise ValueError(
                f"Bar '{bar}' has no valid numeric component for {exchange}:{symbol}. "
                "Expected format: <number><suffix> (e.g., '1m', '15m', '4h')"
            )

        # Validate enabled flag is boolean
        if not isinstance(enabled, bool):
            raise ValueError(
                f"Enabled flag must be boolean for {exchange}:{symbol}:{bar}, "
                f"got {type(enabled).__name__}"
            )

    logger.debug("Time horizons validation passed for %s:%s", exchange, symbol)

# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================


def serialize_dict(
    data: Dict[str, Any],
    sort_keys: bool = True,
    indent: Optional[int] = None,
    ensure_ascii: bool = False,
) -> str:
    """Serialize a dictionary into a JSON string.

    This is a generic serialization utility that handles any dictionary,
    making it suitable for storing complex data structures in database
    TEXT or JSONB columns.

    Args:
        data: Dictionary to serialize. Must contain only JSON-serializable
            types (str, int, float, bool, list, dict, None).
        sort_keys: If True, sort dictionary keys in output. Default True.
            Sorting ensures deterministic output for hashing and comparisons.
        indent: Number of spaces for pretty-printing. None (default) produces
            compact output without newlines or extra spaces.
        ensure_ascii: If True, escape non-ASCII characters. False (default)
            preserves Unicode characters as-is.

    Returns:
        JSON-encoded string representation of the dictionary.

    Raises:
        ValueError: If data is not a dictionary.
        TypeError: If data contains non-serializable objects (e.g., functions,
            custom classes without JSON support).

    Examples:
        >>> data = {"b": 2, "a": 1, "c": {"nested": True}}
        >>> serialize_dict(data)
        '{"a":1,"b":2,"c":{"nested":true}}'

        >>> serialize_dict(data, indent=2)
        '''{
          "a": 1,
          "b": 2,
          "c": {
            "nested": true
          }
        }'''

        >>> data = {"exchange": "binance", "enabled": True}
        >>> json_str = serialize_dict(data)
        >>> # Store in database JSONB column
        >>> conn.execute("INSERT INTO table (config) VALUES (:cfg)", cfg=json_str)

    Notes:
        - Sorted keys ensure deterministic output for testing and hashing
        - Compact output (no indent) minimizes database storage
        - Pretty output (with indent) improves readability for logging
        - For JSONB columns, PostgreSQL handles JSON natively
        - For TEXT columns, you'll need to deserialize on read
        - Boolean values serialize to lowercase "true"/"false" (JSON standard)
    """
    # Type validation
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected dict, got {type(data).__name__}. "
            "serialize_dict only accepts dictionary objects."
        )

    try:
        json_str = json.dumps(
            data,
            sort_keys=sort_keys,
            indent=indent,
            ensure_ascii=ensure_ascii,
        )
        logger.debug(
            "Serialized dictionary with %d keys to JSON (%d bytes)",
            len(data),
            len(json_str),
        )
        return json_str

    except (TypeError, ValueError) as exc:
        logger.error(
            "Failed to serialize dictionary: %s. Dictionary keys: %s",
            exc,
            list(data.keys()) if isinstance(data, dict) else "N/A",
        )
        raise TypeError(
            f"Failed to serialize dictionary: {exc}. "
            "Ensure all values are JSON-serializable "
            "(str, int, float, bool, list, dict, None)."
        ) from exc

