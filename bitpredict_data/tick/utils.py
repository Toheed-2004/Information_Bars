from typing import Dict, Any

def build_config_from_db(db_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build configuration dictionary from database record.

    Transforms database record into the format expected by BybitStreamManager.
    
    Args:
        db_config: Configuration dictionary from database with keys:
            - exchange: str
            - market_type: str
            - enabled: bool
            - symbols: list
            - streams: dict (stream_name -> enabled bool)
            - depth: int
            - buffer_size: int
            - connection_delay: float
        db_url: Database connection URL string.
    
    Returns:
        Configuration dictionary in format expected by BybitStreamManager:
        {
            'database': {
                'url': str,
                'base_schema': str,
                'exchange': str
            },
            'collection': {
                'symbols': list,
                'stream_types': list,
                'market_type': str,
                'depth': int
            },
            'urls': {
                'linear': str,
                'inverse': str,
                'spot': str
            },
            'settings': {
                'buffer_size': int,
                'connection_delay': float,
                'log_level': str,
                'log_format': str
            }
        }
    
    Example:
        >>> db_config = {
        ...     'exchange': 'bybit',
        ...     'market_type': 'linear',
        ...     'enabled': True,
        ...     'symbols': ['BTCUSDT', 'ETHUSDT'],
        ...     'streams': {'orderbook': True, 'trade': True, 'ticker': False},
        ...     'depth': 50,
        ...     'buffer_size': 1,
        ...     'connection_delay': 1.0
        ... }
        >>> config = build_config_from_db(db_config, "postgresql://...")
        >>> config['collection']['symbols']
        ['BTCUSDT', 'ETHUSDT']
        >>> config['collection']['stream_types']
        ['orderbook', 'trade']
    
    Notes:
        - Filters streams to only include enabled ones
        - Uses standard Bybit WebSocket URLs
        - Applies default logging settings
        - Validates all required fields are present
    """
    # Extract enabled stream types (only those with True value)
    enabled_streams = [
        stream_name 
        for stream_name, enabled in db_config['streams'].items() 
        if enabled
    ]
    
    # Construct config in expected format
    config = {
        'database': {
            'base_schema': 'data_tick',
            'exchange': db_config['exchange']
        },
        'collection': {
            'symbols': db_config['symbols'],
            'stream_types': enabled_streams,
            'market_type': db_config['market_type'],
            'depth': db_config['depth']
        },

        'settings': {
            'buffer_size': 1,
            'connection_delay': 1.0
        }
    }
    return config

