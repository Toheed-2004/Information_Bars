"""
vectorbtpro Indicator Registry - 14 indicators
"""

from typing import Dict, Any, Optional
from enum import Enum


# ==============================================================================
# ENUMS - Required by VBT Indicators
# ==============================================================================
class WType(Enum):
    """Weighting types for VBT indicators"""
    SIMPLE = "simple"
    WEIGHTED = "weighted"
    EXP = "exp"
    WILDER = "wilder"
    VIDYA = "vidya"


class HurstMethod(Enum):
    """Hurst calculation methods for VBT"""
    STANDARD = "standard"
    LOGRS = "logrs"
    RS = "rs"
    DMA = "dma"
    DSOD = "dsod"

VBT_INDICATORS = {
    'RSI': {
        'lib': 'vectorbtpro',
        'func_name': 'RSI',
        'category': 'momentum',
        'inputs': ['close'],
        'params': {
            'window': {'default': 14, 'range': (2, 1000), 'type': int},
            'wtype': {'default': 'wilder', 'options': [e.value for e in WType], 'type': str}
        },
        'outputs': ['rsi'],
        'description': 'Relative Strength Index - Momentum oscillator measuring speed and change of price movements',
        'column_examples': ['rsi_14_rsi', 'rsi_21_rsi', 'rsi_28_rsi'],
        'interpretation': 'Values 0-100. >70 overbought, <30 oversold. Divergences signal reversals.'
    },
    
    'MACD': {
        'lib': 'vectorbtpro',
        'func_name': 'MACD',
        'category': 'trend',
        'inputs': ['close'],
        'params': {
            'fast_window': {'default': 12, 'range': (5, 1000), 'type': int},
            'slow_window': {'default': 26, 'range': (10, 1000), 'type': int},
            'signal_window': {'default': 9, 'range': (3, 30), 'type': int},
            'wtype': {'default': 'exp', 'options': [e.value for e in WType], 'type': str},
            'macd_wtype': {'default': 'simple', 'options': [e.value for e in WType] + [None], 'type': Optional[str]},
            'signal_wtype': {'default': 'simple', 'options': [e.value for e in WType] + [None], 'type': Optional[str]}
        },
        'outputs': ['macd', 'signal'],
        'description': 'Moving Average Convergence Divergence - Trend-following momentum indicator',
        'column_examples': ['macd_12_26_9', 'macdsignal_12_26_9'],
        'interpretation': 'MACD line crossovers signal line indicate buy/sell. Histogram shows momentum.'
    },
    
    'BBANDS': {
        'lib': 'vectorbtpro',
        'func_name': 'BBANDS',
        'category': 'volatility',
        'inputs': ['close'],
        'params': {
            'window': {'default': 14, 'range': (5, 1000), 'type': int},
            'wtype': {'default': 'simple', 'options': ['simple', 'exp', 'weighted', 'wilder'], 'type': str},
            'alpha': {'default': 2.0, 'range': (1.0, 4.0), 'type': float}
        },
        'outputs': ['upper', 'middle', 'lower'],
        'description': 'Bollinger Bands - Volatility indicator with moving average and standard deviation bands',
        'column_examples': ['bbands_20_2.0_upper', 'bbands_20_2.0_middle', 'bbands_20_2.0_lower'],
        'interpretation': 'Price near upper band = overbought, near lower band = oversold. Band squeeze = low volatility.'
    },
    
    'ATR': {
        'lib': 'vectorbtpro',
        'func_name': 'ATR',
        'inputs': ['high', 'low', 'close'],
        'params': {
            'window': {'default': 14, 'range': (2, 1000), 'type': int},
            'wtype': {'default': 'wilder', 'options': [e.value for e in WType], 'type': str}
        },
        'outputs': ['tr', 'atr'],
        'description': 'Average True Range'
    },
    
    'ADX': {
        'lib': 'vectorbtpro',
        'func_name': 'ADX',
        'inputs': ['high', 'low', 'close'],
        'params': {
            'window': {'default': 14, 'range': (2, 1000), 'type': int},
            'wtype': {'default': 'wilder', 'options': [e.value for e in WType], 'type': str}
        },
        'outputs': ['plus_di', 'minus_di', 'dx', 'adx'],
        'description': 'Average Directional Index'
    },
    
    'MA': {
        'lib': 'vectorbtpro',
        'func_name': 'MA',
        'inputs': ['close'],
        'params': {
            'window': {'default': 14, 'range': (2, 200), 'type': int},
            'wtype': {'default': 'simple', 'options': [e.value for e in WType], 'type': str}
        },
        'outputs': ['ma'],
        'description': 'Moving Average'
    },
    
    'STOCH': {
        'lib': 'vectorbtpro',
        'func_name': 'STOCH',
        'inputs': ['high', 'low', 'close'],
        'params': {
            'fast_k_window': {'default': 14, 'range': (2, 50), 'type': int},
            'slow_k_window': {'default': 3, 'range': (1, 20), 'type': int},
            'slow_d_window': {'default': 3, 'range': (1, 20), 'type': int},
            'wtype': {'default': 'simple', 'options': [e.value for e in WType], 'type': str},
            'slow_k_wtype': {'default': 'simple', 'options': [e.value for e in WType] + [None], 'type': Optional[str]},
            'slow_d_wtype': {'default': 'simple', 'options': [e.value for e in WType] + [None], 'type': Optional[str]}
        },
        'outputs': ['fast_k', 'slow_k', 'slow_d'],
        'description': 'Stochastic Oscillator'
    },
    
    'SUPERTREND': {
        'lib': 'vectorbtpro',
        'func_name': 'SUPERTREND',
        'inputs': ['high', 'low', 'close'],
        'params': {
            'period': {'default': 7, 'range': (2, 50), 'type': int},
            'multiplier': {'default': 3.0, 'range': (1.0, 10.0), 'type': float}
        },
        'outputs': ['trend', 'direction', 'long', 'short'],
        'description': 'SuperTrend'
    },
    
    'OBV': {
        'lib': 'vectorbtpro',
        'func_name': 'OBV',
        'inputs': ['close', 'volume'],
        'params': {},
        'outputs': ['obv'],
        'description': 'On Balance Volume'
    },
    'VWAP': {
        'lib': 'vectorbtpro',
        'func_name': 'VWAP',
        'inputs': ['high', 'low', 'close', 'volume'],
        'params': {
            'anchor': {'default': 'D', 'options': ['D', 'W', 'M', 'Q', 'Y'], 'type': str}
        },
        'outputs': ['vwap'],
        'description': 'Volume Weighted Average Price with anchor frequency'
    },

    # 'OLS': {
    #     'lib': 'vbt',
    #     'func_name': 'OLS',
    #     'inputs': ['close'],
    #     'params': {
    #         'window': {'default': 14, 'range': (5, 200), 'type': int},
    #         'norm_window': {'default': 20, 'range': (5, 200), 'type': Optional[int]}
    #     },
    #     'outputs': ['slope', 'intercept', 'zscore'],
    #     'description': 'Ordinary Least Squares - Linear regression on close price'
    # },
    
    'HURST': {
        'lib': 'vectorbtpro',
        'func_name': 'HURST',
        'inputs': ['close'],
        'params': {
            'window': {'default': 200, 'range': (50, 500), 'type': int},
            'method': {'default': 'standard', 'options': [e.value for e in HurstMethod], 'type': str},
            'max_lag': {'default': 20, 'range': (5, 100), 'type': int},
            'min_log': {'default': 1, 'range': (1, 3), 'type': int},
            'max_log': {'default': 2, 'range': (2, 5), 'type': int},
            'log_step': {'default': 0.25, 'range': (0.1, 1.0), 'type': float},
            'min_chunk': {'default': 8, 'range': (5, 20), 'type': int},
            'max_chunk': {'default': 100, 'range': (50, 200), 'type': int},
            'num_chunks': {'default': 5, 'range': (3, 10), 'type': int}
        },
        'outputs': ['hurst'],
        'description': 'Hurst Exponent'
    },
    
    'MSD': {
        'lib': 'vectorbtpro',
        'func_name': 'MSD',
        'inputs': ['close'],
        'params': {
            'window': {'default': 14, 'range': (2, 100), 'type': int},
            'wtype': {'default': 'simple', 'options': ['simple', 'exp', 'wilder'], 'type': str}
        },
        'outputs': ['msd'],
        'description': 'Moving Standard Deviation'
    }
}

# Column naming key parameter order for each indicator
COLUMN_NAME_PARAM_ORDER = {
    'RSI': ['window'],
    'MACD': ['fast_window', 'slow_window', 'signal_window'],
    'BBANDS': ['window'],
    'ATR': ['window'],
    'ADX': ['window'],
    'MA': ['window'],
    'KAMA': ['window'],
    'DEMA': ['window'],
    'EMA': ['window'],
    'SUPERTREND': ['period', 'multiplier'],
    'OBV': [],
    'HURST': ['window', 'method'],
    'MSD': ['window'],
    'STOCH': ['fast_k_window', 'slow_k_window', 'slow_d_window'],
    "VWAP" : ['anchor']
}

# Output name mapping for VBT to match TALib naming convention
# Format: {indicator_name: {vbt_output: talib_output}}
OUTPUT_NAME_MAPPING = {
    'MACD': {
        'signal': 'macd_signal',
    },
    'STOCH': {
        'fast_k': 'fastk',
        'slow_k': 'slowk',
        'slow_d': 'slowd',
    }
}

def create_column_name(indicator_name: str, output_name: str, params: Dict[str, Any]) -> str:
    """Create column name following TALib convention
    
    Format: {output_lower}_{numeric_params}
    
    Args:
        indicator_name: Name of the indicator (e.g., 'RSI', 'MACD')
        output_name: Output name (e.g., 'rsi', 'macd', 'signal')
        params: Dict of parameters passed to indicator
    
    Returns:
        Column name string (e.g., 'rsi_14', 'macd_12_26_9', 'macdsignal_12_26_9')
    """
    # Standardize output name mapping and lower-case
    output_lower = output_name.lower()
    if indicator_name in OUTPUT_NAME_MAPPING:
        output_lower = OUTPUT_NAME_MAPPING[indicator_name].get(output_lower, output_lower)

    # Get parameter order for this indicator
    key_params = COLUMN_NAME_PARAM_ORDER.get(indicator_name, sorted(params.keys()))

    # Extract numeric parameter values in order, preserving float precision when needed
    param_values = []
    for param_name in key_params:
        if param_name in params:
            value = params[param_name]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float) and value != int(value):
                    param_values.append(str(value))
                else:
                    param_values.append(str(int(value)))

    # Build base column name (match TALib-style internals)
    base_name = output_lower
    if param_values:
        params_str = '_'.join(param_values)
        col = f"{base_name}_{params_str}"
    else:
        col = base_name

    # Prefix for VBT-sourced indicators
    return f"vbt_ind_{col}"