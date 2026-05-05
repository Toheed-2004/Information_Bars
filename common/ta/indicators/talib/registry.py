"""
TALib Indicator Registry - Complete Set
"""

import numpy as np
from typing import Dict, List, Any
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)

# TALIB INDICATORS - COMPLETE
TALIB_INDICATORS: Dict[str, Dict[str, Any]] = {
    # --- Overlap Studies ---
    "SMA": {
        "lib": "talib",
        "func_name": "SMA",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 30, "range": (2, 100), "type": int}},
        "outputs": ["sma"],
        "description": "Simple Moving Average",
    },
    "EMA": {
        "lib": "talib",
        "func_name": "EMA",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 30, "range": (2, 100), "type": int}},
        "outputs": ["ema"],
        "description": "Exponential Moving Average",
    },
    "DEMA": {
        "lib": "talib",
        "func_name": "DEMA",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 30, "range": (2, 100), "type": int}},
        "outputs": ["dema"],
        "description": "Double Exponential Moving Average",
    },
    "TEMA": {
        "lib": "talib",
        "func_name": "TEMA",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 30, "range": (2, 100), "type": int}},
        "outputs": ["tema"],
        "description": "Triple Exponential Moving Average",
    },
    "TRIMA": {
        "lib": "talib",
        "func_name": "TRIMA",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 30, "range": (2, 100), "type": int}},
        "outputs": ["trima"],
        "description": "Triangular Moving Average",
    },
    "KAMA": {
        "lib": "talib",
        "func_name": "KAMA",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 30, "range": (2, 100), "type": int}},
        "outputs": ["kama"],
        "description": "Kaufman Adaptive Moving Average",
    },
    "MAMA": {
        "lib": "talib",
        "func_name": "MAMA",
        "inputs": ["close"],
        "params": {
            "fastlimit": {"default": 0.5, "range": (0.1, 0.99), "type": float},
            "slowlimit": {"default": 0.05, "range": (0.01, 0.99), "type": float},
        },
        "outputs": ["mama", "fama"],
        "description": "MESA Adaptive Moving Average",
    },
    "T3": {
        "lib": "talib",
        "func_name": "T3",
        "inputs": ["close"],
        "params": {
            "timeperiod": {"default": 5, "range": (2, 100), "type": int},
            "vfactor": {"default": 0.7, "range": (0, 1), "type": float},
        },
        "outputs": ["t3"],
        "description": "Triple Exponential Moving Average (T3)",
    },
    "WMA": {
        "lib": "talib",
        "func_name": "WMA",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 30, "range": (2, 100), "type": int}},
        "outputs": ["wma"],
        "description": "Weighted Moving Average",
    },
    "MA": {
        "lib": "talib",
        "func_name": "MA",
        "inputs": ["close"],
        "params": {
            "timeperiod": {"default": 30, "range": (2, 100), "type": int},
            "matype": {"default": 0, "range": (0, 8), "type": int},
        },
        "outputs": ["ma"],
        "description": "Moving Average",
    },
    "MAVP": {
        "lib": "talib",
        "func_name": "MAVP",
        "inputs": ["close", "periods"],
        "params": {
            "minperiod": {"default": 2, "range": (2, 100), "type": int},
            "maxperiod": {"default": 30, "range": (2, 100), "type": int},
            "matype": {"default": 0, "range": (0, 8), "type": int},
        },
        "outputs": ["mavp"],
        "description": "Moving Average with Variable Period",
    },
    "SAR": {
        "lib": "talib",
        "func_name": "SAR",
        "inputs": ["high", "low"],
        "params": {
            "acceleration": {"default": 0.02, "range": (0.01, 0.2), "type": float},
            "maximum": {"default": 0.2, "range": (0.1, 1.0), "type": float},
        },
        "outputs": ["sar"],
        "description": "Parabolic SAR",
    },
    "SAREXT": {
        "lib": "talib",
        "func_name": "SAREXT",
        "inputs": ["high", "low"],
        "params": {
            "startvalue": {"default": 0.0, "range": (0, 1), "type": float},
            "offsetonreverse": {"default": 0.0, "range": (0, 1), "type": float},
            "accelerationinitlong": {
                "default": 0.02,
                "range": (0.01, 0.2),
                "type": float,
            },
            "accelerationlong": {"default": 0.02, "range": (0.01, 0.2), "type": float},
            "accelerationmaxlong": {"default": 0.2, "range": (0.1, 1.0), "type": float},
            "accelerationinitshort": {
                "default": 0.02,
                "range": (0.01, 0.2),
                "type": float,
            },
            "accelerationshort": {"default": 0.02, "range": (0.01, 0.2), "type": float},
            "accelerationmaxshort": {
                "default": 0.2,
                "range": (0.1, 1.0),
                "type": float,
            },
        },
        "outputs": ["sarext"],
        "description": "Parabolic SAR - Extended",
    },
    "BBANDS": {
        "lib": "talib",
        "func_name": "BBANDS",
        "inputs": ["close"],
        "params": {
            "timeperiod": {"default": 5, "range": (2, 100), "type": int},
            "nbdevup": {"default": 2, "range": (0, 5), "type": float},
            "nbdevdn": {"default": 2, "range": (0, 5), "type": float},
            "matype": {"default": 0, "range": (0, 8), "type": int},
        },
        "outputs": ["upperband", "middleband", "lowerband"],
        "description": "Bollinger Bands",
    },
    # --- Momentum Indicators ---
    "ADX": {
        "lib": "talib",
        "func_name": "ADX",
        "inputs": ["high", "low", "close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["adx"],
        "description": "Average Directional Movement Index",
    },
    "ADXR": {
        "lib": "talib",
        "func_name": "ADXR",
        "inputs": ["high", "low", "close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["adxr"],
        "description": "Average Directional Movement Index Rating",
    },
    "APO": {
        "lib": "talib",
        "func_name": "APO",
        "inputs": ["close"],
        "params": {
            "fastperiod": {"default": 12, "range": (2, 50), "type": int},
            "slowperiod": {"default": 26, "range": (10, 100), "type": int},
            "matype": {"default": 0, "range": (0, 8), "type": int},
        },
        "outputs": ["apo"],
        "description": "Absolute Price Oscillator",
    },
    "AROON": {
        "lib": "talib",
        "func_name": "AROON",
        "inputs": ["high", "low"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["aroondown", "aroonup"],
        "description": "Aroon",
    },
    "AROONOSC": {
        "lib": "talib",
        "func_name": "AROONOSC",
        "inputs": ["high", "low"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["aroonosc"],
        "description": "Aroon Oscillator",
    },
    "BOP": {
        "lib": "talib",
        "func_name": "BOP",
        "inputs": ["open", "high", "low", "close"],
        "params": {},
        "outputs": ["bop"],
        "description": "Balance of Power",
    },
    "CCI": {
        "lib": "talib",
        "func_name": "CCI",
        "inputs": ["high", "low", "close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["cci"],
        "description": "Commodity Channel Index",
    },
    "CMO": {
        "lib": "talib",
        "func_name": "CMO",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["cmo"],
        "description": "Chande Momentum Oscillator",
    },
    "DX": {
        "lib": "talib",
        "func_name": "DX",
        "inputs": ["high", "low", "close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["dx"],
        "description": "Directional Movement Index",
    },
    "MACD": {
        "lib": "talib",
        "func_name": "MACD",
        "inputs": ["close"],
        "params": {
            "fastperiod": {"default": 12, "range": (2, 50), "type": int},
            "slowperiod": {"default": 26, "range": (10, 100), "type": int},
            "signalperiod": {"default": 9, "range": (3, 30), "type": int},
        },
        "outputs": ["macd", "macdsignal", "macdhist"],
        "description": "Moving Average Convergence/Divergence",
    },
    "MACDEXT": {
        "lib": "talib",
        "func_name": "MACDEXT",
        "inputs": ["close"],
        "params": {
            "fastperiod": {"default": 12, "range": (2, 50), "type": int},
            "fastmatype": {"default": 0, "range": (0, 8), "type": int},
            "slowperiod": {"default": 26, "range": (10, 100), "type": int},
            "slowmatype": {"default": 0, "range": (0, 8), "type": int},
            "signalperiod": {"default": 9, "range": (3, 30), "type": int},
            "signalmatype": {"default": 0, "range": (0, 8), "type": int},
        },
        "outputs": ["macd", "macdsignal", "macdhist"],
        "description": "MACD with controllable MA type",
    },
    "MACDFIX": {
        "lib": "talib",
        "func_name": "MACDFIX",
        "inputs": ["close"],
        "params": {"signalperiod": {"default": 9, "range": (3, 30), "type": int}},
        "outputs": ["macd", "macdsignal", "macdhist"],
        "description": "Moving Average Convergence/Divergence Fix 12/26",
    },
    "MFI": {
        "lib": "talib",
        "func_name": "MFI",
        "inputs": ["high", "low", "close", "volume"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["mfi"],
        "description": "Money Flow Index",
    },
    'PLUS_DI': {
        'lib': 'talib',
        'func_name': 'PLUS_DI',
        'inputs': ['high', 'low', 'close'],
        'params': {
            'timeperiod': {'default': 14, 'range': (2, 100), 'type': int}
        },
        'outputs': ['plus_di'],
        'description': 'Plus Directional Indicator'
    },
    "MINUS_DI": {
        "lib": "talib",
        "func_name": "MINUS_DI",
        "inputs": ["high", "low", "close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["minus_di"],
        "description": "Minus Directional Indicator",
    },
    "MINUS_DM": {
        "lib": "talib",
        "func_name": "MINUS_DM",
        "inputs": ["high", "low"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["minus_dm"],
        "description": "Minus Directional Movement",
    },
    "MOM": {
        "lib": "talib",
        "func_name": "MOM",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 10, "range": (1, 100), "type": int}},
        "outputs": ["mom"],
        "description": "Momentum",
    },

    "PPO": {
        "lib": "talib",
        "func_name": "PPO",
        "inputs": ["close"],
        "params": {
            "fastperiod": {"default": 12, "range": (2, 50), "type": int},
            "slowperiod": {"default": 26, "range": (10, 100), "type": int},
            "matype": {"default": 0, "range": (0, 8), "type": int},
        },
        "outputs": ["ppo"],
        "description": "Percentage Price Oscillator",
    },
    "ROC": {
        "lib": "talib",
        "func_name": "ROC",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 10, "range": (1, 100), "type": int}},
        "outputs": ["roc"],
        "description": "Rate of change: ((price/prevPrice)-1)*100",
    },
    "ROCP": {
        "lib": "talib",
        "func_name": "ROCP",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 10, "range": (1, 100), "type": int}},
        "outputs": ["rocp"],
        "description": "Rate of change Percentage: (price-prevPrice)/prevPrice",
    },
    "ROCR": {
        "lib": "talib",
        "func_name": "ROCR",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 10, "range": (1, 100), "type": int}},
        "outputs": ["rocr"],
        "description": "Rate of change ratio: (price/prevPrice)",
    },
    "ROCR100": {
        "lib": "talib",
        "func_name": "ROCR100",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 10, "range": (1, 100), "type": int}},
        "outputs": ["rocr100"],
        "description": "Rate of change ratio 100 scale: (price/prevPrice)*100",
    },
    "RSI": {
        "lib": "talib",
        "func_name": "RSI",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["rsi"],
        "description": "Relative Strength Index",
    },
    "STOCH": {
        "lib": "talib",
        "func_name": "STOCH",
        "inputs": ["high", "low", "close"],
        "params": {
            "fastk_period": {"default": 5, "range": (1, 50), "type": int},
            "slowk_period": {"default": 3, "range": (1, 30), "type": int},
            "slowk_matype": {"default": 0, "range": (0, 8), "type": int},
            "slowd_period": {"default": 3, "range": (1, 30), "type": int},
            "slowd_matype": {"default": 0, "range": (0, 8), "type": int},
        },
        "outputs": ["slowk", "slowd"],
        "description": "Stochastic",
    },
    "STOCHF": {
        "lib": "talib",
        "func_name": "STOCHF",
        "inputs": ["high", "low", "close"],
        "params": {
            "fastk_period": {"default": 5, "range": (1, 50), "type": int},
            "fastd_period": {"default": 3, "range": (1, 30), "type": int},
            "fastd_matype": {"default": 0, "range": (0, 8), "type": int},
        },
        "outputs": ["fastk", "fastd"],
        "description": "Stochastic Fast",
    },
    "STOCHRSI": {
        "lib": "talib",
        "func_name": "STOCHRSI",
        "inputs": ["close"],
        "params": {
            "timeperiod": {"default": 14, "range": (2, 100), "type": int},
            "fastk_period": {"default": 5, "range": (1, 50), "type": int},
            "fastd_period": {"default": 3, "range": (1, 30), "type": int},
            "fastd_matype": {"default": 0, "range": (0, 8), "type": int},
        },
        "outputs": ["fastk", "fastd"],
        "description": "Stochastic Relative Strength Index",
    },
    "TRIX": {
        "lib": "talib",
        "func_name": "TRIX",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 30, "range": (2, 100), "type": int}},
        "outputs": ["trix"],
        "description": "1-day Rate-Of-Change (ROC) of a Triple Smooth EMA",
    },
    "ULTOSC": {
        "lib": "talib",
        "func_name": "ULTOSC",
        "inputs": ["high", "low", "close"],
        "params": {
            "timeperiod1": {"default": 7, "range": (1, 30), "type": int},
            "timeperiod2": {"default": 14, "range": (7, 50), "type": int},
            "timeperiod3": {"default": 28, "range": (14, 100), "type": int},
        },
        "outputs": ["ultosc"],
        "description": "Ultimate Oscillator",
    },
    "WILLR": {
        "lib": "talib",
        "func_name": "WILLR",
        "inputs": ["high", "low", "close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["willr"],
        "description": "Williams' %R",
    },

    # --- Volatility ---
    "ATR": {
        "lib": "talib",
        "func_name": "ATR",
        "inputs": ["high", "low", "close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["atr"],
        "description": "Average True Range",
    },
    "NATR": {
        "lib": "talib",
        "func_name": "NATR",
        "inputs": ["high", "low", "close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["natr"],
        "description": "Normalized Average True Range",
    },
    "TRANGE": {
        "lib": "talib",
        "func_name": "TRANGE",
        "inputs": ["high", "low", "close"],
        "params": {},
        "outputs": ["trange"],
        "description": "True Range",
    },
    "LINEARREG": {
        "lib": "talib",
        "func_name": "LINEARREG",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["linearreg"],
        "description": "Linear Regression",
    },
    "LINEARREG_ANGLE": {
        "lib": "talib",
        "func_name": "LINEARREG_ANGLE",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["linearreg_angle"],
        "description": "Linear Regression Angle",
    },
    "LINEARREG_INTERCEPT": {
        "lib": "talib",
        "func_name": "LINEARREG_INTERCEPT",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["linearreg_intercept"],
        "description": "Linear Regression Intercept",
    },
    "LINEARREG_SLOPE": {
        "lib": "talib",
        "func_name": "LINEARREG_SLOPE",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["linearreg_slope"],
        "description": "Linear Regression Slope",
    },
    "STDDEV": {
        "lib": "talib",
        "func_name": "STDDEV",
        "inputs": ["close"],
        "params": {
            "timeperiod": {"default": 14, "range": (2, 100), "type": int},
            "nbdev": {"default": 1, "range": (0, 5), "type": float},
        },
        "outputs": ["stddev"],
        "description": "Standard Deviation",
    },
    "TSF": {
        "lib": "talib",
        "func_name": "TSF",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["tsf"],
        "description": "Time Series Forecast",
    },
    "VAR": {
        "lib": "talib",
        "func_name": "VAR",
        "inputs": ["close"],
        "params": {"timeperiod": {"default": 14, "range": (2, 100), "type": int}},
        "outputs": ["var"],
        "description": "Variance",
    },
    # --- Volume ---
    "AD": {
        "lib": "talib",
        "func_name": "AD",
        "inputs": ["high", "low", "close", "volume"],
        "params": {},
        "outputs": ["ad"],
        "description": "Chaikin A/D Line",
    },
    "ADOSC": {
        "lib": "talib",
        "func_name": "ADOSC",
        "inputs": ["high", "low", "close", "volume"],
        "params": {
            "fastperiod": {"default": 3, "range": (2, 50), "type": int},
            "slowperiod": {"default": 10, "range": (5, 100), "type": int},
        },
        "outputs": ["adosc"],
        "description": "Chaikin A/D Oscillator",
    },
    "OBV": {
        "lib": "talib",
        "func_name": "OBV",
        "inputs": ["close", "volume"],
        "params": {},
        "outputs": ["obv"],
        "description": "On Balance Volume",
    },
   
}

# Indicator categories mapping
INDICATOR_CATEGORIES = {
    "momentum": [
        "RSI",
        "STOCH",
        "STOCHF",
        "STOCHRSI",
        "WILLR",
        "CMO",
        "MOM",
        "CCI",
        "MFI",
        "ULTOSC",
        "DX",
        "ADX",
        "ADXR",
        "PLUS_DI",
        "MINUS_DI",
        "ROC",
        "ROCP",
        "ROCR",
        "ROCR100",
        "TRIX",
        "APO",
        "PPO",
        "LINEARREG_ANGLE",
        "LINEARREG_SLOPE",
        "AROONOSC",
        "AROON",
        "BOP",
    ],
    "volatility": [
        "ATR",
        "NATR",
        "TRANGE",
        "STDDEV",
        "VAR",
        "BBANDS",
    ],
    "volume": [
        "AD",
        "ADOSC",
        "OBV",
    ],
    "overlap": [
        "MACD",
        "MACDEXT",
        "MACDFIX",
        "MA",
        "EMA",
        "SMA",
        "WMA",
        "DEMA",
        "TEMA",
        "TRIMA",
        "KAMA",
        "T3",
        "MAMA",
        "SAR",
        "SAREXT",
        "TSF",
        "LINEARREG",
        "LINEARREG_INTERCEPT",
        "MAVP",
    ],
}

# Summary counts by category
INDICATOR_COUNTS = {
    "Overlap Studies": 14,
    "Momentum Indicators": 29,
    "Volume Indicators": 3,
    "Cycle Indicators": 5,
    "Volatility Indicators": 3,
    "Total": 54,
}
# Column naming key parameter order for each indicator
COLUMN_NAME_PARAM_ORDER = {
    # Moving Averages / Overlap Studies
    "SMA": ["timeperiod"],
    "EMA": ["timeperiod"],
    "DEMA": ["timeperiod"],
    "TEMA": ["timeperiod"],
    "TRIMA": ["timeperiod"],
    "KAMA": ["timeperiod"],
    "MAMA": ["fastlimit", "slowlimit"],
    "T3": ["timeperiod", "vfactor"],
    "MA": ["timeperiod", "matype"],
    "MAVP": ["minperiod", "maxperiod", "matype"],
    "MIDPOINT": ["timeperiod"],
    "MIDPRICE": ["timeperiod"],
    "BBANDS": ["timeperiod", "nbdevup", "nbdevdn", "matype"],
    # Momentum Indicators
    "ADX": ["timeperiod"],
    "ADXR": ["timeperiod"],
    "APO": ["fastperiod", "slowperiod", "matype"],
    "AROON": ["timeperiod"],
    "AROONOSC": ["timeperiod"],
    "BOP": [],  # No parameters
    "CCI": ["timeperiod"],
    "CMO": ["timeperiod"],
    "DX": ["timeperiod"],
    "MACD": ["fastperiod", "slowperiod", "signalperiod"],
    "MACDEXT": [
        "fastperiod",
        "fastmatype",
        "slowperiod",
        "slowmatype",
        "signalperiod",
        "signalmatype",
    ],
    "MACDFIX": ["signalperiod"],
    "MFI": ["timeperiod"],
    "MINUS_DI": ["timeperiod"],
    "MINUS_DM": ["timeperiod"],
    "MOM": ["timeperiod"],
    "PLUS_DI": ["timeperiod"],
    "PLUS_DM": ["timeperiod"],
    "PPO": ["fastperiod", "slowperiod", "matype"],
    "ROC": ["timeperiod"],
    "ROCP": ["timeperiod"],
    "ROCR": ["timeperiod"],
    "ROCR100": ["timeperiod"],
    "RSI": ["timeperiod"],
    "STOCH": ["fastk_period", "slowk_period", "slowd_period"],
    "STOCHF": ["fastk_period", "fastd_period"],
    "STOCHRSI": ["timeperiod", "fastk_period", "fastd_period", "fastd_matype"],
    "TRIX": ["timeperiod"],
    "ULTOSC": ["timeperiod1", "timeperiod2", "timeperiod3"],
    "WILLR": ["timeperiod"],
    # Cycle Indicators
    "HT_DCPERIOD": [],
    "HT_DCPHASE": [],
    "HT_PHASOR": [],
    "HT_SINE": [],
    "HT_TRENDMODE": [],
    # Volatility Indicators
    "ATR": ["timeperiod"],
    "NATR": ["timeperiod"],
    "TRANGE": [],
    # Price / Math Functions
    "AVGPRICE": [],
    "MEDPRICE": [],
    "TYPPRICE": [],
    "WCLPRICE": [],
    "BETA": ["timeperiod"],
    "CORREL": ["timeperiod"],
    "LINEARREG": ["timeperiod"],
    "LINEARREG_ANGLE": ["timeperiod"],
    "LINEARREG_INTERCEPT": ["timeperiod"],
    "LINEARREG_SLOPE": ["timeperiod"],
    "STDDEV": ["timeperiod", "nbdev"],
    "TSF": ["timeperiod"],
    "VAR": ["timeperiod"],
    # Volume Indicators
    "AD": [],
    "ADOSC": ["fastperiod", "slowperiod"],
    "OBV": [],
    # Arithmetic / Statistical Functions
    "ADD": [],
    "DIV": [],
    "MAX": [],
    "MAXINDEX": [],
    "MIN": [],
    "MININDEX": [],
    "MINMAX": [],
    "MINMAXINDEX": [],
    "MULT": [],
    "SUB": [],
    "ATAN": [],
    "CEIL": [],
    "COS": [],
    "COSH": [],
    "FLOOR": [],
    "LN": [],
    "LOG10": [],
    "SIN": [],
    "SINH": [],
    "SQRT": [],
    "TAN": [],
    "TANH": [],
}
OUTPUT_NAME_MAPPING = {
    'MACD': {
        'signal': 'macd_signal',
        'hist': 'macd_hist',
    },
    'STOCH': {
        'fast_k': 'fastk',
        'slow_k': 'slowk',
        'slow_d': 'slowd',
    }
}

def create_column_name(
    indicator_name: str, output_name: str, params: Dict[str, Any]
) -> str:
    """
    Create column name with intelligent prefixing.

    Rules:
    - If output already starts with indicator name -> keep as is
    - Else prepend indicator name
    - Append numeric params in defined order
    """

    indicator_lower = indicator_name.lower()
    output_lower = output_name.lower()
    if indicator_name in OUTPUT_NAME_MAPPING:
        output_lower = OUTPUT_NAME_MAPPING[indicator_name].get(output_lower, output_lower)

    # Prefix logic (keep existing indicator prefix if present)
    if output_lower.startswith(indicator_lower):
        base_name = output_lower
    else:
        base_name = f"{indicator_lower}_{output_lower}"

    # Parameter ordering
    key_params = COLUMN_NAME_PARAM_ORDER.get(indicator_name, sorted(params.keys()))

    # Preserve float decimals if present, otherwise use ints
    param_values = []
    for param_name in key_params:
        if param_name in params:
            value = params[param_name]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float) and value != int(value):
                    param_values.append(str(value))
                else:
                    param_values.append(str(int(value)))

    if param_values:
        col = f"{base_name}_{'_'.join(param_values)}"
    else:
        col = base_name

    # Prefix for TALib-sourced indicators
    return f"talib_ind_{col}"


def get_indicators_by_category(category: str) -> List[str]:
    """
    Returns a list of indicators based on category name.

    Valid categories:
    - 'momentum'
    - 'volatility'
    - 'cycle'
    - 'volume'
    - 'overlap'
    
    Args:
        category: Category name (case-insensitive)
        
    Returns:
        List of indicator names in the category
        
    Raises:
        ValueError: If category is not valid
    """
    category = category.lower()
    logger.debug(f"Retrieving indicators for category: {category}")

    if category not in INDICATOR_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. "
            f"Choose from {list(INDICATOR_CATEGORIES.keys())}"
        )

    return INDICATOR_CATEGORIES[category]