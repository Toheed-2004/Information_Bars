"""
Pattern Calculator - Functional Design
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Literal, Tuple
from bitpredict.common.constants import SUPPORTED_PATTERN_LIBRARIES
from bitpredict.common.utils.data_validation import validate_ohlcv
from bitpredict.common.ta.patterns.talib.calculator import CandlestickPatternCalculator


def calculate_talib_patterns(
    data: pd.DataFrame,
    patterns: Union[str, List[str], Dict[str, Dict]] = "all",
    return_type: Literal["dataframe", "numpy"] = "dataframe",
    drop_nan: bool = True
) -> Tuple[Union[pd.DataFrame, Dict[str, np.ndarray]], Dict[str, Dict]]:
    """
    Calculate patterns using TA-Lib
    
    Args:
        data: DataFrame with OHLCV columns
        patterns: 'all', pattern name, list of names, or dict with params
        return_type: 'dataframe' for pandas DataFrame, 'numpy' for numpy arrays
        shift: Shift patterns by 1 bar
        prefix: Add library prefix to pattern column names (e.g., 'talib_')
        drop_nan: if True (default), drop rows with NaN; if False, return raw data
        **kwargs: Additional calculator options
        
    Returns:
        Tuple of (results, metadata)
    """
    
    calc = CandlestickPatternCalculator(
        data,
        patterns=patterns,
        return_type=return_type if return_type == "dataframe" else "numpy_array",
        drop_nan=drop_nan
    )
    
    result, config = calc.calculate()
    return result, config

def calculate_patterns(
    data: pd.DataFrame,
    patterns: Union[str, List[str], Dict[str, Dict]] = "all",
    library: str = "talib",
    return_type: Literal["dataframe", "numpy"] = "dataframe",
    drop_nan: bool = True
) -> Tuple[Union[pd.DataFrame, Dict[str, np.ndarray]], Dict[str, Dict]]:
    """
    Main function to calculate candlestick patterns
    
    Args:
        data: DataFrame with OHLCV columns (datetime, open, high, low, close, volume)
        patterns: 'all', pattern name, list of names, or dict with params
        library: 'talib' or 'vectorbt'
        return_type: 'dataframe' for pandas DataFrame, 'numpy' for numpy arrays
        shift: Shift patterns by 1 bar
        prefix: Add library prefix to pattern column names
        drop_nans: Remove NaN rows from data before processing
        **kwargs: Additional calculator options
        
    Returns:
        Tuple of (results, metadata)
        
    Examples:
        # Calculate all patterns using default library (TA-Lib)
        pattern_df, metadata = calculate_patterns(
            data=price_data,
            patterns="all",
            return_type="dataframe"
        )
        
        # Calculate specific patterns
        pattern_df, metadata = calculate_patterns(
            data=price_data,
            patterns=["CDL_DOJI", "CDL_HAMMER"],
            library="talib"
        )
        
        # Calculate with custom parameters
        pattern_df, metadata = calculate_patterns(
            data=price_data,
            patterns={"CDL_DOJI": {}, "CDL_HAMMER": {}},
            shift=False,
            prefix=True
        )
    """
    
    # Validate input data (keep all columns for patterns)
    validated_data = validate_ohlcv(data, ohlc_only=False)
        
    # Normalize library name
    library = library.lower()
    
    # Dispatch to appropriate library calculator
    if library == "talib":
        return calculate_talib_patterns(
            data=validated_data,
            patterns=patterns,
            return_type=return_type,
            drop_nan=drop_nan
        )

    else:
        raise ValueError(f"Unsupported library: {library}. Use 'talib' or 'vectorbt'")

