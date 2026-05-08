"""
Main Indicator Calculator - Functional Design
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple
from bitpredict.common.utils.data_validation import validate_ohlcv
from bitpredict.common.ta.indicators.talib import calculate_talib
from bitpredict.common.ta.indicators.vectorbtpro import calculate_vbt
from bitpredict.common.utils.data_validation import ensure_datetime_column
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


def calculate_talib_indicators(
    data: pd.DataFrame,
    indicators: Union[str, List[str], Dict[str, Dict]] = "all",
    return_type: str = "dataframe",
    drop_nan: bool = True
) -> Tuple[Union[pd.DataFrame, Dict[str, np.ndarray]], Dict]:
    """
    Calculate indicators using TA-Lib
    
    Args:
        data: DataFrame with OHLCV columns
        indicators: 'all', indicator name, list of names, or dict with params
        return_type: 'dataframe' for pandas DataFrame, 'numpy' for numpy arrays
        drop_nan: if True (default), drop rows with NaN; if False, return raw data
        
    Returns:
        Tuple of (results, metadata)
    """
    logger.debug(f"Calculating TA-Lib indicators: {indicators}, return_type={return_type}, drop_nan={drop_nan}")
    
    result, config = calculate_talib(
        data,
        indicators=indicators,
        return_type=return_type,
        drop_nan=drop_nan
    )
    
    if return_type == "dataframe":
        logger.info(f"TA-Lib calculation complete: {len(result.columns)} indicator columns, {len(result)} rows")
    else:
        logger.info(f"TA-Lib calculation complete: {len(result)} indicators, {len(config)} configurations")
    
    return result, config


def calculate_vectorbt_indicators(
    data: pd.DataFrame,
    indicators: Union[str, List[str], Dict[str, Dict]] = "all",
    return_type: str = "dataframe",
    drop_nan: bool = True
) -> Tuple[Union[pd.DataFrame, Dict[str, np.ndarray]], Dict]:
    """
    Calculate indicators using VectorBT Pro
    
    Args:
        data: DataFrame with OHLCV columns
        indicators: 'all', indicator name, list of names, or dict with params
        return_type: 'dataframe' for pandas DataFrame, 'numpy' for numpy arrays
        drop_nan: if True (default), drop rows with NaN; if False, return raw data
        
    Returns:
        Tuple of (results, metadata)
    """
    logger.debug(f"Calculating VectorBT indicators: {indicators}, return_type={return_type}, drop_nan={drop_nan}")
    
    result, config = calculate_vbt(
        data,
        indicators=indicators,
        return_type=return_type,
        drop_nan=drop_nan
    )
    
    if return_type == "dataframe":
        logger.info(f"VectorBT calculation complete: {len(result.columns)} indicator columns, {len(result)} rows")
    else:
        logger.info(f"VectorBT calculation complete: {len(result)} indicators, {len(config)} configurations")
    
    return result, config


def calculate_indicators(
    data: pd.DataFrame,
    indicators: Union[str, List[str], Dict[str, Dict]] = "all",
    library: str = "talib",
    return_type: str = "dataframe",
    drop_nan: bool = True
) -> Tuple[Union[pd.DataFrame, np.ndarray], Dict]:
    """
    Unified function to calculate technical indicators
    
    This function handles:
    - Dataset with all indicators
    - Dataset with specific indicators
    
    Args:
        data: DataFrame with OHLCV columns 
        indicators: 'all', indicator name, list of names, or dict with params
        library: 'talib' or 'vectorbtpro'
        return_type: 'dataframe' for pandas DataFrame, 'numpy' for numpy arrays
        drop_nan: if True (default), drop rows with NaN; if False, return raw data
        
    Returns:
        Tuple of (results, metadata)
        
    Examples:
        # Calculate all indicators
        result, metadata = calculate_indicators(data=price_data)
        
        # Calculate specific indicators with custom params
        result, metadata = calculate_indicators(
            data=price_data,
            indicators={"RSI": {"timeperiod": 14}, "MACD": {}},
            return_type="dataframe"
        )
    """
    logger.info(f"Starting indicator calculation: library={library}, return_type={return_type}, drop_nan={drop_nan}")
    logger.debug(f"Input data shape: {data.shape}, indicators: {indicators}")
    
    # Process indicators parameter
    if isinstance(indicators, str) and indicators != "all":
        indicators = [indicators]
        logger.debug(f"Converted single indicator string to list: {indicators}")
    
    # Log indicator count
    if indicators == "all":
        logger.info(f"Calculating ALL available indicators using {library}")
    elif isinstance(indicators, list):
        logger.info(f"Calculating {len(indicators)} indicators using {library}: {indicators}")
    elif isinstance(indicators, dict):
        logger.info(f"Calculating {len(indicators)} indicators using {library} with custom parameters")
        logger.debug(f"Indicator configurations: {indicators}")
    
    # Validate input data
    logger.debug("Validating OHLCV data structure...")
    data = ensure_datetime_column(data)
    validated_data = validate_ohlcv(data)
    logger.debug(f"Data validation successful: {validated_data.shape}")
    
    # Normalize library name
    library = library.lower()
    
    # Dispatch to appropriate library calculator
    if library == "talib":
        logger.debug("Dispatching to TA-Lib calculator")
        return calculate_talib_indicators(
            data=validated_data,
            indicators=indicators,
            return_type=return_type,
            drop_nan=drop_nan
        )
    elif library in ["vectorbtpro", "vectorbt"]:
        logger.debug("Dispatching to VectorBT Pro calculator")
        return calculate_vectorbt_indicators(
            data=validated_data,
            indicators=indicators,
            return_type=return_type,
            drop_nan=drop_nan
        )
    else:
        error_msg = f"Unsupported library: {library}. Use 'talib' or 'vectorbtpro'"
        logger.error(error_msg)
        raise ValueError(error_msg)