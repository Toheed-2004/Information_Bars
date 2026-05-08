"""
TA-Lib Indicator Calculator
"""

import talib
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple
from .registry import TALIB_INDICATORS, create_column_name
from bitpredict.common.constants import OHLCV_COLUMNS
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


def calculate(
    data: pd.DataFrame,
    indicators: Union[str, List[str], Dict[str, Dict]] = "all",
    return_type: str = "dataframe",
    drop_nan: bool = True
) -> Tuple[Union[pd.DataFrame, Dict[str, np.ndarray]], Dict]:
    """
    Calculate TA-Lib indicators
    
    Args:
        data: OHLCV DataFrame
        indicators: 'all', indicator name, list of names, or dict with params
        return_type: 'dataframe' or 'numpy' output
        drop_nan: if True (default), drop rows with NaN; if False, return raw data
        **kwargs: Additional options
        
    Returns:
        Tuple of (results, config)
    """
    # data = data.copy()
    # Parse indicators input
    if indicators == "all":
        indicator_list = list(TALIB_INDICATORS.keys())
        params = {}
    elif isinstance(indicators, str):
        indicator_list = [indicators]
        params = {}
    elif isinstance(indicators, list):
        indicator_list = indicators
        params = {}
    elif isinstance(indicators, dict):
        indicator_list = list(indicators.keys())
        params = indicators
    else:
        raise ValueError("Invalid indicators input")
    
    # Create config dictionary
    config = {"indicators": {}}
    for name in indicator_list:
        if name in TALIB_INDICATORS:
            config["indicators"][name.lower()] = params.get(name, {})
    
    for name in indicator_list:
        if name not in TALIB_INDICATORS:
            logger.warning(f"Indicator '{name}' not found in TA-Lib registry")
            continue
        
        indicator_def = TALIB_INDICATORS[name]
        
        # Prepare parameters for this indicator
        indicator_params = params.get(name, {})
        
        # Merge with defaults to get full parameter set for column naming
        final_params = {}
        for param_name, param_spec in indicator_def.get('params', {}).items():
            final_params[param_name] = param_spec.get('default')
        final_params.update(indicator_params)
        
        try:
            # Get TA-Lib function
            func = getattr(talib, name)
            
            # Update config with final params actually used
            config["indicators"][name.lower()] = final_params
            
            # Prepare inputs - pass numpy arrays directly
            inputs = []
            for input_type in indicator_def.get("inputs", []):
                if input_type.lower() in data.columns:
                    inputs.append(data[input_type.lower()].astype(np.float64).values)
                else:
                    inputs.append(data['close'].astype(np.float64).values)
            
            # Calculate indicator values
            if indicator_params:
                logger.debug(f"Calculating {name} with params: {indicator_params}")
                values = func(*inputs, **indicator_params)
            else:
                logger.debug(f"Calculating {name} with default params")
                values = func(*inputs)
            
            # Handle output based on number of outputs
            outputs = indicator_def.get("outputs", [name.lower()])
            
            # Shift values to avoid look-ahead bias
            if isinstance(values, tuple):
                for i, output_name in enumerate(outputs):
                    col_name = create_column_name(name, output_name, final_params)
                    shifted = pd.Series(values[i], index=data.index).shift(1)
                    data[col_name] = shifted.values
            else:
                col_name = create_column_name(name, outputs[0], final_params)
                shifted = pd.Series(values, index=data.index).shift(1)
                data[col_name] = shifted.values
                    
        except Exception as e:
            logger.error(f"Indicator '{name}' calculation failed: {e}", exc_info=True)
            # Add NaN columns for failed indicator
            # Here we use final_params which is now defined before the try block
            outputs = indicator_def.get("outputs", [name.lower()])
            for output_name in outputs:
                col_name = create_column_name(name, output_name, final_params)
                data[col_name] = np.nan
    
    # Clean up results - handle sparse indicators and optionally drop NaN
    indicator_cols = [col for col in data.columns if col not in OHLCV_COLUMNS]
    if drop_nan:        
        if indicator_cols:        
            # Drop rows with ANY NaN in indicator columns
            data = data.dropna(subset=indicator_cols)
    else:
        # Fill ALL NaNs in indicator columns with 0
        if indicator_cols:
            data[indicator_cols] = data[indicator_cols].fillna(0)
    
    # Return in requested format
    if return_type == "numpy":
        numpy_dict = {col: data[col].values for col in data.columns}
        return numpy_dict, config
    else:
        return data, config