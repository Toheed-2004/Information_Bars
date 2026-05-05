"""
VectorBT Pro Indicator Calculator
"""

import numpy as np
import pandas as pd
import warnings
import vectorbtpro as vbt

from typing import Union, List, Dict, Tuple
from bitpredict.common.constants import OHLCV_COLUMNS
from bitpredict.common.ta.indicators.vectorbtpro.registry import VBT_INDICATORS, create_column_name


def calculate(
    data: pd.DataFrame,
    indicators: Union[str, List[str], Dict[str, Dict]] = "all",
    return_type: str = "dataframe",
    drop_nan: bool = True
) -> Tuple[Union[pd.DataFrame, Dict[str, np.ndarray]], Dict]:
    """
    Calculate VectorBT Pro indicators
    
    Args:
        data: OHLCV DataFrame
        indicators: 'all', indicator name, list of names, or dict with params
        return_type: 'dataframe' or 'numpy' output
        drop_nan: if True (default), drop rows with NaN; if False, return raw data
        
    Returns:
        Tuple of (results, config)
    """   
    # Parse indicators input
    if indicators == "all":
        indicator_list = list(VBT_INDICATORS.keys())
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
        if name in VBT_INDICATORS:
            config["indicators"][name.lower()] = params.get(name, {})
    
    
    for name in indicator_list:
        if name not in VBT_INDICATORS:
            warnings.warn(f"Indicator '{name}' not found in VBT registry")
            continue
        
        indicator_def = VBT_INDICATORS[name]
        
        # Prepare parameters for this indicator
        indicator_params = params.get(name, {})
        
        # Merge with defaults to get full parameter set for column naming
        final_params = {}
        for param_name, param_spec in indicator_def.get('params', {}).items():
            final_params[param_name] = param_spec.get('default')
        final_params.update(indicator_params)
        
        try:
            # Get VBT indicator class
            indicator_class = getattr(vbt, indicator_def["func_name"])
            
            # Update config with final params actually used
            config["indicators"][name.lower()] = final_params
            
            # Prepare inputs - pass numpy arrays directly
            input_args = {}
            for input_name in indicator_def.get("inputs", []):
                input_lower = input_name.lower()
                if input_lower in data.columns:
                    input_args[input_name] = data[input_lower].values.astype(np.float64)
                else:
                    input_args[input_name] = data["close"].values.astype(np.float64)
            
            # Calculate indicator
            if indicator_params:
                result = indicator_class.run(**input_args, **indicator_params)
            else:
                result = indicator_class.run(**input_args)
            
            # Extract outputs
            outputs = indicator_def.get("outputs", [name.lower()])
            
            for output_name in outputs:
                if hasattr(result, output_name):
                    output_data = getattr(result, output_name)
                    # Create column name with correct argument order
                    col_name = create_column_name(name, output_name, final_params)
                    
                    # Get values and shift to avoid look-ahead bias
                    if hasattr(output_data, 'values'):
                        values = output_data.values
                    else:
                        values = output_data
                    # Use shift(1) to avoid look-ahead bias (same as TALib)
                    shifted = pd.Series(values, index=data.index).shift(1)
                    data[col_name] = shifted.values
                else:
                    warnings.warn(f"Output '{output_name}' not found for indicator '{name}'")
                    
        except Exception as e:
            warnings.warn(f"Indicator '{name}' calculation failed: {e}")
            import traceback
            traceback.print_exc()
            # Add NaN columns for failed indicator (matching TALib behavior)
            outputs = indicator_def.get("outputs", [name.lower()])
            for output_name in outputs:
                col_name = create_column_name(name, output_name, final_params)
                data[col_name] = np.nan
    
    # Clean up results - handle sparse indicators and optionally drop NaN
    indicator_cols = [col for col in data.columns if col not in OHLCV_COLUMNS]

    if drop_nan:        
        if indicator_cols:
        # Supertrend-style sparse signals
            sparse_keywords = ("_trend_", "_direction_", "_long_", "_short_")

            # Fill NaN with 0 ONLY for sparse signal columns
            sparse_cols = [
                col for col in indicator_cols
                if any(k in col for k in sparse_keywords)
            ]

            if sparse_cols:
                data[sparse_cols] = data[sparse_cols].fillna(0)

            # Drop rows that still have NaN in any indicator column
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