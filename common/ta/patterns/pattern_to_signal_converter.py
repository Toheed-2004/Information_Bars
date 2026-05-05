import pandas as pd
import numpy as np
from typing import Union, Dict

try:
    from bitpredict.common.constants import PATTERN_TALIB_COL_PREFIX
except ImportError:
    PATTERN_TALIB_COL_PREFIX = "talib_CDL"


def signals_from_patterns(
    result: Union[pd.DataFrame, Dict[str, np.ndarray]],
    return_type: str = None,
    keep_original: bool = False
) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
    """Convert candlestick pattern values to normalized signal strength indicators.
    
    Parameters:
    -----------
    result : DataFrame or dict
        Input data with pattern columns (prefix: PATTERN_TALIB_COL_PREFIX)
    return_type : str, optional
        Output format. If None, matches input format ('dataframe' or 'numpy_array').
    keep_original : bool, default False
        If True, keep original pattern columns alongside normalized signals.
    
    Returns:
    --------
    DataFrame or dict with normalized signal columns (lowercase names).
    """
    
    # Detect input type if return_type not specified
    is_input_dict = isinstance(result, dict)
    if return_type is None:
        return_type = "numpy_array" if is_input_dict else "dataframe"
    
    # Convert dict to DataFrame if needed
    df = pd.DataFrame(result) if is_input_dict else result
    
    
    # Get pattern columns using constant
    pattern_cols = [
    col for col in df.columns
    if col.lower().startswith(PATTERN_TALIB_COL_PREFIX.lower())]

    # CORE LOGIC:
    # For each candlestick pattern column:
    # 1. Extract unique values (e.g., [-100, -80, 0, 80, 100])
    # 2. Get unique absolute values and sort them (e.g., [0, 80, 100])
    # 3. Map each absolute value to a strength index (0->0, 80->1, 100->2)
    # 4. Apply sign: positive values keep positive signal, negative keep negative signal
    #    Result: {0:0, 80:1, 100:2, -80:-1, -100:-2}
    # 5. Convert raw pattern values to normalized signal strengths
    
    def create_mapping(col):
        unique_vals = sorted(set(df[col].dropna().unique()))
        if not unique_vals:
            signal_col = col.replace(PATTERN_TALIB_COL_PREFIX, 'signal_').lower()
        
        # Map absolute values to strength indices (0, 1, 2, ...)
        abs_vals = sorted(set(abs(v) for v in unique_vals))
        abs_mapping = {abs_vals[i]: i for i in range(len(abs_vals))}
        
        # Preserve sign: apply strength with appropriate sign
        mapping = {v: (abs_mapping[abs(v)] if v >= 0 else -abs_mapping[abs(v)]) 
                   for v in unique_vals}
        
        signal_col = col.replace(PATTERN_TALIB_COL_PREFIX, 'signal_').lower()
        return signal_col, df[col].map(mapping)
    
    # Apply mapping to all columns
    for col in pattern_cols:
        signal_col, signal_data = create_mapping(col)
        if signal_data is not None:
            df[signal_col] = signal_data
    
    # Remove original pattern columns if not keeping them
    if not keep_original:
        df = df.drop(columns=pattern_cols)
    
    # Return based on type
    if return_type.lower() == "numpy_array":
        return {col: df[col].values for col in df.columns}
    else:
        return df


# Test

"""
if __name__ == "__main__":
    # For testing - define the prefix if not available
    TEST_PREFIX = "talib_CDL"
    
    # Create dummy data
    dummy_df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=5),
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        f'{TEST_PREFIX}2CROWS': [-100, 0, -100, 0, 0],
        f'{TEST_PREFIX}ENGULFING': [-100, -80, 0, 80, 100],
        f'{TEST_PREFIX}DOJI': [0, 100, 0, 100, 100],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    print("=" * 70)
    print("ORIGINAL DATA")
    print("=" * 70)
    print(dummy_df)
    print("\n")
    
    # Test 1: Default (no original)
    print("=" * 70)
    print("TEST 1: keep_original=False (default)")
    print("=" * 70)
    result1 = signals_from_patterns(dummy_df)
    print(result1)
    print(f"Columns: {result1.columns.tolist()}\n")
    
    # Test 2: Keep original values
    print("=" * 70)
    print("TEST 2: keep_original=True")
    print("=" * 70)
    result2 = signals_from_patterns(dummy_df, keep_original=True)
    print(result2)
    print(f"Columns: {result2.columns.tolist()}\n")
    
    # Test 3: Convert to numpy array output
    print("=" * 70)
    print("TEST 3: return_type='numpy_array'")
    print("=" * 70)
    result3 = signals_from_patterns(dummy_df, return_type="numpy_array")
    print(f"Type: {type(result3)}")
    print(f"Keys: {list(result3.keys())}")
    print(f"\nSignal values:")
    for key in result3:
        if key.startswith('signal_'):
            print(f"  {key}: {result3[key]}\n")
    
    # Test 4: Dict input
    print("=" * 70)
    print("TEST 4: Dict input (auto returns dict)")
    print("=" * 70)
    dummy_dict = {col: dummy_df[col].values for col in dummy_df.columns}
    result4 = signals_from_patterns(dummy_dict)
    print(f"Type: {type(result4)}")
    print(f"Keys: {list(result4.keys())}\n")
    for key in result4:
        if key.startswith('signal_'):
            print(f"{key}: {result4[key]}")
"""