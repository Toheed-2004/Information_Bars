"""
Utility functions for macro economic data processing.

This module provides helper functions for resampling time series data
and other data transformation operations.
"""
import numpy as np
import pandas as pd

from bitpredict.common.logging import get_logger

logger = get_logger("data.macro.utils")


def resample_to_frequency(df, freq='1H'):
    """
    Resample DataFrame to the desired frequency using timestamp column.
    
    Converts unix timestamps to datetime, resamples to the specified frequency
    using forward fill, then converts back to unix timestamps.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'timestamp' column with
            unix timestamps (seconds since epoch).
        freq (str, optional): Pandas frequency string for resampling.
            Examples: '1H' (hourly), '1D' (daily), '15T' (15 minutes).
            Defaults to '1H'.
    
    Returns:
        pd.DataFrame: Resampled DataFrame with timestamp column and all
            other columns forward-filled to the new frequency.
    
    Raises:
        KeyError: If 'timestamp' column is not present in the DataFrame.
        ValueError: If the frequency string is invalid.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': [1609459200, 1609545600],
        ...     'value': [100, 200]
        ... })
        >>> resampled = resample_to_frequency(df, freq='12H')
    """
    try:
        logger.debug(
            f"Resampling DataFrame with {len(df)} rows to frequency: {freq}"
        )
        
        # Convert timestamp to datetime for resampling
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df.set_index('datetime', inplace=True)
        
        # Resample and forward fill
        resampled_df = df.resample(freq).ffill()
        resampled_df = resampled_df.reset_index()
        
        # Convert back to unix timestamp
        resampled_df['timestamp'] = (
            resampled_df['datetime'].astype(np.int64) // 10**9
        ).astype(int)
        
        # Drop datetime column and keep only timestamp
        resampled_df = resampled_df.drop(columns=['datetime'])
        
        logger.debug(
            f"Resampling complete. Output has {len(resampled_df)} rows"
        )
        
        return resampled_df
        
    except KeyError as e:
        logger.error(f"Missing required column in DataFrame: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid frequency string '{freq}': {e}")
        raise
    except Exception as e:
        logger.error(f"Error during resampling: {e}")
        raise