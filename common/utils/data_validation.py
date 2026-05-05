"""
Utility functions for data validation and normalisation.
"""

import pandas as pd
from bitpredict.common.constants import OHLCV_COLUMNS


def validate_ohlcv(data: pd.DataFrame, ohlc_only: bool = True) -> pd.DataFrame:
    # normalize column names (in-place)
    data.columns = [str(col).lower() for col in data.columns]

    # validation always happens
    missing = [c for c in OHLCV_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # only slice when explicitly requested
    if ohlc_only:
        return data[OHLCV_COLUMNS]

    # otherwise: validation only
    return data


def ensure_datetime_index(df: pd.DataFrame, sort: bool = True) -> pd.DataFrame:
    """
    Ensure the DataFrame has a DatetimeIndex (UTC).

    Accepted input formats:
      1. df already has a DatetimeIndex  → returned as-is (sorted if sort=True)
      2. df has a 'datetime' column      → column is converted, set as index, column dropped
      3. anything else                   → ValueError

    Always returns a copy; the original is never modified.
    """
    df = df.copy()

    if isinstance(df.index, pd.DatetimeIndex):
        if sort:
            df = df.sort_index()
        return df

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime")
        if sort:
            df = df.sort_index()
        return df

    raise ValueError(
        "DataFrame must have either a DatetimeIndex or a 'datetime' column. "
        "Call ensure_datetime_index() before passing data to the feature pipeline."
    )


def ensure_datetime_column(df: pd.DataFrame, sort: bool = True) -> pd.DataFrame:
    """
    Ensure the DataFrame has a 'datetime' column (UTC) and a RangeIndex.

    Accepted input formats:
      1. df has a 'datetime' column      → converted to UTC, index reset
      2. df has a DatetimeIndex          → index moved to 'datetime' column
      3. anything else                   → ValueError

    Always returns a copy; the original is never modified.
    """
    # df = df.copy()

    # Case 1: 'datetime' is already a column
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        if sort:
            df = df.sort_values("datetime")
        return df.reset_index(drop=True)

    # Case 2: Index is a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        # If the index is already UTC, this is a no-op; otherwise, it converts
        df.index = pd.to_datetime(df.index, utc=True)
        if sort:
            df = df.sort_index()
        
        # Move index to column. We name it specifically to 'datetime'
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "datetime"}) 
        
        # Ensure only the 'datetime' column is the time carrier
        return df.reset_index(drop=True)

    raise ValueError(
        "DataFrame must have either a 'datetime' column or a DatetimeIndex. "
        "Ensure data is pre-processed before passing to the feature pipeline."
    )

