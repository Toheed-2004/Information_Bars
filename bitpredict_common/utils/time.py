from bitpredict.common.constants import TIME_UNITS
import pandas as pd
from typing import Union
import numpy as np

def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convert a time horizon string to its equivalent value in minutes.

    Args:
        timeframe (str): Time horizon string, e.g., '1m', '3h', '1d'.

    Returns:
        int: Equivalent value in minutes.

    Raises:
        ValueError: If the input format is invalid or unsupported.
    """
    if not isinstance(timeframe, str) or len(timeframe) < 2:
        raise ValueError(f"Invalid time horizon: {timeframe}")

    # Separate numeric part and unit
    numeric_part = timeframe[:-1]
    unit = timeframe[-1]

    try:
        value = int(numeric_part)
    except ValueError:
        raise ValueError(f"Invalid numeric value in time horizon: {timeframe}")

    # Normalize unit (optional, lowercase)
    if unit not in TIME_UNITS:
        raise ValueError(f"Unsupported time unit: {unit}")

    return value * TIME_UNITS[unit]


def timestamp_to_datetime(
    data: Union[pd.DataFrame, pd.Series, pd.Timestamp, str, int, float, list, np.ndarray],
    column: str = "timestamp"
) -> Union[pd.DataFrame, pd.Series, pd.Timestamp]:
    """
    Convert Unix timestamp(s) in milliseconds to timezone-aware UTC datetime.
    
    Works for:
    - DataFrame: converts specified column to datetime (returns a copy)
    - Series: converts series to datetime
    - Single value (int, float, str, pd.Timestamp)
    - List or numpy array
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, pd.Timestamp, str, int, float, list, np.ndarray]
        Input to convert.
    column : str
        Column name if input is a DataFrame (default: "timestamp").
        
    Returns
    -------
    Converted object in datetime:
    - DataFrame with column converted
    - Series
    - pd.Timestamp for single value
    
    Notes
    -----
    - Numeric values are interpreted as Unix timestamps in milliseconds
    - Non-numeric values are parsed using pd.to_datetime
    - All outputs are timezone-aware UTC
    """
    
    # -----------------------------
    # DataFrame case
    # -----------------------------
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        # Convert numeric columns from Unix ms
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = pd.to_datetime(data[column], unit="ms", utc=True)
        else:
            # Parse non-numeric columns
            data[column] = pd.to_datetime(data[column], utc=True)
            # Ensure UTC if timezone-naive
            if data[column].dt.tz is None:
                data[column] = data[column].dt.tz_localize("UTC")
        return data
    
    # -----------------------------
    # Series case
    # -----------------------------
    if isinstance(data, pd.Series):
        series_copy = data
        if pd.api.types.is_numeric_dtype(series_copy):
            return pd.to_datetime(series_copy, unit="ms", utc=True)
        else:
            # Parse non-numeric
            result = pd.to_datetime(series_copy, utc=True)
            # Ensure UTC
            if result.dt.tz is None:
                result = result.dt.tz_localize("UTC")
            return result
    
    # -----------------------------
    # Single scalar value or list/array
    # -----------------------------
    # Handle list or numpy array
    if isinstance(data, (list, np.ndarray)):
        if len(data) == 0:
            return pd.Series([], dtype='datetime64[ns, UTC]')
        
        # Convert to Series for consistent handling
        series = pd.Series(data)
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_datetime(series, unit="ms", utc=True)
        else:
            result = pd.to_datetime(series, utc=True)
            if result.dt.tz is None:
                result = result.dt.tz_localize("UTC")
            return result
    
    # Single value handling
    # If it's already a Timestamp, ensure UTC
    if isinstance(data, pd.Timestamp):
        if data.tz is None:
            return data.tz_localize("UTC")
        return data.tz_convert("UTC") if data.tz != "UTC" else data
    
    # If it's numeric, treat as Unix ms
    if isinstance(data, (int, float)):
        return pd.to_datetime(data, unit="ms", utc=True)
    
    # If it's a string or other type, parse it
    result = pd.to_datetime(data, utc=True)
    if result.tz is None:
        result = result.tz_localize("UTC")
    return result


# def datetime_to_timestamp(
#     data: Union[pd.DataFrame, pd.Series, pd.Timestamp, str, int, float],
#     column: str = "datetime"
# ) -> Union[pd.DataFrame, pd.Series, int]:
#     """
#     Convert input to Unix timestamp in milliseconds.

#     Works for:
#     - DataFrame: converts specified column to Unix ms (returns a copy)
#     - Series: converts series to Unix ms
#     - Single timestamp (str, pd.Timestamp, int/float)

#     Parameters
#     ----------
#     data : DataFrame, Series, Timestamp, str, int, float
#         Input to convert.
#     column : str
#         Column name if input is a DataFrame.

#     Returns
#     -------
#     Converted object in Unix milliseconds:
#     - DataFrame with column converted
#     - Series
#     - int for single timestamp
#     """

#     # -----------------------------
#     # DataFrame case
#     # -----------------------------
#     if isinstance(data, pd.DataFrame):
#         if column not in data.columns:
#             raise KeyError(f"Column '{column}' not found in DataFrame")
#         if not pd.api.types.is_numeric_dtype(data[column]):
#             data[column] = pd.to_datetime(data[column], utc=True).astype("int64") // 10**6
#         return data

#     # -----------------------------
#     # Series case
#     # -----------------------------
#     if isinstance(data, pd.Series):
#         if not pd.api.types.is_numeric_dtype(data):
#             return pd.to_datetime(data, utc=True).astype("int64") // 10**6
#         return data

#     # -----------------------------
#     # Single scalar timestamp
#     # -----------------------------
#     if isinstance(data, (int, float)):
#         return int(data)
#     if isinstance(data, pd.Timestamp):
#         return int(data.timestamp() * 1000)
#     return int(pd.to_datetime(data, utc=True).timestamp() * 1000)


def datetime_to_timestamp(
    data: Union[pd.DataFrame, pd.Series, pd.Timestamp, str, int, float],
    column: str = "datetime"
) -> Union[pd.DataFrame, pd.Series, int]:
    """
    Convert input to Unix timestamp in milliseconds.

    Works for:
    - DataFrame: converts specified column to Unix ms (returns a copy)
    - Series: converts series to Unix ms
    - Single timestamp (str, pd.Timestamp, int/float)

    Parameters
    ----------
    data : DataFrame, Series, Timestamp, str, int, float
        Input to convert.
    column : str
        Column name if input is a DataFrame.

    Returns
    -------
    Converted object in Unix milliseconds:
    - DataFrame with column converted
    - Series
    - int for single timestamp
    """

    def _series_to_ms(s: pd.Series) -> pd.Series:
        """Convert any Series to Unix milliseconds as int64."""
        if pd.api.types.is_numeric_dtype(s):
            sample = s.dropna().iloc[0] if not s.dropna().empty else 0
            if sample > 1e15:        # nanoseconds
                return (s // 10**6).astype("int64")
            elif sample > 1e12:      # microseconds
                return (s // 10**3).astype("int64")
            elif sample > 1e9:       # already milliseconds
                return s.astype("int64")
            else:                    # seconds
                return (s * 1_000).astype("int64")
        else:
            # datetime-like — convert via epoch to avoid resolution ambiguity
            dt = pd.to_datetime(s, utc=True)
            epoch = pd.Timestamp("1970-01-01", tz="UTC")
            return pd.Series(
                (dt - epoch).dt.total_seconds().mul(1_000).astype("int64"),
                index=s.index,
                name=s.name
            )

    def _scalar_to_ms(value) -> int:
        """Convert a single scalar value to Unix milliseconds."""
        if isinstance(value, (int, float)):
            if value > 1e15:         # nanoseconds
                return int(value // 10**6)
            elif value > 1e12:       # microseconds
                return int(value // 10**3)
            elif value > 1e9:        # already milliseconds
                return int(value)
            else:                    # seconds
                return int(value * 1_000)
        if isinstance(value, pd.Timestamp):
            # Always reliable: use .timestamp() which returns seconds as float
            return int(value.timestamp() * 1_000)
        # str or anything else pandas can parse
        return int(pd.to_datetime(value, utc=True).timestamp() * 1_000)

    # -----------------------------
    # DataFrame case
    # -----------------------------
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        data = data.copy()
        data[column] = _series_to_ms(data[column])
        return data

    # -----------------------------
    # Series case
    # -----------------------------
    if isinstance(data, pd.Series):
        return _series_to_ms(data)

    # -----------------------------
    # Single scalar timestamp
    # -----------------------------
    return _scalar_to_ms(data)





