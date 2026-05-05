"""
Signal combination utilities.

Houses functions for combining multiple signal streams into a single
strategy signal. Import from here rather than duplicating in
simulator/main.py or strategies/ta/utils.py.
"""
import pandas as pd
from typing import List, Optional


def majority_rule(
    signals_df: pd.DataFrame, signal_cols: Optional[List[str]] = None
) -> pd.Series:
    """Apply majority-vote rule across signal columns.

    Rules:
    - Count occurrences of 1 and -1 per row.
    - If 1 has strictly highest count  -> 1 (long).
    - If -1 has strictly highest count -> -1 (short).
    - Otherwise -> 0 (neutral / tie).

    Args:
        signals_df: DataFrame with one or more signal columns (values -1/0/1).
        signal_cols: Explicit list of columns to use.  If None, every column
                     whose name ends with ``_signals`` is used.

    Returns:
        pd.Series of int (values -1, 0, 1) aligned to *signals_df.index*.
    """
    if signal_cols is None:
        signal_cols = [c for c in signals_df.columns if c.endswith("_signals")]

    if len(signal_cols) == 0:
        return pd.Series(0, index=signals_df.index, dtype=int)

    arr = signals_df[signal_cols].fillna(0).astype(int)
    ones = (arr == 1).sum(axis=1)
    negs = (arr == -1).sum(axis=1)

    result = pd.Series(0, index=arr.index, dtype=int)
    result[ones > negs] = 1
    result[negs > ones] = -1
    return result
