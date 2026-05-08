"""
bridge/signal_exporter.py
--------------------------
Exports walk-forward predictions as a VBT-compatible signal CSV.

Signal values: +1 = BUY, -1 = SELL, 0 = HOLD
NaN predictions (bars before first test fold) → 0 (HOLD).

For tick bars: timestamps are rounded to minute precision so they
can be aligned with 1-minute OHLCV data in VBT backtests.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def export_signals(
    predictions: pd.Series,
    bar_df: pd.DataFrame,
    asset: str,
    bar_type: str,
    output_dir: Path,
    is_tick_source: bool = False,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    predictions   : Series of {-1.0, 0.0, 1.0, NaN} from walk-forward.
    bar_df        : Original bar DataFrame (for datetime index alignment).
    asset         : e.g. 'btc'
    bar_type      : e.g. 'dollar_tick'
    output_dir    : Where to write signal CSV.
    is_tick_source: If True, round timestamps to minute for 1m OHLCV alignment.

    Returns
    -------
    pd.DataFrame with columns [datetime, signals].
    """
    sig = predictions.fillna(0).astype(int)
    df_out = pd.DataFrame({
        "datetime": sig.index,
        "signals":  sig.values,
    })

    # Remove NaT rows
    nat_mask = df_out["datetime"].isna()
    if nat_mask.any():
        logger.info("%s: removed %d NaT rows", bar_type, nat_mask.sum())
    df_out = df_out[~nat_mask].copy()

    # For tick bars, round to minute so VBT can align with 1m OHLCV
    if is_tick_source:
        df_out["datetime"] = df_out["datetime"].dt.floor("min")
        before = len(df_out)
        df_out = df_out.drop_duplicates(subset=["datetime"], keep="last")
        collapsed = before - len(df_out)
        if collapsed:
            logger.info("%s: %d duplicates collapsed after rounding", bar_type, collapsed)

    path = output_dir / f"signals_{asset}_{bar_type}.csv"
    df_out.to_csv(path, index=False)

    buy  = (df_out["signals"] ==  1).sum()
    sell = (df_out["signals"] == -1).sum()
    hold = (df_out["signals"] ==  0).sum()
    logger.info("Signals → %s  (BUY=%d SELL=%d HOLD=%d)", path.name, buy, sell, hold)

    return df_out
