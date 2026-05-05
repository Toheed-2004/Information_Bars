"""
ml_module/backtest_bridge/signal_exporter.py
----------------------------------------------
Converts ML predictions into the signal format expected by VBTBacktestOptimized.

The backtest module expects a DataFrame with:
  - A ``datetime`` column (timezone-aware, matching the OHLCV index).
  - One or more signal columns containing values in {-1, 0, 1}.

This module:
  1. Aligns predictions with bar datetimes.
  2. Filters out HOLD (0) signals according to the configured signal map.
  3. Exports a CSV ready for direct use in the backtest pipeline.
  4. Produces a performance report linking ML metrics to backtest inputs.

Usage
-----
    from ml_module.backtest_bridge.signal_exporter import SignalExporter
    exporter = SignalExporter(cfg["backtest"])
    df_signals = exporter.export(predictions, bar_df, output_path="outputs/signals.csv")
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ml_module.utils.helpers import get_logger

logger = get_logger(__name__)


class SignalExporter:
    """
    Converts integer label predictions into a backtest-compatible signal DataFrame.

    Parameters
    ----------
    signal_col : Column name for signals in the output CSV (default: "signals").
    signal_map : Optional mapping from label → signal value.
                 Default: {1: 1, -1: -1, 0: 0}.
    output_dir : Directory to save exported CSV files.
    """

    def __init__(
        self,
        signal_col: str = "signals",
        signal_map: Optional[Dict] = None,
        output_dir: str | Path = "outputs",
    ):
        self.signal_col = signal_col
        self.signal_map = signal_map or {1: 1, -1: -1, 0: 0}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def export(
        self,
        predictions:  pd.Series,
        bar_df:       pd.DataFrame,
        output_path:  Optional[str | Path] = None,
        bar_type:     str = "unknown",
        asset:        str = "unknown",
    ) -> pd.DataFrame:
        """
        Build and optionally save the signal DataFrame.

        Parameters
        ----------
        predictions  : Series of int labels {-1, 0, 1} with a DatetimeIndex
                       (or integer index alignable to bar_df).
        bar_df       : Original bar DataFrame with DatetimeIndex (used to
                       recover the ``datetime`` column for VBT alignment).
        output_path  : If provided, save the DataFrame as CSV here.
        bar_type     : Metadata tag (included in filename if output_path is None).
        asset        : Asset identifier (metadata only).

        Returns
        -------
        pd.DataFrame with columns [``datetime``, ``signals``].
        """
        df_signals = self._build_signal_df(predictions, bar_df)

        if output_path is None:
            output_path = self.output_dir / f"signals_{asset}_{bar_type}.csv"

        output_path = Path(output_path)
        df_signals.to_csv(output_path, index=False)
        logger.info(
            "Exported %d signals → %s  (BUY=%d, SELL=%d, HOLD=%d)",
            len(df_signals),
            output_path.name,
            (df_signals[self.signal_col] == 1).sum(),
            (df_signals[self.signal_col] == -1).sum(),
            (df_signals[self.signal_col] == 0).sum(),
        )
        return df_signals

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_signal_df(
        self,
        predictions: pd.Series,
        bar_df:      pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align predictions to bar timestamps and apply the signal map.
        """
        # Recover datetime from bar_df index or column
        if isinstance(bar_df.index, pd.DatetimeIndex):
            datetimes = bar_df.index
        elif "datetime" in bar_df.columns:
            datetimes = pd.to_datetime(bar_df["datetime"], utc=True)
        else:
            raise ValueError("bar_df must have a DatetimeIndex or a 'datetime' column.")

        # Align predictions to bar_df length
        if len(predictions) == len(datetimes):
            preds_aligned = predictions.values
        else:
            # Reindex by position — predictions may be a subset (walk-forward)
            preds_aligned = np.zeros(len(datetimes), dtype=np.int8)
            pred_arr      = predictions.values

            if isinstance(predictions.index, pd.DatetimeIndex):
                # Align by datetime
                idx_map = {dt: i for i, dt in enumerate(datetimes)}
                for dt, val in zip(predictions.index, pred_arr):
                    pos = idx_map.get(dt)
                    if pos is not None and not np.isnan(val):
                        preds_aligned[pos] = int(val)
            else:
                # Align by integer position
                for pos, val in zip(predictions.index, pred_arr):
                    if 0 <= pos < len(preds_aligned) and not np.isnan(val):
                        preds_aligned[pos] = int(val)

        # Apply signal map
        mapped = np.vectorize(
            lambda x: self.signal_map.get(int(x), 0) if not np.isnan(x) else 0
        )(preds_aligned)

        df_out = pd.DataFrame({
            "datetime":       datetimes,
            self.signal_col:  mapped.astype(np.int8),
        })
        return df_out

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def signal_summary(self, df_signals: pd.DataFrame) -> Dict:
        """
        Return a summary dict suitable for inclusion in a research report.
        """
        s = df_signals[self.signal_col]
        total = len(s)
        return {
            "total_bars":   total,
            "buy_signals":  int((s == 1).sum()),
            "sell_signals": int((s == -1).sum()),
            "hold_signals": int((s == 0).sum()),
            "buy_pct":      round((s == 1).mean() * 100, 2),
            "sell_pct":     round((s == -1).mean() * 100, 2),
            "hold_pct":     round((s == 0).mean() * 100, 2),
            "datetime_start": str(df_signals["datetime"].iloc[0]),
            "datetime_end":   str(df_signals["datetime"].iloc[-1]),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_exporter(cfg: Dict) -> SignalExporter:
    """Instantiate from the ``backtest`` config section."""
    raw_map = cfg.get("signal_map", {})
    # YAML keys are sometimes strings; normalise to int
    signal_map = {}
    for k, v in raw_map.items():
        try:
            signal_map[int(k)] = int(v)
        except (ValueError, TypeError):
            pass
    if not signal_map:
        signal_map = {1: 1, -1: -1, 0: 0}

    return SignalExporter(
        signal_col = cfg.get("signal_col", "signals"),
        signal_map = signal_map,
        output_dir = cfg.get("output_dir", "outputs"),
    )
