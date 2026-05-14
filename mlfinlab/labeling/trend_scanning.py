"""
mlfinlab.labeling.trend_scanning
==================================
Trend-scanning labels – de Prado & Lewis (2019).

For each event timestamp *t*, fits OLS regressions over look-ahead
windows of lengths ``L = [l_min, …, l_max]`` bars and selects the
window that maximises the absolute *t-statistic* of the slope.

Returns ``+1`` (up-trend), ``-1`` (down-trend), or ``0`` (no significant
trend), together with the signed t-statistic used as a confidence proxy.

This provides a model-free alternative to triple-barrier labeling and is
particularly useful when the user cannot pre-specify a good pt/sl ratio.

Reference
---------
de Prado, M. L. & Lewis, M. J. (2019).  "Detection of False Investment
Strategies Using Unsupervised Learning Methods."  *Quantitative Finance*.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


def trend_scanning_labels(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    look_forward_window: Union[int, Sequence[int]] = 20,
    min_sample_length: int = 5,
    step: int = 1,
    t_value_threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute trend-scanning labels for each event in *t_events*.

    Parameters
    ----------
    close : pd.Series
        Bar close prices (log-prices are computed internally).
    t_events : pd.DatetimeIndex
        Event timestamps to label.
    look_forward_window : int | list[int]
        Maximum look-ahead in *bars*.  When a sequence is supplied it is
        used as-is; when an int *L* is supplied, windows range over
        ``range(min_sample_length, L + 1, step)``.
    min_sample_length : int
        Minimum bars required to fit a regression.
    step : int
        Step size when iterating over window lengths.
    t_value_threshold : float
        Absolute t-statistic threshold.  Events below the threshold are
        assigned label ``0``.

    Returns
    -------
    pd.DataFrame  columns:
        * ``t_value``  – signed t-stat of slope at best window
        * ``bin``      – +1 / -1 / 0
        * ``end_time`` – timestamp of the best look-ahead window end
        * ``window``   – window length (bars) chosen
    """
    log_p = np.log(close)

    if isinstance(look_forward_window, int):
        windows = list(range(min_sample_length, look_forward_window + 1, step))
    else:
        windows = sorted(look_forward_window)

    records = []
    bar_index = log_p.index

    for t in t_events:
        if t not in bar_index:
            continue
        t_pos = bar_index.get_loc(t)

        best_t_val = 0.0
        best_window = np.nan
        best_end = pd.NaT

        for w in windows:
            end_pos = t_pos + w
            if end_pos >= len(bar_index):
                break
            slice_ = log_p.iloc[t_pos : end_pos + 1]
            if len(slice_) < min_sample_length:
                continue

            x = np.arange(len(slice_), dtype=float)
            y = slice_.values

            # OLS: y = a + b*x
            x_bar, y_bar = x.mean(), y.mean()
            ss_xx = ((x - x_bar) ** 2).sum()
            if ss_xx == 0:
                continue
            b = ((x - x_bar) * (y - y_bar)).sum() / ss_xx
            a = y_bar - b * x_bar

            resid = y - (a + b * x)
            n = len(x)
            if n < 3:
                continue
            se_b = np.sqrt((resid ** 2).sum() / (n - 2) / ss_xx)
            if se_b == 0:
                continue
            t_val = b / se_b

            if abs(t_val) > abs(best_t_val):
                best_t_val = t_val
                best_window = w
                best_end = bar_index[end_pos]

        if abs(best_t_val) >= t_value_threshold:
            label = int(np.sign(best_t_val))
        else:
            label = 0

        records.append(
            {
                "t_value": best_t_val,
                "bin": label,
                "end_time": best_end,
                "window": best_window,
            }
        )

    df = pd.DataFrame(records, index=t_events[: len(records)])
    return df
