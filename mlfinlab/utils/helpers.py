"""
mlfinlab.utils.helpers
======================
Shared low-level utilities used by labeling and feature modules.

References
----------
de Prado, M. L. (2018). *Advances in Financial Machine Learning*.
Wiley.  Chapters 2, 17.
"""
from __future__ import annotations

import multiprocessing as mp
import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Log-returns
# ---------------------------------------------------------------------------

def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Compute log-returns for a price series.

    Parameters
    ----------
    close : pd.Series
        Price series indexed by datetime.
    periods : int
        Look-back periods.

    Returns
    -------
    pd.Series  (same index, NaN at head)
    """
    return np.log(close / close.shift(periods))


# ---------------------------------------------------------------------------
# Daily volatility estimate  (AFML Ch.3 snippet 3.1)
# ---------------------------------------------------------------------------

def daily_vol(
    close: pd.Series,
    lookback: int = 100,
    span: Optional[int] = None,
) -> pd.Series:
    """Exponentially-weighted daily volatility estimate.

    Computes the standard deviation of log-returns over a rolling window,
    anchored to the daily frequency of the bar series.

    Parameters
    ----------
    close : pd.Series
        Close price series (any bar type).
    lookback : int
        Number of bars used in the EWM span.  Ignored when *span* is given.
    span : int, optional
        EWM span.  When provided, overrides *lookback*.

    Returns
    -------
    pd.Series  Daily volatility, same index as *close*.
    """
    ret = log_returns(close).dropna()
    ewm_span = span if span is not None else lookback
    vol = ret.ewm(span=ewm_span).std()
    return vol.reindex(close.index).ffill()


# ---------------------------------------------------------------------------
# Symmetric CUSUM filter  (AFML Ch.2 snippet 2.4)
# ---------------------------------------------------------------------------

def cusum_filter(
    close: pd.Series,
    threshold: Union[float, pd.Series],
) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter for event sampling.

    Triggers a new event whenever the cumulative positive or negative
    deviation from the running reference crosses *threshold*.

    Parameters
    ----------
    close : pd.Series
        Price (or any signal) series indexed by datetime.
    threshold : float | pd.Series
        Fixed scalar threshold, or a time-aligned series of thresholds
        (e.g. the daily_vol series scaled by some multiple).

    Returns
    -------
    pd.DatetimeIndex  Timestamps of sampled events.
    """
    if isinstance(threshold, pd.Series):
        threshold = threshold.reindex(close.index).ffill().fillna(threshold.median())
    else:
        threshold = pd.Series(threshold, index=close.index)

    t_events: list = []
    s_pos = s_neg = 0.0
    ret = log_returns(close).dropna()

    for dt, r in ret.items():
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        h = float(threshold.loc[dt])
        if s_neg < -h:
            s_neg = 0.0
            t_events.append(dt)
        elif s_pos > h:
            s_pos = 0.0
            t_events.append(dt)

    return pd.DatetimeIndex(t_events)


# ---------------------------------------------------------------------------
# Vertical barriers  (AFML Ch.3 snippet 3.4)
# ---------------------------------------------------------------------------

def get_vertical_barriers(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    num_days: float = 1.0,
) -> pd.Series:
    """Assign a vertical (time) barrier to each event.

    Parameters
    ----------
    close : pd.Series
        Price series – used only for its DatetimeIndex.
    t_events : pd.DatetimeIndex
        Event timestamps (e.g. from CUSUM filter).
    num_days : float
        Maximum holding period in *calendar* days.

    Returns
    -------
    pd.Series  Index = t_events, values = barrier timestamp (NaT if
               beyond last bar).
    """
    dt_index = close.index
    barriers: list = []
    delta = pd.Timedelta(days=num_days)
    for t in t_events:
        future = dt_index[dt_index >= t + delta]
        barriers.append(future[0] if len(future) else pd.NaT)
    return pd.Series(barriers, index=t_events)


# ---------------------------------------------------------------------------
# Multiprocessing helper  (AFML Ch.20 snippet 20.7)
# ---------------------------------------------------------------------------

def _mp_worker(
    func: Callable,
    df: pd.DataFrame,
    molecule: list,
    kwargs: dict,
    out_q: mp.Queue,
) -> None:
    """Internal worker executed in a subprocess."""
    try:
        result = func(df=df, molecule=molecule, **kwargs)
        out_q.put(result)
    except Exception as exc:  # noqa: BLE001
        out_q.put(exc)


def mp_pandas_obj(
    func: Callable,
    pd_obj: Union[pd.Series, pd.DataFrame],
    num_threads: int = 1,
    **kwargs,
) -> Union[pd.Series, pd.DataFrame]:
    """Apply *func* to *pd_obj* using multiprocessing.

    Splits the index of *pd_obj* into *num_threads* roughly equal chunks
    and calls ``func(df=pd_obj, molecule=chunk, **kwargs)`` in each worker.
    Results are concatenated and sorted by index.

    Parameters
    ----------
    func : callable
        Function with signature ``func(df, molecule, **kwargs) -> pd.Series``.
    pd_obj : pd.Series | pd.DataFrame
        Object whose index is partitioned.
    num_threads : int
        Number of parallel workers (capped at CPU count).
    **kwargs
        Forwarded to *func*.

    Returns
    -------
    pd.Series | pd.DataFrame  Concatenated results.
    """
    if num_threads < 2:
        # serial – avoids fork overhead for small datasets
        idx = list(pd_obj.index)
        return func(df=pd_obj, molecule=idx, **kwargs)

    num_threads = min(num_threads, mp.cpu_count(), len(pd_obj))
    parts = np.array_split(pd_obj.index, num_threads)
    out_q: mp.Queue = mp.Queue()
    jobs = []
    for part in parts:
        p = mp.Process(
            target=_mp_worker,
            args=(func, pd_obj, list(part), kwargs, out_q),
        )
        jobs.append(p)
        p.start()

    results = []
    for _ in jobs:
        res = out_q.get()
        if isinstance(res, Exception):
            raise res
        results.append(res)

    for p in jobs:
        p.join()

    out = pd.concat(results).sort_index()
    return out
