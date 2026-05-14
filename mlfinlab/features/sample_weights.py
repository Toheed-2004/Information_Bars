"""
mlfinlab.features.sample_weights
==================================
Sample-weighting schemes for financial ML – AFML Chapter 4.

Standard IID assumptions are violated in financial data because
overlapping outcomes create serial correlation in labels.  Correcting
for this requires estimating sample *uniqueness* and applying appropriate
weights and time-decay.

Key functions
-------------
get_num_concurrent_events   Count simultaneous open events at each bar.
get_avg_uniqueness          Average fraction of non-overlapping bars.
get_ind_matrix              Indicator matrix linking events to bars.
seq_bootstrap               Draw non-redundant bootstrap samples.
get_sample_weights_return   Weight by return-attributed uniqueness.
get_sample_weights_time_decay Combine uniqueness with linear time decay.

References
----------
de Prado, M. L. (2018). *Advances in Financial Machine Learning*, Ch.4.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Concurrent events
# ---------------------------------------------------------------------------

def get_num_concurrent_events(
    close_idx: pd.DatetimeIndex,
    t1: pd.Series,
    molecule: Optional[list] = None,
) -> pd.Series:
    """Count how many events are simultaneously *open* at each bar.

    An event opened at *t0* is open at bar *t* iff ``t0 ≤ t ≤ t1[t0]``.

    Parameters
    ----------
    close_idx : pd.DatetimeIndex
        Full bar index of the price series.
    t1 : pd.Series
        Series mapping event start → event end (from get_events).
    molecule : list, optional
        Subset of t1's index to process (used by mp_pandas_obj).

    Returns
    -------
    pd.Series  Concurrency count for every bar in *close_idx*.
    """
    if molecule is not None:
        t1_ = t1.reindex(molecule).dropna()
    else:
        t1_ = t1.dropna()

    # bars that are active during at least one event
    idx = close_idx[
        (close_idx >= t1_.index.min()) & (close_idx <= t1_.max())
    ]
    count = pd.Series(0, index=idx)

    for t0, t_end in t1_.items():
        count.loc[t0:t_end] += 1

    return count


# ---------------------------------------------------------------------------
# Indicator matrix
# ---------------------------------------------------------------------------

def get_ind_matrix(
    bar_idx: pd.DatetimeIndex,
    t1: pd.Series,
) -> pd.DataFrame:
    """Build the binary indicator matrix linking events to bars.

    Entry (i, j) = 1 iff bar *i* falls within event *j*'s window.

    Parameters
    ----------
    bar_idx : pd.DatetimeIndex
        Bar timestamps.
    t1 : pd.Series
        Event-start → event-end mapping.

    Returns
    -------
    pd.DataFrame  Shape (n_bars, n_events).  Columns = event start timestamps.
    """
    ind_m = pd.DataFrame(0, index=bar_idx, columns=t1.index)
    for t0, t_end in t1.dropna().items():
        ind_m.loc[t0:t_end, t0] = 1
    return ind_m


# ---------------------------------------------------------------------------
# Average uniqueness
# ---------------------------------------------------------------------------

def get_avg_uniqueness(ind_m: pd.DataFrame) -> pd.Series:
    """Compute average uniqueness for each event.

    Uniqueness at bar *i* = 1 / (number of concurrent events at bar *i*).
    Average uniqueness of event *j* = mean uniqueness over its active bars.

    Parameters
    ----------
    ind_m : pd.DataFrame  Indicator matrix from :func:`get_ind_matrix`.

    Returns
    -------
    pd.Series  Average uniqueness per event.
    """
    conc = ind_m.sum(axis=1)          # concurrent events at each bar
    # uniqueness at each bar for each event
    uniq = ind_m.div(conc, axis=0)    # 0 where event not active
    avg_uniq = uniq[uniq > 0].mean()  # mean over active bars only
    return avg_uniq.fillna(0)


# ---------------------------------------------------------------------------
# Sequential Bootstrap
# ---------------------------------------------------------------------------

def seq_bootstrap(
    ind_m: pd.DataFrame,
    sample_length: Optional[int] = None,
    random_state: Optional[int] = None,
) -> list:
    """Draw a bootstrap sample that maximises average uniqueness.

    At each draw, compute the *marginal uniqueness* of each candidate
    event (given already-selected events) and sample with probability
    proportional to uniqueness.

    Parameters
    ----------
    ind_m : pd.DataFrame  Indicator matrix.
    sample_length : int, optional  Defaults to number of events.
    random_state : int, optional  Seed for reproducibility.

    Returns
    -------
    list  Indices (column names) of the bootstrap sample.
    """
    if sample_length is None:
        sample_length = ind_m.shape[1]

    rng = np.random.default_rng(random_state)
    phi: list = []          # indices (int positions) of chosen columns
    col_list = list(ind_m.columns)

    while len(phi) < sample_length:
        avg_u = pd.Series(dtype=float)
        for i, col in enumerate(col_list):
            ind_m_ = ind_m[[col_list[j] for j in phi] + [col]]
            avg_u[i] = get_avg_uniqueness(ind_m_).iloc[-1]

        prob = avg_u / avg_u.sum()
        chosen_pos = int(rng.choice(len(col_list), p=prob.values))
        phi.append(chosen_pos)

    return [col_list[i] for i in phi]


# ---------------------------------------------------------------------------
# Sample weights by return × uniqueness
# ---------------------------------------------------------------------------

def get_sample_weights_return(
    t1: pd.Series,
    close: pd.Series,
    num_threads: int = 1,
) -> pd.Series:
    """Weight each event proportionally to |log-return| × avg uniqueness.

    Gives more weight to large, unique events.

    Parameters
    ----------
    t1 : pd.Series  Event start → event end.
    close : pd.Series  Price series.
    num_threads : int  (unused, kept for API consistency)

    Returns
    -------
    pd.Series  Normalised sample weights (sum = n_samples).
    """
    bar_idx = close.index
    ind_m = get_ind_matrix(bar_idx, t1)
    avg_u = get_avg_uniqueness(ind_m)

    # absolute log-return of each event
    ret = pd.Series(dtype=float)
    for t0, t_end in t1.dropna().items():
        if t_end in close.index:
            ret[t0] = abs(np.log(close.loc[t_end] / close.loc[t0]))
        else:
            ret[t0] = np.nan

    w = avg_u * ret.reindex(avg_u.index).fillna(0)
    w = w / w.mean()    # normalise so mean weight = 1
    return w.rename("weight")


# ---------------------------------------------------------------------------
# Sample weights with time-decay
# ---------------------------------------------------------------------------

def get_sample_weights_time_decay(
    t1: pd.Series,
    close: pd.Series,
    decay: float = 1.0,
) -> pd.Series:
    """Combine return-uniqueness weights with a linear time-decay factor.

    Older observations receive less weight.

    Parameters
    ----------
    t1 : pd.Series  Event start → event end.
    close : pd.Series  Price series.
    decay : float
        Slope of the linear decay.  ``decay=1`` → oldest obs gets weight 0;
        ``decay=0`` → no decay; ``decay<0`` → older obs get *more* weight.
        Values are clipped to [0, 1] per event.

    Returns
    -------
    pd.Series  Normalised time-decayed sample weights.
    """
    w = get_sample_weights_return(t1, close)

    # sort by event start time and apply decay
    w_sorted = w.sort_index()
    n = len(w_sorted)
    if n < 2:
        return w_sorted

    # linear decay factor: c_i = 1 - decay * (1 - (i+1)/n)
    factors = np.array([max(0.0, 1 - decay * (1 - (i + 1) / n)) for i in range(n)])
    w_decayed = pd.Series(
        w_sorted.values * factors,
        index=w_sorted.index,
        name="weight_decay",
    )
    w_decayed = w_decayed / w_decayed.mean()
    return w_decayed.reindex(w.index)
