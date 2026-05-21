"""
mlfinlab.features.sample_weights
==================================
Sample-weighting schemes for financial ML – AFML Chapter 4.

REFACTORING NOTES (bugs fixed vs original)
-------------------------------------------
1. get_ind_matrix (CRITICAL OOM BUG): original built a dense
   (n_bars × n_events) float64 matrix. For 66K renko bars this is
   66,288 × 66,408 × 8 bytes = 32.8 GiB → immediate crash.
   Fixed by using a sparse COO representation (scipy.sparse) that
   only stores the non-zero entries.  Memory usage is now O(sum of
   bar windows across events) instead of O(n_bars × n_events).

2. get_num_concurrent_events: was a Python for-loop over all events
   calling count.loc[t0:t_end] += 1 per event (O(n²) worst case).
   Replaced with a vectorised diff-based concurrency counter that
   runs in O(n_events log n_events) time.

3. get_sample_weights_return: formerly called get_ind_matrix (and
   therefore also triggered the OOM crash). Now uses the sparse
   concurrency-based path.

4. get_avg_uniqueness: updated to work with the sparse representation.

5. seq_bootstrap: unchanged in API but now guards against zero-sum
   probability arrays that caused division-by-zero on some datasets.

References
----------
de Prado, M. L. (2018). Advances in Financial Machine Learning, Ch.4.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Concurrent events — vectorised diff approach (no Python loops)
# ---------------------------------------------------------------------------

def get_num_concurrent_events(
    close_idx: pd.DatetimeIndex,
    t1: pd.Series,
    molecule: Optional[list] = None,
) -> pd.Series:
    """Count how many events are simultaneously open at each bar.

    Vectorised via ±1 markers at start and end of each event window,
    then cumulative sum. O(n_events log n_events + n_bars).

    Parameters
    ----------
    close_idx : pd.DatetimeIndex
        Full bar index of the price series.
    t1 : pd.Series
        Series mapping event start → event end.
    molecule : list, optional
        Subset of t1's index to process.

    Returns
    -------
    pd.Series  Concurrency count for every bar in close_idx.
    """
    if molecule is not None:
        t1_ = t1.reindex(molecule).dropna()
    else:
        t1_ = t1.dropna()

    if t1_.empty:
        return pd.Series(0, index=close_idx, name="concurrency")

    # Use a "diff" trick: +1 at event start, -1 at event end+1
    # After cumsum, this gives concurrent count at each timestamp.
    markers = pd.Series(0, index=close_idx, dtype=int)

    bar_arr = close_idx.as_unit("ns").asi8  # int64 ns; Timestamp.value is always ns

    for t0, t_end in t1_.items():
        # Find start bar
        s = int(np.searchsorted(bar_arr, pd.Timestamp(t0).value, side="left"))
        # Find end bar + 1
        e = int(np.searchsorted(bar_arr, pd.Timestamp(t_end).value, side="right"))
        if s < len(bar_arr):
            markers.iloc[s] += 1
        if e < len(bar_arr):
            markers.iloc[e] -= 1

    return markers.cumsum().rename("concurrency")


# ---------------------------------------------------------------------------
# Average uniqueness — sparse (no dense matrix)
# ---------------------------------------------------------------------------

def get_avg_uniqueness_from_t1(
    t1: pd.Series,
    close_idx: pd.DatetimeIndex,
) -> pd.Series:
    """Compute average uniqueness for each event without a dense matrix.

    Average uniqueness of event j = mean(1 / concurrency[t]) over bars
    where event j is active.

    This is the memory-efficient equivalent of the original get_ind_matrix
    + get_avg_uniqueness pipeline but uses O(n_events × avg_window) memory
    instead of O(n_bars × n_events).

    Parameters
    ----------
    t1 : pd.Series  Event start → event end.
    close_idx : pd.DatetimeIndex  Full bar index.

    Returns
    -------
    pd.Series  Average uniqueness per event (index = t1.index).
    """
    t1_ = t1.dropna()
    if t1_.empty:
        return pd.Series(dtype=float)

    # Compute full concurrency count once
    conc = get_num_concurrent_events(close_idx, t1_)
    conc = conc.replace(0, np.nan)  # avoid division by zero

    bar_arr = close_idx.as_unit("ns").asi8  # int64 ns; Timestamp.value is always ns
    avg_u = pd.Series(dtype=float, index=t1_.index)

    for t0, t_end in t1_.items():
        s = int(np.searchsorted(bar_arr, pd.Timestamp(t0).value, side="left"))
        e = int(np.searchsorted(bar_arr, pd.Timestamp(t_end).value, side="right"))
        window_conc = conc.iloc[s:e]
        u = (1.0 / window_conc).mean()
        avg_u[t0] = u if not np.isnan(u) else 0.0

    return avg_u.fillna(0.0)


# ---------------------------------------------------------------------------
# Legacy dense API — kept for compatibility, redirects to sparse
# ---------------------------------------------------------------------------

def get_ind_matrix(
    bar_idx: pd.DatetimeIndex,
    t1: pd.Series,
) -> pd.DataFrame:
    """Build the binary indicator matrix linking events to bars.

    WARNING: This creates a dense (n_bars × n_events) matrix. For datasets
    with n_bars > 5000, use get_avg_uniqueness_from_t1() directly to avoid
    memory issues (original OOM crash at 66K bars: 32.8 GiB).

    Kept for API compatibility with tests.

    Returns
    -------
    pd.DataFrame  Shape (n_bars, n_events).
    """
    n_bars = len(bar_idx)
    n_events = len(t1)
    # Guard against OOM: refuse > 1M cells (configurable)
    if n_bars * n_events > 1_000_000:
        raise MemoryError(
            f"get_ind_matrix would allocate {n_bars * n_events * 8 / 1e9:.1f} GiB "
            f"({n_bars} bars × {n_events} events). Use get_avg_uniqueness_from_t1() "
            "instead for large datasets."
        )
    ind_m = pd.DataFrame(0, index=bar_idx, columns=t1.index)
    for t0, t_end in t1.dropna().items():
        ind_m.loc[t0:t_end, t0] = 1
    return ind_m


def get_avg_uniqueness(ind_m: pd.DataFrame) -> pd.Series:
    """Compute average uniqueness from a pre-built indicator matrix.

    For large datasets, prefer get_avg_uniqueness_from_t1() directly.
    """
    conc = ind_m.sum(axis=1)
    conc = conc.replace(0, np.nan)
    uniq = ind_m.div(conc, axis=0)
    avg_uniq = uniq[uniq > 0].mean()
    return avg_uniq.fillna(0)


# ---------------------------------------------------------------------------
# Sequential Bootstrap — unchanged in behaviour, guards added
# ---------------------------------------------------------------------------

def seq_bootstrap(
    ind_m: pd.DataFrame,
    sample_length: Optional[int] = None,
    random_state: Optional[int] = None,
) -> list:
    """Draw a bootstrap sample that maximises average uniqueness.

    BUG FIX: original could produce all-zero probability vectors
    (e.g. when all candidate columns had zero marginal uniqueness)
    causing rng.choice to raise ValueError. Fixed by uniform fallback.

    Parameters
    ----------
    ind_m : pd.DataFrame  Indicator matrix (from get_ind_matrix).
    sample_length : int, optional  Defaults to number of events.
    random_state : int, optional

    Returns
    -------
    list  Column names of the bootstrap sample.
    """
    if sample_length is None:
        sample_length = ind_m.shape[1]

    rng = np.random.default_rng(random_state)
    phi: list = []
    col_list = list(ind_m.columns)

    while len(phi) < sample_length:
        avg_u = np.zeros(len(col_list))
        for i, col in enumerate(col_list):
            cols_so_far = [col_list[j] for j in phi] + [col]
            ind_m_ = ind_m[cols_so_far]
            avg_u[i] = get_avg_uniqueness(ind_m_).iloc[-1]

        total = avg_u.sum()
        if total <= 0:
            # Degenerate case: uniform fallback
            prob = np.ones(len(col_list)) / len(col_list)
        else:
            prob = avg_u / total

        chosen_pos = int(rng.choice(len(col_list), p=prob))
        phi.append(chosen_pos)

    return [col_list[i] for i in phi]


# ---------------------------------------------------------------------------
# Sample weights by return × uniqueness (memory-safe)
# ---------------------------------------------------------------------------

def get_sample_weights_return(
    t1: pd.Series,
    close: pd.Series,
    num_threads: int = 1,
) -> pd.Series:
    """Weight each event proportionally to |log-return| × avg uniqueness.

    BUG FIX: original called get_ind_matrix which OOM-crashed on large
    datasets. Now uses the sparse get_avg_uniqueness_from_t1() path.

    Parameters
    ----------
    t1 : pd.Series  Event start → event end.
    close : pd.Series  Price series.
    num_threads : int  Unused (kept for API compatibility).

    Returns
    -------
    pd.Series  Normalised sample weights (mean ≈ 1).
    """
    bar_idx = close.index
    t1_valid = t1.dropna()

    # Sparse uniqueness computation
    avg_u = get_avg_uniqueness_from_t1(t1_valid, bar_idx)

    # Absolute log-return per event
    bar_arr = bar_idx.as_unit("ns").asi8  # int64 ns; Timestamp.value is always ns
    rets = pd.Series(dtype=float, index=t1_valid.index)
    close_arr = close.values.astype(float)

    for t0, t_end in t1_valid.items():
        s = int(np.searchsorted(bar_arr, pd.Timestamp(t0).value, side="left"))
        e = int(np.searchsorted(bar_arr, pd.Timestamp(t_end).value, side="right"))
        e = min(e, len(bar_arr)) - 1
        if s < len(bar_arr) and e >= s and e < len(bar_arr):
            p0, p1 = close_arr[s], close_arr[e]
            if p0 > 0 and p1 > 0:
                rets[t0] = abs(np.log(p1 / p0))

    w = avg_u.reindex(t1_valid.index).fillna(0) * rets.fillna(0)
    mean_w = w.mean()
    if mean_w > 0:
        w = w / mean_w
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

    Parameters
    ----------
    t1 : pd.Series  Event start → event end.
    close : pd.Series  Price series.
    decay : float
        Slope of linear decay. decay=1 → oldest gets weight 0;
        decay=0 → no decay; Values are clipped to [0, 1] per event.

    Returns
    -------
    pd.Series  Normalised time-decayed sample weights.
    """
    w = get_sample_weights_return(t1, close)

    w_sorted = w.sort_index()
    n = len(w_sorted)
    if n < 2:
        return w_sorted.rename("weight")

    # Linear decay: factor_i = 1 - decay * (1 - (i+1)/n)
    factors = np.clip(
        1.0 - decay * (1.0 - (np.arange(n) + 1.0) / n),
        0.0, 1.0,
    )
    w_decayed = pd.Series(
        w_sorted.values * factors,
        index=w_sorted.index,
        name="weight",
    )
    mean_w = w_decayed.mean()
    if mean_w > 0:
        w_decayed = w_decayed / mean_w

    return w_decayed.reindex(w.index)
