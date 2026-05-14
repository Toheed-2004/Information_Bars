"""
mlfinlab.labeling.triple_barrier
=================================
de Prado (2018) triple-barrier labeling method – AFML Chapter 3.

The three barriers are:
  * **Upper** horizontal barrier  – profit-taking at +pt × σ
  * **Lower** horizontal barrier  – stop-loss   at -sl × σ
  * **Vertical** barrier          – maximum holding period t1

Labels
------
+1  upper barrier touched first  (or sign of return if symmetric)
-1  lower barrier touched first
 0  vertical barrier hit, |return| < min(pt, sl)

Meta-labeling (Ch.3.6)
-----------------------
A second pass labels whether a primary signal (side) was *correct*
(1) or not (0).  The *side* series flips the meaning of the horizontal
barriers so that only the barrier on the *signal's side* counts.

Uniqueness / Sample Weights (Ch.4)
------------------------------------
``get_bins`` also returns a *ret* column (log-return over the event
window) and an *out* flag so downstream weight estimators can operate
directly on the label frame.

References
----------
de Prado, M. L. (2018). *Advances in Financial Machine Learning*, Ch.3.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from mlfinlab.utils.helpers import daily_vol, log_returns


# ---------------------------------------------------------------------------
# Step 1: vertical barrier helper
# ---------------------------------------------------------------------------

def add_vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    num_days: float = 1.0,
) -> pd.Series:
    """Return a Series mapping each event → its vertical barrier timestamp.

    Parameters
    ----------
    t_events : pd.DatetimeIndex
        Timestamps of sampled events.
    close : pd.Series
        Close prices (used only for its index).
    num_days : float
        Maximum holding period in calendar days.

    Returns
    -------
    pd.Series  index=t_events, values=barrier timestamps (NaT when
               event falls within *num_days* of the last bar).
    """
    delta = pd.Timedelta(days=num_days)
    idx = close.index
    t1_list: list = []
    for t in t_events:
        future = idx[idx >= t + delta]
        t1_list.append(future[0] if len(future) else pd.NaT)
    return pd.Series(t1_list, index=t_events, name="t1")


# ---------------------------------------------------------------------------
# Step 2: event frame with dynamic barriers
# ---------------------------------------------------------------------------

def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: Union[list, tuple],
    target: pd.Series,
    min_ret: float = 0.0,
    num_threads: int = 1,
    vertical_barrier_times: Optional[pd.Series] = None,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build the event DataFrame driving triple-barrier labeling.

    Parameters
    ----------
    close : pd.Series
        Bar close prices.
    t_events : pd.DatetimeIndex
        Candidate event timestamps (e.g. from CUSUM filter).
    pt_sl : list[float, float]
        ``[profit_take_multiplier, stop_loss_multiplier]``.
        Set either to 0 to disable that barrier.
    target : pd.Series
        Per-event target (e.g. daily_vol), aligned to *t_events*.
    min_ret : float
        Minimum absolute return threshold to include an event.
    num_threads : int
        Workers for multiprocessing (serial when ≤1).
    vertical_barrier_times : pd.Series, optional
        Pre-computed t1 series (from :func:`add_vertical_barrier`).
        When ``None``, no vertical barrier is applied.
    side : pd.Series, optional
        Primary model side prediction (+1 / -1) for meta-labeling.

    Returns
    -------
    pd.DataFrame  columns: ``t1`` (vertical barrier), ``trgt``
                  (volatility target), ``side`` (if meta-labeling).
    """
    # --- align target to t_events
    target = target.reindex(t_events).dropna()
    if min_ret > 0:
        target = target[target >= min_ret]

    # --- vertical barrier
    if vertical_barrier_times is not None:
        t1 = vertical_barrier_times.reindex(target.index)
    else:
        t1 = pd.Series(pd.NaT, index=target.index)

    # --- side for meta-labeling only
    # When side=None  ->  standard triple-barrier: all three labels possible
    #                     (-1 stop-loss, 0 vertical/timeout, +1 profit-take)
    #                     side column is NOT added to events frame so get_bins
    #                     takes the standard np.sign(ret) path, not clip(lower=0)
    # When side given ->  meta-labeling: side flips the barrier perspective so
    #                     only the barrier on the signal's side counts; labels
    #                     become binary 0/1 (was the primary signal correct?)
    if side is None:
        events = pd.concat({"t1": t1, "trgt": target}, axis=1)
        events = events.dropna(subset=["trgt"])
        pt_sl_ = [pt_sl[0], pt_sl[1]]
    else:
        side_ = side.reindex(target.index)
        events = pd.concat({"t1": t1, "trgt": target, "side": side_}, axis=1)
        events = events.dropna(subset=["trgt"])
        pt_sl_ = [pt_sl[0], pt_sl[1]]

    # --- apply barriers
    out = _apply_pt_sl_on_t1(close, events, pt_sl_)

    # re-attach first-touch timestamp
    out = events.join(out.rename("t1_touch"), how="left")
    return out


def _apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: list,
) -> pd.Series:
    """Vectorised barrier touch time for a batch of events.

    Returns a Series with the *first touch* timestamp for each event.
    """
    out = pd.Series(dtype="datetime64[ns]", name="t1_touch")
    for loc, row in events.iterrows():
        t1_barrier = row["t1"]  # vertical barrier
        trgt       = row["trgt"]
        # side=1.0 in standard triple-barrier (no side column in events)
        # side=+1/-1 only in meta-labeling mode (side column present)
        side_val   = float(row["side"]) if "side" in row.index else 1.0

        # slice of prices from event to vertical barrier (inclusive)
        if pd.isna(t1_barrier):
            df0 = close.loc[loc:]
        else:
            df0 = close.loc[loc:t1_barrier]

        if df0.empty:
            out.loc[loc] = pd.NaT
            continue

        cum_ret = (df0 / df0.iloc[0] - 1) * side_val

        # upper barrier (profit-take)
        if pt_sl[0] > 0:
            pt_hit = cum_ret[cum_ret >= trgt * pt_sl[0]]
            pt_time = pt_hit.index[0] if len(pt_hit) else pd.NaT
        else:
            pt_time = pd.NaT

        # lower barrier (stop-loss)
        if pt_sl[1] > 0:
            sl_hit = cum_ret[cum_ret <= -trgt * pt_sl[1]]
            sl_time = sl_hit.index[0] if len(sl_hit) else pd.NaT
        else:
            sl_time = pd.NaT

        touch_times = [t for t in [pt_time, sl_time, t1_barrier] if not pd.isna(t)]
        out.loc[loc] = min(touch_times) if touch_times else pd.NaT

    return out


# ---------------------------------------------------------------------------
# Step 3: assign labels
# ---------------------------------------------------------------------------

def get_bins(
    events: pd.DataFrame,
    close: pd.Series,
    t1_col: str = "t1_touch",
) -> pd.DataFrame:
    """Assign +1 / -1 / 0 labels based on first-touch barrier.

    Parameters
    ----------
    events : pd.DataFrame
        Output of :func:`get_events`.
    close : pd.Series
        Bar close prices.
    t1_col : str
        Column holding the first-touch timestamp.

    Returns
    -------
    pd.DataFrame  columns: ``ret`` (log-return), ``bin`` (label),
                  ``trgt``, ``side``.
    """
    events_ = events.dropna(subset=[t1_col])

    # Use .loc for individual lookups to avoid duplicate-index reindex errors
    out = pd.DataFrame(index=events_.index)
    rets = []
    for t0, row in events_.iterrows():
        t_end = row[t1_col]
        p0 = close.loc[t0] if t0 in close.index else np.nan
        if pd.isna(t_end) or t_end not in close.index:
            # fall back to last available price
            avail = close.index[close.index <= t_end] if not pd.isna(t_end) else close.index[:0]
            p1 = close.loc[avail[-1]] if len(avail) else np.nan
        else:
            p1 = close.loc[t_end]
        if isinstance(p0, pd.Series):
            p0 = p0.iloc[-1]
        if isinstance(p1, pd.Series):
            p1 = p1.iloc[-1]
        rets.append(np.log(p1 / p0) if (p0 > 0 and p1 > 0) else np.nan)
    out["ret"] = rets

    # Standard triple-barrier: bin = sign(ret) -> -1, 0, +1
    #   -1  stop-loss barrier hit first   (price fell by sl * sigma)
    #    0  vertical barrier hit first    (timeout, neither barrier reached)
    #   +1  profit-take barrier hit first (price rose by pt * sigma)
    #
    # Meta-labeling (side column present):
    #   ret is already direction-adjusted (ret * side in event frame)
    #   bin = 0/1 only: was the primary signal correct?
    if "side" in events_.columns:
        # meta-labeling path: direction-adjust ret, then binary label
        out["ret"] *= events_["side"].values
        out["bin"] = np.sign(out["ret"])
        out["bin"] = out["bin"].clip(lower=0)   # 0 = wrong, 1 = correct
        out["side"] = events_["side"]
    else:
        # standard triple-barrier path: three classes -1, 0, +1
        out["bin"] = np.sign(out["ret"])

    out["trgt"] = events_["trgt"]

    return out


# ---------------------------------------------------------------------------
# Step 4: drop rare labels
# ---------------------------------------------------------------------------

def drop_labels(
    events: pd.DataFrame,
    min_pct: float = 0.05,
) -> pd.DataFrame:
    """Iteratively remove the least-frequent label until all exceed *min_pct*.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame with a ``bin`` column.
    min_pct : float
        Minimum fraction of any label.

    Returns
    -------
    pd.DataFrame  Filtered events.
    """
    while True:
        df0 = events["bin"].value_counts(normalize=True)
        if df0.min() > min_pct or len(df0) < 3:
            break
        rare = df0.idxmin()
        events = events[events["bin"] != rare]
    return events


# ---------------------------------------------------------------------------
# Meta-labeling convenience wrapper
# ---------------------------------------------------------------------------

def meta_labeling(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: Union[list, tuple],
    target: pd.Series,
    side: pd.Series,
    num_days: float = 1.0,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """End-to-end meta-labeling pipeline.

    Combines :func:`add_vertical_barrier`, :func:`get_events`, and
    :func:`get_bins` into a single call, returning binary (0/1) labels
    indicating whether the primary signal *side* was correct.

    Parameters
    ----------
    close : pd.Series
        Bar close prices.
    t_events : pd.DatetimeIndex
        Primary-model entry timestamps.
    pt_sl : [float, float]
        Profit-take and stop-loss multipliers.
    target : pd.Series
        Volatility target series.
    side : pd.Series
        Primary model predicted side (+1 / -1).
    num_days : float
        Vertical-barrier look-ahead in calendar days.
    min_ret : float
        Minimum target to include an event.

    Returns
    -------
    pd.DataFrame  ``ret``, ``bin`` (0/1), ``trgt``, ``side``.
    """
    t1 = add_vertical_barrier(t_events, close, num_days)
    events = get_events(
        close=close,
        t_events=t_events,
        pt_sl=pt_sl,
        target=target,
        min_ret=min_ret,
        vertical_barrier_times=t1,
        side=side,
    )
    labels = get_bins(events, close)
    return labels