"""
mlfinlab.features.fractional_diff
===================================
Fractional differencing via the Fixed-width Finite Difference (FFD) method.

Standard integer differencing (d=1) removes non-stationarity but destroys
memory.  Fractional differencing with 0 < d < 1 achieves stationarity while
retaining the maximum amount of memory – crucial for financial ML.

FFD computes weights using a sliding window of fixed length *width* instead
of the full history, making it compatible with real-time pipelines.

Key functions
-------------
frac_diff_ffd   Apply fractional differencing to a price series.
find_min_d      Grid-search the minimum d that achieves stationarity.
plot_min_ffd    Diagnostic plot of ADF statistics vs d.

References
----------
de Prado, M. L. (2018). *Advances in Financial Machine Learning*, Ch.5.
Jensen, T. L. (2019).  "Fractionally differenced features in financial
    machine learning."  *Journal of Financial Data Science*.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------------
# Weight vector for fractional differencing
# ---------------------------------------------------------------------------

def _get_weights_ffd(d: float, threshold: float, max_width: int) -> np.ndarray:
    """Compute FFD weight vector.

    w_k = -w_{k-1} * (d - k + 1) / k,   w_0 = 1

    Parameters
    ----------
    d : float
        Differencing order (0 < d ≤ 2).
    threshold : float
        Drop weights whose absolute value falls below *threshold*.
    max_width : int
        Hard cap on the number of weights (window size).

    Returns
    -------
    np.ndarray  Weight vector (length = width).
    """
    w = [1.0]
    k = 1
    while k < max_width:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])   # oldest weight first


# ---------------------------------------------------------------------------
# FFD transform
# ---------------------------------------------------------------------------

def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
    max_width: int = 10_000,
) -> pd.Series:
    """Apply fractional differencing (FFD) to *series*.

    Parameters
    ----------
    series : pd.Series
        Price series (or any financial time series).  Log-prices are
        recommended to reduce magnitude heterogeneity.
    d : float
        Fractional differencing order.  ``d=1`` reproduces standard
        first differencing; ``0 < d < 1`` is the typical ML range.
    threshold : float
        Weights smaller than this are discarded (controls window width).
    max_width : int
        Maximum window size (safety cap).

    Returns
    -------
    pd.Series  Fractionally differenced series (NaNs at head equal to
               ``width - 1``).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> close = pd.Series(np.random.randn(200).cumsum() + 100)
    >>> fd = frac_diff_ffd(np.log(close), d=0.4)
    """
    if not (0.0 < d <= 2.0):
        raise ValueError(f"d must be in (0, 2], got {d}")

    n = len(series)
    effective_max = min(max_width, n)
    w = _get_weights_ffd(d, threshold, effective_max)
    width = len(w)

    vals = series.values
    n = len(vals)
    out = np.full(n, np.nan)

    for i in range(width - 1, n):
        out[i] = float(np.dot(w, vals[i - width + 1 : i + 1]))

    return pd.Series(out, index=series.index, name=f"frac_diff_d{d:.3f}")


# ---------------------------------------------------------------------------
# Minimum d that passes ADF
# ---------------------------------------------------------------------------

def find_min_d(
    series: pd.Series,
    d_range: tuple = (0.0, 1.0),
    step: float = 0.05,
    threshold: float = 1e-5,
    max_width: int = 10_000,
    adf_significance: float = 0.05,
    verbose: bool = False,
) -> float:
    """Grid-search the minimum *d* that makes *series* ADF-stationary.

    Parameters
    ----------
    series : pd.Series
        Input series (log-prices recommended).
    d_range : tuple[float, float]
        Search interval for *d*.
    step : float
        Grid resolution.
    threshold : float
        FFD weight threshold.
    max_width : int
        Maximum window.
    adf_significance : float
        p-value threshold for the Augmented Dickey-Fuller test.
    verbose : bool
        Print ADF statistics at each *d*.

    Returns
    -------
    float  Minimum *d* achieving stationarity, or ``d_range[1]`` if
           none found.
    """
    d_vals = np.arange(d_range[0], d_range[1] + step / 2, step)
    for d in d_vals:
        if d == 0:
            fd = series
        else:
            fd = frac_diff_ffd(series, d, threshold, max_width).dropna()

        if len(fd) < 20:
            continue

        adf_result = adfuller(fd, maxlag=1, regression="c", autolag=None)
        p_val = adf_result[1]

        if verbose:
            print(f"d={d:.3f}  ADF stat={adf_result[0]:.4f}  p={p_val:.4f}")

        if p_val <= adf_significance:
            return float(d)

    return float(d_range[1])


# ---------------------------------------------------------------------------
# Diagnostic plot data
# ---------------------------------------------------------------------------

def plot_min_ffd(
    series: pd.Series,
    d_range: tuple = (0.0, 1.0),
    step: float = 0.05,
    threshold: float = 1e-5,
    max_width: int = 10_000,
) -> pd.DataFrame:
    """Compute ADF statistics across a range of *d* values.

    Useful for constructing the "min FFD" diagnostic chart (AFML Fig.5.2).

    Parameters
    ----------
    series : pd.Series
        Input series (log-prices recommended).
    d_range : tuple[float, float]
    step : float
    threshold : float
    max_width : int

    Returns
    -------
    pd.DataFrame  columns: ``d``, ``adf_stat``, ``p_value``,
                  ``corr_with_original``, ``n_obs``.
    """
    d_vals = np.arange(d_range[0], d_range[1] + step / 2, step)
    rows = []

    for d in d_vals:
        if d == 0:
            fd = series
        else:
            fd = frac_diff_ffd(series, d, threshold, max_width).dropna()

        if len(fd) < 20:
            continue

        adf_res = adfuller(fd, maxlag=1, regression="c", autolag=None)
        corr = series.reindex(fd.index).corr(fd)

        rows.append(
            {
                "d": round(d, 5),
                "adf_stat": adf_res[0],
                "p_value": adf_res[1],
                "corr_with_original": corr,
                "n_obs": len(fd),
            }
        )

    return pd.DataFrame(rows)
