"""
ml_module/features/fractional_diff.py
---------------------------------------
Fractional Differencing (López de Prado, "Advances in Financial ML", Ch. 5).

Purpose
-------
Achieve stationarity in financial time series while preserving as much
long-range memory as possible.  Integer differencing (d=1) kills memory;
raw series (d=0) are non-stationary.  Fractional differencing finds the
minimum d ∈ (0, 1) that passes an ADF test, maximising autocorrelation
retention.

Key classes / functions
-----------------------
FractionalDifferencer   — transforms one or more columns of a DataFrame.
find_optimal_d()        — searches for the minimum d that achieves stationarity.
build_differencer()     — factory from YAML config dict.

References
----------
López de Prado, M. (2018). Advances in Financial Machine Learning.
  Chapter 5 — "Fractionally Differentiated Features".
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from ml_module.utils.helpers import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core convolution kernel
# ---------------------------------------------------------------------------

def _frac_diff_weights(d: float, threshold: float = 1e-5, max_window: Optional[int] = None) -> np.ndarray:
    """
    Compute the convolution weights for the fractional differencing operator.

    The weights follow the binomial series:
        w_k = ∏_{j=0}^{k-1} (d - j) / (k!)

    Parameters
    ----------
    d         : Differencing order in (0, 1).
    threshold : Drop weights with |w_k| < threshold (efficiency cutoff).
    max_window: Hard cap on the weight vector length.

    Returns
    -------
    1-D array of weights (most recent first: w[0] = 1.0).
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        if max_window is not None and k >= max_window:
            break
    return np.array(weights[::-1])  # oldest-first for np.convolve


def _apply_weights(series: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Apply the fractional differencing weights to a 1-D series.

    The first (len(weights)-1) observations become NaN because the full
    convolution window is not yet available — matching the behaviour of
    pandas ``rolling()``.

    Parameters
    ----------
    series  : Raw 1-D float64 array.
    weights : Output of ``_frac_diff_weights`` (oldest-first).

    Returns
    -------
    Array of same length; leading entries are NaN.
    """
    n      = len(series)
    # Cap kernel so we always produce at least one non-NaN output
    if len(weights) > n:
        weights = weights[-n:]          # keep the highest-magnitude (recent) end
    w_len  = len(weights)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(w_len - 1, n):
        window        = series[i - w_len + 1 : i + 1]
        result[i]     = np.dot(weights, window)

    return result


# ---------------------------------------------------------------------------
# Stationarity search
# ---------------------------------------------------------------------------

def find_optimal_d(
    series: pd.Series,
    d_min: float = 0.1,
    d_max: float = 1.0,
    d_step: float = 0.1,
    adf_significance: float = 0.05,
    threshold: float = 1e-5,
    max_window: Optional[int] = 200,
) -> float:
    """
    Search for the minimum d that achieves ADF stationarity.

    Algorithm
    ---------
    1. Iterate d from d_min to d_max in steps of d_step.
    2. For each d, apply fractional differencing.
    3. Run ADF test on the differenced series (dropping leading NaNs).
    4. Return the first d where ADF p-value ≤ adf_significance.
    5. If no d achieves stationarity, return d_max and warn.

    Parameters
    ----------
    series           : Raw price/volume series (pandas Series).
    d_min / d_max    : Search range.
    d_step           : Grid step size.
    adf_significance : Rejection threshold for the unit-root null hypothesis.
    threshold        : Weight cutoff passed to ``_frac_diff_weights``.

    Returns
    -------
    Optimal d (float).
    """
    values = series.dropna().to_numpy(dtype=np.float64)

    if len(values) < 30:
        logger.warning("Series too short for ADF (%d obs); defaulting to d=1.0", len(values))
        return 1.0

    d_candidates = np.round(np.arange(d_min, d_max + d_step / 2, d_step), 6)

    for d in d_candidates:
        weights = _frac_diff_weights(d, threshold=threshold, max_window=max_window)
        diff_vals = _apply_weights(values, weights)
        diff_clean = diff_vals[~np.isnan(diff_vals)]

        if len(diff_clean) < 20:
            continue  # not enough observations after warmup

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, pvalue, *_ = adfuller(diff_clean, maxlag=1, autolag=None)
            except Exception:
                continue

        logger.debug("  d=%.2f → ADF p-value=%.4f", d, pvalue)

        if pvalue <= adf_significance:
            logger.info("Optimal d=%.2f (ADF p=%.4f ≤ %.2f)", d, pvalue, adf_significance)
            return float(d)

    logger.warning(
        "No d in [%.1f, %.1f] achieved ADF significance=%.2f; using d_max=%.1f",
        d_min, d_max, adf_significance, d_max,
    )
    return float(d_max)


# ---------------------------------------------------------------------------
# Main transformer
# ---------------------------------------------------------------------------

class FractionalDifferencer:
    """
    Apply fractional differencing to selected columns of a DataFrame.

    Parameters
    ----------
    d              : Differencing order. Pass a float to fix it, or ``"auto"``
                     to run ``find_optimal_d`` per column.
    target_columns : Column names to transform. Others are passed through.
    threshold      : Weight cutoff (efficiency parameter).
    max_window     : Hard cap on convolution kernel length.
    d_min / d_max / d_step / adf_significance : Forwarded to ``find_optimal_d``
                     when d == "auto".
    """

    def __init__(
        self,
        d: Union[float, str] = "auto",
        target_columns: Optional[Sequence[str]] = None,
        threshold: float = 1e-5,
        max_window: Optional[int] = None,
        d_min: float = 0.1,
        d_max: float = 1.0,
        d_step: float = 0.1,
        adf_significance: float = 0.05,
    ):
        self.d                = d
        self.target_columns   = list(target_columns) if target_columns else None
        self.threshold        = threshold
        self.max_window       = max_window
        self.d_min            = d_min
        self.d_max            = d_max
        self.d_step           = d_step
        self.adf_significance = adf_significance

        # Populated after fit / fit_transform
        self.d_values_: Dict[str, float] = {}
        self.weights_:  Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FractionalDifferencer":
        """
        Learn the optimal d for each target column (no-op if d is fixed).

        Parameters
        ----------
        df : Bar DataFrame.

        Returns
        -------
        self
        """
        cols = self._resolve_columns(df)
        for col in cols:
            if col not in df.columns:
                logger.warning("Column '%s' not in DataFrame; skipping.", col)
                continue
            d_val = self._find_d(df[col])
            self.d_values_[col] = d_val
            self.weights_[col]  = _frac_diff_weights(d_val, self.threshold, self.max_window)
            logger.info("Column '%s': d=%.3f  kernel_len=%d", col, d_val, len(self.weights_[col]))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fractional differencing to *df* using the fitted d values.

        Columns not in ``target_columns`` are returned unchanged.
        New columns are named ``{col}_fdiff``.

        Parameters
        ----------
        df : Bar DataFrame (must contain the same columns used during fit).

        Returns
        -------
        Copy of *df* with new ``*_fdiff`` columns appended.
        """
        result = df.copy()
        for col, weights in self.weights_.items():
            if col not in df.columns:
                continue
            values    = df[col].to_numpy(dtype=np.float64)
            diff_vals = _apply_weights(values, weights)
            result[f"{col}_fdiff"] = diff_vals
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit then transform in one step."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_columns(self, df: pd.DataFrame) -> List[str]:
        if self.target_columns:
            return [c for c in self.target_columns if c in df.columns]
        # Default: all numeric columns
        return [c for c in df.select_dtypes(include=np.number).columns]

    def _find_d(self, series: pd.Series) -> float:
        if self.d == "auto":
            return find_optimal_d(
                series,
                d_min=self.d_min,
                d_max=self.d_max,
                d_step=self.d_step,
                adf_significance=self.adf_significance,
                threshold=self.threshold,
                max_window=self.max_window,
            )
        return float(self.d)

    # ------------------------------------------------------------------
    # Diagnostics (research use)
    # ------------------------------------------------------------------

    def stationarity_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame showing ADF statistics before and after differencing.
        Useful for documenting the stationarity–memory tradeoff in a paper.
        """
        rows = []
        for col, d_val in self.d_values_.items():
            if col not in df.columns:
                continue
            raw_vals = df[col].dropna().to_numpy(dtype=np.float64)
            diff_col = f"{col}_fdiff"

            # ADF on raw series
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _, p_raw, *_ = adfuller(raw_vals[-500:], maxlag=1, autolag=None)
                except Exception:
                    p_raw = np.nan

            # ADF on differenced series (from transform)
            p_diff = np.nan
            if diff_col in df.columns:
                diff_vals = df[diff_col].dropna().to_numpy(dtype=np.float64)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        _, p_diff, *_ = adfuller(diff_vals[-500:], maxlag=1, autolag=None)
                    except Exception:
                        pass

            rows.append({
                "column":       col,
                "d":            d_val,
                "adf_p_raw":    round(p_raw,  4) if not np.isnan(p_raw)  else None,
                "adf_p_fdiff":  round(p_diff, 4) if not np.isnan(p_diff) else None,
                "kernel_len":   len(self.weights_.get(col, [])),
            })

        return pd.DataFrame(rows)

    def get_params(self) -> Dict:
        return {
            "d":                self.d,
            "target_columns":   self.target_columns,
            "threshold":        self.threshold,
            "max_window":       self.max_window,
            "d_min":            self.d_min,
            "d_max":            self.d_max,
            "d_step":           self.d_step,
            "adf_significance": self.adf_significance,
            "fitted_d_values":  self.d_values_,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_differencer(cfg: Dict) -> FractionalDifferencer:
    """
    Instantiate a FractionalDifferencer from the ``fractional_diff`` config section.
    YAML may deserialise scientific-notation floats (e.g. 1e-5) as strings;
    explicit float() casts guard against this.
    """
    d_raw = cfg.get("d", "auto")
    d     = "auto" if str(d_raw) == "auto" else float(d_raw)
    mw    = cfg.get("max_window", 200)
    return FractionalDifferencer(
        d                = d,
        target_columns   = cfg.get("target_columns", ["close", "volume", "vwap"]),
        threshold        = float(cfg.get("threshold",        1e-5)),
        max_window       = int(mw) if mw is not None else 200,
        d_min            = float(cfg.get("d_min",            0.1)),
        d_max            = float(cfg.get("d_max",            1.0)),
        d_step           = float(cfg.get("d_step",           0.1)),
        adf_significance = float(cfg.get("adf_significance", 0.05)),
    )
