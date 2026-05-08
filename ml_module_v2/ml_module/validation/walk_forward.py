"""
validation/walk_forward.py
--------------------------
Expanding-window walk-forward validation.

Each fold:
  train [0 .. train_end]  →  embargo gap  →  test [test_start .. test_end]

The stitched predictions from all test windows become the trading signal.
No separate inference step is needed.

Confidence filter:
  If model has predict_proba AND confidence_threshold > 0,
  bars where max(P(class)) < threshold are set to HOLD (0).
  This reduces overtrading by acting only on high-conviction signals.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, matthews_corrcoef

logger = logging.getLogger(__name__)


class WalkForward:
    def __init__(
        self,
        initial_train_bars:   int   = 200,
        step_bars:            int   = 100,
        min_test_bars:        int   = 20,
        embargo_bars:         int   = 5,
        confidence_threshold: float = 0.0,
    ):
        self.initial_train_bars   = initial_train_bars
        self.step_bars            = step_bars
        self.min_test_bars        = min_test_bars
        self.embargo_bars         = embargo_bars
        self.confidence_threshold = confidence_threshold

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        factory: Callable,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict], pd.Series]:
        """
        Parameters
        ----------
        X              : Feature matrix, pd.DataFrame, time-sorted.
        y              : Labels, pd.Series, same index as X.
        factory        : Callable(X_tr, y_tr, w_tr) -> fitted model.
                         Model must implement predict(X).
                         predict_proba(X) optional (for confidence filter).
        sample_weights : Float32 array length len(X), mean≈1.

        Returns
        -------
        fold_metrics : list of {fold, accuracy, mcc, train_size, test_start_dt, test_end_dt}
        predictions  : pd.Series float, NaN where no test fold covered the bar.
        """
        n           = len(X)
        preds_out   = pd.Series(np.nan, index=y.index, dtype=float, name="prediction")
        metrics     = []
        train_end   = self.initial_train_bars
        fold        = 0

        while True:
            ts = train_end + self.embargo_bars
            te = min(ts + self.step_bars, n)

            if (te - ts) < self.min_test_bars:
                break

            tr_idx = np.arange(0, train_end)
            te_idx = np.arange(ts, te)

            X_tr = X.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            X_te = X.iloc[te_idx]
            y_te = y.iloc[te_idx].to_numpy()
            w_tr = sample_weights[tr_idx] if sample_weights is not None else None

            if len(np.unique(y_tr)) < 2:
                train_end += self.step_bars; fold += 1; continue

            try:
                model  = factory(X_tr, y_tr, w_tr)
                p      = model.predict(X_te)

                if self.confidence_threshold > 0 and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_te)
                    mask  = proba.max(axis=1) < self.confidence_threshold
                    p     = np.where(mask, 0, p)

            except Exception as e:
                logger.error("Fold %d failed: %s", fold, e, exc_info=True)
                train_end += self.step_bars; fold += 1; continue

            preds_out.iloc[te_idx] = p

            acc = float(accuracy_score(y_te, p))
            mcc = float(matthews_corrcoef(y_te, p))
            metrics.append({
                "fold":          fold,
                "accuracy":      acc,
                "mcc":           mcc,
                "train_size":    len(tr_idx),
                "test_start_dt": str(X.index[ts])[:10],
                "test_end_dt":   str(X.index[te - 1])[:10],
            })
            logger.info(
                "WF fold %d | train=%d | [%s->%s] | acc=%.3f mcc=%+.3f",
                fold, len(tr_idx), metrics[-1]["test_start_dt"],
                metrics[-1]["test_end_dt"], acc, mcc,
            )
            train_end += self.step_bars; fold += 1

        if fold == 0:
            logger.warning("Walk-forward produced 0 folds.")

        return metrics, preds_out
