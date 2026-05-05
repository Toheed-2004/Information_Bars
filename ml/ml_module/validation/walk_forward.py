"""
ml_module/validation/walk_forward.py
--------------------------------------
Expanding-window walk-forward validation.

Architecture
------------
WalkForwardValidator:
  - Trains on [0, train_end]
  - Tests  on [train_end + embargo, train_end + embargo + test_window]
  - Advances train_end by step_bars
  - Expands (never shrinks) the training window

This prevents look-ahead bias because each test fold uses only data that
was genuinely available at the simulated decision point.

The validator integrates cleanly with the MetaEnsemble via a callable
``model_factory``.  The factory receives no state — each fold trains a
fresh model, avoiding cross-fold parameter bleed.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_module.utils.helpers import get_logger, classification_report_dict

logger = get_logger(__name__)


class WalkForwardValidator:
    """
    Expanding-window walk-forward validator.

    Parameters
    ----------
    initial_train_bars : Size of the first training window (bars).
    step_bars          : How many bars to advance before each re-train.
    min_test_bars      : Minimum bars in a test window; shorter folds are skipped.
    embargo_bars       : Gap between training end and test start.
    """

    def __init__(
        self,
        initial_train_bars: int = 300,
        step_bars: int = 50,
        min_test_bars: int = 30,
        embargo_bars: int = 5,
        confidence_threshold: float = 0.0,
    ):
        if initial_train_bars < 50:
            raise ValueError("initial_train_bars must be >= 50")
        self.initial_train_bars = initial_train_bars
        self.step_bars = step_bars
        self.min_test_bars = min_test_bars
        self.embargo_bars = embargo_bars
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable,
    ) -> Tuple[List[Dict], pd.Series]:
        """
        Run walk-forward validation.

        Parameters
        ----------
        X             : Feature matrix (sorted by time, ascending).
        y             : Target labels (same index as X).
        model_factory : Callable() → fitted_model.  Called once per fold with
                        (X_train, y_train).  Signature:
                        ``model_factory(X_train, y_train) -> model``
                        where model has a ``.predict(X_test)`` method.

        Returns
        -------
        fold_metrics  : List of per-fold metric dicts.
        all_preds     : Series of out-of-sample predictions aligned to y's index.
        """
        n = len(X)
        fold_metrics: List[Dict] = []
        all_preds = pd.Series(np.nan, index=y.index, dtype=float, name="prediction")

        train_end = self.initial_train_bars
        fold_num = 0

        while True:
            test_start = train_end + self.embargo_bars
            test_end = test_start + self.step_bars

            if test_end > n:
                test_end = n

            if (test_end - test_start) < self.min_test_bars:
                logger.debug("Walk-forward complete: remaining test window too small.")
                break

            train_pos = np.arange(0, train_end)
            test_pos = np.arange(test_start, test_end)

            X_train = X.iloc[train_pos]
            y_train = y.iloc[train_pos]
            X_test = X.iloc[test_pos]
            y_test = y.iloc[test_pos]

            # Guard: skip fold if any class is missing from training set
            unique_train = np.unique(y_train)
            if len(unique_train) < 2:
                logger.warning(
                    "Fold %d: only %d class(es) in training set — skipping.",
                    fold_num,
                    len(unique_train),
                )
                train_end += self.step_bars
                fold_num += 1
                continue

            try:
                model = model_factory(X_train, y_train)
                preds = model.predict(X_test)

                # Confidence filtering: set low-confidence predictions to HOLD (0)
                # This reduces overtrading and improves signal quality.
                # Only active when model supports predict_proba.
                if self.confidence_threshold > 0.0:
                    try:
                        proba = model.predict_proba(X_test)
                        max_proba = proba.max(axis=1)
                        low_conf = max_proba < self.confidence_threshold
                        preds = np.where(low_conf, 0, preds)
                        logger.debug(
                            "Fold %d: filtered %d/%d low-confidence predictions to HOLD",
                            fold_num,
                            int(low_conf.sum()),
                            len(preds),
                        )
                    except Exception:
                        pass  # model has no predict_proba — skip filtering

            except Exception as e:
                logger.error("Fold %d failed: %s", fold_num, e)
                train_end += self.step_bars
                fold_num += 1
                continue

            # Store predictions
            all_preds.iloc[test_pos] = preds

            # Compute fold metrics
            metrics = classification_report_dict(y_test.to_numpy(), np.array(preds))
            metrics.update(
                {
                    "fold": fold_num,
                    "train_start": int(train_pos[0]),
                    "train_end": int(train_pos[-1]),
                    "test_start": int(test_pos[0]),
                    "test_end": int(test_pos[-1]),
                    "train_size": len(train_pos),
                    "test_size": len(test_pos),
                    "test_start_dt": str(X.index[test_pos[0]]),
                    "test_end_dt": str(X.index[test_pos[-1]]),
                }
            )
            fold_metrics.append(metrics)

            logger.info(
                "WF fold %2d | train=%d bars | test=[%s → %s] | acc=%.3f | mcc=%.3f",
                fold_num,
                len(train_pos),
                metrics["test_start_dt"][:10],
                metrics["test_end_dt"][:10],
                metrics["accuracy"],
                metrics["mcc"],
            )

            train_end += self.step_bars
            fold_num += 1

        if fold_num == 0:
            logger.warning(
                "Walk-forward produced 0 folds. Check initial_train_bars and data length."
            )

        return fold_metrics, all_preds

    # ------------------------------------------------------------------
    # Split preview (no training — useful for research planning)
    # ------------------------------------------------------------------

    def get_splits(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Preview all (train, test) positional splits without running any model.

        Parameters
        ----------
        n : Total number of samples.

        Returns
        -------
        List of (train_positions, test_positions) tuples.
        """
        splits = []
        train_end = self.initial_train_bars
        while True:
            test_start = train_end + self.embargo_bars
            test_end = test_start + self.step_bars
            if test_end > n:
                test_end = n
            if (test_end - test_start) < self.min_test_bars:
                break
            splits.append(
                (
                    np.arange(0, train_end),
                    np.arange(test_start, test_end),
                )
            )
            train_end += self.step_bars
        return splits

    def __repr__(self) -> str:
        return (
            f"WalkForwardValidator(initial_train_bars={self.initial_train_bars}, "
            f"step_bars={self.step_bars}, embargo_bars={self.embargo_bars})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_walk_forward(cfg: Dict) -> WalkForwardValidator:
    """Instantiate from the ``walk_forward`` config section."""
    return WalkForwardValidator(
        initial_train_bars=cfg.get("initial_train_bars", 300),
        step_bars=cfg.get("step_bars", 50),
        min_test_bars=cfg.get("min_test_bars", 30),
        embargo_bars=cfg.get("embargo_bars", 5),
        confidence_threshold=cfg.get("confidence_threshold", 0.0),
    )
