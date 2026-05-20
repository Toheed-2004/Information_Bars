"""
mlfinlab/models/cv.py
=====================
Walk-forward cross-validation with purging and embargo for financial ML.

Standard K-Fold is wrong for financial time series because:
  1. It shuffles randomly, allowing training on future data
  2. Overlapping triple-barrier windows leak label information
     from training events into the test period

This module implements:
  WalkForwardCV   -- time-ordered folds, always train-past / test-future
                     with purging and embargo to remove contaminated events

How it works
------------
Given N labeled events ordered in time, split into k folds:

  Fold 1:  TRAIN = events in fold 1        TEST = events in fold 2
  Fold 2:  TRAIN = events in folds 1-2     TEST = events in fold 3
  ...
  Fold k:  TRAIN = events in folds 1..k-1  TEST = events in fold k

For each fold, purging removes any train event whose triple-barrier
window (t0 -> t1) extends into the test period, because that event's
label was determined partly by prices in the test period.

Embargo removes train events that entered within N bars AFTER the test
period ends, because their rolling features (RSI, ATR etc.) were
computed using bars from the test period.

References
----------
de Prado, M. L. (2018). Advances in Financial Machine Learning, Ch.7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterator, Optional


class WalkForwardCV:
    """Walk-forward cross-validator with purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds. Minimum 2.
    embargo_pct : float
        Fraction of the training set size to use as embargo after
        each test period.  Typical value: 0.01 (1%).
    """

    def __init__(self,
                 n_splits         : int   = 5,
                 embargo_pct      : float = 0.01,
                 initial_train_pct: float = 0.40):
        """
        Parameters
        ----------
        n_splits : int
            Number of folds (train+test pairs = n_splits - 1).
        embargo_pct : float
            Fraction of total events used as embargo buffer after test.
        initial_train_pct : float
            Fraction of ALL events reserved for the FIRST training fold.
            With 6 years of data (2020-2025) and initial_train_pct=0.40:
                First 40% (2020-Jun 2022) always in training.
                Remaining 60% split across 4 test folds.
            This guarantees the first fold has substantial training data
            and avoids the problem of training on near-zero history.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits          = n_splits
        self.embargo_pct       = embargo_pct
        self.initial_train_pct = initial_train_pct

    def split(
        self,
        X          : pd.DataFrame,
        t1         : pd.Series,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index arrays.

        Parameters
        ----------
        X  : pd.DataFrame
            Feature matrix with DatetimeIndex. Rows must be sorted
            chronologically.
        t1 : pd.Series
            Triple-barrier exit timestamps aligned to X.index.
            t1.index = event entry time, t1.values = event exit time.

        Yields
        ------
        train_idx : np.ndarray  Integer positions of train events.
        test_idx  : np.ndarray  Integer positions of test events.
        """
        n            = len(X)
        indices      = np.arange(n)
        embargo_size = max(1, int(n * self.embargo_pct))

        # initial_train_end: position where first training block ends
        # Everything before this is ALWAYS in training (never tested)
        init_end  = int(n * self.initial_train_pct)
        remaining = n - init_end           # events available for test folds
        n_test_folds = self.n_splits - 1   # number of train/test pairs
        fold_size = max(1, remaining // n_test_folds)

        for fold in range(n_test_folds):
            # test = next fold_size events after initial training block
            test_start_pos = init_end + fold * fold_size
            test_end_pos   = min(test_start_pos + fold_size, n)
            test_idx       = indices[test_start_pos:test_end_pos]

            if len(test_idx) == 0:
                continue

            test_start_time = X.index[test_idx[0]]
            test_end_time   = X.index[test_idx[-1]]

            # train = all events BEFORE the test period
            train_candidates = indices[:test_start_pos]

            # --- PURGE -------------------------------------------------
            # Remove any train event whose barrier window ends AFTER
            # the test period starts.  That event's label was decided
            # by prices inside the test period.
            purged_train = []
            for i in train_candidates:
                event_time = X.index[i]
                t1_time    = t1.iloc[i] if i < len(t1) else pd.NaT
                # keep only events whose window closes before test starts
                if pd.isna(t1_time) or t1_time < test_start_time:
                    purged_train.append(i)

            # --- EMBARGO -----------------------------------------------
            # Remove train events that entered within embargo_size events
            # BEFORE the test period. Their rolling features (RSI, ATR)
            # were computed from bars that may include early test-period bars.
            embargo_cutoff = max(0, test_start_pos - embargo_size)
            embargoed_train = [
                i for i in purged_train if i < embargo_cutoff
            ]

            if len(embargoed_train) == 0:
                continue

            yield np.array(embargoed_train), test_idx

    def get_n_splits(self) -> int:
        return self.n_splits - 1   # n_splits folds → n_splits-1 train/test pairs