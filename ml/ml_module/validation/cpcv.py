"""
ml_module/validation/cpcv.py
------------------------------
Purged Combinatorial Purged Cross-Validation (CPCV).

References
----------
López de Prado, M. (2018). Advances in Financial Machine Learning.
  Chapter 12 — "Backtesting through Cross-Validation".

Why not sklearn TimeSeriesSplit?
---------------------------------
sklearn.TimeSeriesSplit does NOT:
  - purge overlapping label windows between train and test
  - apply embargo gaps to prevent leakage from future observations
  - produce combinatorial (path-based) test sets

This implementation does all three.  No scikit-learn CV utilities are used.

Architecture
------------
CPCVSplitter   — generates (train_idx, test_idx) index pairs.
purge_indices  — removes train samples whose label windows overlap test.
embargo_indices— removes train samples within an embargo gap of test start.

Usage
-----
    splitter = CPCVSplitter(n_splits=6, n_test_splits=2, embargo_bars=5)
    for train_idx, test_idx in splitter.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
"""
from __future__ import annotations

import itertools
from typing import Generator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml_module.utils.helpers import get_logger

logger = get_logger(__name__)

# Type aliases
IndexArray = np.ndarray  # integer positional indices


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def purge_indices(
    train_pos: IndexArray,
    test_pos:  IndexArray,
    label_end_pos: Optional[np.ndarray] = None,
) -> IndexArray:
    """
    Remove training samples whose label window overlaps with the test set.

    A label generated at bar *i* uses price information up to bar
    ``label_end_pos[i]``.  If that extends into the test window, the
    sample leaks future price action and must be purged.

    Parameters
    ----------
    train_pos      : Integer positional indices of the training set.
    test_pos       : Integer positional indices of the test set.
    label_end_pos  : For each bar, the last positional index used by its
                     label (e.g. i + max_holding_bars).  If None, no purging
                     beyond position overlap is applied.

    Returns
    -------
    Purged train_pos (indices that do NOT overlap with test set).
    """
    if len(train_pos) == 0 or len(test_pos) == 0:
        return train_pos

    test_start = test_pos.min()
    test_end   = test_pos.max()

    if label_end_pos is None:
        # Simple position-based purge: drop any train bar inside test window
        return train_pos[~np.isin(train_pos, test_pos)]

    # Purge train bars whose label_end touches or exceeds test_start
    label_ends_for_train = label_end_pos[train_pos]
    keep_mask = label_ends_for_train < test_start
    return train_pos[keep_mask]


def embargo_indices(
    train_pos:    IndexArray,
    test_pos:     IndexArray,
    embargo_bars: int,
) -> IndexArray:
    """
    Remove training samples within *embargo_bars* bars before the test set.

    The embargo prevents the model from learning patterns that are
    contaminated by bars immediately adjacent to the test window.

    Parameters
    ----------
    train_pos    : Current training positional indices.
    test_pos     : Test positional indices.
    embargo_bars : Number of bars to exclude before test_start.

    Returns
    -------
    Embargoed train_pos.
    """
    if embargo_bars <= 0 or len(test_pos) == 0:
        return train_pos

    test_start = test_pos.min()
    embargo_start = max(0, test_start - embargo_bars)
    embargo_zone  = np.arange(embargo_start, test_start)
    return train_pos[~np.isin(train_pos, embargo_zone)]


# ---------------------------------------------------------------------------
# CPCV Splitter
# ---------------------------------------------------------------------------

class CPCVSplitter:
    """
    Combinatorial Purged Cross-Validation.

    Splits data into ``n_splits`` groups.  Each iteration holds out
    ``n_test_splits`` adjacent groups as the test set, training on all
    remaining groups (after purging and embargo).

    The combinatorial approach generates C(n_splits, n_test_splits) unique
    test paths — providing a more thorough estimate of out-of-sample
    performance than a single walk-forward pass.

    Parameters
    ----------
    n_splits      : Total number of groups to split data into (like k in k-fold).
    n_test_splits : Number of groups that form each test set.
    embargo_bars  : Bars to exclude from the train end before each test window.
    min_train_size: Minimum number of training samples per fold; folds with
                    fewer training samples are skipped.
    max_holding_bars: Label horizon — used for overlap-aware purging.
                      Pass None to use simple position-overlap purge.
    """

    def __init__(
        self,
        n_splits:        int = 6,
        n_test_splits:   int = 2,
        embargo_bars:    int = 5,
        min_train_size:  int = 100,
        max_holding_bars: Optional[int] = None,
    ):
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be < n_splits")
        self.n_splits         = n_splits
        self.n_test_splits    = n_test_splits
        self.embargo_bars     = embargo_bars
        self.min_train_size   = min_train_size
        self.max_holding_bars = max_holding_bars

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Generator[Tuple[IndexArray, IndexArray], None, None]:
        """
        Generate (train_positions, test_positions) index pairs.

        Parameters
        ----------
        X : Feature DataFrame (sorted by time, ascending).
        y : Target Series (same index as X).  Not used directly but mirrors
            sklearn's API for compatibility.

        Yields
        ------
        train_pos, test_pos : Integer positional arrays into X/y.
        """
        n         = len(X)
        group_idx = self._make_groups(n)
        combos    = list(itertools.combinations(range(self.n_splits), self.n_test_splits))

        # Pre-compute label-end positions for purging
        label_end_pos = None
        if self.max_holding_bars is not None:
            label_end_pos = np.minimum(
                np.arange(n) + self.max_holding_bars, n - 1
            )

        total_yielded = 0
        for combo in combos:
            test_groups  = set(combo)
            train_groups = [g for g in range(self.n_splits) if g not in test_groups]

            test_pos  = np.concatenate([group_idx[g] for g in sorted(test_groups)])
            train_pos = np.concatenate([group_idx[g] for g in train_groups])

            # Purge overlapping labels
            train_pos = purge_indices(train_pos, test_pos, label_end_pos)
            # Apply embargo
            train_pos = embargo_indices(train_pos, test_pos, self.embargo_bars)

            if len(train_pos) < self.min_train_size:
                logger.debug(
                    "Fold skipped: train_size=%d < min_train_size=%d",
                    len(train_pos), self.min_train_size,
                )
                continue

            logger.debug(
                "CPCV fold test_groups=%s  train=%d  test=%d",
                combo, len(train_pos), len(test_pos),
            )
            total_yielded += 1
            yield np.sort(train_pos), np.sort(test_pos)

        logger.info(
            "CPCVSplitter: %d folds generated (C(%d,%d)=%d combos, %d skipped)",
            total_yielded, self.n_splits, self.n_test_splits, len(combos),
            len(combos) - total_yielded,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_groups(self, n: int) -> List[IndexArray]:
        """Split n samples into n_splits roughly equal groups."""
        all_pos = np.arange(n)
        splits  = np.array_split(all_pos, self.n_splits)
        return splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of splitting iterations (sklearn-compatible signature)."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def __repr__(self) -> str:
        return (
            f"CPCVSplitter(n_splits={self.n_splits}, "
            f"n_test_splits={self.n_test_splits}, "
            f"embargo_bars={self.embargo_bars})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_cpcv(cfg: Dict, max_holding_bars: Optional[int] = None) -> CPCVSplitter:
    """
    Instantiate a CPCVSplitter from the ``cpcv`` config section.

    Parameters
    ----------
    cfg              : ``cpcv`` section of the YAML config.
    max_holding_bars : Forwarded from the ``labeling`` section for purge logic.
    """
    from typing import Dict  # local to avoid circular imports
    return CPCVSplitter(
        n_splits         = cfg.get("n_splits",       6),
        n_test_splits    = cfg.get("n_test_splits",  2),
        embargo_bars     = cfg.get("embargo_bars",   5),
        min_train_size   = cfg.get("min_train_size", 100),
        max_holding_bars = max_holding_bars,
    )
