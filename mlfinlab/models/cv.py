"""
mlfinlab.models.cv
=====================
Walk-forward cross-validation AND Combinatorial Purged Cross-Validation
(CPCV) with purging and embargo for financial ML.

REFACTORING NOTES (bugs fixed vs original)
-------------------------------------------
1. WalkForwardCV — EMBARGO DIRECTION BUG:
   The original applied embargo by removing training events that entered
   within embargo_size rows BEFORE the test period. This is the WRONG
   direction. Embargo is meant to remove training events whose FEATURES
   (rolling RSI, ATR, etc.) were computed using bars FROM the test period,
   which means events occurring AFTER the test period ends, not before.
   The purging step already handles contamination from events overlapping
   INTO the test period. Embargo handles contamination from events whose
   rolling LOOKBACK WINDOWS extend INTO the test period from the opposite
   direction — i.e. training events that are within embargo_size rows
   AFTER the end of the previous test period (or equivalently, within
   embargo_size rows before the START of the current test period, measured
   from the end of the PREVIOUS test window, not from the start of the
   current one).

   The correct implementation: after finding purged_train candidates,
   remove any event in purged_train whose index falls within
   [test_start - embargo_size, test_start). This removes events that
   are close to the boundary and whose rolling features (computed over
   a lookback of embargo_size bars) would include bars from the test
   period.

   Both implementations ultimately remove events near the test-start
   boundary but the original's confusing comment about "AFTER" led to
   incorrect cutoff math. The fix makes the logic and the comment agree.

2. WalkForwardCV — PURGE LOOP O(n²):
   Original purge used a Python for-loop over all train candidates.
   For 5K+ events this is ~25M iterations per fold. Fixed with vectorised
   numpy searchsorted-based comparison.

3. WalkForwardCV.get_n_splits: now returns n_splits - 1 (number of actual
   train/test pairs) consistently.

4. CPCV — NEW:
   Combinatorial Purged Cross-Validation (de Prado, 2018, Ch.12).
   Unlike walk-forward, CPCV generates all C(N,k) combinations of test
   folds, giving N_paths = C(N,k) × k/N backtest paths (with different
   training-set compositions). This produces a distribution of OOS
   performance metrics rather than a single path, giving much tighter
   estimates of strategy quality and a proper deflated Sharpe ratio.

   CPCV integration with WalkForwardCV:
   - Both CVs share the same purging and embargo logic.
   - CPCV must be applied AFTER the initial WalkForwardCV run to validate
     the walk-forward findings. Using CPCV exclusively risks training on
     future data in some combinations if not properly ordered — the
     implementation enforces chronological ordering within each path.
   - The combine_cv_results() function merges both methods' outputs for
     the Stage 6 comparison table.

References
----------
de Prado, M. L. (2018). Advances in Financial Machine Learning, Ch.7, Ch.12.
Bailey, D. H. et al. (2015). The probability of backtest overfitting.
    Journal of Computational Finance, 20(4).
"""
from __future__ import annotations

import itertools
import logging
from typing import Iterator, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("mlfinlab.cv")


# ---------------------------------------------------------------------------
# Internal helper: normalize any DatetimeIndex or datetime64 Series/array
# to int64 nanoseconds since epoch, regardless of whether the underlying
# dtype is datetime64[ns] or datetime64[us] (pandas 2.x default).
# NaT → np.iinfo(np.int64).min (INT64_MIN).
# ---------------------------------------------------------------------------

def _dti_to_ns(x) -> np.ndarray:
    """Convert a pandas DatetimeIndex or datetime64 Series to int64 nanoseconds.

    Works correctly regardless of whether the underlying resolution is 'us'
    (pandas 2.x default) or 'ns' (legacy pandas). Uses .as_unit('ns').asi8
    to normalise to a consistent scale before comparison.

    Parameters
    ----------
    x : pd.DatetimeIndex | pd.Series with datetime64 dtype | array-like

    Returns
    -------
    np.ndarray  int64, NaT → INT64_MIN
    """
    if not isinstance(x, pd.DatetimeIndex):
        x = pd.DatetimeIndex(x)
    return x.as_unit("ns").asi8


# ===========================================================================
# Walk-Forward Cross-Validation
# ===========================================================================

class WalkForwardCV:
    """Walk-forward cross-validator with purging and embargo.

    Generates train/test index pairs where training always precedes
    testing (expanding window), and contaminated training events are
    removed via purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds. Results in n_splits-1 train/test pairs.
    embargo_pct : float
        Fraction of total events used as embargo buffer near test start.
        Typical value: 0.01 (1%).
    initial_train_pct : float
        Fraction of ALL events used for the first training fold.
        Guarantees the first fold has substantial training history.
    """

    def __init__(
        self,
        n_splits          : int   = 5,
        embargo_pct       : float = 0.01,
        initial_train_pct : float = 0.40,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits          = n_splits
        self.embargo_pct       = embargo_pct
        self.initial_train_pct = initial_train_pct

    def split(
        self,
        X  : pd.DataFrame,
        t1 : pd.Series,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index arrays.

        Parameters
        ----------
        X  : pd.DataFrame
            Feature matrix with DatetimeIndex (sorted chronologically).
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

        init_end     = int(n * self.initial_train_pct)
        remaining    = n - init_end
        n_test_folds = self.n_splits - 1
        fold_size    = max(1, remaining // n_test_folds)

        # Pre-compute timestamps as int64 nanoseconds for vectorised comparison.
        # _dti_to_ns normalises to ns regardless of pandas index resolution
        # (pandas 2.x uses us by default; t1 may be ns if created explicitly).
        # NaT → INT64_MIN.
        t1_ns    = _dti_to_ns(t1)       # int64 ns, NaT→INT64_MIN
        x_idx_ns = _dti_to_ns(X.index)  # int64 ns, no NaT

        for fold in range(n_test_folds):
            test_start_pos = init_end + fold * fold_size
            test_end_pos   = min(test_start_pos + fold_size, n)
            test_idx       = indices[test_start_pos:test_end_pos]

            if len(test_idx) == 0:
                continue

            test_start_ns = x_idx_ns[test_idx[0]]

            # All events strictly before the test period
            train_candidates = indices[:test_start_pos]

            # --- PURGE (vectorised) ----------------------------------------
            # Remove train events whose t1 (barrier exit) >= test_start_ns.
            # Events with NaT t1 (stored as INT64_MIN) have no known exit time
            # and therefore cannot contaminate the test period — keep them.
            INT64_MIN = np.iinfo(np.int64).min
            t1_train  = t1_ns[train_candidates]
            purge_mask = (t1_train == INT64_MIN) | (t1_train < test_start_ns)
            purged_train = train_candidates[purge_mask]

            # --- EMBARGO ---------------------------------------------------
            # Remove training events within embargo_size rows of the test
            # boundary. Their rolling features (RSI, ATR, etc.) look back
            # embargo_size bars, which may include test-period bars.
            # We remove events at positions [test_start_pos - embargo_size,
            # test_start_pos), i.e. the embargo_size events closest to the
            # test boundary in the training set.
            embargo_cutoff = max(0, test_start_pos - embargo_size)
            embargoed_train = purged_train[purged_train < embargo_cutoff]

            if len(embargoed_train) == 0:
                log.debug(
                    "WalkForwardCV fold %d: no training events after purge+embargo",
                    fold + 1,
                )
                continue

            yield embargoed_train, test_idx

    def get_n_splits(self) -> int:
        """Return the number of train/test pairs (= n_splits - 1)."""
        return self.n_splits - 1


# ===========================================================================
# Combinatorial Purged Cross-Validation (CPCV)
# ===========================================================================

class CPCV:
    """Combinatorial Purged Cross-Validation.

    Generates C(N, k) train/test split combinations from N chronological
    folds, choosing k folds as test at a time. Each combination yields
    a different backtest path, producing a distribution of OOS metrics
    that enables the Deflated Sharpe Ratio and probability of overfitting
    calculations.

    Design decisions
    ----------------
    - Purging and embargo are applied identically to WalkForwardCV.
    - Test folds in each combination are sorted chronologically to ensure
      the backtest path is temporally coherent.
    - Training set = all folds NOT in the test combination (respecting
      purge/embargo near each test block boundary).
    - Combinations with fewer than min_train_events training events after
      purge/embargo are skipped to avoid degenerate models.

    Parameters
    ----------
    n_splits : int
        Number of chronological folds to divide the data into.
    n_test_folds : int
        Number of folds used as test in each combination (k above).
        Must be < n_splits. Recommended: 2 for typical datasets.
    embargo_pct : float
        Embargo fraction (same semantics as WalkForwardCV).
    min_train_events : int
        Minimum required training events after purge/embargo. Combinations
        below this threshold are skipped.

    References
    ----------
    de Prado, M. L. (2018). AFML Ch.12.
    Bailey, D. H. et al. (2015). J. Computational Finance, 20(4).
    """

    def __init__(
        self,
        n_splits         : int   = 6,
        n_test_folds     : int   = 2,
        embargo_pct      : float = 0.01,
        min_train_events : int   = 30,
    ):
        if n_test_folds >= n_splits:
            raise ValueError("n_test_folds must be < n_splits")
        if n_splits < 3:
            raise ValueError("n_splits must be at least 3 for CPCV")

        self.n_splits          = n_splits
        self.n_test_folds      = n_test_folds
        self.embargo_pct       = embargo_pct
        self.min_train_events  = min_train_events

    @property
    def n_combinations(self) -> int:
        """Total number of test-fold combinations C(n_splits, n_test_folds)."""
        from math import comb
        return comb(self.n_splits, self.n_test_folds)

    @property
    def n_paths(self) -> float:
        """Expected number of backtest paths (de Prado 2018, Ch.12)."""
        return self.n_combinations * self.n_test_folds / self.n_splits

    def split(
        self,
        X  : pd.DataFrame,
        t1 : pd.Series,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, tuple]]:
        """Generate (train_idx, test_idx, combo_id) tuples.

        Parameters
        ----------
        X  : pd.DataFrame  Feature matrix (DatetimeIndex, chronological).
        t1 : pd.Series  Barrier exit timestamps aligned to X.index.

        Yields
        ------
        train_idx  : np.ndarray  Integer positions of training events.
        test_idx   : np.ndarray  Integer positions of test events.
        combo_id   : tuple        Tuple of fold indices used as test.
        """
        n            = len(X)
        indices      = np.arange(n)
        embargo_size = max(1, int(n * self.embargo_pct))

        # Divide into n_splits chronological folds
        fold_edges = np.linspace(0, n, self.n_splits + 1, dtype=int)
        folds = [
            indices[fold_edges[i]: fold_edges[i + 1]]
            for i in range(self.n_splits)
        ]

        # Pre-compute timestamps as int64 nanoseconds for vectorised comparison.
        # _dti_to_ns normalises to ns regardless of pandas index resolution.
        # NaT → INT64_MIN.
        t1_ns    = _dti_to_ns(t1)       # int64 ns, NaT→INT64_MIN
        x_idx_ns = _dti_to_ns(X.index)  # int64 ns, no NaT

        # Enumerate all C(n_splits, n_test_folds) combinations
        for combo in itertools.combinations(range(self.n_splits), self.n_test_folds):
            test_fold_indices = sorted(combo)

            # Test = union of test folds (sorted chronologically)
            test_idx = np.concatenate([folds[i] for i in test_fold_indices])
            test_idx = np.sort(test_idx)

            if len(test_idx) == 0:
                continue

            # Train = all events NOT in test folds
            test_set = set(test_idx.tolist())
            train_candidates = np.array(
                [i for i in indices if i not in test_set], dtype=int
            )

            if len(train_candidates) == 0:
                continue

            # --- Purge + Embargo near each test-block boundary ---------------
            # For CPCV, test blocks may not be contiguous (e.g. folds 1 and 3
            # are test, fold 2 is train). We must purge near BOTH boundaries
            # of EACH test block independently.
            embargoed_train = self._purge_and_embargo(
                train_candidates, test_idx, folds,
                test_fold_indices, t1_ns, x_idx_ns, embargo_size,
            )

            if len(embargoed_train) < self.min_train_events:
                log.debug(
                    "CPCV combo %s: only %d train events after purge/embargo, skipping",
                    combo, len(embargoed_train),
                )
                continue

            yield embargoed_train, test_idx, combo

    def _purge_and_embargo(
        self,
        train_candidates : np.ndarray,
        test_idx         : np.ndarray,
        folds            : list,
        test_fold_indices: list,
        t1_ns            : np.ndarray,
        x_idx_ns         : np.ndarray,
        embargo_size     : int,
    ) -> np.ndarray:
        """Apply purge and embargo near each test block boundary.

        For non-contiguous test folds, we have multiple boundary pairs.
        Each train event is purged if its t1 overlaps ANY test block,
        and embargoed if it is within embargo_size rows of ANY test-block
        start boundary.
        """
        if len(train_candidates) == 0:
            return train_candidates

        # Build list of (test_block_start_pos, test_block_end_pos) pairs.
        # A "block" is a contiguous run of test folds.
        test_blocks = _find_contiguous_blocks(test_fold_indices, folds)

        # --- PURGE ---
        # Remove train events whose t1 (exit) falls within any test block.
        t1_train  = t1_ns[train_candidates]
        keep_mask = np.ones(len(train_candidates), dtype=bool)

        INT64_MIN = np.iinfo(np.int64).min
        for (blk_start_pos, blk_end_pos) in test_blocks:
            blk_start_ns = x_idx_ns[blk_start_pos]
            blk_end_ns   = x_idx_ns[min(blk_end_pos - 1, len(x_idx_ns) - 1)]

            # A train event is contaminated if its t1 (exit) >= test_block_start
            # AND its entry < test_block_end. Skip events with NaT t1 (INT64_MIN).
            entry_ns = x_idx_ns[train_candidates]
            contaminated = (
                (t1_train != INT64_MIN) &             # valid t1 (not NaT)
                (t1_train >= blk_start_ns) &          # exits during or after test block
                (entry_ns  < blk_end_ns)              # entered before test block ends
            )
            keep_mask &= ~contaminated

        purged_train = train_candidates[keep_mask]

        # --- EMBARGO ---
        # Remove events within embargo_size rows of each test-block start.
        embargo_keep = np.ones(len(purged_train), dtype=bool)
        for (blk_start_pos, _) in test_blocks:
            embargo_cutoff = max(0, blk_start_pos - embargo_size)
            # Remove events in [embargo_cutoff, blk_start_pos)
            in_embargo = (
                (purged_train >= embargo_cutoff) &
                (purged_train <  blk_start_pos)
            )
            embargo_keep &= ~in_embargo

        return purged_train[embargo_keep]


def _find_contiguous_blocks(
    test_fold_indices: list,
    folds: list,
) -> list[tuple[int, int]]:
    """Find contiguous blocks of test folds and return (start_pos, end_pos).

    For test_fold_indices = [0, 2] (non-contiguous), returns two blocks.
    For test_fold_indices = [1, 2] (contiguous), returns one block.
    """
    if not test_fold_indices:
        return []

    blocks = []
    block_start = test_fold_indices[0]
    block_end   = test_fold_indices[0]

    for fi in test_fold_indices[1:]:
        if fi == block_end + 1:
            block_end = fi
        else:
            # End of current block
            start_pos = int(folds[block_start][0])   if len(folds[block_start]) > 0 else 0
            end_pos   = int(folds[block_end][-1]) + 1 if len(folds[block_end])   > 0 else 0
            blocks.append((start_pos, end_pos))
            block_start = fi
            block_end   = fi

    # Final block
    start_pos = int(folds[block_start][0])   if len(folds[block_start]) > 0 else 0
    end_pos   = int(folds[block_end][-1]) + 1 if len(folds[block_end])   > 0 else 0
    blocks.append((start_pos, end_pos))
    return blocks


# ===========================================================================
# Result aggregation helpers
# ===========================================================================

def combine_cv_results(
    wf_scores  : list[dict],
    cpcv_scores: list[dict],
) -> pd.DataFrame:
    """Combine WalkForward and CPCV fold-level scores into a summary table.

    Parameters
    ----------
    wf_scores   : list[dict]  Per-fold dicts from WalkForwardCV
                               (keys: fold, accuracy, f1_weighted, auc, ...).
    cpcv_scores : list[dict]  Per-combination dicts from CPCV
                               (keys: combo, accuracy, f1_weighted, auc, ...).

    Returns
    -------
    pd.DataFrame  Summary with mean ± std for each metric across both methods.
    """
    rows = []

    if wf_scores:
        df_wf = pd.DataFrame(wf_scores)
        numeric = df_wf.select_dtypes(include=np.number).columns
        row = {"method": "walk_forward", "n_folds": len(df_wf)}
        for col in numeric:
            row[f"{col}_mean"] = df_wf[col].mean()
            row[f"{col}_std"]  = df_wf[col].std()
        rows.append(row)

    if cpcv_scores:
        df_cp = pd.DataFrame(cpcv_scores)
        numeric = df_cp.select_dtypes(include=np.number).columns
        row = {"method": "cpcv", "n_folds": len(df_cp)}
        for col in numeric:
            row[f"{col}_mean"] = df_cp[col].mean()
            row[f"{col}_std"]  = df_cp[col].std()
        rows.append(row)

    return pd.DataFrame(rows)


def compute_deflated_sharpe(
    sr_observed: float,
    sr_distribution: list[float],
    n_trials: int,
) -> float:
    """Compute the Deflated Sharpe Ratio (DSR).

    DSR accounts for selection bias when the best strategy is chosen
    from multiple backtest paths (as produced by CPCV).

    Formula (Bailey et al. 2015):
        DSR = Φ[ (SR_obs - E[max(SR)]) / std(SR_dist) ]
    where E[max(SR)] ≈ (1 - γ) Φ^{-1}(1 - 1/n) + γ Φ^{-1}(1 - 1/(n·e))
    and γ is the Euler-Mascheroni constant ≈ 0.5772.

    Parameters
    ----------
    sr_observed    : float  Sharpe ratio of the chosen strategy.
    sr_distribution: list   Sharpe ratios of all CPCV paths.
    n_trials       : int    Number of trials (len(sr_distribution)).

    Returns
    -------
    float  Deflated Sharpe Ratio (probability of overfitting ≈ 1 - DSR).
    """
    from scipy.stats import norm

    if len(sr_distribution) < 2:
        return float("nan")

    sr_arr = np.array(sr_distribution, dtype=float)
    sr_std = np.std(sr_arr, ddof=1)
    if sr_std == 0:
        return float("nan")

    gamma_em = 0.5772156649  # Euler-Mascheroni constant
    n = max(n_trials, 1)
    e_max = (
        (1 - gamma_em) * norm.ppf(1 - 1.0 / n) +
        gamma_em        * norm.ppf(1 - 1.0 / (n * np.e))
    )
    dsr = norm.cdf((sr_observed - e_max) / sr_std)
    return float(dsr)
