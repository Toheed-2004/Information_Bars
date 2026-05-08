"""
features/sample_weights.py
--------------------------
Sample uniqueness weights (De Prado AFML Ch.4).

Triple-barrier labels overlap: bar i's label uses prices i+1..i+k,
bar i+1's label uses i+2..i+k+1, etc. These are NOT independent.
Training uniformly overstates effective sample size.

Fix: weight each sample by 1 / average_concurrent_windows.
Bars deep in the middle (many overlapping neighbours) get low weight.
Bars near the end (fewer overlaps) get higher weight.
Weights normalised so mean = 1.0 (sklearn convention: sum to n).
"""
from __future__ import annotations
import numpy as np


def compute_sample_weights(n: int, max_holding_bars: int = 10) -> np.ndarray:
    """
    Parameters
    ----------
    n                : Number of training samples.
    max_holding_bars : Label horizon (vertical barrier length).

    Returns
    -------
    float32 array shape (n,), mean = 1.0.
    """
    # Count concurrent label windows at each time point
    concurrent = np.zeros(n, dtype=np.float64)
    for i in range(n):
        concurrent[i: min(i + max_holding_bars, n)] += 1.0

    # Uniqueness of sample i = mean(1/concurrent) over its label window
    uniq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        end = min(i + max_holding_bars, n)
        uniq[i] = (1.0 / concurrent[i:end]).mean()

    # Normalise
    w = uniq / uniq.mean()
    return w.astype(np.float32)
