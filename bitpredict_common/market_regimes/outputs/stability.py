"""
Regime stability: consecutive bars in the current regime_label, normalized to [0,1].
"""
import numpy as np


def update_stability(state, regime_label: str, stability_cap: int) -> float:
    if regime_label == state.last_regime_label:
        state.stability_counter += 1
    else:
        state.stability_counter = 1
        state.last_regime_label = regime_label
    return min(state.stability_counter / stability_cap, 1.0)


def compute_stability_batch(regime_labels: np.ndarray, stability_cap: int) -> np.ndarray:
    """
    Vectorized stability counter.
    stability[i] = consecutive bars (including i) with same regime_label, normalized.
    """
    n = len(regime_labels)
    stability = np.zeros(n, dtype=np.float64)

    counter = 0
    last_label = None
    for i in range(n):
        lbl = regime_labels[i]
        if lbl == last_label:
            counter += 1
        else:
            counter = 1
            last_label = lbl
        stability[i] = min(counter / stability_cap, 1.0)

    return stability
