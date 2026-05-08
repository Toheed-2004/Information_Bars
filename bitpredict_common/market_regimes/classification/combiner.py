"""
Combine three dimension labels into a single regime_label.
TRANSITION suppresses vol and momentum modifiers.
"""


def combine_labels(regime_trend: str, regime_vol: str, regime_momentum: str) -> str:
    if regime_trend == "TRANSITION":
        return "TRANSITION"
    parts = [regime_trend, regime_vol, regime_momentum]
    return "_".join(parts)


def combine_labels_batch(regime_trend, regime_vol, regime_momentum):
    """
    Vectorized batch combine. Works on arrays of strings (object dtype).
    Returns object array of combined labels.
    """
    import numpy as np
    n = len(regime_trend)
    t = np.asarray(regime_trend, dtype=object)
    v = np.asarray(regime_vol, dtype=object)
    m = np.asarray(regime_momentum, dtype=object)
    # Build combined strings using vectorized Python object operations
    combined = np.frompyfunc(lambda a, b, c: f"{a}_{b}_{c}", 3, 1)(t, v, m)
    out = combined.astype(object)
    # Override TRANSITION rows
    out[t == "TRANSITION"] = "TRANSITION"
    return out
