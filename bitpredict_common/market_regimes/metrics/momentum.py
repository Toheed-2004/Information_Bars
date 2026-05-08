"""
Momentum / acceleration dimension metrics.
Computed purely from raw metrics — no dependency on classification outputs.
"""
import numpy as np
from bitpredict.common.market_regimes.metrics.trend import _ewma_lfilter

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def compute_momentum_batch(
    volatility_level: np.ndarray,
    trend_strength_z: np.ndarray,
    alpha_expansion: float,
) -> dict:
    """
    Returns: vol_expansion_factor, d_trend_z, transition_pressure
    """
    n = len(volatility_level)

    # EWMA of volatility_level (expansion denominator)
    ewma_vol_exp = _ewma_lfilter(volatility_level, alpha_expansion, seed=volatility_level[0])

    # vol_expansion_factor = volatility_level / EWMA(volatility_level)
    vol_expansion_factor = volatility_level / (ewma_vol_exp + _EPS)

    # d_trend_z = |trend_strength_z[i] - trend_strength_z[i-1]|
    d_trend_z = np.empty(n)
    d_trend_z[0] = 0.0
    d_trend_z[1:] = np.abs(np.diff(trend_strength_z))

    # transition_pressure = d_trend_z * vol_expansion_factor
    transition_pressure = d_trend_z * vol_expansion_factor

    return {
        "vol_expansion_factor": vol_expansion_factor,
        "d_trend_z": d_trend_z,
        "transition_pressure": transition_pressure,
        "ewma_vol_expansion": ewma_vol_exp,
    }


# ---------------------------------------------------------------------------
# Incremental (O(1))
# ---------------------------------------------------------------------------

def update_momentum(state, volatility_level: float, trend_strength_z: float, prev_trend_strength_z: float, config) -> dict:
    """
    prev_trend_strength_z must be the z-score from the PREVIOUS bar
    (captured before update_trend() overwrites state.prev_trend_strength_z).
    """
    alpha_expansion = config.alpha_expansion

    if state.bars_seen == 0:
        state.ewma_vol_expansion = volatility_level
        vol_expansion_factor = 1.0
        d_trend_z = 0.0
    else:
        state.ewma_vol_expansion = (
            alpha_expansion * volatility_level +
            (1 - alpha_expansion) * state.ewma_vol_expansion
        )
        vol_expansion_factor = volatility_level / (state.ewma_vol_expansion + _EPS)
        d_trend_z = abs(trend_strength_z - prev_trend_strength_z)

    transition_pressure = d_trend_z * vol_expansion_factor

    return {
        "vol_expansion_factor": vol_expansion_factor,
        "d_trend_z": d_trend_z,
        "transition_pressure": transition_pressure,
    }
