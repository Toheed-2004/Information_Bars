"""
Regime confidence score (Layer 1) and per-dimension confidence.
Batch mode is fully vectorized via lfilter — no Python loops over bars.
"""
import numpy as np

from bitpredict.common.market_regimes.metrics.trend import _ewma_lfilter

_EPS = 1e-10
_ALPHA_CONF = 0.02


def compute_confidence(
    trend_strength_z: float,
    vol_percentile: float,
    transition_pressure: float,
    trend_threshold: float,
    accel_threshold: float,
    transition_pressure_ewma_std: float,
    trend_ewma_std: float,
    weights: list,
) -> dict:
    """Compute confidence scalars for one bar (incremental)."""
    trend_conf = float(np.clip(
        (abs(trend_strength_z) - trend_threshold) / (trend_ewma_std + _EPS), 0.0, 1.0
    ))
    vol_conf = float(np.clip(
        abs(vol_percentile - 0.5) / 0.25, 0.0, 1.0
    ))
    accel_conf = float(np.clip(
        (transition_pressure - accel_threshold) / (transition_pressure_ewma_std + _EPS), 0.0, 1.0
    ))

    w = weights
    total_w = sum(w)
    regime_confidence = (w[0] * trend_conf + w[1] * vol_conf + w[2] * accel_conf) / (total_w + _EPS)

    return {
        "regime_confidence": float(np.clip(regime_confidence, 0.0, 1.0)),
        "trend_confidence": trend_conf,
        "vol_confidence": vol_conf,
        "accel_confidence": accel_conf,
    }


def compute_confidence_batch(
    trend_strength_z: np.ndarray,
    vol_percentile: np.ndarray,
    transition_pressure: np.ndarray,
    trend_threshold: float,
    accel_threshold: float,
    weights: list,
) -> dict:
    """
    Vectorized confidence for a batch.
    All EWMAs computed via lfilter — no Python loops over bars.
    """
    abs_tsz = np.abs(trend_strength_z)

    # EWMA mean then variance of |trend_strength_z| — same pattern as compute_trend_batch
    ewma_tsz_mean = _ewma_lfilter(abs_tsz, _ALPHA_CONF, seed=abs_tsz[0])
    residuals2_tsz = (abs_tsz - ewma_tsz_mean) ** 2
    ewma_tsz_var = _ewma_lfilter(residuals2_tsz, _ALPHA_CONF, seed=0.0)
    trend_ewma_std = np.sqrt(np.maximum(ewma_tsz_var, 0.0)) + _EPS

    # EWMA mean then variance of transition_pressure
    ewma_tp_mean = _ewma_lfilter(transition_pressure, _ALPHA_CONF, seed=transition_pressure[0])
    residuals2_tp = (transition_pressure - ewma_tp_mean) ** 2
    ewma_tp_var = _ewma_lfilter(residuals2_tp, _ALPHA_CONF, seed=0.0)
    tp_ewma_std = np.sqrt(np.maximum(ewma_tp_var, 0.0)) + _EPS

    trend_conf = np.clip((np.abs(trend_strength_z) - trend_threshold) / trend_ewma_std, 0.0, 1.0)
    vol_conf = np.clip(np.abs(vol_percentile - 0.5) / 0.25, 0.0, 1.0)
    accel_conf = np.clip((transition_pressure - accel_threshold) / tp_ewma_std, 0.0, 1.0)

    w = weights
    total_w = sum(w)
    regime_confidence = np.clip(
        (w[0] * trend_conf + w[1] * vol_conf + w[2] * accel_conf) / (total_w + _EPS),
        0.0, 1.0
    )

    return {
        "regime_confidence": regime_confidence,
        "trend_confidence": trend_conf,
        "vol_confidence": vol_conf,
        "accel_confidence": accel_conf,
        "_trend_ewma_std": trend_ewma_std,
        "_tp_ewma_std": tp_ewma_std,
        "_ewma_tsz_mean": ewma_tsz_mean,
        "_ewma_tsz_var": ewma_tsz_var,
        "_ewma_tp_mean": ewma_tp_mean,
        "_ewma_tp_var": ewma_tp_var,
    }
