"""
Classify each dimension independently from raw metrics.
No cross-dimension conditioning here.
"""
import numpy as np


def classify_trend_raw(trend_strength_z: float, trend_threshold: float) -> str:
    """Returns BULL / BEAR / RANGE based on z-score magnitude. TRANSITION handled separately."""
    if trend_strength_z > trend_threshold:
        return "BULL"
    elif trend_strength_z < -trend_threshold:
        return "BEAR"
    return "RANGE"


def classify_vol(vol_percentile: float, vol_high_cutoff: float, vol_low_cutoff: float) -> str:
    if vol_percentile > vol_high_cutoff:
        return "HIGH_VOL"
    elif vol_percentile < vol_low_cutoff:
        return "LOW_VOL"
    return "NORMAL_VOL"


def classify_momentum(transition_pressure: float, accel_threshold: float) -> str:
    if transition_pressure > accel_threshold:
        return "ACCELERATING"
    return "STABLE"


# ---------------------------------------------------------------------------
# Vectorized helpers for batch (used after hysteresis and min_duration)
# ---------------------------------------------------------------------------

def classify_trend_raw_batch(trend_strength_z: np.ndarray, trend_threshold: float) -> np.ndarray:
    labels = np.full(len(trend_strength_z), "RANGE", dtype=object)
    labels[trend_strength_z > trend_threshold] = "BULL"
    labels[trend_strength_z < -trend_threshold] = "BEAR"
    return labels


def classify_vol_batch(
    vol_percentile: np.ndarray, vol_high_cutoff: float, vol_low_cutoff: float
) -> np.ndarray:
    labels = np.full(len(vol_percentile), "NORMAL_VOL", dtype=object)
    labels[vol_percentile > vol_high_cutoff] = "HIGH_VOL"
    labels[vol_percentile < vol_low_cutoff] = "LOW_VOL"
    return labels


def classify_momentum_batch(
    transition_pressure: np.ndarray, accel_threshold: float
) -> np.ndarray:
    labels = np.full(len(transition_pressure), "STABLE", dtype=object)
    labels[transition_pressure > accel_threshold] = "ACCELERATING"
    return labels
