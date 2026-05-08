"""
Layer 3 soft scores — independent sigmoid activations.
No softmax; multiple scores can be high simultaneously.
"""
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _sigmoid_scalar(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-max(-50.0, min(50.0, x)))))


def compute_soft_scores(
    trend_strength_z: float,
    vol_percentile: float,
    transition_pressure: float,
    trend_threshold: float,
    vol_high_cutoff: float,
    vol_low_cutoff: float,
    accel_threshold: float,
    transition_high_threshold: float,
    sigmoid_steepness: float,
) -> dict:
    k = sigmoid_steepness
    return {
        "score_bull":         _sigmoid_scalar((trend_strength_z - trend_threshold) * k),
        "score_bear":         _sigmoid_scalar((-trend_strength_z - trend_threshold) * k),
        "score_range":        _sigmoid_scalar((trend_threshold - abs(trend_strength_z)) * k),
        "score_transition":   _sigmoid_scalar((transition_pressure - transition_high_threshold) * k),
        "score_high_vol":     _sigmoid_scalar((vol_percentile - vol_high_cutoff) * k),
        "score_low_vol":      _sigmoid_scalar((vol_low_cutoff - vol_percentile) * k),
        "score_accelerating": _sigmoid_scalar((transition_pressure - accel_threshold) * k),
    }


def compute_soft_scores_batch(
    trend_strength_z: np.ndarray,
    vol_percentile: np.ndarray,
    transition_pressure: np.ndarray,
    trend_threshold: float,
    vol_high_cutoff: float,
    vol_low_cutoff: float,
    accel_threshold: float,
    transition_high_threshold: float,
    sigmoid_steepness: float,
) -> dict:
    k = sigmoid_steepness
    return {
        "score_bull":         _sigmoid((trend_strength_z - trend_threshold) * k),
        "score_bear":         _sigmoid((-trend_strength_z - trend_threshold) * k),
        "score_range":        _sigmoid((trend_threshold - np.abs(trend_strength_z)) * k),
        "score_transition":   _sigmoid((transition_pressure - transition_high_threshold) * k),
        "score_high_vol":     _sigmoid((vol_percentile - vol_high_cutoff) * k),
        "score_low_vol":      _sigmoid((vol_low_cutoff - vol_percentile) * k),
        "score_accelerating": _sigmoid((transition_pressure - accel_threshold) * k),
    }
