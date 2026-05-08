"""
Hysteresis and minimum-duration pending state per dimension.
Applied independently to each dimension (except TRANSITION).
"""
import numpy as np


def compute_hysteresis_buffer(vol_percentile: float, hysteresis_base: float, hysteresis_k: float) -> float:
    return hysteresis_base + hysteresis_k * vol_percentile


# ---------------------------------------------------------------------------
# Incremental hysteresis + min_duration state machine
# ---------------------------------------------------------------------------

def apply_hysteresis_and_min_duration(
    state,
    dim: str,           # "trend", "vol", "momentum"
    raw_label: str,
    activating_metric: float,
    threshold: float,
    vol_percentile: float,
    config,
) -> str:
    """
    Apply hysteresis and min_duration for one dimension, one bar.
    Returns the committed label for this bar.
    Updates state.pending_<dim> and state.committed_<dim>.
    """
    hb = compute_hysteresis_buffer(vol_percentile, config.hysteresis_base, config.hysteresis_k)
    min_dur = config.min_duration_bars

    pending_label_attr = f"pending_{dim}"
    pending_bars_attr = f"pending_{dim}_bars"
    committed_attr = f"committed_{dim}"

    committed = getattr(state, committed_attr)
    pending = getattr(state, pending_label_attr)
    pending_bars = getattr(state, pending_bars_attr)

    # If committed is None (first bar), commit raw_label immediately
    if committed is None:
        setattr(state, committed_attr, raw_label)
        setattr(state, pending_label_attr, None)
        setattr(state, pending_bars_attr, 0)
        return raw_label

    # Check if raw_label differs from current committed
    if raw_label != committed:
        # Must exceed threshold by more than hysteresis_buffer
        # activating_metric is the signed distance from threshold
        if activating_metric > hb:
            # Start or continue pending
            if pending == raw_label:
                pending_bars += 1
            else:
                pending = raw_label
                pending_bars = 1

            if pending_bars >= min_dur:
                # Commit the pending label
                setattr(state, committed_attr, pending)
                setattr(state, pending_label_attr, None)
                setattr(state, pending_bars_attr, 0)
                return pending
            else:
                setattr(state, pending_label_attr, pending)
                setattr(state, pending_bars_attr, pending_bars)
                return committed
        else:
            # Hysteresis buffer not exceeded; cancel pending
            setattr(state, pending_label_attr, None)
            setattr(state, pending_bars_attr, 0)
            return committed
    else:
        # Raw label matches committed; cancel any pending
        setattr(state, pending_label_attr, None)
        setattr(state, pending_bars_attr, 0)
        return committed


# ---------------------------------------------------------------------------
# Batch hysteresis + min_duration simulation
# ---------------------------------------------------------------------------

def apply_hysteresis_and_min_duration_batch(
    raw_labels: np.ndarray,
    activating_distances: np.ndarray,
    vol_percentiles: np.ndarray,
    hysteresis_base: float,
    hysteresis_k: float,
    min_duration_bars: int,
) -> np.ndarray:
    """
    Apply hysteresis + min_duration to a batch of raw labels.
    Returns committed labels array.
    Sequential state machine — single O(n) scan.
    """
    n = len(raw_labels)
    committed_labels = np.empty(n, dtype=object)

    committed = None
    pending = None
    pending_bars = 0

    for i in range(n):
        raw = raw_labels[i]
        dist = activating_distances[i]
        hb = hysteresis_base + hysteresis_k * vol_percentiles[i]

        if committed is None:
            committed = raw
            pending = None
            pending_bars = 0
            committed_labels[i] = committed
            continue

        # TRANSITION bypasses hysteresis and min_duration entirely
        # (mirrors incremental mode which sets committed immediately and clears pending)
        if raw == "TRANSITION":
            committed = "TRANSITION"
            pending = None
            pending_bars = 0
            committed_labels[i] = committed
            continue

        if raw != committed:
            if dist > hb:
                if pending == raw:
                    pending_bars += 1
                else:
                    pending = raw
                    pending_bars = 1

                if pending_bars >= min_duration_bars:
                    committed = pending
                    pending = None
                    pending_bars = 0
            else:
                pending = None
                pending_bars = 0
        else:
            pending = None
            pending_bars = 0

        committed_labels[i] = committed

    return committed_labels


def activation_distance_trend(trend_strength_z: np.ndarray, trend_threshold: float) -> np.ndarray:
    """Absolute distance from nearest trend threshold."""
    return np.abs(np.abs(trend_strength_z) - trend_threshold)


def activation_distance_vol(vol_percentile: np.ndarray, vol_high_cutoff: float, vol_low_cutoff: float) -> np.ndarray:
    """Distance from nearest vol threshold."""
    dist_high = np.abs(vol_percentile - vol_high_cutoff)
    dist_low = np.abs(vol_percentile - vol_low_cutoff)
    return np.minimum(dist_high, dist_low)


def activation_distance_momentum(transition_pressure: np.ndarray, accel_threshold: float) -> np.ndarray:
    """Distance from accel threshold."""
    return np.abs(transition_pressure - accel_threshold)
