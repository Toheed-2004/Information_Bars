"""
TRANSITION state machine for the trend dimension.
Entry: sign change in trend_strength_z AND transition_pressure > transition_high_threshold.
Exit:  transition_pressure < transition_exit_threshold.
"""
import numpy as np


def update_transition_state(
    state,
    trend_strength_z: float,
    transition_pressure: float,
    config,
) -> bool:
    """
    Update transition state machine for one bar (incremental).
    Returns True if we are in TRANSITION after this update.
    Modifies state.in_transition and state.last_trend_sign in place.
    """
    th_high = config.transition_high_threshold
    th_exit = config.transition_exit_threshold

    current_sign = int(np.sign(trend_strength_z))

    if state.in_transition:
        # Exit condition: pressure has subsided
        if transition_pressure < th_exit:
            state.in_transition = False
    else:
        # Entry condition: sign changed AND pressure is high
        sign_changed = (current_sign != state.last_trend_sign) and (state.last_trend_sign != 0)
        if sign_changed and transition_pressure > th_high:
            state.in_transition = True

    state.last_trend_sign = current_sign
    return state.in_transition


def compute_transition_batch(
    trend_strength_z: np.ndarray,
    transition_pressure: np.ndarray,
    th_high: float,
    th_exit: float,
) -> np.ndarray:
    """
    Vectorized TRANSITION state machine simulation over a batch.
    Returns boolean array: True at bars where regime_trend == TRANSITION.
    Must replicate exact causal bar-by-bar logic — no Python loop avoidance here
    because state machine is inherently sequential.
    Uses numpy where possible but requires a single O(n) scan.
    """
    n = len(trend_strength_z)
    in_transition = np.zeros(n, dtype=bool)

    current_in_transition = False
    last_sign = 0

    signs = np.sign(trend_strength_z).astype(int)

    for i in range(n):
        s = signs[i]
        p = transition_pressure[i]

        if current_in_transition:
            if p < th_exit:
                current_in_transition = False
        else:
            sign_changed = (s != last_sign) and (last_sign != 0)
            if sign_changed and p > th_high:
                current_in_transition = True

        in_transition[i] = current_in_transition
        last_sign = s

    return in_transition
