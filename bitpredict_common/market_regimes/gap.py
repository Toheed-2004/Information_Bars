"""
Gap detection and EWMA-based interval tracking.
Works for all bar types: time bars (interval = seconds), non-time bars (interval in native units).
"""
import numpy as np
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


def detect_gap(state, current_interval: float, gap_multiplier: float) -> bool:
    """
    Detect a gap for the current bar.
    Returns True if a gap is detected.
    Updates state.ewma_interval and state.interval_ewma_initialized.
    """
    logger.debug(f"DETECT_GAP ENTRY: current_interval={current_interval}, "
                f"interval_ewma_initialized={state.interval_ewma_initialized}, "
                f"ewma_interval={state.ewma_interval}, gap_multiplier={gap_multiplier}")
    
    if not state.interval_ewma_initialized:
        # Need at least one prior interval before gap detection can start.
        # The EWMA is seeded on the second bar's interval.
        state.ewma_interval = current_interval
        state.interval_ewma_initialized = True
        state.last_interval = current_interval
        logger.debug(f"DETECT_GAP: First interval, seeding ewma_interval={current_interval}")
        return False

    gap_detected = current_interval > gap_multiplier * state.ewma_interval
    
    if gap_detected:
        logger.warning(f"DETECT_GAP: GAP DETECTED! current_interval={current_interval}, "
                      f"ewma_interval={state.ewma_interval}, gap_multiplier={gap_multiplier}, "
                      f"threshold={gap_multiplier * state.ewma_interval}")
    else:
        # Update EWMA with current interval (alpha_expansion as a proxy — use 0.1 for intervals)
        alpha_interval = 0.10
        old_ewma = state.ewma_interval
        state.ewma_interval = alpha_interval * current_interval + (1 - alpha_interval) * state.ewma_interval
        logger.debug(f"DETECT_GAP: No gap, ewma_interval updated {old_ewma:.2f} → {state.ewma_interval:.2f}")

    state.last_interval = current_interval
    return gap_detected


def reset_state_on_gap(state):
    """
    Reset all EWMA state after a gap is detected.
    Preserves ring buffer size and config references.
    """
    logger.warning(f"GAP RESET START: bars_seen={state.bars_seen} → 0, "
                  f"warmup_complete={state.warmup_complete} → False, "
                  f"prev_timestamp={state.prev_timestamp} → 0.0, "
                  f"ewma_interval={state.ewma_interval}, interval_ewma_initialized={state.interval_ewma_initialized}")
    
    state.ewma_fast_close = 0.0
    state.ewma_slow_close = 0.0
    state.ewma_mean = 0.0
    state.ewma_var = 0.0
    state.ewma_vol_up = 0.0
    state.ewma_vol_down = 0.0
    state.ewma_vol_expansion = 0.0
    state.ewma_directional_persistence = 0.0

    # Clear ring buffer
    state.ring_buffer = []
    state.ring_buffer_head = 0
    state.ring_buffer_count = 0

    # Reset transition state
    state.in_transition = False
    state.last_trend_sign = 0

    # Reset pending labels
    state.pending_trend = None
    state.pending_trend_bars = 0
    state.pending_vol = None
    state.pending_vol_bars = 0
    state.pending_momentum = None
    state.pending_momentum_bars = 0

    state.committed_trend = None
    state.committed_vol = None
    state.committed_momentum = None

    # Reset stability
    state.stability_counter = 0
    state.last_regime_label = None

    # Reset warmup
    state.bars_seen = 0
    state.warmup_complete = False
    state.prev_trend_strength_z = 0.0
    state.prev_close = 0.0
    state.vol_floor_count = 0

    # Reset interval tracking so it re-seeds on next bar (CRITICAL FIX)
    state.ewma_interval = 0.0
    state.interval_ewma_initialized = False
    state.last_interval = 0.0
    
    # Reset timestamp so first bar after gap doesn't have stale prev_timestamp (CRITICAL FIX)
    state.prev_timestamp = 0.0
    
    logger.warning(f"GAP RESET COMPLETE: All state reset, ready for new warmup period")


def compute_warmup_bars(alpha_min: float, warmup_epsilon: float) -> int:
    """
    Number of bars required for EWMA seed influence to decay below epsilon.
    (1 - alpha)^N < epsilon  →  N > log(epsilon) / log(1 - alpha)
    Uses alpha_min (most conservative estimate).
    """
    import math
    if alpha_min <= 0 or alpha_min >= 1:
        return 0
    return int(np.ceil(np.log(warmup_epsilon) / np.log(1.0 - alpha_min)))


def is_warmup_complete(bars_seen: int, warmup_bars: int) -> bool:
    return bars_seen >= warmup_bars
