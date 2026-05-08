"""
Trend dimension metrics.
All batch functions are fully vectorized (no Python loops over bars).
Z-score normalization uses PREVIOUS bar's ewma_mean / ewma_var (causal).
"""
import numpy as np
from scipy.signal import lfilter


_EPS = 1e-10


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def compute_trend_batch(
    close: np.ndarray,
    alpha_fast: float,
    alpha_slow: float,
    alpha_z: float,
    alpha_persistence: float,
) -> dict:
    """
    Returns a dict of 1-D arrays, all length n:
      ewma_fast, ewma_slow, trend_raw,
      ewma_mean, ewma_var,
      trend_strength_z, directional_persistence, trend_acceleration
    """
    n = len(close)

    # --- ewma_fast / ewma_slow ---
    ewma_fast = _ewma_lfilter(close, alpha_fast, seed=close[0])
    ewma_slow = _ewma_lfilter(close, alpha_slow, seed=close[0])

    # --- trend_raw = log(ewma_fast / ewma_slow) ---
    trend_raw = np.log(ewma_fast / ewma_slow)

    # --- causal z-score ---
    # ewma_mean[i] = alpha_z*trend_raw[i] + (1-alpha_z)*ewma_mean[i-1], seeded at trend_raw[0]
    ewma_mean = _ewma_lfilter(trend_raw, alpha_z, seed=trend_raw[0])

    # ewma_var[i] = alpha_z*(trend_raw[i] - ewma_mean[i])^2 + (1-alpha_z)*ewma_var[i-1], seed=0
    residuals2 = (trend_raw - ewma_mean) ** 2
    ewma_var = _ewma_lfilter(residuals2, alpha_z, seed=0.0)

    # Shift mean/var by 1 to get *previous* bar's statistics for z-score
    # ewma_mean_prev[0] = trend_raw[0] (the seed), ewma_mean_prev[i] = ewma_mean[i-1]
    ewma_mean_prev = np.empty(n)
    ewma_mean_prev[0] = trend_raw[0]
    ewma_mean_prev[1:] = ewma_mean[:-1]

    ewma_var_prev = np.empty(n)
    ewma_var_prev[0] = 0.0
    ewma_var_prev[1:] = ewma_var[:-1]

    trend_strength_z = np.zeros(n)
    # bar 0 → 0 by definition (no prior stats)
    if n > 1:
        trend_strength_z[1:] = (
            (trend_raw[1:] - ewma_mean_prev[1:]) /
            np.sqrt(ewma_var_prev[1:] + _EPS)
        )

    # --- directional persistence ---
    log_returns = np.empty(n)
    log_returns[0] = 0.0
    log_returns[1:] = np.log(close[1:] / close[:-1])
    sign_returns = np.sign(log_returns)
    directional_persistence = _ewma_lfilter(sign_returns, alpha_persistence, seed=sign_returns[0])

    # --- trend acceleration ---
    trend_acceleration = np.empty(n)
    trend_acceleration[0] = 0.0
    trend_acceleration[1:] = trend_strength_z[1:] - trend_strength_z[:-1]

    return {
        "ewma_fast": ewma_fast,
        "ewma_slow": ewma_slow,
        "trend_raw": trend_raw,
        "ewma_mean": ewma_mean,
        "ewma_var": ewma_var,
        "trend_strength_z": trend_strength_z,
        "directional_persistence": directional_persistence,
        "trend_acceleration": trend_acceleration,
        "log_returns": log_returns,
    }


# ---------------------------------------------------------------------------
# Incremental (O(1))
# ---------------------------------------------------------------------------

def update_trend(state, close: float, prev_close: float, config) -> dict:
    """
    Update trend metrics for a single new bar.
    Returns a dict with scalar metric values for this bar.
    Modifies state in place.
    """
    alpha_fast = config.alpha_fast
    alpha_slow = config.alpha_slow
    alpha_z = config.alpha_z
    alpha_persistence = config.alpha_persistence

    if state.bars_seen == 0:
        # Seed EWMAs
        state.ewma_fast_close = close
        state.ewma_slow_close = close
        trend_raw = 0.0  # log(1) = 0
        # Seed z-score stats: ewma_mean = trend_raw[0], ewma_var = 0
        state.ewma_mean = trend_raw
        state.ewma_var = 0.0
        trend_strength_z = 0.0  # bar 0 by definition
        state.prev_trend_strength_z = 0.0
        trend_acceleration = 0.0
        log_ret = 0.0
        dp = 0.0
        state.ewma_directional_persistence = 0.0
    else:
        # Update fast/slow EWMAs
        state.ewma_fast_close = alpha_fast * close + (1 - alpha_fast) * state.ewma_fast_close
        state.ewma_slow_close = alpha_slow * close + (1 - alpha_slow) * state.ewma_slow_close
        trend_raw = np.log(state.ewma_fast_close / state.ewma_slow_close)

        # Causal z-score: normalize using PREVIOUS bar's stats
        prev_mean = state.ewma_mean
        prev_var = state.ewma_var
        trend_strength_z = (trend_raw - prev_mean) / np.sqrt(prev_var + _EPS)

        # Update stats AFTER computing z-score
        state.ewma_mean = alpha_z * trend_raw + (1 - alpha_z) * state.ewma_mean
        state.ewma_var = (
            alpha_z * (trend_raw - state.ewma_mean) ** 2 +
            (1 - alpha_z) * state.ewma_var
        )

        # Directional persistence
        log_ret = np.log(close / prev_close) if prev_close > 0 else 0.0
        sign_ret = float(np.sign(log_ret))
        state.ewma_directional_persistence = (
            alpha_persistence * sign_ret +
            (1 - alpha_persistence) * state.ewma_directional_persistence
        )
        dp = state.ewma_directional_persistence

        # Trend acceleration
        trend_acceleration = trend_strength_z - state.prev_trend_strength_z

    state.prev_trend_strength_z = trend_strength_z

    return {
        "trend_raw": trend_raw,
        "trend_strength_z": trend_strength_z,
        "directional_persistence": state.ewma_directional_persistence,
        "trend_acceleration": trend_acceleration,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ewma_lfilter(values: np.ndarray, alpha: float, seed: float) -> np.ndarray:
    """
    EWMA via scipy lfilter. Seeded at `seed` (out[-1] = seed before first bar).
    out[i] = alpha * values[i] + (1-alpha) * out[i-1]
    With seed as the "previous" value before bar 0:
      out[0] = alpha * values[0] + (1-alpha) * seed
    When seed == values[0], this gives out[0] = values[0].
    """
    beta = 1.0 - alpha
    # lfilter(b, a, x, zi) with b=[alpha], a=[1,-beta], zi = [beta * seed]
    # → out[0] = alpha*x[0] + beta*seed
    zi = np.array([beta * seed])
    out, _ = lfilter([alpha], [1.0, -beta], values, zi=zi)
    return out
