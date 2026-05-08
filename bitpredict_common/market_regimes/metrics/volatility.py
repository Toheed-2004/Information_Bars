"""
Volatility dimension metrics.
Causal ring-buffer percentile: bar i is ranked against past-only buffer.
All batch computation is vectorized (no Python loops over bars).
"""
import numpy as np
from bitpredict.common.market_regimes.metrics.trend import _ewma_lfilter

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def compute_volatility_batch(
    close: np.ndarray,
    alpha_vol: float,
    ring_buffer_size: int,
    alpha_min: float,
    alpha_max: float,
) -> dict:
    """
    Returns: up_vol, down_vol, volatility_level, volatility_skew,
             vol_percentile, adaptive_alpha, log_returns
    """
    n = len(close)

    # log returns
    log_returns = np.empty(n)
    log_returns[0] = 0.0
    log_returns[1:] = np.log(close[1:] / close[:-1])

    # Semi-deviations (EWMA of squared semi-returns)
    up_ret = np.maximum(log_returns, 0.0)
    dn_ret = np.maximum(-log_returns, 0.0)

    up_vol2 = _ewma_lfilter(up_ret ** 2, alpha_vol, seed=(up_ret[0] ** 2))
    dn_vol2 = _ewma_lfilter(dn_ret ** 2, alpha_vol, seed=(dn_ret[0] ** 2))

    up_vol = np.sqrt(np.maximum(up_vol2, 0.0))
    down_vol = np.sqrt(np.maximum(dn_vol2, 0.0))

    volatility_level = np.maximum(up_vol, down_vol)
    volatility_skew = down_vol / (up_vol + _EPS)

    # Causal vol percentile via ring buffer (vectorized)
    vol_percentile = _vol_percentile_batch(volatility_level, ring_buffer_size)

    # Adaptive alpha
    adaptive_alpha = alpha_min + (alpha_max - alpha_min) * vol_percentile

    return {
        "up_vol": up_vol,
        "down_vol": down_vol,
        "volatility_level": volatility_level,
        "volatility_skew": volatility_skew,
        "vol_percentile": vol_percentile,
        "adaptive_alpha": adaptive_alpha,
        "log_returns": log_returns,
    }


def _vol_percentile_batch(vol_level: np.ndarray, K: int) -> np.ndarray:
    """
    Causal percentile: vol_percentile[i] = rank(vol_level[i] among vol_level[0..i-1]) / i
    For i >= K: rank among vol_level[i-K..i-1] only (ring buffer of size K).
    vol_percentile[0] = 0.0 (no past data).
    Fully vectorized — no Python loops over bars.
    """
    n = len(vol_level)
    percentile = np.zeros(n, dtype=np.float64)

    if n <= 1:
        return percentile

    # ---- Warmup region: bars 1 .. min(n-1, K-1) ----
    # Buffer not yet full; rank vol_level[i] among vol_level[0..i-1]
    m = min(n, K)  # number of bars in warmup region (indices 0..m-1)
    if m > 1:
        # Lower-triangular comparison matrix (vectorized, no loops)
        # comp[i, j] = (vol_level[j] < vol_level[i]) and (j < i), i,j in [0..m-1]
        v = vol_level[:m]
        # strictly lower triangular mask
        lower_tri = np.tri(m, k=-1, dtype=bool)           # True where j < i
        gt_matrix = v[np.newaxis, :] < v[:, np.newaxis]   # gt_matrix[i,j] = v[j] < v[i]
        counts = np.sum(gt_matrix & lower_tri, axis=1).astype(np.float64)
        denom = np.arange(m, dtype=np.float64)             # denom[i] = i (past bar count)
        denom[0] = 1.0                                     # avoid /0 at bar 0
        percentile[:m] = counts / denom
        percentile[0] = 0.0                                # bar 0: no past values

    # ---- Full-buffer region: bars K .. n-1 ----
    if n > K:
        # sliding_window_view(vol_level[:-1], K)[j] = vol_level[j .. j+K-1]
        # For bar i = j+K → past K values are vol_level[i-K .. i-1] ✓
        windows = np.lib.stride_tricks.sliding_window_view(vol_level[:-1], K)
        # windows shape: (n-K, K)  (only available when n-1 >= K, i.e. n > K)
        if windows.shape[0] > 0:
            current = vol_level[K:]          # shape (n-K,)
            # Compare: count how many past values are strictly less
            ranks = np.sum(windows < current[:, np.newaxis], axis=1)
            percentile[K:] = ranks / K

    return percentile


# ---------------------------------------------------------------------------
# Incremental (O(1))
# ---------------------------------------------------------------------------

def update_volatility(state, close: float, prev_close: float, config) -> dict:
    alpha_vol = config.alpha_vol
    alpha_min = config.alpha_min
    alpha_max = config.alpha_max
    K = config.ring_buffer_size

    if state.bars_seen == 0:
        log_ret = 0.0
        up_r = 0.0
        dn_r = 0.0
    else:
        log_ret = np.log(close / prev_close) if prev_close > 0 else 0.0
        up_r = max(log_ret, 0.0)
        dn_r = max(-log_ret, 0.0)

    if state.bars_seen == 0:
        # Seed with bar 0 values
        state.ewma_vol_up = up_r ** 2
        state.ewma_vol_down = dn_r ** 2
    else:
        state.ewma_vol_up = alpha_vol * (up_r ** 2) + (1 - alpha_vol) * state.ewma_vol_up
        state.ewma_vol_down = alpha_vol * (dn_r ** 2) + (1 - alpha_vol) * state.ewma_vol_down

    up_vol = float(np.sqrt(max(state.ewma_vol_up, 0.0)))
    down_vol = float(np.sqrt(max(state.ewma_vol_down, 0.0)))
    volatility_level = max(up_vol, down_vol)
    volatility_skew = down_vol / (up_vol + _EPS)

    # Causal ring buffer percentile
    # Rank current value against PAST buffer contents BEFORE appending
    buf = state.ring_buffer
    buf_count = state.ring_buffer_count
    if buf_count == 0:
        vol_percentile = 0.0
    else:
        # Get the valid buffer contents
        if buf_count < K:
            past = buf[:buf_count]
        else:
            # Full circular buffer
            past = buf  # list of K elements
        past_arr = np.array(past, dtype=np.float64)
        vol_percentile = float(np.sum(past_arr < volatility_level) / len(past_arr))

    # Append current value to ring buffer
    if len(buf) < K:
        buf.append(volatility_level)
        state.ring_buffer_count += 1
    else:
        # Circular overwrite
        buf[state.ring_buffer_head] = volatility_level
        state.ring_buffer_head = (state.ring_buffer_head + 1) % K

    adaptive_alpha = alpha_min + (alpha_max - alpha_min) * vol_percentile

    return {
        "up_vol": up_vol,
        "down_vol": down_vol,
        "volatility_level": volatility_level,
        "volatility_skew": volatility_skew,
        "vol_percentile": vol_percentile,
        "adaptive_alpha": adaptive_alpha,
        "log_return": log_ret,
    }
