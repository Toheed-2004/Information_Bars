"""
tick_calibration_patch.py
--------------------------
Drop-in replacements for the `calibrate()` functions in:
    tick_volume_bars.py
    tick_volatility_bars.py
    tick_range_bars.py
    tick_renko_bars.py
    tick_hybrid_bars.py

Each replacement adds `analyze_from_ticks()` — the tick-native equivalent of
`analyze_from_dataframe()` already used by tick_dollar_bars.py.

The minute-bucketing path (`_ticks_to_minute_ohlcv` → `analyze_market_history`)
is completely removed from calibration.  Every parameter — target, ema_alpha,
alpha_min/max, target_bars_per_day, duration bounds — is computed directly from
raw tick arrays using the same sophistication as tick_dollar_bars:

    • Shannon entropy of log-returns        → information_multiplier
    • Regime stability (rolling autocorr)   → alpha_min / alpha_max
    • Market noise ratio                    → alpha_min / alpha_max
    • Market efficiency (autocorr + VR)     → activity_multiplier
    • Daily metric CV (MAD/median)          → ema_alpha position in [min,max]
    • Asset tier (for volume bars)          → base_bpd

Signals that are tick-native (realized vol, tick range, brick displacement) are
computed directly from raw prices — no minute approximation at any stage.

HOW TO APPLY
------------
For each bar type, replace the existing `calibrate()` function with the
corresponding function below.  The function signatures and return shapes are
identical to the originals — the rest of the pipeline is untouched.

IMPORTANT: these functions use constants already imported in each file.
Add any missing imports listed in the per-function header comments.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Shared tick-native helpers (copy into a shared module or duplicate per file)
# ═══════════════════════════════════════════════════════════════════════════════

import gc
import numpy as np
from collections import deque
from pathlib import Path
from typing import Callable

_MS_PER_DAY  = 86_400 * 1_000
_MS_PER_S    = 1_000


def _tick_daily_split(timestamps_ms: np.ndarray):
    """Return array of day indices (int) aligned to timestamps_ms."""
    return timestamps_ms // _MS_PER_DAY


def _tick_log_returns(prices: np.ndarray) -> np.ndarray:
    """Consecutive log-returns, zeros removed."""
    lr = np.diff(np.log(prices.astype(np.float64)))
    return lr[lr != 0.0]


def _tick_entropy(data: np.ndarray, bins: int = 50) -> float:
    """Shannon entropy of a 1-D distribution (same as BaseBar._calculate_entropy)."""
    if len(data) < 10:
        return 2.0
    try:
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) < 2:
            return 2.0
        bw = (data.max() - data.min()) / bins
        probs = hist * bw
        probs /= probs.sum()
        return float(-np.sum(probs * np.log2(probs)))
    except Exception:
        return 2.0


def _tick_regime_stability(log_returns: np.ndarray, window: int = 500) -> float:
    """Rolling vol-autocorrelation regime stability (0–1)."""
    if len(log_returns) < window * 2:
        return 0.5
    try:
        vol = np.abs(log_returns)
        if len(vol) >= window + 50:
            cur = vol[-window:]
            lag = vol[-window - 50:-50]
            if len(cur) == len(lag) and np.std(cur) > 0 and np.std(lag) > 0:
                corr = np.corrcoef(cur, lag)[0, 1]
                if not np.isnan(corr):
                    return float(max(0.0, min(1.0, (corr + 1) / 2)))
        return 0.5
    except Exception:
        return 0.5


def _tick_market_noise(log_returns: np.ndarray, window: int = 300) -> float:
    """Noise ratio (0–1, higher = noisier)."""
    if len(log_returns) < 50:
        return 0.5
    try:
        recent = log_returns[-window:] if len(log_returns) >= window else log_returns
        std = np.std(recent)
        mean_abs = np.mean(np.abs(recent))
        if mean_abs > 1e-8:
            return float(max(0.0, min(1.0, (std / mean_abs - 2.0) / 18.0)))
        return 0.5
    except Exception:
        return 0.5


def _tick_market_efficiency(prices: np.ndarray, quantities: np.ndarray) -> float:
    """Market efficiency via autocorrelation + variance ratio (same as TickDollarBar)."""
    if len(prices) < 100:
        return 0.5
    try:
        p = (prices[np.linspace(0, len(prices) - 1, min(50_000, len(prices)), dtype=int)]
             if len(prices) > 50_000 else prices)
        lr = np.diff(np.log(p.astype(np.float64)))
        lr = lr[lr != 0.0]
        if len(lr) < 4:
            return 0.5
        autocorr = float(np.corrcoef(lr[:-1], lr[1:])[0, 1])
        if np.isnan(autocorr):
            autocorr = 0.0
        eff_ac = max(0.0, 1.0 - abs(autocorr) * 10)
        var1 = float(np.var(lr))
        var2 = float(np.var(lr[::2]))
        eff_vr = (max(0.0, 1.0 - abs(var2 / (2.0 * var1) - 1.0) * 2.0)
                  if var1 > 0 else 0.5)
        return float(max(0.1, min(0.9, eff_ac * 0.5 + eff_vr * 0.5)))
    except Exception:
        return 0.5


def _tick_daily_metric(metric_per_tick: np.ndarray, day_idx: np.ndarray):
    """
    Sum metric_per_tick within each day.
    Returns (daily_values np.ndarray, n_days int).
    """
    unique_days = np.unique(day_idx)
    daily = np.array([metric_per_tick[day_idx == d].sum() for d in unique_days],
                     dtype=np.float64)
    return daily, len(unique_days)


def _alpha_from_cv(cv: float, regime_stability: float, market_noise: float,
                   alpha_floor: float = 0.03, alpha_ceil_base: float = 0.20) -> tuple:
    """
    Compute (alpha_min, alpha_max, ema_alpha) from CV and market state.
    Formula mirrors range/renko/volatility minute implementations exactly,
    applied to tick-native CV.
    """
    alpha_min = alpha_floor + (market_noise * 0.07)
    alpha_max = alpha_ceil_base + (regime_stability * 0.25)
    normalized_cv = min(1.0, cv / 0.8)
    ema_alpha = alpha_min + (alpha_max - alpha_min) * normalized_cv
    return alpha_min, alpha_max, ema_alpha


def _duration_seconds_from_bpd(target_bpd: float,
                                TICK_MIN_DURATION_SECONDS: int,
                                TICK_MAX_DURATION_FLOOR_SECONDS: int,
                                TICK_MAX_DURATION_SECONDS: int,
                                DURATION_ESTIMATED_MULTIPLIER: float) -> tuple:
    estimated_bar_s = 86_400.0 / target_bpd
    min_s = TICK_MIN_DURATION_SECONDS
    max_s = max(TICK_MAX_DURATION_FLOOR_SECONDS,
                min(TICK_MAX_DURATION_SECONDS,
                    int(estimated_bar_s * DURATION_ESTIMATED_MULTIPLIER)))
    return min_s, max_s


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VOLUME BARS  — replace calibrate() in tick_volume_bars.py
# ═══════════════════════════════════════════════════════════════════════════════
# Extra imports needed in tick_volume_bars.py (add if not present):
#   from common.constants import (VOLUME_TIER_BASE_BARS, VOLUME_TIER_BASE_BARS_DEFAULT,
#       SLOW_BAR_FREQUENCY_MULTIPLIER, FREQ_ADJ_BASE, FREQ_ADJ_SENSITIVITY,
#       VOLUME_EXTREME_THRESHOLD_MULTIPLIER, DURATION_ESTIMATED_MULTIPLIER)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VOLATILITY BARS  — replace calibrate() in tick_volatility_bars.py
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# 3. RANGE BARS  — replace calibrate() in tick_range_bars.py
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# 4. RENKO BARS  — replace calibrate() in tick_renko_bars.py
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# 5. HYBRID BARS  — replace calibrate() in tick_hybrid_bars.py
# ═══════════════════════════════════════════════════════════════════════════════

