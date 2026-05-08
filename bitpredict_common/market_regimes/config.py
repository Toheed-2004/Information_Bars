from dataclasses import dataclass, field
from typing import List


@dataclass
class RegimeConfig:
    # EWMA alphas
    alpha_fast: float = 0.15
    alpha_slow: float = 0.04
    alpha_vol: float = 0.10
    alpha_expansion: float = 0.05
    alpha_z: float = 0.02
    alpha_persistence: float = 0.10
    alpha_min: float = 0.02
    alpha_max: float = 0.20

    # Trend classification
    trend_threshold: float = 1.0

    # Volatility classification
    vol_high_cutoff: float = 0.75
    vol_low_cutoff: float = 0.25

    # Momentum / acceleration
    accel_threshold: float = 1.5
    transition_high_threshold: float = 2.5
    transition_exit_threshold: float = 1.5

    # Hysteresis
    hysteresis_base: float = 0.10
    hysteresis_k: float = 0.15

    # Timing
    min_duration_bars: int = 3

    # Ring buffer
    ring_buffer_size: int = 500

    # Gap detection
    gap_multiplier: float = 5.0

    # Warmup
    warmup_epsilon: float = 0.05

    # Soft scores
    sigmoid_steepness: float = 5.0

    # Stability
    stability_cap: int = 100

    # Confidence weights [trend, vol, momentum]
    confidence_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Failure mode thresholds
    volatility_floor: float = 1e-8
    volatility_floor_bars: int = 20
