from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RegimeState:
    # EWMA scalars
    ewma_fast_close: float = 0.0
    ewma_slow_close: float = 0.0
    ewma_mean: float = 0.0
    ewma_var: float = 0.0
    ewma_vol_up: float = 0.0
    ewma_vol_down: float = 0.0
    ewma_vol_expansion: float = 0.0
    ewma_interval: float = 0.0
    ewma_directional_persistence: float = 0.0

    # Confidence normalization EWMAs (must be in state for correct DB restore)
    conf_ewma_mean_tsz: float = 0.0
    conf_ewma_var_tsz: float = 0.0
    conf_ewma_mean_tp: float = 0.0
    conf_ewma_var_tp: float = 0.0

    # Ring buffer for vol percentile
    ring_buffer: List[float] = field(default_factory=list)
    ring_buffer_head: int = 0      # write index (circular)
    ring_buffer_count: int = 0     # number of valid entries

    # Transition state machine
    in_transition: bool = False
    last_trend_sign: int = 0       # -1, 0, or 1

    # Pending labels (for min_duration)
    pending_trend: Optional[str] = None
    pending_trend_bars: int = 0
    pending_vol: Optional[str] = None
    pending_vol_bars: int = 0
    pending_momentum: Optional[str] = None
    pending_momentum_bars: int = 0

    # Committed labels
    committed_trend: Optional[str] = None
    committed_vol: Optional[str] = None
    committed_momentum: Optional[str] = None

    # Stability
    stability_counter: int = 0
    last_regime_label: Optional[str] = None

    # Warmup
    bars_seen: int = 0
    warmup_complete: bool = False

    # Previous bar values needed for derivatives
    prev_trend_strength_z: float = 0.0
    prev_close: float = 0.0
    prev_timestamp: float = 0.0    # for gap detection in incremental mode

    # Consecutive bars below volatility floor
    vol_floor_count: int = 0

    # Last interval seen (for gap detection)
    last_interval: float = 0.0
    interval_ewma_initialized: bool = False

    def to_dict(self) -> dict:
        return {
            "ewma_fast_close": self.ewma_fast_close,
            "ewma_slow_close": self.ewma_slow_close,
            "ewma_mean": self.ewma_mean,
            "ewma_var": self.ewma_var,
            "ewma_vol_up": self.ewma_vol_up,
            "ewma_vol_down": self.ewma_vol_down,
            "ewma_vol_expansion": self.ewma_vol_expansion,
            "ewma_interval": self.ewma_interval,
            "ewma_directional_persistence": self.ewma_directional_persistence,
            "conf_ewma_mean_tsz": self.conf_ewma_mean_tsz,
            "conf_ewma_var_tsz": self.conf_ewma_var_tsz,
            "conf_ewma_mean_tp": self.conf_ewma_mean_tp,
            "conf_ewma_var_tp": self.conf_ewma_var_tp,
            "ring_buffer": list(self.ring_buffer),
            "ring_buffer_head": self.ring_buffer_head,
            "ring_buffer_count": self.ring_buffer_count,
            "in_transition": self.in_transition,
            "last_trend_sign": self.last_trend_sign,
            "pending_trend": self.pending_trend,
            "pending_trend_bars": self.pending_trend_bars,
            "pending_vol": self.pending_vol,
            "pending_vol_bars": self.pending_vol_bars,
            "pending_momentum": self.pending_momentum,
            "pending_momentum_bars": self.pending_momentum_bars,
            "committed_trend": self.committed_trend,
            "committed_vol": self.committed_vol,
            "committed_momentum": self.committed_momentum,
            "stability_counter": self.stability_counter,
            "last_regime_label": self.last_regime_label,
            "bars_seen": self.bars_seen,
            "warmup_complete": self.warmup_complete,
            "prev_trend_strength_z": self.prev_trend_strength_z,
            "prev_close": self.prev_close,
            "prev_timestamp": self.prev_timestamp,
            "vol_floor_count": self.vol_floor_count,
            "last_interval": self.last_interval,
            "interval_ewma_initialized": self.interval_ewma_initialized,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeState":
        s = cls()
        for k, v in d.items():
            setattr(s, k, v)
        return s
