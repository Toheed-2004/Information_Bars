"""
bars/bar_types/__init__.py
--------------------------
Registry of all minute-level and tick-level bar type classes.

BUG-FIX 1 (Critical — Wrong tick class registry):
    The original file aliased 5 minute-bar classes as their tick equivalents:
        from .hybrid_bars    import HybridBar    as TickHybridBar
        from .volatility_bars import VolatilityBar as TickVolatilityBar
        from .volume_bars    import VolumeBar    as TickVolumeBar
        from .renko_bars     import RenkoBar     as TickRenkoBar
        from .range_bars     import RangeBar     as TickRangeBar
    Then the actual tick-module imports were COMMENTED OUT.
    Effect: TICK_BAR_TYPES contained minute-bar classes for 5 of 6 types.
    When processor.py called get_bar_class(bar_type, source='tick'),
    it got a minute-bar class — causing tick-native calibration, the
    exact tick accumulation logic, and duration_seconds tracking to be
    completely bypassed. Bars were built using minute-bar EMA logic
    and minute-level accumulate_bar_data even on tick data, producing
    structurally incorrect results for the tick pipeline comparison.

    Fix: import each class from its actual tick module. All tick imports
    are now active (not commented out).
"""
from .base import BaseBar

# ── Minute-level bar classes ──────────────────────────────────────────────────
from .dollar_bars     import DollarBar
from .volume_bars     import VolumeBar
from .volatility_bars import VolatilityBar
from .range_bars      import RangeBar
from .renko_bars      import RenkoBar
from .hybrid_bars     import HybridBar

# ── Tick-level bar classes — BUG-FIX 1: import from the correct tick modules ─
from .tick_dollar_bars     import TickDollarBar
from .tick_volume_bars     import TickVolumeBar       # FIX: was VolumeBar alias
from .tick_volatility_bars import TickVolatilityBar   # FIX: was VolatilityBar alias
from .tick_range_bars      import TickRangeBar        # FIX: was RangeBar alias
from .tick_renko_bars      import TickRenkoBar        # FIX: was RenkoBar alias
from .tick_hybrid_bars     import TickHybridBar       # FIX: was HybridBar alias

# ── Registries ────────────────────────────────────────────────────────────────
MINUTE_BAR_TYPES: dict = {
    "dollar":     DollarBar,
    "volume":     VolumeBar,
    "volatility": VolatilityBar,
    "range":      RangeBar,
    "renko":      RenkoBar,
    "hybrid":     HybridBar,
}

TICK_BAR_TYPES: dict = {
    "dollar":     TickDollarBar,
    "volume":     TickVolumeBar,
    "volatility": TickVolatilityBar,
    "range":      TickRangeBar,
    "renko":      TickRenkoBar,
    "hybrid":     TickHybridBar,
}

ALL_BAR_TYPE_NAMES = list(MINUTE_BAR_TYPES.keys())


def get_bar_class(bar_type: str, source: str = "minute") -> type:
    """
    Return the bar class for *bar_type* and *source*.

    Args:
        bar_type: One of dollar / volume / volatility / range / renko / hybrid.
        source:   "minute" or "tick".

    Raises:
        ValueError: If bar_type or source is unrecognised.
    """
    registry = TICK_BAR_TYPES if source == "tick" else MINUTE_BAR_TYPES
    if bar_type not in registry:
        raise ValueError(
            f"Unknown bar type {bar_type!r} for source {source!r}.  "
            f"Available: {list(registry)}"
        )
    return registry[bar_type]
