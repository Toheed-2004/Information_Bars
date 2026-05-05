"""
bars/bar_types/__init__.py
--------------------------
Registry of all minute-level and tick-level bar type classes.
"""
from .base import BaseBar
from .hybrid_bars import HybridBar as TickHybridBar  
from .volatility_bars import VolatilityBar as TickVolatilityBar
from .volume_bars import VolumeBar as TickVolumeBar
from .renko_bars import RenkoBar as TickRenkoBar
from .range_bars import RangeBar as TickRangeBar

# Minute-level bar classes
from .dollar_bars     import DollarBar
from .volume_bars     import VolumeBar
from .volatility_bars import VolatilityBar
from .range_bars      import RangeBar
from .renko_bars      import RenkoBar
from .hybrid_bars     import HybridBar

# Tick-level bar classes
from .tick_dollar_bars     import TickDollarBar
# from .tick_volume_bars     import TickVolumeBar
# from .tick_volatility_bars import TickVolatilityBar
# from .tick_range_bars      import TickRangeBar
# from .tick_renko_bars      import TickRenkoBar
# from .tick_hybrid_bars     import TickHybridBar

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
