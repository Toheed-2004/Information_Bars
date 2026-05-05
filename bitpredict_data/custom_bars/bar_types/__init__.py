"""
Bar types module for creating different types of bars from minute-level data.
"""

from bitpredict.data.custom_bars.bar_types.base import BaseBar
from bitpredict.data.custom_bars.bar_types.volume import VolumeBar
from bitpredict.data.custom_bars.bar_types.volatility import VolatilityBar
from bitpredict.data.custom_bars.bar_types.dollar import DollarBar
from bitpredict.data.custom_bars.bar_types.range_bar import RangeBar
from bitpredict.data.custom_bars.bar_types.renko import RenkoBar
from bitpredict.data.custom_bars.bar_types.hybrid import HybridBar

# Registry of available bar types
BAR_TYPES = {
    "volume":     VolumeBar,
    "volatility": VolatilityBar,
    "dollar":     DollarBar,
    "range":      RangeBar,
    "renko":      RenkoBar,
    "hybrid":     HybridBar,
}


def get_bar_class(bar_type: str):
    """Get the bar class for a given bar type."""
    if bar_type not in BAR_TYPES:
        raise ValueError(
            f"Unknown bar type: {bar_type!r}. Available: {list(BAR_TYPES.keys())}"
        )
    return BAR_TYPES[bar_type]
