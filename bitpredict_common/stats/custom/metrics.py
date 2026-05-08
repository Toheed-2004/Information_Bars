"""
Thin re-export shim — all implementations live in shared/utils.py.
"""
from ..shared.utils import (
    _max_consecutive_numpy,
    _rolling_windows,
    _calculate_avg_return,
    _calculate_geometric_mean,
)

__all__ = [
    '_max_consecutive_numpy',
    '_rolling_windows',
    '_calculate_avg_return',
    '_calculate_geometric_mean',
]
