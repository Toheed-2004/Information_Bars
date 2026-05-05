"""
bars/db.py — DB helpers for the bars module.

Re-exports the common service functions used by bars, and provides
batch_insert_bars for single-transaction bulk inserts.
"""

from bitpredict.common.db.services.data import (
    get_bar_state,
    update_bar_state,
    get_historical_bars_with_regime,
    get_recent_bars_for_regime,
    update_bars_with_regime,
    batch_insert_bars,
    read_bar,
    upsert_bar_stats,
)

__all__ = [
    "get_bar_state",
    "update_bar_state",
    "get_historical_bars_with_regime",
    "get_recent_bars_for_regime",
    "update_bars_with_regime",
    "batch_insert_bars",
    "read_bar",
    "upsert_bar_stats",
]
