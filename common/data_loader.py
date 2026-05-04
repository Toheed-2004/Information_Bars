"""
common/data_loader.py
---------------------
CSV-based data loading — replaces the database layer for
standalone / reproducible use.

All minute OHLCV and tick data are read directly from CSV files
in data/raw_data/.  Bar outputs are persisted to data/processed_bars/.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)


# ── Minute OHLCV ─────────────────────────────────────────────────────────────

def load_minute_csv(
    path: str | Path,
    start_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load minute OHLCV data from a CSV file.

    Expected columns: datetime (or timestamp), open, high, low, close, volume.

    Args:
        path:       Path to the CSV file.
        start_date: If supplied, return only rows after this timestamp.

    Returns:
        DataFrame with columns [datetime, open, high, low, close, volume],
        sorted by datetime, datetime column is UTC-aware.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Minute CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    # Normalise datetime column name
    dt_col = next(
        (c for c in df.columns
         if c.lower() in ("datetime", "timestamp", "time", "date")),
        None,
    )
    if dt_col is None:
        raise ValueError(
            f"No datetime column found in {path}.  Columns: {list(df.columns)}"
        )
    if dt_col != "datetime":
        df = df.rename(columns={dt_col: "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = (
        df.dropna(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if start_date is not None:
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        df = df[df["datetime"] > pd.Timestamp(start_date)]

    logger.info("Loaded %d minute rows from %s", len(df), path.name)
    return df.reset_index(drop=True)


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a minute DataFrame to a list of dicts for bar processors."""
    return [
        {
            "datetime": row.datetime,
            "open":     float(row.open),
            "high":     float(row.high),
            "low":      float(row.low),
            "close":    float(row.close),
            "volume":   float(row.volume),
        }
        for row in df.itertuples(index=False)
    ]


# ── Tick data ─────────────────────────────────────────────────────────────────

def load_tick_csv(
    path: str | Path,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """
    Load Binance aggTrade tick data from a CSV file.

    Expected columns: price, qty, timestamp (ms), is_buyer_maker.
    Reads in chunks to handle year-long CSV files without memory issues.

    Args:
        path:      Path to the tick CSV file.
        chunksize: Rows per chunk.

    Returns:
        DataFrame with columns [price, qty, timestamp_ms, is_buyer_maker].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tick CSV not found: {path}")

    header = pd.read_csv(path, nrows=0)
    cols   = {c.lower(): c for c in header.columns}

    price_col = cols.get("price", cols.get("p"))
    qty_col   = cols.get("qty",   cols.get("q", cols.get("quantity")))
    ts_col    = cols.get("timestamp", cols.get("time", cols.get("t")))
    bm_col    = cols.get("is_buyer_maker", cols.get("m", cols.get("buyer_maker")))

    if not all([price_col, qty_col, ts_col]):
        raise ValueError(
            f"Cannot identify required columns in {path}.  "
            f"Found: {list(header.columns)}"
        )

    usecols = [c for c in [price_col, qty_col, ts_col, bm_col] if c]
    chunks  = []
    for chunk in pd.read_csv(path, usecols=usecols,
                              chunksize=chunksize, low_memory=False):
        rename = {price_col: "price", qty_col: "qty", ts_col: "timestamp_ms"}
        if bm_col:
            rename[bm_col] = "is_buyer_maker"
        chunk = chunk.rename(columns=rename)
        for col in ("price", "qty", "timestamp_ms"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.dropna(subset=["price", "qty", "timestamp_ms"])
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True).sort_values("timestamp_ms")
    logger.info("Loaded %d ticks from %s", len(df), path.name)
    return df.reset_index(drop=True)


# ── Bar CSV persistence ───────────────────────────────────────────────────────

def save_bars_csv(bars: List[Dict[str, Any]], path: str | Path) -> None:
    """Save a list of bar dicts to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bars).to_csv(path, index=False)
    logger.info("Saved %d bars → %s", len(bars), path)


def load_bars_csv(path: str | Path) -> pd.DataFrame:
    """Load bars from a CSV file previously saved by save_bars_csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bars CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    for col in ("datetime", "datetime_start", "datetime_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    logger.info("Loaded %d bars from %s", len(df), path.name)
    return df


# ── EMA state persistence ─────────────────────────────────────────────────────

def save_state(state: Dict[str, Any], path: str | Path) -> None:
    """Persist EMA calibration state to JSON so runs can be resumed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _serial(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "isoformat"):          # pandas Timestamp
            return obj.isoformat()
        raise TypeError(f"Not JSON-serialisable: {type(obj)}")

    with open(path, "w") as fh:
        json.dump(state, fh, default=_serial, indent=2)
    logger.debug("State saved → %s", path)


def load_state(path: str | Path) -> Optional[Dict[str, Any]]:
    """Load EMA state from JSON; return None if file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)
