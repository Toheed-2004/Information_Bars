"""
ml_module/features/feature_engineer.py
----------------------------------------
Feature engineering for alternative bar data.

Computes technical indicators, regime signals, bar-specific statistics,
lag features, and rolling return features — all from OHLCV bar data.

Design principles
-----------------
- Fully vectorised (pandas/numpy); no row loops.
- Only uses information available *at bar close* — no look-ahead.
- Adding a new feature group requires only a new ``_add_*`` method
  and a call in ``transform()``.
- NaN rows produced by rolling windows are dropped at the end.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ml_module.utils.helpers import get_logger

logger = get_logger(__name__)

# Columns that must not be used as features (leakage or metadata)
_DEFAULT_DROP = [
    "datetime_start", "datetime_end", "created_at",
    "Unnamed: 0", "exchange", "symbol",
    "regime_label", "regime_trend", "regime_volatility", "regime_momentum",
]


class FeatureEngineer:
    """
    Transform a bar DataFrame into a model-ready feature matrix.

    Parameters
    ----------
    rsi_windows    : RSI periods (list of ints).
    macd_params    : {'fast': int, 'slow': int, 'signal': int}.
    bb_window      : Bollinger Bands period.
    atr_window     : ATR period.
    ema_windows    : EMA periods.
    return_horizons: Rolling return horizons in bars.
    lag_periods    : Number of bars to lag features.
    drop_columns   : Columns to exclude before feature building.
    """

    def __init__(
        self,
        rsi_windows:     List[int]  = (7, 14, 21),
        macd_params:     Dict       = None,
        bb_window:       int        = 20,
        atr_window:      int        = 14,
        ema_windows:     List[int]  = (9, 21, 50),
        return_horizons: List[int]  = (1, 3, 5, 10),
        lag_periods:     List[int]  = (1, 2, 3),
        drop_columns:    List[str]  = None,
    ):
        self.rsi_windows     = list(rsi_windows)
        self.macd_params     = macd_params or {"fast": 12, "slow": 26, "signal": 9}
        self.bb_window       = bb_window
        self.atr_window      = atr_window
        self.ema_windows     = list(ema_windows)
        self.return_horizons = list(return_horizons)
        self.lag_periods     = list(lag_periods)
        self.drop_columns    = list(drop_columns or _DEFAULT_DROP)

        # Populated after fit/transform — useful for debugging
        self.feature_names_: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features from *df*.

        Parameters
        ----------
        df : Bar DataFrame (sorted ascending by datetime).

        Returns
        -------
        DataFrame with only numeric feature columns; datetime index retained.
        Rows with NaN (from rolling warmup) are dropped.
        """
        logger.info("FeatureEngineer: input shape %s", df.shape)
        out = df.copy()

        # Drop metadata / leaking columns
        drop_existing = [c for c in self.drop_columns if c in out.columns]
        out.drop(columns=drop_existing, inplace=True, errors="ignore")

        # Core indicator groups
        out = self._add_returns(out)
        out = self._add_ema(out)
        out = self._add_rsi(out)
        out = self._add_macd(out)
        out = self._add_bollinger(out)
        out = self._add_atr(out)
        out = self._add_bar_features(out)
        out = self._add_regime_scores(out)
        out = self._encode_categoricals(out)
        out = self._add_lags(out)

        # Drop non-numeric (anything left from bar metadata)
        out = out.select_dtypes(include=[np.number])

        # Drop rows with NaN (rolling warmup period)
        before = len(out)
        out.dropna(inplace=True)
        dropped = before - len(out)
        if dropped:
            logger.info("Dropped %d NaN rows (rolling warmup).", dropped)

        self.feature_names_ = list(out.columns)
        logger.info("FeatureEngineer: output shape %s, features=%d", out.shape, len(self.feature_names_))
        return out

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling log-returns over multiple horizons."""
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["log_return_1"] = log_ret
        for h in self.return_horizons:
            df[f"log_return_{h}"] = np.log(df["close"] / df["close"].shift(h))
        return df

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA values and price-relative ratios."""
        for w in self.ema_windows:
            ema = df["close"].ewm(span=w, adjust=False).mean()
            df[f"ema_{w}"]          = ema
            df[f"close_div_ema_{w}"] = df["close"] / ema - 1.0
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Relative Strength Index for each window."""
        for w in self.rsi_windows:
            delta = df["close"].diff()
            gain  = delta.clip(lower=0).rolling(w, min_periods=1).mean()
            loss  = (-delta.clip(upper=0)).rolling(w, min_periods=1).mean()
            rs    = gain / loss.replace(0, np.nan)
            df[f"rsi_{w}"] = 100.0 - 100.0 / (1.0 + rs)
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD line, signal line, and histogram."""
        fast   = self.macd_params["fast"]
        slow   = self.macd_params["slow"]
        sig    = self.macd_params["signal"]
        ema_f  = df["close"].ewm(span=fast, adjust=False).mean()
        ema_s  = df["close"].ewm(span=slow, adjust=False).mean()
        macd   = ema_f - ema_s
        signal = macd.ewm(span=sig, adjust=False).mean()
        df["macd"]          = macd
        df["macd_signal"]   = signal
        df["macd_hist"]     = macd - signal
        df["macd_norm"]     = macd / df["close"]          # price-relative
        return df

    def _add_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Band width and position."""
        w      = self.bb_window
        mid    = df["close"].rolling(w, min_periods=1).mean()
        std    = df["close"].rolling(w, min_periods=1).std()
        upper  = mid + 2.0 * std
        lower  = mid - 2.0 * std
        df["bb_width"]    = (upper - lower) / mid
        df["bb_position"] = (df["close"] - lower) / (upper - lower + 1e-10)
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average True Range (normalised by close)."""
        w       = self.atr_window
        prev_c  = df["close"].shift(1)
        tr      = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"]  - prev_c).abs(),
        ], axis=1).max(axis=1)
        atr     = tr.rolling(w, min_periods=1).mean()
        df["atr"]      = atr
        df["atr_norm"] = atr / df["close"]
        return df

    def _add_bar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bar-specific structural features."""
        # Intra-bar structure
        range_   = df["high"] - df["low"]
        df["bar_range_norm"] = range_ / df["close"]
        df["upper_shadow"]   = (df["high"] - df[["open", "close"]].max(axis=1)) / (range_ + 1e-10)
        df["lower_shadow"]   = (df[["open", "close"]].min(axis=1) - df["low"]) / (range_ + 1e-10)
        df["body_size"]      = (df["close"] - df["open"]).abs() / (range_ + 1e-10)
        df["close_position"] = (df["close"] - df["low"]) / (range_ + 1e-10)

        # VWAP-relative (already present in some bars)
        if "vwap" in df.columns:
            df["close_vs_vwap"] = df["close"] / df["vwap"] - 1.0

        # Volume features
        if "volume" in df.columns:
            df["volume_pct_change"] = df["volume"].pct_change()
            df["volume_zscore"]     = (
                (df["volume"] - df["volume"].rolling(20, min_periods=1).mean())
                / (df["volume"].rolling(20, min_periods=1).std() + 1e-10)
            )

        # Duration / tick count (information density)
        if "duration_minutes" in df.columns:
            df["duration_norm"] = df["duration_minutes"] / df["duration_minutes"].rolling(20, min_periods=1).mean()
        if "tick_count" in df.columns:
            df["tick_density"] = df["tick_count"] / (df["duration_minutes"].replace(0, 1) + 1e-10)

        # Bar-size momentum
        if "bar_size" in df.columns:
            df["bar_size_zscore"] = (
                (df["bar_size"] - df["bar_size"].rolling(20, min_periods=1).mean())
                / (df["bar_size"].rolling(20, min_periods=1).std() + 1e-10)
            )

        return df

    def _add_regime_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pass through numeric regime score columns (already computed by bar generator)."""
        score_cols = [c for c in df.columns if c.startswith("score_") or c in (
            "trend_strength_z", "vol_percentile", "volatility_skew",
            "transition_pressure", "trend_acceleration", "adaptive_alpha",
            "up_vol", "down_vol", "regime_stability", "directional_persistence",
            "regime_confidence",
        )]
        # They're already numeric — just ensure NaN from early rows are forward-filled
        for col in score_cols:
            df[col] = df[col].ffill()
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode any remaining string columns (e.g. partial regime labels)."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            try:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=np.float32)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            except Exception as e:
                logger.debug("Could not encode column '%s': %s", col, e)
                df.drop(columns=[col], inplace=True, errors="ignore")
        return df

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged copies of all current features (prevents leakage)."""
        cols_to_lag = [c for c in df.columns if c not in ("label",)]
        for lag in self.lag_periods:
            lagged = df[cols_to_lag].shift(lag)
            lagged.columns = [f"{c}_lag{lag}" for c in cols_to_lag]
            df = pd.concat([df, lagged], axis=1)
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_params(self) -> Dict:
        return {
            "rsi_windows":     self.rsi_windows,
            "macd_params":     self.macd_params,
            "bb_window":       self.bb_window,
            "atr_window":      self.atr_window,
            "ema_windows":     self.ema_windows,
            "return_horizons": self.return_horizons,
            "lag_periods":     self.lag_periods,
            "n_features":      len(self.feature_names_),
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bar_csv(
    path: str,
    datetime_col: str = "datetime",
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a bar CSV, parse the datetime index, and sort ascending.

    Handles the 'Unnamed: 0' index column written by pandas .to_csv().
    """
    df = pd.read_csv(path, low_memory=False)

    # Parse datetime
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
        df.set_index(datetime_col, inplace=True)
    else:
        raise ValueError(f"datetime column '{datetime_col}' not found in {path}")

    df.sort_index(inplace=True)

    # Drop user-specified extra columns
    if drop_cols:
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_feature_engineer(cfg: Dict) -> FeatureEngineer:
    """Instantiate a FeatureEngineer from the ``features`` config section."""
    return FeatureEngineer(
        rsi_windows     = cfg.get("rsi_windows",     [7, 14, 21]),
        macd_params     = cfg.get("macd_params",     {"fast": 12, "slow": 26, "signal": 9}),
        bb_window       = cfg.get("bb_window",       20),
        atr_window      = cfg.get("atr_window",      14),
        ema_windows     = cfg.get("ema_windows",     [9, 21, 50]),
        return_horizons = cfg.get("return_horizons", [1, 3, 5, 10]),
        lag_periods     = cfg.get("lag_periods",     [1, 2, 3]),
        drop_columns    = cfg.get("drop_columns",    _DEFAULT_DROP),
    )
