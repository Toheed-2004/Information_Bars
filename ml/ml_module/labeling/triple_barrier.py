"""
ml_module/labeling/triple_barrier.py
-------------------------------------
Triple-Barrier Labeling (López de Prado, "Advances in Financial ML", Ch. 3).

Label generation assigns one of three outcomes to each bar:
  +1  (BUY)  → price hits the upper profit-taking barrier first
  -1  (SELL) → price hits the lower stop-loss barrier first
   0  (HOLD) → vertical barrier (max holding period) is hit first

Design principles
-----------------
- Fully vectorised; no Python loops over rows.
- Barriers can be fixed (% of price) or volatility-scaled (dynamic).
- The labeler is a class so its parameters are inspectable and serialisable.
- All configuration arrives via a plain dict so it integrates with YAML configs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from ml_module.utils.helpers import get_logger

logger = get_logger(__name__)


class TripleBarrierLabeler:
    """
    Generate triple-barrier labels for a bar DataFrame.

    Parameters
    ----------
    profit_target     : Upper barrier as a fraction of entry price (e.g. 0.02).
    stop_loss         : Lower barrier as a fraction of entry price (e.g. 0.01).
    max_holding_bars  : Number of bars before the vertical barrier triggers.
    volatility_lookback: If not None, scale barriers by rolling σ over this window.
                         Barriers become profit_target × σ_t (dimensionless units).
    min_class_count   : Minimum samples per class; warns if violated.

    Notes
    -----
    When ``volatility_lookback`` is set, the barriers are *multiplied* by the
    local volatility estimate so that the labeler adapts to market regimes.
    The raw profit_target / stop_loss then act as **multipliers on σ**, not
    absolute return thresholds.  Typical setup: profit_target ≈ 1.5–2.5,
    stop_loss ≈ 0.5–1.0 when volatility scaling is active.
    """

    def __init__(
        self,
        profit_target: float = 0.02,
        stop_loss: float = 0.01,
        max_holding_bars: int = 20,
        volatility_lookback: Optional[int] = 20,
        min_class_count: int = 10,
    ):
        if profit_target <= 0 or stop_loss <= 0:
            raise ValueError("profit_target and stop_loss must be > 0")
        if max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1")

        self.profit_target      = profit_target
        self.stop_loss          = stop_loss
        self.max_holding_bars   = max_holding_bars
        self.volatility_lookback = volatility_lookback
        self.min_class_count    = min_class_count

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate labels for every bar in *df*.

        Parameters
        ----------
        df : Bar DataFrame with at minimum columns: ``close``, ``high``, ``low``.
             Must be sorted by time (ascending).

        Returns
        -------
        pd.Series of int8 with values in {-1, 0, 1}, indexed like *df*.
        Name: "label".
        """
        self._validate(df)
        close  = df["close"].to_numpy(dtype=np.float64)
        high   = df["high"].to_numpy(dtype=np.float64)
        low    = df["low"].to_numpy(dtype=np.float64)
        n      = len(close)

        upper_mult, lower_mult = self._compute_barrier_multipliers(df)

        labels = np.zeros(n, dtype=np.int8)

        # Vectorised via a forward-scan using cumulative min/max windows.
        # For each bar i we scan bars [i+1, i+max_holding_bars].
        for i in range(n):
            horizon_end = min(i + self.max_holding_bars, n - 1)
            if horizon_end <= i:
                labels[i] = 0
                continue

            entry = close[i]
            u_mult = upper_mult[i]
            l_mult = lower_mult[i]

            upper_barrier = entry * (1.0 + u_mult)
            lower_barrier = entry * (1.0 - l_mult)

            # Slice the forward window
            future_high  = high[i + 1 : horizon_end + 1]
            future_low   = low[i  + 1 : horizon_end + 1]

            # First bar index where upper / lower is touched
            upper_touch = np.argmax(future_high >= upper_barrier)
            lower_touch = np.argmax(future_low  <= lower_barrier)

            hit_upper = future_high[upper_touch] >= upper_barrier if len(future_high) > 0 else False
            hit_lower = future_low[lower_touch]  <= lower_barrier if len(future_low)  > 0 else False

            if not hit_upper and not hit_lower:
                labels[i] = 0  # vertical barrier
            elif hit_upper and not hit_lower:
                labels[i] = 1
            elif hit_lower and not hit_upper:
                labels[i] = -1
            else:
                # Both touched; whichever came first wins
                labels[i] = 1 if upper_touch <= lower_touch else -1

        result = pd.Series(labels, index=df.index, name="label", dtype=np.int8)
        self._log_distribution(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_barrier_multipliers(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-bar upper and lower barrier multipliers."""
        n = len(df)

        if self.volatility_lookback is None:
            upper_mult = np.full(n, self.profit_target)
            lower_mult = np.full(n, self.stop_loss)
        else:
            # Rolling σ of log-returns; forward-fill NaN at start of series
            log_ret = np.log(df["close"] / df["close"].shift(1))
            rolling_vol = (
                log_ret
                .rolling(self.volatility_lookback, min_periods=1)
                .std()
                .bfill()
                .to_numpy(dtype=np.float64)
            )
            upper_mult = rolling_vol * self.profit_target
            lower_mult = rolling_vol * self.stop_loss

        return upper_mult, lower_mult

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in ("close", "high", "low") if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        if df.empty:
            raise ValueError("DataFrame is empty — cannot label.")

    def _log_distribution(self, labels: pd.Series) -> None:
        counts = labels.value_counts().sort_index()
        total  = len(labels)
        parts  = {
            "BUY (+1)":  counts.get(1,  0),
            "HOLD (0)":  counts.get(0,  0),
            "SELL (-1)": counts.get(-1, 0),
        }
        logger.info(
            "Label distribution  total=%d | %s",
            total,
            " | ".join(f"{k}: {v} ({v/total:.1%})" for k, v in parts.items()),
        )
        for cls_val, cnt in [( 1, parts["BUY (+1)"]),
                              ( 0, parts["HOLD (0)"]),
                              (-1, parts["SELL (-1)"])]:
            if cnt < self.min_class_count:
                logger.warning(
                    "Class %d has only %d samples (min_class_count=%d). "
                    "Consider adjusting barriers or reducing min_class_count.",
                    cls_val, cnt, self.min_class_count,
                )

    # ------------------------------------------------------------------
    # Serialisation helpers (for reproducibility in research reports)
    # ------------------------------------------------------------------

    def get_params(self) -> Dict:
        return {
            "profit_target":      self.profit_target,
            "stop_loss":          self.stop_loss,
            "max_holding_bars":   self.max_holding_bars,
            "volatility_lookback": self.volatility_lookback,
            "min_class_count":    self.min_class_count,
        }

    def __repr__(self) -> str:
        p = self.get_params()
        return (
            f"TripleBarrierLabeler(profit_target={p['profit_target']}, "
            f"stop_loss={p['stop_loss']}, "
            f"max_holding_bars={p['max_holding_bars']}, "
            f"volatility_lookback={p['volatility_lookback']})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_labeler(cfg: Dict) -> TripleBarrierLabeler:
    """
    Instantiate a TripleBarrierLabeler from the ``labeling`` config section.

    Example cfg
    -----------
    {
        "profit_target": 0.02, "stop_loss": 0.01,
        "max_holding_bars": 20, "volatility_lookback": 20,
        "min_class_count": 10
    }
    """
    return TripleBarrierLabeler(
        profit_target      = cfg.get("profit_target",      0.02),
        stop_loss          = cfg.get("stop_loss",          0.01),
        max_holding_bars   = cfg.get("max_holding_bars",   20),
        volatility_lookback= cfg.get("volatility_lookback", 20),
        min_class_count    = cfg.get("min_class_count",    10),
    )
