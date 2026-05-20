"""
mlfinlab/signals/generator.py
==============================
Stage 4 – Signal generation from stitched walk-forward predictions.

Input
-----
predictions : pd.DataFrame
    Stitched out-of-sample predictions from Stage 3 (all folds concatenated).
    Columns: y_true | y_pred | prob_m1 | prob_0 | prob_p1 | fold

Output
------
signals : pd.DataFrame
    One row per event, columns:
        signal      int    -1 / 0 / +1
        bet_size    float  fraction of capital to deploy (0.0 to 1.0)
        confidence  float  probability of the predicted class
        y_true      int    true label (kept for backtest P&L verification)
        y_pred      int    model prediction
        prob_m1     float
        prob_0      float
        prob_p1     float

Signal logic
------------
1. Take the predicted class (y_pred) from the model.
2. Apply a confidence threshold: if the probability of the predicted
   class is below the threshold, signal = 0 (do not trade).
3. Size the bet proportionally to confidence using fractional Kelly:
       bet_size = kelly_fraction * (confidence - threshold) / (1 - threshold)
   Capped at max_bet_size for risk management.

Why confidence threshold matters
---------------------------------
A model that predicts +1 with prob=0.51 is barely better than a coin
flip. After the 0.04% Binance fee on entry and exit (0.08% round-trip),
low-confidence trades will lose money even when directionally correct.
Only trade when the model is confident enough to overcome the fee drag.

References
----------
de Prado, M. L. (2018). Advances in Financial Machine Learning, Ch.10.
Kelly, J. L. (1956). A new interpretation of information rate.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("mlfinlab.signals")


def generate_signals(
    predictions       : pd.DataFrame,
    confidence_threshold : float = 0.55,
    kelly_fraction    : float = 0.25,
    max_bet_size      : float = 0.20,
) -> pd.DataFrame:
    """Convert walk-forward predictions into actionable trading signals.

    Parameters
    ----------
    predictions : pd.DataFrame
        Stitched out-of-sample predictions from Stage 3.
        Must contain: y_pred, prob_m1, prob_p1.
        prob_0 used when present (three-class model).
    confidence_threshold : float
        Minimum probability of predicted class required to trade.
        Below this → signal = 0 (hold, no trade).
        Default 0.55 ensures fees are covered on average.
    kelly_fraction : float
        Fraction of full Kelly bet to use (0.25 = quarter Kelly).
        Full Kelly is theoretically optimal but highly volatile in
        practice. Quarter Kelly is the standard conservative choice.
    max_bet_size : float
        Maximum fraction of capital per trade (hard cap = 20%).
        Prevents ruinous concentration even at very high confidence.

    Returns
    -------
    pd.DataFrame  signals with columns described in module docstring.
    """
    preds = predictions.copy()

    # ── Build prob_0 column if not present (binary model) ─────────────────
    if "prob_0" not in preds.columns:
        preds["prob_0"] = 1.0 - preds["prob_m1"] - preds["prob_p1"]
        preds["prob_0"] = preds["prob_0"].clip(lower=0.0)

    # ── Confidence = probability of the predicted class ────────────────────
    def _confidence(row):
        if row["y_pred"] == 1:
            return row["prob_p1"]
        elif row["y_pred"] == -1:
            return row["prob_m1"]
        else:
            return row["prob_0"]

    preds["confidence"] = preds.apply(_confidence, axis=1)

    # ── Signal: 0 if below threshold, else follow y_pred ──────────────────
    preds["signal"] = np.where(
        preds["confidence"] >= confidence_threshold,
        preds["y_pred"].astype(int),
        0,
    )

    # ── Bet sizing: fractional Kelly ───────────────────────────────────────
    # Scale confidence above threshold to [0, 1], apply Kelly fraction
    scaled = (preds["confidence"] - confidence_threshold).clip(lower=0) / (
        1.0 - confidence_threshold + 1e-9
    )
    raw_bet = kelly_fraction * scaled
    preds["bet_size"] = np.where(
        preds["signal"] != 0,
        raw_bet.clip(upper=max_bet_size),
        0.0,
    )

    # ── Summary stats ──────────────────────────────────────────────────────
    n_total  = len(preds)
    n_trade  = (preds["signal"] != 0).sum()
    n_buy    = (preds["signal"] ==  1).sum()
    n_sell   = (preds["signal"] == -1).sum()
    n_hold   = (preds["signal"] ==  0).sum()

    log.info("  Signals generated : %d total events", n_total)
    log.info("  Buy  (+1) : %d  (%.1f%%)", n_buy,  n_buy  / n_total * 100)
    log.info("  Sell (-1) : %d  (%.1f%%)", n_sell, n_sell / n_total * 100)
    log.info("  Hold ( 0) : %d  (%.1f%%)", n_hold, n_hold / n_total * 100)
    log.info("  Threshold : %.2f  Kelly : %.2f  MaxBet : %.2f",
             confidence_threshold, kelly_fraction, max_bet_size)
    if n_trade > 0:
        log.info("  Avg bet size (traded events) : %.4f",
                 preds.loc[preds["signal"] != 0, "bet_size"].mean())

    keep_cols = ["signal", "bet_size", "confidence",
                 "y_true", "y_pred", "prob_m1", "prob_0", "prob_p1"]
    keep_cols = [c for c in keep_cols if c in preds.columns]
    return preds[keep_cols]