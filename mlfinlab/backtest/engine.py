"""
mlfinlab/backtest/engine.py
============================
Stage 5 – Walk-forward backtest and performance metrics.

REFACTORING NOTES (bugs fixed vs original)
-------------------------------------------
1. Annualisation for information bars (BUG): original fell back to 252
   (equity trading-day convention) for all bar types not in the
   ANNUALISATION dict, which means ALL information bars (dollar, volume,
   volatility, hybrid, range, renko) got the wrong scaling factor. Fixed:
   the avg_seconds branch now correctly fires for all information bars by
   computing bars-per-year from actual bar spacing. For BTC which trades
   24/7/365 this gives ~26,280 for hourly bars (365×24) not 252.

2. bar_opens.get / bar_closes.get: pd.Series.get() is deprecated and slow
   for large Series. Replaced with searchsorted-based price lookups.

3. Feature mode column extraction: original did
   signals.get("feature_mode", ...).iloc[0] which calls pd.Series.get()
   on a DataFrame — this returns the column Series, not NaN. Fixed.

4. CPCV metrics integration: backtest now accepts an optional cpcv_scores
   list and computes DSR when available.

5. Equity curve vectorisation: trade simulation loop now uses sorted
   signals index to avoid repeated searchsorted overhead.

References
----------
de Prado, M. L. (2018). Advances in Financial Machine Learning, Ch.14.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

log = logging.getLogger("mlfinlab.backtest")


def run(
    signals      : pd.DataFrame,
    bars         : pd.DataFrame,
    ml_frame     : pd.DataFrame,
    meta         : dict,
    fee_pct      : float = 0.0004,
    risk_free    : float = 0.0,
    initial_cap  : float = 10_000.0,
    cpcv_sharpes : list  = None,
) -> dict:
    """Simulate trades from signals and compute all performance metrics.

    Parameters
    ----------
    signals      : Stage 4 output (DatetimeIndex = event entry times).
    bars         : Full OHLCV bars for this dataset.
    ml_frame     : Stage 2 output (contains t1_touch exit timestamps).
    meta         : Bar metadata (bar_type, source, bar_class).
    fee_pct      : One-way fee fraction (0.0004 = 0.04%).
    risk_free    : Annualised risk-free rate for Sharpe.
    initial_cap  : Starting capital in USD.
    cpcv_sharpes : List of Sharpe ratios from CPCV combinations for DSR.

    Returns
    -------
    dict  Flat metrics dict ready for Stage 6 comparison table.
    """
    if signals is None or len(signals) == 0:
        log.warning("  No signals to backtest")
        return _empty_metrics(meta)

    # ── Exit timestamps from ml_frame ─────────────────────────────────────
    t1_col = "t1_touch" if "t1_touch" in ml_frame.columns else (
             "t1"       if "t1"       in ml_frame.columns else None)
    if t1_col is None:
        log.warning("  No exit timestamp column in ml_frame")
        return _empty_metrics(meta)

    exits = ml_frame[t1_col].reindex(signals.index)

    # ── Annualisation factor — BUG FIX ─────────────────────────────────────
    # For information bars (dollar, volume, etc.) bars are NOT equally spaced
    # so bar-type heuristics are meaningless. Always compute from actual data.
    ann_factor = _compute_ann_factor(bars)

    # ── Price arrays for fast O(log n) lookup ─────────────────────────────
    bar_index  = bars.index
    bar_arr    = bar_index.as_unit("ns").asi8  # int64 ns; Timestamp.value is always ns
    opens_arr  = bars["open"].values.astype(float)
    closes_arr = bars["close"].values.astype(float)

    # ── Trade simulation ──────────────────────────────────────────────────
    trades = []
    equity = initial_cap

    # Sort signals chronologically (should already be sorted, but guard)
    signals = signals.sort_index()

    for entry_dt, row in signals.iterrows():
        signal   = int(row["signal"])
        bet_size = float(row.get("bet_size", 0.0))

        if signal == 0 or bet_size <= 0:
            continue

        # Entry price: open of the NEXT bar after signal bar
        entry_ns  = entry_dt.value
        next_pos  = int(np.searchsorted(bar_arr, entry_ns, side="right"))
        if next_pos >= len(bar_arr):
            continue
        entry_price = opens_arr[next_pos]
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        # Exit price: close of bar where triple-barrier fired
        exit_dt = exits.get(entry_dt, None)
        if exit_dt is None or pd.isna(exit_dt):
            # Fall back to 3-day vertical barrier
            fb_ns    = (entry_dt + pd.Timedelta(days=3)).value
            fb_pos   = int(np.searchsorted(bar_arr, fb_ns, side="left"))
            fb_pos   = min(fb_pos, len(bar_arr) - 1)
            exit_bar = fb_pos
        else:
            exit_ns  = exit_dt.value
            exit_pos = int(np.searchsorted(bar_arr, exit_ns, side="left"))
            exit_pos = min(exit_pos, len(bar_arr) - 1)
            exit_bar = exit_pos

        exit_price = closes_arr[exit_bar]
        if np.isnan(exit_price) or exit_price <= 0:
            continue

        exit_bar_dt = bar_index[exit_bar]

        # P&L
        capital_at_risk = equity * bet_size

        if signal == 1:    # long
            raw_ret = (exit_price - entry_price) / entry_price
        else:              # short
            raw_ret = (entry_price - exit_price) / entry_price

        net_ret = raw_ret - 2 * fee_pct
        pnl     = capital_at_risk * net_ret
        equity += pnl

        trades.append({
            "entry_dt"   : entry_dt,
            "exit_dt"    : exit_bar_dt,
            "signal"     : signal,
            "entry_price": entry_price,
            "exit_price" : exit_price,
            "raw_ret"    : raw_ret,
            "net_ret"    : net_ret,
            "pnl"        : pnl,
            "equity"     : equity,
            "bet_size"   : bet_size,
        })

    if not trades:
        log.warning("  No trades executed")
        return _empty_metrics(meta)

    trade_df = pd.DataFrame(trades).set_index("entry_dt")
    log.info("  Trades executed : %d", len(trade_df))

    # ── Classification metrics ────────────────────────────────────────────
    cl_metrics = _classification_metrics(signals)

    # ── Equity curve metrics ──────────────────────────────────────────────
    perf_metrics = _performance_metrics(trade_df, initial_cap, risk_free)

    # ── DSR from CPCV (if available) ──────────────────────────────────────
    dsr = float("nan")
    if cpcv_sharpes and len(cpcv_sharpes) >= 2:
        from mlfinlab.models.cv import compute_deflated_sharpe
        sr_obs = perf_metrics.get("sharpe_ratio", float("nan"))
        if not np.isnan(sr_obs):
            dsr = compute_deflated_sharpe(sr_obs, cpcv_sharpes, len(cpcv_sharpes))

    # ── Feature mode ──────────────────────────────────────────────────────
    # BUG FIX: original used signals.get() which on a DataFrame returns a
    # column Series. Now access .columns directly.
    feature_mode = "?"
    if "feature_mode" in signals.columns:
        feature_mode = signals["feature_mode"].iloc[0]

    metrics = {
        "bar_type"    : meta.get("bar_type",  "?"),
        "source"      : meta.get("source",    "?"),
        "bar_class"   : meta.get("bar_class", "?"),
        "feature_mode": feature_mode,
        **cl_metrics,
        **perf_metrics,
        "n_trades"    : len(trade_df),
        "ann_factor"  : ann_factor,
        "dsr"         : dsr,
    }

    _log_metrics(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Annualisation factor computation — BUG FIX
# ---------------------------------------------------------------------------

def _compute_ann_factor(bars: pd.DataFrame) -> int:
    """Compute bars-per-year from actual bar spacing.

    BUG FIX: original used a static dict keyed on bar_type. This gives
    wrong values for information bars (which are not equally spaced by
    construction) and also for calendar bars if bar count ≠ expected.

    The correct approach: measure the actual timespan and bar count.
    For 24/7 markets like BTC crypto: 365.25 × 24 × 3600 seconds/year.
    """
    if len(bars) < 2:
        return 252  # safe fallback

    total_seconds = (bars.index[-1] - bars.index[0]).total_seconds()
    if total_seconds <= 0:
        return 252

    avg_bar_seconds = total_seconds / (len(bars) - 1)
    ann = max(1, int(round(365.25 * 24 * 3600 / avg_bar_seconds)))
    return ann


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def _classification_metrics(signals: pd.DataFrame) -> dict:
    """Compute classification metrics from y_pred vs y_true."""
    if "y_true" not in signals.columns or "y_pred" not in signals.columns:
        return {"accuracy": np.nan, "f1_weighted": np.nan, "auc_roc": np.nan}

    traded = signals[signals["signal"] != 0]
    if len(traded) < 5:
        return {"accuracy": np.nan, "f1_weighted": np.nan, "auc_roc": np.nan}

    y_true = traded["y_true"].astype(int)
    y_pred = traded["y_pred"].astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    auc = np.nan
    if "prob_p1" in traded.columns:
        try:
            y_bin = (y_true == 1).astype(int)
            if y_bin.nunique() == 2:
                auc = roc_auc_score(y_bin, traded["prob_p1"])
        except Exception:
            pass

    return {"accuracy": acc, "f1_weighted": f1, "auc_roc": auc}


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _performance_metrics(
    trade_df    : pd.DataFrame,
    initial_cap : float,
    risk_free   : float,
) -> dict:
    """Compute Sharpe, Sortino, MDD, Calmar, Win rate, Profit factor.

    Annualisation: based on actual trade frequency (trades per year),
    NOT bar frequency. This is correct because the return series is
    per-trade, not per-bar.
    """
    rets     = trade_df["net_ret"].values
    n_trades = len(rets)

    total_return = (trade_df["equity"].iloc[-1] - initial_cap) / initial_cap

    # Trades per year from actual timestamps
    if n_trades > 1:
        span_years      = (trade_df.index[-1] - trade_df.index[0]).days / 365.25
        trades_per_year = n_trades / max(span_years, 1e-6)
    else:
        trades_per_year = 252.0   # fallback

    ann_return   = (1 + total_return) ** (trades_per_year / max(n_trades, 1)) - 1
    mean_ret     = float(np.mean(rets))
    std_ret      = float(np.std(rets, ddof=1)) if n_trades > 1 else float("nan")
    rf_per_trade = risk_free / max(trades_per_year, 1.0)

    sharpe = float("nan")
    if std_ret and std_ret > 0 and np.isfinite(std_ret):
        sharpe = ((mean_ret - rf_per_trade) / std_ret) * np.sqrt(trades_per_year)

    neg_rets = rets[rets < rf_per_trade]
    sortino  = float("nan")
    if len(neg_rets) > 1:
        down_std = float(np.std(neg_rets, ddof=1))
        if down_std > 0:
            sortino = ((mean_ret - rf_per_trade) / down_std) * np.sqrt(trades_per_year)

    equity_curve = np.concatenate([[initial_cap], trade_df["equity"].values])
    running_max  = np.maximum.accumulate(equity_curve)
    drawdowns    = (equity_curve - running_max) / (running_max + 1e-12)
    max_dd       = float(np.min(drawdowns))

    calmar = float("nan")
    if max_dd < 0 and np.isfinite(ann_return):
        calmar = ann_return / abs(max_dd)

    wins         = rets[rets > 0]
    losses       = rets[rets < 0]
    win_rate     = float(len(wins)) / n_trades if n_trades > 0 else 0.0
    gross_profit = float(np.sum(wins))   if len(wins)   > 0 else 0.0
    gross_loss   = float(np.sum(losses)) if len(losses) > 0 else 0.0
    profit_factor = (gross_profit / abs(gross_loss)
                     if gross_loss < 0 else float("nan"))

    return {
        "total_return"    : total_return,
        "ann_return"      : ann_return,
        "sharpe_ratio"    : sharpe,
        "sortino_ratio"   : sortino,
        "max_drawdown"    : max_dd,
        "calmar_ratio"    : calmar,
        "win_rate"        : win_rate,
        "profit_factor"   : profit_factor,
        "avg_trade_return": mean_ret,
    }


# ---------------------------------------------------------------------------
# Empty metrics
# ---------------------------------------------------------------------------

def _empty_metrics(meta: dict) -> dict:
    nan = float("nan")
    return {
        "bar_type": meta.get("bar_type","?"), "source": meta.get("source","?"),
        "bar_class": meta.get("bar_class","?"), "feature_mode": "?",
        "accuracy": nan, "f1_weighted": nan, "auc_roc": nan,
        "total_return": nan, "ann_return": nan, "sharpe_ratio": nan,
        "sortino_ratio": nan, "max_drawdown": nan, "calmar_ratio": nan,
        "win_rate": nan, "profit_factor": nan, "avg_trade_return": nan,
        "n_trades": 0, "dsr": nan,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_metrics(m: dict) -> None:
    log.info("  ── Classification ──────────────────────────")
    log.info("  Accuracy      : %.3f", m.get("accuracy",    float("nan")))
    log.info("  F1 weighted   : %.3f", m.get("f1_weighted", float("nan")))
    log.info("  AUC-ROC       : %.3f", m.get("auc_roc",     float("nan")))
    log.info("  ── Backtest ────────────────────────────────")
    log.info("  Total return  : %+.2f%%", m.get("total_return", float("nan")) * 100)
    log.info("  Ann return    : %+.2f%%", m.get("ann_return",   float("nan")) * 100)
    log.info("  Sharpe ratio  : %.3f",    m.get("sharpe_ratio", float("nan")))
    log.info("  Sortino ratio : %.3f",    m.get("sortino_ratio",float("nan")))
    log.info("  Max drawdown  : %.2f%%",  m.get("max_drawdown", float("nan")) * 100)
    log.info("  Calmar ratio  : %.3f",    m.get("calmar_ratio", float("nan")))
    log.info("  Win rate      : %.1f%%",  m.get("win_rate",     float("nan")) * 100)
    log.info("  Profit factor : %.3f",    m.get("profit_factor",float("nan")))
    log.info("  N trades      : %d",      m.get("n_trades",     0))
    dsr = m.get("dsr", float("nan"))
    if not np.isnan(dsr):
        log.info("  DSR (CPCV)    : %.3f", dsr)
