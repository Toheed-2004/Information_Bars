"""
mlfinlab/backtest/engine.py
============================
Stage 5 – Walk-forward backtest and performance metrics.

Input
-----
signals : pd.DataFrame
    Output of Stage 4. Columns: signal | bet_size | y_true | confidence
    Index = event entry datetime (out-of-sample events only).

bars : pd.DataFrame
    Full OHLCV bar DataFrame for the bar type being evaluated.
    Used to get entry/exit prices.

ml_frame : pd.DataFrame
    Output of Stage 2. Contains t1_touch (exit timestamp) per event.
    Used to know when each trade exits.

Output
------
metrics : dict
    All performance metrics in one flat dict, ready for Stage 6 table.

Trade simulation
----------------
Entry : open price of the bar AFTER the signal bar
        (realistic: you see the signal at bar close, enter next bar open)
Exit  : close price of the bar where the triple-barrier fired (t1_touch)
Fee   : 0.04% Binance taker fee at entry AND exit (0.08% round-trip)

For signal = +1 (buy):   profit = (exit - entry) / entry - 2*fee
For signal = -1 (short):  profit = (entry - exit) / entry - 2*fee
Signal = 0: no trade, no P&L, capital unchanged.

Metrics computed
----------------
Classification (from y_pred vs y_true):
    accuracy, f1_weighted, auc_roc

Backtest (from simulated trade P&L):
    sharpe_ratio        annualised, using risk_free_rate from CFG
    sortino_ratio       annualised, downside deviation only
    max_drawdown        maximum peak-to-trough loss on equity curve
    calmar_ratio        annualised_return / max_drawdown
    win_rate            fraction of trades with positive net P&L
    profit_factor       gross_profit / gross_loss
    total_return        cumulative return over out-of-sample period
    n_trades            total number of trades executed
    avg_trade_return    mean P&L per trade

References
----------
de Prado, M. L. (2018). Advances in Financial Machine Learning, Ch.14.
Sharpe, W. F. (1994). The Sharpe ratio. Journal of Portfolio Management.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

log = logging.getLogger("mlfinlab.backtest")

ANNUALISATION = {
    # bars_per_year used to annualise Sharpe/Sortino from per-trade returns
    # approximate values; exact value computed from data in run()
    "1h"         : 365 * 24,
    "4h"         : 365 * 6,
    "6h"         : 365 * 4,
    "8h"         : 365 * 3,
    "12h"        : 365 * 2,
    "default"    : 252,          # fallback: trading-day convention
}


def run(
    signals      : pd.DataFrame,
    bars         : pd.DataFrame,
    ml_frame     : pd.DataFrame,
    meta         : dict,
    fee_pct      : float = 0.0004,
    risk_free    : float = 0.0,
    initial_cap  : float = 10_000.0,
) -> dict:
    """Simulate trades from signals and compute all performance metrics.

    Parameters
    ----------
    signals     : Stage 4 output. Index = event entry datetime.
    bars        : Full OHLCV bars for this dataset.
    ml_frame    : Stage 2 output. Contains exit timestamps per event.
    meta        : Bar metadata (bar_type, source, bar_class).
    fee_pct     : One-way fee fraction (0.0004 = 0.04%).
    risk_free   : Annualised risk-free rate for Sharpe.
    initial_cap : Starting capital in USD.

    Returns
    -------
    dict  Flat metrics dict ready for Stage 6 comparison table.
    """
    if signals is None or len(signals) == 0:
        log.warning("  No signals to backtest")
        return _empty_metrics(meta)

    # ── Align exit timestamps from ml_frame ──────────────────────────────
    # ml_frame has t1_touch column = when the triple-barrier fired
    if "t1_touch" in ml_frame.columns:
        t1_col = "t1_touch"
    elif "t1" in ml_frame.columns:
        t1_col = "t1"
    else:
        log.warning("  No exit timestamp column found in ml_frame")
        return _empty_metrics(meta)

    exits = ml_frame[t1_col].reindex(signals.index)

    # ── Compute annualisation factor from data ────────────────────────────
    bar_type = meta.get("bar_type", "default")
    if bar_type in ANNUALISATION:
        ann_factor = ANNUALISATION[bar_type]
    else:
        # compute from actual bar frequency
        if len(bars) > 2:
            avg_seconds = (bars.index[-1] - bars.index[0]).total_seconds() / len(bars)
            ann_factor  = int(365 * 24 * 3600 / avg_seconds)
        else:
            ann_factor = ANNUALISATION["default"]

    # ── Simulate trades ───────────────────────────────────────────────────
    bar_index  = bars.index
    bar_opens  = bars["open"]
    bar_closes = bars["close"]

    trades = []
    equity = initial_cap

    for entry_dt, row in signals.iterrows():
        signal   = int(row["signal"])
        bet_size = float(row.get("bet_size", 0.0))

        if signal == 0 or bet_size <= 0:
            continue

        # Entry price: open of the NEXT bar after signal
        future_bars = bar_index[bar_index > entry_dt]
        if len(future_bars) == 0:
            continue
        entry_bar_dt = future_bars[0]
        entry_price  = float(bar_opens.get(entry_bar_dt, np.nan))
        if np.isnan(entry_price):
            continue

        # Exit price: close of bar where triple-barrier fired
        exit_dt = exits.get(entry_dt, pd.NaT)
        if pd.isna(exit_dt):
            # No recorded exit: use vertical barrier (3 days forward)
            vb_bars = bar_index[bar_index >= entry_dt + pd.Timedelta(days=3)]
            exit_dt = vb_bars[0] if len(vb_bars) else bar_index[-1]

        # Snap exit to nearest available bar
        avail_exits = bar_index[bar_index >= exit_dt]
        exit_bar_dt = avail_exits[0] if len(avail_exits) else bar_index[-1]
        exit_price  = float(bar_closes.get(exit_bar_dt, np.nan))
        if np.isnan(exit_price):
            continue

        # P&L
        capital_at_risk = equity * bet_size

        if signal == 1:    # buy
            raw_ret = (exit_price - entry_price) / entry_price
        else:              # short
            raw_ret = (entry_price - exit_price) / entry_price

        net_ret = raw_ret - 2 * fee_pct   # entry fee + exit fee
        pnl     = capital_at_risk * net_ret
        equity += pnl

        trades.append({
            "entry_dt"    : entry_dt,
            "exit_dt"     : exit_bar_dt,
            "signal"      : signal,
            "entry_price" : entry_price,
            "exit_price"  : exit_price,
            "raw_ret"     : raw_ret,
            "net_ret"     : net_ret,
            "pnl"         : pnl,
            "equity"      : equity,
            "bet_size"    : bet_size,
        })

    if not trades:
        log.warning("  No trades executed")
        return _empty_metrics(meta)

    trade_df = pd.DataFrame(trades).set_index("entry_dt")
    log.info("  Trades executed : %d", len(trade_df))

    # ── Classification metrics (from y_pred vs y_true) ───────────────────
    cl_metrics = _classification_metrics(signals)

    # ── Equity curve metrics ──────────────────────────────────────────────
    perf_metrics = _performance_metrics(
        trade_df, initial_cap, ann_factor, risk_free)

    # ── Combine ───────────────────────────────────────────────────────────
    metrics = {
        "bar_type"     : meta.get("bar_type",  "?"),
        "source"       : meta.get("source",    "?"),
        "bar_class"    : meta.get("bar_class", "?"),
        "feature_mode" : signals.get("feature_mode", pd.Series(["?"])).iloc[0]
                         if "feature_mode" in signals.columns else "?",
        **cl_metrics,
        **perf_metrics,
        "n_trades"     : len(trade_df),
        "ann_factor"   : ann_factor,
    }

    _log_metrics(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def _classification_metrics(signals: pd.DataFrame) -> dict:
    """Compute classification metrics from y_pred vs y_true."""
    if "y_true" not in signals.columns or "y_pred" not in signals.columns:
        return {"accuracy": np.nan, "f1_weighted": np.nan, "auc_roc": np.nan}

    # Use only events where a trade was made (signal != 0)
    traded = signals[signals["signal"] != 0]
    if len(traded) < 5:
        return {"accuracy": np.nan, "f1_weighted": np.nan, "auc_roc": np.nan}

    y_true = traded["y_true"].astype(int)
    y_pred = traded["y_pred"].astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # AUC: binary +1 vs rest (direction call)
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
# Performance metrics from equity curve
# ---------------------------------------------------------------------------

def _performance_metrics(
    trade_df     : pd.DataFrame,
    initial_cap  : float,
    ann_factor   : int,
    risk_free    : float,
) -> dict:
    """Compute Sharpe, Sortino, MDD, Calmar, Win rate, Profit factor."""
    rets = trade_df["net_ret"].values

    # Annualised return
    total_return = (trade_df["equity"].iloc[-1] - initial_cap) / initial_cap
    n_trades     = len(rets)

    # Trades per year: computed from actual trade timestamps, not bar count.
    # This is the correct annualisation base for per-trade returns.
    # trades are NOT spaced at bar frequency; they have variable holding periods.
    if n_trades > 1:
        span_years = (trade_df.index[-1] - trade_df.index[0]).days / 365.25
        trades_per_year = n_trades / max(span_years, 1e-6)
    else:
        trades_per_year = ann_factor  # fallback for single trade

    ann_return = (1 + total_return) ** (trades_per_year / max(n_trades, 1)) - 1

    # Sharpe ratio: annualised via sqrt(trades_per_year).
    # Formula: (mean_trade_ret - rf_per_trade) / std_trade_ret * sqrt(tpy)
    # This is the standard per-trade Sharpe annualisation (Lo 2002).
    mean_ret     = np.mean(rets)
    std_ret      = np.std(rets, ddof=1) if n_trades > 1 else np.nan
    rf_per_trade = risk_free / max(trades_per_year, 1)
    sharpe = np.nan
    if std_ret and std_ret > 0:
        sharpe = ((mean_ret - rf_per_trade) / std_ret) * np.sqrt(trades_per_year)

    # Sortino ratio (downside std only)
    neg_rets = rets[rets < rf_per_trade]
    sortino  = np.nan
    if len(neg_rets) > 1:
        down_std = np.std(neg_rets, ddof=1)
        if down_std > 0:
            sortino = ((mean_ret - rf_per_trade) / down_std) * np.sqrt(trades_per_year)

    # Maximum drawdown from equity curve
    equity_curve = np.concatenate([[initial_cap], trade_df["equity"].values])
    running_max  = np.maximum.accumulate(equity_curve)
    drawdowns    = (equity_curve - running_max) / running_max
    max_dd       = float(np.min(drawdowns))   # negative number

    # Calmar ratio
    calmar = np.nan
    if max_dd < 0:
        calmar = ann_return / abs(max_dd)

    # Win rate and profit factor
    wins         = rets[rets > 0]
    losses       = rets[rets < 0]
    win_rate     = len(wins) / n_trades if n_trades > 0 else 0.0
    gross_profit = float(np.sum(wins))  if len(wins)   > 0 else 0.0
    gross_loss   = float(np.sum(losses)) if len(losses) > 0 else 0.0
    profit_factor = (gross_profit / abs(gross_loss)
                     if gross_loss < 0 else np.nan)

    return {
        "total_return"    : total_return,
        "ann_return"      : ann_return,
        "sharpe_ratio"    : sharpe,
        "sortino_ratio"   : sortino,
        "max_drawdown"    : max_dd,
        "calmar_ratio"    : calmar,
        "win_rate"        : win_rate,
        "profit_factor"   : profit_factor,
        "avg_trade_return": float(mean_ret),
    }


# ---------------------------------------------------------------------------
# Empty metrics when no trades produced
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
        "n_trades": 0,
    }


# ---------------------------------------------------------------------------
# Log key metrics
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