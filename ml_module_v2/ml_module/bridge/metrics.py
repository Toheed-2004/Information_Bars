"""
bridge/metrics.py
-----------------
All research-grade metrics computed correctly.

Sharpe  — annualised using actual trade frequency, not sqrt(365).
          Reference: Lo (2002) "The Statistics of Sharpe Ratios"
          ann_factor = sqrt(trades_per_year)

Sortino — RMS of negative returns divided by TOTAL n, not loss-count.
          Reference: Sortino & van der Meer (1991)
          downside_dev = sqrt(sum(neg^2) / n_total)

CAGR    — from actual date range of first entry to last exit.

MaxDD   — from cumulative account returns, not position returns.

MCC     — from walk-forward fold metrics, mean ± std reported.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


def compute_backtest_metrics(
    ledger: pd.DataFrame,
    starting_balance: float = 10_000.0,
    trading_days_year: int = 365,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    ledger           : Trade ledger with columns:
                       entry_datetime, exit_datetime,
                       account_return_pct, cum_account_return, balance.
    starting_balance : Initial capital.
    trading_days_year: 365 for crypto, 252 for equities.

    Returns
    -------
    Dict of research metrics.
    """
    if ledger is None or ledger.empty:
        return _empty_metrics()

    ret = ledger["account_return_pct"].dropna() / 100.0
    n   = len(ret)
    if n == 0:
        return _empty_metrics()

    final_bal    = float(ledger["balance"].iloc[-1])
    total_return = (final_bal - starting_balance) / starting_balance

    # ── CAGR from actual date span ────────────────────────────────────────
    try:
        t0    = pd.to_datetime(ledger["entry_datetime"].iloc[0],  utc=True)
        t1    = pd.to_datetime(ledger["exit_datetime"].iloc[-1],  utc=True)
        years = max((t1 - t0).days / 365.25, 1 / 365.25)
        cagr  = (1 + total_return) ** (1 / years) - 1
    except Exception:
        years = 1.0
        cagr  = float("nan")

    # ── Sharpe — annualised by TRADE frequency, not calendar days ─────────
    mean_r = float(ret.mean())
    std_r  = float(ret.std())
    trades_per_year = n / years
    ann    = trades_per_year ** 0.5
    sharpe = (mean_r / std_r * ann) if std_r > 0 else float("nan")

    # ── Sortino — RMS of negative returns / total n (Sortino 1991) ────────
    neg = ret[ret < 0]
    if len(neg) > 0:
        downside_dev = float(((neg ** 2).sum() / n) ** 0.5)
        sortino = (mean_r / downside_dev * ann) if downside_dev > 0 else float("nan")
    else:
        sortino = float("nan")

    # ── MaxDD from cumulative account return ──────────────────────────────
    try:
        cum  = ledger["cum_account_return"].dropna() / 100.0 + 1.0
        peak = cum.cummax()
        dd   = (cum - peak) / peak
        max_dd = float(dd.min())
    except Exception:
        max_dd = float("nan")

    # ── Calmar ───────────────────────────────────────────────────────────
    calmar = (cagr / abs(max_dd)
              if (not np.isnan(cagr) and not np.isnan(max_dd) and max_dd < 0)
              else float("nan"))

    # ── Trade stats ───────────────────────────────────────────────────────
    wins   = ret[ret > 0]
    losses = ret[ret < 0]
    win_rate      = len(wins) / n
    avg_win       = float(wins.mean())   if len(wins)   else 0.0
    avg_loss      = float(losses.mean()) if len(losses) else 0.0
    gross_profit  = float(wins.sum())    if len(wins)   else 0.0
    gross_loss    = float(losses.sum())  if len(losses) else 0.0
    profit_factor = (abs(gross_profit / gross_loss)
                     if gross_loss != 0 else float("nan"))

    return {
        "total_return_pct":  round(total_return * 100, 4),
        "cagr_pct":          _r(cagr * 100),
        "sharpe":            _r(sharpe),
        "sortino":           _r(sortino),
        "calmar":            _r(calmar),
        "max_drawdown_pct":  _r(max_dd * 100),
        "n_trades":          n,
        "win_rate_pct":      round(win_rate * 100, 4),
        "avg_win_pct":       round(avg_win * 100, 4),
        "avg_loss_pct":      round(avg_loss * 100, 4),
        "profit_factor":     _r(profit_factor),
    }


def summarise_walk_forward(fold_metrics: List[Dict]) -> Dict[str, Any]:
    """Summary stats from walk-forward fold metrics."""
    if not fold_metrics:
        return {"accuracy_mean": float("nan"), "mcc_mean": float("nan")}
    accs = [m["accuracy"] for m in fold_metrics]
    mccs = [m["mcc"]      for m in fold_metrics]
    return {
        "n_folds":       len(fold_metrics),
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std":  round(float(np.std(accs)),  4),
        "mcc_mean":      round(float(np.mean(mccs)), 4),
        "mcc_std":       round(float(np.std(mccs)),  4),
    }


def _r(x): return None if (isinstance(x,float) and np.isnan(x)) else round(x, 4)

def _empty_metrics(): return {
    "total_return_pct": None, "cagr_pct": None,
    "sharpe": None, "sortino": None, "calmar": None,
    "max_drawdown_pct": None, "n_trades": 0,
    "win_rate_pct": None, "avg_win_pct": None,
    "avg_loss_pct": None, "profit_factor": None,
}
