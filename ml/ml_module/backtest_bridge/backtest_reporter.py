"""
ml/ml_module/backtest_bridge/backtest_reporter.py
--------------------------------------------------
Computes all research-grade metrics from a VBT portfolio object and
trade ledger, then saves them in formats ready for the paper.

Metrics computed
----------------
Return metrics:
  total_return_pct, cagr_pct, final_balance

Risk-adjusted:
  sharpe_ratio       — annualised, time-series based (from pf)
  sortino_ratio      — annualised, downside deviation
  calmar_ratio       — CAGR / max_drawdown

Drawdown:
  max_drawdown_pct
  avg_drawdown_pct
  max_drawdown_duration_bars

Trade statistics:
  total_trades, win_rate_pct, profit_factor
  avg_win_pct, avg_loss_pct
  avg_trade_return_pct, expectancy_pct

Exit breakdown:
  tp_exits, sl_exits, direction_change_exits, time_exits

Signal stats (from signal CSV):
  buy_signals_pct, sell_signals_pct, hold_signals_pct

Outputs
-------
  outputs/backtest_metrics_{bar_type}.json    ← full metrics dict
  outputs/equity_curve_{bar_type}.csv         ← timestamp, equity
  outputs/research_report.csv                 ← one row per bar type,
                                                 appended on each run

Usage
-----
    from ml.ml_module.backtest_bridge.backtest_reporter import BacktestReporter
    reporter = BacktestReporter(output_dir="outputs")
    metrics  = reporter.compute_and_save(pf, ledger, bar_type="dollar",
                                          df_signals=df_signals)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ml.ml_module.utils.helpers import get_logger, save_json

logger = get_logger(__name__)

# Trading days per year — used for annualisation
TRADING_DAYS_PER_YEAR = 365  # crypto never closes; use 252 for equities


class BacktestReporter:
    """
    Compute and persist all research metrics from a completed backtest.

    Parameters
    ----------
    output_dir         : Directory to write metric files.
    trading_days_year  : Used for annualisation (365 crypto, 252 equities).
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs",
        trading_days_year: int = TRADING_DAYS_PER_YEAR,
    ):
        self.output_dir = Path(output_dir)
        self.trading_days_year = trading_days_year
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute_and_save(
        self,
        pf,  # VBT Portfolio object
        ledger: pd.DataFrame,
        bar_type: str = "unknown",
        df_signals: Optional[pd.DataFrame] = None,
        starting_balance: float = 10_000.0,
    ) -> Dict[str, Any]:
        """
        Compute all metrics, save per-bar-type JSON + equity CSV, and
        append a row to the consolidated research_report.csv.

        Parameters
        ----------
        pf              : Portfolio object returned by VBTBacktestOptimized.run().
        ledger          : Trade ledger DataFrame from VBTBacktestOptimized.run().
        bar_type        : Label used in filenames and the report table.
        df_signals      : Signal DataFrame (datetime + signals) — used for
                          signal distribution stats.
        starting_balance: Initial capital (used if ledger balance is unavailable).

        Returns
        -------
        metrics dict (also saved to disk).
        """
        if ledger is None or ledger.empty:
            logger.warning("[%s] Ledger is empty — no trades to report.", bar_type)
            metrics = self._empty_metrics(bar_type)
        else:
            metrics = self._compute_metrics(
                pf, ledger, bar_type, df_signals, starting_balance
            )

        # Save per-bar-type JSON
        json_path = self.output_dir / f"backtest_metrics_{bar_type}.json"
        save_json(metrics, json_path)
        logger.info("[%s] Backtest metrics → %s", bar_type, json_path.name)

        # Save equity curve CSV
        self._save_equity_curve(
            pf,
            bar_type,
            ledger=(
                ledger
                if not (ledger is None or (hasattr(ledger, "empty") and ledger.empty))
                else None
            ),
        )

        # Append to consolidated research report
        self._append_to_report(metrics)

        return metrics

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        pf,
        ledger: pd.DataFrame,
        bar_type: str,
        df_signals: Optional[pd.DataFrame],
        starting_balance: float,
    ) -> Dict[str, Any]:

        metrics: Dict[str, Any] = {"bar_type": bar_type}

        # ---- Per-trade returns (account %, not position %) ----
        returns = ledger["account_return_pct"].dropna() / 100.0

        # ---- Equity curve ----
        final_balance = float(ledger["balance"].iloc[-1])
        total_return = (final_balance - starting_balance) / starting_balance

        # ---- CAGR ----
        try:
            t_start = pd.to_datetime(ledger["entry_datetime"].iloc[0], utc=True)
            t_end = pd.to_datetime(ledger["exit_datetime"].iloc[-1], utc=True)
            years = max((t_end - t_start).days / 365.25, 1 / 365.25)
            cagr = (1 + total_return) ** (1 / years) - 1
        except Exception:
            cagr = float("nan")

        # ---- Risk-adjusted — computed from ledger trade returns ----
        # VBT Pro pf methods vary by version; use ledger directly for reliability.
        # returns = per-trade account % change series (already computed above)
        trading_periods = self.trading_days_year

        try:
            mean_r = float(returns.mean())
            std_r = float(returns.std())
            down_r = (
                float(returns[returns < 0].std())
                if (returns < 0).any()
                else float("nan")
            )
            n = len(returns)

            # Annualise assuming n trades spread over ~1 year
            ann_factor = trading_periods**0.5
            sharpe = (mean_r / std_r * ann_factor) if std_r > 0 else float("nan")
            sortino = (
                (mean_r / down_r * ann_factor)
                if (down_r and down_r > 0)
                else float("nan")
            )
        except Exception:
            sharpe = sortino = float("nan")

        # ---- Drawdown from cum_account_return column in ledger ----
        try:
            cum = ledger["cum_account_return"].dropna() / 100.0 + 1.0
            peak = cum.cummax()
            dd_series = (cum - peak) / peak
            max_dd = float(dd_series.min())
            avg_dd = (
                float(dd_series[dd_series < 0].mean()) if (dd_series < 0).any() else 0.0
            )

            # Duration: consecutive trades in drawdown
            in_dd = (dd_series < 0).astype(int)
            changes = in_dd.diff().fillna(0)
            durations = []
            start_idx = None
            for i, (s, e) in enumerate(zip(changes == 1, changes == -1)):
                if s:
                    start_idx = i
                if e and start_idx is not None:
                    durations.append(i - start_idx)
                    start_idx = None
            max_dd_dur = int(max(durations)) if durations else 0
        except Exception:
            max_dd = avg_dd = float("nan")
            max_dd_dur = 0

        # ---- Calmar ----
        try:
            calmar = (
                float(cagr / abs(max_dd)) if (max_dd and max_dd < 0) else float("nan")
            )
        except Exception:
            calmar = float("nan")

        # ---- Trade statistics ----
        n_trades = len(ledger)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = float(len(wins) / n_trades) if n_trades else 0.0
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(losses.abs().sum()) if len(losses) else 0.0
        profit_factor = (
            round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf")
        )
        expectancy = float(returns.mean()) if len(returns) else 0.0
        avg_trade_ret = expectancy

        # ---- Exit type breakdown (from action column) ----
        if "action" in ledger.columns:
            actions = ledger["action"].fillna("Unknown")
            tp_exits = int(actions.str.startswith("TP").sum())
            sl_exits = int(actions.str.startswith("SL").sum())
            dir_exits = int(actions.str.contains("Direction", case=False).sum())
            time_exits = int(actions.str.contains("Time", case=False).sum())
        else:
            tp_exits = sl_exits = dir_exits = time_exits = 0

        # ---- Long / short breakdown ----
        if "direction" in ledger.columns:
            long_trades = int((ledger["direction"].str.lower() == "long").sum())
            short_trades = int((ledger["direction"].str.lower() == "short").sum())
        else:
            long_trades = short_trades = 0

        # ---- Signal distribution ----
        sig_stats = self._signal_stats(df_signals)

        # ---- Assemble ----
        metrics.update(
            {
                # Return
                "starting_balance": round(starting_balance, 2),
                "final_balance": round(final_balance, 2),
                "total_return_pct": round(total_return * 100, 4),
                "cagr_pct": round(cagr * 100, 4) if not np.isnan(cagr) else None,
                # Risk-adjusted
                "sharpe_ratio": round(sharpe, 4) if not np.isnan(sharpe) else None,
                "sortino_ratio": round(sortino, 4) if not np.isnan(sortino) else None,
                "calmar_ratio": round(calmar, 4) if not np.isnan(calmar) else None,
                # Drawdown
                "max_drawdown_pct": (
                    round(max_dd * 100, 4) if not np.isnan(max_dd) else None
                ),
                "avg_drawdown_pct": (
                    round(avg_dd * 100, 4) if not np.isnan(avg_dd) else None
                ),
                "max_drawdown_duration_bars": max_dd_dur,
                # Trade stats
                "total_trades": n_trades,
                "long_trades": long_trades,
                "short_trades": short_trades,
                "win_rate_pct": round(win_rate * 100, 4),
                "profit_factor": profit_factor,
                "avg_win_pct": round(avg_win * 100, 4),
                "avg_loss_pct": round(avg_loss * 100, 4),
                "avg_trade_return_pct": round(avg_trade_ret * 100, 4),
                "expectancy_pct": round(expectancy * 100, 4),
                "gross_profit_pct": round(gross_profit * 100, 4),
                "gross_loss_pct": round(gross_loss * 100, 4),
                # Exit breakdown
                "tp_exits": tp_exits,
                "sl_exits": sl_exits,
                "direction_change_exits": dir_exits,
                "time_exits": time_exits,
                # Signal stats
                **sig_stats,
            }
        )

        self._log_summary(metrics, bar_type)
        return metrics

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def _save_equity_curve(
        self, pf, bar_type: str, ledger: "pd.DataFrame" = None
    ) -> None:
        """Save equity curve from ledger balance column (more reliable than pf.value)."""
        try:
            if ledger is not None and not ledger.empty and "balance" in ledger.columns:
                # Use exit_datetime + balance from ledger — trade-by-trade equity
                dt_col = (
                    "exit_datetime"
                    if "exit_datetime" in ledger.columns
                    else ledger.columns[0]
                )
                df_eq = ledger[[dt_col, "balance"]].copy()
                df_eq.columns = ["datetime", "equity"]
            else:
                # Fallback: try pf.value
                equity = pf.value
                if hasattr(equity, "squeeze"):
                    equity = equity.squeeze()
                df_eq = equity.reset_index()
                df_eq.columns = ["datetime", "equity"]
            path = self.output_dir / f"equity_curve_{bar_type}.csv"
            df_eq.to_csv(path, index=False)
            logger.info(
                "[%s] Equity curve → %s  (%d rows)", bar_type, path.name, len(df_eq)
            )
        except Exception as e:
            logger.warning("[%s] Could not save equity curve: %s", bar_type, e)

    # ------------------------------------------------------------------
    # Consolidated research report
    # ------------------------------------------------------------------

    def _append_to_report(self, metrics: Dict[str, Any]) -> None:
        """
        Append one row to outputs/research_report.csv.
        Creates the file with headers on first call; appends thereafter.
        This gives you the full cross-bar-type comparison table for the paper.
        """
        report_path = self.output_dir / "research_report.csv"

        # Flat row — only scalar metrics (exclude nested objects)
        row = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float, str, type(None)))
        }

        df_row = pd.DataFrame([row])

        if report_path.exists():
            # Append without header; align columns to existing file
            existing = pd.read_csv(report_path)
            combined = pd.concat([existing, df_row], ignore_index=True)
            # Deduplicate by bar_type — keep last run
            combined.drop_duplicates(subset=["bar_type"], keep="last", inplace=True)
            combined.to_csv(report_path, index=False)
        else:
            df_row.to_csv(report_path, index=False)

        logger.info("Research report updated → %s", report_path.name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _signal_stats(self, df_signals: Optional[pd.DataFrame]) -> Dict[str, Any]:
        if df_signals is None or df_signals.empty:
            return {}
        sig_col = [c for c in df_signals.columns if c != "datetime"]
        if not sig_col:
            return {}
        s = df_signals[sig_col[0]]
        total = len(s)
        return {
            "total_signal_bars": total,
            "buy_signals_pct": round((s == 1).sum() / total * 100, 2),
            "sell_signals_pct": round((s == -1).sum() / total * 100, 2),
            "hold_signals_pct": round((s == 0).sum() / total * 100, 2),
        }

    def _empty_metrics(self, bar_type: str) -> Dict[str, Any]:
        return {
            "bar_type": bar_type,
            "total_trades": 0,
            "note": "No trades executed",
        }

    @staticmethod
    def _safe(fn, default):
        try:
            return fn()
        except Exception:
            return default

    def _log_summary(self, m: Dict, bar_type: str) -> None:
        logger.info(
            "[%s] Return=%.2f%%  Sharpe=%.3f  Sortino=%.3f  MaxDD=%.2f%%  "
            "Trades=%d  WinRate=%.1f%%  PF=%.3f",
            bar_type,
            m.get("total_return_pct", 0) or 0,
            m.get("sharpe_ratio", 0) or 0,
            m.get("sortino_ratio", 0) or 0,
            m.get("max_drawdown_pct", 0) or 0,
            m.get("total_trades", 0),
            m.get("win_rate_pct", 0) or 0,
            m.get("profit_factor", 0) or 0,
        )
