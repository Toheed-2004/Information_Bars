"""
mlfinlab/reporting/report.py
=============================
Stage 6 – Cross-bar comparison report for the research paper.

Input
-----
all_metrics : list[dict]
    One metrics dict per (bar_type × classifier) from Stage 5.

Output
------
  cross_bar_summary.csv   Machine-readable table (all metrics, all bar types)
  cross_bar_summary.tex   LaTeX table ready to paste into paper
  cross_bar_summary.md    Markdown version for quick reading

Table structure
---------------
Rows    = bar types (dollar_minute, dollar_tick, volume_minute, ..., 1h, 4h, ...)
Columns = Sharpe | Sortino | AUC | MDD | WinRate | Accuracy | F1
        | TotalReturn | NTrades | MinD | BarClass | Source

Each cell = mean across classifiers (RF, GB, SVM, XGB) for that bar type.
A separate detailed table keeps per-classifier breakdown.

Statistical note
----------------
All metrics are computed on out-of-sample predictions only.
In-sample metrics are never reported.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("mlfinlab.reporting")

# Columns included in the paper's main comparison table
PAPER_COLS = [
    "bar_type", "source", "bar_class",
    "n_events", "min_d",
    "accuracy", "f1_weighted", "auc_roc",
    "sharpe_ratio", "sortino_ratio", "max_drawdown",
    "calmar_ratio", "win_rate", "profit_factor",
    "total_return", "n_trades",
]

# Human-readable column names for LaTeX / display
COL_LABELS = {
    "bar_type"      : "Bar Type",
    "source"        : "Source",
    "bar_class"     : "Class",
    "n_events"      : "N Events",
    "min_d"         : "Min FFD d",
    "accuracy"      : "Accuracy",
    "f1_weighted"   : "F1 (wtd)",
    "auc_roc"       : "AUC-ROC",
    "sharpe_ratio"  : "Sharpe",
    "sortino_ratio" : "Sortino",
    "max_drawdown"  : "Max DD",
    "calmar_ratio"  : "Calmar",
    "win_rate"      : "Win %",
    "profit_factor" : "Profit Factor",
    "total_return"  : "Total Return",
    "n_trades"      : "N Trades",
}

# Columns formatted as percentages
PCT_COLS = {"max_drawdown", "win_rate", "total_return"}
# Columns formatted as 3 decimal places
FLOAT3_COLS = {"accuracy", "f1_weighted", "auc_roc", "sharpe_ratio",
               "sortino_ratio", "calmar_ratio", "profit_factor", "min_d"}


def build_report(
    all_metrics  : list[dict],
    run_results  : dict,
    output_dir   : Path,
    timestamp    : str,
) -> pd.DataFrame:
    """Build and save the cross-bar comparison table.

    Parameters
    ----------
    all_metrics : list[dict]
        One dict per (bar_type × classifier) from Stage 5.
    run_results : dict
        Stage 2 summary (n_events, min_d, label_dist per bar file).
    output_dir  : Path
    timestamp   : str

    Returns
    -------
    pd.DataFrame  The summary table (also saved to CSV, TEX, MD).
    """
    if not all_metrics:
        log.warning("  No metrics to report")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(all_metrics)

    # ── Enrich with Stage 2 data (n_events, min_d) ────────────────────────
    for name, res in run_results.items():
        mask = metrics_df["bar_type"] == res.get("meta", {}).get("bar_type", "?")
        if "events" in res:
            metrics_df.loc[mask, "n_events"] = res["events"]
        if "min_d" in res:
            metrics_df.loc[mask, "min_d"] = res["min_d"]
        if "label_dist" in res:
            dist = res["label_dist"]
            total = sum(dist.values())
            metrics_df.loc[mask, "label_m1_pct"] = dist.get(-1.0, 0) / total * 100 if total else 0
            metrics_df.loc[mask, "label_0_pct"]  = dist.get( 0.0, 0) / total * 100 if total else 0
            metrics_df.loc[mask, "label_p1_pct"] = dist.get( 1.0, 0) / total * 100 if total else 0

    # ── Primary table: mean across classifiers per bar type ───────────────
    numeric_cols = [c for c in metrics_df.columns
                    if c not in ("bar_type", "source", "bar_class",
                                 "feature_mode", "classifier")]
    agg = {c: "mean" for c in numeric_cols if c in metrics_df.columns}

    summary = (
        metrics_df
        .groupby(["bar_type", "source", "bar_class"], as_index=False)
        .agg(agg)
        .sort_values(["bar_class", "sharpe_ratio"],
                     ascending=[True, False])
    )

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = output_dir / f"cross_bar_summary__{timestamp}.csv"
    summary.to_csv(csv_path, index=False)
    log.info("  Saved CSV  -> %s", csv_path.name)

    # Detailed per-classifier table
    detail_path = output_dir / f"cross_bar_detail__{timestamp}.csv"
    metrics_df.to_csv(detail_path, index=False)
    log.info("  Saved detail CSV -> %s", detail_path.name)

    # ── Save Markdown ──────────────────────────────────────────────────────
    md_path = output_dir / f"cross_bar_summary__{timestamp}.md"
    _save_markdown(summary, md_path)
    log.info("  Saved MD   -> %s", md_path.name)

    # ── Save LaTeX ────────────────────────────────────────────────────────
    tex_path = output_dir / f"cross_bar_summary__{timestamp}.tex"
    _save_latex(summary, tex_path)
    log.info("  Saved TEX  -> %s", tex_path.name)

    # ── Log console summary ───────────────────────────────────────────────
    _log_console(summary)

    return summary


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val, col: str) -> str:
    """Format a single cell value for display."""
    if pd.isna(val):
        return "—"
    if col in PCT_COLS:
        return f"{val*100:+.1f}%" if col != "win_rate" else f"{val*100:.1f}%"
    if col in FLOAT3_COLS:
        return f"{val:.3f}"
    if col == "n_events" or col == "n_trades":
        return f"{int(val):,}"
    return str(val)


def _display_df(summary: pd.DataFrame) -> pd.DataFrame:
    """Build a formatted display DataFrame."""
    cols = [c for c in PAPER_COLS if c in summary.columns]
    disp = summary[cols].copy()
    for col in disp.columns:
        if col in COL_LABELS and col not in ("bar_type", "source", "bar_class"):
            disp[col] = disp[col].apply(lambda v: _fmt(v, col))
    disp.columns = [COL_LABELS.get(c, c) for c in disp.columns]
    return disp


def _save_markdown(summary: pd.DataFrame, path: Path) -> None:
    disp = _display_df(summary)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Cross-Bar Comparison: ML Performance\n\n")
        f.write("All metrics computed on **out-of-sample** predictions ")
        f.write("(walk-forward CV, values averaged across classifiers).\n\n")
        f.write(disp.to_markdown(index=False))
        f.write("\n\n")
        f.write("**Columns:** Sharpe/Sortino annualised. ")
        f.write("Max DD shown as percentage. ")
        f.write("Win% = fraction of trades with positive net P&L. ")
        f.write("Min FFD d = minimum fractional differencing order (ADF test).\n")


def _save_latex(summary: pd.DataFrame, path: Path) -> None:
    """Write a publication-ready LaTeX table."""
    cols = [c for c in PAPER_COLS if c in summary.columns]
    display_cols = ["bar_type", "source", "n_events",
                    "accuracy", "auc_roc",
                    "sharpe_ratio", "sortino_ratio",
                    "max_drawdown", "win_rate", "n_trades"]
    display_cols = [c for c in display_cols if c in summary.columns]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Out-of-sample ML performance across bar types. "
                 r"Sharpe and Sortino are annualised. "
                 r"Max DD is maximum drawdown. "
                 r"All metrics averaged across Random Forest, "
                 r"Gradient Boost, SVM, and XGBoost classifiers.}")
    lines.append(r"\label{tab:ml_performance}")

    n_cols = len(display_cols)
    col_spec = "ll" + "r" * (n_cols - 2)
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Header
    header = " & ".join(
        COL_LABELS.get(c, c).replace("_", r"\_")
        for c in display_cols
    )
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    # Group: information bars first, then calendar
    prev_class = None
    for _, row in summary.iterrows():
        bc = row.get("bar_class", "?")
        if bc != prev_class:
            if prev_class is not None:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{"
                         r"\textit{" + bc.capitalize() + r" bars}} \\")
            prev_class = bc

        cells = []
        for col in display_cols:
            val = row.get(col, np.nan)
            cells.append(_fmt(val, col).replace("%", r"\%").replace("+", r"+"))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _log_console(summary: pd.DataFrame) -> None:
    """Print a readable summary table to the log."""
    log.info("")
    log.info("  %-35s %8s %8s %8s %8s %8s",
             "Bar Type / Source", "Sharpe", "Sortino", "AUC", "MaxDD", "WinRate")
    log.info("  " + "-" * 77)
    for _, row in summary.iterrows():
        name = f"{row.get('bar_type','?')} / {row.get('source','?')}"
        log.info("  %-35s %8.3f %8.3f %8.3f %7.1f%% %7.1f%%",
                 name,
                 row.get("sharpe_ratio",  float("nan")),
                 row.get("sortino_ratio", float("nan")),
                 row.get("auc_roc",       float("nan")),
                 row.get("max_drawdown",  float("nan")) * 100,
                 row.get("win_rate",      float("nan")) * 100)