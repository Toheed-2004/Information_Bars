"""
compare_bars.py — Research-grade comparison of two bar construction methods.

Primary use case: compare minute-data bars vs tick-data bars of the same type.

Usage
-----
    python compare_bars.py <csv_a> <csv_b> [label_a] [label_b] [output_dir]

    csv_a, csv_b   paths to bar CSVs  (must have: datetime, open, high, low, close, volume)
    label_a/b      display names      (default: "Minute bars", "Tick bars")
    output_dir     output directory   (default: current directory)

Outputs
-------
    comparison_stats.txt    full statistical report with interpretation
    comparison_figure.png   publication-quality 9-panel figure

Figure layout (3×3)
-------------------
Row 1 — Structure
    [A] Close price + bar boundaries (how bars carve up the price series)
    [B] Bar size distribution with CV  (de Prado uniformity criterion)
    [C] Bars per day over time         (does frequency adapt to the market?)

Row 2 — Return quality
    [D] Return distribution vs normal  (fat tails, skew visible)
    [E] Return ACF lags 1–20           (serial correlation — key de Prado test)
    [F] Variance ratio profile q=2,4,8,16  (random walk test)

Row 3 — Volatility structure + scorecard
    [G] Rolling annualised volatility  (correctly annualised per bar frequency)
    [H] Squared-return ACF             (ARCH / volatility clustering)
    [I] Scorecard table                (all criteria, winner highlighted)

How to interpret the results
-----------------------------
Different bar counts between minute and tick pipelines are NORMAL. The two
pipelines calibrate targets from different data sources and will produce
different bar frequencies. What matters is quality, not quantity.

Key quality criteria (de Prado 2018):
  Bar size CV        → lower = more uniform bars = better
  Shannon entropy    → higher = more information per bar = better
  Autocorrelation    → closer to 0 = less serial dependence = better
  Ljung-Box p        → higher = no serial correlation = better
  |VR(q) − 1|        → closer to 0 = random walk = better
  Excess kurtosis    → closer to 0 = more normal returns = better
  Timeout %          → lower = bars close on signal, not time = better

A high timeout % (>10%) is a warning sign: the calibrated target is too
large and bars are mostly closed by the max_duration guard. Such bars
contain no market-structure information — they are just time bars with
extra steps. High serial correlation and BDS rejection often accompany this.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.stats import mannwhitneyu

from statsmodels.tsa.stattools import adfuller, kpss, bds
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# read_ohlcv — CSV fallback for standalone use (no bitpredict DB required).
# Reads pre-exported time bar CSVs from data_dir.
# File naming convention: {exchange}_{symbol}_{timeframe}.csv
# e.g. binance_btc_1h.csv, binance_btc_4h.csv
def read_ohlcv(exchange, symbol, timeframe, bar_type=None,
               start_date=None, end_date=None,
               return_timestamp=False, columns=None,
               **kwargs):
    """CSV-based replacement for bitpredict DB read_ohlcv."""
    import pandas as pd
    from pathlib import Path
    # Look in the same directory as the bar CSVs (data_dir is passed via load_time_bars)
    # We search common locations relative to this script
    candidates = [
        Path(__file__).parent / "data" / f"{exchange}_{symbol}_{timeframe}.csv",
        Path(__file__).parent.parent / "data" / "raw_data" / f"{exchange}_{symbol}_{timeframe}.csv",
        Path("data/raw_data") / f"{exchange}_{symbol}_{timeframe}.csv",
        Path("data/raw_data") / f"binance_btc_{timeframe}.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            if columns:
                present = [c for c in columns if c in df.columns]
                df = df[present] if present else df
            if start_date:
                dt_col = next((c for c in df.columns if "datetime" in c.lower() or "timestamp" in c.lower()), None)
                if dt_col:
                    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
                    df = df[df[dt_col] >= pd.Timestamp(start_date, tz="UTC")]
            if end_date:
                dt_col = next((c for c in df.columns if "datetime" in c.lower() or "timestamp" in c.lower()), None)
                if dt_col:
                    df = df[df[dt_col] <= pd.Timestamp(end_date, tz="UTC")]
            return df
    print(f"  [TIME BAR] CSV not found for {exchange} {symbol} {timeframe}. "
          f"Place {exchange}_{symbol}_{timeframe}.csv in data/raw_data/")
    return None

warnings.filterwarnings("ignore")

# ── Palette ───────────────────────────────────────────────────────────────────
CA = "#1F77B4"  # teal   — series A (minute information bars)
CB = "#D62728"  # amber  — series B (tick information bars)
CC = "#2CA02C"  # violet — series C (time bars — baseline)
DARK_BG = "white"
PANEL_BG = "#F5F5F5"
GRID_COL = "#CCCCCC"
TEXT_COL = "#1A1A1A"
DIM_COL = "#555555"
GREEN = "#4ADE80"
RED_C = "#F87171"
YELLOW = "#FACC15"

plt.rcParams.update(
    {
        # ── Figure & background ──────────────────────────────────────────
        "figure.facecolor":      "white",
        "figure.dpi":            300,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.facecolor":     "white",
        # ── Font — Times New Roman preferred (journal serif standard) ────
        "font.family":           "serif",
        "font.serif":            ["Times New Roman", "DejaVu Serif", "serif"],
        "font.weight":           "bold",
        "text.color":            "#1A1A1A",
        # ── Axes ─────────────────────────────────────────────────────────
        "axes.facecolor":        "#F5F5F5",
        "axes.edgecolor":        "#333333",
        "axes.linewidth":        1.2,
        "axes.labelcolor":       "#1A1A1A",
        "axes.labelweight":      "bold",
        "axes.labelsize":        13,
        "axes.titlecolor":       "#1A1A1A",
        "axes.titlesize":        13,
        "axes.titleweight":      "bold",
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        # ── Ticks — bold, near-black, large enough to read printed ───────
        "xtick.color":           "#1A1A1A",
        "ytick.color":           "#1A1A1A",
        "xtick.labelcolor":      "#1A1A1A",
        "ytick.labelcolor":      "#1A1A1A",
        "xtick.labelsize":       11,
        "ytick.labelsize":       11,
        "xtick.major.width":     1.2,
        "ytick.major.width":     1.2,
        "xtick.major.size":      5,
        "ytick.major.size":      5,
        "xtick.direction":       "out",
        "ytick.direction":       "out",
        # ── Grid ─────────────────────────────────────────────────────────
        "grid.color":            "#CCCCCC",
        "grid.linewidth":        0.5,
        "grid.linestyle":        "--",
        # ── Legend ───────────────────────────────────────────────────────
        "legend.facecolor":      "white",
        "legend.edgecolor":      "#555555",
        "legend.fontsize":       10,
        "legend.framealpha":     1.0,
        "legend.borderpad":      0.7,
        "legend.labelspacing":   0.45,
        "legend.handlelength":   2.5,
        "legend.handletextpad":  0.55,
        # ── Lines ────────────────────────────────────────────────────────
        "lines.linewidth":       1.8,
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_bars(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Find datetime column (prefer plain 'datetime' / 'datetime_end' over start)
    dt_col = next(
        (
            c
            for c in df.columns
            if c.lower() in ("datetime", "datetime_end", "timestamp")
        ),
        None,
    )
    if dt_col is None:
        raise ValueError(f"No datetime column found in {path}")
    df = df.rename(columns={dt_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Derived fields
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_range"] = (df["high"] - df["low"]) / df["open"]
    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.date
    df["label"] = label

    # Optional columns — fill NaN if absent
    for col in (
        "bar_size",
        "duration_seconds",
        "duration_minutes",
        "tick_count",
        "vwap",
        "buy_sell_imbalance",
    ):
        if col not in df.columns:
            df[col] = np.nan

    # Unify duration to seconds
    if df["duration_seconds"].isna().all() and df["duration_minutes"].notna().any():
        df["duration_seconds"] = df["duration_minutes"] * 60.0

    # Bars per year (for correct volatility annualisation)
    day_span = max((df["datetime"].iloc[-1] - df["datetime"].iloc[0]).days, 1)
    df.attrs["bars_per_year"] = len(df) / (day_span / 365.25)
    df.attrs["label"] = label

    bpd = df.groupby("date").size()
    print(
        f"  {label:38s} {len(df):>5} bars  "
        f"{df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}  "
        f"({bpd.mean():.1f} bars/day avg)"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════


def _iqr_filter(arr: np.ndarray) -> np.ndarray:
    """Remove outliers via IQR (so timeout bars don't distort CV)."""
    if len(arr) < 10:
        return arr
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = q75 - q25
    if iqr == 0:
        return arr
    lo, hi = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    filtered = arr[(arr >= lo) & (arr <= hi)]
    if len(filtered) < len(arr) * 0.7:
        lo, hi = q25 - 2.5 * iqr, q75 + 2.5 * iqr
        filtered = arr[(arr >= lo) & (arr <= hi)]
    return filtered if len(filtered) > 0 else arr


def _variance_ratio(r: np.ndarray, q: int) -> tuple:
    """
    Variance Ratio Test: Lo and MacKinlay (1988), Review of Financial Studies.

    Implements the exact paper formulas:
      var1  = (1/T) * sum((r_t - mu)^2)           [biased estimator, per paper]
      m     = q*(T-q+1)*(1-q/T)                   [normalisation in paper]
      var_q = (1/m) * sum((r_q - q*mu)^2)         [q-period return variance]
      VR(q) = var_q / var1
      z(q)  = (VR(q)-1) / sqrt(delta)             [homoskedastic z-stat]
      delta = 2*(2q-1)*(q-1) / (3qT)              [Lo-MacKinlay eq. 11]

    Standard q values for this paper: {2, 4, 8, 16}.
    """
    T = len(r)
    if T < q + 2:
        return np.nan, np.nan
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / T            # paper uses biased 1/T
    if var1 == 0:
        return np.nan, np.nan
    cs = np.concatenate([[0.0], np.cumsum(r)])
    r_q = cs[q:] - cs[:-q]                      # overlapping q-period returns
    m = q * (T - q + 1) * (1 - q / T)           # paper normalisation
    if m <= 0:
        return np.nan, np.nan
    var_q = np.sum((r_q - q * mu) ** 2) / m
    vr = var_q / var1
    delta = 2 * (2 * q - 1) * (q - 1) / (3 * q * T)
    z = (vr - 1) / np.sqrt(delta) if delta > 0 else np.nan
    p = float(2 * (1 - stats.norm.cdf(abs(z)))) if not np.isnan(z) else np.nan
    return round(float(vr), 4), round(p, 4)


def _shannon_entropy(r: np.ndarray, bins: int = 50) -> float:
    if len(r) < 10:
        return np.nan
    hist, _ = np.histogram(r, bins=bins, density=True)
    hist = hist[hist > 0]
    bw = (r.max() - r.min()) / bins
    probs = hist * bw
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log2(probs)))


def run_tests(df: pd.DataFrame, label: str) -> dict:
    nan = float("nan")
    out = {
        "label": label,
        "n_bars": len(df),
        "bars_per_day": round(df.groupby("date").size().mean(), 2),
        "bars_per_year": round(df.attrs.get("bars_per_year", nan), 0),
        "mean": nan,
        "std": nan,
        "skew": nan,
        "kurt": nan,
        "min": nan,
        "max": nan,
        "bs_cv": nan,
        "bs_mean": nan,
        "dur_mean_s": nan,
        "dur_median_s": nan,
        "dur_cv": nan,
        "timeout_pct": nan,
        "entropy": nan,
        "jb_stat": nan,
        "jb_p": nan,
        "sw_stat": nan,
        "sw_p": nan,
        "sw_n": nan,
        "adf_p": nan,
        "kpss_p": nan,
        "lb10_p": nan,
        "lb20_p": nan,
        "lb10_r2_p": nan,
        "ac1": nan,
        "ac2": nan,
        "ac5": nan,
        "arch_p": nan,
        "vr2": nan,
        "vr2_p": nan,
        "vr4": nan,
        "vr4_p": nan,
        "vr8": nan,
        "vr8_p": nan,
        "vr16": nan,
        "vr16_p": nan,
        "bds2_stat": nan,
        "bds2_p": nan,
        "bds3_stat": nan,
        "bds3_p": nan,
    }

    r = df["log_return"].dropna().values
    if len(r) < 20:
        print(f"  WARNING {label}: only {len(r)} returns, skipping most tests")
        return out

    out.update(
        {
            "mean": round(float(r.mean()), 7),
            "std": round(float(r.std(ddof=1)), 7),
            "skew": round(float(stats.skew(r)), 4),
            "kurt": round(float(stats.kurtosis(r)), 4),
            "min": round(float(r.min()), 6),
            "max": round(float(r.max()), 6),
            "entropy": round(_shannon_entropy(r), 4),
        }
    )

    # Bar size CV (IQR-filtered — removes timeout bars from the calculation)
    if df["bar_size"].notna().sum() > 10:
        bs_raw = df["bar_size"].dropna().values.astype(float)
        bs_f = _iqr_filter(bs_raw)
        if bs_f.mean() > 0:
            out["bs_mean"] = round(float(bs_f.mean()), 4)
            out["bs_cv"] = round(float(bs_f.std() / bs_f.mean()), 4)

    # Duration and timeout fraction
    if df["duration_seconds"].notna().sum() > 10:
        dur = df["duration_seconds"].dropna().values.astype(float)
        # Timeout heuristic: any bar within 1 % of the max observed duration
        max_dur = dur.max()
        n_timeout = int((dur >= max_dur * 0.99).sum())
        out["timeout_pct"] = round(100 * n_timeout / len(dur), 1)
        dur_f = _iqr_filter(dur)
        if dur_f.mean() > 0:
            out["dur_mean_s"] = round(float(dur_f.mean()), 0)
            out["dur_median_s"] = round(float(np.median(dur_f)), 0)
            out["dur_cv"] = round(float(dur_f.std() / dur_f.mean()), 3)

    # Normality
    jb, jb_p = stats.jarque_bera(r)
    out["jb_stat"] = round(float(jb), 2)
    out["jb_p"] = round(float(jb_p), 6)
    sw_n = min(5_000, len(r))
    sw_s, sw_p = stats.shapiro(r[:sw_n])
    out.update(
        {"sw_stat": round(float(sw_s), 4), "sw_p": round(float(sw_p), 6), "sw_n": sw_n}
    )

    # Stationarity
    try:
        adf_r = adfuller(r, autolag="AIC")
        out["adf_p"] = round(float(adf_r[1]), 4)
    except Exception:
        pass
    try:
        kpss_r = kpss(r, regression="c", nlags="auto")
        out["kpss_p"] = round(float(kpss_r[1]), 4)
    except Exception:
        pass

    # Serial correlation — returns
    try:
        lb = acorr_ljungbox(r, lags=[10, 20], return_df=True)
        out["lb10_p"] = round(float(lb["lb_pvalue"].iloc[0]), 4)
        out["lb20_p"] = round(float(lb["lb_pvalue"].iloc[1]), 4)
    except Exception:
        pass
    out["ac1"] = round(float(pd.Series(r).autocorr(1)), 4)
    out["ac2"] = round(float(pd.Series(r).autocorr(2)), 4)
    out["ac5"] = round(float(pd.Series(r).autocorr(5)), 4)

    # Serial correlation — squared returns (volatility clustering)
    try:
        lb2 = acorr_ljungbox(r**2, lags=[10], return_df=True)
        out["lb10_r2_p"] = round(float(lb2["lb_pvalue"].iloc[0]), 4)
    except Exception:
        pass

    # ARCH-LM
    try:
        arch = het_arch(r, nlags=10)
        out["arch_p"] = round(float(arch[1]), 4)
    except Exception:
        pass

    # Variance ratio
    for q, k in ((2, "vr2"), (4, "vr4"), (8, "vr8"), (16, "vr16")):
        vr, vr_p = _variance_ratio(r, q)
        out[k] = vr
        out[f"{k}_p"] = vr_p

    # BDS — nonlinear dependence
    # Run on the full series when possible.  When N > 10,000 use the most
    # recent 10,000 bars (tail) rather than a stride-sampled subsample.
    # A stride destroys short-lag temporal structure which is exactly what
    # BDS detects — striding would systematically understate dependence.
    # Taking the tail preserves consecutive ordering and focuses on the
    # most recent market regime.
    try:
        MAX_BDS_N = 10_000
        r_bds = r[-MAX_BDS_N:] if len(r) > MAX_BDS_N else r
        bds_r = bds(r_bds, max_dim=4)
        out["bds2_stat"] = round(float(bds_r[0][0]), 4)
        out["bds2_p"] = round(float(bds_r[1][0]), 4)
        out["bds3_stat"] = round(float(bds_r[0][1]), 4)
        out["bds3_p"] = round(float(bds_r[1][1]), 4)
    except Exception:
        pass

    return out


def compare_distributions(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    ra = a["log_return"].dropna().values
    rb = b["log_return"].dropna().values
    mw_u, mw_p = mannwhitneyu(ra, rb, alternative="two-sided")
    pooled_std = np.sqrt((ra.std(ddof=1) ** 2 + rb.std(ddof=1) ** 2) / 2)
    cohen_d = (ra.mean() - rb.mean()) / pooled_std if pooled_std > 0 else 0.0
    ks_s, ks_p = stats.ks_2samp(ra, rb)
    return {
        "mw_u": round(float(mw_u), 2),
        "mw_p": round(float(mw_p), 4),
        "cohen_d": round(float(cohen_d), 4),
        "ks_stat": round(float(ks_s), 4),
        "ks_p": round(float(ks_p), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════════════


def _fmt(v, d: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def _sig(p) -> str:
    try:
        p = float(p)
        if np.isnan(p):
            return ""
        if p < 0.01:
            return "***"
        if p < 0.05:
            return "**"
        if p < 0.10:
            return "*"
    except Exception:
        pass
    return ""


def _winner(
    va, vb, lower_better: bool, la: str, lb: str, vc=None, lc: str | None = None
) -> str:
    """
    Return the label of the best series among 2 or 3 candidates.

    When vc and lc are supplied (time bar series C is present), compares
    all three and returns the label of the best one.

    Tie threshold: within 2% relative difference.
    All comparisons use abs() so kurtosis / AC values work correctly
    (distance from zero is what matters, not sign).
    """
    try:
        vals: dict[str, float] = {la: abs(float(va)), lb: abs(float(vb))}
        if vc is not None and lc is not None:
            vals[lc] = abs(float(vc))

        # Any NaN → cannot determine winner
        if any(np.isnan(v) for v in vals.values()):
            return "—"

        # For higher-is-better metrics the caller passes negated values so
        # we can always minimise.  But actually we keep the raw values and
        # flip the comparison direction instead.
        if not lower_better:
            # Higher is better → the label with the largest raw (non-abs) value
            raw: dict[str, float] = {la: float(va), lb: float(vb)}
            if vc is not None and lc is not None:
                raw[lc] = float(vc)
            best_val = max(raw.values())
            best_lbl = max(raw, key=raw.get)
            # Tie check: all within 2% of the best
            if all(
                abs(v - best_val) / max(abs(best_val), 1e-12) < 0.02
                for v in raw.values()
            ):
                return "≈ tie"
            return best_lbl
        else:
            # Lower is better → smallest abs value wins
            best_val = min(vals.values())
            best_lbl = min(vals, key=vals.get)
            if all(
                abs(v - best_val) / max(abs(best_val), 1e-12) < 0.02
                for v in vals.values()
            ):
                return "≈ tie"
            return best_lbl
    except Exception:
        return "—"


def write_report(
    ta: dict,
    tb: dict,
    comp: dict,
    out_path: Path,
    tc: dict | None = None,  # time bar stats — optional
    comp_ac: dict | None = None,  # distribution comparison A vs C
    comp_bc: dict | None = None,  # distribution comparison B vs C
) -> tuple:
    """
    Write the statistical comparison report.

    When tc is supplied the report has three columns throughout:
        Minute bars | Tick bars | Time bars (baseline)
    When tc is None the report has two columns (original behaviour).

    Returns (score_a, score_b, criteria).
    criteria is a list of tuples used by plot_figure for the scorecard panel.
    """
    la, lb = ta["label"], tb["label"]
    lc = tc["label"] if tc else None
    has_c = tc is not None

    # Column widths adapt to 2 or 3 series
    W = 100 if has_c else 80
    C1 = 14  # width of each value column

    lines = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def h(txt):
        lines.append(f"\n{'═' * W}\n  {txt}\n{'═' * W}")

    def row(name: str, va, vb, note: str = "", d: int = 4, vc=None):
        """One data row.  vc is the time-bar value; ignored when has_c=False."""
        if has_c:
            c_col = f"{_fmt(vc, d):>{C1}}" if vc is not None else f"{'n/a':>{C1}}"
            lines.append(
                f"  {name:<34} {_fmt(va,d):>{C1}} {_fmt(vb,d):>{C1}} {c_col}  {note}"
            )
        else:
            lines.append(f"  {name:<34} {_fmt(va,d):>{C1}} {_fmt(vb,d):>{C1}}  {note}")

    def win2(va, vb, lower, vc=None):
        """Winner label for a metric — 2-way or 3-way."""
        return _winner(va, vb, lower, la, lb, vc, lc)

    def win_note(va, vb, lower, prefix: str = "", vc=None) -> str:
        w = win2(va, vb, lower, vc)
        return f"{prefix}winner: {w}"

    # ── Header ────────────────────────────────────────────────────────────────
    h("BAR COMPARISON — STATISTICAL REPORT")
    hdr = f"  {'Metric':<34} {la:>{C1}} {lb:>{C1}}"
    if has_c:
        hdr += f" {lc:>{C1}}"
    lines.append(hdr)
    lines.append(f"  {'-' * (W - 2)}")

    # ── Understanding section ─────────────────────────────────────────────────
    h("UNDERSTANDING THIS COMPARISON")
    desc = (
        "  Series A and B are information bars built from the same asset.\n"
        "  They differ only in DATA SOURCE:\n"
        "    A = calibrated and built from 1-minute OHLCV bars\n"
        "    B = calibrated and built from raw tick (aggTrade) data\n"
    )
    if has_c:
        desc += (
            f"    C = {lc} — fixed-interval TIME BARS (the null hypothesis baseline)\n\n"
            "  TIME BAR BASELINE: A time bar closes every N minutes by the clock,\n"
            "  regardless of market activity. It carries no information-sampling\n"
            "  advantage. If A or B cannot beat C on most criteria, the information\n"
            "  bar is not extracting structure the clock cannot already provide.\n"
        )
    desc += (
        "\n  DIFFERENT BAR COUNTS ARE NORMAL. The research question is quality,\n"
        "  not quantity.  Quality = low serial correlation + high entropy +\n"
        "  low kurtosis + uniform sizes + duration CV in adaptive range.\n"
        "  A high TIMEOUT % means bars are closing by time limit, not by signal."
    )
    lines.append(desc)

    # ── Overview ──────────────────────────────────────────────────────────────
    h("OVERVIEW")
    row(
        "Bar count",
        ta["n_bars"],
        tb["n_bars"],
        "",
        0,
        vc=tc["n_bars"] if has_c else None,
    )
    row(
        "Bars / day (mean)",
        ta["bars_per_day"],
        tb["bars_per_day"],
        "",
        4,
        vc=tc["bars_per_day"] if has_c else None,
    )
    row(
        "Timeout bar % (est)",
        _fmt(ta["timeout_pct"], 1),
        _fmt(tb["timeout_pct"], 1),
        "← > 10% means most bars carry no signal",
        vc=_fmt(tc["timeout_pct"], 1) if has_c else None,
    )
    row(
        "Mean log-return",
        ta["mean"],
        tb["mean"],
        "",
        7,
        vc=tc["mean"] if has_c else None,
    )
    row("Std log-return", ta["std"], tb["std"], "", 7, vc=tc["std"] if has_c else None)
    if not np.isnan(ta.get("dur_mean_s", np.nan)):
        row(
            "Duration mean (s)",
            ta["dur_mean_s"],
            tb["dur_mean_s"],
            "",
            0,
            vc=tc["dur_mean_s"] if has_c else None,
        )
        row(
            "Duration median (s)",
            ta["dur_median_s"],
            tb["dur_median_s"],
            "",
            0,
            vc=tc["dur_median_s"] if has_c else None,
        )
        row(
            "Duration CV",
            ta["dur_cv"],
            tb["dur_cv"],
            "← 0.3–0.8 = adaptive; ~0 = time-bar-like; >1 = chaotic",
            vc=tc["dur_cv"] if has_c else None,
        )

    # ── De Prado criteria ─────────────────────────────────────────────────────
    h("DE PRADO INFORMATION BAR CRITERIA  (primary quality metrics)")
    lines.append(
        "  BAR SIZE CV : lower = more uniform bars = better.\n"
        "  ENTROPY     : higher = more information per bar = better.\n"
        "  KURTOSIS    : |value| closer to 0 = more normal returns = better.\n"
        "  AUTOCORR    : closer to 0 = less serial dependence = better."
    )
    lines.append("")

    _g = lambda k: tc[k] if has_c else None  # shorthand: get tc value or None

    row(
        "Bar size CV",
        ta["bs_cv"],
        tb["bs_cv"],
        win_note(ta["bs_cv"], tb["bs_cv"], True, "lower better  |  ", vc=_g("bs_cv")),
        vc=_g("bs_cv"),
    )
    row(
        "Shannon entropy",
        ta["entropy"],
        tb["entropy"],
        win_note(
            ta["entropy"], tb["entropy"], False, "higher better |  ", vc=_g("entropy")
        ),
        vc=_g("entropy"),
    )
    row(
        "Excess kurtosis",
        ta["kurt"],
        tb["kurt"],
        win_note(ta["kurt"], tb["kurt"], True, "0 ideal       |  ", vc=_g("kurt")),
        vc=_g("kurt"),
    )
    row(
        "Skewness",
        ta["skew"],
        tb["skew"],
        win_note(ta["skew"], tb["skew"], True, "0 ideal       |  ", vc=_g("skew")),
        vc=_g("skew"),
    )
    row(
        "Autocorr lag-1",
        ta["ac1"],
        tb["ac1"],
        win_note(ta["ac1"], tb["ac1"], True, "0 ideal       |  ", vc=_g("ac1")),
        vc=_g("ac1"),
    )

    # ── Normality ─────────────────────────────────────────────────────────────
    h("NORMALITY  (returns of good information bars are closer to normal)")
    sw_note = (
        f"  Shapiro-Wilk used {ta['sw_n']} obs for {la}, " f"{tb['sw_n']} obs for {lb}"
    )
    if has_c:
        sw_note += f", {tc['sw_n']} obs for {lc}"
    sw_note += " (max 5000)."
    lines.append(
        "  H0: returns are normally distributed.  p > 0.05 → cannot reject normality.\n"
        "  Compare the JB statistic: LOWER = closer to normal = better quality bars.\n"
        + sw_note
    )
    lines.append("")
    row(
        "Jarque-Bera stat",
        ta["jb_stat"],
        tb["jb_stat"],
        win_note(
            ta["jb_stat"], tb["jb_stat"], True, "lower better  |  ", vc=_g("jb_stat")
        ),
        2,
        vc=_g("jb_stat"),
    )
    row(
        "Jarque-Bera p",
        ta["jb_p"],
        tb["jb_p"],
        _sig(
            min(
                filter(
                    lambda x: not np.isnan(x),
                    [ta["jb_p"], tb["jb_p"]] + ([tc["jb_p"]] if has_c else []),
                )
            )
        ),
        vc=_g("jb_p"),
    )

    # ── Stationarity ──────────────────────────────────────────────────────────
    h("STATIONARITY")
    lines.append(
        "  ADF: H0 = unit root.  p < 0.05 = stationary = good.\n"
        "  KPSS: H0 = stationary.  p > 0.05 = cannot reject stationarity = good."
    )
    lines.append("")
    row("ADF p-value", ta["adf_p"], tb["adf_p"], "< 0.05 desired", vc=_g("adf_p"))
    row("KPSS p-value", ta["kpss_p"], tb["kpss_p"], "> 0.05 desired", vc=_g("kpss_p"))

    # ── Serial correlation ────────────────────────────────────────────────────
    h("SERIAL CORRELATION IN RETURNS  (the most important de Prado criterion)")
    lines.append(
        "  The entire purpose of information bars is to produce returns with LESS\n"
        "  serial correlation than time bars.  H0: no serial correlation.\n"
        "  HIGH p-value = no correlation = GOOD.\n"
        "  Ljung-Box tests multiple lags simultaneously — more powerful than lag-1 alone."
    )
    lines.append("")
    row("Autocorr lag-1", ta["ac1"], tb["ac1"], "0 ideal", vc=_g("ac1"))
    row("Autocorr lag-2", ta["ac2"], tb["ac2"], "", vc=_g("ac2"))
    row("Autocorr lag-5", ta["ac5"], tb["ac5"], "", vc=_g("ac5"))
    row(
        "Ljung-Box p (10 lags)",
        ta["lb10_p"],
        tb["lb10_p"],
        f"{_sig(min(filter(lambda x: not np.isnan(x), [ta['lb10_p'], tb['lb10_p']] + ([tc['lb10_p']] if has_c else []))))}"
        f"  {win_note(ta['lb10_p'], tb['lb10_p'], False, vc=_g('lb10_p'))}",
        vc=_g("lb10_p"),
    )
    row(
        "Ljung-Box p (20 lags)",
        ta["lb20_p"],
        tb["lb20_p"],
        f"{_sig(min(filter(lambda x: not np.isnan(x), [ta['lb20_p'], tb['lb20_p']] + ([tc['lb20_p']] if has_c else []))))}"
        f"  {win_note(ta['lb20_p'], tb['lb20_p'], False, vc=_g('lb20_p'))}",
        vc=_g("lb20_p"),
    )

    # ── Volatility clustering ─────────────────────────────────────────────────
    h("VOLATILITY CLUSTERING  (ARCH / heteroskedasticity)")
    lines.append(
        "  Volatility clustering is expected in crypto — the question is degree.\n"
        "  Ljung-Box on r²: autocorrelation of squared returns."
    )
    lines.append("")
    row(
        "ARCH-LM p",
        ta["arch_p"],
        tb["arch_p"],
        win_note(ta["arch_p"], tb["arch_p"], False, vc=_g("arch_p")),
        vc=_g("arch_p"),
    )
    row(
        "LB p (r², 10 lags)",
        ta["lb10_r2_p"],
        tb["lb10_r2_p"],
        win_note(ta["lb10_r2_p"], tb["lb10_r2_p"], False, vc=_g("lb10_r2_p")),
        vc=_g("lb10_r2_p"),
    )

    # ── Variance ratio ────────────────────────────────────────────────────────
    h("VARIANCE RATIO TEST  (Lo-MacKinlay 1988)")
    lines.append(
        "  VR(q) = 1 under a random walk.  VR > 1: momentum.  VR < 1: mean-reversion.\n"
        "  |VR(q) − 1| closer to 0 = better."
    )
    lines.append("")
    for q, k in ((2, "vr2"), (4, "vr4"), (8, "vr8"), (16, "vr16")):
        da = abs(ta[k] - 1) if not np.isnan(ta.get(k, np.nan)) else np.nan
        db = abs(tb[k] - 1) if not np.isnan(tb.get(k, np.nan)) else np.nan
        dc = abs(tc[k] - 1) if (has_c and not np.isnan(tc.get(k, np.nan))) else None
        dev = f"|VR-1|: {_fmt(da,3)}/{_fmt(db,3)}"
        if has_c:
            dev += f"/{_fmt(dc,3)}"
        row(
            f"VR(q={q:2d})",
            ta[k],
            tb[k],
            f"{dev}  {win_note(da, db, True, vc=dc)}",
            vc=tc[k] if has_c else None,
        )

    # ── BDS ───────────────────────────────────────────────────────────────────
    h("NONLINEAR DEPENDENCE  (BDS Test)")
    lines.append(
        "  H0: returns are i.i.d.  p > 0.05 = cannot reject i.i.d. = GOOD.\n"
        "  BDS detects nonlinear dependence invisible to autocorrelation tests.\n"
        "  Note: run on the full return series (no subsampling)."
    )
    lines.append("")
    row("BDS dim=2 stat", ta["bds2_stat"], tb["bds2_stat"], "", vc=_g("bds2_stat"))
    row(
        "BDS dim=2 p",
        ta["bds2_p"],
        tb["bds2_p"],
        f"{_sig(min(filter(lambda x: not np.isnan(x), [ta['bds2_p'], tb['bds2_p']] + ([tc['bds2_p']] if has_c else []))))}"
        f"  {win_note(ta['bds2_p'], tb['bds2_p'], False, vc=_g('bds2_p'))}",
        vc=_g("bds2_p"),
    )
    row(
        "BDS dim=3 p",
        ta["bds3_p"],
        tb["bds3_p"],
        _sig(
            min(
                filter(
                    lambda x: not np.isnan(x),
                    [ta["bds3_p"], tb["bds3_p"]] + ([tc["bds3_p"]] if has_c else []),
                )
            )
        ),
        vc=_g("bds3_p"),
    )

    # ── Distribution comparisons ──────────────────────────────────────────────
    h("DISTRIBUTION COMPARISONS")
    lines.append(
        "  Mann-Whitney U: central tendency.  p > 0.05 = no significant difference.\n"
        "  KS 2-sample: full distribution shape.  p > 0.05 = similar.\n"
        "  Cohen's d: < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large."
    )
    lines.append("")
    lines.append("  A vs B  (Minute vs Tick):")
    row("  Mann-Whitney p", comp["mw_p"], "", "< 0.05 = distributions differ")
    row("  KS p-value", comp["ks_p"], "", _sig(comp["ks_p"]))
    row("  Cohen's d", comp["cohen_d"], "")
    if has_c and comp_ac and comp_bc:
        lines.append("")
        lines.append(f"  A vs C  (Minute vs {lc}):")
        row("  Mann-Whitney p", comp_ac["mw_p"], "", "< 0.05 = different from time bar")
        row("  KS p-value", comp_ac["ks_p"], "", _sig(comp_ac["ks_p"]))
        row("  Cohen's d", comp_ac["cohen_d"], "")
        lines.append("")
        lines.append(f"  B vs C  (Tick vs {lc}):")
        row("  Mann-Whitney p", comp_bc["mw_p"], "", "< 0.05 = different from time bar")
        row("  KS p-value", comp_bc["ks_p"], "", _sig(comp_bc["ks_p"]))
        row("  Cohen's d", comp_bc["cohen_d"], "")
        lines.append(
            "\n  HOW TO READ: If A and B are clearly different from C (KS p << 0.05)\n"
            "  but similar to each other, the information bars are extracting structure\n"
            "  the clock cannot provide.  If A or B resembles C more than each other,\n"
            "  that pipeline is behaving like a time bar."
        )

    h("SIGNIFICANCE CODES")
    lines.append("  *** p < 0.01   ** p < 0.05   * p < 0.10   (none) p ≥ 0.10")

    # ── Scorecard ─────────────────────────────────────────────────────────────
    # criteria tuples: (name, va, vb, lower_better, vc)
    # vc is None when has_c=False — plot_figure handles both cases.
    def _abs(t, k):
        v = t.get(k, np.nan)
        return abs(v) if not np.isnan(v) else np.nan

    criteria = [
        ("Bar size CV", ta["bs_cv"], tb["bs_cv"], True, _g("bs_cv")),
        ("Entropy", ta["entropy"], tb["entropy"], False, _g("entropy")),
        (
            "|Kurtosis|",
            _abs(ta, "kurt"),
            _abs(tb, "kurt"),
            True,
            _abs(tc, "kurt") if has_c else None,
        ),
        (
            "|AC lag-1|",
            _abs(ta, "ac1"),
            _abs(tb, "ac1"),
            True,
            _abs(tc, "ac1") if has_c else None,
        ),
        ("LB p (10)", ta["lb10_p"], tb["lb10_p"], False, _g("lb10_p")),
        ("JB stat", ta["jb_stat"], tb["jb_stat"], True, _g("jb_stat")),
        (
            "|VR(4)−1|",
            # Lo-MacKinlay (1988) standard q values: {2, 4, 8, 16}
            abs(ta["vr4"] - 1) if not np.isnan(ta.get("vr4", np.nan)) else np.nan,
            abs(tb["vr4"] - 1) if not np.isnan(tb.get("vr4", np.nan)) else np.nan,
            True,
            (
                abs(tc["vr4"] - 1)
                if (has_c and not np.isnan(tc.get("vr4", np.nan)))
                else None
            ),
        ),
        ("Timeout %", ta["timeout_pct"], tb["timeout_pct"], True, _g("timeout_pct")),
    ]

    h("OVERALL SCORECARD  (de Prado information bar criteria)")
    score_a = score_b = score_c = ties = 0

    # Header row
    sc_hdr = f"  {'Criterion':<18} {la[:C1]:>{C1}} {lb[:C1]:>{C1}}"
    if has_c:
        sc_hdr += f" {lc[:C1]:>{C1}}"
    lines.append(sc_hdr)
    lines.append("")

    for name, va, vb, lower_better, vc in criteria:
        w = win2(va, vb, lower_better, vc)
        wa = "←" if w == la else ("↔" if "tie" in w else "  ")
        wb = "←" if w == lb else ("↔" if "tie" in w else "  ")
        sc_row = f"  {name:<18} {_fmt(va,3):>{C1}} {_fmt(vb,3):>{C1}}   {wa} {wb}"
        if has_c:
            wc = "←" if w == lc else ("↔" if "tie" in w else "  ")
            sc_row += f" {_fmt(vc,3) if vc is not None else 'n/a':>{C1}}   {wc}"
        lines.append(sc_row)

        if w == la:
            score_a += 1
        elif w == lb:
            score_b += 1
        elif has_c and w == lc:
            score_c += 1
        else:
            ties += 1

    score_line = (
        f"\n  Scored {len(criteria)} criteria  |  "
        f"{la}: {score_a}  |  {lb}: {score_b}"
    )
    if has_c:
        score_line += f"  |  {lc}: {score_c}"
    score_line += f"  |  ties: {ties}"
    lines.append(score_line)

    # Verdict — compare all available series
    scores = {la: score_a, lb: score_b}
    if has_c:
        scores[lc] = score_c
    best = max(scores, key=scores.get)
    best_n = scores[best]
    is_unique = list(scores.values()).count(best_n) == 1

    if is_unique:
        lines.append(
            f"\n  ► {best} shows better information bar properties "
            f"on {best_n}/{len(criteria)} criteria."
        )
    else:
        lines.append("\n  ► No clear winner — results are mixed across criteria.")

    lines.append(
        "\n  IMPORTANT CAVEAT: 'Better' here means the bar better satisfies\n"
        "  information-bar construction theory (de Prado 2018). It does NOT\n"
        "  imply better out-of-sample trading performance. A strategy backtest\n"
        "  is needed to assess practical trading value."
    )

    text = "\n".join(lines)
    print("\n" + text)
    out_path.write_text(text, encoding="utf-8")
    print(f"\n  Report → {out_path}")
    return score_a, score_b, criteria


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════


def _draw_candles(ax, df: pd.DataFrame, offset: float, width: float, n: int):
    """OHLC candlesticks as coloured rectangles + wicks."""
    subset = df.head(n).reset_index(drop=True)
    for i, row in subset.iterrows():
        x = i * 2 + offset
        o, h, lo, c = row["open"], row["high"], row["low"], row["close"]
        col = GREEN if c >= o else RED_C
        ax.add_patch(
            Rectangle(
                (x - width / 2, min(o, c)),
                width,
                max(abs(c - o), (h - lo) * 0.003),
                facecolor=col,
                edgecolor=col,
                linewidth=0,
                alpha=0.85,
            )
        )
        ax.plot([x, x], [lo, min(o, c)], color=col, lw=0.7, alpha=0.7)
        ax.plot([x, x], [max(o, c), h], color=col, lw=0.7, alpha=0.7)


def _rolling_vol(df: pd.DataFrame, roll: int) -> pd.Series:
    bpy = df.attrs.get("bars_per_year", 252)
    return df["log_return"].rolling(roll).std() * np.sqrt(bpy)


def _label_winner(w: str, la: str, color_a: str, color_b: str, default: str) -> str:
    if w == la:
        return color_a
    if "tie" in w:
        return YELLOW
    return color_b


def plot_figure(
    a: pd.DataFrame,
    b: pd.DataFrame,
    ta: dict,
    tb: dict,
    comp: dict,
    score_a: int,
    score_b: int,
    criteria: list,
    out_path: Path,
    tc: dict | None = None,
):
    """
    Clean 9-panel research figure saved as PDF (vector, print-quality).

    Design principles
    -----------------
    - One title per panel, no subtitle clutter — key stat in the title only.
    - Bar size distribution: each series normalised to z-scores so they share
      one axis regardless of unit differences (USD vs BTC vs dimensionless).
    - Variance ratio: y-axis zoomed to [0.7, 1.3] so deviations from 1.0 are
      clearly visible rather than compressed near the top of a [0, 1] axis.
    - ACF confidence bands: single shared ±95% CI band (shaded region) instead
      of four separate dashed lines.
    - Scorecard: verdict framed as "leads on N/8 criteria" — not "wins/loses",
      consistent with de Prado's framing. Prominent caveat footnote.
    - Output: PDF (vector) for publication; PNG can still be used if needed.
    """
    la, lb = ta["label"], tb["label"]
    lc: str | None = tc["label"] if tc is not None else None
    ra = a["log_return"].dropna().values
    rb = b["log_return"].dropna().values

    # ── Layout ────────────────────────────────────────────────────────────────
    # 18 × 14 in at 300 dpi gives ~5400 × 4200 px — comfortably print-ready
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    fig.suptitle(
        f"Information Bar Comparison:  {la}  vs  {lb}",
        fontsize=16,
        color=TEXT_COL,
        y=0.995,
        fontweight="bold",
        fontfamily="serif",
    )
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        hspace=0.65,    # extra row spacing so panel titles don't crowd tick labels
        wspace=0.42,    # extra column spacing for y-axis labels
        top=0.94,
        bottom=0.06,
        left=0.09,
        right=0.97,
    )

    def _ax(row, col):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
            sp.set_linewidth(1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            colors=TEXT_COL, labelsize=11, length=5, width=1.2,
            direction="out", top=False, right=False,
            labelcolor=TEXT_COL,
        )
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7, linestyle="--")
        return ax

    def _bold_ticks(ax):
        """Enforce bold, near-black tick labels after any draw call."""
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
            lbl.set_color(TEXT_COL)
            lbl.set_fontsize(11)

    def _title(ax, main: str, stat: str = ""):
        """Single-line bold title with optional muted stat suffix."""
        if stat:
            ax.set_title(
                f"{main}   —   {stat}",
                fontsize=13,
                color=TEXT_COL,
                pad=7,
                loc="left",
                fontweight="bold",
            )
        else:
            ax.set_title(
                main, fontsize=13, color=TEXT_COL, pad=7, loc="left", fontweight="bold"
            )

    def _leg(ax, loc="upper right"):
        """Legend inside axes — used for panels with unambiguous space."""
        leg = ax.legend(
            fontsize=10,
            loc=loc,
            facecolor="white",
            edgecolor="#555555",
            framealpha=1.0,
            borderpad=0.6,
        )
        if leg:
            for t in leg.get_texts():
                t.set_color(TEXT_COL)
                t.set_fontweight("bold")

    # ── [A] Close price ───────────────────────────────────────────────────────
    axA = _ax(0, 0)
    axA.plot(a["datetime"], a["close"], color=CA, lw=1.8, alpha=0.95, label=la)
    axA.plot(b["datetime"], b["close"], color=CB, lw=1.8, alpha=0.80, label=lb)
    axA.set_ylabel("Price (USD)", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
    axA.set_xlabel("Date", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
    _leg(axA)
    _bold_ticks(axA)
    _title(
        axA,
        "Close Price",
        f"A: {ta['bars_per_day']:.1f} bars/day  ·  B: {tb['bars_per_day']:.1f} bars/day",
    )
    axA.margins(x=0.01)
    # Annotate: denser line = more bars forming in that period
    axA.text(
        0.01,
        0.03,
        "denser line = more bars formed",
        transform=axA.transAxes,
        fontsize=11,
        color=DIM_COL,
        style="italic",
    )

    # ── [B] Bar size distribution — z-score normalised ────────────────────────
    # Normalising to z-scores removes unit differences (USD vs BTC vs %)
    # while preserving the shape (CV) comparison that actually matters.
    axB = _ax(0, 1)
    has_a = a["bar_size"].notna().sum() > 10
    has_b = b["bar_size"].notna().sum() > 10
    if has_a or has_b:
        if has_a:
            bs_a = _iqr_filter(a["bar_size"].dropna().values.astype(float))
            z_a = (bs_a - bs_a.mean()) / bs_a.std()
            p1, p99 = np.percentile(z_a, [1, 99])
            bins_a = np.linspace(p1, p99, 45)
            axB.hist(
                z_a,
                bins=bins_a,
                color=CA,
                alpha=0.70,
                density=True,
                label=f"{la}   CV={_fmt(ta['bs_cv'],3)}",
            )
        if has_b:
            bs_b = _iqr_filter(b["bar_size"].dropna().values.astype(float))
            z_b = (bs_b - bs_b.mean()) / bs_b.std()
            p1, p99 = np.percentile(z_b, [1, 99])
            bins_b = np.linspace(p1, p99, 45)
            axB.hist(
                z_b,
                bins=bins_b,
                color=CB,
                alpha=0.70,
                density=True,
                label=f"{lb}   CV={_fmt(tb['bs_cv'],3)}",
            )
        w_cv = _winner(ta["bs_cv"], tb["bs_cv"], True, la, lb)
        _leg(axB)
        axB.set_xlabel(
            "Bar size  (z-score, IQR-filtered)",
            fontsize=12,
            color=TEXT_COL,
            fontweight="bold",
            labelpad=5,
        )
        axB.set_ylabel("Density", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
        _title(axB, "Bar Size Uniformity", f"CV lower = more uniform  ·  leads: {w_cv}")
    else:
        axB.hist(
            a["hl_range"].dropna(), bins=40, color=CA, alpha=0.7, density=True, label=la
        )
        axB.hist(
            b["hl_range"].dropna(), bins=40, color=CB, alpha=0.7, density=True, label=lb
        )
        _leg(axB)
        _title(axB, "High-Low Range  (bar_size unavailable)")
    _bold_ticks(axB)

    # ── [C] Bars per day ──────────────────────────────────────────────────────
    axC = _ax(0, 2)
    bpd_a = a.groupby("date").size()
    bpd_b = b.groupby("date").size()
    axC.plot(
        pd.to_datetime(list(bpd_a.index)),
        bpd_a.values,
        color=CA,
        lw=1.8,
        alpha=0.9,
        label=f"{la}   μ = {ta['bars_per_day']:.1f}",
    )
    axC.plot(
        pd.to_datetime(list(bpd_b.index)),
        bpd_b.values,
        color=CB,
        lw=1.8,
        alpha=0.9,
        label=f"{lb}   μ = {tb['bars_per_day']:.1f}",
    )
    axC.set_ylabel("Bars per day", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
    _leg(axC)
    _bold_ticks(axC)
    _title(axC, "Bar Formation Frequency", "variation with market activity = good")

    # ── [D] Return distribution ───────────────────────────────────────────────
    axD = _ax(1, 0)
    lo = min(np.percentile(ra, 0.5), np.percentile(rb, 0.5))
    hi = max(np.percentile(ra, 99.5), np.percentile(rb, 99.5))
    bins = np.linspace(lo, hi, 60)
    axD.hist(ra, bins=bins, color=CA, alpha=0.55, density=True, label=la)
    axD.hist(rb, bins=bins, color=CB, alpha=0.55, density=True, label=lb)
    xr = np.linspace(lo, hi, 300)
    axD.plot(
        xr,
        stats.norm.pdf(xr, ra.mean(), ra.std()),
        color=CA,
        lw=2.0,
        ls="--",
        alpha=0.9,
    )
    axD.plot(
        xr,
        stats.norm.pdf(xr, rb.mean(), rb.std()),
        color=CB,
        lw=2.0,
        ls="--",
        alpha=0.9,
    )
    w_kurt = _winner(ta["kurt"], tb["kurt"], True, la, lb)
    axD.legend(
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        facecolor="white",
        edgecolor="#555555",
        framealpha=1.0,
    )
    for t in axD.get_legend().get_texts():
        t.set_color(TEXT_COL)
        t.set_fontweight("bold")
    axD.set_xlabel("Log return", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
    axD.set_ylabel("Density", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
    _bold_ticks(axD)
    _title(
        axD,
        "Return Distribution  (dashed = fitted normal)",
        f"excess kurtosis  A={_fmt(ta['kurt'],2)}  B={_fmt(tb['kurt'],2)}  ·  lower = better",
    )

    # ── [E] ACF of returns ────────────────────────────────────────────────────
    axE = _ax(1, 1)
    max_lag = max(5, min(20, len(ra) // 4, len(rb) // 4))
    lags = np.arange(1, max_lag + 1)
    ac_a = [float(pd.Series(ra).autocorr(int(l))) for l in lags]
    ac_b = [float(pd.Series(rb).autocorr(int(l))) for l in lags]
    conf_a = 1.96 / np.sqrt(len(ra))
    conf_b = 1.96 / np.sqrt(len(rb))
    bw = 0.35
    axE.bar(
        lags - bw / 2,
        ac_a,
        bw,
        color=CA,
        alpha=0.85,
        label=f"{la}   LB₁₀ p={_fmt(ta['lb10_p'],3)}",
    )
    axE.bar(
        lags + bw / 2,
        ac_b,
        bw,
        color=CB,
        alpha=0.85,
        label=f"{lb}   LB₁₀ p={_fmt(tb['lb10_p'],3)}",
    )
    # Single shared ±95% CI band — much cleaner than 4 dashed lines
    conf_mid = (conf_a + conf_b) / 2
    axE.fill_between(
        [-0.5, max_lag + 0.5], -conf_mid, conf_mid, color="white", alpha=0.06
    )
    axE.axhline(conf_mid, color=DIM_COL, lw=1.2, ls="--", alpha=0.8)
    axE.axhline(-conf_mid, color=DIM_COL, lw=1.2, ls="--", alpha=0.8)
    axE.axhline(0, color="#AAAAAA", lw=0.5, alpha=0.2)
    axE.set_xlim(0.5, max_lag + 0.5)
    w_lb = _winner(ta["lb10_p"], tb["lb10_p"], False, la, lb)
    _leg(axE)
    _bold_ticks(axE)
    axE.set_xlabel("Lag", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
    axE.set_ylabel("Autocorrelation", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
    _title(
        axE,
        "Return ACF  (Ljung-Box test)",
        f"bars outside band = significant  ·  higher p = better  ·  leads: {w_lb}",
    )

    # ── [F] Variance ratio — zoomed y-axis ────────────────────────────────────
    axF = _ax(1, 2)
    qs = [2, 5, 8, 16]
    vr_a = [ta.get(f"vr{q}", np.nan) for q in qs]
    vr_b = [tb.get(f"vr{q}", np.nan) for q in qs]
    x = np.arange(len(qs))
    axF.bar(x - 0.22, vr_a, 0.40, color=CA, alpha=0.85, label=la)
    axF.bar(x + 0.22, vr_b, 0.40, color=CB, alpha=0.85, label=lb)
    axF.axhline(
        1.0, color="white", lw=2.0, ls="--", alpha=0.85, label="Random walk  (VR = 1)"
    )
    axF.set_xticks(x)
    axF.set_xticklabels([f"q = {q}" for q in qs], fontsize=14)
    # Zoom y-axis so deviations from 1 are clearly visible
    all_vr = [v for v in vr_a + vr_b if not np.isnan(v)]
    if all_vr:
        vr_lo = min(0.80, min(all_vr) - 0.05)
        vr_hi = max(1.20, max(all_vr) + 0.05)
        axF.set_ylim(vr_lo, vr_hi)
    d_a = abs(ta.get("vr4", np.nan) - 1)
    d_b = abs(tb.get("vr4", np.nan) - 1)
    w_vr = _winner(d_a, d_b, True, la, lb)
    _leg(axF)
    _bold_ticks(axF)
    axF.set_ylabel("VR(q)", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
    _title(
        axF,
        "Variance Ratio  (Lo-MacKinlay 1988)",
        f"|VR(4)−1|:  A={_fmt(d_a,3)}  B={_fmt(d_b,3)}  ·  closer to 1 = better",
    )

    # ── [G] Rolling annualised volatility ─────────────────────────────────────
    axG = _ax(2, 0)
    roll = max(5, min(20, min(len(a), len(b)) // 10))
    axG.plot(
        a["datetime"],
        _rolling_vol(a, roll) * 100,
        color=CA,
        lw=1.8,
        alpha=0.9,
        label=la,
    )
    axG.plot(
        b["datetime"],
        _rolling_vol(b, roll) * 100,
        color=CB,
        lw=1.8,
        alpha=0.9,
        label=lb,
    )
    axG.set_ylabel("Ann. vol (%)", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
    axG.set_xlabel("Date", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
    axG.margins(x=0.01)
    _leg(axG)
    _bold_ticks(axG)
    _title(
        axG,
        f"Rolling Annualised Volatility  (window = {roll} bars)",
        "annualised with √(bars/year) per series",
    )

    # ── [H] Squared-return ACF (ARCH) ─────────────────────────────────────────
    axH = _ax(2, 1)
    ac_a2 = [float(pd.Series(ra**2).autocorr(int(l))) for l in lags]
    ac_b2 = [float(pd.Series(rb**2).autocorr(int(l))) for l in lags]
    axH.bar(
        lags - bw / 2,
        ac_a2,
        bw,
        color=CA,
        alpha=0.85,
        label=f"{la}   ARCH p={_fmt(ta['arch_p'],3)}",
    )
    axH.bar(
        lags + bw / 2,
        ac_b2,
        bw,
        color=CB,
        alpha=0.85,
        label=f"{lb}   ARCH p={_fmt(tb['arch_p'],3)}",
    )
    axH.fill_between(
        [-0.5, max_lag + 0.5], -conf_mid, conf_mid, color="white", alpha=0.06
    )
    axH.axhline(conf_mid, color=DIM_COL, lw=1.2, ls="--", alpha=0.8)
    axH.axhline(-conf_mid, color=DIM_COL, lw=1.2, ls="--", alpha=0.8)
    axH.axhline(0, color="#AAAAAA", lw=0.5, alpha=0.2)
    axH.set_xlim(0.5, max_lag + 0.5)
    w_arch = _winner(ta["arch_p"], tb["arch_p"], False, la, lb)
    _leg(axH)
    _bold_ticks(axH)
    axH.set_xlabel("Lag", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5)
    axH.set_ylabel(
        "Autocorrelation  (r²)", fontsize=12, color=TEXT_COL, fontweight="bold", labelpad=5
    )
    _title(
        axH,
        "Squared-Return ACF  (volatility clustering)",
        f"higher ARCH p = less clustering = better  ·  leads: {w_arch}",
    )

    # ── [I] Scorecard ─────────────────────────────────────────────────────────
    axI = _ax(2, 2)
    axI.axis("off")
    axI.set_facecolor(PANEL_BG)

    # Verdict: framed as "leads on N criteria" not "wins/loses"
    if score_a > score_b:
        verdict_line = f"{la}  leads on {score_a} of {len(criteria)} criteria"
    elif score_b > score_a:
        verdict_line = f"{lb}  leads on {score_b} of {len(criteria)} criteria"
    else:
        verdict_line = f"No clear leader  ({score_a}–{score_b})"

    cell_rows = []
    color_map = {}
    for i, (name, va, vb, lower_better, *rest) in enumerate(criteria):
        vc = rest[0] if rest else None
        w_c = _winner(va, vb, lower_better, la, lb, vc, lc)
        row_cells = [name, _fmt(va, 3), _fmt(vb, 3)]
        if vc is not None:
            row_cells.append(_fmt(vc, 3))
        cell_rows.append(row_cells)
        color_map[(i, 1)] = (
            GREEN if w_c == la else (YELLOW if "tie" in w_c else PANEL_BG)
        )
        color_map[(i, 2)] = (
            GREEN if w_c == lb else (YELLOW if "tie" in w_c else PANEL_BG)
        )
        if vc is not None:
            color_map[(i, 3)] = (
                GREEN if (lc and w_c == lc) else (YELLOW if "tie" in w_c else PANEL_BG)
            )

    la_short = la[:12] if len(la) > 12 else la
    lb_short = lb[:12] if len(lb) > 12 else lb
    col_labels = ["Criterion", la_short, lb_short]
    if any(len(r) > 3 for r in cell_rows):
        lc_short = lc[:12] if (lc and len(lc) > 12) else (lc or "Time")
        col_labels.append(lc_short)

    tbl = axI.table(
        cellText=cell_rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9 if len(col_labels) > 3 else 10)
    tbl.scale(1.0 if len(col_labels) > 3 else 1.05, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#DDEEFF")
            cell.set_text_props(color=TEXT_COL, fontweight="bold", fontsize=11)
        else:
            bg = color_map.get((r - 1, c), "white")
            cell.set_facecolor(bg)
            cell.set_text_props(color=TEXT_COL, fontsize=9, fontweight="bold")

    axI.set_title(
        verdict_line,
        fontsize=12,
        color=TEXT_COL,
        pad=10,
        loc="center",
        fontweight="bold",
    )
    # Caveat footnote — important for research integrity
    axI.text(
        0.5,
        -0.02,
        "Green = leads on this criterion  ·  Yellow = tie\n"
        "Leading on more criteria ≠ better trading performance",
        transform=axI.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color=DIM_COL,
        style="italic",
        multialignment="center",
    )

    # ── Save as PDF ───────────────────────────────────────────────────────────
    pdf_path = out_path.with_suffix(".pdf")
    # tight_layout pass resolves any remaining label/title overlap
    try:
        fig.set_layout_engine("tight", pad=0.5)
    except Exception:
        fig.tight_layout(pad=0.5)
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.10,
        facecolor="white",
        edgecolor="none",
        backend="pdf",
        dpi=300,
        metadata={"Creator": "compare_bars.py — 9-panel journal figure"},
    )
    plt.close()
    print(f"  Figure (PDF) → {pdf_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE FIGURE (Plotly HTML)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_interactive(
    a: pd.DataFrame,
    b: pd.DataFrame,
    ta: dict,
    tb: dict,
    comp: dict,
    score_a: int,
    score_b: int,
    criteria: list,
    out_path: Path,
):
    """
    9-panel interactive Plotly dashboard — mirrors the static figure exactly
    but is fully zoomable, hoverable, and exportable.

    Layout (3 rows × 3 cols):
      Row 1: Close Price | Bar Size Distribution | Bars per Day
      Row 2: Return Distribution | Return ACF | Variance Ratio
      Row 3: Rolling Volatility | Squared-Return ACF | Scorecard Table
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    la, lb = ta["label"], tb["label"]
    # lc used in scorecard — infer from criteria tuple length
    lc = None
    ra = a["log_return"].dropna().values
    rb = b["log_return"].dropna().values

    PLOTLY_BG = "white"
    PLOTLY_PANE = "#F5F5F5"
    PLOTLY_GRID = "#CCCCCC"
    PLOTLY_TEXT = "#1A1A1A"
    PLOTLY_DIM = "#555555"

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[
            "Close Price Over Time",
            "Bar Size Distribution  (IQR-filtered CV)",
            "Bars per Day Over Time",
            "Return Distribution vs Fitted Normal  (dashed)",
            "Return ACF  (Ljung-Box serial correlation test)",
            "Variance Ratio Profile  (Lo-MacKinlay 1988)",
            "Rolling Annualised Volatility",
            "Squared-Return ACF  (volatility clustering / ARCH)",
            "",  # scorecard table — no title needed, uses table header
        ],
        specs=[
            [{}, {}, {}],
            [{}, {}, {}],
            [{}, {}, {"type": "table"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    kw_a = dict(line_color=CA, legendgroup="a")
    kw_b = dict(line_color=CB, legendgroup="b")

    # ── [1,1] Close price ────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=a["datetime"],
            y=a["close"],
            mode="lines",
            name=la,
            line=dict(color=CA, width=0.9),
            legendgroup="a",
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Close: %{y:,.2f}<extra>"
            + la
            + "</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=b["datetime"],
            y=b["close"],
            mode="lines",
            name=lb,
            line=dict(color=CB, width=0.9),
            opacity=0.85,
            legendgroup="b",
            showlegend=True,
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Close: %{y:,.2f}<extra>"
            + lb
            + "</extra>",
        ),
        row=1,
        col=1,
    )

    # ── [1,2] Bar size distribution ───────────────────────────────────────────
    has_a = a["bar_size"].notna().sum() > 10
    has_b = b["bar_size"].notna().sum() > 10
    if has_a:
        bs_a = _iqr_filter(a["bar_size"].dropna().values.astype(float))
        fig.add_trace(
            go.Histogram(
                x=bs_a,
                histnorm="probability density",
                opacity=0.65,
                marker_color=CA,
                name=f"{la}  CV={_fmt(ta['bs_cv'],3)}",
                legendgroup="a",
                showlegend=False,
                hovertemplate="Bar size: %{x}<br>Density: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    if has_b:
        bs_b = _iqr_filter(b["bar_size"].dropna().values.astype(float))
        fig.add_trace(
            go.Histogram(
                x=bs_b,
                histnorm="probability density",
                opacity=0.65,
                marker_color=CB,
                name=f"{lb}  CV={_fmt(tb['bs_cv'],3)}",
                legendgroup="b",
                showlegend=False,
                hovertemplate="Bar size: %{x}<br>Density: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # ── [1,3] Bars per day ────────────────────────────────────────────────────
    bpd_a = a.groupby("date").size().reset_index(name="n")
    bpd_b = b.groupby("date").size().reset_index(name="n")
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(bpd_a["date"]),
            y=bpd_a["n"],
            mode="lines",
            line=dict(color=CA, width=1),
            name=f"{la}  μ={ta['bars_per_day']:.1f}/day",
            legendgroup="a",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Bars: %{y}<extra>" + la + "</extra>",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(bpd_b["date"]),
            y=bpd_b["n"],
            mode="lines",
            line=dict(color=CB, width=1),
            name=f"{lb}  μ={tb['bars_per_day']:.1f}/day",
            legendgroup="b",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Bars: %{y}<extra>" + lb + "</extra>",
        ),
        row=1,
        col=3,
    )

    # ── [2,1] Return distribution ─────────────────────────────────────────────
    lo = min(np.percentile(ra, 0.5), np.percentile(rb, 0.5))
    hi = max(np.percentile(ra, 99.5), np.percentile(rb, 99.5))
    bin_size = (hi - lo) / 60
    fig.add_trace(
        go.Histogram(
            x=ra,
            xbins=dict(start=lo, end=hi, size=bin_size),
            histnorm="probability density",
            opacity=0.55,
            marker_color=CA,
            legendgroup="a",
            showlegend=False,
            hovertemplate="Return: %{x:.5f}<br>Density: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=rb,
            xbins=dict(start=lo, end=hi, size=bin_size),
            histnorm="probability density",
            opacity=0.55,
            marker_color=CB,
            legendgroup="b",
            showlegend=False,
            hovertemplate="Return: %{x:.5f}<br>Density: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    # Fitted normal curves
    xr = np.linspace(lo, hi, 200)
    fig.add_trace(
        go.Scatter(
            x=xr,
            y=stats.norm.pdf(xr, ra.mean(), ra.std()),
            mode="lines",
            line=dict(color=CA, dash="dash", width=1.5),
            legendgroup="a",
            showlegend=False,
            hovertemplate="Fitted normal %{y:.2f}<extra>" + la + "</extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xr,
            y=stats.norm.pdf(xr, rb.mean(), rb.std()),
            mode="lines",
            line=dict(color=CB, dash="dash", width=1.5),
            legendgroup="b",
            showlegend=False,
            hovertemplate="Fitted normal %{y:.2f}<extra>" + lb + "</extra>",
        ),
        row=2,
        col=1,
    )

    # ── [2,2] Return ACF ──────────────────────────────────────────────────────
    max_lag = max(5, min(20, len(ra) // 4, len(rb) // 4))
    lags = list(range(1, max_lag + 1))
    ac_a = [float(pd.Series(ra).autocorr(l)) for l in lags]
    ac_b = [float(pd.Series(rb).autocorr(l)) for l in lags]
    conf_a = 1.96 / np.sqrt(len(ra))
    conf_b = 1.96 / np.sqrt(len(rb))
    w = 0.38
    fig.add_trace(
        go.Bar(
            x=[l - w / 2 for l in lags],
            y=ac_a,
            width=w,
            marker_color=CA,
            opacity=0.85,
            name=f"{la}  LB10p={_fmt(ta['lb10_p'],3)}",
            legendgroup="a",
            showlegend=False,
            hovertemplate="Lag %{x:.0f}<br>ACF: %{y:.4f}<extra>" + la + "</extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=[l + w / 2 for l in lags],
            y=ac_b,
            width=w,
            marker_color=CB,
            opacity=0.85,
            name=f"{lb}  LB10p={_fmt(tb['lb10_p'],3)}",
            legendgroup="b",
            showlegend=False,
            hovertemplate="Lag %{x:.0f}<br>ACF: %{y:.4f}<extra>" + lb + "</extra>",
        ),
        row=2,
        col=2,
    )
    for sign in (1, -1):
        fig.add_hline(
            y=sign * conf_a,
            line=dict(color=CA, dash="dot", width=1),
            opacity=0.55,
            row=2,
            col=2,
        )
        fig.add_hline(
            y=sign * conf_b,
            line=dict(color=CB, dash="dot", width=1),
            opacity=0.55,
            row=2,
            col=2,
        )
    fig.add_hline(y=0, line=dict(color="white", width=0.5), opacity=0.2, row=2, col=2)

    # ── [2,3] Variance ratio ──────────────────────────────────────────────────
    qs = [2, 5, 8, 16]
    vr_a = [ta.get(f"vr{q}", np.nan) for q in qs]
    vr_b = [tb.get(f"vr{q}", np.nan) for q in qs]
    x_vr = list(range(len(qs)))
    fig.add_trace(
        go.Bar(
            x=[i - 0.22 for i in x_vr],
            y=vr_a,
            width=0.40,
            marker_color=CA,
            opacity=0.85,
            name=la,
            legendgroup="a",
            showlegend=False,
            customdata=[f"q={q}" for q in qs],
            hovertemplate="%{customdata}<br>VR: %{y:.4f}<extra>" + la + "</extra>",
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=[i + 0.22 for i in x_vr],
            y=vr_b,
            width=0.40,
            marker_color=CB,
            opacity=0.85,
            name=lb,
            legendgroup="b",
            showlegend=False,
            customdata=[f"q={q}" for q in qs],
            hovertemplate="%{customdata}<br>VR: %{y:.4f}<extra>" + lb + "</extra>",
        ),
        row=2,
        col=3,
    )
    fig.add_hline(
        y=1.0,
        line=dict(color="white", dash="dash", width=1.5),
        opacity=0.7,
        row=2,
        col=3,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_vr,
        ticktext=[f"q={q}" for q in qs],
        row=2,
        col=3,
    )

    # ── [3,1] Rolling annualised volatility ───────────────────────────────────
    roll = max(5, min(20, min(len(a), len(b)) // 10))
    rv_a = _rolling_vol(a, roll) * 100
    rv_b = _rolling_vol(b, roll) * 100
    fig.add_trace(
        go.Scatter(
            x=a["datetime"],
            y=rv_a,
            mode="lines",
            line=dict(color=CA, width=0.9),
            name=la,
            legendgroup="a",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Ann. vol: %{y:.1f}%<extra>"
            + la
            + "</extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=b["datetime"],
            y=rv_b,
            mode="lines",
            line=dict(color=CB, width=0.9),
            name=lb,
            legendgroup="b",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Ann. vol: %{y:.1f}%<extra>"
            + lb
            + "</extra>",
        ),
        row=3,
        col=1,
    )

    # ── [3,2] Squared-return ACF ──────────────────────────────────────────────
    ac_a2 = [float(pd.Series(ra**2).autocorr(l)) for l in lags]
    ac_b2 = [float(pd.Series(rb**2).autocorr(l)) for l in lags]
    fig.add_trace(
        go.Bar(
            x=[l - w / 2 for l in lags],
            y=ac_a2,
            width=w,
            marker_color=CA,
            opacity=0.85,
            name=f"{la}  ARCH-p={_fmt(ta['arch_p'],3)}",
            legendgroup="a",
            showlegend=False,
            hovertemplate="Lag %{x:.0f}<br>ACF(r²): %{y:.4f}<extra>" + la + "</extra>",
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=[l + w / 2 for l in lags],
            y=ac_b2,
            width=w,
            marker_color=CB,
            opacity=0.85,
            name=f"{lb}  ARCH-p={_fmt(tb['arch_p'],3)}",
            legendgroup="b",
            showlegend=False,
            hovertemplate="Lag %{x:.0f}<br>ACF(r²): %{y:.4f}<extra>" + lb + "</extra>",
        ),
        row=3,
        col=2,
    )
    for sign in (1, -1):
        fig.add_hline(
            y=sign * conf_a,
            line=dict(color=CA, dash="dot", width=1),
            opacity=0.55,
            row=3,
            col=2,
        )
        fig.add_hline(
            y=sign * conf_b,
            line=dict(color=CB, dash="dot", width=1),
            opacity=0.55,
            row=3,
            col=2,
        )
    fig.add_hline(y=0, line=dict(color="white", width=0.5), opacity=0.2, row=3, col=2)

    # ── [3,3] Scorecard table ─────────────────────────────────────────────────
    if score_a > score_b:
        verdict = f"{la} wins  ({score_a}–{score_b})"
    elif score_b > score_a:
        verdict = f"{lb} wins  ({score_b}–{score_a})"
    else:
        verdict = f"Tie  ({score_a}–{score_b})"

    tbl_names = [c[0] for c in criteria]
    tbl_va = [_fmt(c[1], 3) for c in criteria]
    tbl_vb = [_fmt(c[2], 3) for c in criteria]
    tbl_vc = [
        _fmt(c[4], 3) if len(c) > 4 and c[4] is not None else "n/a" for c in criteria
    ]
    has_c_col = any(len(c) > 4 and c[4] is not None for c in criteria)
    tbl_winners = []
    cell_fill_a = []
    cell_fill_b = []
    cell_fill_c = []
    for row_d in criteria:
        name, va, vb, lower_better = row_d[0], row_d[1], row_d[2], row_d[3]
        vc = row_d[4] if len(row_d) > 4 else None
        lc_label = lc if has_c_col else None
        w_c = _winner(va, vb, lower_better, la, lb, vc, lc_label)
        tbl_winners.append(w_c)
        cell_fill_a.append(
            "#4ADE80" if w_c == la else ("#FACC15" if "tie" in w_c else "#F5F5F5")
        )
        cell_fill_b.append(
            "#4ADE80" if w_c == lb else ("#FACC15" if "tie" in w_c else "#F5F5F5")
        )
        cell_fill_c.append(
            "#4ADE80"
            if (lc_label and w_c == lc_label)
            else ("#FACC15" if "tie" in w_c else "#F5F5F5")
        )

    hdr_vals = [f"<b>Criterion</b>", f"<b>{la[:14]}</b>", f"<b>{lb[:14]}</b>"]
    tbl_data = [tbl_names, tbl_va, tbl_vb]
    fill_cols = [[PLOTLY_PANE] * len(criteria), cell_fill_a, cell_fill_b]
    if has_c_col:
        lc_short = lc[:14] if lc else "Time"
        hdr_vals.append(f"<b>{lc_short}</b>")
        tbl_data.append(tbl_vc)
        fill_cols.append(cell_fill_c)
    hdr_vals.append(f"<b>A vs B</b>")
    tbl_data.append(tbl_winners)
    fill_cols.append([PLOTLY_PANE] * len(criteria))

    fig.add_trace(
        go.Table(
            header=dict(
                values=hdr_vals,
                fill_color=PLOTLY_GRID,
                font=dict(color=PLOTLY_TEXT, size=10),
                align="center",
                height=28,
            ),
            cells=dict(
                values=tbl_data,
                fill_color=fill_cols,
                font=dict(color=PLOTLY_TEXT, size=9),
                align="center",
                height=24,
            ),
        ),
        row=3,
        col=3,
    )

    # ── Global layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"Information Bar Comparison — {la} vs {lb}<br>"
            f"<sup>{verdict}</sup>",
            font=dict(size=16, color=PLOTLY_TEXT),
            x=0.5,
        ),
        barmode="overlay",
        height=1500,
        paper_bgcolor=PLOTLY_BG,
        plot_bgcolor=PLOTLY_PANE,
        font=dict(color=PLOTLY_TEXT, family="monospace"),
        legend=dict(
            bgcolor=PLOTLY_PANE,
            bordercolor=PLOTLY_GRID,
            borderwidth=1,
            x=0.01,
            y=0.99,
        ),
        hoverlabel=dict(bgcolor=PLOTLY_PANE, font_color=PLOTLY_TEXT),
    )
    fig.update_xaxes(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID)
    fig.update_yaxes(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID)

    # Axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Bar size", row=1, col=2)
    fig.update_yaxes(title_text="Bars", row=1, col=3)
    fig.update_xaxes(title_text="Log return", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)
    fig.update_yaxes(title_text="Autocorrelation", row=2, col=2)
    fig.update_yaxes(title_text="VR(q)", row=2, col=3)
    fig.update_yaxes(title_text="Ann. vol (%)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Lag", row=3, col=2)
    fig.update_yaxes(title_text="Autocorrelation (r²)", row=3, col=2)

    fig.write_html(str(out_path))
    print(f"  Interactive dashboard → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TIME BAR SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

# Only intervals that map to supported DB timeframes:
# 15m, 30m, 1h, 2h, 3h, 4h, 6h, 8h, 12h, 1d
_TIME_INTERVALS_MIN = [15, 30, 60, 120, 180, 240, 360, 480, 720, 1440]

_SUPPORTED_TIMEFRAMES = {
    "15m",
    "30m",
    "1h",
    "2h",
    "3h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
}


def _minutes_to_timeframe(minutes: int) -> str:
    """
    Convert integer minutes to a supported DB timeframe string.

    Examples:
        15   → '15m'
        30   → '30m'
        60   → '1h'
        120  → '2h'
        1440 → '1d'

    Raises ValueError if result is not in _SUPPORTED_TIMEFRAMES.
    """
    if minutes == 1440:
        tf = "1d"
    elif minutes < 60:
        tf = f"{minutes}m"
    else:
        hours = minutes // 60
        if minutes % 60 != 0:
            raise ValueError(
                f"{minutes} min does not convert cleanly to hours. "
                f"Supported intervals (minutes): {_TIME_INTERVALS_MIN}"
            )
        tf = f"{hours}h"
    if tf not in _SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"'{tf}' is not a supported timeframe. "
            f"Supported: {sorted(_SUPPORTED_TIMEFRAMES)}"
        )
    return tf


def infer_time_bar_interval(reference_df: pd.DataFrame) -> int:
    """
    Return the standard fixed interval (minutes) whose duration best matches the
    mean bar duration of reference_df.

    Works for any bar type — purely based on the observed mean duration.
    The interval is always chosen from _TIME_INTERVALS_MIN so the filename
    convention is predictable (see load_time_bars).
    """
    dur_s = reference_df["duration_seconds"].dropna()
    if not dur_s.empty:
        mean_min = float(dur_s.mean()) / 60.0
    else:
        bpd = reference_df.groupby("date").size().mean()
        mean_min = (24 * 60) / max(float(bpd), 0.1)

    chosen = min(_TIME_INTERVALS_MIN, key=lambda x: abs(x - mean_min))
    print(
        f"  [TIME BAR] mean bar duration ≈ {mean_min:.1f} min  "
        f"→  closest standard interval = {chosen} min"
    )
    return chosen


def load_time_bars(
    interval_minutes: int,
    label: str,
    exchange: str,
    symbol: str,
    data_dir: Path,
    start_date=None,
    end_date=None,
) -> "pd.DataFrame | None":
    """
    Load time bars for the given interval from the database via read_ohlcv.

    interval_minutes must map to a supported DB timeframe:
        15, 30, 60, 120, 180, 240, 360, 480, 720, 1440

    The raw OHLCV data is saved to data_dir as:
        {exchange}_{symbol_lower}_{timeframe}.csv   e.g. binance_btcusdt_1h.csv
    with columns: datetime, open, high, low, close, volume

    The returned DataFrame is passed through load_bars() normalisation so it
    has the same shape (log_return, hl_range, date, hour, attrs) as any other
    bar DataFrame used in this file.

    Returns None if the interval is unsupported or no data is found.
    """
    try:
        timeframe = _minutes_to_timeframe(interval_minutes)
    except ValueError as e:
        print(f"  [TIME BAR] unsupported interval {interval_minutes} min — {e}")
        return None

    print(f"  [TIME BAR] querying {exchange} {symbol} {timeframe} bars  ({label})")

    df = read_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        bar_type="time",
        start_date="2024-01-01",  # FIX: was hardcoded 2024 only
        end_date="2024-12-31",
        return_timestamp=True,
        columns=[
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],  # load_bars() needs a datetime/timestamp column
    )

    if df is None or df.empty:
        print(f"  [TIME BAR] no data found for {exchange} {symbol} {timeframe}")
        return None

    # Convert unix-ms timestamp → ISO datetime string so load_bars() can parse
    # it cleanly via pd.to_datetime(). Also rename to 'datetime' so load_bars()
    # finds it by its preferred column name.
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "datetime"})
    if "datetime" in df.columns:
        col = df["datetime"]
        # If stored as unix ms integers, convert to UTC datetime strings
        if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
            df["datetime"] = pd.to_datetime(col, unit="ms", utc=True)
        # Ensure it's a proper datetime (handles string timestamps too)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

    # Save raw OHLCV to data_dir: e.g. binance_btcusdt_1h.csv
    csv_cols = ["datetime", "open", "high", "low", "close", "volume"]
    save_cols = [c for c in csv_cols if c in df.columns]
    csv_name = f"{exchange.lower()}_{symbol.lower()}_{timeframe}.csv"
    csv_path = data_dir / csv_name
    df[save_cols].to_csv(csv_path, index=False)
    print(f"  [TIME BAR] saved → {csv_path}")

    # Pass through load_bars() so all derived fields and attrs are set
    # identically to how series A and B are loaded from CSVs.
    import io, tempfile, os

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    tmp.write(buf.getvalue())
    tmp.close()
    try:
        result = load_bars(tmp.name, label)
    finally:
        os.unlink(tmp.name)

    return result


def _add_timebar_to_figure(fig, c: "pd.DataFrame", tc: dict):
    """
    Overlay the time bar series (violet, dashed) on every panel of an existing
    matplotlib figure produced by plot_figure().

    Called after plot_figure() returns — opens the axes by index and adds
    traces without touching the existing A/B series.
    """
    import matplotlib
    from matplotlib.patches import Rectangle as Rect

    axes = fig.get_axes()
    if len(axes) < 9:
        return

    lc = tc["label"]
    rc = c["log_return"].dropna().values

    # ── [A] Close price ───────────────────────────────────────────────────────
    axes[0].plot(
        c["datetime"], c["close"], color=CC, lw=1.8, alpha=0.85, ls="--", label=lc
    )
    _leg_kw = dict(fontsize=10, facecolor="white", edgecolor="#555555",
                   framealpha=1.0, labelcolor="#1A1A1A")
    axes[0].legend(**_leg_kw)

    # ── [B] Bar size distribution ─────────────────────────────────────────────
    if c["bar_size"].notna().sum() > 10:
        bs_c = _iqr_filter(c["bar_size"].dropna().values.astype(float))
        if bs_c.std() > 0:
            z_c = (bs_c - bs_c.mean()) / bs_c.std()
            p1, p99 = np.percentile(z_c, [1, 99])
            axes[1].hist(
                z_c,
                bins=np.linspace(p1, p99, 45),
                color=CC,
                alpha=0.55,
                density=True,
                histtype="step",
                lw=1.5,
                label=f"{lc}  CV={_fmt(tc['bs_cv'], 3)}",
            )
            axes[1].legend(**_leg_kw)

    # ── [C] Bars per day ──────────────────────────────────────────────────────
    bpd_c = c.groupby("date").size()
    axes[2].plot(
        pd.to_datetime(list(bpd_c.index)),
        bpd_c.values,
        color=CC,
        lw=0.8,
        alpha=0.70,
        ls="--",
        label=f"{lc}  μ={tc['bars_per_day']:.1f}",
    )
    axes[2].legend(**_leg_kw)

    # ── [D] Return distribution ───────────────────────────────────────────────
    lo = min(np.percentile(rc, 0.5), axes[3].get_xlim()[0])
    hi = max(np.percentile(rc, 99.5), axes[3].get_xlim()[1])
    axes[3].hist(
        rc,
        bins=60,
        color=CC,
        alpha=0.40,
        density=True,
        histtype="step",
        lw=1.5,
        label=lc,
    )
    from scipy import stats as _stats

    xr = np.linspace(axes[3].get_xlim()[0], axes[3].get_xlim()[1], 300)
    axes[3].plot(
        xr,
        _stats.norm.pdf(xr, rc.mean(), rc.std()),
        color=CC,
        lw=1.3,
        ls=":",
        alpha=0.80,
    )
    axes[3].legend(**_leg_kw)

    # ── [E] ACF ───────────────────────────────────────────────────────────────
    max_lag = min(20, len(rc) // 4)
    lags = np.arange(1, max_lag + 1)
    ac_c = [float(pd.Series(rc).autocorr(int(l))) for l in lags]
    axes[4].bar(
        lags + 0.38 * 0.5 + 0.13,
        ac_c,
        0.26,
        color=CC,
        alpha=0.75,
        label=f"{lc}  LB₁₀={_fmt(tc['lb10_p'], 3)}",
    )
    axes[4].legend(**_leg_kw)

    # ── [F] Variance ratio ────────────────────────────────────────────────────
    qs = [2, 5, 8, 16]
    vr_c = [tc.get(f"vr{q}", np.nan) for q in qs]
    x = np.arange(len(qs))
    axes[5].bar(x + 0.27, vr_c, 0.26, color=CC, alpha=0.75, label=lc)
    axes[5].legend(**_leg_kw)

    # ── [G] Rolling vol ───────────────────────────────────────────────────────
    bpy_c = c.attrs.get("bars_per_year", 252)
    roll = max(5, min(20, len(c) // 10))
    rv_c = c["log_return"].rolling(roll).std() * np.sqrt(bpy_c) * 100
    axes[6].plot(c["datetime"], rv_c, color=CC, lw=1.8, alpha=0.85, ls="--", label=lc)
    axes[6].legend(**_leg_kw)

    # ── [H] Squared ACF ───────────────────────────────────────────────────────
    ac_c2 = [float(pd.Series(rc**2).autocorr(int(l))) for l in lags]
    axes[7].bar(
        lags + 0.38 * 0.5 + 0.13,
        ac_c2,
        0.26,
        color=CC,
        alpha=0.75,
        label=f"{lc}  ARCH={_fmt(tc['arch_p'], 3)}",
    )
    axes[7].legend(**_leg_kw)


def _add_timebar_to_interactive(fig, c: "pd.DataFrame", tc: dict, la: str, lb: str):
    """
    Add the time bar series as a violet dashed overlay to an existing Plotly figure
    produced by plot_interactive().
    """
    import plotly.graph_objects as go

    lc = tc["label"]
    rc = c["log_return"].dropna().values

    # [1,1] Close price
    fig.add_trace(
        go.Scatter(
            x=c["datetime"],
            y=c["close"],
            mode="lines",
            name=lc,
            line=dict(color=CC, width=0.8, dash="dash"),
            opacity=0.75,
            legendgroup="c",
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Close: %{y:,.2f}<extra>"
            + lc
            + "</extra>",
        ),
        row=1,
        col=1,
    )

    # [1,2] Bar size
    if c["bar_size"].notna().sum() > 10:
        bs_c = _iqr_filter(c["bar_size"].dropna().values.astype(float))
        fig.add_trace(
            go.Histogram(
                x=bs_c,
                histnorm="probability density",
                opacity=0.50,
                marker_color=CC,
                name=f"{lc} CV={_fmt(tc['bs_cv'],3)}",
                legendgroup="c",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # [1,3] Bars per day
    bpd_c = c.groupby("date").size().reset_index(name="n")
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(bpd_c["date"]),
            y=bpd_c["n"],
            mode="lines",
            line=dict(color=CC, width=0.9, dash="dash"),
            name=lc,
            legendgroup="c",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Bars: %{y}<extra>" + lc + "</extra>",
        ),
        row=1,
        col=3,
    )

    # [2,1] Return distribution
    lo = float(np.percentile(rc, 0.5))
    hi = float(np.percentile(rc, 99.5))
    bs = (hi - lo) / 60
    fig.add_trace(
        go.Histogram(
            x=rc,
            xbins=dict(start=lo, end=hi, size=bs),
            histnorm="probability density",
            opacity=0.45,
            marker_color=CC,
            legendgroup="c",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    from scipy import stats as _stats

    xr = np.linspace(lo, hi, 200)
    fig.add_trace(
        go.Scatter(
            x=xr,
            y=_stats.norm.pdf(xr, rc.mean(), rc.std()),
            mode="lines",
            line=dict(color=CC, dash="dot", width=1.2),
            legendgroup="c",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # [2,2] ACF
    max_lag = min(20, len(rc) // 4)
    lags = list(range(1, max_lag + 1))
    ac_c = [float(pd.Series(rc).autocorr(l)) for l in lags]
    fig.add_trace(
        go.Bar(
            x=[l + 0.32 for l in lags],
            y=ac_c,
            width=0.26,
            marker_color=CC,
            opacity=0.75,
            name=f"{lc} LB10={_fmt(tc['lb10_p'],3)}",
            legendgroup="c",
            showlegend=False,
            hovertemplate="Lag %{x:.0f}<br>ACF: %{y:.4f}<extra>" + lc + "</extra>",
        ),
        row=2,
        col=2,
    )

    # [2,3] Variance ratio
    qs = [2, 5, 8, 16]
    vr_c = [tc.get(f"vr{q}", np.nan) for q in qs]
    fig.add_trace(
        go.Bar(
            x=[i + 0.27 for i in range(len(qs))],
            y=vr_c,
            width=0.26,
            marker_color=CC,
            opacity=0.75,
            name=lc,
            legendgroup="c",
            showlegend=False,
            customdata=[f"q={q}" for q in qs],
            hovertemplate="%{customdata}<br>VR: %{y:.4f}<extra>" + lc + "</extra>",
        ),
        row=2,
        col=3,
    )

    # [3,1] Rolling vol
    bpy_c = c.attrs.get("bars_per_year", 252)
    roll = max(5, min(20, len(c) // 10))
    rv_c = (c["log_return"].rolling(roll).std() * np.sqrt(bpy_c) * 100).values
    fig.add_trace(
        go.Scatter(
            x=c["datetime"],
            y=rv_c,
            mode="lines",
            name=lc,
            line=dict(color=CC, width=0.8, dash="dash"),
            legendgroup="c",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Ann. vol: %{y:.1f}%<extra>"
            + lc
            + "</extra>",
        ),
        row=3,
        col=1,
    )

    # [3,2] Squared ACF
    ac_c2 = [float(pd.Series(rc**2).autocorr(l)) for l in lags]
    fig.add_trace(
        go.Bar(
            x=[l + 0.32 for l in lags],
            y=ac_c2,
            width=0.26,
            marker_color=CC,
            opacity=0.75,
            name=f"{lc} ARCH={_fmt(tc['arch_p'],3)}",
            legendgroup="c",
            showlegend=False,
            hovertemplate="Lag %{x:.0f}<br>ACF(r²): %{y:.4f}<extra>" + lc + "</extra>",
        ),
        row=3,
        col=2,
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


# ── Bar type registry ─────────────────────────────────────────────────────────
BAR_TYPES = {
    "dollar": (
        "binance_btc_dollar_minute_bars.csv",
        "binance_btc_dollar_tick_bars.csv",
        "Minute Dollar",
        "Tick Dollar",
    ),
    "volume": (
        "binance_btc_volume_minute_bars.csv",
        "binance_btc_volume_tick_bars.csv",
        "Minute Volume",
        "Tick Volume",
    ),
    "volatility": (
        "binance_btc_volatility_minute_bars.csv",
        "binance_btc_volatility_tick_bars.csv",
        "Minute Volatility",
        "Tick Volatility",
    ),
    "range": (
        "binance_btc_range_minute_bars.csv",
        "binance_btc_range_tick_bars.csv",
        "Minute Range",
        "Tick Range",
    ),
    "renko": (
        "binance_btc_renko_minute_bars.csv",
        "binance_btc_renko_tick_bars.csv",
        "Minute Renko",
        "Tick Renko",
    ),
    "hybrid": (
        "binance_btc_hybrid_minute_bars.csv",
        "binance_btc_hybrid_tick_bars.csv",
        "Minute Hybrid",
        "Tick Hybrid",
    ),
}

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "processed_bars"
)


def _run_one(
    bar_type: str,
    data_dir: Path,
    out_dir: Path,
    with_time_bars: bool = False,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    panels: str | None = None,
    figures: str | None = None,
) -> None:
    min_file, tick_file, label_a, label_b = BAR_TYPES[bar_type]
    csv_a = data_dir / min_file
    csv_b = data_dir / tick_file

    if not csv_a.exists():
        print(f"  [SKIP] {bar_type}: {csv_a.name} not found")
        return
    if not csv_b.exists():
        print(f"  [SKIP] {bar_type}: {csv_b.name} not found")
        return

    type_out = out_dir / bar_type
    type_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}\n  BAR TYPE: {bar_type.upper()}\n{'═'*60}")

    print("\nLoading bar data ...")
    a = load_bars(str(csv_a), label_a)
    b = load_bars(str(csv_b), label_b)

    # ── Optional: load closest-matching time bar ──────────────────────────────
    c = tc = comp_ac = comp_bc = None
    if with_time_bars:
        interval = infer_time_bar_interval(a)  # match minute bar mean duration
        lc = f"Time {interval}min"
        c = load_time_bars(
            interval_minutes=interval,
            label=lc,
            exchange=exchange,
            symbol=symbol,
            data_dir=data_dir,
        )
        if c is not None:
            print("\nRunning statistical tests on time bars ...")
            tc = run_tests(c, lc)
            comp_ac = compare_distributions(a, c)
            comp_bc = compare_distributions(b, c)

    # ── Statistical tests ─────────────────────────────────────────────────────
    print("\nRunning statistical tests ...")
    ta = run_tests(a, label_a)
    tb = run_tests(b, label_b)
    comp = compare_distributions(a, b)

    # ── Unified report — 2 or 3 columns depending on whether time bars loaded ─
    score_a, score_b, criteria = write_report(
        ta,
        tb,
        comp,
        type_out / "comparison_stats.txt",
        tc=tc,
        comp_ac=comp_ac,
        comp_bc=comp_bc,
    )

    # ── Static figure ─────────────────────────────────────────────────────────
    print("\nGenerating static figure ...")
    plot_figure(
        a,
        b,
        ta,
        tb,
        comp,
        score_a,
        score_b,
        criteria,
        type_out / "comparison_figure.pdf",
        tc=tc,
    )

    if c is not None and tc is not None:
        pdf_3 = type_out / "comparison_figure_with_timebar.pdf"
        fig_obj = _make_figure_with_timebar(
            a, b, ta, tb, comp, score_a, score_b, criteria, c, tc
        )
        fig_obj.savefig(
            str(pdf_3),
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            backend="pdf",
        )
        import matplotlib.pyplot as _plt_close; _plt_close.close(fig_obj)
        print(f"  Figure+time bar (PDF) → {pdf_3}")

    # ── Interactive dashboard ─────────────────────────────────────────────────
    print("\nGenerating interactive dashboard ...")
    plot_interactive(
        a,
        b,
        ta,
        tb,
        comp,
        score_a,
        score_b,
        criteria,
        type_out / "comparison_interactive.html",
    )

    if c is not None and tc is not None:
        html_3 = type_out / "comparison_interactive_with_timebar.html"
        _gen_interactive_3series(
            a, b, ta, tb, comp, score_a, score_b, criteria, c, tc, html_3
        )

    # ── Extract individual panels if requested ───────────────────────────────
    if panels:
        import fitz  # PyMuPDF

        _PANEL_POSITIONS = {
            "A": (0, 0),
            "B": (0, 1),
            "C": (0, 2),
            "D": (1, 0),
            "E": (1, 1),
            "F": (1, 2),
            "G": (2, 0),
            "H": (2, 1),
            "I": (2, 2),
        }
        _PANEL_NAMES = {
            "A": "close_price",
            "B": "bar_size_dist",
            "C": "bars_per_day",
            "D": "return_dist",
            "E": "return_acf",
            "F": "variance_ratio",
            "G": "rolling_vol",
            "H": "squared_acf",
            "I": "scorecard",
        }
        # Choose the with-timebar figure if available, else fall back
        panel_src = type_out / "comparison_figure_with_timebar.pdf"
        if not panel_src.exists():
            panel_src = type_out / "comparison_figure.pdf"
        if panel_src.exists():
            panel_dir = type_out / "panels"
            panel_dir.mkdir(exist_ok=True)
            # Build dummy figure to get axes positions at the correct GridSpec
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            dummy = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(
                3,
                3,
                figure=dummy,
                hspace=0.52,
                wspace=0.38,
                top=0.93,
                bottom=0.05,
                left=0.08,
                right=0.97,
            )
            ax_pos = {}
            for letter, (row, col) in _PANEL_POSITIONS.items():
                ax = dummy.add_subplot(gs[row, col])
                dummy.canvas.draw()
                pos = ax.get_position()
                pad_x, pad_y_b, pad_y_t = 0.015, 0.030, 0.035
                ax_pos[letter] = (
                    max(0.0, pos.x0 - pad_x),
                    max(0.0, pos.y0 - pad_y_b),
                    min(1.0, pos.x1 + pad_x),
                    min(1.0, pos.y1 + pad_y_t),
                )
            plt.close(dummy)
            doc = fitz.open(str(panel_src))
            page = doc[0]
            pw, ph = page.rect.width, page.rect.height
            for letter in [p.strip().upper() for p in panels.split(",") if p.strip()]:
                if letter not in _PANEL_POSITIONS:
                    continue
                x0_f, y0_f, x1_f, y1_f = ax_pos[letter]
                crop = fitz.Rect(x0_f * pw, (1 - y1_f) * ph, x1_f * pw, (1 - y0_f) * ph)
                # Vector-preserving crop: show_pdf_page retains all paths and fonts
                out_doc = fitz.open()
                new_pg = out_doc.new_page(width=crop.width, height=crop.height)
                new_pg.show_pdf_page(new_pg.rect, doc, 0, clip=crop)
                out_path = panel_dir / f"panel_{letter}_{_PANEL_NAMES[letter]}.pdf"
                out_doc.save(str(out_path), garbage=4, deflate=True)
                out_doc.close()
                print(f"  Panel {letter} → {out_path.name}")
            doc.close()
        else:
            print(f"  [WARN] No figure PDF found for panel extraction")

    # ── Standalone publication figures if requested ──────────────────────────
    if figures:
        print("\nGenerating standalone figures ...")
        _generate_standalone_figures(
            figures=figures,
            a=a,
            b=b,
            ta=ta,
            tb=tb,
            c=c,
            tc=tc,
            criteria=criteria,
            score_a=score_a,
            score_b=score_b,
            out_dir=type_out,
        )

    print(f"✓  {bar_type} — outputs in {type_out.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE PUBLICATION FIGURES  (--figures flag)
# ═══════════════════════════════════════════════════════════════════════════════
# Generates self-contained publication-ready vector PDFs at IEEE full width.
# All fonts bold, legends below axes — legible on physical B&W print.
#
# Usage examples:
#   --figures D          return distribution only, all bar types
#   --figures E          return ACF only
#   --figures D,E        both panels as separate files
#   --figures DE         D and E side-by-side in one combined figure
#   --figures all        D, E, and DE
#
# Output: comparisons/figures/fig_D_<bartype>.pdf
#                              fig_E_<bartype>.pdf
#                              fig_DE_<bartype>.pdf   (combined)
# ═══════════════════════════════════════════════════════════════════════════════

_STANDALONE_PANELS = {"D": "return_dist", "E": "return_acf", "DE": "return_dist_acf"}

# ── Journal typography constants ─────────────────────────────────────────────
# Sized for IEEE double-column (7.16 in) at 300 dpi.  All text is bold so a
# physical B&W print remains fully legible; colours still distinguish series
# on screen / colour print.
_FIG_W   = 7.16   # single-panel width  (IEEE/Elsevier double-column = 7.16 in)
_FIG_H   = 5.4    # single-panel height — extra vertical room for below-legend
_FIG_W2  = 14.32  # combined DE width   (2 × 7.16 in, two panels side by side)
_FIG_H2  = 5.6    # combined DE height
_FS_TTL  = 16     # panel title — bold, readable at 300 dpi on physical print
_FS_LBL  = 15     # axis labels
_FS_TCK  = 15     # tick labels — bold, clearly readable on A4 printout
_FS_LEG  = 14     # legend text
_LW_FIT  = 2.4    # fitted curve line width
_LW_CNF  = 1.8    # confidence-band dashed line width
_LW_SP   = 1.2    # spine / tick line width
_ALPHA_H = 0.28   # histogram fill alpha
_ALPHA_B = 0.85   # bar alpha
# Bottom margins: must clear the tallest legend block without clipping
_BOT_1   = 0.30   # single panel — 3 legend rows fit comfortably
_BOT_2   = 0.34   # combined DE — 4 legend rows (series + CI band)

_INK = "#1A1A1A"  # near-black for all text/spines — prints as solid black


def _sa_style(ax):
    """Apply clean, journal-ready axis style (white background, bold black ink)."""
    ax.set_facecolor("white")
    ax.grid(True, color="#CCCCCC", lw=0.5, alpha=0.7, zorder=0, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_edgecolor(_INK)
        ax.spines[sp].set_linewidth(_LW_SP)
    ax.tick_params(
        colors=_INK, labelsize=_FS_TCK, length=5, width=_LW_SP,
        direction="out", top=False, right=False,
        labelcolor=_INK,
    )
    # Force bold on all existing tick labels immediately
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_color(_INK)
        lbl.set_fontsize(_FS_TCK)


def _sa_bold_ticks(ax):
    """Make every tick label bold, near-black, and correctly sized — survives B&W printing."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_color(_INK)
        lbl.set_fontsize(_FS_TCK)


def _sa_thicken_legend_lines(leg):
    """
    Make line handles in a legend visibly thick and dashes clearly visible —
    critical for B&W physical print where thin dashes disappear entirely.
    """
    if leg is None:
        return
    for handle in leg.legend_handles:
        # Line2D handles (plot, axhline)
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(3.0)
        # Rectangle handles (hist, bar) — just ensure full alpha
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(0.85)


def _sa_legend_below(ax, fig, ncol, bottom_margin):
    """
    Place the legend below the axes, fully outside the plot area, never overlapping data.
    The anchor is set in axes-fraction coordinates so it is immune to data scaling.
    Returns the legend object.
    """
    # -0.22 clears the x-axis tick labels; increase for taller tick fonts
    leg = ax.legend(
        fontsize=_FS_LEG,
        facecolor="white",
        edgecolor="#555555",
        framealpha=1.0,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=max(1, ncol),
        borderpad=0.8,
        labelspacing=0.5,
        handlelength=2.8,
        handletextpad=0.6,
        columnspacing=1.4,
    )
    if leg:
        for txt in leg.get_texts():
            txt.set_color(_INK)
            txt.set_fontweight("bold")
        _sa_thicken_legend_lines(leg)
    return leg


def _sa_shared_legend_below(fig, axes, ncol):
    """
    Single shared legend for a multi-axes figure, centred below the whole row.
    Collects handles/labels from all axes (deduplicates by label).
    """
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    leg = fig.legend(
        handles,
        labels,
        fontsize=_FS_LEG,
        facecolor="white",
        edgecolor="#555555",
        framealpha=1.0,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=ncol,
        borderpad=0.7,
        labelspacing=0.45,
        handlelength=2.4,
        handletextpad=0.55,
        columnspacing=1.2,
    )
    if leg:
        for txt in leg.get_texts():
            txt.set_color(_INK)
            txt.set_fontweight("bold")
    return leg


def _sa_save(fig, path: Path):
    """Save figure as publication-quality PDF — vector, 300 dpi, tight crop."""
    # Constrain layout ensures no label/tick clipping before final save
    try:
        fig.set_layout_engine("tight", pad=0.4)
    except Exception:
        fig.tight_layout(pad=0.4)
    fig.savefig(
        str(path),
        bbox_inches="tight",
        pad_inches=0.08,        # minimal white border — journal ready
        facecolor="white",
        edgecolor="none",
        backend="pdf",
        dpi=300,
        metadata={"Creator": "compare_bars.py — journal figure export"},
    )
    plt.close(fig)
    print(f"  Fig → {path.name}")


# ── Shared drawing helpers (draw onto a supplied axes, no figure creation) ───

def _draw_panel_D(ax, ra, rb, rc, ta, tb, tc):
    """
    Draw return-distribution panel onto *ax*.

    Visual design (journal standard):
      solid line  = empirical KDE    (data)
      dashed line = fitted Gaussian  (null hypothesis)
    KDE curves instead of filled histograms eliminate alpha-blending artefacts
    when three overlapping distributions share the same centre region.
    """
    from scipy import stats as _stats
    from scipy.stats import gaussian_kde
    import matplotlib.ticker as _mt

    # Compute x-axis range from per-series percentiles (union of widest range).
    # Using the combined-array percentile fails when series have very different
    # return scales (e.g. Renko: tick std~0.002 vs minute std~0.016), because
    # 98%+ of combined returns come from the narrower series, setting the x-axis
    # so tight that 32% of the wider series returns fall outside — making its
    # KDE and Gaussian appear as an invisible flat line near density=0.
    _series_r = [ra, rb] + ([rc] if rc is not None else [])
    lo = min(np.percentile(r, 0.5)  for r in _series_r)
    hi = max(np.percentile(r, 99.5) for r in _series_r)
    xfit = np.linspace(lo, hi, 500)

    series = [
        (ra, CA, ta["label"], ta.get("kurt", float("nan"))),
        (rb, CB, tb["label"], tb.get("kurt", float("nan"))),
    ]
    if rc is not None and tc is not None:
        series.append((rc, CC, tc["label"], tc.get("kurt", float("nan"))))

    for r, col, lab, kurt in series:
        kde = gaussian_kde(r, bw_method="scott")
        ax.plot(
            xfit, kde(xfit),
            color=col, lw=2.4, ls="-", zorder=3,
            label=f"{lab}  (Kurt = {kurt:+.3f})",
        )
        # Fitted Gaussian — dashed, same colour, slightly thinner
        ax.plot(
            xfit,
            _stats.norm.pdf(xfit, float(r.mean()), float(r.std())),
            color=col, lw=1.6, ls="--", zorder=2, alpha=0.75,
        )

    ax.set_xlabel("Log Return", fontsize=_FS_LBL, fontweight="bold", color=_INK,
                  labelpad=6)
    ax.set_ylabel("Density",    fontsize=_FS_LBL, fontweight="bold", color=_INK,
                  labelpad=6)
    ax.set_xlim(lo, hi)
    ax.margins(y=0.05)

    # Y-axis formatter — auto-scale decimal places to avoid scientific notation
    ymax = ax.get_ylim()[1]
    fmt = (f"{{x:.1f}}" if ymax >= 10 else f"{{x:.2f}}" if ymax >= 1 else f"{{x:.3f}}")
    ax.yaxis.set_major_formatter(_mt.FuncFormatter(lambda x, _: fmt.format(x=x)))

    ax.set_title(
        "(a)  Return Distribution  [dashed = fitted Gaussian]",
        fontsize=_FS_TTL, fontweight="bold", color=_INK, loc="left", pad=8,
    )


def _draw_panel_E(ax, ra, rb, rc, ta, tb, tc):
    """Draw return-ACF bar chart onto *ax* (journal standard)."""
    import matplotlib.ticker as _mt

    max_lag = 15
    lags = np.arange(1, max_lag + 1)
    conf = 1.96 / np.sqrt(min(len(ra), len(rb)))

    series = [
        (ra, CA, ta["label"], ta.get("lb10_p", float("nan")), -0.27),
        (rb, CB, tb["label"], tb.get("lb10_p", float("nan")),  0.00),
    ]
    bw = 0.25
    if rc is not None and tc is not None:
        series.append((rc, CC, tc["label"], tc.get("lb10_p", float("nan")), 0.27))
        bw = 0.23  # narrower when 3 series so bars never overlap

    for r, col, lab, lb_p, offset in series:
        ac  = [float(pd.Series(r).autocorr(int(l))) for l in lags]
        lbl = f"{lab}  (LB p = {lb_p:.3f})" if not np.isnan(lb_p) else lab
        ax.bar(lags + offset, ac, width=bw * 0.90,
               color=col, alpha=_ALPHA_B, zorder=2, label=lbl)

    ax.axhline( conf, color="#444444", lw=_LW_CNF, ls="--", zorder=3,
                label="95 % confidence band")
    ax.axhline(-conf, color="#444444", lw=_LW_CNF, ls="--", zorder=3)
    ax.axhline(0,     color=_INK,      lw=0.8,     zorder=1)

    ax.set_xlabel("Lag",             fontsize=_FS_LBL, fontweight="bold", color=_INK,
                  labelpad=6)
    ax.set_ylabel("Autocorrelation", fontsize=_FS_LBL, fontweight="bold", color=_INK,
                  labelpad=6)
    ax.set_xlim(0.3, max_lag + 0.7)
    ax.xaxis.set_major_locator(_mt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(_mt.MultipleLocator(1))
    ax.set_title(
        "(b)  Return Autocorrelation Function  [dashed = 95 % CI]",
        fontsize=_FS_TTL, fontweight="bold", color=_INK, loc="left", pad=8,
    )


# ── Public panel functions ────────────────────────────────────────────────────

def _sa_panel_D(ra, rb, rc, ta, tb, tc, out_path: Path):
    """Standalone figure D: return distribution — journal ready."""
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), facecolor="white")
    # Reserve bottom space for the legend block before drawing
    fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=_BOT_1)
    _sa_style(ax)
    _draw_panel_D(ax, ra, rb, rc, ta, tb, tc)
    # Bold ticks AFTER drawing (draw may create new tick labels)
    _sa_bold_ticks(ax)
    handles_D, labels_D = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles_D, labels_D,
        fontsize=_FS_LEG, facecolor="white", edgecolor="#555555", framealpha=1.0,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),   # clear x-axis label + tick labels
        ncol=1,
        borderpad=0.8, labelspacing=0.5,
        handlelength=2.8, handletextpad=0.6, columnspacing=1.4,
    )
    if leg:
        for txt in leg.get_texts():
            txt.set_color(_INK); txt.set_fontweight("bold")
    _sa_thicken_legend_lines(leg)
    _sa_save(fig, out_path)


def _sa_panel_E(ra, rb, rc, ta, tb, tc, out_path: Path):
    """Standalone figure E: return ACF — journal ready."""
    n_series = 2 + (1 if rc is not None else 0) + 1   # +1 for CI band entry
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=_BOT_1)
    _sa_style(ax)
    _draw_panel_E(ax, ra, rb, rc, ta, tb, tc)
    _sa_bold_ticks(ax)
    leg = _sa_legend_below(ax, fig, ncol=min(n_series, 4), bottom_margin=_BOT_1)
    _sa_thicken_legend_lines(leg)
    _sa_save(fig, out_path)


def _sa_panel_DE(ra, rb, rc, ta, tb, tc, out_path: Path):
    """
    Combined figure DE: panel D (left) and panel E (right) side by side.
    Landscape format at 14.32 in wide — journal standard for two-panel figures.
    Each panel has its own legend below its x-axis with thick visible handles.
    Output: fig_DE_<bartype>.pdf
    """
    fig, (axD, axE) = plt.subplots(
        1, 2,
        figsize=(_FIG_W2, _FIG_H2),
        facecolor="white",
    )
    # Generous bottom margin so below-axis legends are never clipped at any DPI
    fig.subplots_adjust(
        left=0.07, right=0.98,
        top=0.89,  bottom=_BOT_2,
        wspace=0.30,
    )

    _sa_style(axD)
    _sa_style(axE)

    _draw_panel_D(axD, ra, rb, rc, ta, tb, tc)
    _draw_panel_E(axE, ra, rb, rc, ta, tb, tc)

    # Bold ticks AFTER drawing — tick labels may be regenerated by the draw calls
    _sa_bold_ticks(axD)
    _sa_bold_ticks(axE)

    # ── Panel D legend — one row per series, below x-axis ────────────────────
    custom_handles_D, custom_labels_D = axD.get_legend_handles_labels()
    legD = axD.legend(
        custom_handles_D, custom_labels_D,
        fontsize=_FS_LEG,
        facecolor="white", edgecolor="#555555", framealpha=1.0,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),   # clears x-label + tick labels
        ncol=1,
        borderpad=0.7, labelspacing=0.45,
        handlelength=2.6, handletextpad=0.5,
    )
    if legD:
        for txt in legD.get_texts():
            txt.set_color(_INK); txt.set_fontweight("bold")
        _sa_thicken_legend_lines(legD)

    # ── Panel E legend — series + CI band, sorted so CI is last ─────────────
    handles_E, labels_E = axE.get_legend_handles_labels()
    ci_idx = next((i for i, l in enumerate(labels_E) if "confidence" in l.lower()), None)
    if ci_idx is not None and ci_idx != len(handles_E) - 1:
        handles_E.append(handles_E.pop(ci_idx))
        labels_E.append(labels_E.pop(ci_idx))
    legE = axE.legend(
        handles_E, labels_E,
        fontsize=_FS_LEG,
        facecolor="white", edgecolor="#555555", framealpha=1.0,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
        borderpad=0.7, labelspacing=0.45,
        handlelength=2.6, handletextpad=0.5,
    )
    if legE:
        for txt in legE.get_texts():
            txt.set_color(_INK); txt.set_fontweight("bold")
        _sa_thicken_legend_lines(legE)

    _sa_save(fig, out_path)


def _generate_standalone_figures(
    figures: str,
    a: pd.DataFrame,
    b: pd.DataFrame,
    ta: dict,
    tb: dict,
    c,
    tc,
    criteria: list,
    score_a: int,
    score_b: int,
    out_dir: Path,
):
    """
    Generate standalone publication figures.

    figures : comma-separated panel letters or 'all'
              Valid: D (return dist), E (return ACF), DE (combined side-by-side)
    out_dir : comparisons/<bar_type>/ — figures saved to out_dir.parent/figures/
    """
    letters = (
        list(_STANDALONE_PANELS.keys())
        if figures.strip().lower() == "all"
        else [p.strip().upper() for p in figures.split(",") if p.strip()]
    )

    bar_name = out_dir.name
    fig_dir = out_dir.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    ra = a["log_return"].dropna().values
    rb = b["log_return"].dropna().values
    rc = c["log_return"].dropna().values if c is not None else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _plt.rcParams.update(
        {
            # ── Font (Times New Roman — IEEE/Elsevier preferred) ─────────
            "font.family":        "serif",
            "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
            "font.weight":        "bold",
            "figure.dpi":         300,
            "savefig.dpi":        300,
            "savefig.bbox":       "tight",
            "text.color":         _INK,
            # ── Axes ─────────────────────────────────────────────────────
            "axes.facecolor":     "white",
            "axes.edgecolor":     "#333333",
            "axes.linewidth":     _LW_SP,
            "axes.titleweight":   "bold",
            "axes.labelweight":   "bold",
            "axes.titlesize":     _FS_TTL,
            "axes.labelsize":     _FS_LBL,
            "axes.spines.top":    False,
            "axes.spines.right":  False,
            # ── Ticks — bold near-black, large enough for A4 print ───────
            "xtick.labelsize":    _FS_TCK,
            "ytick.labelsize":    _FS_TCK,
            "xtick.color":        _INK,
            "ytick.color":        _INK,
            "xtick.labelcolor":   _INK,
            "ytick.labelcolor":   _INK,
            "xtick.major.width":  _LW_SP,
            "ytick.major.width":  _LW_SP,
            "xtick.major.size":   5,
            "ytick.major.size":   5,
            "xtick.direction":    "out",
            "ytick.direction":    "out",
            # ── Legend ───────────────────────────────────────────────────
            "legend.fontsize":    _FS_LEG,
            "legend.framealpha":  1.0,
        }
    )

    for letter in letters:
        if letter not in _STANDALONE_PANELS:
            print(
                f"  [WARN] Unknown panel '{letter}'. "
                f"Valid: {', '.join(_STANDALONE_PANELS)} or all"
            )
            continue
        path = fig_dir / f"fig_{letter}_{bar_name}.pdf"
        if letter == "D":
            _sa_panel_D(ra, rb, rc, ta, tb, tc, path)
        elif letter == "E":
            _sa_panel_E(ra, rb, rc, ta, tb, tc, path)
        elif letter == "DE":
            _sa_panel_DE(ra, rb, rc, ta, tb, tc, path)


def _make_figure_with_timebar(a, b, ta, tb, comp, score_a, score_b, criteria, c, tc):
    """
    Re-create the 9-panel matplotlib figure and overlay the violet time bar
    series on each panel.  Returns the figure object (not saved yet).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # FIX: produce the full 9-panel A-vs-B figure, then overlay series C.
    # Monkeypatch savefig to capture the fig object before plt.close().
    import tempfile, os

    tmp_pdf = Path(tempfile.mktemp(suffix=".pdf"))
    captured_fig = [None]
    _orig_sf = plt.Figure.savefig

    def _cap(self, *a, **k):
        captured_fig[0] = self
        _orig_sf(self, *a, **k)

    plt.Figure.savefig = _cap
    try:
        plot_figure(a, b, ta, tb, comp, score_a, score_b, criteria, tmp_pdf, tc=None)
    finally:
        plt.Figure.savefig = _orig_sf
        try:
            os.unlink(str(tmp_pdf))
        except Exception:
            pass
    fig = captured_fig[0]
    if fig is None:
        raise RuntimeError("Could not capture figure from plot_figure()")
    # Overlay time bar series C on all 9 panels.
    _add_timebar_to_figure(fig, c, tc)
    la_l, lb_l, lc_l = ta["label"], tb["label"], tc["label"]
    fig.suptitle(
        f"Information Bar Comparison:  {la_l}  vs  {lb_l}  vs  {lc_l}  (time bar baseline)",
        fontsize=18,
        color="#1A1A1A",
        y=0.99,
        fontweight="bold",
    )
    return fig


def _gen_interactive_3series(
    a, b, ta, tb, comp, score_a, score_b, criteria, c, tc, out_path: Path
):
    """
    Generate the Plotly interactive dashboard with the time bar series (C)
    overlaid as violet dashed traces.  Uses plot_interactive for the base
    then calls _add_timebar_to_interactive to overlay C.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    la, lb = ta["label"], tb["label"]
    lc = tc["label"]

    # Build the base figure in-memory (without writing to disk)
    import io

    tmp = io.StringIO()
    # We cannot easily capture the Plotly figure from plot_interactive because
    # it writes directly.  Instead, rebuild minimally and call the overlay.
    # For the 3-series interactive, use the overlay helper on a fresh figure.

    # Create a fresh base figure via plot_interactive but save to temp
    import tempfile, os

    tmp_path = Path(tempfile.mktemp(suffix=".html"))
    plot_interactive(a, b, ta, tb, comp, score_a, score_b, criteria, tmp_path)

    # Read back (Plotly HTML contains the full figure JSON)
    import plotly.io as pio

    # Since we can't easily deserialise from HTML, we generate the 3-series
    # figure from scratch using a compact approach.
    ra = a["log_return"].dropna().values
    rb = b["log_return"].dropna().values
    rc = c["log_return"].dropna().values

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[
            "Close Price",
            "Bar Size Dist.",
            "Bars per Day",
            "Return Distribution",
            "Return ACF",
            "Variance Ratio",
            "Rolling Volatility",
            "Squared-Return ACF",
            "Scorecard",
        ],
        specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {"type": "table"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    BG, PANE, GRID, TEXT = "white", "#F5F5F5", "#CCCCCC", "#1A1A1A"

    kws = [(a, CA, "a", la, ta), (b, CB, "b", lb, tb), (c, CC, "c", lc, tc)]

    # [1,1] Close price
    for df, col, grp, lbl, _ in kws:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["close"],
                mode="lines",
                name=lbl,
                line=dict(color=col, width=0.9, dash="dash" if grp == "c" else "solid"),
                legendgroup=grp,
                opacity=0.85 if grp != "a" else 1.0,
                hovertemplate=f"<b>%{{x|%Y-%m-%d %H:%M}}</b><br>Close: %{{y:,.2f}}<extra>{lbl}</extra>",
            ),
            row=1,
            col=1,
        )

    # [1,2] Bar size
    for df, col, grp, lbl, t in kws:
        if df["bar_size"].notna().sum() > 10:
            bs = _iqr_filter(df["bar_size"].dropna().values.astype(float))
            fig.add_trace(
                go.Histogram(
                    x=bs,
                    histnorm="probability density",
                    opacity=0.60,
                    marker_color=col,
                    name=f"{lbl} CV={_fmt(t['bs_cv'],3)}",
                    legendgroup=grp,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    # [1,3] Bars per day
    for df, col, grp, lbl, t in kws:
        bpd = df.groupby("date").size().reset_index(name="n")
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(bpd["date"]),
                y=bpd["n"],
                mode="lines",
                line=dict(color=col, width=0.9, dash="dash" if grp == "c" else "solid"),
                name=f"{lbl} μ={t['bars_per_day']:.1f}",
                legendgroup=grp,
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    # [2,1] Return distribution
    lo = min(np.percentile(ra, 0.5), np.percentile(rb, 0.5), np.percentile(rc, 0.5))
    hi = max(np.percentile(ra, 99.5), np.percentile(rb, 99.5), np.percentile(rc, 99.5))
    bs_ret = (hi - lo) / 60
    for r, col, grp, lbl, _ in [
        (ra, CA, "a", la, ta),
        (rb, CB, "b", lb, tb),
        (rc, CC, "c", lc, tc),
    ]:
        fig.add_trace(
            go.Histogram(
                x=r,
                xbins=dict(start=lo, end=hi, size=bs_ret),
                histnorm="probability density",
                opacity=0.55,
                marker_color=col,
                legendgroup=grp,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # [2,2] Return ACF
    max_lag = max(5, min(20, min(len(ra), len(rb), len(rc)) // 4))
    lags_l = list(range(1, max_lag + 1))
    offsets = [-0.26, 0, 0.26]
    for (r, col, grp, lbl, t), off in zip(
        [(ra, CA, "a", la, ta), (rb, CB, "b", lb, tb), (rc, CC, "c", lc, tc)], offsets
    ):
        acf = [float(pd.Series(r).autocorr(l)) for l in lags_l]
        fig.add_trace(
            go.Bar(
                x=[l + off for l in lags_l],
                y=acf,
                width=0.24,
                marker_color=col,
                opacity=0.85,
                name=f"{lbl} LB10={_fmt(t['lb10_p'],3)}",
                legendgroup=grp,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # [2,3] VR
    qs = [2, 5, 8, 16]
    x_vr = list(range(len(qs)))
    for (df, col, grp, lbl, t), off in zip(kws, [-0.27, 0, 0.27]):
        vr_vals = [t.get(f"vr{q}", np.nan) for q in qs]
        fig.add_trace(
            go.Bar(
                x=[i + off for i in x_vr],
                y=vr_vals,
                width=0.25,
                marker_color=col,
                opacity=0.85,
                name=lbl,
                legendgroup=grp,
                showlegend=False,
                customdata=[f"q={q}" for q in qs],
                hovertemplate="%{customdata}: %{y:.4f}<extra>" + lbl + "</extra>",
            ),
            row=2,
            col=3,
        )
    fig.add_hline(
        y=1.0,
        line=dict(color="white", dash="dash", width=1.5),
        opacity=0.7,
        row=2,
        col=3,
    )
    fig.update_xaxes(
        tickmode="array", tickvals=x_vr, ticktext=[f"q={q}" for q in qs], row=2, col=3
    )

    # [3,1] Rolling vol
    for df, col, grp, lbl, t in kws:
        bpy = df.attrs.get("bars_per_year", 252)
        roll = max(5, min(20, len(df) // 10))
        rv = (df["log_return"].rolling(roll).std() * np.sqrt(bpy) * 100).values
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=rv,
                mode="lines",
                name=lbl,
                line=dict(color=col, width=0.9, dash="dash" if grp == "c" else "solid"),
                legendgroup=grp,
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # [3,2] Squared-return ACF
    for (r, col, grp, lbl, t), off in zip(
        [(ra, CA, "a", la, ta), (rb, CB, "b", lb, tb), (rc, CC, "c", lc, tc)], offsets
    ):
        acf2 = [float(pd.Series(r**2).autocorr(l)) for l in lags_l]
        fig.add_trace(
            go.Bar(
                x=[l + off for l in lags_l],
                y=acf2,
                width=0.24,
                marker_color=col,
                opacity=0.85,
                name=f"{lbl} ARCH={_fmt(t['arch_p'],3)}",
                legendgroup=grp,
                showlegend=False,
            ),
            row=3,
            col=2,
        )

    # [3,3] Scorecard table
    scores = {la: score_a, lb: score_b}
    best = max(scores, key=scores.get)
    verdict = f"{best} leads on {scores[best]}/{len(criteria)} criteria"
    tbl_rows = [r[0] for r in criteria]
    tbl_va = [_fmt(r[1], 3) for r in criteria]
    tbl_vb = [_fmt(r[2], 3) for r in criteria]
    tbl_vc = [
        _fmt(r[4], 3) if len(r) > 4 and r[4] is not None else "n/a" for r in criteria
    ]
    tbl_w = [_winner(r[1], r[2], r[3], la, lb) for r in criteria]
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "<b>Criterion</b>",
                    f"<b>{la[:12]}</b>",
                    f"<b>{lb[:12]}</b>",
                    f"<b>{lc[:12]}</b>",
                    "<b>Leader</b>",
                ],
                fill_color=GRID,
                font=dict(color=TEXT, size=10),
                height=28,
            ),
            cells=dict(
                values=[tbl_rows, tbl_va, tbl_vb, tbl_vc, tbl_w],
                fill_color=PANE,
                font=dict(color=TEXT, size=9),
                height=24,
            ),
        ),
        row=3,
        col=3,
    )

    fig.update_layout(
        title=dict(
            text=f"3-Series Comparison — {la}  ·  {lb}  ·  {lc}<br>"
            f"<sup>{verdict}</sup>",
            font=dict(size=15, color=TEXT),
            x=0.5,
        ),
        barmode="overlay",
        height=1500,
        paper_bgcolor=BG,
        plot_bgcolor=PANE,
        font=dict(color=TEXT, family="monospace"),
        legend=dict(bgcolor=PANE, bordercolor=GRID, x=0.01, y=0.99),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)

    fig.write_html(str(out_path))
    print(f"  Interactive 3-series → {out_path}")
    try:
        os.unlink(str(tmp_path))
    except Exception:
        pass


def main():
    """
    Usage
    -----
    python compare_bars.py                                # all types, default dir
    python compare_bars.py --data-dir PATH               # all types, custom dir
    python compare_bars.py --types dollar volume         # specific types only
    python compare_bars.py --types dollar --time-bars    # add time bar baseline
    python compare_bars.py --data-dir PATH --out-dir PATH --types dollar --time-bars
    python compare_bars.py --types volatility --time-bars --figures D
    python compare_bars.py --types all --time-bars --figures D,E,I
    python compare_bars.py --types dollar --time-bars --figures all
    """
    import argparse

    parser = argparse.ArgumentParser(description="Information bar comparison")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing bar CSV files",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <data-dir>/comparisons)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=list(BAR_TYPES.keys()),
        metavar="TYPE",
        help=f"Bar types: {', '.join(BAR_TYPES)}",
    )
    parser.add_argument(
        "--time-bars",
        action="store_true",
        help="Load closest-matching time bar from DB and add as baseline series C",
    )
    parser.add_argument(
        "--exchange",
        default="binance",
        help="Exchange name for DB query (default: binance)",
    )
    parser.add_argument(
        "--symbol", default="btc", help="Symbol for DB query (default: BTCUSDT)"
    )
    parser.add_argument(
        "--figures",
        default=None,
        metavar="LETTERS",
        help=(
            "Generate standalone publication-ready figure PDFs. "
            "Comma-separated panel letters or 'all'. "
            "D=return-dist  E=return-acf  DE=D+E combined side-by-side  "
            "Saved to comparisons/<type>/figures/fig_<X>_<name>.pdf. "
            "Example: --figures D  or  --figures D,E  or  --figures DE  or  --figures all"
        ),
    )
    parser.add_argument(
        "--panels",
        default=None,
        metavar="LETTERS",
        help=(
            "Extract individual panel PDFs alongside the full figure. "
            "Comma-separated letters A-I, e.g. --panels D,E. "
            "A=price B=bar-size C=bars/day D=return-dist "
            "E=ACF F=VR G=vol H=sq-ACF I=scorecard. "
            "Saved to comparisons/<type>/panels/"
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = (
    Path(args.out_dir) if args.out_dir
    else Path(__file__).resolve().parents[1] / "data" / "comparison_results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_types = args.types if args.types else list(BAR_TYPES.keys())

    print(f"\nData dir    : {data_dir}")
    print(f"Out dir     : {out_dir}")
    print(f"Types       : {', '.join(run_types)}")
    print(f"Time bars   : {'yes' if args.time_bars else 'no'}")
    print(f"Exchange    : {args.exchange}")
    print(f"Symbol      : {args.symbol}")

    for bar_type in run_types:
        _run_one(
            bar_type,
            data_dir,
            out_dir,
            with_time_bars=args.time_bars,
            exchange=args.exchange,
            symbol=args.symbol,
            panels=args.panels,
            figures=args.figures,
        )

    print(f"\n{'═'*60}\n  ALL DONE — {out_dir.resolve()}\n{'═'*60}")


if __name__ == "__main__":
    main()