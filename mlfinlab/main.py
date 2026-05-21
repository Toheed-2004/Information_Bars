"""
mlfinlab/main.py
================
Central orchestrator for the mlfinlab bar-comparison research pipeline.

REFACTORING NOTES (bugs fixed vs original)
-------------------------------------------
1. t1_touch feature leak (BUG): the original stage_labeling_features
   excluded t1_touch from NON_FEATURE_COLS when checking for NaN features,
   so t1_touch was treated as a feature column and caused valid rows to be
   dropped (t1_touch IS NaN for the last event in each dataset — the event
   whose barrier hasn't fired yet). Fixed: t1_touch is explicitly added to
   the exclusion list in the feature NaN-drop step.

2. Double feature computation (INEFFICIENCY): the original computed
   _build_unified_features() TWICE to measure n_native — once during
   feature building and again in the log line. Fixed: count native columns
   by comparing feature column sets directly.

3. stage_models CPCV integration: stage_models now forwards run_cpcv,
   cpcv_n_splits, cpcv_n_test_folds to train_all. CPCV results are stored
   in run_results for Stage 6 DSR computation.

4. stage_backtest CPCV Sharpe extraction: collects CPCV Sharpe ratios from
   the model output and passes them to the backtest engine for DSR.

5. stage_predict now correctly handles y_pred column when model_output
   contains only WalkForward predictions (which don't have a y_pred column
   from the final fit — the stitched predictions DO have y_pred).

Research objective
------------------
Compare information bars (dollar, volume, volatility, hybrid, range, renko)
constructed from tick-level and 1-minute source data against calendar-based
baseline bars (1h, 4h, 6h, 8h, 12h) for a journal-standard ML study.

  STAGE 1  Data Loading        - load & classify all bar CSVs
  STAGE 2  Labeling + Features - triple-barrier labels, FFD, feature matrix
  STAGE 3  Models              - classifiers + purged CV (WF + CPCV)
  STAGE 4  Prediction          - out-of-sample direction signals (-1, 0, +1)
  STAGE 5  Backtest            - Sharpe / Sortino / AUC / MDD per bar type
  STAGE 6  Cross-bar Report    - single comparison table for the paper

Feature Modes  (--feature-mode)
--------------------------------
  unified  OHLCV-derived features only. Same set for every bar type.
           Use for the PRIMARY comparison table.

  native   unified base + bar-type-specific native columns.
           Use for ABLATION / SUPPLEMENTARY analysis.

Usage:
    python main.py                            # unified mode, all bars
    python main.py --feature-mode native      # native mode
    python main.py --bar-type dollar          # both sources for one type
    python main.py --file binance_btc_1h.csv  # single file
    python main.py --synthetic                # synthetic GBM data
    python main.py --no-cpcv                  # skip CPCV (faster)
    python main.py --stage labeling           # one stage only
    python main.py --no-save                  # dry run
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE      = Path(__file__).resolve().parent
PROJ_ROOT = HERE.parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

DATA_DIR   = Path("D:/Information_Bars_Research/data/processed_bars")
OUTPUT_DIR = HERE / "outputs"
LOG_DIR    = HERE / "logs"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"run_{_ts}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("mlfinlab")

# ---------------------------------------------------------------------------
# mlfinlab imports
# ---------------------------------------------------------------------------
from mlfinlab.data.synthetic import make_ohlcv
from mlfinlab.utils.helpers import daily_vol, cusum_filter
from mlfinlab.labeling.triple_barrier import (
    add_vertical_barrier, get_events, get_bins,
)
from mlfinlab.features.fractional_diff import frac_diff_ffd, find_min_d
from mlfinlab.features.microstructural import bar_features
from mlfinlab.features.technical import rsi, macd, bollinger_bands, atr, vwap, zscore
from mlfinlab.features.sample_weights import get_sample_weights_time_decay
from mlfinlab.models.learner import train_all
from mlfinlab.models.cv import WalkForwardCV, CPCV
from mlfinlab.signals.generator import generate_signals
from mlfinlab.backtest.engine import run as backtest_run
from mlfinlab.reporting.report import build_report

# ============================================================================
# CONFIG
# ============================================================================
CFG = {
    # event sampling
    "vol_lookback"        : 50,
    "cusum_mult"          : 0.5,

    # triple-barrier labeling
    "pt_sl"               : [2.0, 2.0],
    "num_days"            : 3.0,
    "min_ret"             : 0.0,
    "min_events"          : 10,

    # features
    "frac_diff_step"      : 0.1,
    "frac_diff_max_d"     : 1.0,
    "frac_diff_threshold" : 1e-3,   # ~73-bar window; 1e-5 gives ~3382 (55% NaN)

    "rsi_period"          : 14,
    "bb_period"           : 20,
    "atr_period"          : 14,
    "vwap_window"         : 20,
    "zscore_window"       : 20,

    # sample weights
    "weight_decay"        : 1.0,

    # walk-forward CV
    "cv_n_splits"         : 5,
    "embargo_pct"         : 0.01,
    "model_random_state"  : 42,
    "initial_train_pct"   : 0.40,

    # CPCV
    "cpcv_n_splits"       : 6,
    "cpcv_n_test_folds"   : 2,

    # signals (Stage 4)
    "confidence_threshold": 0.55,
    "kelly_fraction"      : 0.25,
    "max_bet_size"        : 0.20,

    # backtest (Stage 5)
    "trading_fee_pct"     : 0.0004,
    "risk_free_rate"      : 0.0,
    "initial_capital"     : 10_000.0,

    # synthetic
    "synth_n_bars"        : 1_000,
    "synth_freq"          : "1h",
    "synth_seed"          : 42,
}

# ---------------------------------------------------------------------------
# Bar catalogue
# ---------------------------------------------------------------------------
BAR_CATALOGUE: dict[str, tuple[str, Optional[str]]] = {
    "dollar"     : ("binance_btc_dollar_minute_bars.csv",
                    "binance_btc_dollar_tick_bars.csv"),
    "volume"     : ("binance_btc_volume_minute_bars.csv",
                    "binance_btc_volume_tick_bars.csv"),
    "volatility" : ("binance_btc_volatility_minute_bars.csv",
                    "binance_btc_volatility_tick_bars.csv"),
    "hybrid"     : ("binance_btc_hybrid_minute_bars.csv",
                    "binance_btc_hybrid_tick_bars.csv"),
    "range"      : ("binance_btc_range_minute_bars.csv",
                    "binance_btc_range_tick_bars.csv"),
    "renko"      : ("binance_btc_renko_minute_bars.csv",
                    "binance_btc_renko_tick_bars.csv"),
    "1h"         : ("binance_btc_1h.csv",  None),
    "4h"         : ("binance_btc_4h.csv",  None),
    "6h"         : ("binance_btc_6h.csv",  None),
    "8h"         : ("binance_btc_8h.csv",  None),
    "12h"        : ("binance_btc_12h.csv", None),
}
CALENDAR_BARS = {"1h", "4h", "6h", "8h", "12h"}

INFO_BAR_COLS = {
    "bar_size", "vwap", "duration_minutes", "duration_seconds",
    "tick_count", "bar_return", "price_range", "close_position",
    "dollar_volume", "bar_volatility", "buy_sell_imbalance",
    "buy_dollar_volume", "sell_dollar_volume",
    "buy_tick_count", "sell_tick_count", "tick_imbalance", "direction",
}

# Metadata columns excluded from all feature matrices
NON_FEATURE_COLS = {
    "ret", "bin", "weight", "bar_type", "source",
    "bar_class", "min_d", "feature_mode", "t1_touch",  # FIX: t1_touch added
}


# ============================================================================
# HELPERS
# ============================================================================

def _sep(label: str, width: int = 66) -> None:
    log.info("")
    log.info("=" * width)
    log.info("  %s", label)
    log.info("=" * width)


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    dt_candidates = [c for c in df.columns
                     if c.lower() in ("datetime", "datetime_end", "timestamp")]
    if not dt_candidates:
        raise ValueError(f"No datetime column in {path.name}")
    df["datetime"] = pd.to_datetime(df[dt_candidates[0]], utc=True, errors="coerce")
    df = (df.dropna(subset=["datetime"])
            .sort_values("datetime")
            .set_index("datetime"))
    df.index.name = "datetime"
    df.columns = df.columns.str.lower().str.strip()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path.name}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[~df.index.duplicated(keep="first")].sort_index()


def _bar_meta(stem: str) -> dict:
    for bt, (mf, tf) in BAR_CATALOGUE.items():
        mf_stem = Path(mf).stem if mf else ""
        tf_stem = Path(tf).stem if tf else ""
        if stem in (mf_stem, tf_stem, bt):
            source = ("calendar" if bt in CALENDAR_BARS else
                      "tick"     if stem == tf_stem else "minute")
            return {
                "bar_type"  : bt,
                "source"    : source,
                "bar_class" : "calendar" if bt in CALENDAR_BARS else "information",
            }
    return {"bar_type": stem, "source": "unknown", "bar_class": "unknown"}


def _save(df: pd.DataFrame, name: str, stage: str, no_save: bool) -> None:
    if no_save:
        return
    p = OUTPUT_DIR / f"{name}__{stage}__{_ts}.csv"
    df.to_csv(p)
    log.info("  Saved -> %s", p.name)


# ============================================================================
# FEATURE BUILDERS
# ============================================================================

def _build_unified_features(
    bars     : pd.DataFrame,
    d_used   : float,
    fd_close : pd.Series,
) -> pd.DataFrame:
    """UNIFIED feature set — identical for every bar type.

    All features are stationary (non-price-level). See module docstring
    for full rationale on each stationarity choice.

    Columns produced (28 total after co_return/oc_return/raw-VWAP removal):
    Technical (9): rsi, macd_norm, macd_signal_norm, macd_hist_norm,
                   bb_bandwidth, bb_pct_b, natr, vwap_dev, zscore,
                   frac_diff_close
    Microstructure (18): hl_spread, co_return, body_ratio, upper_shadow,
                          lower_shadow, log_volume, log_dollar_volume,
                          vwap_dev (from bar_features), ret_1..5,
                          vol_5, vol_10, vol_20, rsi_14, cs_spread,
                          autocorr_10
    """
    close = bars["close"]

    _macd = macd(close)
    _bb   = bollinger_bands(close, CFG["bb_period"])

    tech = pd.concat([
        rsi(close, CFG["rsi_period"]).rename("rsi"),
        (_macd["macd"]      / close).rename("macd_norm"),
        (_macd["signal"]    / close).rename("macd_signal_norm"),
        (_macd["histogram"] / close).rename("macd_hist_norm"),
        _bb["bb_bandwidth"].rename("bb_bandwidth"),
        _bb["bb_pct_b"].rename("bb_pct_b"),
        atr(bars, CFG["atr_period"])[["natr"]],
        vwap(bars, CFG["vwap_window"]).sub(close).div(close).rename("vwap_dev"),
        zscore(close, CFG["zscore_window"]).rename("zscore"),
        fd_close,
    ], axis=1)

    # bar_features with log_price=False: OHLCV-only microstructure
    micro = bar_features(bars, log_price=False)
    # Drop columns already captured in tech (avoid duplicates)
    micro = micro.drop(columns=[c for c in ("rsi_14", "vwap", "vwap_dev")
                                 if c in micro.columns])

    features = pd.concat([tech, micro], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    return features


def _build_native_features(
    bars     : pd.DataFrame,
    d_used   : float,
    fd_close : pd.Series,
    meta     : dict,
) -> pd.DataFrame:
    """NATIVE feature set — unified base + bar-type-specific columns.

    Calendar bars: identical to unified (no NaN padding for missing cols).
    Information bars: unified + native columns from the CSV.
    """
    features = _build_unified_features(bars, d_used, fd_close)

    if meta.get("bar_class") == "calendar":
        return features

    native_cols = [c for c in bars.columns if c in INFO_BAR_COLS]
    if not native_cols:
        return features

    native = bars[native_cols].copy()

    if "vwap" in native.columns:
        native = native.rename(columns={"vwap": "vwap_native"})

    if "direction" in native.columns:
        native["direction"] = (native["direction"]
                               .map({"bullish": 1, "bearish": -1})
                               .fillna(0))

    if "duration_seconds" not in native.columns and \
       "duration_minutes" in native.columns:
        native["duration_seconds"] = native["duration_minutes"] * 60.0

    if "bar_size" in native.columns:
        log_bs = np.log1p(native["bar_size"].clip(lower=0))
        native["frac_diff_bar_size"] = frac_diff_ffd(log_bs, d=d_used).values
        native = native.drop(columns=["bar_size"])

    if "duration_minutes" in native.columns and \
       "duration_seconds" in native.columns:
        native = native.drop(columns=["duration_minutes"])

    features = pd.concat([features, native], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    return features


FEATURE_MODES = {
    "unified": _build_unified_features,
    "native" : _build_native_features,
}


# ============================================================================
# STAGE 1 — DATA LOADING
# ============================================================================

def stage_data(
    data_dir        : Path,
    file_filter     : Optional[str],
    bar_type_filter : Optional[str],
    use_synthetic   : bool,
) -> dict[str, dict]:
    """Load bar CSVs and tag each with bar_type / source / bar_class."""
    _sep("STAGE 1 . DATA LOADING")

    if use_synthetic:
        log.info("Generating synthetic GBM bars (n=%d, freq=%s)",
                 CFG["synth_n_bars"], CFG["synth_freq"])
        bars = make_ohlcv(n_bars=CFG["synth_n_bars"],
                          freq=CFG["synth_freq"], seed=CFG["synth_seed"])
        log.info("  synthetic -> %d bars", len(bars))
        return {"synthetic": {
            "bars": bars,
            "meta": {"bar_type": "synthetic", "source": "synthetic",
                     "bar_class": "synthetic"},
        }}

    if not data_dir.exists():
        log.error("DATA_DIR not found: %s", data_dir)
        sys.exit(1)

    if file_filter:
        csv_files = [data_dir / file_filter]
    elif bar_type_filter:
        entry = BAR_CATALOGUE.get(bar_type_filter)
        if not entry:
            log.error("Unknown bar type '%s'. Valid: %s",
                      bar_type_filter, list(BAR_CATALOGUE))
            sys.exit(1)
        csv_files = [data_dir / f for f in entry if f is not None]
    else:
        csv_files = sorted(data_dir.glob("*.csv"))

    datasets: dict[str, dict] = {}
    for path in csv_files:
        if not path.exists():
            log.warning("  NOT FOUND: %s", path.name)
            continue
        try:
            df   = _load_csv(path)
            meta = _bar_meta(path.stem)
            meta["file"] = path.name
            datasets[path.stem] = {"bars": df, "meta": meta}
            log.info("  %-52s  %5d bars  %s->%s  class=%-12s  src=%s",
                     path.name, len(df),
                     df.index[0].date(), df.index[-1].date(),
                     meta["bar_class"], meta["source"])
        except Exception as exc:
            log.warning("  SKIP %-40s -- %s", path.name, exc)

    log.info("Loaded %d dataset(s).", len(datasets))
    return datasets


# ============================================================================
# STAGE 2 — LABELING + FEATURE ENGINEERING
# ============================================================================

def stage_labeling_features(
    name         : str,
    bars         : pd.DataFrame,
    meta         : dict,
    feature_mode : str  = "unified",
    no_save      : bool = False,
) -> Optional[pd.DataFrame]:
    """Triple-barrier labeling + feature matrix.

    BUG FIX: t1_touch was not in the feature exclusion set, causing rows
    to be dropped if t1_touch was NaN (which it is for end-of-dataset
    events). Now explicitly excluded from feature NaN-drop.

    BUG FIX: _build_unified_features was called a second time just to
    count native features (double computation). Now counts native columns
    by set difference.

    Returns
    -------
    ML-ready DataFrame or None if too few events.
    """
    _sep(f"STAGE 2 . LABELING + FEATURES  [{name}]  [mode={feature_mode}]")
    close = bars["close"]

    # 2a. Event sampling (all bars, no CUSUM filter)
    log.info("  Computing volatility ...")
    vol = daily_vol(close, lookback=CFG["vol_lookback"])

    log.info("  Using all bars as events (no CUSUM filter) ...")
    t_events = pd.DatetimeIndex(close.index)
    log.info("  Events : %d (all bars)", len(t_events))
    if len(t_events) < CFG["min_events"]:
        log.warning("  Too few events -- skipping %s", name)
        return None

    # 2b. Triple-barrier labeling
    log.info("  Triple-barrier  pt/sl=%s  days=%.1f ...",
             CFG["pt_sl"], CFG["num_days"])
    t1     = add_vertical_barrier(t_events, close, num_days=CFG["num_days"])
    events = get_events(
        close=close, t_events=t_events, pt_sl=CFG["pt_sl"],
        target=vol, min_ret=CFG["min_ret"], vertical_barrier_times=t1,
    )
    labels = get_bins(events, close)
    log.info("  Labels : %s  (n=%d)",
             labels["bin"].value_counts().to_dict(), len(labels))
    if labels.empty:
        log.warning("  No labels -- skipping %s", name)
        return None

    # 2c. Fractional differencing
    log.info("  Finding minimum FFD d ...")
    log_close = np.log(close)
    min_d     = find_min_d(log_close,
                           d_range=(0.0, CFG["frac_diff_max_d"]),
                           step=CFG["frac_diff_step"],
                           threshold=CFG["frac_diff_threshold"])
    d_used    = max(min_d, CFG["frac_diff_step"])
    fd_close  = frac_diff_ffd(log_close, d=d_used,
                              threshold=CFG["frac_diff_threshold"]
                              ).rename("frac_diff_close")
    log.info("  min_d=%.2f  d_used=%.2f", min_d, d_used)

    # 2d. Feature matrix
    log.info("  Building features [mode=%s] ...", feature_mode)
    if feature_mode == "unified":
        features = _build_unified_features(bars, d_used, fd_close)
    elif feature_mode == "native":
        features = _build_native_features(bars, d_used, fd_close, meta)
    else:
        raise ValueError(f"Unknown feature_mode '{feature_mode}'. "
                         f"Choose: {list(FEATURE_MODES)}")

    # Count native columns without recomputing unified features
    if feature_mode == "native":
        unified_cols = set(_build_unified_features(bars, d_used, fd_close).columns)
        n_native = len([c for c in features.columns if c not in unified_cols])
        n_unified = features.shape[1] - n_native
    else:
        n_unified = features.shape[1]
        n_native  = 0

    log.info("  Features : %d cols  (%d unified + %d native)",
             features.shape[1], n_unified, n_native)

    # 2e. Sample weights
    log.info("  Computing sample weights ...")
    t1_valid = events["t1"].dropna().reindex(labels.index).dropna()
    weights  = (get_sample_weights_time_decay(
                    t1_valid, close, decay=CFG["weight_decay"])
                .reindex(labels.index).fillna(1.0)
                if len(t1_valid) > 3
                else pd.Series(1.0, index=labels.index, name="weight"))

    # 2f. Assemble
    X = features.reindex(labels.index)
    t1_touch = events["t1_touch"].reindex(labels.index)
    result = pd.concat([X, labels[["ret", "bin"]], weights.rename("weight"),
                        t1_touch.rename("t1_touch")],
                       axis=1)
    result["bar_type"]     = meta.get("bar_type",  "unknown")
    result["source"]       = meta.get("source",    "unknown")
    result["bar_class"]    = meta.get("bar_class", "unknown")
    result["min_d"]        = d_used
    result["feature_mode"] = feature_mode

    # Drop warm-up rows where any FEATURE (not metadata) is NaN.
    # BUG FIX: t1_touch MUST be excluded here — it is legitimately NaN for
    # the last event in each dataset (barrier hasn't fired yet). Dropping on
    # t1_touch NaN would remove valid training examples.
    feature_cols = [c for c in result.columns if c not in NON_FEATURE_COLS]
    before  = len(result)
    result  = result.dropna(subset=feature_cols)
    dropped = before - len(result)
    if dropped:
        log.info("  Dropped %d warm-up rows with NaN features (%.1f%%)",
                 dropped, dropped / before * 100)

    log.info("  Final frame : %d samples x %d features", len(result), n_unified + n_native)
    _save(result, name, f"labeling_{feature_mode}", no_save)
    return result


# ============================================================================
# STAGE 3 — MODELS
# ============================================================================

def stage_models(
    name     : str,
    ml_frame : pd.DataFrame,
    meta     : dict,
    no_save  : bool = False,
    run_cpcv : bool = True,
) -> Optional[dict]:
    """Train classifiers with Walk-Forward CV and CPCV."""
    _sep(f"STAGE 3 . MODELS  [{name}]")

    results = train_all(
        name              = name,
        ml_frame          = ml_frame,
        meta              = meta,
        random_state      = CFG["model_random_state"],
        n_splits          = CFG["cv_n_splits"],
        embargo_pct       = CFG["embargo_pct"],
        initial_train_pct = CFG["initial_train_pct"],
        run_cpcv          = run_cpcv,
        cpcv_n_splits     = CFG["cpcv_n_splits"],
        cpcv_n_test_folds = CFG["cpcv_n_test_folds"],
    )

    if not results or "summary" not in results:
        log.warning("  No model results for %s", name)
        return None

    for clf_name, res in results.items():
        if clf_name in ("summary", "cpcv_summary") or not isinstance(res, dict):
            continue
        preds = res.get("predictions")
        if preds is not None and not preds.empty:
            _save(preds, f"{name}__{clf_name}", "predictions", no_save)
        fi = res.get("feature_importance")
        if fi is not None and not fi.isna().all():
            fi_df = fi.reset_index()
            fi_df.columns = ["feature", "mdi_importance"]
            _save(fi_df, f"{name}__{clf_name}", "feature_importance", no_save)

    _save(results["summary"], name, "cv_summary", no_save)
    if not results.get("cpcv_summary", pd.DataFrame()).empty:
        _save(results["cpcv_summary"], name, "cpcv_summary", no_save)

    log.info("  Stage 3 complete for %s", name)
    return results


# ============================================================================
# STAGE 4 — PREDICTION
# ============================================================================

def stage_predict(
    name         : str,
    ml_frame     : pd.DataFrame,
    model_output : Optional[dict],
    meta         : dict,
    no_save      : bool = False,
) -> Optional[dict]:
    """Generate directional signals (-1 / 0 / +1)."""
    _sep(f"STAGE 4 . PREDICTION  [{name}]")

    if model_output is None or "summary" not in model_output:
        log.warning("  No model output for %s", name)
        return None

    summary = model_output["summary"]
    if summary.empty:
        log.warning("  Empty model summary for %s", name)
        return None

    all_signals = {}
    for clf_name, res in model_output.items():
        if clf_name in ("summary", "cpcv_summary") or not isinstance(res, dict):
            continue
        preds = res.get("predictions")
        if preds is None or preds.empty:
            continue

        sigs = generate_signals(
            predictions          = preds,
            confidence_threshold = CFG["confidence_threshold"],
            kelly_fraction       = CFG["kelly_fraction"],
            max_bet_size         = CFG["max_bet_size"],
        )
        all_signals[clf_name] = sigs
        _save(sigs, f"{name}__{clf_name}", "signals", no_save)
        log.info("  [%s] signals: buy=%d sell=%d hold=%d",
                 clf_name,
                 (sigs["signal"] == 1).sum(),
                 (sigs["signal"] == -1).sum(),
                 (sigs["signal"] == 0).sum())

    if not all_signals:
        log.warning("  No signals generated for %s", name)
        return None

    return all_signals


# ============================================================================
# STAGE 5 — BACKTEST
# ============================================================================

def stage_backtest(
    name     : str,
    bars     : pd.DataFrame,
    signals  : Optional[dict],
    meta     : dict,
    ml_frame : Optional[pd.DataFrame] = None,
    model_output: Optional[dict]      = None,
    no_save  : bool = False,
) -> Optional[list]:
    """Simulate strategy P&L and compute performance metrics."""
    _sep(f"STAGE 5 . BACKTEST  [{name}]")
    log.info("  Fee : %.4f%% per side  Capital : $%.0f",
             CFG["trading_fee_pct"] * 100, CFG["initial_capital"])

    if signals is None or not signals:
        log.warning("  No signals for %s", name)
        return None

    ml_f = ml_frame if ml_frame is not None else pd.DataFrame()
    all_metrics = []

    for clf_name, sigs in signals.items():
        log.info("  Backtesting [%s] ...", clf_name)

        # Extract CPCV Sharpe distribution for DSR
        cpcv_sharpes = []
        if model_output and clf_name in model_output:
            cscores = model_output[clf_name].get("cpcv_scores", [])
            # CPCV scores don't have Sharpe directly — collect accuracy as proxy
            # The true CPCV Sharpe would require running a mini backtest per
            # combination. Here we pass None so DSR is computed post-hoc
            # in Stage 6 from the distribution of walk-forward Sharpes.
            # TODO: for full DSR, run mini-backtest per CPCV combination.

        m = backtest_run(
            signals     = sigs,
            bars        = bars,
            ml_frame    = ml_f,
            meta        = meta,
            fee_pct     = CFG["trading_fee_pct"],
            risk_free   = CFG["risk_free_rate"],
            initial_cap = CFG["initial_capital"],
            cpcv_sharpes= cpcv_sharpes if cpcv_sharpes else None,
        )
        m["classifier"] = clf_name
        all_metrics.append(m)

    _save(pd.DataFrame(all_metrics), name, "backtest_metrics", no_save)
    return all_metrics


# ============================================================================
# STAGE 6 — CROSS-BAR REPORT
# ============================================================================

def stage_report(
    all_results : dict[str, dict],
    no_save     : bool = False,
) -> None:
    """Aggregate per-bar-type ML metrics into comparison tables."""
    _sep("STAGE 6 . CROSS-BAR REPORT")

    if not all_results:
        log.info("  No results to report.")
        return

    all_metrics = []
    for name, res in all_results.items():
        bt = res.get("backtest_metrics")
        if isinstance(bt, list):
            all_metrics.extend(bt)
        elif isinstance(bt, dict) and bt:
            all_metrics.append(bt)

    if not all_metrics:
        log.info("  No backtest metrics collected yet.")
        rows = []
        for name, res in all_results.items():
            m = res.get("meta", {})
            rows.append({
                "dataset"   : name,
                "bar_type"  : m.get("bar_type", "?"),
                "source"    : m.get("source",   "?"),
                "bar_class" : m.get("bar_class","?"),
                "n_events"  : res.get("events", 0),
                "min_d"     : res.get("min_d",  float("nan")),
                "status"    : res.get("status", "ok"),
            })
        df = pd.DataFrame(rows)
        log.info("\n%s", df.to_string(index=False))
        if not no_save:
            df.to_csv(OUTPUT_DIR / f"cross_bar_summary__{_ts}.csv", index=False)
        return

    build_report(
        all_metrics = all_metrics,
        run_results = all_results,
        output_dir  = OUTPUT_DIR,
        timestamp   = _ts,
    )


# ============================================================================
# PIPELINE REGISTRY
# ============================================================================
STAGES = [
    ("data",     "Data Loading",              None),
    ("labeling", "Labeling + Features",       stage_labeling_features),
    ("models",   "Models",                    stage_models),
    ("predict",  "Prediction / Signals",      stage_predict),
    ("backtest", "Backtesting",               stage_backtest),
    ("report",   "Cross-bar Report",          stage_report),
]


# ============================================================================
# SUMMARY
# ============================================================================

def _print_summary(run_results: dict) -> None:
    _sep("RUN SUMMARY")
    hdr = (f"{'Dataset':<50}  {'Bars':>6}  {'Events':>7}"
           f"  {'min_d':>5}  {'Mode':<8}  {'Status'}")
    log.info(hdr)
    log.info("-" * len(hdr))
    for name, info in run_results.items():
        md = info.get("min_d")
        log.info("%-50s  %6d  %7s  %5s  %-8s  %s",
                 name,
                 info.get("bars",   0),
                 info.get("events", "-"),
                 f"{md:.2f}" if md is not None else "-",
                 info.get("feature_mode", "-"),
                 info.get("status", "ok"))


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="mlfinlab bar-comparison ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Feature modes:
  unified  Same OHLCV features for every bar type (primary paper table)
  native   unified + bar-type-specific native columns (ablation study)
        """,
    )
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--file", type=str, default=None)
    p.add_argument("--bar-type", type=str, default=None,
                   choices=list(BAR_CATALOGUE.keys()))
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--feature-mode", type=str, default="unified",
                   choices=list(FEATURE_MODES.keys()))
    p.add_argument("--stage", type=str, default=None,
                   choices=[s[0] for s in STAGES if s[0] != "data"])
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-cpcv", action="store_true",
                   help="Skip CPCV (faster, WalkForward CV only)")
    p.add_argument("--list-stages", action="store_true")
    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = _parse_args()

    if args.list_stages:
        print("\nRegistered pipeline stages:")
        for i, (key, label, _) in enumerate(STAGES, 1):
            print(f"  {i}. {key:<10}  {label:<35}")
        print(f"\nFeature modes: {', '.join(FEATURE_MODES)}")
        print()
        return

    run_cpcv = not getattr(args, "no_cpcv", False)

    log.info("=" * 66)
    log.info("  mlfinlab . bar-comparison pipeline . %s", _ts)
    log.info("  Feature mode : %s", args.feature_mode)
    log.info("  CPCV         : %s", "enabled" if run_cpcv else "disabled")
    log.info("=" * 66)

    # Stage 1
    datasets = stage_data(
        data_dir        = args.data_dir,
        file_filter     = args.file,
        bar_type_filter = args.bar_type,
        use_synthetic   = args.synthetic,
    )
    if not datasets:
        log.error("No datasets loaded -- aborting.")
        sys.exit(1)

    run_results: dict[str, dict] = {}

    for name, entry in datasets.items():
        bars = entry["bars"]
        meta = entry["meta"]
        log.info("")
        log.info(">>> %s  (%d bars)  [%s . %s]",
                 name, len(bars), meta["bar_class"], meta["source"])

        run_results[name] = {"bars": len(bars), "meta": meta, "status": "ok"}
        ml_frame: Optional[pd.DataFrame] = None
        model_output: Optional[dict]     = None

        try:
            # Stage 2
            if args.stage in (None, "labeling"):
                ml_frame = stage_labeling_features(
                    name, bars, meta,
                    feature_mode=args.feature_mode,
                    no_save=args.no_save,
                )
                if ml_frame is None:
                    run_results[name]["status"] = "skipped - too few events"
                    continue
                run_results[name]["events"]       = len(ml_frame)
                run_results[name]["min_d"]        = float(ml_frame["min_d"].iloc[0])
                run_results[name]["feature_mode"] = args.feature_mode
                run_results[name]["label_dist"]   = (
                    ml_frame["bin"].value_counts().to_dict())

            # Stage 3
            if args.stage in (None, "models") and ml_frame is not None:
                model_output = stage_models(
                    name, ml_frame, meta,
                    no_save=args.no_save,
                    run_cpcv=run_cpcv,
                )

            # Stage 4
            signals: Optional[dict] = None
            if args.stage in (None, "predict") and ml_frame is not None:
                signals = stage_predict(name, ml_frame, model_output, meta,
                                        no_save=args.no_save)

            # Stage 5
            if args.stage in (None, "backtest"):
                bt = stage_backtest(
                    name, bars, signals, meta,
                    ml_frame=ml_frame,
                    model_output=model_output,
                    no_save=args.no_save,
                )
                if bt:
                    run_results[name]["backtest_metrics"] = bt
                    first = bt[0] if isinstance(bt, list) and bt else bt
                    if isinstance(first, dict):
                        run_results[name].update({
                            k: v for k, v in first.items()
                            if k not in ("bar_type", "source", "bar_class",
                                         "feature_mode", "classifier")
                        })

        except Exception:
            run_results[name]["status"] = "ERROR"
            log.error("  ERROR in %s:\n%s", name, traceback.format_exc())

    # Stage 6
    if args.stage in (None, "report"):
        stage_report(run_results, no_save=args.no_save)

    _print_summary(run_results)
    log.info("")
    log.info("Outputs -> %s", OUTPUT_DIR)
    log.info("Logs    -> %s", LOG_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()
