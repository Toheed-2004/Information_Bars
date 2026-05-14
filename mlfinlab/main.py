"""
mlfinlab/main.py
================
Central orchestrator for the mlfinlab bar-comparison research pipeline.

Research objective
------------------
Compare information bars (dollar, volume, volatility, hybrid, range, renko)
constructed from tick-level and 1-minute source data against calendar-based
baseline bars (1h, 4h, 6h, 8h, 12h) for a journal-standard ML study.

Statistical bar-quality comparison is handled by compare_bars.py.
This pipeline covers the ML layer:

  STAGE 1  Data Loading        - load & classify all bar CSVs
  STAGE 2  Labeling + Features - triple-barrier labels, FFD, feature matrix
  STAGE 3  Models              - classifiers + purged CV per bar type
  STAGE 4  Prediction          - out-of-sample direction signals (-1, 0, +1)
  STAGE 5  Backtest            - Sharpe / Sortino / AUC / MDD per bar type
  STAGE 6  Cross-bar Report    - single comparison table for the paper

Feature Modes  (--feature-mode)
--------------------------------
Two mutually exclusive modes control what goes into the feature matrix.
Every other part of the pipeline (labeling, models, backtest, report) is
identical regardless of which mode is chosen.

  unified  -- OHLCV-derived features only.  Exactly the same feature set
              is computed for every bar type.  Calendar bars and information
              bars are treated identically.
              Use this for the PRIMARY comparison table in the paper.
              Differences in model performance are attributable solely to
              bar construction quality, not to feature availability.

              Features: RSI, MACD, Bollinger Bands, ATR, rolling VWAP,
                        Z-score, FFD close, OHLCV microstructure
                        (hl_spread, body_ratio, shadows, log_volume, ...)

  native   -- unified base PLUS bar-type-specific columns appended.
              Information bars contribute tick_count, duration_seconds,
              buy_sell_imbalance, bar_volatility, close_position, etc.
              Calendar bars still get only the unified feature set;
              they are NOT padded with NaN for native columns because
              NaN-padding would introduce a systematic difference that
              is an artefact of the feature engineering, not of the bars.
              Use this for ABLATION / SUPPLEMENTARY analysis to quantify
              how much the richer information-bar data adds beyond OHLCV.

              Features: all unified + available native columns per bar type
                        (see INFO_BAR_COLS for the complete list)

Usage (from D:/Information_Bars_Research/mlfinlab/):
    python main.py                                   # unified mode, all bars
    python main.py --feature-mode native             # native mode, all bars
    python main.py --bar-type dollar                 # both sources for one type
    python main.py --file binance_btc_1h.csv         # single file
    python main.py --synthetic                       # synthetic GBM data
    python main.py --stage labeling                  # one stage only
    python main.py --list-stages                     # show stages and exit
    python main.py --no-save                         # dry run
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
HERE      = Path(__file__).resolve().parent     # .../mlfinlab/
PROJ_ROOT = HERE.parent                         # .../Information_Bars_Research/
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

# ============================================================================
# CONFIG
# ============================================================================
CFG = {
    # event sampling
    "vol_lookback"       : 50,
    "cusum_mult"         : 0.5,

    # triple-barrier labeling
    "pt_sl"              : [2.0, 2.0],
    "num_days"           : 3.0,
    "min_ret"            : 0.0,
    "min_events"         : 10,

    # features
    "frac_diff_step"     : 0.1,
    "frac_diff_max_d"    : 1.0,
    "rsi_period"         : 14,
    "bb_period"          : 20,
    "atr_period"         : 14,
    "vwap_window"        : 20,
    "zscore_window"      : 20,

    # sample weights
    "weight_decay"       : 1.0,

    # models (Stage 3 placeholder)
    "cv_n_splits"        : 5,
    "embargo_pct"        : 0.01,
    "model_random_state" : 42,

    # backtest (Stage 5 placeholder)
    "trading_fee_pct"    : 0.0004,
    "risk_free_rate"     : 0.0,
    "initial_capital"    : 10_000.0,

    # synthetic
    "synth_n_bars"       : 1_000,
    "synth_freq"         : "1h",
    "synth_seed"         : 42,
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

# Native columns present in information bars but not in calendar bars.
# Used only in 'native' feature mode.
INFO_BAR_COLS = {
    "bar_size", "vwap", "duration_minutes", "duration_seconds",
    "tick_count", "bar_return", "price_range", "close_position",
    "dollar_volume", "bar_volatility", "buy_sell_imbalance",
    "buy_dollar_volume", "sell_dollar_volume",
    "buy_tick_count", "sell_tick_count", "tick_imbalance", "direction",
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
            if bt in CALENDAR_BARS:
                source = "calendar"
            elif stem == tf_stem:
                source = "tick"
            else:
                source = "minute"
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
# Two named functions, called exclusively by stage_labeling_features.
# Adding a third mode means adding one function and one elif below.
# ============================================================================

def _build_unified_features(
    bars     : pd.DataFrame,
    d_used   : float,
    fd_close : pd.Series,
) -> pd.DataFrame:
    """
    UNIFIED feature set -- identical for every bar type.

    Computed purely from OHLCV columns (open, high, low, close, volume).
    Calendar bars and all information bars go through exactly this code path.
    Use for the PRIMARY paper comparison table: any performance difference
    between bar types reflects bar construction quality alone.

    frac_diff_close is INCLUDED here -- FFD is applied to every bar type
    identically. The d value differs per bar type (found by ADF in Stage 2)
    but the transformation itself is universal.

    vwap_rolling is an OHLCV-derived proxy: rolling(H+L+C)/3 * volume.
    This is different from bars["vwap"] which is the tick-level execution
    VWAP recorded during bar formation (only in dollar/hybrid/range/
    renko/volatility/volume tick bars). The native vwap column belongs
    in native mode only, under the name "vwap_native" to avoid collision.

    Columns produced (33 total)
    ---------------------------
    Technical:
        rsi, macd_macd, macd_signal, macd_histogram,
        bb_bb_upper, bb_bb_mid, bb_bb_lower, bb_bb_bandwidth, bb_bb_pct_b,
        atr, natr, vwap_rolling, zscore, frac_diff_close

    OHLCV microstructure (bar_features, OHLCV-only):
        hl_spread, co_return, oc_return, body_ratio,
        upper_shadow, lower_shadow, log_volume, log_dollar_volume,
        ret_1..ret_5, vol_5, vol_10, vol_20, autocorr_10
    """
    close = bars["close"]

    tech = pd.concat([
        rsi(close, CFG["rsi_period"]).rename("rsi"),
        macd(close).add_prefix("macd_"),
        bollinger_bands(close, CFG["bb_period"]).add_prefix("bb_"),
        atr(bars, CFG["atr_period"]),
        vwap(bars, CFG["vwap_window"]).rename("vwap_rolling"),  # OHLCV proxy
        zscore(close, CFG["zscore_window"]).rename("zscore"),
        fd_close,                                               # FFD on log-close
    ], axis=1)

    # bar_features with log_price=False stays OHLCV-only
    micro = bar_features(bars, log_price=False)
    # Drop columns already in tech to avoid duplicates
    micro = micro.drop(columns=[c for c in ("rsi_14", "vwap")
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
    """
    NATIVE feature set -- unified base + bar-type-specific columns.

    Starts with the full unified feature set, then appends native columns
    that are present in this specific bar type's CSV (tick_count,
    duration_seconds, buy_sell_imbalance, bar_volatility, etc.).

    Calendar bars receive only the unified set -- they are NOT given NaN
    placeholders for native columns. Padding with NaN would introduce a
    systematic missing-data signal that is an artefact of feature
    engineering rather than of the bar construction method itself.

    Use this mode for ablation analysis and supplementary material:
    the performance delta between unified and native isolates the
    contribution of the structural information unique to each bar type.

    Additional columns (when available per bar type)
    ------------------------------------------------
    All information bars:
        tick_count, close_position, price_range, bar_return,
        duration_seconds, frac_diff_bar_size

    Dollar / Volume:
        + dollar_volume (volume bars have this too)

    Dollar tick / Hybrid tick / Range tick / Volatility tick:
        + buy_sell_imbalance, buy_dollar_volume, sell_dollar_volume

    Hybrid tick / Volatility tick:
        + bar_volatility

    Renko tick:
        + direction (encoded: bullish=1, bearish=-1),
          tick_imbalance, buy_tick_count, sell_tick_count

    All native numeric columns are stationary-checked:
        bar_size -> log1p -> FFD at d_used
    """
    # Start from unified
    features = _build_unified_features(bars, d_used, fd_close)

    # Calendar bars: return unified as-is (no NaN padding)
    if meta.get("bar_class") == "calendar":
        return features

    # Information bars: append native columns present in this CSV
    native_cols = [c for c in bars.columns if c in INFO_BAR_COLS]
    if not native_cols:
        return features

    native = bars[native_cols].copy()

    # Rename native vwap to avoid collision with vwap_rolling in unified.
    # bars["vwap"] = tick-level execution VWAP (precise, from bar formation).
    # vwap_rolling = (H+L+C)/3 rolling proxy computed from OHLCV after the fact.
    # They measure different things; both are valid; names must be distinct.
    if "vwap" in native.columns:
        native = native.rename(columns={"vwap": "vwap_native"})

    # Encode renko direction as numeric
    if "direction" in native.columns:
        native["direction"] = (native["direction"]
                               .map({"bullish": 1, "bearish": -1})
                               .fillna(0))

    # Unify duration to seconds
    if "duration_seconds" not in native.columns and \
       "duration_minutes" in native.columns:
        native["duration_seconds"] = native["duration_minutes"] * 60.0

    # FFD on bar_size -- the raw target threshold is non-stationary.
    # Same d as log-close FFD (found by ADF in Stage 2) for consistency.
    if "bar_size" in native.columns:
        log_bs = np.log1p(native["bar_size"].clip(lower=0))
        native["frac_diff_bar_size"] = frac_diff_ffd(log_bs, d=d_used).values
        native = native.drop(columns=["bar_size"])

    # Drop duration_minutes now that duration_seconds is present
    if "duration_minutes" in native.columns and \
       "duration_seconds" in native.columns:
        native = native.drop(columns=["duration_minutes"])

    features = pd.concat([features, native], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    return features


# Feature mode registry -- extend here to add new modes
FEATURE_MODES = {
    "unified": _build_unified_features,   # primary comparison
    "native" : _build_native_features,    # ablation / supplementary
}


# ============================================================================
# STAGE 1 -- DATA LOADING
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
# STAGE 2 -- LABELING + FEATURE ENGINEERING
# ============================================================================

def stage_labeling_features(
    name         : str,
    bars         : pd.DataFrame,
    meta         : dict,
    feature_mode : str  = "unified",
    no_save      : bool = False,
) -> Optional[pd.DataFrame]:
    """Triple-barrier labeling + feature matrix.

    Parameters
    ----------
    name         : Dataset identifier.
    bars         : OHLCV DataFrame with DatetimeIndex.
    meta         : Bar metadata dict (bar_type, source, bar_class).
    feature_mode : 'unified' or 'native' -- see module docstring.
    no_save      : Skip writing output CSV when True.

    Returns
    -------
    ML-ready DataFrame or None if too few events.
    Columns: <features> | ret | bin | weight | bar_type | source
                        | bar_class | min_d | feature_mode

    Labeling design decisions
    -------------------------
    - drop_labels() is NOT called. Class-0 proportion varies across bar
      types and is itself a research finding.
    - One labeling method (triple-barrier) applied identically to every
      bar type for a fair comparison.
    - FFD d is found per-dataset and stored in min_d column for the paper
      stationarity comparison table.
    """
    _sep(f"STAGE 2 . LABELING + FEATURES  [{name}]  [mode={feature_mode}]")
    close = bars["close"]

    # -- 2a. Event sampling
    log.info("  Computing volatility & sampling events ...")
    vol      = daily_vol(close, lookback=CFG["vol_lookback"])
    t_events = cusum_filter(close, threshold=vol * CFG["cusum_mult"])
    log.info("  Events sampled : %d", len(t_events))
    if len(t_events) < CFG["min_events"]:
        log.warning("  Too few events -- skipping %s", name)
        return None

    # -- 2b. Triple-barrier labeling
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

    # -- 2c. Fractional differencing (per-dataset min d)
    log.info("  Finding minimum FFD d ...")
    log_close = np.log(close)
    min_d     = find_min_d(log_close,
                           d_range=(0.0, CFG["frac_diff_max_d"]),
                           step=CFG["frac_diff_step"])
    d_used    = max(min_d, CFG["frac_diff_step"])
    fd_close  = frac_diff_ffd(log_close, d=d_used).rename("frac_diff_close")
    log.info("  min_d=%.2f  d_used=%.2f", min_d, d_used)

    # -- 2d. Feature matrix via selected mode
    log.info("  Building features [mode=%s] ...", feature_mode)
    if feature_mode == "unified":
        features = _build_unified_features(bars, d_used, fd_close)
    elif feature_mode == "native":
        features = _build_native_features(bars, d_used, fd_close, meta)
    else:
        raise ValueError(f"Unknown feature_mode '{feature_mode}'. "
                         f"Choose: {list(FEATURE_MODES)}")

    n_native = features.shape[1] - _build_unified_features(
        bars, d_used, fd_close).shape[1]
    log.info("  Features : %d cols  (%d unified + %d native)",
             features.shape[1],
             features.shape[1] - max(n_native, 0),
             max(n_native, 0))

    # -- 2e. Sample weights
    log.info("  Computing sample weights ...")
    t1_valid = events["t1"].dropna().reindex(labels.index).dropna()
    weights  = (get_sample_weights_time_decay(
                    t1_valid, close, decay=CFG["weight_decay"])
                .reindex(labels.index).fillna(1.0)
                if len(t1_valid) > 3
                else pd.Series(1.0, index=labels.index, name="weight"))

    # -- 2f. Assemble
    X = features.reindex(labels.index)
    result = pd.concat([X, labels[["ret", "bin"]], weights.rename("weight")],
                       axis=1)
    result["bar_type"]     = meta.get("bar_type",  "unknown")
    result["source"]       = meta.get("source",    "unknown")
    result["bar_class"]    = meta.get("bar_class", "unknown")
    result["min_d"]        = d_used
    result["feature_mode"] = feature_mode

    log.info("  Final frame : %d samples x %d features", len(result), X.shape[1])
    _save(result, name, f"labeling_{feature_mode}", no_save)
    return result


# ============================================================================
# STAGE 3 -- MODELS  (placeholder)
# ============================================================================

def stage_models(
    name    : str,
    ml_frame: pd.DataFrame,
    meta    : dict,
    no_save : bool = False,
) -> Optional[pd.DataFrame]:
    """Train classifiers on the labelled feature matrix with purged CV.

    Implement in mlfinlab/models/

    Sub-stages
    ----------
    3a. Feature prep
          NON_FEATURE_COLS = {"ret","bin","weight","bar_type","source",
                               "bar_class","min_d","feature_mode"}
          X = ml_frame.drop(columns=NON_FEATURE_COLS)
          y = ml_frame["bin"]      # -1 / 0 / +1
          w = ml_frame["weight"]

    3b. Purged K-fold CV  (AFML Ch.7)
          PurgedKFold(n_splits=CFG["cv_n_splits"],
                      pct_embargo=CFG["embargo_pct"])

    3c. Classifiers
          RandomForestClassifier     primary de Prado recommendation
          GradientBoostingClassifier strong gradient-based baseline
          SVC(kernel="rbf")          classical benchmark

    3d. Feature importance: MDI, MDA, SFI

    3e. Persist model per bar type + feature mode
          joblib.dump(model, OUTPUT_DIR / f"{name}__{feature_mode}__model.pkl")

    Returns
    -------
    pd.DataFrame  y_true | y_pred | prob_m1 | prob_0 | prob_p1
    """
    _sep(f"STAGE 3 . MODELS  [{name}]  -- implement in mlfinlab/models/")
    log.info("  Input  : %d samples x %d cols", *ml_frame.shape)
    log.info("  Labels : %s", ml_frame["bin"].value_counts().to_dict())
    log.info("  Mode   : %s", ml_frame["feature_mode"].iloc[0])
    log.info("  [ stub ]")
    return None


# ============================================================================
# STAGE 4 -- PREDICTION  (placeholder)
# ============================================================================

def stage_predict(
    name         : str,
    ml_frame     : pd.DataFrame,
    model_output : Optional[pd.DataFrame],
    meta         : dict,
    no_save      : bool = False,
) -> Optional[pd.DataFrame]:
    """Generate directional signals (-1 / 0 / +1).

    Implement in mlfinlab/signals/

    Sub-stages
    ----------
    4a. Probability calibration (isotonic / Platt)
    4b. Signal thresholding:
          prob(+1) > thresh  ->  signal = +1
          prob(-1) > thresh  ->  signal = -1
          else               ->  signal =  0  (no trade)
    4c. Bet sizing: fixed-fraction or fractional Kelly

    Returns
    -------
    pd.DataFrame  signal (-1/0/+1) | bet_size | confidence
    """
    _sep(f"STAGE 4 . PREDICTION  [{name}]  -- implement in mlfinlab/signals/")
    log.info("  [ stub ]")
    return None


# ============================================================================
# STAGE 5 -- BACKTEST  (placeholder)
# ============================================================================

def stage_backtest(
    name     : str,
    bars     : pd.DataFrame,
    signals  : Optional[pd.DataFrame],
    meta     : dict,
    no_save  : bool = False,
) -> Optional[dict]:
    """Simulate strategy P&L and compute performance metrics.

    Implement in mlfinlab/backtest/

    Sub-stages
    ----------
    5a. Walk-forward simulation; fee = CFG["trading_fee_pct"] per side
    5b. Metrics (stored per bar type + feature mode for Stage 6 table):
          Sharpe ratio, Sortino ratio, Max drawdown, Calmar ratio,
          Win rate, Profit factor, AUC-ROC, Deflated Sharpe Ratio
    5c. Equity curve + drawdown plot

    Returns
    -------
    dict  {metric: value, "bar_type": ..., "source": ..., "feature_mode": ...}
    """
    _sep(f"STAGE 5 . BACKTEST  [{name}]  -- implement in mlfinlab/backtest/")
    log.info("  Fee     : %.4f%%  (%.4f%% round-trip)",
             CFG["trading_fee_pct"] * 100,
             CFG["trading_fee_pct"] * 200)
    log.info("  Capital : $%.0f", CFG["initial_capital"])
    log.info("  [ stub ]")
    return None


# ============================================================================
# STAGE 6 -- CROSS-BAR REPORT  (placeholder)
# ============================================================================

def stage_report(
    all_results : dict[str, dict],
    no_save     : bool = False,
) -> None:
    """Aggregate per-bar-type ML metrics into a comparison table.

    Implement in mlfinlab/reporting/

    Sub-stages
    ----------
    6a. Collect backtest dicts for all bar types from Stage 5
    6b. Summary DataFrame:
          Rows    = bar types  (dollar_minute, dollar_tick, ..., 1h, 4h, ...)
          Columns = Sharpe | Sortino | AUC | MDD | WinRate
                  | min_d  | n_events | label_0_pct | feature_mode
    6c. Pairwise Sharpe significance: Jobson-Korkie or block bootstrap
    6d. Export:
          summary_table.csv   machine-readable
          summary_table.tex   LaTeX table for paper
          ml_comparison.pdf   grouped bar-chart figure
    """
    _sep("STAGE 6 . CROSS-BAR REPORT  -- implement in mlfinlab/reporting/")

    if not all_results:
        log.info("  No results to report.")
        return

    rows = []
    for name, res in all_results.items():
        m = res.get("meta", {})
        rows.append({
            "dataset"      : name,
            "bar_type"     : m.get("bar_type",  "?"),
            "source"       : m.get("source",    "?"),
            "bar_class"    : m.get("bar_class", "?"),
            "feature_mode" : res.get("feature_mode", "?"),
            "n_bars"       : res.get("bars",    0),
            "n_events"     : res.get("events",  0),
            "min_d"        : res.get("min_d",   float("nan")),
            "status"       : res.get("status",  "ok"),
        })

    df = pd.DataFrame(rows).set_index("dataset")
    log.info("\n%s", df.to_string())

    if not no_save:
        p = OUTPUT_DIR / f"cross_bar_summary__{_ts}.csv"
        df.to_csv(p)
        log.info("  Saved -> %s", p.name)

    log.info("  [ full metrics table -- implement in mlfinlab/reporting/ ]")


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
    p.add_argument("--file", type=str, default=None,
                   help="Single CSV filename, e.g. binance_btc_1h.csv")
    p.add_argument("--bar-type", type=str, default=None,
                   choices=list(BAR_CATALOGUE.keys()),
                   help="Run both minute + tick for one bar type")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--feature-mode", type=str, default="unified",
                   choices=list(FEATURE_MODES.keys()),
                   help="Feature set to use (default: unified)")
    p.add_argument("--stage", type=str, default=None,
                   choices=[s[0] for s in STAGES if s[0] != "data"],
                   help="Run only one stage (data always runs first)")
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing output CSVs")
    p.add_argument("--list-stages", action="store_true",
                   help="Print stages and exit")
    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = _parse_args()

    if args.list_stages:
        print("\nRegistered pipeline stages:")
        active = {"data", "labeling"}
        for i, (key, label, _) in enumerate(STAGES, 1):
            s = "active" if key in active else "placeholder"
            print(f"  {i}. {key:<10}  {label:<35}  [{s}]")
        print(f"\nFeature modes: {', '.join(FEATURE_MODES)}")
        print()
        return

    log.info("=" * 66)
    log.info("  mlfinlab . bar-comparison pipeline . %s", _ts)
    log.info("  Feature mode : %s", args.feature_mode)
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
            model_output: Optional[pd.DataFrame] = None
            if args.stage in (None, "models") and ml_frame is not None:
                model_output = stage_models(name, ml_frame, meta,
                                            no_save=args.no_save)

            # Stage 4
            signals: Optional[pd.DataFrame] = None
            if args.stage in (None, "predict") and ml_frame is not None:
                signals = stage_predict(name, ml_frame, model_output, meta,
                                        no_save=args.no_save)

            # Stage 5
            if args.stage in (None, "backtest"):
                bt = stage_backtest(name, bars, signals, meta,
                                    no_save=args.no_save)
                if bt:
                    run_results[name].update(bt)

        except Exception:
            run_results[name]["status"] = "ERROR"
            log.error("  ERROR in %s:\n%s", name, traceback.format_exc())

    # Stage 6 -- once, after all datasets
    if args.stage in (None, "report"):
        stage_report(run_results, no_save=args.no_save)

    _print_summary(run_results)
    log.info("")
    log.info("Outputs -> %s", OUTPUT_DIR)
    log.info("Logs    -> %s", LOG_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()