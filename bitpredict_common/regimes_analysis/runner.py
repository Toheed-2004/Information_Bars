"""
runner.py — Single entry point for regimes analysis.

Usage:
    from bitpredict.common.regimes_analysis.runner import run_regimes_analysis
    result = run_regimes_analysis(df_ohlcv, df_ledger)

Handles:
    1. Mapping regime columns from OHLCV bars onto each trade (entry + exit)
    2. Running all analysis modules on the enriched ledger
    3. Returning a fully serialisable dict ready for DB storage
"""

import pandas as pd
import numpy as np

from bitpredict.common.regimes_analysis.performance_by_regime_label import calculate_regime_performance
from bitpredict.common.regimes_analysis.regime_transition import compute_regime_transition_matrix
from bitpredict.common.regimes_analysis.exit_regime_breakdown import compute_exit_type_regime_breakdown
from bitpredict.common.regimes_analysis.continuous_metric_quartile import compute_metric_quartile_performance
from bitpredict.common.regimes_analysis.score_threshold_analysis import compute_score_thresholds
from bitpredict.common.regimes_analysis.regime_confidence_analysis import compute_confidence_analysis
from bitpredict.common.regimes_analysis.transition_pressure_analysis import compute_transition_pressure_analysis
from bitpredict.common.regimes_analysis.volatility_asymmetry_analysis import compute_volatility_asymmetry_analysis
from bitpredict.common.regimes_analysis.directional_persistence_analysis import compute_directional_persistence_analysis
from bitpredict.common.regimes_analysis.rolling_regime_performance import compute_rolling_regime_performance, get_latest_rolling_performance
from bitpredict.common.regimes_analysis.trade_duration_analysis import compute_trade_duration_analysis
from bitpredict.common.regimes_analysis.regime_fitness_score import compute_regime_fitness
from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    LEDGER_ENTRY_DATETIME_COL,
    LEDGER_EXIT_DATETIME_COL,
    LEDGER_STATUS_COL,
    LEDGER_OPEN_STATUS,
    OHLCV_DATETIME_COL,
    REGIME_COLUMN_PREFIXES,
    REGIME_EXACT_COLUMNS,
    ENTRY_PREFIX,
    EXIT_PREFIX,
    REGIME_LABEL_COL,
    ROLLING_WINDOW_SIZE,
    ROLLING_MIN_TRADES,
)


# ---------------------------------------------------------------------------
# Step 1: Map regime columns from bars onto each trade
# ---------------------------------------------------------------------------

def map_regimes_to_trades(df_ohlcv: pd.DataFrame, df_ledger: pd.DataFrame) -> pd.DataFrame:
    """
    For each trade's entry and exit datetime, find the last completed bar
    (bar.datetime <= trade.datetime) and copy all regime/score/metric columns
    onto the trade row with entry_* and exit_* prefixes.

    Works for all bar types (time-based and non-time-based) since we only
    use the single bar datetime column.
    """
    bars = df_ohlcv if OHLCV_DATETIME_COL in df_ohlcv.columns else df_ohlcv.reset_index()
    if OHLCV_DATETIME_COL not in bars.columns:
        raise ValueError(f"OHLCV DataFrame must have a '{OHLCV_DATETIME_COL}' column")

    # Detect all regime/score/metric columns
    regime_cols = [
        col for col in bars.columns
        if any(col.startswith(p) for p in REGIME_COLUMN_PREFIXES) or col in REGIME_EXACT_COLUMNS
    ]
    if not regime_cols:
        raise ValueError("No regime/score/metric columns found in df_ohlcv")

    # Normalise bar datetimes → tz-naive int64 nanoseconds
    bars_dt = pd.to_datetime(bars[OHLCV_DATETIME_COL])
    if bars_dt.dt.tz is not None:
        bars_dt = bars_dt.dt.tz_convert('UTC').dt.tz_localize(None)
    bars_ts = bars_dt.astype('int64').values

    # Sort bars for searchsorted
    sort_idx = np.argsort(bars_ts)
    bars_ts_sorted = bars_ts[sort_idx]
    regime_arrays = {col: bars[col].values[sort_idx] for col in regime_cols}

    # Normalise trade datetimes → tz-naive int64 nanoseconds
    def _to_ts(col: str) -> np.ndarray:
        dt = pd.to_datetime(df_ledger[col])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_convert('UTC').dt.tz_localize(None)
        return dt.astype('int64').values

    entry_ts = _to_ts(LEDGER_ENTRY_DATETIME_COL)
    exit_ts = _to_ts(LEDGER_EXIT_DATETIME_COL)

    # Last bar whose datetime <= trade datetime  (side='right' - 1)
    n_bars = len(bars_ts_sorted)
    
    # Handle empty bars case
    if n_bars == 0:
        # No bars available, cannot map regimes
        return pd.DataFrame()
    
    entry_idx = np.clip(np.searchsorted(bars_ts_sorted, entry_ts, side='right') - 1, 0, n_bars - 1)
    exit_idx = np.clip(np.searchsorted(bars_ts_sorted, exit_ts, side='right') - 1, 0, n_bars - 1)

    for col in regime_cols:
        df_ledger[f'{ENTRY_PREFIX}{col}'] = regime_arrays[col][entry_idx]
        df_ledger[f'{EXIT_PREFIX}{col}'] = regime_arrays[col][exit_idx]

    # Drop rows where regime label is missing or invalid
    entry_label_col = ENTRY_PREFIX + REGIME_LABEL_COL
    exit_label_col = EXIT_PREFIX + REGIME_LABEL_COL

    if entry_label_col in df_ledger.columns and exit_label_col in df_ledger.columns:
        e_labels = df_ledger[entry_label_col].values.astype(str)
        x_labels = df_ledger[exit_label_col].values.astype(str)
        bad = {'', 'nan', 'INSUFFICIENT_DATA'}
        valid = (
            pd.notna(df_ledger[entry_label_col].values) &
            pd.notna(df_ledger[exit_label_col].values) &
            ~np.isin(e_labels, list(bad)) &
            ~np.isin(x_labels, list(bad))
        )
        df_ledger = df_ledger[valid].reset_index(drop=True)

    return df_ledger


# ---------------------------------------------------------------------------
# Step 2: Run all analysis modules on the enriched ledger
# ---------------------------------------------------------------------------

def run_regimes_analysis(df_ohlcv: pd.DataFrame, df_ledger: pd.DataFrame) -> dict:
    """
    Full pipeline: map regimes → run all modules → return serialisable dict.

    Parameters
    ----------
    df_ohlcv   : OHLCV + regime columns for the strategy's instrument/bar type
    df_ledger  : Closed trades ledger (open trades are filtered out here)

    Returns
    -------
    dict with all analysis results, fully serialisable (no DataFrames).
    Empty dict if no usable trades after filtering.
    """
    # Filter out open trades
    if LEDGER_STATUS_COL in df_ledger.columns:
        df_ledger = df_ledger[df_ledger[LEDGER_STATUS_COL] != LEDGER_OPEN_STATUS]

    if df_ledger.empty:
        return {}

    # Map regime columns onto trades
    enriched = map_regimes_to_trades(df_ohlcv, df_ledger)
    # enriched_path = os.path.join(os.path.dirname(__file__), "enriched.csv")
    # enriched.to_csv(enriched_path, index=False)

    if enriched.empty:
        return {}

    # Module 1: Performance by regime label / trend / volatility / momentum
    perf = calculate_regime_performance(enriched)
    # Module 2: Regime transition matrix (entry regime → exit regime)
    transition = compute_regime_transition_matrix(enriched)

    # Module 3: Exit type breakdown by regime
    exit_breakdown = compute_exit_type_regime_breakdown(enriched)

    # Module 4: Continuous metric quartile performance
    quartile_perf = compute_metric_quartile_performance(enriched)

    # Module 5: Score threshold analysis
    score_thresholds = compute_score_thresholds(enriched)

    # Module 6: Regime confidence analysis
    confidence = compute_confidence_analysis(enriched)

    # Module 7: Transition pressure analysis
    pressure = compute_transition_pressure_analysis(enriched)

    # Module 8: Volatility asymmetry analysis
    vol_asymmetry = compute_volatility_asymmetry_analysis(enriched)

    # Module 9: Directional persistence analysis
    persistence = compute_directional_persistence_analysis(enriched)

    # Module 10: Rolling regime performance (latest snapshot — serialisable)
    rolling_raw = compute_rolling_regime_performance(
        enriched,
        window_size=ROLLING_WINDOW_SIZE,
        min_trades=ROLLING_MIN_TRADES,
    )
    rolling_latest = get_latest_rolling_performance(rolling_raw, window_size=ROLLING_WINDOW_SIZE)
    rolling_perf = rolling_latest.to_dict(orient='records') if not rolling_latest.empty else []

    # Module 11: Trade duration analysis
    duration = compute_trade_duration_analysis(enriched)

    # Module 12: Regime fitness score
    fitness = {
        'by_regime_label': compute_regime_fitness(perf.get('by_regime_label', {})),
        'by_trend': compute_regime_fitness(perf.get('by_trend', {})),
        'by_volatility': compute_regime_fitness(perf.get('by_volatility', {})),
        'by_momentum': compute_regime_fitness(perf.get('by_momentum', {})),
        'by_direction_trend': compute_regime_fitness(perf.get('by_direction_trend', {})),
    }

    return {
        'performance_per_label': perf,
        'transition_matrix': transition,
        'exit_type_breakdown': exit_breakdown,
        'quartile_performance': quartile_perf,
        'score_thresholds': score_thresholds,
        'confidence_analysis': confidence,
        'transition_pressure': pressure,
        'volatility_asymmetry': vol_asymmetry,
        'directional_persistence': persistence,
        'rolling_performance': rolling_perf,
        'trade_duration': duration,
        'regime_fitness': fitness,
    }

