import numpy as np
import pandas as pd
import numpy as np
from typing import Dict

from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    LEDGER_EXIT_DATETIME_COL,
    ENTRY_REGIME_LABEL_COL,
    ROLLING_WINDOW_SIZE,
    ROLLING_MIN_TRADES,
)


def compute_rolling_regime_performance(
    ledger: pd.DataFrame,
    window_size: int = ROLLING_WINDOW_SIZE,
    min_trades: int = ROLLING_MIN_TRADES,
) -> Dict[str, pd.DataFrame]:
    """
    Compute rolling performance metrics within each regime label.

    Uses O(n) cumsum trick — no Python loops over windows.

    Returns a dict of {regime_label: DataFrame} where each DataFrame has columns:
        datetime, rolling_avg_return_pct, rolling_win_rate_pct, trade_count_in_window
    Only rows where a full window is available are included.
    """
    returns = pd.to_numeric(ledger[LEDGER_PNL_COL], errors='coerce').values
    datetimes = pd.to_datetime(ledger[LEDGER_EXIT_DATETIME_COL].values)
    regimes = ledger[ENTRY_REGIME_LABEL_COL].values.astype(str)

    unique_regimes = np.unique(regimes)
    results: Dict[str, pd.DataFrame] = {}

    for regime in unique_regimes:
        mask = regimes == regime
        r = returns[mask]
        dt = datetimes[mask]

        if len(r) < min_trades:
            continue

        sort_idx = np.argsort(dt)
        r = r[sort_idx]
        dt = dt[sort_idx]

        # Exclude NaN returns from rolling computation
        valid = ~np.isnan(r)
        r_clean = r[valid]
        dt_clean = dt[valid]
        n = len(r_clean)

        if n < window_size:
            continue

        # O(n) cumsum rolling
        cumsum = np.concatenate([[0.0], np.cumsum(r_clean)])
        cumsum_wins = np.concatenate([[0.0], np.cumsum(r_clean > 0)])

        idx = np.arange(window_size - 1, n)
        rolling_avg = (cumsum[idx + 1] - cumsum[idx - window_size + 1]) / window_size
        rolling_win_rate = (
            (cumsum_wins[idx + 1] - cumsum_wins[idx - window_size + 1]) / window_size
        ) * 100.0

        results[str(regime)] = pd.DataFrame({
            'datetime': dt_clean[idx],
            'rolling_avg_return_pct': np.round(rolling_avg, 4),
            'rolling_win_rate_pct': np.round(rolling_win_rate, 4),
            'trade_count_in_window': window_size,  # every row represents exactly one full window
        })

    return results

def get_latest_rolling_performance(
    rolling_results: Dict[str, pd.DataFrame],
    window_size: int = ROLLING_WINDOW_SIZE,
    avg_return_threshold: float = 0.0,
    win_rate_threshold: float = 40.0,
) -> pd.DataFrame:
    """
    Extract the most recent rolling performance snapshot for each regime.

    Returns a DataFrame suitable for `.to_dict(orient='records')` before DB storage.
    Columns: regime, last_datetime, rolling_avg_return_pct, rolling_win_rate_pct,
             trade_count_in_window, window_size, status
    """
    rows = []
    for regime, df in rolling_results.items():
        if df.empty:
            continue
        last = df.iloc[-1]
        avg_r = float(last['rolling_avg_return_pct'])
        wr = float(last['rolling_win_rate_pct'])

        if avg_r < avg_return_threshold and wr < win_rate_threshold:
            status = 'critical'
        elif avg_r < avg_return_threshold:
            status = 'deteriorating'
        elif wr < win_rate_threshold:
            status = 'poor_win_rate'
        else:
            status = 'healthy'

        rows.append({
            'regime': str(regime),
            'last_datetime': str(last['datetime']),
            'rolling_avg_return_pct': round(avg_r, 4),
            'rolling_win_rate_pct': round(wr, 4),
            'trade_count_in_window': int(last['trade_count_in_window']),
            'window_size': window_size,
            'status': status,
        })
    return pd.DataFrame(rows)


def print_rolling_regime_performance(
    rolling_results: Dict[str, pd.DataFrame], top_n: int = 5
) -> None:
    print("\n=== ROLLING REGIME PERFORMANCE ===")
    for regime, df in rolling_results.items():
        print(f"\n  Regime: {regime}  ({len(df)} windows)")
        print(df.tail(top_n).to_string(index=False))
