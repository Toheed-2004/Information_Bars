import numpy as np
import pandas as pd
from typing import Dict, Any

from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    ENTRY_PREFIX,
    CONTINUOUS_METRICS,
)


def compute_metric_quartile_performance(ledger: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    For each continuous metric, bucket trades into quartiles at entry and compute performance.
    Quartile 1 = lowest values, Quartile 4 = highest values.

    Returns
    -------
    {
      metric_name: {
        'quartile_1': {trade_count, avg_return_pct, win_rate_pct, profit_factor},
        ...
        'quartile_4': {...}
      }
    }
    """
    returns = pd.to_numeric(ledger[LEDGER_PNL_COL], errors='coerce').values
    results = {}

    for metric in CONTINUOUS_METRICS:
        col_name = f"{ENTRY_PREFIX}{metric}"
        if col_name not in ledger.columns:
            continue

        metric_arr = pd.to_numeric(ledger[col_name], errors='coerce').values

        # Combined valid mask: both metric and return must be non-NaN
        valid = ~np.isnan(metric_arr) & ~np.isnan(returns)
        if np.sum(valid) < 4:
            continue

        m = metric_arr[valid]
        r = returns[valid]

        # Quartile edges (25th, 50th, 75th percentiles as inner boundaries)
        edges = np.percentile(m, [25, 50, 75])
        if len(np.unique(edges)) < 3:
            # Not enough variation to form meaningful quartiles
            continue

        # Assign quartiles 1..4 via searchsorted (O(n log n), faster than digitize for sorted edges)
        quartile_arr = np.searchsorted(edges, m, side='left') + 1  # 1,2,3,4
        quartile_arr = np.clip(quartile_arr, 1, 4)

        # Vectorised aggregation
        trade_counts = np.bincount(quartile_arr, minlength=5)[1:]
        return_sums = np.bincount(quartile_arr, weights=r, minlength=5)[1:]
        win_counts = np.bincount(quartile_arr, weights=(r > 0).astype(float), minlength=5)[1:]
        profit_sums = np.bincount(quartile_arr, weights=np.where(r > 0, r, 0.0), minlength=5)[1:]
        loss_sums = np.bincount(quartile_arr, weights=np.where(r < 0, -r, 0.0), minlength=5)[1:]

        if np.sum(trade_counts > 0) < 2:
            continue

        metric_dict = {}
        for q in range(1, 5):
            cnt = int(trade_counts[q - 1])
            if cnt == 0:
                continue
            avg_r = float(return_sums[q - 1]) / cnt
            win_rate = (float(win_counts[q - 1]) / cnt) * 100.0
            pf = (float(profit_sums[q - 1]) / float(loss_sums[q - 1])
                  if loss_sums[q - 1] > 0 else 0.0)
            metric_dict[f'quartile_{q}'] = {
                'trade_count': cnt,
                'avg_return_pct': round(avg_r, 4),
                'win_rate_pct': round(win_rate, 4),
                'profit_factor': round(pf, 4),
            }

        results[metric] = metric_dict

    return results


def print_quartile_analysis(results: dict) -> None:
    for metric, quartiles in results.items():
        print(f"\n=== {metric.upper()} QUARTILE PERFORMANCE ===")
        for q, data in quartiles.items():
            print(f"  {q}: {data}")
