import numpy as np
import pandas as pd
from typing import Dict, Any

from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    LEDGER_DIRECTION_COL,
    ENTRY_PREFIX,
)

_PERSISTENCE_COL = ENTRY_PREFIX + 'directional_persistence'


def compute_directional_persistence_analysis(ledger: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse how directional persistence at trade entry affects outcomes,
    split by direction and combined.

    Quartile edges are computed globally across all valid trades so that
    Q1–Q4 represent the same persistence ranges across Long, Short, and combined.
    Quartile 1 = lowest persistence, Quartile 4 = highest persistence.

    Returns
    -------
    {
      'by_direction': {
          direction: {
              'quartile_1': {trade_count, avg_return_pct, win_rate_pct, profit_factor},
              ...
          }
      },
      'combined': {
          'quartile_1': {...},
          ...
      }
    }
    """
    returns = pd.to_numeric(ledger[LEDGER_PNL_COL], errors='coerce').values
    directions = ledger[LEDGER_DIRECTION_COL].values.astype(str)
    persistence = (pd.to_numeric(ledger[_PERSISTENCE_COL], errors='coerce').values
                   if _PERSISTENCE_COL in ledger.columns else np.full(len(returns), np.nan))

    valid = ~np.isnan(persistence) & ~np.isnan(returns)
    if np.sum(valid) < 8:
        return {'by_direction': {}, 'combined': {}}

    p = persistence[valid]
    r = returns[valid]
    d = directions[valid]

    # Global quartile edges — consistent across all groups
    edges = np.percentile(p, [25, 50, 75])
    if len(np.unique(edges)) < 3:
        return {'by_direction': {}, 'combined': {}}

    q_ids = np.clip(np.searchsorted(edges, p, side='left') + 1, 1, 4)

    # -----------------------------------------------------------------------
    # Helper: bincount-based stats for a given group array
    # -----------------------------------------------------------------------
    def _quartile_stats(q_arr, r_arr):
        counts = np.bincount(q_arr, minlength=5)[1:]
        r_sums = np.bincount(q_arr, weights=r_arr, minlength=5)[1:]
        w_sums = np.bincount(q_arr, weights=(r_arr > 0).astype(float), minlength=5)[1:]
        p_sums = np.bincount(q_arr, weights=np.where(r_arr > 0, r_arr, 0.0), minlength=5)[1:]
        l_sums = np.bincount(q_arr, weights=np.where(r_arr < 0, -r_arr, 0.0), minlength=5)[1:]
        out = {}
        for q in range(1, 5):
            cnt = int(counts[q - 1])
            if cnt == 0:
                continue
            pf = (round(float(p_sums[q - 1]) / float(l_sums[q - 1]), 4)
                  if l_sums[q - 1] > 0 else 0.0)
            out[f'quartile_{q}'] = {
                'trade_count': cnt,
                'avg_return_pct': round(float(r_sums[q - 1]) / cnt, 4),
                'win_rate_pct': round((float(w_sums[q - 1]) / cnt) * 100.0, 4),
                'profit_factor': pf,
            }
        return out

    # -----------------------------------------------------------------------
    # By direction — vectorised via combined direction×quartile id
    # -----------------------------------------------------------------------
    unique_dirs, dir_ids = np.unique(d, return_inverse=True)
    by_direction: Dict[str, Dict] = {}

    for i, direc in enumerate(unique_dirs):
        mask = (dir_ids == i)
        if np.sum(mask) < 4:
            continue
        by_direction[str(direc)] = _quartile_stats(q_ids[mask], r[mask])

    # -----------------------------------------------------------------------
    # Combined (all trades)
    # -----------------------------------------------------------------------
    combined = _quartile_stats(q_ids, r)

    return {'by_direction': by_direction, 'combined': combined}


def print_directional_persistence_analysis(results: dict) -> None:
    print("\n=== DIRECTIONAL PERSISTENCE ANALYSIS ===")
    print("Q1 = lowest persistence, Q4 = highest persistence")
    for direc, quartiles in results['by_direction'].items():
        print(f"\n  {direc}:")
        for q, data in quartiles.items():
            print(f"    {q}: {data}")
    print("\n  Combined:")
    for q, data in results['combined'].items():
        print(f"    {q}: {data}")
