import numpy as np
import pandas as pd
from typing import Dict

from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    ENTRY_REGIME_LABEL_COL,
    EXIT_REGIME_LABEL_COL,
)


def compute_regime_transition_matrix(ledger: pd.DataFrame) -> Dict[str, Dict[str, Dict]]:
    """
    Build entry_regime → exit_regime transition matrix.
    Only non-zero transitions are stored (missing = 0).

    Returns
    -------
    {
      'count':         {entry_label: {exit_label: int}},
      'avg_return_pct': {entry_label: {exit_label: float}},
      'win_rate_pct':  {entry_label: {exit_label: float}},
    }
    All numeric values rounded to 4 decimals.
    """
    for col in (ENTRY_REGIME_LABEL_COL, EXIT_REGIME_LABEL_COL, LEDGER_PNL_COL):
        if col not in ledger.columns:
            raise ValueError(f"Missing required column: {col}")

    entry_regimes = ledger[ENTRY_REGIME_LABEL_COL].values.astype(str)
    exit_regimes = ledger[EXIT_REGIME_LABEL_COL].values.astype(str)
    returns = ledger[LEDGER_PNL_COL].values.astype(float)

    # Sorted unique labels — enables searchsorted (O(n log m) vs O(n) dict lookup)
    all_labels = np.unique(np.concatenate([entry_regimes, exit_regimes]))
    n_labels = len(all_labels)

    entry_idx = np.searchsorted(all_labels, entry_regimes)
    exit_idx = np.searchsorted(all_labels, exit_regimes)
    flat_idx = entry_idx * n_labels + exit_idx
    flat_size = n_labels * n_labels

    # Vectorised aggregation
    count_flat = np.bincount(flat_idx, minlength=flat_size)
    return_sum_flat = np.bincount(flat_idx, weights=returns, minlength=flat_size)
    win_sum_flat = np.bincount(flat_idx, weights=(returns > 0).astype(float), minlength=flat_size)

    count_matrix = count_flat.reshape(n_labels, n_labels)
    return_sum_matrix = return_sum_flat.reshape(n_labels, n_labels)
    win_sum_matrix = win_sum_flat.reshape(n_labels, n_labels)

    # Only iterate non-zero cells
    nz_i, nz_j = np.where(count_matrix > 0)

    counts = count_matrix[nz_i, nz_j].astype(float)
    avg_returns = np.round(return_sum_matrix[nz_i, nz_j] / counts, 4)
    win_rates = np.round((win_sum_matrix[nz_i, nz_j] / counts) * 100.0, 4)

    count_dict: Dict[str, Dict] = {}
    avg_return_dict: Dict[str, Dict] = {}
    win_rate_dict: Dict[str, Dict] = {}

    for k in range(len(nz_i)):
        entry_label = str(all_labels[nz_i[k]])
        exit_label = str(all_labels[nz_j[k]])

        if entry_label not in count_dict:
            count_dict[entry_label] = {}
            avg_return_dict[entry_label] = {}
            win_rate_dict[entry_label] = {}

        count_dict[entry_label][exit_label] = int(counts[k])
        avg_return_dict[entry_label][exit_label] = float(avg_returns[k])
        win_rate_dict[entry_label][exit_label] = float(win_rates[k])

    return {
        'count': count_dict,
        'avg_return_pct': avg_return_dict,
        'win_rate_pct': win_rate_dict,
    }


def print_transition_matrix(transition: dict) -> None:
    for name, data in transition.items():
        print(f"\n=== {name.upper()} ===")
        df = pd.DataFrame(data).T.fillna(0)
        print(df.round(4))
