import numpy as np
import pandas as pd
from typing import Dict

from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    LEDGER_ACTION_COL,
    LEDGER_DIRECTION_COL,
    EXIT_REGIME_LABEL_COL,
    EXIT_ACTION_SL_PREFIX,
    EXIT_ACTION_TP_PREFIX,
)


def compute_exit_type_regime_breakdown(ledger: pd.DataFrame) -> Dict[str, Dict]:
    """
    Exit type breakdown per exit-regime and per exit-regime × direction.
    Action categorisation is generic: SL* = stop loss, TP* = take profit, else = other.

    Returns
    -------
    {
      'by_regime': {
          regime: {
              trade_count, pct_SL, pct_TP, pct_other, TP_SL_ratio,
              action_breakdown: {action_value: {count, pct}}
          }
      },
      'by_regime_direction': {
          regime: {
              direction: {trade_count, SL_rate, TP_rate, TP_SL_ratio}
          }
      }
    }
    """
    regimes = ledger[EXIT_REGIME_LABEL_COL].values.astype(str)
    actions = ledger[LEDGER_ACTION_COL].values.astype(str)
    directions = ledger[LEDGER_DIRECTION_COL].values.astype(str)

    # --- Vectorised prefix-based action categorisation ---
    actions_s = pd.Series(actions)
    sl_mask = actions_s.str.startswith(EXIT_ACTION_SL_PREFIX).values
    tp_mask = actions_s.str.startswith(EXIT_ACTION_TP_PREFIX).values
    other_mask = ~sl_mask & ~tp_mask

    unique_regimes, regime_ids = np.unique(regimes, return_inverse=True)
    n_regimes = len(unique_regimes)

    # --- Per-regime aggregation ---
    total_per_regime = np.bincount(regime_ids, minlength=n_regimes)
    sl_per_regime = np.bincount(regime_ids[sl_mask], minlength=n_regimes)
    tp_per_regime = np.bincount(regime_ids[tp_mask], minlength=n_regimes)
    other_per_regime = np.bincount(regime_ids[other_mask], minlength=n_regimes)

    with np.errstate(divide='ignore', invalid='ignore'):
        pct_sl = np.divide(sl_per_regime * 100.0, total_per_regime,
                           out=np.zeros(n_regimes), where=total_per_regime > 0)
        pct_tp = np.divide(tp_per_regime * 100.0, total_per_regime,
                           out=np.zeros(n_regimes), where=total_per_regime > 0)
        pct_other = np.divide(other_per_regime * 100.0, total_per_regime,
                              out=np.zeros(n_regimes), where=total_per_regime > 0)

    tp_sl_ratio = np.where(
        sl_per_regime > 0,
        tp_per_regime / sl_per_regime.astype(float),
        0.0
    )

    # --- Per-regime specific action breakdown (fully vectorised) ---
    unique_actions, action_ids = np.unique(actions, return_inverse=True)
    n_actions = len(unique_actions)
    flat_ra = regime_ids * n_actions + action_ids
    count_ra = np.bincount(flat_ra, minlength=n_regimes * n_actions).reshape(n_regimes, n_actions)

    # --- Per-regime × direction (vectorised combined id) ---
    unique_dirs, dir_ids = np.unique(directions, return_inverse=True)
    n_dirs = len(unique_dirs)

    combined_id = regime_ids * n_dirs + dir_ids
    unique_combined, combined_inv = np.unique(combined_id, return_inverse=True)
    n_comb = len(unique_combined)

    total_comb = np.bincount(combined_inv, minlength=n_comb)
    sl_comb = np.bincount(combined_inv[sl_mask], minlength=n_comb)
    tp_comb = np.bincount(combined_inv[tp_mask], minlength=n_comb)

    with np.errstate(divide='ignore', invalid='ignore'):
        sl_rate_comb = np.divide(sl_comb, total_comb, out=np.zeros(n_comb), where=total_comb > 0)
        tp_rate_comb = np.divide(tp_comb, total_comb, out=np.zeros(n_comb), where=total_comb > 0)

    tp_sl_ratio_comb = np.where(sl_comb > 0, tp_comb / sl_comb.astype(float), 0.0)

    regime_of_comb = unique_combined // n_dirs
    dir_of_comb = unique_combined % n_dirs

    # --- Assemble by_regime ---
    by_regime: Dict[str, Dict] = {}
    for i in range(n_regimes):
        if total_per_regime[i] == 0:
            continue
        reg = str(unique_regimes[i])
        total = int(total_per_regime[i])

        # Specific action breakdown — only non-zero actions
        action_breakdown = {}
        for j in range(n_actions):
            cnt = int(count_ra[i, j])
            if cnt > 0:
                action_breakdown[str(unique_actions[j])] = {
                    'count': cnt,
                    'pct': round(cnt / total * 100.0, 4),
                }

        by_regime[reg] = {
            'trade_count': total,
            'SL_pct': round(float(pct_sl[i]), 4),
            'TP_pct': round(float(pct_tp[i]), 4),
            'other_pct': round(float(pct_other[i]), 4),
            'TP_SL_ratio': round(float(tp_sl_ratio[i]), 4),
            'action_breakdown': action_breakdown,
        }

    # --- Assemble by_regime_direction ---
    by_regime_direction: Dict[str, Dict] = {}
    for k in range(n_comb):
        if total_comb[k] == 0:
            continue
        reg = str(unique_regimes[regime_of_comb[k]])
        direc = str(unique_dirs[dir_of_comb[k]])
        if reg not in by_regime_direction:
            by_regime_direction[reg] = {}
        by_regime_direction[reg][direc] = {
            'trade_count': int(total_comb[k]),
            'SL_rate': round(float(sl_rate_comb[k]), 4),
            'TP_rate': round(float(tp_rate_comb[k]), 4),
            'TP_SL_ratio': round(float(tp_sl_ratio_comb[k]), 4),
        }

    return {
        'by_regime': by_regime,
        'by_regime_direction': by_regime_direction,
    }


def print_exit_type_analysis(result: dict) -> None:
    print("\n=== EXIT TYPE BREAKDOWN BY REGIME ===")
    for regime, data in result['by_regime'].items():
        print(f"{regime}: {data}")
    print("\n=== EXIT TYPE BREAKDOWN BY REGIME × DIRECTION ===")
    for regime, dirs in result['by_regime_direction'].items():
        for direction, data in dirs.items():
            print(f"{regime} | {direction}: {data}")
