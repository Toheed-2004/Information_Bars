import numpy as np
import pandas as pd
from typing import Dict, Any

from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    LEDGER_ACTION_COL,
    ENTRY_PREFIX,
    EXIT_PREFIX,
    ENTRY_REGIME_LABEL_COL,
    EXIT_REGIME_LABEL_COL,
    EXIT_ACTION_SL_PREFIX,
    EXIT_ACTION_TP_PREFIX,
)

_PRESSURE_COL = 'transition_pressure'
_ENTRY_PRESSURE_COL = ENTRY_PREFIX + _PRESSURE_COL
_EXIT_PRESSURE_COL = EXIT_PREFIX + _PRESSURE_COL


def compute_transition_pressure_analysis(ledger: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse how transition pressure at entry/exit relates to trade outcomes.

    Returns
    -------
    {
      'entry_pressure_quartiles': {
          'quartile_1': {trade_count, avg_return_pct, win_rate_pct, profit_factor, SL_rate_pct},
          ...
      },
      'exit_pressure_by_action': {
          action_value: {trade_count, avg_exit_pressure}
      },
      'regime_change_pressure': {
          trade_count, avg_entry_pressure, avg_exit_pressure, avg_return_pct, win_rate_pct
      }
    }
    """
    returns = pd.to_numeric(ledger[LEDGER_PNL_COL], errors='coerce').values
    actions = ledger[LEDGER_ACTION_COL].values.astype(str)
    entry_regimes = ledger[ENTRY_REGIME_LABEL_COL].values.astype(str)
    exit_regimes = ledger[EXIT_REGIME_LABEL_COL].values.astype(str)

    entry_pressure = (pd.to_numeric(ledger[_ENTRY_PRESSURE_COL], errors='coerce').values
                      if _ENTRY_PRESSURE_COL in ledger.columns else np.full(len(returns), np.nan))
    exit_pressure = (pd.to_numeric(ledger[_EXIT_PRESSURE_COL], errors='coerce').values
                     if _EXIT_PRESSURE_COL in ledger.columns else np.full(len(returns), np.nan))

    is_sl = pd.Series(actions).str.startswith(EXIT_ACTION_SL_PREFIX).values.astype(float)
    is_tp = pd.Series(actions).str.startswith(EXIT_ACTION_TP_PREFIX).values.astype(float)

    # -----------------------------------------------------------------------
    # 1. Entry pressure quartiles — performance + SL rate
    # -----------------------------------------------------------------------
    entry_pressure_quartiles: Dict[str, Dict] = {}
    valid_entry = ~np.isnan(entry_pressure) & ~np.isnan(returns)

    if np.sum(valid_entry) >= 8:
        ep = entry_pressure[valid_entry]
        r = returns[valid_entry]
        sl = is_sl[valid_entry]
        tp = is_tp[valid_entry]

        edges = np.percentile(ep, [25, 50, 75])
        if len(np.unique(edges)) >= 3:
            q_ids = np.searchsorted(edges, ep, side='left') + 1
            q_ids = np.clip(q_ids, 1, 4)

            counts = np.bincount(q_ids, minlength=5)[1:]
            r_sums = np.bincount(q_ids, weights=r, minlength=5)[1:]
            w_sums = np.bincount(q_ids, weights=(r > 0).astype(float), minlength=5)[1:]
            p_sums = np.bincount(q_ids, weights=np.where(r > 0, r, 0.0), minlength=5)[1:]
            l_sums = np.bincount(q_ids, weights=np.where(r < 0, -r, 0.0), minlength=5)[1:]
            sl_sums = np.bincount(q_ids, weights=sl, minlength=5)[1:]
            tp_sums = np.bincount(q_ids, weights=tp, minlength=5)[1:]

            for q in range(1, 5):
                cnt = int(counts[q - 1])
                if cnt == 0:
                    continue
                pf = (float(p_sums[q - 1]) / float(l_sums[q - 1])
                      if l_sums[q - 1] > 0 else 0.0)
                entry_pressure_quartiles[f'quartile_{q}'] = {
                    'trade_count': cnt,
                    'avg_return_pct': round(float(r_sums[q - 1]) / cnt, 4),
                    'win_rate_pct': round((float(w_sums[q - 1]) / cnt) * 100.0, 4),
                    'profit_factor': round(pf, 4),
                    'SL_rate_pct': round((float(sl_sums[q - 1]) / cnt) * 100.0, 4),
                    'TP_rate_pct': round((float(tp_sums[q - 1]) / cnt) * 100.0, 4),
                }

    # -----------------------------------------------------------------------
    # 2. Exit pressure by action — vectorised
    # -----------------------------------------------------------------------
    valid_exit = ~np.isnan(exit_pressure)
    exit_pressure_by_action: Dict[str, Dict] = {}

    if np.any(valid_exit):
        actions_valid = actions[valid_exit]
        ep_valid = exit_pressure[valid_exit]
        unique_actions, action_ids = np.unique(actions_valid, return_inverse=True)
        n_actions = len(unique_actions)

        act_counts = np.bincount(action_ids, minlength=n_actions)
        ep_sums = np.bincount(action_ids, weights=ep_valid, minlength=n_actions)
        avg_ep = np.divide(ep_sums, act_counts, out=np.zeros(n_actions), where=act_counts > 0)

        for i, act in enumerate(unique_actions):
            if act_counts[i] > 0:
                exit_pressure_by_action[str(act)] = {
                    'trade_count': int(act_counts[i]),
                    'avg_exit_pressure': round(float(avg_ep[i]), 4),
                }

    # -----------------------------------------------------------------------
    # 3. Regime-change trades pressure
    # -----------------------------------------------------------------------
    regime_change_mask = (entry_regimes != exit_regimes) & ~np.isnan(returns)
    regime_change_pressure: Dict[str, Any] = {}

    if np.any(regime_change_mask):
        rc_returns = returns[regime_change_mask]
        rc_ep = entry_pressure[regime_change_mask]
        rc_xp = exit_pressure[regime_change_mask]
        cnt = int(np.sum(regime_change_mask))

        avg_ep_rc = round(float(np.nanmean(rc_ep)), 4) if not np.all(np.isnan(rc_ep)) else None
        avg_xp_rc = round(float(np.nanmean(rc_xp)), 4) if not np.all(np.isnan(rc_xp)) else None

        regime_change_pressure = {
            'trade_count': cnt,
            'avg_entry_pressure': avg_ep_rc,
            'avg_exit_pressure': avg_xp_rc,
            'avg_return_pct': round(float(np.mean(rc_returns)), 4),
            'win_rate_pct': round(float(np.mean(rc_returns > 0) * 100.0), 4),
        }
    else:
        regime_change_pressure = {
            'trade_count': 0,
            'avg_entry_pressure': None,
            'avg_exit_pressure': None,
            'avg_return_pct': None,
            'win_rate_pct': None,
        }

    return {
        'entry_pressure_quartiles': entry_pressure_quartiles,
        'exit_pressure_by_action': exit_pressure_by_action,
        'regime_change_pressure': regime_change_pressure,
    }


def print_transition_pressure_analysis(results: dict) -> None:
    import json
    print("\n=== TRANSITION PRESSURE ANALYSIS ===")
    print(json.dumps(results, indent=2, default=str))
