import numpy as np
import pandas as pd
from typing import Dict, Any

from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    LEDGER_DIRECTION_COL,
    ENTRY_PREFIX,
)

_UP_VOL_COL = ENTRY_PREFIX + 'up_vol'
_DOWN_VOL_COL = ENTRY_PREFIX + 'down_vol'
_SKEW_BINS = [0.999, 1.001]   # boundaries for downside / symmetric / upside skew
_SKEW_LABELS = ['downside_dominant', 'symmetric', 'upside_dominant']
_ADVERSE_VOL_PERCENTILE = 75   # threshold for "high" adverse vol


def compute_volatility_asymmetry_analysis(ledger: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse how volatility asymmetry (up_vol / down_vol skew) affects trade outcomes.

    Returns
    -------
    {
      'skew_by_direction': {
          direction: {
              skew_category: {trade_count, avg_return_pct, win_rate_pct, profit_factor}
          }
      },
      'up_vol_quartiles':   {quartile_1..4: {trade_count, avg_return_pct, win_rate_pct, profit_factor}},
      'down_vol_quartiles': {quartile_1..4: {...}},
      'adverse_vol_risk': {
          'short_high_up_vol':   {trade_count, avg_return_pct, win_rate_pct},
          'short_low_up_vol':    {trade_count, avg_return_pct, win_rate_pct},
          'long_high_down_vol':  {trade_count, avg_return_pct, win_rate_pct},
          'long_low_down_vol':   {trade_count, avg_return_pct, win_rate_pct},
      }
    }
    """
    returns = pd.to_numeric(ledger[LEDGER_PNL_COL], errors='coerce').values
    directions = ledger[LEDGER_DIRECTION_COL].values.astype(str)

    up_vol = (pd.to_numeric(ledger[_UP_VOL_COL], errors='coerce').values
              if _UP_VOL_COL in ledger.columns else np.full(len(returns), np.nan))
    down_vol = (pd.to_numeric(ledger[_DOWN_VOL_COL], errors='coerce').values
                if _DOWN_VOL_COL in ledger.columns else np.full(len(returns), np.nan))

    # Volatility skew: up_vol / down_vol (NaN when either is NaN or down_vol == 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_skew = np.where(
            np.isnan(up_vol) | np.isnan(down_vol) | (down_vol == 0),
            np.nan,
            up_vol / down_vol
        )

    valid_returns = ~np.isnan(returns)

    # -----------------------------------------------------------------------
    # Helper: vectorised stats from pre-filtered arrays
    # -----------------------------------------------------------------------
    def _stats(r):
        cnt = len(r)
        if cnt == 0:
            return None
        avg_r = float(np.mean(r))
        win_rate = float(np.mean(r > 0) * 100.0)
        gross_p = float(np.sum(r[r > 0])) if np.any(r > 0) else 0.0
        gross_l = float(np.sum(-r[r < 0])) if np.any(r < 0) else 0.0
        pf = round(gross_p / gross_l, 4) if gross_l > 0 else 0.0
        return {
            'trade_count': cnt,
            'avg_return_pct': round(avg_r, 4),
            'win_rate_pct': round(win_rate, 4),
            'profit_factor': pf,
        }

    # -----------------------------------------------------------------------
    # 1. Skew performance by direction — vectorised combined-id bincount
    # -----------------------------------------------------------------------
    valid_skew = ~np.isnan(vol_skew) & valid_returns
    skew_by_direction: Dict[str, Dict] = {}

    if np.any(valid_skew):
        s = vol_skew[valid_skew]
        r = returns[valid_skew]
        d = directions[valid_skew]

        skew_ids = np.searchsorted(_SKEW_BINS, s, side='left')   # 0, 1, 2
        n_skew = len(_SKEW_LABELS)
        unique_dirs, dir_ids = np.unique(d, return_inverse=True)
        n_dirs = len(unique_dirs)

        combined = dir_ids * n_skew + skew_ids
        n_comb = n_dirs * n_skew

        counts = np.bincount(combined, minlength=n_comb)
        r_sums = np.bincount(combined, weights=r, minlength=n_comb)
        w_sums = np.bincount(combined, weights=(r > 0).astype(float), minlength=n_comb)
        p_sums = np.bincount(combined, weights=np.where(r > 0, r, 0.0), minlength=n_comb)
        l_sums = np.bincount(combined, weights=np.where(r < 0, -r, 0.0), minlength=n_comb)

        for di, direc in enumerate(unique_dirs):
            skew_by_direction[str(direc)] = {}
            for si, skew_label in enumerate(_SKEW_LABELS):
                idx = di * n_skew + si
                cnt = int(counts[idx])
                if cnt == 0:
                    continue
                pf = (round(float(p_sums[idx]) / float(l_sums[idx]), 4)
                      if l_sums[idx] > 0 else 0.0)
                skew_by_direction[str(direc)][skew_label] = {
                    'trade_count': cnt,
                    'avg_return_pct': round(float(r_sums[idx]) / cnt, 4),
                    'win_rate_pct': round((float(w_sums[idx]) / cnt) * 100.0, 4),
                    'profit_factor': pf,
                }

    # -----------------------------------------------------------------------
    # Helper: quartile performance for a vol series
    # -----------------------------------------------------------------------
    def _vol_quartiles(vol_arr):
        valid = ~np.isnan(vol_arr) & valid_returns
        if np.sum(valid) < 8:
            return {}
        v = vol_arr[valid]
        r = returns[valid]
        edges = np.percentile(v, [25, 50, 75])
        if len(np.unique(edges)) < 3:
            return {}
        q_ids = np.clip(np.searchsorted(edges, v, side='left') + 1, 1, 4)
        counts = np.bincount(q_ids, minlength=5)[1:]
        r_sums = np.bincount(q_ids, weights=r, minlength=5)[1:]
        w_sums = np.bincount(q_ids, weights=(r > 0).astype(float), minlength=5)[1:]
        p_sums = np.bincount(q_ids, weights=np.where(r > 0, r, 0.0), minlength=5)[1:]
        l_sums = np.bincount(q_ids, weights=np.where(r < 0, -r, 0.0), minlength=5)[1:]
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
    # 2 & 3. up_vol / down_vol quartile performance
    # -----------------------------------------------------------------------
    up_vol_quartiles = _vol_quartiles(up_vol)
    down_vol_quartiles = _vol_quartiles(down_vol)

    # -----------------------------------------------------------------------
    # 4. Adverse vol risk
    #    Short trades with high up_vol  → squeeze risk
    #    Long  trades with high down_vol → whipsaw risk
    # -----------------------------------------------------------------------
    adverse_vol_risk: Dict[str, Any] = {}

    def _adverse(vol_arr, dir_filter):
        valid = ~np.isnan(vol_arr) & valid_returns & (directions == dir_filter)
        if np.sum(valid) < 4:
            return
        v = vol_arr[valid]
        r = returns[valid]
        threshold = np.percentile(v, _ADVERSE_VOL_PERCENTILE)
        high_mask = v >= threshold
        low_mask = ~high_mask
        for mask, label in [(high_mask, 'high'), (low_mask, 'low')]:
            s = _stats(r[mask])
            if s:
                key = f"{dir_filter.lower()}_{label}_{('up_vol' if dir_filter == 'Short' else 'down_vol')}"
                adverse_vol_risk[key] = s

    _adverse(up_vol, 'Short')
    _adverse(down_vol, 'Long')

    return {
        'skew_by_direction': skew_by_direction,
        'up_vol_quartiles': up_vol_quartiles,
        'down_vol_quartiles': down_vol_quartiles,
        'adverse_vol_risk': adverse_vol_risk,
    }


def print_volatility_asymmetry_analysis(results: dict) -> None:
    import json
    print("\n=== VOLATILITY ASYMMETRY ANALYSIS ===")
    print(json.dumps(results, indent=2, default=str))
