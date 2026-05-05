import numpy as np
import pandas as pd
from typing import Dict, Any

from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    LEDGER_ENTRY_DATETIME_COL,
    LEDGER_EXIT_DATETIME_COL,
    LEDGER_ACTION_COL,
    ENTRY_REGIME_LABEL_COL,
    ENTRY_REGIME_TREND_COL,
    ENTRY_REGIME_VOLATILITY_COL,
)

_DURATION_BINS = [1, 2, 4, 8, 12, 24]      # upper bounds of each bucket (hours)
_DURATION_BUCKET_LABELS = ['<1h', '1-2h', '2-4h', '4-8h', '8-12h', '12-24h', '>24h']


def compute_trade_duration_analysis(ledger: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse trade hold duration and its relationship to performance and regime conditions.

    Returns
    -------
    {
      'by_regime_label': {label: {trade_count, avg_duration_hours, avg_return_pct, win_rate_pct}},
      'duration_distribution': {label: {bucket: count}},
      'performance_by_duration_bucket': {label: {bucket: {trade_count, avg_return_pct}}},
      'by_action_duration': {action: {trade_count, avg_duration_hours, avg_return_pct}},
      'by_trend_hold_type': {trend: {'short_hold'|'long_hold': {trade_count, avg_duration_hours, avg_return_pct, win_rate_pct}}},
      'by_volatility_hold_type': {volatility: {'short_hold'|'long_hold': {...}}},
    }
    """
    entry_dt = pd.to_datetime(ledger[LEDGER_ENTRY_DATETIME_COL].values)
    exit_dt = pd.to_datetime(ledger[LEDGER_EXIT_DATETIME_COL].values)
    duration_hours = (exit_dt.astype('int64') - entry_dt.astype('int64')).astype(float) / (3600 * 1e9)

    returns = pd.to_numeric(ledger[LEDGER_PNL_COL], errors='coerce').values
    actions = ledger[LEDGER_ACTION_COL].values.astype(str)
    regime_labels = ledger[ENTRY_REGIME_LABEL_COL].values.astype(str)
    regime_trends = (ledger[ENTRY_REGIME_TREND_COL].values.astype(str)
                     if ENTRY_REGIME_TREND_COL in ledger.columns else None)
    regime_vols = (ledger[ENTRY_REGIME_VOLATILITY_COL].values.astype(str)
                   if ENTRY_REGIME_VOLATILITY_COL in ledger.columns else None)

    valid_ret = ~np.isnan(returns)
    valid_dur = ~np.isnan(duration_hours) & (duration_hours >= 0)
    both_valid = valid_ret & valid_dur

    unique_labels, label_ids = np.unique(regime_labels, return_inverse=True)
    n_labels = len(unique_labels)
    n_buckets = len(_DURATION_BUCKET_LABELS)

    # Arrays filtered to rows where both duration and return are valid
    lbl_b = label_ids[both_valid]
    r_b = returns[both_valid]
    d_b = duration_hours[both_valid]
    act_b = actions[both_valid]

    # -----------------------------------------------------------------------
    # 1. By regime label — duration only (return/win_rate already in performance_by_regime_label)
    # -----------------------------------------------------------------------
    counts = np.bincount(lbl_b, minlength=n_labels)
    d_sums = np.bincount(lbl_b, weights=d_b, minlength=n_labels)

    by_label: Dict[str, Any] = {}
    for i, label in enumerate(unique_labels):
        cnt = int(counts[i])
        if cnt == 0:
            continue
        by_label[str(label)] = {
            'trade_count': cnt,
            'avg_duration_hours': round(float(d_sums[i]) / cnt, 4),
        }

    # -----------------------------------------------------------------------
    # 2. Duration distribution buckets (count only — valid_duration, not requiring return)
    # -----------------------------------------------------------------------
    lbl_d = label_ids[valid_dur]
    d_d = duration_hours[valid_dur]
    bucket_ids_d = np.clip(np.searchsorted(_DURATION_BINS, d_d, side='right'), 0, n_buckets - 1)
    flat_d = lbl_d * n_buckets + bucket_ids_d
    flat_counts_d = np.bincount(flat_d, minlength=n_labels * n_buckets).reshape(n_labels, n_buckets)

    duration_distribution: Dict[str, Any] = {
        str(unique_labels[i]): {
            _DURATION_BUCKET_LABELS[j]: int(flat_counts_d[i, j])
            for j in range(n_buckets)
        }
        for i in range(n_labels)
    }

    # -----------------------------------------------------------------------
    # 3. Performance per duration bucket — NaN returns properly excluded
    # -----------------------------------------------------------------------
    bucket_ids_b = np.clip(np.searchsorted(_DURATION_BINS, d_b, side='right'), 0, n_buckets - 1)
    flat_b = lbl_b * n_buckets + bucket_ids_b
    flat_r_counts = np.bincount(flat_b, minlength=n_labels * n_buckets).reshape(n_labels, n_buckets)
    flat_r_sums = np.bincount(flat_b, weights=r_b, minlength=n_labels * n_buckets).reshape(n_labels, n_buckets)

    perf_by_bucket: Dict[str, Any] = {}
    for i, label in enumerate(unique_labels):
        label_out: Dict[str, Any] = {}
        for j, bucket in enumerate(_DURATION_BUCKET_LABELS):
            cnt = int(flat_r_counts[i, j])
            if cnt == 0:
                continue
            label_out[bucket] = {
                'trade_count': cnt,
                'avg_return_pct': round(float(flat_r_sums[i, j]) / cnt, 4),
            }
        if label_out:
            perf_by_bucket[str(label)] = label_out

    # -----------------------------------------------------------------------
    # 4. By exit action
    # -----------------------------------------------------------------------
    unique_actions, action_ids = np.unique(act_b, return_inverse=True)
    n_actions = len(unique_actions)
    act_counts = np.bincount(action_ids, minlength=n_actions)
    act_r_sums = np.bincount(action_ids, weights=r_b, minlength=n_actions)
    act_d_sums = np.bincount(action_ids, weights=d_b, minlength=n_actions)

    by_action: Dict[str, Any] = {}
    for i, act in enumerate(unique_actions):
        cnt = int(act_counts[i])
        if cnt == 0:
            continue
        by_action[str(act)] = {
            'trade_count': cnt,
            'avg_duration_hours': round(float(act_d_sums[i]) / cnt, 4),
            'avg_return_pct': round(float(act_r_sums[i]) / cnt, 4),
        }

    # -----------------------------------------------------------------------
    # Helper: split a regime dimension into short_hold / long_hold buckets
    # using global median duration across all valid trades as the split point
    # -----------------------------------------------------------------------
    global_median = float(np.median(d_b)) if len(d_b) > 0 else 0.0

    def _hold_split(dim_arr: np.ndarray) -> Dict[str, Any]:
        dim_b = dim_arr[both_valid].astype(str)
        unique_dims, dim_ids = np.unique(dim_b, return_inverse=True)
        n_dims = len(unique_dims)
        hold_ids = (d_b >= global_median).astype(int)  # 0=short_hold, 1=long_hold
        combined = dim_ids * 2 + hold_ids
        n_comb = n_dims * 2

        c = np.bincount(combined, minlength=n_comb)
        ds = np.bincount(combined, weights=d_b, minlength=n_comb)

        out: Dict[str, Any] = {}
        hold_labels = ['short_hold', 'long_hold']
        for di, dim in enumerate(unique_dims):
            dim_out: Dict[str, Any] = {}
            for hi, hold_label in enumerate(hold_labels):
                idx = di * 2 + hi
                cnt = int(c[idx])
                if cnt == 0:
                    continue
                dim_out[hold_label] = {
                    'trade_count': cnt,
                    'avg_duration_hours': round(float(ds[idx]) / cnt, 4),
                }
            if dim_out:
                out[str(dim)] = dim_out
        return out

    # -----------------------------------------------------------------------
    # 5. By regime trend × hold type (generic — all unique trend values)
    # -----------------------------------------------------------------------
    by_trend_hold: Dict[str, Any] = {}
    if regime_trends is not None:
        by_trend_hold = _hold_split(regime_trends)

    # -----------------------------------------------------------------------
    # 6. By regime volatility × hold type (generic — all unique vol values)
    # -----------------------------------------------------------------------
    by_vol_hold: Dict[str, Any] = {}
    if regime_vols is not None:
        by_vol_hold = _hold_split(regime_vols)

    return {
        'by_regime_label': by_label,
        'duration_distribution': duration_distribution,
        'performance_by_duration_bucket': perf_by_bucket,
        'by_action_duration': by_action,
        'by_trend_hold_type': by_trend_hold,
        'by_volatility_hold_type': by_vol_hold,
    }


def print_trade_duration_analysis(results: dict) -> None:
    import json
    print("\n=== TRADE DURATION ANALYSIS ===")
    print(json.dumps(results, indent=2, default=str))
