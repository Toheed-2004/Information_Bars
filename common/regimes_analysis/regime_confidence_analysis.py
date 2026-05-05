import numpy as np
import pandas as pd
from typing import Dict, Any

from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    ENTRY_REGIME_CONFIDENCE_COL,
    ENTRY_REGIME_TREND_COL,
    ENTRY_REGIME_LABEL_COL,
    CONFIDENCE_BUCKET_EDGES,
    CONFIDENCE_BUCKET_LABELS,
)


def compute_confidence_analysis(ledger: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse performance by regime confidence bucket.
    Confidence is bucketed using CONFIDENCE_BUCKET_EDGES from config.
    NaN confidence rows are excluded (not filled).

    Returns
    -------
    {
      'overall':  {bucket_label: {trade_count, avg_return_pct, win_rate_pct, profit_factor}},
      'by_trend': {bucket_label: {trend: {stats}}},
      'by_label': {bucket_label: {regime_label: {stats}}},
    }
    """
    returns = pd.to_numeric(ledger[LEDGER_PNL_COL], errors='coerce').values
    confidence = pd.to_numeric(ledger[ENTRY_REGIME_CONFIDENCE_COL], errors='coerce').values
    regime_trends = ledger[ENTRY_REGIME_TREND_COL].values.astype(str)
    regime_labels = ledger[ENTRY_REGIME_LABEL_COL].values.astype(str)

    # Exclude rows where confidence or return is NaN
    valid = ~np.isnan(confidence) & ~np.isnan(returns)
    confidence = confidence[valid]
    returns = returns[valid]
    regime_trends = regime_trends[valid]
    regime_labels = regime_labels[valid]

    if len(returns) == 0:
        return {'overall': {}, 'by_trend': {}, 'by_label': {}}

    # Assign bucket index 0,1,2 using config edges
    edges = np.array(CONFIDENCE_BUCKET_EDGES)
    bucket_ids = np.searchsorted(edges, confidence, side='left')   # 0, 1, 2
    n_buckets = len(CONFIDENCE_BUCKET_LABELS)

    # -----------------------------------------------------------------------
    # Helper: vectorised stats for a group array + returns
    # -----------------------------------------------------------------------
    def _agg(group_ids, n_groups):
        counts = np.bincount(group_ids, minlength=n_groups)
        r_sums = np.bincount(group_ids, weights=returns, minlength=n_groups)
        w_sums = np.bincount(group_ids, weights=(returns > 0).astype(float), minlength=n_groups)
        p_sums = np.bincount(group_ids, weights=np.where(returns > 0, returns, 0.0), minlength=n_groups)
        l_sums = np.bincount(group_ids, weights=np.where(returns < 0, -returns, 0.0), minlength=n_groups)
        return counts, r_sums, w_sums, p_sums, l_sums

    def _build_stats(cnt, r_sum, w_sum, p_sum, l_sum):
        cnt = int(cnt)
        if cnt == 0:
            return None
        avg_r = round(float(r_sum) / cnt, 4)
        win_rate = round((float(w_sum) / cnt) * 100.0, 4)
        pf = round(float(p_sum) / float(l_sum), 4) if l_sum > 0 else 0.0
        return {'trade_count': cnt, 'avg_return_pct': avg_r, 'win_rate_pct': win_rate, 'profit_factor': pf}

    # -----------------------------------------------------------------------
    # 1. Overall by confidence bucket
    # -----------------------------------------------------------------------
    counts, r_sums, w_sums, p_sums, l_sums = _agg(bucket_ids, n_buckets)
    overall = {}
    for i, label in enumerate(CONFIDENCE_BUCKET_LABELS):
        s = _build_stats(counts[i], r_sums[i], w_sums[i], p_sums[i], l_sums[i])
        if s:
            overall[label] = s

    # -----------------------------------------------------------------------
    # 2. By confidence bucket × regime trend (vectorised combined id)
    # -----------------------------------------------------------------------
    unique_trends, trend_ids = np.unique(regime_trends, return_inverse=True)
    n_trends = len(unique_trends)
    combined_bt = bucket_ids * n_trends + trend_ids
    counts_bt, r_bt, w_bt, p_bt, l_bt = _agg(combined_bt, n_buckets * n_trends)

    by_trend: Dict[str, Dict] = {}
    for i, label in enumerate(CONFIDENCE_BUCKET_LABELS):
        for j, trend in enumerate(unique_trends):
            idx = i * n_trends + j
            s = _build_stats(counts_bt[idx], r_bt[idx], w_bt[idx], p_bt[idx], l_bt[idx])
            if s:
                if label not in by_trend:
                    by_trend[label] = {}
                by_trend[label][str(trend)] = s

    # -----------------------------------------------------------------------
    # 3. By confidence bucket × full regime label (vectorised combined id)
    # -----------------------------------------------------------------------
    unique_labels, label_ids = np.unique(regime_labels, return_inverse=True)
    n_labels = len(unique_labels)
    combined_bl = bucket_ids * n_labels + label_ids
    counts_bl, r_bl, w_bl, p_bl, l_bl = _agg(combined_bl, n_buckets * n_labels)

    by_label: Dict[str, Dict] = {}
    for i, label in enumerate(CONFIDENCE_BUCKET_LABELS):
        for j, reg_label in enumerate(unique_labels):
            idx = i * n_labels + j
            s = _build_stats(counts_bl[idx], r_bl[idx], w_bl[idx], p_bl[idx], l_bl[idx])
            if s:
                if label not in by_label:
                    by_label[label] = {}
                by_label[label][str(reg_label)] = s

    return {'overall': overall, 'by_trend': by_trend, 'by_label': by_label}


def print_confidence_analysis(results: dict) -> None:
    print("\n=== CONFIDENCE ANALYSIS (OVERALL) ===")
    for bucket, data in results['overall'].items():
        print(f"  {bucket}: {data}")
    print("\n=== CONFIDENCE × REGIME TREND ===")
    for bucket, trends in results['by_trend'].items():
        for trend, data in trends.items():
            print(f"  {bucket} | {trend}: {data}")
    print("\n=== CONFIDENCE × REGIME LABEL ===")
    for bucket, labels in results['by_label'].items():
        for lbl, data in labels.items():
            print(f"  {bucket} | {lbl}: {data}")
