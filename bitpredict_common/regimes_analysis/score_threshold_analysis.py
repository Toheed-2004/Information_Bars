import numpy as np
import pandas as pd
from typing import Dict, Any

from bitpredict.common.regimes_analysis.config import (
    LEDGER_PNL_COL,
    ENTRY_PREFIX,
    SCORE_COLUMN_PREFIX,
)

_N_DECILES = 10
_MIN_TRADES_PER_DECILE = 5   # minimum trades in a decile to be considered for optimal threshold


def compute_score_thresholds(ledger: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    For each score column detected in the enriched ledger, bin trades into
    10 equal-width deciles (0→1) and compute performance per decile.

    Score columns are auto-detected by the entry_ + score_ prefix pattern.

    Returns
    -------
    {
      score_name: {
        'deciles': [
            {'decile': 1, 'range_low': 0.0, 'range_high': 0.1,
             'trade_count': int, 'avg_return_pct': float, 'win_rate_pct': float},
            ...
        ],
        'higher_is_better': bool,   # True if upper deciles outperform lower
        'optimal_threshold': float, # lower bound of the best-performing decile
        'best_avg_return_pct': float,
      }
    }
    """
    returns = pd.to_numeric(ledger[LEDGER_PNL_COL], errors='coerce').values

    # Auto-detect score columns: entry_score_*
    score_prefix = ENTRY_PREFIX + SCORE_COLUMN_PREFIX
    score_cols = [col for col in ledger.columns if col.startswith(score_prefix)]

    bin_edges = np.round(np.linspace(0.0, 1.0, _N_DECILES + 1), 4)

    results = {}

    for col in score_cols:
        score_arr = pd.to_numeric(ledger[col], errors='coerce').values

        valid = ~np.isnan(score_arr) & ~np.isnan(returns)
        if np.sum(valid) < _MIN_TRADES_PER_DECILE * 2:
            continue

        s = score_arr[valid]
        r = returns[valid]

        # Assign deciles 1..10 via searchsorted on inner edges
        bin_ids = np.searchsorted(bin_edges[1:-1], s, side='left') + 1
        bin_ids = np.clip(bin_ids, 1, _N_DECILES)

        trade_counts = np.bincount(bin_ids, minlength=_N_DECILES + 1)[1:]
        return_sums = np.bincount(bin_ids, weights=r, minlength=_N_DECILES + 1)[1:]
        win_sums = np.bincount(bin_ids, weights=(r > 0).astype(float), minlength=_N_DECILES + 1)[1:]

        avg_returns = np.divide(
            return_sums, trade_counts,
            out=np.full(_N_DECILES, np.nan), where=trade_counts > 0
        )
        win_rates = np.divide(
            win_sums * 100.0, trade_counts,
            out=np.full(_N_DECILES, np.nan), where=trade_counts > 0
        )

        # Per-decile list
        deciles = []
        for d in range(_N_DECILES):
            cnt = int(trade_counts[d])
            deciles.append({
                'decile': d + 1,
                'range_low': float(bin_edges[d]),
                'range_high': float(bin_edges[d + 1]),
                'trade_count': cnt,
                'avg_return_pct': round(float(avg_returns[d]), 4) if not np.isnan(avg_returns[d]) else None,
                'win_rate_pct': round(float(win_rates[d]), 4) if not np.isnan(win_rates[d]) else None,
            })

        # Optimal threshold: decile with highest avg_return_pct among those with enough trades
        eligible = np.where(trade_counts >= _MIN_TRADES_PER_DECILE)[0]
        if len(eligible) > 0:
            eligible_avg = avg_returns[eligible]
            valid_eligible = ~np.isnan(eligible_avg)
            if np.any(valid_eligible):
                best_local = eligible[valid_eligible][np.argmax(eligible_avg[valid_eligible])]
                optimal_threshold = float(bin_edges[best_local])
                best_avg = round(float(avg_returns[best_local]), 4)
            else:
                optimal_threshold = None
                best_avg = None
        else:
            optimal_threshold = None
            best_avg = None

        # Higher is better: Pearson correlation between decile rank and avg_return_pct
        valid_deciles = ~np.isnan(avg_returns) & (trade_counts >= _MIN_TRADES_PER_DECILE)
        if np.sum(valid_deciles) >= 3:
            ranks = np.arange(1, _N_DECILES + 1)[valid_deciles].astype(float)
            avgs = avg_returns[valid_deciles]
            corr = float(np.corrcoef(ranks, avgs)[0, 1])
            higher_is_better = corr >= 0.0
        else:
            higher_is_better = None

        score_name = col[len(ENTRY_PREFIX):]   # strip entry_ prefix for cleaner key
        results[score_name] = {
            'deciles': deciles,
            'higher_is_better': higher_is_better,
            'optimal_threshold': optimal_threshold,
            'best_avg_return_pct': best_avg,
        }

    return results


def print_score_thresholds(results: dict) -> None:
    for score, data in results.items():
        print(f"\n=== {score.upper()} ===")
        print(f"  Higher is better: {data['higher_is_better']}")
        print(f"  Optimal threshold: {data['optimal_threshold']}  "
              f"(best avg return: {data['best_avg_return_pct']})")
        print(f"  {'Decile':<8} {'Range':<14} {'Count':>6} {'AvgRet':>8} {'WinRate':>8}")
        for d in data['deciles']:
            avg = f"{d['avg_return_pct']:.4f}" if d['avg_return_pct'] is not None else '  N/A  '
            wr = f"{d['win_rate_pct']:.2f}" if d['win_rate_pct'] is not None else ' N/A'
            print(f"  {d['decile']:<8} {d['range_low']:.1f}-{d['range_high']:.1f}      "
                  f"{d['trade_count']:>6} {avg:>8} {wr:>8}")
