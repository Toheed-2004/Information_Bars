import numpy as np
from typing import Dict, Any

from bitpredict.common.regimes_analysis.config import (
    FITNESS_WEIGHT_WIN_RATE,
    FITNESS_WEIGHT_PROFIT_FACTOR,
    FITNESS_WEIGHT_AVG_PNL,
    FITNESS_WEIGHT_TRADE_COUNT,
    FITNESS_PROFIT_FACTOR_CAP,
    FITNESS_AVG_PNL_CAP,
    FITNESS_MIN_TRADES_RELIABLE,
)


def compute_regime_fitness(regime_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute a fitness score [0, 1] for each regime.

    Input is the `by_regime_label` dict from compute_regime_performance.
    Required keys per regime: trade_count, win_rate_pct, profit_factor,
    avg_return_pct, max_drawdown_pct.

    Formula
    -------
    base_fitness = weighted sum of 4 normalised components (weights from config):
        win_rate      : win_rate_pct / 100
        profit_factor : clipped to [0, FITNESS_PROFIT_FACTOR_CAP], then normalised
        avg_return    : clipped to [0, FITNESS_AVG_PNL_CAP], then normalised
        trade_count   : min(count / FITNESS_MIN_TRADES_RELIABLE, 1.0)

    drawdown_penalty = 1 - min(abs(max_drawdown_pct) / 100, 1.0)
    fitness = base_fitness * drawdown_penalty

    Returns
    -------
    {regime_label: {fitness_score, trade_count, reliable}}
    """
    results: Dict[str, Any] = {}

    for regime, stats in regime_data.items():
        trade_count = int(stats.get('trade_count', 0))
        win_rate_pct = float(stats.get('win_rate_pct', 0.0))
        profit_factor = float(stats.get('profit_factor', 0.0))
        avg_return_pct = float(stats.get('avg_return_pct', 0.0))
        max_drawdown_pct = float(stats.get('max_drawdown_pct', 0.0))

        win_rate_norm = float(np.clip(win_rate_pct / 100.0, 0.0, 1.0))
        pf_norm = float(np.clip(profit_factor, 0.0, FITNESS_PROFIT_FACTOR_CAP)) / FITNESS_PROFIT_FACTOR_CAP
        avg_r_norm = float(np.clip(avg_return_pct, 0.0, FITNESS_AVG_PNL_CAP)) / FITNESS_AVG_PNL_CAP
        trade_norm = min(trade_count / FITNESS_MIN_TRADES_RELIABLE, 1.0)

        base_fitness = (
            win_rate_norm * FITNESS_WEIGHT_WIN_RATE
            + pf_norm * FITNESS_WEIGHT_PROFIT_FACTOR
            + avg_r_norm * FITNESS_WEIGHT_AVG_PNL
            + trade_norm * FITNESS_WEIGHT_TRADE_COUNT
        )

        # High drawdown reduces fitness proportionally
        drawdown_penalty = 1.0 - min(abs(max_drawdown_pct) / 100.0, 1.0)
        fitness = round(float(base_fitness * drawdown_penalty), 4)

        results[str(regime)] = {
            'fitness_score': fitness,
            'trade_count': trade_count,
            'reliable': trade_count >= FITNESS_MIN_TRADES_RELIABLE,
        }

    return results


def print_regime_fitness(results: dict) -> None:
    print("\n=== REGIME FITNESS SCORES ===")
    for regime, data in sorted(results.items(), key=lambda x: x[1]['fitness_score'], reverse=True):
        rel = '' if data['reliable'] else ' [unreliable]'
        print(f"  {regime}: {data['fitness_score']:.4f}  (n={data['trade_count']}){rel}")
