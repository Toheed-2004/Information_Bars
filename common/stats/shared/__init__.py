"""
Shared utilities and ratio calculations used by both
custom/ and vectorbt_pro/ stats modules.
"""
from .utils import _max_consecutive_numpy, _rolling_windows
from .ratios import (
    _get_empty_risk_adjusted,
    _calculate_cagr,
    _calculate_expected_return,
    _calculate_adjusted_sortino,
    _calculate_serenity_index,
    _calculate_probabilistic_sharpe_ratio,
    _calculate_kelly_criterion,
    _calculate_risk_adjusted_ratios,
)
