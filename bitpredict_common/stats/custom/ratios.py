"""
Thin re-export shim — all implementations live in shared/ratios.py.
"""
from ..shared.ratios import (
    _get_empty_risk_adjusted,
    _calculate_cagr,
    _calculate_expected_return,
    _calculate_adjusted_sortino,
    _calculate_serenity_index,
    _calculate_probabilistic_sharpe_ratio,
    _calculate_kelly_criterion,
    _calculate_risk_adjusted_ratios,
)

__all__ = [
    '_get_empty_risk_adjusted',
    '_calculate_cagr',
    '_calculate_expected_return',
    '_calculate_adjusted_sortino',
    '_calculate_serenity_index',
    '_calculate_probabilistic_sharpe_ratio',
    '_calculate_kelly_criterion',
    '_calculate_risk_adjusted_ratios',
]
