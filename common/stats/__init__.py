import pandas as pd
from bitpredict.common.stats.vectorbt_pro.vbt_stats import (
    calculate_essential_stats_only as _vbt_essential_stats,
    calculate_comprehensive_vbt_stats_optimized as _vbt_comprehensive_stats
)
from bitpredict.common.stats.custom.custom_stats import (
    calculate_essential_stats as _custom_essential_stats,
    calculate_comprehensive_stats as _custom_comprehensive_stats,
    calculate_regime_stats
)

__all__ = [
    'calculate_essential_stats',
    'calculate_comprehensive_stats',
    # Underlying functions (optional, for debugging or special cases)
    '_vbt_essential_stats',
    '_vbt_comprehensive_stats',
    '_custom_essential_stats',
    '_custom_comprehensive_stats',
    "calculate_regime_stats"
]


def calculate_essential_stats(data):
    """
    Unified essential statistics calculator.

    Automatically detects input type:
    - If `data` is a pandas DataFrame (ledger), uses the custom pure‑NumPy implementation.
    - Otherwise (assumes a vectorbt Portfolio object), uses the optimized vbt-based function.

    Args:
        data: Either a ledger DataFrame or a vectorbt Portfolio object.

    Returns:
        Dictionary of essential statistics.
    """
    if isinstance(data, pd.DataFrame):
        return _custom_essential_stats(data)
    else:
        return _vbt_essential_stats(data)['0']


def calculate_comprehensive_stats(data, **kwargs):
    """
    Unified comprehensive statistics calculator.

    Automatically detects input type:
    - If `data` is a pandas DataFrame (ledger), uses the custom pure‑NumPy implementation.
    - Otherwise (assumes a vectorbt Portfolio object), uses the optimized vbt-based function,
      passing any additional keyword arguments (e.g., ledger_input, bar_type, benchmark_returns).

    Args:
        data: Either a ledger DataFrame or a vectorbt Portfolio object.
        **kwargs: Additional arguments for the vbt comprehensive function:
            - ledger_input: DataFrame (required when using vbt)
            - bar_type: str, default 'time'
            - benchmark_returns: array-like, optional

    Returns:
        Nested dictionary of comprehensive statistics.
    """
    if isinstance(data, pd.DataFrame):
        return _custom_comprehensive_stats(data, **kwargs)['0']
    else:
        # Strip params that were removed from vbt stats (bar_type, benchmark_returns)
        vbt_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ('ledger_input', 'calculate_monte_carlo')
        }
        return _vbt_comprehensive_stats(portfolio_obj=data, **vbt_kwargs)['0']