import numpy as np
import pandas as pd
from typing import Dict, Any


def _extract_essential_core_stats(cache: Dict) -> Dict[str, Any]:
    """Extract only the 9 essential stats from vbt_stats."""
    try:
        vbt_stats = cache.get('vbt_stats', pd.Series(dtype=float))
        if vbt_stats is None or len(vbt_stats) == 0:
            return {}

        essential_mapping = {
            'Sharpe Ratio':      'sharpe_ratio',
            'Sortino Ratio':     'sortino_ratio',
            'Calmar Ratio':      'calmar_ratio',
            'Total Return [%]':  'total_return_pct',
            'Total Trades':      'total_trades',
            'Win Rate [%]':      'win_rate_pct',
            'Max Drawdown [%]':  'max_drawdown_pct',
            'Profit Factor':     'profit_factor',
            'Expectancy':        'expectancy',
        }
        result: Dict[str, Any] = {}
        for original_key, mapped_key in essential_mapping.items():
            if original_key in vbt_stats.index:
                value = vbt_stats[original_key]
                if hasattr(value, 'item') and not isinstance(value, str):
                    try:
                        value = value.item()
                    except Exception:
                        pass
                result[mapped_key] = value
        return result
    except Exception:
        return {}
