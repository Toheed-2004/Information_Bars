"""
Utility functions for statistics processing
"""
import numpy as np
import pandas as pd
from typing import Any, Dict


def round_stats(stats_dict: Dict[str, Any], decimals: int = 4) -> Dict[str, Any]:
    """
    Round all numeric values in a stats dictionary to specified decimal places.
    
    Args:
        stats_dict: Dictionary containing statistics
        decimals: Number of decimal places (default: 4)
    
    Returns:
        Dictionary with rounded values
    """
    rounded = {}
    for key, value in stats_dict.items():
        if isinstance(value, (int, np.integer)):
            # Keep integers as-is
            rounded[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            # Round floats
            if np.isnan(value) or np.isinf(value):
                rounded[key] = 0.0
            else:
                rounded[key] = round(float(value), decimals)
        elif isinstance(value, dict):
            # Recursively round nested dictionaries
            rounded[key] = round_stats(value, decimals)
        elif isinstance(value, list):
            # Round list items if they're numeric
            rounded[key] = [
                round(float(item), decimals) if isinstance(item, (float, np.floating)) else item
                for item in value
            ]
        else:
            # Keep other types as-is (strings, timestamps, etc.)
            rounded[key] = value
    
    return rounded
