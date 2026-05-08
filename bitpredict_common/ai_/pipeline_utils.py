"""
Pipeline Utility Functions

Helper functions used across all pipeline phases.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from statsmodels.tsa.stattools import adfuller
from scipy.stats import spearmanr


def check_stationarity(series: pd.Series, significance: float = 0.05) -> Tuple[bool, float]:
    """
    Run ADF test for stationarity.
    
    Args:
        series: Time series to test
        significance: P-value threshold
        
    Returns:
        Tuple of (is_stationary, p_value)
    """
    clean_series = series.dropna()
    if len(clean_series) < 10:
        return False, 1.0
    try:
        result = adfuller(clean_series, autolag='AIC')
        return result[1] < significance, result[1]
    except:
        return False, 1.0


def get_domain_transform(domain: str) -> Optional[str]:
    """
    Get transformation method for domain.
    
    Args:
        domain: Feature domain ('positive', 'bounded', 'signed')
        
    Returns:
        Transformation method name or None
    """
    if domain == "positive":
        return "signed_log"
    elif domain == "bounded":
        return "logit_transform"
    return None


def create_forward_returns(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """
    Create forward returns for multiple horizons.
    
    Args:
        df: DataFrame with 'close' column
        horizons: List of forward periods
        
    Returns:
        DataFrame with forward returns
    """
    returns = pd.DataFrame(index=df.index)
    for n in horizons:
        returns[f'return_{n}'] = np.log(df['close'].shift(-n) / df['close'])
    return returns


def calculate_spearman_safe(feature: pd.Series, returns: pd.Series, 
                            min_samples: int = 30) -> Tuple[float, int]:
    """
    Calculate Spearman correlation with safety checks.
    
    Args:
        feature: Feature series
        returns: Returns series
        min_samples: Minimum valid samples required
        
    Returns:
        Tuple of (correlation, n_samples)
    """
    # Remove NaN values
    valid = ~(feature.isna() | returns.isna())
    n_samples = valid.sum()
    
    if n_samples < min_samples:
        return np.nan, n_samples
    
    feature_valid = feature[valid]
    returns_valid = returns[valid]
    
    # Check for constant values (zero variance)
    if feature_valid.std() < 1e-10 or returns_valid.std() < 1e-10:
        return np.nan, n_samples
    
    try:
        ric, _ = spearmanr(feature_valid, returns_valid)
        return ric, n_samples
    except:
        return np.nan, n_samples


def get_feature_columns(df: pd.DataFrame, feature_metadata: Dict) -> List[str]:
    """
    Get list of feature columns for processing (excludes OHLCV and regime columns).
    
    Args:
        df: DataFrame with all columns
        feature_metadata: Feature metadata dictionary
        
    Returns:
        List of feature column names
    """
    # Columns to exclude from feature processing
    exclude = [
        # OHLCV base columns
        'open', 'high', 'low', 'close', 'volume', 'datetime',
        # Regime metadata (needed for Phase 2, but not features to transform)
        'primary_regime', 'regime_confidence', 'regime_duration', 
        'regime_stability', 'is_transitioning'
    ]
    
    # Regime probability columns (prob_*)
    exclude.extend([c for c in df.columns if c.startswith('prob_')])
    
    # Get only feature columns from feature_metadata
    feature_cols = list(feature_metadata.get('features', {}).keys())
    
    # Return features that exist in df and are not in exclude list
    return [c for c in feature_cols if c in df.columns and c not in exclude]


def get_cluster_id(feature_name: str, clusters: Dict) -> int:
    """
    Helper to find which cluster a feature belongs to.
    
    Args:
        feature_name: Name of the feature
        clusters: Dictionary of cluster_id -> [feature_names]
        
    Returns:
        Cluster ID or -1 if not found
    """
    for cluster_id, members in clusters.items():
        if feature_name in members:
            return cluster_id
    return -1
