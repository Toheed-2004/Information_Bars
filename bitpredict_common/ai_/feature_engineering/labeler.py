"""
Universal Labeler for ML Strategies

Supports multiple labeling methods:
- Triple Barrier Method (with ATR or percentage barriers)
- Fixed Percentage Returns
- Regression Targets
"""

import pandas as pd
import numpy as np
from typing import Literal
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


class UniversalLabeler:
    """
    Universal labeling system for ML strategies.
    
    Supports:
    - Triple Barrier Method (meta-labeling)
    - Fixed percentage returns
    - Regression targets
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize labeler.
        
        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()
        
        # Pre-calculate ATR for ATR-based barriers
        if 'atr' not in self.df.columns:
            self.df['atr'] = self._calculate_atr(14)
    
    def _calculate_atr(self, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    def generate(self, 
                method: Literal['triple_barrier', 'fixed_pct', 'regression'] = 'triple_barrier',
                **kwargs) -> pd.Series:
        """
        Generic entry point for label generation.
        
        Args:
            method: Labeling method
                - 'triple_barrier': Triple barrier method (classification)
                - 'fixed_pct': Fixed percentage returns (classification)
                - 'regression': Future returns (regression)
            **kwargs: Method-specific parameters
            
        Returns:
            Series of labels
        """
        if method == 'triple_barrier':
            return self._triple_barrier(**kwargs)
        elif method == 'fixed_pct':
            return self._fixed_pct(**kwargs)
        elif method == 'regression':
            return self._regression(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _triple_barrier(self,
                       pt_mult: float = 2.0,
                       sl_mult: float = 1.0,
                       horizon: int = 20,
                       barrier_type: Literal['atr', 'pct'] = 'atr',
                       pct_base: float = 0.01) -> pd.Series:
        """
        Triple Barrier Method for meta-labeling.
        
        Labels:
        - 1: Hit upper barrier first (profit target)
        - -1: Hit lower barrier first (stop loss)
        - 0: Hit time barrier (no clear signal)
        
        Args:
            pt_mult: Profit target multiplier
            sl_mult: Stop loss multiplier
            horizon: Time barrier (bars)
            barrier_type: 'atr' or 'pct'
            pct_base: Base percentage for 'pct' barrier (e.g., 0.01 = 1%)
            
        Returns:
            Series of labels {-1, 0, 1}
        """
        logger.info(f"Generating triple barrier labels: PT={pt_mult}, SL={sl_mult}, "
                   f"horizon={horizon}, type={barrier_type}")
        
        labels = pd.Series(index=self.df.index, data=0, dtype=int)
        prices = self.df['close'].values
        
        # Determine barrier widths
        if barrier_type == 'atr':
            widths = self.df['atr'].values
        else:  # percentage
            widths = prices * pct_base
        
        # Calculate labels
        n_upper = 0
        n_lower = 0
        n_time = 0
        
        for i in range(len(prices) - horizon):
            if np.isnan(widths[i]) or np.isnan(prices[i]):
                continue
            
            # Define barriers
            upper = prices[i] + (widths[i] * pt_mult)
            lower = prices[i] - (widths[i] * sl_mult)
            
            # Future price window
            future_window = prices[i+1 : i+1+horizon]
            
            # Find first barrier hit
            hit_up = np.where(future_window >= upper)[0]
            hit_lo = np.where(future_window <= lower)[0]
            
            first_up = hit_up[0] if len(hit_up) > 0 else horizon
            first_lo = hit_lo[0] if len(hit_lo) > 0 else horizon
            
            # Assign label
            if first_up < first_lo and first_up < horizon:
                labels.iloc[i] = 1
                n_upper += 1
            elif first_lo < first_up and first_lo < horizon:
                labels.iloc[i] = -1
                n_lower += 1
            else:
                labels.iloc[i] = 0  # Time barrier
                n_time += 1
        
        # Log distribution
        total = n_upper + n_lower + n_time
        if total > 0:
            logger.info(f"Label distribution: "
                       f"Upper={n_upper} ({n_upper/total*100:.1f}%), "
                       f"Lower={n_lower} ({n_lower/total*100:.1f}%), "
                       f"Time={n_time} ({n_time/total*100:.1f}%)")
        
        return labels
    
    def _fixed_pct(self,
                  horizon: int = 20,
                  threshold: float = 0.02) -> pd.Series:
        """
        Fixed percentage return labeling.
        
        Labels:
        - 1: Return > threshold
        - -1: Return < -threshold
        - 0: |Return| <= threshold
        
        Args:
            horizon: Forward looking period
            threshold: Percentage threshold (e.g., 0.02 = 2%)
            
        Returns:
            Series of labels {-1, 0, 1}
        """
        logger.info(f"Generating fixed percentage labels: horizon={horizon}, threshold={threshold}")
        
        # Calculate forward returns
        forward_returns = self.df['close'].pct_change(horizon).shift(-horizon)
        
        # Classify
        labels = pd.Series(index=self.df.index, data=0, dtype=int)
        labels[forward_returns > threshold] = 1
        labels[forward_returns < -threshold] = -1
        
        # Log distribution
        n_up = (labels == 1).sum()
        n_down = (labels == -1).sum()
        n_neutral = (labels == 0).sum()
        total = len(labels)
        
        logger.info(f"Label distribution: "
                   f"Up={n_up} ({n_up/total*100:.1f}%), "
                   f"Down={n_down} ({n_down/total*100:.1f}%), "
                   f"Neutral={n_neutral} ({n_neutral/total*100:.1f}%)")
        
        return labels
    
    def _regression(self,
                   horizon: int = 20,
                   target_type: Literal['return', 'price', 'log_return'] = 'return') -> pd.Series:
        """
        Regression target generation.
        
        Args:
            horizon: Forward looking period
            target_type: Type of target
                - 'return': Percentage return
                - 'price': Future price
                - 'log_return': Log return
                
        Returns:
            Series of continuous targets
        """
        logger.info(f"Generating regression targets: horizon={horizon}, type={target_type}")
        
        if target_type == 'return':
            target = self.df['close'].pct_change(horizon).shift(-horizon)
        elif target_type == 'price':
            target = self.df['close'].shift(-horizon)
        elif target_type == 'log_return':
            target = np.log(self.df['close'] / self.df['close'].shift(horizon)).shift(-horizon)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        logger.info(f"Target stats: mean={target.mean():.4f}, std={target.std():.4f}, "
                   f"min={target.min():.4f}, max={target.max():.4f}")
        
        return target

def create_labels(df: pd.DataFrame, label_config: dict) -> pd.Series:
    """
    Create labels from configuration dictionary.
    
    This is a unified entry point that reads the config and calls the appropriate
    labeling method with the correct parameters.
    
    Args:
        df: DataFrame with OHLCV data
        label_config: Configuration dictionary with method-specific nested configs
            
    Returns:
        Series of labels
        
    Example config (new structure):
        {
            'triple_barrier': {
                'pt_mult': 2.0,
                'sl_mult': 1.0,
                'horizon': 20,
                'barrier_type': 'atr'
            },
            'fixed_pct': {
                'horizon': 20,
                'threshold': 0.02
            },
            'regression': {
                'horizon': 20,
                'target_type': 'return'
            }
        }
    """
    # Determine which method to use based on which config is present
    if 'triple_barrier' in label_config:
        method = 'triple_barrier'
        params = label_config['triple_barrier']
    elif 'fixed_pct' in label_config:
        method = 'fixed_pct'
        params = label_config['fixed_pct']
    elif 'regression' in label_config:
        method = 'regression'
        params = label_config['regression']
    else:
        raise ValueError("No valid labeling method found in config. Expected 'triple_barrier', 'fixed_pct', or 'regression'")
    
    logger.info(f"Creating labels from config: method={method}, params={params}")
    labeler = UniversalLabeler(df)
    return labeler.generate(method=method, **params)
