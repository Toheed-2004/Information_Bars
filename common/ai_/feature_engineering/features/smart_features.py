"""
Smart Core Features (Category: SMART)

This module provides the SmartFeatures class for foundational features that every 
trading system needs. These include:
1. Returns & Risk-Adjusted Features
2. Price Position Features
3. Volume Transformation
4. Volatility Features
5. Smart Money Proxies (critical for crypto)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, List
from bitpredict.common.ta.indicators.base import calculate_indicators
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


class SmartFeatures:
    """
    Core foundational features for trading systems.
    
    This class implements Phase 2 features including:
    - Returns and risk-adjusted metrics across multiple windows
    - Price position and intraday strength indicators
    - Volume transformations and smart money proxies
    - Volatility-based features for crypto markets
    """

    def __init__(self, df: pd.DataFrame, close: np.ndarray, high: np.ndarray, 
                 low: np.ndarray, volume: np.ndarray, n: int, eps: float):
        """
        Initializes the SmartFeatures calculator.

        Args:
            df (pd.DataFrame): Input OHLCV data.
            close, high, low, volume (np.ndarray): Price/volume arrays.
            n (int): Length of data.
            eps (float): Stability constant.
        """
        self.df = df
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.n = n
        self.eps = eps
        
        # Pre-calculate 1-period returns for efficiency
        self.returns_1 = self._calculate_1period_returns()
        
        # Store indicator results for reuse
        self._cached_atr = {}
        self._cached_bbands = {}

    def _calculate_1period_returns(self) -> np.ndarray:
        """Calculate 1-period arithmetic returns."""
        returns = np.full(self.n, np.nan)
        for i in range(1, self.n):
            if self.close[i-1] > self.eps:
                returns[i] = (self.close[i] - self.close[i-1]) / self.close[i-1]
        return returns

    def _extract_indicator(self, result: Union[pd.DataFrame, pd.Series, np.ndarray], 
                           indicator_name: str) -> np.ndarray:
        """
        Internal Helper: Robust extraction of indicator values from diverse return types.
        Same as in TechnicalIndicators class for consistency.
        
        Args:
            result: The raw output from the indicator calculation.
            indicator_name (str): Target name or substring for filtering.

        Returns:
            np.ndarray: One-dimensional array of indicator values.
        """
        if isinstance(result, pd.DataFrame):
            # Attempt to find column by name match
            for col in result.columns:
                if indicator_name.lower() in col.lower():
                    return result[col].values
            # Fallback: return the first available column
            if len(result.columns) > 0:
                return result.iloc[:, 0].values
        elif isinstance(result, pd.Series):
            return result.values
        elif isinstance(result, np.ndarray):
            return result
        # Return empty NaN array if extraction fails
        return np.full(self.n, np.nan)

    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        rolling_std = np.full(self.n, np.nan)
        for i in range(window, self.n):
            segment = data[i-window+1:i+1]
            if not np.any(np.isnan(segment)):
                rolling_std[i] = np.std(segment)
        return rolling_std

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean."""
        rolling_mean = np.full(self.n, np.nan)
        for i in range(window, self.n):
            segment = data[i-window+1:i+1]
            if not np.any(np.isnan(segment)):
                rolling_mean[i] = np.mean(segment)
        return rolling_mean

    def _rolling_median(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling median."""
        rolling_median = np.full(self.n, np.nan)
        for i in range(window, self.n):
            segment = data[i-window+1:i+1]
            if not np.any(np.isnan(segment)):
                rolling_median[i] = np.median(segment)
        return rolling_median

    # =========================================================================
    # 2.1 RETURNS & RISK-ADJUSTED FEATURES
    # =========================================================================

    def return_window(self, window: int) -> np.ndarray:
        """
        Calculate arithmetic return over specified window.
        
        Args:
            window (int): Lookback period for return calculation
            
        Returns:
            np.ndarray: Arithmetic returns over window
        """
        returns = np.full(self.n, np.nan)
        for i in range(window, self.n):
            if self.close[i-window] > self.eps:
                returns[i] = (self.close[i] - self.close[i-window]) / self.close[i-window]
        return returns

    def log_return_window(self, window: int) -> np.ndarray:
        """
        Calculate log return over specified window.
        
        Args:
            window (int): Lookback period for log return calculation
            
        Returns:
            np.ndarray: Log returns over window
        """
        log_returns = np.full(self.n, np.nan)
        for i in range(window, self.n):
            if self.close[i-window] > self.eps and self.close[i] > self.eps:
                log_returns[i] = np.log(self.close[i] / self.close[i-window])
        return log_returns

    def volatility_window(self, window: int) -> np.ndarray:
        """
        Calculate volatility as rolling standard deviation of 1-period returns.
        
        Args:
            window (int): Window for rolling standard deviation
            
        Returns:
            np.ndarray: Rolling volatility
        """
        return self._rolling_std(self.returns_1, window)

    def risk_adjusted_momentum(self, window: int) -> np.ndarray:
        """
        Risk-adjusted momentum: return_W / (volatility_W + epsilon).
        
        Args:
            window (int): Window for both return and volatility
            
        Returns:
            np.ndarray: Risk-adjusted momentum values
        """
        returns = self.return_window(window)
        volatility = self.volatility_window(window)
        
        risk_adj_momentum = np.full(self.n, np.nan)
        for i in range(window, self.n):
            if not np.isnan(returns[i]) and not np.isnan(volatility[i]):
                risk_adj_momentum[i] = returns[i] / (volatility[i] + self.eps)
        
        return risk_adj_momentum

    def sharpe_ratio(self, window: int) -> np.ndarray:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            window (int): Window for return and volatility calculation
            
        Returns:
            np.ndarray: Annualized Sharpe ratios
        """
        returns_mean = self._rolling_mean(self.returns_1, window)
        volatility = self.volatility_window(window)
        
        # Annualization factor for crypto (assuming daily data)
        # Adjust based on your timeframe (minute, hour, day, etc.)
        annualization_factor = np.sqrt(365)  # For daily data
        
        sharpe = np.full(self.n, np.nan)
        for i in range(window, self.n):
            if not np.isnan(returns_mean[i]) and not np.isnan(volatility[i]):
                if volatility[i] > self.eps:
                    sharpe[i] = (returns_mean[i] / volatility[i]) * annualization_factor
        
        return sharpe

    # =========================================================================
    # 2.2 PRICE POSITION FEATURES
    # =========================================================================

    def price_position(self) -> np.ndarray:
        """
        Calculate price position within the bar.
        
        Returns:
            np.ndarray: (close - low) / (high - low + epsilon)
        """
        position = np.full(self.n, np.nan)
        for i in range(self.n):
            denominator = self.high[i] - self.low[i] + self.eps
            if denominator > self.eps:
                position[i] = (self.close[i] - self.low[i]) / denominator
        return position

    def close_position(self) -> np.ndarray:
        """
        Calculate close position within the bar (same as price_position).
        
        Returns:
            np.ndarray: (close - low) / (high - low + epsilon)
        """
        return self.price_position()

    def typical_price(self) -> np.ndarray:
        """
        Calculate typical price: (high + low + close) / 3.
        
        Returns:
            np.ndarray: Typical price values
        """
        typical_price = np.full(self.n, np.nan)
        for i in range(self.n):
            typical_price[i] = (self.high[i] + self.low[i] + self.close[i]) / 3
        return typical_price

    def median_price(self) -> np.ndarray:
        """
        Calculate median price: (high + low) / 2.
        
        Returns:
            np.ndarray: Median price values
        """
        median_price = np.full(self.n, np.nan)
        for i in range(self.n):
            median_price[i] = (self.high[i] + self.low[i]) / 2
        return median_price

    # =========================================================================
    # 2.3 VOLUME TRANSFORMATION
    # =========================================================================

    def log_volume(self) -> np.ndarray:
        """
        Calculate natural logarithm of (1 + volume).
        
        Returns:
            np.ndarray: Log-transformed volume
        """
        log_vol = np.full(self.n, np.nan)
        for i in range(self.n):
            if self.volume[i] >= 0:
                log_vol[i] = np.log1p(self.volume[i])
        return log_vol

    def volume_ratio(self, window: int) -> np.ndarray:
        """
        Calculate volume ratio: volume / rolling_median(volume, window).
        
        Args:
            window (int): Window for rolling median
            
        Returns:
            np.ndarray: Volume ratio values
        """
        vol_median = self._rolling_median(self.volume, window)
        
        ratio = np.full(self.n, np.nan)
        for i in range(window, self.n):
            if not np.isnan(vol_median[i]) and vol_median[i] > self.eps:
                ratio[i] = self.volume[i] / vol_median[i]
        
        return ratio

    def volume_zscore(self, window: int) -> np.ndarray:
        """
        Calculate volume z-score: (log_volume - mean) / std.
        
        Args:
            window (int): Window for rolling statistics
            
        Returns:
            np.ndarray: Volume z-scores
        """
        log_vol = self.log_volume()
        vol_mean = self._rolling_mean(log_vol, window)
        vol_std = self._rolling_std(log_vol, window)
        
        zscore = np.full(self.n, np.nan)
        for i in range(window, self.n):
            if not np.isnan(vol_mean[i]) and not np.isnan(vol_std[i]):
                if vol_std[i] > self.eps:
                    zscore[i] = (log_vol[i] - vol_mean[i]) / vol_std[i]
        
        return zscore

    # =========================================================================
    # 2.4 VOLATILITY FEATURES
    # =========================================================================

    def atr(self, window: int = 14) -> np.ndarray:
        """
        Calculate Average True Range using the TALib indicator.
        
        Args:
            window (int): ATR window period
            
        Returns:
            np.ndarray: ATR values
        """
        # Check cache first
        cache_key = f"atr_{window}"
        if cache_key in self._cached_atr:
            return self._cached_atr[cache_key]
        
        # Use the existing TA indicator calculator
        result, _ = calculate_indicators(
            drop_nan=False, 
            data=self.df, 
            indicators={"ATR": {"timeperiod": window}}
        )
        
        # Extract ATR values using the same pattern as TechnicalIndicators
        atr_values = self._extract_indicator(result, "atr")
        
        # Cache the result
        self._cached_atr[cache_key] = atr_values
        
        return atr_values

    def atr_percent(self, window: int = 14) -> np.ndarray:
        """
        Calculate ATR as percentage of close price.
        
        Args:
            window (int): ATR window period
            
        Returns:
            np.ndarray: ATR percentage values
        """
        atr_values = self.atr(window)
        
        atr_percent = np.full(self.n, np.nan)
        for i in range(self.n):
            if not np.isnan(atr_values[i]) and self.close[i] > self.eps:
                atr_percent[i] = (atr_values[i] / self.close[i]) * 100
        
        return atr_percent

    def volatility_ratio(self, short_window: int = 20, long_window: int = 100) -> np.ndarray:
        """
        Calculate ratio of short-term to long-term volatility.
        
        Args:
            short_window (int): Short-term volatility window
            long_window (int): Long-term volatility window
            
        Returns:
            np.ndarray: Volatility ratio values
        """
        short_vol = self.volatility_window(short_window)
        long_vol = self.volatility_window(long_window)
        
        ratio = np.full(self.n, np.nan)
        for i in range(long_window, self.n):
            if not np.isnan(short_vol[i]) and not np.isnan(long_vol[i]):
                if long_vol[i] > self.eps:
                    ratio[i] = short_vol[i] / long_vol[i]
        
        return ratio

    def bb_width_percent(self, window: int = 20, nbdev: float = 2.0) -> np.ndarray:
        """
        Calculate Bollinger Band width as percentage of middle band.
        
        Args:
            window (int): Bollinger Band window
            nbdev (float): Number of standard deviations
            
        Returns:
            np.ndarray: BB width percentage
        """
        # Check cache first
        cache_key = f"bbands_{window}_{nbdev}"
        if cache_key in self._cached_bbands:
            upper, middle, lower = self._cached_bbands[cache_key]
        else:
            # Use the existing TA indicator calculator for Bollinger Bands
            result, _ = calculate_indicators(
                drop_nan=False,
                data=self.df,
                indicators={
                    "BBANDS": {
                        "timeperiod": window, 
                        "nbdevup": nbdev, 
                        "nbdevdn": nbdev
                    }
                },
            )
            
            # Extract bands
            upper = self._extract_indicator(result, "upper")
            middle = self._extract_indicator(result, "middle")
            lower = self._extract_indicator(result, "lower")
            
            # Cache the result
            self._cached_bbands[cache_key] = (upper, middle, lower)
        
        # Calculate width percentage
        width_pct = np.full(self.n, np.nan)
        for i in range(self.n):
            if (not np.isnan(upper[i]) and not np.isnan(lower[i]) and 
                not np.isnan(middle[i]) and middle[i] > self.eps):
                width_pct[i] = ((upper[i] - lower[i]) / middle[i]) * 100
        
        return width_pct

    # =========================================================================
    # 2.5 SMART MONEY PROXIES (CRITICAL FOR CRYPTO)
    # =========================================================================

    def pv_divergence(self, return_window: int = 5, volume_window: int = 10) -> np.ndarray:
        """
        Calculate price-volume divergence.
        
        Args:
            return_window (int): Window for return calculation
            volume_window (int): Window for volume ratio
            
        Returns:
            np.ndarray: Price-volume divergence values
        """
        returns = self.return_window(return_window)
        vol_ratio = self.volume_ratio(volume_window)
        
        divergence = np.full(self.n, np.nan)
        for i in range(max(return_window, volume_window), self.n):
            if not np.isnan(returns[i]) and not np.isnan(vol_ratio[i]):
                sign = 1 if returns[i] >= 0 else -1
                divergence[i] = sign * (abs(returns[i]) - abs(vol_ratio[i]))
        
        return divergence

    def efficiency_ratio(self, window: int = 10) -> np.ndarray:
        """
        Calculate efficiency ratio: abs(price_change) / sum(abs(daily_returns)).
        
        Args:
            window (int): Window for calculation
            
        Returns:
            np.ndarray: Efficiency ratio values (0-1)
        """
        price_change = np.full(self.n, np.nan)
        sum_abs_returns = np.full(self.n, np.nan)
        
        for i in range(window, self.n):
            price_change[i] = abs(self.close[i] - self.close[i-window])
            
            # Sum of absolute daily returns
            returns_segment = self.returns_1[i-window+1:i+1]
            if not np.any(np.isnan(returns_segment)):
                sum_abs_returns[i] = np.sum(np.abs(returns_segment))
        
        efficiency = np.full(self.n, np.nan)
        for i in range(window, self.n):
            if not np.isnan(price_change[i]) and not np.isnan(sum_abs_returns[i]):
                if sum_abs_returns[i] > self.eps:
                    efficiency[i] = price_change[i] / sum_abs_returns[i]
        
        return efficiency

    def intraday_strength(self) -> np.ndarray:
        """
        Calculate intraday strength indicator.
        
        Returns:
            np.ndarray: Strength values (-1, 0, 1)
        """
        close_pos = self.close_position()
        
        strength = np.zeros(self.n)
        for i in range(self.n):
            if not np.isnan(close_pos[i]):
                if close_pos[i] > 0.7:
                    strength[i] = 1
                elif close_pos[i] < 0.3:
                    strength[i] = -1
        
        return strength

    def accumulation_score(self) -> np.ndarray:
        """
        Calculate smart money accumulation score.
        
        Returns:
            np.ndarray: Accumulation score values
        """
        score = np.full(self.n, np.nan)
        
        for i in range(self.n):
            denominator = self.high[i] - self.low[i] + self.eps
            if denominator > self.eps:
                # ((close-low) - (high-close)) / (high-low) * volume
                numerator = (self.close[i] - self.low[i]) - (self.high[i] - self.close[i])
                score[i] = (numerator / denominator) * self.volume[i]
        
        return score