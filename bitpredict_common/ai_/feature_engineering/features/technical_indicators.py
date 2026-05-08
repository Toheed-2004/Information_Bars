"""
Standard Technical Indicators (Category: TI)

This module provides the TechnicalIndicators class, which wraps various 
standard technical analysis tools like Moving Averages, MACD, RSI, and Bollinger Bands.
It integrates with the project's internal TA engine while providing optimized 
helper functions for slope and distance calculations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, List
from bitpredict.common.ta.indicators.base import calculate_indicators
from bitpredict.common.logging import get_logger

# Initialize logger for debugging and status updates
logger = get_logger(__name__)


class TechnicalIndicators:
    """
    Calculates technical analysis indicators for financial time-series.
    
    Categorized into:
    1. Trend-Following (SMA, EMA, MACD, etc.)
    2. Momentum Oscillators (RSI, Stochastic, Williams %R, etc.)
    3. Volatility-Based (Bollinger Bands, Keltner Channels, ATR, etc.)

    Attributes:
        df (pd.DataFrame): Copy of the input OHLCV DataFrame.
        close, high, low, volume (np.ndarray): Price and volume series.
        n (int): Total number of data points.
        eps (float): Epsilon for numerical stability (avoiding division by zero).
    """

    def __init__(self, df: pd.DataFrame, close: np.ndarray, high: np.ndarray, 
                 low: np.ndarray, volume: np.ndarray, n: int, eps: float):
        """
        Initializes the TechnicalIndicators calculator.

        Args:
            df (pd.DataFrame): Input OHLCV data.
            close, high, low, volume (np.ndarray): Price/volume arrays.
            n (int): Length of data.
            eps (float): Stability constant.
        """
        self.df = df.copy()
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.n = n
        self.eps = eps

    def _extract_indicator(self, result: Union[pd.DataFrame, pd.Series, np.ndarray], 
                           indicator_name: str) -> np.ndarray:
        """
        Internal Helper: Robust extraction of indicator values from diverse return types.
        
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

    # =========================================================================
    # TREND-FOLLOWING INDICATORS
    # =========================================================================

    def sma(self, window: Union[int, List[int]] = 20) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI01: Simple Moving Average (SMA).
        Calculates the arithmetic mean of prices over the last N periods.
        """
        if isinstance(window, list):
            return [self._single_sma(w) for w in window]
        return self._single_sma(window)

    def _single_sma(self, window: int) -> np.ndarray:
        """Helper for calculating a single SMA window."""
        result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"SMA": {"timeperiod": window}}
        )
        return self._extract_indicator(result, "sma")

    def ema(self, window: Union[int, List[int]] = 20) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI02: Exponential Moving Average (EMA).
        Weighted average that prioritizes recent prices using a decay factor (alpha).
        """
        if isinstance(window, list):
            return [self._single_ema(w) for w in window]
        return self._single_ema(window)

    def _single_ema(self, window: int) -> np.ndarray:
        """Helper for calculating a single EMA window."""
        result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"EMA": {"timeperiod": window}}
        )
        return self._extract_indicator(result, "ema")

    def price_sma_distance(self, sma_window: Union[int, List[int]] = 20) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI03: Price-to-SMA Distance.
        Percentage distance between the current Close and the SMA.
        Values > 0 imply price is above average; < 0 imply price is below average.
        """
        if isinstance(sma_window, list):
            return [self._single_price_sma_distance(w) for w in sma_window]
        return self._single_price_sma_distance(sma_window)

    def _single_price_sma_distance(self, sma_window: int) -> np.ndarray:
        """Helper for price-SMA distance."""
        sma = self._single_sma(sma_window)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Distance = (Price / Average) - 1
            distance = np.where(np.abs(sma) > self.eps, (self.close - sma) / sma, np.nan)
        return distance

    def ema_slope(self, ema_window: Union[int, List[int]] = 20, 
                  deriv_window: int = 5) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI04: EMA Slope.
        Measures the velocity of the EMA trend over a 'deriv_window' range.
        Used to identify trend acceleration or exhaustion.
        """
        if isinstance(ema_window, list):
            return [self._single_ema_slope(w, deriv_window) for w in ema_window]
        return self._single_ema_slope(ema_window, deriv_window)

    def _single_ema_slope(self, ema_window: int, deriv_window: int) -> np.ndarray:
        """Helper for calculating EMA slope."""
        ema = self._single_ema(ema_window)
        slope = np.full(self.n, np.nan)
        for i in range(deriv_window, self.n):
            if not np.isnan(ema[i]) and not np.isnan(ema[i - deriv_window]):
                # Rise over Run approximation
                slope[i] = (ema[i] - ema[i - deriv_window]) / deriv_window
        slope = np.roll(slope, 1)
        slope[0] = np.nan

        return slope

    def macd(self, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        TI05: MACD (Moving Average Convergence Divergence).
        Returns the trio: MACD line, Signal line, and Histogram.
        MACD = EMA(fast) - EMA(slow). Histogram = MACD - Signal.
        """
        result, _ = calculate_indicators(
            drop_nan=False,
            data=self.df,
            indicators={
                "MACD": {"fastperiod": fast, "slowperiod": slow, "signalperiod": signal}
            },
        )
        macd_line = self._extract_indicator(result, "macd")
        signal_line = self._extract_indicator(result, "signal")
        histogram = self._extract_indicator(result, "hist")
        return macd_line, signal_line, histogram

    def adx(self, window: Union[int, List[int]] = 14) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI06: Average Directional Index (ADX).
        Measures trend strength on a scale of 0-100.
        > 25 often signifies a strong trending environment.
        """
        if isinstance(window, list):
            return [self._single_adx(w) for w in window]
        return self._single_adx(window)

    def _single_adx(self, window: int) -> np.ndarray:
        """Helper for single window ADX."""
        result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"ADX": {"timeperiod": window}}
        )
        return self._extract_indicator(result, "adx")

    def adx_slope(self, adx_window: Union[int, List[int]] = 14, 
                  slope_window: int = 5) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI07: ADX Slope.
        Rate of change of the ADX. Positive slope indicates a strengthening trend.
        """
        if isinstance(adx_window, list):
            return [self._single_adx_slope(w, slope_window) for w in adx_window]
        return self._single_adx_slope(adx_window, slope_window)

    def _single_adx_slope(self, adx_window: int, slope_window: int) -> np.ndarray:
        """Helper for ADX slope."""
        adx = self._single_adx(adx_window)
        slope = np.full(self.n, np.nan)
        for i in range(slope_window, self.n):
            if not np.isnan(adx[i]) and not np.isnan(adx[i - slope_window]):
                slope[i] = (adx[i] - adx[i - slope_window]) / slope_window
        return slope

    def supertrend(self, period: Union[int, List[int]] = 10, 
                   multiplier: float = 3.0) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI08: SuperTrend.
        A popular trend indicator that combines volatility (ATR) and price midpoint.
        Acts as a trailing stop/support level.
        """
        if isinstance(period, list):
            return [self._single_supertrend(p, multiplier) for p in period]
        return self._single_supertrend(period, multiplier)

    def _single_supertrend(self, period: int, multiplier: float) -> np.ndarray:
        """Helper for calculating single SuperTrend."""
        # Calculate ATR for volatility bands
        result_atr, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"ATR": {"timeperiod": period}}
        )
        atr = self._extract_indicator(result_atr, "atr")

        hl2 = (self.high + self.low) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        supertrend = np.full(self.n, np.nan)
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()

        trend = 1  # State: 1 (Bullish), -1 (Bearish)

        # Iterative update logic for non-repaint SuperTrend
        for i in range(period, self.n):
            if i > period:
                # Update Upper Band
                if basic_upper[i] < final_upper[i - 1] or self.close[i - 1] > final_upper[i - 1]:
                    final_upper[i] = basic_upper[i]
                else:
                    final_upper[i] = final_upper[i - 1]

                # Update Lower Band
                if basic_lower[i] > final_lower[i - 1] or self.close[i - 1] < final_lower[i - 1]:
                    final_lower[i] = basic_lower[i]
                else:
                    final_lower[i] = final_lower[i - 1]

            # Detect Trend Flip
            if trend == 1:
                if self.close[i] <= final_lower[i]:
                    trend = -1
                    supertrend[i] = final_upper[i]
                else:
                    supertrend[i] = final_lower[i]
            else:
                if self.close[i] >= final_upper[i]:
                    trend = 1
                    supertrend[i] = final_lower[i]
                else:
                    supertrend[i] = final_upper[i]

        return supertrend

    # =========================================================================
    # MOMENTUM OSCILLATORS
    # =========================================================================

    def rsi(self, window: Union[int, List[int]] = 14) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI09: Relative Strength Index (RSI).
        Oscillator measuring velocity and magnitude of price changes.
        Range: 0-100. > 70 is Overbought, < 30 is Oversold.
        """
        if isinstance(window, list):
            return [self._single_rsi(w) for w in window]
        return self._single_rsi(window)

    def _single_rsi(self, window: int) -> np.ndarray:
        """Helper for RSI."""
        result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"RSI": {"timeperiod": window}}
        )
        return self._extract_indicator(result, "rsi")

    def rsi_slope(self, rsi_window: Union[int, List[int]] = 14, 
                  slope_window: int = 3) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI10: RSI Slope.
        Rate of change of RSI. Useful for early detection of momentum reversals.
        """
        if isinstance(rsi_window, list):
            return [self._single_rsi_slope(w, slope_window) for w in rsi_window]
        return self._single_rsi_slope(rsi_window, slope_window)

    def _single_rsi_slope(self, rsi_window: int, slope_window: int) -> np.ndarray:
        """Helper for RSI slope."""
        rsi = self._single_rsi(rsi_window)
        slope = np.full(self.n, np.nan)
        for i in range(slope_window, self.n):
            if not np.isnan(rsi[i]) and not np.isnan(rsi[i - slope_window]):
                slope[i] = (rsi[i] - rsi[i - slope_window]) / slope_window
        return slope

    def stochastic(self, k_window: Union[int, List[int]] = 14, 
                   d_window: int = 3) -> Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        TI11: Stochastic Oscillator.
        Compares closing price to price range over time. Returns %K and %D lines.
        """
        if isinstance(k_window, list):
            return [self._single_stochastic(k, d_window) for k in k_window]
        return self._single_stochastic(k_window, d_window)

    def _single_stochastic(self, k_window: int, d_window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Helper for Stochastic."""
        result, _ = calculate_indicators(
            drop_nan=False,
            data=self.df,
            indicators={
                "STOCH": {
                    "fastk_period": k_window,
                    "slowk_period": d_window,
                    "slowd_period": d_window,
                }
            },
        )
        k = self._extract_indicator(result, "slowk")
        d = self._extract_indicator(result, "slowd")
        return k, d

    def cci(self, window: Union[int, List[int]] = 20) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI12: Commodity Channel Index (CCI).
        Measures the current price level relative to an average price level over time.
        """
        if isinstance(window, list):
            return [self._single_cci(w) for w in window]
        return self._single_cci(window)

    def _single_cci(self, window: int) -> np.ndarray:
        """Helper for CCI."""
        result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"CCI": {"timeperiod": window}}
        )
        return self._extract_indicator(result, "cci")

    def williams_r(self, window: Union[int, List[int]] = 14) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI13: Williams %R.
        A momentum indicator that measures overbought/oversold levels.
        Identical interpretation to Slow Stochastic but on -100 to 0 scale.
        """
        if isinstance(window, list):
            return [self._single_williams_r(w) for w in window]
        return self._single_williams_r(window)

    def _single_williams_r(self, window: int) -> np.ndarray:
        """Helper for Williams %R."""
        result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"WILLR": {"timeperiod": window}}
        )
        return self._extract_indicator(result, "willr")

    def roc(self, window: Union[int, List[int]] = 10) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI14: Rate of Change (ROC).
        Percentage change in price between the current price and the price N periods ago.
        """
        if isinstance(window, list):
            return [self._single_roc(w) for w in window]
        return self._single_roc(window)

    def _single_roc(self, window: int) -> np.ndarray:
        """Helper for ROC."""
        result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"ROC": {"timeperiod": window}}
        )
        return self._extract_indicator(result, "roc")

    def mfi(self, window: Union[int, List[int]] = 14) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI15: Money Flow Index (MFI).
        A volume-weighted RSI. Uses both price and volume to measure buying/selling pressure.
        """
        if isinstance(window, list):
            return [self._single_mfi(w) for w in window]
        return self._single_mfi(window)

    def _single_mfi(self, window: int) -> np.ndarray:
        """Helper for MFI."""
        result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"MFI": {"timeperiod": window}}
        )
        return self._extract_indicator(result, "mfi")

    def awesome_osc(self, fast_window: int = 5, slow_window: int = 34) -> np.ndarray:
        """
        TI16: Awesome Oscillator (AO).
        The difference between a 5-period and a 34-period SMA of the bar's midpoint.
        Highlights market momentum.
        """
        median_price = (self.high + self.low) / 2
        
        # Internal vectorized SMA tool
        def _simple_sma(arr, w):
            res = np.full(len(arr), np.nan)
            if len(arr) < w: return res
            # Use rolling mean logic
            for i in range(w, len(arr) + 1):
                res[i - 1] = np.mean(arr[i - w : i])
            return res

        sma_fast = _simple_sma(median_price, fast_window)
        sma_slow = _simple_sma(median_price, slow_window)
        return sma_fast - sma_slow

    def ultimate_osc(self, t1: int = 7, t2: int = 14, t3: int = 28) -> np.ndarray:
        """
        TI17: Ultimate Oscillator.
        Combines three different timeframes to minimize false divergence signals.
        """
        result, _ = calculate_indicators(
            drop_nan=False,
            data=self.df,
            indicators={
                "ULTOSC": {"timeperiod1": t1, "timeperiod2": t2, "timeperiod3": t3}
            },
        )
        return self._extract_indicator(result, "ultosc")

    # =========================================================================
    # VOLATILITY-BASED INDICATORS
    # =========================================================================

    def bb_percent_b(self, window: Union[int, List[int]] = 20, 
                     nbdev: float = 2.0) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI18: Bollinger %B.
        Quantifies price position within the bands. 
        1.0 = Upper Band, 0.0 = Lower Band.
        """
        if isinstance(window, list):
            return [self._single_bb_percent_b(w, nbdev) for w in window]
        return self._single_bb_percent_b(window, nbdev)

    def _single_bb_percent_b(self, window: int, nbdev: float) -> np.ndarray:
        """Helper for Bollinger %B."""
        result, _ = calculate_indicators(
            drop_nan=False,
            data=self.df,
            indicators={
                "BBANDS": {"timeperiod": window, "nbdevup": nbdev, "nbdevdn": nbdev}
            },
        )
        upper = self._extract_indicator(result, "upper")
        lower = self._extract_indicator(result, "lower")

        with np.errstate(divide="ignore", invalid="ignore"):
            percent_b = np.where((upper - lower) > self.eps, (self.close - lower) / (upper - lower), np.nan)
        return percent_b

    def bb_bandwidth(self, window: Union[int, List[int]] = 20, 
                     nbdev: float = 2.0) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI19: Bollinger Bandwidth.
        The relative width of the bands. (Upper - Lower) / Middle.
        Low bandwidth indicates low volatility or 'squeeze'.
        """
        if isinstance(window, list):
            return [self._single_bb_bandwidth(w, nbdev) for w in window]
        return self._single_bb_bandwidth(window, nbdev)

    def _single_bb_bandwidth(self, window: int, nbdev: float) -> np.ndarray:
        """Helper for BB Bandwidth."""
        result, _ = calculate_indicators(
            drop_nan=False,
            data=self.df,
            indicators={
                "BBANDS": {"timeperiod": window, "nbdevup": nbdev, "nbdevdn": nbdev}
            },
        )
        upper = self._extract_indicator(result, "upper")
        lower = self._extract_indicator(result, "lower")
        middle = self._extract_indicator(result, "middle")

        with np.errstate(divide="ignore", invalid="ignore"):
            bandwidth = np.where(np.abs(middle) > self.eps, (upper - lower) / middle, np.nan)
        return bandwidth

    def bb_squeeze(self, window: Union[int, List[int]] = 20, nbdev: float = 2.0, 
                   keltner_mult: float = 1.5) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI20: BB Squeeze.
        Ratio of Bollinger Band width to Keltner Channel width.
        Values < 1.0 imply a 'squeeze' state (volatility compression).
        """
        if isinstance(window, list):
            return [self._single_bb_squeeze(w, nbdev, keltner_mult) for w in window]
        return self._single_bb_squeeze(window, nbdev, keltner_mult)

    def _single_bb_squeeze(self, window: int, nbdev: float, keltner_mult: float) -> np.ndarray:
        """Helper for BB Squeeze."""
        bb_width = self._single_bb_bandwidth(window, nbdev)

        # Need ATR for Keltner Channel width
        atr_result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"ATR": {"timeperiod": window}}
        )
        atr = self._extract_indicator(atr_result, "atr")
        kc_width = 2 * keltner_mult * atr

        with np.errstate(divide="ignore", invalid="ignore"):
            squeeze = np.where(kc_width > self.eps, bb_width / kc_width, np.nan)
        return squeeze

    def keltner_upper(self, ema_window: Union[int, List[int]] = 20, 
                      atr_window: int = 10, mult: float = 2.0) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI21: Keltner Channel Upper Band.
        Volatility-based channel where the middle is an EMA and bands are offset by ATR.
        """
        if isinstance(ema_window, list):
            return [self._single_keltner_upper(w, atr_window, mult) for w in ema_window]
        return self._single_keltner_upper(ema_window, atr_window, mult)

    def _single_keltner_upper(self, ema_window: int, atr_window: int, mult: float) -> np.ndarray:
        """Helper for Keltner Upper."""
        ema = self._single_ema(ema_window)
        # Calculate ATR for the offset
        atr_result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"ATR": {"timeperiod": atr_window}}
        )
        atr = self._extract_indicator(atr_result, "atr")
        return ema + (mult * atr)

    def donchian_upper(self, window: Union[int, List[int]] = 20) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI22: Donchian Channel Upper.
        The highest High observed over the last N periods.
        """
        if isinstance(window, list):
            return [self._single_donchian_upper(w) for w in window]
        return self._single_donchian_upper(window)

    def _single_donchian_upper(self, window: int) -> np.ndarray:
        """Helper for Donchian Upper."""
        upper = np.full(self.n, np.nan)
        for i in range(window, self.n):
            upper[i] = np.max(self.high[i - window : i])
        return upper

    def choppiness_index(self, window: Union[int, List[int]] = 14) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI23: Choppiness Index.
        Scale of 0-100 measuring if the market is trending (low) or range-bound (high).
        Formula uses ATR and price range.
        """
        if isinstance(window, list):
            return [self._single_choppiness_index(w) for w in window]
        return self._single_choppiness_index(window)

    def _single_choppiness_index(self, window: int) -> np.ndarray:
        """Helper for Choppiness Index."""
        # Calculate single-period True Range (TR)
        tr_result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"ATR": {"timeperiod": 1}}
        )
        tr = self._extract_indicator(tr_result, "atr")
        chop = np.full(self.n, np.nan)

        for i in range(window, self.n):
            sum_tr = np.sum(tr[i - window + 1 : i + 1])
            max_h, min_l = np.max(self.high[i - window + 1 : i + 1]), np.min(self.low[i - window + 1 : i + 1])
            range_hl = max_h - min_l

            if range_hl > self.eps and sum_tr > self.eps:
                x = sum_tr / range_hl
                if x > 0:
                    chop[i] = 100 * np.log10(x) / np.log10(window)
        return chop

    # =========================================================================
    # ADDITIONAL INDICATORS
    # =========================================================================

    def atr(self, window: Union[int, List[int]] = 14) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI24: Average True Range (ATR).
        The standard measure of market volatility, accounting for gaps.
        """
        if isinstance(window, list):
            return [self._single_atr(w) for w in window]
        return self._single_atr(window)

    def _single_atr(self, window: int) -> np.ndarray:
        """Helper for ATR."""
        result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"ATR": {"timeperiod": window}}
        )
        return self._extract_indicator(result, "atr")

    def obv(self) -> np.ndarray:
        """
        TI25: On-Balance Volume (OBV).
        Relates volume flow to price changes.
        """
        result, _ = calculate_indicators(
            drop_nan=False, data=self.df, indicators={"OBV": {}}
        )
        return self._extract_indicator(result, "obv")

    def vwap(self) -> np.ndarray:
        """
        TI26: VWAP (Simplified).
        Volume Weighted Average Price. This version is cumulative for the entire series.
        """
        typical_price = (self.high + self.low + self.close) / 3
        # VWAP = Sum(Price * Volume) / Sum(Volume)
        vwap = np.cumsum(typical_price * self.volume) / (np.cumsum(self.volume) + self.eps)
        return vwap

    def volume_ratio(self, window: Union[int, List[int]] = 20) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI27: Volume Ratio.
        Ratio of current volume to its moving average. Indicates relative activity levels.
        """
        if isinstance(window, list):
            return [self._single_volume_ratio(w) for w in window]
        return self._single_volume_ratio(window)

    def _single_volume_ratio(self, window: int) -> np.ndarray:
        """Helper for Volume Ratio."""
        vol_sma = self._single_volume_sma(window)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(vol_sma > self.eps, self.volume / vol_sma, np.nan)
        return ratio

    def volume_sma(self, window: Union[int, List[int]] = 20) -> Union[np.ndarray, List[np.ndarray]]:
        """
        TI28: Volume SMA.
        Simple Moving Average of volume over N periods.
        """
        if isinstance(window, list):
            return [self._single_volume_sma(w) for w in window]
        return self._single_volume_sma(window)

    def _single_volume_sma(self, window: int) -> np.ndarray:
        """Helper for Volume SMA."""
        volume_sma = np.full(self.n, np.nan)
        for i in range(window, self.n):
            volume_sma[i] = np.mean(self.volume[i - window : i])
        return volume_sma
