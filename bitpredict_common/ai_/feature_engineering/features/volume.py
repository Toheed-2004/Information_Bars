"""
Volume & Liquidity Features (Category: V)

This module provides the VolumeFeatures class, which calculates volume-based
indicators and liquidity proxies for financial time series. It covers basic 
volume momentum, standardized volume surprises, volume-price relationships, 
and advanced liquidity estimators like Amihud illiquidity and Roll spread.
"""

import numpy as np
import pandas as pd
from bitpredict.common.ta.indicators.base import calculate_indicators

class VolumeFeatures:
    """
    Calculates various volume-based features and market liquidity proxies.
    
    This class leverages both custom NumPy implementations and the project's
    internal Technical Analysis (TA) infrastructure to generate features 
    representing participation, conviction, and ease of trading.

    Attributes:
        df (pd.DataFrame): DataFrame containing OHLCV data.
        open, high, low, close (np.ndarray): Price series data.
        volume (np.ndarray): Array of trading volumes.
        n (int): Length of the data series.
        eps (float): Small constant for numerical stability.
    """

    def __init__(self, df: pd.DataFrame, open: np.ndarray, high: np.ndarray, 
                 low: np.ndarray, close: np.ndarray, volume: np.ndarray, 
                 n: int, eps: float):
        """
        Initializes the VolumeFeatures calculator.

        Args:
            df (pd.DataFrame): The source DataFrame.
            open, high, low, close (np.ndarray): Price arrays.
            volume (np.ndarray): Volume array.
            n (int): Total number of samples.
            eps (float): Epsilon for numerical stability.
        """
        self.df = df
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.n = n
        self.eps = eps

    def _extract_indicator(self, result: pd.DataFrame, indicator_name: str) -> np.ndarray:
        """
        Helper: Extracts specific indicator values from a result DataFrame.
        
        Args:
            result (pd.DataFrame): The DataFrame returned by TA functions.
            indicator_name (str): Substring to match in column names.

        Returns:
            np.ndarray: Extracted values or NaNs if not found.
        """
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                if indicator_name.lower() in col.lower():
                    return result[col].values
            # Fallback to first column if no name match
            if len(result.columns) > 0:
                return result.iloc[:, 0].values
        return np.full(self.n, np.nan)

    def _shift_array(self, arr: np.ndarray, k: int = 1) -> np.ndarray:
        """Helper: Shift forward by k to remove lookahead bias."""
        out = np.full_like(arr, np.nan)
        out[k:] = arr[:-k]
        return out
    
    # =========================================================================
    # VOLUME-BASED MOMENTUM & STATS
    # =========================================================================

    def volume_roc(self, window: int = 5) -> np.ndarray:
        """
        V01: Volume Rate of Change (ROC).
        Formula: (Vol_t / Vol_{t-n}) - 1
        Measures the percentage change in volume over a fixed horizon.
        """
        roc = np.full(self.n, np.nan)
        for i in range(window, self.n):
            if self.volume[i - window] > self.eps:
                roc[i] = (self.volume[i] / self.volume[i - window]) - 1
        return roc

    def volume_surprise(self, window: int = 20) -> np.ndarray:
        """
        V02: Volume Surprise (Z-Score).
        Formula: (Current Vol - Mean Vol) / Std Dev Vol
        Quantifies how unusual the current volume is relative to its recent distribution.
        """
        surprise = np.full(self.n, np.nan)

        for i in range(window, self.n):
            vol_window = self.volume[i - window:i]
            mean_vol = np.mean(vol_window)
            std_vol = np.std(vol_window)

            if std_vol > self.eps:
                surprise[i] = (self.volume[i] - mean_vol) / std_vol

        return surprise

    def volume_momentum(self, window: int = 20) -> np.ndarray:
        """
        V03: Volume Momentum.
        Formula: Current Vol - Vol_{t-n}
        Simple absolute difference in volume levels.
        """
        momentum = np.full(self.n, np.nan)
        for i in range(window, self.n):
            momentum[i] = self.volume[i] - self.volume[i - window]
        return momentum

    def volume_ema_ratio(self, fast_window: int = 5, slow_window: int = 20) -> np.ndarray:
        """
        V04: Volume EMA Ratio (Fast / Slow).
        Ratio of short-term average volume to long-term average volume.
        Ratios > 1.0 indicate expanding participation.
        """
        ratio = np.full(self.n, np.nan)
        ema_fast = np.full(self.n, np.nan)
        ema_slow = np.full(self.n, np.nan)

        alpha_fast = 2 / (fast_window + 1)
        alpha_slow = 2 / (slow_window + 1)

        ema_fast[0] = self.volume[0]
        ema_slow[0] = self.volume[0]

        for i in range(1, self.n):
            ema_fast[i] = alpha_fast * self.volume[i] + (1 - alpha_fast) * ema_fast[i - 1]
            ema_slow[i] = alpha_slow * self.volume[i] + (1 - alpha_slow) * ema_slow[i - 1]

            if ema_slow[i] > self.eps:
                ratio[i] = ema_fast[i] / ema_slow[i]

        return ratio

    def volume_skew(self, window: int = 20) -> np.ndarray:
        """
        V05: Volume Skewness.
        Measures the asymmetry of the volume distribution over a window.
        High skewness suggests presence of extreme volume 'shocks'.
        """
        skew = np.full(self.n, np.nan)
        for i in range(window, self.n):
            vol_window = self.volume[i - window:i]
            mean_vol = np.mean(vol_window)
            std_vol = np.std(vol_window)
            if std_vol > self.eps:
                # 3rd moment standardized
                skew[i] = np.mean(((vol_window - mean_vol) / std_vol) ** 3)
        return skew

    def volume_kurtosis(self, window: int = 20) -> np.ndarray:
        """
        V06: Volume Excess Kurtosis.
        Measures the 'tail-heaviness' of the volume distribution.
        Indicates frequency and severity of volume outliers.
        """
        kurt = np.full(self.n, np.nan)
        for i in range(window, self.n):
            vol_window = self.volume[i - window:i]
            mean_vol = np.mean(vol_window)
            std_vol = np.std(vol_window)
            if std_vol > self.eps:
                # 4th moment standardized (minus 3 for excess)
                kurt[i] = np.mean(((vol_window - mean_vol) / std_vol) ** 4) - 3
        return kurt

    # =========================================================================
    # VOLUME-PRICE RELATIONSHIP
    # =========================================================================

    def volume_price_corr(self, window: int = 20) -> np.ndarray:
        """
        V07: Volume-Return Correlation.
        Rolling Pearson correlation between volume and absolute log returns.
        Measures link between activity and volatility.
        """
        # Calculate log returns first
        log_returns = np.log(self.close[1:] / (self.close[:-1] + self.eps))
        returns = np.concatenate([[np.nan], log_returns])
        abs_returns = np.abs(returns)

        corr = np.full(self.n, np.nan)
        for i in range(window, self.n):
            vol_window = self.volume[i - window:i]
            ret_window = abs_returns[i - window:i]

            mask = ~np.isnan(ret_window)
            if np.sum(mask) > 1:
                vol_valid, ret_valid = vol_window[mask], ret_window[mask]
                if np.std(vol_valid) > self.eps and np.std(ret_valid) > self.eps:
                    corr[i] = np.corrcoef(vol_valid, ret_valid)[0, 1]
        return corr

    def volume_weighted_return(self, window: int = 20) -> np.ndarray:
        """
        V08: Volume-Weighted Return.
        The window average of returns, weighted by the volume of each bar.
        Ensures returns with higher conviction (more volume) have more influence.
        """
        log_returns = np.log(self.close[1:] / (self.close[:-1] + self.eps))
        returns = np.concatenate([[np.nan], log_returns])

        vw_ret = np.full(self.n, np.nan)
        for i in range(window, self.n):
            vol_window, ret_window = self.volume[i - window:i], returns[i - window:i]

            valid = ~np.isnan(ret_window)
            vol_sum = np.sum(vol_window[valid])
            if np.sum(valid) > 0 and vol_sum > self.eps:
                vw_ret[i] = np.sum(vol_window[valid] * ret_window[valid]) / vol_sum
        return vw_ret

    def obv(self) -> np.ndarray:
        """
        V09: On-Balance Volume (OBV).
        Cumulative volume where volume is added if close > prev_close, 
        and subtracted if close < prev_close. Used to detect divergence.
        """
        result, _ = calculate_indicators(data=self.df, indicators={"OBV": {}}, drop_nan=False)
        obv_vals = self._extract_indicator(result, "obv")
        
        # Ensure array length matches input series
        if len(obv_vals) < self.n:
            obv_vals = np.concatenate([np.full(self.n - len(obv_vals), np.nan), obv_vals])
        elif len(obv_vals) > self.n:
            obv_vals = obv_vals[-self.n:]
            
        return obv_vals

    def obv_slope(self, window: int = 14) -> np.ndarray:
        """
        V10: OBV Slope.
        The slope of a linear regression line fitted to the OBV values.
        Identifies whether cumulative volume trend is positive or negative.
        """
        obv_vals = self.obv()
        slope = np.full(self.n, np.nan)

        for i in range(window, self.n):
            y = obv_vals[i - window:i]
            if not np.any(np.isnan(y)):
                # Fit linear regression poly1
                slope[i] = np.polyfit(np.arange(window), y, 1)[0]

        return slope

    def money_flow_multiplier(self) -> np.ndarray:
        """
        V11: Money Flow Multiplier (MFM).
        Position of close relative to the range: ((C - L) - (H - C)) / (H - L)
        Ranges from -1 to 1; measures buying/selling pressure within a single bar.
        """
        mfm = np.full(self.n, np.nan)
        for i in range(self.n):
            hl_range = self.high[i] - self.low[i]
            if hl_range > self.eps:
                mfm[i] = ((self.close[i] - self.low[i]) - (self.high[i] - self.close[i])) / hl_range
            else:
                mfm[i] = 0.0
        mfm = self._shift_array(mfm, 1)
        return mfm

    def adl(self) -> np.ndarray:
        """
        V12: Accumulation/Distribution Line (ADL).
        Cumulative sum of (MFM * Volume). Tracks the flow of money in/out of the asset.
        """
        result, _ = calculate_indicators(data=self.df, indicators={"AD": {}}, drop_nan=False)
        adl_vals = self._extract_indicator(result, "ad")
        
        if len(adl_vals) < self.n:
            adl_vals = np.concatenate([np.full(self.n - len(adl_vals), np.nan), adl_vals])
        elif len(adl_vals) > self.n:
            adl_vals = adl_vals[-self.n:]
            
        return adl_vals

    def chaikin_oscillator(self, fast_window: int = 3, slow_window: int = 10) -> np.ndarray:
        """
        V13: Chaikin Oscillator.
        Difference between 3-day and 10-day EMAs of the ADL.
        Measures the momentum of the accumulation/distribution line.
        """
        result, _ = calculate_indicators(
            data=self.df,
            indicators={"ADOSC": {"fastperiod": fast_window, "slowperiod": slow_window}},
            drop_nan=False,
        )
        adosc_vals = self._extract_indicator(result, "adosc")
        
        if len(adosc_vals) < self.n:
            adosc_vals = np.concatenate([np.full(self.n - len(adosc_vals), np.nan), adosc_vals])
        elif len(adosc_vals) > self.n:
            adosc_vals = adosc_vals[-self.n:]
            
        return adosc_vals

    # =========================================================================
    # LIQUIDITY PROXIES
    # =========================================================================

    def amihud_illiquidity(self, window: int = 20) -> np.ndarray:
        """
        V14: Amihud Illiquidity Ratio.
        Formula: Mean(|Return| / Volume)
        The price impact of one unit of volume. High values = low liquidity.
        """
        log_returns = np.log(self.close[1:] / (self.close[:-1] + self.eps))
        abs_returns = np.concatenate([[np.nan], np.abs(log_returns)])

        illiq = np.full(self.n, np.nan)
        for i in range(window, self.n):
            vol_window, ret_window = self.volume[i - window:i], abs_returns[i - window:i]

            valid = (vol_window > self.eps) & (~np.isnan(ret_window))
            if np.sum(valid) > 0:
                illiq[i] = np.mean(ret_window[valid] / vol_window[valid])
        return illiq

    def roll_spread(self, window: int = 5) -> np.ndarray:
        """
        V15: Roll Spread Estimator.
        Estimated Spread = 2 * sqrt(-Cov(dP_t, dP_{t-1}))
        Estimates the effective bid-ask spread from serial covariance of price changes.
        """
        price_changes = np.diff(self.close)
        spread = np.full(self.n, np.nan)

        for i in range(window + 1, self.n):
            dp_window = price_changes[i - window:i]
            if len(dp_window) > 1:
                # Calculate covariance between consecutive price changes
                lag1_cov = np.cov(dp_window[:-1], dp_window[1:])[0, 1]
                if lag1_cov < 0:
                    spread[i] = 2 * np.sqrt(-lag1_cov)
                else:
                    spread[i] = 0.0 # Covariance must be negative for valid spread estimate
        return spread

    def corwin_schultz_spread(self, window: int = 20) -> np.ndarray:
        """
        V16: Corwin-Schultz Spread.
        Estimates spread using only High and Low prices over two consecutive periods.
        More robust to noise than simple range-based measures.
        """
        hl_ratio = np.log((self.high + self.eps) / (self.low + self.eps))
        cs_spread = np.full(self.n, np.nan)

        for i in range(window, self.n):
            hl_window = hl_ratio[i - window:i]
            beta = np.sum(hl_window ** 2) / window
            if beta > self.eps:
                gamma = 0.04 * beta
                cs_spread[i] = 2 * (np.exp(gamma) - 1) / (1 + np.exp(gamma))
        return cs_spread

    def effective_tick_proxy(self, window: int = 20) -> np.ndarray:
        """
        V17: Effective Tick Size Proxy.
        Measures clusters of prices as a proxy for the actual discrete price steps.
        """
        tick_proxy = np.full(self.n, np.nan)
        for i in range(window, self.n):
            high_w, low_w, close_w = self.high[i-window:i], self.low[i-window:i], self.close[i-window:i]
            # Standard proxy for effective tick: distance from price midpoint
            tick_proxy[i] = np.mean(np.abs(2 * close_w - high_w - low_w))
        return tick_proxy

    def overnight_gap(self) -> np.ndarray:
        """
        V18: Overnight Gap Measure.
        Measures the absolute gap between Open_t and Close_{t-1}, normalized by High-Low.
        Captures informational leaps that occur when the market is closed or thin.
        """
        gap = np.full(self.n, np.nan)
        for i in range(1, self.n):
            hl_range = self.high[i] - self.low[i]
            if hl_range > self.eps:
                gap[i] = np.abs(self.open[i] - self.close[i - 1]) / hl_range
            else:
                gap[i] = 0.0
        return gap
