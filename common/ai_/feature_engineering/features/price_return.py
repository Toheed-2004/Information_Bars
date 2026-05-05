"""
Price & Return Features (Category: PR)

This module provides the PriceReturnFeatures class, which calculates various
price-based and return-based features using NumPy for high performance.
Features include arithmetic returns, log returns, higher moments (skewness/kurtosis),
and path geometry metrics like fractal dimension and curvature.
"""

import numpy as np
from bitpredict.common.logging import get_logger

# Initialize logger for tracking execution and potential issues
logger = get_logger(__name__)


class PriceReturnFeatures:
    """
    Calculates various price and return-based features for financial time series.
    
    This class implements a suite of features derived from price movement,
    ranging from basic returns to advanced path geometry and statistical moments.
    All calculations are performed using vectorized NumPy operations where possible.

    Attributes:
        close (np.ndarray): Array of closing prices.
        volume (np.ndarray): Array of volumes.
        n (int): Length of the price series.
        eps (float): Small constant to avoid division by zero.
    """
    
    def __init__(self, close: np.ndarray, volume: np.ndarray, n: int, eps: float):
        """
        Initializes the PriceReturnFeatures calculator.

        Args:
            close (np.ndarray): Array of closing prices.
            volume (np.ndarray): Array of trading volumes.
            n (int): Total number of data points.
            eps (float): Epsilon value for numerical stability (e.g., 1e-10).
        """
        self.close = close
        self.volume = volume
        self.n = n
        self.eps = eps
    
    # =========================================================================
    # BASIC RETURNS
    # =========================================================================

    def _shift_array(self, arr: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Helper: Shifts an array by k periods to remove lookahead bias or align data.

        Args:
            arr (np.ndarray): The array to shift.
            k (int): Number of periods to shift forward.

        Returns:
            np.ndarray: Shifted array with NaNs at the beginning.
        """
        out = np.full_like(arr, np.nan)
        out[k:] = arr[:-k]
        return out

    def returns_arithmetic(self) -> np.ndarray:
        """
        PR01: Arithmetic returns.
        Formula: R_t = (P_t / P_{t-1}) - 1
        
        Note: Shifted by 1 to ensure that at time t, the feature represents 
        the return that occurred from t-1 to t.
        """
        # Handle non-positive prices by adding 1 if necessary (avoiding div by zero)
        if np.any(self.close <= 0):
            shifted_close = self.close + 1
        else:
            shifted_close = self.close
        
        returns = np.full(self.n, np.nan)
        # Calculate percent change: (P_t / P_{t-1}) - 1
        returns[1:] = (shifted_close[1:] / shifted_close[:-1]) - 1
        
        # Remove lookahead bias: aligns return from (t-1 -> t) to timestamp t
        returns = self._shift_array(returns, 1)
        return returns
    
    def returns_log(self, shift = True) -> np.ndarray:
        """
        PR02: Logarithmic returns.
        Formula: r_t = log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
        
        Log returns are preferred in financial analysis for their time-additivity
        and better statistical properties (closer to normal distribution).
        """
        if np.any(self.close <= 0):
            shifted_close = self.close + 1
        else:
            shifted_close = self.close
        
        # Calculate natural log of prices
        log_prices = np.log(shifted_close)
        returns = np.full(self.n, np.nan)
        # Compute difference in log prices
        returns[1:] = log_prices[1:] - log_prices[:-1]
        if shift:
            #  Shift to remove lookahead bias: aligns return from t-1 -> t to timestamp t
            returns = self._shift_array(returns, 1)
        return returns
    
    def returns_signed(self) -> np.ndarray:
        """
        PR03: Signed returns.
        Returns the logarithmic return preserved with its original sign.
        Formula: sign(r_t) * |r_t|
        """
        returns = self.returns_log()
        # Preserve the direction (sign) while potentially transforming magnitude
        signed = np.sign(returns) * np.abs(returns)
        return signed
    
    def returns_absolute(self) -> np.ndarray:
        """
        PR04: Absolute returns.
        Magnitude of return regardless of direction. Formula: |r_t|
        Often used as a proxy for realized volatility or market activity.
        """
        returns = self.returns_log()
        abs_returns = np.abs(returns)
        return abs_returns
    
    def returns_squared(self) -> np.ndarray:
        """
        PR05: Squared returns.
        Formula: (r_t)^2
        Used as a proxy for variance and to capture high-volatility events.
        """
        returns = self.returns_log()
        squared = returns ** 2
        return squared
    
    # =========================================================================
    # MULTI-PERIOD RETURNS
    # =========================================================================
    
    def _returns_multiperiod_log(self, periods: int) -> np.ndarray:
        """
        Internal Helper: Calculates log returns over a specified number of periods.
        
        Args:
            periods (int): The number of periods back for the return calculation.

        Returns:
            np.ndarray: Log returns over 'periods'.
        """
        if np.any(self.close <= 0):
            shifted_close = self.close + 1
        else:
            shifted_close = self.close
        
        log_prices = np.log(shifted_close)
        returns = np.full(self.n, np.nan)
        # Difference between current log price and log price 'periods' ago
        returns[periods:] = log_prices[periods:] - log_prices[:-periods]
        
        return returns
    
    def returns_5min(self) -> np.ndarray:
        """PR06: 5-minute log returns."""
        return self._returns_multiperiod_log(5)
    
    def returns_15min(self) -> np.ndarray:
        """PR07: 15-minute log returns."""
        return self._returns_multiperiod_log(15)
    
    def returns_1hr(self) -> np.ndarray:
        """PR08: 1-hour log returns (assuming 1-minute bars)."""
        return self._returns_multiperiod_log(60)
    
    def returns_4hr(self) -> np.ndarray:
        """PR09: 4-hour log returns (assuming 1-minute bars)."""
        return self._returns_multiperiod_log(240)
    
    def returns_1d(self) -> np.ndarray:
        """PR10: Daily log returns (assuming 1-minute bars)."""
        return self._returns_multiperiod_log(1440)
    
    def returns_forward_1(self) -> np.ndarray:
        """
        PR11: Target Return (t-1 to t).
        Note: This is strictly the return calculated from (t-1) to (t),
        aligned exactly at timestamp t. Used for standard prediction targets.
        """
        if np.any(self.close <= 0):
            shifted_close = self.close + 1
        else:
            shifted_close = self.close
        
        log_prices = np.log(shifted_close)
        returns = np.full(self.n, np.nan)
        returns[:-1] = log_prices[1:] - log_prices[:-1]

        # Shift to ensure p_t - p_{t-1} is at index t.
        returns = self._shift_array(returns, 1)
        return returns
    
    def returns_forward_5(self, window: int = 5) -> np.ndarray:
        """
        PR12: Forward Return relative to a sliding average window.
        Calculates the difference between the next price and the mean of the current window.
        """
        result = np.full(self.n, np.nan)

        # Handle zero or negative values for log transformation
        shifted_close = np.where(self.close <= 0, self.close + 1, self.close)
        log_prices = np.log(shifted_close)

        for i in range(self.n - window):
            # Average log price over the current window
            current_avg = np.nanmean(log_prices[i:i+window])
            # Price at the step immediately following the window
            next_val = log_prices[i + window]
            # Difference representing the "forward" move relative to window mean
            result[i + window - 1] = next_val - current_avg  

        # Align to current time step (end of window)
        result = self._shift_array(result, 1)
        return result
    
    # =========================================================================
    # HIGHER MOMENT & STATISTICAL RETURNS
    # =========================================================================
    
    def returns_cubed(self) -> np.ndarray:
        """
        PR13: Cubic returns. Formula: (r_t)^3
        Useful for capturing asymmetries and extreme tail events (skewness proxy).
        """
        returns = self.returns_log()
        cubed = returns ** 3
        return cubed
    
    def returns_quartic(self) -> np.ndarray:
        """
        PR14: Quartic returns. Formula: (r_t)^4
        Highly sensitive to outliers and fat tails (kurtosis proxy).
        """
        returns = self.returns_log()
        quartic = returns ** 4
        return quartic
    
    def gain_loss_ratio(self, window: int = 20) -> np.ndarray:
        """
        PR15: Gain/Loss Ratio.
        Measures the sum of positive returns relative to the sum of absolute negative returns
        over a rolling window. Higher values indicate bullish dominance.
        """
        returns = self.returns_log()
        ratio = np.full(self.n, np.nan)
        
        for i in range(window, self.n):
            window_returns = returns[i - window:i]
            gains = np.sum(window_returns[window_returns > 0])
            losses = np.sum(np.abs(window_returns[window_returns < 0]))
            
            # Avoid division by zero with epsilon
            if losses > self.eps:
                ratio[i] = gains / losses
            else:
                ratio[i] = np.nan
        
        return ratio
    
    def upside_downside_var(self, window: int = 20) -> np.ndarray:
        """
        PR16: Upside/Downside Variance Ratio.
        The ratio of the variance of positive returns to the variance of negative returns.
        Helps distinguish between 'good' volatility and 'bad' volatility.
        """
        returns = self.returns_log()
        ratio = np.full(self.n, np.nan)
        
        for i in range(window, self.n):
            window_returns = returns[i - window:i]
            upside = window_returns[window_returns > 0]
            downside = window_returns[window_returns < 0]
            
            # Compute variance if sufficient data points exist
            upside_var = np.var(upside) if len(upside) > 1 else np.nan
            downside_var = np.var(downside) if len(downside) > 1 else np.nan
            
            if not np.isnan(downside_var) and downside_var > self.eps:
                ratio[i] = upside_var / downside_var
            else:
                ratio[i] = np.nan
        
        return ratio
    
    # =========================================================================
    # PATH GEOMETRY FEATURES
    # =========================================================================
    
    def net_displacement_ratio(self, window: int = 20) -> np.ndarray:
        """
        PR17: Net Displacement Ratio (Efficiency Ratio).
        Ratio of net price change to total path length traveled.
        1.0 = Perfectly straight trend, 0.0 = Pure noise/mean reversion.
        """
        returns = self.returns_log(shift= False)
        ratio = np.full(self.n, np.nan)
        
        for i in range(window, self.n):
            window_returns = returns[i - window:i]
            # Direct distance from start to end of window
            net_change = np.sum(window_returns)
            # Total zig-zag distance covered
            total_movement = np.sum(np.abs(window_returns))
            
            if total_movement > self.eps:
                ratio[i] = net_change / total_movement
            else:
                ratio[i] = np.nan
        
        return ratio
    
    def fractal_dimension(self, window: int = 20) -> np.ndarray:
        """
        PR18: Fractal Dimension (Higuchi-inspired approximation).
        Measures the "roughness" or complexity of the price path.
        Range [1, 2]: 1 = smooth trend, 2 = maximally noisy/space-filling path.
        """
        # Work with absolute returns to measure path "length"
        returns = np.abs(self.returns_log(shift=False))
        fd = np.full(self.n, np.nan)
        
        for i in range(window, self.n):
            window_abs_ret = returns[i - window:i]
            total_movement = np.sum(window_abs_ret)
            
            if total_movement > self.eps:
                # Compare total movement to net displacement
                net_displacement = np.abs(np.sum(self.returns_log(shift=False)[i - window:i]))
                # Scaling factor representing complexity
                ratio = total_movement / (net_displacement + self.eps)
                # Map complexity to [1, 2] dimension scale
                fd[i] = 1 + np.log(ratio) / np.log(window)
            else:
                fd[i] = np.nan
        
        return fd
    
    def turning_point_density(self, window: int = 20) -> np.ndarray:
        """
        PR19: Turning Point Density.
        Percentage of bars that constitute an 'inflection point' (local max/min).
        High density suggests range-bound or highly choppy conditions.
        """
        density = np.full(self.n, np.nan)
        
        for i in range(window, self.n):
            window_close = self.close[i - window:i]
            turning_points = 0
            
            # Count points where the price direction switches
            for j in range(1, len(window_close) - 1):
                if ((window_close[j] > window_close[j - 1] and window_close[j] > window_close[j + 1]) or
                    (window_close[j] < window_close[j - 1] and window_close[j] < window_close[j + 1])):
                    turning_points += 1
            
            density[i] = turning_points / window
        
        return density
    
    def path_curvature(self, window: int = 20) -> np.ndarray:
        """
        PR20: Path Curvature.
        Approximates the second derivative (acceleration) of the price path.
        Uses a Savitzky-Golay inspired polynomial fitting approach.
        """
        curvature = np.full(self.n, np.nan)
        
        for i in range(window + 1, self.n):
            window_close = self.close[i - window:i + 1]
            
            # Use fixed 5-point window for localized curvature calculation
            if len(window_close) >= 5:
                # Coefficients for calculating 2nd derivative of a quadratic fit
                coeffs = np.array([2, 1, 0, -1, -2])
                # Dot product to estimate rate of change of the slope
                curvature[i] = np.sum(coeffs * window_close[-5:]) / (window / 2)
            else:
                curvature[i] = np.nan
        
        return curvature
    
    def autocorr_decay_rate(self, window: int = 20) -> np.ndarray:
        """
        PR21: Autocorrelation Decay Rate.
        Estimates how quickly the correlation between current and past returns fades.
        High decay = efficient market; Low decay = strong momentum or persistence.
        """
        returns = self.returns_log()
        decay_rate = np.full(self.n, np.nan)
        
        for i in range(window * 2, self.n):
            window_returns = returns[i - window * 2:i]
            
            # Calculate Autocorrelation Function (ACF) for lags 1, 2, 3
            acf_vals = []
            for lag in range(1, min(4, len(window_returns))):
                # Pearson-style correlation for the given lag
                numerator = np.sum(
                    (window_returns[:-lag] - np.mean(window_returns)) *
                    (window_returns[lag:] - np.mean(window_returns))
                )
                denominator = np.sum((window_returns - np.mean(window_returns)) ** 2)
                acf = numerator / (denominator + self.eps)
                acf_vals.append(acf)
            
            # Model decay as an exponential process: ACF(k) = exp(-lambda * k)
            if len(acf_vals) > 1 and acf_vals[0] > self.eps:
                # Solve for lambda (decay constant)
                decay_rate[i] = -np.log(
                    np.abs(acf_vals[1]) / (np.abs(acf_vals[0]) + self.eps) + self.eps
                )
            else:
                decay_rate[i] = np.nan
        
        return decay_rate
    
    def directional_persistence(self, window: int = 20) -> np.ndarray:
        """
        PR22: Directional Persistence (Trendiness).
        Similar to Net Displacement Ratio, measures orientation of path.
        Formula: Sum(Returns) / Sum(|Returns|)
        Range [-1, 1]: 1 = strong up trend, -1 = strong down trend.
        """
        returns = self.returns_log(shift=False)
        persistence = np.full(self.n, np.nan)
        
        for i in range(window, self.n):
            window_returns = returns[i - window:i]
            net_return = np.sum(window_returns)
            total_movement = np.sum(np.abs(window_returns))
            
            if total_movement > self.eps:
                persistence[i] = net_return / total_movement
            else:
                persistence[i] = np.nan
    
        return persistence
