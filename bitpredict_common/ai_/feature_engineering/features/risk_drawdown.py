"""
Risk & Drawdown Features (Category: RISK)

This module provides the RiskFeatures class, which calculates indicators 
related to asset risk management, including drawdown depth/duration, 
tail risk, and performance ratios (Sharpe, Sortino, Calmar).
"""

import numpy as np

class RiskFeatures:
    """
    Calculates downside risk and risk-adjusted return metrics.
    
    Includes:
    1. Drawdown Analysis: Peak-to-trough calculations and recovery metrics.
    2. Path Risk: Indicators like Ulcer Index and Pain Index that penalize duration.
    3. Performance Ratios: Standard measures (Sharpe) and downside-focused (Sortino, Calmar).
    4. Extremes: Tail ratios to identify asymmetric risk profiles.

    Attributes:
        close (np.ndarray): Price series.
        n (int): Total data points.
        eps (float): Numerical stability factor.
    """

    def __init__(self, close: np.ndarray, n: int, eps: float):
        """
        Initializes the RiskFeatures calculator.

        Args:
            close (np.ndarray): Asset prices.
            n (int): Length of data.
            eps (float): Stability constant.
        """
        self.close = close
        self.n = n
        self.eps = eps

    def _shift_array(self, arr: np.ndarray, k: int = 1) -> np.ndarray:
        """Helper: Shifts array forward by k to prevent lookahead bias in features."""
        out = np.full_like(arr, np.nan)
        out[k:] = arr[:-k]
        return out
    
    def _returns_log(self) -> np.ndarray:
        """Helper: Calculate log returns, shifted to prevent bias."""
        log_close = np.log(self.close + self.eps)
        returns = np.full(self.n, np.nan)
        returns[1:] = log_close[1:] - log_close[:-1]
        
        # Shift back once to ensure we only use information available at t-1
        return returns

    # =========================================================================
    # DRAWDOWN METRICS
    # =========================================================================

    def max_drawdown(self, window: int = 100) -> np.ndarray:
        """
        RISK01: Maximum Drawdown (MDD).
        Formula: Min( (Price - Peak) / Peak ) over window.
        Measures the worst historical loss an investor would have faced.
        """
        mdd = np.full(self.n, np.nan)

        for i in range(window, self.n):
            price_w = self.close[i - window:i]
            # Running peak
            cummax = np.maximum.accumulate(price_w)
            # Drawdown series
            dd = (price_w - cummax) / (cummax + self.eps)
            mdd[i] = np.min(dd)

        return mdd

    def current_drawdown(self, window: int = 100) -> np.ndarray:
        """
        RISK02: Current Drawdown.
        Percentage drop from the rolling peak to the current observation.
        """
        curr_dd = np.full(self.n, np.nan)

        for i in range(window, self.n):
            peak = np.max(self.close[i-window:i])
            # Current price relative to recent peak
            curr_dd[i] = (self.close[i] - peak) / (peak + self.eps)
        
        return curr_dd

    def drawdown_duration(self, window: int = 100) -> np.ndarray:
        """
        RISK03: Drawdown Duration.
        Count of bars elapsed since the last 'High Water Mark' (Rolling Peak).
        Identifies period length of underwater positions.
        """
        duration = np.full(self.n, np.nan)

        for i in range(window, self.n):
            price_w = self.close[i - window:i]
            # Index of the maximum value
            peak_idx = np.argmax(price_w)
            # Distance from the end of window to peak
            duration[i] = (window - 1) - peak_idx

        return duration

    def ulcer_index(self, window: int = 100) -> np.ndarray:
        """
        RISK04: Ulcer Index.
        Formula: Sqrt( Mean( ( (Price - Peak) / Peak )^2 ) )
        Squares drawdowns to penalize deep and long-lasting losses heavily.
        """
        ulcer = np.full(self.n, np.nan)

        for i in range(window, self.n):
            price_w = self.close[i - window:i]
            cummax = np.maximum.accumulate(price_w)
            dd = (price_w - cummax) / (cummax + self.eps)
            # Sum of squared negative drawdowns
            ulcer[i] = np.sqrt(np.mean(dd**2))

        return ulcer

    def pain_index(self, window: int = 100) -> np.ndarray:
        """
        RISK05: Pain Index.
        Formula: Mean of all negative drawdowns.
        Aggregates constant 'suffering' relative to price peaks.
        """
        pain = np.full(self.n, np.nan)

        for i in range(window, self.n):
            price_w = self.close[i - window:i]
            cummax = np.maximum.accumulate(price_w)
            dd = (price_w - cummax) / (cummax + self.eps)
            # Subset of values where we are actually in a drawdown
            neg_dd = dd[dd < 0]
            pain[i] = np.mean(neg_dd) if len(neg_dd) > 0 else 0.0
                
        return pain

    def recovery_slope(self, window: int = 100, min_dd: float = -0.05) -> np.ndarray:
        """
        RISK06: Post-Drawdown Recovery Slope.
        Measures the steepness of the recovery once the trough is hit.
        Returns the slope coefficient (m) of the recovery path.
        """
        rec_slope = np.full(self.n, np.nan)

        for i in range(window, self.n):
            price_w = self.close[i - window:i]
            cummax = np.maximum.accumulate(price_w)
            dd = (price_w - cummax) / (cummax + self.eps)

            # Only calculate if the drawdown was meaningful (e.g. > 5%)
            if np.min(dd) < min_dd:
                trough_idx = np.argmin(dd)
                # Ensure we have enough data points after the trough to calculate slope
                if trough_idx < len(dd) - 2:
                    recovery_path = dd[trough_idx:]
                    x = np.arange(len(recovery_path))
                    # Linear regression slope: m
                    rec_slope[i] = np.polyfit(x, recovery_path, 1)[0]
        
        return rec_slope

    # =========================================================================
    # RISK-ADJUSTED RETURNS
    # =========================================================================

    def sharpe_ratio(self, window: int = 60, rf_rate: float = 0.0) -> np.ndarray:
        """
        RISK07: Annualized Sharpe Ratio.
        Formula: (Mean Return - Rf) / StdDev.
        Standard measure of return per unit of total risk.
        """
        returns = self._returns_log()
        sharpe = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_w = returns[i - window:i]
            std_r = np.nanstd(ret_w)
            if std_r > self.eps:
                # Annualization factor: Sqrt(252) for daily data
                sharpe[i] = (np.nanmean(ret_w) - rf_rate) / std_r * np.sqrt(252)
        sharpe = self._shift_array(sharpe, 1)
        return sharpe

    def sortino_ratio(self, window: int = 60, rf_rate: float = 0.0) -> np.ndarray:
        """
        RISK08: Annualized Sortino Ratio.
        Formula: (Mean Return - Rf) / Downside Deviation.
        Better than Sharpe for asymmetric returns as it ignores 'upside risk'.
        """
        returns = self._returns_log()
        sortino = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_w = returns[i - window:i]
            # Standard deviation calculated only on negative returns
            down_ret = ret_w[ret_w < 0]
            if len(down_ret) > 1:
                down_vol = np.std(down_ret)
                if down_vol > self.eps:
                    sortino[i] = (np.mean(ret_w) - rf_rate) / down_vol * np.sqrt(252)

        return sortino
    def calmar_ratio(self, window: int = 100, rf_rate: float = 0.0) -> np.ndarray:
        """
        RISK09: Calmar Ratio.
        Formula: Annualized Return / Max Drawdown.
        Relates portfolio performance specifically to tail risk (drawdowns).
        """
        returns = self._returns_log()
        calmar = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_w = returns[i - window:i]
            ann_ret = np.nanmean(ret_w) * 252

            price_w = self.close[i - window:i]
            peak = np.maximum.accumulate(price_w)
            mdd = np.abs(np.min((price_w - peak) / (peak + self.eps)))

            if mdd > self.eps:
                calmar[i] = (ann_ret - rf_rate) / mdd

        return calmar


    def omega_ratio(self, window: int = 100, mar: float = 0.0) -> np.ndarray:
        """
        RISK10: Omega Ratio.
        Ratio of the integral of gains vs integral of losses relative to MAR.
        Captures the entire shape of the return distribution.
        """
        returns = self._returns_log()
        omega = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_w = returns[i - window:i]
            # Integral of gains relative to MAR
            up = np.sum(np.maximum(ret_w - mar, 0))
            # Integral of losses relative to MAR
            down = np.sum(np.maximum(mar - ret_w, 0))

            if down > self.eps:
                omega[i] = up / down
            else:
                omega[i] = np.inf if up > 0 else 1.0

        return omega


    def tail_ratio_95_5(self, window: int = 100) -> np.ndarray:
        """
        RISK11: Multi-period Tail Ratio.
        Formula: 95th Percentile Gain / |5th Percentile Loss|.
        Measures the skewness of outcomes in the extreme tails.
        """
        returns = self._returns_log()
        tail = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_w = returns[i - window:i]
            q95, q05 = np.nanpercentile(ret_w, 95), np.nanpercentile(ret_w, 5)

            if abs(q05) > self.eps:
                tail[i] = q95 / abs(q05)
            else:
                tail[i] = np.nan  # or 1.0 if you prefer

        return tail
