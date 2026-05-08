"""
Volatility Features & Regime Indicators (Category: VOL)

This module provides the VolatilityFeatures class, which implements various
volatility estimators (Range-based, Realized) and regime identification 
tools (Volatility of Volatility, Jump components, Tail indices).
"""

import numpy as np

class VolatilityFeatures:
    """
    Calculates advanced volatility indicators and market risk metrics.
    
    Includes:
    1. OHLC-based efficient estimators (Parkinson, Yang-Zhang).
    2. Realized volatility and term structure.
    3. Distribution shape features (Skewness, Semivariance).
    4. Jump components and Extreme Value Theory (EVT) indicators (VaR, Tail Index).

    Attributes:
        open, high, low, close, volume (np.ndarray): Price and volume series.
        n (int): Total number of data points.
        eps (float): Epsilon for numerical stability.
    """

    def __init__(self, open: np.ndarray, high: np.ndarray, low: np.ndarray, 
                 close: np.ndarray, volume: np.ndarray, n: int, eps: float):
        """
        Initializes the VolatilityFeatures calculator.

        Args:
            open, high, low, close (np.ndarray): OHLC prices.
            volume (np.ndarray): Trading volume.
            n (int): Length of data.
            eps (float): Stability constant.
        """
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.n = n
        self.eps = eps

    def _log_hl_ratio(self) -> np.ndarray:
        """Helper: log(high/low) ratios for range-based estimators."""
        return np.log((self.high + self.eps) / (self.low + self.eps))

    
    def _shift_array(self, arr: np.ndarray, k: int = 1) -> np.ndarray:
        """Helper: Shift forward by k to remove lookahead bias."""
        out = np.full_like(arr, np.nan)
        out[k:] = arr[:-k]
        return out

    def _log_co_ratio(self) -> np.ndarray:
        """Helper: log(close/open) ratios for intra-bar dynamics."""
        return np.log((self.close + self.eps) / (self.open + self.eps))

    def _log_hc_ratio(self) -> np.ndarray:
        """Helper: log(high/close) ratios."""
        return np.log((self.high + self.eps) / (self.close + self.eps))

    def _log_lc_ratio(self) -> np.ndarray:
        """Helper: log(low/close) ratios."""
        return np.log((self.close + self.eps) / (self.low + self.eps))

    def _true_range(self) -> np.ndarray:
        """Helper: Calculate True Range (TR) accounting for gaps."""
        tr = np.zeros(self.n)
        tr[0] = self.high[0] - self.low[0]

        for i in range(1, self.n):
            # TR = max(H-L, |H-Cp|, |L-Cp|)
            tr[i] = max(
                self.high[i] - self.low[i],
                abs(self.high[i] - self.close[i - 1]),
                abs(self.low[i] - self.close[i - 1]),
            )
        return tr

    def _returns_log(self) -> np.ndarray:
        """Helper: Calculate log returns (discrete approximation of continuous volatility)."""
        log_close = np.log(self.close + self.eps)
        returns = np.full(self.n, np.nan)
        returns[1:] = log_close[1:] - log_close[:-1]
        return returns

    # =========================================================================
    # OHLC-BASED VOLATILITY ESTIMATORS
    # =========================================================================

    def vol_parkinson(self, window: int = 20) -> np.ndarray:
        """
        VOL01: Parkinson Volatility.
        Formula: sqrt( (1 / (4 * ln(2) * n)) * sum( ln(H_i / L_i)^2 ) )
        Uses High/Low range. ~5x more efficient than close-to-close volatility.
        """
        log_hl = self._log_hl_ratio()
        parkinson = np.full(self.n, np.nan)

        coeff = 1 / (4 * np.log(2))
        for i in range(window, self.n):
            hl_window = log_hl[i - window:i]
            # Sum of squared log H/L ratios over window
            parkinson[i] = np.sqrt(coeff * np.mean(hl_window ** 2))
        return parkinson

    def vol_garman_klass(self, window: int = 20) -> np.ndarray:
        """
        VOL02: Garman-Klass Volatility.
        Refined version of Parkinson that includes Open and Close prices.
        Accounts for intra-bar price action more comprehensively.
        """
        log_hl = self._log_hl_ratio()
        log_co = self._log_co_ratio()
        gk = np.full(self.n, np.nan)

        for i in range(window, self.n):
            hl_w = log_hl[i - window:i]
            co_w = log_co[i - window:i]

            # Composite estimator using OHLC squared ratios
            term1 = 0.5 * np.mean(hl_w ** 2)
            term2 = (2 * np.log(2) - 1) * np.mean(co_w ** 2)
            gk[i] = np.sqrt(np.abs(term1 - term2))
        return gk

    def vol_yang_zhang(self, window: int = 20) -> np.ndarray:
        """
        VOL03: Yang-Zhang Volatility.
        The most efficient OHLC estimator; handles both opening gaps and drift.
        Combines Overnight, Open-Close, and Rogers-Satchell volatility.
        """
        log_hl = self._log_hl_ratio()
        log_co = self._log_co_ratio()
        log_hc = self._log_hc_ratio()
        log_lc = self._log_lc_ratio()

        yz = np.full(self.n, np.nan)

        for i in range(window, self.n):
            hl_w, co_w = log_hl[i - window:i], log_co[i - window:i]
            hc_w, lc_w = log_hc[i - window:i], log_lc[i - window:i]

            # 1. Rogers-Satchell (Drift independent)
            rs = np.mean(hc_w * (hc_w - co_w) + lc_w * (lc_w - co_w))
            # 2. Open-to-Close Variance
            v_oc = np.var(co_w)
            # 3. Overnight Variance (Gap)
            log_oo = np.log(
                    (self.open[i-window+1:i+1] + self.eps) /
                    (self.close[i-window:i] + self.eps)
                )
            v_overnight = np.var(log_oo)

            # Combined Yang-Zhang weighted sum
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            yz[i] = np.sqrt(v_overnight + k * v_oc + (1 - k) * rs)
        return yz

    def vol_rogers_satchell(self, window: int = 20) -> np.ndarray:
        """
        VOL04: Rogers-Satchell Volatility.
        Formula: Mean( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) )
        Superior to Parkinson/GK in the presence of strong price trends (drift).
        """
        hc = self._log_hc_ratio()
        lc = self._log_lc_ratio()
        co = self._log_co_ratio()
        
        rs = np.full(self.n, np.nan)
        for i in range(window, self.n):
            hc_w = hc[i-window:i]
            lc_w = lc[i-window:i]
            co_w = co[i-window:i]
            rs[i] = np.sqrt(np.abs(np.mean(hc_w * (hc_w - co_w) + lc_w * (lc_w - co_w))))
        return rs

    def atr(self, window: int = 14) -> np.ndarray:
        """
        VOL05: Average True Range (ATR).
        Exponentially weighted average of the True Range. 
        Measures total price excursion per candle, including overnight gaps.
        """
        tr = self._true_range()
        atr = np.full(self.n, np.nan)

        if window <= self.n:
            atr[window - 1] = np.mean(tr[:window])

        for i in range(window, self.n):
            # Wider's Smoothing: EMA with alpha = 1/window
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
        return atr

    def atr_normalized(self, window: int = 14) -> np.ndarray:
        """
        VOL06: Normalized ATR (NATR).
        ATR expressed as a percentage of the current price. 
        Allows comparison of volatility across different price levels or assets.
        """
        atr_vals = self.atr(window=window)
        normalized = (atr_vals / (self.close + self.eps)) * 100
        
        normalized = self._shift_array(normalized, 1)
        return normalized

    def realized_vol(self, window: int = 20) -> np.ndarray:
        """
        VOL07: Realized Volatility.
        The annualized standard deviation of log returns. 
        Baseline measure of market historical volatility.
        """
        returns = self._returns_log()
        rvol = np.full(self.n, np.nan)

        # 252 is typical trading days; scaling factor adjusts per candle frequency
        annual_scale = np.sqrt(252) 
        for i in range(window, self.n):
            rvol[i] = annual_scale * np.nanstd(returns[i - window:i])
        return rvol

    def vol_of_vol(self, vol_window: int = 20, outer_window: int = 20) -> np.ndarray:
        """
        VOL09: Volatility of Volatility (VoV).
        The standard deviation of realized volatility levels. 
        Quantifies the stability of the current volatility regime.
        """
        vols = self.realized_vol(window=vol_window)
        vov = np.full(self.n, np.nan)

        for i in range(vol_window + outer_window, self.n):
            vov[i] = np.nanstd(vols[i - outer_window:i])
        return vov

    # =========================================================================
    # VOLATILITY SHAPE & DISTRIBUTION
    # =========================================================================

    def vol_skew(self, window: int = 20) -> np.ndarray:
        """
        VOL13: Return Skewness.
        The 3rd standardized moment of returns. 
        Negative skew = frequent small gains and few large losses ('Left Tail').
        """
        returns = self._returns_log()
        skew = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_w = returns[i - window:i]
            std_r = np.nanstd(ret_w)
            if std_r > self.eps:
                skew[i] = np.nanmean(((ret_w - np.nanmean(ret_w)) / std_r) ** 3)
        return skew

    def vol_of_vol_of_vol(self, window: int = 20) -> np.ndarray:
        """
        VOL14: Cubic Volatility Dynamics.
        Measures higher-order 'acceleration' of the volatility of volatility.
        """
        vov = self.vol_of_vol(vol_window=5, outer_window=10)
        vovov = np.full(self.n, np.nan)
        for i in range(window, self.n):
            vovov[i] = np.nanstd(vov[i - window:i])
        return vovov

    def realized_semivariance_down(self, window: int = 20) -> np.ndarray:
        """
        VOL15: Downside Semivariance.
        Variation calculated only on negative returns. 
        Pure measure of downside risk, ignoring upside 'volatility'.
        """
        returns = self._returns_log()
        sv_down = np.full(self.n, np.nan)

        for i in range(window, self.n):
            r_w = returns[i - window:i]
            neg_r = r_w[r_w < 0]
            sv_down[i] = np.sum(neg_r ** 2) / window if len(neg_r) > 0 else 0.0
        return sv_down

    def realized_semivariance_up(self, window: int = 20) -> np.ndarray:
        """
        VOL16: Upside Semivariance.
        Variation calculated only on positive returns. 
        Measures the 'good' volatility associated with aggressive upward moves.
        """
        returns = self._returns_log()
        sv_up = np.full(self.n, np.nan)

        for i in range(window, self.n):
            r_w = returns[i - window:i]
            pos_r = r_w[r_w > 0]
            sv_up[i] = np.sum(pos_r ** 2) / window if len(pos_r) > 0 else 0.0
        return sv_up

    def asymmetry_index(self, window: int = 20) -> np.ndarray:
        """
        VOL17: Volatility Asymmetry Index.
        (Upside Vol - Downside Vol) / Total Vol.
        Indicates which direction is driving the current volatility surge.
        """
        returns = self._returns_log()
        asym = np.full(self.n, np.nan)

        for i in range(window, self.n):
            r_w = returns[i - window:i]
            u_vol = np.std(r_w[r_w > 0]) if np.sum(r_w > 0) > 1 else 0.0
            d_vol = np.std(r_w[r_w < 0]) if np.sum(r_w < 0) > 1 else 0.0
            t_vol = np.nanstd(r_w)

            if t_vol > self.eps:
                asym[i] = (u_vol - d_vol) / t_vol
        return asym

    def intrabar_vol_skew(self) -> np.ndarray:
        """
        VOL18: Intra-bar Volatility Skew.
        Ratio of Upper Shadow to Lower Shadow in log-space.
        High values = rejection of highs; Low values = rejection of lows.
        """
        log_hc = self._log_hc_ratio()
        log_lc = self._log_lc_ratio()

        skew = np.full(self.n, np.nan)
        for i in range(self.n):
            if log_lc[i] > self.eps:
                skew[i] = log_hc[i] / log_lc[i]
            else:
                skew[i] = 1.0 # Symmetric if no shadows
        skew = self._shift_array(skew, 1)
        return skew

    # =========================================================================
    # JUMP DETECTION & EXTREME VALUES
    # =========================================================================

    def bipower_variation(self, window: int = 20) -> np.ndarray:
        """
        VOL19: Bipower Variation (BPV).
        Jump-robust estimator of integrated variance. 
        Calculates volatility while filtering out large, discrete price jumps.
        """
        returns = self._returns_log()
        bpv = np.full(self.n, np.nan)

        # Coefficient (pi/2) ensures BPV converges to integrated variance
        coeff = np.pi / 2.0
        for i in range(window, self.n):
            abs_r = np.abs(returns[i - window:i])
            # Product of consecutive absolute returns filters jumps
            bpv[i] = coeff * np.mean(abs_r[:-1] * abs_r[1:])
        return bpv

    def jump_component(self, window: int = 20) -> np.ndarray:
        """
        VOL20: Jump Variation Component.
        Difference between Realized Variance and Bipower Variation.
        High values indicate that a large portion of volatility is due to jumps.
        """
        returns = self._returns_log()
        bpv = self.bipower_variation(window=window)
        
        rv = np.full(self.n, np.nan)
        for i in range(window, self.n):
            rv[i] = np.mean(returns[i - window:i] ** 2)

        # Only positive difference is considered jump component
        return np.maximum(0.0, rv - bpv)

    def z_score_jump(self, window: int = 20) -> np.ndarray:
        """
        VOL21: Jump Z-Score.
        Standardizes the current return by the recent rolling standard deviation.
        Detects statistically significant price outliers (potential jumps).
        """
        returns = self._returns_log()
        z_score = np.full(self.n, np.nan)

        for i in range(window, self.n):
            vol = np.nanstd(returns[i - window:i])
            if vol > self.eps:
                z_score[i] = np.abs(returns[i]) / vol
        return z_score

    def tail_index_hill(self, window: int = 100) -> np.ndarray:
        """
        VOL22: Hill Tail Index (α).
        Maximum Likelihood Estimator for the power-law tail of return distributions.
        Measures the 'fatness' of tails. α < 3 often implies infinite variance.
        """
        abs_ret = np.abs(self._returns_log())
        tail_idx = np.full(self.n, np.nan)

        for i in range(window, self.n):
            r_w = abs_ret[i - window:i]
            # Use top 10% as the extreme tail for Hill estimation
            thresh = np.nanpercentile(r_w, 90)
            extremes = r_w[r_w > thresh]

            if len(extremes) > 2:
                # α = 1 / mean( ln(X_i / thresh) )
                tail_idx[i] = 1.0 / np.mean(np.log(extremes / (thresh + self.eps)))
        return tail_idx

    def var_95(self, window: int = 100) -> np.ndarray:
        """
        VOL23: Value at Risk (VaR 95%).
        The return level such that there is only a 5% probability of a worse loss.
        Calculated using historical simulation with rolling window.
        """
        returns = self._returns_log()
        var95 = np.full(self.n, np.nan)

        for i in range(window, self.n):
            r_w = returns[i - window:i]
            if np.any(~np.isnan(r_w)):
                var95[i] = np.nanpercentile(r_w, 5)
        return var95

    def cvar_95(self, window: int = 100) -> np.ndarray:
        """
        VOL24: Conditional Value at Risk (CVaR / Expected Shortfall).
        The expected return given that the loss has exceeded the 95% VaR level.
        Captures the 'tail of the tail' risk.
        """
        returns = self._returns_log()
        cvar95 = np.full(self.n, np.nan)

        for i in range(window, self.n):
            r_w = returns[i - window:i]
            if np.any(~np.isnan(r_w)):
                var_level = np.nanpercentile(r_w, 5)
                tail_events = r_w[r_w <= var_level]
                cvar95[i] = np.mean(tail_events) if len(tail_events) > 0 else var_level
        return cvar95
