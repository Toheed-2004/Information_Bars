"""
Geometric & Topological Features (Category: GEO)

This module provides the GeometricTopologicalFeatures class, which extracts 
structural and shape-based characteristics from price action using 
classical chart patterns, Topological Data Analysis (TDA), and Path Signatures.
"""

import numpy as np

class GeometricTopologicalFeatures:
    """
    Calculates features based on geometry (shapes) and topology (persistence).
    
    Category breakdown:
    1. Pattern Quantifiers: Numerical proxies for triangles, channels, and dojis.
    2. Topological Data Analysis (TDA): Betti numbers and Wasserstein distance to 
       identify regime shifts in return clusters.
    3. Path Signatures: Theoretical tools for analyzing order-dependent data 
       through iterated integrals (Chen's Theorem).

    Attributes:
        open, high, low, close (np.ndarray): Price series data.
        volume (np.ndarray): Trading volume data.
        n (int): Data length.
        eps (float): Numerical stability factor.
    """

    def __init__(self, open: np.ndarray, high: np.ndarray, low: np.ndarray, 
                 close: np.ndarray, volume: np.ndarray, n: int, eps: float):
        """
        Initializes the GeometricTopologicalFeatures calculator.

        Args:
            open, high, low, close (np.ndarray): OHLC prices.
            volume (np.ndarray): Volumes.
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

    def _shift_array(self, arr: np.ndarray, k: int = 1) -> np.ndarray:
        """Helper: Shift array forward to prevent lookahead bias."""
        out = np.full_like(arr, np.nan)
        out[k:] = arr[:-k]
        return out

    # =========================================================================
    # PATTERN QUANTIFIERS
    # =========================================================================

    def triangle_convergence(self, window: int = 20) -> np.ndarray:
        """
        GEO01: Triangle Convergence.
        Calculates Slope(Highs) - Slope(Lows).
        - Negative: Converging (Symmetrical Triangle / Pennant).
        - Positive: Diverging (Broadening Wedge).
        """
        conv = np.full(self.n, np.nan)

        for i in range(window, self.n):
            h_w = self.high[i - window : i]
            l_w = self.low[i - window : i]
            x = np.arange(window)
            
            # Linear regression on upper and lower bounds
            m_h = np.polyfit(x, h_w, 1)[0] if not np.any(np.isnan(h_w)) else 0.0
            m_l = np.polyfit(x, l_w, 1)[0] if not np.any(np.isnan(l_w)) else 0.0
            
            conv[i] = m_h - m_l

        return conv

    def channel_parallelism(self, window: int = 20) -> np.ndarray:
        """
        GEO02: Channel Parallelism.
        Correlation between Highs and Lows in the window.
        High correlation (+1.0) suggests price is moving within a neat parallel channel.
        """
        par = np.full(self.n, np.nan)

        for i in range(window, self.n):
            h_w, l_w = self.high[i - window : i], self.low[i - window : i]
            if not (np.any(np.isnan(h_w)) or np.any(np.isnan(l_w))):
                corr = np.corrcoef(h_w, l_w)[0, 1]
                par[i] = corr if not np.isnan(corr) else 0.0

        return par

    def pattern_completion(self, window: int = 20) -> np.ndarray:
        """
        GEO03: Relative Range Position (Pattern Completion).
        Measures where current price is within the window's total range (0.0=bottom, 1.0=top).
        Quantifies 'breakout potential' or 'overextension'.
        """
        compl = np.full(self.n, np.nan)

        for i in range(window, self.n):
            p_w = self.close[i - window : i + 1]
            p_max, p_min = np.max(p_w), np.min(p_w)
            rng = p_max - p_min

            if rng > self.eps:
                compl[i] = (self.close[i] - p_min) / rng

        return compl

    def engulfing_ratio(self) -> np.ndarray:
        """
        GEO04: Candle Body Engulfing Ratio.
        Formula: |Close_t - Open_t| / |Close_t-1 - Open_t-1|.
        Values > 1.0 indicate current momentum has 'engulfed' previous candle's range.
        """
        eng = np.full(self.n, np.nan)
        body = np.abs(self.close - self.open)

        for i in range(1, self.n):
            if body[i-1] > self.eps:
                eng[i] = body[i] / body[i-1]

        return self._shift_array(eng, 1)

    def doji_strength(self) -> np.ndarray:
        """
        GEO05: Doji Intensity Ratio.
        Formula: Total Shadows / Body Size.
        Measures market indecision. A true Doji has Body Size close to 0 (very high ratio).
        """
        doji = np.full(self.n, np.nan)
        body = np.abs(self.close - self.open)
        # Sum of upper and lower shadows
        shadow = (self.high - np.maximum(self.open, self.close)) + \
                 (np.minimum(self.open, self.close) - self.low)

        for i in range(self.n):
            if body[i] > self.eps:
                doji[i] = shadow[i] / body[i]
            else:
                doji[i] = shadow[i] / self.eps if shadow[i] > 0 else 0.0

        return self._shift_array(doji, 1)

    def head_shoulders_symmetry(self, window: int = 30) -> np.ndarray:
        """
        GEO06: Head-and-Shoulders Profile Symmetry.
        Compares max(Left 1/3) vs max(Right 1/3) of window.
        A ratio near 1.0 suggests a symmetrical H&S or Triple Top structural development.
        """
        symm = np.full(self.n, np.nan)

        for i in range(window, self.n):
            p_w = self.high[i - window : i]
            third = window // 3
            if third > 0:
                l_peak = np.max(p_w[:third])
                r_peak = np.max(p_w[-third:])
                if r_peak > self.eps:
                    symm[i] = l_peak / r_peak
        return symm

    # =========================================================================
    # TOPOLOGICAL DATA ANALYSIS (TDA)
    # =========================================================================

    def betti_0_mean_life(self, window: int = 50, threshold: float = 0.01) -> np.ndarray:
        """
        GEO07: Betti-0 Component Persistence (Mean Life).
        Measures the average duration of 'clusters' in the return series.
        Identifies period lengths between significant market activity 'bursts'.
        """
        life = np.full(self.n, np.nan)

        for i in range(window, self.n):
            # Log returns log(P_t/P_t-1)
            rets = np.diff(np.log(self.close[i-window:i] + self.eps))
            # Find indices where returns exceed threshold magnitude
            peaks = np.where(np.abs(rets) > threshold)[0]

            if len(peaks) > 1:
                # 'Life' is the time gap between connected components in 1D
                life[i] = np.mean(np.diff(peaks))

        return life

    def betti_1_count(self, window: int = 50, threshold: float = 0.1) -> np.ndarray:
        """
        GEO08: Betti-1 Cycle Count 

        Counts significant up/down alternations in returns using
        a noise-robust sign definition. This approximates Betti-1
        (cycle count) behavior in 1D time series.
        """
        betti = np.full(self.n, np.nan)

        for i in range(window, self.n):
            rets = np.diff(np.log(self.close[i - window:i] + self.eps))

            if len(rets) < 2:
                continue

            eps = threshold * np.std(rets)
            s = np.where(
                rets > eps, 1,
                np.where(rets < -eps, -1, 0)
            )

            betti[i] = np.sum((s[1:] * s[:-1]) == -1)

        return betti

    def wasserstein_distance(self, window: int = 50, ref_window: int = 100) -> np.ndarray:
        """
        GEO09: 1st Wasserstein Distance (EMD).
        Compares the sorted distribution of 'current' window vs 'historical' reference.
        A high value indicates the current distribution of prices has 'drifted' significantly.
        """
        wash = np.full(self.n, np.nan)

        for i in range(ref_window, self.n):
            curr = np.sort(self.close[i-window:i])
            past = np.sort(self.close[i-ref_window:i-window])
            
            # Linear interp to same length if windows differ
            if len(curr) > 0 and len(past) > 0:
                past_interp = np.interp(np.linspace(0, 1, len(curr)), 
                                        np.linspace(0, 1, len(past)), past)
                # Area between CDFs
                wash[i] = np.mean(np.abs(curr - past_interp))

        return wash

    def persistence_entropy(self, window: int = 50) -> np.ndarray:
        """
        GEO10: Distributional Persistence Entropy.
        Shannon entropy applied to the normalized return magnitudes.
        Measures the diversity of return sizes. High entropy = broad range of move sizes.
        """
        ent = np.full(self.n, np.nan)

        for i in range(window, self.n):
            rets = np.diff(np.log(self.close[i-window:i] + self.eps))
            mag = np.abs(rets)
            sum_mag = np.sum(mag)
            if sum_mag > self.eps:
                p = mag / sum_mag
                # -Sum(p * log(p))
                ent[i] = -np.sum(p[p > 0] * np.log(p[p > 0] + self.eps))
        return ent

    # =========================================================================
    # PATH SIGNATURE FEATURES
    # =========================================================================

    def signature_level1(self, window: int = 20) -> np.ndarray:
        """
        GEO11: Path Signature Level 1 (Linear Increment).
        Formula: Integral(dX) = Sum(Returns).
        Captures the net displacement of price over the window.
        """
        sig1 = np.full(self.n, np.nan)

        for i in range(window, self.n):
            rets = np.diff(np.log(self.close[i-window:i] + self.eps))
            sig1[i] = np.sum(rets)

        return sig1

    def signature_level2(self, window: int = 20) -> np.ndarray:
        """
        GEO12: Path Signature Level 2 (Area/Curvature).
        Iterated integral of second order. 
        Measures the accumulated area under the cumulative return path.
        Detects 'convexity' or 'trending' acceleration in the path.
        """
        sig2 = np.full(self.n, np.nan)

        for i in range(window, self.n):
            rets = np.diff(np.log(self.close[i-window:i] + self.eps))
            # Double integral proxy
            sig2[i] = np.sum(np.cumsum(rets))

        return sig2

    def signature_level3(self, window: int = 20) -> np.ndarray:
        """
        GEO13: Path Signature Level 3 (Higher-Order Moment).
        Triple iterated integral proxy.
        Captures more complex dependencies and 'twists' in the high-frequency path.
        """
        sig3 = np.full(self.n, np.nan)

        for i in range(window, self.n):
            rets = np.diff(np.log(self.close[i-window:i] + self.eps))
            sig3[i] = np.sum(np.cumsum(np.cumsum(rets)))

        return sig3

    def lead_lag_signature(self, window: int = 20, lag: int = 5) -> np.ndarray:
        """
        GEO14: Lead-Lag path signature cross-correlation.
        Compares the current path's signature with a lagged path's signature.
        Identifies if current market geometry 'rehearses' a recent structural pattern.
        """
        ll = np.full(self.n, np.nan)

        for i in range(window + lag, self.n):
            path_now = np.cumsum(np.diff(np.log(self.close[i-window:i] + self.eps)))
            path_lag = np.cumsum(np.diff(np.log(self.close[i-window-lag:i-lag] + self.eps)))
            
            if len(path_now) == len(path_lag):
                ll[i] = np.corrcoef(path_now, path_lag)[0, 1]
        
        return ll

    def log_signature_coords(self, window: int = 20, level: int = 2) -> np.ndarray:
        """
        GEO15: Log-Signature Magnitude.
        Log transformation of signature components to normalize scale.
        Used for feeding recursive path features into non-linear models.
        """
        if level == 1: sig = self.signature_level1(window)
        elif level == 2: sig = self.signature_level2(window)
        else: sig = self.signature_level3(window)

        # Vectorized log-abs map
        return np.log(np.abs(sig) + self.eps)
