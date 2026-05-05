"""
Compression & Phase Transition Features (Category: COMP)

This module provides the CompressionFeatures class, which calculates features
identifying market states like consolidation, compression, and chaotic transitions.
It uses advanced metrics such as Lyapunov exponents, multifractal spectrum width,
and entropy collapse to detect shifts in market regime.
"""

import numpy as np


class CompressionFeatures:
    """
    Calculates features related to market compression and phase transitions.

    This class identifies periods when the market is "coiling" (compressing)
    or undergoing structural changes (phase transitions) using statistical
    and dynamical systems theory metrics.

    Attributes:
        open, high, low, close (np.ndarray): Price series data.
        volume (np.ndarray): Array of trading volumes.
        n (int): Length of the data series.
        eps (float): Small constant for numerical stability.
    """

    def __init__(
        self,
        open: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        n: int,
        eps: float,
    ):
        """
        Initializes the CompressionFeatures calculator.

        Args:
            open, high, low, close (np.ndarray): Price arrays.
            volume (np.ndarray): Volume array.
            n (int): Total number of samples.
            eps (float): Numerical stability constant (e.g., 1e-10).
        """
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.n = n
        self.eps = eps

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _true_range(self) -> np.ndarray:
        """
        Helper: Calculates the True Range (TR) of the price series.
        Formula: max(High - Low, |High - PrevClose|, |Low - PrevClose|)
        """
        tr = np.zeros(self.n)
        # Initial value is just the bar range
        tr[0] = self.high[0] - self.low[0]

        for i in range(1, self.n):
            tr[i] = max(
                self.high[i] - self.low[i],
                abs(self.high[i] - self.close[i - 1]),
                abs(self.low[i] - self.close[i - 1]),
            )
        return tr

    def _shift_array(self, arr: np.ndarray, k: int = 1) -> np.ndarray:
        """Helper: Shifts an array forward by k steps, padding with NaNs."""
        out = np.full_like(arr, np.nan)
        out[k:] = arr[:-k]
        return out

    def _returns_log(self) -> np.ndarray:
        """Helper: Calculates logarithmic returns."""
        log_close = np.log(self.close + self.eps)
        returns = np.full(self.n, np.nan)
        returns[1:] = log_close[1:] - log_close[:-1]

        return returns

    def _ema(self, arr: np.ndarray, window: int) -> np.ndarray:
        """
        Helper: Calculates Exponential Moving Average (EMA).
        Alpha = 2 / (window + 1)
        """
        ema = np.full(len(arr), np.nan)
        alpha = 2.0 / (window + 1)

        # Find first non-NaN index
        start_idx = np.where(~np.isnan(arr))[0]
        if len(start_idx) == 0:
            return ema

        start = start_idx[0]
        ema[start] = arr[start]

        for i in range(start + 1, len(arr)):
            if not np.isnan(arr[i]):
                # Standard EMA recursive formula
                ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
            else:
                ema[i] = ema[i - 1] if i > start else np.nan

        return ema

    # =========================================================================
    # COMPRESSION INDICATORS
    # =========================================================================

    def range_compression_ratio(
        self, short_window: int = 5, long_window: int = 20
    ) -> np.ndarray:
        """
        COMP01: Range Compression Ratio.
        Ratio of short-term ATR to long-term ATR.
        Values < 1.0 indicate that the market is currently in a state of compression.
        """
        tr = self._true_range()

        # ATR calculation using Wilder's smoothing logic
        def get_atr(data, w):
            atr_arr = np.full(self.n, np.nan)
            atr_arr[w - 1] = np.mean(data[:w])
            for i in range(w, self.n):
                atr_arr[i] = (atr_arr[i - 1] * (w - 1) + data[i]) / w
            return atr_arr

        atr_short = get_atr(tr, short_window)
        atr_long = get_atr(tr, long_window)

        # Compute ratio (Short-term Volatility / Long-term Volatility)
        ratio = np.full(self.n, np.nan)
        valid_mask = (~np.isnan(atr_long)) & (atr_long > self.eps)
        ratio[valid_mask] = atr_short[valid_mask] / atr_long[valid_mask]

        return self._shift_array(ratio, 1)

    def bollinger_area(self, window: int = 20, nbdev: float = 2.0) -> np.ndarray:
        """
        COMP02: Bollinger Area.
        Measures the total 'surface area' between Bollinger Bands over a window.
        Useful for quantifying the energy available for a potential breakout.
        """
        bb_area = np.full(self.n, np.nan)

        for i in range(window, self.n):
            price_window = self.close[i - window : i]
            # Standard Deviation as width proxy
            std = np.std(price_window)
            # Area = (Upper - Lower) * Window Length
            width = 2 * nbdev * std
            bb_area[i] = width * window

        return bb_area

    def entropy_collapse_rate(
        self, window: int = 20, outer_window: int = 60
    ) -> np.ndarray:
        """
        COMP03: Entropy Collapse Rate.
        Measures the rate of change in the Shannon Entropy of the price distribution.
        A rapid decrease (collapse) in entropy often precedes large directional moves.
        """
        entropy = np.full(self.n, np.nan)

        for i in range(window, self.n):
            price_window = self.close[i - window : i]

            p_min, p_max = np.min(price_window), np.max(price_window)
            if p_max - p_min > self.eps:
                normalized = (price_window - p_min) / (p_max - p_min)
            else:
                normalized = np.full_like(price_window, 0.5)

            # Histogram-based entropy calculation
            hist, _ = np.histogram(normalized, bins=10, range=(0, 1))
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            entropy[i] = -np.sum(probs * np.log(probs + self.eps))

        collapse_rate = np.full(self.n, np.nan)
        # Calculate the linear slope (velocity) of entropy change
        for i in range(window + outer_window, self.n):
            entropy_window = entropy[i - outer_window : i]
            valid = ~np.isnan(entropy_window)
            if np.sum(valid) > 2:
                x = np.arange(len(entropy_window[valid]))
                slope = np.polyfit(x, entropy_window[valid], 1)[0]
                collapse_rate[i] = slope

        return collapse_rate

    def consolidation_slope(
        self, window: int = 20, outer_window: int = 60
    ) -> np.ndarray:
        """
        COMP04: Consolidation Slope.
        Specifically tracks the narrowing of the absolute price range (High - Low).
        Returns the slope of the price range over an outer window.
        """
        price_range = np.full(self.n, np.nan)
        for i in range(window, self.n):
            price_window = self.close[i - window : i]
            price_range[i] = np.max(price_window) - np.min(price_window)

        cons_slope = np.full(self.n, np.nan)
        for i in range(window + outer_window, self.n):
            range_window = price_range[i - outer_window : i]
            valid = ~np.isnan(range_window)
            if np.sum(valid) > 2:
                x = np.arange(len(range_window[valid]))
                slope = np.polyfit(x, range_window[valid], 1)[0]
                cons_slope[i] = slope

        return cons_slope

    def volume_compression(self, window: int = 20) -> np.ndarray:
        """
        COMP05: Volume Compression.
        Ratio of current volume to its EMA. Low ratios suggest market participants
        are waiting for a catalyst, typical of major consolidation patterns.
        """
        vol_ema = self._ema(self.volume, window)
        compression = np.full(self.n, np.nan)

        valid_mask = (~np.isnan(vol_ema)) & (vol_ema > self.eps)
        compression[valid_mask] = self.volume[valid_mask] / vol_ema[valid_mask]

        compression = self._shift_array(compression)
        return compression

    # =========================================================================
    # PHASE TRANSITION INDICATORS
    # =========================================================================

    def lyapunov_exponent(self, window: int = 50, delay: int = 1) -> np.ndarray:
        """
        COMP06: Lyapunov Exponent (Dynamic Chaos Metric).
        Measures the exponential rate of divergence of adjacent price trajectories.
        Positve values suggest chaotic dynamics; near-zero values suggest equilibrium/stasis.
        """
        returns = self._returns_log()
        lyap = np.full(self.n, np.nan)

        for i in range(window + delay, self.n):
            segment = returns[i - window:i]

            if np.any(np.isnan(segment)):
                continue

            # distance from last point
            ref = segment[-1]
            dists = np.abs(segment[:-delay] - ref)

            idx = np.argmin(dists)
            nearest_dist = dists[idx]

            if nearest_dist < self.eps:
                continue

            future_ref = returns[i]
            future_near = returns[i - window + idx + delay]

            divergence = np.abs(future_ref - future_near)

            lyap[i] = np.log((divergence + self.eps) / (nearest_dist + self.eps))

        return self._shift_array(lyap, 1)

    def multifractal_spectrum_width(self, window: int = 100) -> np.ndarray:
        """
        COMP07: Multifractal Spectrum Width.
        Quantifies the diversity of scaling exponents in the return series.
        A wider spectrum indicates complex, multifractal dynamics (non-linear regimes).
        """
        returns = self._returns_log()
        mf_width = np.full(self.n, np.nan)

        q_vals = np.linspace(-3, 3, 13)
        q_vals = q_vals[q_vals != 0]

        for i in range(window, self.n):
            ret_window = returns[i - window:i]
            ret_window = ret_window[~np.isnan(ret_window)]

            if len(ret_window) < 10:
                continue

            abs_ret = np.abs(ret_window) + self.eps

            tau = []
            for q in q_vals:
                moment = np.mean(abs_ret ** q)
                tau.append(np.log(moment + self.eps))

            tau = np.array(tau)

            # slope of tau(q) curve → scaling variability
            slope = np.gradient(tau, q_vals)

            mf_width[i] = np.max(slope) - np.min(slope)

        return self._shift_array(mf_width, 1)

    def scaling_function_tau(self, window: int = 100) -> np.ndarray:
        """
        COMP08: Scaling Function Tau (q-order Scaling).
        Computes the average scaling exponent across moments.
        Used to identify long-range dependence and deviation from Brownian motion.
        """
        returns = self._returns_log()
        tau_avg = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_window = returns[i - window : i]
            if np.all(np.isnan(ret_window)):
                continue

            q_values = np.linspace(0.5, 5, 10)
            tau_vals = []
            for q in q_values:
                moment = np.mean(np.abs(ret_window) ** q)
                # Scaling exponent tau(q) approximation
                tau = q * np.log(moment + self.eps) / np.log(window)
                tau_vals.append(tau)

            tau_avg[i] = np.mean(tau_vals)
    
        return tau_avg

    def universal_multifractal_c1(self, window: int = 100) -> np.ndarray:
        """
        COMP09: Universal Multifractal C1 (Intermittency).
        Measures the codified intermittency of market volatility.
        Higher values indicate a higher degree of burstiness (extreme event clustering).
        """
        returns = self._returns_log()
        c1 = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_window = returns[i - window : i]
            ret_window = ret_window[~np.isnan(ret_window)]
            if len(ret_window) < 10:
                continue

            abs_ret = np.abs(ret_window) + self.eps

            # Compute positive and negative moments
            m_pos = np.mean(abs_ret ** 1.0)
            m_neg = np.mean(abs_ret ** -1.0)

            if m_pos > self.eps and m_neg > self.eps:
                # Stabilized log ratio
                log_ratio = np.log(m_pos + self.eps) - np.log(m_neg + self.eps)
                if np.isfinite(log_ratio):
                    c1[i] = log_ratio / 2

        # Shift once after loop
        c1 = self._shift_array(c1, 1)
        return c1

    def phase_space_recurrence(self, window: int = 50, embed_dim: int = 3) -> np.ndarray:
        """
        COMP10: Phase Space Recurrence Rate (RQA).
        The density of recurrent points in an embedded phase space.
        Reveals hidden periodicities and deterministic structures in noisy price data.
        """
        returns = self._returns_log()
        recurrence = np.full(self.n, np.nan)

        from scipy.spatial.distance import pdist

        for i in range(window + embed_dim, self.n):
            # Embed return series into higher dimensional space
            embedded = np.array([
                returns[i - window + j : i - window + j + embed_dim]
                for j in range(window - embed_dim + 1)
            ])

            if len(embedded) <= 1:
                continue

            # Compute distances
            distances = pdist(embedded)
            if len(distances) == 0:
                continue

            # Local threshold
            threshold = np.std(returns[i - window : i]) * 0.5
            recurrence_count = np.sum(distances < threshold)
            recurrence[i] = recurrence_count / len(distances)

        # Shift once after loop
        recurrence = self._shift_array(recurrence, 1)
        return recurrence
