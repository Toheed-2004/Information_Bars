"""
Event & Survival Features (Category: EVENT)

This module provides the EventSurvivalFeatures class, which calculates timing 
metrics and probability rates for specific market occurrences (e.g. breakouts, 
regime shifts) using survival analysis concepts and point processes (Hawkes).
"""

import numpy as np

class EventSurvivalFeatures:
    """
    Calculates features related to market events, survival rates, and point processes.
    
    Category breakdown:
    1. Time-to-Event: Distance (in bars) since specific milestones (High Break, Regime Shift).
    2. Hazard Analysis: Probability of a future event (Breakout, Extreme Move) based on history.
    3. Self-Exciting Processes: Clustering and contagion analysis using Hawkes Process models.

    Attributes:
        high, low, close (np.ndarray): Price series data.
        n (int): Total samples.
        eps (float): Stability epsilon.
    """

    def __init__(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 n: int, eps: float):
        """
        Initializes the EventSurvivalFeatures calculator.

        Args:
            high, low, close (np.ndarray): Price arrays.
            n (int): Data length.
            eps (float): Stability constant.
        """
        self.high = high
        self.low = low
        self.close = close
        self.n = n
        self.eps = eps

    def _shift_array(self, arr: np.ndarray, k: int = 1) -> np.ndarray:
        """Helper: Shift forward by k to remove lookahead bias."""
        out = np.full_like(arr, np.nan)
        out[k:] = arr[:-k]
        return out

    def _returns_log(self) -> np.ndarray:
        """Helper: Calculate log returns properly shifted for bias removal."""
        log_close = np.log(self.close + self.eps)
        returns = np.full(self.n, np.nan)
        returns[1:] = log_close[1:] - log_close[:-1]

        return returns

    # =========================================================================
    # TIME-TO-EVENT (BAR COUNTERS)
    # =========================================================================

    def time_since_high_break(self, window: int = 20) -> np.ndarray:
        """
        EVENT01: Time Since High Breakout.
        Counts consecutive bars where current close <= rolling max(High).
        Resets to 0 when a new high is printed. Identifies the age of the current 'local top'.
        """
        ts_break = np.full(self.n, np.nan)
        counter = 0

        for i in range(window, self.n):
            # Rolling peak in the period preceding the current bar
            peak = np.max(self.high[i - window : i])

            if self.close[i] > peak:
                counter = 0
            else:
                counter += 1
            ts_break[i] = float(counter)

        return ts_break

    def time_since_regime_change(self, window: int = 50, threshold: float = 0.02) -> np.ndarray:
        """
        EVENT02: Time Since Regime Shift.
        Identifies regime (Bull/Bear/Static) via mean returns.
        Counts bars since the regime status last toggled. 
        Detects trending maturity and consolidation age.
        """
        returns = self._returns_log()
        ts_regime = np.full(self.n, np.nan)
        counter = 0
        prev_r = None

        for i in range(window, self.n):
            mean_r = np.nanmean(returns[i - window : i])
            # Categorization: 1 (Bull), 0 (Static), -1 (Bear)
            curr_r = 1 if mean_r > threshold else (-1 if mean_r < -threshold else 0)

            if prev_r is not None and curr_r != prev_r:
                counter = 0
            else:
                counter += 1
            
            ts_regime[i] = float(counter)
            prev_r = curr_r

        return self._shift_array(ts_regime, 1)

    def time_since_drawdown_5pct(self, window: int = 100, threshold: float = -0.05) -> np.ndarray:
        """
        EVENT03: Time Since Significant Drawdown.
        Counts bars since price dropped by more than threshold (5%) from recent peak.
        Measures recovery duration and distance to last 'scare' event.
        """
        ts_dd = np.full(self.n, np.nan)
        last_event_idx = 0

        for i in range(window, self.n):
            lookback = self.close[i - window : i]
            peak = np.max(lookback)
            # Current drawdown depth from window peak
            drawdown = (self.close[i] - peak) / (peak + self.eps)

            if drawdown < threshold:
                last_event_idx = i

            ts_dd[i] = float(i - last_event_idx)

        return ts_dd

    def expected_time_to_breakout(self, window: int = 20, min_compression: float = 0.01) -> np.ndarray:
        """
        EVENT04: Expected Time to Breakout (Survival Proxy).
        Estimates 'waiting time' based on range compression speed.
        If range is shrinking (ranges[0] > ranges[-1]), we expect a breakout sooner.
        """
        ttb = np.full(self.n, np.nan)

        for i in range(window, self.n):
            h_w, l_w = self.high[i-window:i], self.low[i-window:i]
            # Compression = (StartRange - EndRange) / StartRange
            rng_start = h_w[0] - l_w[0]
            rng_end = h_w[-1] - l_w[-1]
            compress = (rng_start - rng_end) / (rng_start + self.eps)

            if compress > min_compression:
                # Survival proxy: Inverse of the compression rate
                ttb[i] = 1.0 / (compress + self.eps)

        return ttb

    def hazard_rate_weibull(self, window: int = 100) -> np.ndarray:
        """
        EVENT05: Extreme Event Hazard (Naive Frequency).
        Measures the empirical density of 'outlier' events (> 2 StdDev).
        High hazard indicates an environment prone to jumps/shocks.
        """
        returns = self._returns_log()
        haz = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_w = returns[i - window : i]
            std_r = np.nanstd(ret_w)
            
            if std_r > self.eps:
                # Count frequency of 2-sigma events
                outliers = np.sum(np.abs(ret_w) > 2.0 * std_r)
                haz[i] = outliers / float(window)

        return haz

    # =========================================================================
    # HAWKES & POINT PROCESS (CONTAGION)
    # =========================================================================

    def hawkes_intensity(self, window: int = 100, alpha: float = 0.5, 
                         beta: float = 0.5) -> np.ndarray:
        """
        EVENT06: Hawkes Self-Exciting Intensity.
        Formula: Intensity(t) = Sum( alpha * exp(-beta * (t - t_event)) ).
        Models the 'clustering' of volatility spikes. Intensity rises sharply after 
        a spike and decays over time. Captures momentum contagion.
        """
        returns = self._returns_log()
        intens = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_w = returns[i - window : i]
            std_r = np.std(ret_w)
            if std_r < self.eps: 
                continue

            # Lower threshold to catch more events
            events = np.abs(ret_w) > max(0.5 * std_r, self.eps)
            h_val = 0.0
            for j, is_event in enumerate(events):
                if is_event:
                    dt = len(ret_w) - j - 1
                    h_val += alpha * np.exp(-beta * dt)
            intens[i] = h_val

        return intens
    
    def hawkes_branching_ratio(self, window: int = 100) -> np.ndarray:
        """
        EVENT07: Endogenous Branching Proxy.
        Estimates the 'self-feeding' nature of price moves.
        High ratio suggests moves are driven by market reflexivity (endogenous) 
        rather than external news (exogenous). 
        Calculated via Sign-Change persistence.
        """
        returns = self._returns_log()
        ratio = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_w = returns[i - window : i]
            if len(ret_w) < 5: 
                continue

            # Count sign changes
            chg = np.sum(np.diff(np.sign(ret_w)) != 0)
            a_est = 1.0 - (chg / max(len(ret_w) - 1, 1))
            b_est = 1.0 / (1.0 + chg)
            ratio[i] = a_est / (b_est + self.eps)

        return ratio


    def extreme_event_clustering(self, window: int = 100, threshold: float = 2.0) -> np.ndarray:
        """
        EVENT08: Extreme Jump Autocorrelation.
        Correlation between current extreme event binary flag and previous bar extreme flag.
        Identifies whether crashes/surges are 'lonely' or 'clustered' (contagious).
        """
        returns = self._returns_log()
        clust = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_w = returns[i - window : i]
            std_r = np.std(ret_w)
            if std_r < self.eps: 
                continue

            extremes = (np.abs(ret_w) > max(threshold * std_r, self.eps)).astype(float)
            if np.sum(extremes) > 1:
                # Lag-1 autocorrelation of binary extremes
                c = np.corrcoef(extremes[:-1], extremes[1:])[0, 1]
                clust[i] = c if np.isfinite(c) else 0.0

        return clust
