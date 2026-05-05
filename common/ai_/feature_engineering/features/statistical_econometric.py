"""
Statistical & Econometric Features (Category: STAT)

This module provides the StatisticalEconometric class, which implements
advanced distribution tests, stationarity checks (ADF, Hurst, Variance Ratio), 
and multi-variate relationship measures (Beta, Cointegration).
"""

import numpy as np

class StatisticalEconometric:
    """
    Calculates statistical and econometric properties of price/volume data.
    
    Categorized into:
    1. Distribution Moments (Skewness, Kurtosis, Jarque-Bera).
    2. Stationarity & Ergodicity (Hurst, ADF, Variance Ratio).
    3. Serial Dependence (Autocorrelation, Absolute Autocorrelation).
    4. Asset Relationships (Beta, Cointegration, Eigenvalue analysis).

    Attributes:
        close (np.ndarray): Asset closing prices.
        volume (np.ndarray): Trading volume series.
        n (int): Total number of data samples.
        eps (float): Numerical stability constant.
    """

    def __init__(self, close: np.ndarray, volume: np.ndarray, n: int, eps: float):
        """
        Initializes the StatisticalEconometric calculator.

        Args:
            close (np.ndarray): Price array.
            volume (np.ndarray): Volume array.
            n (int): Length of data.
            eps (float): Stability epsilon.
        """
        self.close = close
        self.volume = volume
        self.n = n
        self.eps = eps


    def _shift_array(self, arr: np.ndarray, k: int = 1) -> np.ndarray:
        """Helper: Shift forward by k to remove lookahead bias."""
        out = np.full_like(arr, np.nan)
        out[k:] = arr[:-k]
        return out
    
    def _returns_log(self) -> np.ndarray:
        """Helper: Calculate log returns, the standard input for many stat tests."""
        log_close = np.log(self.close + self.eps)
        returns = np.full(self.n, np.nan)
        returns[1:] = log_close[1:] - log_close[:-1]
        return returns

    # =========================================================================
    # DISTRIBUTION MOMENTS & NORMALITY
    # =========================================================================

    def skewness(self, window: int = 30) -> np.ndarray:
        """
        STAT01: Return Skewness.
        Formula: E[( (X - mu) / sigma )^3]
        Identifies return asymmetry. Positive = many small losses, few large gains.
        """
        returns = self._returns_log()
        skew = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_window = returns[i - window:i]
            mean_ret = np.nanmean(ret_window)
            std_ret = np.nanstd(ret_window)

            if std_ret > self.eps:
                # 3rd standardized moment
                skew[i] = np.nanmean(((ret_window - mean_ret) / std_ret) ** 3)

        return skew

    def kurtosis(self, window: int = 30) -> np.ndarray:
        """
        STAT02: Excess Kurtosis.
        Formula: E[( (X - mu) / sigma )^4] - 3
        Measures 'fat-tails'. Normal distribution = 0. High values = crash risk.
        """
        returns = self._returns_log()
        kurt = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_window = returns[i - window:i]
            mean_ret = np.nanmean(ret_window)
            std_ret = np.nanstd(ret_window)

            if std_ret > self.eps:
                # 4th standardized moment (excess)
                kurt[i] = np.nanmean(((ret_window - mean_ret) / std_ret) ** 4) - 3

        return kurt

    def jarque_bera(self, window: int = 60) -> np.ndarray:
        """
        STAT03: Jarque-Bera Statistic.
        Formula: (n/6) * (Skew^2 + (Kurtosis^2 / 4))
        Standard test for normality. Higher values indicate non-normal (fat-tailed) returns.
        """
        returns = self._returns_log()
        jb = np.full(self.n, np.nan)

        for i in range(window, self.n):
            window_slice = returns[i - window:i]
            std_val = np.nanstd(window_slice)

            if std_val > self.eps:
                standardized = (window_slice - np.nanmean(window_slice)) / std_val
                s = np.nanmean(standardized ** 3)
                k = np.nanmean(standardized ** 4) - 3
                jb[i] = (window / 6.0) * (s**2 + (k**2 / 4.0))
        
        return jb

    def quantile(self, q: float = 0.1, window: int = 60) -> np.ndarray:
        """
        STAT04-05: Return Quantile.
        Determines the return value below which q% of the data falls.
        Useful for measuring Value-at-Risk (VaR) or extreme upside potential.
        """
        returns = self._returns_log()
        quantile = np.full(self.n, np.nan)

        for i in range(window, self.n):
            quantile[i] = np.nanpercentile(returns[i - window:i], q * 100)
        
        return quantile

    def tail_ratio(self, window: int = 60) -> np.ndarray:
        """
        STAT06: Tail Ratio.
        Formula: Percentile(95) / abs(Percentile(5))
        Measures the balance between extreme gains and extreme losses.
        """
        returns = self._returns_log()
        tail_ratio = np.full(self.n, np.nan)

        for i in range(window, self.n):
            ret_w = returns[i - window:i]
            q95, q05 = np.nanpercentile(ret_w, 95), np.nanpercentile(ret_w, 5)

            if abs(q05) > self.eps:
                tail_ratio[i] = q95 / abs(q05)
        
      
        return tail_ratio

    # =========================================================================
    # STATIONARITY & MEMORY (TIME-DOMAIN)
    # =========================================================================

    def adf_statistic(self, window: int = 60, lag: int = 1) -> np.ndarray:
        """
        STAT07: Augmented Dickey-Fuller Statistic (Approximate).
        Tests for a unit root. Identifies if the series is stationary.
        Implementation uses a simple OLS regression approach: y_t = c + beta*y_{t-1} + e.
        """
        returns = self._returns_log()
        adf = np.full(self.n, np.nan)

        for i in range(window + lag, self.n):
            y = returns[i - window:i]
            y_lag = returns[i - window - lag:i - lag]

            mask = ~(np.isnan(y) | np.isnan(y_lag))
            if np.sum(mask) > lag + 1:
                # OLS: y = beta * y_lag
                X = np.column_stack([np.ones(np.sum(mask)), y_lag[mask]])
                try:
                    beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
                    res = y[mask] - X @ beta
                    se = np.sqrt(np.sum(res**2) / (len(res)-2)) / (np.sqrt(np.sum((y_lag[mask]-np.mean(y_lag[mask]))**2)) + self.eps)
                    adf[i] = (beta[1] - 1) / (se + self.eps)
                except: continue

        return adf

    def hurst_exponent(self, window: int = 100, lag_range: tuple = (2, 20)) -> np.ndarray:
        """
        STAT08: Hurst Exponent (H).
        Calculates the rescaled range (R/S) to measure long-term memory.
        H = 0.5: Random Walk. H > 0.5: Trending (Persistent). H < 0.5: Mean-Reverting.
        """
      
        returns = self._returns_log()
        hurst = np.full(self.n, np.nan)

        for i in range(window, self.n):
            r = returns[i - window:i]
            r = r[~np.isnan(r)]

            if len(r) < 20:
                continue

            lags = np.arange(lag_range[0], lag_range[1])
            tau = []

            for lag in lags:
                diff = r[lag:] - r[:-lag]
                tau.append(np.std(diff))

            tau = np.array(tau)

            if np.all(tau > self.eps):
                hurst[i] = np.polyfit(np.log(lags), np.log(tau), 1)[0]

        return hurst

    def variance_ratio(self, lag: int = 2, window: int = 100) -> np.ndarray:
        """
        STAT09: Variance Ratio Test.
        Checks if Var(Sum of k periods) == k * Var(1 period).
        Values != 1.0 indicate presence of autocorrelation or non-random behavior.
        """
        returns = self._returns_log()
        vr = np.full(self.n, np.nan)

        for i in range(window + lag, self.n):
            ret_w = returns[i - window:i]
            # 1-period variance:
            var_1 = np.nanvar(ret_w)
            
            # k-period variance (rolling sums):
            k_ret = [np.sum(ret_w[j:j+lag]) for j in range(len(ret_w) - lag + 1)]
            var_k = np.var(k_ret)

            if var_1 > self.eps:
                vr[i] = var_k / (lag * var_1)

        return vr

    def return_autocorr(self, lag: int = 1, window: int = 60) -> np.ndarray:
        """
        STAT10: Serial Autocorrelation (ACF).
        Measures the correlation between current returns and returns at 'lag'.
        High autocorrelation suggests price momentum or persistence.
        """
        returns = self._returns_log()
        acf = np.full(self.n, np.nan)

        for i in range(window + lag, self.n):
            r_w = returns[i - window:i]
            mu = np.nanmean(r_w)
            denom = np.nansum((r_w - mu)**2)
            
            if denom > self.eps:
                # Correlate slice(0, -lag) with slice(lag, end)
                num = np.sum((r_w[:-lag] - mu) * (r_w[lag:] - mu))
                acf[i] = num / denom

        return acf

    def abs_autocorr(self, lag: int = 1, window: int = 60) -> np.ndarray:
        """
        STAT11: Absolute Return Autocorrelation.
        Autocorrelation of magnitude (|Returns|).
        Standard evidence for 'volatility clustering' if significantly positive.
        """
        abs_returns = np.abs(self._returns_log())
        abs_acf = np.full(self.n, np.nan)

        for i in range(window + lag, self.n):
            r_w = abs_returns[i - window:i]
            mu = np.nanmean(r_w)

            denom = np.nansum((r_w - mu) ** 2)
            if denom > self.eps:
                num = np.nansum((r_w[:-lag] - mu) * (r_w[lag:] - mu))
                abs_acf[i] = num / denom

        return self._shift_array(abs_acf, 1)


    # =========================================================================
    # MULTI-VARIATE RELATIONSHIPS
    # =========================================================================

    def coint_residual(self, window: int = 100) -> np.ndarray:
        """
        STAT12: Cointegration Residual (Spread).
        Uses a roll-lag benchmark as a proxy for a cointegrated pair.
        residual = Price - (alpha + beta * LaggedPrice). 
        Mean-reverting residuals imply a tradable spread.
        """
        # Benchmark proxy: Price rolled by offset
        ref_asset = np.roll(self.close, 5) 
        residual = np.full(self.n, np.nan)

        for i in range(window, self.n):
            y, x = self.close[i-window:i], ref_asset[i-window:i]
            mask = ~(np.isnan(y) | np.isnan(x))
            if np.sum(mask) > 5:
                # OLS to find alpha, beta
                X_mat = np.column_stack([np.ones(np.sum(mask)), x[mask]])
                try:
                    beta = np.linalg.lstsq(X_mat, y[mask], rcond=None)[0]
                    # Current residual
                    residual[i] = self.close[i-1] - (beta[0] + beta[1] * ref_asset[i-1])
                except: continue

        return residual

    def coint_t_stat(self, window: int = 100) -> np.ndarray:
        """
        STAT13: Cointegration t-statistic (CADF).
        The t-statistic from an ADF test on the residuals of a cointegrating regression.
        Highly negative values signify a valid cointegration relationship.
        """
        # (Internal calculation simplified to reuse spread logic)
        spread = self.coint_residual(window=window)
        t_stat = np.full(self.n, np.nan)
        
        for i in range(window + 20, self.n):
            s_w = spread[i-20:i] # Check stationarity of spread over short window
            if np.any(np.isnan(s_w)): continue
            
            diff_s = np.diff(s_w)
            lag_s = s_w[:-1]
            try:
                # Dickey-Fuller regression on spread
                X = np.column_stack([np.ones(len(lag_s)), lag_s])
                beta = np.linalg.lstsq(X, diff_s, rcond=None)[0]
                res = diff_s - X @ beta
                se = np.sqrt(np.sum(res**2) / (len(res)-2)) / (np.sqrt(np.sum((lag_s-np.mean(lag_s))**2)) + self.eps)
                t_stat[i] = beta[1] / (se + self.eps)
            except: continue

        t_stat = self._shift_array(t_stat)
        return t_stat

    def beta_to_asset(self, window: int = 60) -> np.ndarray:
        """
        STAT14: Rolling Regression Beta.
        Measures sensitivity of the asset to a theoretical 'market' (lagged proxy).
        Beta = Cov(Returns_Self, Returns_Ref) / Var(Returns_Ref).
        """
        # here add the other coin data as the reference asset (e.g., a market index or another coin)
        ref_price = np.roll(self.close, 1)
        r_self = np.diff(np.log(self.close + self.eps))
        r_ref = np.diff(np.log(ref_price + self.eps))
        
        beta = np.full(self.n, np.nan)
        for i in range(window, self.n-1):
            s_s, s_r = r_self[i-window:i], r_ref[i-window:i]
            cov = np.cov(s_s, s_r)[0, 1]
            v_r = np.var(s_r)
            if v_r > self.eps:
                beta[i] = cov / v_r
        
        return beta

    def corr_matrix_eigenval(self, window: int = 60) -> np.ndarray:
        """
        STAT15: Maximum Correlation Eigenvalue.
        Analyzes the correlation between returns and absolute returns. 
        A high max eigenvalue suggests a single dominant factor (e.g., global volatility surge).
        """
        r = self._returns_log()
        ar = np.abs(r)
        
        e_val = np.full(self.n, np.nan)
        for i in range(window, self.n):
            r_w, ar_w = r[i-window:i], ar[i-window:i]
            mask = ~(np.isnan(r_w) | np.isnan(ar_w))
            if np.sum(mask) > 10:
                # Matrix of [Returns, AbsReturns]
                data = np.column_stack([r_w[mask], ar_w[mask]])
                corr = np.corrcoef(data.T)
                # Max eigenvalue of 2x2 correlation matrix
                # Note: For 2x2, max val is 1 + |rho|
                e_val[i] = 1 + np.abs(corr[0, 1])
        e_val = self._shift_array(e_val, 1)
        return e_val
