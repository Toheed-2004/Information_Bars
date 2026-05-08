"""
Information & Causal Features (Category: INFO)
Vectorized version for performance optimization
"""

import numpy as np
from math import factorial


class InformationCausalFeatures:
    """
    Calculate Information & Causal Features using NumPy only - VECTORIZED VERSION.
    """

    def __init__(self, close, volume, n, eps):
        self.close = close
        self.volume = volume
        self.n = n
        self.eps = eps
        
    def _returns_log(self) -> np.ndarray:
        """Helper: Calculate log returns."""
        log_close = np.log(self.close + self.eps)
        returns = np.full(self.n, np.nan)
        returns[1:] = log_close[1:] - log_close[:-1]
        return returns
    
    def _vectorized_sample_entropy_core(self, data, m, r):
        """Vectorized core for sample entropy calculation."""
        N = len(data)
        if N <= m:
            return np.nan
            
        # Create all templates of length m
        templates_m = np.lib.stride_tricks.sliding_window_view(data, m)
        
        # Create all templates of length m+1
        templates_m1 = np.lib.stride_tricks.sliding_window_view(data, m + 1)
        
        # Calculate distances between all pairs of m-length templates
        # Using broadcasting to avoid loops
        if len(templates_m) > 0:
            # Reshape for broadcasting
            tm1 = templates_m[:, np.newaxis, :]
            tm2 = templates_m[np.newaxis, :, :]
            
            # Calculate Chebyshev distances
            dist_m = np.max(np.abs(tm1 - tm2), axis=2)
            
            # Count matches (excluding self-matches)
            np.fill_diagonal(dist_m, np.inf)  # Exclude self-matches
            count_m = np.sum(dist_m <= r) / 2  # Divide by 2 because matrix is symmetric
            
            # For m+1 templates
            if len(templates_m1) > 0:
                tm1_1 = templates_m1[:, np.newaxis, :]
                tm2_1 = templates_m1[np.newaxis, :, :]
                
                dist_m1 = np.max(np.abs(tm1_1 - tm2_1), axis=2)
                np.fill_diagonal(dist_m1, np.inf)
                count_m1 = np.sum(dist_m1 <= r) / 2
                
                if count_m > 0 and count_m1 > 0:
                    return -np.log((count_m1 + self.eps) / (count_m + self.eps))
        
        return np.nan

    # ===================== ENTROPY & COMPLEXITY =====================

    def sample_entropy(self, window: int = 100, m: int = 2, r: float = 0.2) -> np.ndarray:
        """
        INFO01: Sample entropy - Measuring time series regularity.
        VECTORIZED VERSION
        """
        returns = self._returns_log()
        sample_ent = np.full(self.n, np.nan)
        
        # Pre-calculate windows using sliding_window_view
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(returns, (window-1, 0), constant_values=np.nan), 
            window
        )
        
        for i in range(window, min(self.n, len(windows) + window - 1)):
            ret_window = windows[i - window]
            ret_window = ret_window[~np.isnan(ret_window)]
            
            if len(ret_window) < m + 1:
                continue
                
            ret_std = np.std(ret_window)
            if ret_std < self.eps:
                continue
                
            ret_norm = ret_window / ret_std
            sample_ent[i] = self._vectorized_sample_entropy_core(ret_norm, m, r)
            
        return sample_ent

    def permutation_entropy(self, window: int = 100, m: int = 3) -> np.ndarray:
        """
        INFO02: Permutation entropy - Entropy of ordinal patterns.
        VECTORIZED VERSION
        """
        returns = self._returns_log()
        perm_ent = np.full(self.n, np.nan)
        
        # Pre-calculate all windows
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(returns, (window-1, 0), constant_values=np.nan), 
            window
        )
        
        for i in range(window, min(self.n, len(windows) + window - 1)):
            ret_window = windows[i - window]
            ret_window = ret_window[~np.isnan(ret_window)]
            
            if len(ret_window) < m:
                continue
                
            # Create all patterns using vectorized operations
            patterns = np.lib.stride_tricks.sliding_window_view(ret_window, m)
            
            # Get ordinal patterns in one step
            ordinal_patterns = np.argsort(np.argsort(patterns, axis=1), axis=1)
            
            # Convert patterns to tuples for counting
            pattern_tuples = [tuple(row) for row in ordinal_patterns]
            
            # Count occurrences
            unique_patterns, counts = np.unique(pattern_tuples, axis=0, return_counts=True)
            
            # Calculate entropy
            probs = counts / counts.sum()
            entropy_val = -np.sum(probs * np.log(probs + self.eps))
            
            max_entropy = np.log(factorial(m))
            perm_ent[i] = entropy_val / max_entropy if max_entropy > 0 else 0.0
            
        return perm_ent

    def multiscale_entropy(self, window: int = 100, m: int = 2, r: float = 0.2, scale: int = 2) -> np.ndarray:
        """
        INFO03: Multiscale entropy - Sample entropy at different scales.
        VECTORIZED VERSION
        """
        returns = self._returns_log()
        ms_ent = np.full(self.n, np.nan)
        
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(returns, (window-1, 0), constant_values=np.nan), 
            window
        )
        
        for i in range(window, min(self.n, len(windows) + window - 1)):
            ret_window = windows[i - window]
            ret_window = ret_window[~np.isnan(ret_window)]
            
            if len(ret_window) < m * scale:
                continue
                
            # Coarse-graining using reshape and mean
            trim_len = len(ret_window) - (len(ret_window) % scale)
            if trim_len >= scale:
                coarse = ret_window[:trim_len].reshape(-1, scale).mean(axis=1)
                
                coarse_std = np.std(coarse)
                if coarse_std < self.eps:
                    continue
                    
                coarse_norm = coarse / coarse_std
                ms_ent[i] = self._vectorized_sample_entropy_core(coarse_norm, m, r)
                
        return ms_ent

    def fisher_information(self, window: int = 100) -> np.ndarray:
        """
        INFO04: Fisher information - Measuring local predictability.
        VECTORIZED VERSION
        """
        returns = self._returns_log()
        fisher = np.full(self.n, np.nan)
        
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(returns, (window-1, 0), constant_values=np.nan), 
            window
        )
        
        for i in range(window, min(self.n, len(windows) + window - 1)):
            ret_window = windows[i - window]
            ret_window = ret_window[~np.isnan(ret_window)]
            
            if len(ret_window) < 2:
                continue
                
            ret_std = np.std(ret_window)
            if ret_std < self.eps:
                continue
                
            ret_norm = ret_window / ret_std
            bins = max(2, int(np.sqrt(len(ret_norm))))
            
            # Use optimized histogram calculation
            hist, bin_edges = np.histogram(ret_norm, bins=bins, density=True)
            bin_width = bin_edges[1] - bin_edges[0]
            
            # Vectorized Fisher information calculation
            dp_dx = np.gradient(hist, bin_width)
            valid_mask = hist > self.eps
            fisher_val = np.sum((dp_dx[valid_mask] ** 2) / hist[valid_mask]) * bin_width
            
            fisher[i] = fisher_val
            
        return fisher

    def approximate_entropy(self, window: int = 100, m: int = 2, r: float = 0.2) -> np.ndarray:
        """
        INFO05: Approximate entropy - Similar to sample entropy.
        VECTORIZED VERSION
        """
        returns = self._returns_log()
        approx_ent = np.full(self.n, np.nan)
        
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(returns, (window-1, 0), constant_values=np.nan), 
            window
        )
        
        for i in range(window, min(self.n, len(windows) + window - 1)):
            ret_window = windows[i - window]
            ret_window = ret_window[~np.isnan(ret_window)]
            
            if len(ret_window) < m + 1:
                continue
                
            ret_std = np.std(ret_window)
            if ret_std < self.eps:
                continue
                
            ret_norm = ret_window / ret_std
            L = len(ret_norm)
            
            # Create templates for m and m+1
            templates_m = np.lib.stride_tricks.sliding_window_view(ret_norm, m)
            templates_m1 = np.lib.stride_tricks.sliding_window_view(ret_norm, m + 1)
            
            # Calculate distances for m
            if len(templates_m) > 0:
                tm1 = templates_m[:, np.newaxis, :]
                tm2 = templates_m[np.newaxis, :, :]
                dist_m = np.max(np.abs(tm1 - tm2), axis=2)
                
                # Calculate phi(m)
                matches_m = np.sum(dist_m <= r, axis=1)
                valid_m = matches_m > 0
                if np.any(valid_m):
                    phi_m = np.mean(np.log(matches_m[valid_m] / (L - m + 1 + self.eps)))
                else:
                    continue
                    
                # Calculate distances for m+1
                if len(templates_m1) > 0:
                    tm1_1 = templates_m1[:, np.newaxis, :]
                    tm2_1 = templates_m1[np.newaxis, :, :]
                    dist_m1 = np.max(np.abs(tm1_1 - tm2_1), axis=2)
                    
                    # Calculate phi(m+1)
                    matches_m1 = np.sum(dist_m1 <= r, axis=1)
                    valid_m1 = matches_m1 > 0
                    if np.any(valid_m1):
                        phi_m1 = np.mean(np.log(matches_m1[valid_m1] / (L - m + self.eps)))
                        approx_ent[i] = phi_m - phi_m1
                        
        return approx_ent

    # ===================== CAUSAL & INFORMATION FLOW =====================

    def transfer_entropy(self, source_lag: int = 5, target_lag: int = 15, window: int = 100, m: int = 2) -> np.ndarray:
        """
        INFO06: Transfer entropy - Information flow from faster to slower timeframes.
        VECTORIZED VERSION
        """
        returns = self._returns_log()
        te = np.full(self.n, np.nan)
        
        start_idx = max(source_lag, target_lag) + window
        
        # Pre-calculate windows for slow and fast series
        for i in range(start_idx, self.n):
            # Use slicing to get windows
            slow_start = i - target_lag - window
            fast_start = i - source_lag - window
            
            slow = returns[slow_start:i - target_lag:max(1, target_lag // source_lag)]
            fast = returns[fast_start:i - source_lag]
            
            if len(slow) > m and len(fast) > m:
                min_len = min(len(slow), len(fast))
                slow_trim = slow[-min_len:]
                fast_trim = fast[-min_len:]
                
                if np.std(slow_trim) > self.eps and np.std(fast_trim) > self.eps:
                    corr = np.corrcoef(slow_trim, fast_trim)[0, 1]
                    te[i] = 0.0 if np.isnan(corr) else corr
                    
        return te

    def granger_f_stat(self, lag: int = 5, window: int = 100) -> np.ndarray:
        """
        INFO07: Granger F-statistic - Vector autoregression significance.
        VECTORIZED VERSION
        """
        returns = self._returns_log()
        f_stats = np.full(self.n, np.nan)
        
        # Create lagged matrix using vectorized operations
        start_idx = lag + window
        
        for i in range(start_idx, self.n):
            ret_window = returns[i - window:i]
            ret_window = ret_window[~np.isnan(ret_window)]
            
            if len(ret_window) < lag + 1:
                continue
                
            y = ret_window[lag:]
            X_lags = []
            
            # Create lagged variables
            for l in range(1, lag + 1):
                X_lags.append(ret_window[lag - l:-l] if l < lag else ret_window[:-l])
            
            # Stack lagged variables
            X = np.column_stack(X_lags) if X_lags else np.array([]).reshape(len(y), 0)
            
            if len(X) > 0:
                # Calculate residuals using linear algebra
                X_with_const = np.column_stack([np.ones(len(X)), X])
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                y_pred = X_with_const @ beta
                residuals = y - y_pred
                
                var_full = np.var(y)
                var_residual = np.var(residuals)
                
                if var_full > self.eps:
                    # Calculate F-statistic
                    n = len(y)
                    k = X.shape[1] + 1  # +1 for constant
                    f_val = ((var_full - var_residual) / (k - 1)) / (var_residual / (n - k))
                    f_stats[i] = f_val
                    
        return f_stats

    def convergent_cross_map(self, lag: int = 10, window: int = 100) -> np.ndarray:
        """
        INFO08: Convergent Cross-Mapping - Bidirectional causal relationship strength.
        VECTORIZED VERSION
        """
        returns = self._returns_log()
        ccm = np.full(self.n, np.nan)
        
        start_idx = lag + window
        
        for i in range(start_idx, self.n):
            ret_window = returns[i - window:i]
            ret_window = ret_window[~np.isnan(ret_window)]
            
            if len(ret_window) <= lag:
                continue
                
            X = ret_window[:-lag]
            X_lagged = ret_window[lag:]
            
            if len(X) > 1 and len(X_lagged) > 1:
                if np.std(X) > self.eps and np.std(X_lagged) > self.eps:
                    # Use numpy's correlation with validation
                    cov_matrix = np.cov(X, X_lagged)
                    if cov_matrix[0, 0] > self.eps and cov_matrix[1, 1] > self.eps:
                        corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
                        ccm[i] = 0.0 if np.isnan(corr) else corr
                        
        return ccm

    def partial_directed_coherence(self, window: int = 100, lag: int = 5) -> np.ndarray:
        """
        INFO09: Partial Directed Coherence - Frequency-domain directed information.
        VECTORIZED VERSION
        """
        returns = self._returns_log()
        pdc = np.full(self.n, np.nan)
        
        start_idx = lag + window
        
        for i in range(start_idx, self.n):
            ret_window = returns[i - window:i]
            ret_window = ret_window[~np.isnan(ret_window)]
            
            if len(ret_window) <= lag:
                continue
                
            # Use FFT for spectral analysis
            fft_vals = np.fft.fft(ret_window - np.mean(ret_window))
            fft_mag = np.abs(fft_vals)
            
            if len(fft_mag) > lag:
                # Calculate phase difference
                phase_diff = np.angle(fft_vals[lag]) - np.angle(fft_vals[lag // 2])
                pdc[i] = phase_diff
                
        return pdc

    def information_geometry_distance(self, window: int = 100, ref_window: int = 200) -> np.ndarray:
        """
        INFO10: Information geometry distance - KL divergence between return distributions.
        VECTORIZED VERSION
        """
        returns = self._returns_log()
        kl_div = np.full(self.n, np.nan)
        
        for i in range(ref_window, self.n):
            current_ret = returns[i - window:i]
            ref_ret = returns[i - ref_window:i - (ref_window - window)]
            
            current_ret = current_ret[~np.isnan(current_ret)]
            ref_ret = ref_ret[~np.isnan(ref_ret)]
            
            if len(current_ret) < 2 or len(ref_ret) < 2:
                continue
                
            # Use common bins for both distributions
            all_data = np.concatenate([current_ret, ref_ret])
            bins = max(5, int(np.sqrt(min(len(current_ret), len(ref_ret)))))
            
            hist_current, bin_edges = np.histogram(current_ret, bins=bins, density=True, range=(np.min(all_data), np.max(all_data)))
            hist_ref, _ = np.histogram(ref_ret, bins=bin_edges, density=True)
            
            # Add epsilon to avoid division by zero
            hist_current = hist_current + self.eps
            hist_ref = hist_ref + self.eps
            
            # Normalize
            hist_current = hist_current / np.sum(hist_current)
            hist_ref = hist_ref / np.sum(hist_ref)
            
            # Vectorized KL divergence calculation
            valid_mask = (hist_current > self.eps) & (hist_ref > self.eps)
            if np.any(valid_mask):
                kl_val = np.sum(hist_current[valid_mask] * np.log(hist_current[valid_mask] / hist_ref[valid_mask]))
                kl_div[i] = kl_val
                
        return kl_div