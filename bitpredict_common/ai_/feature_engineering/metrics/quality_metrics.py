"""
Feature Quality Metrics Evaluator
OPTIMIZED high-performance module for computing IC-based feature evaluation metrics.
All calculations are vectorized using NumPy for optimal performance.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import savgol_filter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class FeatureMetricsEvaluator:
    """
    HIGH-PERFORMANCE OPTIMIZED feature evaluation metrics calculator.
    
    This class computes multiple IC-based metrics for feature quality assessment.
    All operations are fully vectorized across features for maximum performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing all required columns:
        - Feature columns to evaluate
        - Return columns (return_1, return_3, return_5, return_10, return_20, return_t+1, return_t+5)
        - Market columns (market_cap, volume, bid_ask_spread, market_return, dollar_volume)
    feature_columns : List[str]
        List of feature column names to evaluate
    config : Dict, optional
        Configuration dictionary with parameters:
        - transaction_cost: float (default=0.001 for 10bps)
        - halflife: int (default=252 for business days)
        - rolling_window: int (default=100 for stability calculations)
        - regimes: Dict with regime masks (optional)
        - cluster_labels: np.array for clustering metrics (optional)
    """
    
    # Required columns validation
    REQUIRED_COLUMNS = [
        'return_1', 'return_3', 'return_5', 'return_10', 'return_20',
        'return_t+1', 'return_t+5',
        'market_cap', 'volume', 'bid_ask_spread', 'market_return', 'dollar_volume'
    ]
    
    # Metric thresholds from the specification
    THRESHOLDS = {
        'rank_ic_spearman': 0.02,
        'ic_t_statistic': 2.0,
        'persistence_ratio': 1.0,
        'feature_turnover_rate': 0.3,
        'cost_adjusted_ic': 0.015,
        'rolling_ic_stability_ratio': 2.0,
        'maximum_ic_drawdown': 0.7,
        'time_to_recovery': 180,  # 6 months in days
        'regime_ic_consistency': 0.5,
        'regime_transition_sensitivity': 0.5,
        'skew_exposure': 0.1,
        'volume_liquidity_correlation': 0.0,
        'data_availability': 0.95,
        'signal_to_noise_ratio': 0.5,
        'delay_sensitivity': 0.3,
        'predictive_orthogonality': 0.7,
        'cluster_purity': 0.3
    }
    
    # Weights for final score calculation
    SCORE_WEIGHTS = {
        'predictive_power': 0.4,
        'economic_viability': 0.25,
        'stability': 0.2,
        'risk_management': 0.1,
        'operational': 0.05
    }
    
    def __init__(self, df: pd.DataFrame, feature_columns: List[str], config: Optional[Dict] = None):
        """Initialize the evaluator with data and configuration."""
        
        # Calculate required features BEFORE validation (skips existing columns)
        self.df = self._calculate_required_features(df, shift=1)
        
        self.feature_columns = feature_columns
        self.config = config or {}
        
        # Set default configuration values
        self.transaction_cost = self.config.get('transaction_cost', 0.001)  # 10 bps
        self.halflife = self.config.get('halflife', 252)  # business days
        self.rolling_window = min(self.config.get('rolling_window', 100), len(df) // 2)
        self.regimes = self.config.get('regimes', self._create_default_regimes())
        self.cluster_labels = self.config.get('cluster_labels', None)
        
        # Validate input data
        self._validate_input()
        
        # Convert to NumPy arrays for performance
        self._prepare_arrays()
        
        # Cache for rolling IC computations
        self._rolling_ic_cache = {}

    
    def _prepare_arrays(self) -> None:
        """Convert DataFrame columns to NumPy arrays for vectorized operations."""
        
        # Drop datetime columns if present, as they are not needed for metric calculations
        self.feature_columns = self.df.select_dtypes(exclude=["datetime64[ns]", "datetime64[ns, UTC]"]).select_dtypes(include="number").columns.tolist()

        # Ensure feature columns are numeric; coerce non-numeric values to NaN
        self.df[self.feature_columns] = self.df[self.feature_columns].apply(pd.to_numeric, errors='coerce')
        
        # Optionally warn about features that became all NaN after conversion
        non_numeric_features = [col for col in self.feature_columns if self.df[col].isna().all()]
        if non_numeric_features:
            print(f"Warning: The following feature columns are non-numeric and will be all NaN: {non_numeric_features}")
        
        # Feature arrays (n_samples x n_features)
        self.features = self.df[self.feature_columns].values

        # Return arrays
        self.returns = {
            'return_1': self.df['return_1'].values,
            'return_3': self.df['return_3'].values,
            'return_5': self.df['return_5'].values,
            'return_10': self.df['return_10'].values,
            'return_20': self.df['return_20'].values,
            'return_t+1': self.df['return_t+1'].values,
            'return_t+5': self.df['return_t+5'].values
        }
        
        # Market data arrays
        self.market_data = {
            'market_cap': self.df['market_cap'].values,
            'volume': self.df['volume'].values,
            'bid_ask_spread': self.df['bid_ask_spread'].values,
            'market_return': self.df['market_return'].values,
            'dollar_volume': self.df['dollar_volume'].values
        }
        
        self.n_samples, self.n_features = self.features.shape
        
    def _validate_input(self) -> None:
        """Validate that all required columns exist in the DataFrame."""
        missing_cols = []
        
        # Check required columns
        for col in self.REQUIRED_COLUMNS:
            if col not in self.df.columns:
                missing_cols.append(col)
        
        # Check feature columns
        for col in (self.feature_columns or []):
            if col not in self.df.columns:
                missing_cols.append(col)

        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for NaN alignment
        if self.df.isnull().any().any():
            print("Warning: DataFrame contains NaN values. Metrics will be computed on available data.")
    
    def _calculate_required_features(self, result: pd.DataFrame, shift=1):
        """
        Calculate only the required columns for the model with shift applied.
        Only calculates columns that don't already exist in the DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV columns: open, high, low, close, volume
        shift : int, default=1
            Number of periods to shift features (1 = shift by 1 period)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with required columns added (all shifted)
        """
        
        # Historical returns (with shift) - only if not already present
        if 'return_1' not in result.columns:
            result['return_1'] = result['close'].pct_change(1).shift(shift)
        
        if 'return_3' not in result.columns:
            result['return_3'] = result['close'].pct_change(3).shift(shift)
        
        if 'return_5' not in result.columns:
            result['return_5'] = result['close'].pct_change(5).shift(shift)
        
        if 'return_10' not in result.columns:
            result['return_10'] = result['close'].pct_change(10).shift(shift)
        
        if 'return_20' not in result.columns:
            result['return_20'] = result['close'].pct_change(20).shift(shift)
        
        if 'return_50' not in result.columns:
            result['return_50'] = result['close'].pct_change(50).shift(shift)
        
        # Future returns (targets - NO shift) - only if not already present
        if 'return_t+1' not in result.columns:
            result['return_t+1'] = result['close'].pct_change(1).shift(-1).shift(shift)
        
        if 'return_t+5' not in result.columns:
            result['return_t+5'] = result['close'].pct_change(5).shift(-5).shift(shift)
        
        # Dollar volume (with shift) - only if not already present
        if 'dollar_volume' not in result.columns:
            result['dollar_volume'] = (result['close'] * result['volume']).shift(shift)
        
        # Dummy market data columns (with shift)
        # Add realistic dummy data for missing external columns

        if 'market_cap' not in result.columns:
            # Simulate market cap as 1000x average daily volume
            result['market_cap'] = (result['close'] * result['volume'].rolling(20).mean() * 1000).shift(shift)
        
        if 'bid_ask_spread' not in result.columns:
            # Create realistic bid-ask spread as percentage of price
            spread_pct = 0.001  # 10 bps
            result['bid_ask_spread'] = spread_pct * result['close'].shift(shift)
        
        if 'market_return' not in result.columns:
            result['market_return'] = result['close'].pct_change(1).shift(shift)
            
        # Ensure volume column exists
        if 'volume' not in result.columns and 'volume' in result.columns:
            pass  # Already exists
        
        elif 'volume' not in result.columns:
            result['volume'] = result.get('volume', 1000000)  # Default volume
            
        return result
    
    def _create_default_regimes(self):
        """
        Create default market regimes if none are provided.
        Based on volatility and market return characteristics.
        """
        if len(self.df) < 50:
            return {}
            
        # Calculate rolling volatility
        returns = self.df['close'].pct_change()
        rolling_vol = returns.rolling(20).std()
        vol_regime = (rolling_vol > rolling_vol.median()).fillna(False)
        
        # Calculate market trend
        rolling_return = returns.rolling(10).mean()
        trend_regime = (rolling_return > 0).fillna(False)
        
        # Create regime masks
        regimes = {
            'high_vol_bull': (vol_regime & trend_regime).values,
            'high_vol_bear': (vol_regime & ~trend_regime).values,
            'low_vol_bull': (~vol_regime & trend_regime).values,
            'low_vol_bear': (~vol_regime & ~trend_regime).values,
        }
        
        # Only keep regimes with sufficient data
        regimes = {k: v for k, v in regimes.items() if np.sum(v) > 20}
        
        return regimes

    def _fast_spearman_correlation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute Spearman correlation between X (n_samples x n_features) and y (n_samples).
        Optimized version using numpy operations.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_features = X.shape[1]
        corr = np.zeros(n_features)
        
        # Rank y once for all features
        y_valid_mask = ~np.isnan(y)
        y_valid = y[y_valid_mask]
        if len(y_valid) < 2:
            return corr
            
        y_ranks = stats.rankdata(y_valid)
        
        for i in range(n_features):
            # Get valid observations for this feature
            x_i = X[:, i]
            mask = ~np.isnan(x_i) & y_valid_mask
            
            if np.sum(mask) < 2:
                corr[i] = 0
                continue
            
            x_valid = x_i[mask]
            y_masked = y[mask]
            
            # Rank x
            x_ranks = stats.rankdata(x_valid)
            
            # Use numpy's corrcoef for efficiency
            corr_matrix = np.corrcoef(x_ranks, y_ranks[:len(x_ranks)])
            if corr_matrix.shape == (2, 2):
                corr[i] = corr_matrix[0, 1]
        
        return corr

    def _fast_correlation_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fast correlation between X (n_samples x n_features) and y (n_samples).
        Optimized for many features.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Center the data
        X_centered = X - np.nanmean(X, axis=0, keepdims=True)
        y_centered = y - np.nanmean(y)
        
        # Compute covariance
        cov = np.nansum(X_centered * y_centered[:, np.newaxis], axis=0)
        
        # Compute standard deviations
        X_std = np.sqrt(np.nansum(X_centered ** 2, axis=0))
        y_std = np.sqrt(np.nansum(y_centered ** 2))
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = cov / (X_std * y_std)
            corr[X_std == 0] = 0
        
        return np.nan_to_num(corr, nan=0.0)

    def _rank_data_fast(self, data: np.ndarray) -> np.ndarray:
        """Convert data to ranks efficiently, handling NaNs."""
        ranks = np.empty_like(data)
        
        for i in range(data.shape[1]):
            col = data[:, i]
            valid_mask = ~np.isnan(col)
            
            if np.sum(valid_mask) < 2:
                ranks[:, i] = 0
                continue
            
            # Rank only valid values
            valid_values = col[valid_mask]
            sorted_indices = np.argsort(valid_values)
            rank_values = np.zeros_like(valid_values)
            rank_values[sorted_indices] = np.arange(len(valid_values))
            
            # Normalize to [0, 1]
            if len(rank_values) > 0:
                rank_values = rank_values / (len(rank_values) - 1) if len(rank_values) > 1 else 0
            
            # Place ranks back
            col_ranked = np.empty_like(col)
            col_ranked[valid_mask] = rank_values
            col_ranked[~valid_mask] = np.nan
            ranks[:, i] = col_ranked
        
        return ranks
    
    def compute_rank_ic_spearman(self) -> np.ndarray:
        """
        Compute Rank IC (Spearman correlation).
        
        Formula: IC = corr(rank(feature_t), rank(return_{t+5}))
        Returns correlation coefficients for each feature.
        """
        # Use optimized correlation calculation
        rank_ic = self._fast_spearman_correlation(self.features, self.returns['return_t+5'])
        return rank_ic
    
    def compute_ic_decay_profile(self) -> Dict[str, np.ndarray]:
        """
        Compute IC decay profile for multiple horizons.
        
        Formula: IC_k = corr(feature_t, return_{t+k}) for k in [1, 3, 5, 10, 20]
        Returns dictionary with IC values for each horizon.
        """
        horizons = [1, 3, 5, 10, 20]
        decay_profile = {}
        
        for horizon in horizons:
            return_key = f'return_{horizon}'
            decay_profile[f'ic_decay_{horizon}'] = self._fast_correlation_matrix(
                self.features, self.returns[return_key]
            )
        
        return decay_profile
    
    def compute_weighted_ic(self) -> np.ndarray:
        """
        Compute Weighted IC with exponential decay using rolling IC.
        
        Formula: Weight = exp(-(T-t)/halflife)
                 IC_weighted = Σ(IC_t * Weight_t) / Σ(Weight_t)
        """
        # Compute rolling IC first
        rolling_ic = self._compute_rolling_ic_fast(window=min(60, self.n_samples // 4))
        
        if rolling_ic.shape[0] == 0:
            return np.zeros(self.n_features)
        
        n_periods = rolling_ic.shape[0]
        
        # Create exponential weights
        t = np.arange(n_periods)[::-1]  # Reverse so recent periods have higher weight
        weights = np.exp(-t / min(self.halflife, n_periods))
        
        # Weighted average along time axis (vectorized)
        valid_mask = ~np.isnan(rolling_ic)
        weighted_sum = np.nansum(rolling_ic * weights[:, np.newaxis], axis=0)
        weight_sum = np.nansum(weights[:, np.newaxis] * valid_mask, axis=0)
        
        # Avoid division by zero
        weighted_ic = np.zeros(self.n_features)
        mask = weight_sum > 0
        weighted_ic[mask] = weighted_sum[mask] / weight_sum[mask]
        
        return weighted_ic
    
    def compute_ic_t_statistic(self) -> np.ndarray:
        """
        Compute IC t-statistic for significance testing.
        
        Formula: t_stat = mean(IC_rolling) / (std(IC_rolling) / sqrt(n_periods))
        """
        rolling_ic = self._compute_rolling_ic_fast(window=min(60, self.n_samples // 4))
        
        if rolling_ic.shape[0] == 0:
            return np.zeros(self.n_features)
        
        # Compute t-statistics (vectorized)
        valid_mask = ~np.isnan(rolling_ic)
        n_valid = np.sum(valid_mask, axis=0)
        
        mean_ic = np.nanmean(rolling_ic, axis=0)
        std_ic = np.nanstd(rolling_ic, axis=0, ddof=1)
        
        t_stats = np.zeros(self.n_features)
        mask = (std_ic > 0) & (n_valid >= 2)
        t_stats[mask] = mean_ic[mask] / (std_ic[mask] / np.sqrt(n_valid[mask]))
        
        return t_stats
    
    def compute_persistence_ratio(self) -> np.ndarray:
        """
        Compute Persistence Ratio across quarters.
        
        Binary: 1 if same sign in first and last third of data
        """
        rolling_ic = self._compute_rolling_ic_fast(window=min(60, self.n_samples // 4))
        
        if rolling_ic.shape[0] < 3:
            return np.ones(self.n_features)
        
        # Split into thirds
        split_idx = rolling_ic.shape[0] // 3
        
        persistence = np.ones(self.n_features)
        
        # Vectorized computation
        first_third_mean = np.nanmean(rolling_ic[:split_idx], axis=0)
        last_third_mean = np.nanmean(rolling_ic[-split_idx:], axis=0)
        
        # Check if signs are consistent
        sign_consistent = (first_third_mean * last_third_mean) >= 0
        persistence[~sign_consistent] = 0
        
        return persistence
    
    def compute_feature_turnover_rate(self) -> np.ndarray:
        """
        Compute Feature Turnover Rate using z-score signs.
        
        Formula: turnover = mean(|sign(zscore_t) - sign(zscore_{t-1})|)
        """
        # Standardize features first to get signs that change
        feature_means = np.nanmean(self.features, axis=0, keepdims=True)
        feature_stds = np.nanstd(self.features, axis=0, keepdims=True)
        feature_stds[feature_stds == 0] = 1  # Avoid division by zero
        
        zscores = (self.features - feature_means) / feature_stds
        
        # Compute signs of z-scores
        signs = np.sign(zscores)
        signs[np.isnan(signs)] = 0
        
        # Compute absolute difference in signs (vectorized)
        sign_diff = np.abs(signs[1:] - signs[:-1])
        
        # Average along time axis
        turnover = np.nanmean(sign_diff, axis=0) / 2  # Divide by 2 as diff can be 0, 1, or 2
        
        return turnover
    
    def compute_cost_adjusted_ic(self, rank_ic: np.ndarray, turnover: np.ndarray) -> np.ndarray:
        """
        Compute Cost-Adjusted IC.
        
        Formula: IC_adj = IC_raw - (0.5 * turnover * transaction_cost)
        """
        cost_adjusted_ic = rank_ic - (0.5 * turnover * self.transaction_cost)
        return cost_adjusted_ic
    
    def compute_dollar_volume_sensitivity(self) -> np.ndarray:
        """
        Compute Dollar Volume Sensitivity.
        
        Perform regression: IC ~ log(market_cap) + log(volume)
        Return coefficient magnitude.
        """
        # Use rolling IC and market data
        rolling_ic = self._compute_rolling_ic_fast(window=min(60, self.n_samples // 4))
        
        if rolling_ic.shape[0] < 10:
            return np.zeros(self.n_features)
        
        sensitivities = np.zeros(self.n_features)
        
        # Align market data with rolling IC
        start_idx = self.n_samples - rolling_ic.shape[0]
        market_cap_aligned = self.market_data['market_cap'][start_idx:]
        volume_aligned = self.market_data['volume'][start_idx:]
        
        # Log transform
        log_mcap = np.log(market_cap_aligned + 1)
        log_vol = np.log(volume_aligned + 1)
        
        # Compute correlations for all features at once
        for i in range(self.n_features):
            ic_series = rolling_ic[:, i]
            valid_mask = ~np.isnan(ic_series)
            
            if np.sum(valid_mask) < 10:
                continue
            
            ic_valid = ic_series[valid_mask]
            mcap_valid = log_mcap[valid_mask]
            vol_valid = log_vol[valid_mask]
            
            # Simple regression using correlation
            corr_mcap = np.corrcoef(ic_valid, mcap_valid)[0, 1] if len(ic_valid) > 1 else 0
            corr_vol = np.corrcoef(ic_valid, vol_valid)[0, 1] if len(ic_valid) > 1 else 0
            
            sensitivities[i] = np.abs(corr_mcap + corr_vol) / 2
        
        return sensitivities
    
    def compute_bid_ask_spread_sensitivity(self) -> np.ndarray:
        """
        Compute Bid-Ask Spread Sensitivity.
        
        Formula: delta_IC = |IC_low_spread - IC_high_spread|
        """
        # Get spread data
        spread = self.market_data['bid_ask_spread']
        median_spread = np.nanmedian(spread)
        
        # Create masks (exclude exactly median)
        low_spread_mask = spread < median_spread
        high_spread_mask = spread > median_spread
        
        # Ensure sufficient samples
        if np.sum(low_spread_mask) < 10 or np.sum(high_spread_mask) < 10:
            return np.zeros(self.n_features)
        
        # Compute IC in both regimes
        ic_low = self._fast_correlation_matrix(
            self.features[low_spread_mask], 
            self.returns['return_t+5'][low_spread_mask]
        )
        ic_high = self._fast_correlation_matrix(
            self.features[high_spread_mask], 
            self.returns['return_t+5'][high_spread_mask]
        )
        
        delta_ic = np.abs(ic_low - ic_high)
        return delta_ic
    
    def _compute_rolling_ic_fast(self, window: int = 100) -> np.ndarray:
        """
        OPTIMIZED: Compute rolling Rank IC (Spearman correlation) series for each feature.
        
        This function computes the Spearman correlation between feature ranks
        and return ranks over rolling windows, consistent with the Rank IC metric.
        
        Parameters
        ----------
        window : int
            Rolling window size
            
        Returns
        -------
        np.ndarray
            Rolling IC array of shape (n_windows, n_features)
        """
        # Check cache first
        cache_key = f'rolling_ic_{window}'
        if hasattr(self, '_rolling_ic_cache') and cache_key in self._rolling_ic_cache:
            return self._rolling_ic_cache[cache_key]
        
        window = min(window, self.n_samples // 2)
        
        if window < 10 or self.n_samples < window * 2:
            result = np.zeros((0, self.n_features))
            self._rolling_ic_cache[cache_key] = result
            return result
        
        n_windows = self.n_samples - window + 1
        rolling_ic = np.full((n_windows, self.n_features), np.nan)
        
        # Get the target returns
        target_returns = self.returns['return_t+5']
        
        # Pre-allocate arrays for efficiency
        features_window = np.empty((window, self.n_features))
        
        # OPTIMIZATION: Process in chunks to reduce memory overhead
        chunk_size = min(1000, n_windows)
        
        for chunk_start in range(0, n_windows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_windows)
            chunk_size_actual = chunk_end - chunk_start
            
            # Pre-allocate chunk arrays
            chunk_ic = np.zeros((chunk_size_actual, self.n_features))
            
            for i in range(chunk_size_actual):
                actual_idx = chunk_start + i
                window_slice = slice(actual_idx, actual_idx + window)
                
                # Extract window data
                features_window = self.features[window_slice]
                returns_window = target_returns[window_slice]
                
                # Get valid mask for this window
                valid_mask = ~np.isnan(returns_window)
                
                # Compute correlation for all features at once
                for j in range(self.n_features):
                    feature_col = features_window[:, j]
                    feature_valid_mask = ~np.isnan(feature_col)
                    combined_mask = valid_mask & feature_valid_mask
                    
                    if np.sum(combined_mask) < 2:
                        chunk_ic[i, j] = 0
                        continue
                    
                    # Use numpy's correlation for speed
                    try:
                        corr = np.corrcoef(feature_col[combined_mask], returns_window[combined_mask])[0, 1]
                        chunk_ic[i, j] = corr if not np.isnan(corr) else 0
                    except:
                        chunk_ic[i, j] = 0
            
            # Store chunk results
            rolling_ic[chunk_start:chunk_end] = chunk_ic
        
        # Cache the result
        if not hasattr(self, '_rolling_ic_cache'):
            self._rolling_ic_cache = {}
        self._rolling_ic_cache[cache_key] = rolling_ic
        
        return rolling_ic

    def compute_rolling_ic_stability_ratio(self) -> np.ndarray:
        """
        Compute Rolling IC Stability Ratio.
        
        Formula: stability = mean(IC_rolling) / std(IC_rolling)
        """
        rolling_ic = self._compute_rolling_ic_fast(window=self.rolling_window)
        
        if rolling_ic.shape[0] == 0:
            return np.zeros(self.n_features)
        
        # Compute stability ratio (vectorized)
        mean_ic = np.nanmean(rolling_ic, axis=0)
        std_ic = np.nanstd(rolling_ic, axis=0, ddof=1)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            stability = np.abs(mean_ic) / (std_ic + 1e-10)
            stability[std_ic == 0] = np.inf
        
        # Cap extreme values
        stability = np.clip(stability, -100, 100)
        
        return stability
    
    def compute_maximum_ic_drawdown(self) -> np.ndarray:
        """
        Compute Maximum IC Drawdown.
        
        Formula: IC_drawdown = (IC_peak - IC_trough) / IC_peak
        Measures worst historical decay from peak.
        Range: [0, 1] where 1 = 100% drawdown
        """
        rolling_ic = self._compute_rolling_ic_fast(window=self.rolling_window)
        
        if rolling_ic.shape[0] == 0:
            return np.zeros(self.n_features)
        
        max_drawdown = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            ic_series = rolling_ic[:, i]
            valid_mask = ~np.isnan(ic_series)
            
            if np.sum(valid_mask) < 10:
                continue
            
            ic_valid = ic_series[valid_mask]
            
            # Compute running maximum
            running_max = np.maximum.accumulate(ic_valid)
            
            # Only compute drawdown if peak is significant (> 1%)
            peak_value = np.nanmax(running_max)
            if np.abs(peak_value) < 0.01:
                max_drawdown[i] = 0
                continue
            
            # Compute drawdowns using formula: (IC_peak - IC_trough) / IC_peak
            drawdowns = (running_max - ic_valid) / running_max
            
            # Clip to valid range [0, 1]
            drawdowns = np.clip(drawdowns, 0, 1)
            
            # Get maximum drawdown
            if len(drawdowns) > 0:
                max_drawdown[i] = np.nanmax(drawdowns)
        
        return max_drawdown

    def compute_time_to_recovery(self) -> np.ndarray:
        """
        Compute Time to Recovery from IC drawdown.
        
        How long after decay does feature recover to 90% of peak?
        """
        rolling_ic = self._compute_rolling_ic_fast(window=self.rolling_window)
        
        if rolling_ic.shape[0] == 0:
            return np.zeros(self.n_features)
        
        recovery_times = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            ic_series = rolling_ic[:, i]
            valid_mask = ~np.isnan(ic_series)
            
            if np.sum(valid_mask) < 20:
                continue
            
            ic_valid = ic_series[valid_mask]
            
            # Find global peak
            peak_idx = np.nanargmax(ic_valid)
            peak_value = ic_valid[peak_idx]
            
            # Look for subsequent trough and recovery
            if peak_idx < len(ic_valid) - 5:
                post_peak = ic_valid[peak_idx:]
                trough_idx = np.nanargmin(post_peak)
                trough_value = post_peak[trough_idx]
                
                # Recovery target
                target_value = 0.9 * peak_value
                
                # Find when IC recovers to target
                recovery_mask = post_peak[trough_idx:] >= target_value
                
                if np.any(recovery_mask):
                    recovery_idx = np.argmax(recovery_mask)
                    recovery_times[i] = recovery_idx
                else:
                    recovery_times[i] = len(post_peak) - trough_idx
        
        return recovery_times
    
    def compute_regime_ic_consistency(self) -> np.ndarray:
        """
        Compute Regime IC Consistency.
        
        Should work across multiple regimes.
        """
        if not self.regimes:
            return np.zeros(self.n_features)
        
        regime_ics = []
        
        for regime_name, regime_mask in self.regimes.items():
            if np.sum(regime_mask) > 10:  # Minimum samples
                ic = self._fast_correlation_matrix(
                    self.features[regime_mask], 
                    self.returns['return_t+5'][regime_mask]
                )
                regime_ics.append(ic)
        
        if len(regime_ics) < 2:
            return np.zeros(self.n_features)
        
        regime_ics_array = np.column_stack(regime_ics)
        
        # Compute consistency (vectorized)
        mean_ic = np.nanmean(regime_ics_array, axis=1)
        std_ic = np.nanstd(regime_ics_array, axis=1, ddof=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            consistency = 1 - (std_ic / (np.abs(mean_ic) + 1e-10))
        
        consistency = np.nan_to_num(consistency, nan=0.0, posinf=1.0, neginf=0.0)
        consistency = np.clip(consistency, 0, 1)
        
        return consistency
    
    def compute_regime_transition_sensitivity(self) -> np.ndarray:
        """
        Compute Regime Transition Sensitivity.
        
        Should not break during regime transitions.
        """
        if not self.regimes or len(self.regimes) < 2:
            return np.zeros(self.n_features)
        
        # Find regime transitions
        regime_changes = np.zeros(self.n_samples, dtype=bool)
        regime_array = np.zeros(self.n_samples)
        
        for i, (regime_name, regime_mask) in enumerate(self.regimes.items()):
            regime_array[regime_mask] = i + 1
        
        regime_changes[1:] = regime_array[1:] != regime_array[:-1]
        
        if np.sum(regime_changes) < 5:
            return np.zeros(self.n_features)
        
        sensitivities = np.zeros(self.n_features)
        
        # Use rolling correlation with small window
        window = min(10, self.n_samples // 20)
        
        # Pre-compute rolling correlations for efficiency
        rolling_corrs = np.zeros((self.n_samples - window + 1, self.n_features))
        
        for i in range(self.n_features):
            # Compute rolling correlation
            feature_series = self.features[:, i]
            return_series = self.returns['return_t+5']
            
            for j in range(self.n_samples - window + 1):
                window_slice = slice(j, j + window)
                feature_window = feature_series[window_slice]
                return_window = return_series[window_slice]
                
                valid_mask = ~np.isnan(feature_window) & ~np.isnan(return_window)
                if np.sum(valid_mask) < 5:
                    rolling_corrs[j, i] = np.nan
                else:
                    corr = np.corrcoef(feature_window[valid_mask], return_window[valid_mask])[0, 1]
                    rolling_corrs[j, i] = corr
        
        # Find changes around regime transitions
        change_indices = np.where(regime_changes)[0]
        
        for i in range(self.n_features):
            changes = []
            
            for idx in change_indices:
                if idx > window and idx < len(rolling_corrs) - window:
                    before = np.nanmean(rolling_corrs[idx-window:idx, i])
                    after = np.nanmean(rolling_corrs[idx:idx+window, i])
                    
                    if not np.isnan(before) and not np.isnan(after) and abs(before) > 0.01:
                        change = abs(after - before) / (abs(before) + 1e-10)
                        changes.append(change)
            
            if changes:
                sensitivities[i] = np.median(changes)
        
        return sensitivities
    
    def compute_conditional_ic(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Conditional IC for normal and tail market conditions.
        
        IC_normal when |market_return| < 1%
        IC_tail when |market_return| > 3%
        """
        market_return = self.market_data['market_return']
        
        # Normal market condition
        normal_mask = np.abs(market_return) < 0.01
        if np.sum(normal_mask) > 20:
            ic_normal = self._fast_correlation_matrix(
                self.features[normal_mask], 
                self.returns['return_t+5'][normal_mask]
            )
        else:
            ic_normal = np.zeros(self.n_features)
        
        # Tail market condition
        tail_mask = np.abs(market_return) > 0.03
        if np.sum(tail_mask) > 5:
            ic_tail = self._fast_correlation_matrix(
                self.features[tail_mask], 
                self.returns['return_t+5'][tail_mask]
            )
        else:
            ic_tail = np.zeros(self.n_features)
        
        return ic_normal, ic_tail
    
    def compute_skew_exposure(self) -> np.ndarray:
        """
        Compute Skew Exposure to market crash risk.
        
        Formula: beta_to_market_skew = cov(feature, market^3) / var(market^3)
        """
        market_return = self.market_data['market_return']
        
        # Remove NaN values
        valid_mask = ~np.isnan(market_return)
        market_valid = market_return[valid_mask]
        
        if len(market_valid) < 20:
            return np.zeros(self.n_features)
        
        market_skew = market_valid ** 3
        
        skew_exposure = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            feature_valid = self.features[valid_mask, i]
            feature_clean_mask = ~np.isnan(feature_valid)
            feature_clean = feature_valid[feature_clean_mask]
            skew_clean = market_skew[feature_clean_mask]
            
            if len(feature_clean) < 10:
                continue
            
            # Center the data
            feature_centered = feature_clean - np.mean(feature_clean)
            skew_centered = skew_clean - np.mean(skew_clean)
            
            # Compute covariance and variance
            cov = np.sum(feature_centered * skew_centered) / len(feature_clean)
            var_skew = np.sum(skew_centered ** 2) / len(skew_clean)
            
            if var_skew > 0:
                skew_exposure[i] = cov / var_skew
        
        # Normalize to reasonable range
        skew_exposure = np.clip(skew_exposure, -10, 10)
        
        return skew_exposure
    
    def compute_volume_liquidity_correlation(self) -> np.ndarray:
        """
        Compute Volume-Liquidity Correlation.
        
        Formula: corr(feature, log(dollar_volume))
        """
        dollar_volume = self.market_data['dollar_volume']
        valid_mask = ~np.isnan(dollar_volume) & (dollar_volume > 0)
        
        if np.sum(valid_mask) < 20:
            return np.zeros(self.n_features)
        
        dollar_volume_valid = dollar_volume[valid_mask]
        dollar_volume_log = np.log(dollar_volume_valid + 1)  # Add 1 to avoid log(0)
        
        correlation = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            feature_valid = self.features[valid_mask, i]
            feature_clean_mask = ~np.isnan(feature_valid)
            feature_clean = feature_valid[feature_clean_mask]
            volume_clean = dollar_volume_log[feature_clean_mask]
            
            if len(feature_clean) < 10:
                continue
            
            # Simple correlation
            corr_matrix = np.corrcoef(feature_clean, volume_clean)
            if corr_matrix.shape == (2, 2):
                correlation[i] = corr_matrix[0, 1]
        
        return correlation
    
    def compute_flash_crash_performance(self) -> np.ndarray:
        """
        Compute Flash Crash Performance.
        
        IC during -5% or worse market moves.
        """
        market_return = self.market_data['market_return']
        crash_mask = market_return < -0.05  # Use -5% for more events
        
        if np.sum(crash_mask) > 3:  # Minimum crash events
            flash_crash_ic = self._fast_correlation_matrix(
                self.features[crash_mask], 
                self.returns['return_t+5'][crash_mask]
            )
        else:
            flash_crash_ic = np.zeros(self.n_features)
        
        return flash_crash_ic
    
    def compute_data_availability(self) -> np.ndarray:
        """
        Compute Data Availability.
        
        Formula: availability = 1 - (nan_count / total_periods)
        """
        nan_count = np.sum(np.isnan(self.features), axis=0)
        availability = 1 - (nan_count / self.n_samples)
        return availability
    
    def compute_signal_to_noise_ratio(self) -> np.ndarray:
        """
        Compute Signal-to-Noise Ratio.
        
        Signal = feature smoothed with Savitzky-Golay filter
        Noise = feature - signal
        SNR = std(signal) / std(noise)
        """
        window_size = min(21, self.n_samples // 10)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size < 5:
            return np.ones(self.n_features) * 0.5
        
        snr = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            feature_series = self.features[:, i].copy()
            valid_mask = ~np.isnan(feature_series)
            
            if np.sum(valid_mask) < window_size:
                snr[i] = 0.5
                continue
            
            feature_valid = feature_series[valid_mask]
            
            try:
                # Use Savitzky-Golay filter for smoothing
                signal = savgol_filter(feature_valid, window_size, 3)
                noise = feature_valid - signal
                
                std_signal = np.std(signal, ddof=1)
                std_noise = np.std(noise, ddof=1)
                
                if std_noise > 0:
                    snr[i] = std_signal / std_noise
                else:
                    snr[i] = 1.0
            except:
                # Fallback to simple moving average
                kernel = np.ones(5) / 5
                signal = np.convolve(feature_valid, kernel, mode='same')
                noise = feature_valid - signal
                
                std_signal = np.std(signal, ddof=1)
                std_noise = np.std(noise, ddof=1)
                
                if std_noise > 0:
                    snr[i] = std_signal / std_noise
                else:
                    snr[i] = 1.0
        
        # Clip to reasonable range
        snr = np.clip(snr, 0, 10)
        
        return snr
    
    def compute_delay_sensitivity(self) -> np.ndarray:
        """
        Compute Delay Sensitivity.
        
        Measures sensitivity to execution delay.
        """
        # Use different lags of features
        degradation = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            feature_series = self.features[:, i].copy()
            return_series = self.returns['return_t+1'].copy()
            
            valid_mask = ~np.isnan(feature_series) & ~np.isnan(return_series)
            feature_valid = feature_series[valid_mask]
            return_valid = return_series[valid_mask]
            
            if len(feature_valid) < 20:
                continue
            
            # Immediate correlation
            corr_immediate = np.corrcoef(feature_valid, return_valid)[0, 1]
            
            # Delayed correlation (shift feature by 1 period)
            if len(feature_valid) > 1:
                feature_delayed = np.roll(feature_valid, 1)
                feature_delayed[0] = np.nan
                
                delayed_mask = ~np.isnan(feature_delayed)
                corr_delayed = np.corrcoef(feature_delayed[delayed_mask], 
                                          return_valid[delayed_mask])[0, 1]
                
                if abs(corr_immediate) > 0.01:
                    degradation[i] = abs(corr_immediate - corr_delayed) / (abs(corr_immediate) + 1e-10)
        
        degradation = np.clip(degradation, 0, 1)
        return degradation
    
    def compute_predictive_orthogonality(self) -> np.ndarray:
        """
        Compute Predictive Orthogonality.
        
        Measures unique predictive power relative to other features.
        """
        # Compute correlation matrix of features
        corr_matrix = np.corrcoef(self.features, rowvar=False)
        np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlation
        
        # Average absolute correlation with other features
        avg_corr = np.nanmean(np.abs(corr_matrix), axis=1)
        orthogonality = 1 - avg_corr
        
        orthogonality = np.clip(orthogonality, 0, 1)
        return orthogonality
    
    def compute_cluster_purity(self) -> np.ndarray:
        """
        Compute Cluster Purity (post-clustering).
        
        Validates clustering effectiveness.
        """
        if self.cluster_labels is None or len(self.cluster_labels) != self.n_features:
            # Create simple clusters based on feature correlation
            corr_matrix = np.corrcoef(self.features, rowvar=False)
            # Simple clustering: group highly correlated features
            try:
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import pdist
                
                distance_matrix = 1 - np.abs(corr_matrix)
                np.fill_diagonal(distance_matrix, 0)
                
                # Perform hierarchical clustering
                Z = linkage(pdist(distance_matrix), method='average')
                cluster_labels = fcluster(Z, t=0.5, criterion='distance')
            except:
                # Fallback: assign all features to same cluster
                cluster_labels = np.ones(self.n_features, dtype=int)
        else:
            cluster_labels = self.cluster_labels
        
        # Compute purity
        purity = np.zeros(self.n_features)
        corr_matrix = np.corrcoef(self.features, rowvar=False)
        
        for i in range(self.n_features):
            same_cluster = (cluster_labels == cluster_labels[i])
            diff_cluster = (cluster_labels != cluster_labels[i])
            
            same_cluster[i] = False  # Exclude self
            
            if np.any(same_cluster) and np.any(diff_cluster):
                intra_corr = np.nanmean(np.abs(corr_matrix[i, same_cluster]))
                inter_corr = np.nanmean(np.abs(corr_matrix[i, diff_cluster]))
                purity[i] = intra_corr - inter_corr
        
        purity = np.clip(purity, -1, 1)
        return purity
    
    def compute_final_score(self, metrics: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute final weighted score for each feature.
        
        Formula: score = Σ(metric_i * weight_i * I(threshold_passed))
        """
        # Sub-metric weights from specification
        sub_weights = {
            'rank_ic_spearman': 0.15,
            'ic_decay_1': 0.02,
            'ic_decay_3': 0.02,
            'ic_decay_5': 0.02,
            'ic_decay_10': 0.02,
            'ic_decay_20': 0.02,
            'ic_t_statistic': 0.15,
            'cost_adjusted_ic': 0.15,
            'feature_turnover_rate': 0.10,
            'rolling_ic_stability_ratio': 0.10,
            'regime_ic_consistency': 0.10,
            'conditional_ic_normal': 0.025,
            'conditional_ic_tail': 0.025,
            'skew_exposure': 0.05,
            'volume_liquidity_correlation': 0.05,
            'data_availability': 0.03,
            'signal_to_noise_ratio': 0.02
        }
        
        # Initialize score array
        final_score = np.zeros(self.n_features)
        
        # Apply thresholds and weights (vectorized)
        for metric_name, weight in sub_weights.items():
            if metric_name in metrics:
                metric_values = metrics[metric_name]
                threshold = self.THRESHOLDS.get(metric_name, 0)
                
                # Check if metric passes threshold
                if metric_name in ['rank_ic_spearman', 'ic_t_statistic', 'cost_adjusted_ic',
                                 'rolling_ic_stability_ratio', 'regime_ic_consistency',
                                 'data_availability', 'signal_to_noise_ratio']:
                    # Higher is better
                    passed = metric_values > threshold
                elif metric_name in ['feature_turnover_rate', 'maximum_ic_drawdown',
                                   'time_to_recovery', 'regime_transition_sensitivity',
                                   'skew_exposure', 'delay_sensitivity']:
                    # Lower is better
                    passed = metric_values < threshold
                elif metric_name in ['volume_liquidity_correlation']:
                    # Near zero is better
                    passed = np.abs(metric_values) < abs(threshold)
                else:
                    passed = np.ones_like(metric_values, dtype=bool)
                
                # Add weighted contribution
                final_score += weight * passed.astype(float)
        
        # Normalize to percentage
        total_weight = sum(sub_weights.values())
        final_score = (final_score / total_weight) * 100
        
        return final_score
    
    def compute_all_metrics(self, methods: list[str] | None = None) -> pd.DataFrame:
        """
        Compute selected metrics (or all if methods=None) and return as a DataFrame.
        
        Parameters
        ----------
        methods : list[str] | None
            List of metric names to compute. If None, compute all metrics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with one row per feature and computed metrics.
        """
        print(f"Computing metrics for {self.n_features} features with {self.n_samples} samples...")

        # Helper to check if a metric should be computed
        def should_compute(metric_name: str) -> bool:
            return methods is None or metric_name in methods

        metrics = {}

        # --- 1. Predictive Power Metrics ---
        if should_compute('rank_ic_spearman'):
            metrics['rank_ic_spearman'] = self.compute_rank_ic_spearman()

        if should_compute('ic_decay_profile'):
            metrics.update(self.compute_ic_decay_profile())

        if should_compute('weighted_ic'):
            metrics['weighted_ic'] = self.compute_weighted_ic()
        if should_compute('ic_t_statistic'):
            metrics['ic_t_statistic'] = self.compute_ic_t_statistic()
        if should_compute('persistence_ratio'):
            metrics['persistence_ratio'] = self.compute_persistence_ratio()

        # --- 2. Economic Metrics ---
        if should_compute('feature_turnover_rate'):
            metrics['feature_turnover_rate'] = self.compute_feature_turnover_rate()
        if should_compute('cost_adjusted_ic') and 'rank_ic_spearman' in metrics and 'feature_turnover_rate' in metrics:
            metrics['cost_adjusted_ic'] = self.compute_cost_adjusted_ic(
                metrics['rank_ic_spearman'], metrics['feature_turnover_rate']
            )
        if should_compute('dollar_volume_sensitivity'):
            metrics['dollar_volume_sensitivity'] = self.compute_dollar_volume_sensitivity()
        if should_compute('bid_ask_spread_sensitivity'):
            metrics['bid_ask_spread_sensitivity'] = self.compute_bid_ask_spread_sensitivity()

        # --- 3. Stability Metrics ---
        if should_compute('rolling_ic_stability_ratio'):
            metrics['rolling_ic_stability_ratio'] = self.compute_rolling_ic_stability_ratio()
        if should_compute('maximum_ic_drawdown'):
            metrics['maximum_ic_drawdown'] = self.compute_maximum_ic_drawdown()
        if should_compute('time_to_recovery'):
            metrics['time_to_recovery'] = self.compute_time_to_recovery()
        if should_compute('regime_ic_consistency'):
            metrics['regime_ic_consistency'] = self.compute_regime_ic_consistency()
        if should_compute('regime_transition_sensitivity'):
            metrics['regime_transition_sensitivity'] = self.compute_regime_transition_sensitivity()

        # --- 4. Risk Metrics ---
        if should_compute('conditional_ic_normal') or should_compute('conditional_ic_tail'):
            ic_normal, ic_tail = self.compute_conditional_ic()
            if should_compute('conditional_ic_normal'):
                metrics['conditional_ic_normal'] = ic_normal
            if should_compute('conditional_ic_tail'):
                metrics['conditional_ic_tail'] = ic_tail
        if should_compute('skew_exposure'):
            metrics['skew_exposure'] = self.compute_skew_exposure()
        if should_compute('volume_liquidity_correlation'):
            metrics['volume_liquidity_correlation'] = self.compute_volume_liquidity_correlation()
        if should_compute('flash_crash_performance'):
            metrics['flash_crash_performance'] = self.compute_flash_crash_performance()

        # --- 5. Operational Metrics ---
        if should_compute('data_availability'):
            metrics['data_availability'] = self.compute_data_availability()
        if should_compute('signal_to_noise_ratio'):
            metrics['signal_to_noise_ratio'] = self.compute_signal_to_noise_ratio()
        if should_compute('delay_sensitivity'):
            metrics['delay_sensitivity'] = self.compute_delay_sensitivity()

        # --- 6. Cross-Feature Metrics ---
        if should_compute('predictive_orthogonality'):
            metrics['predictive_orthogonality'] = self.compute_predictive_orthogonality()
        if should_compute('cluster_purity'):
            metrics['cluster_purity'] = self.compute_cluster_purity()

        # --- Final Score ---
        if should_compute('final_score'):
            metrics['final_score'] = self.compute_final_score(metrics)

        # Build results DataFrame
        results_df = pd.DataFrame(metrics, index=self.feature_columns)
        results_df.index.name = 'feature_name'
        results_df = results_df.reset_index()

        print(f"Metrics computation completed. Generated {len(results_df)} feature evaluations.")
        return results_df


def calculate_feature_quality(df: pd.DataFrame, feature_columns: Optional[List[str]] = None, 
                             config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate feature importance using the FeatureMetricsEvaluator.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with all required columns
    feature_columns : List[str], optional
        List of feature column names to evaluate.
        If None, uses non-OHLCV technical features.
    config : Dict, optional
        Configuration dictionary
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature importance scores and metrics
    """
    obj = FeatureMetricsEvaluator(df, feature_columns=feature_columns, config=config)
    results = obj.compute_all_metrics()
    return results
