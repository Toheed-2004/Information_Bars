"""
ML-Focused Feature Engineering Pipeline

This pipeline is optimized for machine learning:
1. Adaptive Fractional Differencing (AFD) for stationarity
2. Gaussian Rank Transform for normalization
3. Feature Neutralization (remove market beta)
4. Information Clustering & PCA (eigen features named after representative originals)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist

from bitpredict.common.logging import get_logger
from bitpredict.common.utils.file_system import read_yaml_config
from bitpredict.common.ai.feature_engineering.features import create_features
from bitpredict.common.constants import OHLCV_COLUMNS

logger = get_logger(__name__)


class MLFeaturePipeline:
    """
    ML-optimized feature engineering pipeline.
    
    Steps:
    1. Calculate raw features
    2. Adaptive Fractional Differencing (stationarity)
    3. Gaussian Rank Transform (normalization)
    4. Feature Neutralization (remove beta)
    5. Information Clustering & PCA (orthogonalization with original names)
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[Dict] = None, config_path: Optional[str] = None, **kwargs):
        """
        Initialize ML pipeline.
        
        Args:
            df: Input DataFrame with features
            config: Configuration dictionary (takes precedence over config_path)
            config_path: Path to YAML config file (used if config is None)
            **kwargs: Override config parameters
                - skip_clustering: If True, skip clustering and eigen feature extraction (default: False)
        """
        # Load configuration
        if config is not None:
            # Use provided config dict
            self.config = config
        elif config_path is not None:
            # Load from file
            self.config = read_yaml_config(str(config_path))
        else:

            config_path = Path(__file__).parent / "feature_pipeline_config.yaml"
            self.config = read_yaml_config(str(config_path))
        
        # Extract parameters
        self.afd_significance = kwargs.get('afd_significance', 
            self.config.get('afd', {}).get('significance', 0.05))
        self.afd_max_d = kwargs.get('afd_max_d',
            self.config.get('afd', {}).get('max_d', 1.0))
        
        self.gaussian_window = kwargs.get('gaussian_window',
            self.config.get('gaussian_transform', {}).get('window', 252))
        
        self.neutralize_against = kwargs.get('neutralize_against',
            self.config.get('neutralization', {}).get('against', ['close']))
        
        self.cluster_threshold = kwargs.get('cluster_threshold',
            self.config.get('clustering', {}).get('distance_threshold', 0.3))
        self.cluster_method = kwargs.get('cluster_method',
            self.config.get('clustering', {}).get('method', 'correlation'))
        
        # Skip clustering for prediction (inference mode)
        self.skip_clustering = kwargs.get('skip_clustering', False)

        self.df = df
        self.feature_metadata = {
            'afd_transformations': {},
            'gaussian_transform': {},
            'neutralization': {},
            'clustering': {}
        }
        self.eigen_features = None
        self.cluster_info = {}
    
    def run_pipeline(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Run complete ML feature pipeline.
        
        Executes transformation steps in sequence:
        1. Filter to OHLCV + datetime only
        2. Calculate features from OHLCV data
        3. Extract feature columns (exclude OHLCV, datetime, metadata)
        4. Adaptive Fractional Differencing for stationarity
        5. Gaussian Rank Transform for normalization
        6. Feature Neutralization to remove market beta
        7. (Optional) Information Clustering and PCA for orthogonalization
        
        If skip_clustering=True (inference mode), steps 1-6 are executed and
        raw features are returned. Clustering is skipped for efficiency.
        
        Returns:
            Tuple of (features_df, metadata):
                - features_df: DataFrame with features (eigen features if clustering enabled, raw features if skipped)
                - metadata: Dict with pipeline statistics and configuration
                
        Raises:
            ValueError: If feature calculation fails or produces no valid features
        """
        # Get feature selection from config
        features = self.config.get('features', 'all')
        logger.info("=" * 80)
        logger.info("ML FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Configuration: features={features}, afd_max_d={self.afd_max_d}, "
                   f"cluster_threshold={self.cluster_threshold}, skip_clustering={self.skip_clustering}")
        
        # Step 1: Filter to OHLCV + datetime only
        logger.info("\nStep 1: Preparing OHLCV data...")
        
        # Build list of columns to keep
        cols_to_keep = [col for col in OHLCV_COLUMNS if col in self.df.columns]
        
        logger.info(f"  OHLCV columns available: {cols_to_keep}")
        
        # Filter self.df to OHLCV + datetime only
        self.df = self.df[cols_to_keep].copy()
        logger.debug(f"  Filtered to OHLCV columns: {list(self.df.columns)}")
        
        # Step 2: Calculate raw features from OHLCV data
        logger.info("\nStep 2: Calculating features...")
        logger.info(f"  Requested features: {features}")
        try:

            self.df, feature_calc_metadata = create_features(self.df, features, drop_nan=True)
            # Merge feature calculation metadata into our metadata structure
            self.feature_metadata['feature_calculation'] = feature_calc_metadata
        except Exception as e:
            logger.error(f"Feature calculation failed: {e}", exc_info=True)
            raise
        
        logger.info(f"  Generated {len(self.df.columns)} columns from create_features")
        
        # Step 3: Extract feature columns (exclude OHLCV, datetime, and metadata columns)
        logger.info("\nStep 3: Extracting feature columns...")
        
        # Columns to exclude
        exclude_cols = {'datetime', 'exchange', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume'}
        
        # Extract features (exclude known non-feature columns and patterns)
        feature_cols = [col for col in self.df.columns 
                       if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("No valid features calculated")
        
        logger.info(f"  Extracted {len(feature_cols)} features: {feature_cols}")
        

        logger.info(f"  DataFrame for processing: {list(self.df.columns)}")
        
        # Step 4: Adaptive Fractional Differencing for stationarity
        logger.info("\nStep 4: Adaptive Fractional Differencing...")
        logger.info(f"  ADF significance level: {self.afd_significance}")
        logger.info(f"  Max differencing order: {self.afd_max_d}")
        self.df = self._apply_afd(self.df, feature_cols)
        
        # Step 5: Gaussian Rank Transform for normalization
        logger.info("\nStep 5: Gaussian Rank Transform...")
        logger.info(f"  Rolling window: {self.gaussian_window}")
        self.df = self._apply_gaussian_transform(self.df, feature_cols)
        
        # Step 6: Feature Neutralization to remove market beta
        logger.info("\nStep 6: Feature Neutralization (remove beta)...")
        logger.info(f"  Neutralizing against: {self.neutralize_against}")
        self.df = self._neutralize_features(self.df, feature_cols)
        
        # Step 7: Information Clustering & PCA (optional, skip for inference)
        if self.skip_clustering:
            logger.info("\nStep 7: Skipping clustering (inference mode)")
            # Include datetime column along with features for inference
            cols_to_return = feature_cols.copy()
            if 'datetime' in self.df.columns:
                cols_to_return.insert(0, 'datetime')
            self.eigen_features = self.df[cols_to_return].copy()
            logger.info(f"  Returning {len(feature_cols)} raw features (no clustering)")
            if 'datetime' in cols_to_return:
                logger.info(f"  Including datetime column for inference")
        else:
            logger.info("\nStep 7: Information Clustering & PCA...")
            logger.info(f"  Clustering method: {self.cluster_method}")
            logger.info(f"  Distance threshold: {self.cluster_threshold}")
            self.eigen_features, self.cluster_info = self._create_eigen_features(
                self.df, feature_cols
            )
        
        # Build comprehensive metadata package (no redundancy)
        metadata = {
            'n_original_features': len(feature_cols),
            'n_clusters': len(self.cluster_info) if not self.skip_clustering else 0,
            'n_eigen_features': self.eigen_features.shape[1],
            'skip_clustering': self.skip_clustering,
            'cluster_info': self.cluster_info if not self.skip_clustering else {},
            'feature_calculation': self.feature_metadata.get('feature_calculation', {}),
            'afd': self.feature_metadata.get('afd_transformations', {}),
            'gaussian_transform': self.feature_metadata.get('gaussian_transform', {}),
            'neutralization': self.feature_metadata.get('neutralization', {}),
            'config': {
                'features': feature_cols,
                'afd': {
                    'enabled': self.config.get('afd', {}).get('enabled', True),
                    'significance': self.afd_significance,
                    'max_d': self.afd_max_d,
                    'step': self.config.get('afd', {}).get('step', 0.05)
                },
                'gaussian_transform': {
                    'enabled': self.config.get('gaussian_transform', {}).get('enabled', True),
                    'window': self.gaussian_window,
                    'n_quantiles': self.config.get('gaussian_transform', {}).get('n_quantiles', 200)
                },
                'neutralization': {
                    'enabled': self.config.get('neutralization', {}).get('enabled', True),
                    'against': self.neutralize_against
                },
                'clustering': {
                    'enabled': self.config.get('clustering', {}).get('enabled', True),
                    'method': self.cluster_method,
                    'distance_threshold': self.cluster_threshold,
                    'linkage': self.config.get('clustering', {}).get('linkage', 'average'),
                    'pca_components': self.config.get('clustering', {}).get('pca_components', 1),
                    'skipped': self.skip_clustering
                }
            }
        }
        # Calculate and log summary statistics
        afd_stats = self.feature_metadata.get('afd_transformations', {})
        n_stationary = sum(1 for v in afd_stats.values() if v.get('is_stationary', False))
        n_transformed = sum(1 for v in afd_stats.values() if v.get('applied', False))
        n_failed = len(afd_stats) - n_stationary - n_transformed
        
        gaussian_stats = self.feature_metadata.get('gaussian_transform', {})
        n_gaussian_applied = sum(1 for v in gaussian_stats.values() if v.get('applied', False))
        
        neutral_stats = self.feature_metadata.get('neutralization', {})
        n_neutralized = sum(1 for v in neutral_stats.values() if v.get('applied', False))
        
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Original features: {metadata['n_original_features']}")
        logger.info(f"AFD - Already stationary: {n_stationary}")
        logger.info(f"AFD - Transformed: {n_transformed}")
        logger.info(f"AFD - Failed: {n_failed}")
        logger.info(f"Gaussian Transform - Applied: {n_gaussian_applied}")
        logger.info(f"Neutralization - Applied: {n_neutralized}")
        if not self.skip_clustering:
            logger.info(f"Clusters formed: {metadata['n_clusters']}")
            logger.info(f"Eigen features: {metadata['n_eigen_features']}")
        else:
            logger.info(f"Clustering skipped (inference mode)")
            logger.info(f"Raw features returned: {metadata['n_eigen_features']}")
        
        # Remove OHLCV columns from eigen_features (keep only features)
        ohlcv_columns = ["open", "high", "low", "close", "volume"]
        ohlcv_to_remove = [col for col in ohlcv_columns if col in self.eigen_features.columns]
        if ohlcv_to_remove:
            self.eigen_features = self.eigen_features.drop(columns=ohlcv_to_remove)
            logger.info(f"Removed OHLCV columns: {ohlcv_to_remove}")

        # Return only eigen_features and metadata (cluster_info is inside metadata)
        return self.eigen_features, metadata
    
    def _apply_afd(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Apply Adaptive Fractional Differencing to achieve stationarity.
        
        This method implements a two-step process:
        1. Test each feature for stationarity using Augmented Dickey-Fuller (ADF) test
        2. If non-stationary, find the minimum differencing order d that achieves stationarity
        
        The fractional differencing preserves more memory than integer differencing,
        which is crucial for machine learning models that need historical patterns.
        
        Args:
            df: DataFrame containing features
            feature_cols: List of feature column names to process
            
        Returns:
            DataFrame with transformed features (in-place modification)
            
        Side Effects:
            Updates self.feature_metadata['afd_transformations'] with transformation details
            
        References:
            "Advances in Financial Machine Learning" by Marcos López de Prado, Chapter 5
        """
        from statsmodels.tsa.stattools import adfuller
        
        logger.info(f"  Processing {len(feature_cols)} features for stationarity...")
        n_processed = 0
        
        for col in feature_cols:
            # Work with the full series (including NaNs)
            series = df[col].copy()
            series_clean = series.dropna()
            
            # Skip features with insufficient data
            if len(series_clean) < 50:
                logger.debug(f"  {col}: Skipped (insufficient data: {len(series_clean)} samples)")
                continue
            
            # Step 1: Test for stationarity using ADF test
            try:
                adf_result = adfuller(series_clean, maxlag=20, regression='c')
                p_value = adf_result[1]
                is_stationary = p_value <= self.afd_significance
                
                if is_stationary:
                    # Feature is already stationary, no transformation needed
                    logger.debug(f"  {col}: Already stationary (p={p_value:.4f})")
                    self.feature_metadata['afd_transformations'][col] = {
                        'is_stationary': True,
                        'p_value': p_value,
                        'd': 0.0,
                        'applied': False
                    }
                    n_processed += 1
                    continue
                
                logger.debug(f"  {col}: Non-stationary (p={p_value:.4f}), finding optimal d...")
                
            except Exception as e:
                logger.warning(f"  {col}: ADF test failed - {str(e)}")
                continue
            
            # Step 2: Find minimum d that achieves stationarity
            optimal_d = None
            optimal_p_value = None
            last_p_value = None
            
            # Iterate through differencing orders from 0.1 to max_d
            d = 0.1
            while d <= self.afd_max_d:
                try:
                    # Apply fractional differencing to clean series
                    diff_series_clean = self._fractional_diff(series_clean, d)
                    
                    # Test transformed series for stationarity
                    adf_result = adfuller(diff_series_clean.dropna(), maxlag=20, regression='c')
                    p_value_diff = adf_result[1]
                    last_p_value = p_value_diff
                    
                    # Check if we achieved stationarity
                    if p_value_diff <= self.afd_significance:
                        optimal_d = d
                        optimal_p_value = p_value_diff
                        logger.debug(f"  {col}: Found optimal d={d:.2f} (p={p_value_diff:.4f})")
                        break
                        
                except Exception as e:
                    logger.debug(f"  {col}: d={d:.2f} failed - {str(e)}")
                
                d += 0.1
            
            # Step 3: Apply transformation
            if optimal_d is not None:
                # Achieved stationarity before max_d
                # Apply to clean series and preserve NaN positions
                diff_series_clean = self._fractional_diff(series_clean, optimal_d)
                # Reconstruct full series with NaNs in original positions
                result = series.copy()
                result[series_clean.index] = diff_series_clean
                df[col] = result
                
                self.feature_metadata['afd_transformations'][col] = {
                    'is_stationary': False,
                    'original_p_value': p_value,
                    'd': optimal_d,
                    'transformed_p_value': optimal_p_value,
                    'applied': True
                }
                
                logger.debug(f"  {col}: Applied d={optimal_d:.2f} (p: {p_value:.4f} → {optimal_p_value:.4f})")
                n_processed += 1
            else:
                # Could not achieve stationarity within max_d limit
                # Apply AFD at max_d anyway to reduce non-stationarity
                logger.warning(f"  {col}: Could not achieve stationarity, applying d={self.afd_max_d} anyway")
                try:
                    diff_series_clean = self._fractional_diff(series_clean, self.afd_max_d)
                    # Reconstruct full series with NaNs in original positions
                    result = series.copy()
                    result[series_clean.index] = diff_series_clean
                    df[col] = result
                    
                    self.feature_metadata['afd_transformations'][col] = {
                        'is_stationary': False,
                        'original_p_value': p_value,
                        'd': self.afd_max_d,
                        'transformed_p_value': last_p_value,
                        'applied': True,
                        'reason': 'max_d_applied_without_stationarity'
                    }
                    
                    logger.debug(f"  {col}: Applied d={self.afd_max_d} (p: {p_value:.4f} → {last_p_value:.4f})")
                    n_processed += 1
                except Exception as e:
                    logger.warning(f"  {col}: Failed to apply d={self.afd_max_d} - {str(e)}")
                    self.feature_metadata['afd_transformations'][col] = {
                        'is_stationary': False,
                        'original_p_value': p_value,
                        'd': None,
                        'applied': False,
                        'reason': 'max_d_application_failed',
                        'error': str(e)
                    }
        
        logger.info(f"  Processed {n_processed}/{len(feature_cols)} features")
        
        # TEMPORARY DEBUG: Log AFD summary
        stationary_count = sum(1 for v in self.feature_metadata['afd_transformations'].values() 
                              if v.get('is_stationary', False))
        transformed_count = sum(1 for v in self.feature_metadata['afd_transformations'].values() 
                               if v.get('applied', False))
        failed_count = len(feature_cols) - stationary_count - transformed_count
        
        logger.info(f"  AFD SUMMARY:")
        logger.info(f"    Already stationary: {stationary_count}")
        logger.info(f"    Transformed to stationary: {transformed_count}")
        logger.info(f"    Failed to achieve stationarity: {failed_count}")
        
        if transformed_count > 0:
            logger.info(f"  Transformed features:")
            for col, info in self.feature_metadata['afd_transformations'].items():
                if info.get('applied', False):
                    logger.info(f"    - {col}: d={info['d']:.2f}")
        
        return df
    
    def _fractional_diff(self, series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
        """
        Apply fractional differentiation to a time series.
        
        Fractional differentiation generalizes integer differencing to non-integer orders,
        allowing us to achieve stationarity while preserving more memory than standard
        differencing. This is critical for ML models that need historical patterns.
        
        The method uses binomial expansion to calculate weights and applies them via
        numpy convolution for maximum performance.
        
        Args:
            series: Time series to difference
            d: Differencing order (0 < d <= 1)
                - d=0: No differencing (original series)
                - d=1: Standard first-order differencing
                - 0<d<1: Fractional differencing (preserves more memory)
            threshold: Weight truncation threshold for computational efficiency
            
        Returns:
            Fractionally differenced series with same index as input
            
        Note:
            Initial values with insufficient history are forward-filled from first valid value.
            
        References:
            "Advances in Financial Machine Learning" by Marcos López de Prado, Chapter 5
        """
        # Calculate binomial weights using recursive formula
        # w_k = -w_{k-1} * (d - k + 1) / k
        weights = [1.0]
        k = 1
        
        # Continue until weights become negligible
        while abs(weights[-1]) > threshold:
            weight = -weights[-1] * (d - k + 1) / k
            weights.append(weight)
            k += 1
        
        # Convert to numpy array (no need to reverse for convolution)
        weights = np.array(weights)
        
        # Use numpy convolution for vectorized operation (10-50x faster)
        # mode='full' to get all values, then trim to original length
        series_values = series.values
        result_values = np.convolve(series_values, weights, mode='full')[:len(series_values)]
        
        # Create result series with same index
        result = pd.Series(result_values, index=series.index, dtype=float)
        
        # Forward-fill initial NaN values (insufficient history)
        # Only fill the first len(weights) positions with the first valid value
        n_weights = len(weights)
        if n_weights > 0 and len(result) > n_weights:
            first_valid = result.iloc[n_weights]
            result.iloc[:n_weights] = first_valid
        
        return result
    
    def _apply_gaussian_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Apply Gaussian Rank Transform on rolling window basis.
        
        This transformation converts feature distributions to standard normal (Gaussian)
        using quantile transformation. Applied on a rolling window to adapt to changing
        market conditions over time.
        
        Benefits:
        - Removes outliers' influence
        - Normalizes features to comparable scales
        - Makes features more suitable for neural networks
        - Adapts to regime changes via rolling window
        
        Args:
            df: DataFrame containing features
            feature_cols: List of feature column names to transform
            
        Returns:
            DataFrame with Gaussian-transformed features (in-place modification)
            
        Note:
            First `gaussian_window` rows will be NaN due to insufficient history.
            n_quantiles is automatically adjusted per window to avoid warnings.
        """
        logger.info(f"  Transforming {len(feature_cols)} features...")
        n_transformed = 0
        n_failed = 0
        
        window_size = self.gaussian_window
        
        for col in feature_cols:
            try:
                series = df[col].to_numpy()
                length = len(series)

                # Apply rolling window transformation
                transformed = np.full(length, np.nan)

                for i in range(window_size, length):
                    window = series[i-window_size:i]

                    # Skip windows with NaN values
                    if np.isnan(window).any():
                        continue

                    # Adaptive n_quantiles: use min of actual window size and 1000
                    n_quantiles = min(len(window), 1000)
                    
                    # Create and fit transformer for this window
                    transformer = QuantileTransformer(
                        output_distribution='normal',
                        n_quantiles=n_quantiles
                    )
                    transformer.fit(window.reshape(-1, 1))
                    transformed[i] = transformer.transform([[series[i]]])[0, 0]

                df[col] = transformed
                n_transformed += 1
                
                # Store metadata
                self.feature_metadata['gaussian_transform'][col] = {
                    'applied': True,
                    'window_size': self.gaussian_window,
                    'n_valid_values': int((~np.isnan(transformed)).sum())
                }
                
            except Exception as e:
                logger.warning(f"  {col}: Gaussian transform failed - {str(e)}")
                n_failed += 1
                self.feature_metadata['gaussian_transform'][col] = {
                    'applied': False,
                    'error': str(e)
                }
                continue
        
        logger.info(f"  Transformed {n_transformed}/{len(feature_cols)} features (failed: {n_failed})")
        return df
    
    def _neutralize_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Neutralize features against market returns to remove systematic beta.
        
        This process removes the component of each feature that can be explained by
        market movements, forcing the model to find alpha (excess returns) independent
        of market direction. This is crucial for market-neutral strategies.
        
        Process:
        1. Calculate market returns from specified columns (typically 'close')
        2. For each feature, regress it against market returns
        3. Replace feature with regression residuals (beta-neutral component)
        
        Args:
            df: DataFrame containing features and market data
            feature_cols: List of feature column names to neutralize
            
        Returns:
            DataFrame with neutralized features (in-place modification)
            
        Mathematical Formula:
            feature_neutralized = feature - beta * market_returns
            where beta is estimated via OLS regression
            
        Note:
            Features with insufficient valid data (<50 samples) are skipped.
        """
        # Calculate market returns from specified columns
        market_returns = df[self.neutralize_against].pct_change()
        
        logger.info(f"  Neutralizing {len(feature_cols)} features...")
        n_neutralized = 0
        n_skipped = 0
        
        for col in feature_cols:
            try:
                feature = df[col].values
                
                # Create valid mask (no NaN in feature or market returns)
                valid_mask = ~(np.isnan(feature) | np.isnan(market_returns.values).any(axis=1))
                
                # Skip if insufficient valid data
                if valid_mask.sum() < 50:
                    logger.debug(f"  {col}: Skipped (insufficient valid data)")
                    n_skipped += 1
                    self.feature_metadata['neutralization'][col] = {
                        'applied': False,
                        'reason': 'insufficient_data',
                        'n_valid_samples': int(valid_mask.sum())
                    }
                    continue
                
                # Extract valid samples
                X = market_returns.values[valid_mask]
                y = feature[valid_mask]
                
                # Regression: feature = beta * market_returns + residual
                # We keep only the residual (market-neutral component)
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
                
                # Replace feature with neutralized version
                df.loc[valid_mask, col] = residuals
                n_neutralized += 1
                
                # Store metadata
                self.feature_metadata['neutralization'][col] = {
                    'applied': True,
                    'beta': float(beta[0]) if len(beta) == 1 else beta.tolist(),
                    'n_valid_samples': int(valid_mask.sum())
                }
                
            except Exception as e:
                logger.warning(f"  {col}: Neutralization failed - {str(e)}")
                self.feature_metadata['neutralization'][col] = {
                    'applied': False,
                    'error': str(e)
                }
                continue
        
        logger.info(f"  Neutralized {n_neutralized}/{len(feature_cols)} features (skipped: {n_skipped})")
        return df
    
    def _create_eigen_features(self, df: pd.DataFrame, 
                              feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Create orthogonalized eigen-features through clustering and PCA.
        
        This process reduces feature redundancy and creates uncorrelated features:
        1. Calculate feature similarity (correlation or euclidean distance)
        2. Hierarchical clustering to group similar features
        3. Apply PCA within each cluster to extract principal component (PC1)
        4. Name each eigen-feature after the most representative original feature
        
        Benefits:
        - Reduces multicollinearity
        - Decreases dimensionality while preserving information
        - Maintains interpretability (uses original feature names)
        - Improves model stability and generalization
        
        Args:
            df: DataFrame containing features
            feature_cols: List of feature column names to cluster
            
        Returns:
            Tuple of (eigen_features_df, cluster_info):
                - eigen_features_df: DataFrame with PC1 per cluster (named after representative feature)
                - cluster_info: Dict mapping cluster_id to member features, selected name, and variance
                
        Note:
            Single-feature clusters use the feature directly without PCA.
            The representative feature name is selected based on highest average correlation
            with other cluster members (for correlation method) or lowest average distance
            (for euclidean method).
        """
        # Prepare feature matrix (remove columns with all NaN)
        feature_matrix = df[feature_cols].dropna(how='all', axis=1)
        feature_matrix = feature_matrix.ffill().fillna(0)
        
        logger.info(f"  Feature matrix shape: {feature_matrix.shape}")
        
        # Calculate distance matrix based on chosen method
        if self.cluster_method == 'correlation':
            # Correlation-based distance: d = 1 - |corr|
            corr_matrix = feature_matrix.corr().abs()
            corr_matrix = corr_matrix.clip(lower=-1.0, upper=1.0)
            distance_matrix = 1 - corr_matrix
            logger.debug("  Using correlation-based distance")
        else:
            # Euclidean distance between feature vectors
            distances = pdist(feature_matrix.T.values, metric='euclidean')
            distance_matrix = pd.DataFrame(
                squareform(distances),
                index=feature_matrix.columns,
                columns=feature_matrix.columns
            )
            logger.debug("  Using euclidean distance")
        
        # Perform hierarchical clustering
        distance_matrix = distance_matrix.clip(lower=0.0)
        np.fill_diagonal(distance_matrix.values, 0.0)

        # Symmetrize — floating point can make distance[i,j] slightly != distance[j,i]
        distance_array = (distance_matrix.values + distance_matrix.values.T) / 2
        np.fill_diagonal(distance_array, 0.0)

        # Perform hierarchical clustering
        linkage_matrix = linkage(squareform(distance_array), method='average')
        cluster_labels = fcluster(linkage_matrix, t=self.cluster_threshold, criterion='distance')
        
        # Group features by cluster
        clusters = {}
        for feature, label in zip(feature_matrix.columns, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(feature)
        
        logger.info(f"  Formed {len(clusters)} clusters")
        
        # Apply PCA per cluster and name after representative feature
        eigen_features = pd.DataFrame(index=df.index)
        cluster_info = {}
        
        for cluster_id, members in clusters.items():
            logger.debug(f"  Cluster {cluster_id}: {len(members)} features")
            
            if len(members) == 1:
                # Single feature - use directly without PCA
                feature_name = members[0]
                eigen_features[feature_name] = feature_matrix[feature_name]
                cluster_info[cluster_id] = {
                    'members': members,
                    'representative_feature': feature_name,
                    'explained_variance': 1.0,
                    'n_components': 1,
                    'pca_applied': False
                }
                logger.debug(f"    Using: {feature_name} (single member)")
            else:
                # Multiple features - apply PCA and name after most representative feature
                try:
                    # Find most representative feature (minimum average distance to others)
                    cluster_distance = distance_matrix.loc[members, members]
                    avg_distances = cluster_distance.mean(axis=1)
                    representative_feature = avg_distances.idxmin()
                    
                    # Apply PCA to get PC1
                    cluster_data = feature_matrix[members].values
                    pca = PCA(n_components=1)
                    pc1 = pca.fit_transform(cluster_data)
                    
                    # Use representative feature name for the eigen feature
                    eigen_features[representative_feature] = pc1.flatten()
                    cluster_info[cluster_id] = {
                        'members': members,
                        'representative_feature': representative_feature,
                        'explained_variance': float(pca.explained_variance_ratio_[0]),
                        'n_components': 1,
                        'pca_applied': True,
                        'avg_distance': float(avg_distances[representative_feature])
                    }
                    
                    logger.debug(f"    PC1 named: {representative_feature} "
                               f"(var={pca.explained_variance_ratio_[0]:.4f}, "
                               f"dist={avg_distances[representative_feature]:.4f})")
                    
                except Exception as e:
                    logger.warning(f"  Cluster {cluster_id}: PCA failed - {str(e)}")
                    # Fallback: use first feature in cluster
                    feature_name = members[0]
                    eigen_features[feature_name] = feature_matrix[feature_name]
                    cluster_info[cluster_id] = {
                        'members': members,
                        'representative_feature': feature_name,
                        'explained_variance': 1.0,
                        'n_components': 1,
                        'pca_applied': False,
                        'note': 'PCA failed, using first feature'
                    }
        
        logger.info(f"  Created {len(eigen_features.columns)} eigen features with original names")
        return eigen_features, cluster_info


def run_feature_pipeline(df: pd.DataFrame, config: Optional[Dict] = None, config_path: Optional[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to run ML feature pipeline.
    
    Args:
        df: Input DataFrame with OHLCV data
        config: Configuration dictionary (takes precedence over config_path)
        config_path: Path to config file (used if config is None)
        **kwargs: Override config parameters
        
    Returns:
        Tuple of (eigen_features_df, metadata)
            - eigen_features_df: DataFrame with eigen features
            - metadata: Dict containing all pipeline metadata including cluster_info
    """
    pipeline = MLFeaturePipeline(df=df, config=config, config_path=config_path, **kwargs)
    return pipeline.run_pipeline()
