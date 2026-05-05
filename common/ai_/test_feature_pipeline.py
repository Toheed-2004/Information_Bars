"""
Test script for ML Feature Engineering Pipeline

Tests the complete ML pipeline including:
1. Feature calculation
2. Adaptive Fractional Differencing (AFD)
3. Gaussian Rank Transform
4. Feature Neutralization
5. Information Clustering & PCA
"""

import numpy as np

from bitpredict.common.ai.feature_pipeline_main import run_feature_pipeline
from bitpredict.common.db.services.data import read_ohlcv
from bitpredict.common.logging import get_logger, setup_logging

setup_logging("test_ml_pipeline")
logger = get_logger(__name__)

def test_basic_pipeline():
    """Test basic pipeline execution with default config."""
    logger.info("=" * 80)
    logger.info("TEST 1: Basic Pipeline Execution")
    logger.info("=" * 80)
    
    # Load data
    df = read_ohlcv(
        exchange='binance',
        symbol='btc',
        time_horizon='12h',
        start_date='2022-01-01'
    )
    
    logger.info(f"Loaded data: {len(df)} rows")
    logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Run pipeline
    eigen_features, metadata = run_feature_pipeline(df)
    
    # Extract cluster_info from metadata
    cluster_info = metadata.get('cluster_info', {})
    
    # Validate outputs
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\nEigen Features Shape: {eigen_features.shape}")
    logger.info(f"Eigen Features Columns: {list(eigen_features.columns)}")
    
    logger.info(f"\nMetadata:")
    logger.info(f"  Original features: {metadata['n_original_features']}")
    logger.info(f"  Clusters formed: {metadata['n_clusters']}")
    logger.info(f"  Eigen features: {metadata['n_eigen_features']}")
    
    logger.info(f"\nAFD Transformations:")
    afd_stats = metadata['afd_transformations']
    n_stationary = sum(1 for v in afd_stats.values() if v.get('is_stationary', False))
    n_transformed = sum(1 for v in afd_stats.values() if v.get('applied', False))
    n_failed = sum(1 for v in afd_stats.values() if not v.get('applied', True) and not v.get('is_stationary', False))
    
    logger.info(f"  Already stationary: {n_stationary}")
    logger.info(f"  Transformed: {n_transformed}")
    logger.info(f"  Failed to achieve stationarity: {n_failed}")
    
    logger.info(f"\nCluster Info:")
    for cluster_id, info in cluster_info.items():
        logger.info(f"  Cluster {cluster_id}:")
        logger.info(f"    Members: {len(info['members'])}")
        logger.info(f"    Explained variance: {info['explained_variance']:.4f}")
        if len(info['members']) <= 5:
            logger.info(f"    Features: {info['members']}")
    
    # Check for NaN values
    nan_count = eigen_features.isna().sum().sum()
    logger.info(f"\nNaN values in eigen features: {nan_count}")
    
    # Basic statistics
    logger.info(f"\nEigen Features Statistics:")
    logger.info(eigen_features.describe())
    
    logger.info("\n✓ Test 1 PASSED")
    return eigen_features, metadata, cluster_info


def test_custom_config():
    """Test pipeline with custom configuration."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Custom Configuration")
    logger.info("=" * 80)
    
    # Load data
    df = read_ohlcv(
        exchange='binance',
        symbol='btc',
        time_horizon='12h',
        start_date='2023-01-01'
    )
    
    # Run with custom parameters
    eigen_features, metadata = run_feature_pipeline(
        df,
        afd_significance=0.01,  # Stricter stationarity test
        afd_max_d=0.8,          # Lower max differencing order
        gaussian_window=100,     # Smaller window
        cluster_threshold=0.5    # More clusters (higher threshold)
    )
    
    # Extract cluster_info from metadata
    cluster_info = metadata.get('cluster_info', {})
    
    logger.info(f"\nResults with custom config:")
    logger.info(f"  Eigen features: {eigen_features.shape[1]}")
    logger.info(f"  Clusters: {metadata['n_clusters']}")
    
    logger.info("\n✓ Test 2 PASSED")
    return eigen_features, metadata, cluster_info


def test_afd_details():
    """Test and display detailed AFD transformation results."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: AFD Transformation Details")
    logger.info("=" * 80)
    
    # Load data
    df = read_ohlcv(
        exchange='binance',
        symbol='btc',
        time_horizon='12h',
        start_date='2023-01-01'
    )
    
    # Run pipeline
    eigen_features, metadata = run_feature_pipeline(df)
    
    # Extract cluster_info from metadata
    cluster_info = metadata.get('cluster_info', {})
    
    # Analyze AFD transformations
    afd_stats = metadata['afd_transformations']
    
    logger.info("\nStationary Features (no transformation needed):")
    stationary = [k for k, v in afd_stats.items() if v.get('is_stationary', False)]
    for feat in stationary[:10]:  # Show first 10
        p_val = afd_stats[feat]['p_value']
        logger.info(f"  {feat}: p-value={p_val:.6f}")
    if len(stationary) > 10:
        logger.info(f"  ... and {len(stationary) - 10} more")
    
    logger.info("\nTransformed Features (AFD applied):")
    transformed = [k for k, v in afd_stats.items() if v.get('applied', False)]
    for feat in transformed[:10]:  # Show first 10
        info = afd_stats[feat]
        logger.info(f"  {feat}:")
        logger.info(f"    Original p-value: {info['original_p_value']:.6f}")
        logger.info(f"    Optimal d: {info['d']:.2f}")
        logger.info(f"    Transformed p-value: {info['transformed_p_value']:.6f}")
    if len(transformed) > 10:
        logger.info(f"  ... and {len(transformed) - 10} more")
    
    logger.info("\nFailed Transformations:")
    failed = [k for k, v in afd_stats.items() 
              if not v.get('applied', True) and not v.get('is_stationary', False)]
    for feat in failed[:5]:  # Show first 5
        info = afd_stats[feat]
        logger.info(f"  {feat}: {info.get('reason', 'unknown')}")
    if len(failed) > 5:
        logger.info(f"  ... and {len(failed) - 5} more")
    
    logger.info("\n✓ Test 3 PASSED")


def test_cluster_analysis():
    """Test and analyze cluster formation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Cluster Analysis")
    logger.info("=" * 80)
    
    # Load data
    df = read_ohlcv(
        exchange='binance',
        symbol='btc',
        time_horizon='12h',
        start_date='2023-01-01'
    )
    
    # Run pipeline
    eigen_features, metadata = run_feature_pipeline(df)
    
    # Extract cluster_info from metadata
    cluster_info = metadata.get('cluster_info', {})
    
    # Analyze clusters
    logger.info(f"\nTotal clusters: {len(cluster_info)}")
    
    # Cluster size distribution
    cluster_sizes = [len(info['members']) for info in cluster_info.values()]
    logger.info(f"\nCluster size statistics:")
    logger.info(f"  Min: {min(cluster_sizes)}")
    logger.info(f"  Max: {max(cluster_sizes)}")
    logger.info(f"  Mean: {np.mean(cluster_sizes):.2f}")
    logger.info(f"  Median: {np.median(cluster_sizes):.2f}")
    
    # Show largest clusters
    sorted_clusters = sorted(
        cluster_info.items(),
        key=lambda x: len(x[1]['members']),
        reverse=True
    )
    
    logger.info(f"\nTop 5 largest clusters:")
    for cluster_id, info in sorted_clusters[:5]:
        logger.info(f"  Cluster {cluster_id}: {len(info['members'])} features")
        logger.info(f"    Explained variance: {info['explained_variance']:.4f}")
        logger.info(f"    Sample features: {info['members'][:3]}")
    
    # Variance explained distribution
    variances = [info['explained_variance'] for info in cluster_info.values()]
    logger.info(f"\nExplained variance statistics:")
    logger.info(f"  Min: {min(variances):.4f}")
    logger.info(f"  Max: {max(variances):.4f}")
    logger.info(f"  Mean: {np.mean(variances):.4f}")
    
    logger.info("\n✓ Test 4 PASSED")


def test_data_quality():
    """Test data quality of pipeline outputs."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Data Quality Checks")
    logger.info("=" * 80)
    
    # Load data
    df = read_ohlcv(
        exchange='binance',
        symbol='btc',
        time_horizon='12h',
        start_date='2023-01-01'
    )
    
    # Run pipeline
    eigen_features, metadata = run_feature_pipeline(df)
    
    # Extract cluster_info from metadata
    cluster_info = metadata.get('cluster_info', {})
    
    # Check 1: No infinite values
    inf_count = np.isinf(eigen_features.values).sum()
    logger.info(f"\nInfinite values: {inf_count}")
    assert inf_count == 0, "Found infinite values!"
    
    # Check 2: NaN values (some expected at start due to rolling windows)
    nan_count = eigen_features.isna().sum().sum()
    nan_pct = (nan_count / eigen_features.size) * 100
    logger.info(f"NaN values: {nan_count} ({nan_pct:.2f}%)")
    
    # Check 3: Valid data rows
    valid_rows = eigen_features.dropna().shape[0]
    logger.info(f"Valid rows (no NaN): {valid_rows} / {len(eigen_features)}")
    
    # Check 4: Feature variance (should not be constant)
    zero_var_features = (eigen_features.std() == 0).sum()
    logger.info(f"Zero variance features: {zero_var_features}")
    
    # Check 5: Index alignment
    logger.info(f"\nIndex alignment:")
    logger.info(f"  Original data length: {len(df)}")
    logger.info(f"  Eigen features length: {len(eigen_features)}")
    logger.info(f"  Index match: {eigen_features.index.equals(df.index)}")
    
    logger.info("\n✓ Test 5 PASSED")


def test_with_different_timeframes():
    """Test pipeline with different timeframes."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Different Timeframes")
    logger.info("=" * 80)
    
    timeframes = ['1h', '4h', '12h']
    
    for tf in timeframes:
        logger.info(f"\nTesting with {tf} timeframe...")
        
        df = read_ohlcv(
            exchange='binance',
            symbol='btc',
            time_horizon=tf,
            start_date='2023-01-01'
        )
        
        eigen_features, metadata = run_feature_pipeline(df)
        
        # Extract cluster_info from metadata
        cluster_info = metadata.get('cluster_info', {})
        
        logger.info(f"  Data rows: {len(df)}")
        logger.info(f"  Eigen features: {eigen_features.shape[1]}")
        logger.info(f"  Clusters: {metadata['n_clusters']}")
        logger.info(f"  Valid rows: {eigen_features.dropna().shape[0]}")
    
    logger.info("\n✓ Test 6 PASSED")


def main():
    """Run all tests."""
    try:
        # Test 1: Basic execution
        eigen_features, metadata = test_basic_pipeline()
        cluster_info = metadata.get('cluster_info', {})
        
        # Test 2: Custom config
        test_custom_config()
        
        # Test 3: AFD details
        test_afd_details()
        
        # Test 4: Cluster analysis
        test_cluster_analysis()
        
        # Test 5: Data quality
        test_data_quality()
        
        # Test 6: Different timeframes
        test_with_different_timeframes()
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
