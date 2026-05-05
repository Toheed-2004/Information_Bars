"""
Phase 2: Performance Signature Generation

Evaluates features across market regimes and classifies them as Universal or Inverse predictors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from bitpredict.common.ai.pipeline_utils import (
    create_forward_returns,
    calculate_spearman_safe,
    get_feature_columns
)
from bitpredict.common.market_regimes.config import REGIME_NAMES
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


def run_phase_2(df: pd.DataFrame, feature_metadata: Dict,
                horizons: List[int] = [1, 5, 20],
                min_ir: float = 0.3,
                min_mean_ric: float = 0.015,
                max_mean_ric: float = -0.015,
                min_regimes_with_data: int = 2,
                min_samples_per_regime: int = 50,
                use_regime_analysis: bool = False) -> Dict:
    """
    Phase 2: Generate performance signatures.
    
    Two modes:
    - Regime-based (use_regime_analysis=True): Analyze RIC across market regimes
    - Simple (use_regime_analysis=False): Calculate overall RIC on full dataset
    
    Steps:
    1. Create forward returns
    2. Calculate overall Rank IC (always)
    3. Calculate regime-specific Rank IC (optional)
    4. Analyze IC stability and decay
    5. Classify features as Universal or Inverse predictors
    
    Args:
        df: DataFrame with features and regime data
        feature_metadata: Feature metadata
        horizons: Forward return horizons
        min_ir: Minimum Information Ratio for classification
        min_mean_ric: Minimum mean RIC for universal predictors
        max_mean_ric: Maximum mean RIC for inverse predictors (negative)
        min_regimes_with_data: Minimum regimes with sufficient data
        min_samples_per_regime: Minimum samples per regime
        use_regime_analysis: If True, calculate regime-specific RIC; if False, use overall RIC only
        
    Returns:
        Phase 2 results dictionary
    """
    logger.info("PHASE 2: PERFORMANCE SIGNATURE")
    logger.info(f"Mode: {'Regime-based analysis' if use_regime_analysis else 'Simple overall RIC'}")
    
    # Verify regime column exists if using regime analysis
    if use_regime_analysis and 'primary_regime' not in df.columns:
        logger.warning("'primary_regime' column not found. Falling back to simple mode.")
        use_regime_analysis = False
    
    feature_cols = get_feature_columns(df, feature_metadata)
    phase_2_results = {}
    
    # Create forward returns for all horizons
    forward_returns = create_forward_returns(df, horizons)
    
    for feature_name in feature_cols:
        logger.info(f"\nEvaluating: {feature_name}")
        
        feature = df[feature_name]
        
        # 1. Calculate overall Rank IC (always calculated)
        overall_ric, n_samples = calculate_spearman_safe(feature, forward_returns[f'return_{horizons[1]}'])
        
        # 2. Calculate regime-specific Rank IC (optional)
        regime_ric = None
        regime_values = []
        if use_regime_analysis:
            regime_ric = _calculate_regime_ric(df, feature, forward_returns, min_samples_per_regime)
            regime_values = [
                r for r in regime_ric.values() 
                if not np.isnan(r['ric']) and r['n_samples'] >= min_samples_per_regime
            ]
        
        # 3. Calculate IC decay across horizons
        ic_decay = _calculate_ic_decay(feature, forward_returns, horizons)
        
        # 4. Calculate stability (Information Ratio)
        stability = _calculate_stability(feature, forward_returns[f'return_{horizons[1]}'])
        
        # 5. Determine mean RIC (use regime mean if available, otherwise overall)
        if use_regime_analysis and regime_values:
            regime_ric_values = [r['ric'] for r in regime_values]
            mean_ric = np.mean(regime_ric_values)
        else:
            mean_ric = overall_ric
        
        # 6. Get Information Ratio
        ir = stability['information_ratio']
        
        # 7. Classify feature
        is_universal, is_inverse, prediction_type, signal_multiplier = _classify_feature(
            regime_ric, mean_ric, ir, 
            min_ir, min_mean_ric, max_mean_ric, 
            min_regimes_with_data, min_samples_per_regime,
            use_regime_analysis
        )
        
        # Store results
        phase_2_results[feature_name] = {
            'overall_ric': overall_ric,
            'regime_ric': regime_ric,
            'ic_decay': ic_decay,
            'stability': stability,
            'mean_ric': mean_ric,
            'is_universal': is_universal,
            'is_inverse': is_inverse,
            'prediction_type': prediction_type,
            'signal_multiplier': signal_multiplier,
            'pass_phase_2': is_universal or is_inverse
        }
        
        # Logging
        logger.info(f"  Overall RIC: {overall_ric:.4f}")
        if use_regime_analysis:
            logger.info(f"  Mean Regime RIC: {mean_ric:.4f}")
            logger.info(f"  Regimes with data (>={min_samples_per_regime} samples): {len(regime_values)}")
            if regime_values:
                ric_values = ", ".join(f"{r['ric']:.3f}" for r in regime_values)
                logger.info(f"  Regime RICs: [{ric_values}]")
        logger.info(f"  IR: {ir:.2f}")
        logger.info(f"  Type: {prediction_type}")
        logger.info(f"  Pass: {phase_2_results[feature_name]['pass_phase_2']}")
    
    # Summary
    passed_features = [f for f, r in phase_2_results.items() if r['pass_phase_2']]
    universal_features = [f for f in passed_features if phase_2_results[f]['is_universal']]
    inverse_features = [f for f in passed_features if phase_2_results[f]['is_inverse']]
    
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total features evaluated: {len(feature_cols)}")
    logger.info(f"Features passed: {len(passed_features)}")
    logger.info(f"  - Universal: {len(universal_features)}")
    logger.info(f"  - Inverse: {len(inverse_features)}")
    
    if passed_features:
        logger.info("\nPassed features:")
        for f in passed_features:
            logger.info(f"  - {f}: {phase_2_results[f]['prediction_type']} "
                    f"(RIC={phase_2_results[f]['mean_ric']:.4f}, "
                    f"IR={phase_2_results[f]['stability']['information_ratio']:.2f})")
    
    logger.info("\nPhase 2 complete!")
    logger.info("=" * 60)
    
    return phase_2_results


def _calculate_regime_ric(df: pd.DataFrame, feature: pd.Series, 
                          returns: pd.DataFrame, min_samples: int = 30) -> Dict:
    """
    Calculate Rank IC for each regime.
    
    Args:
        df: DataFrame with regime column
        feature: Feature series
        returns: Forward returns DataFrame
        min_samples: Minimum samples required per regime
        
    Returns:
        Dictionary mapping regime names to {'ric': float, 'n_samples': int}
    """
    regime_ric = {}
    
    for regime_name in REGIME_NAMES:
        # Filter by regime
        mask = df['primary_regime'] == regime_name
        
        if mask.sum() < min_samples:
            regime_ric[regime_name] = {'ric': np.nan, 'n_samples': mask.sum()}
            continue
        
        # Calculate Spearman correlation
        feature_regime = feature[mask]
        returns_regime = returns.loc[mask, 'return_5']
        
        ric, n_samples = calculate_spearman_safe(feature_regime, returns_regime)
        regime_ric[regime_name] = {'ric': ric, 'n_samples': n_samples}
    
    return regime_ric


def _calculate_ic_decay(feature: pd.Series, returns: pd.DataFrame, 
                       horizons: List[int]) -> Dict:
    """Calculate IC decay across horizons."""
    ic_decay = {}
    for n in horizons:
        ric, _ = calculate_spearman_safe(feature, returns[f'return_{n}'])
        ic_decay[f'horizon_{n}'] = ric
    return ic_decay


def _calculate_stability(feature: pd.Series, returns: pd.Series, 
                        window: int = 60, use_spearman: bool = False) -> Dict:
    """
    Calculate Information Ratio (IC stability) using vectorized rolling correlation.
    
    OPTIMIZED: Uses pandas rolling correlation instead of loop.
    
    Args:
        feature: Feature series
        returns: Returns series
        window: Rolling window size
        use_spearman: If True, use Spearman (slower). If False, use Pearson (faster, default)
        
    Note:
        Pearson correlation is ~10x faster than Spearman for rolling calculations.
        For stability measurement, Pearson is usually sufficient.
    """
    # Remove NaN values
    valid_mask = ~(feature.isna() | returns.isna())
    
    if valid_mask.sum() < window + 10:
        return {'information_ratio': 0.0, 'mean_ic': 0.0, 'std_ic': 0.0}
    
    # Create aligned series
    feature_clean = feature[valid_mask].reset_index(drop=True)
    returns_clean = returns[valid_mask].reset_index(drop=True)
    
    # Check for constant values
    if feature_clean.std() < 1e-10 or returns_clean.std() < 1e-10:
        return {'information_ratio': 0.0, 'mean_ic': 0.0, 'std_ic': 0.0}
    
    if use_spearman:
        # Spearman: Convert to ranks then correlate (slower but rank-based)
        feature_rank = feature_clean.rolling(window=window).apply(
            lambda x: pd.Series(x).rank().iloc[-1], raw=False
        )
        returns_rank = returns_clean.rolling(window=window).apply(
            lambda x: pd.Series(x).rank().iloc[-1], raw=False
        )
        rolling_ic = feature_rank.rolling(window=window).corr(returns_rank)
    else:
        # Pearson: Direct correlation (much faster, ~10x)
        rolling_ic = feature_clean.rolling(window=window).corr(returns_clean)
    
    # Remove NaN values from rolling correlation
    rolling_ic_clean = rolling_ic.dropna()
    
    if len(rolling_ic_clean) < 2:
        return {'information_ratio': 0.0, 'mean_ic': 0.0, 'std_ic': 0.0}
    
    # Calculate statistics
    mean_ic = rolling_ic_clean.mean()
    std_ic = rolling_ic_clean.std()
    ir = mean_ic / (std_ic + 1e-10)
    
    return {'information_ratio': ir, 'mean_ic': mean_ic, 'std_ic': std_ic}


def _classify_feature(regime_ric: Dict, mean_ric: float, ir: float,
                     min_ir: float = 0.3,
                     min_mean_ric: float = 0.015,
                     max_mean_ric: float = -0.015,
                     min_regimes_with_data: int = 2,
                     min_samples_per_regime: int = 50,
                     use_regime_analysis: bool = True) -> tuple:
    """
    Classify feature as Universal or Inverse predictor.
    
    Two modes:
    - Regime-based: Requires consistent sign across regimes (70% agreement)
    - Simple: Uses overall RIC and IR only
    
    Args:
        regime_ric: Dictionary of regime RIC values (None if simple mode)
        mean_ric: Mean RIC (overall or regime average)
        ir: Information Ratio
        min_ir: Minimum Information Ratio threshold
        min_mean_ric: Minimum mean RIC for universal predictors
        max_mean_ric: Maximum mean RIC for inverse predictors (negative)
        min_regimes_with_data: Minimum regimes with sufficient data
        min_samples_per_regime: Minimum samples per regime
        use_regime_analysis: If True, check regime consistency; if False, use simple thresholds
    
    Returns:
        Tuple of (is_universal, is_inverse, prediction_type, signal_multiplier)
    """
    # Simple mode: Just check IR and mean RIC thresholds
    if not use_regime_analysis or regime_ric is None:
        is_universal = (abs(ir) > min_ir and mean_ric > min_mean_ric)
        is_inverse = (abs(ir) > min_ir and mean_ric < max_mean_ric)
        
        if is_universal:
            return True, False, 'universal', 1.0
        elif is_inverse:
            return False, True, 'inverse', -1.0
        else:
            return False, False, 'none', 0.0
    
    # Regime-based mode: Check regime consistency
    regimes_with_data = [
        r for r in regime_ric.values() 
        if not np.isnan(r['ric']) and r['n_samples'] >= min_samples_per_regime
    ]
    
    if len(regimes_with_data) < min_regimes_with_data:
        return False, False, 'none', 0.0
    
    # Count regimes by sign (ignore very weak correlations < 0.01)
    positive_regimes = sum(1 for r in regimes_with_data if r['ric'] > 0.01)
    negative_regimes = sum(1 for r in regimes_with_data if r['ric'] < -0.01)
    neutral_regimes = len(regimes_with_data) - positive_regimes - negative_regimes
    
    # Require at least 70% of regimes to agree on sign (excluding neutral)
    total_non_neutral = positive_regimes + negative_regimes
    if total_non_neutral == 0:
        predominant_sign_agreement = False
        max_agreement = 0.0
    else:
        max_agreement = max(positive_regimes, negative_regimes) / total_non_neutral
        predominant_sign_agreement = max_agreement >= 0.7
    
    # Debug logging
    logger.debug(f"    Pos: {positive_regimes}, Neg: {negative_regimes}, Neutral: {neutral_regimes}, Agreement: {max_agreement:.2f}")
    logger.debug(f"    Checks: |IR|>{min_ir}? {abs(ir)>min_ir}, mean_ric>{min_mean_ric}? {mean_ric>min_mean_ric}, mean_ric<{max_mean_ric}? {mean_ric<max_mean_ric}")
    
    # Universal Predictor (Positive)
    is_universal = (
        abs(ir) > min_ir and
        mean_ric > min_mean_ric and
        predominant_sign_agreement and
        positive_regimes > negative_regimes  # Majority positive
    )
    
    # Inverse Predictor (Negative)
    is_inverse = (
        abs(ir) > min_ir and
        mean_ric < max_mean_ric and
        predominant_sign_agreement and
        negative_regimes > positive_regimes  # Majority negative
    )
    
    # Determine prediction type and signal multiplier
    if is_universal:
        return True, False, 'universal', 1.0
    elif is_inverse:
        return False, True, 'inverse', -1.0
    else:
        return False, False, 'none', 0.0
