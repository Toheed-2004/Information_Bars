"""
Phase 1: Pre-Processing & Memory Preservation

Handles feature transformation to make them learner-ready while preserving memory.
"""

import pandas as pd
from typing import Dict
from bitpredict.common.ai.feature_engineering.transformation.calculator import apply_transformations
from bitpredict.common.ai.pipeline_utils import (
    check_stationarity, 
    get_domain_transform,
    get_feature_columns
)
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


def run_phase_1(df: pd.DataFrame, feature_metadata: Dict, 
                adf_significance: float = 0.05, 
                fracdiff_d: float = 0.4,
                scaling_method: str = "frozen_tanh_scaler") -> tuple[pd.DataFrame, Dict]:
    """
    Phase 1: Pre-process features for learner readiness.
    
    Steps:
    1. Check stationarity (ADF test)
    2. Apply fractional differencing if non-stationary
    3. Apply domain transformation
    4. Apply canonical scaling
    
    Args:
        df: DataFrame with features
        feature_metadata: Feature metadata from calculator
        adf_significance: P-value threshold for stationarity test
        fracdiff_d: Fractional differencing parameter
        scaling_method: Scaling method name
        
    Returns:
        Tuple of (transformed_df, phase_1_results)
    """
    logger.info("PHASE 1: PRE-PROCESSING")
    
    phase_1_config = {"features": {}}
    feature_cols = get_feature_columns(df, feature_metadata)
    
    for feature_name in feature_cols:
        logger.info(f"\nProcessing: {feature_name}")
        
        # Get metadata
        feature_meta = feature_metadata.get("features", {}).get(feature_name, {})
        domain = feature_meta.get("domain", "signed")
        
        # Check stationarity
        is_stationary, p_value = check_stationarity(df[feature_name], adf_significance)
        logger.info(f"  Stationary: {is_stationary} (p={p_value:.4f})")
        
        # Build transformation sequence
        transforms = {}
        seq = 1
        
        # Fractional differencing if non-stationary
        if not is_stationary:
            # logger.info(f"  Adding fracdiff (d={fracdiff_d})")
            transforms[str(seq)] = {
                "method": "adaptive_fractional_differencing",
                "params": {        
                    "max_d": 1.0,
                    "step": 0.05,
                    "significance": adf_significance
                }
            }
            seq += 1
        
        # Domain transformation
        domain_method = get_domain_transform(domain)
        if domain_method:
            logger.info(f"  Adding {domain_method}")
            transforms[str(seq)] = {"method": domain_method, "params": {}}
            seq += 1
        
        # Canonical scaling
        logger.info(f"  Adding {scaling_method}")
        transforms[str(seq)] = {"method": scaling_method, "params": {}}
        
        phase_1_config["features"][feature_name] = {
            "is_stationary": is_stationary,
            "p_value": p_value,
            "domain": domain,
            "transformations": transforms
        }
        print(f" phase 1 config for {feature_name}: {phase_1_config['features'][feature_name]}")
    # Apply transformations
    logger.info("\nApplying transformations...")
    df_transformed, applied_config = apply_transformations(df, phase_1_config)
    
    logger.info(f"\nPhase 1 complete! Processed {len(feature_cols)} features")
    logger.info("=" * 60)
    
    return df_transformed, applied_config
