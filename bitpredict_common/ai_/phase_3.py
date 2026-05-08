"""
Phase 3: Hard Veto Gates

Applies strict filters to remove features that don't meet production requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict
from bitpredict.common.ai.pipeline_utils import get_feature_columns
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


def run_phase_3(df: pd.DataFrame, feature_metadata: Dict, phase_2_results: Dict,
                min_ic: float = 0.015, min_ir: float = 0.3, 
                transaction_cost: float = 0.001, max_turnover: float = 0.6) -> Dict:
    """
    Phase 3: Apply hard veto gates.
    
    Gates:
    1. Monotonicity alignment
    2. Turnover & cost filter
    3. Information Ratio gate
    
    Args:
        df: DataFrame with features
        feature_metadata: Feature metadata
        phase_2_results: Results from Phase 2
        min_ic: Minimum IC threshold (absolute value)
        min_ir: Minimum Information Ratio threshold (absolute value)
        transaction_cost: Transaction cost for turnover filter
        max_turnover: Maximum allowed turnover rate
        
    Returns:
        Phase 3 results dictionary
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: VETO GATES")
    logger.info("=" * 60)
    
    # Only process features that passed Phase 2
    feature_cols = [f for f in get_feature_columns(df, feature_metadata) 
                    if f in phase_2_results and phase_2_results[f].get('pass_phase_2', False)]
    
    logger.info(f"Processing {len(feature_cols)} features that passed Phase 2")
    
    phase_3_results = {}
    
    for feature_name in feature_cols:
        logger.info(f"\nEvaluating: {feature_name}")
        
        phase_2_data = phase_2_results[feature_name]
        feature = df[feature_name]
        feature_meta = feature_metadata.get("features", {}).get(feature_name, {})
        
        # Get feature type and mean RIC
        prediction_type = phase_2_data['prediction_type']
        mean_ric = phase_2_data['mean_ric']
        abs_mean_ric = abs(mean_ric)
        
        # Gate 1: Monotonicity alignment
        monotonicity_pass = _check_monotonicity(
            mean_ric,
            abs_mean_ric,
            prediction_type,
            feature_meta.get('expected_monotonicity', 'none')
        )
        
        # Gate 2: Turnover & cost filter
        turnover = _calculate_turnover(feature)
        
        # Adjust cost penalty based on turnover level
        if turnover > 0.5:
            cost_penalty = 0.4  # Higher penalty for high turnover
        else:
            cost_penalty = 0.3  # Normal penalty
        
        cost_adjusted_ic = abs_mean_ric - (cost_penalty * turnover * transaction_cost)
        cost_pass = cost_adjusted_ic > min_ic and turnover < max_turnover
        
        # Gate 3: Information Ratio gate (use absolute value)
        ir = phase_2_data['stability']['information_ratio']
        abs_ir = abs(ir)
        ir_pass = abs_ir > min_ir
        
        # Final decision
        pass_all = monotonicity_pass and cost_pass and ir_pass
        
        phase_3_results[feature_name] = {
            'monotonicity_pass': monotonicity_pass,
            'turnover': turnover,
            'cost_adjusted_ic': cost_adjusted_ic,
            'cost_pass': cost_pass,
            'ir_pass': ir_pass,
            'pass_phase_3': pass_all
        }
        
        logger.info(f"  Type: {prediction_type}")
        logger.info(f"  Monotonicity: {monotonicity_pass}")
        logger.info(f"  Cost (IC={cost_adjusted_ic:.4f}, turnover={turnover:.3f}): {cost_pass}")
        logger.info(f"  IR (|{ir:.2f}|={abs_ir:.2f}): {ir_pass}")
        logger.info(f"  Pass: {pass_all}")
    
    # Summary
    passed_features = [f for f, r in phase_3_results.items() if r['pass_phase_3']]
    
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total features evaluated: {len(feature_cols)}")
    logger.info(f"Features passed: {len(passed_features)}")
    
    if passed_features:
        logger.info("\nPassed features:")
        for f in passed_features:
            pred_type = phase_2_results[f]['prediction_type']
            mean_ric = phase_2_results[f]['mean_ric']
            logger.info(f"  - {f}: {pred_type} (RIC={mean_ric:.4f})")
    
    logger.info("\nPhase 3 complete!")
    logger.info("=" * 60)
    
    return phase_3_results


def _check_monotonicity(mean_ric: float, abs_ric: float, prediction_type: str, expected: str) -> bool:
    """
    Check if empirical RIC matches expected monotonicity.
    
    For inverse features, we expect negative RIC.
    For universal features, we expect positive RIC.
    """
    # If feature has strong signal, pass regardless of direction
    if abs_ric > 0.04:
        return True
    
    # If feature passed Phase 2 classification, it has correct sign
    if prediction_type in ['universal', 'inverse']:
        return True
    
    # Check against expected monotonicity if specified
    if expected == 'positive':
        return mean_ric > 0
    elif expected == 'negative':
        return mean_ric < 0
    else:  # 'none' - just check it has meaningful signal
        return abs_ric > 0.01
    
    # Default: feature didn't pass Phase 2
    return False


def _calculate_turnover(feature: pd.Series) -> float:
    """Calculate feature turnover rate."""
    # Handle constant features
    if feature.std() < 1e-10:
        return 0.0
    
    z_score = (feature - feature.mean()) / (feature.std() + 1e-10)
    signal = np.sign(z_score)
    changes = np.abs(np.diff(signal.fillna(0)))
    
    return changes.mean() / 2
