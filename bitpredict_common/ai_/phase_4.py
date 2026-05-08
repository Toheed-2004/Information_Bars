"""
Phase 4: Orthogonality & Interaction Control

Ensures feature orthogonality through clustering and redundancy pruning.
"""

import numpy as np
from typing import Dict, List
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


def run_phase_4(phase_2_results: Dict, phase_3_results: Dict,
                distance_threshold: float = 0.05) -> Dict:
    """
    Phase 4: Ensure feature orthogonality.
    
    Steps:
    1. Feature neutralization (skipped - no control variables)
    2. Signature clustering
    3. Redundancy pruning
    
    Args:
        phase_2_results: Results from Phase 2
        phase_3_results: Results from Phase 3
        distance_threshold: Distance threshold for clustering
        
    Returns:
        Phase 4 results dictionary
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: ORTHOGONALITY")
    logger.info("=" * 60)
    
    # Get features that passed Phase 3
    passed_features = [
        f for f, r in phase_3_results.items()
        if r['pass_phase_3']
    ]
    
    if len(passed_features) < 2:
        logger.info("Less than 2 features passed Phase 3, skipping Phase 4")
        return {
            'passed_features': passed_features,
            'clusters': {0: passed_features} if passed_features else {},
            'selected_features': passed_features,
            'n_original': len(passed_features),
            'n_selected': len(passed_features)
        }
    
    logger.info(f"Processing {len(passed_features)} features")
    
    # Step 1: Neutralization (skipped)
    logger.info("\nStep 1: Neutralization (skipped - no control variables)")
    
    # Step 2: Signature clustering
    logger.info("\nStep 2: Signature clustering")
    
    # Check if regime analysis was used
    has_regime_data = all(
        phase_2_results[f]['regime_ric'] is not None 
        for f in passed_features
    )
    
    if has_regime_data:
        # Use regime RIC signatures for clustering
        logger.info("  Using regime RIC signatures")
        signatures = {}
        for f in passed_features:
            regime_ric = phase_2_results[f]['regime_ric']
            # Extract RIC values from regime_ric dict
            signatures[f] = [v['ric'] for v in regime_ric.values()]
        clusters = _cluster_signatures(signatures, distance_threshold)
    else:
        # Simple mode: Use IC decay as signature
        logger.info("  Using IC decay signatures (regime analysis disabled)")
        signatures = {
            f: list(phase_2_results[f]['ic_decay'].values())
            for f in passed_features
        }
        clusters = _cluster_signatures(signatures, distance_threshold)
    
    # Step 3: Redundancy pruning
    logger.info("\nStep 3: Redundancy pruning")
    selected_features = _prune_redundant(clusters, passed_features, phase_2_results)
    
    phase_4_results = {
        'passed_features': passed_features,
        'clusters': clusters,
        'selected_features': selected_features,
        'n_original': len(passed_features),
        'n_selected': len(selected_features)
    }
    
    logger.info(f"\nPhase 4 complete! Selected {len(selected_features)}/{len(passed_features)} features")
    logger.info("=" * 60)
    
    return phase_4_results


def _cluster_signatures(signatures: Dict[str, List], 
                        distance_threshold: float) -> Dict:
    """Cluster features by performance signature."""
    if len(signatures) < 2:
        return {0: list(signatures.keys())}
    
    # Convert to matrix (handle NaN)
    feature_names = list(signatures.keys())
    signature_matrix = []
    for name in feature_names:
        sig = [r['ric'] if isinstance(r, dict) else r for r in signatures[name]]
        sig = [0.0 if np.isnan(x) else x for x in sig]
        signature_matrix.append(sig)
    
    signature_matrix = np.array(signature_matrix)
    
    # Calculate distances
    distances = pdist(signature_matrix, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='average',
        metric='precomputed'
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    # Group by cluster
    clusters = {}
    for feature_name, label in zip(feature_names, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(feature_name)
    
    return clusters


def _prune_redundant(clusters: Dict, feature_names: List[str], 
                    phase_2_results: Dict) -> List[str]:
    """Prune redundant features within clusters."""
    selected = []
    
    for cluster_id, members in clusters.items():
        if len(members) == 1:
            selected.append(members[0])
            continue
        
        # Select feature with highest mean RIC
        best_feature = max(
            members,
            key=lambda f: abs(phase_2_results[f]['mean_ric'])
        )
        selected.append(best_feature)
    
    return selected
