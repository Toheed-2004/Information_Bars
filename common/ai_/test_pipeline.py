"""
Test script for Ultimate Feature Evaluation Pipeline (Phases 1-4).

Tests the complete pipeline with sample features using config file.
"""

from bitpredict.common.ai.pipeline_main import FeatureEvaluationPipeline, run_pipeline
from bitpredict.common.logging import get_logger, setup_logging

setup_logging("test_pipeline")
logger = get_logger(__name__)


def test_with_default_config():
    """Test pipeline using all defaults from config file."""
    logger.info("=" * 80)
    logger.info("TEST 1: DEFAULT CONFIG")
    logger.info("=" * 80)
    
    # Simply run with defaults from config
    config_path = "common/ai/pipeline_config.yaml"

    results = run_pipeline(config_path=config_path)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("DEFAULT CONFIG TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total features evaluated: {results['metadata']['total_features_evaluated']}")
    logger.info(f"Features passed Phase 2: {results['metadata']['features_passed_phase_2']}")
    logger.info(f"Features passed Phase 3: {results['metadata']['features_passed_phase_3']}")
    logger.info(f"Features selected: {results['metadata']['features_selected_phase_4']}")
    logger.info(f"Selected features: {list(results['selected_features'].keys())}")
    logger.info("=" * 80)
    return results


def test_with_parameter_overrides():
    """Test pipeline with parameter overrides."""
    logger.info("\n\n" + "=" * 80)
    logger.info("TEST 2: PARAMETER OVERRIDES")
    logger.info("=" * 80)

    config_path = "common/ai/pipeline_config.yaml"

    # Override specific parameters
    results = run_pipeline(
        config_path=config_path,
        adf_significance=0.05,
        fracdiff_d=0.4,
        min_ic=0.015,
        min_ir=0.5
    )
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER OVERRIDE TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total features evaluated: {results['metadata']['total_features_evaluated']}")
    logger.info(f"Features passed Phase 2: {results['metadata']['features_passed_phase_2']}")
    logger.info(f"Features passed Phase 3: {results['metadata']['features_passed_phase_3']}")
    logger.info(f"Features selected: {results['metadata']['features_selected_phase_4']}")
    logger.info(f"Selected features: {list(results['selected_features'].keys())}")
    logger.info("=" * 80)
    return results


def test_with_pipeline_class():
    """Test using FeatureEvaluationPipeline class directly."""
    logger.info("\n\n" + "=" * 80)
    logger.info("TEST 3: PIPELINE CLASS")
    logger.info("=" * 80)
    
    # Create pipeline with parameter overrides
    config_path = "common/ai/pipeline_config.yaml"
    pipeline = FeatureEvaluationPipeline(
        config_path=config_path,
        adf_significance=0.05,
        fracdiff_d=0.4,
        scaling_method="frozen_tanh_scaler",
        transaction_cost=0.001,
        min_ic=0.015,
        min_ir=0.5,
        distance_threshold=0.05
    )
    
    # Run pipeline (uses config for data/features)
    results = pipeline.run_pipeline()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE CLASS TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total features evaluated: {results['metadata']['total_features_evaluated']}")
    logger.info(f"Features passed Phase 2: {results['metadata']['features_passed_phase_2']}")
    logger.info(f"Features passed Phase 3: {results['metadata']['features_passed_phase_3']}")
    logger.info(f"Features selected: {results['metadata']['features_selected_phase_4']}")
    logger.info(f"Selected features: {list(results['selected_features'].keys())}")
    logger.info("=" * 80)
    return results


if __name__ == "__main__":
    try:
        # Run tests
        # results1 = test_with_default_config()
        results2 = test_with_parameter_overrides()
        # results3 = test_with_pipeline_class()
        
        logger.info("\n\n" + "=" * 80)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
