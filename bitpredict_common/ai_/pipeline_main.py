"""
Ultimate Feature Evaluation Pipeline - Main Orchestrator

Coordinates all phases of the feature evaluation pipeline.
"""

import yaml
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union, Optional

from bitpredict.common.logging import get_logger
from bitpredict.common.utils.file_system import read_yaml_config
from bitpredict.common.utils.json_encoder import RobustJSONEncoder
from bitpredict.common.ai.feature_engineering.features.calculator import FeatureCalculator
from bitpredict.common.db.services.data import read_ohlcv
from bitpredict.common.ai.feature_engineering.features import create_features
# Import phase modules
from bitpredict.common.ai.phase_1 import run_phase_1
from bitpredict.common.ai.phase_2 import run_phase_2
from bitpredict.common.ai.phase_3 import run_phase_3
from bitpredict.common.ai.phase_4 import run_phase_4
from bitpredict.common.ai.pipeline_utils import get_cluster_id

logger = get_logger(__name__)


# ============================================================================
# SIMPLE CONFIG LOADER
# ============================================================================


def get_config_value(config: Dict, key: str, default=None):
    """
    Get value from config using dot notation.
    
    Args:
        config: Configuration dictionary
        key: Dot-separated key (e.g., 'phase_1.functions.check_stationarity.params.adf_significance')
        default: Default value if not found
        
    Returns:
        Configuration value or default
    """
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


class FeatureEvaluationPipeline:
    """
    Complete feature evaluation pipeline (Phases 1-4).
    
    Orchestrates the entire feature evaluation process from data loading
    through final feature selection.
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to YAML config file (default: common/ai/pipeline_config.yaml)
            **kwargs: Override config parameters (e.g., adf_significance=0.05)
        """
        # Load configuration
        self.config = read_yaml_config(config_path)
        
        # Extract parameters from config with kwargs override
        self.adf_significance = kwargs.get('adf_significance', 
            get_config_value(self.config, 'phase_1.functions.check_stationarity.params.adf_significance', 0.05))
        self.fracdiff_d = kwargs.get('fracdiff_d',
            get_config_value(self.config, 'phase_1.functions.fractional_differencing.params.fracdiff_d', 0.4))
        self.scaling_method = kwargs.get('scaling_method',
            get_config_value(self.config, 'phase_1.functions.canonical_scaling.params.scaling_method', 'frozen_tanh_scaler'))
        self.transaction_cost = kwargs.get('transaction_cost',
            get_config_value(self.config, 'phase_3.gates.turnover_cost_filter.params.transaction_cost', 0.001))
        self.min_ic = kwargs.get('min_ic',
            get_config_value(self.config, 'phase_3.gates.turnover_cost_filter.params.min_cost_adjusted_ic', 0.02))
        self.min_ir = kwargs.get('min_ir',
            get_config_value(self.config, 'phase_3.gates.information_ratio_gate.params.min_ir', 0.5))
        self.distance_threshold = kwargs.get('distance_threshold',
            get_config_value(self.config, 'phase_4.functions.signature_clustering.params.distance_threshold', 0.05))
        
        # Phase enable flags
        self.phase_1_enabled = get_config_value(self.config, 'phase_1.enabled', True)
        self.phase_2_enabled = get_config_value(self.config, 'phase_2.enabled', True)
        self.phase_3_enabled = get_config_value(self.config, 'phase_3.enabled', True)
        self.phase_4_enabled = get_config_value(self.config, 'phase_4.enabled', True)
        
        # Phase 2 parameters (initialized here, set during run_pipeline)
        self.phase_2_min_ir = None
        self.phase_2_min_mean_ric = None
        self.phase_2_max_mean_ric = None
        self.phase_2_min_regimes = None
        self.phase_2_min_samples = None
        
        self.df = None
        self.feature_metadata = {}
        self.results = {}
    
    # def calculate_features(self, features: Union[str, List[str]]) -> pd.DataFrame:
    #     """
    #     Calculate features using FeatureCalculator.
        
    #     Args:
    #         features: Feature names or 'all'
            
    #     Returns:
    #         DataFrame with calculated features
    #     """
    #     if self.df is None:
    #         raise ValueError("Load data first")
        
    #     logger.info(f"Calculating features: {features}")
        
    #     calculator = FeatureCalculator(self.df)
    #     self.df, self.feature_metadata = calculator.calculate_features(features=features)
        
    #     logger.info(f"Calculated {len(self.feature_metadata.get('features', {}))} features")
    #     return self.df
    
    def run_pipeline(self) -> Dict:
        """
        Run complete pipeline (Phases 1-4).
        
        All parameters are loaded from config file.
        
        Returns:
            Final results dictionary with only features that passed all phases
        """
        # Get all values from config
        exchange = get_config_value(self.config, 'global.data.exchange', 'bybit')
        symbol = get_config_value(self.config, 'global.data.symbol', 'btc')
        time_horizon = get_config_value(self.config, 'global.data.time_horizon', '1h')
        start_date = get_config_value(self.config, 'global.data.start_date', '2021-01-01')
        features = get_config_value(self.config, 'global.features', 'all')
        horizons = get_config_value(self.config, 'global.horizons', [1, 5, 20])
        
        logger.info("=" * 80)
        logger.info("STARTING FEATURE EVALUATION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Data: {exchange}/{symbol}/{time_horizon} from {start_date}")
        logger.info(f"Features: {features}")
        logger.info(f"Horizons: {horizons}")
        logger.info(f"Phases enabled: 1={self.phase_1_enabled}, 2={self.phase_2_enabled}, "
                   f"3={self.phase_3_enabled}, 4={self.phase_4_enabled}")
        
        # Load data
        logger.info(f"\nLoading data...")
        self.df = read_ohlcv(exchange=exchange, symbol=symbol, 
                            time_horizon=time_horizon, start_date=start_date)
        
        # Calculate features
        self.df, self.feature_metadata = create_features(self.df, features)
        # self.calculate_features(features)
        
        # Run Phase 1: Pre-processing (if enabled)
        if self.phase_1_enabled:
            self.df, phase_1_results = run_phase_1(
                self.df, self.feature_metadata,
                self.adf_significance, self.fracdiff_d, self.scaling_method
            )
            self.results['phase_1'] = phase_1_results
        else:
            logger.warning("Phase 1 is disabled, skipping...")
            self.results['phase_1'] = {'features': {}}
        
        # Run Phase 2: Performance Signature (if enabled)
        if self.phase_2_enabled:
            # Read Phase 2 classification parameters from config
            self.phase_2_min_ir = get_config_value(self.config, 'phase_2.functions.classify_features.params.universal.min_ir', 0.3)
            self.phase_2_min_mean_ric = get_config_value(self.config, 'phase_2.functions.classify_features.params.universal.min_mean_ric', 0.015)
            self.phase_2_max_mean_ric = get_config_value(self.config, 'phase_2.functions.classify_features.params.inverse.max_mean_ric', -0.015)
            self.phase_2_min_regimes = get_config_value(self.config, 'phase_2.functions.classify_features.params.universal.min_regimes_with_data', 2)
            self.phase_2_min_samples = get_config_value(self.config, 'phase_2.functions.classify_features.params.universal.min_samples_per_regime', 50)
            use_regime_analysis = get_config_value(self.config, 'phase_2.use_regime_analysis', True)
            
            phase_2_results = run_phase_2(
                self.df, self.feature_metadata, horizons,
                min_ir=self.phase_2_min_ir,
                min_mean_ric=self.phase_2_min_mean_ric,
                max_mean_ric=self.phase_2_max_mean_ric,
                min_regimes_with_data=self.phase_2_min_regimes,
                min_samples_per_regime=self.phase_2_min_samples,
                use_regime_analysis=use_regime_analysis
            )
            self.results['phase_2'] = phase_2_results
        else:
            logger.warning("Phase 2 is disabled, skipping...")
            self.results['phase_2'] = {}
        
        # Run Phase 3: Veto Gates (if enabled)
        if self.phase_3_enabled and self.phase_2_enabled:
            phase_3_results = run_phase_3(
                self.df, self.feature_metadata, self.results['phase_2'],
                self.min_ic, self.min_ir, self.transaction_cost
            )
            self.results['phase_3'] = phase_3_results
        else:
            if not self.phase_3_enabled:
                logger.warning("Phase 3 is disabled, skipping...")
            else:
                logger.warning("Phase 3 skipped (Phase 2 is disabled)")
            self.results['phase_3'] = {}
        
        # Run Phase 4: Orthogonality (if enabled)
        if self.phase_4_enabled and self.phase_2_enabled and self.phase_3_enabled:
            phase_4_results = run_phase_4(
                self.results['phase_2'], self.results['phase_3'], self.distance_threshold
            )
            self.results['phase_4'] = phase_4_results
        else:
            if not self.phase_4_enabled:
                logger.warning("Phase 4 is disabled, skipping...")
            else:
                logger.warning("Phase 4 skipped (Phase 2 or 3 is disabled)")
            self.results['phase_4'] = {'selected_features': []}
        
        # Create final consolidated results
        final_results = self._create_final_results()
        
        # Save results if enabled
        if get_config_value(self.config, 'output.save_results', True):
            self._save_results(final_results, exchange, symbol, time_horizon)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Features calculated: {final_results['metadata']['total_features_evaluated']}")
        logger.info(f"Features passed Phase 2: {final_results['metadata']['features_passed_phase_2']}")
        logger.info(f"Features passed Phase 3: {final_results['metadata']['features_passed_phase_3']}")
        logger.info(f"Features selected (Phase 4): {final_results['metadata']['features_selected_phase_4']}")
        logger.info(f"Final selected features: {list(final_results['selected_features'].keys())}")
        logger.info("=" * 80)
        
        return final_results
    
    def _create_final_results(self) -> Dict:
        """
        Create consolidated results dictionary containing only features that passed all phases.
        
        Returns:
            Dictionary with complete information for each selected feature
        """
        from bitpredict.common.ai.pipeline_utils import get_feature_columns
        
        selected_features = self.results['phase_4']['selected_features']
        
        final_results = {
            'metadata': {
                'total_features_evaluated': len(get_feature_columns(self.df, self.feature_metadata)),
                'features_passed_phase_2': sum(1 for r in self.results['phase_2'].values() if r['pass_phase_2']),
                'features_passed_phase_3': sum(1 for r in self.results['phase_3'].values() if r['pass_phase_3']),
                'features_selected_phase_4': len(selected_features),
                'pipeline_config': {
                    'phase_1': {
                        'adf_significance': self.adf_significance,
                        'fracdiff_d': self.fracdiff_d,
                        'scaling_method': self.scaling_method
                    },
                    'phase_2': {
                        'min_ir': self.phase_2_min_ir,
                        'min_mean_ric': self.phase_2_min_mean_ric,
                        'max_mean_ric': self.phase_2_max_mean_ric,
                        'min_regimes_with_data': self.phase_2_min_regimes,
                        'min_samples_per_regime': self.phase_2_min_samples
                    },
                    'phase_3': {
                        'transaction_cost': self.transaction_cost,
                        'min_cost_adjusted_ic': self.min_ic,
                        'min_ir': self.min_ir
                    },
                    'phase_4': {
                        'distance_threshold': self.distance_threshold
                    }
                }
            },
            'selected_features': {}
        }
        
        # Populate results for each selected feature
        for feature_name in selected_features:
            final_results['selected_features'][feature_name] = {
                # Phase 1: Transformation info
                'phase_1': {
                    'is_stationary': self.results['phase_1']['features'][feature_name]['is_stationary'],
                    'p_value': self.results['phase_1']['features'][feature_name]['p_value'],
                    'domain': self.results['phase_1']['features'][feature_name]['domain'],
                    'transformations': self.results['phase_1']['features'][feature_name]['transformations']
                },
                
                # Phase 2: Performance signature
                'phase_2': {
                    'overall_ric': self.results['phase_2'][feature_name]['overall_ric'],
                    'mean_ric': self.results['phase_2'][feature_name]['mean_ric'],
                    'information_ratio': self.results['phase_2'][feature_name]['stability']['information_ratio'],
                    'mean_ic': self.results['phase_2'][feature_name]['stability']['mean_ic'],
                    'std_ic': self.results['phase_2'][feature_name]['stability']['std_ic'],
                    'prediction_type': self.results['phase_2'][feature_name]['prediction_type'],
                    'signal_multiplier': self.results['phase_2'][feature_name]['signal_multiplier'],
                    'regime_ric': self.results['phase_2'][feature_name]['regime_ric'],
                    'ic_decay': self.results['phase_2'][feature_name]['ic_decay']
                },
                
                # Phase 3: Veto gates
                'phase_3': {
                    'monotonicity_pass': self.results['phase_3'][feature_name]['monotonicity_pass'],
                    'turnover': self.results['phase_3'][feature_name]['turnover'],
                    'cost_adjusted_ic': self.results['phase_3'][feature_name]['cost_adjusted_ic'],
                    'cost_pass': self.results['phase_3'][feature_name]['cost_pass'],
                    'ir_pass': self.results['phase_3'][feature_name]['ir_pass']
                },
                
                # Phase 4: Cluster info
                'phase_4': {
                    'cluster_id': get_cluster_id(feature_name, self.results['phase_4']['clusters'])
                }
            }
        
        return final_results
    
    def _save_results(self, results: Dict, exchange: str, symbol: str, time_horizon: str):
        """
        Save results to JSON file.
        
        Args:
            results: Final results dictionary
            exchange: Exchange name
            symbol: Symbol name
            time_horizon: Time horizon
        """
        # Get output settings from config
        output_dir = get_config_value(self.config, 'output.output_dir', 'common/ai/results')
        save_json = get_config_value(self.config, 'output.formats.json', True)
        
        if not save_json:
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"pipeline_results_{exchange}_{symbol}_{time_horizon}_{timestamp}.json"
        filepath = output_path / filename
        
        # Add timestamp to results
        results['metadata']['timestamp'] = timestamp
        results['metadata']['exchange'] = exchange
        results['metadata']['symbol'] = symbol
        results['metadata']['time_horizon'] = time_horizon
          
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, cls=RobustJSONEncoder)
        
        logger.info(f"\nResults saved to: {filepath}")

# ========================================================================
# CONVENIENCE FUNCTION
# ========================================================================

def run_pipeline(config_path: Optional[str] = None, **kwargs) -> Dict:
    """
    Convenience function to run complete pipeline.
    
    Args:
        config_path: Path to config file (default: common/ai/pipeline_config.yaml)
        **kwargs: Override pipeline configuration parameters (e.g., min_ic=0.03)
        
    Returns:
        Final results dictionary containing only features that passed all phases
        
    Example:
        # Use default config
        results = run_pipeline()
        
        # Use custom config
        results = run_pipeline(config_path="my_config.yaml")
        
        # Override specific parameters
        results = run_pipeline(min_ic=0.03, min_ir=0.6)
        
        # Access selected features
        for name, data in results['selected_features'].items():
            print(f"{name}: {data['phase_2']['prediction_type']}")
    """
    pipeline = FeatureEvaluationPipeline(config_path=config_path, **kwargs)
    return pipeline.run_pipeline()
