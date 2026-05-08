"""
Unified Feature Calculator

This module provides the central FeatureCalculator class and convenience functions 
to compute a wide array of technical, statistical, and advanced features for 
financial OHLCV datasets by leveraging the feature registry.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple
from bitpredict.common.ai.feature_engineering.features.registry import FEATURES, validate_features, get_features_by_category
from bitpredict.common.constants import OHLCV_COLUMNS
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)

class FeatureCalculator:
    """
    Orchestrates the calculation of features from various specialized modules.
    
    Acts as a bridge between the feature registry (configuration) and the 
    implementation classes (PriceReturn, Volatility, etc.).
    
    Initialization converts the input pandas DataFrame into efficient numpy 
    arrays to speed up downstream rolling calculations.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the FeatureCalculator with OHLCV data.

        Args:
            df (pd.DataFrame): Input data containing 'open', 'high', 'low', 
                               'close', 'volume' columns and optionally 'datetime'.
        """
        self.df = df
        self.n = len(df)
        self.eps = 1e-8

        # Vectorize primary series for high-speed calculation
        self.open = df["open"].values.astype(np.float64)
        self.high = df["high"].values.astype(np.float64)
        self.low = df["low"].values.astype(np.float64)
        self.close = df["close"].values.astype(np.float64)
        self.volume = df["volume"].values.astype(np.float64)

        # Handle temporal axis for TimeFeatures
        if "datetime" in df.columns:
            self.datetime = pd.to_datetime(df["datetime"])
        elif isinstance(df.index, pd.DatetimeIndex):
            self.datetime = df.index
        else:
            self.datetime = None
        
        # Cache for instantiated feature category classes (Lazy Loading)
        self.category_instances = {}

        # Flatten the nested registry for faster lookup
        self.all_features = self._flatten_registry()
    
    def _get_category_instance(self, class_name: str):
        """
        Returns a (cached) instance of the required feature implementation class.
        Uses local imports to avoid circular dependencies and heavy startup.
        
        Args:
            class_name (str): Identifier for the category (e.g., 'PR', 'VOL', 'INFO').
        """
        if class_name not in self.category_instances:
            # Import and instantiate the specific class based on code
            if class_name == "PR":
                from bitpredict.common.ai.feature_engineering.features.price_return import PriceReturnFeatures
                self.category_instances[class_name] = PriceReturnFeatures(
                    close=self.close, volume=self.volume, n=self.n, eps=self.eps
                )
            elif class_name == "VOL":
                from bitpredict.common.ai.feature_engineering.features.volatility import VolatilityFeatures
                self.category_instances[class_name] = VolatilityFeatures(
                    open=self.open, high=self.high, low=self.low, 
                    close=self.close, volume=self.volume, n=self.n, eps=self.eps
                )
            elif class_name == "V":
                from bitpredict.common.ai.feature_engineering.features.volume import VolumeFeatures
                self.category_instances[class_name] = VolumeFeatures(
                    df=self.df, open=self.open, high=self.high, low=self.low,
                    close=self.close, volume=self.volume, n=self.n, eps=self.eps
                )
            elif class_name == "TI":
                from bitpredict.common.ai.feature_engineering.features.technical_indicators import TechnicalIndicators
                self.category_instances[class_name] = TechnicalIndicators(
                    df=self.df, high=self.high, low=self.low, 
                    close=self.close, volume=self.volume, n=self.n, eps=self.eps
                )
            elif class_name == "STAT":
                from bitpredict.common.ai.feature_engineering.features.statistical_econometric import StatisticalEconometric
                self.category_instances[class_name] = StatisticalEconometric(
                    close=self.close, volume=self.volume, n=self.n, eps=self.eps
                )
            elif class_name == "TIME":
                from bitpredict.common.ai.feature_engineering.features.time_multi_timeframe import TimeFeatures
                self.category_instances[class_name] = TimeFeatures(
                    close=self.close, n=self.n, eps=self.eps, datetime=self.datetime
                )
            elif class_name == "GEO":
                from bitpredict.common.ai.feature_engineering.features.geometric_topological import GeometricTopologicalFeatures
                self.category_instances[class_name] = GeometricTopologicalFeatures(
                    open=self.open, high=self.high, low=self.low, 
                    close=self.close, volume=self.volume, n=self.n, eps=self.eps
                )
            elif class_name == "INFO":
                from bitpredict.common.ai.feature_engineering.features.information_causal import InformationCausalFeatures
                self.category_instances[class_name] = InformationCausalFeatures(
                    close=self.close, volume=self.volume, n=self.n, eps=self.eps
                )
            elif class_name == "COMP":
                from bitpredict.common.ai.feature_engineering.features.compression_phase import CompressionFeatures
                self.category_instances[class_name] = CompressionFeatures(
                    open=self.open, high=self.high, low=self.low, 
                    close=self.close, volume=self.volume, n=self.n, eps=self.eps
                )
            elif class_name == "EVENT":
                from bitpredict.common.ai.feature_engineering.features.event_survival import EventSurvivalFeatures
                self.category_instances[class_name] = EventSurvivalFeatures(
                    high=self.high, low=self.low, close=self.close, 
                    n=self.n, eps=self.eps
                )
            elif class_name == "RISK":
                from bitpredict.common.ai.feature_engineering.features.risk_drawdown import RiskFeatures
                self.category_instances[class_name] = RiskFeatures(
                    close=self.close, n=self.n, eps=self.eps
                )
            elif class_name == "SMART":
                from bitpredict.common.ai.feature_engineering.features.smart_features import SmartFeatures
                self.category_instances[class_name] = SmartFeatures(
                    df=self.df, high=self.high, low=self.low,
                    close=self.close, volume=self.volume, n=self.n, eps=self.eps
                )
            else:
                raise ValueError(f"Feature category code '{class_name}' not implemented.")
        
        return self.category_instances[class_name]
    
    def _flatten_registry(self) -> Dict[str, Dict]:
        """
        Flatten registry for fast lookup.
        Since FEATURES is already a dict, this is now a simple pass-through.
        Kept for backward compatibility.
        """
        # FEATURES is already a dict with feature names as keys
        # No flattening needed - just return it
        return FEATURES

    def calculate_features(
    self,
    features: Union[str, List[str], Dict[str, Dict]] = "all",
    drop_nan: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
        """
        Main entry point for feature engineering. Executes methods defined in registry.
        Uses batched column addition to avoid DataFrame fragmentation.
        """

        # --------------------------------------------------
        # Resolve feature list + params
        # --------------------------------------------------
        if features == "all":
            feature_list = list(self.all_features.keys())
            params = {}
        elif isinstance(features, str):
            feature_list = [features]
            params = {}
        elif isinstance(features, list):
            feature_list = features
            params = {}
        elif isinstance(features, dict):
            feature_list = list(features.keys())
            params = features
        else:
            raise ValueError("Invalid format for 'features' argument.")

        final_config = {"features": {}}
        
        # Dictionary to collect all new feature columns
        new_features = {}

        # --------------------------------------------------
        # Feature execution loop - collect results
        # --------------------------------------------------
        for feature_name in feature_list:

            info = self.all_features.get(feature_name)

            # -------- Registry lookup guard (CRITICAL) --------
            if info is None:
                logger.error(f"Feature '{feature_name}' not found in registry")
                new_features[feature_name] = np.full(self.n, np.nan)
                continue

            try:
                inst = self._get_category_instance(info["class_name"])
                func = getattr(inst, info["method"])

                # Merge params (registry defaults + overrides)
                merged_params = dict(info.get("params", {}))
                if isinstance(features, dict):
                    merged_params.update(params.get(feature_name, {}))

                logger.debug(f"Computing feature: {feature_name}")
                result = func(**merged_params) if merged_params else func()

                outputs = info.get("outputs", [feature_name])

                # -------- Output handling - store in dict --------
                if isinstance(result, tuple):
                    for idx, out in enumerate(outputs):
                        new_features[out] = result[idx] if idx < len(result) else np.full(self.n, np.nan)

                elif isinstance(result, np.ndarray):
                    new_features[outputs[0]] = result

                elif isinstance(result, list):
                    for idx, series in enumerate(result):
                        col = outputs[idx] if idx < len(outputs) else f"{outputs[0]}_{idx}"
                        new_features[col] = series

                else:
                    raise TypeError(f"Unsupported return type {type(result)}")

                domain = info.get("metadata", {}).get("domain", None)
                expected_monotonicity = info.get("metadata", {}).get("expected_monotonicity", "none")
                final_config["features"][feature_name] = {
                    "params": merged_params,
                    "domain": domain,
                    "expected_monotonicity": expected_monotonicity
                }

            except Exception as e:
                logger.error(f"Failed to calculate '{feature_name}': {e}", exc_info=True)
                for out in info.get("outputs", [feature_name]):
                    new_features[out] = np.full(self.n, np.nan)

        # --------------------------------------------------
        # Batch add all features at once (avoids fragmentation)
        # --------------------------------------------------
        if new_features:
            features_df = pd.DataFrame(new_features, index=self.df.index)
            self.df = pd.concat([self.df, features_df], axis=1)

        # --------------------------------------------------
        # Final NaN handling
        # --------------------------------------------------
        feature_cols = [
            c for c in self.df.columns
            if c not in OHLCV_COLUMNS and c != "datetime"
        ]
        if feature_cols:
            if drop_nan:
                self.df = self.df.dropna(subset=feature_cols, how="any")
            else:
                self.df[feature_cols] = self.df[feature_cols].fillna(0.0)

        return self.df, final_config
    
    def calculate_features_by_category(
        self,
        categories: Union[str, List[str]],
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """Executes all features belonging to the specified categories."""
        if isinstance(categories, str):
            categories = [categories]

        full_list = []
        for cat in categories:
            try:
                full_list.extend(get_features_by_category(cat))
            except ValueError as e:
                logger.error(f"Category error: {e}")

        return self.calculate_features(features=full_list, **kwargs)

# =============================================================================
# CONVENIENCE WRAPPERS (Functional Interface)
# =============================================================================

def create_features(
    df: pd.DataFrame,
    features: Union[str, List[str], Dict[str, Dict]] = "all",
    drop_nan: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """Helper: Instantiate calculator and run specific features."""
    return FeatureCalculator(df).calculate_features(features=features, drop_nan=drop_nan)

def create_features_by_category(
    df: pd.DataFrame,
    categories: Union[str, List[str]],
    **kwargs,
) -> Tuple[pd.DataFrame, Dict]:
    """Helper: Instantiate calculator and run all features in categories."""
    return FeatureCalculator(df).calculate_features_by_category(categories=categories, **kwargs)


    


    