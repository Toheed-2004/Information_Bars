import pandas as pd
from typing import Dict, Tuple
from bitpredict.common.ai.feature_engineering.transformation.core import DataTransformationCore
from bitpredict.common.ai.feature_engineering.transformation.registry import TransformationRegistry
from bitpredict.common.logging import get_logger
import copy


logger = get_logger(__name__)


def apply_transformations(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply transformations to DataFrame based on nested config.

    Only numeric columns are transformed. Non-numeric columns are skipped.
    Saves learned metadata in applied_config.

    Args:
        df: Input DataFrame
        config: Nested feature transformation configuration

    Returns:
        Tuple of:
        - Transformed DataFrame
        - Updated config with saved metadata for successful transforms
    """
    result_df = df.copy()
    applied_config = copy.deepcopy(config)

    if "features" not in config or not isinstance(config["features"], dict):
        raise ValueError("Config must contain 'features' dict")

    for feature_name, feature_cfg in config["features"].items():
        if feature_name not in result_df.columns:
            logger.warning(f"Column '{feature_name}' not found, skipping feature.")
            continue

        transformations = feature_cfg.get("transformations", {})
        if not transformations:
            logger.info(f"No transformations defined for '{feature_name}', skipping.")
            continue

        # Apply transformations in sorted sequence
        for seq in sorted(transformations.keys(), key=int):
            transform_spec = transformations[seq]
            method = transform_spec.get("method")
            params = dict(transform_spec.get("params", {}))

            if not TransformationRegistry.is_transformation_available(method):
                logger.warning(f"Transformation '{method}' not registered for '{feature_name}', skipping.")
                continue

            # Skip non-numeric columns for all numeric transforms
            if not pd.api.types.is_numeric_dtype(result_df[feature_name]):
                logger.warning(f"Skipping '{method}' on non-numeric column '{feature_name}'")
                continue

            try:
                func = getattr(DataTransformationCore, method)
                output = func(result_df, columns=[feature_name], **params)

                # Update DataFrame and learned metadata
                if isinstance(output, tuple):
                    result_df, learned_meta = output
                    if feature_name in learned_meta:
                        params.update(learned_meta[feature_name])
                else:
                    result_df = output

                applied_config["features"][feature_name]["transformations"][seq]["params"] = params
                logger.debug(f"Applied '{method}' to '{feature_name}' (seq={seq})")

            except Exception as e:
                logger.error(f"Failed '{method}' on '{feature_name}' (seq={seq}): {e}")
                break  # Stop remaining transforms for this feature

    return result_df, applied_config
