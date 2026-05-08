# Import the main function to apply data transformations on a DataFrame
from bitpredict.common.ai.feature_engineering.transformation.calculator import apply_transformations

# Import the registry that contains all available transformations and their metadata
from bitpredict.common.ai.feature_engineering.transformation.registry import TransformationRegistry  

# Define what will be exported when someone does `from <module> import *`
__all__ = ["apply_transformations", "TransformationRegistry"]

# ------------------------------------------------------------------------------
# Explanation:
# 1. `apply_transformations` - function that applies one or more transformations
#    to columns in a DataFrame based on a nested configuration.
# 2. `TransformationRegistry` - central registry holding all transformations,
#    their default parameters, input types, and descriptions.
# 3. `__all__` - controls the public API of this module. Only these two names
#    will be accessible when using wildcard imports.
# ------------------------------------------------------------------------------
