"""
Example with Nested Feature Config Format
"""

import pandas as pd
import json
from bitpredict.common.ai.feature_engineering.transformation import apply_transformations

# Create sample data
df = pd.DataFrame({
    'skewness_60': [100.0, 102.5, 98.3, 105.1, 103.2, 101.5, 99.8],
    'kurtosis_60': [1000, 1200, 900, 1100, 1050, 980, 1020]
})

print("Original DataFrame:")
print(df)
print("\n" + "="*80 + "\n")

# Your exact nested config format
Config = {
    "features": {
        "skewness_60": {
            "params": {"window": 60},
            "use_meta": False,
            "transformations": {
                "1": {"method": "differencing", "params": {}},
                "2": {"method": "log_transform", "params": {}}
            }
        },
        "kurtosis_60": {   # ← INSIDE features
            "params": {"window": 60},
            "use_meta": False,
            "transformations": {
                "1": {"method": "differencing", "params": {}}
            }
        }
    }
}

# Config = {
#   "features": {
#     "skewness_60": {
#       "params": {
#         "window": 60
#       },
#       "use_meta": True,
#       "transformations": {
#         "1": {
#           "method": "differencing",
#           "params": {
#             "last_value": 99.8,
#             "shift_amount": 1
#           }
#         }
#       }
#     },
#     "kurtosis_60": {
#       "params": {
#         "window": 60
#       },
#       "use_meta": True,
#       "transformations": {
#         "1": {
#           "method": "differencing",
#           "params": {
#             "last_value": 1020.0,
#             "shift_amount": 1
#           }
#         }
#       }
#     }
#   }
# }


# Apply transformations - auto-detects nested format!
result_df, applied_config = apply_transformations(df, Config)

print("Transformed DataFrame:")
print(result_df)
print("\n" + "="*80 + "\n")

print("Applied Config:")
print(json.dumps(applied_config, indent=2))
print("\n" + "="*80 + "\n")

print("✓ Auto-detected nested config format")
print("✓ Extracted transformations from features.skewness_60.transformations")
print("✓ Applied to price and volume columns")