"""
Feature Engineering Example & Unit Testing Script

This script serves as a working example and a lightweight integration test for the 
feature engineering pipeline. It demonstrates how to load OHLCV data, resample it, 
and compute features using both category-based and specific feature selection.
"""

from bitpredict.common.ai.feature_engineering.features import create_features_by_category, create_features
import pandas as pd

# 1. DATA PREPARATION
# Load sample raw market data (SOL-USDT 1m)
RAW_DATA_PATH = r"D:\trading\market_regime\data\sol_1m.csv"

try:
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 2. RESAMPLING
    # Downsample 1m data to 8h bars for macro feature analysis.
    # Uses 'first' for open, 'max' for high, etc., to maintain OHLCV integrity.
    df = (
        df.set_index("datetime")
        .resample("8h")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
    )
    df = df.dropna().reset_index()

    # 3. FEATURE CALCULATION EXAMPLE
    # Here we demonstrate calculating all features within a specific category.
    # The 'compression_phase' category includes indicators like Squeeze, ADX, etc.
    TARGET_CATEGORIES = ["smart_features"]

    for category in TARGET_CATEGORIES:
        print(f"--- Calculating Category: {category} ---")
        
        # calculate_features_by_category returns (DataFrame, AppliedConfig)
        df_featured, config = create_features_by_category(df, category)
        
        print(f"Applied Config for {category}:\n{config}\n")
        # Export for manual inspection
        OUTPUT_FILE = f"{category}_test_output.csv"
        df_featured.to_csv(OUTPUT_FILE, index=False)
        
        print(f"Calculation complete. Samples:\n{df_featured.head()}")
        print(f"Results saved to: {OUTPUT_FILE}")

except FileNotFoundError:
    print(f"Warning: Test data not found at {RAW_DATA_PATH}. Skipping execution.")

# -----------------------------------------------------------------------------
# ADDITIONAL USAGE EXAMPLES (Commented Out for Batch Testing)
# -----------------------------------------------------------------------------

"""
# EXAMPLE: Calculate ALL available features in the registry
df_all, _ = create_features(df, features="all")

# EXAMPLE: Calculate SPECIFIC features with default parameters
df_subset, _ = create_features(df, features=["returns_log", "vix_fix"])

# EXAMPLE: Calculate features with CUSTOM parameter overrides
custom_params = {
    "sma_7": {"window": 30},   # Override default 7 to 30
    "rsi": {"window": 14}      # Keep default 14
}
df_custom, _ = create_features(df, features=custom_params)
"""
