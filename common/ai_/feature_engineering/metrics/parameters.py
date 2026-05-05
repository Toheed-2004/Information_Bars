"""
Feature Selection and Model Parameters Configuration
"""

# ============================================================
# MODEL PARAMETERS - XGBRegressor
# ============================================================
MODEL_PARAMS = {
    "model_name": "XGBRegressor",
    "n_estimators": 300,
    # Number of boosting rounds. Higher = better performance but slower training
    "max_depth": 5,
    # Maximum tree depth. Higher = more complex patterns but risk overfitting
    "learning_rate": 0.05,
    # Step size per iteration. Lower = more stable but slower convergence
    "subsample": 0.8,
    # Fraction of samples used per tree. Lower = better generalization
    "colsample_bytree": 0.8,
    # Fraction of features used per tree. Lower = better regularization
    "eval_metric": "rmse",
    # Evaluation metric: "rmse", "mae", "mape"
}

# ============================================================
# PERMUTATION IMPORTANCE PARAMETERS
# ============================================================
PERMUTATION_PARAMS = {
    "n_repeats": 5,
    # Number of shuffle iterations. Higher = more stable importance scores
    "scoring": "r2",
    # Scoring metric: "r2", "neg_mean_squared_error", "neg_mean_absolute_error"
}

# ============================================================
# FEATURE SELECTION PARAMETERS
# ============================================================
FEATURE_SELECTION_PARAMS = {
    "top_n": 20,
    # Number of top features to select. Higher = more information, lower = simpler model
    "correlation_threshold": 0.15,
    # Distance threshold for grouping correlated features
    # 0.15 → correlation > 0.85 | 0.20 → correlation > 0.80 | 0.30 → correlation > 0.70
    "mrmr_top_k": 10,
    # Maximum features for MRMR selection. Higher = more feature diversity
    "granger_maxlag": 5,
    # Maximum lag for Granger causality test. Higher = capture longer-term dependencies
}

# ============================================================
# GLOBAL SETTINGS
# ============================================================
RANDOM_SEED = 42
# For reproducibility across runs

