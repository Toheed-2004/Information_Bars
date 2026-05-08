import json
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from mrmr import mrmr_regression
from xgboost import XGBRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests
from bitpredict.common.ai.feature_engineering.metrics.parameters import MODEL_PARAMS, PERMUTATION_PARAMS, FEATURE_SELECTION_PARAMS

class FeatureSelector:
    """
    FeatureSelector class for selecting important features from a dataframe.
    
    This class computes multiple importance metrics to identify the most predictive features:
    - Tree Importance: Which features the model splits on
    - Permutation Importance: Performance drop when shuffling features
    - Mutual Information: Statistical dependency with target
    - SHAP Importance: Feature contribution to predictions
    - MRMR: Relevant but non-redundant features
    - Granger Causality: Past values predicting future (time series)
    
    The final ranking combines all 6 methods for robust feature selection.
    """

    def __init__(self, df: pd.DataFrame, target_col: str = "target"):
        """
        Initialize the FeatureSelector.

        Args:
            df (pd.DataFrame): Input dataframe with features and target
            target_col (str): Name of target column (default: "target")
        
        Attributes:
            self.X: Features dataframe (all columns except target)
            self.y: Target series
        """
        self.df = df
        self.target_col = target_col

        self.X = (
            df
            .drop(columns=[target_col])
            .select_dtypes(include=["number", "bool"])
        )
        self.y = df[target_col]

   
    # FEATURE GROUPING METHOD
   
    def correlation_groups(self):
        """
        Group correlated features using hierarchical clustering.
        
        This method identifies features that are highly correlated with each other
        to help with feature redundancy reduction. Highly correlated features 
        contain similar information and only one from each group may be needed.
        
        Returns:
            tuple: (correlated_groups, uncorrelated_features)
                - correlated_groups: Dict of groups with 2+ correlated features
                - uncorrelated_features: List of standalone features
        """
     
        # Step 1: Compute correlation matrix and convert to distance
     
        """
        Calculate absolute correlation between all features.
        Why absolute? Because -0.8 correlation is as strong as +0.8.
        Then convert to distance: dist = 1 - correlation
        (distance of 0 = identical, distance of 1 = no correlation)
        """
        corr = self.X.corr(method="pearson").abs().fillna(0)
        dist = 1 - corr

     
        # Step 2: Perform hierarchical clustering on distance matrix
     
        """
        Use AgglomerativeClustering to group similar features:
        - metric="precomputed": Uses the distance matrix we created
        - linkage="average": Average distance between clusters
        - distance_threshold: Features with distance < 0.15 are grouped together
          (meaning correlation > 0.85 are considered correlated)
        """
        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=FEATURE_SELECTION_PARAMS["correlation_threshold"],
            n_clusters=None,
            compute_full_tree=True
        )
        labels = clustering.fit_predict(dist)

     
        # Step 3: Group features by their cluster label
     
        """
        For each feature, assign it to a group based on its label.
        groups[label] contains all features in that cluster.
        Example: {0: ['feature_1', 'feature_2'], 1: ['feature_5']}
        """
        groups = {}
        for col, label in zip(self.X.columns, labels):
            groups.setdefault(label, []).append(col)

     
        # Step 4: Separate correlated and uncorrelated features
     
        """
        - correlated: Groups with 2+ features (redundant features)
          Example: {0: ['feature_1', 'feature_2'], 2: ['feature_5', 'feature_6']}
          
        - uncorrelated: Groups with only 1 feature (unique features)
          Example: ['feature_3', 'feature_4', 'feature_7']
          
        Only take first feature from single-feature groups [v[0]]
        """
        correlated = {int(k): v for k, v in groups.items() if len(v) > 1}
        uncorrelated = [v[0] for v in groups.values() if len(v) == 1]

        return correlated, uncorrelated

   
    # IMPORTANCE CALCULATION METHODS
   
    
    def tree_importance(self, model):
        """
        Calculate Tree-based Feature Importance.
        
        This uses the built-in feature importance from XGBoost which measures
        how much each feature contributed to splits in the decision trees.
        
        Method: Which features the tree split on the most?
        Higher values = more important for model decisions
        
        Args:
            model: Trained XGBRegressor model
        
        Returns:
            pd.Series: Importance score for each feature
        """
     
        # Extract importance scores from the trained model
     
        """
        model.feature_importances_ is built-in to XGBoost.
        Creates a Series with feature names as index for easy lookup.
        """
        return pd.Series(model.feature_importances_, index=self.X.columns)

    def permutation_importance(self, model):
        """
        Calculate Permutation-based Feature Importance.
        
        Shuffles each feature one at a time and measures the drop in model 
        performance. Bigger performance drop = feature is more important.
        
        Method: What breaks the model if I scramble it?
        This is often more reliable than tree importance.
        
        Args:
            model: Trained XGBRegressor model
        
        Returns:
            pd.Series: Mean importance from n_repeats iterations
        """
     
        # Compute permutation importance using sklearn
     
        """
        For each feature:
        1. Shuffle/scramble that feature's values
        2. Measure how model performance drops
        3. Repeat n_repeats times (default: 5) for stability
        4. Average the results
        
        Parameters from config:
        - n_repeats: Number of shuffles per feature
        - scoring: Metric to measure performance (r2, mse, etc)
        - random_state: For reproducibility
        """
        perm_imp = permutation_importance(
            model, self.X, self.y,
            n_repeats=PERMUTATION_PARAMS["n_repeats"],
            scoring=PERMUTATION_PARAMS["scoring"],
            random_state=42
        )
        return pd.Series(perm_imp.importances_mean, index=self.X.columns)

    def mutual_info(self):
        """
        Calculate Mutual Information between features and target.
        
        Measures how much knowing a feature helps predict the target.
        Captures both linear and non-linear relationships.
        
        Method: How much does this feature know about the target?
        This is model-independent, only looks at dependency.
        
        Returns:
            pd.Series: Mutual information score for each feature (higher = more related)
        """
     
        # Compute mutual information for regression
     
        """
        Mutual Information measures dependency between feature and target.
        Information Theory: How much does feature reduce uncertainty about target?
        
        Advantages:
        - Captures non-linear relationships
        - No model training required
        - Fast computation
        """
        return pd.Series(mutual_info_regression(self.X, self.y), index=self.X.columns)

    def shap_importance(self, model):
        """
        Calculate SHAP-based Feature Importance.
        
        Uses game theory to explain model predictions. Shows how much each 
        feature pushes the prediction up or down from the baseline.
        
        Method: Who pushed the prediction up or down?
        This is the most theoretically sound method.
        
        Args:
            model: Trained XGBRegressor model
        
        Returns:
            pd.Series: Mean absolute SHAP values (importance magnitude)
        """
     
        # Create SHAP explainer and compute SHAP values
     
        """
        Step 1: Create SHAP explainer for the model
        Step 2: Calculate SHAP values for all samples
        Step 3: Take absolute value (magnitude of impact)
        Step 4: Average across all samples
        
        SHAP values are Shapley values from game theory.
        Shows each feature's contribution to each prediction.
        """
        explainer = shap.Explainer(model)
        shap_vals = explainer(self.X)
        return pd.Series(np.abs(shap_vals.values).mean(axis=0), index=self.X.columns)

    def mrmr_selection(self):
        """
        Minimum Redundancy Maximum Relevance (MRMR) Feature Selection.
        
        Selects features that are:
        1. Highly relevant to the target
        2. Not redundant with each other
        
        Method: Pick useful features without duplicates.
        This balances feature quality vs feature diversity.
        
        Returns:
            pd.Series: Binary scores (1 = selected, 0 = not selected)
        """
     
        # Determine max features to select (K parameter)
     
        """
        k = minimum of:
        - mrmr_top_k from config (default: 10)
        - Total number of features available
        
        Prevents trying to select more features than exist.
        """
        k = min(FEATURE_SELECTION_PARAMS.get("mrmr_top_k", 10), len(self.X.columns))

     
        # Run MRMR algorithm and assign binary scores
     
        """
        selected = list of feature names chosen by MRMR
        Create Series with all 0s, then set selected features to 1.
        
        This gives us a binary score (0 or 1) that can be combined
        with continuous scores from other methods.
        """
        selected = mrmr_regression(X=self.X, y=self.y, K=k)
        scores = pd.Series(0, index=self.X.columns)
        scores[selected] = 1
        return scores

    def granger_causality(self, maxlag: int = 5):
        """
        Granger Causality Test for Time Series Features.
        
        Tests whether past values of a feature help forecast the target.
        Only meaningful for time-series data with temporal ordering.
        
        Method: Does the past of this feature predict the target?
        This is useful for time series forecasting models.
        
        Args:
            maxlag: Maximum number of past time steps to check (default: 5)
        
        Returns:
            pd.Series: Log-transformed p-values (higher = more causal)
        """
        scores = {}
        
     
        # Test each feature for Granger causality

        """
        For each feature:
        1. Prepare data with feature and target
        2. Run Granger causality test with multiple lags
        3. Extract p-values for each lag
        4. Use smallest (best) p-value as the feature's score
        5. Convert to -log10(p) for easier interpretation
           (higher value = smaller p-value = more significant)
        """
        for col in self.X.columns:
            try:
             
                # Prepare data and run Granger causality test
             
                """
                test_data: 2D array with [feature, target]
                grangercausalitytests: Runs test at lags 1 to maxlag
                Results structure: results[lag][0]['ssr_ftest'][1] = p-value
                """
                test_data = pd.concat([self.X[col], self.y], axis=1)
                results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                
             
                # Extract p-values and compute score
             
                """
                pvals: List of p-values for lags 1 to maxlag
                min(pvals): Use the best (smallest) p-value
                -log10(p): Transform to importance scale
                  p=0.05 → -log10(0.05) ≈ 1.3
                  p=0.001 → -log10(0.001) ≈ 3.0
                Add 1e-10 to prevent log(0)
                """
                pvals = [results[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1)]
                scores[col] = -np.log10(min(pvals) + 1e-10)
                
            except Exception:
             
                # Handle errors (invalid data, etc)
             
                """
                If Granger test fails (e.g., insufficient data, non-stationary):
                Set score to 0 (not causal)
                Continue with other features instead of crashing
                """
                scores[col] = 0
        
        return pd.Series(scores)

   
    # COMPUTE ALL IMPORTANCE SCORES
    def compute_all_importance(self):
        """
        Compute all 6 importance metrics and combine them.
        
        This method:
        1. Trains an XGBoost model on the data
        2. Computes 6 different importance metrics
        3. Combines them into a single ranking by averaging
        
        Returns:
            pd.DataFrame: Importance scores for each feature
                Columns: [tree, perm, mutual_info, shap, mrmr, granger, mean]
                Index: Feature names (sorted by mean importance)
        """
     
        # Step 1: Initialize and train XGBoost model
        """
        Create XGBRegressor with parameters from config.
        These parameters control model complexity and training.
        Train on all available data (X, y).
        """
        model = XGBRegressor(
            n_estimators=MODEL_PARAMS["n_estimators"],
            max_depth=MODEL_PARAMS["max_depth"],
            learning_rate=MODEL_PARAMS["learning_rate"],
            subsample=MODEL_PARAMS["subsample"],
            colsample_bytree=MODEL_PARAMS["colsample_bytree"],
            eval_metric=MODEL_PARAMS["eval_metric"]
        )
        model.fit(self.X, self.y)

     
        # Step 2: Compute all 6 importance metrics
        """
        Call each importance method:
        1. tree_importance: Uses model.feature_importances_
        2. permutation_importance: Shuffles and measures drop
        3. mutual_info: Statistical dependency (model-independent)
        4. shap_importance: SHAP values from game theory
        5. mrmr_selection: Redundancy-aware selection
        6. granger_causality: Time series causal relationships
        
        Each returns a pd.Series with one score per feature.
        """
        model_importance = self.tree_importance(model)
        permutation_importance = self.permutation_importance(model)
        shap_importance = self.shap_importance(model)
        mutual_infomation = self.mutual_info()
        mrmr_selection = self.mrmr_selection()
        granger_causality = self.granger_causality(maxlag=FEATURE_SELECTION_PARAMS.get("granger_maxlag", 5))

     
        # Step 3: Combine all metrics into one dataframe
     
        """
        Concatenate all Series side-by-side to create a dataframe:
        Features × 6 methods matrix
        
        Each row = one feature
        Each column = one importance method
        """
        imp = pd.concat([model_importance, permutation_importance, mutual_infomation, 
                        shap_importance, mrmr_selection, granger_causality], axis=1)
        
     
        # Step 4: Add column names and compute mean importance
     
        """
        Name each column for clarity.
        Add "mean" column = average of all 6 methods.
        This is the final robust ranking.
        """
        imp.columns = ["model_importance", "permutation_importance", "mutual_infomation", 
                      "shap_importance", "mrmr_selection", "granger_causality"]
        
        # Normalize all metrics (0-1 scaling)
        # This scales each column independently so that the smallest value becomes 0
        # and the largest becomes 1. It ensures that all importance metrics
        # contribute equally when computing the mean, preventing any single metric
        # (like granger_causality) from dominating the final score.
        imp = (imp - imp.min()) / (imp.max() - imp.min())

        imp["mean"] = imp.mean(axis=1)

     
        # Step 5: Sort by mean importance (highest first)
     
        """
        ascending=False: Highest scores first
        This gives us the final feature ranking.
        """
        return imp.sort_values("mean", ascending=False)

   

def calculate_feature_importance(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """
    Calculate feature importance using the FeatureSelector class.

    Args:
        df (pd.DataFrame): Input dataframe with features and target
        target_col (str): Name of target column (default: "target")
    
    Returns:
        pd.DataFrame: DataFrame with feature importance scores
    """
    
    selector = FeatureSelector(df, target_col)
    
    # Compute importance matrix (features × methods)
    importance = selector.compute_all_importance()

    # Drop cross-method mean (not needed)
    if "mean" in importance.columns:
        importance = importance.drop(columns=["mean"])

    # Ensure clean ordering
    importance.index.name = "feature"
    importance.columns.name = "method"

    return importance

