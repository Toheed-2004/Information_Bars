import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from bitpredict.common.logging import get_logger
from statsmodels.tsa.stattools import adfuller

logger = get_logger(__name__)


class DataTransformationCore:
    """
    Core data transformation methods implemented using pure NumPy and pandas.
    
    All transformations return a tuple (df, learned) where `learned` contains
    important metadata for stateful transforms (for later inference).
    
    Categories:
    - Basic/stateless: signed_log
    - Differencing: differencing, log_difference, fractional_differencing
    - Scaling/normalization: z_score_scaling, frozen_robust_scaler, frozen_tanh_scaler, frozen_linear_bounder
    - Distribution transforms: frozen_quantile_transform, frozen_winsorizer
    - Crypto-specific: frozen_signed_power
    - Time-series: frozen_differencer, frozen_rolling_zscore
    - Positive features: frozen_log1p_transformer
    - Range mapping: binary_mapping
    """

    
    # =====================================================================
    # BASIC / STATELESS TRANSFORMS
    # =====================================================================
    
    @staticmethod
    def signed_log(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply signed log transform: sign(x) * log(1 + |x|).

        Args:
            df: Input DataFrame
            columns: List of columns to transform (None = all numeric)
            **kwargs: extra parameters (ignored)

        Returns:
            df: Transformed DataFrame
            learned: empty dict (no metadata)
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
            except Exception as e:
                logger.error(f"Failed to apply signed_log to column '{col}': {str(e)}")

        return df, {}

    
    # =====================================================================
    # DIFFERENCING / TIME-SERIES TRANSFORMS
    # =====================================================================
    
    @staticmethod
    def differencing(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        shift_amount: int = 1,
        last_value: Optional[float] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply differencing: x_t - x_{t-shift_amount}.
        Saves last value for later inference.

        Args:
            df: Input DataFrame
            columns: Columns to transform
            shift_amount: order of differencing (default: 1)
            last_value: single-row inference value

        Returns:
            df: Transformed DataFrame
            learned: metadata containing last_value and shift_amount
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                if len(df) == 1 and last_value is not None:
                    df[col] = df[col] - last_value
                else:
                    learned[col] = {
                        "last_value": float(df[col].iloc[-1]),
                        "shift_amount": shift_amount
                    }
                    df[col] = df[col].diff(shift_amount)
            except Exception as e:
                logger.error(f"Failed to apply differencing to column '{col}': {str(e)}")

        return df, learned

    
    @staticmethod
    def log_difference(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        periods: int = 1,
        last_value: Optional[float] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply log difference: log(x_t) - log(x_{t-periods}).
        Only works on positive values. Saves last value for inference.

        Args:
            df: Input DataFrame
            columns: Columns to transform
            periods: number of periods to difference (default: 1)
            last_value: single-row inference value

        Returns:
            df: Transformed DataFrame
            learned: metadata containing last_value and periods
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                if (df[col] <= 0).any():
                    logger.warning(f"Column '{col}' contains non-positive values, skipping log_difference")
                    continue

                if len(df) == 1 and last_value is not None:
                    df[col] = np.log(df[col]) - np.log(last_value)
                else:
                    learned[col] = {
                        "last_value": float(df[col].iloc[-1]),
                        "periods": periods
                    }
                    df[col] = np.log(df[col]).diff(periods)
            except Exception as e:
                logger.error(f"Failed to apply log_difference to column '{col}': {str(e)}")

        return df, learned

    
    @staticmethod
    def fractional_differencing(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        d: float = 0.5,
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply fractional differencing of order d.
        Maintains memory while improving stationarity.

        Args:
            df: Input DataFrame
            columns: Columns to transform
            d: Fractional differencing parameter (0 < d < 1)
            weights: Pre-computed weights for inference

        Returns:
            df: Transformed DataFrame
            learned: metadata containing weights and d
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                series = df[col].dropna().values
                if len(series) == 0:
                    continue

                # Compute fractional differencing weights
                if weights is None:
                    weights_list = [1.0]
                    for k in range(1, len(series)):
                        weights_list.append(-weights_list[-1] * (d - k + 1) / k)
                    learned[col] = {"weights": weights_list[:4], "d": d}
                else:
                    weights_list = weights

                # Apply fractional differencing
                weights_array = np.array(weights_list[:len(series)])
                result = np.zeros_like(series)

                for i in range(len(series)):
                    result[i] = np.dot(series[:i+1][::-1], weights_array[:i+1])

                # Assign result back to DataFrame
                df.loc[df[col].notna(), col] = result
                
                logger.debug(f"Applied fractional_differencing (d={d}) to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply fractional_differencing to column '{col}': {str(e)}")

        return df, learned

    @staticmethod
    def adaptive_fractional_differencing(
        df: pd.DataFrame,
        columns: List[str] = None,
        d: Optional[float] = None,  # harmless injection safety
        max_d: float = 1.0,
        step: float = 0.05,
        significance: float = 0.05,
        n_window: Optional[int] = None,  # optional truncation window
        weights: Optional[np.ndarray] = None,  # swallow injected param
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:

        from statsmodels.tsa.stattools import adfuller

        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:

            series = df[col].dropna()
            if len(series) < 20:
                continue

            # ---------------------------------
            # 1️⃣ Check if already stationary
            # ---------------------------------
            try:
                pval_orig = adfuller(
                    series,
                    regression="ct",   # allow deterministic trend
                    autolag="AIC"
                )[1]
            except Exception:
                continue

            if pval_orig < significance:
                learned[col] = {
                    "d": 0.0,
                    "weights": [1.0]
                }
                continue

            # ---------------------------------
            # 2️⃣ Search smallest stationary d
            # ---------------------------------
            best_d = None

            for candidate_d in np.arange(0.0, max_d + step, step):

                # compute weights
                w = [1.0]
                for k in range(1, len(series)):
                    w.append(-w[-1] * (candidate_d - k + 1) / k)

                    # truncate very small weights for efficiency
                    if abs(w[-1]) < 1e-6:
                        break

                w = np.array(w)

                if len(w) < 2:
                    continue

                # apply convolution (fast, consistent)
                diffed = np.convolve(series.values, w[::-1], mode="valid")

                if len(diffed) < 20:
                    continue

                try:
                    pval = adfuller(
                        diffed,
                        regression="ct",
                        autolag="AIC"
                    )[1]
                except Exception:
                    continue

                if pval < significance:
                    best_d = candidate_d
                    break

            # if nothing found, fallback to max_d
            if best_d is None:
                best_d = max_d

            # ---------------------------------
            # 3️⃣ Apply best_d to full series
            # ---------------------------------
            w_final = [1.0]
            for k in range(1, len(series)):
                w_final.append(-w_final[-1] * (best_d - k + 1) / k)

                if abs(w_final[-1]) < 1e-6:
                    break

            w_final = np.array(w_final)

            result = np.convolve(series.values, w_final[::-1], mode="valid")

            # align index
            aligned_index = series.index[len(w_final) - 1:]
            df.loc[aligned_index, col] = result
            df.loc[series.index[:len(w_final) - 1], col] = np.nan

            # ---------------------------------
            # 4️⃣ Store learned params
            # ---------------------------------
            learned[col] = {
                "d": float(best_d),
                "weights": np.round(w_final[:4], 6).tolist()
            }

        return df, learned

    # =====================================================================
    # SCALING / NORMALIZATION TRANSFORMS
    # =====================================================================
    
    @staticmethod
    def z_score_scaling(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply Z-score normalization: (x - mean) / std.
        Saves mean and std as metadata.

        Args:
            df: Input DataFrame
            columns: Columns to transform
            mean: Pre-computed mean (for inference)
            std: Pre-computed std (for inference)

        Returns:
            df: Transformed DataFrame
            learned: dict with mean and std for each column
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                if mean is None or std is None:
                    mean_val = float(df[col].mean())
                    std_val = float(df[col].std()) or 1.0
                    learned[col] = {"mean": mean_val, "std": std_val}
                else:
                    mean_val, std_val = mean, std or 1.0

                if std_val == 0:
                    logger.warning(f"Column '{col}' has zero standard deviation. Skipping.")
                    continue

                df[col] = (df[col] - mean_val) / std_val
                
                logger.debug(f"Applied z_score_scaling to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply z_score_scaling to column '{col}': {str(e)}")

        return df, learned

    
    @staticmethod
    def binary_mapping(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        from_range: tuple = (-1, 1),
        to_range: tuple = (0, 1),
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Map numeric values linearly from one range to another.
        Useful for normalizing bounded features.

        Args:
            df: Input DataFrame
            columns: Columns to transform
            from_range: Source range (min, max)
            to_range: Target range (min, max)

        Returns:
            df: Transformed DataFrame
            learned: metadata containing range information
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        from_min, from_max = from_range
        to_min, to_max = to_range

        if from_max - from_min == 0:
            logger.error("Source range has zero width. Cannot apply binary_mapping.")
            return df, learned

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                df[col] = (df[col] - from_min) / (from_max - from_min) * (to_max - to_min) + to_min
                learned[col] = {
                    "from_range": from_range,
                    "to_range": to_range
                }
                logger.debug(f"Applied binary_mapping to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply binary_mapping to column '{col}': {str(e)}")

        return df, learned

    
    @staticmethod
    def frozen_robust_scaler(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        median: Optional[float] = None,
        mad: Optional[float] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Robust scaling: (x - median) / MAD (Median Absolute Deviation).
        Robust to outliers. Saves median and MAD as metadata.

        Args:
            df: Input DataFrame
            columns: Columns to transform
            median: Pre-computed median (for inference)
            mad: Pre-computed MAD (for inference)

        Returns:
            df: Transformed DataFrame
            learned: dict with median and mad for each column
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                if median is None or mad is None:
                    median_val = float(np.median(df[col]))
                    mad_val = float(np.median(np.abs(df[col] - median_val))) or 1.0
                    learned[col] = {"median": median_val, "mad": mad_val}
                else:
                    median_val, mad_val = median, mad or 1.0

                if mad_val == 0:
                    mad_val = float(np.std(df[col].values))
                    if mad_val == 0:
                        logger.warning(f"Column '{col}' has zero variance. Skipping.")
                        continue

                df[col] = (df[col] - median_val) / mad_val
                logger.debug(f"Applied frozen_robust_scaler to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply frozen_robust_scaler to column '{col}': {str(e)}")

        return df, learned

    
    @staticmethod
    def frozen_tanh_scaler(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        scale: Optional[float] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Smooth compression to [-1, 1]: tanh(x / scale).
        Preserves sign, differentiable everywhere. Saves scale for inference.

        Args:
            df: Input DataFrame
            columns: Columns to transform
            scale: Scale parameter (if None, computed from MAD)

        Returns:
            df: Transformed DataFrame
            learned: dict with scale for each column
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                if scale is None:
                    values = df[col].dropna().values
                    median = np.median(values)
                    mad = np.median(np.abs(values - median))
                    scale_val = mad * 1.4826 or 1.0
                    learned[col] = {"scale": float(scale_val)}
                else:
                    scale_val = scale

                df[col] = np.tanh(df[col] / scale_val)
                logger.debug(f"Applied frozen_tanh_scaler (scale={scale_val:.4f}) to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply frozen_tanh_scaler to column '{col}': {str(e)}")

        return df, learned

    
    @staticmethod
    def frozen_linear_bounder(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Linear bound to [-1, 1]: 2 * (x - min)/(max - min) - 1.
        For features with known bounds (e.g., RSI 0-100, correlations -1 to 1).

        Args:
            df: Input DataFrame
            columns: Columns to transform
            min_val: Minimum value (if None, computed from data)
            max_val: Maximum value (if None, computed from data)

        Returns:
            df: Transformed DataFrame
            learned: dict with min and max for each column
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                if min_val is None:
                    min_val_computed = float(df[col].min())
                else:
                    min_val_computed = min_val

                if max_val is None:
                    max_val_computed = float(df[col].max())
                else:
                    max_val_computed = max_val

                if max_val_computed <= min_val_computed:
                    logger.warning(f"Column '{col}' has invalid range [{min_val_computed}, {max_val_computed}]. Skipping.")
                    continue

                df[col] = 2 * (df[col] - min_val_computed) / (max_val_computed - min_val_computed) - 1
                learned[col] = {"min_val": min_val_computed, "max_val": max_val_computed}
                logger.debug(f"Applied frozen_linear_bounder to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply frozen_linear_bounder to column '{col}': {str(e)}")

        return df, learned

    
    # =====================================================================
    # DISTRIBUTION TRANSFORMS
    # =====================================================================
    
    @staticmethod
    def frozen_quantile_transform(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        n_quantiles: int = 1000,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Map values to uniform distribution using precomputed quantiles.
        Guarantees output in [0, 1] range. Robust to outliers and distribution shape.
        Saves quantiles for inference.

        Args:
            df: Input DataFrame
            columns: Columns to transform
            quantiles: Pre-computed quantiles (for inference)
            n_quantiles: Number of quantiles to compute (training mode only)

        Returns:
            df: Transformed DataFrame
            learned: dict with quantiles for each column
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                if quantiles is None:
                    series = df[col].dropna().values
                    if len(series) < 2:
                        continue
                    
                    n_quantiles_adjusted = min(n_quantiles, len(series))
                    qs = np.percentile(series, np.linspace(0, 100, n_quantiles_adjusted))
                    learned[col] = {"quantiles": [float(q) for q in qs]}
                else:
                    qs = np.array(quantiles)

                # Use linear interpolation for quantile transform
                df[col] = np.interp(df[col], qs, np.linspace(0, 1, len(qs)))
                logger.debug(f"Applied frozen_quantile_transform to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply frozen_quantile_transform to column '{col}': {str(e)}")

        return df, learned

    
    @staticmethod
    def frozen_winsorizer(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        q1: Optional[float] = None,
        q3: Optional[float] = None,
        iqr: Optional[float] = None,
        k: float = 1.5,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Clip values at [Q1 - k*IQR, Q3 + k*IQR] to remove extreme outliers.
        Saves q1, q3, iqr as metadata.

        Args:
            df: Input DataFrame
            columns: Columns to transform
            q1: Pre-computed first quartile (for inference)
            q3: Pre-computed third quartile (for inference)
            iqr: Pre-computed IQR (for inference)
            k: IQR multiplier (default: 1.5)

        Returns:
            df: Transformed DataFrame
            learned: dict with q1, q3, iqr for each column
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                if q1 is None or q3 is None or iqr is None:
                    series = df[col].dropna().values
                    if len(series) < 4:
                        logger.warning(f"Column '{col}' has insufficient data for winsorizing. Skipping.")
                        continue
                    
                    q1_val = float(np.percentile(series, 25))
                    q3_val = float(np.percentile(series, 75))
                    iqr_val = q3_val - q1_val
                    learned[col] = {"q1": q1_val, "q3": q3_val, "iqr": iqr_val}
                else:
                    q1_val, q3_val, iqr_val = q1, q3, iqr

                lower_bound = q1_val - k * iqr_val
                upper_bound = q3_val + k * iqr_val
                
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                logger.debug(f"Applied frozen_winsorizer (k={k}) to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply frozen_winsorizer to column '{col}': {str(e)}")

        return df, learned

    
    # =====================================================================
    # CRYPTO-SPECIFIC TRANSFORMS
    # =====================================================================
    
    @staticmethod
    def frozen_signed_power(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        power: float = 0.3,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Signed power transform: sign(x) * |x|^power.
        Handles crypto's heavy-tailed returns. Power ≈ 0.3 compresses extremes
        while preserving sign information.

        Args:
            df: Input DataFrame
            columns: Columns to transform
            power: Power parameter (0 < power < 1)

        Returns:
            df: Transformed DataFrame
            learned: dict with power value
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                df[col] = np.sign(df[col]) * np.abs(df[col]) ** power
                learned[col] = {"power": power}
                logger.debug(f"Applied frozen_signed_power (power={power}) to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply frozen_signed_power to column '{col}': {str(e)}")

        return df, learned

    
    # =====================================================================
    # TIME-SERIES SPECIFIC TRANSFORMS
    # =====================================================================
    
    @staticmethod
    def frozen_differencer(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        order: int = 1,
        initial_values: Optional[List[float]] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Differencing for non-stationary series: Δx, Δ²x.
        For prices, cumulative indicators that need stationarity.

        Args:
            df: Input DataFrame
            columns: Columns to transform
            order: Order of differencing (1 or 2)
            initial_values: Initial values for differencing (for inference)

        Returns:
            df: Transformed DataFrame
            learned: dict with order and initial values
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                # Save initial values before differencing
                if initial_values is None:
                    initial_vals = [float(df[col].iloc[0])]
                    if order > 1:
                        initial_vals.append(float(df[col].iloc[1]))
                    learned[col] = {"order": order, "initial_values": initial_vals}
                
                # Apply differencing
                for i in range(order):
                    df[col] = df[col].diff()
                    
                logger.debug(f"Applied frozen_differencer (order={order}) to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply frozen_differencer to column '{col}': {str(e)}")

        return df, learned

    
    @staticmethod
    def frozen_rolling_zscore(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        window: int = 20,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Rolling z-score: (x - rolling_mean) / rolling_std.
        For features that need local normalization (relative to recent window).

        Args:
            df: Input DataFrame
            columns: Columns to transform
            window: Rolling window size

        Returns:
            df: Transformed DataFrame
            learned: dict with window size
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = df[col].rolling(window=window, min_periods=1).std()
                
                rolling_std = rolling_std.replace(0, 1)
                
                df[col] = (df[col] - rolling_mean) / rolling_std
                learned[col] = {"window": window}
                logger.debug(f"Applied frozen_rolling_zscore (window={window}) to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply frozen_rolling_zscore to column '{col}': {str(e)}")

        return df, learned

    
    # =====================================================================
    # POSITIVE FEATURE HANDLING
    # =====================================================================
    
    @staticmethod
    def frozen_log1p_transformer(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Log(1 + x) transform for positive features.
        For volume, volatility, and other positive features before normalization.
        Handles zeros gracefully (log(1+0) = 0).

        Args:
            df: Input DataFrame
            columns: Columns to transform

        Returns:
            df: Transformed DataFrame
            learned: empty dict (stateless transform)
        """
        learned = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue
            try:
                df[col] = np.log1p(df[col])
                logger.debug(f"Applied frozen_log1p_transformer to column '{col}'")
                
            except Exception as e:
                logger.error(f"Failed to apply frozen_log1p_transformer to column '{col}': {str(e)}")

        return df, learned