from typing import Dict, List, Any


class TransformationRegistry:
    """Complete registry containing ALL available data transformations"""

    # ==============================================================================
    # DATA TRANSFORMATIONS
    # ==============================================================================
    TRANSFORMATIONS = {
        # ======================================================================
        # BASIC / STATELESS TRANSFORMS
        # ======================================================================
        "signed_log": {
            "func_name": "signed_log",
            "inputs": ["numeric_columns"],
            "params": {},
            "description": "Apply signed log transform: sign(x) * log(1 + |x|)",
            "use_case": "Financial returns and heavy-tailed distributions",
            "output_range": [],
        },
        
        # ======================================================================
        # DIFFERENCING / TIME-SERIES TRANSFORMS
        # ======================================================================
        "differencing": {
            "func_name": "differencing",
            "inputs": ["numeric_columns"],
            "params": {
                "shift_amount": {"default": 1, "type": int}
            },
            "description": "Apply differencing: x_t - x_{t-shift_amount}",
            "use_case": "Remove trends and make time-series stationary",
            "output_range": [],
        },
        "log_difference": {
            "func_name": "log_difference",
            "inputs": ["numeric_columns"],
            "params": {
                "periods": {"default": 1, "type": int}
            },
            "description": "Apply log difference: log(x_t) - log(x_{t-periods})",
            "use_case": "Percentage change for strictly positive values (prices, volumes)",
            "output_range": [],
        },
        "fractional_differencing": {
            "func_name": "fractional_differencing",
            "inputs": ["numeric_columns"],
            "params": {
                "d": {"default": 0.4, "range": (0.0, 1.0), "type": float}
            },
            "description": "Apply fractional differencing of order d",
            "use_case": "Maintain memory while improving stationarity",
            "output_range": [],
        },
                
        "adaptive_fractional_differencing": {
            "func_name": "adaptive_fractional_differencing",
            "inputs": ["numeric_columns"],
            "params": {
                "max_d": {"default": 1.0, "range": (0.0, 1.0), "type": float},
                "step": {"default": 0.05, "range": (0.001, 0.1), "type": float},
                "significance": {"default": 0.05, "range": (0.001, 0.1), "type": float},
                "n_window": {"default": 200, "range": (50, 1000), "type": int},
            },
            "description": "Apply adaptive fractional differencing with significance test",
            "use_case": "Maintain memory while improving stationarity and automatically select minimal differencing order",
            "output_range": [],
        },
    
        
        # ======================================================================
        # SCALING / NORMALIZATION TRANSFORMS
        # ======================================================================
        "z_score_scaling": {
            "func_name": "z_score_scaling",
            "inputs": ["numeric_columns"],
            "params": {
                "mean": {"default": None, "type": float},
                "std": {"default": None, "type": float}
            },
            "description": "Apply Z-score normalization: (x - mean) / std",
            "use_case": "Standardize features to zero mean and unit variance",
            "output_range": [],
        },
        "binary_mapping": {
            "func_name": "binary_mapping",
            "inputs": ["numeric_columns"],
            "params": {
                "from_range": {"default": (-1, 1), "type": tuple},
                "to_range": {"default": (0, 1), "type": tuple}
            },
            "description": "Map numeric values linearly from one range to another",
            "use_case": "Normalize bounded features (e.g., RSI 0-100 to -1 to 1)",
            "output_range": [],
        },
        "frozen_robust_scaler": {
            "func_name": "frozen_robust_scaler",
            "inputs": ["numeric_columns"],
            "params": {
                "median": {"default": None, "type": float},
                "mad": {"default": None, "type": float}
            },
            "description": "Robust scaling: (x - median) / MAD (Median Absolute Deviation)",
            "use_case": "Heavy-tailed crypto data, robust to outliers",
            "output_range": [],
        },
        "frozen_tanh_scaler": {
            "func_name": "frozen_tanh_scaler",
            "inputs": ["numeric_columns"],
            "params": {
                "scale": {"default": None, "type": float}
            },
            "description": "Smooth compression to [-1, 1]: tanh(x / scale)",
            "use_case": "Bounded output with sign preservation, differentiable everywhere",
            "output_range": [-1.0, 1.0],
        },
        "frozen_linear_bounder": {
            "func_name": "frozen_linear_bounder",
            "inputs": ["numeric_columns"],
            "params": {
                "min_val": {"default": None, "type": float},
                "max_val": {"default": None, "type": float}
            },
            "description": "Linear bound to [-1, 1]: 2 * (x - min)/(max - min) - 1",
            "use_case": "Features with known bounds (RSI 0-100, correlations -1 to 1)",
            "output_range": [-1.0, 1.0],
        },
        
        # ======================================================================
        # DISTRIBUTION TRANSFORMS
        # ======================================================================
        "frozen_quantile_transform": {
            "func_name": "frozen_quantile_transform",
            "inputs": ["numeric_columns"],
            "params": {
                "quantiles": {"default": None, "type": "array"},
                "n_quantiles": {"default": 1000, "type": int}
            },
            "description": "Map to uniform distribution using pre-computed quantiles",
            "use_case": "Output in [0, 1] range, robust to outliers and distribution shape",
            "output_range": [0.0, 1.0],
        },
        "frozen_winsorizer": {
            "func_name": "frozen_winsorizer",
            "inputs": ["numeric_columns"],
            "params": {
                "k": {"default": 1.5, "type": float},
                "q1": {"default": None, "type": float},
                "q3": {"default": None, "type": float},
                "iqr": {"default": None, "type": float}
            },
            "description": "Clip outliers at [Q1 - k*IQR, Q3 + k*IQR]",
            "use_case": "Prevent extreme values from distorting subsequent transformations",
            "output_range": [],
        },
        
        # ======================================================================
        # CRYPTO-SPECIFIC TRANSFORMS
        # ======================================================================
        "frozen_signed_power": {
            "func_name": "frozen_signed_power",
            "inputs": ["numeric_columns"],
            "params": {
                "power": {"default": 0.3, "range": (0.0, 1.0), "type": float}
            },
            "description": "Signed power transform: sign(x) * |x|^power",
            "use_case": "Crypto's heavy-tailed returns, compress extremes while preserving sign",
            "output_range": [],
        },
        
        # ======================================================================
        # TIME-SERIES SPECIFIC TRANSFORMS
        # ======================================================================
        "frozen_differencer": {
            "func_name": "frozen_differencer",
            "inputs": ["numeric_columns"],
            "params": {
                "order": {"default": 1, "type": int},
                "initial_values": {"default": None, "type": "array"}
            },
            "description": "Differencing for non-stationary series: Δx, Δ²x",
            "use_case": "Prices and cumulative indicators needing stationarity",
            "output_range": [],
        },
        "frozen_rolling_zscore": {
            "func_name": "frozen_rolling_zscore",
            "inputs": ["numeric_columns"],
            "params": {
                "window": {"default": 20, "type": int}
            },
            "description": "Rolling z-score: (x - rolling_mean) / rolling_std",
            "use_case": "Local normalization relative to recent window",
            "output_range": [],
        },
        
        # ======================================================================
        # POSITIVE FEATURE HANDLING
        # ======================================================================
        "frozen_log1p_transformer": {
            "func_name": "frozen_log1p_transformer",
            "inputs": ["numeric_columns"],
            "params": {},
            "description": "Log(1 + x) transform for positive features",
            "use_case": "Volume, volatility, and other positive features before normalization",
            "output_range": [0.0, None],  # minimum 0, maximum unbounded
        },
    }

    # ==============================================================================
    # UTILITY METHODS
    # ==============================================================================
    @classmethod
    def get_all_transformations(cls) -> Dict[str, Dict]:
        """Get all available transformations"""
        return cls.TRANSFORMATIONS

    @classmethod
    def get_all(cls) -> Dict[str, Dict]:
        """Get all transformations (alias)"""
        return cls.TRANSFORMATIONS

    @classmethod
    def get_transformation(cls, name: str) -> Dict:
        """Get specific transformation definition"""
        if name not in cls.TRANSFORMATIONS:
            raise ValueError(f"Transformation '{name}' not found")
        return cls.TRANSFORMATIONS[name]

    @classmethod
    def get_transformation_info(cls, transformation_name: str) -> Dict[str, Any]:
        """Get information about a specific transformation"""
        return cls.get_transformation(transformation_name)

    @classmethod
    def get_transformation_description(cls, transformation_name: str) -> str:
        """Get description of a transformation"""
        transformation_def = cls.get_transformation(transformation_name)
        return transformation_def.get("description", "")
    
    @classmethod
    def get_transformation_use_case(cls, transformation_name: str) -> str:
        """Get use case for a transformation"""
        transformation_def = cls.get_transformation(transformation_name)
        return transformation_def.get("use_case", "")

    @classmethod
    def validate_params(cls, transformation_name: str, params: Dict) -> bool:
        """Validate parameter values for a transformation"""
        transformation_def = cls.get_transformation(transformation_name)

        for param_name, param_value in params.items():
            if param_name not in transformation_def["params"]:
                raise ValueError(
                    f"Unknown parameter '{param_name}' for transformation '{transformation_name}'"
                )

            param_spec = transformation_def["params"][param_name]

            # Handle default value extraction
            if isinstance(param_spec, dict) and "default" in param_spec:
                # Type validation if type is specified
                if "type" in param_spec:
                    expected_type = param_spec["type"]
                    if expected_type not in ["array"] and not isinstance(param_value, expected_type):
                        raise ValueError(
                            f"Parameter '{param_name}' must be of type {expected_type.__name__}"
                        )

                # Range validation if range is specified
                if "range" in param_spec and param_value is not None:
                    min_val, max_val = param_spec["range"]
                    if not (min_val <= param_value <= max_val):
                        raise ValueError(
                            f"Parameter '{param_name}' must be between {min_val} and {max_val}"
                        )
            else:
                # Direct value comparison
                if param_value != param_spec:
                    raise ValueError(f"Parameter '{param_name}' must be {param_spec}")

        return True

    @classmethod
    def is_transformation_available(cls, transformation_name: str) -> bool:
        """Check if a transformation is available"""
        return transformation_name in cls.TRANSFORMATIONS
    
    @classmethod
    def get_transformations_by_category(cls, category: str) -> Dict[str, Dict]:
        """
        Get transformations by category.
        
        Categories:
        - basic: signed_log
        - differencing: differencing, log_difference, fractional_differencing
        - scaling: z_score_scaling, binary_mapping, frozen_robust_scaler, frozen_tanh_scaler, frozen_linear_bounder
        - distribution: frozen_quantile_transform, frozen_winsorizer
        - crypto: frozen_signed_power
        - timeseries: frozen_differencer, frozen_rolling_zscore
        - positive: frozen_log1p_transformer
        """
        categories = {
            "basic": ["signed_log"],
            "differencing": ["differencing", "log_difference", "fractional_differencing"],
            "scaling": ["z_score_scaling", "binary_mapping", "frozen_robust_scaler", 
                       "frozen_tanh_scaler", "frozen_linear_bounder"],
            "distribution": ["frozen_quantile_transform", "frozen_winsorizer"],
            "crypto": ["frozen_signed_power"],
            "timeseries": ["frozen_differencer", "frozen_rolling_zscore"],
            "positive": ["frozen_log1p_transformer"],
        }
        
        if category not in categories:
            raise ValueError(f"Unknown category '{category}'")
        
        result = {}
        for transform_name in categories[category]:
            result[transform_name] = cls.TRANSFORMATIONS[transform_name]
        
        return result
    
    @classmethod
    def list_transformations(cls) -> List[str]:
        """List all available transformation names"""
        return sorted(list(cls.TRANSFORMATIONS.keys()))
    
    @classmethod
    def get_output_range(cls, transformation_name: str) -> List:
        """Get expected output range for a transformation"""
        transformation_def = cls.get_transformation(transformation_name)
        return transformation_def.get("output_range", [])