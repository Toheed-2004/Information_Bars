import talib
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Literal, Tuple
from bitpredict.common.ta.patterns.talib.registry import PatternRegistry
from bitpredict.common.constants import PATTERN_TALIB_COL_PREFIX, OHLCV_COLUMNS
from bitpredict.common.logging import get_logger

# Logger setup
logger = get_logger(__name__)


class CandlestickPatternCalculator:
    """TA-Lib candlestick pattern calculator"""

    def __init__(
        self,
        data: pd.DataFrame,
        patterns: Union[str, List[str], Dict[str, Dict]] = "all",
        return_type: Literal["dataframe", "numpy_array"] = "dataframe",
        drop_nan: bool = True
    ):
        self.data = data
        self.patterns = patterns
        self.return_type = return_type
        self.drop_nan = drop_nan

        logger.info(
            "Initialized CandlestickPatternCalculator | rows=%d | patterns=%s | return_type=%s",
            len(self.data), patterns, return_type
        )

    def calculate(self) -> Tuple[Union[pd.DataFrame, Dict[str, np.ndarray]], Dict[str, Dict]]:
        """Calculate candlestick patterns and return (results, config)"""
        logger.info("Starting candlestick pattern calculation")
        self.data[['open', 'high', 'low', 'close']] = (self.data[['open', 'high', 'low', 'close']].astype(np.float64)
)
        # Parse patterns input
        if self.patterns == "all":
            pattern_list = list(PatternRegistry.get_all_patterns().keys())
            params = {}
            logger.info("Calculating ALL patterns (%d total)", len(pattern_list))
        elif isinstance(self.patterns, str):
            pattern_list = [self.patterns]
            params = {}
            logger.info("Calculating single pattern: %s", self.patterns)
        elif isinstance(self.patterns, list):
            pattern_list = self.patterns
            params = {}
            logger.info("Calculating %d patterns", len(pattern_list))
        elif isinstance(self.patterns, dict):
            pattern_list = list(self.patterns.keys())
            params = self.patterns
            logger.info("Calculating patterns with custom params: %s", pattern_list)
        else:
            logger.error("Invalid patterns input: %s", type(self.patterns))
            raise ValueError("Invalid patterns input")

        # Build config metadata
        config = {name: {"params": params.get(name, {})} for name in pattern_list}

        # Calculate each pattern
        for name in pattern_list:
            # Validate pattern availability
            if not PatternRegistry.is_pattern_available(name):
                logger.warning("Pattern '%s' not available, skipping", name)
                continue

            col_name = f"{PATTERN_TALIB_COL_PREFIX}{name}".lower()

            try:
                logger.debug("Calculating pattern: %s", name)
                func = getattr(talib, name)
                pattern_params = params.get(name, {})

                if pattern_params:
                    values = func(
                        self.data['open'].values,
                        self.data['high'].values,
                        self.data['low'].values,
                        self.data['close'].values,
                        **pattern_params
                    )
                else:
                    values = func(
                        self.data['open'].values,
                        self.data['high'].values,
                        self.data['low'].values,
                        self.data['close'].values
                    )

                # Apply shift to avoid look-ahead bias
                if f"{PATTERN_TALIB_COL_PREFIX}".lower() in col_name:
                    values = values.astype(float)
                    values = np.roll(values, 1)
                    values[0] = np.nan
                    logger.debug("Applied shift to pattern: %s", name)

                self.data[col_name] = values

            except Exception as e:
                logger.warning("Pattern '%s' failed: %s, continuing", name, e)
                continue

        logger.info("Pattern calculation complete")

        patterns_cols = [col for col in self.data.columns if col not in OHLCV_COLUMNS]
        # Optionally remove NaN values
        if self.drop_nan:
            if patterns_cols:
                self.data = self.data.dropna(subset=patterns_cols)
                logger.debug("Dropped NaN rows from results")
        else:
            if patterns_cols:
                self.data[patterns_cols] = self.data[patterns_cols].fillna(0)
                logger.debug("Filled NaN values in pattern columns with 0")

        # Return results
        if self.return_type == "numpy_array":
            logger.debug("Returning result as numpy_array dict")
            return {col: self.data[col].values for col in self.data.columns}, config
        else:
            logger.debug("Returning result as DataFrame")
            return self.data, config