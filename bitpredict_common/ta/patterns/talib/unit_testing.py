import unittest
import numpy as np
import pandas as pd
import os

from bitpredict.common.ta.patterns import calculate_patterns
from bitpredict.common.ta.patterns.plot import plot_patterns


class TestCalculatePatterns(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 200

        datetimes = pd.date_range("2023-01-01", periods=n, freq="1h")
        close = np.cumsum(np.random.randn(n)) + 50_000
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = np.random.randint(100, 1000, n)

        cls.df = pd.DataFrame({
            "datetime": datetimes,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    # --------------------------------------------------
    # Core functionality
    # --------------------------------------------------
    def test_single_pattern(self):
        result, config = calculate_patterns(self.df, "CDLDOJI", library="talib")
        self.assertIn("talib_cdldoji", result.columns)
        self.assertIn("CDLDOJI", config)

    def test_multiple_patterns(self):
        result, _ = calculate_patterns(
            self.df,
            ["CDLDOJI", "CDLHAMMER"],
            library="talib",
        )
        self.assertIn("talib_cdldoji", result.columns)
        self.assertIn("talib_cdlhammer", result.columns)

    def test_custom_parameters(self):
        patterns = {"CDLDARKCLOUDCOVER": {"penetration": 0.7}}
        _, config = calculate_patterns(self.df, patterns, library="talib")
        self.assertEqual(config["CDLDARKCLOUDCOVER"]["params"]["penetration"], 0.7)

    # --------------------------------------------------
    # Return types
    # --------------------------------------------------
    def test_dataframe_return(self):
        result, _ = calculate_patterns(
            self.df,
            ["CDLDOJI"],
            library="talib",
            return_type="dataframe",
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_numpy_return(self):
        result, _ = calculate_patterns(
            self.df,
            ["CDLDOJI"],
            library="talib",
            return_type="numpy",
        )
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["talib_cdldoji"], np.ndarray)

    # --------------------------------------------------
    # Options
    # --------------------------------------------------
    def test_all_patterns(self):
        result, config = calculate_patterns(
            self.df.head(50),
            patterns="all",
            library="talib",
        )
        self.assertGreater(len(config), 0)
        self.assertTrue(any(c.startswith("talib_") for c in result.columns))

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------
    def test_plot_patterns(self):
        result, _ = calculate_patterns(
            self.df.head(50),
            ["CDLDOJI"],
            library="talib",
        )

        plot_patterns(result, title="Test Plot")
        # If no exception → pass
        self.assertTrue(True)

    # --------------------------------------------------
    # Error handling
    # --------------------------------------------------
    def test_invalid_library(self):
        with self.assertRaises(ValueError):
            calculate_patterns(self.df, "CDLDOJI", library="invalid")

    def test_missing_columns(self):
        bad_df = self.df[["datetime", "open", "close"]]
        with self.assertRaises(ValueError):
            calculate_patterns(bad_df, "CDLDOJI", library="talib")

    def test_invalid_pattern_name(self):
        result, _ = calculate_patterns(
            self.df,
            ["NOT_A_PATTERN"],
            library="talib",
        )
        self.assertNotIn("talib_cdl_not_a_pattern", result.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
