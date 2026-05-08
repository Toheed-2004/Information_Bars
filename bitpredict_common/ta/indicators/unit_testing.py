"""
Unified Technical Indicators Test Suite

Features:
- Unit tests: TA-Lib, VectorBT, data validation, registry, main calculator
- Visual tests: candlestick charts, single/multiple indicators, trend patterns
- Custom parameter handling
- Full report generation (unit tests)
- CLI options: --unit-only, --visual-only, --no-plots, --verbose, --no-report
"""

import unittest
import numpy as np
import pandas as pd
import warnings
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ------------------------------
# Project path setup
# ------------------------------
project_root = Path(__file__).parent.parent.parent.parent
script_dir = Path(__file__).parent
if str(script_dir) in sys.path:
    sys.path.remove(str(script_dir))
sys.path.insert(0, str(project_root))

# ------------------------------
# Imports for testing
# ------------------------------
import talib
import vectorbtpro as vbt

from bitpredict.common.ta.indicators import (
    calculate_indicators,
    calculate_talib_indicators,
    calculate_vectorbt_indicators
)
from bitpredict.common.ta.indicators.talib.registry import (
    get_indicators_by_category, create_column_name, TALIB_INDICATORS, INDICATOR_CATEGORIES
)
from bitpredict.common.utils.data_validation import validate_ohlcv
from bitpredict.common.ta.indicators.plot import plot_indicators

warnings.filterwarnings('ignore')

# ------------------------------
# Test Data Generator
# ------------------------------
class DataGenerator:
    @staticmethod
    def generate_ohlcv(n_periods=200, start_price=100, volatility=0.02, trend=0.0001, seed=42):
        np.random.seed(seed)
        datetimes = [datetime(2023,1,1)+timedelta(days=i) for i in range(n_periods)]
        returns = np.random.normal(trend, volatility, n_periods)
        close = start_price * np.exp(np.cumsum(returns))
        open_ = np.zeros(n_periods); high = np.zeros(n_periods); low = np.zeros(n_periods)
        for i in range(n_periods):
            open_[i] = close[i-1]*(1+np.random.normal(0,volatility/2)) if i>0 else close[i]*(1+np.random.normal(0,volatility/2))
            intraday = close[i]*abs(np.random.normal(0,volatility))
            high[i] = max(open_[i], close[i]) + intraday/2
            low[i] = min(open_[i], close[i]) - intraday/2
        vol = 1+np.abs(np.diff(close, prepend=close[0]))/close*10
        volume = np.clip(1_000_000*vol*(1+np.random.normal(0,0.2,n_periods)), 500_000, 3_000_000)
        return pd.DataFrame({'datetime': datetimes,'open': open_,'high': high,'low': low,'close': close,'volume': volume})

    @staticmethod
    def generate_trending_data(n_periods=200, trend_type='uptrend'):
        trends = {'uptrend': 0.001,'downtrend': -0.001,'sideways':0.0,'volatile':0.0}
        volatility = {'uptrend':0.015,'downtrend':0.015,'sideways':0.01,'volatile':0.05}[trend_type]
        return DataGenerator.generate_ohlcv(n_periods, trend=trends[trend_type], volatility=volatility)

# ------------------------------
# UNIT TESTS
# ------------------------------
class TestRegistry(unittest.TestCase):
    def test_talib_structure(self):
        self.assertIsInstance(TALIB_INDICATORS, dict)
        for name, cfg in TALIB_INDICATORS.items():
            for key in ['lib','func_name','inputs','params','outputs','description']:
                self.assertIn(key, cfg)
    def test_categories(self):
        self.assertIsInstance(INDICATOR_CATEGORIES, dict)
        for cat, inds in INDICATOR_CATEGORIES.items():
            self.assertIsInstance(inds, list)
            self.assertGreater(len(inds),0)
    def test_get_by_category(self):
        for cat in INDICATOR_CATEGORIES.keys():
            self.assertIsInstance(get_indicators_by_category(cat), list)
        with self.assertRaises(ValueError):
            get_indicators_by_category('invalid')
    def test_create_column_name(self):
        self.assertEqual(create_column_name("RSI","rsi",{"timeperiod":14}),"talib_ind_rsi_14")
        self.assertEqual(create_column_name("BOP","bop",{}),"talib_ind_bop")

class TestDataValidation(unittest.TestCase):
    def setUp(self): self.data = DataGenerator.generate_ohlcv(100)
    def test_validate_ohlc_only(self):
        validated = validate_ohlcv(self.data.copy(), ohlc_only=True)
        self.assertTrue(set(['datetime','open','high','low','close','volume']).issubset(validated.columns))
    def test_validate_missing_cols(self):
        bad = self.data[['datetime','open','close']].copy()
        with self.assertRaises(ValueError): validate_ohlcv(bad)

class TestTalibCalculator(unittest.TestCase):
    def setUp(self): self.data = DataGenerator.generate_ohlcv(200)
    def test_single_indicator(self):
        result, _ = calculate_talib_indicators(self.data.copy(), indicators="RSI")
        # --- Print full DataFrame sample ---
        print("\nSingle Indicator - RSI full DataFrame sample:")
        print(result.head(5))  # first 5 rows
        self.assertTrue(any('rsi' in c.lower() for c in result.columns))

    def test_multiple_indicators(self):
        indicators = ["RSI", "MACD", "SMA"]
        result, _ = calculate_talib_indicators(self.data.copy(), indicators=indicators)
        # --- Print full DataFrame sample ---
        print("\nMultiple Indicators full DataFrame sample:")
        print(result.head(5))
        self.assertTrue(all(any(i.lower() in c.lower() for c in result.columns) for i in ['rsi','macd','sma']))

    def test_custom_params(self):
        indicators = {
            'RSI': {'timeperiod': 21},
            'MACD': {'fastperiod': 8, 'slowperiod': 21, 'signalperiod': 5}
        }
        result, _ = calculate_talib_indicators(self.data.copy(), indicators=indicators)
        # --- Print full DataFrame sample ---
        print("\nCustom Params Indicators full DataFrame sample:")
        print(result.head(5))
        self.assertTrue(any('rsi_21' in c.lower() for c in result.columns))

class TestBaseFunctions(unittest.TestCase):
    def setUp(self): self.data = DataGenerator.generate_ohlcv(150)
    def test_talib_main(self):
        df,_ = calculate_indicators(self.data.copy(), indicators=["RSI","MACD"], library="talib")
        print("HELLLLO")
        print(df.head(10))
        self.assertTrue(any('rsi' in col.lower() for col in df.columns))
    def test_invalid_library(self):
        with self.assertRaises(ValueError): calculate_indicators(self.data.copy(), indicators="RSI", library="invalid")

class TestVectorBTCalculator(unittest.TestCase):
    def setUp(self): self.data = DataGenerator.generate_ohlcv(200)
    def test_vectorbt_indicators(self):
        df,_ = calculate_vectorbt_indicators(self.data.copy(), indicators=["RSI","MACD"])
        self.assertTrue(isinstance(df,pd.DataFrame))
    def test_calculate_main_vectorbt(self):
        df,_ = calculate_indicators(self.data.copy(), indicators=["RSI"], library="vectorbt")
        self.assertTrue('rsi' in _.get('indicators',{}))

# ------------------------------
# VISUAL TESTS
# ------------------------------
class VisualTestRunner:
    def __init__(self, output_dir='./visual_test_output'):
        self.output_dir = Path(output_dir); self.output_dir.mkdir(exist_ok=True)
        self.plot = plot_indicators
    def _save_fig(self, fig, name): 
        path = self.output_dir / f"{name}.png"; fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig); print(f"Saved: {path}")
    def run_all(self):
        # Basic candlestick
        df = DataGenerator.generate_ohlcv(200)
        fig = self.plot(df); self._save_fig(fig,'basic_candlestick')
        # RSI
        df,_ = calculate_indicators(df, indicators="SUPERTREND", library="talib")
        fig = self.plot(df); self._save_fig(fig,'rsi')
        # Multiple indicators
        df2 = DataGenerator.generate_ohlcv(300)
        df2,_ = calculate_indicators(df2, indicators=["RSI","MACD","BBANDS"], library="talib")
        fig = self.plot(df2); self._save_fig(fig,'multiple_indicators')
        # Trend patterns
        fig,axes=plt.subplots(2,2,figsize=(16,12)); fig.suptitle("Trend Patterns")
        trends = ['uptrend','downtrend','sideways','volatile']; positions=[(0,0),(0,1),(1,0),(1,1)]
        for t,p in zip(trends,positions):
            df = DataGenerator.generate_trending_data(200,t)
            df,_ = calculate_indicators(df, indicators="RSI", library="talib")
            axes[p].plot(df['close']); axes[p].set_title(t)
        plt.tight_layout(); self._save_fig(fig,'trend_patterns')

# ------------------------------
# TEST RUNNERS
# ------------------------------
def run_unit_tests(verbosity=2, save_report=True):
    loader = unittest.TestLoader(); suite = unittest.TestSuite()
    for cls in [TestRegistry, TestDataValidation, TestTalibCalculator, TestBaseFunctions, TestVectorBTCalculator]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    # Save report
    if save_report:
        report_path = Path('unit_test_report.txt')
        with open(report_path,'w') as f:
            f.write(f"Tests run: {result.testsRun}\nFailures:{len(result.failures)}\nErrors:{len(result.errors)}\n")
        print(f"Report saved to: {report_path}")
    return result

def run_visual_tests():
    VisualTestRunner().run_all()

# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description='Technical Indicators Test Suite')
    parser.add_argument('--unit-only', action='store_true')
    parser.add_argument('--visual-only', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-report', action='store_true')
    args = parser.parse_args()
    verbosity = 2 if args.verbose else 1
    unit_result = None
    if not args.visual_only: unit_result = run_unit_tests(verbosity, save_report=not args.no_report)
    if not args.unit_only and not args.no_plots: run_visual_tests()
    return 0 if (unit_result is None or unit_result.wasSuccessful()) else 1

if __name__ == '__main__': sys.exit(main())
