"""
ml_module — Production-Grade ML System for High-Frequency Bar Analysis
=======================================================================

Architecture overview:
  labeling/       → Triple-barrier label generation
  features/       → Fractional differencing + feature engineering
  validation/     → Purged & Combinatorial Purged Cross-Validation (CPCV)
  models/         → Primary learners + meta-model stacking ensemble
  backtest_bridge/→ Prediction → backtest signal adapter
  utils/          → Logging, serialization, metrics
  config/         → YAML-driven configuration
  tests/          → Unit & integration tests

Quick start:
    from ml_module import MLPipeline
    pipeline = MLPipeline.from_config("config/ml_config.yaml")
    results  = pipeline.run(bar_csv="data/dollar_bars.csv", bar_type="dollar")
"""

from ml_module.pipeline import MLPipeline

__all__ = ["MLPipeline"]
__version__ = "1.0.0"
