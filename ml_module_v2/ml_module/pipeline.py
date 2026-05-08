"""
pipeline.py
-----------
Main orchestrator.

Flow per bar type:
  1. Load CSV                          → pd.DataFrame
  2. Triple-barrier labels             → pd.Series {-1, 0, +1}
  3. Fractional differencing (close, volume) → adds *_fdiff columns
  4. Feature engineering               → numeric matrix, all bar columns kept
  5. Align labels ↔ features           → (X, y) same index
  6. Sample uniqueness weights         → float32 array
  7. Walk-forward validation           → fold_metrics, predictions
  8. Export signals                    → CSV for VBT backtest

Research question:
  Tier 1 (baseline):  calendar bars    — OHLCV only
  Tier 2:             minute-source    — OHLCV + vwap + tick_count + duration
  Tier 3:             tick-source      — Tier 2 + buy_sell_imbalance (order flow)

Each tier uses ALL its naturally available features.
The comparison is fair because calendar bars simply do not
have microstructure columns — this is their structural limitation,
not an artificial restriction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_module.labeling.triple_barrier import label_bars
from ml_module.features.fractional_diff import find_min_d, frac_diff
from ml_module.features.engineer import engineer
from ml_module.features.sample_weights import compute_sample_weights
from ml_module.model.lgbm import model_factory
from ml_module.validation.walk_forward import WalkForward
from ml_module.bridge.signal_exporter import export_signals
from ml_module.bridge.metrics import summarise_walk_forward
from ml_module.utils.helpers import get_logger, save_json, Timer

logger = get_logger(__name__)

# Tick-source bar types — need timestamp rounding for 1m OHLCV alignment
_TICK_TYPES = {
    "dollar_tick", "volume_tick", "volatility_tick",
    "hybrid_tick", "range_tick", "renko_tick",
}


class Pipeline:
    """
    Parameters
    ----------
    cfg : Loaded ml_config.yaml dict.
    """

    def __init__(self, cfg: Dict):
        self.cfg        = cfg
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run_all(self) -> Dict[str, Any]:
        """Run pipeline for every bar type in config."""
        results: Dict[str, Any] = {}
        data_cfg = self.cfg["data"]
        bar_dir  = Path(data_cfg["bar_data_dir"])
        asset    = data_cfg.get("asset", "btc")

        for bar_type, filename in data_cfg["bar_files"].items():
            path = bar_dir / filename
            if not path.exists():
                logger.warning("File not found: %s — skipping.", path)
                continue
            logger.info("\n%s\n  %s\n%s", "="*60, bar_type, "="*60)
            try:
                results[bar_type] = self.run_one(str(path), bar_type, asset)
            except Exception as e:
                logger.error("%s failed: %s", bar_type, e, exc_info=True)
                results[bar_type] = {"error": str(e)}

        save_json(results, self.output_dir / "pipeline_summary.json")
        return results

    def run_one(
        self,
        bar_csv:  str,
        bar_type: str,
        asset:    str = "btc",
    ) -> Dict[str, Any]:
        """Run full pipeline for one bar CSV."""
        result = {"bar_type": bar_type}
        cfg    = self.cfg

        # 1. Load
        with Timer() as t:
            df = self._load(bar_csv)
        logger.info("[%s] Loaded %d bars (%.2fs)", bar_type, len(df), t.elapsed)
        if len(df) < cfg.get("min_bars", 150):
            raise ValueError(f"Only {len(df)} bars — need {cfg.get('min_bars',150)}")
        result["n_bars"] = len(df)

        # 2. Label
        with Timer() as t:
            lab_cfg = cfg["labeling"]
            y_all   = label_bars(
                df,
                profit_target    = lab_cfg["profit_target"],
                stop_loss        = lab_cfg["stop_loss"],
                max_holding_bars = lab_cfg["max_holding_bars"],
                vol_lookback     = lab_cfg.get("vol_lookback", 10),
            )
        vc = y_all.value_counts().sort_index()
        logger.info("[%s] Labels (%.2fs): %s",
                    bar_type, t.elapsed,
                    {int(k): int(v) for k, v in vc.items()})
        result["label_counts"] = {int(k): int(v) for k, v in vc.items()}

        # 3. Fractional differencing
        with Timer() as t:
            fd_cfg   = cfg.get("fractional_diff", {})
            d_c, cfd = find_min_d(df["close"],
                                  d_min  = fd_cfg.get("d_min",  0.1),
                                  d_max  = fd_cfg.get("d_max",  1.0),
                                  d_step = fd_cfg.get("d_step", 0.1),
                                  adf_sig= fd_cfg.get("adf_sig", 0.05))
            d_v = d_c  # reuse same d for volume (conservative)
            vfd  = frac_diff(df["volume"], d_v) if "volume" in df.columns else None
        logger.info("[%s] Frac-diff (%.2fs): d_close=%.1f", bar_type, t.elapsed, d_c)
        result["frac_diff_d"] = d_c

        # 4. Features
        with Timer() as t:
            X_all = engineer(df, close_fdiff=cfd, volume_fdiff=vfd)
        logger.info("[%s] Features (%.2fs): shape=%s", bar_type, t.elapsed, X_all.shape)
        result["n_features"] = X_all.shape[1]

        # 5. Align
        combined = X_all.join(y_all.rename("label"), how="inner").dropna(subset=["label"])
        y = combined.pop("label").astype(np.int8)
        X = combined
        logger.info("[%s] Aligned: %d samples", bar_type, len(X))
        result["n_aligned"] = len(X)

        if len(X) < cfg.get("min_bars", 150):
            raise ValueError(f"Only {len(X)} aligned samples")

        # 6. Sample weights
        max_hold = lab_cfg["max_holding_bars"]
        w = compute_sample_weights(len(X), max_holding_bars=max_hold)

        # 7. Walk-forward
        wf_cfg = cfg["walk_forward"]
        wf     = WalkForward(
            initial_train_bars   = wf_cfg.get("initial_train_bars",   200),
            step_bars            = wf_cfg.get("step_bars",            100),
            min_test_bars        = wf_cfg.get("min_test_bars",         20),
            embargo_bars         = wf_cfg.get("embargo_bars",           5),
            confidence_threshold = wf_cfg.get("confidence_threshold",  0.0),
        )
        model_params = cfg["model"]["params"]
        fac = model_factory(model_params)

        with Timer() as t:
            fold_metrics, predictions = wf.run(X, y, fac, sample_weights=w)
        wf_sum = summarise_walk_forward(fold_metrics)
        logger.info(
            "[%s] Walk-forward (%.2fs): folds=%d acc=%.3f mcc=%+.3f",
            bar_type, t.elapsed,
            wf_sum.get("n_folds", 0),
            wf_sum.get("accuracy_mean", float("nan")),
            wf_sum.get("mcc_mean", float("nan")),
        )
        result["wf_summary"]   = wf_sum
        result["fold_metrics"] = fold_metrics

        # 8. Export signals
        sig_df = export_signals(
            predictions    = predictions,
            bar_df         = df,
            asset          = asset,
            bar_type       = bar_type,
            output_dir     = self.output_dir,
            is_tick_source = bar_type in _TICK_TYPES,
        )
        result["signal_counts"] = {
            "buy":  int((sig_df["signals"] ==  1).sum()),
            "sell": int((sig_df["signals"] == -1).sum()),
            "hold": int((sig_df["signals"] ==  0).sum()),
        }

        # Save per-bar-type diagnostics
        save_json(result, self.output_dir / f"diagnostics_{bar_type}.json")
        return result

    # ------------------------------------------------------------------
    def _load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        dt_col = self.cfg["data"].get("datetime_col", "datetime")
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
        df = df.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
        return df
