"""
ml_module/pipeline.py
----------------------
Central orchestrator that wires all components together.

Flow per bar type
-----------------
1.  Load bar CSV                    → pd.DataFrame
2.  Generate triple-barrier labels  → pd.Series
3.  Apply fractional differencing   → DataFrame (with *_fdiff cols)
4.  Engineer features               → numeric feature matrix
5.  Align labels ↔ features         → (X, y)
6.  CPCV cross-validation           → per-fold metrics
7.  Walk-forward validation         → out-of-sample predictions
8.  Export signals                  → CSV for backtest bridge
9.  Save diagnostics & metrics      → JSON / CSV

All components are swappable via config — no subclassing required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_module.utils.helpers import (
    get_logger, load_config, save_json,
    classification_report_dict, walk_forward_summary, Timer,
)
from ml_module.labeling.triple_barrier import build_labeler
from ml_module.features.fractional_diff import build_differencer
from ml_module.features.feature_engineer import build_feature_engineer, load_bar_csv
from ml_module.validation.cpcv import build_cpcv
from ml_module.validation.walk_forward import build_walk_forward
from ml_module.models.ensemble import build_ensemble, MetaEnsemble
from ml_module.backtest_bridge.signal_exporter import build_exporter

logger = get_logger(__name__)


class MLPipeline:
    """
    Production ML pipeline for high-frequency bar analysis.

    Parameters
    ----------
    cfg : Fully-loaded config dict (from ml_config.yaml).

    Usage
    -----
        pipeline = MLPipeline.from_config("ml_module/config/ml_config.yaml")
        results = pipeline.run_all()
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        log_cfg = cfg.get("logging", {})
        self.logger = get_logger(
            __name__,
            level    = log_cfg.get("level", "INFO"),
            log_file = log_cfg.get("log_file", None),
        )
        self.output_dir = Path(cfg.get("backtest", {}).get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Constructor helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "MLPipeline":
        """Load config from YAML and return a ready pipeline."""
        cfg = load_config(config_path)
        return cls(cfg)

    # ------------------------------------------------------------------
    # Top-level entry points
    # ------------------------------------------------------------------

    def run_all(self) -> Dict[str, Any]:
        """
        Run the full pipeline for every bar type defined in the config.

        Returns
        -------
        Dict mapping bar_type → per-bar-type result dict.
        """
        data_cfg   = self.cfg["data"]
        bar_files  = data_cfg["bar_files"]
        bar_dir    = Path(data_cfg["bar_data_dir"])
        min_bars   = data_cfg.get("min_bars", 150)

        all_results: Dict[str, Any] = {}

        for bar_type, filename in bar_files.items():
            csv_path = bar_dir / filename
            if not csv_path.exists():
                self.logger.warning("Bar file not found: %s — skipping.", csv_path)
                continue

            self.logger.info("\n%s\n  Processing bar type: %s\n%s", "="*60, bar_type, "="*60)
            try:
                result = self.run(
                    bar_csv  = str(csv_path),
                    bar_type = bar_type,
                    min_bars = min_bars,
                )
                all_results[bar_type] = result
            except Exception as e:
                self.logger.error("Pipeline failed for bar_type=%s: %s", bar_type, e, exc_info=True)
                all_results[bar_type] = {"error": str(e)}

        # Save consolidated summary
        summary_path = self.output_dir / "pipeline_summary.json"
        save_json({k: self._jsonify(v) for k, v in all_results.items()}, summary_path)
        self.logger.info("\nAll bar types complete. Summary → %s", summary_path)
        return all_results

    def run(
        self,
        bar_csv:  str,
        bar_type: str = "unknown",
        min_bars: int = 150,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline for a single bar CSV.

        Parameters
        ----------
        bar_csv  : Path to the pre-built bar CSV.
        bar_type : Label used in output filenames and logs.
        min_bars : Skip if the CSV contains fewer bars.

        Returns
        -------
        Dict with keys: bar_type, n_bars, labels, cpcv_metrics,
                        wf_summary, signal_summary, stationarity_report.
        """
        t_total = Timer()
        result  = {"bar_type": bar_type}

        with t_total:
            # --- 1. Load ---
            bar_df = self._load(bar_csv, bar_type, min_bars)
            if bar_df is None:
                return {**result, "error": "Insufficient data"}

            result["n_bars"] = len(bar_df)

            # --- 2. Label ---
            with Timer() as t:
                labels = self._label(bar_df)
            self.logger.info("[%s] Labeling: %.2fs", bar_type, t.elapsed)
            result["label_counts"] = labels.value_counts().to_dict()

            # --- 3. Fractional Differencing ---
            with Timer() as t:
                bar_df_diff, diff_report = self._diff(bar_df)
            self.logger.info("[%s] Frac-diff: %.2fs", bar_type, t.elapsed)
            result["stationarity_report"] = diff_report.to_dict(orient="records")

            # --- 4. Feature Engineering ---
            with Timer() as t:
                X_raw = self._features(bar_df_diff)
            self.logger.info("[%s] Features: %.2fs  shape=%s", bar_type, t.elapsed, X_raw.shape)

            # --- 5. Align labels ↔ features ---
            X, y = self._align(X_raw, labels)
            if len(X) < min_bars:
                self.logger.warning("[%s] Only %d aligned samples — skipping.", bar_type, len(X))
                return {**result, "error": "Insufficient aligned samples"}

            result["n_aligned"] = len(X)

            # --- 6. CPCV Cross-Validation ---
            with Timer() as t:
                cpcv_metrics = self._cpcv_eval(X, y, bar_type)
            self.logger.info("[%s] CPCV: %.2fs  folds=%d", bar_type, t.elapsed, len(cpcv_metrics))
            result["cpcv_metrics"] = cpcv_metrics

            # --- 7. Walk-Forward Validation ---
            with Timer() as t:
                wf_summary, predictions = self._walk_forward(X, y, bar_type)
            self.logger.info("[%s] Walk-forward: %.2fs", bar_type, t.elapsed)
            result["wf_summary"] = wf_summary

            # --- 8. Export Signals ---
            with Timer() as t:
                sig_df = self._export_signals(predictions, bar_df, bar_type)
            result["signal_summary"] = self._exporter.signal_summary(sig_df)

            # --- 9. Save per-bar diagnostics ---
            self._save_diagnostics(result, bar_type)

        result["total_time_s"] = round(t_total.elapsed, 2)
        self.logger.info(
            "[%s] DONE in %.1fs | acc=%.3f | mcc=%.3f",
            bar_type,
            t_total.elapsed,
            wf_summary.get("accuracy_mean", float("nan")),
            wf_summary.get("mcc_mean", float("nan")),
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _load(
        self, bar_csv: str, bar_type: str, min_bars: int
    ) -> Optional[pd.DataFrame]:
        datetime_col = self.cfg["data"].get("datetime_col", "datetime")
        df = load_bar_csv(bar_csv, datetime_col=datetime_col)
        self.logger.info("[%s] Loaded %d bars from %s", bar_type, len(df), Path(bar_csv).name)
        if len(df) < min_bars:
            self.logger.warning(
                "[%s] Only %d bars (min=%d) — skipping.", bar_type, len(df), min_bars
            )
            return None
        return df

    def _label(self, bar_df: pd.DataFrame) -> pd.Series:
        labeler = build_labeler(self.cfg["labeling"])
        return labeler.fit_transform(bar_df)

    def _diff(self, bar_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        differencer = build_differencer(self.cfg["fractional_diff"])
        df_diff     = differencer.fit_transform(bar_df)
        report      = differencer.stationarity_report(df_diff)
        return df_diff, report

    def _features(self, bar_df: pd.DataFrame) -> pd.DataFrame:
        fe = build_feature_engineer(self.cfg["features"])
        return fe.fit_transform(bar_df)

    def _align(
        self, X_raw: pd.DataFrame, labels: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Inner-join features and labels on their shared index.
        Both may have different lengths due to NaN dropping in feature engineering.
        """
        X = X_raw.copy()
        y = labels.rename("label")
        combined = X.join(y, how="inner")
        combined.dropna(subset=["label"], inplace=True)
        y_aligned = combined.pop("label").astype(np.int8)
        return combined, y_aligned

    def _cpcv_eval(
        self, X: pd.DataFrame, y: pd.Series, bar_type: str
    ) -> List[Dict]:
        """Run CPCV folds and return per-fold metrics."""
        cpcv_cfg = self.cfg["cpcv"]
        max_hold = self.cfg["labeling"].get("max_holding_bars", 20)
        splitter = build_cpcv(cpcv_cfg, max_holding_bars=max_hold)

        fold_metrics = []
        X_arr = X.to_numpy(dtype=np.float32)
        y_arr = y.to_numpy()

        for fold_i, (train_pos, test_pos) in enumerate(splitter.split(X)):
            X_tr, y_tr = X_arr[train_pos], y_arr[train_pos]
            X_te, y_te = X_arr[test_pos],  y_arr[test_pos]

            if len(np.unique(y_tr)) < 2:
                continue

            try:
                ensemble = build_ensemble(self.cfg["ensemble"])
                ensemble.fit(X_tr, y_tr)
                preds    = ensemble.predict(X_te)
                metrics  = classification_report_dict(y_te, preds)
                metrics["fold"]       = fold_i
                metrics["train_size"] = len(train_pos)
                metrics["test_size"]  = len(test_pos)
                fold_metrics.append(metrics)
            except Exception as e:
                self.logger.error("CPCV fold %d failed: %s", fold_i, e)

        return fold_metrics

    def _walk_forward(
        self, X: pd.DataFrame, y: pd.Series, bar_type: str
    ) -> Tuple[Dict, pd.Series]:
        """Run expanding-window walk-forward validation."""
        wf = build_walk_forward(self.cfg["walk_forward"])

        X_arr = X.to_numpy(dtype=np.float32)
        y_arr = y.to_numpy()
        y_series = pd.Series(y_arr, index=y.index)

        def model_factory(X_train, y_train):
            ens = build_ensemble(self.cfg["ensemble"])
            ens.fit(X_train, y_train)
            # Wrap to match walk_forward API (predict from numpy)
            return _NumpyModelWrapper(ens)

        fold_metrics, all_preds = wf.validate(
            X         = pd.DataFrame(X_arr, index=X.index),
            y         = y_series,
            model_factory = model_factory,
        )
        summary = walk_forward_summary(fold_metrics)

        # Save walk-forward fold predictions
        preds_path = self.output_dir / f"wf_predictions_{bar_type}.csv"
        all_preds.to_csv(preds_path, header=True)

        return summary, all_preds

    def _export_signals(
        self,
        predictions: pd.Series,
        bar_df:      pd.DataFrame,
        bar_type:    str,
    ) -> pd.DataFrame:
        self._exporter = build_exporter(self.cfg["backtest"])
        asset = self.cfg["data"].get("asset", "unknown")
        return self._exporter.export(
            predictions = predictions,
            bar_df      = bar_df,
            bar_type    = bar_type,
            asset       = asset,
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _save_diagnostics(self, result: Dict, bar_type: str) -> None:
        path = self.output_dir / f"diagnostics_{bar_type}.json"
        save_json(self._jsonify(result), path)
        self.logger.debug("[%s] Diagnostics → %s", bar_type, path)

    @staticmethod
    def _jsonify(obj: Any) -> Any:
        """Recursively convert numpy types for JSON serialisation."""
        if isinstance(obj, dict):
            return {k: MLPipeline._jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MLPipeline._jsonify(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

class _NumpyModelWrapper:
    """Thin wrapper so MetaEnsemble.predict() accepts numpy arrays from WalkForward."""

    def __init__(self, ensemble: MetaEnsemble):
        self._ens = ensemble

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)
        return self._ens.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)
        return self._ens.predict_proba(X)
