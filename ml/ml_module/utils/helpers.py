"""
ml_module/utils/helpers.py
--------------------------
Shared utilities: structured logging, YAML config loading,
model serialization, and research metrics.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a module-level logger with consistent formatting.

    Parameters
    ----------
    name     : Module ``__name__`` or any identifier.
    level    : Logging level string ("DEBUG", "INFO", …).
    log_file : Optional path; if set, logs are also written to this file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file and return as nested dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* into *base*.
    Override values take precedence; nested dicts are merged, not replaced.
    """
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str | Path) -> None:
    """Pickle a model object to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str | Path) -> Any:
    """Load a pickled model from *path*."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_json(data: Any, path: str | Path) -> None:
    """Save *data* as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Return a rich classification metrics dict suitable for research reporting.

    Includes: accuracy, per-class precision/recall/f1, macro/weighted averages,
    confusion matrix, and Matthews correlation coefficient.
    """
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        matthews_corrcoef, cohen_kappa_score,
    )

    labels = sorted(set(y_true) | set(y_pred))
    label_names = {1: "BUY", 0: "HOLD", -1: "SELL"}
    target_names = [label_names.get(l, str(l)) for l in labels]

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "mcc":              float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa":      float(cohen_kappa_score(y_true, y_pred)),
        "per_class":        report,
        "confusion_matrix": cm.tolist(),
        "label_order":      labels,
    }


def walk_forward_summary(wf_results: list[Dict]) -> Dict[str, Any]:
    """
    Aggregate walk-forward fold metrics into summary statistics.

    Parameters
    ----------
    wf_results : List of per-fold metric dicts (from WalkForwardValidator).
    """
    if not wf_results:
        return {}

    accs  = [r["accuracy"] for r in wf_results if "accuracy" in r]
    mccs  = [r["mcc"]      for r in wf_results if "mcc"      in r]
    kappas = [r["cohen_kappa"] for r in wf_results if "cohen_kappa" in r]

    return {
        "n_folds":       len(wf_results),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std":  float(np.std(accs)),
        "mcc_mean":      float(np.mean(mccs)),
        "mcc_std":       float(np.std(mccs)),
        "kappa_mean":    float(np.mean(kappas)),
        "kappa_std":     float(np.std(kappas)),
        "fold_results":  wf_results,
    }


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

class Timer:
    """Context-manager wall-clock timer."""
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t0

    def __str__(self):
        return f"{self.elapsed:.2f}s"
