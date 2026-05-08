"""
model/lgbm.py
-------------
Single LightGBM directional classifier.

One well-regularised LightGBM on all available features with sample
weights outperforms a stacked ensemble on small financial datasets
because:
  - No meta-train split wasting 30% of already small training data
  - No diversity theater when all sub-models see same features
  - LightGBM natively handles 40-70 mixed features well

Trained with class_weight='balanced' + De Prado sample weights.
Returns predict_proba so walk-forward can apply confidence filter.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict


class DirectionalModel:
    """LightGBM multiclass directional predictor {-1, 0, +1}."""

    def __init__(self, **kwargs):
        defaults = dict(
            n_estimators      = 500,
            learning_rate     = 0.02,
            num_leaves        = 31,
            min_child_samples = 20,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.1,
            reg_lambda        = 1.0,
            class_weight      = "balanced",
            random_state      = 42,
            n_jobs            = -1,
            verbose           = -1,
        )
        defaults.update(kwargs)
        self._params = defaults
        self._clf    = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "DirectionalModel":
        from lightgbm import LGBMClassifier
        self._clf = LGBMClassifier(**self._params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._clf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        X = _to_np(X)
        return self._clf.predict(X).astype(np.int8)

    def predict_proba(self, X) -> np.ndarray:
        X = _to_np(X)
        return self._clf.predict_proba(X)

    @property
    def feature_importances_(self):
        return self._clf.feature_importances_


def _to_np(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def model_factory(params: Dict):
    """Returns a callable(X_tr, y_tr, w_tr) -> fitted DirectionalModel."""
    def _factory(X_tr, y_tr, w_tr=None):
        m = DirectionalModel(**params)
        m.fit(_to_np(X_tr),
              y_tr.to_numpy() if hasattr(y_tr, "to_numpy") else np.array(y_tr),
              sample_weight=w_tr)
        return m
    return _factory
