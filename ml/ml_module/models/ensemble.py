"""
ml_module/models/ensemble.py
------------------------------
Stacking Meta-Ensemble for directional bar prediction.

Architecture
============

Level-0  — Three specialist primary learners, each trained on the full
           feature matrix but optimised for a different sub-task:

  direction_model   (LightGBM classifier)
    → Outputs class probabilities P(BUY), P(HOLD), P(SELL) per bar.
    → Specialises in capturing regime-specific directional patterns.

  confidence_model  (XGBoost regressor)
    → Outputs predicted |log-return| magnitude for the next bar.
    → Acts as a "conviction filter": high confidence → stronger signal.

  regime_model      (RandomForest classifier)
    → Trained on regime scores (score_bull, score_bear, score_transition…)
    → Outputs market-state probabilities that the meta-learner uses to
      weight the direction_model outputs per regime.

Level-1  — Meta-learner (Logistic Regression)
    → Receives all Level-0 outputs as input features.
    → Learns which primary model to trust, and when.
    → Outputs final 3-class label: BUY (+1), HOLD (0), SELL (-1).

Why this design?
----------------
- Each specialist sees the same raw features but is evaluated on a
  different loss → diversity of inductive bias reduces ensemble variance.
- The meta-learner is intentionally simple (LogReg) to avoid overfitting
  the stacking layer on small financial datasets.
- Swapping any component requires only a change to the YAML config.

Extension guide (for future researchers)
-----------------------------------------
To add a new primary learner:
  1. Add a new entry under ``ensemble.primary_learners`` in ml_config.yaml.
  2. Give it a ``role`` and a ``type`` from MODEL_REGISTRY.
  3. No Python changes needed.

To change the meta-learner:
  Change ``ensemble.meta_learner.type`` in ml_config.yaml.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_module.utils.helpers import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model registry  — maps config type strings to constructor callables
# ---------------------------------------------------------------------------

def _lgbm_clf(params: Dict) -> Any:
    from lightgbm import LGBMClassifier
    p = {k: v for k, v in params.items() if k != "verbose"}
    return LGBMClassifier(**p, verbose=-1)


def _xgb_clf(params: Dict) -> Any:
    from xgboost import XGBClassifier
    return XGBClassifier(**params, eval_metric="mlogloss", verbosity=0)


def _rf_clf(params: Dict) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**params)


def _lgbm_reg(params: Dict) -> Any:
    from lightgbm import LGBMRegressor
    p = {k: v for k, v in params.items() if k != "verbose"}
    return LGBMRegressor(**p, verbose=-1)


def _xgb_reg(params: Dict) -> Any:
    from xgboost import XGBRegressor
    return XGBRegressor(**params, verbosity=0)


def _rf_reg(params: Dict) -> Any:
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**params)


def _logreg(params: Dict) -> Any:
    from sklearn.linear_model import LogisticRegression
    # multi_class param was removed in sklearn 1.5; strip it for forward compat
    p = {k: v for k, v in params.items() if k != 'multi_class'}
    return LogisticRegression(**p)


MODEL_REGISTRY: Dict[str, Callable[[Dict], Any]] = {
    "lightgbm_clf": _lgbm_clf,
    "xgboost_clf":  _xgb_clf,
    "rf_clf":       _rf_clf,
    "lightgbm_reg": _lgbm_reg,
    "xgboost_reg":  _xgb_reg,
    "rf_reg":       _rf_reg,
    "logreg":       _logreg,
}


def build_model(type_str: str, params: Dict) -> Any:
    """Instantiate a model from the registry."""
    if type_str not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{type_str}'. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[type_str](params)


# ---------------------------------------------------------------------------
# Primary learner wrapper
# ---------------------------------------------------------------------------

class PrimaryLearner:
    """
    Wraps a single Level-0 model with its role metadata.

    Parameters
    ----------
    name   : Unique identifier (e.g. "direction_model").
    role   : One of {"direction", "confidence", "regime"}.
    model  : Unfitted sklearn-compatible estimator.
    """

    def __init__(self, name: str, role: str, model: Any):
        self.name  = name
        self.role  = role
        self.model = model
        self._is_regressor = role == "confidence"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrimaryLearner":
        """Fit the underlying model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        return self

    def predict_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate the features that the meta-learner will use.

        - Classifiers → class probabilities (n_samples × n_classes).
        - Regressors  → scalar prediction (n_samples × 1).
        """
        if self._is_regressor:
            preds = self.model.predict(X)
            return preds.reshape(-1, 1)
        else:
            return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard class prediction (for inspection)."""
        if self._is_regressor:
            return self.model.predict(X)
        return self.model.predict(X)

    def feature_importances(self) -> Optional[np.ndarray]:
        return getattr(self.model, "feature_importances_", None)


# ---------------------------------------------------------------------------
# Meta-Ensemble (stacking)
# ---------------------------------------------------------------------------

class MetaEnsemble:
    """
    Two-level stacking ensemble.

    Level-0: List of PrimaryLearner objects.
    Level-1: A meta-learner trained on Level-0 outputs.

    Training protocol
    -----------------
    1. Split training data into L0-train (1 - meta_train_fraction) and
       meta-train (meta_train_fraction).
    2. Fit each primary learner on L0-train.
    3. Generate primary predictions on meta-train.
    4. Fit meta-learner on (primary predictions, meta-train labels).

    This avoids training-set contamination of the meta-learner while
    keeping the implementation simple.  For larger datasets, k-fold
    out-of-fold stacking is preferred and can be enabled in a future version
    by setting ``use_oof_stacking=True``.

    Parameters
    ----------
    primary_learners     : List of PrimaryLearner objects.
    meta_learner         : sklearn-compatible classifier for Level-1.
    meta_train_fraction  : Fraction of training data reserved for meta-training.
    scaler               : Optional StandardScaler applied to meta-features.
    """

    def __init__(
        self,
        primary_learners:    List[PrimaryLearner],
        meta_learner:        Any,
        meta_train_fraction: float = 0.3,
        scaler:              Optional[StandardScaler] = None,
    ):
        self.primary_learners    = primary_learners
        self.meta_learner        = meta_learner
        self.meta_train_fraction = meta_train_fraction
        self.scaler              = scaler or StandardScaler()
        self.classes_:           Optional[np.ndarray] = None
        self._fitted             = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MetaEnsemble":
        """
        Train all primary learners, then train the meta-learner.

        Parameters
        ----------
        X : Feature matrix (n_samples × n_features).
        y : Integer labels in {-1, 0, 1}.
        """
        n = len(X)
        split = int(n * (1.0 - self.meta_train_fraction))
        split = max(split, 50)  # guarantee at least 50 samples for L0

        X_l0, y_l0 = X[:split], y[:split]
        X_meta, y_meta = X[split:], y[split:]

        if len(X_meta) < 10:
            logger.warning(
                "Meta-train set has only %d samples; increasing meta_train_fraction "
                "or providing more data is recommended.", len(X_meta),
            )

        logger.info(
            "Fitting %d primary learners on %d samples; meta-train on %d samples",
            len(self.primary_learners), len(X_l0), len(X_meta),
        )

        # --- Level-0 training ---
        for learner in self.primary_learners:
            y_l0_role = self._prepare_target(y_l0, learner.role)
            logger.info("  Fitting primary learner '%s' (role=%s) …", learner.name, learner.role)
            learner.fit(X_l0, y_l0_role)

        # --- Meta-feature generation ---
        meta_features = self._build_meta_features(X_meta)
        meta_features_scaled = self.scaler.fit_transform(meta_features)

        # --- Level-1 training ---
        logger.info("Fitting meta-learner on %d meta-features …", meta_features_scaled.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.meta_learner.fit(meta_features_scaled, y_meta)

        self.classes_ = np.unique(y)
        self._fitted  = True
        logger.info("MetaEnsemble training complete.")
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate final class predictions.

        Returns
        -------
        1-D array of int labels in {-1, 0, 1}.
        """
        self._check_fitted()
        meta_features = self._build_meta_features(X)
        meta_scaled   = self.scaler.transform(meta_features)
        return self.meta_learner.predict(meta_scaled).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return meta-learner class probabilities (n_samples × n_classes).
        """
        self._check_fitted()
        meta_features = self._build_meta_features(X)
        meta_scaled   = self.scaler.transform(meta_features)
        if hasattr(self.meta_learner, "predict_proba"):
            return self.meta_learner.predict_proba(meta_scaled)
        # Fall back to hard predictions as one-hot
        preds = self.predict(X)
        n_cls = len(self.classes_)
        out = np.zeros((len(preds), n_cls))
        for i, p in enumerate(preds):
            j = np.searchsorted(self.classes_, p)
            out[i, j] = 1.0
        return out

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def primary_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return raw primary learner predictions for interpretability."""
        self._check_fitted()
        return {lrn.name: lrn.predict(X) for lrn in self.primary_learners}

    def meta_feature_names(self) -> List[str]:
        """Column names of the meta-feature matrix (for research documentation)."""
        names = []
        for lrn in self.primary_learners:
            if lrn.role == "confidence":
                names.append(f"{lrn.name}_pred")
            else:
                # We don't know n_classes until after fitting; approximate
                n_out = getattr(lrn.model, "n_classes_", 3)
                names += [f"{lrn.name}_p{i}" for i in range(n_out)]
        return names

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_meta_features(self, X: np.ndarray) -> np.ndarray:
        parts = [lrn.predict_meta_features(X) for lrn in self.primary_learners]
        return np.hstack(parts)

    @staticmethod
    def _prepare_target(y: np.ndarray, role: str) -> np.ndarray:
        """
        Transform labels for each primary learner's loss function.

        direction  → original {-1, 0, 1} labels
        confidence → |log-return| proxy  (here: magnitude of label * 1, a
                     placeholder until actual return data is passed — see
                     fit_with_returns() for the production version)
        regime     → binarise as 0=bearish/hold, 1=bullish  (for regime clf)
        """
        if role == "direction":
            return y
        elif role == "confidence":
            # Use |label| as a signal-strength proxy (0 = HOLD = low conf)
            return np.abs(y).astype(np.float64)
        elif role == "regime":
            # Consecutive-label consistency over last 5 non-HOLD bars.
            # 1 = trending  (majority of recent bars same direction)
            # 0 = ranging   (mixed directions or HOLD)
            # Genuinely different from confidence target (|label|).
            n      = len(y)
            regime = np.zeros(n, dtype=np.int32)
            window = 5
            for i in range(n):
                recent = [y[j] for j in range(max(0, i - window), i) if y[j] != 0]
                if len(recent) < 2:
                    regime[i] = 1
                else:
                    most_recent = recent[0]
                    same = sum(1 for v in recent if v == most_recent)
                    regime[i] = 1 if same > len(recent) / 2 else 0
            return regime
        return y

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("MetaEnsemble must be fitted before calling predict()")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ensemble(cfg: Dict) -> MetaEnsemble:
    """
    Instantiate a MetaEnsemble from the ``ensemble`` config section.

    Parameters
    ----------
    cfg : ``ensemble`` section of ml_config.yaml.
    """
    primary_cfg = cfg.get("primary_learners", {})
    learners = []
    for name, spec in primary_cfg.items():
        role   = spec.get("role", "direction")
        mtype  = spec.get("type", "lightgbm_clf")
        params = spec.get("params", {})
        model  = build_model(mtype, params)
        learners.append(PrimaryLearner(name=name, role=role, model=model))
        logger.info("Registered primary learner '%s' (%s, role=%s)", name, mtype, role)

    meta_cfg  = cfg.get("meta_learner", {})
    meta_type = meta_cfg.get("type", "logreg")
    meta_p    = meta_cfg.get("params", {})
    meta_mdl  = build_model(meta_type, meta_p)
    logger.info("Meta-learner: %s", meta_type)

    return MetaEnsemble(
        primary_learners    = learners,
        meta_learner        = meta_mdl,
        meta_train_fraction = cfg.get("meta_train_fraction", 0.3),
    )
