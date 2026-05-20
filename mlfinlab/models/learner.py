"""
mlfinlab/models/learner.py
==========================
Classifier training and evaluation for the bar-comparison study.

Three classifiers are evaluated per bar type:
  RandomForest    -- primary choice from de Prado (AFML Ch.6)
                     handles non-linearity, gives feature importance (MDI)
  GradientBoost   -- strong competitor, often outperforms RF on tabular data
  SVM             -- classical benchmark with RBF kernel

Training protocol
-----------------
  Walk-forward CV with purging and embargo (WalkForwardCV in cv.py).
  Sample weights passed to fit() so overlapping events contribute less.
  Probability calibration (isotonic regression) applied after fitting.

Outputs per classifier per bar type
-------------------------------------
  predictions DataFrame : event_time | y_true | y_pred | prob_m1 | prob_1
  cv_scores dict        : accuracy, f1_weighted per fold
  feature_importance    : MDI from RandomForest (MDA computed separately)

All outputs are returned to stage_models in main.py which saves them.

References
----------
de Prado, M. L. (2018). Advances in Financial Machine Learning, Ch.6-8.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
)
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from mlfinlab.models.cv import WalkForwardCV

log = logging.getLogger("mlfinlab.models")

# Columns that are metadata, not features
NON_FEATURE_COLS = {
    "ret", "bin", "weight", "bar_type", "source",
    "bar_class", "min_d", "feature_mode", "t1_touch",
}


class XGBWrapper(BaseEstimator, ClassifierMixin):
    """Wraps XGBClassifier to remap labels -1/+1 -> 0/1 for XGBoost compatibility."""

    def __init__(self, xgb_clf):
        self.xgb_clf   = xgb_clf
        self.classes_  = None
        self._label_map = {}
        self._inv_map   = {}

    def fit(self, X, y, sample_weight=None):
        classes = sorted(np.unique(y))
        self.classes_   = np.array(classes)
        self._label_map = {c: i for i, c in enumerate(classes)}
        self._inv_map   = {i: c for i, c in enumerate(classes)}
        y_mapped = np.array([self._label_map[v] for v in y])
        kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.xgb_clf.fit(X, y_mapped, **kw)
        return self

    def predict(self, X):
        y_mapped = self.xgb_clf.predict(X)
        return np.array([self._inv_map[v] for v in y_mapped])

    def predict_proba(self, X):
        return self.xgb_clf.predict_proba(X)

    def get_params(self, deep=True):
        return {"xgb_clf": self.xgb_clf}

    def set_params(self, **params):
        if "xgb_clf" in params:
            self.xgb_clf = params["xgb_clf"]
        return self



# ---------------------------------------------------------------------------
# Build classifiers
# ---------------------------------------------------------------------------

def _get_classifiers(random_state: int = 42) -> dict:
    """Return the three classifiers used in the study.

    RandomForest and GradientBoost are wrapped in isotonic calibration
    so that predict_proba outputs well-calibrated probabilities suitable
    for bet sizing in Stage 4.

    SVM requires StandardScaler (distance-based, sensitive to feature scale).
    """
    rf = RandomForestClassifier(
        n_estimators   = 500,
        max_depth      = None,      # grow full trees, rely on ensemble averaging
        min_samples_leaf = 5,       # prevent single-sample leaves
        max_features   = "sqrt",    # standard RF feature subsampling
        class_weight   = "balanced",# handles label imbalance (+1 vs -1)
        n_jobs         = -1,
        random_state   = random_state,
    )

    gb = GradientBoostingClassifier(
        n_estimators   = 300,
        max_depth      = 3,         # shallow trees prevent overfitting
        learning_rate  = 0.05,
        subsample      = 0.8,       # row subsampling like RF
        max_features   = "sqrt",
        random_state   = random_state,
    )

    svm = Pipeline([
        ("scaler", StandardScaler()),   # SVM needs scaled features
        ("svc",    SVC(
            kernel       = "rbf",
            C            = 1.0,
            gamma        = "scale",
            probability  = True,        # needed for predict_proba
            class_weight = "balanced",
            random_state = random_state,
        )),
    ])

    classifiers = {
        "random_forest"    : CalibratedClassifierCV(rf, method="isotonic", cv=3),
        "gradient_boost"   : CalibratedClassifierCV(gb, method="isotonic", cv=3),
        "svm"              : svm,   # Pipeline handles scaling; SVC has probability=True
    }

    if HAS_XGB:
        # XGBoost requires labels 0,1 not -1,1.
        # XGBWrapper handles remapping and provides predict_proba natively.
        # Do NOT wrap in CalibratedClassifierCV - XGB has built-in probability.
        classifiers["xgboost"] = XGBWrapper(XGBClassifier(
            n_estimators     = 300,
            max_depth        = 3,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            eval_metric      = "logloss",
            random_state     = random_state,
            verbosity        = 0,
        ))

    return classifiers


# ---------------------------------------------------------------------------
# Feature importance (MDI)
# ---------------------------------------------------------------------------

def _mdi_importance(model, feature_names: list) -> pd.Series:
    """Extract Mean Decrease in Impurity from a fitted RandomForest.

    For CalibratedClassifierCV wrapping RF, we access the base estimator.
    Returns NaN series for non-tree models.
    """
    try:
        # CalibratedClassifierCV wraps the base estimator
        if hasattr(model, "estimator"):
            base = model.estimator
        elif hasattr(model, "calibrated_classifiers_"):
            base = model.calibrated_classifiers_[0].estimator
        else:
            base = model

        if hasattr(base, "feature_importances_"):
            return pd.Series(
                base.feature_importances_,
                index=feature_names,
                name="mdi_importance",
            ).sort_values(ascending=False)
    except Exception:
        pass
    return pd.Series(np.nan, index=feature_names, name="mdi_importance")


# ---------------------------------------------------------------------------
# Single classifier: walk-forward CV + final fit
# ---------------------------------------------------------------------------

def _train_one(
    clf_name    : str,
    clf,
    X           : pd.DataFrame,
    y           : pd.Series,
    w           : pd.Series,
    t1          : pd.Series,
    cv          : WalkForwardCV,
    random_state: int,
) -> dict:
    """Train one classifier with walk-forward CV.

    Returns
    -------
    dict with keys:
        predictions     pd.DataFrame  y_true | y_pred | prob_m1 | prob_1
        cv_scores       list of dicts, one per fold
        feature_importance pd.Series
        clf_fitted      fitted classifier (trained on full data)
    """
    classes   = sorted(y.unique())
    all_preds = []
    cv_scores = []

    log.info("    [%s] Walk-forward CV ...", clf_name)

    for fold_idx, (train_pos, test_pos) in enumerate(cv.split(X, t1)):
        if len(train_pos) < 30:
            log.warning("    [%s] Fold %d: only %d train events, skipping",
                        clf_name, fold_idx + 1, len(train_pos))
            continue

        X_tr = X.iloc[train_pos]
        y_tr = y.iloc[train_pos]
        w_tr = w.iloc[train_pos]
        X_te = X.iloc[test_pos]
        y_te = y.iloc[test_pos]

        # Fill any remaining NaN with column median (safety net)
        col_medians = X_tr.median()
        X_tr = X_tr.fillna(col_medians)
        X_te = X_te.fillna(col_medians)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                clf.fit(X_tr, y_tr, **_sample_weight_kwargs(clf, w_tr))
            except Exception as e:
                log.warning("    [%s] Fold %d fit failed: %s", clf_name, fold_idx+1, e)
                continue

        proba      = clf.predict_proba(X_te)
        y_pred     = clf.predict(X_te)
        clf_classes = clf.classes_

        # Build probability columns for -1 and +1
        prob_m1 = proba[:, list(clf_classes).index(-1)] if -1 in clf_classes else np.zeros(len(y_te))
        prob_p1 = proba[:, list(clf_classes).index( 1)] if  1 in clf_classes else np.zeros(len(y_te))

        fold_df = pd.DataFrame({
            "y_true"  : y_te.values,
            "y_pred"  : y_pred,
            "prob_m1" : prob_m1,
            "prob_p1" : prob_p1,
            "fold"    : fold_idx + 1,
        }, index=X_te.index)
        all_preds.append(fold_df)

        # Fold metrics
        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        fold_scores = {"fold": fold_idx + 1, "accuracy": acc, "f1_weighted": f1,
                       "n_train": len(train_pos), "n_test": len(test_pos)}

        # AUC only if both classes present in test fold
        if len(set(y_te)) == 2:
            fold_scores["auc"] = roc_auc_score(y_te, prob_p1)

        cv_scores.append(fold_scores)
        log.info("    [%s] Fold %d: acc=%.3f  f1=%.3f  train=%d  test=%d",
                 clf_name, fold_idx + 1, acc, f1,
                 len(train_pos), len(test_pos))

    if not all_preds:
        log.warning("    [%s] No valid folds produced predictions", clf_name)
        return {}

    predictions = pd.concat(all_preds).sort_index()

    # --- Final fit on full dataset for Stage 4 prediction
    log.info("    [%s] Final fit on full dataset ...", clf_name)
    col_medians = X.median()
    X_clean     = X.fillna(col_medians)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_clean, y, **_sample_weight_kwargs(clf, w))

    return {
        "predictions"        : predictions,
        "cv_scores"          : cv_scores,
        "feature_importance" : _mdi_importance(clf, list(X.columns)),
        "clf_fitted"         : clf,
        "col_medians"        : col_medians,
        "classes"            : classes,
    }


def _sample_weight_kwargs(clf, w: pd.Series) -> dict:
    """Return sample_weight kwarg dict if the classifier supports it."""
    try:
        # Pipeline: pass weight to the last step
        if hasattr(clf, "steps"):
            last_step = clf.steps[-1][0]
            return {f"{last_step}__sample_weight": w.values}
        return {"sample_weight": w.values}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main entry point called by stage_models
# ---------------------------------------------------------------------------

def train_all(
    name               : str,
    ml_frame           : pd.DataFrame,
    meta               : dict,
    random_state       : int   = 42,
    n_splits           : int   = 5,
    embargo_pct        : float = 0.01,
    initial_train_pct  : float = 0.40,
) -> dict:
    """Train all classifiers on ml_frame and return results.

    Parameters
    ----------
    name         : Dataset name (for logging).
    ml_frame     : Output of stage_labeling_features. Must contain
                   feature columns + ret, bin, weight, bar_type,
                   source, bar_class, min_d, feature_mode.
    meta         : Bar metadata dict.
    random_state : Seed for reproducibility.
    n_splits     : Walk-forward CV folds.
    embargo_pct  : Embargo fraction of training set size.

    Returns
    -------
    dict  {
        classifier_name: {
            predictions, cv_scores, feature_importance, clf_fitted
        },
        "summary": pd.DataFrame  one row per classifier with mean CV metrics
    }
    """
    # --- Separate features from metadata
    feature_cols = [c for c in ml_frame.columns if c not in NON_FEATURE_COLS]
    X = ml_frame[feature_cols].copy()
    y = ml_frame["bin"].copy()
    w = ml_frame["weight"].copy()

    # t1 for purging: use actual barrier exit timestamps from ml_frame.
    # t1_touch = the bar where the triple-barrier actually fired.
    # This is the correct time to use for purging because it is exactly
    # the bar whose price determined the label.
    if "t1_touch" in ml_frame.columns:
        t1 = ml_frame["t1_touch"].copy()
        # fallback to 3-day proxy for any NaT values
        fallback = ml_frame.index + pd.Timedelta(days=3)
        t1 = t1.fillna(pd.Series(fallback, index=ml_frame.index))
    else:
        t1 = pd.Series(
            ml_frame.index + pd.Timedelta(days=3),
            index=ml_frame.index,
            name="t1",
        )

    log.info("  Features : %d  Events : %d  Classes : %s",
             X.shape[1], len(y), sorted(y.unique()))
    log.info("  Label distribution : %s", y.value_counts().sort_index().to_dict())

    # Ensure y is integer for sklearn
    y = y.astype(int)

    cv          = WalkForwardCV(n_splits=n_splits, embargo_pct=embargo_pct, initial_train_pct=initial_train_pct)
    classifiers = _get_classifiers(random_state)
    results     = {}

    for clf_name, clf in classifiers.items():
        log.info("  Training [%s] ...", clf_name)
        result = _train_one(clf_name, clf, X, y, w, t1, cv, random_state)
        if result:
            results[clf_name] = result

    # --- Summary table: mean CV metrics across folds per classifier
    summary_rows = []
    for clf_name, res in results.items():
        scores = res.get("cv_scores", [])
        if not scores:
            continue
        df_scores = pd.DataFrame(scores)
        row = {
            "classifier"  : clf_name,
            "bar_type"    : meta.get("bar_type", "?"),
            "source"      : meta.get("source",   "?"),
            "bar_class"   : meta.get("bar_class","?"),
            "feature_mode": ml_frame["feature_mode"].iloc[0],
            "n_events"    : len(ml_frame),
            "accuracy_mean": df_scores["accuracy"].mean(),
            "accuracy_std" : df_scores["accuracy"].std(),
            "f1_mean"      : df_scores["f1_weighted"].mean(),
            "f1_std"       : df_scores["f1_weighted"].std(),
            "n_folds"      : len(scores),
        }
        if "auc" in df_scores.columns:
            row["auc_mean"] = df_scores["auc"].mean()
            row["auc_std"]  = df_scores["auc"].std()
        summary_rows.append(row)

    results["summary"] = pd.DataFrame(summary_rows)

    if not summary_rows:
        log.warning("  No classifiers produced valid results for %s", name)
    else:
        log.info("  CV Summary:")
        for row in summary_rows:
            log.info("    %-20s  acc=%.3f±%.3f  f1=%.3f±%.3f",
                     row["classifier"],
                     row["accuracy_mean"], row["accuracy_std"],
                     row["f1_mean"],       row["f1_std"])

    return results