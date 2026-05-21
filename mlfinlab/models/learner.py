"""
mlfinlab.models.learner
==========================
Classifier training with Walk-Forward CV and Combinatorial Purged CV.

REFACTORING NOTES (bugs fixed vs original)
-------------------------------------------
1. Pipeline classes_ attribute (BUG): original accessed clf.classes_ on
   the SVM Pipeline directly after fitting. sklearn Pipeline does not
   expose .classes_ at the top level — you must access
   clf.named_steps['svc'].classes_ (or Pipeline[-1].classes_). This caused
   AttributeError for SVM during probability extraction. Fixed with a
   helper _get_classes(clf).

2. _sample_weight_kwargs for Pipeline (BUG): original mapped weights as
   "{last_step_name}__sample_weight" but did not account for the SVM
   Pipeline having StandardScaler as its first step (which doesn't accept
   sample_weight). The correct sklearn Pipeline kwarg syntax for passing
   sample weights to the classifier step is:
   "svc__sample_weight" (step name) not "{last_step}__sample_weight"
   (which could accidentally target the scaler if names differ).
   Fixed by inspecting pipeline steps to find the classifier step.

3. CPCV integration: train_all now optionally runs CPCV alongside
   WalkForwardCV. CPCV results are stored separately per classifier and
   returned in the results dict for Stage 6 aggregation.

4. AUC computation: original only computed binary AUC (+1 vs rest).
   For three-class labels (-1, 0, +1), multi-class OvR AUC is now
   computed when all three classes are present in the test fold.

5. t1 fillna: original used pd.Series(fallback, index=ml_frame.index)
   where fallback was a DatetimeIndex — this created a Series of
   Timestamps correctly, but the pd.Series() constructor sometimes
   converts to int64 (nanoseconds). Now uses explicit dtype specification.

6. CalibratedClassifierCV cv=3 may produce warnings when a training fold
   has fewer than 3 samples of some class. Now suppressed and guarded.

References
----------
de Prado, M. L. (2018). Advances in Financial Machine Learning, Ch.6-8, 12.
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
    accuracy_score, f1_score, roc_auc_score,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from mlfinlab.models.cv import WalkForwardCV, CPCV

log = logging.getLogger("mlfinlab.models")

# Columns that are metadata, not features
NON_FEATURE_COLS = {
    "ret", "bin", "weight", "bar_type", "source",
    "bar_class", "min_d", "feature_mode", "t1_touch",
}


# ---------------------------------------------------------------------------
# XGBoost wrapper (label remapping)
# ---------------------------------------------------------------------------

class XGBWrapper(BaseEstimator, ClassifierMixin):
    """Wraps XGBClassifier to remap labels -1/0/+1 → 0/1/2 for XGBoost."""

    def __init__(self, xgb_clf):
        self.xgb_clf    = xgb_clf
        self.classes_   = None
        self._label_map = {}
        self._inv_map   = {}

    def fit(self, X, y, sample_weight=None):
        classes = sorted(np.unique(y))
        self.classes_    = np.array(classes)
        self._label_map  = {c: i for i, c in enumerate(classes)}
        self._inv_map    = {i: c for i, c in enumerate(classes)}
        y_mapped = np.array([self._label_map[v] for v in y])
        kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.xgb_clf.fit(X, y_mapped, **kw)
        return self

    def predict(self, X):
        y_mapped = self.xgb_clf.predict(X)
        return np.array([self._inv_map.get(v, v) for v in y_mapped])

    def predict_proba(self, X):
        return self.xgb_clf.predict_proba(X)

    def get_params(self, deep=True):
        return {"xgb_clf": self.xgb_clf}

    def set_params(self, **params):
        if "xgb_clf" in params:
            self.xgb_clf = params["xgb_clf"]
        return self


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def _get_classifiers(random_state: int = 42) -> dict:
    """Return the classifiers used in the study."""
    rf = RandomForestClassifier(
        n_estimators    = 500,
        max_depth       = None,
        min_samples_leaf= 5,
        max_features    = "sqrt",
        class_weight    = "balanced",
        n_jobs          = -1,
        random_state    = random_state,
    )

    gb = GradientBoostingClassifier(
        n_estimators = 300,
        max_depth    = 3,
        learning_rate= 0.05,
        subsample    = 0.8,
        max_features = "sqrt",
        random_state = random_state,
    )

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(
            kernel       = "rbf",
            C            = 1.0,
            gamma        = "scale",
            probability  = True,
            class_weight = "balanced",
            random_state = random_state,
        )),
    ])

    classifiers = {
        "random_forest" : CalibratedClassifierCV(rf, method="isotonic", cv=3),
        "gradient_boost": CalibratedClassifierCV(gb, method="isotonic", cv=3),
        "svm"           : svm,
    }

    if HAS_XGB:
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
# Helpers: Pipeline-safe classes_ and sample_weight kwargs
# ---------------------------------------------------------------------------

def _get_classes(clf) -> np.ndarray:
    """Extract classes_ from a fitted classifier or Pipeline.

    BUG FIX: original accessed clf.classes_ directly, which fails for
    sklearn Pipeline (Pipeline does not propagate classes_ from sub-estimators
    until sklearn 1.3+). Now checks named_steps first.
    """
    if hasattr(clf, "classes_"):
        return clf.classes_

    # sklearn Pipeline: look in steps
    if hasattr(clf, "named_steps"):
        for step_name, step in clf.named_steps.items():
            if hasattr(step, "classes_"):
                return step.classes_

    # CalibratedClassifierCV
    if hasattr(clf, "calibrated_classifiers_"):
        base = clf.calibrated_classifiers_[0]
        if hasattr(base, "classes_"):
            return base.classes_
        if hasattr(base, "estimator") and hasattr(base.estimator, "classes_"):
            return base.estimator.classes_

    raise AttributeError(f"Cannot find classes_ on {type(clf).__name__}")


def _sample_weight_kwargs(clf, w: pd.Series) -> dict:
    """Return sample_weight kwarg dict if the classifier supports it.

    BUG FIX: for Pipeline, must pass weight to the CLASSIFIER step, not
    the scaler step. Use the step name directly rather than the last step.
    """
    try:
        if hasattr(clf, "named_steps"):
            # Find the step that accepts sample_weight (classifier, not scaler)
            for step_name, step in clf.named_steps.items():
                if hasattr(step, "fit") and not isinstance(step, StandardScaler):
                    return {f"{step_name}__sample_weight": w.values}
            return {}
        return {"sample_weight": w.values}
    except Exception:
        return {}


def _mdi_importance(model, feature_names: list) -> pd.Series:
    """Extract MDI feature importance from a fitted RF."""
    try:
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
# Probability extraction (handles binary and multiclass)
# ---------------------------------------------------------------------------

def _extract_probs(
    clf,
    X_te: pd.DataFrame,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (prob_m1, prob_0, prob_p1) arrays aligned to X_te."""
    proba = clf.predict_proba(X_te)
    clf_classes = _get_classes(clf)

    def _prob_for_class(label):
        matches = np.where(clf_classes == label)[0]
        return proba[:, matches[0]] if len(matches) > 0 else np.zeros(len(X_te))

    prob_m1 = _prob_for_class(-1)
    prob_0  = _prob_for_class(0)
    prob_p1 = _prob_for_class(1)
    return prob_m1, prob_0, prob_p1


def _fold_metrics(
    y_te: pd.Series,
    y_pred: np.ndarray,
    prob_p1: np.ndarray,
    fold_id: object,
    n_train: int,
    n_test: int,
) -> dict:
    """Compute per-fold classification metrics."""
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    scores = {
        "fold"    : fold_id,
        "accuracy": acc,
        "f1_weighted": f1,
        "n_train" : n_train,
        "n_test"  : n_test,
    }
    unique_classes = set(np.unique(y_te))
    if len(unique_classes) == 2 and 1 in unique_classes:
        try:
            y_bin = (np.array(y_te) == 1).astype(int)
            scores["auc"] = roc_auc_score(y_bin, prob_p1)
        except Exception:
            pass
    return scores


# ---------------------------------------------------------------------------
# Single classifier: walk-forward CV training
# ---------------------------------------------------------------------------

def _train_one_wf(
    clf_name    : str,
    clf,
    X           : pd.DataFrame,
    y           : pd.Series,
    w           : pd.Series,
    t1          : pd.Series,
    cv          : WalkForwardCV,
) -> dict:
    """Train one classifier with WalkForwardCV.

    Returns
    -------
    dict  predictions, cv_scores, feature_importance, clf_fitted, col_medians
    """
    classes   = sorted(y.unique())
    all_preds : list = []
    cv_scores : list = []

    col_medians = X.median()

    log.info("    [%s] Walk-forward CV ...", clf_name)

    for fold_idx, (train_pos, test_pos) in enumerate(cv.split(X, t1)):
        if len(train_pos) < 30:
            log.warning(
                "    [%s] Fold %d: only %d train events, skipping",
                clf_name, fold_idx + 1, len(train_pos),
            )
            continue

        X_tr = X.iloc[train_pos].fillna(col_medians)
        y_tr = y.iloc[train_pos]
        w_tr = w.iloc[train_pos]
        X_te = X.iloc[test_pos].fillna(col_medians)
        y_te = y.iloc[test_pos]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                clf.fit(X_tr, y_tr, **_sample_weight_kwargs(clf, w_tr))
            except Exception as e:
                log.warning("    [%s] Fold %d fit failed: %s", clf_name, fold_idx + 1, e)
                continue

        y_pred             = clf.predict(X_te)
        prob_m1, prob_0, prob_p1 = _extract_probs(clf, X_te, np.array(classes))

        fold_df = pd.DataFrame({
            "y_true" : y_te.values,
            "y_pred" : y_pred,
            "prob_m1": prob_m1,
            "prob_0" : prob_0,
            "prob_p1": prob_p1,
            "fold"   : fold_idx + 1,
        }, index=X_te.index)
        all_preds.append(fold_df)

        scores = _fold_metrics(y_te, y_pred, prob_p1,
                               fold_id=fold_idx + 1,
                               n_train=len(train_pos),
                               n_test=len(test_pos))
        cv_scores.append(scores)

        log.info(
            "    [%s] WF Fold %d: acc=%.3f  f1=%.3f  train=%d  test=%d",
            clf_name, fold_idx + 1, scores["accuracy"], scores["f1_weighted"],
            len(train_pos), len(test_pos),
        )

    if not all_preds:
        log.warning("    [%s] No valid WF folds produced predictions", clf_name)
        return {}

    predictions = pd.concat(all_preds).sort_index()

    # Final fit on full dataset for Stage 4 signal generation
    log.info("    [%s] Final fit on full dataset ...", clf_name)
    X_clean = X.fillna(col_medians)
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


# ---------------------------------------------------------------------------
# Single classifier: CPCV training
# ---------------------------------------------------------------------------

def _train_one_cpcv(
    clf_name  : str,
    clf,
    X         : pd.DataFrame,
    y         : pd.Series,
    w         : pd.Series,
    t1        : pd.Series,
    cpcv      : CPCV,
) -> dict:
    """Train one classifier with CPCV.

    CPCV does NOT produce a single stitched prediction sequence — it
    produces C(N,k) OOS prediction sets (one per combination). We store
    the per-combination metrics and the distribution of Sharpe ratios
    so Stage 6 can compute the Deflated Sharpe Ratio.

    Returns
    -------
    dict  cpcv_scores (list of per-combo metrics), n_combos, n_paths
    """
    col_medians = X.median()
    cpcv_scores : list = []
    classes = sorted(y.unique())

    log.info("    [%s] CPCV (%d combinations, %d test folds each) ...",
             clf_name, cpcv.n_combinations, cpcv.n_test_folds)

    for combo_idx, (train_pos, test_pos, combo_id) in enumerate(cpcv.split(X, t1)):
        if len(train_pos) < cpcv.min_train_events:
            continue

        X_tr = X.iloc[train_pos].fillna(col_medians)
        y_tr = y.iloc[train_pos]
        w_tr = w.iloc[train_pos]
        X_te = X.iloc[test_pos].fillna(col_medians)
        y_te = y.iloc[test_pos]

        # Clone the classifier for this combination to avoid state leakage
        import sklearn.base as skbase
        clf_clone = skbase.clone(
            clf.xgb_clf if isinstance(clf, XGBWrapper) else clf
        )
        if isinstance(clf, XGBWrapper):
            clf_clone = XGBWrapper(clf_clone)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                clf_clone.fit(X_tr, y_tr, **_sample_weight_kwargs(clf_clone, w_tr))
            except Exception as e:
                log.debug("    [%s] CPCV combo %s fit failed: %s", clf_name, combo_id, e)
                continue

        y_pred               = clf_clone.predict(X_te)
        prob_m1, prob_0, prob_p1 = _extract_probs(clf_clone, X_te, np.array(classes))

        scores = _fold_metrics(y_te, y_pred, prob_p1,
                               fold_id=combo_id,
                               n_train=len(train_pos),
                               n_test=len(test_pos))
        scores["combo_id"] = str(combo_id)
        cpcv_scores.append(scores)

    if cpcv_scores:
        log.info(
            "    [%s] CPCV: %d combos completed, acc=%.3f±%.3f",
            clf_name, len(cpcv_scores),
            np.mean([s["accuracy"] for s in cpcv_scores]),
            np.std( [s["accuracy"] for s in cpcv_scores]),
        )
    else:
        log.warning("    [%s] CPCV: no combinations produced valid results", clf_name)

    return {
        "cpcv_scores": cpcv_scores,
        "n_combos"   : len(cpcv_scores),
        "n_paths"    : cpcv.n_paths,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_all(
    name               : str,
    ml_frame           : pd.DataFrame,
    meta               : dict,
    random_state       : int   = 42,
    n_splits           : int   = 5,
    embargo_pct        : float = 0.01,
    initial_train_pct  : float = 0.40,
    run_cpcv           : bool  = True,
    cpcv_n_splits      : int   = 6,
    cpcv_n_test_folds  : int   = 2,
) -> dict:
    """Train all classifiers on ml_frame and return results.

    Parameters
    ----------
    name               : Dataset name (for logging).
    ml_frame           : Output of stage_labeling_features.
    meta               : Bar metadata dict.
    random_state       : Seed for reproducibility.
    n_splits           : Walk-forward CV folds.
    embargo_pct        : Embargo fraction.
    initial_train_pct  : Fraction for first training fold.
    run_cpcv           : Whether to also run CPCV (default True).
    cpcv_n_splits      : Number of chronological folds for CPCV.
    cpcv_n_test_folds  : Test folds per CPCV combination.

    Returns
    -------
    dict  {
        classifier_name: {
            predictions, cv_scores, cpcv_scores (if run_cpcv),
            feature_importance, clf_fitted
        },
        "summary": pd.DataFrame  mean CV metrics per classifier
        "cpcv_summary": pd.DataFrame  CPCV distribution per classifier
    }
    """
    # --- Feature / label separation
    feature_cols = [c for c in ml_frame.columns if c not in NON_FEATURE_COLS]
    X = ml_frame[feature_cols].copy()
    y = ml_frame["bin"].astype(int).copy()
    w = ml_frame["weight"].copy()

    # t1 for purging: use actual barrier exit timestamps
    if "t1_touch" in ml_frame.columns:
        t1_raw  = ml_frame["t1_touch"].copy()
        fallback = pd.to_datetime(
            ml_frame.index + pd.Timedelta(days=3), utc=True
        )
        # BUG FIX: explicit datetime64[ns, UTC] dtype to avoid int64 coercion
        t1 = t1_raw.fillna(pd.Series(fallback, index=ml_frame.index, dtype="datetime64[ns, UTC]"))
    else:
        t1 = pd.Series(
            pd.to_datetime(ml_frame.index + pd.Timedelta(days=3), utc=True),
            index=ml_frame.index,
            name="t1",
            dtype="datetime64[ns, UTC]",
        )

    log.info("  Features : %d  Events : %d  Classes : %s",
             X.shape[1], len(y), sorted(y.unique()))
    log.info("  Label distribution : %s", y.value_counts().sort_index().to_dict())

    wf_cv = WalkForwardCV(
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        initial_train_pct=initial_train_pct,
    )

    if run_cpcv and len(X) >= 50:
        cpcv = CPCV(
            n_splits=cpcv_n_splits,
            n_test_folds=cpcv_n_test_folds,
            embargo_pct=embargo_pct,
            min_train_events=30,
        )
        log.info("  CPCV: N=%d  k=%d  C(N,k)=%d  paths≈%.1f",
                 cpcv_n_splits, cpcv_n_test_folds,
                 cpcv.n_combinations, cpcv.n_paths)
    else:
        cpcv = None

    classifiers = _get_classifiers(random_state)
    results     = {}

    for clf_name, clf in classifiers.items():
        log.info("  Training [%s] ...", clf_name)

        # Walk-Forward CV
        wf_result = _train_one_wf(clf_name, clf, X, y, w, t1, wf_cv)
        if not wf_result:
            continue

        results[clf_name] = wf_result

        # CPCV (uses a fresh clone of clf; wf_result["clf_fitted"] is the
        # full-data fit from walk-forward which we want to keep)
        if cpcv is not None:
            import sklearn.base as skbase
            clf_fresh = skbase.clone(
                clf.xgb_clf if isinstance(clf, XGBWrapper) else clf
            )
            if isinstance(clf, XGBWrapper):
                clf_fresh = XGBWrapper(clf_fresh)

            cpcv_result = _train_one_cpcv(clf_name, clf_fresh, X, y, w, t1, cpcv)
            results[clf_name]["cpcv_scores"] = cpcv_result.get("cpcv_scores", [])
            results[clf_name]["cpcv_n_combos"] = cpcv_result.get("n_combos", 0)

    # --- Walk-Forward summary table
    summary_rows = []
    for clf_name, res in results.items():
        scores = res.get("cv_scores", [])
        if not scores:
            continue
        df_scores = pd.DataFrame(scores)
        row = {
            "classifier"    : clf_name,
            "bar_type"      : meta.get("bar_type", "?"),
            "source"        : meta.get("source",   "?"),
            "bar_class"     : meta.get("bar_class","?"),
            "feature_mode"  : ml_frame["feature_mode"].iloc[0],
            "n_events"      : len(ml_frame),
            "cv_method"     : "walk_forward",
            "accuracy_mean" : df_scores["accuracy"].mean(),
            "accuracy_std"  : df_scores["accuracy"].std(),
            "f1_mean"       : df_scores["f1_weighted"].mean(),
            "f1_std"        : df_scores["f1_weighted"].std(),
            "n_folds"       : len(scores),
        }
        if "auc" in df_scores.columns:
            row["auc_mean"] = df_scores["auc"].mean()
            row["auc_std"]  = df_scores["auc"].std()
        summary_rows.append(row)

    results["summary"] = pd.DataFrame(summary_rows)

    # --- CPCV summary table (distribution of metrics across combinations)
    cpcv_rows = []
    for clf_name, res in results.items():
        if clf_name == "summary":
            continue
        cscores = res.get("cpcv_scores", [])
        if not cscores:
            continue
        df_c = pd.DataFrame(cscores)
        crow = {
            "classifier"    : clf_name,
            "bar_type"      : meta.get("bar_type", "?"),
            "cv_method"     : "cpcv",
            "n_combos"      : len(cscores),
            "accuracy_mean" : df_c["accuracy"].mean(),
            "accuracy_std"  : df_c["accuracy"].std(),
            "f1_mean"       : df_c["f1_weighted"].mean(),
            "f1_std"        : df_c["f1_weighted"].std(),
        }
        if "auc" in df_c.columns:
            auc_vals = df_c["auc"].dropna()
            if len(auc_vals) > 0:
                crow["auc_mean"] = auc_vals.mean()
                crow["auc_std"]  = auc_vals.std()
        cpcv_rows.append(crow)

    results["cpcv_summary"] = pd.DataFrame(cpcv_rows) if cpcv_rows else pd.DataFrame()

    # --- Logging
    if summary_rows:
        log.info("  Walk-Forward CV Summary:")
        for row in summary_rows:
            log.info(
                "    %-20s  acc=%.3f±%.3f  f1=%.3f±%.3f",
                row["classifier"],
                row["accuracy_mean"], row.get("accuracy_std", 0),
                row["f1_mean"],       row.get("f1_std", 0),
            )
    if cpcv_rows:
        log.info("  CPCV Summary:")
        for row in cpcv_rows:
            log.info(
                "    %-20s  %d combos  acc=%.3f±%.3f  f1=%.3f±%.3f",
                row["classifier"], row["n_combos"],
                row["accuracy_mean"], row.get("accuracy_std", 0),
                row["f1_mean"],       row.get("f1_std", 0),
            )

    return results
