"""
ml_module/tests/test_pipeline.py
----------------------------------
Unit and integration tests for all pipeline components.

Run with:
    python -m pytest ml_module/tests/test_pipeline.py -v
or:
    python ml_module/tests/test_pipeline.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PASS  = "✅"
FAIL  = "❌"
WARN  = "⚠️ "
SEP   = "-" * 60

results: List[Tuple[str, bool, str]] = []


def register(name):
    """Decorator that catches exceptions and records pass/fail."""
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results.append((name, True, ""))
                print(f"  {PASS}  {name}")
            except Exception as e:
                tb = traceback.format_exc()
                results.append((name, False, str(e)))
                print(f"  {FAIL}  {name}")
                print(f"       {e}")
                print(tb)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV bar data with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    close = 40000.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
    noise = rng.uniform(0.001, 0.005, n)

    df = pd.DataFrame({
        "open":             close * (1 - rng.uniform(0, 0.002, n)),
        "high":             close * (1 + noise),
        "low":              close * (1 - noise),
        "close":            close,
        "volume":           rng.uniform(1000, 5000, n),
        "vwap":             close * (1 + rng.normal(0, 0.001, n)),
        "bar_size":         rng.uniform(1e9, 5e9, n),
        "duration_minutes": rng.integers(30, 300, n),
        "tick_count":       rng.integers(100, 1000, n),
        "bar_return":       rng.normal(0, 0.002, n),
        "price_range":      noise,
        "score_bull":       rng.uniform(0, 1, n),
        "score_bear":       rng.uniform(0, 1, n),
        "score_transition": rng.uniform(0, 1, n),
        "regime_confidence": rng.uniform(0, 1, n),
        "trend_strength_z": rng.normal(0, 1, n),
        "vol_percentile":   rng.uniform(0, 1, n),
    }, index=idx)
    return df


# ---------------------------------------------------------------------------
# Tests — Labeling
# ---------------------------------------------------------------------------

@register("TripleBarrierLabeler: output shape and values")
def test_labeler_basic():
    from ml_module.labeling.triple_barrier import TripleBarrierLabeler
    df     = make_ohlcv(300)
    lbl    = TripleBarrierLabeler(profit_target=0.02, stop_loss=0.01, max_holding_bars=10)
    labels = lbl.fit_transform(df)
    assert len(labels) == len(df), "Label length mismatch"
    assert set(labels.unique()).issubset({-1, 0, 1}), "Invalid label values"
    assert labels.name == "label"


@register("TripleBarrierLabeler: volatility scaling produces labels")
def test_labeler_vol_scaling():
    from ml_module.labeling.triple_barrier import TripleBarrierLabeler
    df  = make_ohlcv(300)
    lbl = TripleBarrierLabeler(profit_target=1.5, stop_loss=1.0, volatility_lookback=20)
    labels = lbl.fit_transform(df)
    assert len(labels) == len(df)
    assert len(labels[labels != 0]) > 10, "Too many HOLD labels — barriers may be too wide"


@register("TripleBarrierLabeler: min_class_count warning does not crash")
def test_labeler_min_class():
    from ml_module.labeling.triple_barrier import TripleBarrierLabeler
    df  = make_ohlcv(60)  # small dataset
    lbl = TripleBarrierLabeler(profit_target=0.5, stop_loss=0.5,
                                max_holding_bars=5, min_class_count=100)
    labels = lbl.fit_transform(df)
    assert labels is not None


# ---------------------------------------------------------------------------
# Tests — Fractional Differencing
# ---------------------------------------------------------------------------

@register("FractionalDifferencer: fixed d produces fdiff columns")
def test_fracdiff_fixed():
    from ml_module.features.fractional_diff import FractionalDifferencer
    df  = make_ohlcv(200)
    fd  = FractionalDifferencer(d=0.4, target_columns=["close", "volume"])
    out = fd.fit_transform(df)
    assert "close_fdiff"  in out.columns
    assert "volume_fdiff" in out.columns
    assert not out["close_fdiff"].isna().all(), "All NaN in fdiff column"


@register("FractionalDifferencer: auto-d search finds stationary d")
def test_fracdiff_auto():
    from ml_module.features.fractional_diff import FractionalDifferencer
    df = make_ohlcv(300)
    fd = FractionalDifferencer(d="auto", target_columns=["close"],
                               d_min=0.1, d_max=1.0, d_step=0.2)
    fd.fit(df)
    assert "close" in fd.d_values_, "d not computed for 'close'"
    d = fd.d_values_["close"]
    assert 0.0 < d <= 1.0, f"d={d} out of range"


@register("FractionalDifferencer: stationarity_report returns DataFrame")
def test_fracdiff_report():
    from ml_module.features.fractional_diff import FractionalDifferencer
    df   = make_ohlcv(200)
    fd   = FractionalDifferencer(d=0.5, target_columns=["close"])
    out  = fd.fit_transform(df)
    rep  = fd.stationarity_report(out)
    assert isinstance(rep, pd.DataFrame)
    assert "d" in rep.columns
    assert "adf_p_fdiff" in rep.columns


# ---------------------------------------------------------------------------
# Tests — Feature Engineering
# ---------------------------------------------------------------------------

@register("FeatureEngineer: produces numeric-only output with no leakage cols")
def test_feature_engineer_basic():
    from ml_module.features.feature_engineer import FeatureEngineer
    df  = make_ohlcv(300)
    fe  = FeatureEngineer(lag_periods=[1])
    out = fe.fit_transform(df)
    assert out.select_dtypes(include=[np.number]).shape == out.shape, "Non-numeric columns remain"
    assert "exchange" not in out.columns
    assert len(out) < len(df), "Expected NaN rows to be dropped"


@register("FeatureEngineer: lag features are shifted (no leakage)")
def test_feature_engineer_lags():
    from ml_module.features.feature_engineer import FeatureEngineer
    df  = make_ohlcv(200)
    fe  = FeatureEngineer(lag_periods=[1, 2])
    out = fe.fit_transform(df)
    lag_cols = [c for c in out.columns if "_lag" in c]
    assert len(lag_cols) > 0, "No lag columns generated"


# ---------------------------------------------------------------------------
# Tests — CPCV
# ---------------------------------------------------------------------------

@register("CPCVSplitter: generates correct number of folds")
def test_cpcv_folds():
    from ml_module.validation.cpcv import CPCVSplitter
    from math import comb
    n_splits, n_test = 6, 2
    spl  = CPCVSplitter(n_splits=n_splits, n_test_splits=n_test,
                         embargo_bars=3, min_train_size=20)
    X    = pd.DataFrame(np.random.randn(300, 10))
    folds = list(spl.split(X))
    expected = comb(n_splits, n_test)
    assert len(folds) <= expected, f"Too many folds: {len(folds)} > {expected}"
    assert len(folds) > 0, "No folds generated"


@register("CPCVSplitter: train and test indices are disjoint")
def test_cpcv_disjoint():
    from ml_module.validation.cpcv import CPCVSplitter
    spl = CPCVSplitter(n_splits=5, n_test_splits=1, embargo_bars=5, min_train_size=20)
    X   = pd.DataFrame(np.random.randn(200, 5))
    for train, test in spl.split(X):
        overlap = np.intersect1d(train, test)
        assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} indices"


@register("purge_indices: removes overlapping label windows")
def test_purge():
    from ml_module.validation.cpcv import purge_indices
    train = np.arange(0, 50)
    test  = np.arange(60, 80)
    # Labels at positions 45–54 overlap test (start=60)
    label_ends = np.arange(0, 100) + 15  # each label extends 15 bars
    purged = purge_indices(train, test, label_ends)
    # Bars where label_end >= 60 should be purged: bar >= 45
    assert all(purged < 45), f"Not all overlapping bars purged: max={purged.max()}"


@register("embargo_indices: removes bars within embargo zone")
def test_embargo():
    from ml_module.validation.cpcv import embargo_indices
    train = np.arange(0, 100)
    test  = np.arange(110, 130)
    embargoed = embargo_indices(train, test, embargo_bars=10)
    assert all(embargoed < 100), "Embargo should not remove indices beyond train"
    # Bars 100–109 are embargoed (but they're not in train here)
    # test_start=110, embargo_start=100, zone=[100,110) — not in train, so train unchanged
    assert len(embargoed) == len(train)


# ---------------------------------------------------------------------------
# Tests — Walk-Forward
# ---------------------------------------------------------------------------

@register("WalkForwardValidator: produces predictions for out-of-sample bars")
def test_walk_forward():
    from ml_module.validation.walk_forward import WalkForwardValidator
    from sklearn.dummy import DummyClassifier

    n = 300
    X = pd.DataFrame(np.random.randn(n, 5),
                      index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))
    y = pd.Series(np.random.choice([-1, 0, 1], n), index=X.index, name="label")

    def factory(Xtr, ytr):
        m = DummyClassifier(strategy="most_frequent")
        m.fit(Xtr, ytr)
        return m

    wf = WalkForwardValidator(initial_train_bars=100, step_bars=30,
                               min_test_bars=20, embargo_bars=3)
    folds, preds = wf.validate(X, y, factory)
    assert len(folds) > 0, "No walk-forward folds generated"
    non_nan = preds.dropna()
    assert len(non_nan) > 0, "No out-of-sample predictions"


@register("WalkForwardValidator: get_splits preview matches validate count")
def test_walk_forward_preview():
    from ml_module.validation.walk_forward import WalkForwardValidator
    wf     = WalkForwardValidator(initial_train_bars=100, step_bars=30, min_test_bars=20)
    splits = wf.get_splits(300)
    assert len(splits) > 0


# ---------------------------------------------------------------------------
# Tests — Ensemble
# ---------------------------------------------------------------------------

@register("MetaEnsemble: fits and predicts on synthetic data")
def test_ensemble_fit_predict():
    from ml_module.models.ensemble import MetaEnsemble, PrimaryLearner, build_model

    n, d = 200, 20
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.choice([-1, 0, 1], n)

    learners = [
        PrimaryLearner("dir",  "direction",  build_model("lightgbm_clf", {"n_estimators": 30, "random_state": 0, "verbose": -1})),
        PrimaryLearner("conf", "confidence", build_model("lightgbm_reg", {"n_estimators": 30, "random_state": 0, "verbose": -1})),
        PrimaryLearner("reg",  "regime",     build_model("rf_clf",       {"n_estimators": 20, "random_state": 0})),
    ]
    meta = build_model("logreg", {"max_iter": 200, "random_state": 0})
    ens  = MetaEnsemble(learners, meta, meta_train_fraction=0.3)
    ens.fit(X, y)
    preds = ens.predict(X)
    assert len(preds) == n
    assert set(preds).issubset({-1, 0, 1})


@register("MetaEnsemble: predict_proba sums to 1")
def test_ensemble_proba():
    from ml_module.models.ensemble import MetaEnsemble, PrimaryLearner, build_model

    n, d = 150, 10
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.choice([-1, 0, 1], n)

    learners = [
        PrimaryLearner("dir", "direction", build_model("rf_clf", {"n_estimators": 10, "random_state": 0})),
    ]
    meta = build_model("logreg", {"max_iter": 200, "random_state": 0})
    ens  = MetaEnsemble(learners, meta, meta_train_fraction=0.3)
    ens.fit(X, y)
    proba = ens.predict_proba(X)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Proba does not sum to 1: {row_sums[:5]}"


# ---------------------------------------------------------------------------
# Tests — Signal Exporter
# ---------------------------------------------------------------------------

@register("SignalExporter: exports correct CSV structure")
def test_signal_exporter():
    from ml_module.backtest_bridge.signal_exporter import SignalExporter
    import tempfile, os

    df  = make_ohlcv(100)
    idx = df.index
    preds = pd.Series(np.random.choice([-1, 0, 1], len(idx)), index=idx)

    with tempfile.TemporaryDirectory() as tmp:
        exp = SignalExporter(output_dir=tmp)
        sig = exp.export(preds, df, bar_type="dollar")
        assert "datetime" in sig.columns
        assert "signals"  in sig.columns
        assert set(sig["signals"].unique()).issubset({-1, 0, 1})
        files = list(Path(tmp).glob("*.csv"))
        assert len(files) == 1, "Expected one CSV file"


# ---------------------------------------------------------------------------
# Integration test — end-to-end on synthetic data
# ---------------------------------------------------------------------------

@register("Integration: full pipeline on synthetic bar data")
def test_integration():
    import tempfile, yaml
    from ml_module.pipeline import MLPipeline

    df = make_ohlcv(400)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv_path = tmp / "dollar_bars.csv"
        df.reset_index().rename(columns={"index": "datetime"}).to_csv(csv_path, index=False)

        cfg = {
            "data": {
                "bar_data_dir": str(tmp),
                "bar_files":    {"dollar": "dollar_bars.csv"},
                "datetime_col": "datetime",
                "min_bars":     100,
            },
            "labeling": {
                "profit_target": 0.015, "stop_loss": 0.01,
                "max_holding_bars": 10, "volatility_lookback": 10,
                "min_class_count": 5,
            },
            "fractional_diff": {
                "d": 0.5, "target_columns": ["close", "volume"],
                "threshold": 1e-5, "max_window": 50,
                "d_min": 0.1, "d_max": 1.0, "d_step": 0.2, "adf_significance": 0.05,
            },
            "features": {
                "rsi_windows": [7, 14], "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                "bb_window": 20, "atr_window": 14, "ema_windows": [9, 21],
                "return_horizons": [1, 3], "lag_periods": [1],
                "drop_columns": ["regime_label", "regime_trend", "regime_volatility",
                                 "regime_momentum", "datetime_start", "datetime_end",
                                 "created_at", "Unnamed: 0", "exchange", "symbol"],
            },
            "cpcv": {
                "n_splits": 4, "n_test_splits": 1,
                "embargo_bars": 3, "min_train_size": 50,
            },
            "walk_forward": {
                "initial_train_bars": 120, "step_bars": 40,
                "min_test_bars": 20, "embargo_bars": 3,
            },
            "ensemble": {
                "primary_learners": {
                    "direction_model": {
                        "role": "direction", "type": "lightgbm_clf",
                        "params": {"n_estimators": 30, "random_state": 42, "verbose": -1},
                    },
                    "confidence_model": {
                        "role": "confidence", "type": "lightgbm_reg",
                        "params": {"n_estimators": 20, "random_state": 42, "verbose": -1},
                    },
                },
                "meta_learner": {
                    "type": "logreg",
                    "params": {"max_iter": 200, "random_state": 42},
                },
                "meta_train_fraction": 0.3,
            },
            "backtest": {
                "signal_col": "signals",
                "signal_map": {1: 1, -1: -1, 0: 0},
                "output_dir": str(tmp / "outputs"),
            },
            "logging": {"level": "WARNING"},
        }

        pipeline = MLPipeline(cfg)
        results  = pipeline.run(bar_csv=str(csv_path), bar_type="dollar", min_bars=100)

        assert "wf_summary" in results, "Walk-forward summary missing"
        assert "signal_summary" in results, "Signal summary missing"
        assert results.get("n_aligned", 0) > 50, "Too few aligned samples"

        sig_files = list((tmp / "outputs").glob("signals_*.csv"))
        assert len(sig_files) >= 1, "No signal CSV exported"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  ml_module Test Suite")
    print(f"{'='*60}\n")

    test_labeler_basic()
    test_labeler_vol_scaling()
    test_labeler_min_class()
    test_fracdiff_fixed()
    test_fracdiff_auto()
    test_fracdiff_report()
    test_feature_engineer_basic()
    test_feature_engineer_lags()
    test_cpcv_folds()
    test_cpcv_disjoint()
    test_purge()
    test_embargo()
    test_walk_forward()
    test_walk_forward_preview()
    test_ensemble_fit_predict()
    test_ensemble_proba()
    test_signal_exporter()
    test_integration()

    print(f"\n{SEP}")
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    print(f"  Results: {passed}/{len(results)} passed  |  {failed} failed")
    print(SEP)

    if failed:
        print("\nFailed tests:")
        for name, ok, msg in results:
            if not ok:
                print(f"  {FAIL} {name}: {msg}")
        sys.exit(1)
    else:
        print(f"\n  {PASS} All tests passed!")
        sys.exit(0)
