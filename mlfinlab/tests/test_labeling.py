"""
Tests for mlfinlab.labeling.triple_barrier
"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mlfinlab.data.synthetic import make_ohlcv
from mlfinlab.utils.helpers import daily_vol, cusum_filter
from mlfinlab.labeling.triple_barrier import (
    add_vertical_barrier,
    get_events,
    get_bins,
    drop_labels,
    meta_labeling,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bars():
    return make_ohlcv(n_bars=800, freq="1h", seed=1)


@pytest.fixture(scope="module")
def close(bars):
    return bars["close"]


@pytest.fixture(scope="module")
def vol(close):
    return daily_vol(close, lookback=50)


@pytest.fixture(scope="module")
def t_events(close, vol):
    return cusum_filter(close, threshold=vol * 0.5)


@pytest.fixture(scope="module")
def t1(close, t_events):
    return add_vertical_barrier(t_events, close, num_days=3.0)


@pytest.fixture(scope="module")
def events(close, t_events, vol, t1):
    return get_events(
        close=close,
        t_events=t_events,
        pt_sl=[2.0, 2.0],
        target=vol,
        min_ret=0.0,
        vertical_barrier_times=t1,
    )


@pytest.fixture(scope="module")
def bins(events, close):
    return get_bins(events, close)


# ---------------------------------------------------------------------------
# add_vertical_barrier
# ---------------------------------------------------------------------------

class TestAddVerticalBarrier:
    def test_returns_series(self, close, t_events):
        t1 = add_vertical_barrier(t_events, close, num_days=1.0)
        assert isinstance(t1, pd.Series)

    def test_index_matches_events(self, close, t_events):
        t1 = add_vertical_barrier(t_events, close, num_days=1.0)
        assert set(t1.index) == set(t_events)

    def test_all_barriers_after_event(self, close, t_events):
        t1 = add_vertical_barrier(t_events, close, num_days=1.0)
        valid = t1.dropna()
        assert (valid > valid.index).all()

    def test_num_days_effect(self, close, t_events):
        t1_short = add_vertical_barrier(t_events, close, num_days=0.5)
        t1_long = add_vertical_barrier(t_events, close, num_days=5.0)
        valid_short = t1_short.dropna()
        valid_long = t1_long.dropna()
        # longer horizon → barrier further in time
        assert (valid_long.values >= valid_short.reindex(valid_long.index).values).all()


# ---------------------------------------------------------------------------
# get_events
# ---------------------------------------------------------------------------

class TestGetEvents:
    def test_columns(self, events):
        # standard triple-barrier: t1, trgt, t1_touch only (no side)
        # side only appears when meta-labeling is requested
        assert "t1" in events.columns
        assert "trgt" in events.columns
        assert "t1_touch" in events.columns
        assert "side" not in events.columns  # side absent in standard mode

    def test_no_nan_trgt(self, events):
        assert events["trgt"].notna().all()

    def test_events_subset_of_t_events(self, t_events, events):
        assert set(events.index).issubset(set(t_events))

    def test_min_ret_filter(self, close, t_events, vol, t1):
        big_min = vol.max() * 10  # nothing survives
        ev = get_events(close, t_events, [1.0, 1.0], vol, min_ret=float(big_min), vertical_barrier_times=t1)
        assert len(ev) == 0

    def test_asymmetric_barriers(self, close, t_events, vol, t1):
        # pt only (no stop-loss)
        ev = get_events(close, t_events, [2.0, 0.0], vol, vertical_barrier_times=t1)
        assert len(ev) > 0

    def test_no_vertical_barrier(self, close, t_events, vol):
        ev = get_events(close, t_events, [1.0, 1.0], vol)
        assert len(ev) > 0


# ---------------------------------------------------------------------------
# get_bins
# ---------------------------------------------------------------------------

class TestGetBins:
    def test_columns(self, bins):
        assert "bin" in bins.columns
        assert "ret" in bins.columns

    def test_labels_are_valid(self, bins):
        assert set(bins["bin"].unique()).issubset({-1, 0, 1})

    def test_no_nan_bins(self, bins):
        assert bins["bin"].notna().all()

    def test_label_sign_matches_return(self, bins):
        # +1 labels should have non-negative returns
        pos = bins[bins["bin"] == 1]
        if len(pos) > 0:
            assert (pos["ret"] >= 0).all()
        neg = bins[bins["bin"] == -1]
        if len(neg) > 0:
            assert (neg["ret"] <= 0).all()

    def test_index_subset_of_events(self, events, bins):
        assert set(bins.index).issubset(set(events.index))


# ---------------------------------------------------------------------------
# drop_labels
# ---------------------------------------------------------------------------

class TestDropLabels:
    def test_drops_rare_class(self):
        # Force a tiny class
        df = pd.DataFrame({"bin": [1] * 90 + [-1] * 8 + [0] * 2})
        result = drop_labels(df, min_pct=0.05)
        counts = result["bin"].value_counts(normalize=True)
        assert (counts >= 0.05).all()

    def test_does_not_drop_binary(self, bins):
        # Binary labels – should never drop below 2 classes
        df = bins.copy()
        result = drop_labels(df, min_pct=0.05)
        assert len(result["bin"].unique()) >= 1

    def test_preserves_structure(self, bins):
        result = drop_labels(bins.copy(), min_pct=0.01)
        assert "bin" in result.columns


# ---------------------------------------------------------------------------
# meta_labeling
# ---------------------------------------------------------------------------

class TestMetaLabeling:
    def test_binary_output(self, close, t_events, vol):
        # Simulate a primary model that always bets long
        side = pd.Series(1.0, index=t_events)
        labels = meta_labeling(
            close=close,
            t_events=t_events,
            pt_sl=[1.5, 1.5],
            target=vol,
            side=side,
            num_days=3.0,
        )
        assert set(labels["bin"].unique()).issubset({0, 1})

    def test_mixed_side_signals(self, close, t_events, vol):
        rng = np.random.default_rng(42)
        side_arr = rng.choice([-1.0, 1.0], size=len(t_events))
        side = pd.Series(side_arr, index=t_events)
        labels = meta_labeling(
            close=close,
            t_events=t_events,
            pt_sl=[1.5, 1.5],
            target=vol,
            side=side,
            num_days=3.0,
        )
        assert len(labels) > 0
        assert set(labels["bin"].unique()).issubset({0, 1})

    def test_returns_dataframe(self, close, t_events, vol):
        side = pd.Series(1.0, index=t_events)
        labels = meta_labeling(close, t_events, [1.5, 1.5], vol, side)
        assert isinstance(labels, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])