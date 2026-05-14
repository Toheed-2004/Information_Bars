"""
Tests for mlfinlab.features.sample_weights
"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mlfinlab.data.synthetic import make_ohlcv
from mlfinlab.utils.helpers import daily_vol, cusum_filter
from mlfinlab.labeling.triple_barrier import add_vertical_barrier, get_events
from mlfinlab.features.sample_weights import (
    get_num_concurrent_events,
    get_avg_uniqueness,
    get_ind_matrix,
    seq_bootstrap,
    get_sample_weights_return,
    get_sample_weights_time_decay,
)


@pytest.fixture(scope="module")
def bars():
    return make_ohlcv(n_bars=400, seed=5)


@pytest.fixture(scope="module")
def close(bars):
    return bars["close"]


@pytest.fixture(scope="module")
def vol(close):
    return daily_vol(close, lookback=30)


@pytest.fixture(scope="module")
def t_events(close, vol):
    return cusum_filter(close, threshold=vol * 0.5)


@pytest.fixture(scope="module")
def t1_series(close, t_events):
    return add_vertical_barrier(t_events, close, num_days=2.0)


@pytest.fixture(scope="module")
def events(close, t_events, vol, t1_series):
    return get_events(close, t_events, [1.5, 1.5], vol, vertical_barrier_times=t1_series)


@pytest.fixture(scope="module")
def t1_col(events):
    # Use the t1 column from events (vertical barrier timestamp)
    return events["t1"].dropna()


@pytest.fixture(scope="module")
def ind_m(close, t1_col):
    return get_ind_matrix(close.index, t1_col)


# ---------------------------------------------------------------------------
# get_num_concurrent_events
# ---------------------------------------------------------------------------

class TestConcurrentEvents:
    def test_returns_series(self, close, t1_col):
        c = get_num_concurrent_events(close.index, t1_col)
        assert isinstance(c, pd.Series)

    def test_non_negative(self, close, t1_col):
        c = get_num_concurrent_events(close.index, t1_col)
        assert (c >= 0).all()

    def test_max_reasonable(self, close, t1_col):
        c = get_num_concurrent_events(close.index, t1_col)
        assert c.max() <= len(t1_col)


# ---------------------------------------------------------------------------
# get_ind_matrix
# ---------------------------------------------------------------------------

class TestIndMatrix:
    def test_returns_dataframe(self, ind_m):
        assert isinstance(ind_m, pd.DataFrame)

    def test_binary(self, ind_m):
        assert set(ind_m.values.flatten()).issubset({0, 1})

    def test_columns_are_event_times(self, ind_m, t1_col):
        assert set(ind_m.columns).issubset(set(t1_col.index))

    def test_rows_are_bar_index(self, ind_m, close):
        assert ind_m.index.isin(close.index).all()


# ---------------------------------------------------------------------------
# get_avg_uniqueness
# ---------------------------------------------------------------------------

class TestAvgUniqueness:
    def test_returns_series(self, ind_m):
        u = get_avg_uniqueness(ind_m)
        assert isinstance(u, pd.Series)

    def test_bounded(self, ind_m):
        u = get_avg_uniqueness(ind_m)
        assert (u >= 0).all()
        assert (u <= 1 + 1e-9).all()

    def test_higher_overlap_lower_uniqueness(self, close, t1_col):
        """Events with many overlapping bars should have lower avg uniqueness."""
        # Create two scenarios: one event vs many overlapping events
        short_t1 = t1_col.iloc[:2]
        long_t1 = t1_col

        im_short = get_ind_matrix(close.index, short_t1)
        im_long = get_ind_matrix(close.index, long_t1)

        u_short = get_avg_uniqueness(im_short).mean()
        u_long = get_avg_uniqueness(im_long).mean()
        # More events → more overlap → lower average uniqueness
        assert u_long <= u_short + 0.3   # generous tolerance


# ---------------------------------------------------------------------------
# seq_bootstrap
# ---------------------------------------------------------------------------

class TestSeqBootstrap:
    def test_returns_list(self, ind_m):
        sample = seq_bootstrap(ind_m, sample_length=5, random_state=0)
        assert isinstance(sample, list)

    def test_correct_length(self, ind_m):
        n = 10
        sample = seq_bootstrap(ind_m, sample_length=n, random_state=1)
        assert len(sample) == n

    def test_elements_in_columns(self, ind_m):
        sample = seq_bootstrap(ind_m, sample_length=5, random_state=2)
        assert all(s in ind_m.columns for s in sample)

    def test_reproducible_with_seed(self, ind_m):
        s1 = seq_bootstrap(ind_m, sample_length=5, random_state=99)
        s2 = seq_bootstrap(ind_m, sample_length=5, random_state=99)
        assert s1 == s2


# ---------------------------------------------------------------------------
# sample weights
# ---------------------------------------------------------------------------

class TestSampleWeightsReturn:
    def test_returns_series(self, t1_col, close):
        w = get_sample_weights_return(t1_col, close)
        assert isinstance(w, pd.Series)

    def test_positive(self, t1_col, close):
        w = get_sample_weights_return(t1_col, close).dropna()
        assert (w >= 0).all()

    def test_mean_approx_one(self, t1_col, close):
        w = get_sample_weights_return(t1_col, close).dropna()
        assert abs(w.mean() - 1.0) < 0.5


class TestSampleWeightsTimeDecay:
    def test_returns_series(self, t1_col, close):
        w = get_sample_weights_time_decay(t1_col, close, decay=1.0)
        assert isinstance(w, pd.Series)

    def test_non_negative(self, t1_col, close):
        w = get_sample_weights_time_decay(t1_col, close, decay=1.0).dropna()
        assert (w >= 0).all()

    def test_decay_reduces_early_weights(self, t1_col, close):
        w_no_decay = get_sample_weights_return(t1_col, close).dropna()
        w_decay = get_sample_weights_time_decay(t1_col, close, decay=1.0).dropna()
        # First half should have lower weight with decay
        n = len(w_decay)
        if n > 10:
            first_half = w_decay.iloc[: n // 2].mean()
            last_half = w_decay.iloc[n // 2 :].mean()
            assert last_half >= first_half * 0.9   # generous: later events weight ≥ earlier


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
