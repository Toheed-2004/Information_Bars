"""
Tests for mlfinlab.utils.helpers
"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mlfinlab.data.synthetic import make_ohlcv
from mlfinlab.utils.helpers import (
    log_returns,
    daily_vol,
    cusum_filter,
    get_vertical_barriers,
)


@pytest.fixture(scope="module")
def bars():
    return make_ohlcv(n_bars=500, freq="1h", seed=0)


@pytest.fixture(scope="module")
def close(bars):
    return bars["close"]


# ---------------------------------------------------------------------------
# log_returns
# ---------------------------------------------------------------------------

class TestLogReturns:
    def test_shape(self, close):
        ret = log_returns(close)
        assert len(ret) == len(close)

    def test_first_is_nan(self, close):
        ret = log_returns(close)
        assert np.isnan(ret.iloc[0])

    def test_values_correct(self, close):
        ret = log_returns(close, periods=1)
        expected = np.log(close.iloc[5] / close.iloc[4])
        assert abs(ret.iloc[5] - expected) < 1e-12

    def test_periods(self, close):
        ret = log_returns(close, periods=5)
        # first 5 should be NaN
        assert ret.iloc[:5].isna().all()

    def test_positive_trend_positive_return(self):
        s = pd.Series([1.0, 1.1, 1.2, 1.3])
        ret = log_returns(s)
        assert (ret.dropna() > 0).all()


# ---------------------------------------------------------------------------
# daily_vol
# ---------------------------------------------------------------------------

class TestDailyVol:
    def test_shape(self, close):
        vol = daily_vol(close)
        assert len(vol) == len(close)

    def test_all_positive(self, close):
        vol = daily_vol(close).dropna()
        assert (vol > 0).all()

    def test_span_override(self, close):
        vol = daily_vol(close, span=50)
        assert not vol.dropna().empty

    def test_no_index_loss(self, close):
        vol = daily_vol(close)
        assert vol.index.equals(close.index)


# ---------------------------------------------------------------------------
# CUSUM filter
# ---------------------------------------------------------------------------

class TestCUSUMFilter:
    def test_returns_datetimeindex(self, close):
        vol = daily_vol(close)
        events = cusum_filter(close, threshold=vol)
        assert isinstance(events, pd.DatetimeIndex)

    def test_events_subset_of_index(self, close):
        events = cusum_filter(close, threshold=0.01)
        assert set(events).issubset(set(close.index))

    def test_fixed_threshold(self, close):
        events = cusum_filter(close, threshold=0.02)
        assert len(events) > 0

    def test_high_threshold_fewer_events(self, close):
        e_low = cusum_filter(close, threshold=0.005)
        e_high = cusum_filter(close, threshold=0.05)
        assert len(e_low) >= len(e_high)

    def test_series_threshold(self, close):
        thresh = pd.Series(0.01, index=close.index)
        events = cusum_filter(close, threshold=thresh)
        assert isinstance(events, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Vertical barriers
# ---------------------------------------------------------------------------

class TestVerticalBarriers:
    def test_output_series(self, close):
        events = cusum_filter(close, threshold=0.01)
        t1 = get_vertical_barriers(close, events, num_days=2.0)
        assert isinstance(t1, pd.Series)

    def test_index_equals_events(self, close):
        events = cusum_filter(close, threshold=0.01)
        t1 = get_vertical_barriers(close, events, num_days=2.0)
        assert t1.index.equals(events)

    def test_barriers_after_events(self, close):
        events = cusum_filter(close, threshold=0.01)
        t1 = get_vertical_barriers(close, events, num_days=2.0)
        valid = t1.dropna()
        assert (valid > valid.index).all()

    def test_nat_for_late_events(self, close):
        # Create an event very close to the end
        late_event = pd.DatetimeIndex([close.index[-2]])
        t1 = get_vertical_barriers(close, late_event, num_days=10.0)
        assert t1.isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
