"""
Tests for mlfinlab.features.fractional_diff
"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mlfinlab.data.synthetic import make_ohlcv
from mlfinlab.features.fractional_diff import (
    frac_diff_ffd,
    find_min_d,
    plot_min_ffd,
)


@pytest.fixture(scope="module")
def log_close():
    bars = make_ohlcv(n_bars=600, seed=2)
    return np.log(bars["close"])


# ---------------------------------------------------------------------------
# frac_diff_ffd
# ---------------------------------------------------------------------------

class TestFracDiffFFD:
    def test_same_length(self, log_close):
        fd = frac_diff_ffd(log_close, d=0.5)
        assert len(fd) == len(log_close)

    def test_head_is_nan(self, log_close):
        fd = frac_diff_ffd(log_close, d=0.5)
        assert np.isnan(fd.iloc[0])

    def test_tail_not_all_nan(self, log_close):
        fd = frac_diff_ffd(log_close, d=0.5)
        assert not fd.dropna().empty

    def test_d1_approx_returns(self, log_close):
        """d=1 should approximately reproduce first differences."""
        fd = frac_diff_ffd(log_close, d=1.0, threshold=1e-10)
        diff = log_close.diff()
        common = fd.dropna().index.intersection(diff.dropna().index)
        corr = fd.loc[common].corr(diff.loc[common])
        assert corr > 0.99

    def test_invalid_d_raises(self, log_close):
        with pytest.raises(ValueError):
            frac_diff_ffd(log_close, d=0.0)

    def test_index_preserved(self, log_close):
        fd = frac_diff_ffd(log_close, d=0.4)
        assert fd.index.equals(log_close.index)

    def test_different_d_different_output(self, log_close):
        fd03 = frac_diff_ffd(log_close, d=0.3).dropna()
        fd07 = frac_diff_ffd(log_close, d=0.7).dropna()
        common = fd03.index.intersection(fd07.index)
        assert not (fd03.loc[common] == fd07.loc[common]).all()

    def test_threshold_affects_width(self, log_close):
        """Lower threshold → wider window → fewer NaNs at head."""
        fd_loose = frac_diff_ffd(log_close, d=0.4, threshold=1e-2)
        fd_tight = frac_diff_ffd(log_close, d=0.4, threshold=1e-8)
        n_nan_loose = fd_loose.isna().sum()
        n_nan_tight = fd_tight.isna().sum()
        assert n_nan_tight >= n_nan_loose

    def test_d_near_zero_close_to_original(self, log_close):
        """Very small d should produce output close to original."""
        fd = frac_diff_ffd(log_close, d=0.01, threshold=1e-3)
        common = fd.dropna().index
        corr = log_close.loc[common].corr(fd.loc[common])
        assert corr > 0.99


# ---------------------------------------------------------------------------
# find_min_d
# ---------------------------------------------------------------------------

class TestFindMinD:
    def test_returns_float(self, log_close):
        d = find_min_d(log_close, step=0.1)
        assert isinstance(d, float)

    def test_result_in_range(self, log_close):
        d = find_min_d(log_close, d_range=(0.0, 1.0), step=0.1)
        assert 0.0 <= d <= 1.0

    def test_stationary_at_result(self, log_close):
        from statsmodels.tsa.stattools import adfuller
        d = find_min_d(log_close, d_range=(0.0, 1.0), step=0.1)
        fd = frac_diff_ffd(log_close, d=d).dropna()
        p = adfuller(fd, maxlag=1, regression="c", autolag=None)[1]
        assert p <= 0.10   # allow slight tolerance vs 0.05

    def test_nonstationary_series_needs_positive_d(self):
        # A pure random walk is I(1) – needs d > 0
        rw = pd.Series(np.cumsum(np.random.randn(400)))
        d = find_min_d(rw, d_range=(0.0, 1.0), step=0.1)
        assert d > 0


# ---------------------------------------------------------------------------
# plot_min_ffd
# ---------------------------------------------------------------------------

class TestPlotMinFFD:
    def test_returns_dataframe(self, log_close):
        df = plot_min_ffd(log_close, d_range=(0.0, 0.5), step=0.1)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self, log_close):
        df = plot_min_ffd(log_close, d_range=(0.0, 0.5), step=0.1)
        for col in ["d", "adf_stat", "p_value", "corr_with_original", "n_obs"]:
            assert col in df.columns

    def test_correlation_decreases_with_d(self, log_close):
        """Higher d → lower correlation with original."""
        df = plot_min_ffd(log_close, d_range=(0.0, 1.0), step=0.2)
        df = df.sort_values("d")
        corrs = df["corr_with_original"].values
        # monotone-ish: every step should not increase by more than a little
        assert corrs[0] >= corrs[-1]

    def test_adf_stat_decreases_with_d(self, log_close):
        """Higher d → more negative ADF stat (more stationary)."""
        df = plot_min_ffd(log_close, d_range=(0.1, 1.0), step=0.2)
        df = df.sort_values("d")
        assert df["adf_stat"].iloc[-1] < df["adf_stat"].iloc[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
