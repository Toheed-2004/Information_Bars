"""
Integration test: full pipeline
  raw OHLCV → events → labels → features → sample weights

Verifies that all modules compose correctly and produce a
machine-learning-ready DataFrame suitable for downstream models.
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
)
from mlfinlab.labeling.trend_scanning import trend_scanning_labels
from mlfinlab.features.fractional_diff import frac_diff_ffd, find_min_d
from mlfinlab.features.microstructural import bar_features
from mlfinlab.features.technical import (
    rsi, macd, bollinger_bands, atr, vwap, zscore
)
from mlfinlab.features.sample_weights import (
    get_ind_matrix,
    get_avg_uniqueness,
    get_sample_weights_time_decay,
)


@pytest.fixture(scope="module")
def bars():
    return make_ohlcv(n_bars=1000, freq="1h", seed=42)


class TestFullPipeline:
    """End-to-end pipeline producing an ML-ready dataset."""

    def test_pipeline_runs(self, bars):
        close = bars["close"]

        # ----- Step 1: Event Sampling -----
        vol = daily_vol(close, lookback=50)
        t_events = cusum_filter(close, threshold=vol * 0.5)
        assert len(t_events) > 10, "Too few events sampled"

        # ----- Step 2: Triple-barrier labeling -----
        t1 = add_vertical_barrier(t_events, close, num_days=3.0)
        events = get_events(
            close=close,
            t_events=t_events,
            pt_sl=[2.0, 2.0],
            target=vol,
            min_ret=0.0,
            vertical_barrier_times=t1,
        )
        bins = get_bins(events, close)
        bins = drop_labels(bins, min_pct=0.05)
        assert len(bins) > 5
        assert set(bins["bin"].unique()).issubset({-1, 0, 1})

        # ----- Step 3: Feature engineering -----
        # Fractional differencing
        log_close = np.log(close)
        min_d = find_min_d(log_close, step=0.1, verbose=False)
        fd_close = frac_diff_ffd(log_close, d=max(min_d, 0.1)).rename("frac_diff")

        # Technical indicators
        rsi_feat = rsi(close, 14)
        macd_feat = macd(close)
        bb_feat = bollinger_bands(close, 20)
        atr_feat = atr(bars)
        vwap_feat = vwap(bars, 20)
        z_feat = zscore(close, 20)

        # Microstructural
        micro = bar_features(bars)

        # ----- Step 4: Assemble feature matrix -----
        features = pd.concat(
            [fd_close, rsi_feat, macd_feat, bb_feat, atr_feat, vwap_feat, z_feat],
            axis=1,
        )
        # Align to event timestamps
        X = features.reindex(bins.index)
        y = bins["bin"]

        # ----- Step 5: Sample weights -----
        t1_valid = events["t1"].dropna()
        # Use only events present in bins
        t1_for_weights = t1_valid.reindex(bins.index).dropna()
        if len(t1_for_weights) > 3:
            w = get_sample_weights_time_decay(t1_for_weights, close, decay=1.0)
            w = w.reindex(bins.index).fillna(w.mean())
        else:
            w = pd.Series(1.0, index=bins.index)

        # ----- Assertions -----
        assert len(X) == len(y), "Feature/label length mismatch"
        assert len(w) == len(y), "Weight/label length mismatch"
        assert not X.empty
        assert not y.empty

        # No label should be all-NaN
        assert y.notna().all()

        # Weights should be positive
        assert (w > 0).all()

        print(f"\n  Events: {len(t_events)}, Labels: {len(bins)}")
        print(f"  Label distribution:\n{y.value_counts()}")
        print(f"  Feature shape: {X.shape}")
        print(f"  Min d (FFD): {min_d:.3f}")
        print(f"  Weight stats: mean={w.mean():.3f}  std={w.std():.3f}")

    def test_trend_scanning_pipeline(self, bars):
        close = bars["close"]
        vol = daily_vol(close, lookback=50)
        t_events = cusum_filter(close, threshold=vol * 0.5)

        # Trend-scanning labels as alternative to triple-barrier
        ts_labels = trend_scanning_labels(
            close,
            t_events,
            look_forward_window=20,
            min_sample_length=5,
            t_value_threshold=1.5,
        )
        assert isinstance(ts_labels, pd.DataFrame)
        assert "bin" in ts_labels.columns
        assert "t_value" in ts_labels.columns
        assert set(ts_labels["bin"].unique()).issubset({-1, 0, 1})

    def test_feature_matrix_no_lookahead(self, bars):
        """All features must use only past data (no lookahead bias)."""
        close = bars["close"]

        # Each feature computed at time t uses only data ≤ t
        rsi_feat = rsi(close, 14)
        fd = frac_diff_ffd(np.log(close), d=0.4)

        # If there were lookahead, correlations with future returns would be
        # implausibly high on an i.i.d. GBM series.
        future_ret = np.log(close / close.shift(-1)).shift(0)
        common = rsi_feat.dropna().index.intersection(future_ret.dropna().index)

        if len(common) < 20:
            pytest.skip("Not enough common observations for correlation test")

        corr_rsi = abs(rsi_feat.loc[common].corr(future_ret.loc[common]))
        fd_valid = fd.dropna()
        common_fd = fd_valid.index.intersection(future_ret.dropna().index)
        corr_fd = abs(fd_valid.loc[common_fd].corr(future_ret.loc[common_fd])) if len(common_fd) > 5 else 0.0

        # On a GBM, neither should have >0.15 correlation with future returns
        assert corr_rsi < 0.15, f"RSI lookahead suspicion: corr={corr_rsi:.3f}"
        assert corr_fd < 0.15, f"FFD lookahead suspicion: corr={corr_fd:.3f}"

    def test_bar_type_agnostic(self):
        """All modules should work on differently-sized bar series."""
        for n in [100, 300, 1000]:
            b = make_ohlcv(n_bars=n, freq="5min", seed=n)
            c = b["close"]
            vol = daily_vol(c, lookback=20)
            evts = cusum_filter(c, threshold=vol * 0.5)
            if len(evts) == 0:
                continue
            t1 = add_vertical_barrier(evts, c, num_days=1.0)
            ev = get_events(c, evts, [1.5, 1.5], vol, vertical_barrier_times=t1)
            if ev.empty:
                continue
            bn = get_bins(ev, c)
            assert not bn.empty
            fd = frac_diff_ffd(np.log(c), d=0.4)
            assert len(fd) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
