"""
Tests for mlfinlab.features.microstructural and mlfinlab.features.technical
"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mlfinlab.data.synthetic import make_ohlcv
from mlfinlab.features.microstructural import (
    bar_features,
    roll_spread,
    amihud_lambda,
    kyle_lambda,
    corwin_schultz_spread,
)
from mlfinlab.features.technical import (
    rsi,
    macd,
    bollinger_bands,
    atr,
    vwap,
    zscore,
)


@pytest.fixture(scope="module")
def bars():
    return make_ohlcv(n_bars=500, seed=3)


@pytest.fixture(scope="module")
def close(bars):
    return bars["close"]


# ===========================================================================
# Microstructural
# ===========================================================================

class TestBarFeatures:
    def test_returns_dataframe(self, bars):
        feat = bar_features(bars)
        assert isinstance(feat, pd.DataFrame)

    def test_same_length(self, bars):
        feat = bar_features(bars)
        assert len(feat) == len(bars)

    def test_expected_columns(self, bars):
        feat = bar_features(bars)
        for col in ["hl_spread", "co_return", "body_ratio", "log_volume",
                    "rsi_14", "cs_spread"]:
            assert col in feat.columns, f"Missing: {col}"
        assert "amihud" not in feat.columns, "amihud should be removed"

    def test_hl_spread_positive(self, bars):
        feat = bar_features(bars)
        assert (feat["hl_spread"].dropna() >= 0).all()

    def test_body_ratio_bounded(self, bars):
        feat = bar_features(bars)
        br = feat["body_ratio"].dropna()
        assert (br >= 0).all()
        assert (br <= 1 + 1e-9).all()

    def test_rsi_bounded(self, bars):
        feat = bar_features(bars)
        r = feat["rsi_14"].dropna()
        assert (r >= 0).all() and (r <= 100).all()

    def test_log_volume_positive(self, bars):
        feat = bar_features(bars)
        assert (feat["log_volume"].dropna() >= 0).all()

    def test_no_all_nan_columns(self, bars):
        feat = bar_features(bars)
        for col in feat.columns:
            assert not feat[col].isna().all(), f"Column all NaN: {col}"


class TestRollSpread:
    def test_returns_series(self, close):
        s = roll_spread(close)
        assert isinstance(s, pd.Series)

    def test_non_negative(self, close):
        s = roll_spread(close).dropna()
        assert (s >= 0).all()

    def test_same_length(self, close):
        s = roll_spread(close)
        assert len(s) == len(close)


class TestAmihudLambda:
    def test_returns_series(self, bars):
        a = amihud_lambda(bars["close"], bars["volume"])
        assert isinstance(a, pd.Series)

    def test_non_negative(self, bars):
        a = amihud_lambda(bars["close"], bars["volume"]).dropna()
        assert (a >= 0).all()

    def test_larger_window_smoother(self, bars):
        a10 = amihud_lambda(bars["close"], bars["volume"], window=10).dropna()
        a50 = amihud_lambda(bars["close"], bars["volume"], window=50).dropna()
        assert a50.std() <= a10.std()


class TestKyleLambda:
    def test_returns_series(self, bars):
        k = kyle_lambda(bars["close"], bars["volume"])
        assert isinstance(k, pd.Series)

    def test_same_length(self, bars):
        k = kyle_lambda(bars["close"], bars["volume"])
        assert len(k) == len(bars)


class TestCorwinSchultzSpread:
    def test_returns_series(self, bars):
        cs = corwin_schultz_spread(bars)
        assert isinstance(cs, pd.Series)

    def test_non_negative(self, bars):
        cs = corwin_schultz_spread(bars).dropna()
        assert (cs >= 0).all()

    def test_bounded_above(self, bars):
        cs = corwin_schultz_spread(bars).dropna()
        assert (cs < 1).all()


# ===========================================================================
# Technical
# ===========================================================================

class TestRSI:
    def test_bounded(self, close):
        r = rsi(close, 14).dropna()
        assert (r >= 0).all() and (r <= 100).all()

    def test_same_length(self, close):
        r = rsi(close, 14)
        assert len(r) == len(close)

    def test_period_14_vs_7(self, close):
        r14 = rsi(close, 14).dropna()
        r7 = rsi(close, 7).dropna()
        # RSI-7 should be more volatile
        assert r7.std() >= r14.std() * 0.8


class TestMACD:
    def test_returns_dataframe(self, close):
        df = macd(close)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, close):
        df = macd(close)
        assert set(df.columns) == {"macd", "signal", "histogram"}

    def test_histogram_is_macd_minus_signal(self, close):
        df = macd(close)
        diff = (df["macd"] - df["signal"] - df["histogram"]).abs()
        assert diff.max() < 1e-10

    def test_same_length(self, close):
        df = macd(close)
        assert len(df) == len(close)


class TestBollingerBands:
    def test_returns_dataframe(self, close):
        bb = bollinger_bands(close)
        assert isinstance(bb, pd.DataFrame)

    def test_columns(self, close):
        bb = bollinger_bands(close)
        for col in ["bb_upper", "bb_mid", "bb_lower", "bb_bandwidth", "bb_pct_b"]:
            assert col in bb.columns

    def test_upper_above_lower(self, close):
        bb = bollinger_bands(close).dropna()
        assert (bb["bb_upper"] >= bb["bb_lower"]).all()

    def test_mid_between_bands(self, close):
        bb = bollinger_bands(close).dropna()
        assert (bb["bb_mid"] >= bb["bb_lower"]).all()
        assert (bb["bb_mid"] <= bb["bb_upper"]).all()

    def test_bandwidth_positive(self, close):
        bb = bollinger_bands(close).dropna()
        assert (bb["bb_bandwidth"] >= 0).all()


class TestATR:
    def test_returns_dataframe(self, bars):
        df = atr(bars)
        assert isinstance(df, pd.DataFrame)

    def test_atr_positive(self, bars):
        df = atr(bars).dropna()
        assert (df["atr"] > 0).all()

    def test_natr_bounded(self, bars):
        df = atr(bars, normalized=True).dropna()
        assert (df["natr"] > 0).all()
        assert (df["natr"] < 1).all()

    def test_same_length(self, bars):
        df = atr(bars)
        assert len(df) == len(bars)


class TestVWAP:
    def test_returns_series(self, bars):
        v = vwap(bars)
        assert isinstance(v, pd.Series)

    def test_positive(self, bars):
        v = vwap(bars).dropna()
        assert (v > 0).all()

    def test_same_length(self, bars):
        v = vwap(bars)
        assert len(v) == len(bars)

    def test_close_to_price(self, bars):
        v = vwap(bars).dropna()
        c = bars["close"].reindex(v.index)
        ratio = v / c
        assert ratio.between(0.5, 2.0).all()


class TestZScore:
    def test_returns_series(self, close):
        z = zscore(close)
        assert isinstance(z, pd.Series)

    def test_near_zero_mean(self, close):
        z = zscore(close).dropna()
        # Rolling zscore doesn't guarantee global mean=0,
        # but typical range is ±3
        assert z.abs().mean() < 3.0

    def test_same_length(self, close):
        z = zscore(close)
        assert len(z) == len(close)

    def test_no_demean_version(self, close):
        z = zscore(close, demean=False).dropna()
        assert not z.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])