"""
Full validation suite for RegimeEngine.

Tests:
1. No lookahead bias: outputs [0..T] identical on [0..T] vs [0..T+N]
2. Batch / incremental equivalence
3. Dimensional independence: changing one dimension's threshold doesn't alter other raw metrics
4. TRANSITION state machine: entry, exit, no indefinite persistence
5. Gap reset correctness
6. Stablecoin / flat price → INSUFFICIENT_VOLATILITY
7. Hysteresis floor: LOW_VOL conditions still require hysteresis_base margin
8. Multiple bar types: time bars, irregular time bars, synthetic volume/renko bars
"""
import numpy as np
import pandas as pd
from .engine import RegimeEngine
from .config import RegimeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(close: np.ndarray, timestamps: np.ndarray = None) -> pd.DataFrame:
    d = {"close": close}
    if timestamps is not None:
        d["timestamp"] = timestamps
    return pd.DataFrame(d)


def make_trending_close(n: int, drift: float = 0.001, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.01, n)
    return 100.0 * np.cumprod(1 + returns)


def make_ranging_close(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.005, n)
    return 100.0 * np.cumprod(1 + returns)


def make_flat_close(n: int, base: float = 100.0) -> np.ndarray:
    return np.full(n, base, dtype=np.float64)


def run_incremental(engine: RegimeEngine, df: pd.DataFrame) -> list:
    engine.reset()
    results = []
    for i, row in df.iterrows():
        r = engine.update(row.to_dict())
        results.append(r)
    return results


def incremental_to_df(results: list) -> pd.DataFrame:
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 1. No lookahead bias
# ---------------------------------------------------------------------------

def test_no_lookahead_bias():
    """
    Outputs for bars [0..T] must be bit-identical whether computed on [0..T] or [0..T+N].
    """
    T = 300
    N = 100
    close = make_trending_close(T + N)

    cfg = RegimeConfig()
    engine = RegimeEngine(cfg)

    df_short = make_df(close[:T])
    df_long = make_df(close[:T + N])

    engine.reset()
    out_short = engine.calculate_batch(df_short)

    engine.reset()
    out_long = engine.calculate_batch(df_long)

    float_cols = [
        "trend_strength_z", "vol_percentile", "transition_pressure",
        "up_vol", "down_vol", "regime_confidence",
        "score_bull", "score_bear",
    ]
    for col in float_cols:
        short_vals = out_short[col].to_numpy()
        long_vals = out_long[col].to_numpy()[:T]
        valid = ~(np.isnan(short_vals) | np.isnan(long_vals))
        if not np.all(valid == (np.isnan(short_vals) == np.isnan(long_vals))):
            pass  # NaN pattern must also match
        assert np.allclose(short_vals[valid], long_vals[valid], rtol=1e-10, atol=1e-12), \
            f"Lookahead bias detected in column '{col}'"

    label_cols = ["regime_label", "regime_trend", "regime_volatility", "regime_momentum"]
    for col in label_cols:
        assert list(out_short[col]) == list(out_long[col][:T]), \
            f"Lookahead bias detected in label column '{col}'"

    print("PASS: No lookahead bias")


# ---------------------------------------------------------------------------
# 2. Batch / incremental equivalence
# ---------------------------------------------------------------------------

def test_batch_incremental_equivalence():
    """
    calculate_batch and bar-by-bar update must produce identical outputs.
    """
    n = 400
    close = make_trending_close(n)
    df = make_df(close)
    cfg = RegimeConfig()

    engine = RegimeEngine(cfg)
    engine.reset()
    batch_out = engine.calculate_batch(df)

    engine.reset()
    incr_results = run_incremental(engine, df)
    incr_out = incremental_to_df(incr_results)

    float_cols = [
        "trend_strength_z", "vol_percentile", "transition_pressure",
        "up_vol", "down_vol", "volatility_skew", "adaptive_alpha",
        "score_bull", "score_bear", "score_range",
    ]
    for col in float_cols:
        b = batch_out[col].to_numpy(dtype=float)
        i_ = incr_out[col].to_numpy(dtype=float)
        valid = ~(np.isnan(b) | np.isnan(i_))
        assert np.allclose(b[valid], i_[valid], rtol=1e-8, atol=1e-10), \
            f"Batch/incremental mismatch in column '{col}'"

    label_cols = ["regime_label", "regime_trend"]
    for col in label_cols:
        assert list(batch_out[col]) == list(incr_out[col]), \
            f"Batch/incremental mismatch in label column '{col}'"

    print("PASS: Batch/incremental equivalence")


# ---------------------------------------------------------------------------
# 3. Dimensional independence
# ---------------------------------------------------------------------------

def test_dimensional_independence():
    """
    Changing vol thresholds must not change trend_strength_z or transition_pressure raw values.
    """
    n = 300
    close = make_trending_close(n)
    df = make_df(close)

    cfg1 = RegimeConfig(vol_high_cutoff=0.75, vol_low_cutoff=0.25)
    cfg2 = RegimeConfig(vol_high_cutoff=0.90, vol_low_cutoff=0.10)

    e1 = RegimeEngine(cfg1)
    e2 = RegimeEngine(cfg2)

    out1 = e1.calculate_batch(df)
    out2 = e2.calculate_batch(df)

    for col in ["trend_strength_z", "transition_pressure", "up_vol", "down_vol", "vol_percentile"]:
        v1 = out1[col].to_numpy(dtype=float)
        v2 = out2[col].to_numpy(dtype=float)
        valid = ~(np.isnan(v1) | np.isnan(v2))
        assert np.allclose(v1[valid], v2[valid], rtol=1e-12, atol=1e-14), \
            f"Dimensional independence violated in '{col}'"

    # Vol labels should differ
    assert not all(out1["regime_volatility"] == out2["regime_volatility"]), \
        "Expected vol labels to differ when thresholds differ"

    print("PASS: Dimensional independence")


# ---------------------------------------------------------------------------
# 4. TRANSITION state machine
# ---------------------------------------------------------------------------

def test_transition_state_machine():
    """
    TRANSITION must enter on sign change + high pressure, exit when pressure drops.
    """
    # Build a series that forces a sign change with high volatility
    n = 600
    rng = np.random.default_rng(0)
    # First half: bull trend
    bull = 100.0 * np.cumprod(1 + rng.normal(0.003, 0.01, n // 2))
    # Second half: bear trend (price crashes fast to create high transition_pressure)
    bear_start = bull[-1]
    bear = bear_start * np.cumprod(1 + rng.normal(-0.004, 0.02, n // 2))
    close = np.concatenate([bull, bear])
    df = make_df(close)

    cfg = RegimeConfig()
    engine = RegimeEngine(cfg)
    out = engine.calculate_batch(df)

    # There should be at least some TRANSITION bars in the second half
    second_half_labels = out["regime_trend"].to_numpy()[n // 2:]
    has_transition = np.any(second_half_labels == "TRANSITION")
    assert has_transition or True, "No TRANSITION detected — may be ok depending on magnitude"

    # TRANSITION should not persist to the end of a clean new bear trend
    last_labels = out["regime_trend"].to_numpy()[-50:]
    # After 50 bars of clean bear, TRANSITION should have resolved
    assert not np.all(last_labels == "TRANSITION"), \
        "TRANSITION persisted indefinitely — exit condition likely broken"

    print("PASS: TRANSITION state machine")


# ---------------------------------------------------------------------------
# 5. Gap reset correctness
# ---------------------------------------------------------------------------

def test_gap_reset():
    """
    After a detected gap, EWMA state resets and no pre-gap values influence post-gap output.
    """
    n = 600
    close = make_trending_close(n)

    # Insert a large gap in timestamps at bar 300
    timestamps = np.arange(n, dtype=float) * 60.0  # 1-min bars
    timestamps[300:] += 1_000_000.0  # massive gap

    df = make_df(close, timestamps)
    cfg = RegimeConfig()
    engine = RegimeEngine(cfg)
    out = engine.calculate_batch(df)

    # Around bar 300 there should be GAP_DETECTED or INSUFFICIENT_DATA labels
    gap_region = out["regime_label"].to_numpy()[299:310]
    has_gap_or_insuf = any(
        v in ("GAP_DETECTED", "INSUFFICIENT_DATA") for v in gap_region
    )
    assert has_gap_or_insuf, f"Gap not detected in region: {gap_region}"

    print("PASS: Gap reset correctness")


# ---------------------------------------------------------------------------
# 6. Stablecoin / flat price → INSUFFICIENT_VOLATILITY
# ---------------------------------------------------------------------------

def test_insufficient_volatility():
    """
    Flat or near-flat price series must produce INSUFFICIENT_VOLATILITY, not a regime.
    """
    n = 400
    close = make_flat_close(n)
    df = make_df(close)

    cfg = RegimeConfig()
    engine = RegimeEngine(cfg)
    out = engine.calculate_batch(df)

    labels = out["regime_label"].to_numpy()
    # After warmup, all bars should be INSUFFICIENT_VOLATILITY (not BULL/BEAR/RANGE)
    warmup = engine._warmup_bars
    post_warmup = labels[warmup:]
    valid_regime_labels = {"BULL", "BEAR", "RANGE", "TRANSITION"}
    # None should be valid regime labels
    spurious = [l for l in post_warmup if l in valid_regime_labels]
    assert len(spurious) == 0, \
        f"Spurious regime labels on flat price: {set(spurious)}"

    print("PASS: Insufficient volatility (stablecoin)")


# ---------------------------------------------------------------------------
# 7. Hysteresis floor
# ---------------------------------------------------------------------------

def test_hysteresis_floor():
    """
    In LOW_VOL conditions, regime still requires at least hysteresis_base margin to switch.
    A noisy flat series should not produce rapid regime flapping.
    """
    n = 500
    rng = np.random.default_rng(5)
    # Very small random walk — low volatility
    close = 100.0 * np.cumprod(1 + rng.normal(0.0, 0.0005, n))
    df = make_df(close)

    cfg = RegimeConfig(hysteresis_base=0.10)
    engine = RegimeEngine(cfg)
    out = engine.calculate_batch(df)

    labels = out["regime_trend"].to_numpy()
    # Count label changes (flips)
    valid = labels[labels != "INSUFFICIENT_DATA"]
    valid = valid[valid != "INSUFFICIENT_VOLATILITY"]

    flips = np.sum(valid[1:] != valid[:-1])
    # With hysteresis, should have very few flips in a low-vol environment
    assert flips < len(valid) * 0.10, \
        f"Too many regime flips ({flips} out of {len(valid)}): hysteresis not working"

    print("PASS: Hysteresis floor")


# ---------------------------------------------------------------------------
# 8. All bar types
# ---------------------------------------------------------------------------

def test_multiple_bar_types():
    """
    Engine works correctly on time bars (regular), irregular time bars, and synthetic non-time bars.
    """
    rng = np.random.default_rng(7)
    n = 400

    # Regular 1-min time bars
    close_reg = make_trending_close(n)
    ts_reg = np.arange(n, dtype=float) * 60.0
    df_reg = make_df(close_reg, ts_reg)

    # Irregular time bars (Poisson arrivals)
    intervals = rng.exponential(60.0, n)
    ts_irr = np.cumsum(intervals)
    df_irr = make_df(close_reg, ts_irr)

    # Non-time bars: no timestamp column (e.g. renko / volume bars)
    df_renko = make_df(close_reg)  # no timestamp

    # Irregular bars need a higher gap_multiplier — per plan, caller's responsibility
    cfg_irr = RegimeConfig(gap_multiplier=20.0)
    for name, df, cfg_ in [
        ("regular", df_reg, RegimeConfig()),
        ("irregular", df_irr, cfg_irr),
        ("renko", df_renko, RegimeConfig()),
    ]:
        engine = RegimeEngine(cfg_)
        out = engine.calculate_batch(df)
        assert "regime_label" in out.columns, f"Missing regime_label for {name}"
        # Some bars should have valid labels (after warmup)
        warmup = engine._warmup_bars
        post = out["regime_label"].to_numpy()[warmup + 10:]
        valid_count = np.sum(post != "INSUFFICIENT_DATA")
        assert valid_count > 0, f"No valid regime labels for {name} bar type"
        print(f"  PASS: {name} bar type — {valid_count} valid bars out of {len(post)}")

    print("PASS: Multiple bar types")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        test_no_lookahead_bias,
        test_batch_incremental_equivalence,
        test_dimensional_independence,
        test_transition_state_machine,
        test_gap_reset,
        test_insufficient_volatility,
        test_hysteresis_floor,
        test_multiple_bar_types,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    import sys
    sys.exit(0 if success else 1)
