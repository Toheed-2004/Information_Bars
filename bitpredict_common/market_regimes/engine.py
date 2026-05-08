"""
RegimeEngine — main entry point.

Two modes that produce identical outputs:
  - calculate_batch(df) → pd.DataFrame
  - update(bar: dict)   → dict

Both respect:
  - Zero lookahead bias
  - Causal z-score (uses prev bar's EWMA stats)
  - Causal ring-buffer percentile (ranks current against past-only)
  - No Python loops over bars in batch mode (where possible)
"""
import numpy as np
import pandas as pd
from typing import Optional

from bitpredict.common.market_regimes.config import RegimeConfig
from bitpredict.common.market_regimes.state import RegimeState
from bitpredict.common.market_regimes.gap import (
    detect_gap, reset_state_on_gap,
    compute_warmup_bars, is_warmup_complete,
)
from bitpredict.common.market_regimes.metrics.trend import compute_trend_batch, update_trend
from bitpredict.common.market_regimes.metrics.volatility import compute_volatility_batch, update_volatility
from bitpredict.common.market_regimes.metrics.momentum import compute_momentum_batch, update_momentum
from bitpredict.common.market_regimes.classification.dimensions import (
    classify_trend_raw, classify_vol, classify_momentum,
    classify_trend_raw_batch, classify_vol_batch, classify_momentum_batch,
)
from bitpredict.common.market_regimes.classification.transition import update_transition_state, compute_transition_batch
from bitpredict.common.market_regimes.classification.hysteresis import (
    apply_hysteresis_and_min_duration,
    apply_hysteresis_and_min_duration_batch,
    activation_distance_trend,
    activation_distance_vol,
    activation_distance_momentum,
)
from bitpredict.common.market_regimes.classification.combiner import combine_labels, combine_labels_batch
from bitpredict.common.market_regimes.outputs.confidence import compute_confidence, compute_confidence_batch
from bitpredict.common.market_regimes.outputs.soft_scores import compute_soft_scores, compute_soft_scores_batch
from bitpredict.common.market_regimes.outputs.stability import update_stability, compute_stability_batch
from bitpredict.common.logging import get_logger
logger = get_logger(__name__)
_INSUFFICIENT = "INSUFFICIENT_DATA"
_GAP = "GAP_DETECTED"
_INSUF_VOL = "INSUFFICIENT_VOLATILITY"


def _datetime_col_to_epoch(col: pd.Series) -> np.ndarray:
    """Convert a datetime column (Timestamp, datetime, or tz-aware) to float seconds since epoch."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return col.astype("int64").to_numpy() / 1e9
    # Already numeric (e.g. milliseconds stored as int) — try direct float conversion
    try:
        return col.astype(np.float64).to_numpy()
    except Exception:
        return pd.to_datetime(col).astype("int64").to_numpy() / 1e9


def _extract_bar_timestamp(bar: dict) -> Optional[float]:
    """Extract a float epoch-seconds timestamp from a bar dict, supporting both 'timestamp' and 'datetime' keys."""
    if "timestamp" in bar:
        ts = float(bar["timestamp"])
        logger.debug(f"_extract_bar_timestamp: Using 'timestamp' key = {ts}")
        return ts
    if "datetime" in bar:
        val = bar["datetime"]
        if hasattr(val, "timestamp"):        # pandas Timestamp or datetime object
            ts = val.timestamp()
            logger.debug(f"_extract_bar_timestamp: Using 'datetime' key (has .timestamp()) = {ts}")
            return ts
        try:
            ts = float(pd.Timestamp(val).timestamp())
            logger.debug(f"_extract_bar_timestamp: Using 'datetime' key (pd.Timestamp) = {ts}")
            return ts
        except Exception as e:
            logger.warning(f"_extract_bar_timestamp: Failed to parse datetime: {e}")
            return None
    logger.warning(f"_extract_bar_timestamp: No 'timestamp' or 'datetime' key found in bar")
    return None


class RegimeEngine:
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._state = RegimeState()
        self._warmup_bars = compute_warmup_bars(
            self.config.alpha_min, self.config.warmup_epsilon
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        self._state = RegimeState()

    @property
    def state(self) -> RegimeState:
        return self._state

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full historical calculation over a DataFrame with a 'close' column.
        Optionally uses 'timestamp' column (seconds) for gap detection.
        Returns all three output layers merged with input columns.
        """
        cfg = self.config
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        
        logger.info(f"CALCULATE_BATCH START: n={n} bars, warmup_bars={self._warmup_bars}, "
                   f"current_bars_seen={self._state.bars_seen}, current_warmup_complete={self._state.warmup_complete}")

        timestamps = None
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].to_numpy(dtype=np.float64)
        elif "datetime" in df.columns:
            timestamps = _datetime_col_to_epoch(df["datetime"])

        if timestamps is not None:
            intervals = np.empty(n)
            intervals[0] = 0.0
            intervals[1:] = np.diff(timestamps)
        else:
            intervals = np.ones(n)

        # ---- Compute all raw metrics (vectorized) ----
        trend_m = compute_trend_batch(
            close, cfg.alpha_fast, cfg.alpha_slow, cfg.alpha_z, cfg.alpha_persistence
        )
        vol_m = compute_volatility_batch(
            close, cfg.alpha_vol, cfg.ring_buffer_size, cfg.alpha_min, cfg.alpha_max
        )
        mom_m = compute_momentum_batch(
            vol_m["volatility_level"], trend_m["trend_strength_z"], cfg.alpha_expansion
        )

        trend_strength_z = trend_m["trend_strength_z"]
        vol_percentile = vol_m["vol_percentile"]
        transition_pressure = mom_m["transition_pressure"]

        # ---- Warmup mask ----
        # In incremental, bar i is the first valid bar when bars_seen = i+1 >= _warmup_bars,
        # i.e. i >= _warmup_bars - 1. So bars 0.._warmup_bars-2 are warmup.
        warmup_mask = np.arange(n) < (self._warmup_bars - 1)

        # ---- Gap detection (sequential — depends on prior intervals) ----
        gap_mask, final_ewma_interval, final_initialized = self._detect_gaps_batch(intervals, cfg.gap_multiplier)
        # After a gap, warmup restarts. Expand warmup_mask to cover post-gap warmup windows.
        gap_mask, warmup_mask = self._expand_warmup_after_gaps(gap_mask, warmup_mask, n)

        # ---- TRANSITION state machine ----
        in_transition = compute_transition_batch(
            trend_strength_z,
            transition_pressure,
            cfg.transition_high_threshold,
            cfg.transition_exit_threshold,
        )

        # ---- Raw dimension labels ----
        raw_trend = classify_trend_raw_batch(trend_strength_z, cfg.trend_threshold)
        raw_vol = classify_vol_batch(vol_percentile, cfg.vol_high_cutoff, cfg.vol_low_cutoff)
        raw_momentum = classify_momentum_batch(transition_pressure, cfg.accel_threshold)

        # Override trend with TRANSITION where applicable
        raw_trend[in_transition] = "TRANSITION"

        # ---- Hysteresis + min_duration (sequential state machine) ----
        # Only applied to post-warmup bars — matches incremental mode which never
        # touches hysteresis during warmup (committed starts as None at first valid bar).
        ws = self._warmup_bars - 1  # index of first valid bar

        dist_trend = activation_distance_trend(trend_strength_z, cfg.trend_threshold)
        dist_vol = activation_distance_vol(vol_percentile, cfg.vol_high_cutoff, cfg.vol_low_cutoff)
        dist_mom = activation_distance_momentum(transition_pressure, cfg.accel_threshold)

        committed_trend = np.full(n, _INSUFFICIENT, dtype=object)
        committed_vol = np.full(n, _INSUFFICIENT, dtype=object)
        committed_momentum = np.full(n, _INSUFFICIENT, dtype=object)

        if n > ws:
            committed_trend[ws:] = apply_hysteresis_and_min_duration_batch(
                raw_trend[ws:], dist_trend[ws:], vol_percentile[ws:],
                cfg.hysteresis_base, cfg.hysteresis_k, cfg.min_duration_bars,
            )
            committed_vol[ws:] = apply_hysteresis_and_min_duration_batch(
                raw_vol[ws:], dist_vol[ws:], vol_percentile[ws:],
                cfg.hysteresis_base, cfg.hysteresis_k, cfg.min_duration_bars,
            )
            committed_momentum[ws:] = apply_hysteresis_and_min_duration_batch(
                raw_momentum[ws:], dist_mom[ws:], vol_percentile[ws:],
                cfg.hysteresis_base, cfg.hysteresis_k, cfg.min_duration_bars,
            )

        # TRANSITION always overrides hysteresis for trend
        committed_trend[in_transition] = "TRANSITION"

        # ---- Combined label ----
        regime_label = combine_labels_batch(committed_trend, committed_vol, committed_momentum)

        # ---- INSUFFICIENT_VOLATILITY ----
        insuf_vol_mask = self._compute_insuf_vol_mask(
            vol_m["volatility_level"], cfg.volatility_floor, cfg.volatility_floor_bars
        )
        regime_label[insuf_vol_mask & ~warmup_mask & ~gap_mask] = _INSUF_VOL
        committed_trend[insuf_vol_mask & ~warmup_mask & ~gap_mask] = _INSUF_VOL
        committed_vol[insuf_vol_mask & ~warmup_mask & ~gap_mask] = _INSUF_VOL
        committed_momentum[insuf_vol_mask & ~warmup_mask & ~gap_mask] = _INSUF_VOL

        # ---- Confidence ----
        conf_out = compute_confidence_batch(
            trend_strength_z, vol_percentile, transition_pressure,
            cfg.trend_threshold, cfg.accel_threshold, cfg.confidence_weights,
        )

        # ---- Soft scores ----
        scores = compute_soft_scores_batch(
            trend_strength_z, vol_percentile, transition_pressure,
            cfg.trend_threshold, cfg.vol_high_cutoff, cfg.vol_low_cutoff,
            cfg.accel_threshold, cfg.transition_high_threshold, cfg.sigmoid_steepness,
        )

        # ---- Stability ----
        regime_stability = compute_stability_batch(regime_label, cfg.stability_cap)

        # ---- Apply warmup / gap masks (overwrite outputs with INSUFFICIENT_DATA) ----
        insufficient_mask = warmup_mask | gap_mask
        regime_label[insufficient_mask] = _INSUFFICIENT
        committed_trend[insufficient_mask] = _INSUFFICIENT
        committed_vol[insufficient_mask] = _INSUFFICIENT
        committed_momentum[insufficient_mask] = _INSUFFICIENT

        nan_mask = insufficient_mask
        float_nan = np.full(n, np.nan)
        conf_out["regime_confidence"][nan_mask] = np.nan
        for k in scores:
            scores[k][nan_mask] = np.nan
        regime_stability[nan_mask] = np.nan

        # ---- Build output DataFrame ----
        out = df.copy()
        # Layer 1
        out["regime_trend"] = committed_trend
        out["regime_volatility"] = committed_vol
        out["regime_momentum"] = committed_momentum
        out["regime_label"] = regime_label
        out["regime_confidence"] = conf_out["regime_confidence"]
        # Layer 2
        out["trend_strength_z"] = trend_strength_z
        out["vol_percentile"] = vol_percentile
        out["volatility_skew"] = vol_m["volatility_skew"]
        out["transition_pressure"] = transition_pressure
        out["trend_acceleration"] = trend_m["trend_acceleration"]
        out["adaptive_alpha"] = vol_m["adaptive_alpha"]
        out["up_vol"] = vol_m["up_vol"]
        out["down_vol"] = vol_m["down_vol"]
        out["regime_stability"] = regime_stability
        out["directional_persistence"] = trend_m["directional_persistence"]
        # Layer 3
        for k, v in scores.items():
            out[k] = v

        # Apply NaN to all numeric Layer 2 columns for insufficient bars
        l2_cols = [
            "trend_strength_z", "vol_percentile", "volatility_skew",
            "transition_pressure", "trend_acceleration", "adaptive_alpha",
            "up_vol", "down_vol", "directional_persistence",
        ]
        for col in l2_cols:
            out.loc[nan_mask, col] = np.nan

        # Update internal state to end-of-batch state for live continuation
        self._restore_state_from_batch(
            close, trend_m, vol_m, mom_m,
            raw_trend, raw_vol, raw_momentum,
            committed_trend, committed_vol, committed_momentum,
            regime_label, regime_stability, conf_out, n,
            intervals, timestamps, final_ewma_interval, final_initialized,
        )

        logger.info(f"CALCULATE_BATCH END: Processed {n} bars, final_bars_seen={self._state.bars_seen}, "
                   f"final_warmup_complete={self._state.warmup_complete}")
        return out

    def update(self, bar: dict) -> dict:
        """
        Single-bar incremental update.
        bar must have 'close'. Optionally 'timestamp' (seconds) or 'interval'.
        Returns a dict with all three output layers for this bar.
        """
        cfg = self.config
        state = self._state
        close = float(bar["close"])

        
        logger.debug(f"UPDATE START: close={close}, bars_seen={state.bars_seen}, "
                    f"warmup_complete={state.warmup_complete}, prev_timestamp={state.prev_timestamp}, "
                    f"bar_keys={list(bar.keys())}")

        # ---- Gap detection ----
        gap_detected = False
        bar_ts = _extract_bar_timestamp(bar)  # None if no time info
        if bar_ts is not None and state.bars_seen > 0:
            interval = bar_ts - state.prev_timestamp
            logger.debug(f"UPDATE: Gap detection - bar_ts={bar_ts}, prev_timestamp={state.prev_timestamp}, "
                        f"interval={interval}, bars_seen={state.bars_seen}, "
                        f"interval_ewma_initialized={state.interval_ewma_initialized}")
            gap_detected = False #detect_gap(state, interval, cfg.gap_multiplier)
            if gap_detected:
                logger.warning(f"UPDATE: GAP DETECTED! interval={interval}, ewma_interval={state.ewma_interval}, "
                              f"gap_multiplier={cfg.gap_multiplier}, bars_seen={state.bars_seen}, "
                              f"warmup_complete={state.warmup_complete}")
        elif "interval" in bar and state.bars_seen > 0:
            interval_val = float(bar["interval"])
            logger.debug(f"UPDATE: Gap detection from interval field - interval={interval_val}, bars_seen={state.bars_seen}, "
                        f"interval_ewma_initialized={state.interval_ewma_initialized}")
            gap_detected = detect_gap(state, interval_val, cfg.gap_multiplier)
            if gap_detected:
                logger.warning(f"UPDATE: GAP DETECTED (from interval field)! interval={interval_val}, "
                              f"ewma_interval={state.ewma_interval}, bars_seen={state.bars_seen}")
        else:
            if bar_ts is not None:
                logger.debug(f"UPDATE: Skipping gap detection - bar_ts={bar_ts}, bars_seen={state.bars_seen} (need > 0)")
            elif "interval" in bar:
                logger.debug(f"UPDATE: Skipping gap detection - has interval field but bars_seen={state.bars_seen} (need > 0)")
            else:
                logger.debug(f"UPDATE: Skipping gap detection - no timestamp or interval info")

        if gap_detected:
            logger.warning(f"UPDATE: Resetting state due to gap detection")
            reset_state_on_gap(state)
            result = self._insufficient_result()
            result["regime_label"] = _GAP
            result["regime_trend"] = _GAP
            result["regime_volatility"] = _GAP
            result["regime_momentum"] = _GAP
            state.bars_seen += 1
            state.prev_close = close
            if bar_ts is not None:
                state.prev_timestamp = bar_ts
            logger.warning(f"UPDATE: After gap reset - bars_seen={state.bars_seen}, warmup_complete={state.warmup_complete}")
            return result

        prev_close = state.prev_close if state.bars_seen > 0 else close
        # Capture BEFORE update_trend overwrites state.prev_trend_strength_z
        prev_z = state.prev_trend_strength_z

        # ---- Compute metrics ----
        trend_out = update_trend(state, close, prev_close, cfg)
        vol_out = update_volatility(state, close, prev_close, cfg)
        mom_out = update_momentum(
            state,
            vol_out["volatility_level"],
            trend_out["trend_strength_z"],
            prev_z,
            cfg,
        )

        trend_strength_z = trend_out["trend_strength_z"]
        vol_percentile = vol_out["vol_percentile"]
        transition_pressure = mom_out["transition_pressure"]

        # ---- Update bars_seen AFTER seeding ----
        state.bars_seen += 1
        state.prev_close = close
        if bar_ts is not None:
            state.prev_timestamp = bar_ts
            logger.debug(f"UPDATE: Updated prev_timestamp to {state.prev_timestamp}")
        else:
            logger.warning(f"UPDATE: bar_ts is None! Cannot update prev_timestamp. Bar dict keys: {bar.keys()}")

        # ---- Warmup check ----
        if not state.warmup_complete:
            state.warmup_complete = is_warmup_complete(state.bars_seen, self._warmup_bars)
            if state.warmup_complete:
                logger.info(f"UPDATE: WARMUP COMPLETE at bars_seen={state.bars_seen} >= warmup_bars={self._warmup_bars}")
            else:
                logger.debug(f"UPDATE: Still in warmup: bars_seen={state.bars_seen} < warmup_bars={self._warmup_bars}")
        else:
            logger.debug(f"UPDATE: Warmup already complete, bars_seen={state.bars_seen}")

        if not state.warmup_complete:
            logger.debug(f"UPDATE: Returning INSUFFICIENT_DATA (warmup not complete)")
            return self._insufficient_result()

        # ---- INSUFFICIENT_VOLATILITY check ----
        vol_level = vol_out["volatility_level"]
        if vol_level < cfg.volatility_floor:
            state.vol_floor_count += 1
        else:
            state.vol_floor_count = 0

        if state.vol_floor_count >= cfg.volatility_floor_bars:
            result = self._layer2_result(trend_out, vol_out, mom_out)
            result.update(self._layer3_result(trend_strength_z, vol_percentile, transition_pressure))
            result["regime_trend"] = _INSUF_VOL
            result["regime_volatility"] = _INSUF_VOL
            result["regime_momentum"] = _INSUF_VOL
            result["regime_label"] = _INSUF_VOL
            result["regime_confidence"] = np.nan
            result["regime_stability"] = np.nan
            return result

        # ---- TRANSITION state machine ----
        in_trans = update_transition_state(state, trend_strength_z, transition_pressure, cfg)

        # ---- Raw dimension labels ----
        if in_trans:
            raw_trend = "TRANSITION"
        else:
            raw_trend = classify_trend_raw(trend_strength_z, cfg.trend_threshold)

        raw_vol = classify_vol(vol_percentile, cfg.vol_high_cutoff, cfg.vol_low_cutoff)
        raw_momentum = classify_momentum(transition_pressure, cfg.accel_threshold)

        # ---- Hysteresis + min_duration ----
        # For TRANSITION, bypass hysteresis on trend dimension
        if in_trans:
            state.committed_trend = "TRANSITION"
            state.pending_trend = None
            state.pending_trend_bars = 0
            trend_label = "TRANSITION"
        else:
            dist_trend = abs(abs(trend_strength_z) - cfg.trend_threshold)
            trend_label = apply_hysteresis_and_min_duration(
                state, "trend", raw_trend, dist_trend,
                cfg.trend_threshold, vol_percentile, cfg,
            )

        dist_vol = min(
            abs(vol_percentile - cfg.vol_high_cutoff),
            abs(vol_percentile - cfg.vol_low_cutoff),
        )
        vol_label = apply_hysteresis_and_min_duration(
            state, "vol", raw_vol, dist_vol,
            cfg.vol_high_cutoff, vol_percentile, cfg,
        )

        dist_mom = abs(transition_pressure - cfg.accel_threshold)
        momentum_label = apply_hysteresis_and_min_duration(
            state, "momentum", raw_momentum, dist_mom,
            cfg.accel_threshold, vol_percentile, cfg,
        )

        # ---- Combined label ----
        regime_label = combine_labels(trend_label, vol_label, momentum_label)

        # ---- Confidence ----
        self._update_conf_ewma(trend_strength_z, transition_pressure)
        trend_ewma_std = float(np.sqrt(max(state.conf_ewma_var_tsz, 0.0))) + 1e-10
        tp_ewma_std = float(np.sqrt(max(state.conf_ewma_var_tp, 0.0))) + 1e-10
        conf_out = compute_confidence(
            trend_strength_z, vol_percentile, transition_pressure,
            cfg.trend_threshold, cfg.accel_threshold,
            tp_ewma_std, trend_ewma_std, cfg.confidence_weights,
        )

        # ---- Stability ----
        regime_stability = update_stability(state, regime_label, cfg.stability_cap)

        # ---- Soft scores ----
        scores = compute_soft_scores(
            trend_strength_z, vol_percentile, transition_pressure,
            cfg.trend_threshold, cfg.vol_high_cutoff, cfg.vol_low_cutoff,
            cfg.accel_threshold, cfg.transition_high_threshold, cfg.sigmoid_steepness,
        )

        # ---- Assemble result ----
        result = {
            # Layer 1
            "regime_trend": trend_label,
            "regime_volatility": vol_label,
            "regime_momentum": momentum_label,
            "regime_label": regime_label,
            "regime_confidence": conf_out["regime_confidence"],
            # Layer 2
            "trend_strength_z": trend_strength_z,
            "vol_percentile": vol_percentile,
            "volatility_skew": vol_out["volatility_skew"],
            "transition_pressure": transition_pressure,
            "trend_acceleration": trend_out["trend_acceleration"],
            "adaptive_alpha": vol_out["adaptive_alpha"],
            "up_vol": vol_out["up_vol"],
            "down_vol": vol_out["down_vol"],
            "regime_stability": regime_stability,
            "directional_persistence": trend_out["directional_persistence"],
        }
        result.update(scores)
        
        logger.debug(f"UPDATE END: bars_seen={state.bars_seen}, prev_timestamp={state.prev_timestamp}, "
                    f"warmup_complete={state.warmup_complete}, regime_label={result['regime_label']}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insufficient_result(self) -> dict:
        nan = float("nan")
        return {
            "regime_trend": _INSUFFICIENT,
            "regime_volatility": _INSUFFICIENT,
            "regime_momentum": _INSUFFICIENT,
            "regime_label": _INSUFFICIENT,
            "regime_confidence": nan,
            "trend_strength_z": nan,
            "vol_percentile": nan,
            "volatility_skew": nan,
            "transition_pressure": nan,
            "trend_acceleration": nan,
            "adaptive_alpha": nan,
            "up_vol": nan,
            "down_vol": nan,
            "regime_stability": nan,
            "directional_persistence": nan,
            "score_bull": nan,
            "score_bear": nan,
            "score_range": nan,
            "score_transition": nan,
            "score_high_vol": nan,
            "score_low_vol": nan,
            "score_accelerating": nan,
        }

    def _layer2_result(self, trend_out, vol_out, mom_out) -> dict:
        return {
            "trend_strength_z": trend_out["trend_strength_z"],
            "vol_percentile": vol_out["vol_percentile"],
            "volatility_skew": vol_out["volatility_skew"],
            "transition_pressure": mom_out["transition_pressure"],
            "trend_acceleration": trend_out["trend_acceleration"],
            "adaptive_alpha": vol_out["adaptive_alpha"],
            "up_vol": vol_out["up_vol"],
            "down_vol": vol_out["down_vol"],
            "directional_persistence": trend_out["directional_persistence"],
        }

    def _layer3_result(self, trend_strength_z, vol_percentile, transition_pressure) -> dict:
        cfg = self.config
        return compute_soft_scores(
            trend_strength_z, vol_percentile, transition_pressure,
            cfg.trend_threshold, cfg.vol_high_cutoff, cfg.vol_low_cutoff,
            cfg.accel_threshold, cfg.transition_high_threshold, cfg.sigmoid_steepness,
        )

    def _update_conf_ewma(self, trend_strength_z: float, transition_pressure: float):
        a = 0.02
        b = 1 - a
        state = self._state
        abs_tsz = abs(trend_strength_z)
        state.conf_ewma_mean_tsz = a * abs_tsz + b * state.conf_ewma_mean_tsz
        state.conf_ewma_var_tsz = a * (abs_tsz - state.conf_ewma_mean_tsz) ** 2 + b * state.conf_ewma_var_tsz
        state.conf_ewma_mean_tp = a * transition_pressure + b * state.conf_ewma_mean_tp
        state.conf_ewma_var_tp = a * (transition_pressure - state.conf_ewma_mean_tp) ** 2 + b * state.conf_ewma_var_tp

    def _detect_gaps_batch(self, intervals: np.ndarray, gap_multiplier: float) -> tuple:
        """
        Sequential gap detection over interval array. 
        Returns (gap_mask, final_ewma_interval, final_initialized).
        """
        n = len(intervals)
        gap_mask = np.zeros(n, dtype=bool)
        alpha_interval = 0.10
        ewma_interval = 0.0
        initialized = False

        for i in range(n):
            iv = intervals[i]
            if iv <= 0:
                continue
            if not initialized:
                ewma_interval = iv
                initialized = True
                continue
            if iv > gap_multiplier * ewma_interval:
                gap_mask[i] = True
                # Re-seed EWMA after gap so next interval is evaluated fresh
                initialized = False
            else:
                ewma_interval = alpha_interval * iv + (1 - alpha_interval) * ewma_interval

        logger.debug(f"_detect_gaps_batch: final_ewma_interval={ewma_interval}, final_initialized={initialized}")
        return gap_mask, ewma_interval, initialized

    def _expand_warmup_after_gaps(self, gap_mask, warmup_mask, n):
        """
        After each gap, the next _warmup_bars bars are also marked as insufficient.
        """
        expanded = warmup_mask.copy()
        gap_positions = np.where(gap_mask)[0]
        for gp in gap_positions:
            # After a gap bar, _warmup_bars bars must be processed before valid output.
            # Gap bar itself (gp) is already covered by gap_mask; mark gp+1..gp+_warmup_bars-2
            # plus gp itself for consistency (innocent overlap with gap_mask).
            end = min(gp + self._warmup_bars - 1, n)
            expanded[gp:end] = True
        return gap_mask, expanded

    def _compute_insuf_vol_mask(
        self, volatility_level: np.ndarray, vfloor: float, vfloor_bars: int
    ) -> np.ndarray:
        """
        Mark bars where volatility_level < vfloor for more than vfloor_bars consecutive bars.
        """
        n = len(volatility_level)
        mask = np.zeros(n, dtype=bool)
        count = 0
        for i in range(n):
            if volatility_level[i] < vfloor:
                count += 1
            else:
                count = 0
            if count >= vfloor_bars:
                mask[i] = True
        return mask

    def _restore_state_from_batch(
        self, close, trend_m, vol_m, mom_m,
        raw_trend, raw_vol, raw_momentum,
        committed_trend, committed_vol, committed_momentum,
        regime_label, regime_stability, conf_out, n,
        intervals: np.ndarray,
        timestamps: Optional[np.ndarray],
        final_ewma_interval: float = 0.0,
        final_initialized: bool = False,
    ):
        """
        After calculate_batch, restore state to the end-of-batch values
        so that subsequent update() calls continue correctly.
        """
        cfg = self.config
        state = self._state
        
        logger.info(f"_restore_state_from_batch START: n={n}, warmup_bars={self._warmup_bars}, "
                   f"prev_bars_seen={state.bars_seen}, prev_warmup_complete={state.warmup_complete}")
        
        state.bars_seen = n
        old_warmup = state.warmup_complete
        state.warmup_complete = (n >= self._warmup_bars)
        
        logger.info(f"_restore_state_from_batch: bars_seen changed {old_warmup} → {state.bars_seen}, "
                   f"warmup_complete changed {old_warmup} → {state.warmup_complete} "
                   f"(n={n} >= warmup_bars={self._warmup_bars})")
        
        # Restore interval tracking state (CRITICAL FIX)
        state.ewma_interval = final_ewma_interval
        state.interval_ewma_initialized = final_initialized
        state.last_interval = intervals[-1] if len(intervals) > 0 else 0.0
        logger.info(f"_restore_state_from_batch: Restored interval tracking - "
                   f"ewma_interval={state.ewma_interval}, interval_ewma_initialized={state.interval_ewma_initialized}")
        
        # Restore prev_timestamp for next incremental update (CRITICAL FIX)
        if timestamps is not None and len(timestamps) > 0:
            state.prev_timestamp = float(timestamps[-1])
            logger.info(f"_restore_state_from_batch: Restored prev_timestamp={state.prev_timestamp}")
        
        state.prev_close = float(close[-1])
        state.prev_trend_strength_z = float(trend_m["trend_strength_z"][-1])

        state.ewma_fast_close = float(trend_m["ewma_fast"][-1])
        state.ewma_slow_close = float(trend_m["ewma_slow"][-1])
        state.ewma_mean = float(trend_m["ewma_mean"][-1])
        state.ewma_var = float(trend_m["ewma_var"][-1])
        state.ewma_vol_up = float(vol_m["up_vol"][-1] ** 2)
        state.ewma_vol_down = float(vol_m["down_vol"][-1] ** 2)
        state.ewma_vol_expansion = float(mom_m["ewma_vol_expansion"][-1])
        state.ewma_directional_persistence = float(trend_m["directional_persistence"][-1])

        # Restore ring buffer (last ring_buffer_size values of vol_level)
        K = cfg.ring_buffer_size
        vol_level_arr = vol_m["volatility_level"]
        buf_vals = vol_level_arr[-K:].tolist() if n >= K else vol_level_arr.tolist()
        state.ring_buffer = buf_vals
        state.ring_buffer_count = len(buf_vals)
        state.ring_buffer_head = len(buf_vals) % K if len(buf_vals) == K else len(buf_vals)

        state.committed_trend = str(committed_trend[-1])
        state.committed_vol = str(committed_vol[-1])
        state.committed_momentum = str(committed_momentum[-1])
        state.last_regime_label = str(regime_label[-1])
        last_stab = regime_stability[-1]
        state.stability_counter = int(last_stab * cfg.stability_cap) if not np.isnan(last_stab) else 0

        # Transition state: re-derive from last bar
        last_z = float(trend_m["trend_strength_z"][-1])
        state.last_trend_sign = int(np.sign(last_z))
        state.in_transition = (str(committed_trend[-1]) == "TRANSITION")

        # Restore confidence normalization EWMAs from last batch values
        state.conf_ewma_mean_tsz = float(conf_out["_ewma_tsz_mean"][-1])
        state.conf_ewma_var_tsz = float(conf_out["_ewma_tsz_var"][-1])
        state.conf_ewma_mean_tp = float(conf_out["_ewma_tp_mean"][-1])
        state.conf_ewma_var_tp = float(conf_out["_ewma_tp_var"][-1])

        # Restore pending labels by scanning the last min_duration_bars of the batch.
        # For each dimension, a pending label exists if the tail of raw labels differs
        # from the final committed label and the run hasn't yet met min_duration_bars.
        self._restore_pending_state(
            raw_trend, committed_trend, "trend",
            raw_vol, committed_vol, "vol",
            raw_momentum, committed_momentum, "momentum",
            cfg.min_duration_bars,
        )

    def _restore_pending_state(
        self,
        raw_trend, committed_trend, dim_trend,
        raw_vol, committed_vol, dim_vol,
        raw_momentum, committed_momentum, dim_mom,
        min_duration_bars: int,
    ):
        """
        Re-derive pending label and pending_bars for each dimension by scanning
        the tail of the batch. A pending label exists when the last K raw labels
        are a consistent new label that differs from the last committed label,
        and K < min_duration_bars (i.e. it hasn't committed yet).
        """
        logger.debug(f"_restore_pending_state: Restoring pending state from batch tail")
        for raw_arr, committed_arr, dim in [
            (raw_trend, committed_trend, dim_trend),
            (raw_vol, committed_vol, dim_vol),
            (raw_momentum, committed_momentum, dim_mom),
        ]:
            self._derive_pending_for_dim(raw_arr, committed_arr, dim, min_duration_bars)

    def _derive_pending_for_dim(self, raw_arr, committed_arr, dim, min_duration_bars):
        state = self._state
        # Scan backwards through the last min_duration_bars bars (valid only)
        # to find a run of consistent raw != committed at the tail.
        n = len(raw_arr)
        # Find valid (non-INSUFFICIENT) tail
        tail_end = n
        tail_start = max(0, tail_end - min_duration_bars)

        final_committed = str(committed_arr[tail_end - 1])
        pending = None
        pending_bars = 0

        # Walk backwards to find how many consecutive bars at the end have
        # raw != final_committed and all the same raw value
        for i in range(tail_end - 1, tail_start - 1, -1):
            raw = str(raw_arr[i])
            committed = str(committed_arr[i])
            if raw == committed:
                # Committed label changed here — stop
                break
            if raw == _INSUFFICIENT:
                break
            if pending is None:
                pending = raw
            elif raw != pending:
                # Inconsistent raw labels — no clean pending run
                pending = None
                pending_bars = 0
                break
            pending_bars += 1

        # Only set pending if it's a genuine partial run (< min_duration would commit)
        if pending is not None and pending_bars < min_duration_bars and pending != final_committed:
            setattr(state, f"pending_{dim}", pending)
            setattr(state, f"pending_{dim}_bars", pending_bars)
        else:
            setattr(state, f"pending_{dim}", None)
            setattr(state, f"pending_{dim}_bars", 0)
