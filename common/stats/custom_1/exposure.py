import numpy as np
from bitpredict.common.stats.custom_1.utils import LedgerArrays, BatchedReturns, COL_POS_SIZE


def calculate_exposure_analysis(
    stacked: LedgerArrays,
    batched: BatchedReturns,
) -> np.ndarray:

    n_strats = len(stacked.names)
    max_days = batched.daily_returns_2d.shape[0]

    entry_dt = stacked.datetime_3d[:, :, 0].astype("int64")
    exit_dt  = stacked.datetime_3d[:, :, 1].astype("int64")

    pos_size = stacked.numeric_3d[:, :, COL_POS_SIZE] / 100.0
    sign     = stacked.sign_2d
    lengths  = stacked.lengths

    day_ts = batched.day_timestamps_ns.astype("int64")
    n_days = batched.n_days_per_strat.astype("int64")
    valid_mask_daily = batched.valid_mask_2d

    # ─────────────────────────────────────────────
    # FAST PATH: sweep-line exposure construction (VECTORIZED)
    # ─────────────────────────────────────────────

    # Flatten all trades across strategies for batch searchsorted
    max_trades = stacked.numeric_3d.shape[1]
    trade_idx = np.arange(max_trades)[np.newaxis, :]
    trade_valid = trade_idx < lengths[:, np.newaxis]  # (n_strats, max_trades)
    
    # Flatten timestamps and metadata
    entry_flat = entry_dt.ravel()  # (n_strats * max_trades,)
    exit_flat = exit_dt.ravel()
    sz_flat = pos_size.ravel()
    sg_flat = sign.ravel()
    valid_flat = trade_valid.ravel()
    
    # Strategy indices for each trade
    strat_idx = np.repeat(np.arange(n_strats), max_trades)  # (n_strats * max_trades,)
    
    # Batch searchsorted (much faster than per-strategy)
    entry_idx_flat = np.searchsorted(day_ts, entry_flat)
    exit_idx_flat = np.searchsorted(day_ts, exit_flat)
    
    # Build delta arrays using flat indexing
    daily_gross = np.zeros((max_days, n_strats), dtype=np.float64)
    daily_net = np.zeros((max_days, n_strats), dtype=np.float64)
    long_cnt = np.zeros((max_days, n_strats), dtype=np.int32)
    short_cnt = np.zeros((max_days, n_strats), dtype=np.int32)
    
    # Filter to valid trades only
    valid_mask = valid_flat & (entry_idx_flat < max_days) & (exit_idx_flat < max_days)
    
    if valid_mask.any():
        entry_idx_v = entry_idx_flat[valid_mask]
        exit_idx_v = exit_idx_flat[valid_mask]
        sz_v = sz_flat[valid_mask]
        sg_v = sg_flat[valid_mask]
        strat_v = strat_idx[valid_mask]
        
        # 2D indices for np.add.at: (day_idx, strat_idx)
        entry_2d = (entry_idx_v, strat_v)
        exit_2d = (exit_idx_v, strat_v)
        
        # Net exposure deltas
        np.add.at(daily_net, entry_2d, sz_v * sg_v)
        np.add.at(daily_net, exit_2d, -sz_v * sg_v)
        
        # Gross exposure deltas
        np.add.at(daily_gross, entry_2d, np.abs(sz_v))
        np.add.at(daily_gross, exit_2d, -np.abs(sz_v))
        
        # Long/short counts
        is_long = (sg_v == 1).astype(np.int8)
        is_short = (sg_v == -1).astype(np.int8)
        
        np.add.at(long_cnt, entry_2d, is_long)
        np.add.at(long_cnt, exit_2d, -is_long)
        
        np.add.at(short_cnt, entry_2d, is_short)
        np.add.at(short_cnt, exit_2d, -is_short)

    # cumulative exposure (THIS replaces full simulation loop)
    daily_net   = np.cumsum(daily_net, axis=0)
    daily_gross = np.cumsum(daily_gross, axis=0)
    long_cnt    = np.cumsum(long_cnt, axis=0)
    short_cnt   = np.cumsum(short_cnt, axis=0)

    # apply mask
    daily_gross = np.where(valid_mask_daily, daily_gross, np.nan)
    daily_net   = np.where(valid_mask_daily, daily_net, np.nan)

    # ─────────────────────────────────────────────
    # METRICS (fully vectorized, no loops)
    # ─────────────────────────────────────────────

    # Get last valid day per strategy (correct indexing for transposed arrays)
    last_day_idx = (n_days - 1).clip(0)  # (n_strats,)
    gross_last = daily_gross[last_day_idx, np.arange(n_strats)]  # (n_strats,)
    net_last   = daily_net[last_day_idx, np.arange(n_strats)]    # (n_strats,)

    gross_max = np.nanmax(daily_gross, axis=0)
    gross_avg = np.nanmean(daily_gross, axis=0)

    net_max = np.nanmax(daily_net, axis=0)
    net_min = np.nanmin(daily_net, axis=0)
    net_avg = np.nanmean(daily_net, axis=0)

    gross_std = np.nanstd(daily_gross, axis=0, ddof=1)
    net_std   = np.nanstd(daily_net, axis=0, ddof=1)

    total = n_days.astype(np.float64)

    position_mask = (daily_gross > 0) & valid_mask_daily
    position_periods = np.sum(position_mask, axis=0)

    long_periods  = np.sum((long_cnt > 0) & valid_mask_daily, axis=0)
    short_periods = np.sum((short_cnt > 0) & valid_mask_daily, axis=0)

    idle_periods = total - position_periods

    # ratios
    avg_util = np.divide(gross_avg, gross_max, out=np.zeros_like(gross_avg), where=gross_max > 0)
    coeff_var = np.divide(gross_std, gross_avg, out=np.zeros_like(gross_std), where=gross_avg > 0)
    consistency = 1 - coeff_var

    directional_bias = np.divide(np.abs(net_avg), gross_avg, out=np.zeros_like(net_avg), where=gross_avg > 0)

    # percent coverage
    pos_cov = position_periods / total * 100
    long_cov = long_periods / total * 100
    short_cov = short_periods / total * 100

    # long/short exposure
    long_exp = np.divide(long_periods.astype(np.float64), position_periods.astype(np.float64), 
                         out=np.zeros(n_strats, dtype=np.float64), 
                         where=position_periods > 0) * 100
    short_exp = np.divide(short_periods.astype(np.float64), position_periods.astype(np.float64), 
                          out=np.zeros(n_strats, dtype=np.float64), 
                          where=position_periods > 0) * 100

    # percentiles (single pass for all)
    pcts = np.nanpercentile(daily_gross, [25, 50, 75, 90, 95], axis=0)  # (5, n_strats)
    p25, p50, p75, p90, p95 = pcts

    # ─────────────────────────────────────────────
    # OUTPUT
    # ─────────────────────────────────────────────

    dtype = [
        ('gross_exposure_current_pct', 'f8'),
        ('gross_exposure_max_pct', 'f8'),
        ('gross_exposure_avg_pct', 'f8'),
        ('net_exposure_current_pct', 'f8'),
        ('net_exposure_max_pct', 'f8'),
        ('net_exposure_min_pct', 'f8'),
        ('net_exposure_avg_pct', 'f8'),
        ('net_exposure_range_pct', 'f8'),
        ('position_coverage_pct', 'f8'),
        ('long_position_coverage_pct', 'f8'),
        ('short_position_coverage_pct', 'f8'),
        ('long_exposure_pct', 'f8'),
        ('short_exposure_pct', 'f8'),
        ('exposure_volatility_pct', 'f8'),
        ('net_exposure_volatility_pct', 'f8'),
        ('exposure_coefficient_of_variation', 'f8'),
        ('avg_exposure_utilization', 'f8'),
        ('exposure_consistency', 'f8'),
        ('exposure_directional_bias', 'f8'),
        ('exposure_p25_pct', 'f8'),
        ('exposure_p50_pct', 'f8'),
        ('exposure_p75_pct', 'f8'),
        ('exposure_p90_pct', 'f8'),
        ('exposure_p95_pct', 'f8'),
        ('total_periods', 'i8'),
        ('position_periods', 'i8'),
        ('long_periods', 'i8'),
        ('short_periods', 'i8'),
        ('idle_periods', 'i8'),
    ]

    result = np.zeros(n_strats, dtype=dtype)

    result['gross_exposure_current_pct'] = gross_last * 100
    result['gross_exposure_max_pct']     = gross_max * 100
    result['gross_exposure_avg_pct']     = gross_avg * 100

    result['net_exposure_current_pct'] = net_last * 100
    result['net_exposure_max_pct']     = net_max * 100
    result['net_exposure_min_pct']     = net_min * 100
    result['net_exposure_avg_pct']     = net_avg * 100
    result['net_exposure_range_pct']   = (net_max - net_min) * 100

    result['position_coverage_pct']      = pos_cov
    result['long_position_coverage_pct'] = long_cov
    result['short_position_coverage_pct']= short_cov

    result['long_exposure_pct']  = long_exp
    result['short_exposure_pct'] = short_exp

    result['exposure_volatility_pct']     = gross_std * 100
    result['net_exposure_volatility_pct'] = net_std * 100

    result['exposure_coefficient_of_variation'] = coeff_var
    result['avg_exposure_utilization'] = avg_util
    result['exposure_consistency'] = consistency
    result['exposure_directional_bias'] = directional_bias

    result['exposure_p25_pct'] = p25 * 100
    result['exposure_p50_pct'] = p50 * 100
    result['exposure_p75_pct'] = p75 * 100
    result['exposure_p90_pct'] = p90 * 100
    result['exposure_p95_pct'] = p95 * 100

    result['total_periods'] = total.astype(np.int64)
    result['position_periods'] = position_periods.astype(np.int64)
    result['long_periods'] = long_periods.astype(np.int64)
    result['short_periods'] = short_periods.astype(np.int64)
    result['idle_periods'] = idle_periods.astype(np.int64)

    return result