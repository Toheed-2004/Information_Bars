"""
Fully vectorized performance analysis for custom_1.
Calculates monthly breakdown and recent performance metrics.
NO per-strategy loops — all strategies processed simultaneously.
"""

import numpy as np
from bitpredict.common.stats.custom_1.utils import BatchedReturns


def calculate_performance_batched(batched: BatchedReturns) -> tuple:
    """
    Calculate performance metrics for ALL strategies at once (batched).
    Zero per-strategy Python loops — fully vectorized across strategies.
    """
    monthly_breakdowns = calculate_monthly_breakdown_batched(batched)
    recent_perfs       = calculate_recent_performance_batched(batched)
    monthly_stats_list = calculate_monthly_statistics_batched(batched)
    monthly_heatmaps   = calculate_monthly_heatmap_matrix_batched(batched)

    return monthly_breakdowns, recent_perfs, monthly_stats_list, monthly_heatmaps


# ── Shared helper: build (strategy, month) compound returns ──────────────────

def _compound_monthly_returns(batched: BatchedReturns):
    """
    Returns
    -------
    monthly_rets_2d : (n_strats, n_unique_months)  float64
    unique_months   : (n_unique_months,)            datetime64[M]
    valid_cell_mask : (n_strats, n_unique_months)   bool
        True where the strategy actually had data in that month.
    month_labels_ns : (n_unique_months,)            datetime64[ns]  (month-end)
    """
    max_days, n_strats = batched.daily_returns_2d.shape

    # ── Timestamps → month integer codes (one shared timeline) ───────────
    ts_ns    = batched.day_timestamps_ns                     # (max_days,)
    months   = ts_ns.astype('datetime64[ns]').astype('datetime64[M]')   # (max_days,)
    unique_months, month_codes = np.unique(months, return_inverse=True)  # codes: (max_days,)
    n_months = len(unique_months)

    # ── Valid mask: day d is valid for strategy s ─────────────────────────
    day_idx  = np.arange(max_days)                           # (max_days,)
    n_days   = batched.n_days_per_strat                      # (n_strats,)
    valid    = day_idx[:, np.newaxis] < n_days[np.newaxis, :]  # (max_days, n_strats)

    r        = batched.daily_returns_2d                      # (max_days, n_strats)

    # ── log-returns trick: sum log(1+r) per (strategy, month) then exp ───
    # Shape broadcasts: (max_days, n_strats) × (max_days,) index
    log1r    = np.log1p(np.where(valid, r, 0.0))             # (max_days, n_strats)

    # Accumulate into (n_strats, n_months) using np.add.at equivalent —
    # but faster with bincount per strategy via einsum + sparse index trick.
    # We flatten strategy dimension and use np.add.at on (n_strats*n_months).
    n_flat   = n_strats * n_months
    # cell index for each (day, strat) pair
    strat_idx = np.arange(n_strats)[np.newaxis, :]            # (1, n_strats)
    cell_idx  = month_codes[:, np.newaxis] * n_strats + strat_idx  # (max_days, n_strats)

    log_sum  = np.zeros(n_flat, dtype=np.float64)
    count    = np.zeros(n_flat, dtype=np.int32)

    np.add.at(log_sum, cell_idx[valid], log1r[valid])
    np.add.at(count,   cell_idx[valid], 1)

    log_sum  = log_sum.reshape(n_months, n_strats).T         # (n_strats, n_months)
    count    = count.reshape(n_months, n_strats).T           # (n_strats, n_months)

    monthly_rets_2d  = (np.expm1(log_sum)) * 100             # (n_strats, n_months)
    valid_cell_mask  = count > 0                              # (n_strats, n_months)

    # ── Month-end labels ──────────────────────────────────────────────────
    # unique_months is datetime64[M]; month_end = first_of_next_month - 1 day
    months_as_days   = unique_months.astype('datetime64[D]')
    # add 32 days then truncate to month, then back to day, gives first of next month
    next_month_start = (unique_months + np.timedelta64(1, 'M')).astype('datetime64[D]')
    month_end_days   = next_month_start - np.timedelta64(1, 'D')
    month_labels_ns  = month_end_days.astype('datetime64[ns]')

    return monthly_rets_2d, unique_months, valid_cell_mask, month_labels_ns


# ── Monthly breakdown ─────────────────────────────────────────────────────────

def calculate_monthly_breakdown_batched(batched: BatchedReturns) -> list:
    """
    Returns list[dict] — one dict per strategy, keyed by YYYY-MM-DD month-end.
    No per-strategy Python loops.
    """
    monthly_rets_2d, _, valid_cell_mask, month_labels_ns = _compound_monthly_returns(batched)

    # Convert month-end ns to YYYY-MM-DD strings — vectorized
    label_strs = np.datetime_as_string(month_labels_ns, unit='D')   # (n_months,)

    n_strats, n_months = monthly_rets_2d.shape

    # Build list of dicts — dict construction is O(n_months) per strategy,
    # unavoidable in Python, but no numpy loop over strategies.
    results = []
    for s in range(n_strats):
        mask = valid_cell_mask[s]
        results.append(dict(zip(label_strs[mask].tolist(),
                                monthly_rets_2d[s, mask].tolist())))
    return results


# ── Recent performance ────────────────────────────────────────────────────────

def calculate_recent_performance_batched(batched: BatchedReturns) -> list:
    """
    Returns list[dict] with keys 1d_pnl_pct … 90d_pnl_pct, fully vectorized.
    """
    periods   = [1, 7, 15, 30, 45, 60, 90]
    max_days, n_strats = batched.daily_returns_2d.shape
    n_days    = batched.n_days_per_strat                     # (n_strats,)
    r         = batched.daily_returns_2d                     # (max_days, n_strats)

    # For each period p and strategy s we want compound(r[max(0,nd-p):nd, s])
    # = exp(sum(log1p(r[max(0,nd-p):nd, s]))) - 1
    # Build a suffix log-sum array: suffix_log[d, s] = sum log1p(r[d:nd, s])
    # Then period return = expm1(suffix_log[max(0,nd-p), s] - suffix_log[nd, s])
    # suffix_log[nd, s] = 0 by definition.

    log1r     = np.log1p(r)                                  # (max_days, n_strats)
    # Mask days beyond n_days
    day_idx   = np.arange(max_days)
    valid     = day_idx[:, np.newaxis] < n_days[np.newaxis, :]
    log1r     = np.where(valid, log1r, 0.0)

    # Reverse-cumsum from the end up to n_days
    # We want: for each s, suffix_logsum[d, s] = sum_{i=d}^{n_days[s]-1} log1r[i,s]
    # = total_log[s] - prefix_log[d, s]
    prefix    = np.cumsum(log1r, axis=0)                     # (max_days, n_strats)
    # total per strategy = prefix at index n_days[s]-1
    nd_clamp  = np.maximum(n_days - 1, 0).astype(int)        # (n_strats,)
    total     = prefix[nd_clamp, np.arange(n_strats)]        # (n_strats,)

    # For period p, start_day = max(0, n_days - p)
    # suffix_log from start_day = total - prefix[start_day-1]  (0 if start_day==0)
    results   = []
    period_rets = {}

    for p in periods:
        start_days = np.maximum(n_days - p, 0).astype(int)   # (n_strats,)
        # prefix[start_day - 1] — guard for start_day == 0
        prev_prefix = np.where(
            start_days > 0,
            prefix[np.maximum(start_days - 1, 0), np.arange(n_strats)],
            0.0
        )
        log_sum_p  = total - prev_prefix                      # (n_strats,)
        # zero out strategies with no data
        log_sum_p  = np.where(n_days > 0, log_sum_p, 0.0)
        period_rets[f'{p}d_pnl_pct'] = np.expm1(log_sum_p) * 100  # (n_strats,)

    # Assemble list of dicts — columns → rows
    keys = [f'{p}d_pnl_pct' for p in periods]
    vals = np.stack([period_rets[k] for k in keys], axis=1)  # (n_strats, n_periods)

    for s in range(n_strats):
        results.append(dict(zip(keys, vals[s].tolist())))

    return results


# ── Monthly statistics ────────────────────────────────────────────────────────

def calculate_monthly_statistics_batched(batched: BatchedReturns) -> list:
    """
    Returns list[dict] with mean/std/best/worst/win_rate, vectorized.
    """
    monthly_rets_2d, _, valid_cell_mask, _ = _compound_monthly_returns(batched)
    n_strats = monthly_rets_2d.shape[0]

    # Mask invalid cells with NaN so nanfunctions work correctly
    mr    = np.where(valid_cell_mask, monthly_rets_2d, np.nan)  # (n_strats, n_months)
    counts = valid_cell_mask.sum(axis=1).astype(float)           # (n_strats,)

    mean_ret  = np.nanmean(mr, axis=1)                           # (n_strats,)
    # ddof=1 std via nanvar
    std_ret   = np.sqrt(np.nanvar(mr, axis=1) * np.where(counts > 1, counts / (counts - 1), 0.0))
    best_ret  = np.nanmax(mr, axis=1)
    worst_ret = np.nanmin(mr, axis=1)
    pos_count = np.nansum(mr > 0, axis=1)
    win_rate  = np.where(counts > 0, pos_count / counts * 100, 0.0)

    # Replace NaN (all-empty strategies) with 0
    def _z(x): return np.where(np.isfinite(x), x, 0.0)

    keys = ["monthly_returns_mean_pct", "monthly_returns_std_pct",
            "best_month_return_pct", "worst_month_return_pct",
            "positive_months_pct", "monthly_win_rate"]
    vals = np.stack([_z(mean_ret), _z(std_ret), _z(best_ret),
                     _z(worst_ret), win_rate, win_rate], axis=1)  # (n_strats, 6)

    return [dict(zip(keys, vals[s].tolist())) for s in range(n_strats)]


# ── Monthly heatmap matrix ────────────────────────────────────────────────────

def calculate_monthly_heatmap_matrix_batched(batched: BatchedReturns) -> list:
    """
    Returns list[dict] with monthly_matrix / years / months, vectorized.
    All pivot logic done with numpy — no per-strategy loop over months.
    """
    monthly_rets_2d, unique_months, valid_cell_mask, _ = _compound_monthly_returns(batched)
    n_strats, n_months = monthly_rets_2d.shape

    if n_months == 0:
        empty = {"monthly_matrix": [], "years": [], "months": list(range(1, 13))}
        return [empty] * n_strats

    # ── Decode year and month-of-year from unique_months ─────────────────
    years_dt  = unique_months.astype('datetime64[Y]')
    year_vals = years_dt.astype(int) + 1970                   # (n_months,) int
    mon_vals  = (unique_months.astype(int) % 12) + 1          # (n_months,) 1..12

    unique_years = np.unique(year_vals)                        # (n_years,)
    n_years      = len(unique_years)

    # Map each month to its year-row and month-column in the pivot
    year_row = np.searchsorted(unique_years, year_vals)        # (n_months,)
    mon_col  = mon_vals - 1                                    # (n_months,) 0..11

    # ── Build pivot: (n_strats, n_years, 12) ─────────────────────────────
    pivot = np.zeros((n_strats, n_years, 12), dtype=np.float64)

    # Vectorized scatter: for each (month_idx), fill all strategies at once
    # pivot[:, year_row[m], mon_col[m]] = monthly_rets_2d[:, m]
    pivot[:, year_row, mon_col] = np.where(valid_cell_mask,
                                           monthly_rets_2d, 0.0)

    # ── Assemble output dicts ─────────────────────────────────────────────
    year_list = unique_years.tolist()
    mon_list  = list(range(1, 13))

    results = []
    for s in range(n_strats):
        results.append({
            "monthly_matrix": pivot[s].tolist(),
            "years":          year_list,
            "months":         mon_list,
        })
    return results