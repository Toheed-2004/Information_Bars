import numpy as np
from typing import Dict, Any, List
from bitpredict.common.stats.custom_1.utils import BatchedReturns, LedgerArrays, COL_ACC_RET
from bitpredict.common.stats.custom_1.config import ANN_FACTOR, SQRT_ANN


def generate_plot_data_batched(
    stacked: LedgerArrays,
    batched: BatchedReturns,
) -> List[Dict[str, Any]]:
    n_strats  = len(stacked.names)
    max_days  = batched.daily_returns_2d.shape[0]
    valid     = batched.valid_mask_2d
    r         = batched.daily_returns_2d
    bal       = batched.daily_balances_2d
    bm        = batched.benchmark_returns_1d
    ts_ns     = batched.day_timestamps_ns
    n_days    = batched.n_days_per_strat

    # ── Timestamps ────────────────────────────────────────────────────────
    iso_all     = np.char.replace(
        np.datetime_as_string(ts_ns.astype('datetime64[ns]'), unit='s'), 'T', ' '
    )
    ts_list_all = iso_all.tolist()

    # ── Cumret / DD ───────────────────────────────────────────────────────
    r_safe   = np.where(valid, r, 0.0)
    cum_ret  = np.cumprod(1.0 + r_safe, axis=0)
    bm_pct   = bm * 100
    bal_safe = np.where(valid, bal, 0.0)
    peak     = np.maximum.accumulate(bal_safe, axis=0)
    dd       = np.where(valid & (peak > 0), bal_safe / np.where(peak > 0, peak, 1.0) - 1.0, 0.0)
    dd_pct   = dd * 100

    # ── Rolling ratios — optimised (see helper) ───────────────────────────
    window         = 30
    rs_all, rt_all = _rolling_ratios_batched(r_safe, valid, window)
    rc_all         = _rolling_corr_batched(r_safe, bm, valid, window)

    # ── Bulk tolist for daily series ──────────────────────────────────────
    cum_ret_list = cum_ret.T.tolist()
    bm_pct_list  = bm_pct.tolist()
    dd_pct_list  = dd_pct.T.tolist()
    rs_list      = rs_all.T.tolist()
    rt_list      = rt_all.T.tolist()
    rc_list      = rc_all.T.tolist()

    # ── Trade arrays ──────────────────────────────────────────────────────
    max_trades     = stacked.numeric_3d.shape[1]
    trade_lengths  = stacked.lengths
    trade_returns  = stacked.numeric_3d[:, :, COL_ACC_RET]
    sign_2d        = stacked.sign_2d
    exit_ts_all_ns = stacked.datetime_3d[:, :, 1]             # (n_strats, max_trades) int64

    trade_idx_arr  = np.arange(max_trades)[np.newaxis, :]
    trade_valid    = trade_idx_arr < trade_lengths[:, np.newaxis]
    long_mask_2d   = trade_valid & (sign_2d == 1.0)
    short_mask_2d  = trade_valid & (sign_2d == -1.0)
    tr_masked      = np.where(trade_valid, trade_returns, 0.0)
    long_cum_2d    = np.cumsum(np.where(long_mask_2d,  tr_masked, 0.0), axis=1)
    short_cum_2d   = np.cumsum(np.where(short_mask_2d, tr_masked, 0.0), axis=1)

    # ── FIX #1: exit timestamps — convert ONLY valid trades, not full matrix
    # Collect all valid exit timestamps in one flat array, then split back.
    # Avoids converting n_strats * max_trades slots when most are padding.
    exit_iso_per_strat = _convert_exit_timestamps(
        exit_ts_all_ns, trade_valid, trade_lengths, n_strats
    )

    # ── KDE ───────────────────────────────────────────────────────────────
    kde_results = _kde_batched(tr_masked, trade_valid, n_strats)

    # ── Pre-convert trade arrays to Python lists (avoids per-element cast) ─
    trade_returns_list = trade_returns.tolist()
    long_cum_list      = long_cum_2d.tolist()
    short_cum_list     = short_cum_2d.tolist()
    long_mask_list     = long_mask_2d.tolist()
    short_mask_list    = short_mask_2d.tolist()

    # ── Assemble results ──────────────────────────────────────────────────
    results = []
    for s in range(n_strats):
        nd = int(n_days[s])
        if nd == 0:
            results.append(_empty_plot_data())
            continue

        ts_s     = ts_list_all[:nd]
        cum_dict = dict(zip(ts_s, cum_ret_list[s][:nd]))
        bm_dict  = dict(zip(ts_s, bm_pct_list[:nd]))
        dd_dict  = dict(zip(ts_s, dd_pct_list[s][:nd]))
        rs_dict  = dict(zip(ts_s, rs_list[s][:nd]))
        rt_dict  = dict(zip(ts_s, rt_list[s][:nd]))
        rc_dict  = dict(zip(ts_s, rc_list[s][:nd]))

        dd_periods = _detect_drawdown_periods_fast(bal[:nd, s], ts_ns[:nd], dd[:nd, s])

        nt = int(trade_lengths[s])
        if nt > 0:
            ets  = exit_iso_per_strat[s]           # already a Python list of nt strings
            tr_s = trade_returns_list[s][:nt]
            lc_s = long_cum_list[s][:nt]
            sc_s = short_cum_list[s][:nt]
            lm_s = long_mask_list[s][:nt]
            sm_s = short_mask_list[s][:nt]

            mfe_dict = {}
            mae_dict = {}
            for i in range(nt):
                v = tr_s[i]
                if   v > 0: mfe_dict[ets[i]] = v
                elif v < 0: mae_dict[ets[i]] = v

            long_dict  = {ets[i]: lc_s[i] for i in range(nt) if lm_s[i]}
            short_dict = {ets[i]: sc_s[i] for i in range(nt) if sm_s[i]}
            pnl_dist   = kde_results[s]
        else:
            mfe_dict = mae_dict = long_dict = short_dict = {}
            pnl_dist = {"histogram_values": [], "kde_curve": []}

        results.append({
            "cumulative_return":   cum_dict,
            "drawdown_series":     dd_dict,
            "drawdown_periods":    dd_periods,
            "mfe_pct":             mfe_dict,
            "mae_pct":             mae_dict,
            "pnl_distribution":    pnl_dist,
            "directional_pnl":     {"long": long_dict, "short": short_dict},
            "rolling_sharpe":      rs_dict,
            "rolling_sortino":     rt_dict,
            "benchmark_returns":   bm_dict,
            "rolling_correlation": rc_dict,
        })

    return results


# ── FIX #1: convert only valid exit timestamps ────────────────────────────────

def _convert_exit_timestamps(
    exit_ts_all_ns: np.ndarray,   # (n_strats, max_trades) int64
    trade_valid:    np.ndarray,   # (n_strats, max_trades) bool
    trade_lengths:  np.ndarray,   # (n_strats,) int
    n_strats:       int,
) -> list:
    """
    Convert only the valid trade exit timestamps to ISO strings.
    Instead of converting the full (n_strats × max_trades) matrix (mostly zeros),
    we gather only valid entries into a flat array, convert once, then split.

    Avoids ~34% of total runtime caused by converting padding slots.
    """
    # Gather valid timestamps into one flat array
    # flat_ts[i] = exit timestamp for the i-th valid (strat, trade) pair
    flat_ns    = exit_ts_all_ns[trade_valid]                   # (total_valid_trades,)

    if flat_ns.size == 0:
        return [[] for _ in range(n_strats)]

    # Single bulk conversion — only valid trades, no padding
    flat_iso   = np.char.replace(
        np.datetime_as_string(flat_ns.astype('datetime64[ns]'), unit='s'), 'T', ' '
    ).tolist()                                                  # list[total_valid_trades]

    # Split back into per-strategy lists using trade_lengths
    result     = [None] * n_strats
    offset     = 0
    for s in range(n_strats):
        nt           = int(trade_lengths[s])
        result[s]    = flat_iso[offset: offset + nt]
        offset      += nt

    return result


# ── FIX #2: rolling ratios — avoid redundant sliding_window_view passes ───────

def _rolling_ratios_batched(
    r: np.ndarray,
    valid: np.ndarray,
    window: int,
) -> tuple:
    from numpy.lib.stride_tricks import sliding_window_view
    max_days, n_strats = r.shape
    sharpe  = np.zeros((max_days, n_strats), dtype=np.float32)
    sortino = np.zeros((max_days, n_strats), dtype=np.float32)

    if max_days < 2:
        return sharpe, sortino

    if max_days >= window:
        r_t    = r.T.astype(np.float32, copy=False)            # (n_strats, max_days)
        v_t    = valid.T.astype(np.float32)                    # (n_strats, max_days)

        # sliding_window_view is zero-copy (view, not allocation)
        wins   = sliding_window_view(r_t, window, axis=1)      # (n_strats, steps, window)
        v_wins = sliding_window_view(v_t, window, axis=1)

        win_n  = v_wins.sum(axis=2)                            # (n_strats, steps)
        win_ns = np.maximum(win_n, 1.0)

        # Compute mean and variance in ONE pass using Welford-style identity:
        # var = (Σx² - (Σx)²/n) / (n-1)  — avoids materialising (wins - mean)
        w_sum  = (wins * v_wins).sum(axis=2)                   # masked sum
        w_sum2 = (wins * wins * v_wins).sum(axis=2)
        means  = w_sum / win_ns
        var    = (w_sum2 - w_sum ** 2 / win_ns) / np.maximum(win_n - 1, 1.0)
        stds   = np.sqrt(np.maximum(var, 0.0))

        sharpe[window-1:, :]  = np.where(stds > 0, means / stds * SQRT_ANN, 0.0).T

        # Sortino: downside deviation — only one extra pass needed
        neg_sq = np.where(wins < 0, wins * wins * v_wins, 0.0).sum(axis=2)
        ds_dev = np.sqrt(neg_sq / win_ns)
        sortino[window-1:, :] = np.where(ds_dev > 0, means / ds_dev * SQRT_ANN, 0.0).T

    wu = min(window - 1, max_days)
    if wu > 0:
        r_wu   = r[:wu].astype(np.float32, copy=False)
        v_wu   = valid[:wu].astype(np.float32)
        counts = v_wu.cumsum(axis=0)
        n_s    = np.maximum(counts, 1.0)
        rv     = r_wu * v_wu
        cum_s  = rv.cumsum(axis=0)
        means  = cum_s / n_s
        cum_sq = (rv * rv).cumsum(axis=0)
        var    = np.where(counts > 1, (cum_sq - cum_s ** 2 / n_s) / np.maximum(counts - 1, 1.0), 0.0)
        stds   = np.sqrt(np.maximum(var, 0.0))
        sharpe[:wu]  = np.where(stds > 0, means / stds * SQRT_ANN, 0.0)
        neg_sq_wu    = np.where(r_wu < 0, rv * r_wu, 0.0).cumsum(axis=0)
        ds_dev_wu    = np.sqrt(neg_sq_wu / n_s)
        sortino[:wu] = np.where(ds_dev_wu > 0, means / ds_dev_wu * SQRT_ANN, 0.0)

    return sharpe, sortino


def _rolling_corr_batched(r, bm, valid, window):
    from numpy.lib.stride_tricks import sliding_window_view
    max_days, n_strats = r.shape
    corr = np.zeros((max_days, n_strats), dtype=np.float32)
    if max_days < 2:
        return corr
    if max_days >= window:
        r_t    = r.T.astype(np.float32, copy=False)
        bm_f   = bm.astype(np.float32, copy=False)
        p_wins = sliding_window_view(r_t,  window, axis=1)
        b_wins = sliding_window_view(bm_f, window)[np.newaxis, :, :]
        v_wins = sliding_window_view(valid.T.astype(np.float32), window, axis=1)
        win_n  = v_wins.sum(axis=2)
        n_s    = np.maximum(win_n, 1.0)
        pm     = p_wins.sum(axis=2) / n_s
        bm_    = b_wins.sum(axis=2) / window
        pc     = p_wins - pm[:, :, np.newaxis]
        bc     = b_wins - bm_[:, :, np.newaxis]
        cov    = (pc * bc * v_wins).sum(axis=2) / n_s
        std_p  = np.sqrt((pc ** 2 * v_wins).sum(axis=2) / n_s)
        std_b  = np.sqrt((bc ** 2).mean(axis=2))
        corr[window-1:, :] = np.where((std_p > 0) & (std_b > 0), cov / (std_p * std_b), 0.0).T
    wu = min(window - 1, max_days)
    if wu > 0:
        r_wu  = r[:wu].astype(np.float32, copy=False)
        b_wu  = bm[:wu].astype(np.float32, copy=False)
        v_wu  = valid[:wu].astype(np.float32)
        n_s   = np.maximum(v_wu.cumsum(axis=0), 1.0)
        pm_wu = (r_wu * v_wu).cumsum(axis=0) / n_s
        bm_wu = b_wu.cumsum() / np.arange(1, wu + 1, dtype=np.float32)
        pc_wu = np.where(valid[:wu], r_wu - pm_wu, 0.0)
        bc_wu = b_wu[:, np.newaxis] - bm_wu[:, np.newaxis]
        cov_wu   = (pc_wu * bc_wu * v_wu).cumsum(axis=0) / n_s
        std_p_wu = np.sqrt(np.maximum((pc_wu ** 2 * v_wu).cumsum(axis=0) / n_s, 0.0))
        std_b_wu = np.sqrt(np.maximum(((b_wu[:, np.newaxis] - bm_wu[:, np.newaxis]) ** 2).cumsum(axis=0) / np.arange(1, wu + 1, dtype=np.float32)[:, np.newaxis], 0.0))
        corr[:wu] = np.where((std_p_wu > 0) & (std_b_wu > 0), cov_wu / (std_p_wu * std_b_wu), 0.0)
    return corr


def _kde_batched(tr, trade_valid, n_strats):
    N_BINS = 40; N_POINTS = 50
    kernel  = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    counts  = trade_valid.sum(axis=1)
    tr_nan  = np.where(trade_valid, tr, np.nan)
    tr_min  = np.nanmin(tr_nan, axis=1, initial=0.0)
    tr_max  = np.nanmax(tr_nan, axis=1, initial=0.0)
    results = []
    for s in range(n_strats):
        if not counts[s]:
            results.append({"histogram_values": [], "kde_curve": []}); continue
        pnl    = tr[s, trade_valid[s]]
        lo, hi = float(tr_min[s]), float(tr_max[s])
        if lo == hi:
            results.append({"histogram_values": pnl.tolist(), "kde_curve": [{"x": lo, "density": 1.0}]}); continue
        hist, edges = np.histogram(pnl, bins=N_BINS, range=(lo, hi), density=True)
        hist        = hist.astype(np.float32)
        smoothed    = np.convolve(np.pad(hist, 1, mode='reflect'), kernel, mode='valid')
        x_out       = np.linspace(lo, hi, N_POINTS, dtype=np.float32)
        density     = np.interp(x_out, (edges[:-1] + edges[1:]) * 0.5, smoothed).astype(np.float32)
        dx          = (hi - lo) / N_POINTS
        total       = float(density.sum()) * dx
        if total > 0: density /= total
        results.append({"histogram_values": pnl.tolist(),
                         "kde_curve": [{"x": float(x_out[i]), "density": float(density[i])} for i in range(N_POINTS)]})
    return results


def _detect_drawdown_periods_fast(balances, timestamps_ns, drawdown):
    threshold   = -1e-10
    no_dd       = drawdown >= threshold
    transitions = np.diff(no_dd.astype(np.int8), prepend=1)
    starts      = np.flatnonzero(transitions == -1)
    ends        = np.flatnonzero(transitions == 1)
    if starts.size == 0:
        return []
    if ends.size and starts[0] > ends[0]:
        starts = np.insert(starts, 0, 0)
    if starts.size and (ends.size == 0 or starts[-1] > ends[-1]):
        ends = np.append(ends, len(drawdown) - 1)
    valley_indices = np.array([s + int(np.argmin(drawdown[s:e+1])) for s, e in zip(starts, ends)], dtype=np.int64)
    si_indices     = np.maximum(starts - 1, 0)
    all_idx        = np.concatenate([si_indices, valley_indices, ends])
    all_ts         = np.char.replace(
        np.datetime_as_string(timestamps_ns[all_idx].astype('datetime64[ns]'), unit='s'), 'T', ' '
    )
    n_p     = len(starts)
    si_strs = all_ts[:n_p].tolist()
    vl_strs = all_ts[n_p:2*n_p].tolist()
    en_strs = all_ts[2*n_p:].tolist()
    last    = len(drawdown) - 1
    return [{"Drawdown Id": i, "Column": 0,
             "Start Index": si_strs[i], "Valley Index": vl_strs[i], "End Index": en_strs[i],
             "Start Value": float(balances[si_indices[i]]), "Valley Value": float(balances[valley_indices[i]]),
             "End Value": float(balances[ends[i]]),
             "Status": "Active" if (ends[i] == last and drawdown[-1] < threshold) else "Recovered"}
            for i in range(n_p)]


def _empty_plot_data():
    return {"cumulative_return": {}, "benchmark_returns": {}, "drawdown_series": {},
            "drawdown_periods": [], "mfe_pct": {}, "mae_pct": {},
            "pnl_distribution": {"histogram_values": [], "kde_curve": []},
            "directional_pnl": {"long": {}, "short": {}},
            "rolling_sharpe": {}, "rolling_sortino": {}, "rolling_correlation": {}}