import numpy as np
from bitpredict.common.stats.custom_1.utils import BatchedReturns
from bitpredict.common.stats.custom_1.config import ANN_FACTOR, SQRT_ANN


def calculate_benchmark_analysis(batched: BatchedReturns) -> np.ndarray:
    max_days, n_strats = batched.daily_returns_2d.shape

    P = batched.daily_returns_2d          # (max_days, n_strats)
    B = batched.benchmark_returns_1d      # (max_days,)
    valid = batched.valid_mask_2d         # (max_days, n_strats)
    n_days = batched.n_days_per_strat     # (n_strats,)

    B2d = B[:, np.newaxis] * np.ones((1, n_strats))   # (max_days, n_strats)

    # Zero out invalid days
    P_m = np.where(valid, P, 0.0)
    B_m = np.where(valid, B2d, 0.0)
    excess_m = P_m - B_m                              # (max_days, n_strats)

    # ── Total returns ─────────────────────────────────────────────────────
    log_p = np.where(valid, np.log1p(P), 0.0)
    log_b = np.where(valid, np.log1p(B2d), 0.0)

    portfolio_total  = (np.exp(log_p.sum(axis=0)) - 1) * 100   # (n_strats,)
    benchmark_total  = (np.exp(log_b.sum(axis=0)) - 1) * 100

    # ── Outperformance ────────────────────────────────────────────────────
    outperformance = portfolio_total - benchmark_total
    outperformance_ratio = np.where(
        benchmark_total != -100,
        (1 + portfolio_total / 100) / (1 + benchmark_total / 100),
        0.0
    )

    # ── Tracking error / IR ───────────────────────────────────────────────
    n_f = n_days.astype(np.float64)
    n_valid = np.maximum(n_f, 1.0)

    excess_mean = excess_m.sum(axis=0) / n_valid                # (n_strats,)
    excess_sq   = ((excess_m - np.where(valid, excess_mean, 0.0)) ** 2)
    excess_var  = np.where(
        n_days > 1,
        (np.where(valid, excess_sq, 0.0)).sum(axis=0) / np.maximum(n_f - 1, 1.0),
        0.0
    )
    te_frac         = np.sqrt(excess_var) * SQRT_ANN
    tracking_error  = te_frac * 100
    information_ratio = np.where(
        te_frac > 0,
        excess_mean * ANN_FACTOR / te_frac,
        0.0
    )
    active_return_mean = excess_mean * ANN_FACTOR * 100
    active_return_std  = tracking_error

    # ── Capture ratios ────────────────────────────────────────────────────
    B_col = B[:, np.newaxis]                                     # (max_days, 1)
    up_mask   = (B_col > 0) & valid                              # (max_days, n_strats)
    down_mask = (B_col < 0) & valid

    up_count   = up_mask.sum(axis=0).astype(np.float64)
    down_count = down_mask.sum(axis=0).astype(np.float64)

    mean_p_up   = np.where(up_mask,   P,   0.0).sum(axis=0) / np.maximum(up_count,   1.0)
    mean_b_up   = np.where(up_mask,   B2d, 0.0).sum(axis=0) / np.maximum(up_count,   1.0)
    mean_p_down = np.where(down_mask, P,   0.0).sum(axis=0) / np.maximum(down_count, 1.0)
    mean_b_down = np.where(down_mask, B2d, 0.0).sum(axis=0) / np.maximum(down_count, 1.0)

    up_capture   = np.where(mean_b_up   != 0, mean_p_up   / mean_b_up,   0.0)
    down_capture = np.where(mean_b_down != 0, mean_p_down / mean_b_down, 0.0)
    capture_ratio = np.where(down_capture != 0, up_capture / np.abs(down_capture), 0.0)

    # ── Correlation, Beta, Alpha ──────────────────────────────────────────
    p_mean = P_m.sum(axis=0) / n_valid                          # (n_strats,)
    b_mean = B_m.sum(axis=0) / n_valid

    P_c = np.where(valid, P - p_mean, 0.0)
    B_c = np.where(valid, B2d - b_mean, 0.0)

    cov_pb = (P_c * B_c).sum(axis=0) / np.maximum(n_f - 1, 1.0)
    var_b  = (B_c ** 2).sum(axis=0)  / np.maximum(n_f - 1, 1.0)
    var_p  = (P_c ** 2).sum(axis=0)  / np.maximum(n_f - 1, 1.0)

    std_p = np.sqrt(var_p)
    std_b = np.sqrt(var_b)

    correlation = np.where(
        (std_p > 0) & (std_b > 0) & (n_days > 1),
        cov_pb / (std_p * std_b),
        0.0
    )
    beta  = np.where((var_b > 0) & (n_days > 1), cov_pb / var_b, 0.0)
    alpha = (p_mean * ANN_FACTOR - beta * b_mean * ANN_FACTOR) * 100

    # ── Rolling correlation (stride-tricks, no Python loops) ──────────────
    window = 30
    rolling_correlation = correlation.copy()

    if max_days >= window * 2:
        from numpy.lib.stride_tricks import sliding_window_view

        P_sw = sliding_window_view(P, window, axis=0)   # (max_days-w+1, n_strats, w)
        B_sw = sliding_window_view(B, window)            # (max_days-w+1, w)
        B_sw = B_sw[:, np.newaxis, :]                    # broadcast

        p_w_mean = P_sw.mean(axis=2, keepdims=True)
        b_w_mean = B_sw.mean(axis=2, keepdims=True)

        P_wc = P_sw - p_w_mean
        B_wc = B_sw - b_w_mean

        cov_w  = (P_wc * B_wc).mean(axis=2)
        std_pw = P_wc.std(axis=2)
        std_bw = B_wc.std(axis=2)

        rc_all = np.where(
            (std_pw > 0) & (std_bw > 0),
            cov_w / (std_pw * std_bw),
            np.nan
        )                                                # (steps, n_strats)

        valid_sw = sliding_window_view(valid.astype(np.float32), window, axis=0)
        full_win = valid_sw.min(axis=2).astype(bool)     # (steps, n_strats)

        rc_masked = np.where(full_win, rc_all, np.nan)
        with np.errstate(all='ignore'):
            rc_mean = np.nanmean(rc_masked, axis=0)

        has_rc = ~np.isnan(rc_mean) & (n_days >= window * 2)
        rolling_correlation = np.where(has_rc, rc_mean, correlation)

    # ── Period comparison ─────────────────────────────────────────────────
    out_mask  = valid & (P > B2d)
    under_mask = valid & (P < B2d)
    tie_mask  = valid & (P == B2d)

    outperforming_periods  = out_mask.sum(axis=0).astype(np.int64)
    underperforming_periods = under_mask.sum(axis=0).astype(np.int64)
    tie_periods            = tie_mask.sum(axis=0).astype(np.int64)

    outperforming_pct   = np.where(n_days > 0, outperforming_periods  / n_f * 100, 0.0)
    underperforming_pct = np.where(n_days > 0, underperforming_periods / n_f * 100, 0.0)

    # ── MAE / MFE ─────────────────────────────────────────────────────────
    cum_p = np.cumprod(np.where(valid, 1 + P, 1.0), axis=0) - 1
    cum_b = np.cumprod(np.where(valid, 1 + B2d, 1.0), axis=0) - 1
    cum_excess = np.where(valid, cum_p - cum_b, np.nan)

    with np.errstate(all='ignore'):
        max_adverse_excursion  = np.nanmin(cum_excess, axis=0) * 100
        max_favorable_excursion = np.nanmax(cum_excess, axis=0) * 100

    max_adverse_excursion  = np.where(n_days > 0, max_adverse_excursion,  0.0)
    max_favorable_excursion = np.where(n_days > 0, max_favorable_excursion, 0.0)

    # ── Benchmark volatility ──────────────────────────────────────────────
    benchmark_volatility = np.where(n_days > 1, std_b * SQRT_ANN * 100, 0.0)

    # ── Build structured array ────────────────────────────────────────────
    dtype = [
        ('portfolio_total_return_pct', 'f8'), ('benchmark_total_return_pct', 'f8'),
        ('outperformance', 'f8'), ('outperformance_ratio', 'f8'),
        ('tracking_error_pct', 'f8'), ('information_ratio', 'f8'),
        ('active_return_mean_pct', 'f8'), ('active_return_std_pct', 'f8'),
        ('up_capture_ratio', 'f8'), ('down_capture_ratio', 'f8'),
        ('capture_ratio', 'f8'),
        ('correlation', 'f8'), ('beta', 'f8'), ('alpha_pct', 'f8'),
        ('rolling_correlation', 'f8'),
        ('outperforming_periods', 'i8'), ('underperforming_periods', 'i8'),
        ('tie_periods', 'i8'), ('outperforming_pct', 'f8'), ('underperforming_pct', 'f8'),
        ('max_adverse_excursion_pct', 'f8'), ('max_favorable_excursion_pct', 'f8'),
        ('comparison_periods', 'i8'), ('benchmark_volatility_pct', 'f8'),
    ]

    result = np.zeros(n_strats, dtype=dtype)
    result['portfolio_total_return_pct']  = portfolio_total
    result['benchmark_total_return_pct']  = benchmark_total
    result['outperformance']              = outperformance
    result['outperformance_ratio']        = outperformance_ratio
    result['tracking_error_pct']          = tracking_error
    result['information_ratio']           = information_ratio
    result['active_return_mean_pct']      = active_return_mean
    result['active_return_std_pct']       = active_return_std
    result['up_capture_ratio']            = up_capture
    result['down_capture_ratio']          = down_capture
    result['capture_ratio']               = capture_ratio
    result['correlation']                 = correlation
    result['beta']                        = beta
    result['alpha_pct']                   = alpha
    result['rolling_correlation']         = rolling_correlation
    result['outperforming_periods']       = outperforming_periods
    result['underperforming_periods']     = underperforming_periods
    result['tie_periods']                 = tie_periods
    result['outperforming_pct']           = outperforming_pct
    result['underperforming_pct']         = underperforming_pct
    result['max_adverse_excursion_pct']   = max_adverse_excursion
    result['max_favorable_excursion_pct'] = max_favorable_excursion
    result['comparison_periods']          = n_days.astype(np.int64)
    result['benchmark_volatility_pct']    = benchmark_volatility

    return result