"""
Fully vectorized distribution analysis for ALL strategies at once.
Operates on daily returns arrays with pure numpy - no pandas after input.
"""

import numpy as np
from bitpredict.common.stats.custom_1.utils import BatchedReturns


def calculate_distribution_analysis(batched: BatchedReturns) -> np.ndarray:
    max_days, n_strats = batched.daily_returns_2d.shape

    r         = batched.daily_returns_2d   # (max_days, n_strats)
    valid     = batched.valid_mask_2d      # (max_days, n_strats)
    n_days    = batched.n_days_per_strat.astype(np.float64)  # (n_strats,)
    n_s       = np.maximum(n_days, 1.0)
    n1_s      = np.maximum(n_days - 1.0, 1.0)

    r_v       = np.where(valid, r, 0.0)   # zeros for invalid (safe for sums)

    # ── Central moments (single pass, no nan overhead) ────────────────────
    mean_r    = r_v.sum(axis=0) / n_s                         # (n_strats,)
    rc        = np.where(valid, r - mean_r, 0.0)              # centered, invalid=0
    m2        = (rc ** 2).sum(axis=0) / n1_s                  # variance ddof=1
    std_r     = np.sqrt(m2)

    # Standardized moments for skew/kurt
    std_safe  = np.where(std_r > 0, std_r, 1.0)
    zc        = np.where(valid, rc / std_safe, 0.0)           # (max_days, n_strats)
    m3        = (zc ** 3).sum(axis=0) / n_s
    m4        = (zc ** 4).sum(axis=0) / n_s

    skewness  = np.where(
        (std_r > 0) & (n_days > 2),
        m3 * np.sqrt(n_days * (n_days - 1)) / np.maximum(n_days - 2, 1.0),
        0.0
    )
    kurtosis  = np.where(
        (std_r > 0) & (n_days > 3),
        (n_days - 1) / (np.maximum(n_days - 2, 1.0) * np.maximum(n_days - 3, 1.0))
        * ((n_days + 1) * m4 - 3.0 * (n_days - 1)),
        0.0
    )

    # ── All percentiles in ONE call → (11, n_strats) ──────────────────────
    r_nan     = np.where(valid, r, np.nan)
    pcts      = np.nanpercentile(r_nan, [1, 5, 10, 25, 50, 75, 90, 95, 99], axis=0) * 100
    # pcts[i] shape: (n_strats,)
    p1, p5, p10, p25, p50, p75, p90, p95, p99 = pcts

    # VaR reuses percentiles (already computed)
    var_95    = p5    # (n_strats,)
    var_99    = p1

    # ── CVaR — sorted-array trick, zero loops ─────────────────────────────
    r_sorted  = np.sort(r_nan, axis=0)                        # NaNs → end
    # threshold per strategy broadcast: (1, n_strats)
    thresh_95 = (var_95 / 100)[np.newaxis, :]
    thresh_99 = (var_99 / 100)[np.newaxis, :]

    tail95_m  = r_sorted <= thresh_95
    tail99_m  = r_sorted <= thresh_99
    # also exclude NaNs
    not_nan   = ~np.isnan(r_sorted)
    tail95_m &= not_nan
    tail99_m &= not_nan

    tail95_n  = tail95_m.sum(axis=0).astype(np.float64)
    tail99_n  = tail99_m.sum(axis=0).astype(np.float64)

    cvar_95   = np.where(
        tail95_n > 0,
        np.where(tail95_m, r_sorted, 0.0).sum(axis=0) / np.maximum(tail95_n, 1.0) * 100,
        var_95
    )
    cvar_99   = np.where(
        tail99_n > 0,
        np.where(tail99_m, r_sorted, 0.0).sum(axis=0) / np.maximum(tail99_n, 1.0) * 100,
        var_99
    )

    # ── Tail ratio ────────────────────────────────────────────────────────
    tail_ratio = np.where(p5 != 0, p95 / np.abs(p5), 0.0)

    # ── Downside / upside deviation (vectorized) ──────────────────────────
    dn_mask   = valid & (r <= 0)
    up_mask_d = valid & (r >= 0)
    dn_n      = dn_mask.sum(axis=0).astype(np.float64)
    up_n      = up_mask_d.sum(axis=0).astype(np.float64)

    dn_mean   = np.where(dn_mask, r, 0.0).sum(axis=0) / np.maximum(dn_n, 1.0)
    up_mean   = np.where(up_mask_d, r, 0.0).sum(axis=0) / np.maximum(up_n, 1.0)

    dn_rc     = np.where(dn_mask, r - dn_mean, 0.0)
    up_rc     = np.where(up_mask_d, r - up_mean, 0.0)

    downside_deviation = np.where(
        dn_n > 1,
        np.sqrt((dn_rc ** 2).sum(axis=0) / np.maximum(dn_n - 1, 1.0)) * 100,
        0.0
    )
    upside_deviation = np.where(
        up_n > 1,
        np.sqrt((up_rc ** 2).sum(axis=0) / np.maximum(up_n - 1, 1.0)) * 100,
        0.0
    )

    # ── Return frequency ──────────────────────────────────────────────────
    pos_mask  = valid & (r > 0)
    neg_mask  = valid & (r < 0)
    zero_mask = valid & (r == 0)

    pos_count  = pos_mask.sum(axis=0).astype(np.int64)
    neg_count  = neg_mask.sum(axis=0).astype(np.int64)
    zero_count = zero_mask.sum(axis=0).astype(np.int64)

    pos_pct    = np.where(n_days > 0, pos_count  / n_s * 100, 0.0)
    neg_pct    = np.where(n_days > 0, neg_count  / n_s * 100, 0.0)
    zero_pct   = np.where(n_days > 0, zero_count / n_s * 100, 0.0)

    # ── Gain/loss means ───────────────────────────────────────────────────
    pos_n_f    = pos_count.astype(np.float64)
    neg_n_f    = neg_count.astype(np.float64)

    pos_mean   = np.where(pos_n_f > 0,
                          np.where(pos_mask, r, 0.0).sum(axis=0) / np.maximum(pos_n_f, 1.0) * 100, 0.0)
    neg_mean   = np.where(neg_n_f > 0,
                          np.where(neg_mask, r, 0.0).sum(axis=0) / np.maximum(neg_n_f, 1.0) * 100, 0.0)

    gain_loss_ratio = np.where(neg_mean != 0, pos_mean / np.abs(neg_mean), 0.0)

    # ── Shape / normality ─────────────────────────────────────────────────
    mean_pct   = mean_r * 100
    std_pct    = std_r  * 100
    var_pct    = m2     * 10000

    coeff_var  = np.where(mean_pct != 0, std_r / np.abs(mean_r), 0.0)
    jarque_bera = np.where(
        (n_days > 2) & (std_r > 0),
        n_days * (skewness ** 2 / 6.0 + kurtosis ** 2 / 24.0),
        0.0
    )

    # ── Extreme values ────────────────────────────────────────────────────
    max_ret    = np.where(valid, r, -np.inf).max(axis=0) * 100
    min_ret    = np.where(valid, r,  np.inf).min(axis=0) * 100
    ret_range  = max_ret - min_ret

    # ── Outliers (IQR, vectorized) ────────────────────────────────────────
    iqr        = (p75 - p25) / 100.0
    upper_thr  = (p75 / 100.0) + 1.5 * iqr                   # (n_strats,)
    lower_thr  = (p25 / 100.0) - 1.5 * iqr

    out_upper  = (valid & (r > upper_thr[np.newaxis, :])).sum(axis=0).astype(np.int64)
    out_lower  = (valid & (r < lower_thr[np.newaxis, :])).sum(axis=0).astype(np.int64)
    tot_out    = out_upper + out_lower
    out_pct    = np.where(n_days > 0, tot_out / n_s * 100, 0.0)

    # ── Build result ──────────────────────────────────────────────────────
    dtype = [
        ('returns_mean_pct', 'f8'), ('returns_std_pct', 'f8'), ('returns_variance_pct', 'f8'),
        ('skewness', 'f8'), ('kurtosis', 'f8'),
        ('percentile_1', 'f8'), ('percentile_5', 'f8'), ('percentile_10', 'f8'),
        ('percentile_25', 'f8'), ('percentile_50', 'f8'), ('percentile_75', 'f8'),
        ('percentile_90', 'f8'), ('percentile_95', 'f8'), ('percentile_99', 'f8'),
        ('var_95_pct', 'f8'), ('var_99_pct', 'f8'), ('cvar_95_pct', 'f8'), ('cvar_99_pct', 'f8'),
        ('tail_ratio', 'f8'), ('downside_deviation_pct', 'f8'), ('upside_deviation_pct', 'f8'),
        ('positive_returns_count', 'i8'), ('negative_returns_count', 'i8'),
        ('zero_returns_count', 'i8'), ('positive_returns_pct', 'f8'),
        ('negative_returns_pct', 'f8'), ('zero_returns_pct', 'f8'),
        ('positive_mean_pct', 'f8'), ('negative_mean_pct', 'f8'), ('gain_loss_ratio', 'f8'),
        ('coefficient_of_variation', 'f8'), ('jarque_bera_statistic', 'f8'),
        ('max_return_pct', 'f8'), ('min_return_pct', 'f8'), ('return_range_pct', 'f8'),
        ('outliers_upper', 'i8'), ('outliers_lower', 'i8'),
        ('total_outliers', 'i8'), ('outlier_pct', 'f8'),
        ('total_observations', 'i8'), ('distribution_analysis_complete', 'f8'),
    ]

    result = np.zeros(n_strats, dtype=dtype)
    result['returns_mean_pct']              = mean_pct
    result['returns_std_pct']               = std_pct
    result['returns_variance_pct']          = var_pct
    result['skewness']                      = skewness
    result['kurtosis']                      = kurtosis
    result['percentile_1']                  = p1
    result['percentile_5']                  = p5
    result['percentile_10']                 = p10
    result['percentile_25']                 = p25
    result['percentile_50']                 = p50
    result['percentile_75']                 = p75
    result['percentile_90']                 = p90
    result['percentile_95']                 = p95
    result['percentile_99']                 = p99
    result['var_95_pct']                    = var_95
    result['var_99_pct']                    = var_99
    result['cvar_95_pct']                   = cvar_95
    result['cvar_99_pct']                   = cvar_99
    result['tail_ratio']                    = tail_ratio
    result['downside_deviation_pct']        = downside_deviation
    result['upside_deviation_pct']          = upside_deviation
    result['positive_returns_count']        = pos_count
    result['negative_returns_count']        = neg_count
    result['zero_returns_count']            = zero_count
    result['positive_returns_pct']          = pos_pct
    result['negative_returns_pct']          = neg_pct
    result['zero_returns_pct']              = zero_pct
    result['positive_mean_pct']             = pos_mean
    result['negative_mean_pct']             = neg_mean
    result['gain_loss_ratio']               = gain_loss_ratio
    result['coefficient_of_variation']      = coeff_var
    result['jarque_bera_statistic']         = jarque_bera
    result['max_return_pct']                = max_ret
    result['min_return_pct']                = min_ret
    result['return_range_pct']              = ret_range
    result['outliers_upper']                = out_upper
    result['outliers_lower']                = out_lower
    result['total_outliers']                = tot_out
    result['outlier_pct']                   = out_pct
    result['total_observations']            = n_days.astype(np.int64)
    result['distribution_analysis_complete']= np.ones(n_strats)

    return result