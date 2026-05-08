"""
Fully vectorized cash flow analysis for ALL strategies at once.
Operates on LedgerArrays and BatchedReturns with pure numpy - no pandas after input.
"""

import numpy as np
from bitpredict.common.stats.custom_1.utils import (
    LedgerArrays, BatchedReturns, COL_POS_SIZE, COL_ENTRY_FEE, 
    COL_EXIT_FEE, COL_ACC_RET, COL_BALANCE
)


def calculate_cash_flow_analysis(stacked: LedgerArrays, batched: BatchedReturns) -> np.ndarray:
    n_strats   = len(stacked.names)
    max_trades = stacked.numeric_3d.shape[1]
    max_days   = batched.daily_balances_2d.shape[0]

    # ── Raw fields ────────────────────────────────────────────────────────
    balance_vals = stacked.numeric_3d[:, :, COL_BALANCE]        # (n_strats, max_trades)
    acc_ret      = stacked.numeric_3d[:, :, COL_ACC_RET]  / 100.0
    pos_size     = stacked.numeric_3d[:, :, COL_POS_SIZE] / 100.0
    entry_fee    = stacked.numeric_3d[:, :, COL_ENTRY_FEE]/ 100.0
    exit_fee     = stacked.numeric_3d[:, :, COL_EXIT_FEE] / 100.0
    lengths      = stacked.lengths                               # (n_strats,)

    daily_balances = batched.daily_balances_2d                   # (max_days, n_strats)
    daily_returns  = batched.daily_returns_2d                    # (max_days, n_strats)
    valid_mask     = batched.valid_mask_2d                       # (max_days, n_strats)
    n_days         = batched.n_days_per_strat.astype(np.float64) # (n_strats,)

    trade_idx      = np.arange(max_trades)[np.newaxis, :]        # (1, max_trades)
    trade_valid    = trade_idx < lengths[:, np.newaxis]          # (n_strats, max_trades)

    # ── Initial balance ───────────────────────────────────────────────────
    denom           = 1.0 + acc_ret[:, 0]
    initial_balance = np.where(denom != 0, balance_vals[:, 0] / denom, balance_vals[:, 0])

    # ── Balance before each trade ─────────────────────────────────────────
    balance_before          = np.empty_like(balance_vals)
    balance_before[:, 0]    = initial_balance
    balance_before[:, 1:]   = balance_vals[:, :-1]

    # ── Per-trade P&L and fees ────────────────────────────────────────────
    pos_val             = np.where(trade_valid, balance_before * pos_size,  0.0)
    e_fees              = np.where(trade_valid, pos_val * entry_fee,        0.0)
    x_fees              = np.where(trade_valid, pos_val * exit_fee,         0.0)
    total_fees_pt       = e_fees + x_fees
    pnl_pt              = np.where(trade_valid, balance_before * acc_ret,   0.0)

    entry_flows         = -(pos_val + e_fees)                    # (n_strats, max_trades)
    exit_flows          = pos_val + pnl_pt - x_fees

    # Stack entry+exit side by side → (n_strats, max_trades*2)
    all_flows           = np.concatenate([entry_flows, exit_flows], axis=1)
    flow_valid          = np.concatenate([trade_valid, trade_valid],   axis=1)
    all_flows_m         = np.where(flow_valid, all_flows, np.nan)    # NaN = invalid

    # ── Daily balance stats ───────────────────────────────────────────────
    bal_m               = np.where(valid_mask, daily_balances, np.nan)  # (max_days, n_strats)
    last_idx            = (batched.n_days_per_strat - 1).clip(0).astype(np.int64)
    current_balance     = daily_balances[last_idx, np.arange(n_strats)]

    n_days_i            = np.maximum(n_days, 1.0)
    min_balance         = np.nanmin(bal_m, axis=0)
    max_balance         = np.nanmax(bal_m, axis=0)
    mean_balance        = np.nansum(np.where(valid_mask, daily_balances, 0.0), axis=0) / n_days_i
    # variance of balance
    bal_rc              = np.where(valid_mask, daily_balances - mean_balance, 0.0)
    balance_vol         = np.sqrt((bal_rc ** 2).sum(axis=0) / np.maximum(n_days - 1, 1.0))

    total_cash_flow     = current_balance - initial_balance
    avg_cash_flow       = np.where(n_days > 0, total_cash_flow / n_days_i, 0.0)

    # ── Flow stats (vectorized moments, no NaN overhead) ──────────────────
    flow_n              = flow_valid.sum(axis=1).astype(np.float64)    # (n_strats,)
    flow_n_s            = np.maximum(flow_n, 1.0)
    flow_n1_s           = np.maximum(flow_n - 1.0, 1.0)

    cf_mean             = np.where(flow_valid, all_flows, 0.0).sum(axis=1) / flow_n_s

    fc                  = np.where(flow_valid, all_flows - cf_mean[:, np.newaxis], 0.0)
    cf_var              = (fc ** 2).sum(axis=1) / flow_n1_s
    cf_vol              = np.sqrt(cf_var)

    # Skew / Kurt — reuse fc
    m3                  = (fc ** 3).sum(axis=1) / flow_n_s
    m4                  = (fc ** 4).sum(axis=1) / flow_n_s
    skew_raw            = np.where(cf_var > 0, m3 / np.maximum(cf_var ** 1.5, 1e-30), 0.0)
    kurt_raw            = np.where(cf_var > 0, m4 / np.maximum(cf_var ** 2,   1e-30) - 3.0, 0.0)
    cf_skew             = np.where(
        flow_n > 2,
        skew_raw * np.sqrt(flow_n * (flow_n - 1)) / np.maximum(flow_n - 2, 1.0),
        0.0
    )
    cf_kurt             = np.where(
        flow_n > 3,
        (flow_n - 1) / (np.maximum(flow_n - 2, 1.0) * np.maximum(flow_n - 3, 1.0))
        * ((flow_n + 1) * kurt_raw + 6.0),
        0.0
    )

    # ── Positive / negative flows ─────────────────────────────────────────
    pos_mask            = flow_valid & (all_flows > 0)
    neg_mask            = flow_valid & (all_flows < 0)

    gross_inflow        = np.where(pos_mask, all_flows,  0.0).sum(axis=1)
    gross_outflow       = np.abs(np.where(neg_mask, all_flows, 0.0).sum(axis=1))
    net_cash_flow       = gross_inflow - gross_outflow

    pos_n               = pos_mask.sum(axis=1).astype(np.float64)
    neg_n               = neg_mask.sum(axis=1).astype(np.float64)

    max_pos_flow        = np.where(pos_n > 0, np.where(pos_mask,  all_flows, -np.inf).max(axis=1), 0.0)
    max_neg_flow        = np.where(neg_n > 0, np.where(neg_mask,  all_flows,  np.inf).min(axis=1), 0.0)
    avg_pos_flow        = np.where(pos_n > 0, np.where(pos_mask,  all_flows, 0.0).sum(axis=1) / np.maximum(pos_n, 1.0), 0.0)
    avg_neg_flow        = np.where(neg_n > 0, np.where(neg_mask,  all_flows, 0.0).sum(axis=1) / np.maximum(neg_n, 1.0), 0.0)

    # ── Period counts ─────────────────────────────────────────────────────
    pos_periods         = (valid_mask & (daily_returns > 0)).sum(axis=0).astype(np.int64)
    neg_periods         = (valid_mask & (daily_returns < 0)).sum(axis=0).astype(np.int64)
    neutral_periods     = (valid_mask & (daily_returns == 0)).sum(axis=0).astype(np.int64)
    pos_periods_pct     = np.where(n_days > 0, pos_periods / n_days_i * 100, 0.0)
    neg_periods_pct     = np.where(n_days > 0, neg_periods / n_days_i * 100, 0.0)
    neutral_periods_pct = np.where(n_days > 0, neutral_periods / n_days_i * 100, 0.0)

    # ── Fee analysis ──────────────────────────────────────────────────────
    total_fees          = np.where(trade_valid, total_fees_pt, 0.0).sum(axis=1)
    avg_fee_per_trade   = np.where(lengths > 0, total_fees / np.maximum(lengths, 1), 0.0)
    fee_pct             = np.where(initial_balance > 0, total_fees / initial_balance * 100, 0.0)

    # ── Derived metrics ───────────────────────────────────────────────────
    cf_coeff_var        = np.where(np.abs(cf_mean) > 0, cf_vol / np.abs(cf_mean), 0.0)
    cf_efficiency       = np.where(gross_outflow > 0, gross_inflow / gross_outflow, 0.0)
    cf_sharpe           = np.where(cf_vol > 0, cf_mean / cf_vol, 0.0)
    cf_stability        = np.where(np.abs(mean_balance) > 0, 1.0 - cf_vol / np.abs(mean_balance), 0.0)

    util_avg            = np.where(initial_balance != 0, mean_balance    / initial_balance * 100, 0.0)
    util_min            = np.where(initial_balance != 0, min_balance     / initial_balance * 100, 0.0)
    util_max            = np.where(initial_balance != 0, max_balance     / initial_balance * 100, 0.0)

    # ── Percentiles (axis=1 → per strategy) ──────────────────────────────
    # nanpercentile over axis=1 on (n_strats, max_trades*2)
    pcts                = np.nanpercentile(all_flows_m, [5, 10, 25, 50, 75, 90, 95], axis=1)
    # pcts shape: (7, n_strats)

    # ── Time metrics — vectorized over strategies (OPTIMIZED) ────────────
    # exit timestamps: (n_strats, max_trades)
    exit_ts = stacked.datetime_3d[:, :, 1].astype(np.int64)  # Use int64 directly, no float conversion

    # For each strategy: diff of exit_ts across valid trades
    # Valid diff: both trade t and t+1 must be valid
    diff_valid = (trade_idx[:, 1:] < lengths[:, np.newaxis]) & \
                 (trade_idx[:, :-1] < lengths[:, np.newaxis])     # (n_strats, max_trades-1)
    
    # Compute diffs only where valid (avoid unnecessary computation)
    shifted = np.diff(exit_ts, axis=1)  # (n_strats, max_trades-1) - faster than manual shift
    
    diff_n = diff_valid.sum(axis=1).astype(np.float64)
    diff_sum = np.where(diff_valid, shifted, 0.0).sum(axis=1)
    avg_time_hours = np.where(diff_n > 0, diff_sum / np.maximum(diff_n, 1.0) / (3600 * 1e9), 0.0)

    # Frequency: trades per day = lengths / span_in_days
    # Use advanced indexing to get last valid timestamp per strategy
    last_idx = (lengths - 1).clip(0)
    first_valid_ts = exit_ts[:, 0]
    last_valid_ts = exit_ts[np.arange(n_strats), last_idx]
    
    span_ns = np.maximum(last_valid_ts - first_valid_ts, 0.0)
    cf_frequency = np.where(span_ns > 0, lengths / span_ns * 86400 * 1e9, 0.0)

    # ── Drawdown on daily balances ────────────────────────────────────────
    bal_filled          = np.where(valid_mask, daily_balances, 0.0)
    peak                = np.maximum.accumulate(bal_filled, axis=0)
    dd_arr              = np.where(valid_mask, peak - bal_filled, 0.0)
    max_dd              = dd_arr.max(axis=0)
    avg_dd              = dd_arr.sum(axis=0) / n_days_i

    # ── Build result ──────────────────────────────────────────────────────
    dtype = [
        ('total_cash_flow_dollar', 'f8'),       ('avg_cash_flow_dollar', 'f8'),
        ('cash_flow_volatility_dollar', 'f8'),  ('cash_flow_mean_dollar', 'f8'),
        ('positive_cash_flow_dollar', 'f8'),    ('negative_cash_flow_dollar', 'f8'),
        ('net_cash_flow_dollar', 'f8'),         ('gross_cash_inflow_dollar', 'f8'),
        ('gross_cash_outflow_dollar', 'f8'),    ('max_positive_flow_dollar', 'f8'),
        ('max_negative_flow_dollar', 'f8'),     ('avg_positive_flow_dollar', 'f8'),
        ('avg_negative_flow_dollar', 'f8'),
        ('positive_periods_count', 'i8'),       ('negative_periods_count', 'i8'),
        ('neutral_periods_count', 'i8'),        ('positive_periods_pct', 'f8'),
        ('negative_periods_pct', 'f8'),         ('neutral_periods_pct', 'f8'),
        ('cash_balance_current_dollar', 'f8'),  ('cash_balance_initial_dollar', 'f8'),
        ('cash_balance_min_dollar', 'f8'),      ('cash_balance_max_dollar', 'f8'),
        ('cash_balance_mean_dollar', 'f8'),     ('cash_balance_volatility_dollar', 'f8'),
        ('total_fees_paid', 'f8'),              ('avg_fee_per_trade', 'f8'),
        ('fee_pct', 'f8'),
        ('cash_flow_coefficient_of_variation', 'f8'), ('cash_flow_skewness', 'f8'),
        ('cash_flow_kurtosis', 'f8'),           ('cash_flow_efficiency', 'f8'),
        ('cash_flow_sharpe', 'f8'),             ('cash_flow_stability', 'f8'),
        ('cash_utilization_avg_pct', 'f8'),     ('cash_utilization_min_pct', 'f8'),
        ('cash_utilization_max_pct', 'f8'),
        ('flow_p5_dollar', 'f8'),  ('flow_p10_dollar', 'f8'), ('flow_p25_dollar', 'f8'),
        ('flow_p50_dollar', 'f8'), ('flow_p75_dollar', 'f8'), ('flow_p90_dollar', 'f8'),
        ('flow_p95_dollar', 'f8'),
        ('avg_time_between_flows_hours', 'f8'), ('cash_flow_frequency', 'f8'),
        ('max_balance_drawdown_dollar', 'f8'),  ('avg_balance_drawdown_dollar', 'f8'),
    ]

    result = np.zeros(n_strats, dtype=dtype)
    result['total_cash_flow_dollar']              = total_cash_flow
    result['avg_cash_flow_dollar']                = avg_cash_flow
    result['cash_flow_volatility_dollar']         = cf_vol
    result['cash_flow_mean_dollar']               = cf_mean
    result['positive_cash_flow_dollar']           = gross_inflow
    result['negative_cash_flow_dollar']           = gross_outflow
    result['net_cash_flow_dollar']                = net_cash_flow
    result['gross_cash_inflow_dollar']            = gross_inflow
    result['gross_cash_outflow_dollar']           = gross_outflow
    result['max_positive_flow_dollar']            = max_pos_flow
    result['max_negative_flow_dollar']            = max_neg_flow
    result['avg_positive_flow_dollar']            = avg_pos_flow
    result['avg_negative_flow_dollar']            = avg_neg_flow
    result['positive_periods_count']              = pos_periods
    result['negative_periods_count']              = neg_periods
    result['neutral_periods_count']               = neutral_periods
    result['positive_periods_pct']                = pos_periods_pct
    result['negative_periods_pct']                = neg_periods_pct
    result['neutral_periods_pct']                 = neutral_periods_pct
    result['cash_balance_current_dollar']         = current_balance
    result['cash_balance_initial_dollar']         = initial_balance
    result['cash_balance_min_dollar']             = min_balance
    result['cash_balance_max_dollar']             = max_balance
    result['cash_balance_mean_dollar']            = mean_balance
    result['cash_balance_volatility_dollar']      = balance_vol
    result['total_fees_paid']                     = total_fees
    result['avg_fee_per_trade']                   = avg_fee_per_trade
    result['fee_pct']                             = fee_pct
    result['cash_flow_coefficient_of_variation']  = cf_coeff_var
    result['cash_flow_skewness']                  = cf_skew
    result['cash_flow_kurtosis']                  = cf_kurt
    result['cash_flow_efficiency']                = cf_efficiency
    result['cash_flow_sharpe']                    = cf_sharpe
    result['cash_flow_stability']                 = cf_stability
    result['cash_utilization_avg_pct']            = util_avg
    result['cash_utilization_min_pct']            = util_min
    result['cash_utilization_max_pct']            = util_max
    result['flow_p5_dollar']                      = pcts[0]
    result['flow_p10_dollar']                     = pcts[1]
    result['flow_p25_dollar']                     = pcts[2]
    result['flow_p50_dollar']                     = pcts[3]
    result['flow_p75_dollar']                     = pcts[4]
    result['flow_p90_dollar']                     = pcts[5]
    result['flow_p95_dollar']                     = pcts[6]
    result['avg_time_between_flows_hours']        = avg_time_hours
    result['cash_flow_frequency']                 = cf_frequency
    result['max_balance_drawdown_dollar']         = max_dd
    result['avg_balance_drawdown_dollar']         = avg_dd

    return result