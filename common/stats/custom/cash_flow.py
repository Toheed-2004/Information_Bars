import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def _get_empty_cash_flow() -> Dict[str, Any]:
    return {
        'total_cash_flow_dollar': 0.0, 'avg_cash_flow_dollar': 0.0,
        'cash_flow_volatility_dollar': 0.0, 'cash_flow_mean_dollar': 0.0,
        'positive_cash_flow_dollar': 0.0, 'negative_cash_flow_dollar': 0.0,
        'net_cash_flow_dollar': 0.0, 'gross_cash_inflow_dollar': 0.0,
        'gross_cash_outflow_dollar': 0.0, 'max_positive_flow_dollar': 0.0,
        'max_negative_flow_dollar': 0.0, 'avg_positive_flow_dollar': 0.0,
        'avg_negative_flow_dollar': 0.0,
        'positive_periods_count': 0, 'negative_periods_count': 0, 'neutral_periods_count': 0,
        'positive_periods_pct': 0.0, 'negative_periods_pct': 0.0, 'neutral_periods_pct': 0.0,
        'cash_balance_current_dollar': 0.0, 'cash_balance_initial_dollar': 0.0,
        'cash_balance_min_dollar': 0.0, 'cash_balance_max_dollar': 0.0,
        'cash_balance_mean_dollar': 0.0, 'cash_balance_volatility_dollar': 0.0,
        'total_fees_paid': 0.0, 'avg_fee_per_trade': 0.0, 'fee_pct': 0.0,
        'cash_flow_coefficient_of_variation': 0.0, 'cash_flow_skewness': 0.0,
        'cash_flow_kurtosis': 0.0, 'cash_flow_efficiency': 0.0,
        'cash_flow_sharpe': 0.0, 'cash_flow_stability': 0.0,
        'cash_utilization_avg_pct': 0.0, 'cash_utilization_min_pct': 0.0,
        'cash_utilization_max_pct': 0.0,
        'flow_p5_dollar': 0.0, 'flow_p10_dollar': 0.0, 'flow_p25_dollar': 0.0,
        'flow_p50_dollar': 0.0, 'flow_p75_dollar': 0.0, 'flow_p90_dollar': 0.0,
        'flow_p95_dollar': 0.0,
        'avg_time_between_flows_hours': 0.0, 'cash_flow_frequency': 0.0,
        'max_balance_drawdown_dollar': 0.0, 'avg_balance_drawdown_dollar': 0.0,
    }


def _calculate_cash_flow(
    balance_array: np.ndarray,
    pnl_array: np.ndarray,
    df_ledger: pd.DataFrame,
    returns_df: Optional[pd.DataFrame] = None,
    true_initial: Optional[float] = None
) -> Dict[str, Any]:
    """Cash flow analysis matching VBT's approach using actual trade cash flows."""

    if len(balance_array) == 0 or len(df_ledger) == 0:
        return _get_empty_cash_flow()

    # ── Use daily value array from returns_df for balance stats ───────────
    if returns_df is not None and not returns_df.empty and 'balance' in returns_df.columns:
        val_arr = returns_df['balance'].values
    else:
        val_arr = balance_array

    cb_initial = float(true_initial) if true_initial is not None else float(val_arr[0])
    cb_current = float(val_arr[-1])
    cb_min     = float(np.min(val_arr))
    cb_max     = float(np.max(val_arr))
    cb_mean    = float(np.mean(val_arr))
    cb_vol     = float(np.std(val_arr, ddof=1)) if len(val_arr) > 1 else 0.0

    # ── Reconstruct cash flows matching VBT's daily_pf.cash_flow ─────────
    # VBT's cash_flow = daily changes in the CASH account (not portfolio value)
    # On entry day: cash drops by position_value + entry_fee
    # On exit day: cash rises by position_value + PnL - exit_fee
    # These are tracked per day (not netted when same-day entry+exit)
    
    balance_vals = df_ledger['balance'].values
    balance_before = np.empty(len(balance_vals))
    if 'account_return_pct' in df_ledger.columns:
        first_ret = df_ledger['account_return_pct'].iloc[0] / 100.0
        balance_before[0] = balance_vals[0] / (1.0 + first_ret) if (1.0 + first_ret) != 0 else balance_vals[0]
    else:
        balance_before[0] = balance_vals[0]
    if len(balance_vals) > 1:
        balance_before[1:] = balance_vals[:-1]

    pos_frac = df_ledger['position_size_pct'].values / 100.0 if 'position_size_pct' in df_ledger.columns else np.ones(len(df_ledger))
    position_values = balance_before * pos_frac

    entry_fee_pct = df_ledger['entry_fee_pct'].values / 100.0 if 'entry_fee_pct' in df_ledger.columns else np.zeros(len(df_ledger))
    exit_fee_pct  = df_ledger['exit_fee_pct'].values  / 100.0 if 'exit_fee_pct'  in df_ledger.columns else np.zeros(len(df_ledger))
    entry_fees = position_values * entry_fee_pct
    exit_fees  = position_values * exit_fee_pct
    actual_fees = entry_fees + exit_fees

    if 'account_return_pct' in df_ledger.columns:
        pnl_per_trade = balance_before * (df_ledger['account_return_pct'].values / 100.0)
    else:
        pnl_per_trade = np.zeros(len(df_ledger))

    # Build separate entry and exit cash flow events (NOT netted per day)
    # Each entry = negative cash flow, each exit = positive cash flow
    entry_flows = -(position_values + entry_fees)
    exit_flows  =  (position_values + pnl_per_trade - exit_fees)

    # Combine into single array of all cash flow events
    all_flows = np.concatenate([entry_flows, exit_flows])

    # ── Cash flow statistics ──────────────────────────────────────────────
    n_cf    = len(all_flows)
    total_cf = float(val_arr[-1] - cb_initial)
    avg_cf   = float(total_cf / len(val_arr))
    cf_vol   = float(np.std(all_flows, ddof=1)) if n_cf > 1 else 0.0
    cf_mean  = float(np.mean(all_flows)) if n_cf > 0 else 0.0

    pos_cf  = all_flows[all_flows > 0]
    neg_cf  = all_flows[all_flows < 0]
    zero_cf = all_flows[all_flows == 0]

    gross_in  = float(np.sum(pos_cf))      if len(pos_cf) > 0 else 0.0
    gross_out = float(abs(np.sum(neg_cf))) if len(neg_cf) > 0 else 0.0
    net_cf    = gross_in - gross_out

    # Period counts: aggregate by day first (matching VBT's daily_pf.cash_flow)
    if returns_df is not None and not returns_df.empty:
        daily_dates = pd.to_datetime(returns_df['datetime']).dt.normalize()
        daily_cf_series = pd.Series(0.0, index=daily_dates)
    else:
        all_dates = pd.to_datetime(df_ledger['entry_datetime']).dt.normalize().tolist() + \
                    pd.to_datetime(df_ledger['exit_datetime']).dt.normalize().tolist()
        date_range = pd.date_range(min(all_dates), max(all_dates), freq='D')
        daily_cf_series = pd.Series(0.0, index=date_range)

    # Add entry flows by day
    entry_dates = pd.to_datetime(df_ledger['entry_datetime']).dt.normalize()
    for date, flow in zip(entry_dates, entry_flows):
        if date in daily_cf_series.index:
            daily_cf_series[date] += flow

    # Add exit flows by day
    exit_dates = pd.to_datetime(df_ledger['exit_datetime']).dt.normalize()
    for date, flow in zip(exit_dates, exit_flows):
        if date in daily_cf_series.index:
            daily_cf_series[date] += flow

    daily_cf_vals = daily_cf_series.values
    pos_count  = int(np.sum(daily_cf_vals > 0))
    neg_count  = int(np.sum(daily_cf_vals < 0))
    zero_count = int(np.sum(daily_cf_vals == 0))
    n_days     = len(daily_cf_vals)
    pos_pct    = float(pos_count  / n_days * 100) if n_days > 0 else 0.0
    neg_pct    = float(neg_count  / n_days * 100) if n_days > 0 else 0.0
    zero_pct   = float(zero_count / n_days * 100) if n_days > 0 else 0.0

    max_pos = float(np.max(pos_cf))  if len(pos_cf) > 0 else 0.0
    max_neg = float(np.min(neg_cf))  if len(neg_cf) > 0 else 0.0
    avg_pos = float(np.mean(pos_cf)) if len(pos_cf) > 0 else 0.0
    avg_neg = float(np.mean(neg_cf)) if len(neg_cf) > 0 else 0.0

    # ── Fees ──────────────────────────────────────────────────────────────
    total_fees = float(np.sum(actual_fees))
    avg_fee    = total_fees / len(df_ledger)
    fee_pct    = float(total_fees / cb_initial * 100) if cb_initial > 0 else 0.0

    # ── Advanced metrics ──────────────────────────────────────────────────
    cf_coeff_var = float(cf_vol / abs(cf_mean)) if abs(cf_mean) > 0 else 0.0
    try:
        from scipy.stats import skew, kurtosis
        cf_skew = float(skew(all_flows, bias=False))                  if n_cf > 2 else 0.0
        cf_kurt = float(kurtosis(all_flows, bias=False, fisher=True)) if n_cf > 2 else 0.0
    except Exception:
        cf_skew = cf_kurt = 0.0

    cf_efficiency = float(gross_in / gross_out) if gross_out > 0 else 0.0
    cf_sharpe     = float(cf_mean / cf_vol)     if cf_vol   > 0 else 0.0
    cf_stability  = float(1.0 - cf_vol / abs(cb_mean)) if abs(cb_mean) > 0 else 0.0

    # ── Utilization ───────────────────────────────────────────────────────
    util_avg = float(cb_mean / cb_initial * 100) if cb_initial != 0 else 0.0
    util_min = float(cb_min  / cb_initial * 100) if cb_initial != 0 else 0.0
    util_max = float(cb_max  / cb_initial * 100) if cb_initial != 0 else 0.0

    # ── Percentiles ───────────────────────────────────────────────────────
    fp = np.percentile(all_flows, [5, 10, 25, 50, 75, 90, 95]) if n_cf > 0 else np.zeros(7)

    # ── Time-based ────────────────────────────────────────────────────────
    avg_time_hours = 0.0
    cf_frequency   = 0.0
    if 'exit_datetime' in df_ledger.columns and len(df_ledger) > 1:
        exits = pd.to_datetime(df_ledger['exit_datetime']).sort_values()
        diffs = exits.diff().dt.total_seconds().dropna().values
        avg_time_hours = float(np.mean(diffs)) / 3600.0 if len(diffs) > 0 else 0.0
        span = (exits.iloc[-1] - exits.iloc[0]).total_seconds()
        cf_frequency = float(len(exits) / span * 86400) if span > 0 else 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────
    peak   = np.maximum.accumulate(val_arr)
    dd_arr = peak - val_arr
    max_dd = float(np.max(dd_arr))
    avg_dd = float(np.mean(dd_arr))

    return {
        'total_cash_flow_dollar':             float(total_cf),
        'avg_cash_flow_dollar':               float(avg_cf),
        'cash_flow_volatility_dollar':        float(cf_vol),
        'cash_flow_mean_dollar':              float(cf_mean),
        'positive_cash_flow_dollar':          float(gross_in),
        'negative_cash_flow_dollar':          float(gross_out),
        'net_cash_flow_dollar':               float(net_cf),
        'gross_cash_inflow_dollar':           float(gross_in),
        'gross_cash_outflow_dollar':          float(gross_out),
        'max_positive_flow_dollar':           float(max_pos),
        'max_negative_flow_dollar':           float(max_neg),
        'avg_positive_flow_dollar':           float(avg_pos),
        'avg_negative_flow_dollar':           float(avg_neg),
        'positive_periods_count':             pos_count,
        'negative_periods_count':             neg_count,
        'neutral_periods_count':              zero_count,
        'positive_periods_pct':               pos_pct,
        'negative_periods_pct':               neg_pct,
        'neutral_periods_pct':                zero_pct,
        'cash_balance_current_dollar':        cb_current,
        'cash_balance_initial_dollar':        cb_initial,
        'cash_balance_min_dollar':            cb_min,
        'cash_balance_max_dollar':            cb_max,
        'cash_balance_mean_dollar':           cb_mean,
        'cash_balance_volatility_dollar':     cb_vol,
        'total_fees_paid':                    float(total_fees),
        'avg_fee_per_trade':                  float(avg_fee),
        'fee_pct':                            float(fee_pct),
        'cash_flow_coefficient_of_variation': float(cf_coeff_var),
        'cash_flow_skewness':                 float(cf_skew),
        'cash_flow_kurtosis':                 float(cf_kurt),
        'cash_flow_efficiency':               float(cf_efficiency),
        'cash_flow_sharpe':                   float(cf_sharpe),
        'cash_flow_stability':                float(cf_stability),
        'cash_utilization_avg_pct':           float(util_avg),
        'cash_utilization_min_pct':           float(util_min),
        'cash_utilization_max_pct':           float(util_max),
        'flow_p5_dollar':  float(fp[0]), 'flow_p10_dollar': float(fp[1]),
        'flow_p25_dollar': float(fp[2]), 'flow_p50_dollar': float(fp[3]),
        'flow_p75_dollar': float(fp[4]), 'flow_p90_dollar': float(fp[5]),
        'flow_p95_dollar': float(fp[6]),
        'avg_time_between_flows_hours':       float(avg_time_hours),
        'cash_flow_frequency':                float(cf_frequency),
        'max_balance_drawdown_dollar':        float(max_dd),
        'avg_balance_drawdown_dollar':        float(avg_dd),
    }
