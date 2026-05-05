"""
Cash flow analysis — all from cache arrays (value, cash_flow).
No ledger or core_stats_raw required.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any


def _get_empty_cash_flow() -> Dict[str, Any]:
    return {
        'total_cash_flow_dollar':    0.0,
        'avg_cash_flow_dollar':      0.0,
        'cash_flow_volatility_dollar': 0.0,
        'cash_flow_mean_dollar':     0.0,
        'positive_cash_flow_dollar': 0.0,
        'negative_cash_flow_dollar': 0.0,
        'net_cash_flow_dollar':      0.0,
        'gross_cash_inflow_dollar':  0.0,
        'gross_cash_outflow_dollar': 0.0,
        'max_positive_flow_dollar':  0.0,
        'max_negative_flow_dollar':  0.0,
        'avg_positive_flow_dollar':  0.0,
        'avg_negative_flow_dollar':  0.0,
        'positive_periods_count':    0,
        'negative_periods_count':    0,
        'neutral_periods_count':     0,
        'positive_periods_pct':      0.0,
        'negative_periods_pct':      0.0,
        'neutral_periods_pct':       0.0,
        'cash_balance_current_dollar': 0.0,
        'cash_balance_initial_dollar': 0.0,
        'cash_balance_min_dollar':   0.0,
        'cash_balance_max_dollar':   0.0,
        'cash_balance_mean_dollar':  0.0,
        'cash_balance_volatility_dollar': 0.0,
        'total_fees_paid':           0.0,
        'avg_fee_per_trade':         0.0,
        'fee_pct':                   0.0,
        'cash_flow_coefficient_of_variation': 0.0,
        'cash_flow_skewness':        0.0,
        'cash_flow_kurtosis':        0.0,
        'cash_flow_efficiency':      0.0,
        'cash_flow_sharpe':          0.0,
        'cash_flow_stability':       0.0,
        'cash_utilization_avg_pct':  0.0,
        'cash_utilization_min_pct':  0.0,
        'cash_utilization_max_pct':  0.0,
        'flow_p5_dollar':  0.0, 'flow_p10_dollar': 0.0,
        'flow_p25_dollar': 0.0, 'flow_p50_dollar': 0.0,
        'flow_p75_dollar': 0.0, 'flow_p90_dollar': 0.0,
        'flow_p95_dollar': 0.0,
        'avg_time_between_flows_hours': 0.0,
        'cash_flow_frequency':       0.0,
        'max_balance_drawdown_dollar': 0.0,
        'avg_balance_drawdown_dollar': 0.0,
    }


def _extract_cash_flow_stats_full(cache: Dict) -> Dict[str, Any]:
    """
    Comprehensive cash flow stats from cache only.
    Sources:
      cache['value_array']      — portfolio equity curve (D)
      cache['cash_flow_array']  — daily cash flow changes (D)
      cache['total_fees_paid']  — pre-computed scalar (V)
      cache['trades_df']        — trade count + exit timestamps (O)
    """
    val_arr = cache.get('value_array', np.array([]))
    cf_arr  = cache.get('cash_flow_array', np.array([]))

    if len(val_arr) == 0:
        result = _get_empty_cash_flow()
        result['total_fees_paid'] = float(cache.get('total_fees_paid', 0.0))
        return result

    # Cash flow changes: prefer cash_flow_array, else derive from value diff
    balance_changes = cf_arr if len(cf_arr) > 0 else np.diff(val_arr)
    n_cf = len(balance_changes)

    total_cf   = float(val_arr[-1] - val_arr[0])
    avg_cf     = float(total_cf / len(val_arr))
    cf_vol     = float(np.std(balance_changes, ddof=1)) if n_cf > 1 else 0.0
    cf_mean    = float(np.mean(balance_changes)) if n_cf > 0 else 0.0

    pos_cf  = balance_changes[balance_changes > 0]
    neg_cf  = balance_changes[balance_changes < 0]
    zero_cf = balance_changes[balance_changes == 0]

    gross_in  = float(np.sum(pos_cf))       if len(pos_cf) > 0 else 0.0
    gross_out = float(abs(np.sum(neg_cf)))  if len(neg_cf) > 0 else 0.0
    net_cf    = gross_in - gross_out

    pos_count  = int(len(pos_cf))
    neg_count  = int(len(neg_cf))
    zero_count = int(len(zero_cf))
    pos_pct    = float(pos_count  / n_cf * 100) if n_cf > 0 else 0.0
    neg_pct    = float(neg_count  / n_cf * 100) if n_cf > 0 else 0.0
    zero_pct   = float(zero_count / n_cf * 100) if n_cf > 0 else 0.0

    max_pos = float(np.max(pos_cf))  if len(pos_cf) > 0 else 0.0
    max_neg = float(np.min(neg_cf))  if len(neg_cf) > 0 else 0.0
    avg_pos = float(np.mean(pos_cf)) if len(pos_cf) > 0 else 0.0
    avg_neg = float(np.mean(neg_cf)) if len(neg_cf) > 0 else 0.0

    # Balance stats
    cb_current = float(val_arr[-1])
    cb_initial = float(val_arr[0])
    cb_min     = float(np.min(val_arr))
    cb_max     = float(np.max(val_arr))
    cb_mean    = float(np.mean(val_arr))
    cb_vol     = float(np.std(val_arr, ddof=1)) if len(val_arr) > 1 else 0.0

    # Fees — pre-computed in cache
    total_fees = float(cache.get('total_fees_paid', 0.0))
    trades_df  = cache.get('trades_df', pd.DataFrame())
    n_trades   = len(trades_df) or 1
    avg_fee    = total_fees / n_trades
    fee_pct    = float(total_fees / cb_initial * 100) if cb_initial > 0 else 0.0

    # Advanced metrics
    cf_coeff_var = float(cf_vol / abs(cf_mean)) if abs(cf_mean) > 0 else 0.0
    try:
        from scipy.stats import skew, kurtosis
        cf_skew = float(skew(balance_changes, bias=False))     if n_cf > 2 else 0.0
        cf_kurt = float(kurtosis(balance_changes, bias=False, fisher=True)) if n_cf > 2 else 0.0
    except Exception:
        cf_skew = cf_kurt = 0.0
    cf_efficiency = float(gross_in / gross_out) if gross_out > 0 else 0.0
    cf_sharpe     = float(cf_mean / cf_vol)     if cf_vol   > 0 else 0.0
    cf_stability  = float(1.0 - cf_vol / abs(cb_mean)) if abs(cb_mean) > 0 else 0.0

    # Utilization (% of initial)
    util_avg = float(cb_mean / cb_initial * 100) if cb_initial != 0 else 0.0
    util_min = float(cb_min  / cb_initial * 100) if cb_initial != 0 else 0.0
    util_max = float(cb_max  / cb_initial * 100) if cb_initial != 0 else 0.0

    # Percentiles
    fp = np.percentile(balance_changes, [5, 10, 25, 50, 75, 90, 95]) if n_cf > 0 else np.zeros(7)

    # Time-based from trades_df Exit Index
    avg_time_hours = 0.0
    cf_frequency   = 0.0
    if not trades_df.empty and 'Exit Index' in trades_df.columns:
        try:
            exits = pd.to_datetime(trades_df['Exit Index']).sort_values()
            if len(exits) > 1:
                diffs = exits.diff().dt.total_seconds().dropna().values
                avg_time_hours = float(np.mean(diffs)) / 3600.0
                span = (exits.iloc[-1] - exits.iloc[0]).total_seconds()
                cf_frequency = float(len(exits) / span * 86400) if span > 0 else 0.0
        except Exception:
            pass

    # Balance drawdown
    peak     = np.maximum.accumulate(val_arr)
    dd_arr   = peak - val_arr
    max_dd   = float(np.max(dd_arr))
    avg_dd   = float(np.mean(dd_arr))

    return {
        'total_cash_flow_dollar':         float(total_cf),
        'avg_cash_flow_dollar':           float(avg_cf),
        'cash_flow_volatility_dollar':    float(cf_vol),
        'cash_flow_mean_dollar':          float(cf_mean),
        'positive_cash_flow_dollar':      float(gross_in),
        'negative_cash_flow_dollar':      float(abs(np.sum(neg_cf)) if len(neg_cf) > 0 else 0.0),
        'net_cash_flow_dollar':           float(net_cf),
        'gross_cash_inflow_dollar':       float(gross_in),
        'gross_cash_outflow_dollar':      float(gross_out),
        'max_positive_flow_dollar':       float(max_pos),
        'max_negative_flow_dollar':       float(max_neg),
        'avg_positive_flow_dollar':       float(avg_pos),
        'avg_negative_flow_dollar':       float(avg_neg),
        'positive_periods_count':         pos_count,
        'negative_periods_count':         neg_count,
        'neutral_periods_count':          zero_count,
        'positive_periods_pct':           pos_pct,
        'negative_periods_pct':           neg_pct,
        'neutral_periods_pct':            zero_pct,
        'cash_balance_current_dollar':    cb_current,
        'cash_balance_initial_dollar':    cb_initial,
        'cash_balance_min_dollar':        cb_min,
        'cash_balance_max_dollar':        cb_max,
        'cash_balance_mean_dollar':       cb_mean,
        'cash_balance_volatility_dollar': cb_vol,
        'total_fees_paid':                float(total_fees),
        'avg_fee_per_trade':              float(avg_fee),
        'fee_pct':                        float(fee_pct),
        'cash_flow_coefficient_of_variation': float(cf_coeff_var),
        'cash_flow_skewness':             float(cf_skew),
        'cash_flow_kurtosis':             float(cf_kurt),
        'cash_flow_efficiency':           float(cf_efficiency),
        'cash_flow_sharpe':               float(cf_sharpe),
        'cash_flow_stability':            float(cf_stability),
        'cash_utilization_avg_pct':       float(util_avg),
        'cash_utilization_min_pct':       float(util_min),
        'cash_utilization_max_pct':       float(util_max),
        'flow_p5_dollar':  float(fp[0]), 'flow_p10_dollar': float(fp[1]),
        'flow_p25_dollar': float(fp[2]), 'flow_p50_dollar': float(fp[3]),
        'flow_p75_dollar': float(fp[4]), 'flow_p90_dollar': float(fp[5]),
        'flow_p95_dollar': float(fp[6]),
        'avg_time_between_flows_hours':   float(avg_time_hours),
        'cash_flow_frequency':            float(cf_frequency),
        'max_balance_drawdown_dollar':    float(max_dd),
        'avg_balance_drawdown_dollar':    float(avg_dd),
    }
