import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from typing import Dict, Any
from ..shared.utils import _rolling_windows, ANN_FACTOR

def get_data_cumulative_returns_custom(returns_df: pd.DataFrame = None, df_ledger: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Cumulative returns series - daily resampled, starting from 1.0 like VBT.
    
    Args:
        returns_df: Daily returns DataFrame with 'datetime', 'portfolio_return', 'balance' columns
        df_ledger: Fallback to ledger-based calculation if returns_df not provided
    """
    # Use daily returns if available (preferred)
    if returns_df is not None and not returns_df.empty and 'portfolio_return' in returns_df.columns:
        # Calculate cumulative returns starting from 1.0
        cumulative = (1 + returns_df['portfolio_return']).cumprod()
        
        # Format timestamps
        timestamps = pd.to_datetime(returns_df['datetime']).dt.strftime("%Y-%m-%d %H:%M:%S")
        
        return {"cumulative_return": dict(zip(timestamps, cumulative.astype(float)))}
    
    # Fallback to ledger-based calculation (trade-level, not daily)
    if df_ledger is not None and not df_ledger.empty:
        balance_vals = df_ledger['balance'].values
        if 'account_return_pct' in df_ledger.columns:
            first_ret = df_ledger['account_return_pct'].iloc[0] / 100.0
            true_initial = balance_vals[0] / (1.0 + first_ret) if (1.0 + first_ret) != 0 else balance_vals[0]
        else:
            true_initial = float(balance_vals[0])

        if true_initial == 0:
            return {"cumulative_return": {}}

        # Cumulative returns starting from 1.0
        cumulative_returns = df_ledger['balance'] / true_initial
        exit_times = pd.to_datetime(df_ledger['exit_datetime']).dt.strftime("%Y-%m-%d %H:%M:%S")
        return {"cumulative_return": dict(zip(exit_times, cumulative_returns.astype(float)))}
    
    return {"cumulative_return": {}}


def get_benchmark_returns_custom(returns_df: pd.DataFrame = None, benchmark_data: Any = None) -> Dict[str, Any]:
    """
    Daily benchmark returns as percentage, matching VBT format.
    Prefers returns_df['benchmark_return'] (already daily), falls back to raw benchmark_data series.
    """
    # Use daily benchmark returns from returns_df (preferred - already aligned)
    if returns_df is not None and not returns_df.empty and 'benchmark_return' in returns_df.columns:
        timestamps = pd.to_datetime(returns_df['datetime']).dt.strftime("%Y-%m-%d %H:%M:%S")
        bm_pct = returns_df['benchmark_return'].values * 100  # convert to percentage
        return {"benchmark_returns": dict(zip(timestamps, bm_pct.astype(float)))}

    # Fallback to raw benchmark_data series
    if benchmark_data is None:
        return {"benchmark_returns": {}}
    if isinstance(benchmark_data, pd.DataFrame):
        benchmark_series = benchmark_data.iloc[:, 0]
    elif not isinstance(benchmark_data, pd.Series):
        return {"benchmark_returns": {}}
    else:
        benchmark_series = benchmark_data
    if benchmark_series.empty:
        return {"benchmark_returns": {}}
    idx = pd.to_datetime(benchmark_series.index)
    # If price series, convert to returns
    if np.mean(np.abs(benchmark_series.values)) > 2.0:
        bm_returns = benchmark_series.pct_change().fillna(0) * 100
    else:
        bm_returns = benchmark_series * 100
    return {"benchmark_returns": dict(zip(idx.strftime("%Y-%m-%d %H:%M:%S"), bm_returns.astype(float).values))}


def get_data_underwater_custom(returns_df: pd.DataFrame = None, df_ledger: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Drawdown series and periods - daily resampled like VBT.
    """
    if returns_df is not None and not returns_df.empty and 'balance' in returns_df.columns:
        balance_array = returns_df['balance'].values
        dates = pd.to_datetime(returns_df['datetime'])
        peak = np.maximum.accumulate(balance_array)
        drawdown = (balance_array / peak) - 1
        timestamps = dates.dt.strftime("%Y-%m-%d %H:%M:%S")

        # Build drawdown periods (matching VBT's drawdowns.records_readable format)
        # VBT: Start = last peak day (day before drop), End = first recovery day
        no_dd = drawdown >= -1e-10
        prev_no_dd = np.empty_like(no_dd)
        prev_no_dd[0] = True
        prev_no_dd[1:] = no_dd[:-1]

        # Start: last day at peak (day before drawdown begins) = index before first dd day
        starts_dd = np.flatnonzero((~no_dd) & prev_no_dd)  # first day IN drawdown
        # End: first day back at peak (recovery day) = index after last dd day
        ends_dd = np.flatnonzero(no_dd & (~prev_no_dd))    # first day OUT of drawdown

        if starts_dd.size and ends_dd.size and starts_dd[0] > ends_dd[0]:
            starts_dd = np.insert(starts_dd, 0, 0)
        if starts_dd.size and (ends_dd.size == 0 or starts_dd[-1] > ends_dd[-1]):
            ends_dd = np.append(ends_dd, len(drawdown) - 1)

        drawdown_periods = []
        for dd_id, (s, e) in enumerate(zip(starts_dd, ends_dd)):
            # Start Value: peak value = value on the day before drawdown (VBT uses last peak day)
            start_idx   = max(0, s - 1)  # day before drawdown starts = last peak day
            valley_idx  = s + int(np.argmin(drawdown[s:e+1]))
            # End: first recovery day (VBT) = e (first day no_dd after drawdown)
            end_idx     = e

            start_val  = float(balance_array[start_idx])
            valley_val = float(balance_array[valley_idx])
            end_val    = float(balance_array[end_idx])
            is_active  = (e == len(drawdown) - 1) and drawdown[-1] < -1e-10

            drawdown_periods.append({
                "Drawdown Id":  dd_id,
                "Column":       0,
                "Start Index":  dates.iloc[start_idx].isoformat(),
                "Valley Index": dates.iloc[valley_idx].isoformat(),
                "End Index":    dates.iloc[end_idx].isoformat(),
                "Start Value":  start_val,
                "Valley Value": valley_val,
                "End Value":    end_val,
                "Status":       "Active" if is_active else "Recovered",
            })

        return {
            "drawdown_series": dict(zip(timestamps, (drawdown * 100).astype(float))),
            "drawdown_periods": drawdown_periods
        }

    # Fallback to ledger-based calculation (trade-level)
    if df_ledger is not None and not df_ledger.empty:
        balance_array = df_ledger['balance'].values
        peak = np.maximum.accumulate(balance_array)
        drawdown = (balance_array / peak) - 1
        exit_times = pd.to_datetime(df_ledger['exit_datetime']).dt.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "drawdown_series": dict(zip(exit_times, (drawdown * 100).astype(float))),
            "drawdown_periods": []
        }

    return {"drawdown_series": {}, "drawdown_periods": []}


def get_mfe_mae_custom(df_ledger: pd.DataFrame) -> Dict[str, Any]:
    """
    MFE/MAE using trade_return_pct as proxy (no mfe/mae columns in ledger).
    """
    if df_ledger.empty or 'trade_return_pct' not in df_ledger.columns:
        return {"mfe_pct": {}, "mae_pct": {}}

    exit_times = pd.to_datetime(df_ledger['exit_datetime']).dt.strftime("%Y-%m-%d %H:%M:%S")
    ret = df_ledger['trade_return_pct'].values.astype(float)
    mfe_pct = np.where(ret > 0, ret, 0.0)
    mae_pct = np.where(ret < 0, ret, 0.0)

    return {
        "mfe_pct": dict(zip(exit_times, mfe_pct.astype(float))),
        "mae_pct": dict(zip(exit_times, mae_pct.astype(float))),
    }


def get_data_pnl_distribution_custom(df_ledger: pd.DataFrame) -> Dict[str, Any]:
    """PnL (account_return_pct) distribution with KDE."""
    # Use account_return_pct for actual account impact; fall back to trade_return_pct
    if df_ledger.empty:
        return {"pnl_distribution": {"histogram_values": [], "kde_curve": []}}
    if 'account_return_pct' in df_ledger.columns:
        pnl_pct = df_ledger['account_return_pct'].values
    elif 'trade_return_pct' in df_ledger.columns:
        pnl_pct = df_ledger['trade_return_pct'].values
    else:
        return {"pnl_distribution": {"histogram_values": [], "kde_curve": []}}

    n_trades = len(pnl_pct)
    n_points = 30 if n_trades <= 10 else (50 if n_trades <= 50 else 100)

    if len(np.unique(pnl_pct)) == 1:
        pnl_pct = pnl_pct + np.random.normal(0, 0.0001, len(pnl_pct))

    try:
        kde = gaussian_kde(pnl_pct)
        x_range = np.linspace(pnl_pct.min(), pnl_pct.max(), n_points)
        return {
            "pnl_distribution": {
                "histogram_values": pnl_pct.tolist(),
                "kde_curve": [{"x": float(x), "density": float(y)} for x, y in zip(x_range, kde(x_range))],
            }
        }
    except Exception:
        return {"pnl_distribution": {"histogram_values": pnl_pct.tolist(), "kde_curve": []}}


def get_data_directional_pnl_custom(df_ledger: pd.DataFrame) -> Dict[str, Any]:
    """Directional cumulative account_return_pct series."""
    if df_ledger.empty:
        return {"directional_pnl": {"long": {}, "short": {}}}

    ret_col = 'account_return_pct' if 'account_return_pct' in df_ledger.columns else (
        'trade_return_pct' if 'trade_return_pct' in df_ledger.columns else None
    )
    if ret_col is None:
        return {"directional_pnl": {"long": {}, "short": {}}}

    def build(direction):
        # Filter creates a view, no copy needed
        sub = df_ledger[df_ledger["direction"].str.lower() == direction.lower()]
        if sub.empty:
            return {}
        sub["cum_pnl"] = sub[ret_col].cumsum()
        exit_times = pd.to_datetime(sub["exit_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return dict(zip(exit_times, sub["cum_pnl"].astype(float)))

    return {"directional_pnl": {"long": build("Long"), "short": build("Short")}}


def get_data_rolling_sharpe_custom(returns_df: pd.DataFrame = None, df_ledger: pd.DataFrame = None, window: int = 30) -> Dict[str, Any]:
    """
    Rolling Sharpe and Sortino - daily resampled, matching VBT's exact formula.
    Sortino uses RMS downside deviation: sqrt(mean(neg_returns²))
    """
    from ..shared.utils import _rolling_windows

    # Use daily returns if available (preferred)
    if returns_df is not None and not returns_df.empty and 'portfolio_return' in returns_df.columns:
        returns = returns_df['portfolio_return'].values
        timestamps = pd.to_datetime(returns_df['datetime']).dt.strftime("%Y-%m-%d %H:%M:%S").values
    elif df_ledger is not None and not df_ledger.empty:
        if 'account_return_pct' in df_ledger.columns:
            returns = df_ledger['account_return_pct'].values / 100.0
        elif 'trade_return_pct' in df_ledger.columns:
            returns = df_ledger['trade_return_pct'].values / 100.0
        else:
            return {"rolling_sharpe": {}, "rolling_sortino": {}}
        timestamps = pd.to_datetime(df_ledger['exit_datetime']).dt.strftime("%Y-%m-%d %H:%M:%S").values
    else:
        return {"rolling_sharpe": {}, "rolling_sortino": {}}

    n   = len(returns)
    ann = np.sqrt(ANN_FACTOR)
    sharpe_arr  = np.zeros(n)
    sortino_arr = np.zeros(n)

    # Full rolling window (vectorized)
    if n >= window:
        wins  = _rolling_windows(returns, window)          # (n-w+1, w)
        means = np.mean(wins, axis=1)
        stds  = np.std(wins, axis=1, ddof=1)
        sharpe_arr[window - 1:] = np.where(stds > 0, means / stds * ann, 0.0)
        # Sortino: RMS downside deviation (matching VBT)
        neg_sq        = np.where(wins < 0, wins ** 2, 0.0)
        downside_devs = np.sqrt(np.mean(neg_sq, axis=1))
        sortino_arr[window - 1:] = np.where(downside_devs > 0, means / downside_devs * ann, 0.0)

    # Expanding warm-up (vectorized, matching VBT)
    warmup_n = min(window - 1, n)
    if warmup_n > 0:
        wr     = returns[:warmup_n]
        counts = np.arange(1, warmup_n + 1, dtype=np.float64)
        cum_s  = np.cumsum(wr)
        cum_sq = np.cumsum(wr ** 2)
        means  = cum_s / counts
        variances = np.where(counts > 1, (cum_sq - cum_s ** 2 / counts) / (counts - 1), 0.0)
        stds = np.sqrt(np.maximum(variances, 0.0))
        sharpe_arr[:warmup_n] = np.where(stds > 0, means / stds * ann, 0.0)
        cum_neg_sq = np.cumsum(np.where(wr < 0, wr ** 2, 0.0))
        ds_devs = np.sqrt(cum_neg_sq / counts)
        sortino_arr[:warmup_n] = np.where(ds_devs > 0, means / ds_devs * ann, 0.0)

    return {
        "rolling_sharpe":  dict(zip(timestamps, sharpe_arr.tolist())),
        "rolling_sortino": dict(zip(timestamps, sortino_arr.tolist())),
    }


def get_data_rolling_correlation_custom(df_ledger, benchmark_returns, window: int = 3) -> Dict[str, Any]:
    """Rolling correlation with benchmark using account_return_pct."""
    if df_ledger.empty or benchmark_returns is None:
        return {"rolling_correlation": {}}

    exit_datetimes = pd.to_datetime(df_ledger['exit_datetime'])
    if exit_datetimes.dt.tz is not None:
        exit_datetimes = exit_datetimes.dt.tz_localize(None)

    if 'account_return_pct' in df_ledger.columns:
        returns = df_ledger['account_return_pct'].values / 100.0
    else:
        returns = np.zeros(len(df_ledger))

    # No copy needed - we only read from benchmark_returns
    bm = benchmark_returns
    if hasattr(bm, 'index') and bm.index.tz is not None:
        bm.index = bm.index.tz_localize(None)

    try:
        benchmark_at_exits = bm.reindex(exit_datetimes, method='nearest').values
    except Exception:
        benchmark_at_exits = np.zeros(len(returns))

    corr_dict = {}
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        if i - start >= 1:
            s = returns[start: i + 1]
            b = benchmark_at_exits[start: i + 1]
            if len(s) > 1 and np.std(s) > 0 and np.std(b) > 0:
                corr = np.corrcoef(s, b)[0, 1]
                corr = 0.0 if np.isnan(corr) else corr
            else:
                corr = 0.0
        else:
            corr = 0.0
        corr_dict[exit_datetimes.iloc[i].strftime('%Y-%m-%d %H:%M:%S')] = float(corr)

    return {"rolling_correlation": corr_dict}


def get_data_monthly_heatmap_custom(df_ledger, returns_df=None) -> Dict[str, Any]:
    """Monthly returns heatmap - uses monthly_breakdown from returns_df for consistency."""
    from .performance import _calculate_monthly_breakdown

    # Get monthly breakdown (uses returns_df if available, matching VBT)
    monthly_breakdown = _calculate_monthly_breakdown(df_ledger, returns_df=returns_df)

    if not monthly_breakdown:
        return {"heatmaps_data": {}}

    # Build pivot from monthly_breakdown dict {YYYY-MM-DD: return_pct}
    records = []
    for date_str, ret in monthly_breakdown.items():
        dt = pd.to_datetime(date_str)
        records.append({'Year': dt.year, 'Month': dt.month, 'return': ret})

    monthly = pd.DataFrame(records)
    pivot = monthly.pivot_table(index='Year', columns='Month', values='return', fill_value=0)
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = 0.0
    pivot = pivot[sorted(pivot.columns)]

    return {
        "heatmaps_data": {
            "monthly_matrix": pivot.values.tolist(),
            "years": pivot.index.tolist(),
            "months": pivot.columns.tolist(),
            "monthly_returns_series": {}
        }
    }
