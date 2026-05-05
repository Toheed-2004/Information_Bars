"""Plot data functions for VBT stats module.

This module provides chart-data helper functions. Implementations are
embedded here (mirrored from `common.stats.custom.plots`) so callers
can import a single `vectorbtpro.plots` module.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from typing import Dict, Any

from ..shared.utils import _max_consecutive_numpy, ANN_FACTOR, _rolling_windows
from bitpredict.common.monte_carlo import monte_carlo_simulations

def get_data_monte_carlo(cache, ledger_input, n_simulations=1000, method="shuffle"):
    """
    Run Monte Carlo simulations by shuffling trade order.
    Uses daily-resampled equity (cache['daily_pf']) — consistent with all other stats.
    """
    if ledger_input is None or len(ledger_input) == 0:
        return None

    # Get actual equity curve from daily-resampled portfolio (master, req 2)
    daily_pf = cache.get('daily_pf')
    actual_equity = cache.get('value_array', np.array([]))
    if len(actual_equity) == 0 or daily_pf is None:
        return None
    actual_dates = daily_pf.value.index
    n_points     = len(actual_equity)
    
    # Initial balance for simulations
    initial_balance = ledger_input["balance"].iloc[0] if "balance" in ledger_input.columns else 10000
    n_trades        = len(ledger_input)

    if n_trades < 2:
        return None

    # Run simulations
    simulated_equities = np.zeros((n_simulations, n_points))
    
    rng = np.random.RandomState(42)  # for reproducibility
    
    for i in range(n_simulations):
        try:
            # Generate one simulation
            sim_ledger = monte_carlo_simulations(
                ledger=ledger_input,
                method=method,
                random_state=rng
            )
            
            # Extract equity curve from simulation
            sim_equity = sim_ledger["balance"].values
            
            # Align simulation to actual timeline
            if len(sim_equity) == n_points:
                simulated_equities[i] = sim_equity
            else:
                # Create normalized index (0 to 1) for interpolation
                sim_x = np.linspace(0, 1, len(sim_equity))
                actual_x = np.linspace(0, 1, n_points)
                simulated_equities[i] = np.interp(actual_x, sim_x, sim_equity)
        
        except Exception:
            # Fill with initial balance (flat line)
            simulated_equities[i] = initial_balance

    # Calculate percentiles
    percentile_5  = np.percentile(simulated_equities, 5,  axis=0)
    percentile_95 = np.percentile(simulated_equities, 95, axis=0)
    
    # Calculate statistics
    final_actual     = actual_equity[-1]
    final_median_sim = np.median(simulated_equities[:, -1])
    final_p5         = percentile_5[-1]
    final_p95        = percentile_95[-1]

    # Convert to {timestamp: value} format
    actual_dates_str = pd.to_datetime(actual_dates).strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "mc_data": {
            "actual_equity": dict(zip(actual_dates_str, actual_equity.astype(float))),
            "percentile_5": dict(zip(actual_dates_str, percentile_5.astype(float))),
            "percentile_95": dict(zip(actual_dates_str, percentile_95.astype(float))),
            # Return subset of simulated paths to keep payload size reasonable
            "simulated_paths": [
                dict(zip(actual_dates_str, simulated_equities[i].astype(float)))
                for i in range(min(10, n_simulations))
            ],
            "stats": {
                "final_actual": round(float(final_actual), 2),
                "final_median_sim": round(float(final_median_sim), 2),
                "final_p5": round(float(final_p5), 2),
                "final_p95": round(float(final_p95), 2),
                "n_trades": n_trades,
                "initial_balance": float(initial_balance),
                "n_simulations": n_simulations,
            }
        }
    }

def get_data_rolling_sharpe(cache: Dict, window: int = 126):
    """
    Vectorized rolling Sharpe and Sortino from daily_returns (cache).
    Time-indexed (not trade-indexed) — consistent with all other daily stats.
    annualisation: sqrt(ANN_FACTOR) applied to daily Sharpe.
    """
    daily_pf = cache.get('daily_pf')
    returns  = cache.get('daily_returns', np.array([]))

    if len(returns) == 0:
        return {"rolling_sharpe": {}, "rolling_sortino": {}}

    # Get datetime index from daily_pf
    try:
        dates = daily_pf.returns.dropna().index
        # Align with NaN-stripped returns
        dr_full = daily_pf.returns.values
        nan_mask = ~np.isnan(dr_full)
        dates = daily_pf.returns.index[nan_mask]
    except Exception:
        dates = pd.RangeIndex(len(returns))

    n   = len(returns)
    ann = np.sqrt(ANN_FACTOR)

    sharpe_arr  = np.zeros(n)
    sortino_arr = np.zeros(n)

    if n >= window:
        wins  = _rolling_windows(returns, window)
        means = np.mean(wins, axis=1)
        stds  = np.std(wins, axis=1, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            sharpe_arr[window - 1:] = np.where(stds > 0, means / stds * ann, 0.0)
        # Vectorized Sortino: RMS downside deviation
        neg_sq        = np.where(wins < 0, wins ** 2, 0.0)
        downside_devs = np.sqrt(np.mean(neg_sq, axis=1))
        sortino_arr[window - 1:] = np.where(downside_devs > 0, means / downside_devs * ann, 0.0)

    # Vectorised expanding-window warm-up (no Python loop)
    warmup_n = min(window - 1, n)
    if warmup_n > 0:
        wr     = returns[:warmup_n]
        counts = np.arange(1, warmup_n + 1, dtype=np.float64)
        cum_s  = np.cumsum(wr)
        cum_sq = np.cumsum(wr ** 2)
        means  = cum_s / counts
        # Sample variance: (Σx² − n·μ²) / (n−1), safe for n=1
        variances = np.where(
            counts > 1,
            (cum_sq - cum_s ** 2 / counts) / (counts - 1),
            0.0,
        )
        stds = np.sqrt(np.maximum(variances, 0.0))
        sharpe_arr[:warmup_n] = np.where(stds > 0, means / stds * ann, 0.0)
        # Sortino: expanding downside-deviation via prefix sums of neg² returns
        cum_neg_sq = np.cumsum(np.where(wr < 0, wr ** 2, 0.0))
        ds_devs = np.sqrt(cum_neg_sq / counts)
        sortino_arr[:warmup_n] = np.where(ds_devs > 0, means / ds_devs * ann, 0.0)

    try:
        dt_strs = pd.DatetimeIndex(dates).strftime('%Y-%m-%d %H:%M:%S').tolist()
    except Exception:
        dt_strs = [str(i) for i in range(n)]

    return {
        "rolling_sharpe":  dict(zip(dt_strs, sharpe_arr.tolist())),
        "rolling_sortino": dict(zip(dt_strs, sortino_arr.tolist())),
    }

def get_data_rolling_correlation(cache: Dict, window: int = 3):
    """
    Rolling correlation between daily strategy returns and buy-and-hold benchmark.
    Both series from cache — consistent, always available, no external param needed.
    """
    daily_pf   = cache.get('daily_pf')
    bm_series  = cache.get('bm_returns_series')

    if daily_pf is None or bm_series is None:
        return {"rolling_correlation": {}}

    try:
        strat_daily = daily_pf.returns.dropna()
        bm_daily    = bm_series.dropna()

        # Align on common dates
        common_idx     = strat_daily.index.intersection(bm_daily.index)
        if len(common_idx) < 2:
            return {"rolling_correlation": {}}
        strategy_returns = strat_daily.loc[common_idx].values
        benchmark_vals   = bm_daily.loc[common_idx].values
        exit_datetimes   = common_idx
    except Exception:
        return {"rolling_correlation": {}}
    
    # Vectorized rolling correlation
    n        = len(strategy_returns)
    corr_arr = np.zeros(n)

    if n >= window:
        wins_s = _rolling_windows(strategy_returns, window)
        wins_b = _rolling_windows(benchmark_vals, window)
        ds = wins_s - np.mean(wins_s, axis=1, keepdims=True)
        db = wins_b - np.mean(wins_b, axis=1, keepdims=True)
        num = np.sum(ds * db, axis=1)
        den = np.sqrt(np.sum(ds ** 2, axis=1) * np.sum(db ** 2, axis=1))
        corr_arr[window - 1:] = np.where(den > 0, num / den, 0.0)

    try:
        dt_strs = pd.DatetimeIndex(exit_datetimes).strftime('%Y-%m-%d %H:%M:%S').tolist()
    except Exception:
        dt_strs = [str(i) for i in range(n)]

    return {"rolling_correlation": dict(zip(dt_strs, corr_arr.tolist()))}

def get_data_directional_pnl(cache: Dict):
    """
    Get directional (Long/Short) cumulative trade return % series.
    Uses trades_df['Direction'], trades_df['Return'], trades_df['Exit Index'].
    """
    trades_df = cache.get('trades_df', pd.DataFrame())
    if trades_df.empty or 'Direction' not in trades_df.columns:
        return {"directional_pnl": {"long": {}, "short": {}}}

    def build(direction):
        # Filter creates a view, no copy needed
        sub = trades_df[trades_df['Direction'] == direction]
        if sub.empty:
            return {}
        sub['cum_pnl'] = (sub['Return'] * 100).cumsum()
        exit_times = pd.to_datetime(sub['Exit Index']).dt.strftime("%Y-%m-%d %H:%M:%S")
        return dict(zip(exit_times, sub['cum_pnl'].astype(float)))

    return {
        "directional_pnl": {
            "long":  build("Long"),
            "short": build("Short"),
        }
    }

def get_data_underwater(cache: Dict):
    """
    Daily underwater (drawdown %) from daily_pf — full time-series.

    daily_pf timestamps are midnight; ledger exit timestamps are intraday — matching is meaningless.
    Returns full daily drawdown series for smooth frontend chart.
    drawdown_periods from VBT's native records (durations already in days).
    """
    daily_pf = cache.get('daily_pf')
    drawdown_series: dict = {}
    drawdown_periods: list = []

    if daily_pf is None:
        return {"drawdown_series": drawdown_series, "drawdown_periods": drawdown_periods}

    # Full daily drawdown series from daily_pf.drawdown (always ≤ 0 by VBT convention)
    try:
        dd = daily_pf.drawdown.dropna()
        idx_strs = dd.index.strftime('%Y-%m-%d %H:%M:%S')
        drawdown_series = dict(zip(idx_strs, (dd * 100).astype(float).tolist()))
    except Exception:
        # Fallback: derive from value
        try:
            val = daily_pf.value.values
            running_max = np.maximum.accumulate(val)
            dd_arr = (val - running_max) / running_max * 100
            idx_strs = daily_pf.value.index.strftime('%Y-%m-%d %H:%M:%S')
            drawdown_series = dict(zip(idx_strs, dd_arr.astype(float).tolist()))
        except Exception:
            pass

    # Drawdown period records
    try:
        df = daily_pf.drawdowns.records_readable
        if not df.empty:
            drawdown_periods = df.to_dict('records')
    except Exception:
        pass

    return {
        "drawdown_series":  drawdown_series,
        "drawdown_periods": drawdown_periods,
    }

def get_data_cumulative_returns(cache: Dict):
    """
    Daily cumulative returns from daily_pf — full time-series, not filtered to exit timestamps.
    daily_pf timestamps are midnight; ledger exit timestamps are intraday — matching is meaningless.
    Returns {timestamp_str: cumulative_return} for every trading day.
    """
    daily_pf = cache.get('daily_pf')
    if daily_pf is None:
        return {"cumulative_return": {}}
    try:
        cum_ret = daily_pf.cumulative_returns
        cum_ret = cum_ret.dropna()
        idx_strs = cum_ret.index.strftime('%Y-%m-%d %H:%M:%S')
        return {"cumulative_return": dict(zip(idx_strs, cum_ret.astype(float).tolist()))}
    except Exception:
        return {"cumulative_return": {}}

def get_data_monthly_heatmap(cache: Dict):
    """
    Monthly returns heatmap from daily_returns (cache).
    Uses daily_pf.returns index for timestamps.
    """
    daily_returns = cache.get('daily_returns', np.array([]))
    if len(daily_returns) == 0:
        return {"heatmaps_data": {"monthly_matrix": [], "years": [], "months": []}}

    daily_pf = cache.get('daily_pf')
    try:
        # Build aligned series: drop NaN from daily_pf.returns, use that index
        dr_series_full = daily_pf.returns.dropna()
        if len(dr_series_full) != len(daily_returns):
            # Lengths differ — fall back to RangeIndex (won't have meaningful monthly dates)
            return {"heatmaps_data": {"monthly_matrix": [], "years": [], "months": []}}
        index = dr_series_full.index
    except Exception:
        return {"heatmaps_data": {"monthly_matrix": [], "years": [], "months": []}}

    returns_series = pd.Series(daily_returns, index=index)
    
    # Calculate monthly compounded returns using month-end frequency "M"
    monthly = (1 + returns_series).resample("M").prod() - 1
    monthly *= 100  # convert to percentage
    
    # Extra monthly stats
    stats = {}
    if len(monthly) > 0:
        stats["monthly_returns_mean_pct"] = float(np.mean(monthly))
        stats["monthly_returns_std_pct"] = float(np.std(monthly))
        stats["best_month_return_pct"] = float(np.max(monthly))
        stats["worst_month_return_pct"] = float(np.min(monthly))
        stats["positive_months_pct"] = float(np.sum(monthly > 0) / len(monthly) * 100)
        stats["monthly_win_rate"] = float(np.sum(monthly > 0) / len(monthly) * 100)
    
    # Prepare pivot table
    monthly_df = monthly.reset_index()
    monthly_df.columns = ['datetime', 'return']
    monthly_df["Year"] = monthly_df["datetime"].dt.year
    monthly_df["Month"] = monthly_df["datetime"].dt.month
    
    pivot = monthly_df.pivot_table(
        values='return',
        index="Year",
        columns="Month",
        fill_value=0
    )
    
    # Ensure all months 1-12 are present
    for month in range(1, 13):
        if month not in pivot.columns:
            pivot[month] = 0
    pivot = pivot[sorted(pivot.columns)]
    
    res = {
        "heatmaps_data": {
            "monthly_matrix": pivot.values.tolist(),
            # "years": pivot.index.tolist(),
            # "months": pivot.columns.tolist(),
            "monthly_returns_series": dict(zip(monthly.index.strftime("%Y-%m-%d"), monthly.astype(float).tolist()))
        }
    }
    res["heatmaps_data"].update(stats)
    return res

def get_mfe_mae(cache: Dict):
    """
    MFE and MAE percentages indexed by exit datetime.
    Uses cached mfe_values/mae_values from original pf.trades.
    Entry prices and exit timestamps from trades_df.
    """
    mfe_values = cache.get('mfe_values', np.array([]))
    mae_values = cache.get('mae_values', np.array([]))
    if len(mfe_values) == 0 or len(mae_values) == 0:
        return {"mfe_pct": {}, "mae_pct": {}}

    trades_df = cache.get('trades_df', pd.DataFrame())
    if trades_df.empty or 'Avg Entry Price' not in trades_df.columns or 'Exit Index' not in trades_df.columns:
        return {"mfe_pct": {}, "mae_pct": {}}

    entry_prices = trades_df['Avg Entry Price'].values.astype(float)
    exit_times   = pd.to_datetime(trades_df['Exit Index'])

    # Compute percentages safely
    with np.errstate(divide="ignore", invalid="ignore"):
        mfe_pct = np.where(entry_prices != 0, mfe_values / entry_prices * 100, np.nan)
        mae_pct = np.where(entry_prices != 0, mae_values / entry_prices * 100, np.nan)

    mfe_series = pd.Series(mfe_pct, index=exit_times).dropna()
    mae_series = pd.Series(mae_pct, index=exit_times).dropna()

    mfe_series.index = mfe_series.index.strftime("%Y-%m-%d %H:%M:%S")
    mae_series.index = mae_series.index.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "mfe_pct": mfe_series.astype(float).to_dict(),
        "mae_pct": mae_series.astype(float).to_dict(),
    }

def get_data_pnl_distribution(cache: Dict):
    """
    Get PnL distribution with adaptive point count based on trade count.
    Uses trades_df['Return'] * 100 (trade return %).
    """
    trades_df = cache.get('trades_df', pd.DataFrame())
    if trades_df.empty or 'Return' not in trades_df.columns:
        return {"pnl_distribution": {"histogram_values": [], "kde_curve": []}}

    pnl_pct  = trades_df['Return'].values.astype(float) * 100
    n_trades = len(pnl_pct)
    
    # Adaptive point count: more trades = more points, but cap it
    if n_trades <= 10:
        n_points = 30
    elif n_trades <= 50:
        n_points = 50
    elif n_trades <= 100:
        n_points = 75
    else:
        n_points = 100  # Cap at 100 points max
    
    # Add small jitter if all values are identical
    if len(np.unique(pnl_pct)) == 1:
        pnl_pct = pnl_pct + np.random.normal(0, 0.0001, len(pnl_pct))
    
    try:
        kde = gaussian_kde(pnl_pct)
        x_range = np.linspace(pnl_pct.min(), pnl_pct.max(), n_points)

        return {
            "pnl_distribution": {
                "histogram_values": pnl_pct.tolist(),
                "kde_curve": [
                    {"x": float(x), "density": float(y)}
                    for x, y in zip(x_range, kde(x_range))
                ],
            }
        }
    except:
        return {"pnl_distribution": {"histogram_values": pnl_pct.tolist(), "kde_curve": []}}
