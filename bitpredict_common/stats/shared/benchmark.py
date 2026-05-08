"""
Single source of truth for benchmark analysis.
Both custom/ and vectorbt_pro/ modules call calculate_benchmark_analysis().
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def _empty_benchmark_analysis() -> Dict[str, Any]:
    return {
        'portfolio_total_return_pct': 0.0, 'benchmark_total_return_pct': 0.0,
        'outperformance': 0.0, 'outperformance_ratio': 0.0,
        'tracking_error_pct': 0.0, 'information_ratio': 0.0,
        'active_return_mean_pct': 0.0, 'active_return_std_pct': 0.0,
        'up_capture_ratio': 0.0, 'down_capture_ratio': 0.0,
        'capture_ratio': 0.0,
        'correlation': 0.0, 'beta': 0.0, 'alpha_pct': 0.0,
        'rolling_correlation': 0.0,
        'outperforming_periods': 0, 'underperforming_periods': 0,
        'tie_periods': 0, 'outperforming_pct': 0.0, 'underperforming_pct': 0.0,
        'max_adverse_excursion_pct': 0.0, 'max_favorable_excursion_pct': 0.0,
        'comparison_periods': 0, 'benchmark_volatility_pct': 0.0,
    }


def _build_daily_returns(
    ledger: pd.DataFrame,
    benchmark_series: Optional[pd.Series] = None,
) -> tuple:
    """
    Build (portfolio_returns, benchmark_returns) as daily numpy arrays from ledger.
    Both aligned to same date range (first exit day → last exit day), idle days = 0.

    portfolio_returns: account_return_pct / 100 per trade, compounded daily.
    benchmark_returns: external series aligned to exit times (if provided),
                       else exit_price / entry_price - 1 per trade (buy-and-hold).
    """
    exit_dt = pd.to_datetime(ledger['exit_datetime'].values)
    if exit_dt.tz is not None:
        exit_dt = exit_dt.tz_localize(None)

    portfolio_per_trade = ledger['account_return_pct'].values / 100.0

    # --- Benchmark per trade ---
    if benchmark_series is not None:
        if isinstance(benchmark_series, pd.DataFrame):
            benchmark_series = benchmark_series.iloc[:, 0]
        bench_idx = pd.to_datetime(benchmark_series.index)
        if bench_idx.tz is not None:
            bench_idx = bench_idx.tz_localize(None)
        temp = pd.Series(benchmark_series.values, index=bench_idx).sort_index()
        # handle price series vs return series
        bench_cum = temp / temp.iloc[0] if np.mean(np.abs(temp.values)) > 2.0 else (1 + temp).cumprod()
        cum_at_exit = bench_cum.reindex(exit_dt, method='ffill').fillna(1.0).values
        cum_with_start = np.concatenate([[1.0], cum_at_exit])
        bm_per_trade = cum_with_start[1:] / cum_with_start[:-1] - 1
    elif 'avg_entry_price' in ledger.columns and 'avg_exit_price' in ledger.columns:
        entry_px = ledger['avg_entry_price'].values.astype(float)
        exit_px  = ledger['avg_exit_price'].values.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            bm_per_trade = np.where(entry_px > 0, exit_px / entry_px - 1, 0.0)
    else:
        bm_per_trade = np.zeros(len(portfolio_per_trade))

    # --- Resample to daily ---
    trade_df = pd.DataFrame(
        {'p': portfolio_per_trade, 'b': bm_per_trade},
        index=exit_dt,
    )
    daily = trade_df.resample('D').agg(
        p=('p', lambda x: float((1 + x).prod() - 1) if len(x) > 0 else 0.0),
        b=('b', lambda x: float((1 + x).prod() - 1) if len(x) > 0 else 0.0),
    )
    # drop days with no trades (NaN from empty resample buckets)
    daily = daily.dropna()

    # fill calendar gaps with 0
    date_range = pd.date_range(daily.index[0], daily.index[-1], freq='D')
    daily = daily.reindex(date_range, fill_value=0.0)

    return daily['p'].values, daily['b'].values


def _compute_stats(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    ann_factor: float,
) -> Dict[str, Any]:
    n = min(len(portfolio_returns), len(benchmark_returns))
    if n == 0:
        return _empty_benchmark_analysis()

    p = portfolio_returns[:n]
    b = benchmark_returns[:n]

    portfolio_total = float((np.prod(1 + p) - 1) * 100)
    benchmark_total = float((np.prod(1 + b) - 1) * 100)
    outperformance  = portfolio_total - benchmark_total
    outperformance_ratio = float(
        (1 + portfolio_total / 100) / (1 + benchmark_total / 100)
        if benchmark_total != -100 else 0.0
    )

    excess = p - b
    te_frac     = float(np.std(excess, ddof=1) * np.sqrt(ann_factor)) if n > 1 else 0.0
    te          = te_frac * 100
    ir          = float(np.mean(excess) * ann_factor / te_frac) if te_frac > 0 else 0.0
    active_mean = float(np.mean(excess) * ann_factor * 100)
    active_std  = float(np.std(excess, ddof=1) * np.sqrt(ann_factor) * 100) if n > 1 else 0.0

    up   = b > 0
    down = b < 0
    up_cap = 0.0
    down_cap = 0.0
    if np.any(up):
        up_cap = float(np.mean(p[up]) / np.mean(b[up])) if np.mean(b[up]) != 0 else 0.0
    if np.any(down):
        down_cap = float(np.mean(p[down]) / np.mean(b[down])) if np.mean(b[down]) != 0 else 0.0

    corr = beta = alpha = 0.0
    if n > 1:
        cm = np.corrcoef(p, b)
        corr = float(cm[0, 1]) if not np.isnan(cm[0, 1]) else 0.0
        bm_var = float(np.var(b, ddof=1))
        if bm_var > 0:
            beta = float(np.cov(p, b, ddof=1)[0, 1] / bm_var)
        alpha = float((np.mean(p) * ann_factor - beta * np.mean(b) * ann_factor) * 100)

    win  = int(np.sum(p > b))
    lose = int(np.sum(p < b))
    tie  = int(np.sum(p == b))

    win_pct  = float(win  / n * 100)
    lose_pct = float(lose / n * 100)

    # rolling_correlation: mean of 30-day rolling correlations (matching VBT scalar)
    w = 30
    if n >= w * 2:
        roll_corrs = []
        for i in range(w, n):
            window_p = p[i-w:i]
            window_b = b[i-w:i]
            if np.std(window_p) > 0 and np.std(window_b) > 0:
                rc = float(np.corrcoef(window_p, window_b)[0, 1])
                if not np.isnan(rc):
                    roll_corrs.append(rc)
        rolling_corr = float(np.mean(roll_corrs)) if roll_corrs else corr
    else:
        rolling_corr = corr

    # MAE/MFE: cumulative excess return (compounded) series
    cum_p = np.cumprod(1 + p) - 1
    cum_b = np.cumprod(1 + b) - 1
    cum_excess = cum_p - cum_b
    mae = float(np.min(cum_excess)  * 100)
    mfe = float(np.max(cum_excess)  * 100)
    bm_vol = float(np.std(b, ddof=1) * np.sqrt(ann_factor) * 100) if n > 1 else 0.0

    return {
        'portfolio_total_return_pct':  portfolio_total,
        'benchmark_total_return_pct':  benchmark_total,
        'outperformance':              outperformance,
        'outperformance_ratio':        outperformance_ratio,
        'tracking_error_pct':          te,
        'information_ratio':           ir,
        'active_return_mean_pct':      active_mean,
        'active_return_std_pct':       active_std,
        'up_capture_ratio':            up_cap,
        'down_capture_ratio':          down_cap,
        'capture_ratio':               float(up_cap / abs(down_cap)) if down_cap != 0 else 0.0,
        'correlation':                 corr,
        'beta':                        beta,
        'alpha_pct':                   alpha,
        'rolling_correlation':         rolling_corr,
        'outperforming_periods':       win,
        'underperforming_periods':     lose,
        'tie_periods':                 tie,
        'outperforming_pct':           win_pct,
        'underperforming_pct':         lose_pct,
        'max_adverse_excursion_pct':   mae,
        'max_favorable_excursion_pct': mfe,
        'comparison_periods':          n,
        'benchmark_volatility_pct':    bm_vol,
    }


def calculate_benchmark_analysis(
    ledger: Optional[pd.DataFrame] = None,
    benchmark_series: Optional[pd.Series] = None,
    ann_factor: float = 365.25,
    portfolio_daily_returns: Optional[np.ndarray] = None,
    benchmark_daily_returns: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate benchmark analysis.

    VBT path: pass portfolio_daily_returns + benchmark_daily_returns (cache arrays)
              — skips _build_daily_returns entirely.
    Custom path: pass ledger (+ optional benchmark_series) as before.
    """
    # VBT fast path — use pre-built daily arrays from cache
    if portfolio_daily_returns is not None and len(portfolio_daily_returns) > 0:
        bm = benchmark_daily_returns if (benchmark_daily_returns is not None and len(benchmark_daily_returns) > 0) \
             else np.zeros(len(portfolio_daily_returns))
        if np.all(bm == 0):
            result = _empty_benchmark_analysis()
            result['portfolio_total_return_pct'] = float((np.prod(1 + portfolio_daily_returns) - 1) * 100)
            return result
        return _compute_stats(portfolio_daily_returns, bm, ann_factor)

    # Custom / ledger path
    if ledger is None or len(ledger) == 0:
        return _empty_benchmark_analysis()
    try:
        portfolio_returns, benchmark_returns = _build_daily_returns(ledger, benchmark_series)
    except Exception:
        return _empty_benchmark_analysis()
    if len(portfolio_returns) == 0:
        return _empty_benchmark_analysis()

    bm_all_zero = np.all(benchmark_returns == 0)
    result = _compute_stats(portfolio_returns, benchmark_returns, ann_factor) if not bm_all_zero \
        else _empty_benchmark_analysis()
    result['portfolio_total_return_pct'] = float((np.prod(1 + portfolio_returns) - 1) * 100)
    return result
