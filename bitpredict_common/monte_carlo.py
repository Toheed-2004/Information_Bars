"""
Monte Carlo Analysis — Common Module.

Two public functions:

``monte_carlo_simulations(ledger, method, random_state)``
    Generate ONE simulated ledger by resampling trade returns.
    Used internally by ``run_monte_carlo_analysis`` and available
    for ad-hoc use (research notebooks, unit tests, etc.).

``run_monte_carlo_analysis(ledger, config)``
    Full Monte Carlo analysis: runs n_sims simulations, collects
    comprehensive statistics across the return / Sharpe / drawdown
    distributions.  Config-driven, no gates — gates belong in the
    pipeline layer.

Config keys  (``monte_carlo`` section):
  n_simulations         : int   (default 500)
  method                : str   (default "stationary")
  random_seed           : int   (default 42)
  worst_case_percentile : int   (default 5)

Simulation methods
------------------
shuffle        : Permute returns — i.i.d. null model
bootstrap      : i.i.d. resample with replacement
hybrid         : Single random contiguous block
moving_block   : Moving block bootstrap (fixed block length)
stationary     : Stationary bootstrap / Politis-Romano (random block
                 lengths, preserves autocorrelation) — recommended default
parametric     : Fit t-distribution to returns and sample from it
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as scipy_t
from typing import Any, Dict, Optional

# Avoid circular import — only imported inside run_monte_carlo_analysis
# from bitpredict.common.stats import calculate_essential_stats


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

def monte_carlo_simulations(
    ledger: pd.DataFrame,
    method: str = "stationary",
    random_state: Optional[np.random.RandomState] = None,
    initial_balance: float = 10000.0,
) -> pd.DataFrame:
    """
    Generate a single simulated ledger by resampling trade returns.

    Parameters
    ----------
    ledger : pd.DataFrame
        Original trade ledger.  Must contain columns:
        ``returns``       — percentage return per trade,
        ``exit_datetime`` — timestamp of trade close,
        ``balance``       — account balance AFTER each trade.
    method : str
        Resampling method (see module docstring).
    random_state : np.random.RandomState, optional
        For reproducibility.  If None, uses the global NumPy RNG.
    initial_balance : float
        Starting balance before any trades.  Pass
        ``config['backtest']['starting_balance']`` — do not derive
        from the ledger.

    Returns
    -------
    pd.DataFrame
        Simulated ledger with the same columns as the input.
    """
    rng = random_state if random_state is not None else np.random

    # Use the correct column names from constants
    # Prefer per-trade account return for MC simulation
    if "account_return_pct" in ledger.columns:
        original_returns = ledger["account_return_pct"].astype(float).values
    elif "trade_return_pct" in ledger.columns:
        original_returns = ledger["trade_return_pct"].astype(float).values
    else:
        raise ValueError("Ledger must contain per-trade return column.")

    original_dates = ledger["exit_datetime"].values
    n_trades         = len(original_returns)

    # ------------------------------------------------------------------
    # Method selection — produce sampled_returns and sim_dates
    # ------------------------------------------------------------------
    if method in ("shuffle", "returns"):
        idx             = rng.permutation(n_trades)
        sampled_returns = original_returns[idx]
        sim_dates       = original_dates[idx]

    elif method == "bootstrap":
        idx             = rng.choice(n_trades, size=n_trades, replace=True)
        sampled_returns = original_returns[idx]
        sim_dates       = original_dates[idx]

    elif method == "hybrid":
        min_window      = max(5, int(0.6 * n_trades))
        window          = rng.randint(min_window, n_trades + 1)
        start           = rng.randint(0, n_trades - window + 1)
        idx             = np.arange(start, start + window)
        sampled_returns = original_returns[idx]
        sim_dates       = original_dates[idx]

    elif method == "moving_block":
        block_len = max(2, min(20, n_trades // 4))
        n_blocks  = int(np.ceil(n_trades / block_len))
        idx_list  = []
        for _ in range(n_blocks):
            start = rng.randint(0, max(1, n_trades - block_len + 1))
            idx_list.extend(range(start, start + block_len))
        idx             = np.array(idx_list[:n_trades])
        sampled_returns = original_returns[idx]
        sim_dates       = original_dates[idx]

    elif method == "stationary":
        # Politis-Romano (1994): geometric block lengths, mean = block_len
        block_len = max(2, min(20, n_trades // 4))
        p         = 1.0 / block_len
        idx_list  = []
        while len(idx_list) < n_trades:
            start = rng.randint(0, n_trades)
            k     = int(rng.geometric(p))
            for i in range(k):
                idx_list.append((start + i) % n_trades)
        idx             = np.array(idx_list[:n_trades])
        sampled_returns = original_returns[idx]
        sim_dates       = original_dates[idx]

    elif method == "parametric":
        df_t, loc, scale = scipy_t.fit(original_returns)
        sampled_returns  = scipy_t.rvs(df_t, loc=loc, scale=scale,
                                        size=n_trades, random_state=rng)
        sim_dates        = original_dates[:n_trades]

    else:
        raise ValueError(f"Unknown Monte Carlo method: {method!r}")

    # ------------------------------------------------------------------
    # Build synthetic equity curve
    # ------------------------------------------------------------------
    growth_factors   = 1.0 + sampled_returns
    cumulative_growth = np.cumprod(growth_factors)
    equity_values    = initial_balance * cumulative_growth
    pnl_values       = np.diff(equity_values, prepend=initial_balance)

    n_sim = len(sampled_returns)

    return pd.DataFrame({
        "entry_datetime":  sim_dates,
        "entry_fee_pct":   np.zeros(n_sim),
        "avg_entry_price": np.zeros(n_sim),
        "exit_datetime":   sim_dates,
        "exit_fee_pct":    np.zeros(n_sim),
        "avg_exit_price":  np.zeros(n_sim),
        "position_size_pct":            np.zeros(n_sim),
        "trade_return_pct": sampled_returns,
        "account_return_pct": sampled_returns,  # MC sim uses per-trade return as account return
        "cum_account_return": np.cumsum(sampled_returns),
        "direction":       ["Long"] * n_sim,
        "status":          ["Closed"] * n_sim,
        "action":          ["close_position"] * n_sim,
        "balance":         equity_values,
    })


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def run_monte_carlo_analysis(
    ledger: pd.DataFrame,
    config: dict,
) -> Dict[str, Any]:
    """
    Run the full Monte Carlo analysis and return comprehensive statistics.

    This is the SINGLE SOURCE OF TRUTH for all Monte Carlo computation.
    It runs the simulation loop, collects return / Sharpe / drawdown
    distributions, and computes percentiles.  It does NOT apply pass/fail
    gates — that is the pipeline layer's responsibility.

    Parameters
    ----------
    ledger : pd.DataFrame
        Trade ledger (e.g. aggregated OOS ledger from walk-forward).
    config : dict
        Full pipeline config.  Reads ``config['monte_carlo']``.

    Returns
    -------
    dict
        Comprehensive statistics dict.  All keys are described below.

    Keys returned
    -------------
    n_simulations, n_trades, method
    original_return_pct, original_sharpe, original_drawdown_pct
    profitable_ratio                — fraction of sims with return > 0
    mean_return_pct, median_return_pct, std_return_pct
    return_percentiles              — dict {p1, p5, p10, p25, p50, p75, p90, p95, p99}
    mean_sharpe, sharpe_percentiles
    mean_drawdown_pct, drawdown_percentiles
    worst_case_percentile           — from config (e.g. 5)
    worst_case_return               — return at worst_case_percentile
    worst_case_sharpe               — sharpe at worst_case_percentile
    worst_case_drawdown             — drawdown at (100 - worst_case_percentile)
    sim_returns, sim_sharpes, sim_drawdowns  — raw arrays (lists)
    """
    from bitpredict.common.stats import calculate_essential_stats

    mc_cfg  = config.get('monte_carlo', {})
    n_sims  = int(mc_cfg.get('n_simulations', 500))
    method  = str(mc_cfg.get('method', 'stationary'))
    seed    = int(mc_cfg.get('random_seed', 42))
    wc_pct  = int(mc_cfg.get('worst_case_percentile', 5))
    initial_balance = float(config.get('backtest', {}).get('starting_balance', 10000))

    pct_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    # Original (observed) performance
    original_stats   = calculate_essential_stats(ledger)
    original_return  = float(original_stats.get('total_return_pct', 0) or 0)
    original_sharpe  = float(original_stats.get('sharpe_ratio', 0) or 0)
    original_dd      = float(original_stats.get('max_drawdown_pct', 0) or 0)

    rng = np.random.RandomState(seed)

    sim_returns:   list[float] = []
    sim_sharpes:   list[float] = []
    sim_drawdowns: list[float] = []

    for _ in range(n_sims):
        try:
            sim_ledger = monte_carlo_simulations(ledger, method=method, random_state=rng,
                                                   initial_balance=initial_balance)
            sim_stats  = calculate_essential_stats(sim_ledger)
            sim_returns.append(float(sim_stats.get('total_return_pct', 0) or 0))
            sim_sharpes.append(float(sim_stats.get('sharpe_ratio', 0) or 0))
            sim_drawdowns.append(float(sim_stats.get('max_drawdown_pct', 0) or 0))
        except Exception:
            pass  # failed sims are silently dropped; caller sees n_simulations < n_sims

    if not sim_returns:
        return {
            'n_simulations': 0, 'n_trades': len(ledger), 'method': method,
            'error': 'All simulations failed',
        }

    arr_ret = np.array(sim_returns, dtype=float)
    arr_sh  = np.array(sim_sharpes, dtype=float)
    arr_dd  = np.array(sim_drawdowns, dtype=float)

    def _pct_dict(arr: np.ndarray) -> Dict[str, float]:
        return {f'p{p}': float(np.percentile(arr, p)) for p in pct_levels}

    return {
        # --- metadata ---------------------------------------------------
        'n_simulations':          len(sim_returns),
        'n_trades':               len(ledger),
        'method':                 method,

        # --- observed performance ---------------------------------------
        'original_return_pct':    original_return,
        'original_sharpe':        original_sharpe,
        'original_drawdown_pct':  original_dd,

        # --- return distribution ----------------------------------------
        'profitable_ratio':       float(np.mean(arr_ret > 0)),
        'mean_return_pct':        float(np.mean(arr_ret)),
        'median_return_pct':      float(np.median(arr_ret)),
        'std_return_pct':         float(np.std(arr_ret)),
        'return_percentiles':     _pct_dict(arr_ret),

        # --- Sharpe distribution ----------------------------------------
        'mean_sharpe':            float(np.mean(arr_sh)),
        'sharpe_percentiles':     _pct_dict(arr_sh),

        # --- drawdown distribution --------------------------------------
        'mean_drawdown_pct':      float(np.mean(arr_dd)),
        'drawdown_percentiles':   _pct_dict(arr_dd),

        # --- worst-case summary (configurable percentile) ---------------
        'worst_case_percentile':  wc_pct,
        'worst_case_return':      float(np.percentile(arr_ret, wc_pct)),
        'worst_case_sharpe':      float(np.percentile(arr_sh, wc_pct)),
        # Worst drawdown = high percentile (larger drawdown = worse)
        'worst_case_drawdown':    float(np.percentile(arr_dd, 100 - wc_pct)),

        # --- raw arrays (for custom gate logic in pipeline) -------------
        'sim_returns':            arr_ret.tolist(),
        'sim_sharpes':            arr_sh.tolist(),
        'sim_drawdowns':          arr_dd.tolist(),
    }
