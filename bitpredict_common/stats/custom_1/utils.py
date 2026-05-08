"""
Pure NumPy utility functions for custom_1 stats.
No pandas used after the initial array preparation step.
"""

import numpy as np
import pandas as pd
from collections import namedtuple
from typing import Tuple
import time
NS_PER_DAY = np.int64(86_400_000_000_000)

LEDGER_NUMERIC_COLS = [
    "avg_entry_price", "avg_exit_price",
    "position_size_pct", "entry_fee_pct", "exit_fee_pct",
    "account_return_pct", "trade_return_pct", "balance",
]
COL_ENTRY_PX  = 0
COL_EXIT_PX   = 1
COL_POS_SIZE  = 2
COL_ENTRY_FEE = 3
COL_EXIT_FEE  = 4
COL_ACC_RET   = 5
COL_TRADE_RET = 6
COL_BALANCE   = 7

BarArrays       = namedtuple("BarArrays",       ["ts", "open", "high", "low", "close", "volume"])
DailyIndex      = namedtuple("DailyIndex",      ["day_ns", "unique_days", "first_idx", "last_idx", "bar_day_idx"])
LedgerArrays    = namedtuple("LedgerArrays",    ["numeric_3d", "datetime_3d", "sign_2d", "lengths", "names"])
DailyReturns    = namedtuple("DailyReturns",    ["daily_returns", "daily_balances", "benchmark_returns", "day_timestamps_ns"])
BatchedReturns  = namedtuple("BatchedReturns",  ["daily_returns_2d", "daily_balances_2d", "benchmark_returns_1d", "valid_mask_2d", "day_timestamps_ns", "n_days_per_strat"])


def prepare_bar_arrays(df_1m: pd.DataFrame) -> BarArrays:
    """Convert 1-minute bars to BarArrays namedtuple. Called once, shared across strategies."""
    m = len(df_1m)
    return BarArrays(
        ts = df_1m["datetime"].values.view('int64'),  # Assumes datetime64[ns] column
        open   = df_1m["open"].to_numpy(dtype=np.float64)   if "open"   in df_1m.columns else np.zeros(m),
        high   = df_1m["high"].to_numpy(dtype=np.float64)   if "high"   in df_1m.columns else np.zeros(m),
        low    = df_1m["low"].to_numpy(dtype=np.float64)    if "low"    in df_1m.columns else np.zeros(m),
        close  = df_1m["close"].to_numpy(dtype=np.float64),
        volume = df_1m["volume"].to_numpy(dtype=np.float64) if "volume" in df_1m.columns else np.zeros(m),
    )

def prepare_daily_index(bars: BarArrays) -> DailyIndex:
    """Precompute daily resampling indices from bar timestamps. Called once, shared across strategies."""
    ts      = bars.ts
    n       = len(ts)
    day_int = (ts // NS_PER_DAY)  # integer day number per bar

    # Detect day boundaries (vectorized, no Python list)
    day_changes        = np.empty(n, dtype=np.bool_)
    day_changes[0]     = True
    day_changes[1:]    = day_int[1:] != day_int[:-1]

    first_idx   = np.where(day_changes)[0]
    unique_days = day_int[first_idx]

    last_idx        = np.empty(len(first_idx), dtype=np.int64)
    last_idx[:-1]   = first_idx[1:] - 1
    last_idx[-1]    = n - 1

    # Per-bar day index: cumsum of day_changes gives 0-based day index per bar
    bar_day_idx = np.cumsum(day_changes) - 1

    return DailyIndex(
        day_ns      = unique_days * NS_PER_DAY,
        unique_days = unique_days,
        first_idx   = first_idx,
        last_idx    = last_idx,
        bar_day_idx = bar_day_idx.astype(np.int32),
    )


def prepare_ledger_arrays(ledgers: dict) -> LedgerArrays:
    names = list(ledgers.keys())
    dfs   = list(ledgers.values())
    n_strats = len(dfs)
 
    lengths    = np.fromiter((len(df) for df in dfs), dtype=np.int64)
    max_trades = int(lengths.max())
    n_cols     = len(LEDGER_NUMERIC_COLS)
 
    numeric_3d  = np.full((n_strats, max_trades, n_cols), np.nan, dtype=np.float64)
    datetime_3d = np.zeros((n_strats, max_trades, 2), dtype=np.int64)
    sign_2d     = np.zeros((n_strats, max_trades), dtype=np.float64)
 
    # Pre-resolve column positions once — avoids repeated __getitem__ on df
    for i, (df, n) in enumerate(zip(dfs, lengths)):
        n = int(n)
 
        # --- numeric: same as before, already optimal
        numeric_3d[i, :n] = df[LEDGER_NUMERIC_COLS].to_numpy(dtype=np.float64, copy=False)
 
        # --- datetime: use __array__ directly, skip .to_numpy() overhead
        datetime_3d[i, :n, 0] = df["entry_datetime"].values.view("int64")
        datetime_3d[i, :n, 1] = df["exit_datetime"].values.view("int64")
 
        # --- direction: avoid re-allocating np.where output when all same sign
        if "direction" in df.columns:
            d = df["direction"].values          # raw numpy array, no copy
            # Use view trick: 'Long'[0] == 'L' — single-char compare is faster
            # than full string compare on large arrays
            first_chars = d.astype("U1")        # truncate to first char only
            sign_2d[i, :n] = np.where(first_chars == "L", 1.0, -1.0)
        else:
            sign_2d[i, :n] = 1.0
 
    return LedgerArrays(
        numeric_3d  = numeric_3d,
        datetime_3d = datetime_3d,
        sign_2d     = sign_2d,
        lengths     = lengths,
        names       = names,
    )

# first optimized

# def build_returns_all(stacked: LedgerArrays, bars: BarArrays, di: DailyIndex) -> BatchedReturns:
#     """
#     Batched build_returns for ALL strategies at once.
#     Returns TRANSPOSED arrays: (max_days, n_strats) for full vectorization.
    
#     Each strategy uses its own trade date range (first trade to last trade).
#     No artificial idle days are prepended.
    
#     Layout optimized for thousands of strategies:
#     - Shape: (max_days, n_strats) - each column is one strategy's time series
#     - Enables vectorized operations across all strategies simultaneously
#     - Includes validity mask for variable-length strategies
    
#     Optimizations:
#     - Benchmark computed once (shared across all strategies)
#     - searchsorted called once for all strategies (vectorized)
#     - Per-strategy loop only for operations that require different indices
#     """
#     n_strats   = len(stacked.names)
#     n_bars     = len(bars.ts)
#     n_days_raw = len(di.unique_days)
#     max_trades = stacked.numeric_3d.shape[1]
    
#     bar_ts    = bars.ts
#     close_arr = bars.close
    
#     # Pre-allocate output arrays - TRANSPOSED: (max_days, n_strats)
#     max_days = n_days_raw
#     daily_returns_2d   = np.zeros((max_days, n_strats), dtype=np.float64)
#     daily_balances_2d  = np.zeros((max_days, n_strats), dtype=np.float64)
#     valid_mask_2d      = np.zeros((max_days, n_strats), dtype=bool)
#     n_days_per_strat   = np.zeros(n_strats, dtype=np.int32)
    
#     # ═══════════════════════════════════════════════════════════════════════
#     # SHARED COMPUTATION (once for all strategies)
#     # ═══════════════════════════════════════════════════════════════════════
    
#     # Benchmark is shared (same for all strategies) - 1D array
#     daily_close = close_arr[di.last_idx]
#     benchmark_returns_1d = np.empty(n_days_raw, dtype=np.float64)
#     benchmark_returns_1d[0] = 0.0
#     benchmark_returns_1d[1:] = daily_close[1:] / daily_close[:-1] - 1.0
    
#     # Vectorized searchsorted for ALL strategies at once
#     # Shape: (n_strats, max_trades)
#     entry_idx_all = np.searchsorted(bar_ts, stacked.datetime_3d[:, :, 0], side="left").clip(0, n_bars - 1)
#     exit_idx_all  = np.searchsorted(bar_ts, stacked.datetime_3d[:, :, 1], side="left").clip(0, n_bars - 1)
    
#     # ═══════════════════════════════════════════════════════════════════════
#     # PER-STRATEGY LOOP (only for operations requiring different indices)
#     # ═══════════════════════════════════════════════════════════════════════
    
#     for s in range(n_strats):
#         n = int(stacked.lengths[s])
#         if n == 0:
#             continue
        
#         # Extract slices (already vectorized via numpy indexing)
#         entry_px  = stacked.numeric_3d[s, :n, COL_ENTRY_PX]
#         exit_px   = stacked.numeric_3d[s, :n, COL_EXIT_PX]
#         sign      = stacked.sign_2d[s, :n]
#         bal_vals  = stacked.numeric_3d[s, :n, COL_BALANCE]
#         acc_ret   = stacked.numeric_3d[s, :n, COL_ACC_RET]   / 100.0
#         pos_size  = stacked.numeric_3d[s, :n, COL_POS_SIZE]  / 100.0
#         entry_fee = stacked.numeric_3d[s, :n, COL_ENTRY_FEE] / 100.0
#         exit_fee  = stacked.numeric_3d[s, :n, COL_EXIT_FEE]  / 100.0
        
#         # Use pre-computed indices
#         entry_idx = entry_idx_all[s, :n]
#         exit_idx  = exit_idx_all[s, :n]
        
#         # Initial balance calculation
#         denom = 1.0 + acc_ret[0]
#         initial_balance = bal_vals[0] / denom if denom != 0.0 else bal_vals[0]
        
#         # Position values and fees (vectorized)
#         prev_bal   = np.where((1.0 + acc_ret) != 0.0, bal_vals / (1.0 + acc_ret), bal_vals)
#         pos_val    = prev_bal * pos_size
#         entry_fees = pos_val * entry_fee
#         exit_fees  = pos_val * exit_fee
        
#         # Cash changes (vectorized with np.add.at)
#         cash_changes = np.zeros(n_bars, dtype=np.float64)
#         cash_changes[0] = initial_balance
#         np.add.at(cash_changes, entry_idx, -entry_fees)
#         pnl = pos_val * sign * ((exit_px / entry_px) - 1.0)
#         np.add.at(cash_changes, exit_idx, pnl - exit_fees)
#         cash = np.cumsum(cash_changes)
        
#         # Unrealized P&L (vectorized using cumsum trick)
#         coef  = pos_val * sign / entry_px
#         const = -pos_val * sign

#         diff_coef  = np.zeros(n_bars + 1, dtype=np.float64)
#         diff_const = np.zeros(n_bars + 1, dtype=np.float64)

#         np.add.at(diff_coef, entry_idx, coef)
#         np.add.at(diff_coef, exit_idx, -coef)

#         np.add.at(diff_const, entry_idx, const)
#         np.add.at(diff_const, exit_idx, -const)

#         cum_coef  = np.cumsum(diff_coef[:-1])
#         cum_const = np.cumsum(diff_const[:-1])

#         unrealized = cum_coef * close_arr + cum_const
        
#         # Portfolio values
#         portfolio_values = cash + unrealized
        
#         # Minute returns (vectorized)
#         minute_returns = np.empty(n_bars, dtype=np.float64)
#         minute_returns[0] = 0.0
#         prev = portfolio_values[:-1]
#         minute_returns[1:] = np.where(prev != 0.0, (portfolio_values[1:] - prev) / prev, 0.0)
        
#         # Daily resampling (shared bar_day_idx)
#         log_returns = np.log1p(minute_returns)
#         daily_log = np.bincount(di.bar_day_idx, weights=log_returns, minlength=n_days_raw)
#         daily_ret = np.expm1(daily_log)
#         daily_bal = portfolio_values[di.last_idx]
        
#         # Store results in TRANSPOSED layout: column s = strategy s
#         n_days = len(daily_ret)
#         daily_returns_2d[:n_days, s] = daily_ret
#         daily_balances_2d[:n_days, s] = daily_bal
#         valid_mask_2d[:n_days, s] = True
#         n_days_per_strat[s] = n_days
    
#     return BatchedReturns(
#         daily_returns_2d     = daily_returns_2d,      # (max_days, n_strats)
#         daily_balances_2d    = daily_balances_2d,     # (max_days, n_strats)
#         benchmark_returns_1d = benchmark_returns_1d,  # (max_days,)
#         valid_mask_2d        = valid_mask_2d,         # (max_days, n_strats)
#         day_timestamps_ns    = di.unique_days * NS_PER_DAY,
#         n_days_per_strat     = n_days_per_strat,
#     )


def build_returns_all_vectorized(
    stacked: LedgerArrays, bars: BarArrays, di: DailyIndex
) -> BatchedReturns:

    n_strategies = len(stacked.names)
    n_bars = len(bars.ts)
    n_days = len(di.unique_days)

    dtype = np.float32

    # ====================== ALLOCATIONS ======================
    cash_flow = np.zeros((n_strategies, n_bars), dtype=dtype)

    coef_diff = np.zeros((n_strategies, n_bars + 1), dtype=dtype)
    const_diff = np.zeros((n_strategies, n_bars + 1), dtype=dtype)

    daily_returns = np.zeros((n_days, n_strategies), dtype=dtype)
    daily_balances = np.zeros((n_days, n_strategies), dtype=dtype)
    valid_mask = np.zeros((n_days, n_strategies), dtype=bool)
    n_days_per_strategy = np.zeros(n_strategies, dtype=np.int32)

    close_prices = bars.close.astype(dtype)
    bar_to_day_index = di.bar_day_idx

    # ====================== BENCHMARK ======================
    daily_close = bars.close[di.last_idx]
    benchmark_returns = np.empty(n_days, dtype=np.float64)
    benchmark_returns[0] = 0.0
    benchmark_returns[1:] = daily_close[1:] / daily_close[:-1] - 1.0

    # Map timestamps → bar indices
    entry_idx_all = np.searchsorted(
        bars.ts, stacked.datetime_3d[:, :, 0], side="left"
    ).clip(0, n_bars - 1)

    exit_idx_all = np.searchsorted(
        bars.ts, stacked.datetime_3d[:, :, 1], side="left"
    ).clip(0, n_bars - 1)

    # ====================== PREP ======================
    trade_counts = stacked.lengths.astype(np.int32)
    cumulative_counts = np.cumsum(trade_counts)

    strategy_ids = np.repeat(np.arange(n_strategies), trade_counts)

    trade_indices = (
        np.concatenate([np.arange(n) for n in trade_counts])
        if len(trade_counts) > 0
        else np.array([], dtype=np.int32)
    )

    flat_entry_idx = entry_idx_all[strategy_ids, trade_indices]
    flat_exit_idx = exit_idx_all[strategy_ids, trade_indices]

    # Flattened trade data
    entry_price = stacked.numeric_3d[strategy_ids, trade_indices, COL_ENTRY_PX].astype(dtype)
    exit_price = stacked.numeric_3d[strategy_ids, trade_indices, COL_EXIT_PX].astype(dtype)
    position_sign = stacked.sign_2d[strategy_ids, trade_indices].astype(dtype)

    balance_values = stacked.numeric_3d[strategy_ids, trade_indices, COL_BALANCE].astype(dtype)
    accumulated_return = (
        stacked.numeric_3d[strategy_ids, trade_indices, COL_ACC_RET].astype(dtype) * 0.01
    )

    position_size = (
        stacked.numeric_3d[strategy_ids, trade_indices, COL_POS_SIZE].astype(dtype) * 0.01
    )

    entry_fee = (
        stacked.numeric_3d[strategy_ids, trade_indices, COL_ENTRY_FEE].astype(dtype) * 0.01
    )
    exit_fee = (
        stacked.numeric_3d[strategy_ids, trade_indices, COL_EXIT_FEE].astype(dtype) * 0.01
    )

    # ====================== SCATTER ======================

    # Identify first trade per strategy (for initial cash)
    first_trade_mask = np.zeros(len(strategy_ids), dtype=bool)
    if len(first_trade_mask) > 0:
        first_trade_mask[0] = True
        if len(cumulative_counts) > 0:
            first_trade_mask[cumulative_counts[:-1]] = True

    if np.any(first_trade_mask):
        denom = 1.0 + accumulated_return[first_trade_mask]
        strat_idx = strategy_ids[first_trade_mask]

        cash_flow[strat_idx, 0] = np.where(
            denom != 0,
            balance_values[first_trade_mask] / denom,
            balance_values[first_trade_mask],
        )

    # Position value
    previous_balance = np.where(
        accumulated_return != -1.0,
        balance_values / (1.0 + accumulated_return),
        balance_values,
    )

    position_value = previous_balance * position_size

    # Entry fees
    np.add.at(cash_flow, (strategy_ids, flat_entry_idx), -position_value * entry_fee)

    # Exit PnL + fees
    pnl = position_value * position_sign * ((exit_price / entry_price) - 1.0)
    np.add.at(cash_flow, (strategy_ids, flat_exit_idx), pnl - position_value * exit_fee)

    # Linear exposure model
    coef = position_value * position_sign / entry_price
    const = -position_value * position_sign

    np.add.at(coef_diff, (strategy_ids, flat_entry_idx), coef)
    np.add.at(coef_diff, (strategy_ids, flat_exit_idx), -coef)

    np.add.at(const_diff, (strategy_ids, flat_entry_idx), const)
    np.add.at(const_diff, (strategy_ids, flat_exit_idx), -const)

    # ====================== PORTFOLIO ======================
    cash_cumsum = np.empty_like(cash_flow)
    np.cumsum(cash_flow, axis=1, out=cash_cumsum)

    coef_cumsum = np.empty((n_strategies, n_bars), dtype=dtype)
    np.cumsum(coef_diff[:, :-1], axis=1, out=coef_cumsum)

    const_cumsum = np.empty((n_strategies, n_bars), dtype=dtype)
    np.cumsum(const_diff[:, :-1], axis=1, out=const_cumsum)

    portfolio_values = np.empty_like(cash_cumsum)
    np.multiply(coef_cumsum, close_prices, out=portfolio_values)
    np.add(portfolio_values, const_cumsum, out=portfolio_values)
    np.add(portfolio_values, cash_cumsum, out=portfolio_values)

    # ====================== RETURNS ======================
    shifted_portfolio = portfolio_values[:, :-1]

    minute_returns = np.zeros((n_strategies, n_bars), dtype=dtype)

    diff = portfolio_values[:, 1:] - shifted_portfolio
    np.divide(
        diff,
        shifted_portfolio,
        out=minute_returns[:, 1:],
        where=(shifted_portfolio != 0),
    )

    log_returns = np.empty_like(minute_returns)
    np.log1p(minute_returns, out=log_returns)

    # ====================== DAILY ======================
    daily_log = np.zeros((n_strategies, n_days), dtype=dtype)

    strat_idx = np.arange(n_strategies)
    np.add.at(
        daily_log,
        (strat_idx[:, None], bar_to_day_index[None, :]),
        log_returns,
    )

    daily_return_matrix = np.expm1(daily_log)

    # Fill outputs
    daily_returns[:] = daily_return_matrix.T
    daily_balances[:] = portfolio_values[:, di.last_idx].T

    valid_mask[:] = True
    n_days_per_strategy[:] = n_days

    # ====================== OUTPUT ======================
    return BatchedReturns(
        daily_returns_2d=daily_returns,
        daily_balances_2d=daily_balances,
        benchmark_returns_1d=benchmark_returns,
        valid_mask_2d=valid_mask,
        day_timestamps_ns=di.unique_days * NS_PER_DAY,
        n_days_per_strat=n_days_per_strategy,
    )
    
#2.2 sec
def build_returns_all(stacked: LedgerArrays, bars: BarArrays, di: DailyIndex) -> BatchedReturns:
    n_strategies = len(stacked.names)
    n_bars = len(bars.ts)
    n_days = len(di.unique_days)

    dtype = np.float32

    # ====================== ALLOCATIONS ======================
    cash_flow = np.zeros((n_strategies, n_bars), dtype=dtype)

    coef_diff = np.zeros((n_strategies, n_bars + 1), dtype=dtype)
    const_diff = np.zeros((n_strategies, n_bars + 1), dtype=dtype)

    daily_returns = np.zeros((n_days, n_strategies), dtype=dtype)
    daily_balances = np.zeros((n_days, n_strategies), dtype=dtype)
    valid_mask = np.zeros((n_days, n_strategies), dtype=bool)
    n_days_per_strategy = np.zeros(n_strategies, dtype=np.int32)

    close_prices = bars.close.astype(dtype)
    bar_to_day_index = di.bar_day_idx

    # ====================== BENCHMARK ======================
    daily_close = bars.close[di.last_idx]
    benchmark_returns = np.empty(n_days, dtype=np.float64)
    benchmark_returns[0] = 0.0
    benchmark_returns[1:] = daily_close[1:] / daily_close[:-1] - 1.0

    # Map trade timestamps → bar indices
    entry_indices_all = np.searchsorted(
        bars.ts, stacked.datetime_3d[:, :, 0], side="left"
    ).clip(0, n_bars - 1)

    exit_indices_all = np.searchsorted(
        bars.ts, stacked.datetime_3d[:, :, 1], side="left"
    ).clip(0, n_bars - 1)

    # ====================== PREP ======================
    trade_counts = stacked.lengths.astype(np.int32)
    cumulative_counts = np.cumsum(trade_counts)

    strategy_ids = np.repeat(np.arange(n_strategies), trade_counts)
    trade_indices = (
        np.concatenate([np.arange(n) for n in trade_counts])
        if len(trade_counts) > 0
        else np.array([], dtype=np.int32)
    )

    flat_entry_idx = entry_indices_all[strategy_ids, trade_indices]
    flat_exit_idx = exit_indices_all[strategy_ids, trade_indices]

    # ====================== SCATTER TRADES ======================
    for s in range(n_strategies):
        n_trades = trade_counts[s]
        if n_trades == 0:
            continue

        start = 0 if s == 0 else cumulative_counts[s - 1]
        end = cumulative_counts[s]

        entry_idx = flat_entry_idx[start:end]
        exit_idx = flat_exit_idx[start:end]

        # Trade data
        entry_price = stacked.numeric_3d[s, :n_trades, COL_ENTRY_PX].astype(dtype)
        exit_price = stacked.numeric_3d[s, :n_trades, COL_EXIT_PX].astype(dtype)
        position_sign = stacked.sign_2d[s, :n_trades].astype(dtype)

        balance_values = stacked.numeric_3d[s, :n_trades, COL_BALANCE].astype(dtype)
        accumulated_return = (
            stacked.numeric_3d[s, :n_trades, COL_ACC_RET].astype(dtype) * 0.01
        )

        position_size = (
            stacked.numeric_3d[s, :n_trades, COL_POS_SIZE].astype(dtype) * 0.01
        )

        entry_fee = (
            stacked.numeric_3d[s, :n_trades, COL_ENTRY_FEE].astype(dtype) * 0.01
        )
        exit_fee = (
            stacked.numeric_3d[s, :n_trades, COL_EXIT_FEE].astype(dtype) * 0.01
        )

        # ====================== INITIAL CASH ======================
        initial_denominator = 1.0 + accumulated_return[0]
        cash_flow[s, 0] = (
            balance_values[0] / initial_denominator
            if initial_denominator != 0
            else balance_values[0]
        )

        # ====================== POSITION VALUE ======================
        previous_balance = np.where(
            accumulated_return != -1.0,
            balance_values / (1.0 + accumulated_return),
            balance_values,
        )

        position_value = previous_balance * position_size

        # ====================== CASH FLOWS ======================
        # Entry fees
        np.add.at(cash_flow[s], entry_idx, -position_value * entry_fee)

        # PnL + exit fees
        pnl = position_value * position_sign * ((exit_price / entry_price) - 1.0)
        np.add.at(cash_flow[s], exit_idx, pnl - position_value * exit_fee)

        # ====================== LINEAR EXPOSURE MODEL ======================
        coef = position_value * position_sign / entry_price
        const = -position_value * position_sign

        coef_diff[s, :] = 0
        const_diff[s, :] = 0

        np.add.at(coef_diff[s], entry_idx, coef)
        np.add.at(coef_diff[s], exit_idx, -coef)

        np.add.at(const_diff[s], entry_idx, const)
        np.add.at(const_diff[s], exit_idx, -const)

    # ====================== PORTFOLIO CONSTRUCTION ======================
    cash_cumsum = np.cumsum(cash_flow, axis=1)
    coef_cumsum = np.cumsum(coef_diff[:, :-1], axis=1)
    const_cumsum = np.cumsum(const_diff[:, :-1], axis=1)

    portfolio_values = cash_cumsum + coef_cumsum * close_prices + const_cumsum

    # ====================== RETURNS ======================
    shifted_portfolio = portfolio_values[:, :-1]

    minute_returns = np.zeros((n_strategies, n_bars), dtype=dtype)

    valid_mask_nonzero = shifted_portfolio != 0
    diff = portfolio_values[:, 1:] - shifted_portfolio

    np.divide(
        diff,
        shifted_portfolio,
        out=minute_returns[:, 1:],
        where=valid_mask_nonzero,
    )

    log_returns = np.log1p(minute_returns)

    # ====================== DAILY AGGREGATION ======================
    for s in range(n_strategies):
        daily_log_sum = np.bincount(
            bar_to_day_index,
            weights=log_returns[s],
            minlength=n_days,
        )

        strategy_daily_returns = np.expm1(daily_log_sum)
        strategy_daily_balance = portfolio_values[s, di.last_idx]

        nd = len(strategy_daily_returns)

        daily_returns[:nd, s] = strategy_daily_returns
        daily_balances[:nd, s] = strategy_daily_balance
        valid_mask[:nd, s] = True
        n_days_per_strategy[s] = nd

    # ====================== OUTPUT ======================
    return BatchedReturns(
        daily_returns_2d=daily_returns,
        daily_balances_2d=daily_balances,
        benchmark_returns_1d=benchmark_returns,
        valid_mask_2d=valid_mask,
        day_timestamps_ns=di.unique_days * NS_PER_DAY,
        n_days_per_strat=n_days_per_strategy,
    )


 