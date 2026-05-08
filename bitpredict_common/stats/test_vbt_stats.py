import pandas as pd
import numpy as np
import sys
from pathlib import Path

from bitpredict.common.ta.indicators import calculate_indicators
from bitpredict.signals import SignalGenerator
from bitpredict.common.db.services.data import read_ohlcv
from bitpredict.backtest import run_backtest
from bitpredict.common.stats import calculate_essential_stats, calculate_comprehensive_stats
import vectorbtpro as vbt



price_data = read_ohlcv(exchange="bybit", symbol="eth", timeframe="1h")
minute_data = read_ohlcv(exchange="bybit", symbol="eth", timeframe="1m")
# Ensure datetime column is proper datetime (UTC-aware)
price_data["datetime"] = pd.to_datetime(price_data["datetime"], utc=True)
minute_data["datetime"] = pd.to_datetime(minute_data["datetime"], utc=True)
cutoff = pd.Timestamp("2026-01-01", tz="UTC")
price_data = price_data[price_data["datetime"] >= cutoff].reset_index(drop=True)
minute_data = minute_data[minute_data["datetime"] >= cutoff].reset_index(drop=True)

print("Generating indicators....")
indicator_df, metadata = calculate_indicators(
    data=price_data,
    indicators=["RSI"],
    # library='vectorbtpro'
)

print("Generating signals....")
signal_gen = SignalGenerator()
signal_params_batch = {
    "RSI": {
        "upper_threshold": 60,
        "lower_threshold": 40,
        "exit_threshold": 50,
        "use_crossover": True
    },
}

signals, info = signal_gen.generate_signals(
    indicators_data=indicator_df, 
    signal_params_batch=signal_params_batch
)

backtest_params = {
    "starting_balance": 10000,
    "take_profit": 1.5,          # still used
    "stop_loss": 0.5,            # ignored when trailing is enabled
    "transaction_fee": 0.0,
    "leverage": 1.0,
    "position_size": 1,
    "slippage": 0.0,
    "create_ledger": True,
    'risk_mgmt':{
        # 'trailing_stop': {'enabled': True, 'activation_pct': 0.5, 'trail_pct': 0.3},
        # 'atr_stop': {'enabled': True, 'period': 14, 'multiplier': 2.0},
        # 'time_stop': {'enabled': True, 'max_duration': '2h'} # minutes - xm, hours - xh, days - xd, weeks - xw, months - xM

    },
    "zero_signal_mode": "hold_position",
}

signals = signals.reset_index()
signals = signals[['datetime', 'rsi_signals']]

print(f"signals (head):{signals.head(5)}")
print(f"minute_data (head):{minute_data.head(5)}")

pf, ledger = run_backtest(
    df_signals=signals, 
    df_ohlcv=minute_data, 
    config=backtest_params
)

print("\n" + "="*80)
print("LEDGER PREVIEW:")
print("="*80)
print(ledger.head())
ledger.to_csv("ledger.csv")
# exit()

minute_data_indexed = minute_data.set_index('datetime')
benchmark_returns = vbt.Portfolio.from_holding(
    close=minute_data_indexed['close'],  
    init_cash=backtest_params['starting_balance']
).value



comprehensive_stats_vbt = calculate_comprehensive_stats(
    data=pf,
    ledger_input=ledger,
    bar_type='time', 
    benchmark_returns= benchmark_returns,
    # calculate_monte_carlo = True
)

comprehensive_stats_cusotm = calculate_comprehensive_stats(
    data=ledger)

# --- quick sanity checks ---------------------------------------------------
from bitpredict.common.stats.vectorbtpro.vbt_stats import (
    _build_vbt_cache,
    _extract_risk_adjusted_stats_vectorized
)

vbt_cache_test = _build_vbt_cache(pf, 'time', benchmark_returns)
ra_extra_check = _extract_risk_adjusted_stats_vectorized(pf, vbt_cache_test, ledger)
ra_group = comprehensive_stats_vbt['risk_adjusted']
for key, val in ra_extra_check.items():
    assert np.isclose(ra_group.get(key, 0.0), val, atol=1e-9), (
        f"Expected {key} {val} but got {ra_group.get(key)}"
    )

# verify vbt/custom ratio modules agree on simple data
from bitpredict.common.stats.vectorbtpro.ratios import _calculate_risk_adjusted_ratios as _vbt_ratios
from bitpredict.common.stats.custom.ratios import _calculate_risk_adjusted_ratios as _custom_ratios
simple_returns = np.array([0.01, -0.02, 0.015, 0.0])
timestamps = np.arange(len(simple_returns))
ann = 365.25
vbt_res = _vbt_ratios(simple_returns, timestamps, 0.0, 0.0, ann)
custom_res = _custom_ratios(simple_returns, timestamps, 0.0, 0.0, ann)
for key in ['calmar_ratio', 'sharpe_ratio', 'sortino_ratio', 'omega_ratio']:
    assert np.isclose(vbt_res[key], custom_res[key], atol=1e-9), (
        f"{key} mismatch: vbt {vbt_res[key]} vs custom {custom_res[key]}"
    )

print("\n" + "="*80)
print(comprehensive_stats_vbt.keys())
print("\n" + "="*80)
print(comprehensive_stats_cusotm.keys())
print("\n============VBT" + "="*80)
print(comprehensive_stats_vbt['exposure'])
print("\n" + "="*80)
print("\n============custom" + "="*80)
print(comprehensive_stats_cusotm['exposure'])
print("\n" + "="*80)
# print(comprehensive_stats_vbt['profit_loss'])

# vbt_stats = comprehensive_stats_vbt

# single_values_vbt = []
# dict_values_vbt = []

# for k, v in vbt_stats.items():
#     if isinstance(v, dict):
#         dict_values_vbt.append(k)
#     else:
#         single_values_vbt.append(k)
# print("="*80)
# print("----------VBT----------------------------------")
# print("Single value keys:", single_values_vbt)
# print("="*80)
# print("Dict keys:", dict_values_vbt)
# print("="*80)

# custom_stats = comprehensive_stats_cusotm

# single_values_custom = []
# dict_values_custom = []

# # First pass: separate top-level keys
# for k, v in custom_stats.items():
#     if isinstance(v, dict):
#         dict_values_custom.append(k)
#     else:
#         single_values_custom.append(k)

# # Second pass: explore nested dicts and add their keys
# nested_dict_keys = []
# for top_key in dict_values_custom:
#     nested_dict = custom_stats[top_key]
#     for k in nested_dict.keys():
#         nested_dict_keys.append(f"{top_key}.------.{k}")  # Use dotted notation to preserve hierarchy

# print("="*80)
# print("----------CUSTOM----------------------------------")
# print("Single value keys:", single_values_custom)
# print("="*80)
# print("Top-level dict keys:", dict_values_custom)
# print("="*80)
# print("Nested dict keys:", nested_dict_keys)
# print("="*80)




# print("-|=X=|-"*30)
# # Flattened keys from your earlier results
# vbt_keys = ['start_date', 'end_date', 'total_duration_days', 'initial_value', 'min_value', 'max_value', 'final_value', 'total_return_pct', 'benchmark_return_pct', 'position_coverage_pct', 'max_gross_exposure_pct', 'max_drawdown_pct', 'max_drawdown_duration_days', 'total_fees_paid', 'total_trades', 'win_rate_pct', 'best_trade_pct', 'worst_trade_pct', 'avg_winning_trade_pct', 'avg_losing_trade_pct', 'avg_winning_trade_duration_days', 'avg_losing_trade_duration_days', 'profit_factor', 'expectancy', 'sharpe_ratio', 'calmar_ratio', 'omega_ratio', 'sortino_ratio', 'portfolio_value_current', 'portfolio_value_initial', 'portfolio_value_min', 'portfolio_value_max', 'portfolio_value_mean', 'portfolio_value_volatility', 'cash_balance_current', 'cash_balance_initial', 'period_return_mean', 'period_return_volatility', 'period_return_min', 'period_return_max', 'period_return_skewness', 'period_return_kurtosis', 'cumulative_return_final', 'daily_return_mean', 'daily_return_volatility', 'current_drawdown', 'avg_drawdown', 'drawdown_volatility', 'min_drawdown', 'drawdown_periods', 'mfe_pct', 'mae_pct', 'pnl_distribution', 'directional_pnl', 'directional_metrics', 'rolling_sharpe', 'rolling_sortino', 'benchmark_returns', 'rolling_correlation', 'heatmaps_data', 'gross_exposure_current', 'gross_exposure_max', 'gross_exposure_avg', 'net_exposure_current', 'net_exposure_max', 'net_exposure_avg', 'total_cash_flow', 'avg_cash_flow', 'cash_flow_volatility', 'positive_cash_flow', 'negative_cash_flow', 'total_wins', 'total_losses', 'losses_percentage', 'average_trade_return', 'consecutive_wins', 'consecutive_losses', 'gain_to_pain_ratio', 'r2', 'pnl_1_days', 'pnl_7_days', 'pnl_15_days', 'pnl_30_days', 'pnl_45_days', 'pnl_60_days']

# custom_keys = nested_dict_keys  # already flattened

# # Convert to sets
# vbt_set = set(vbt_keys)
# custom_set = set(custom_keys)

# # Compute differences and intersection
# custom_only = sorted(list(custom_set - vbt_set))
# vbt_only = sorted(list(vbt_set - custom_set))
# common_stats = sorted(list(vbt_set & custom_set))

# print("Stats in custom but not in VBT:", custom_only)
# print("="*80)
# print("Stats in VBT but not in custom:", vbt_only)
# print("="*80)
# print("Common stats:", common_stats)