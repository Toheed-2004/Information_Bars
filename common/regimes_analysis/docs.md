# Regimes Analysis

Analyses how market regimes affect trade outcomes for a given strategy. Takes a closed-trades ledger and an OHLCV bar DataFrame (with pre-computed regime columns), and returns a fully serialisable dict ready for database storage.

## Entry Point

```python
from bitpredict.common.regimes_analysis.runner import run_regimes_analysis

result = run_regimes_analysis(df_ohlcv, df_ledger)
```

`df_ohlcv` must have a `datetime` column plus regime/score/metric columns (auto-detected by prefix or exact name — see `config.py`). `df_ledger` is the closed-trades ledger. Open trades are filtered out automatically.

## How It Works

**Bar matching** (`map_regimes_to_trades`): for each trade's `entry_datetime` and `exit_datetime`, the last completed bar (`bar.datetime <= trade.datetime`) is found via `searchsorted`. All regime/score columns from that bar are copied onto the trade with `entry_` and `exit_` prefixes. Works for any bar type (time-based or non-time-based).

## Output Modules

| Key | Module | What it answers |
|-----|--------|-----------------|
| `performance_per_label` | `performance_by_regime_label` | Win rate, avg return, Sharpe, Sortino, Calmar, max drawdown, max consecutive losses — grouped by regime label, trend, volatility, momentum, and direction×trend |
| `transition_matrix` | `regime_transition` | When a trade enters regime A and exits in regime B, what are the avg return and win rate? |
| `exit_type_breakdown` | `exit_regime_breakdown` | SL/TP/other rates per regime; specific action breakdown (e.g. `SL - ATR`) per regime |
| `quartile_performance` | `continuous_metric_quartile` | For each continuous metric (trend strength, vol percentile, etc.), does performance improve across quartiles Q1→Q4? |
| `score_thresholds` | `score_threshold_analysis` | Per score column: which decile threshold maximises avg return? Is higher score better? |
| `confidence_analysis` | `regime_confidence_analysis` | Does higher regime confidence (Low / Medium / High) improve trade outcomes? Broken down by regime trend and label. |
| `transition_pressure` | `transition_pressure_analysis` | Do high-pressure regime transitions (near a change) produce better/worse entries? SL/TP rates per pressure quartile. |
| `volatility_asymmetry` | `volatility_asymmetry_analysis` | How does up_vol / down_vol skew affect outcomes? Adverse vol risk for Short (squeeze) and Long (whipsaw) trades. |
| `directional_persistence` | `directional_persistence_analysis` | Do high-persistence trend entries (Q4) outperform low-persistence ones (Q1)? Split by direction and combined. |
| `rolling_performance` | `rolling_regime_performance` | Latest 20-trade rolling avg return and win rate per regime. Status: `healthy / deteriorating / poor_win_rate / critical`. |
| `trade_duration` | `trade_duration_analysis` | Avg hold time per regime; duration bucket distribution; performance by bucket (does holding longer help?); hold time by exit action, trend, and volatility. |
| `regime_fitness` | `regime_fitness_score` | Composite score [0–1] per regime combining win rate, profit factor, avg return, and trade count. Penalised by max drawdown. Includes `reliable` flag (≥30 trades). |

## Configuration

All column names, thresholds, weights, and prefixes are in `config.py`. Nothing is hardcoded in the modules.

Key constants:
- `LEDGER_PNL_COL` — return column name in the ledger (`trade_return_pct`)
- `ENTRY_PREFIX` / `EXIT_PREFIX` — `entry_` / `exit_`
- `EXIT_ACTION_SL_PREFIX` / `EXIT_ACTION_TP_PREFIX` — `SL` / `TP` (prefix-based, generic)
- `FITNESS_MIN_TRADES_RELIABLE` — minimum trades for a regime to be considered reliable (30)
- `ROLLING_WINDOW_SIZE` — rolling window in trades (20)
- `CONFIDENCE_BUCKET_EDGES` — confidence bucket boundaries `[0.25, 0.45]`
