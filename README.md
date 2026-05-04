# Information Bars for Machine Learning — Research Repository

> **Companion code for:**  
> *"Information Bars: A Systematic Comparison of Tick-Level and Minute-Aggregated Bar Construction Methods for Cryptocurrency Markets"*  
> Muhammad Toheed Fayyaz, Abdul Jabbar, Syed Qaisar Jalil

---

## What Are Information Bars?

Traditional finance uses **calendar bars** (1-minute, 1-hour candles) where each bar covers a fixed time interval regardless of market activity. This introduces statistical problems: serial autocorrelation, non-Gaussian return distributions, and regime-dependent bar sizes that make machine learning models harder to train.

**Information bars** close when a market-activity threshold is reached rather than when a clock ticks. Six types are studied:

| Bar Type | Closes when… | Signal |
|---|---|---|
| **Dollar** | Cumulative dollar volume (Σ price × qty) hits target | Liquidity |
| **Volume** | Cumulative traded quantity hits target | Supply/demand |
| **Volatility** | Cumulative absolute price movement hits target | Information arrival |
| **Range** | High-low price excursion hits target | Price discovery |
| **Renko** | Price displacement from reference hits target | Trend direction |
| **Hybrid** | Dollar-volume AND volatility both hit targets | Multi-dimensional |

Each bar type is implemented for two data resolutions:
- **Minute pipeline** — uses 1-minute OHLCV pre-aggregates
- **Tick pipeline** — uses raw Binance aggTrade records (exact price × qty per trade)

---

## Repository Structure

```
research_paper/
├── common/                  # Shared utilities
│   ├── constants.py         # All calibration constants
│   ├── data_loader.py       # CSV-based data I/O
│   └── logging.py           # Logging setup
├── bars/
│   ├── main.py              # ← Entry point: generate bars
│   ├── compare.py           # ← Entry point: run comparisons
│   ├── compare_bars.py      # Full 9-panel comparison engine
│   ├── processor.py         # Core accumulation loop
│   ├── cli_control.py       # Argument parsers
│   └── bar_types/
│       ├── base.py                  # Abstract BaseBar
│       ├── dollar_bars.py           # Minute dollar bars
│       ├── volume_bars.py           # Minute volume bars
│       ├── volatility_bars.py       # Minute volatility bars
│       ├── range_bars.py            # Minute range bars
│       ├── renko_bars.py            # Minute renko bars
│       ├── hybrid_bars.py           # Minute hybrid bars
│       ├── tick_dollar_bars.py      # Tick dollar bars
│       ├── tick_volume_bars.py      # Tick volume bars
│       ├── tick_volatility_bars.py  # Tick volatility bars
│       ├── tick_range_bars.py       # Tick range bars
│       ├── tick_renko_bars.py       # Tick renko bars
│       └── tick_hybrid_bars.py      # Tick hybrid bars
├── data/
│   ├── raw_data/            # ← Place your CSV files here
│   ├── processed_bars/      # Generated bar CSVs (auto-created)
│   └── comparison_results/  # Comparison outputs (auto-created)
├── ml/                      # ML pipeline (future work)
├── config/
│   └── custom_bars.yaml     # Bar type enable/disable configuration
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
git clone https://github.com/<your-org>/information-bars-research
cd information-bars-research

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Data Requirements

Place your CSV files in `data/raw_data/`:

**Minute OHLCV CSV** — columns: `datetime`, `open`, `high`, `low`, `close`, `volume`
```
data/raw_data/1minute_btcusdt_2024.csv
```

**Tick CSV** (Binance aggTrades format) — columns: `price`, `qty`, `timestamp` (ms), `is_buyer_maker`
```
data/raw_data/btcusdt_aggTrades_2024.csv
```

---

## Usage

### Generate bars

```bash
# All minute-level bar types
python bars/main.py \
    --source minute \
    --types all \
    --minute-csv data/raw_data/1minute_btcusdt_2024.csv

# Tick-level dollar bars only
python bars/main.py \
    --source tick \
    --types dollar \
    --tick-csv   data/raw_data/btcusdt_aggTrades_2024.csv \
    --minute-csv data/raw_data/1minute_btcusdt_2024.csv

# Multiple specific types
python bars/main.py \
    --source minute \
    --types dollar volatility hybrid \
    --minute-csv data/raw_data/1minute_btcusdt_2024.csv

# Full help
python bars/main.py --help
```

Output: `data/processed_bars/<exchange>_<symbol>_<type>_<source>_bars.csv`

### Compare minute vs tick pipelines

```bash
python bars/compare.py \
    --types all \
    --time-bars \
    --minute-csv data/raw_data/1minute_btcusdt_2024.csv \
    --tick-csv   data/raw_data/btcusdt_aggTrades_2024.csv \
    --output-dir data/comparison_results
```

Or use the full comparison engine directly:

```bash
python bars/compare_bars.py \
    --types dollar \
    --time-bars \
    --figures DE
```

Output: `data/comparison_results/` — HTML figures, PDF panels, stats TXT files.

---

## Calibration Design

Both pipelines use a **shared calibration framework**: the lookback window (14 days), EMA adaptation rate, and duration bounds are computed from minute OHLCV data for both pipelines.

For the tick pipeline, the activity **threshold value** is subsequently replaced with a tick-native equivalent:

| Bar type | Tick-native threshold |
|---|---|
| Dollar | Median daily Σ(p×q) ÷ target bars/day |
| Volume | Median daily Σq ÷ target bars/day |
| Volatility | Median daily Σ\|log(pᵢ/pᵢ₋₁)\| ÷ target bars/day |
| Range | Median daily (H−L)/pₒₚₑₙ ÷ target bars/day |

This isolates **signal resolution** as the sole experimental variable.

---

## Reproducibility

All results are generated by running the code — no pre-computed outputs are committed to the repository.

To reproduce the paper's results:
1. Place the 2024 BTCUSDT minute and tick CSVs in `data/raw_data/`
2. Run `python bars/main.py --source minute --types all --minute-csv ...`
3. Run `python bars/main.py --source tick --types all --tick-csv ... --minute-csv ...`
4. Run `python bars/compare_bars.py --time-bars --figures DE`

---

## Citation

```bibtex
@article{fayyaz2025infobars,
  title   = {Information Bars: A Systematic Comparison of Tick-Level and
             Minute-Aggregated Bar Construction Methods for Cryptocurrency Markets},
  author  = {Fayyaz, Muhammad Toheed and Jabbar, Abdul and Jalil, Syed Qaisar},
  journal = {IEEE Transactions},
  year    = {2025}
}
```
