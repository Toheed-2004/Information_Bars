import numpy as np
import pandas as pd
from pathlib import Path
from bitpredict.common.market_regimes import calculate_regimes

rng = np.random.default_rng(42)
n = 500

df = pd.DataFrame({
    "datetime": pd.date_range("2024-01-01", periods=n, freq="1h"),
    "close": 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n)),
})

df_out = calculate_regimes(df, exchange="binance", symbol="btc", bar_type="time", bar_timeframe="1h")

out_path = Path(__file__).parent / "regime_output.csv"
df_out.to_csv(out_path, index=False)
# print(df_out[["datetime", "close", "regime_label", "trend_strength_z", "vol_percentile"]].tail(10))
