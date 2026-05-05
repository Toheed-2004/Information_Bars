import warnings
import pandas as pd
from common.db.services.data import read_ohlcv
from common.ta.indicators import calculate_indicators


def extract_atr_series(atr_df):
    if isinstance(atr_df, tuple):
        atr_df = atr_df[0]

    if not isinstance(atr_df, pd.DataFrame):
        raise ValueError("ATR calculation did not return a DataFrame")

    df = atr_df.copy()

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    atr_cols = [
        col for col in df.columns
        if col.lower().startswith(("talib_ind_atr", "vbt_ind_atr"))
    ]

    if not atr_cols:
        raise ValueError("ATR column not found.")

    return df[atr_cols[0]].astype(float)


def calculate_atr_for_trailing_stop(df_signals, period, exchange, bar_type, timeframe, symbol, df_bars=None):
    """
    Compute ATR at the signal bar frequency.

    If df_bars is provided and covers the required history it is used directly,
    avoiding a DB fetch. Otherwise bar-level OHLCV is fetched from the DB.

    ATR must be at the same resolution as the signal bar data so that stop distances
    reflect actual trading-timeframe volatility. Using minute-level ATR would produce
    stops far too tight for bar-level (e.g. 1h) signals.
    """
    start_time_signals = df_signals.index[0]
    start_time = start_time_signals - pd.Timedelta(days=period + 1)

    if df_bars is not None:
        price_data = df_bars.copy()
        if "datetime" in price_data.columns:
            price_data["datetime"] = pd.to_datetime(price_data["datetime"])
            dt_index = price_data["datetime"]
        else:
            dt_index = pd.to_datetime(price_data.index)

        if dt_index.min() <= start_time:
            # Pre-loaded data covers the required ATR lookback — skip DB fetch
            pass
        else:
            warnings.warn(
                f"df_bars does not cover ATR lookback (need from {start_time}). Fetching from DB.",
                UserWarning
            )
            price_data = read_ohlcv(exchange, symbol, timeframe, bar_type, start_date=start_time)
    else:
        price_data = read_ohlcv(exchange, symbol, timeframe, bar_type, start_date=start_time)

    atr_df, _ = calculate_indicators(
        data=price_data,
        indicators={"ATR": {"timeperiod": period}},
    )
    return extract_atr_series(atr_df)
