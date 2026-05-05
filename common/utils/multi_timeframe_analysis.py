import pandas as pd
from bitpredict.common.db.services.data import read_ohlcv


def multitimeframe_analysis(
    *,
    symbol: str,
    exchange: str,
    base_tf: str,
    lower_tfs: list[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Multi-timeframe OHLCV alignment WITHOUT lookahead bias.

    Assumptions:
    - All OHLCV data is already resampled and stored in DB
    - `timestamp` represents candle CLOSE time
    - `read_ohlcv` returns completed candles only

    Returns:
    - DataFrame indexed by base timeframe
    - Lower timeframe OHLCV appended as columns (open_1h, close_2h, etc.)
    """

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Convert bigint milliseconds to proper UTC timestamp
        if pd.api.types.is_integer_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df.sort_values("timestamp")


    # ========================
    # Fetch base timeframe
    # ========================
    base_df = read_ohlcv(
        symbol=symbol,
        exchange=exchange,
        timeframe=base_tf,
        start_date=start,
        end_date=end,
    )
    base_df = _prepare(base_df)

    result = base_df.copy()

    # ========================
    # Align lower timeframes
    # ========================
    for tf in lower_tfs:
        lower_df = read_ohlcv(
            symbol=symbol,
            exchange=exchange,
            timeframe=tf,
            start_date=start,
            end_date=end,
        )
        lower_df = _prepare(lower_df)

        # Rename columns (except timestamp)
        rename_map = {
            c: f"{c}_{tf}"
            for c in lower_df.columns
            if c != "timestamp"
        }
        lower_df = lower_df.rename(columns=rename_map)

        # ASOF JOIN (backward only)
        result = pd.merge_asof(
            result.sort_values("timestamp"),
            lower_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

    return result

def main():

    result = multitimeframe_analysis(symbol='btc', exchange="binance", base_tf='8h', lower_tfs=['3h', '15m'])
    result.to_csv("multitimframe_analysis.csv", index=False)

if __name__ == "__main__":
    main()