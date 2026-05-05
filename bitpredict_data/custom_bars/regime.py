"""
bars/regime.py — Simplified regime calculation for bars.

Single entry point: calculate_and_update_regimes(table_name, exchange, symbol, bar_type, new_bars_df).
"""
import pandas as pd

from bitpredict.common.market_regimes import calculate_regimes
from bitpredict.common.db.services.data import update_bars_with_regime
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


def calculate_and_update_regimes(
    table_name: str,
    exchange: str,
    symbol: str,
    bar_type: str,
    new_bars_df: pd.DataFrame
) -> int:
    """
    Calculate market regimes for new bars and persist to DB.

    Uses the market regime state table for continuity, so no additional context bars are needed.

    Args:
        table_name: Fully-qualified bars table name.
        new_bars_df: DataFrame of newly created bars (datetime index, OHLCV columns).

    Returns:
        Number of bars updated with regime data.
    """
    if new_bars_df is None or new_bars_df.empty:
        return 0

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    # print(new_bars_df)
    try:
        combined_df = new_bars_df[ohlcv_cols].copy()
        # print(combined_df)
        # MarketRegimeAnalyzer expects a 'datetime' column
        analysis_df = combined_df.reset_index().rename(columns={"index": "datetime"})
        if "datetime" not in analysis_df.columns and combined_df.index.name:
            analysis_df = analysis_df.rename(columns={combined_df.index.name: "datetime"})
        # print(analysis_df)

        results_df = calculate_regimes(analysis_df, exchange=exchange, symbol=symbol, bar_type=bar_type, bar_timeframe=None)
        results_df.set_index('datetime', inplace=True)
        # # assigning the datetime index so reset_index() in update_bars_with_regime
        # # doesn't encounter a duplicate column name.
        # if "datetime" in results_df.columns:
        #     results_df = results_df.drop(columns=["datetime"])
        # results_df.index = combined_df.index

        # # Only update bars in new_bars_df
        # new_bars_with_regime = results_df[results_df.index.isin(new_bars_df.index)]
        # if new_bars_with_regime.empty:
        #     logger.debug("No new bars matched in regime results for %s", table_name)
        #     return 0
        # print(results_df)
        updated = update_bars_with_regime(table_name, results_df)
        logger.info("Updated %d bars with regime data in %s", updated, table_name)
        return updated

    except Exception as e:
        logger.error("Error calculating regimes for %s: %s", table_name, e)
        return 0
