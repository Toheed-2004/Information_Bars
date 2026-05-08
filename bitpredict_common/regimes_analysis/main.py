from bitpredict.common.regimes_analysis.config import LEDGER_STATUS_COL, LEDGER_OPEN_STATUS
from bitpredict.common.regimes_analysis.runner import run_regimes_analysis
from bitpredict.common.logging import setup_logging, get_logger
from bitpredict.common.db.services import (
    read_ohlcv,
    read_ledger,
    read_strategy_metadata,
    upsert_into_regime_analysis_col
)

import pandas as pd

setup_logging("regimes_analysis")
logger = get_logger(__name__)


def group_strategies_by_exchange_symbol_bar_type_timeframe(strategy_metadata: pd.DataFrame) -> dict:
    grouped = {}
    for key, group in strategy_metadata.groupby(['exchange', 'symbol', 'bar_type', 'timeframe']):
        grouped[key] = list(zip(group['id'], group['strategy_type']))
    return grouped


def _process_group(exchange, symbol, bar_type, timeframe, strategy_ids):
    df_ohlcv = read_ohlcv(exchange=exchange, symbol=symbol, bar_type=bar_type, timeframe=timeframe)
    for sid, stype in strategy_ids:
        logger.info(f"Processing strategy {sid} ({stype}) — {exchange} {symbol}")
        df_ledger = read_ledger(strategy_id=sid)
        result = run_regimes_analysis(df_ohlcv, df_ledger)
        if result:
            upsert_into_regime_analysis_col(strategy_id=sid, dict_data=result)
            logger.info(f"Upserted regime analysis for strategy {sid}")
        else:
            logger.warning(f"No result for strategy {sid}, skipping upsert")


def main():
    metadata = read_strategy_metadata()
    grouped_strategies = group_strategies_by_exchange_symbol_bar_type_timeframe(metadata)
    for (exchange, symbol, bar_type, timeframe), strategy_ids in grouped_strategies.items():
        _process_group(exchange, symbol, bar_type, timeframe, strategy_ids)


if __name__ == "__main__":
    main()
