import logging
from datetime import datetime, timezone
from typing import Optional, Union, List, Dict, Any
import pandas as pd
import pickle
import json
import math
import os
import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from bitpredict.common.db.config import get_engine
from bitpredict.common.db.utils import  filter_all_timeseries

from bitpredict.common.db.utils import  sanitize_fields, update_strategy_metadata, parse_timeseries_rows, clean_for_json
from bitpredict.common.db.utils import read_df
from bitpredict.common.constants import PORTFOLIO_DIR, SIMULATOR_SCHEMA
from bitpredict.common.utils.json_encoder import RobustJSONEncoder
from bitpredict.common.db.exceptions import QueryError
from bitpredict.common.db.services.shared import (
    upsert_analytics, 
    upsert_graphs, 
    upsert_graphs_daily, 
    upsert_ledgers,
    read_ledger,
    read_graphs,
    read_graphs_daily,
    read_analytics,
    get_last_trade,
    get_open_trades,
    get_all_open_trades
    )
logger = logging.getLogger(__name__)


TIMESERIES_SCALAR_METRICS = {
    "daily_cumulative_return",
    "drawdown_pct",
    "benchmark_return_pct",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_correlation",
}

TIMESERIES_TRADE_METRICS = {
    "mfe_pct",
    "mae_pct",
    "long_return_pct",
    "short_return_pct",
}


def upsert_simulator_ledgers(
    ledger: pd.DataFrame,
    engine=None,
    skip_conflict_handling: bool = False
) -> None:
    """
    Bulk upsert ledger entries for simulator schema.
    
    Args:
        ledger: DataFrame with 'id' column (strategy_id) and optional 'trial_id'
        engine: Database engine (optional)
        skip_conflict_handling: If True, uses plain INSERT (faster)
    """
    return upsert_ledgers(
        ledger=ledger,
        schema_name=SIMULATOR_SCHEMA,
        engine=engine,
        skip_conflict_handling=skip_conflict_handling
    )


def read_simulator_ledger(
    id_value: Union[str, int],
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    trial_id: Optional[bool] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read ledger entries for a specific strategy id.
    """
    return read_ledger(
        id_value=id_value,
        schema_name=SIMULATOR_SCHEMA,
        start_date=start_date,
        end_date=end_date,
        trial_id=trial_id,
        columns=columns,
        limit=limit
    )


def get_simulator_last_trade(
    id_value: Union[str, int],
    trial_id: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the most recent trade for a specific strategy id.
    """
    return get_last_trade(
        id_value=id_value,
        schema_name=SIMULATOR_SCHEMA,
        trial_id=trial_id
    )


def get_simulator_open_trades(
    id_value: Union[str, int],
    trial_id: Optional[bool] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get all open trades for a specific strategy id.
    """
    return get_open_trades(
        id_value=id_value,
        schema_name=SIMULATOR_SCHEMA,
        trial_id=trial_id,
        columns=columns
    )


def get_simulator_all_open_trades(
    trial_id: Optional[bool] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get all open trades across all strategy ids.
    """
    return get_all_open_trades(
        schema_name=SIMULATOR_SCHEMA,
        trial_id=trial_id,
        columns=columns
    )


# ============================================================================
# GRAPHS OPERATIONS
# ============================================================================

def upsert_simulator_graphs(
    df: pd.DataFrame,
    engine=None
) -> None:
    """
    Bulk upsert graph entries for simulator schema.
    
    Args:
        df: DataFrame with 'id' column (strategy_id) and optional 'trial_id'
        engine: Database engine (optional)
    """
    return upsert_graphs(
        df=df,
        schema_name=SIMULATOR_SCHEMA,
        engine=engine
    )


def read_simulator_graphs(
    id_value: Union[str, int],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    trial_id: Optional[bool] = None,
    cols: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Read graph data for a specific strategy id.
    """
    return read_graphs(
        id_value=id_value,
        schema_name=SIMULATOR_SCHEMA,
        start_date=start_date,
        end_date=end_date,
        trial_id=trial_id,
        cols=cols
    )


def upsert_simulator_graphs_daily(
    df: pd.DataFrame,
    engine=None
) -> None:
    """
    Bulk upsert daily graph entries for simulator schema.
    
    Args:
        df: DataFrame with 'id' column (strategy_id) and optional 'trial_id'
        engine: Database engine (optional)
    """
    return upsert_graphs_daily(
        df=df,
        schema_name=SIMULATOR_SCHEMA,
        engine=engine
    )


def read_simulator_graphs_daily(
    id_value: Union[str, int],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    trial_id: Optional[bool] = None,
    cols: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Read daily graph data for a specific strategy id.
    """
    return read_graphs_daily(
        id_value=id_value,
        schema_name=SIMULATOR_SCHEMA,
        start_date=start_date,
        end_date=end_date,
        trial_id=trial_id,
        cols=cols
    )


# ============================================================================
# ANALYTICS OPERATIONS
# ============================================================================

def upsert_simulator_analytics(
    rows: List[Dict[str, Any]],
    engine=None
) -> None:
    """
    Bulk upsert analytics for simulator schema.
    
    Args:
        rows: List of dicts with 'id' and optional 'trial_id'
        engine: Database engine (optional)
    """
    return upsert_analytics(
        rows=rows,
        schema_name=SIMULATOR_SCHEMA,
        engine=engine
    )


def read_simulator_analytics(
    id_value: Union[str, int],
    trial_id: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Read analytics for a specific strategy id.
    """
    return read_analytics(
        id_value=id_value,
        schema_name=SIMULATOR_SCHEMA,
        trial_id=trial_id
    )

#======================================================================================================
# portfolio object read/write using pickle for now, can switch to JSON if needed later
#======================================================================================================

def save_portfolio_object(
    strategy_id: str,
    pf,
) -> None:
    """
    Upsert ONLY portfolio_obj by saving it to the _portfolio_objects directory.
    Overwrites existing file if strategy_id already exists.
    """

    if pf is None:
        raise ValueError("Portfolio object cannot be None")

    os.makedirs(PORTFOLIO_DIR, exist_ok=True)

    try:
        portfolio_blob = pickle.dumps(pf, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise ValueError(f"Failed to serialize Portfolio object: {exc}") from exc

    file_path = os.path.join(PORTFOLIO_DIR, f"{strategy_id}.pkl")

    with open(file_path, "wb") as f:
        f.write(portfolio_blob)

def get_portfolio_object(
    strategy_id: str,
    obj_path: str = None,
):
    """
    Fetch and return deserialized Portfolio object from the portfolio objects directory.
    Returns None if file not found.

    Args:
        strategy_id: Unique identifier for the strategy.
        obj_path:    Optional complete path to portfolio pickle file.
                     If not provided, uses PORTFOLIO_DIR + strategy_id + .pkl
    """
    # If obj_path is provided and is a complete file path, use it directly
    if obj_path is not None and obj_path.endswith('.pkl'):
        file_path = obj_path
    else:
        # Otherwise, construct the path from PORTFOLIO_DIR and strategy_id
        base_dir = obj_path if obj_path is not None else PORTFOLIO_DIR
        file_path = os.path.join(base_dir, f"{strategy_id}.pkl")

    if not os.path.exists(file_path):
        return None

    with open(file_path, "rb") as f:
        blob = f.read()

    if not blob:
        return None

    try:
        pf = pickle.loads(blob)
    except Exception as exc:
        raise ValueError(f"Failed to deserialize Portfolio object: {exc}") from exc

    return pf

# ============================================================================
# SIMULATOR OPERATIONS
# ============================================================================

def update_strategies(
    strategy_id: str,
    metrics: Dict[str, Any],
    ledger: pd.DataFrame,
    portfolio_state_path: Optional[str] = None,
    regimes_analysis: Optional[Dict[str, Any]] = None
) -> None:
    """
    Update a strategy's metadata, analytics, and graphs using the provided metrics,
    ledger, benchmark returns, and optional regimes_analysis.
    
    Args:
        strategy_id: The strategy identifier
        metrics: Dictionary of calculated metrics
        ledger: DataFrame containing trade ledger data
        portfolio_state_path: Optional path to the portfolio object pickle file
        regimes_analysis: Optional regime analysis dictionary
    """
    logger.info(f"Starting update_strategies for strategy {strategy_id}")
    metadata_updates, analytics_updates, graphs_updates, graphs_daily_updates = _build_updates_from_metrics(
        metrics, ledger,  regimes_analysis
    )

    logger.debug(f"Built updates: metadata_keys={list(metadata_updates.keys())}, "
                f"analytics_keys={list(analytics_updates.keys())}, "
                f"graph_keys={list(graphs_updates.keys())}")
    
    if portfolio_state_path:
        analytics_updates['portfolio_state_path'] = portfolio_state_path
        logger.info(f"Added portfolio_state_path to analytics: {portfolio_state_path}")

    try:
        if metadata_updates:
            logger.info(f"Upserting metadata for strategy {strategy_id}")
            update_strategy_metadata(strategy_id, metadata_updates)
        if analytics_updates:
            logger.info(f"Upserting analytics for strategy {strategy_id} with {len(analytics_updates)} keys")
            upsert_analytics(strategy_id, analytics_updates,  schema_name=SIMULATOR_SCHEMA)
        else:
            logger.warning(f"No analytics updates to insert for strategy {strategy_id}")
        if graphs_updates:
            logger.info(f"Upserting graphs for strategy {strategy_id} with {len(graphs_updates)} keys")
            upsert_graphs(strategy_id, graphs_updates, schema_name=SIMULATOR_SCHEMA)
        else:
            logger.warning(f"No graph updates to insert for strategy {strategy_id}")
        if graphs_daily_updates:
            logger.info(f"Upserting daily graphs for strategy {strategy_id} with {len(graphs_daily_updates)} keys")
            upsert_graphs_daily(strategy_id, graphs_daily_updates, schema_name=SIMULATOR_SCHEMA)
        else:
            logger.warning(f"No daily graph updates to insert for strategy {strategy_id}")

        logger.info(f"Successfully updated strategy ID {strategy_id}")
    except Exception as exc:
        logger.exception(f"Failed to update strategy ID {strategy_id}")
        raise

# ============================================================================
# METRICS OPERATIONS
# ============================================================================

def _build_updates_from_metrics(stats, ledger, regimes_analysis=None):
    """
    Build metadata, analytics, and graph update dictionaries from a metrics
    dictionary (as returned by calculate_comprehensive_stats) and optional
    ledger, benchmark data, and regimes_analysis.
    """
    _ra = stats.get('risk_adjusted', {})
    trade_analysis = stats.get('trade_analysis', {})
    profit_loss = dict(stats.get('profit_loss', {}))
    # profit_loss.pop('monthly_returns', None)  # Remove large structures not needed in metadata

    # Get regime fitness score from regimes_analysis if available
    regime_fitness_score = 0.0
    best_regime = 'place_holder'
    if regimes_analysis:
        try:
            regimes_fitness = regimes_analysis['regime_fitness']['by_regime_label']
            
            # Get regime names and their fitness scores, excluding TRANSITION
            regimes = [r for r in regimes_fitness.keys() if r != 'TRANSITION']
            
            if regimes:
                fitness_scores = np.array([regimes_fitness[r]['fitness_score'] for r in regimes])
                
                # Find index of maximum fitness score
                best_idx = np.argmax(fitness_scores)
                best_regime = regimes[best_idx]
                regime_fitness_score = float(fitness_scores[best_idx])
        except (KeyError, IndexError, TypeError, ValueError):
            # Fallback if structure is different
            regime_fitness_score = regimes_analysis.get('best_score', 0.0)
            best_regime = regimes_analysis.get('best_regime', 'place_holder')
    
    # Get direction bias from stats if available
    direction_bias = stats.get('direction_bias', 'both')
    
    # Get trade frequency from stats if available
    trade_frequency = stats.get('trade_frequency', 'swing')
    
    # ---------------- ROLLING STATUS ----------------
    win_rate = trade_analysis.get('win_rate_pct', 0)
    sharpe = _ra.get('sharpe_ratio', 0)
    
    if win_rate < 40:
        rolling_status = 'poor_win_rate'
    elif sharpe < 0.5 and win_rate < 50:
        rolling_status = 'deteriorating'
    elif sharpe < 0:
        rolling_status = 'critical'
    else:
        rolling_status = 'healthy'
    

    # ---- Metadata updates with all new columns ----
    metadata = {
        "total_return_pct": profit_loss.get("total_return_pct", 0),
        "sharpe_ratio": _ra.get("sharpe_ratio", stats.get("sharpe_ratio", 0)),
        "sortino_ratio": _ra.get("sortino_ratio", stats.get("sortino_ratio", 0)),
        "calmar_ratio": _ra.get("calmar_ratio", stats.get("calmar_ratio", 0)),
        "profit_factor": stats.get("profit_loss", {}).get("profit_factor", stats.get("profit_factor", 0)),
        "max_drawdown_pct": stats.get("drawdown_analysis", {}).get("max_drawdown_pct", stats.get("max_drawdown_pct", 0)),
        "win_rate_pct": win_rate,
        
        # Validation & Regime
        "best_regime": best_regime,
        "regime_fitness_score": regime_fitness_score,
        "rolling_status": rolling_status,
        
        # Trading Parameters
        "direction_bias": direction_bias,
        "trade_frequency": trade_frequency,
    }


    # ---- Analytics updates ----

    EXCLUDE_ANALYTICS = {
        # graph/time-series (NOT DB)
        "cumulative_return",
        "drawdown_series",
        "rolling_sharpe",
        "rolling_sortino",
        "rolling_correlation",
        "mfe_pct",
        "mae_pct",
        "monthly_returns",
        "monthly_matrix",
        "heatmaps_data",
        "directional_pnl",
    }

    analytics = {k: v for k, v in stats.items() if k not in EXCLUDE_ANALYTICS}

    analytics.update({
        "directional_metrics": stats.get('pnl_distribution', {}),
        "regimes_analysis": regimes_analysis or {} ,
        "profit_loss": profit_loss,
        })
    logger.debug(f"Built analytics dict with keys: {list(analytics.keys())}")

    # ---- Graph updates (time series data) ----
    graphs = {}
    graphs_daily = {}
    if ledger is not None:
        first_trade_time = ledger["entry_datetime"].min() 

        filtered_series = filter_all_timeseries(
            stats,
            first_trade_time,
            ["cumulative_return", "drawdown_series",
            "rolling_sharpe", "rolling_sortino", "rolling_correlation", "benchmark_returns"]
        )


        graphs = {
            "mfe_pct": stats.get("mfe_pct", {}),
            "mae_pct": stats.get("mae_pct", {}),
            "long_return_pct": stats.get('directional_pnl', {}).get('long', {}),
            "short_return_pct": stats.get('directional_pnl', {}).get('short', {}),
        }

        graphs_daily = {
            "daily_cumulative_return": filtered_series["cumulative_return"],
            "drawdown_pct": filtered_series["drawdown_series"],
            "benchmark_return_pct": filtered_series["benchmark_returns"],
            "rolling_sharpe": filtered_series["rolling_sharpe"],
            "rolling_sortino": filtered_series["rolling_sortino"],
            "rolling_correlation": filtered_series["rolling_correlation"],
        }

    return metadata, analytics, graphs, graphs_daily