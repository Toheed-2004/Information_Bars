"""
Functional service layer for strategy database operations.

All functions follow the new schemas with strategy_id as primary key
in analytics, configs, and time_series tables.
"""
import random
import time
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import json
import math
import numpy as np
import pandas as pd
from datetime import timezone
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from collections import defaultdict
from bitpredict.common.db.config import get_engine
from bitpredict.common.utils.json_encoder import RobustJSONEncoder
from bitpredict.common.logging import get_logger
from bitpredict.common.db.exceptions import QueryError
from bitpredict.common.db.models import ensure_strategies_training_table
from bitpredict.common.constants import STRATEGIES_SCHEMA, SIMULATOR_SCHEMA
from bitpredict.common.db.models import (
    ensure_strategies_metadata_table,
    ensure_strategies_configs_table,

)
from bitpredict.common.db.utils import downsample_to_avg, sanitize_fields, clean_for_json

logger = get_logger(__name__)



# ----------------------------------------------------------------------------
# METADATA OPERATIONS
# ----------------------------------------------------------------------------
def upsert_strategy_metadata(data: Dict[str, Any]) -> int:
    """
    Upsert strategy metadata. If ID is not provided, a new auto-increment ID will be generated.
    Returns the strategy ID (either provided or auto-generated).
    """
    engine = get_engine()
    ensure_strategies_metadata_table(engine)
    
    # Complete list of all columns from the table
    columns = [
        # User & Ownership
        'user_id', 'owner_type', 'display_name', 'code',
        
        # Strategy Classification
        'strategy_type', 'mtf' ,'exchange', 'symbol',
        
        # Trading Parameters
        'bar_type', 'timeframe', 'direction_bias', 'trade_frequency',
        
        # Validation & Regime
        'best_regime', 'regime_fitness_score', 'rolling_status',
        
        # Status Flags
        'public','simulator', 'access_level',
        
        # Tags & Metadata
        'tags', 'version', 'description',
        
        # Performance Metrics
        'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 
        'win_rate_pct', 'sortino_ratio', 'calmar_ratio', 'profit_factor',
        'total_trades', 'cagr', 'expectancy', 'avg_trade_return_pct',
        'recovery_factor', 'sqn', 'exposure_pct', 'avg_monthly_return_pct',
        
        # Sparkline & Hierarchy
        'sparkline_data', 'parent_id',
        
        # Deployment Counters
        'demo_count', 'live_count',
        
        # Timestamps (created_at is usually not updated on upsert)
        'updated_at', 'created_at'
    ]

    # Prepare row data with defaults for missing values
    row_data = {}
    for col in columns:
        if col in data and data[col] is not None:
            row_data[col] = data[col]
        else:
            # Apply defaults based on column type
            if col in ['user_id', 'parent_id', 'timeframe']:
                row_data[col] = None  # Foreign keys and timeframe can be NULL
            elif col in ['owner_type']:
                row_data[col] = 'platform'
            elif col in ['direction_bias']:
                row_data[col] = 'both'
            elif col in ['trade_frequency']:
                row_data[col] = 'swing'
            elif col in ['best_regime']:
                row_data[col] = 'place_holder'
            elif col in ['rolling_status']:
                row_data[col] = 'healthy'
            elif col in ['access_level']:
                row_data[col] = 'admin_only'
            elif col in ['public', 'simulator']:
                row_data[col] = True
            elif col in ['mtf']:
                row_data[col] = False
            elif col in ['tags']:
                row_data[col] = []
            elif col in ['sparkline_data']:
                row_data[col] = []
            elif col in ['version', 'description', 'code', 'display_name']:
                # Required string fields - use 'place_holder' if missing
                row_data[col] = data.get(col, 'place_holder')
            elif col == 'updated_at':
                row_data[col] = datetime.now()
            elif col in ['created_at']:
                row_data[col] = datetime.now()
            else:
                # Numeric fields default to 0
                row_data[col] = 0
    
    # Ensure required fields are never None
    required_string_fields = ['display_name', 'code', 'strategy_type', 
                              'exchange', 'symbol', 'bar_type']
    for field in required_string_fields:
        if field not in row_data or row_data[field] is None or row_data[field] == '':
            row_data[field] = 'place_holder'

    
    # Filter to only columns that exist in row_data
    existing_cols = [col for col in columns if col in row_data]
    col_names = ', '.join(existing_cols)
    placeholders = ', '.join([f':{col}' for col in existing_cols])
    
    # Update clause excludes created_at and id
    update_cols = [col for col in existing_cols if col not in ('created_at',)]
    update_clause = ', '.join([f'{col} = EXCLUDED.{col}' for col in update_cols])

    # Check if ID is provided for update mode
    strategy_id = data.get('id')
    
    if strategy_id is not None:
        # Update existing strategy
        strategy_id = int(strategy_id) if isinstance(strategy_id, str) else strategy_id
        row_data['id'] = strategy_id
        col_names_with_id = 'id, ' + col_names
        placeholders_with_id = ':id, ' + placeholders
        
        query = text(f"""
            INSERT INTO {STRATEGIES_SCHEMA}.metadata ({col_names_with_id})
            VALUES ({placeholders_with_id})
            ON CONFLICT (id)
            DO UPDATE SET {update_clause}
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(query, row_data)
            logger.debug(f"Updated strategy with ID: {strategy_id}")
            return strategy_id
        except SQLAlchemyError as exc:
            logger.exception("Failed to upsert strategy metadata")
            raise QueryError("Failed to upsert strategy metadata") from exc
    else:
        # Insert new strategy and get the auto-generated ID
        query = text(f"""
            INSERT INTO {STRATEGIES_SCHEMA}.metadata ({col_names})
            VALUES ({placeholders})
            RETURNING id
        """)

        try:
            with engine.begin() as conn:
                result = conn.execute(query, row_data)
                strategy_id = result.scalar()
            logger.debug(f"Created new strategy with auto-generated ID: {strategy_id}")
            return strategy_id
        except SQLAlchemyError as exc:
            logger.exception("Failed to insert strategy metadata")
            raise QueryError("Failed to insert strategy metadata") from exc

def bulk_update_strategy_metadata(updates_list: List[Tuple[int, Dict[str, Any]]]) -> None:
    """
    Bulk update strategy metadata for multiple strategies using CASE statements.
    More efficient than individual updates.
    
    Args:
        updates_list: List of tuples (strategy_id, metadata_dict) to update
    """
    if not updates_list:
        return

    engine = get_engine()
    
    valid_columns = {
        'strategy_type', 'exchange', 'symbol',
        'bar_type', 'timeframe',  'simulator',  'access_level', 'tags',
        'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct',
        'sortino_ratio', 'calmar_ratio', 'profit_factor', 'sparkline_data', 'version',
        'description'
    }
    
    # Collect all columns that need updating across all strategies
    all_columns = set()
    strategy_data = {}
    
    for strategy_id, updates in updates_list:
        if not updates:
            continue
            
        row_data = {col: updates[col] for col in updates if col in valid_columns}
        if not row_data:
            continue
            
        strategy_data[strategy_id] = row_data
        all_columns.update(row_data.keys())
    
    if not strategy_data:
        return
    
    all_columns = list(all_columns)
    strategy_ids = list(strategy_data.keys())
    
    try:
        with engine.begin() as conn:
            # Build CASE statements for each column
            set_parts = []
            params = {}
            
            for col in all_columns:
                case_parts = []
                for idx, strategy_id in enumerate(strategy_ids):
                    if col in strategy_data[strategy_id]:
                        value = strategy_data[strategy_id][col]
                        param_name = f"{col}_{idx}"
                        case_parts.append(f"WHEN id = :sid_{idx} THEN :{param_name}")
                        params[param_name] = value
                        params[f"sid_{idx}"] = strategy_id
                
                if case_parts:
                    case_clause = f"{col} = CASE " + " ".join(case_parts) + " ELSE " + col + " END"
                    set_parts.append(case_clause)
            
            if not set_parts:
                return
            
            set_clause = ", ".join(set_parts)
            placeholders = ", ".join([f":sid_{idx}" for idx in range(len(strategy_ids))])
            
            query = text(f"""
                UPDATE {STRATEGIES_SCHEMA}.metadata 
                SET {set_clause}
                WHERE id IN ({placeholders})
            """)
            
            conn.execute(query, params)
            
        logger.info(f"Bulk updated metadata for {len(strategy_data)} strategies")
        
    except SQLAlchemyError as exc:
        logger.exception("Failed to bulk update strategy metadata")
        raise QueryError("Failed to bulk update strategy metadata") from exc

def read_strategy_metadata(
    exchange: Optional[str] = None,
    symbol: Optional[str] = None,
    bar_type: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: Optional[int] = None,
    order_by: str = "id ASC"
) -> pd.DataFrame:
    engine = get_engine()
    
    # Build filters dictionary from non-None parameters
    filters = {}
    if exchange is not None:
        filters["exchange"] = exchange
    if symbol is not None:
        filters["symbol"] = symbol
    if bar_type is not None:
        filters["bar_type"] = bar_type
    # timeframe filter only applied when bar_type == 'time'
    if timeframe is not None and bar_type == 'time':
        filters["timeframe"] = timeframe

    where_parts = []
    params = {}
    for key, value in filters.items():
        where_parts.append(f"{key} = :{key}")
        params[key] = value

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
    query = f"SELECT * FROM {STRATEGIES_SCHEMA}.metadata {where_clause} ORDER BY {order_by}"
    if limit:
        query += f" LIMIT {limit}"

    try:
        return pd.read_sql(text(query), engine, params=params)
    except SQLAlchemyError as exc:
        logger.exception("Failed to read strategy metadata")
        raise QueryError("Failed to read strategy metadata") from exc


def get_strategy_metadata_by_id(strategy_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a single strategy metadata row by ID.

    Args:
        strategy_id: The strategy ID to fetch.

    Returns:
        Dictionary containing the strategy metadata row, or None if not found.

    Raises:
        QueryError: If the database query fails.

    Example:
        >>> metadata = get_strategy_metadata_by_id(42)
        >>> if metadata:
        ...     print(metadata['display_name'], metadata['sharpe_ratio'])
    """
    engine = get_engine()
    query = text(f"SELECT * FROM {STRATEGIES_SCHEMA}.metadata WHERE id = :id")

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"id": strategy_id}).fetchone()
        return dict(result._mapping) if result else None
    except SQLAlchemyError as exc:
        logger.exception("Failed to read strategy metadata for id %s", strategy_id)
        raise QueryError(f"Failed to read strategy metadata for id {strategy_id}") from exc


# ----------------------------------------------------------------------------
# TRAINING OPERATIONS
# ----------------------------------------------------------------------------
def upsert_strategy_training(strategy_id: str, training: Dict[str, Any]) -> None:
    """
    Upsert training data for a strategy.
    """
    
    engine = get_engine()
    # Ensure the training table exists before inserting
    ensure_strategies_training_table(engine)
    ensure_strategies_configs_table(engine)

    numeric_fields = {
        'cagr', 'sortino_ratio', 'calmar_ratio', 'omega_ratio', 'tail_ratio',
        'profit_factor', 'expectancy', 'recovery_factor', 'sqn',
        'max_consecutive_losses', 'avg_trade_duration', 'risk_of_ruin'
    }
    # Sanitize numeric fields
    training = sanitize_fields(training, numeric_fields)

    individual_metrics = {
        'cagr', 'sortino_ratio', 'calmar_ratio', 'omega_ratio', 'tail_ratio',
        'profit_factor', 'expectancy', 'recovery_factor', 'sqn',
        'max_consecutive_losses', 'avg_trade_duration', 'risk_of_ruin'
    }

    jsonb_columns = {'walk_forward', 'monte_carlo', 'holdout', 'regimes_analysis'}

    # Initialize row_data with strategy_id and created_at
    row_data = {'strategy_id': strategy_id, 'created_at': datetime.now()}

    for key, value in training.items():
        if key in individual_metrics:
            row_data[key] = value
        elif key in jsonb_columns:
            cleaned = clean_for_json(value)
            row_data[key] = json.dumps(cleaned, cls=RobustJSONEncoder) if cleaned is not None else None

    columns = list(row_data.keys())
    col_names = ', '.join(columns)
    placeholders = ', '.join([f':{col}' for col in columns])
    update_cols = [col for col in columns if col != 'strategy_id']
    
    if not update_cols:
        logger.debug("No training metrics to update for strategy %s", strategy_id)
        return

    update_clause = ', '.join([f'{col} = EXCLUDED.{col}' for col in update_cols])

    query = text(f"""
        INSERT INTO {STRATEGIES_SCHEMA}.training ({col_names})
        VALUES ({placeholders})
        ON CONFLICT (strategy_id)
        DO UPDATE SET {update_clause}
    """)

    try:
        with engine.begin() as conn:
            conn.execute(query, row_data)
    except SQLAlchemyError as exc:
        logger.exception("Failed to upsert strategy training")
        raise QueryError("Failed to upsert strategy training") from exc


def read_strategy_training(strategy_id: str) -> pd.DataFrame:
    """
    Read training data for a strategy.
    """
    engine = get_engine()
    query = text(f"SELECT * FROM {STRATEGIES_SCHEMA}.training WHERE strategy_id = :strategy_id")

    try:
        return pd.read_sql(query, engine, params={"strategy_id": strategy_id})
    except SQLAlchemyError as exc:
        logger.exception("Failed to read strategy training")
        raise QueryError("Failed to read strategy training") from exc


# ----------------------------------------------------------------------------
# CONFIGURATION OPERATIONS
# ----------------------------------------------------------------------------
def upsert_strategy_config(strategy_id: str, config: Dict[str, Any]) -> None:
    engine = get_engine()
    ensure_strategies_configs_table(engine)

    row_data = {
        "strategy_id": strategy_id,
        "data": json.dumps(clean_for_json(config.get("data")), cls=RobustJSONEncoder) if config.get("data") else None,
        "input": json.dumps(clean_for_json(config.get("input")), cls=RobustJSONEncoder) if config.get("input") else None,
        "backtest": json.dumps(clean_for_json(config.get("backtest")), cls=RobustJSONEncoder) if config.get("backtest") else None,
        "training": json.dumps(clean_for_json(config.get("training")), cls=RobustJSONEncoder) if config.get("training") else None,
        "model_path": config.get("model_path"),
        "updated_at": datetime.now(),
    }

    query = text(f"""
        INSERT INTO {STRATEGIES_SCHEMA}.configs (strategy_id, data, input, backtest, training, model_path, updated_at)
        VALUES (:strategy_id, :data, :input, :backtest, :training, :model_path, :updated_at)
        ON CONFLICT (strategy_id)
        DO UPDATE SET
            data = EXCLUDED.data,
            input = EXCLUDED.input,
            backtest = EXCLUDED.backtest,
            training = EXCLUDED.training,
            model_path = EXCLUDED.model_path,
            updated_at = EXCLUDED.updated_at
    """)

    try:
        with engine.begin() as conn:
            conn.execute(query, row_data)
    except SQLAlchemyError as exc:
        logger.exception("Failed to upsert strategy config")
        raise QueryError("Failed to upsert strategy config") from exc


def read_strategy_config(strategy_id: str) -> pd.DataFrame:
    engine = get_engine()
    query = text(f"SELECT * FROM {STRATEGIES_SCHEMA}.configs WHERE strategy_id = :strategy_id")

    try:
        return pd.read_sql(query, engine, params={"strategy_id": strategy_id})
    except SQLAlchemyError as exc:
        logger.exception("Failed to read strategy config")
        raise QueryError("Failed to read strategy config") from exc


# ----------------------------------------------------------------------------
# Save Strategy
# ----------------------------------------------------------------------------

def generate_unique_strategy_code(strategy_name: str, max_length: int = 16) -> str:
    """
    Generate a unique code from the strategy name.
    
    Args:
        strategy_name: Unique strategy name
        max_length: Maximum length of the code (default 16)
    
    Returns:
        Unique code string
    
    Examples:
        >>> generate_unique_strategy_code('test_strategy_1')
        'TESTSTRAT1'
        
        >>> generate_unique_strategy_code('my_awesome_strategy_v2')
        'MYAWESOMESTR'
    """
    # Remove special characters and convert to uppercase
    code = ''.join(c for c in strategy_name if c.isalnum() or c == '_')
    code = code.replace('_', '').upper()
    
    # Take the last max_length characters
    return code[-max_length:] if len(code) > max_length else code


def save_strategy(
    exchange: str, 
    symbol: str, 
    bar_type: str,
    timeframe: str, 
    strategy_name: str,
    strategy_type: str, 
    strategy_config: Dict[str, Any], 
    metrics: Dict[str, Any], 
    dict_regime_analysis: Dict[str, Any],
    backtest_config: Dict[str, Any], 
    training_config: Dict[str, Any], 
    data_config: Dict[str, Any], 
    hold_out: Dict[str, Any], 
    walk_forward: Dict[str, Any], 
    monte_carlo: Dict[str, Any], 
    ledger: pd.DataFrame,
    description: Optional[str] = None
    ):
        
    # Helper for sparkline and other time series
    cumulative_pnl_data = ledger["cum_account_return"].fillna(0).cumsum().tolist()

    # convenience reference to risk_adjusted group (may be missing)
    _ra = metrics.get('risk_adjusted', {})
    
    # Extract best regime and fitness score from regime analysis
    best_regime = 'place_holder'
    best_fitness = 0.0
    try:
        regimes_fitness = dict_regime_analysis['regime_fitness']['by_regime_label']
        
        # Get regime names and their fitness scores, excluding TRANSITION
        regimes = [r for r in regimes_fitness.keys() if r != 'TRANSITION']
        
        if regimes:
            fitness_scores = np.array([regimes_fitness[r]['fitness_score'] for r in regimes])
            
            # Find index of maximum fitness score
            best_idx = np.argmax(fitness_scores)
            best_regime = regimes[best_idx]
            best_fitness = float(fitness_scores[best_idx])
    except (KeyError, IndexError, TypeError, ValueError):
        # Keep defaults if extraction fails
        pass

    metadata_payload = {
        # User & Ownership
        "user_id": None,
        "owner_type": "platform",
        "display_name": strategy_name,
        "code": generate_unique_strategy_code(strategy_name),
        
        # Strategy Classification
        "strategy_type": strategy_type,
        "mtf": False,
        "exchange": exchange,
        "symbol": symbol,
        
        # Trading Parameters
        "bar_type": bar_type,
        "timeframe": timeframe if bar_type == "time" else None,
        "direction_bias": "both",
        "trade_frequency": "swing",
        
        # Validation & Regime
        "best_regime": best_regime,
        "regime_fitness_score": best_fitness,
        "rolling_status": "healthy",
        
        # Status Flags
        "simulator": True,
        "access_level": "admin_only",
        
        # Tags & Metadata
        "tags": [],
        "version": "0",
        "description": description if description else "place_holder",
        
        # Performance Metrics
        "total_return_pct": metrics.get("profit_loss", {}).get("total_return_pct", metrics.get("total_return_pct", 0)),
        "sharpe_ratio": _ra.get("sharpe_ratio", metrics.get("sharpe_ratio", 0)),
        "sortino_ratio": _ra.get("sortino_ratio", metrics.get("sortino_ratio", 0)),
        "calmar_ratio": _ra.get("calmar_ratio", metrics.get("calmar_ratio", 0)),
        "profit_factor": metrics.get("profit_loss", {}).get("profit_factor", metrics.get("profit_factor", 0)),
        "max_drawdown_pct": metrics.get("drawdown_analysis", {}).get("max_drawdown_pct", metrics.get("max_drawdown_pct", 0)),
        "win_rate_pct": metrics.get("trade_analysis", {}).get("win_rate_pct", metrics.get("win_rate_pct", 0)),
        "total_trades": metrics.get("trade_analysis", {}).get("total_trades", metrics.get("total_trades", 0)),
        "cagr": _ra.get("cagr", metrics.get("cagr", 0)),
        "expectancy": metrics.get("trade_analysis", {}).get("expectancy", metrics.get("expectancy", 0)),
        "avg_trade_return_pct": round(metrics.get("trade_analysis", {}).get("avg_return_all_trades", metrics.get("avg_return_all_trades", 0)) * 100, 2),
        "recovery_factor": metrics.get("drawdown_analysis", {}).get("recovery_factor", metrics.get("recovery_factor", 0)),
        "sqn": metrics.get("trade_analysis", {}).get("sqn", metrics.get("sqn", 0)),
        "exposure_pct": metrics.get("exposure", {}).get("gross_exposure_avg_pct", metrics.get("gross_exposure_avg_pct", 0)),
        "avg_monthly_return_pct": sum(metrics.get("monthly_returns", {}).values()) / len(metrics.get("monthly_returns", {})) if metrics.get("monthly_returns") else 0,
        
        # Sparkline & Hierarchy
        "sparkline_data": downsample_to_avg(cumulative_pnl_data) if cumulative_pnl_data else [],
        "parent_id": None,
        
        # Deployment Counters
        "demo_count": 0,
        "live_count": 0,
    }
    

    training_payload = {
        "cagr": metrics.get("cagr_pct", _ra.get("cagr", 0)),
        "sortino_ratio": _ra.get("sortino_ratio", metrics.get("sortino_ratio", 0)),
        "calmar_ratio": _ra.get("calmar_ratio", metrics.get("calmar_ratio", 0)),
        "omega_ratio": _ra.get("omega_ratio", metrics.get("omega_ratio", 0)),
        "tail_ratio": metrics.get("risk_metrics", {}).get("tail_ratio", 0),
        "profit_factor": metrics.get("profit_loss", {}).get("profit_factor", 0),
        "expectancy": _ra.get("expected_return_pct", 0),
        "recovery_factor": metrics.get("drawdown_analysis", {}).get("recovery_factor", 0),
        "sqn": metrics.get("trade_analysis", {}).get("sqn", 0),
        "max_consecutive_losses": metrics.get("trade_analysis", {}).get("consecutive_losses", 0),
        "avg_trade_duration": metrics.get("trade_analysis", {}).get("avg_duration_trades", 0),
        "risk_of_ruin": metrics.get("risk_metrics", {}).get("risk_of_ruin", metrics.get("risk_of_ruin", 0)),
        "walk_forward": walk_forward,
        "monte_carlo": monte_carlo,
        "regimes_analysis": dict_regime_analysis,
        "holdout": hold_out,
    }

    configs_payload = {
        "data": data_config,
        "input": strategy_config,
        "backtest": backtest_config,
        "training": {
            "config"  : training_config,
            "holdout": hold_out,
            "walkforward": walk_forward,
            "monte_carlo": monte_carlo,
        },
    }

    try:
        # Sanitize numeric fields to prevent database overflow errors
        numeric_fields = {
        'total_return_pct', 'sharpe_ratio', 
        'max_drawdown_pct', 'win_rate_pct', 'sortino_ratio', 
        'calmar_ratio', 'profit_factor', 'regime_fitness_score'
        }
    
        metadata_payload = sanitize_fields(metadata_payload, numeric_fields)
        
        # Upsert metadata and get the auto-generated (or provided) strategy_id
        strategy_id = upsert_strategy_metadata(metadata_payload)
        upsert_strategy_training(strategy_id, training_payload)
        upsert_strategy_config(strategy_id, configs_payload)
        logger.info(f"Successfully saved strategy '{strategy_name}' with ID {strategy_id} to database")
    except Exception as exc:
        logger.exception(f"Failed to save strategy '{strategy_name}' to database")
        raise

    return strategy_id





