from bitpredict.common.db.models import ensure_regime_analysis_column_in_analytics_table
from bitpredict.common.db.exceptions import SchemaError, TableError
from bitpredict.common.db.utils import ensure_schema, ensure_table
from bitpredict.common.db.config import get_engine
from bitpredict.common.logging import get_logger
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Any
from sqlalchemy import text
import pandas as pd 
import numpy as np
import json
import math


logger = get_logger(__name__)

def upsert_into_regime_analysis_col(strategy_id: str,
                                    dict_data: Dict[str, Any],
                                    schema: str = 'simulator',
                                    table: str = 'analytics') -> None:
    """
    Upsert regime analysis data into the 'regime_analysis' column of the analytics table.
    Only interacts with the regime_analysis column, leaving all other columns unchanged.
    """

    def convert_to_serializable(obj):
        """
        Recursively convert pandas objects to JSON serializable format.
        Sanitizes Infinity and NaN to None, as PostgreSQL JSON does not support them.
        Preserves dictionary keys when converting DataFrames.
        """
        try:
            import numpy as np
            HAS_NUMPY = True
        except ImportError:
            HAS_NUMPY = False

        if isinstance(obj, pd.DataFrame):
            # Check if index has meaningful names (not just numeric range)
            if isinstance(obj.index, pd.RangeIndex):
                # Numeric index - use 'records' to avoid numeric string keys
                return convert_to_serializable(obj.to_dict(orient='records'))
            else:
                # Named index - preserve it
                return convert_to_serializable(obj.to_dict(orient='index'))
        elif isinstance(obj, pd.Series):
            return convert_to_serializable(obj.to_dict())
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif HAS_NUMPY:
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                v = float(obj)
                return None if (math.isnan(v) or math.isinf(v)) else v
            elif isinstance(obj, np.ndarray):
                return convert_to_serializable(obj.tolist())
        
        # Handle plain Python floats with Infinity / NaN
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj

        # Must come after numpy checks to avoid TypeError on numpy scalars
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass

        return obj

    engine = get_engine()

    # Ensure the column exists first
    ensure_regime_analysis_column_in_analytics_table(engine, schema, table)

    # Convert any DataFrames or non-serializable objects in dict_data
    serializable_data = convert_to_serializable(dict_data)

    # Convert to JSON string for storage
    json_data = json.dumps(serializable_data, default=str)

    with engine.connect() as conn:
        if engine.dialect.name == 'postgresql':
            upsert_sql = text(f"""
                INSERT INTO {schema}.{table} (strategy_id, regime_analysis)
                VALUES (:sid, CAST(:jdata AS jsonb))
                ON CONFLICT (strategy_id) DO UPDATE 
                SET regime_analysis = EXCLUDED.regime_analysis
            """)
            conn.execute(upsert_sql, {"sid": str(strategy_id), "jdata": json_data})

        elif engine.dialect.name == 'sqlite':
            check_sql = text(f"SELECT 1 FROM {schema}.{table} WHERE strategy_id = :sid")
            exists = conn.execute(check_sql, {"sid": str(strategy_id)}).fetchone()

            if exists:
                update_sql = text(f"UPDATE {schema}.{table} SET regime_analysis = :jdata WHERE strategy_id = :sid")
                conn.execute(update_sql, {"sid": str(strategy_id), "jdata": json_data})
            else:
                insert_sql = text(f"INSERT INTO {schema}.{table} (strategy_id, regime_analysis) VALUES (:sid, :jdata)")
                conn.execute(insert_sql, {"sid": str(strategy_id), "jdata": json_data})

        else:
            check_sql = text(f"SELECT 1 FROM {schema}.{table} WHERE strategy_id = :sid")
            exists = conn.execute(check_sql, {"sid": str(strategy_id)}).fetchone()

            if exists:
                update_sql = text(f"UPDATE {schema}.{table} SET regime_analysis = :jdata WHERE strategy_id = :sid")
                conn.execute(update_sql, {"sid": str(strategy_id), "jdata": json_data})
            else:
                insert_sql = text(f"INSERT INTO {schema}.{table} (strategy_id, regime_analysis) VALUES (:sid, :jdata)")
                conn.execute(insert_sql, {"sid": str(strategy_id), "jdata": json_data})

        conn.commit()
        logger.info(f"Successfully upserted regime_analysis data for strategy {strategy_id}")