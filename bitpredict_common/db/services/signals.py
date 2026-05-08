from datetime import datetime
from typing import Optional, Union, List, Dict, Any

import pandas as pd
import json
import psycopg2.extras

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from bitpredict.common.logging import get_logger
from bitpredict.common.db.config import get_engine
from bitpredict.common.db.utils import ensure_schema, read_df
from bitpredict.common.db.models import ensure_signal_table

logger = get_logger(__name__)

def read_signals(
    strategy_id: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read signals for a given strategy_id, optionally filtered by datetime range.
    """
    engine = get_engine()

    ensure_schema(engine, "strategies")
    ensure_signal_table(engine)

    where_parts = [f"strategy_id = '{strategy_id}'"]

    if start_date:
        where_parts.append(f"datetime >= '{start_date}'")
    if end_date:
        where_parts.append(f"datetime <= '{end_date}'")

    where_clause = " AND ".join(where_parts)

    df = read_df(
        engine=engine,
        schema="strategies",
        table_name="signals",
        columns=columns,
        where=where_clause,
        method="copy"  # or "sql"
    )

    if not df.empty:
        df = df.sort_values("datetime")

    if limit and not df.empty:
        df = df.head(limit)

    return df

def get_last_signal(strategy_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent signal for a given strategy_id.
    """
    engine = get_engine()

    ensure_schema(engine, "strategies")
    ensure_signal_table(engine)

    sql = """
        SELECT *
        FROM strategies.signals
        WHERE strategy_id = :strategy_id
        ORDER BY datetime DESC
        LIMIT 1
    """

    with engine.connect() as conn:
        result = conn.execute(text(sql), {"strategy_id": strategy_id}).fetchone()

    if result is None:
        return None

    return dict(result._mapping)

def upsert_signals(
    strategy_id: str,
    df_signals: pd.DataFrame,
    batch_size: int = 10000
) -> None:
    if df_signals is None or df_signals.empty:
        return

    try:
        engine = get_engine()
        ensure_schema(engine, "strategies")
        ensure_signal_table(engine)

        df = df_signals.copy().sort_values("datetime")
        component_cols = [c for c in df.columns if c not in ['datetime', 'signals']]

        # Vectorized JSON serialization — one orjson/json call per row via numpy
        if component_cols:
            # Build list of dicts from numpy arrays — no per-row Python overhead
            keys = component_cols
            arrays = [df[c].astype(int).to_numpy() for c in keys]
            components_list = [
                json.dumps(dict(zip(keys, (int(a[i]) for a in arrays))))
                for i in range(len(df))
            ]
        else:
            components_list = ['{}'] * len(df)

        datetimes = df['datetime'].to_numpy()
        signals   = df['signals'].astype(int).to_numpy()

        # Build tuples directly — no intermediate dict creation
        tuples = [
            (strategy_id, dt, comp, int(sig))
            for dt, comp, sig in zip(datetimes, components_list, signals)
        ]

        sql = """
            INSERT INTO strategies.signals (
                strategy_id, datetime, components, signals
            )
            VALUES %s
            ON CONFLICT (strategy_id, datetime)
            DO UPDATE SET
                components = EXCLUDED.components,
                signals    = EXCLUDED.signals,
                updated_at = NOW()
        """

        # Use a template that casts components to JSONB at DB level
        template = "(%s, %s, %s::jsonb, %s)"

        raw_conn = engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, sql, tuples,
                    template=template,
                    page_size=batch_size
                )
            raw_conn.commit()
        except Exception:
            raw_conn.rollback()
            raise
        finally:
            raw_conn.close()

        logger.info("Successfully upserted %d signals for strategy %s", len(tuples), strategy_id)

    except SQLAlchemyError as e:
        logger.error("DB error in upsert_signals | strategy_id=%s | error=%s", strategy_id, e)
        raise
    except Exception:
        logger.exception("Unexpected error in upsert_signals | strategy_id=%s", strategy_id)
        raise