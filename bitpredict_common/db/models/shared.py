from sqlalchemy import text, Engine
from sqlalchemy.exc import SQLAlchemyError
from bitpredict.common.db.exceptions import TableError
from bitpredict.common.db.utils import hypertable_exists, create_hypertable
from bitpredict.common.constants import SIMULATOR_SCHEMA
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)

def ensure_ledger_table(
    engine: Engine,
    schema_name: str = "simulator",
    table_name: str = "ledgers",
    is_timeseries: bool = True
) -> bool:

    """
    Ensure ledgers table exists under the given schema_name.
    Creates table and hypertable if needed.
    Includes updated_at column with auto-update.
    """
    if engine is None:
        raise TypeError("Database engine is None; cannot create table")
    
    # ---- CREATE TABLE ----
    try:
        with engine.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE {schema_name}.{table_name} (
                    id BIGINT NOT NULL,
                    trial_id BOOLEAN DEFAULT NULL,
                    entry_datetime TIMESTAMPTZ NOT NULL,
                    entry_fee_pct DOUBLE PRECISION,
                    avg_entry_price DOUBLE PRECISION,
                    exit_datetime TIMESTAMPTZ,
                    exit_fee_pct DOUBLE PRECISION,
                    avg_exit_price DOUBLE PRECISION,
                    position_size_pct DOUBLE PRECISION,
                    trade_return_pct DOUBLE PRECISION,
                    account_return_pct DOUBLE PRECISION,
                    cum_account_return DOUBLE PRECISION,
                    direction TEXT,
                    status TEXT,
                    action TEXT,
                    balance DOUBLE PRECISION,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (id, entry_datetime)
                )
            """))
            logger.debug(f"Created table {schema_name}.{table_name}")

    except Exception as e:
        if "duplicate key value" in str(e):
            logger.debug(f"Race condition: {schema_name}.{table_name} exists")
        else:
            logger.error(f"Failed to create table {schema_name}.{table_name}: {e}")
            raise

    # ---- CREATE HYPERTABLE ----
    if is_timeseries:
        if not hypertable_exists(engine, schema_name, table_name):
            try:
                create_hypertable(
                    engine=engine,
                    schema_name=schema_name,
                    table_name=table_name,
                    time_column="entry_datetime",
                    compress=True,
                    compress_segmentby="id"
                )
                logger.debug(f"Created hypertable {schema_name}.{table_name}")
            except Exception as e:
                error_msg = str(e).lower()
                if not any(x in error_msg for x in ["already", "duplicate"]):
                    logger.error(f"Failed hypertable creation: {e}")
                    raise

    # ---- CREATE INDEXES AFTER HYPERTABLE ----
    try:
        with engine.begin() as conn:
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{schema_name}_{table_name}_id_time
                ON {schema_name}.{table_name} (id, entry_datetime DESC);
            """))
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{schema_name}_{table_name}_time
                ON {schema_name}.{table_name} (entry_datetime DESC);
            """))
        logger.debug(f"Indexes created for {schema_name}.{table_name}")

    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")
        raise

    return True


def ensure_analytics_table(
    engine: Engine, 
    schema_name: str = SIMULATOR_SCHEMA,
    table_name: str = "analytics"
) -> None:
    """
    Ensure the analytics table exists.

    Columns:
        - id (BIGINT, PK): Strategy ID or Request ID
        - trial_id (BOOLEAN, DEFAULT NULL): Trial run flag
        - risk_adjusted (JSONB): Risk-adjusted performance metrics
        - risk_metrics (JSONB): Volatility, VaR, and other risk indicators
        - drawdown_analysis (JSONB): Drawdown statistics and curves
        - trade_analysis (JSONB): Trade-level statistics and summaries
        - profit_loss (JSONB): PnL breakdowns
        - long_short (JSONB): Long vs short performance
        - portfolio_values (JSONB): Portfolio equity curve
        - exposure (JSONB): Market exposure over time
        - cash_flow (JSONB): Cash movement tracking
        - time_series_analysis (JSONB): Time-based analytics
        - benchmark_analysis (JSONB): Benchmark comparison metrics
        - distribution_analysis (JSONB): Return distributions
        - drawdown_periods (JSONB): Individual drawdown periods
        - directional_metrics (JSONB): Direction-based metrics
        - regimes_analysis (JSONB, nullable): Market regime analysis
        - portfolio_state_path (VARCHAR): Path to serialized portfolio state
        - created_at (TIMESTAMPTZ): Creation timestamp
        - updated_at (TIMESTAMPTZ): Last update timestamp
    """    
    if engine is None:
        raise TypeError("Database engine is None; cannot create table")
    
    # No foreign key clause - removed entirely

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
        id BIGINT NOT NULL,
        trial_id BOOLEAN DEFAULT NULL,
        risk_adjusted JSONB,
        risk_metrics JSONB,
        drawdown_analysis JSONB,
        trade_analysis JSONB,
        profit_loss JSONB,
        long_short JSONB, 
        portfolio_values JSONB,
        exposure JSONB,
        cash_flow JSONB,
        time_series_analysis JSONB,
        benchmark_analysis JSONB,
        distribution_analysis JSONB,
        drawdown_periods JSONB,
        directional_metrics JSONB,
        regimes_analysis JSONB,
        portfolio_state_path VARCHAR(500),
        created_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        PRIMARY KEY (id)
    );

    CREATE INDEX IF NOT EXISTS idx_{schema_name}_{table_name}_updated_at
        ON {schema_name}.{table_name} (updated_at);
    
    CREATE INDEX IF NOT EXISTS idx_{schema_name}_{table_name}_trial_id
        ON {schema_name}.{table_name} (trial_id);

    CREATE INDEX IF NOT EXISTS idx_{schema_name}_{table_name}_id
        ON {schema_name}.{table_name} (id);
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info("Table %s.%s created successfully", schema_name, table_name)
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.%s", schema_name, table_name)
        raise TableError(f"Failed to create table {schema_name}.{table_name}") from exc
    

def ensure_graphs_table(
    engine: Engine, 
    schema_name: str = SIMULATOR_SCHEMA,
    table_name: str = "graphs", 
    is_timeseries: bool = True
) -> None:
    """
    Ensure the graphs table exists.

    Columns:
        - id: BIGINT (strategy_id or request_id)
        - trial_id: BOOLEAN (DEFAULT NULL)
        - datetime: TIMESTAMPTZ (the bucket, e.g., day)
        - mfe_pct: DOUBLE PRECISION
        - mae_pct: DOUBLE PRECISION
        - long_return_pct: DOUBLE PRECISION
        - short_return_pct: DOUBLE PRECISION
        - created_at: TIMESTAMPTZ NOT NULL DEFAULT now()
        - updated_at: TIMESTAMPTZ NOT NULL DEFAULT now()
        Primary key: (id, datetime)
    """
    
    if engine is None:
        raise TypeError("Database engine is None; cannot create table")
    
    # No foreign key clause - removed entirely
    
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
        id BIGINT NOT NULL,
        trial_id BOOLEAN DEFAULT NULL,
        "datetime" TIMESTAMPTZ NOT NULL,
        mfe_pct DOUBLE PRECISION,
        mae_pct DOUBLE PRECISION,
        long_return_pct DOUBLE PRECISION,
        short_return_pct DOUBLE PRECISION,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (id, "datetime")
    );
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
            
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_id_datetime
                ON {schema_name}.{table_name} (id, "datetime" DESC);
            """))
            
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_trial_id
                ON {schema_name}.{table_name} (trial_id);
            """))

        logger.debug("Table %s.%s created successfully", schema_name, table_name)
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.%s", schema_name, table_name)
        raise TableError(f"Failed to create table {schema_name}.{table_name}") from exc

    if is_timeseries:
        if hypertable_exists(engine, schema_name, table_name):
            logger.debug(f"Hypertable for {schema_name}.{table_name} already exists")
            return

        try:
            create_hypertable(
                engine=engine,
                schema_name=schema_name,
                table_name=table_name,
                time_column="datetime",
                compress=True,
                compress_segmentby="id"
            )
            logger.debug(f"Successfully created hypertable for {schema_name}.{table_name}")
        except Exception as e:
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in [
                "already a hypertable", "already exists", "duplicate key value"
            ]):
                if hypertable_exists(engine, schema_name, table_name):
                    logger.debug(f"Race condition creating hypertable for {schema_name}.{table_name}, exists now")
                    return
                else:
                    logger.warning(f"Could not create hypertable {schema_name}.{table_name}: {e}")
            else:
                logger.error(f"Failed to create hypertable {schema_name}.{table_name}: {e}")
                raise


def ensure_graphs_daily_table(
    engine: Engine, 
    schema_name: str = SIMULATOR_SCHEMA,
    table_name: str = "graphs_daily", 
    is_timeseries: bool = True
) -> None:
    """
    Ensure the graphs_daily table exists.

    Columns:
        - id: BIGINT (strategy_id or request_id)
        - trial_id: BOOLEAN (DEFAULT NULL)
        - datetime: TIMESTAMPTZ (the bucket, e.g., day)
        - daily_cumulative_return: DOUBLE PRECISION
        - drawdown_pct: DOUBLE PRECISION
        - benchmark_return_pct: DOUBLE PRECISION
        - rolling_sharpe: DOUBLE PRECISION
        - rolling_sortino: DOUBLE PRECISION
        - rolling_correlation: DOUBLE PRECISION
        - created_at: TIMESTAMPTZ NOT NULL DEFAULT now()
        - updated_at: TIMESTAMPTZ NOT NULL DEFAULT now()
        Primary key: (id, datetime)
    """
    
    if engine is None:
        raise TypeError("Database engine is None; cannot create table")
    
    # No foreign key clause - removed entirely
    
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
        id BIGINT NOT NULL,
        trial_id BOOLEAN DEFAULT NULL,
        "datetime" TIMESTAMPTZ NOT NULL,
        daily_cumulative_return DOUBLE PRECISION,
        drawdown_pct DOUBLE PRECISION,
        benchmark_return_pct DOUBLE PRECISION,
        rolling_sharpe DOUBLE PRECISION,
        rolling_sortino DOUBLE PRECISION,
        rolling_correlation DOUBLE PRECISION,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (id, "datetime")
    );
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))

            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_id_datetime
                ON {schema_name}.{table_name} (id, "datetime" DESC);
            """))

            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_trial_id
                ON {schema_name}.{table_name} (trial_id);
            """))

        logger.debug("Table %s.%s created successfully", schema_name, table_name)
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.%s", schema_name, table_name)
        raise TableError(f"Failed to create table {schema_name}.{table_name}") from exc

    if is_timeseries:
        if hypertable_exists(engine, schema_name, table_name):
            logger.debug(f"Hypertable for {schema_name}.{table_name} already exists")
            return

        try:
            create_hypertable(
                engine=engine,
                schema_name=schema_name,
                table_name=table_name,
                time_column="datetime",
                compress=True,
                compress_segmentby="id"
            )
            logger.debug(f"Successfully created hypertable for {schema_name}.{table_name}")
        except Exception as e:
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in [
                "already a hypertable", "already exists", "duplicate key value"
            ]):
                if hypertable_exists(engine, schema_name, table_name):
                    logger.debug(f"Race condition creating hypertable for {schema_name}.{table_name}, exists now")
                    return
                else:
                    logger.warning(f"Could not create hypertable {schema_name}.{table_name}: {e}")
            else:
                logger.error(f"Failed to create hypertable {schema_name}.{table_name}: {e}")
                raise


def ensure_requests_table(engine: Engine, schema_name: str = "backtest") -> None:
    """
    Ensure the backtest.requests table exists.

    Columns:
        - id (BIGSERIAL, PK)
        - trial_id (BOOLEAN, DEFAULT NULL)
        - user_id (TEXT, NOT NULL)
        - user_email (TEXT, nullable)
        - service_type (TEXT, NOT NULL, default 'backtest')
        - strategy_id (BIGINT, nullable)
        - backtest_config (JSONB, nullable)
        - strategy_config (JSONB, nullable)
        - priority (INTEGER, NOT NULL, default 0)
        - status (TEXT, NOT NULL, default 'pending')
        - error_message (TEXT, nullable)
        - started_at (TIMESTAMPTZ, nullable)
        - completed_at (TIMESTAMPTZ, nullable)
        - created_at (TIMESTAMPTZ, NOT NULL, default now())
        - updated_at (TIMESTAMPTZ, NOT NULL, default now())
    """
    if engine is None:
        raise TypeError("Database engine is None; cannot create table")

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.requests (
        id BIGSERIAL PRIMARY KEY,
        trial_id BOOLEAN DEFAULT NULL,
        user_id TEXT NOT NULL,
        user_email TEXT,
        service_type TEXT NOT NULL DEFAULT 'backtest',
        strategy_id BIGINT,
        backtest_config JSONB,
        strategy_config JSONB,
        priority INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending',
        error_message TEXT,
        started_at TIMESTAMPTZ,
        completed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_{schema_name}_requests_user_id ON {schema_name}.requests (user_id);
    CREATE INDEX IF NOT EXISTS idx_{schema_name}_requests_strategy_id ON {schema_name}.requests (strategy_id);
    CREATE INDEX IF NOT EXISTS idx_{schema_name}_requests_status ON {schema_name}.requests (status);
    CREATE INDEX IF NOT EXISTS idx_{schema_name}_requests_priority ON {schema_name}.requests (priority);
    CREATE INDEX IF NOT EXISTS idx_{schema_name}_requests_created_at ON {schema_name}.requests (created_at);
    CREATE INDEX IF NOT EXISTS idx_{schema_name}_requests_updated_at ON {schema_name}.requests (updated_at);
    CREATE INDEX IF NOT EXISTS idx_{schema_name}_requests_trial_id ON {schema_name}.requests (trial_id);
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info("Table %s.requests created successfully", schema_name)
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.requests", schema_name)
        raise TableError(f"Failed to create table {schema_name}.requests") from exc