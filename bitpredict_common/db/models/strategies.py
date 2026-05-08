"""
Database setup for strategies schema.

Ensures all tables exist with the correct column definitions and indexes.
"""

from sqlalchemy.engine import Engine
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from bitpredict.common.db.utils import ensure_schema, ensure_table
from bitpredict.common.constants import STRATEGIES_SCHEMA
from bitpredict.common.db.exceptions import SchemaError, TableError
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# METADATA TABLE
# ---------------------------------------------------------------------------
def ensure_strategies_metadata_table(engine: Engine) -> None:
    """
    Ensure the strategies.metadata table exists.
    """

    if engine is None:
        raise TypeError("Database engine is None; cannot create table")

    logger.debug("Ensuring schema '%s' exists", STRATEGIES_SCHEMA)

    try:
        if ensure_table(engine, STRATEGIES_SCHEMA, "metadata"):
            logger.debug("Table %s.metadata already exists", STRATEGIES_SCHEMA)
            return
        ensure_schema(engine, STRATEGIES_SCHEMA)
    except Exception as exc:
        logger.error("Failed to ensure schema '%s' exists", STRATEGIES_SCHEMA)
        raise SchemaError(
            f"Failed to create or verify schema {STRATEGIES_SCHEMA}"
        ) from exc

    ddl = f"""
    CREATE TABLE {STRATEGIES_SCHEMA}.metadata (
        id BIGSERIAL PRIMARY KEY,

        -- User & Ownership
        user_id BIGINT NULL,  -- No foreign key constraint, just a regular BIGINT
        owner_type VARCHAR(20) NOT NULL DEFAULT 'platform',
        display_name VARCHAR(150),
        code VARCHAR(16) UNIQUE NOT NULL,

        -- Strategy Classification
        strategy_type VARCHAR(50) NOT NULL,
        mtf BOOLEAN NOT NULL DEFAULT false,
        exchange VARCHAR(50) NOT NULL,
        symbol VARCHAR(50) NOT NULL,

        -- Trading Parameters
        bar_type VARCHAR(50) NOT NULL,
        timeframe VARCHAR(20),
        direction_bias VARCHAR(20),
        trade_frequency VARCHAR(20),

        -- Validation & Regime
        best_regime VARCHAR(50),
        regime_fitness_score NUMERIC(10, 4),
        rolling_status VARCHAR(20),

        -- Status Flags
        public BOOLEAN NOT NULL DEFAULT false,
        simulator BOOLEAN NOT NULL DEFAULT true,
        access_level VARCHAR(20) NOT NULL DEFAULT 'admin_only',

        -- Tags & Metadata
        tags TEXT[],
        version VARCHAR(50) DEFAULT NULL,
        description TEXT,

        -- Performance Metrics (from simulator.analytics)
        total_return_pct NUMERIC(10, 4),
        sharpe_ratio NUMERIC(10, 4),
        max_drawdown_pct NUMERIC(10, 4),
        win_rate_pct NUMERIC(10, 4),
        sortino_ratio NUMERIC(10, 4),
        calmar_ratio NUMERIC(10, 4),
        profit_factor NUMERIC(10, 4),
        total_trades INTEGER,
        cagr NUMERIC(10, 4),
        expectancy NUMERIC(10, 4),
        avg_trade_return_pct NUMERIC(10, 4),
        recovery_factor NUMERIC(10, 4),
        sqn NUMERIC(10, 4),
        exposure_pct NUMERIC(10, 4),
        avg_monthly_return_pct NUMERIC(10, 4),

        -- Sparkline & Hierarchy
        sparkline_data FLOAT[],
        parent_id BIGINT NULL,

        -- Deployment Counters
        demo_count SMALLINT,
        live_count SMALLINT,

        -- Timestamps
        created_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),

        -- Constraints
        CONSTRAINT chk_bar_type_timeframe
        CHECK (
            (bar_type = 'time' AND timeframe IS NOT NULL)
            OR
            (bar_type <> 'time' AND timeframe IS NULL)
        ),

        CONSTRAINT chk_owner_type
        CHECK (owner_type IN ('platform', 'user')),

        CONSTRAINT chk_direction_bias
        CHECK (direction_bias IN ('long_only', 'short_only', 'both')),

        CONSTRAINT chk_rolling_status
        CHECK (rolling_status IN ('healthy', 'deteriorating', 'poor_win_rate', 'critical')),

        -- Foreign Keys (only parent_id has FK constraint now)
        FOREIGN KEY (parent_id) REFERENCES {STRATEGIES_SCHEMA}.metadata(id) ON DELETE SET NULL
    );

    -- Indexes
    CREATE INDEX idx_strategies_metadata_user_id
        ON {STRATEGIES_SCHEMA}.metadata (user_id);

    CREATE INDEX idx_strategies_metadata_owner_type
        ON {STRATEGIES_SCHEMA}.metadata (owner_type);


    CREATE INDEX idx_strategies_metadata_code
        ON {STRATEGIES_SCHEMA}.metadata (code);

    CREATE INDEX idx_strategies_metadata_mtf
        ON {STRATEGIES_SCHEMA}.metadata (mtf);
        
    CREATE INDEX idx_strategies_metadata_strategy_type
        ON {STRATEGIES_SCHEMA}.metadata (strategy_type);

    CREATE INDEX idx_strategies_metadata_exchange
        ON {STRATEGIES_SCHEMA}.metadata (exchange);

    CREATE INDEX idx_strategies_metadata_symbol
        ON {STRATEGIES_SCHEMA}.metadata (symbol);

    CREATE INDEX idx_strategies_metadata_public
        ON {STRATEGIES_SCHEMA}.metadata (public);

    CREATE INDEX idx_strategies_metadata_access_level
        ON {STRATEGIES_SCHEMA}.metadata (access_level);

    CREATE INDEX idx_strategies_metadata_direction_bias
        ON {STRATEGIES_SCHEMA}.metadata (direction_bias);

    CREATE INDEX idx_strategies_metadata_trade_frequency
        ON {STRATEGIES_SCHEMA}.metadata (trade_frequency);

    CREATE INDEX idx_strategies_metadata_rolling_status
        ON {STRATEGIES_SCHEMA}.metadata (rolling_status);

    CREATE INDEX idx_strategies_metadata_parent_id
        ON {STRATEGIES_SCHEMA}.metadata (parent_id);

    -- Performance metrics indexes
    CREATE INDEX idx_strategies_metadata_total_return
        ON {STRATEGIES_SCHEMA}.metadata (total_return_pct);

    CREATE INDEX idx_strategies_metadata_sharpe
        ON {STRATEGIES_SCHEMA}.metadata (sharpe_ratio);

    CREATE INDEX idx_strategies_metadata_drawdown
        ON {STRATEGIES_SCHEMA}.metadata (max_drawdown_pct);

    CREATE INDEX idx_strategies_metadata_win_rate
        ON {STRATEGIES_SCHEMA}.metadata (win_rate_pct);

    CREATE INDEX idx_strategies_metadata_sortino
        ON {STRATEGIES_SCHEMA}.metadata (sortino_ratio);

    CREATE INDEX idx_strategies_metadata_calmar
        ON {STRATEGIES_SCHEMA}.metadata (calmar_ratio);

    CREATE INDEX idx_strategies_metadata_profit_factor
        ON {STRATEGIES_SCHEMA}.metadata (profit_factor);

    CREATE INDEX idx_strategies_metadata_total_trades
        ON {STRATEGIES_SCHEMA}.metadata (total_trades);

    CREATE INDEX idx_strategies_metadata_cagr
        ON {STRATEGIES_SCHEMA}.metadata (cagr);

    CREATE INDEX idx_strategies_metadata_sqn
        ON {STRATEGIES_SCHEMA}.metadata (sqn);

    CREATE INDEX idx_strategies_metadata_regime_fitness
        ON {STRATEGIES_SCHEMA}.metadata (regime_fitness_score);

    CREATE INDEX idx_strategies_metadata_version
        ON {STRATEGIES_SCHEMA}.metadata (version);

    CREATE INDEX idx_strategies_metadata_tags
        ON {STRATEGIES_SCHEMA}.metadata USING GIN (tags);

    CREATE INDEX idx_strategies_metadata_created_at
        ON {STRATEGIES_SCHEMA}.metadata (created_at);

    CREATE INDEX idx_strategies_metadata_updated_at
        ON {STRATEGIES_SCHEMA}.metadata (updated_at);

    CREATE INDEX idx_strategies_metadata_bar_type
        ON {STRATEGIES_SCHEMA}.metadata (bar_type);

    CREATE INDEX idx_strategies_metadata_timeframe
        ON {STRATEGIES_SCHEMA}.metadata (timeframe);

    CREATE INDEX idx_strategies_metadata_best_regime
        ON {STRATEGIES_SCHEMA}.metadata (best_regime);

    CREATE INDEX idx_strategies_metadata_simulator
        ON {STRATEGIES_SCHEMA}.metadata (simulator);

    CREATE INDEX idx_strategies_metadata_expectancy
        ON {STRATEGIES_SCHEMA}.metadata (expectancy);

    CREATE INDEX idx_strategies_metadata_recovery
        ON {STRATEGIES_SCHEMA}.metadata (recovery_factor);

    CREATE INDEX idx_strategies_metadata_display_name
        ON {STRATEGIES_SCHEMA}.metadata (display_name);

    CREATE INDEX idx_strategies_metadata_description
        ON {STRATEGIES_SCHEMA}.metadata (description);

    CREATE INDEX idx_strategies_metadata_avg_trade_return
        ON {STRATEGIES_SCHEMA}.metadata (avg_trade_return_pct);

    CREATE INDEX idx_strategies_metadata_exposure
        ON {STRATEGIES_SCHEMA}.metadata (exposure_pct);

    CREATE INDEX idx_strategies_metadata_avg_monthly_return
        ON {STRATEGIES_SCHEMA}.metadata (avg_monthly_return_pct);

    CREATE INDEX idx_strategies_metadata_sparkline
        ON {STRATEGIES_SCHEMA}.metadata USING GIN (sparkline_data);

    CREATE INDEX idx_strategies_metadata_demo_count
        ON {STRATEGIES_SCHEMA}.metadata (demo_count);

    CREATE INDEX idx_strategies_metadata_live_count
        ON {STRATEGIES_SCHEMA}.metadata (live_count);
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))

        logger.info("Table %s.metadata created successfully", STRATEGIES_SCHEMA)

    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.metadata", STRATEGIES_SCHEMA)
        raise TableError(
            f"Failed to create table {STRATEGIES_SCHEMA}.metadata"
        ) from exc


# ---------------------------------------------------------------------------
# TRAINING TABLE (strategy_id as PRIMARY KEY)
# ---------------------------------------------------------------------------
def ensure_strategies_training_table(engine: Engine) -> None:
    """
    Ensure the strategies.training table exists.

    Columns:
        - strategy_id: BIGINT PRIMARY KEY REFERENCES metadata(id) ON DELETE CASCADE
        - cagr, sortino_ratio, calmar_ratio, omega_ratio, tail_ratio,
          profit_factor, expectancy, recovery_factor, sqn,
          max_consecutive_losses, avg_trade_duration, risk_of_ruin
        - walk_forward: JSONB
        - holdout: JSONB
        - regimes_analysis: JSONB (nullable)
        - created_at: TIMESTAMPTZ NOT NULL DEFAULT now()
        - updated_at: TIMESTAMPTZ NOT NULL DEFAULT now()
    """
    if engine is None:
        raise TypeError("Database engine is None; cannot create table")
    
    if ensure_table(engine, STRATEGIES_SCHEMA, "training"):
        logger.debug("Table %s.training already exists", STRATEGIES_SCHEMA)
        return
    
    logger.debug("Ensuring schema '%s' exists", STRATEGIES_SCHEMA)
    try:
        ensure_schema(engine, STRATEGIES_SCHEMA)
    except Exception as exc:
        raise SchemaError(f"Failed to create or verify schema {STRATEGIES_SCHEMA}") from exc

    try:
        if ensure_table(engine, STRATEGIES_SCHEMA, "training"):
            logger.debug("Table %s.training already exists", STRATEGIES_SCHEMA)
            return
    except Exception as exc:
        logger.error("Failed to check if table %s.training exists", STRATEGIES_SCHEMA)
        raise TableError(f"Failed to verify table existence for {STRATEGIES_SCHEMA}.training") from exc

    ddl = f"""
    CREATE TABLE {STRATEGIES_SCHEMA}.training (
        strategy_id BIGINT PRIMARY KEY,
        cagr NUMERIC(10, 4),
        sortino_ratio NUMERIC(10, 4),
        calmar_ratio NUMERIC(10, 4),
        omega_ratio NUMERIC(10, 4),
        tail_ratio NUMERIC(10, 4),
        profit_factor NUMERIC(10, 4),
        expectancy NUMERIC(10, 4),
        recovery_factor NUMERIC(10, 4),
        sqn NUMERIC(10, 4),
        max_consecutive_losses NUMERIC(10, 4),
        avg_trade_duration NUMERIC(10, 4),
        risk_of_ruin NUMERIC(10, 6),
        walk_forward JSONB,
        monte_carlo JSONB,
        holdout JSONB,
        regimes_analysis JSONB,
        created_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        CONSTRAINT fk_training_strategy
            FOREIGN KEY (strategy_id)
            REFERENCES {STRATEGIES_SCHEMA}.metadata(id)
            ON DELETE CASCADE
    );

    CREATE INDEX idx_strategies_training_created_at ON {STRATEGIES_SCHEMA}.training (created_at);
    CREATE INDEX idx_strategies_training_updated_at ON {STRATEGIES_SCHEMA}.training (updated_at);
    CREATE INDEX idx_strategies_training_walk_forward
        ON {STRATEGIES_SCHEMA}.training USING GIN (walk_forward);
    CREATE INDEX idx_strategies_training_monte_carlo
        ON {STRATEGIES_SCHEMA}.training USING GIN (monte_carlo);
    CREATE INDEX idx_strategies_training_holdout
        ON {STRATEGIES_SCHEMA}.training USING GIN (holdout);
    CREATE INDEX idx_strategies_training_regimes_analysis
        ON {STRATEGIES_SCHEMA}.training USING GIN (regimes_analysis);
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info("Table %s.training created successfully", STRATEGIES_SCHEMA)
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.training", STRATEGIES_SCHEMA)
        raise TableError(f"Failed to create table {STRATEGIES_SCHEMA}.training") from exc


# ---------------------------------------------------------------------------
# CONFIGS TABLE (strategy_id as PRIMARY KEY)
# ---------------------------------------------------------------------------
def ensure_strategies_configs_table(engine: Engine) -> None:
    """
    Ensure the strategies.configs table exists.

    Columns:
        - strategy_id: BIGINT PRIMARY KEY REFERENCES metadata(id) ON DELETE CASCADE
        - data: JSONB
        - input: JSONB
        - backtest: JSONB
        - training: JSONB
        - model_path: VARCHAR(500) nullable
        - created_at: TIMESTAMPTZ NOT NULL DEFAULT now()
        - updated_at: TIMESTAMPTZ NOT NULL DEFAULT now()
    """
    if engine is None:
        raise TypeError("Database engine is None; cannot create table")
    if ensure_table(engine, STRATEGIES_SCHEMA, "configs"):
        logger.debug("Table %s.configs already exists", STRATEGIES_SCHEMA)
        return
    logger.debug("Ensuring schema '%s' exists", STRATEGIES_SCHEMA)
    try:
        ensure_schema(engine, STRATEGIES_SCHEMA)
    except Exception as exc:
        raise SchemaError(f"Failed to create or verify schema {STRATEGIES_SCHEMA}") from exc

    try:
        if ensure_table(engine, STRATEGIES_SCHEMA, "configs"):
            logger.debug("Table %s.configs already exists", STRATEGIES_SCHEMA)
            return
    except Exception as exc:
        raise TableError(f"Failed to verify table existence for {STRATEGIES_SCHEMA}.configs") from exc

    ddl = f"""
    CREATE TABLE {STRATEGIES_SCHEMA}.configs (
        strategy_id BIGINT PRIMARY KEY,
        data JSONB,
        input JSONB,
        backtest JSONB,
        training JSONB,
        model_path VARCHAR(500),
        created_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT now(),
        CONSTRAINT fk_configs_strategy
            FOREIGN KEY (strategy_id)
            REFERENCES {STRATEGIES_SCHEMA}.metadata(id)
            ON DELETE CASCADE
    );

    CREATE INDEX idx_strategies_configs_input
        ON {STRATEGIES_SCHEMA}.configs USING GIN (input);
    CREATE INDEX idx_strategies_configs_data
        ON {STRATEGIES_SCHEMA}.configs USING GIN (data);
    CREATE INDEX idx_strategies_configs_backtest
        ON {STRATEGIES_SCHEMA}.configs USING GIN (backtest);
    CREATE INDEX idx_strategies_configs_training
        ON {STRATEGIES_SCHEMA}.configs USING GIN (training);
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info("Table %s.configs created successfully", STRATEGIES_SCHEMA)
    except SQLAlchemyError as exc:
        logger.exception("Failed to create table %s.configs", STRATEGIES_SCHEMA)
        raise TableError(f"Failed to create table {STRATEGIES_SCHEMA}.configs") from exc





