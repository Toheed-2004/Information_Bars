"""
Database configuration and engine management for BitPredict framework.

This module provides centralized database connection and engine management
with connection pooling, health checks, and environment-driven configuration.

Example:
    Basic usage with default configuration::

        from common.db.config import get_engine, test_connection

        engine = get_engine()
        if test_connection(engine):
            # Use engine for database operations
            pass

    Custom pool configuration::

        engine = get_engine(
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            echo=False
        )

    Multiple database profiles::

        # Production database
        prod_engine = get_engine(db_url=PROD_DB_URL)

        # Analytics database
        analytics_engine = get_engine(db_url=ANALYTICS_DB_URL)

Note:
    - All functions are safe to import (no side effects)
    - Engines are cached per connection string
    - Connection pooling is enabled by default
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine, Engine, text, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from dotenv import load_dotenv
from bitpredict.common.logging import get_logger
from .exceptions import *
from sqlalchemy import text

# Load the .env file from project root (adjust path as needed)
dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)  # override existing env vars

# Initialize module logger
logger = get_logger(__name__)

# Engine cache to prevent duplicate engine creation
_engine_cache: Dict[str, Engine] = {}


def get_db_url() -> str:
    """
    Retrieve database URL from environment variables.

    Constructs a database URL from individual environment variables or
    returns a complete DATABASE_URL if provided.

    Returns:
        str: Database connection URL in SQLAlchemy format.

    Raises:
        ValueError: If required environment variables are missing.

    Note:
        Expected environment variables:
        - DATABASE_URL (complete URL) OR
        - DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD (components)
    """
    # Check for complete DATABASE_URL first
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        logger.debug("Using DATABASE_URL from environment")
        return db_url

    # Build from components
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_driver = os.getenv("DB_DRIVER", "postgresql")

    # Validate required components
    if not all([db_host, db_name, db_user, db_password]):
        missing = []
        if not db_host:
            missing.append("DB_HOST")
        if not db_name:
            missing.append("DB_NAME")
        if not db_user:
            missing.append("DB_USER")
        if not db_password:
            missing.append("DB_PASSWORD")

        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Please set DATABASE_URL or individual DB_* variables in .env file."
        )

    # Construct URL
    db_url = f"{db_driver}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    logger.debug(
        f"Constructed database URL for {db_driver}://{db_host}:{db_port}/{db_name}"
    )

    return db_url


def get_engine(
    db_url: Optional[str] = None,
    pool_size: Optional[int] = None,
    max_overflow: Optional[int] = None,
    pool_pre_ping: bool = True,
    pool_recycle: int = 3600,
    echo: bool = False,
    **kwargs: Any,
) -> Engine:
    """
    Create or retrieve a cached SQLAlchemy engine with connection pooling.

    Engines are cached per unique connection string to prevent duplicate
    connection pools. All configuration follows best practices for production
    database connections.

    Args:
        db_url: Database connection URL. If None, loads from environment.
        pool_size: Number of connections to maintain in the pool.
            Defaults to DB_POOL_SIZE env var or 5.
        max_overflow: Maximum additional connections beyond pool_size.
            Defaults to DB_MAX_OVERFLOW env var or 10.
        pool_pre_ping: Test connections before using them.
            Recommended for production to handle stale connections.
        pool_recycle: Recycle connections after N seconds.
            Prevents issues with database timeout policies.
        echo: Log all SQL statements. Use for debugging only.
        **kwargs: Additional arguments passed to create_engine.

    Returns:
        Engine: Configured SQLAlchemy engine with connection pooling.

    Raises:
        ValueError: If database configuration is invalid.
        SQLAlchemyError: If engine creation fails.

    Example:
        >>> engine = get_engine(pool_size=20, max_overflow=40)
        >>> # Engine is cached and reused for subsequent calls
        >>> same_engine = get_engine()
        >>> assert engine is same_engine
    """

    # Get database URL
    if db_url is None:
        db_url = get_db_url()

    # Check cache
    if db_url in _engine_cache:
        logger.debug(f"Returning cached engine for database")
        return _engine_cache[db_url]

    # Get pool configuration from environment or use defaults
    if pool_size is None:
        pool_size = int(os.getenv("DB_POOL_SIZE", "1"))

    if max_overflow is None:
        max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))

    # Validate pool configuration
    if pool_size < 1:
        raise ValueError(f"pool_size must be at least 1, got {pool_size}")

    if max_overflow < 0:
        raise ValueError(f"max_overflow must be non-negative, got {max_overflow}")

    try:
        # Create engine with connection pooling
        engine = create_engine(
            db_url,
            poolclass=pool.QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,
            pool_recycle=pool_recycle,
            connect_args={"options": "-c timezone=UTC"},
            echo=echo,
            **kwargs,
        )

        logger.info(
            f"Created database engine with pool_size={pool_size}, "
            f"max_overflow={max_overflow}, pool_pre_ping={pool_pre_ping}"
        )

        # Cache the engine
        _engine_cache[db_url] = engine

        return engine

    except SQLAlchemyError as e:
        logger.error(f"Failed to create database engine: {e}")
        raise


def check_connection(engine: Engine, timeout: float = 5.0) -> bool:
    """
    Check database connectivity by executing a simple query.

    Performs a lightweight query to verify the database connection is working.
    This is useful for health checks and startup validation.

    Args:
        engine: SQLAlchemy engine to test.
        timeout: Maximum time in seconds to wait for connection.

    Returns:
        bool: True if connection successful.

    Raises:
        OperationalError: If connection fails or times out.
        SQLAlchemyError: If query execution fails.

    Example:
        >>> engine = get_engine()
        >>> if check_connection(engine):
        ...     print("Database is accessible")
    """
    try:
        with engine.connect() as conn:
            # Set statement timeout for PostgreSQL
            if "postgresql" in engine.dialect.name:
                conn.execute(text(f"SET statement_timeout = {int(timeout * 1000)}"))

            # Execute simple query
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        logger.info("Database connection test successful")
        return True

    except OperationalError as e:
        logger.error(f"Database connection test failed (operational error): {e}")
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database connection test failed: {e}")
        raise


def get_session_maker(
    engine: Engine,
    autocommit: bool = False,
    autoflush: bool = False,
    expire_on_commit: bool = False,
) -> sessionmaker:
    """
    Create a configured session factory for ORM operations.

    Returns a sessionmaker that can be used to create individual sessions.
    The default configuration is suitable for most use cases.

    Args:
        engine: SQLAlchemy engine to bind sessions to.
        autocommit: Enable autocommit mode (not recommended).
        autoflush: Automatically flush before queries.
        expire_on_commit: Expire all instances after commit.

    Returns:
        sessionmaker: Factory for creating database sessions.

    Example:
        >>> engine = get_engine()
        >>> SessionMaker = get_session_maker(engine)
        >>> session = SessionMaker()
        >>> try:
        ...     # Use session for ORM operations
        ...     session.commit()
        ... finally:
        ...     session.close()
    """
    return sessionmaker(
        bind=engine,
        autocommit=autocommit,
        autoflush=autoflush,
        expire_on_commit=expire_on_commit,
    )


@contextmanager
def get_session(engine: Optional[Engine] = None, commit_on_success: bool = True):
    """
    Context manager for database sessions with automatic cleanup.

    Provides a session that automatically commits on success and rolls back
    on exceptions. Always closes the session when exiting the context.

    Args:
        engine: SQLAlchemy engine. If None, creates one from environment.
        commit_on_success: Automatically commit if no exceptions occur.

    Yields:
        Session: Active database session.

    Raises:
        Any exception raised within the context is re-raised after rollback.

    Example:
        >>> with get_session() as session:
        ...     result = session.execute(text("SELECT * FROM users"))
        ...     # Automatically commits on successful completion

        >>> with get_session(commit_on_success=False) as session:
        ...     result = session.execute(text("SELECT * FROM users"))
        ...     session.commit()  # Manual commit
    """
    if engine is None:
        engine = get_engine()

    SessionMaker = get_session_maker(engine)
    session: Session = SessionMaker()

    try:
        yield session

        if commit_on_success:
            session.commit()
            logger.debug("Session committed successfully")

    except Exception as e:
        session.rollback()
        logger.warning(f"Session rolled back due to exception: {e}")
        raise

    finally:
        session.close()
        logger.debug("Session closed")


def dispose_engine(
    engine: Optional[Engine] = None, db_url: Optional[str] = None
) -> None:
    """
    Dispose of an engine and remove it from cache.

    Closes all connections in the pool and removes the engine from the cache.
    Useful for cleanup in testing or when switching database configurations.

    Args:
        engine: Engine to dispose. If None, uses db_url to find cached engine.
        db_url: Database URL of cached engine to dispose.

    Raises:
        ValueError: If neither engine nor db_url is provided.

    Example:
        >>> engine = get_engine()
        >>> # ... use engine ...
        >>> dispose_engine(engine)
        >>> # Engine is disposed and removed from cache
    """
    if engine is None and db_url is None:
        raise ValueError("Either engine or db_url must be provided")

    if engine is None:
        if db_url not in _engine_cache:
            logger.warning(f"No cached engine found for URL")
            return
        engine = _engine_cache[db_url]

    # Find and remove from cache
    cache_key = None
    for key, cached_engine in _engine_cache.items():
        if cached_engine is engine:
            cache_key = key
            break

    if cache_key:
        del _engine_cache[cache_key]
        logger.debug(f"Removed engine from cache")

    # Dispose engine
    engine.dispose()
    logger.info("Engine disposed and connections closed")


def clear_engine_cache() -> None:
    """
    Dispose all cached engines and clear the cache.

    Useful for cleanup in testing or application shutdown.

    Example:
        >>> # At application shutdown
        >>> clear_engine_cache()
    """
    logger.info(f"Disposing {len(_engine_cache)} cached engines")

    for engine in _engine_cache.values():
        try:
            engine.dispose()
        except Exception as e:
            logger.error(f"Error disposing engine: {e}")

    _engine_cache.clear()
    logger.info("Engine cache cleared")


def get_engine_info(engine: Engine) -> Dict[str, Any]:
    """
    Get diagnostic information about an engine.

    Returns pool statistics and configuration details useful for monitoring
    and debugging.

    Args:
        engine: Engine to inspect.

    Returns:
        dict: Engine configuration and pool statistics.

    Example:
        >>> engine = get_engine()
        >>> info = get_engine_info(engine)
        >>> print(f"Pool size: {info['pool_size']}")
        >>> print(f"Checked out connections: {info['checked_out']}")
    """
    info = {
        "driver": engine.dialect.name,
        "pool_size": engine.pool.size() if hasattr(engine.pool, "size") else None,
        "checked_out": (
            engine.pool.checkedout() if hasattr(engine.pool, "checkedout") else None
        ),
        "overflow": (
            engine.pool.overflow() if hasattr(engine.pool, "overflow") else None
        ),
        "echo": engine.echo,
    }

    return info
