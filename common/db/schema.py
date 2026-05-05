from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from .exceptions import *


def drop_schema(
    engine: Engine,
    schema: str,
    cascade: bool = False,
    if_exists: bool = True,
) -> None:
    """
    Drop a PostgreSQL schema.

    This function removes an entire schema from the database. Optionally,
    it can cascade the drop operation to remove all dependent objects
    (tables, views, sequences, functions) within the schema.

    The operation is executed inside a transaction to ensure atomicity.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        An active SQLAlchemy Engine connected to the target PostgreSQL database.

    schema : str
        Name of the schema to be dropped.
        Must be a non-empty string.

    cascade : bool, optional (default=False)
        If True, drops the schema along with all dependent objects
        using `DROP SCHEMA ... CASCADE`.
        If False, PostgreSQL will raise an error if the schema is not empty.

    if_exists : bool, optional (default=True)
        If True, uses `DROP SCHEMA IF EXISTS` to avoid errors when the schema
        does not exist.
        If False, PostgreSQL will raise an error if the schema is missing.

    Returns
    -------
    None
        This function does not return a value.
        Successful execution means the schema was dropped (or did not exist
        when `if_exists=True`).

    Raises
    ------
    SchemaError
        - If `schema` is empty or invalid.
        - If the DROP SCHEMA operation fails due to database errors.
        - If the schema does not exist and `if_exists=False`.

    Notes
    -----
    - This is a **destructive operation**. Use with extreme caution,
      especially when `cascade=True`.
    - Common production use cases:
        * Dropping temporary or test schemas
        * Resetting metadata schemas during development
    - This function intentionally does NOT check for schema existence
      beforehand; PostgreSQL handles existence checks efficiently.
    """

    # ------------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------------
    if not schema:
        raise SchemaError("Schema name cannot be empty")

    # ------------------------------------------------------------------------
    # Build DROP SCHEMA statement safely
    # ------------------------------------------------------------------------
    drop_sql = f"DROP SCHEMA {'IF EXISTS ' if if_exists else ''}{schema}"
    if cascade:
        drop_sql += " CASCADE"

    try:
        # --------------------------------------------------------------------
        # Execute schema drop atomically
        # --------------------------------------------------------------------
        with engine.begin() as conn:
            conn.execute(text(drop_sql))
    except SQLAlchemyError as exc:
        # Wrap SQLAlchemy exceptions into domain-specific error
        raise SchemaError(f"Failed to drop schema '{schema}'") from exc
