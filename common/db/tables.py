from typing import Optional, Dict, List, Iterable
from bitpredict.common.db.utils import ensure_schema, ensure_table
from sqlalchemy import (
    MetaData,
    inspect,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from .exceptions import *

# ============================================================================ #
# Table Utilities
# ============================================================================ #


def drop_table(
    engine: Engine,
    table_name: str,
    schema: str,
    cascade: bool = False,
) -> None:
    """
    Drop a single table safely.
    """

    # ------------------------------------------------------------------------
    # Validate required parameters
    # ------------------------------------------------------------------------
    if not table_name or not schema:
        raise TableError("Table name and schema must be provided")

    # ------------------------------------------------------------------------
    # Optionally include CASCADE to remove dependent objects
    # ------------------------------------------------------------------------
    option = "CASCADE" if cascade else ""

    try:
        # --------------------------------------------------------------------
        # Use a transaction context for safety
        # Atomic drop operation
        # --------------------------------------------------------------------
        with engine.begin() as conn:
            # Execute the DROP TABLE IF EXISTS statement
            conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{table_name} {option}"))
    except SQLAlchemyError as exc:
        # Wrap exceptions in custom TableError for clarity
        raise TableError(f"Failed to drop table {schema}.{table_name}") from exc


def drop_all_tables(
    engine: Engine,
    schema: str,
    exclude: Optional[Iterable[str]] = None,
) -> None:
    """
    Drop all tables in a schema.
    """

    # ------------------------------------------------------------------------
    # Validate schema parameter
    # ------------------------------------------------------------------------
    if not schema:
        raise SchemaError("Schema must be provided")

    try:
        # --------------------------------------------------------------------
        # Inspect schema to get all table names
        # --------------------------------------------------------------------
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema=schema)

        # --------------------------------------------------------------------
        # Optionally exclude specific tables from dropping
        # --------------------------------------------------------------------
        if exclude:
            tables = [t for t in tables if t not in exclude]

        # Nothing to drop
        if not tables:
            return

        # --------------------------------------------------------------------
        # Reflect only the tables we want to drop
        # MetaData collects table objects for SQLAlchemy operations
        # --------------------------------------------------------------------
        metadata = MetaData(schema=schema)
        metadata.reflect(bind=engine, only=tables)

        # --------------------------------------------------------------------
        # Drop tables in reverse dependency order
        # - Reversing sorted_tables ensures child tables are dropped before parents
        # - CASCADE ensures foreign key constraints are handled
        # --------------------------------------------------------------------
        with engine.begin() as conn:
            for table in reversed(metadata.sorted_tables):
                conn.execute(
                    text(f"DROP TABLE IF EXISTS {schema}.{table.name} CASCADE")
                )

    except SQLAlchemyError as exc:
        # Wrap any SQLAlchemy errors in a custom TableError
        raise TableError(f"Failed to drop tables in schema '{schema}'") from exc


def create_table(
    engine: Engine,
    table_name: str,
    schema: str,
    columns: Dict[str, str],
    primary_key: Optional[List[str]] = None,
    foreign_keys: Optional[Dict[str, tuple]] = None,
    unique_constraints: Optional[List[List[str]]] = None,
    indexes: Optional[List[List[str]]] = None,
    if_exists: str = "fail",
) -> None:
    """
    Create a PostgreSQL table with specified schema, columns, and constraints.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy Engine connected to PostgreSQL.
    table_name : str
        Name of the table to create.
    schema : str
        Schema where the table will be created.
    columns : Dict[str, str]
        Dictionary mapping column names to PostgreSQL data types.
        Example: {"id": "SERIAL", "name": "VARCHAR(100)", "age": "INTEGER"}
    primary_key : List[str], optional
        List of column names that form the primary key.
        Example: ["id"] or ["user_id", "order_id"] for composite key
    foreign_keys : Dict[str, tuple], optional
        Dictionary mapping column names to (ref_schema, ref_table, ref_column) tuples.
        Example: {"user_id": ("public", "users", "id")}
    unique_constraints : List[List[str]], optional
        List of column lists that should have unique constraints.
        Example: [["email"], ["username", "domain"]]
    indexes : List[List[str]], optional
        List of column lists to create indexes on.
        Example: [["created_at"], ["user_id", "status"]]
    if_exists : str, default "fail"
        Behavior when the table already exists.
        - "fail": Raise an error if table exists
        - "replace": Drop and recreate the table
        - "skip": Do nothing if table exists

    Raises
    ------
    DataSaveError
        If validation fails or table creation fails.

    Examples
    --------
    Create a simple users table:
    >>> create_table(
    ...     engine=engine,
    ...     table_name="users",
    ...     schema="public",
    ...     columns={
    ...         "id": "SERIAL",
    ...         "username": "VARCHAR(50) NOT NULL",
    ...         "email": "VARCHAR(100) NOT NULL",
    ...         "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    ...     },
    ...     primary_key=["id"],
    ...     unique_constraints=[["email"], ["username"]]
    ... )

    Create an orders table with foreign key:
    >>> create_table(
    ...     engine=engine,
    ...     table_name="orders",
    ...     schema="public",
    ...     columns={
    ...         "order_id": "SERIAL",
    ...         "user_id": "INTEGER NOT NULL",
    ...         "amount": "DECIMAL(10,2)",
    ...         "status": "VARCHAR(20)"
    ...     },
    ...     primary_key=["order_id"],
    ...     foreign_keys={"user_id": ("public", "users", "id")},
    ...     indexes=[["user_id"], ["status"]]
    ... )
    """

    # ------------------------------------------------------------------------
    # Validate required parameters
    # ------------------------------------------------------------------------
    if not table_name or not schema:
        raise DataSaveError("Table name and schema must be provided")

    if not columns:
        raise DataSaveError("At least one column must be specified")

    # ------------------------------------------------------------------------
    # Ensure schema exists
    # ------------------------------------------------------------------------
    ensure_schema(engine, schema)

    qualified_table = f"{schema}.{table_name}"

    try:
        # --------------------------------------------------------------------
        # Check if table exists
        # --------------------------------------------------------------------

        table_exists = ensure_table(engine, schema, table_name)

        # --------------------------------------------------------------------
        # Handle if_exists parameter
        # --------------------------------------------------------------------
        if table_exists:
            if if_exists == "fail":
                raise DataSaveError(
                    f"Table {qualified_table} already exists and if_exists='fail'"
                )
            elif if_exists == "skip":
                print(f"Table {qualified_table} already exists, skipping creation")
                return
            elif if_exists == "replace":
                with engine.begin() as conn:
                    conn.execute(
                        text(f"DROP TABLE IF EXISTS {qualified_table} CASCADE")
                    )
            else:
                raise DataSaveError(
                    f"Invalid if_exists value: {if_exists}. Must be 'fail', 'replace', or 'skip'"
                )

        # --------------------------------------------------------------------
        # Build CREATE TABLE statement
        # --------------------------------------------------------------------
        # Column definitions
        column_defs = [
            f"{col_name} {col_type}" for col_name, col_type in columns.items()
        ]

        # Primary key constraint
        if primary_key:
            pk_cols = ", ".join(primary_key)
            column_defs.append(f"PRIMARY KEY ({pk_cols})")

        # Foreign key constraints
        if foreign_keys:
            for fk_col, (ref_schema, ref_table, ref_col) in foreign_keys.items():
                fk_def = (
                    f"FOREIGN KEY ({fk_col}) REFERENCES "
                    f"{ref_schema}.{ref_table}({ref_col})"
                )
                column_defs.append(fk_def)

        # Unique constraints
        if unique_constraints:
            for unique_cols in unique_constraints:
                unique_def = f"UNIQUE ({', '.join(unique_cols)})"
                column_defs.append(unique_def)

        # Combine all definitions
        columns_sql = ",\n    ".join(column_defs)
        create_sql = f"""
        CREATE TABLE {qualified_table} (
            {columns_sql}
        )
        """

        # --------------------------------------------------------------------
        # Execute CREATE TABLE
        # --------------------------------------------------------------------
        with engine.begin() as conn:
            conn.execute(text(create_sql))

        # --------------------------------------------------------------------
        # Create indexes
        # --------------------------------------------------------------------
        if indexes:
            with engine.begin() as conn:
                for idx_num, idx_cols in enumerate(indexes):
                    idx_name = f"{table_name}_{'_'.join(idx_cols)}_idx"
                    idx_sql = (
                        f"CREATE INDEX {idx_name} ON {qualified_table} "
                        f"({', '.join(idx_cols)})"
                    )
                    conn.execute(text(idx_sql))

        print(f"✓ Successfully created table {qualified_table}")

    except SQLAlchemyError as exc:
        raise DataSaveError(f"Failed to create table {qualified_table}: {exc}") from exc
