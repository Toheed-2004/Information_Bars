# ============================================================================
# Custom Exceptions
# ============================================================================


class DatabaseError(Exception):
    """
    Base exception for all database-related errors.

    All custom exceptions in this module inherit from this class to allow
    consistent catching at higher layers.
    """


class SchemaError(DatabaseError):
    """
    Raised when schema-level operations fail.

    Examples:
    - Creating a schema
    - Validating schema existence
    """


class TableError(DatabaseError):
    """
    Raised when table-level operations fail.

    Examples:
    - Dropping tables
    - Checking table existence
    - Inspecting metadata
    """


class DataLoadError(DatabaseError):
    """
    Raised when reading data from the database fails.

    Examples:
    - SELECT queries
    - pandas.read_sql failures
    """


class DataSaveError(DatabaseError):
    """
    Raised when writing data to the database fails.

    Examples:
    - DataFrame.to_sql
    - Constraint violations
    - TimescaleDB hypertable creation
    """


class QueryError(DatabaseError):
    """
    Raised when database query operations fail.

    Examples:
    - SELECT queries fail
    - INSERT/UPDATE/DELETE operations fail
    - Query execution errors
    - Query result processing failures
    """


class DataReadError(Exception):
    """
    Raised when reading data from the database fails.
    """

    pass
