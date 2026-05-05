"""
Data Deletion Utilities for Unified Data Schema

Provides flexible deletion operations for the unified data schema:
- Delete by exchange
- Delete by exchange + symbol
- Delete by exchange + symbol + timeframe
- Delete entire tables
- Delete entire schema

Can be used as a standalone script or imported as a module.
"""

import argparse
import sys
from typing import Optional, List
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from bitpredict.common.logging import get_logger
from bitpredict.common.db.config import get_engine
from bitpredict.common.db.exceptions import DataDeleteError
from bitpredict.common.constants import DATA_SCHEMA
logger = get_logger(__name__)

# Schema and table names
ALL_TABLES = [
    "time_bar",
    "dollar_bar",
    "volume_bar",
    "volatility_bar",
    "renko_bar",
    "hybrid_bar",
    "range_bar"
]


# ============================================================================
# Deletion Functions
# ============================================================================

def delete_by_exchange(
    engine: Engine,
    table_name: str,
    exchange: str,
    dry_run: bool = False
) -> int:
    """Delete all data for a specific exchange from a table."""
    qualified_table = f"{DATA_SCHEMA}.{table_name}"
    
    try:
        with engine.begin() as conn:
            count_query = text(f"SELECT COUNT(*) FROM {qualified_table} WHERE exchange = :exchange")
            result = conn.execute(count_query, {"exchange": exchange})
            row_count = result.scalar()
            
            if dry_run:
                logger.info(f"[DRY RUN] Would delete {row_count:,} rows from {qualified_table} where exchange='{exchange}'")
                return row_count
            
            delete_query = text(f"DELETE FROM {qualified_table} WHERE exchange = :exchange")
            conn.execute(delete_query, {"exchange": exchange})
            logger.info(f"✓ Deleted {row_count:,} rows from {qualified_table} where exchange='{exchange}'")
            return row_count
            
    except SQLAlchemyError as exc:
        raise DataDeleteError(f"Failed to delete from {qualified_table} where exchange='{exchange}'") from exc


def delete_by_exchange_symbol(
    engine: Engine,
    table_name: str,
    exchange: str,
    symbol: str,
    dry_run: bool = False
) -> int:
    """Delete all data for a specific exchange and symbol from a table."""
    qualified_table = f"{DATA_SCHEMA}.{table_name}"
    
    try:
        with engine.begin() as conn:
            count_query = text(f"SELECT COUNT(*) FROM {qualified_table} WHERE exchange = :exchange AND symbol = :symbol")
            result = conn.execute(count_query, {"exchange": exchange, "symbol": symbol})
            row_count = result.scalar()
            
            if dry_run:
                logger.info(f"[DRY RUN] Would delete {row_count:,} rows from {qualified_table} where exchange='{exchange}' and symbol='{symbol}'")
                return row_count
            
            delete_query = text(f"DELETE FROM {qualified_table} WHERE exchange = :exchange AND symbol = :symbol")
            conn.execute(delete_query, {"exchange": exchange, "symbol": symbol})
            logger.info(f"✓ Deleted {row_count:,} rows from {qualified_table} where exchange='{exchange}' and symbol='{symbol}'")
            return row_count
            
    except SQLAlchemyError as exc:
        raise DataDeleteError(f"Failed to delete from {qualified_table} where exchange='{exchange}' and symbol='{symbol}'") from exc



def delete_by_exchange_symbol_timeframe(
    engine: Engine,
    table_name: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    dry_run: bool = False
) -> int:
    """Delete all data for a specific exchange, symbol, and timeframe from a table."""
    qualified_table = f"{DATA_SCHEMA}.{table_name}"
    
    try:
        with engine.begin() as conn:
            count_query = text(f"""
                SELECT COUNT(*) FROM {qualified_table}
                WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
            """)
            result = conn.execute(count_query, {"exchange": exchange, "symbol": symbol, "timeframe": timeframe})
            row_count = result.scalar()
            
            if dry_run:
                logger.info(f"[DRY RUN] Would delete {row_count:,} rows from {qualified_table} where exchange='{exchange}', symbol='{symbol}', timeframe='{timeframe}'")
                return row_count
            
            delete_query = text(f"""
                DELETE FROM {qualified_table}
                WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
            """)
            conn.execute(delete_query, {"exchange": exchange, "symbol": symbol, "timeframe": timeframe})
            logger.info(f"✓ Deleted {row_count:,} rows from {qualified_table} where exchange='{exchange}', symbol='{symbol}', timeframe='{timeframe}'")
            return row_count
            
    except SQLAlchemyError as exc:
        raise DataDeleteError(f"Failed to delete from {qualified_table}") from exc


def drop_table(engine: Engine, table_name: str, cascade: bool = False, dry_run: bool = False) -> bool:
    """Drop a specific table from the data schema."""
    qualified_table = f"{DATA_SCHEMA}.{table_name}"
    
    try:
        with engine.begin() as conn:
            check_query = text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = :schema AND table_name = :table)")
            result = conn.execute(check_query, {"schema": DATA_SCHEMA, "table": table_name})
            exists = result.scalar()
            
            if not exists:
                logger.warning(f"Table {qualified_table} does not exist")
                return False
            
            if dry_run:
                logger.info(f"[DRY RUN] Would drop table {qualified_table}{' CASCADE' if cascade else ''}")
                return True
            
            cascade_sql = " CASCADE" if cascade else ""
            drop_query = text(f"DROP TABLE {qualified_table}{cascade_sql}")
            conn.execute(drop_query)
            logger.info(f"✓ Dropped table {qualified_table}")
            return True
            
    except SQLAlchemyError as exc:
        raise DataDeleteError(f"Failed to drop table {qualified_table}") from exc



def drop_all_tables(engine: Engine, cascade: bool = False, dry_run: bool = False) -> int:
    """Drop all tables in the data schema."""
    dropped_count = 0
    for table_name in ALL_TABLES:
        try:
            if drop_table(engine, table_name, cascade=cascade, dry_run=dry_run):
                dropped_count += 1
        except DataDeleteError as exc:
            logger.error(f"Failed to drop {table_name}: {exc}")
            continue
    
    if not dry_run:
        logger.info(f"✓ Dropped {dropped_count} tables from {DATA_SCHEMA} schema")
    else:
        logger.info(f"[DRY RUN] Would drop {dropped_count} tables from {DATA_SCHEMA} schema")
    return dropped_count


def drop_schema(engine: Engine, cascade: bool = False, dry_run: bool = False) -> bool:
    """Drop the entire data schema."""
    try:
        with engine.begin() as conn:
            check_query = text("SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = :schema)")
            result = conn.execute(check_query, {"schema": DATA_SCHEMA})
            exists = result.scalar()
            
            if not exists:
                logger.warning(f"Schema {DATA_SCHEMA} does not exist")
                return False
            
            if dry_run:
                logger.info(f"[DRY RUN] Would drop schema {DATA_SCHEMA}{' CASCADE' if cascade else ''}")
                return True
            
            cascade_sql = " CASCADE" if cascade else ""
            drop_query = text(f"DROP SCHEMA {DATA_SCHEMA}{cascade_sql}")
            conn.execute(drop_query)
            logger.info(f"✓ Dropped schema {DATA_SCHEMA}")
            return True
            
    except SQLAlchemyError as exc:
        raise DataDeleteError(f"Failed to drop schema {DATA_SCHEMA}") from exc


def delete_from_all_tables(
    engine: Engine,
    exchange: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    dry_run: bool = False
) -> dict:
    """Delete data from all tables based on filters."""
    results = {}
    for table_name in ALL_TABLES:
        try:
            if timeframe and symbol and exchange:
                count = delete_by_exchange_symbol_timeframe(engine, table_name, exchange, symbol, timeframe, dry_run)
            elif symbol and exchange:
                count = delete_by_exchange_symbol(engine, table_name, exchange, symbol, dry_run)
            elif exchange:
                count = delete_by_exchange(engine, table_name, exchange, dry_run)
            else:
                logger.warning(f"No filters provided, skipping {table_name}")
                continue
            results[table_name] = count
        except DataDeleteError as exc:
            logger.error(f"Failed to delete from {table_name}: {exc}")
            results[table_name] = 0
            continue
    
    total_deleted = sum(results.values())
    if not dry_run:
        logger.info(f"✓ Total rows deleted across all tables: {total_deleted:,}")
    else:
        logger.info(f"[DRY RUN] Total rows that would be deleted: {total_deleted:,}")
    return results



# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for data deletion."""
    parser = argparse.ArgumentParser(
        description="Delete data from unified data schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python delete_data.py --table time_bar --exchange binance
  python delete_data.py --all-tables --exchange binance --symbol BTC
  python delete_data.py --table time_bar --exchange binance --symbol BTC --timeframe 1h
  python delete_data.py --drop-table time_bar
  python delete_data.py --drop-all-tables
  python delete_data.py --drop-schema --cascade
  python delete_data.py --table time_bar --exchange binance --dry-run
        """
    )
    
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--table", help="Table name to delete from")
    action_group.add_argument("--all-tables", action="store_true", help="Delete from all tables")
    action_group.add_argument("--drop-table", metavar="TABLE", help="Drop a specific table")
    action_group.add_argument("--drop-all-tables", action="store_true", help="Drop all tables")
    action_group.add_argument("--drop-schema", action="store_true", help="Drop the entire data schema")
    
    parser.add_argument("--exchange", help="Exchange name (e.g., binance, bybit)")
    parser.add_argument("--symbol", help="Symbol name (e.g., BTC, ETH)")
    parser.add_argument("--timeframe", help="Timeframe (e.g., 1m, 1h, 1d)")
    parser.add_argument("--cascade", action="store_true", help="Use CASCADE when dropping")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    
    args = parser.parse_args()
    
    if args.symbol and not args.exchange:
        parser.error("--symbol requires --exchange")
    if args.timeframe and not (args.exchange and args.symbol):
        parser.error("--timeframe requires --exchange and --symbol")
    if (args.exchange or args.symbol or args.timeframe) and (args.drop_table or args.drop_all_tables or args.drop_schema):
        parser.error("Cannot use filters with drop operations")
    
    try:
        engine = get_engine()
    except Exception as exc:
        logger.error(f"Failed to connect to database: {exc}")
        sys.exit(1)
    
    try:
        if args.drop_schema:
            drop_schema(engine, cascade=args.cascade, dry_run=args.dry_run)
        elif args.drop_all_tables:
            drop_all_tables(engine, cascade=args.cascade, dry_run=args.dry_run)
        elif args.drop_table:
            drop_table(engine, args.drop_table, cascade=args.cascade, dry_run=args.dry_run)
        elif args.all_tables:
            if not args.exchange:
                parser.error("--all-tables requires at least --exchange")
            results = delete_from_all_tables(engine, exchange=args.exchange, symbol=args.symbol, timeframe=args.timeframe, dry_run=args.dry_run)
            print("\nDeletion Summary:")
            print("-" * 50)
            for table, count in results.items():
                print(f"  {table:20s}: {count:>10,} rows")
            print("-" * 50)
            print(f"  {'TOTAL':20s}: {sum(results.values()):>10,} rows")
        else:
            if not args.exchange:
                parser.error("--table requires at least --exchange")
            if args.timeframe:
                count = delete_by_exchange_symbol_timeframe(engine, args.table, args.exchange, args.symbol, args.timeframe, args.dry_run)
            elif args.symbol:
                count = delete_by_exchange_symbol(engine, args.table, args.exchange, args.symbol, args.dry_run)
            else:
                count = delete_by_exchange(engine, args.table, args.exchange, args.dry_run)
            print(f"\n{'Would delete' if args.dry_run else 'Deleted'} {count:,} rows")
        sys.exit(0)
    except DataDeleteError as exc:
        logger.error(f"Deletion failed: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        sys.exit(130)
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
