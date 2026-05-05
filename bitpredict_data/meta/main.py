"""
Meta module entry point - orchestration only.

This script is the entry point for loading cryptocurrency exchange metadata.
It orchestrates the complete workflow by calling dedicated loader utilities.

Usage:
    python main.py

Environment:
    Requires .env file with database configuration at project root.
"""

from pathlib import Path
from typing import Callable
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# MODULE SETUP
# ---------------------------------------------------------------------------

# Resolve project root and load environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

# Load environment variables before any database operations occur
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

from bitpredict.common.logging import get_logger, setup_logging
from bitpredict.common.db.services.meta import (
    insert_symbols,
    insert_time_bars,
    insert_tick,
    insert_custom_bars,
    insert_blockchain,
    insert_macro,
)
from bitpredict.common.utils.file_system import read_yaml_config

# Initialize module-level logger
# Use a unified logger name for the entire meta pipeline
# All modules in the meta workflow should log to this same logger
setup_logging("data.meta")
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# META LOADER REGISTRY
# ---------------------------------------------------------------------------

"""
Registry mapping configuration filenames to their corresponding loader
functions. This acts as the single source of truth for all supported
meta ingestion workflows.

IMPORTANT: Order matters! symbols.yaml must be processed FIRST because
ohlcv and bars tables have foreign key dependencies on symbols_master.

To add a new meta loader:
    1. Create the loader function
    2. Add its YAML file and function here
    3. Ensure dependencies are respected in ordering
"""

META_LOADERS: dict[str, Callable] = {
    # Master table - MUST be first
    "symbols.yaml": insert_symbols,
    
    # Tables with foreign key dependencies on symbols_master
    "time_bars.yaml": insert_time_bars,
    "custom_bars.yaml": insert_custom_bars,
    
    # Independent tables - can be in any order
    "tick.yaml": insert_tick,
    "blockchain.yaml": insert_blockchain,
    "macro.yaml": insert_macro,
}

# ---------------------------------------------------------------------------
# MAIN ORCHESTRATION
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Orchestrate the complete meta loading process.

    Responsibilities:
        1. Initialize the database engine using environment variables
        2. Locate and load all meta YAML configuration files
        3. Dispatch each configuration to its corresponding loader IN ORDER
        4. Log progress and errors consistently

    The function expects:
        - A `.env` file at project root containing database credentials
        - YAML configuration files located in the same directory as this script

    CRITICAL: The order of execution matters because:
        - symbols_master must be populated before ohlcv and bars
        - ohlcv and bars tables reference symbols_master via foreign keys

    Raises:
        FileNotFoundError:
            If any required YAML configuration file is missing
        ValueError:
            If configuration validation fails
        Exception:
            Any unexpected error during the loading process

    Example:
        >>> python main.py
        INFO: Running loader: insert_symbols
        INFO: Running loader: insert_ohlcv
        INFO: Running loader: insert_custom_bars
        INFO: Meta loading completed successfully

    Notes:
        - Environment variables are loaded at module import time
        - Database engine initialization is done once
        - Exit code 0 on success, non-zero on failure
        - Suitable for cron jobs, CI pipelines, or manual execution
    """
    try:
        # Initialize database connection (engine creation validates env config)
        logger.debug("Initializing database engine")

        # Directory containing this script and all YAML configs
        base_path = Path(__file__).parent

        # Iterate through all registered meta loaders IN ORDER
        for yaml_filename, loader_func in META_LOADERS.items():
            yaml_path = base_path / yaml_filename

            logger.debug("Loading configuration file: %s", yaml_path)

            # Load YAML configuration
            config = read_yaml_config(yaml_path)

            # Execute corresponding loader
            logger.info("Running loader: %s", loader_func.__name__)
            loader_func(config)

        logger.info("Meta loading completed successfully")

    except FileNotFoundError as exc:
        logger.error("Configuration file not found: %s", exc)
        raise

    except ValueError as exc:
        logger.error("Configuration validation error: %s", exc)
        raise

    except Exception:
        logger.exception("Unexpected error during meta loading")
        raise


# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()