"""
Hyperliquid Data Fetcher Main Entry Point - Database-Driven Version.

This module serves as the entry point for the Hyperliquid cryptocurrency data
collection system. It reads configuration from the database instead of YAML
files, handles logging setup, signal handling for graceful shutdown, and
system initialization.

Key Changes from YAML version:
- Reads configuration from meta.data_tick table
- No longer uses config.yaml file
- Constructs config dict from database records
- Supports multiple market types from database
"""

import time
from typing import List

from fetcher import KrakenStreamManager
from bitpredict.common.db.services.meta import get_tick_meta
from bitpredict.data.tick.utils import build_config_from_db
from dotenv import load_dotenv
from bitpredict.common.logging import get_logger, setup_logging

load_dotenv()

# Configure module logger
setup_logging("data.tick.kraken")
logger = get_logger(__name__)



def main():
    """
    Main entry point for the Hyperliquid data fetcher system (database-driven).

    This function orchestrates the following operations:
    1. Initializes database engine
    2. Reads configuration from meta.data_tick table
    3. Sets up logging based on configuration
    4. Creates HyperliquidStreamManager instance(s)
    5. Starts all WebSocket streams for enabled markets
    6. Keeps main thread alive until interrupted

    The system can be stopped gracefully using Ctrl+C (SIGINT) or
    termination signal (SIGTERM).
    
    Key Differences from YAML Version:
    - No config file loading
    - Reads from database instead
    - Can handle multiple market types
    - Validates configuration exists in database
    
    Raises:
        RuntimeError: If no enabled Hyperliquid configurations found in database.
        Exception: Any database or initialization errors are logged and re-raised.
    """
    try:

        # -----------------------------------------------------------------------
        # STEP 1: Fetch Configuration from Database
        # -----------------------------------------------------------------------

        # Get all enabled configurations for Hyperliquid
        kraken_configs = get_tick_meta(exchange='kraken')
 
        # -----------------------------------------------------------------------
        # STEP 2: Create Stream Managers for Each Market
        # -----------------------------------------------------------------------

        systems: List[KrakenStreamManager] = []
        
        for db_config in kraken_configs:
            # Build config dict from database record
            config = build_config_from_db(db_config)
            
            logger.info(
                "Creating HyperliquidStreamManager for %s market...",
                db_config['market_type']
            )
            
            # Create stream manager
            system = KrakenStreamManager(config)
            systems.append(system)
            
            logger.info(
                "✓ Manager created for %s market (%d symbols, %d streams)",
                db_config['market_type'],
                len(db_config['symbols']),
                len(config['collection']['stream_types'])
            )
        
        # -----------------------------------------------------------------------
        # STEP 3: Start All Stream Managers
        # -----------------------------------------------------------------------
        
        for system in systems:
            market_type = system.market_type
            logger.info("Starting streams for %s market...", market_type)
            system.start_all()
        

        logger.info("Press Ctrl+C to stop")
        logger.info("")
        
        # -----------------------------------------------------------------------
        # STEP 4: Keep Main Thread Alive
        # -----------------------------------------------------------------------
        # Keep running until interrupted
        while all(system.is_running for system in systems):
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Ctrl+C detected - Initiating shutdown...")

    
    except Exception as exc:
        logger.exception("Fatal error during initialization or runtime")
        raise
    
    finally:
        # -----------------------------------------------------------------------
        # CLEANUP: Stop All Systems
        # -----------------------------------------------------------------------
        if 'systems' in locals():
            logger.info("Stopping all stream managers...")
            for system in systems:
                try:
                    system.stop_all()
                    logger.info("✓ Stopped %s market", system.market_type)
                except Exception as exc:
                    logger.error("Error stopping %s market: %s", system.market_type, exc)
            

            logger.info("SHUTDOWN COMPLETE")



if __name__ == "__main__":
    main()
