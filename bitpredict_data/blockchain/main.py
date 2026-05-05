"""
Blockchain Data Pipeline Main Module

Entry point for fetching, managing, and analyzing blockchain data.
This module provides a simple interface to run the blockchain data pipeline.

The pipeline reads configuration from the meta.blockchain table in the database
instead of the YAML file, allowing for dynamic configuration management.

Usage:
    python main.py           # Default: run the pipeline (handles full/update automatically)

The pipeline will:
1. Load configuration from meta.blockchain table
2. Initialize the BlockchainDataManager
3. Process all configured blockchain charts (automatically determines if full or incremental needed)
4. Print results summary
"""

import warnings
import sys
from pathlib import Path

from bitpredict.common.logging import get_logger, setup_logging
from bitpredict.common.db.services.meta import get_blockchain_meta
from bitpredict.data.blockchain.fetcher import BlockchainDataManager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

setup_logging("data.blockchain")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logger = get_logger(__name__)


def main():
    """
    Main entry point for the blockchain data pipeline.
    
    The pipeline automatically determines whether to do a full download
    or incremental update based on existing data in the database.
    """
    logger.info("Starting blockchain data pipeline...")
    
    try:
        # Load configuration from meta.blockchain table
        logger.debug("Loading blockchain configuration from meta.blockchain table...")
        blockchain_charts = get_blockchain_meta()
        print(f"{(blockchain_charts)} ")
        if not blockchain_charts:
            logger.error("No blockchain charts found in meta.blockchain table")
            logger.error("Please run 'python -m bitpredict.data.meta' to load blockchain configuration")
            return
        
        logger.info(f"Found {len(blockchain_charts)} chart categories in database")
        
        # Reconstruct config in the expected format
        config = {
            "blockchain_charts": blockchain_charts,
            "settings": {
                "schema_name": "data_blockchain"
            }
        }
        
        # Initialize the blockchain data manager with configuration
        logger.info("Initializing BlockchainDataManager...")
        manager = BlockchainDataManager(config)
        
        # Process all charts (automatically handles full vs incremental logic)
        logger.info("Processing blockchain charts...")
        results = manager.update_all()  # This method already has the smart logic
        
        # Log summary of results
        successful = sum(1 for count in results.values() if count > 0)
        total_records = sum(results.values())
        logger.info(f"Pipeline completed: {successful} charts updated, {total_records} total records inserted")
        
    except Exception as exc:
        logger.exception("Error in blockchain data pipeline")
        raise


if __name__ == "__main__":
    """
    Execute the main function when script is run directly.
    
    This allows the module to be both imported and run as a script:
    - Import: from main import main
    - Run: python main.py
    """
    main()