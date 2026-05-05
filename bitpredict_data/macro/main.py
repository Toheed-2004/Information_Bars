"""
Main Economic Data Pipeline.

Functions for downloading, updating, and retrieving economic indicators.
The pipeline reads indicator configuration from the meta.macro table in the
database instead of YAML files, allowing for dynamic configuration management.

Usage:
    python main.py [mode] [start_date] [end_date] [timeframe]

Modes:
    full: Download all economic indicators from scratch
    update: Update all economic indicators (default)
    macro-fetch: Fetch aggregated macro economic data
    macro-update: Update aggregated macro economic data

Examples:
    python main.py update
    python main.py macro-fetch 2020-01-01 2023-12-31 1d
    python main.py full
"""
import sys
import warnings

from dotenv import load_dotenv

from bitpredict.common.db.services.meta import get_macro_meta
from bitpredict.common.logging import get_logger, setup_logging
from economic_indicator import EconomicIndicator
from macro_indicator import MacroEconomicData

# Load environment variables from .env file
load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
setup_logging("data.macro")
logger = get_logger(__name__)


def download_all_indicators(indicators_config):
    """
    Download and save all economic indicators from FRED.
    
    Iterates through all configured indicators, fetches data from FRED API,
    and saves to the database. Continues processing even if individual
    indicators fail.
    
    Args:
        indicators_config (list): List of indicator configuration dictionaries.
            Each dict should contain: 'indicator_key', 'fred_series', 'frequency'.
    
    Example:
        >>> config = [
        ...     {
        ...         'indicator_key': 'unemployment_rate',
        ...         'fred_series': 'UNRATE',
        ...         'frequency': 'monthly'
        ...     }
        ... ]
        >>> download_all_indicators(config)
    """
    logger.info(f"Processing {len(indicators_config)} economic indicators...")
    
    successful = 0
    failed = 0
    
    for indicator_config in indicators_config:
        indicator_key = indicator_config['indicator_key']
        indicator_name = indicator_config['fred_series']
        frequency = indicator_config['frequency']
        
        logger.info(f"Processing: {indicator_key} ({indicator_name})")
        
        try:
            ei_object = EconomicIndicator(
                name=indicator_name,
                frequency=frequency,
                table_name=indicator_key
            )
            ei_object.process_and_save()
            successful += 1
            
        except Exception as e:
            logger.error(f"Error processing {indicator_key}: {str(e)}")
            failed += 1
            continue
    
    logger.info(
        f"All indicators processed. Successful: {successful}, Failed: {failed}"
    )


def update_indicator(indicator_key, indicators_config):
    """
    Update a specific economic indicator.
    
    Fetches the latest data for a single indicator and updates the database.
    
    Args:
        indicator_key (str): Key of the indicator to update
            (e.g., 'unemployment_rate').
        indicators_config (list): List of indicator configuration dictionaries.
    
    Raises:
        ValueError: If indicator_key is not found in the configuration.
    
    Example:
        >>> update_indicator('unemployment_rate', config_list)
    """
    logger.info(f"Updating indicator: {indicator_key}")
    
    indicator_config = None
    for config in indicators_config:
        if config['indicator_key'] == indicator_key:
            indicator_config = config
            break
    
    if not indicator_config:
        logger.error(f"Indicator '{indicator_key}' not found in config")
        raise ValueError(f"Indicator '{indicator_key}' not found in config")
    
    indicator_name = indicator_config['fred_series']
    frequency = indicator_config['frequency']
    
    try:
        ei_object = EconomicIndicator(
            name=indicator_name,
            frequency=frequency,
            table_name=indicator_key
        )
        ei_object.process_and_save()
        logger.info(f"Successfully updated {indicator_key}")
        
    except Exception as e:
        logger.error(f"Failed to update {indicator_key}: {e}")
        raise


def get_macro_data(config, start_date="2020-01-01", end_date="now",
                   timeframe='1h'):
    """
    Get aggregated macro economic data.
    
    Retrieves multiple economic indicators from the database, aligns them
    to a common time range, and resamples to the specified frequency.
    
    Args:
        config (dict): Configuration dictionary with 'economic_indicators' key.
        start_date (str, optional): Start date for data retrieval in
            'YYYY-MM-DD' format. Defaults to "2020-01-01".
        end_date (str, optional): End date for data retrieval in 'YYYY-MM-DD'
            format, or 'now' for current date. Defaults to "now".
        timeframe (str, optional): Time frequency for resampling
            (e.g., '1h', '1d'). Defaults to '1h'.
    
    Returns:
        pd.DataFrame: DataFrame with aggregated macro economic indicators.
            Contains a timestamp column and columns for each indicator.
    
    Example:
        >>> config = {'economic_indicators': {...}}
        >>> df = get_macro_data(config, start_date='2021-01-01', end_date='now')
    """
    logger.info(
        f"Fetching macro data: "
        f"date_range={start_date} to {end_date}, "
        f"frequency={timeframe}"
    )
    
    try:
        macro_data = MacroEconomicData(
            config_dict=config,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        df_macro = macro_data.get_data()
        
        logger.info(
            f"Macro data loaded successfully: "
            f"{len(df_macro)} rows, {len(df_macro.columns)} columns"
        )
        
        if 'datetime' in df_macro.columns:
            logger.info(
                f"Date range: {df_macro['datetime'].min()} to "
                f"{df_macro['datetime'].max()}"
            )
        elif 'timestamp' in df_macro.columns:
            logger.info(
                f"Timestamp range: {df_macro['timestamp'].min()} to "
                f"{df_macro['timestamp'].max()}"
            )
        
        return df_macro
        
    except Exception as e:
        logger.error(f"Failed to fetch macro data: {e}")
        raise


def update_macro_data(config, next_fetch_start_date, resample_func=None):
    """
    Update macro economic data by fetching new records.
    
    Fetches fresh data for all indicators starting from the specified date
    and returns the updated DataFrame.
    
    Args:
        config (dict): Configuration dictionary with 'economic_indicators' key.
        next_fetch_start_date (str or int): Date from which to fetch new data.
            Can be 'YYYY-MM-DD' string or unix timestamp.
        resample_func (callable, optional): Function to resample data to
            desired frequency. Currently unused. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame with updated macro economic data.
    
    Example:
        >>> config = {'economic_indicators': {...}}
        >>> df_updated = update_macro_data(config, '2023-01-01')
    """
    logger.info(f"Updating macro data from {next_fetch_start_date}")
    
    try:
        macro_data = MacroEconomicData(config_dict=config)
        
        df_updated = macro_data.update_data(
            next_fetch_start_timestamp=next_fetch_start_date
        )
        
        logger.info(f"Macro data updated: {len(df_updated)} rows")
        
        return df_updated
        
    except Exception as e:
        logger.error(f"Failed to update macro data: {e}")
        raise


def main():
    """
    Main function to run the economic data pipeline.
    
    Reads configuration from meta.macro database table and executes
    the appropriate operation based on the mode argument.
    
    Command-line Arguments:
        mode (str, optional): Operation mode. Defaults to "update".
            Options: 'full', 'update', 'macro-fetch', 'macro-update'
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            Defaults to "2020-01-01".
        end_date (str, optional): End date in 'YYYY-MM-DD' format or 'now'.
            Defaults to "now".
        timeframe (str, optional): Resampling frequency (e.g., '1h', '1d').
            Defaults to "1h".
    
    Raises:
        SystemExit: If configuration cannot be loaded or processing fails.
    """
    try:
        logger.info("Starting macro economic data pipeline...")
        
        # Determine operation mode from CLI argument or default to "update"
        mode = sys.argv[1].lower() if len(sys.argv) > 1 else "full"
        start_date = sys.argv[2] if len(sys.argv) > 2 else "2020-01-01"
        end_date = sys.argv[3] if len(sys.argv) > 3 else "now"
        timeframe = sys.argv[4] if len(sys.argv) > 4 else "1h"
        
        logger.info(
            f"Pipeline configuration: mode={mode}, "
            f"start_date={start_date}, end_date={end_date}, "
            f"timeframe={timeframe}"
        )
        
        # Load configuration from meta.macro table
        logger.info("Loading macro indicators from meta.macro table...")
        macro_meta = get_macro_meta()
        
        if not macro_meta or 'economic_indicators' not in macro_meta:
            logger.error("No macro indicators found in meta.macro table")
            logger.error(
                "Please run 'python -m bitpredict.data.meta' to load "
                "macro configuration"
            )
            sys.exit(1)
        
        # Extract economic_indicators from the meta dict
        economic_indicators = macro_meta.get('economic_indicators', {})
        config = {'economic_indicators': economic_indicators}
        
        # Convert to list format for download_all_indicators
        macro_config = [
            {
                'indicator_key': key,
                'fred_series': value.get('fred_series'),
                'frequency': value.get('frequency')
            }
            for key, value in economic_indicators.items()
        ]
        
        logger.info(
            f"Loaded {len(economic_indicators)} economic indicators "
            "from database"
        )
        
        # Route to appropriate function based on mode
        if mode == 'full':
            logger.info("[MODE: full] Downloading all economic indicators...")
            download_all_indicators(macro_config)
            
        elif mode == 'update':
            logger.info("[MODE: update] Updating all economic indicators...")
            download_all_indicators(macro_config)
            
        elif mode == 'macro-fetch':
            logger.info("[MODE: macro-fetch] Fetching aggregated macro data...")
            df_macro = get_macro_data(
                config,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
        elif mode == 'macro-update':
            logger.info(
                "[MODE: macro-update] Updating aggregated macro data..."
            )
            # Using start_date as the next_fetch_start_date
            update_macro_data(config, start_date, None)
        
        else:
            logger.error(f"Invalid mode: {mode}")
            logger.error(
                "Valid modes: full, update, macro-fetch, macro-update"
            )
            sys.exit(1)
        
        logger.info("Macro economic data pipeline completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
        
    except Exception as exc:
        logger.exception("Error in macro economic data pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()