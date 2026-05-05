"""
Economic Indicator Module.

Handles fetching and saving individual economic indicators from FRED API.
Each indicator is stored in a dedicated database table with timestamp-based
records that are forward-filled to create daily continuity.
"""
import os

import numpy as np
import pandas as pd
from fredapi import Fred

from bitpredict.common.db.config import get_engine
from bitpredict.common.db.utils import get_last_timestamp, insert_df, ensure_schema, ensure_table
from bitpredict.common.logging import get_logger
from bitpredict.common.utils.time import datetime_to_timestamp
logger = get_logger(__name__)


class EconomicIndicator:
    """
    Manages individual economic indicators from the FRED API.
    
    This class handles fetching data from the Federal Reserve Economic Data
    (FRED) API, processing it into a standardized format, and storing it in
    a PostgreSQL database with forward-filled daily values.
    
    Attributes:
        name (str): FRED series ID (e.g., 'UNRATE', 'GDP').
        frequency (str): Data frequency (e.g., 'monthly', 'quarterly', 'daily').
        table_name (str): Database table name with 'ei_' prefix.
        schema_name (str): Database schema name.
        engine: SQLAlchemy database engine.
    """
    
    def __init__(self, name, frequency, table_name, schema_name='data_indicators'):
        """
        Initialize Economic Indicator.
        
        Args:
            name (str): FRED series ID (e.g., 'UNRATE' for unemployment rate).
            frequency (str): Data frequency (e.g., 'monthly', 'quarterly', 'daily').
            table_name (str): Base database table name (will be prefixed with 'ei_').
            schema_name (str, optional): Database schema name.
                Defaults to 'data_indicators'.
        """
        self.name = name
        self.frequency = frequency
        self.table_name = f"ei_{table_name.lower()}"
        self.schema_name = schema_name
        self.engine = get_engine()
        
        logger.info(
            f"Initialized EconomicIndicator: {name} "
            f"(frequency={frequency}, table={self.table_name})"
        )
    
    def fetch_data(self, observation_start='2010-01-01'):
        """
        Fetch data from FRED API with all revisions (ALFRED).
        
        Retrieves historical releases for the specified FRED series ID.
        This provides 'realtime_start' and 'realtime_end' which are essential
        for eliminating lookahead bias in backtests.
        
        Args:
            observation_start (str, optional): Start date for data retrieval
                in 'YYYY-MM-DD' format. Defaults to '2010-01-01'.
        
        Returns:
            pd.DataFrame: DataFrame with 'date', 'realtime_start', 'realtime_end',
                and indicator value columns.
        
        Raises:
            ValueError: If FRED_API_KEY environment variable is not set.
            Exception: If FRED API request fails.
        """
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            logger.error("FRED_API_KEY environment variable not set")
            raise ValueError("FRED_API_KEY environment variable not set")
        
        logger.info(
            f"Fetching archival (ALFRED) data for {self.name} from FRED "
            f"(start_date={observation_start})"
        )
        
        try:
            fred = Fred(api_key)
            # Fetch all releases to get realtime_start (publication date)
            # Some versions of fredapi do not support observation_start here, 
            # so we fetch all and filter in pandas to be safe.
            data_df = fred.get_series_all_releases(self.name)
            
            if observation_start:
                # Filter by observation date manually
                data_df['date'] = pd.to_datetime(data_df['date'])
                obs_start_dt = pd.to_datetime(observation_start)
                data_df = data_df[data_df['date'] >= obs_start_dt]
                
            logger.info(
                f"Successfully fetched and filtered {len(data_df)} release records for {self.name}"
            )
            return data_df
            
        except Exception as e:
            logger.error(f"Failed to fetch archival data for {self.name}: {e}")
            raise
    
    def _prepare_data(self, data_df):
        """
        Prepare data using its release date (realtime_start) to avoid lookahead bias.
        
        Processes ALFRED data by:
        1. Sorting by realtime_start and observation date
        2. Grouping by realtime_start to find the latest available information at each release
        3. Using 'realtime_start' as the source of truth for when data became known
        4. Forward-filling to create daily continuity
        5. Converting to unix timestamps in MILLISECONDS
        6. Retaining human-readable datetime for better DB inspection
        7. Ensuring values are strictly numeric
        
        Args:
            data_df (pd.DataFrame): Raw ALFRED data with 'realtime_start', 'date', and 'value'.
        
        Returns:
            pd.DataFrame: Processed DataFrame with columns [timestamp, datetime, indicator_col]
        """
        logger.debug(f"Preparing {len(data_df)} archival records for {self.name}")
        
        if len(data_df) == 0:
            logger.warning(f"No data to prepare for {self.name}")
            return data_df

        # Use realtime_start (the release date) as our primary time anchor
        if 'value' in data_df.columns:
            data_df = data_df.rename(columns={'value': self.name})

        # Ensure datetime columns are proper datetime objects
        data_df['realtime_start'] = pd.to_datetime(data_df['realtime_start']).dt.tz_localize('UTC')
        data_df['date'] = pd.to_datetime(data_df['date']).dt.tz_localize('UTC')
        
        # Sort to ensure we process information in the order it was released
        data_df = data_df.sort_values(['realtime_start', 'date'])
        
        # Strategy: For each release date, what is the newest information we have?
        realtime_series = data_df.groupby('realtime_start').last().reset_index()
        
        # Ensure values are strictly numeric (float)
        indicator_col = self.name.lower()
        if self.name in realtime_series.columns:
            realtime_series = realtime_series.rename(columns={self.name: indicator_col})
        
        # Convert to numeric, force errors to NaN then drop or fill
        realtime_series[indicator_col] = pd.to_numeric(realtime_series[indicator_col], errors='coerce')
        realtime_series[indicator_col] = realtime_series[indicator_col].round(2)
        
        # Set release date as index for forward filling
        realtime_series = realtime_series.set_index('realtime_start')
        
        # Create a daily date range from the first release to the last release
        start_time = realtime_series.index.min().floor('D')
        end_time = realtime_series.index.max().floor('D')
        idx = pd.date_range(start_time, end_time, freq="D", tz='UTC')
        
        # Reindex and forward fill
        data_df_prepared = realtime_series.reindex(idx, method='ffill')
        
        # Reset index and rename to datetime
        data_df_prepared = data_df_prepared.reset_index().rename(columns={"index": "datetime"})
        
        # ELIMINATE INTRADAY LOOKAHEAD BIAS:
        # Since FRED only provides the release DATE (not exact time), we set the 
        # release time to the END of the day (23:59:59) to ensure it's not used 
        # by a backtest until the data was actually known to the public.
        data_df_prepared['datetime'] = data_df_prepared['datetime'] + pd.Timedelta(hours=23, minutes=59, seconds=59)
        
        # Convert datetime to unix timestamp in MILLISECONDS using fixed utility
        data_df_prepared['timestamp'] = datetime_to_timestamp(data_df_prepared['datetime'])
        
        # Reorder columns: timestamp first, then readable datetime, then value
        data_df_prepared = data_df_prepared[['timestamp', 'datetime', indicator_col]]
        
        logger.debug(
            f"Data preparation complete for {self.name}. "
            f"Release-based continuity with millisecond timestamps and {len(data_df_prepared)} records."
        )
        
        return data_df_prepared
    
    def save_data(self, data_df):
        """
        Save data to database.
        
        Creates schema and table if they don't exist, prepares the data,
        and inserts only new records (those with timestamps greater than
        the last saved timestamp).
        
        Args:
            data_df (pd.DataFrame): DataFrame to save with 'datetime' column
                and indicator value column.
        
        Note:
            Only records newer than the last saved timestamp are inserted
            to avoid duplicates.
        """
        logger.info(f"Saving data for {self.name} to {self.table_name}")
        
        try:
            # Create schema if not exists
            ensure_schema(engine=self.engine, schema_name=self.schema_name)
            
            # Create table if not exists
            ensure_table(
                engine=self.engine,
                schema_name=self.schema_name,
                table_name=self.table_name
            )
            
            # Prepare data
            data_df = self._prepare_data(data_df)
            
            if len(data_df) == 0:
                logger.warning(f"{self.name}: No data available for processing")
                return
            
            # Filter out rows based on last saved timestamp
            last_timestamp = get_last_timestamp(
                engine=self.engine,
                schema_name=self.schema_name,
                table_name=self.table_name
            )
            
            if last_timestamp:
                logger.info(
                    f"Last saved timestamp for {self.name}: {last_timestamp}"
                )
                # Filter out records that are already in the database
                data_df = data_df[data_df['timestamp'] > last_timestamp]
            
            data_df.reset_index(drop=True, inplace=True)
            
            if len(data_df) > 0:
                # Insert data into the database
                insert_df(
                    df=data_df,
                    engine=self.engine,
                    schema_name=self.schema_name,
                    table_name=self.table_name,
                    if_exists="append",
                    is_timeseries=True
                )
                logger.info(
                    f"{self.name}: Successfully inserted {len(data_df)} records"
                )
            else:
                logger.info(f"{self.name}: No new data to insert")
                
        except Exception as e:
            logger.error(f"Error saving data for {self.name}: {e}")
            raise
    
    def process_and_save(self):
        """
        Fetch and save data in one step.
        
        Convenience method that combines fetching data from FRED API
        and saving it to the database.
        
        Raises:
            Exception: If fetching or saving fails.
        """
        logger.info(f"Processing and saving {self.name}")
        
        try:
            data_df = self.fetch_data()
            self.save_data(data_df)
            logger.info(f"Successfully processed and saved {self.name}")
            
        except Exception as e:
            logger.error(f"Failed to process and save {self.name}: {e}")
            raise