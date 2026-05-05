"""
Macro Economic Data Module.

Handles aggregation and retrieval of multiple economic indicators from the
database, resampling them to a common frequency and time range.
"""
import warnings
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd

from bitpredict.common.db.config import get_engine
from bitpredict.common.db.utils import read_df
from bitpredict.common.logging import get_logger
from bitpredict.data.macro.utils import resample_to_frequency
from economic_indicator import EconomicIndicator

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


class MacroEconomicData:
    """
    Manages aggregated macro economic indicators.
    
    This class retrieves multiple economic indicators from the database,
    aligns them to a common time range, resamples them to a specified
    frequency, and merges them into a single DataFrame.
    
    Attributes:
        schema_name (str): Database schema name.
        config_dict (dict): Configuration dictionary with economic indicators.
        start_date (str): Start date for data retrieval.
        end_date (str): End date for data retrieval.
        timeframe (str): Time frequency for resampling.
        engine: SQLAlchemy database engine.
        start_date_ (pd.Timestamp): Start date as UTC timestamp.
        end_date_ (pd.Timestamp): End date as UTC timestamp.
    """
    
    def __init__(
        self,
        config_dict,
        start_date="2020-01-01",
        end_date="now",
        timeframe='1h',
        schema_name='data_indicators'
    ):
        """
        Initialize Macro Economic Data Manager.
        
        Args:
            config_dict (dict): Dictionary with economic indicator configurations.
                Expected format: {'economic_indicators': {indicator_key: config}}
            start_date (str, optional): Start date for data retrieval in
                'YYYY-MM-DD' format. Defaults to "2020-01-01".
            end_date (str, optional): End date for data retrieval in 'YYYY-MM-DD'
                format, or 'now' for current date. Defaults to "now".
            timeframe (str, optional): Time frequency for resampling
                (e.g., '1h', '1d'). Defaults to '1h'.
            schema_name (str, optional): Database schema name.
                Defaults to 'data_indicators'.
        """
        self.schema_name = schema_name
        self.config_dict = config_dict
        self.start_date = start_date
        self.timeframe = timeframe
        self.engine = get_engine()
        
        if end_date == "now":
            self.end_date = (
                dt.utcnow().date() - timedelta(days=1)
            ).strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
        
        self.start_date_ = pd.to_datetime(self.start_date).tz_localize('UTC')
        self.end_date_ = pd.to_datetime(self.end_date).tz_localize('UTC')
        
        logger.info(
            f"Initialized MacroEconomicData: "
            f"date_range={self.start_date} to {self.end_date}, "
            f"frequency={timeframe}"
        )
    
    def _forward_fill_until_today(self, df, timestamp_col='timestamp'):
        """
        Forward fill data until today.
        
        Extends the DataFrame with forward-filled values from the last
        timestamp in the data up to today's date.
        
        Args:
            df (pd.DataFrame): DataFrame to forward fill.
            timestamp_col (str, optional): Name of the timestamp column.
                Defaults to 'timestamp'.
        
        Returns:
            pd.DataFrame: Forward-filled DataFrame extending to today.
        
        Note:
            This ensures that indicators with infrequent updates (e.g., monthly)
            have values available for recent dates.
        """
        logger.debug(f"Forward-filling data until today for {len(df)} records")
        
        # Set the timestamp column as index
        df.set_index(timestamp_col, inplace=True)
        
        # Define the current date in UTC (as unix timestamp)
        today_timestamp = int(pd.Timestamp.utcnow().normalize().timestamp())
        
        # Create a timestamp range from the last timestamp to today (daily)
        if df.index.max() < today_timestamp:
            new_timestamps = pd.date_range(
                start=pd.Timestamp.fromtimestamp(df.index.max()),
                end=pd.Timestamp.fromtimestamp(today_timestamp),
                freq='D',
                tz='UTC'
            )
            new_timestamps = (
                new_timestamps.astype(np.int64) // 10**9
            ).astype(int)
            
            # Reindex with the new timestamps
            df = df.reindex(df.index.union(new_timestamps))
            
            logger.debug(
                f"Added {len(new_timestamps) - 1} forward-filled days"
            )
        
        # Forward-fill missing values
        df = df.fillna(method='ffill')
        
        # Reset the index to restore the timestamp column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)
        
        return df
    
    def update_data(self, next_fetch_start_timestamp):
        """
        Update data by fetching new records from FRED API (ALFRED).
        
        Fetches fresh archival data for all indicators starting from the specified
        timestamp, aligns them by release date, and returns a merged DataFrame.
        
        Args:
            next_fetch_start_timestamp (int or str): Unix timestamp or date string
                from which to fetch new data.
        
        Returns:
            pd.DataFrame: Updated DataFrame with new data for all indicators,
                aligned by publication date and resampled to the specified time horizon.
        
        Raises:
            Exception: If fetching or processing data fails.
        """
        logger.info(
            f"Updating data from timestamp: {next_fetch_start_timestamp}"
        )
        
        if isinstance(next_fetch_start_timestamp, (int, float)):
            # Convert timestamp to date string for FRED API
            next_fetch_start_date = pd.Timestamp.fromtimestamp(
                next_fetch_start_timestamp, tz='UTC'
            ).strftime('%Y-%m-%d')
        else:
            next_fetch_start_date = pd.to_datetime(
                next_fetch_start_timestamp
            ).strftime('%Y-%m-%d')
        
        indicators_config = self.config_dict.get('economic_indicators', {})
        
        ei_df_list = []
        ei_df_timestamp_list = []
        
        for indicator_key, indicator_config in indicators_config.items():
            indicator_name = indicator_config.get('fred_series') or indicator_config.get('name')
            frequency = indicator_config['frequency']
            
            logger.info(f"Fetching update for {indicator_key} ({indicator_name})")
            
            try:
                ei_object = EconomicIndicator(
                    name=indicator_name,
                    frequency=frequency,
                    table_name=indicator_key
                )
                
                # Fetch archival data (returns realtime_start, date, value)
                raw_df = ei_object.fetch_data(
                    observation_start=next_fetch_start_date
                )
                
                # Process archival data to be point-in-time (release date based)
                # This logic mirrors EconomicIndicator._prepare_data
                if 'value' in raw_df.columns:
                    raw_df = raw_df.rename(columns={'value': indicator_name})
                
                raw_df['realtime_start'] = pd.to_datetime(raw_df['realtime_start']).dt.tz_localize('UTC')
                raw_df['date'] = pd.to_datetime(raw_df['date']).dt.tz_localize('UTC')
                
                # Filter out rows before our fetch start timestamp to save processing
                # realtime_start is our source of truth for "known at"
                raw_df = raw_df[raw_df['realtime_start'].astype(np.int64) // 10**9 >= next_fetch_start_timestamp]
                
                if raw_df.empty:
                    logger.warning(f"No new releases found for {indicator_key}")
                    continue

                # Sort and pick the latest observation released on each date
                raw_df = raw_df.sort_values(['realtime_start', 'date'])
                processed_df = raw_df.groupby('realtime_start').last().reset_index()
                
                # Rename value column to indicator_key for consistency in merged DF
                indicator_col = indicator_key.lower()
                if indicator_name in processed_df.columns:
                    processed_df = processed_df.rename(columns={indicator_name: indicator_col})
                
                processed_df['datetime'] = processed_df['realtime_start']
                processed_df['timestamp'] = (
                    processed_df['datetime'].astype(np.int64) // 10**9
                ).astype(int)
                
                # Drop extra columns
                processed_df = processed_df[['timestamp', indicator_col]]
                
                # Resample to desired frequency
                ei_df = resample_to_frequency(processed_df, self.timeframe)
                
                ei_df_timestamp_list.append(ei_df['timestamp'].min())
                ei_df_list.append(ei_df)
                
            except Exception as e:
                logger.error(f"Error updating {indicator_key}: {e}")
                continue
        
        if not ei_df_list:
            logger.error("No data could be fetched for any indicator")
            # Return an empty dataframe with timestamp column to avoid crashing if possible, 
            # or raise if essential. Here we raise as per original code.
            raise ValueError("No data could be fetched for any indicator")
        
        # Align all dataframes to the latest start timestamp among indicators
        start_timestamp = max(ei_df_timestamp_list)
        logger.debug(f"Aligning all indicators to timestamp: {start_timestamp}")
        
        ei_df_list_aligned = []
        for ei_df in ei_df_list:
            ei_df = ei_df[ei_df['timestamp'] >= start_timestamp]
            ei_df.reset_index(drop=True, inplace=True)
            ei_df_list_aligned.append(ei_df)
        
        # Merge all dataframes
        merged_df = pd.concat(ei_df_list_aligned, axis=1)
        # Forward fill across the merged indicators to handle different release timings
        merged_df = merged_df.fillna(method='ffill')
        df_data = merged_df.loc[:, ~merged_df.columns.duplicated()]
        
        logger.info(
            f"Update complete: {len(df_data)} rows, "
            f"{len(df_data.columns)} columns"
        )
        
        return df_data
    
    def get_data(self):
        """
        Get macro economic data from database.
        
        Retrieves all configured indicators from the database, processes them
        (forward-filling, resampling, aligning), and merges them into a single
        DataFrame. If data is missing for the requested date range, it fetches
        updates from the FRED API.
        
        Returns:
            pd.DataFrame: Aggregated DataFrame with all economic indicators,
                aligned to the specified date range and resampled to the
                specified time horizon.
        
        Raises:
            ValueError: If no data could be loaded from the database.
            Exception: If processing fails.
        """
        logger.info("Retrieving macro economic data from database")
        
        indicators_config = self.config_dict.get('economic_indicators', {})
        ei_df_list = []
        
        # Convert start_date to unix timestamp
        start_timestamp = int(
            pd.to_datetime(self.start_date, utc=True).timestamp()
        )
        end_timestamp = int(
            pd.to_datetime(self.end_date, utc=True).timestamp()
        )
        
        for indicator_key, indicator_config in indicators_config.items():
            table_name = f"ei_{indicator_key.lower()}"
            
            logger.info(f"Loading {indicator_key} from {table_name}")
            
            try:
                # Read data from database
                df_data = read_df(
                    engine=self.engine,
                    schema=self.schema_name,
                    table_name=table_name,
                )
                
                if df_data is None or len(df_data) == 0:
                    logger.warning(f"No data found for {indicator_key}")
                    continue
                
                logger.debug(
                    f"Loaded {len(df_data)} records for {indicator_key}"
                )
                
                # Process data - work with timestamp column
                df_data = df_data[df_data['timestamp'] > start_timestamp]
                df_data.reset_index(drop=True, inplace=True)
                
                # Forward fill until today
                df_data = self._forward_fill_until_today(df_data, 'timestamp')
                
                # Resample to desired frequency
                df_data = resample_to_frequency(df_data, self.timeframe)
                ei_df_list.append(df_data)
                
            except Exception as e:
                logger.error(f"Error processing {indicator_key}: {e}")
                continue
        
        if not ei_df_list:
            logger.error("No data could be loaded from database")
            raise ValueError("No data could be loaded from database")
        
        # Align all dataframes
        logger.debug("Aligning all indicator dataframes")
        ei_df_list_aligned = []
        for ei_df in ei_df_list:
            ei_df = ei_df[ei_df['timestamp'] >= start_timestamp]
            ei_df.reset_index(drop=True, inplace=True)
            ei_df_list_aligned.append(ei_df)
        
        # Merge all dataframes
        merged_df = pd.concat(ei_df_list_aligned, axis=1)
        df_data = merged_df.loc[:, ~merged_df.columns.duplicated()]
        
        # Check if we have data for the full timestamp range
        start_exists = df_data['timestamp'].isin([start_timestamp]).any()
        end_exists = df_data['timestamp'].isin([end_timestamp]).any()
        
        if start_exists and end_exists:
            logger.info(
                f"Data retrieval complete: {len(df_data)} rows, "
                f"{len(df_data.columns)} columns"
            )
            df_data.reset_index(drop=True, inplace=True)
            return df_data
        else:
            # Update with missing data
            logger.warning(
                "Data range incomplete, fetching updates from FRED API"
            )
            next_fetch_start_timestamp = df_data['timestamp'].min()
            df_new = self.update_data(next_fetch_start_timestamp)
            
            # Concatenate only new data
            df_new = df_new.loc[
                (df_new['timestamp'] > df_data['timestamp'].iloc[-1])
            ]
            df_data = pd.concat([df_data, df_new])
            df_data.reset_index(drop=True, inplace=True)
            
            logger.info(
                f"Data retrieval complete with updates: {len(df_data)} rows, "
                f"{len(df_data.columns)} columns"
            )
            
            return df_data