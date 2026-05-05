"""
Base data handler for exchange market data operations.

Provides core functionality for fetching, cleaning, and resampling 
cryptocurrency market data from exchanges using a unified interface.
"""

import traceback
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import pytz
import json

from bitpredict.common.db.exceptions import DataSaveError
from bitpredict.common.utils.time import (
    timestamp_to_datetime,
    datetime_to_timestamp,
    timeframe_to_minutes
)
from bitpredict.common.constants import (
    DATA_SCHEMA,
    DEFAULT_TIMEZONE
)
from bitpredict.common.logging import get_logger
from bitpredict.data.time_bars.utils import custom_round_series
from bitpredict.common.db.services import insert_ohlcv, read_ohlcv
from bitpredict.common.market_regimes import calculate_regimes
from bitpredict.common.db.services import get_ohlcv_last_datetime
from bitpredict.common.db.services.market_regime import load_last_bar_id, get_symbol_id

# Logger setup
logger = get_logger(__name__)



class Base(ABC):
    """
    Unified base class for handling exchange market data operations.
    
    Combines fetching and processing into a single class with abstract methods
    for exchange-specific implementations.
    
    Attributes:
        exchange: Exchange name (e.g., 'binance', 'bybit')
        symbol: Trading symbol (e.g., 'BTC', 'ETH')
        engine: SQLAlchemy engine for database operations
        fill_missing_method: Method to fill missing timestamps
        interpolation_method: Interpolation method for missing data
        fill_zero_volume: Method to fill zero volume entries
        retries: Number of retry attempts for failed operations
        retry_delay: Delay in seconds between retry attempts
        timezone: Timezone for timestamp operations
    """
    
    def __init__(
        self,
        exchange: str,
        symbol: str,
        fill_missing_method: str = 'interpolate',
        interpolation_method: str = 'linear',
        fill_zero_volume: str = 'ffill',
        retries: int = 5,
        retry_delay: int = 10,
    ):
        """
        Initialize Base handler.
        
        Args:
            exchange: Exchange name (e.g., 'bybit', 'binance')
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            fill_missing_method: Method to fill missing timestamps
            interpolation_method: Interpolation method for missing data
            fill_zero_volume: Method to fill zero volume entries
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.exchange = exchange
        self.symbol = symbol
        # self.engine = get_engine()
        self.fill_missing_method = fill_missing_method
        self.interpolation_method = interpolation_method
        self.fill_zero_volume = fill_zero_volume
        self.retries = retries
        self.retry_delay = retry_delay
        self.timezone = pytz.timezone(DEFAULT_TIMEZONE)

        logger.info(
            "Initialized Base for %s:%s",
            exchange,
            symbol
        )
    
    @abstractmethod
    def fetch_ohlc(
        self,
        symbol: str,
        start_datetime: Union[datetime, int, None],
        end_datetime: Union[datetime, int]
    ) -> pd.DataFrame:
        """
        Abstract method to fetch OHLC data from the exchange.
        
        Must be implemented by exchange-specific subclasses.
        
        Args:
            symbol: Trading symbol
            start_datetime: Start time for data fetch
            end_datetime: End time for data fetch
            
        Returns:
            DataFrame with OHLC data including 'timestamp' column in unix ms
        """
        pass

    def clean_data(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame by removing duplicates and handling missing values.

        Performs the following operations in order:
        1. Removes duplicate timestamps (keeps first occurrence)
        2. Handles zero volume entries:
           - If ALL rows have volume=0: replaces all with 1
           - If first row has volume=0: replaces with 1 and applies filling
           - If zeros in middle/end: applies filling method
        3. Fills missing timestamps in the time series
        4. Removes unnamed columns
        
        Uses the 'datetime' column for processing, maintains both 'timestamp'
        and 'datetime' columns in the output.
        
        Args:
            df_1m: Input DataFrame with datetime and timestamp columns
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            ValueError: If df_1m is None or doesn't contain required columns
        """
        # Validate input DataFrame
        if df_1m is None:
            error_msg = "No data set. Call set_df_ohlc_1m() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if 'datetime' not in df_1m.columns:
            error_msg = "DataFrame must contain 'datetime' column"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(
            "Cleaning data for %s:%s (%d rows)",
            self.exchange,
            self.symbol,
            len(df_1m),
        )
        
        # Set datetime as index for time-series operations
        df_1m.set_index('datetime', inplace=True)
        
        # Step 1: Remove duplicate timestamps
        duplicate_count = df_1m.index.duplicated(keep='first').sum()
        if duplicate_count > 0:
            logger.info("Removing %d duplicate timestamps", duplicate_count)
            df_1m = df_1m[~df_1m.index.duplicated(keep='first')]
        
        # Step 2: Handle zero volume entries
        if 'volume' in df_1m.columns:
            df_1m = self._handle_zero_volume(df_1m)
        
        # Step 3: Fill missing timestamps
        if self.exchange != "metatrader5":
            df_1m = self._fill_missing_timestamps(df_1m)
        
        # Step 4: Reset index and clean up
        df_1m.reset_index(inplace=True)
        df_1m.rename(columns={'index': 'datetime'}, inplace=True)
        
        # Step 5: Remove unnamed columns (artifacts from previous operations)
        unnamed_columns = df_1m.columns[
            df_1m.columns.str.contains('^Unnamed')
        ]
        if len(unnamed_columns) > 0:
            logger.debug("Removing %d unnamed columns", len(unnamed_columns))
            df_1m = df_1m.loc[:, ~df_1m.columns.str.contains('^Unnamed')]
        
        logger.info(
            "Data cleaning complete for %s:%s - final shape: %s",
            self.exchange,
            self.symbol,
            df_1m.shape,
        )
        
        df_1m.fillna(0, inplace=True)
        
        return df_1m
    
    def _handle_zero_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle zero volume entries in the DataFrame.
        
        Strategy:
        - If all rows have zero volume: replace all with 1
        - If first row has zero volume: replace with 1, then fill remaining
        - Otherwise: replace zeros with NaN and apply filling method
        
        Args:
            df: DataFrame with volume column (modified in-place)
        """
        zero_volume_mask = df['volume'] <= 0
        zero_volume_count = zero_volume_mask.sum()
        
        if zero_volume_count > 0:
            logger.debug("Found %d zero-volume rows", zero_volume_count)
            
            # Case 1: All rows have zero volume
            if zero_volume_count == len(df):
                logger.info("ALL rows have zero volume - replacing all with 1")
                df['volume'] = 1
            
            # Case 2: First row has zero volume
            elif zero_volume_mask.iloc[0]:
                logger.info("First row has zero volume - replacing with 1")
                df.loc[df.index[0], 'volume'] = 1
                
                # Fill remaining zeros
                remaining_zeros = (df['volume'] <= 0).sum()
                if remaining_zeros > 0:
                    df['volume'] = df['volume'].replace(0, np.nan)
                    if self.fill_zero_volume == "ffill":
                        df['volume'] = df['volume'].ffill()
                    elif self.fill_zero_volume == "bfill":
                        df['volume'] = df['volume'].bfill()
                    else:
                        raise ValueError(f"Unsupported fill_zero_volume method: {self.fill_zero_volume}")
            
            # Case 3: Zeros in middle or end
            else:
                df['volume'] = df['volume'].replace(0, np.nan)
                if self.fill_zero_volume == "ffill":
                    df['volume'] = df['volume'].ffill()
                elif self.fill_zero_volume == "bfill":
                    df['volume'] = df['volume'].bfill()
                else:
                    raise ValueError(f"Unsupported fill_zero_volume method: {self.fill_zero_volume}")
        
        return df

    
    def _fill_missing_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing timestamps in the time series.
        
        Creates a complete datetime range from min to max and fills
        missing data points using the specified filling method.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with filled missing timestamps (both datetime and timestamp columns)
        """
        start_time = df.index.min()
        end_time = df.index.max()
        
        # Create complete 1-minute range with timezone awareness
        complete_time_range = pd.date_range(
            start_time, 
            end_time, 
            freq='min',
            tz=timezone.utc  # Ensure timezone awareness
        )
        
        missing_timestamp_count = len(complete_time_range) - len(df)
        
        if missing_timestamp_count > 0:
            logger.info(
                "Filling %d missing timestamps using method: %s",
                missing_timestamp_count,
                self.fill_missing_method,
            )
            
            # Reindex to include all timestamps
            df_reindexed = df.reindex(complete_time_range)
            
            # Apply filling method based on configuration
            if self.fill_missing_method == 'interpolate':
                numeric_columns = df_reindexed.select_dtypes(
                    include='number'
                ).columns
                for col in numeric_columns:
                    df_reindexed[col].interpolate(
                        method=self.interpolation_method,
                        inplace=True,
                        limit_direction='both',
                    )
            
            elif self.fill_missing_method == 'ffill':
                df_reindexed.ffill(inplace=True)
            
            elif self.fill_missing_method == 'bfill':
                df_reindexed.bfill(inplace=True)
            
            else:
                logger.warning(
                    "Unknown fill_missing_method: %s",
                    self.fill_missing_method
                )
                return df
            
            # Handle timestamp column - generate from datetime index
            if 'timestamp' in df_reindexed.columns:
                # Convert datetime index to milliseconds timestamp
                # Only update rows where timestamp is missing (newly added rows)
                missing_timestamp_mask = df_reindexed['timestamp'].isna()
                if missing_timestamp_mask.any():
                    df_reindexed.loc[missing_timestamp_mask, 'timestamp'] = (
                        df_reindexed.index[missing_timestamp_mask]
                        .astype('int64') // 10**6  # Convert nanoseconds to milliseconds
                    )
                
                # Ensure timestamp is int64
                df_reindexed['timestamp'] = df_reindexed['timestamp'].astype('int64')
            
            # Ensure datetime column is properly formatted (if it exists as a column)
            # Note: The index is already datetime, but we might want to ensure it's in the column too
            if 'datetime' not in df_reindexed.columns:
                # Add datetime column from index if it doesn't exist
                df_reindexed['datetime'] = df_reindexed.index
            
            return df_reindexed
        
        return df
    
    def insert_data(
        self,
        df_to_insert: Optional[pd.DataFrame] = None,
        timeframe: str = '1m',
        resample_insert: bool = False
    ) -> None:
        """
        Insert market data into the database using fast COPY method or upsert.
        
        Performs the following operations:
        1. Validates input DataFrame
        2. Removes datetime column (database uses timestamp only)
        3. Rounds numeric values according to custom rules
        4. Inserts data using COPY for 1m data or upsert for resampled data
        
        Args:
            df_to_insert: DataFrame to insert into database
            timeframe: Time Frame for the data (e.g., '1m', '5m', '1h')
            resample_insert: Whether this is resampled data (uses upsert)
            
        Raises:
            ValueError: If DataFrame is missing required columns
            DataSaveError: If database insertion fails
        """
        # Validate input
        if df_to_insert is None or df_to_insert.empty:
            logger.warning(
                "Empty DataFrame provided for %s:%s",
                self.exchange,
                self.symbol
            )
            return
        
        if 'timestamp' not in df_to_insert.columns:
            error_msg = "DataFrame must contain 'timestamp' column"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Prepare database identifiers
        schema_name = f"{DATA_SCHEMA}"
        table_name = f"time"
        
        logger.info(
            "Preparing to insert data into %s.%s (%d rows)",
            schema_name,
            table_name,
            len(df_to_insert),
        )
       
        # attempt up to two times if the first insert fails
        attempts = 2
        for attempt in range(1, attempts + 1):
            try:
                # Round numeric values according to custom precision rules
                numeric_cols = df_to_insert.select_dtypes(
                    include=[np.number]
                ).columns
                for col in numeric_cols:
                    df_to_insert[col] = custom_round_series(df_to_insert[col])

                logger.info(
                    "Rounded numeric values according to custom rules for "
                    "timeseries data (attempt %d)",
                    attempt,
                )

                # Use upsert for resampled data to handle updates properly
                if resample_insert:
                    insert_ohlcv(
                        df=df_to_insert,
                        exchange=self.exchange,
                        symbol=self.symbol,
                        timeframe=timeframe,
                        method="executemany"
                    )
                else:
                    # Use COPY for 1m data which is always complete
                    insert_ohlcv(
                        df=df_to_insert,
                        exchange=self.exchange,
                        symbol=self.symbol,
                        timeframe=timeframe
                    )

                logger.info(
                    "Successfully inserted %d rows into %s.%s",
                    len(df_to_insert),
                    schema_name,
                    table_name,
                )
                # exit the retry loop on success
                break

            except Exception as e:
                logger.error(
                    "Attempt %d: failed to insert data into %s.%s: %s",
                    attempt,
                    schema_name,
                    table_name,
                    str(e),
                )
                traceback.print_exc()
                if attempt == attempts:
                    # all attempts failed, raise error
                    raise DataSaveError(
                        f"Error inserting into {schema_name}.{table_name} after {attempts} attempts"
                    ) from e
                else:
                    logger.warning(
                        "Retrying insert for %s.%s (attempt %d of %d)",
                        schema_name,
                        table_name,
                        attempt + 1,
                        attempts,
                    )
    
    def resample_data(
        self,
        df_1m: pd.DataFrame,
        timeframe_minutes: int
    ) -> pd.DataFrame:
        """
        Resample 1-minute market data to a higher timeframe.
        
        Aggregation rules:
        - open: first value in period
        - high: maximum value in period
        - low: minimum value in period
        - close: last value in period
        - volume: sum of all volumes in period
        
        Args:
            df_1m: Input DataFrame with 1-minute OHLCV data
            timeframe_minutes: Target time frame in minutes
            
        Returns:
            Resampled DataFrame with timestamp and datetime columns
            
        Raises:
            ValueError: If DataFrame is missing required columns
        """
        # Validate input
        if df_1m.empty:
            logger.warning("Empty DataFrame provided for resampling")
            return df_1m
        
        if 'datetime' not in df_1m.columns:
            error_msg = "DataFrame must contain 'datetime' column"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(
            "Resampling %s:%s from 1m to %dm (%d rows)",
            self.exchange,
            self.symbol,
            timeframe_minutes,
            len(df_1m),
        )
        df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])

        
        # Resample using pandas groupby with time-based grouper
        df_resampled = df_1m.groupby(
            pd.Grouper(key='datetime', freq=f'{timeframe_minutes}Min')
        ).agg({
            "open": "first",
            "high": np.max,
            "low": np.min,
            "close": "last",
            "volume": np.sum,
        }).reset_index()
        
        return df_resampled
        
    def resample_and_insert_multiple(self, timeframes_dict: Dict[str, bool], last_datetime: Optional[datetime] = None) -> None:
        """
        Resample 1-minute data into multiple time frames and calculate market regime.
        Handles both initial load and incremental updates without duplicates or gaps.
        """
        for timeframe, enabled in timeframes_dict.items():
            if not enabled:
                continue

            logger.info(
                "Resampling data for %s:%s to %s",
                self.exchange,
                self.symbol,
                timeframe
            )

            # Get last stored timestamp
            last_tf_datetime = get_ohlcv_last_datetime(
                exchange=self.exchange,
                symbol=self.symbol,
                timeframe=timeframe,
            )

            timeframe_minutes = timeframe_to_minutes(timeframe)

            # Fetch 1-minute data
            if last_tf_datetime is None:
                df_1m = read_ohlcv(
                    exchange=self.exchange,
                    symbol=self.symbol,
                    timeframe="1m"
                )
            else:
                df_1m = read_ohlcv(
                    exchange=self.exchange,
                    symbol=self.symbol,
                    timeframe="1m",
                    start_date=last_tf_datetime
                )

            if df_1m.empty:
                logger.info("No new 1-minute data to process")
                continue

            # Resample to target timeframe
            df_resampled = self.resample_data(df_1m, timeframe_minutes)

            # For incremental, combine with existing data from DB for lookback
            if last_tf_datetime is not None:
                symbol_id = get_symbol_id(self.exchange, self.symbol)
                regime_start = load_last_bar_id(symbol_id=symbol_id, bar_type="time", bar_timeframe=timeframe)
                regime_start_dt = pd.to_datetime(regime_start, utc=True) if regime_start else None

                df_existing = read_ohlcv(
                    exchange=self.exchange,
                    symbol=self.symbol,
                    timeframe=timeframe,
                    start_date=regime_start_dt
                )
                if not df_existing.empty:
                    df_resampled = pd.concat([df_existing, df_resampled])

                # Ensure datetime is timezone-aware
                df_resampled['datetime'] = pd.to_datetime(df_resampled['datetime'], utc=True)

                # Keep the newly resampled version (last) not the stale DB version
                df_resampled = df_resampled.drop_duplicates(subset="datetime", keep='last').sort_values("datetime").reset_index(drop=True)


            # Check for incomplete candles and calculate regimes only for complete ones
            if last_datetime is not None:
                mask_complete = df_resampled['datetime'] + pd.Timedelta(minutes=timeframe_minutes) < last_datetime
                df_complete = df_resampled[mask_complete].copy()
                df_incomplete = df_resampled[~mask_complete].copy()
                logger.debug(
                    "[%s] %s: %d complete candles, %d incomplete (last_datetime=%s)",
                    self.symbol, timeframe, len(df_complete), len(df_incomplete), last_datetime
                )

            # Calculate regimes for complete candles
            if not df_complete.empty:
                logger.info(
                    "[%s] %s: calculating regimes for %d complete candles "
                    "(range %s → %s)",
                    self.symbol, timeframe, len(df_complete),
                    df_complete['datetime'].iloc[0], df_complete['datetime'].iloc[-1],
                )
                df_complete = calculate_regimes(df_complete, exchange=self.exchange, symbol=self.symbol,
                                               bar_type="time", bar_timeframe=timeframe)
                regime_counts = df_complete['regime_label'].value_counts().to_dict() if 'regime_label' in df_complete.columns else {}
                logger.info(
                    "[%s] %s: regime calculation done — labels: %s",
                    self.symbol, timeframe, regime_counts,
                )
            else:
                logger.debug("[%s] %s: no complete candles to calculate regimes for", self.symbol, timeframe)

            # Recombine, keeping regime columns only for complete candles
            df_resampled = pd.concat([df_complete, df_incomplete], ignore_index=True).sort_values('datetime').reset_index(drop=True)

            df_resampled["timestamp"] = datetime_to_timestamp(df_resampled["datetime"])

            df_resampled["datetime"] = pd.to_datetime(df_resampled["datetime"], utc=True).dt.floor('ms')
            # Insert/update and log

            self.insert_data(df_resampled, timeframe=timeframe, resample_insert = True)

            logger.info(
                "Processed %s:%s - Inserted/Updated %d candles",
                self.exchange,
                timeframe,
                len(df_resampled)
            )


    def run(
        self,
        run_mode: str,
        timeframes: Union[Dict, str]
    ) -> None:
        """
        Main execution method for fetching and processing data.
        
        Supports three run modes:
        - 'init': Full historical data fetch (first-time setup)
        - 'update': Incremental update (fetch new data since last run)
        - 'resample': Only resample existing 1m data to other timeframes
        
        Args:
            run_mode: One of "init", "update", or "resample"
            timeframes: Dictionary of time frames to resample or
                          JSON string representation of the dictionary
        """
        # Parse timeframes if provided as JSON string
        if isinstance(timeframes, str):
            timeframes = json.loads(timeframes)
        
        # Check for existing data
        last_datetime = get_ohlcv_last_datetime(
            exchange=self.exchange,
            symbol=self.symbol,
        )
        is_existing_data = last_datetime is not None
        
        # CASE 1: Resample mode
        if run_mode == "resample" and is_existing_data:
            logger.info("Resampling data started for %s", self.symbol)
            self.resample_and_insert_multiple(
                timeframes_dict=timeframes,
                last_datetime=last_datetime
            )
            return
        
        # CASE 2: Init mode (first-time fetch)
        elif run_mode == "init" and not is_existing_data:
            # Set start timestamp for historical data
            start_datetime = None
            logger.info("INIT: full fetch for %s", self.symbol)
        
        # CASE 3: Update mode (incremental fetch)
        elif run_mode == "update" and is_existing_data:
            # Fetch from 1 minute after last stored timestamp
            start_datetime = last_datetime

            logger.info(
                "UPDATE from %s for %s",
                start_datetime.isoformat() if start_datetime else None,
                self.symbol
            )
        
        # CASE 4: Invalid mode/state combination
        else:
            logger.info(
                "Skipping %s for run_mode: %s (existing_data: %s)",
                self.symbol,
                run_mode,
                is_existing_data
            )
            return
        
        # Set end timestamp to current time
        end_datetime = datetime.now(self.timezone)
        # Fetch data using exchange-specific implementation
        df_1m = self.fetch_ohlc(
            symbol=self.symbol,
            start_datetime=start_datetime,
            end_datetime=end_datetime
        )
        
        # Validate fetched data
        if df_1m is None or df_1m.empty:
            logger.warning("No data fetched for %s", self.symbol)
            return
        
        # Process the fetched data
        df_1m["datetime"] = timestamp_to_datetime(df_1m["timestamp"])

        df_1m = self.clean_data(df_1m)

        if last_datetime is not None:
            df_1m = df_1m[df_1m["datetime"] > last_datetime]

        # Insert into database
        self.insert_data(df_1m)
        
        logger.info(
            "Completed processing for %s (mode: %s)",
            self.symbol,
            run_mode,
        )