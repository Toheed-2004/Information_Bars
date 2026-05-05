"""
Blockchain Data Fetcher Module

Simplified module for fetching and processing blockchain data from blockchain.info API.
Stores data with Unix timestamps (seconds) for consistency with API.

This module handles:
- Fetching blockchain metrics from blockchain.info API
- Processing and cleaning the raw data
- Storing data in PostgreSQL database with timestamps
- Incremental updates to avoid duplicate data
"""

import logging
import time
from typing import Optional, Dict, List, Union, Any
from datetime import datetime, timedelta

import pandas as pd
import requests
from bitpredict.common.db.utils import insert_df, get_last_timestamp
from bitpredict.common.db.config import get_engine

# Configure module logger
logger = logging.getLogger(__name__)


class BlockchainDataManager:
    """
    Manages blockchain data fetching, processing, and storage.
    
    This class orchestrates the entire pipeline for fetching blockchain data
    from blockchain.info API, processing it, and storing it in a database.
    It supports multiple blockchain metrics and handles incremental updates.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the blockchain data manager with configuration.
        
        Args:
            config: Configuration dictionary containing chart definitions
                   with full metadata (including per-chart start/end dates)
                   and global settings.
        """
        self.config = config
        # Extract chart configuration organized by categories with full metadata
        self.charts_config = config.get('blockchain_charts', {})
        # Extract global settings
        self.settings = config.get('settings', {})
        # Database schema name for storing blockchain data
        self.schema_name = self.settings.get('schema_name', 'data_blockchain')
        # Global fallback start date for initial data fetch
        self.default_start_date = '2010-01-01'
        # Database engine for SQL operations
        self.engine = get_engine()
        # Batch size for continuous fetching (in days)
        self.batch_size_days = 365
        
        # Count total charts
        total_charts = sum(len(charts) for charts in self.charts_config.values())
        logger.debug(f"Initialized BlockchainDataManager with {len(self.charts_config)} categories, {total_charts} total charts")
    
    def get_chart_config(self, chart_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the full configuration for a specific chart.
        
        Args:
            chart_name: Name of the chart to find
            
        Returns:
            Chart metadata dict with start_date, end_date, etc., or None if not found
        """
        for category, charts in self.charts_config.items():
            for chart in charts:
                if isinstance(chart, dict) and chart.get("chart_name") == chart_name:
                    return chart
                elif isinstance(chart, str) and chart == chart_name:
                    # Fallback for backward compatibility
                    return {
                        "chart_name": chart_name,
                        "start_date": self.default_start_date,
                        "end_date": "now"
                    }
        return None
    
    def fetch_chart_data(self, chart_name: str, start_time: str = None, 
                        end_time: str = None, retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single chart from blockchain.info API.
        
        Args:
            chart_name: Name of the blockchain chart to fetch (e.g., 'market-price')
            start_time: Start date for data retrieval (YYYY-MM-DD format). 
                    If None, fetches all available data from default_start_date.
            end_time: End date for data retrieval (YYYY-MM-DD format). 
                    If None, fetches up to current date.
            retries: Number of retry attempts if API call fails
            
        Returns:
            pandas.DataFrame with columns: 'timestamp' (Unix seconds), 'datetime' (timestamp as datetime),
            and value column, or None if fetch fails after all retries
        """
        # Determine fetch range
        if start_time is None:
            start_time = self.default_start_date
            logger.info(f"{chart_name}: Performing full historical fetch from {start_time}")
        
        if end_time is None or end_time.lower() == 'now':
            end_time = datetime.now().strftime('%Y-%m-%d')
        
        # Convert to datetime objects for comparison
        start_dt = datetime.strptime(start_time, '%Y-%m-%d')
        end_dt = datetime.strptime(end_time, '%Y-%m-%d')
        
        # Calculate total days in range
        total_days = (end_dt - start_dt).days
        
        # If range is within batch size, fetch in one go
        if total_days <= self.batch_size_days:
            return self._fetch_single_batch(chart_name, start_time, end_time, retries)
        
        # Otherwise, break into multiple batches
        logger.info(f"{chart_name}: Large date range detected ({total_days} days). Breaking into {self.batch_size_days}-day batches.")
        
        all_dfs = []
        current_start = start_dt
        batch_number = 1
        empty_batch_count = 0
        max_empty_batches = 3  # Stop after 3 consecutive empty batches to avoid infinite loops
        
        while current_start < end_dt and empty_batch_count < max_empty_batches:
            # Calculate batch end date
            batch_end = min(current_start + timedelta(days=self.batch_size_days), end_dt)
            
            # Format dates for API
            start_str = current_start.strftime('%Y-%m-%d')
            end_str = batch_end.strftime('%Y-%m-%d')
            
            logger.info(f"{chart_name}: Batch {batch_number} - Fetching {start_str} to {end_str}")
            
            # Fetch batch data
            batch_df = self._fetch_single_batch(chart_name, start_str, end_str, retries)
            
            if batch_df is None:
                logger.error(f"{chart_name}: Failed to fetch batch {start_str} to {end_str}")
                # Don't break on API error, try next batch
                empty_batch_count += 1
            elif batch_df.empty:
                logger.info(f"{chart_name}: No data in batch {start_str} to {end_str}")
                # Empty batch is fine, just increment counter and continue
                empty_batch_count += 1
            else:
                all_dfs.append(batch_df)
                logger.info(f"{chart_name}: Batch {batch_number} complete - {len(batch_df)} records")
                empty_batch_count = 0  # Reset counter when we get data
            
            batch_number += 1
            
            # Move to next batch
            current_start = batch_end
            
            # Small delay to avoid rate limiting
            time.sleep(1)
        
        if empty_batch_count >= max_empty_batches:
            logger.warning(f"{chart_name}: Stopped after {max_empty_batches} consecutive empty batches")
        
        if not all_dfs:
            logger.warning(f"{chart_name}: No data fetched for any batch in range {start_time} to {end_time}")
            return None
        
        # Combine all batches
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove any duplicates that might span batch boundaries
        final_df = final_df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Sort by timestamp
        final_df = final_df.sort_values('timestamp')
        
        logger.info(f"{chart_name}: Fetch completed with {len(final_df)} total records in {batch_number-1} batches")
        return final_df
    
    def _fetch_single_batch(self, chart_name: str, start_time: str, 
                           end_time: str, retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Internal method to fetch a single batch of data.
        
        Args:
            chart_name: Name of the chart to fetch
            start_time: Start date for this batch
            end_time: End date for this batch
            retries: Number of retry attempts
            
        Returns:
            DataFrame with batch data or None if failed
        """
        for attempt in range(retries + 1):
            try:
                start_dt = datetime.strptime(start_time, '%Y-%m-%d')
                end_dt = datetime.strptime(end_time, '%Y-%m-%d')
                days_diff = (end_dt - start_dt).days
                
                # Construct timespan string (e.g., "365days")
                timespan = f"{days_diff}days"
                # Construct API URL with chart name and parameters
                params = {
                    'format': 'json',
                    'start': start_time,          # Keep start to pinpoint the beginning
                    'timespan': timespan,          # Use timespan to limit the range
                    'sampled': 'false'              # Add this to get all data points
                }
                
                url = f'https://api.blockchain.info/charts/{chart_name}'
                
                # Make HTTP GET request with timeout and parameters
                resp = requests.get(url, params=params, timeout=30)
                # Raise exception for HTTP errors (4xx, 5xx)
                resp.raise_for_status()
                
                # Parse response
                response_data = resp.json()

                # Check if values exist in response
                if 'values' not in response_data or not response_data['values']:
                    logger.info(f"{chart_name}: No data available for period {start_time} to {end_time}")
                    return pd.DataFrame()
                
                # Extract values array from JSON response
                data = response_data['values']
                
                # Extract timestamps (x) and values (y) from API response
                # Timestamps are Unix timestamps in seconds
                timestamps = [d['x'] for d in data]
                values = [d['y'] for d in data]
                
                # Convert chart name to database-safe column name (replace hyphens with underscores)
                column_name = chart_name.replace('-', '_')
                
                # Create DataFrame with timestamp, datetime, and value columns
                df = pd.DataFrame({
                    'timestamp': timestamps,  # Unix timestamp in seconds
                    'datetime': pd.to_datetime(timestamps, unit='s'),  # Datetime column for TimescaleDB
                    column_name: values       # Metric value
                })
                logger.info(f"{chart_name}: Fetched {len(df)} records for {start_time} to {end_time}")
                return df
                
            except requests.exceptions.RequestException as e:
                # Handle API failures with retry logic
                if attempt >= retries:
                    logger.error(f"{chart_name}: Failed after {retries} retries - {e}")
                    return None
                logger.warning(f"{chart_name}: Attempt {attempt + 1} failed - {e}")
                # Wait before retry with exponential backoff
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"{chart_name}: Unexpected error - {e}")
                return None
    
    def _prepare_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data for database insertion.
        
        Args:
            data_df: Raw DataFrame from API with 'timestamp', 'datetime', and value columns
            
        Returns:
            Cleaned DataFrame ready for database storage with:
            - Duplicate timestamps removed
            - Data sorted by timestamp
            - Numeric values rounded to 2 decimal places
        """
        if data_df.empty:
            return data_df
        
        # Work on a copy to avoid modifying original data
        data_df = data_df.copy()
        
        # Remove duplicate timestamps (keep first occurrence)
        # Prevents primary key violations in database
        data_df = data_df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Sort by timestamp for chronological order
        data_df = data_df.sort_values('timestamp')
        
        # Round numeric columns to 2 decimal places for consistency
        numeric_cols = data_df.select_dtypes(include=['float64', 'int64']).columns
        # Exclude timestamp and datetime columns from rounding
        numeric_cols = [col for col in numeric_cols if col not in ['timestamp', 'datetime']]
        if numeric_cols:
            data_df[numeric_cols] = data_df[numeric_cols].round(2)
        
        return data_df
    
    def save_chart_data(self, chart_name: str, data_df: pd.DataFrame) -> int:
        """
        Save chart data to database.
        
        Args:
            chart_name: Name of the chart being saved
            data_df: DataFrame containing data to save
            
        Returns:
            Number of records inserted into database (0 if no new data)
        """
        if data_df.empty:
            logger.info(f"{chart_name}: No data to save")
            return 0
        
        # Prepare data for database insertion
        data_df = self._prepare_data(data_df)
        
        if data_df.empty:
            logger.info(f"{chart_name}: No data after preparation")
            return 0
        
        # Generate database table name from chart name
        # Format: onchain_{chart_name_with_underscores}
        table_name = f"onchain_{chart_name.lower().replace('-', '_')}"
        
        # Get last saved timestamp to avoid duplicate inserts
        last_timestamp = get_last_timestamp(
            engine=self.engine, 
            schema_name=self.schema_name,
            table_name=table_name 
        )

        # Filter out records with timestamp <= last_timestamp
        # This ensures only new data is inserted
        if last_timestamp:
            data_df = data_df[data_df['timestamp'] > last_timestamp]
        
        if data_df.empty:
            logger.info(f"{chart_name}: No new data")
            return 0
        
        # Insert data into database
        insert_df(
            data_df, 
            engine=self.engine, 
            table_name=table_name, 
            schema_name=self.schema_name,
            is_timeseries=True
        )
        
        logger.info(f"{chart_name}: Inserted {len(data_df)} records")
        return len(data_df)
    
    def process_chart(self, chart_name: str) -> int:
        """
        Fetch and save data for a single chart (complete pipeline).
        
        Args:
            chart_name: Name of the chart to process
            
        Returns:
            Number of records inserted into database
            
        Note:
            This method handles the complete workflow:
            1. Get chart-specific configuration (start_date, end_date)
            2. Check for existing data to determine fetch strategy
            3. Fetch data from API (full history if no data exists, incremental otherwise)
            4. Save to database
        """
        # Get chart-specific configuration
        chart_config = self.get_chart_config(chart_name)
        if not chart_config:
            logger.error(f"{chart_name}: No configuration found")
            return 0
        
        # Get chart-specific dates
        chart_start_date = chart_config.get("start_date", self.default_start_date)
        chart_end_date = chart_config.get("end_date", "now")
        
        logger.info(f"{chart_name}: Configured range {chart_start_date} to {chart_end_date}")
        
        # Generate database table name for this chart
        table_name = f"onchain_{chart_name.lower().replace('-', '_')}"
        
        # Get last saved timestamp to determine fetch strategy
        last_timestamp = get_last_timestamp(
            engine=self.engine, 
            schema_name=self.schema_name,
            table_name=table_name 
        )
        
        if last_timestamp is None:
            # No data exists - perform full historical fetch using chart-specific start_date
            logger.info(f"{chart_name}: No existing data found. Performing full historical fetch from {chart_start_date}")
            data_df = self.fetch_chart_data(
                chart_name, 
                start_time=chart_start_date,
                end_time=chart_end_date
            )
        else:
            # Data exists - fetch only new records since last timestamp
            last_date = datetime.utcfromtimestamp(last_timestamp).strftime('%Y-%m-%d')
            logger.info(f"{chart_name}: Existing data found. Fetching new records since {last_date} up to {chart_end_date}")
            data_df = self.fetch_chart_data(
                chart_name, 
                start_time=last_date,
                end_time=chart_end_date
            )
        
        if data_df is None or data_df.empty:
            logger.warning(f"{chart_name}: No data fetched")
            return 0
        
        # Save fetched data to database
        return self.save_chart_data(chart_name, data_df)
    
    def update_all(self) -> Dict[str, int]:
        """
        Update all charts from their configured start dates.
        
        Returns:
            Dictionary mapping chart names to number of new records inserted
            
        Note:
            This method processes all charts, automatically determining whether
            to do a full download or incremental update based on existing data.
            It uses per-chart start_date and end_date from the configuration.
        """
        results = {}
        
        # Process each chart in each category
        for category, charts_list in self.charts_config.items():
            for chart_item in charts_list:
                # Handle both string chart names and dict chart configs
                if isinstance(chart_item, dict):
                    chart_name = chart_item.get("chart_name")
                else:
                    chart_name = chart_item
                
                if not chart_name:
                    continue
                    
                try:
                    # Process chart with its specific configuration
                    count = self.process_chart(chart_name)
                    results[chart_name] = count
                except Exception as e:
                    logger.error(f"Error updating {chart_name}: {e}")
                    results[chart_name] = 0
        
        return results