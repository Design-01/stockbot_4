
from datetime import datetime, timedelta
import pytz
from dateutil.relativedelta import relativedelta
from twelvedata import TDClient
import pandas as pd
import os
import csv
from typing import Tuple, List , Optional, Union
import time
from ib_insync import IB, Stock, util

import pandas as pd
import re

from project_paths import get_project_path


def map_to_storage_interval(interval: str, source: str = 'ib') -> str:
    """
    Maps intervals from various data sources to standardized storage intervals.
    
    Args:
        interval (str): The interval string from the data source
        source (str): The source of the interval ('ib' for Interactive Brokers or 'twelve' for Twelve Data)
    
    Returns:
        str: Mapped storage interval ('1_minute', '1_hour', or '1_day')
    
    Raises:
        ValueError: If the interval is invalid or the source is not recognized
    """
    # Validate source
    if source not in ('ib', 'twelve'):
        raise ValueError("Source must be either 'ib' or 'twelve'")
    
    # Convert to lowercase for case-insensitive comparison
    interval = interval.lower()
    
    # Define mapping rules
    ib_minute_intervals = {'1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
                          '1 min', '2 mins', '3 mins', '5 mins', '10 mins',
                          '15 mins', '20 mins', '30 mins'}
    
    ib_hour_intervals = {'1 hour', '2 hours', '3 hours', '4 hours', '8 hours'}
    
    ib_day_intervals = {'1 day', '1w', '1m'}
    
    twelve_minute_intervals = {'1min', '5min', '15min', '30min', '45min'}
    
    twelve_hour_intervals = {'1h', '2h', '4h', '8h'}
    
    twelve_day_intervals = {'1day', '1week', '1month'}
    
    # Map intervals based on source
    if source == 'ib':
        if interval in {i.lower() for i in ib_minute_intervals}:
            return '1_min'
        elif interval in {i.lower() for i in ib_hour_intervals}:
            return '1_hour'
        elif interval in {i.lower() for i in ib_day_intervals}:
            return '1_day'
    else:  # twelve data
        if interval in {i.lower() for i in twelve_minute_intervals}:
            return '1_min'
        elif interval in {i.lower() for i in twelve_hour_intervals}:
            return '1_hour'
        elif interval in {i.lower() for i in twelve_day_intervals}:
            return '1_day'
    
    raise ValueError(f"Invalid interval '{interval}' for source '{source}'")


def calculate_past_date(start_date, end_date):
    """
    Calculate a past date relative to a specified end date.
    
    Args:
        start_date: Can be either:
            - datetime object
            - string in format 'YYYY-MM-DD'
            - string in format '5 daysAgo' or '3 weeksAgo'
        end_date: Required reference date:
            - string 'now' for current datetime
            - string in format 'YYYY-MM-DD'
    
    Returns:
        str: Formatted date string in 'YYYY-MM-DD HH:MM:SS' format
    """
    # Set reference date (end_date)
    if end_date == 'now':
        reference_date = datetime.now()
    else:
        try:
            reference_date = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("end_date must be 'now' or in 'YYYY-MM-DD' format")

    # Handle start_date as datetime object
    if isinstance(start_date, datetime):
        return start_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Handle relative time string (e.g., '5 daysAgo')
    pattern = r'(\d+)\s*(days?Ago|weeks?Ago)'
    match = re.match(pattern, start_date, re.IGNORECASE)
    
    if match:
        number = int(match.group(1))
        unit = match.group(2).lower()
        
        if 'week' in unit:
            # Go back specified weeks and find Monday
            past_date = reference_date - timedelta(weeks=number)
            days_since_monday = past_date.weekday()
            past_date = past_date - timedelta(days=days_since_monday)
        else:  # days
            past_date = reference_date - timedelta(days=number)
            
        past_date = past_date.replace(hour=0, minute=0, second=1, microsecond=0)
        return past_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Try parsing as regular date string
    try:
        parsed_date = datetime.strptime(start_date, '%Y-%m-%d')
        return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        raise ValueError("start_date must be a datetime, 'YYYY-MM-DD', or 'X daysAgo/weeksAgo'")

class IntervalHandler:
    # Define standard mappings
    TIME_MAPPINGS = {
        # Standard time unit mappings
        'sec': 'S',
        'secs': 'S',
        'second': 'S',
        'seconds': 'S',
        'min': 'T',
        'mins': 'T',
        'minute': 'T',
        'minutes': 'T',
        'h': 'H',
        'hour': 'H',
        'hours': 'H',
        'day': 'D',
        'days': 'D',
        'W': 'W',
        'week': 'W',
        'M': 'M',
        'month': 'M',
        # Additional IB specific mappings
        '1W': 'W',
        '1M': 'M'
    }
    
    # Known intervals for validation
    IB_INTERVALS = {
        '1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
        '1 min', '2 mins', '3 mins', '5 mins', '10 mins',
        '15 mins', '20 mins', '30 mins', '1 hour', '2 hours',
        '3 hours', '4 hours', '8 hours', '1 day', '1W', '1M'
    }
    
    TWELVE_DATA_INTERVALS = {
        '1min', '5min', '15min', '30min', '45min', '1h', '2h',
        '4h', '8h', '1day', '1week', '1month'
    }
    
    @staticmethod
    def parse_interval(interval):
        """
        Parse interval string to get number and unit
        
        Parameters:
        interval (str): Interval string in either IB or 12data format
        
        Returns:
        tuple: (number, pandas_freq_str)
        """
        # Remove any whitespace and convert to lowercase
        interval = interval.lower().strip()
        
        # Try to extract number and unit using regex
        match = re.match(r'^(\d+)?\s*([a-zA-Z]+)$', interval)
        if not match:
            raise ValueError(f"Invalid interval format: {interval}")
        
        number, unit = match.groups()
        number = '1' if number is None else number
        number = int(number)
        
        # Handle special cases
        if interval == '1w':
            return 1, 'W'
        if interval == '1m':
            return 1, 'M'
            
        # Get pandas frequency string
        pandas_freq = IntervalHandler.TIME_MAPPINGS.get(unit)
        if pandas_freq is None:
            raise ValueError(f"Unsupported time unit: {unit}")
            
        return number, pandas_freq
    
    @staticmethod
    def convert_to_pandas_freq(interval):
        """
        Convert interval string to pandas frequency string
        
        Parameters:
        interval (str): Interval string in either IB or 12data format
        
        Returns:
        str: Pandas frequency string
        """
        number, freq = IntervalHandler.parse_interval(interval)
        return f"{number}{freq}"
    
    @staticmethod
    def resample_data(df, interval):
        """
        Resample dataframe to new interval
        
        Parameters:
        df (pd.DataFrame): DataFrame with datetime index
        interval (str): Target interval in either IB or 12data format
        
        Returns:
        pd.DataFrame: Resampled DataFrame
        """
        # Validate input DataFrame
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
            
        # Convert interval to pandas frequency
        pandas_freq = IntervalHandler.convert_to_pandas_freq(interval)
        
        # Perform resampling
        resampled = df.resample(pandas_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled

# Example usage
def resample_to_interval(df, interval):
    """
    Wrapper function to resample data to given interval
    
    Parameters:
    df (pd.DataFrame): DataFrame with datetime index
    interval (str): Target interval in either IB or 12data format
    
    Returns:
    pd.DataFrame: Resampled DataFrame
    """
    return IntervalHandler.resample_data(df, interval)

class BaseHistoricalData:

    def get_batch_dates(self, start_date, end_date, batch:str='monthly'):
        """ Generate a list of dates in a given batch whereby batch is either weekly, monthly or yearly. 
        each batch will be rounded to the nearest week, month or year.
        
        Examples: 
        batch_dates('2024-01-10', '2021-01-20', batch='monthly') --> [(2024-01-01, 2024-01-31)]
        batch_dates('2024-01-10', '2021-01-20', batch='weekly') --> [(2024-01-08, 2024-01-14), (2024-01-15, 2024-01-21)] # Week is from Mon to Sun
        batch_dates('2024-01-10', '2021-01-20', batch='yearly') --> [(2024-01-01, 2024-12-31)]
        batch_dates('2024-01-10', '2021-03-20', batch='monthly') --> [(2024-01-01, 2024-01-31), (2024-02-01, 2024-02-29), (2024-03-01, 2024-03-31)]
        """ 

        start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        
        if batch == 'monthly':
            start_date = start_date.replace(day=1)
            end_date = end_date.replace(day=1) + relativedelta(months=1, days=-1)
            dates = []
            while start_date <= end_date:
                month_end = start_date + relativedelta(day=31)
                dates.append((start_date.strftime('%Y-%m-%d'), month_end.strftime('%Y-%m-%d')))
                start_date = start_date + relativedelta(months=1)
            return dates
        
        if batch == 'weekly':
            start_date = start_date - timedelta(days=start_date.weekday())
            end_date = end_date - timedelta(days=end_date.weekday())
            dates = []
            while start_date <= end_date:
                dates.append((start_date.strftime('%Y-%m-%d'), (start_date + timedelta(days=6)).strftime('%Y-%m-%d')))
                start_date += timedelta(days=7)
            return dates
        
        if batch == 'yearly':
            start_date = start_date.replace(month=1, day=1)
            end_date = end_date.replace(month=12, day=31)
            return [(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))]
        
        return None
    

"https://github.com/twelvedata/twelvedata-python"

class TwelveDataHistoricalData(BaseHistoricalData):
    def __init__(self, api_key):
        self.td = TDClient(apikey=api_key)

    def get_historical_data(self, symbol, end_date, duration, barsize):
        data =  self.td.time_series(
            symbol=symbol,
            interval=barsize,
            start_date=end_date,
            end_date=end_date,
            outputsize=5000,
            timezone="America/New_York",
        ).as_pandas().sort_index(ascending=True)
        return data

    def get_next_lower_common_denominator_as_interval(self, interval, minHourDay_only=True):
            # Define available intervals based on minHourDay_only
            if minHourDay_only:
                available_intervals = ['1min', '1h', '1day']
                interval_map = {
                    '1min': 1,
                    '1h': 60,
                    '1day': 1440
                }
            else:
                available_intervals = ['1min', '5min', '15min', '30min', '45min', '1h', '2h', '4h', '8h', '1day', '1week', '1month']
                interval_map = {
                    '1min': 1,
                    '5min': 5,
                    '15min': 15,
                    '30min': 30,
                    '45min': 45,
                    '1h': 60,
                    '2h': 120,
                    '4h': 240,
                    '8h': 480,
                    '1day': 1440,
                    '1week': 10080,
                    '1month': 43200
                }
            
            # Convert input interval to minutes
            if 'min' in interval:
                interval_minutes = int(interval.replace('min', ''))
            elif 'h' in interval:
                interval_minutes = int(interval.replace('h', '')) * 60
            elif 'day' in interval:
                interval_minutes = int(interval.replace('day', '')) * 1440
            elif 'week' in interval:
                interval_minutes = int(interval.replace('week', '')) * 10080
            elif 'month' in interval:
                interval_minutes = int(interval.replace('month', '')) * 43200
            else:
                raise ValueError("Invalid interval format")
            
            # Find the largest available interval that is less than or equal to the input interval
            next_lower_interval = available_intervals[0]  # Default to smallest interval
            for available_interval in available_intervals:
                if interval_map[available_interval] <= interval_minutes:
                    if minHourDay_only:
                        # When minHourDay_only is True, we don't need to check if it's divisible
                        next_lower_interval = available_interval
                    else:
                        # When minHourDay_only is False, we check if it's divisible
                        if interval_minutes % interval_map[available_interval] == 0:
                            next_lower_interval = available_interval
            
            return next_lower_interval

    def get_batch_size(self, interval):
        """available_intervals = ['1min', '5min', '15min', '30min', '45min', '1h', '2h', '4h', '8h', '1day', '1week', '1month']
            max_rows = 5000"""
        sizes = {
                '1min': 'weekly',   # 1 minute interval, 5000 rows = ~5.2 days of trading data (considering 16 hours/day)
                '5min': 'monthly',  # 5 minute interval, 5000 rows = ~26 days of trading data
                '15min': 'monthly', # 15 minute interval, 5000 rows = ~78 days of trading data
                '30min': 'monthly', # 30 minute interval, 5000 rows = ~156 days of trading data
                '45min': 'monthly', # 45 minute interval, 5000 rows = ~234 days of trading data
                '1h': 'monthly',    # 1 hour interval, 5000 rows = ~312 days of trading data
                '2h': 'yearly',     # 2 hour interval, 5000 rows = ~625 days of trading data
                '4h': 'yearly',     # 4 hour interval, 5000 rows = ~1250 days of trading data
                '8h': 'yearly',     # 8 hour interval, 5000 rows = ~2500 days of trading data
                '1day': 'yearly',   # 1 day interval, 5000 rows = ~5000 days of trading data
                '1week': 'yearly',  # 1 week interval, 5000 rows = ~5000 weeks of trading data
                '1month': 'yearly'  # 1 month interval, 5000 rows = ~5000 months of trading data
            }
        return sizes[interval]


class IBHistoricalData(BaseHistoricalData):
    def __init__(self, host='127.0.0.1', port=7496, client_id=1):
        self.ib = None
        self.host = host
        self.port = port
        self.client_id = client_id
        self.data = None
        self.batch_data = []
        self.available_intervals = [
                '1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
                '1 min', '2 mins', '3 mins', '5 mins', '10 mins', 
                '15 mins', '20 mins', '30 mins', '1 hour', '2 hours',
                '3 hours', '4 hours', '8 hours', '1 day', '1W', '1M']
            
    def is_running_in_notebook(self):
                try:
                    from IPython import get_ipython
                    shell = get_ipython().__class__.__name__
                    return shell == 'ZMQInteractiveShell'
                except (NameError, ImportError):
                    return False

    def _format_datetime(self, date_str):
        """
        Format datetime string to IB's expected format: YYYYMMDD HH:MM:SS US/Eastern
        
        Args:
            date_str (str): Date string in format 'YYYY-MM-DD'
        
        Returns:
            str: Formatted datetime string
        """
        # Convert to datetime object first to ensure proper formatting
        dt = pd.Timestamp(date_str)
        # Format as YYYYMMDD HH:MM:SS US/Eastern
        return f"{dt.strftime('%Y%m%d')} 23:59:59 US/Eastern"
    
    def process_ib_data(self, df):
        """
        Process IB historical data by setting proper datetime index and cleaning columns.
        
        Args:
            df (pandas.DataFrame): Raw dataframe from IB historical data request
        
        Returns:
            pandas.DataFrame: Processed dataframe with proper datetime index
        """
        # Convert date column to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Sort index in ascending order
        df = df.sort_index()
        
        # Optionally: Remove average and barCount columns if not needed
        # df = df.drop(['average', 'barCount'], axis=1)
        
        return df

    def get_batched_data_as_combined_df(self):
        # Concatenate all dataframes and sort by date
            combined_data = pd.concat(self.batch_data)
            return self.process_ib_data(combined_data)

    def get_historical_data(self, symbol, end_date, duration, barsize):
        self.ib = IB()
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        contract = Stock(symbol, 'SMART', 'USD')
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_date,
            barSizeSetting=barsize,
            durationStr=duration,
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )
        self.ib.disconnect()
        self.data = util.df(bars)
        return self.data
    
    def get_batch_historical_data(self, symbol, date_ranges, barsize='1 day', minHourDay_only=True):
        """
        Fetch historical data for multiple date ranges and concatenate the results.
        Automatically handles Jupyter notebook event loop if needed.
        
        Args:
            symbol (str): Stock symbol
            date_ranges (list): List of tuples containing (start_date, end_date)
            barsize (str): Bar size setting (default: '1 day')
        
        Returns:
            tuple: (pandas.DataFrame, str) - Concatenated historical data and interval used
        """
 
        
        # Start IB's event loop if we're in a notebook
        if self.is_running_in_notebook():
            util.startLoop()
        
        self.batch_data = []
        lowest_interval = self.get_next_lower_common_denominator_as_interval(barsize, minHourDay_only=minHourDay_only)
        batch_size = self.get_batch_size(lowest_interval)
        converted_batch_dates = convert_ranges(date_ranges, batch_size)

        
        def get_duration(delta):
            if delta.days <= 1:
                return '1 D'
            elif delta.days <= 7:
                return '1 W'
            elif delta.days <= 31:
                return '1 M'
            elif delta.days <= 365:
                return '1 Y'
            else:
                return '2 Y'  # Add support for longer durations if needed
            
        
        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            contract = Stock(symbol, 'SMART', 'USD')
            
            for start_date, end_date in converted_batch_dates:
                start = pd.Timestamp(start_date)
                end = pd.Timestamp(end_date)
                delta = end - start
                duration = get_duration(delta)
                end_date = self._format_datetime(end_date)

                # rth  = False if not intraday
                useRTH = True if lowest_interval  in ['1 day', '1W', '1M'] else False
                
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_date,
                    barSizeSetting=lowest_interval,
                    durationStr=duration,
                    whatToShow='TRADES',
                    useRTH=useRTH,
                    formatDate=1
                )
                
                if bars:
                    df = util.df(bars)
                    self.batch_data.append(df)
                
                self.ib.sleep(1)
                
        finally:
            if hasattr(self, 'ib') and self.ib.isConnected():
                self.ib.disconnect()
        
        if self.batch_data:
            return self.get_batched_data_as_combined_df(), lowest_interval
        
        return pd.DataFrame(), lowest_interval
    
    def get_next_lower_common_denominator_as_interval(self, interval, minHourDay_only=True):
        # Define available intervals based on minHourDay_only
        if minHourDay_only:
            available_intervals = ['1 min', '1 hour', '1 day']
            interval_map = {
                '1 min': 1,
                '1 hour': 60,
                '1 day': 1440
            }
        else:
            available_intervals = [
                '1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
                '1 min', '2 mins', '3 mins', '5 mins', '10 mins', 
                '15 mins', '20 mins', '30 mins', '1 hour', '2 hours',
                '3 hours', '4 hours', '8 hours', '1 day', '1W', '1M'
            ]
            interval_map = {
                '1 secs': 1/60,
                '5 secs': 5/60,
                '10 secs': 10/60,
                '15 secs': 15/60,
                '30 secs': 30/60,
                '1 min': 1,
                '2 mins': 2,
                '3 mins': 3,
                '5 mins': 5,
                '10 mins': 10,
                '15 mins': 15,
                '20 mins': 20,
                '30 mins': 30,
                '1 hour': 60,
                '2 hours': 120,
                '3 hours': 180,
                '4 hours': 240,
                '8 hours': 480,
                '1 day': 1440,
                '1W': 10080,
                '1M': 43200
            }
        
        # Convert input interval to minutes
        if 'secs' in interval:
            interval_minutes = float(interval.replace('secs', '').strip()) / 60
        elif 'mins' in interval or 'min' in interval:
            interval_minutes = float(interval.replace('mins', '').replace('min', '').strip())
        elif 'hours' in interval or 'hour' in interval:
            interval_minutes = float(interval.replace('hours', '').replace('hour', '').strip()) * 60
        elif 'day' in interval:
            interval_minutes = float(interval.replace('day', '').strip()) * 1440
        elif 'W' in interval:
            interval_minutes = float(interval.replace('W', '').strip()) * 10080
        elif 'M' in interval:
            interval_minutes = float(interval.replace('M', '').strip()) * 43200
        else:
            raise ValueError("Invalid interval format")
        
        # Find the largest available interval that is less than or equal to the input interval
        next_lower_interval = available_intervals[0]  # Default to smallest interval
        for available_interval in available_intervals:
            if interval_map[available_interval] <= interval_minutes:
                if minHourDay_only:
                    # When minHourDay_only is True, we don't need to check if it's divisible
                    next_lower_interval = available_interval
                else:
                    # When minHourDay_only is False, we check if it's divisible
                    if abs(interval_minutes % interval_map[available_interval]) < 0.0001:  # Using small epsilon for float comparison
                        next_lower_interval = available_interval
        
        return next_lower_interval
    
    def get_batch_size(self, interval):
        """
        Returns the appropriate batch size for downloading historical data from Interactive Brokers
        based on the specified interval.
        
        Parameters:
        interval (str): Time interval for the data
            Possible values: '1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
            '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins',
            '30 mins', '1 hour', '2 hours', '3 hours', '4 hours', '8 hours', '1 day',
            '1W', '1M'
        
        Returns:
        str: Batch size period ('weekly', 'monthly', 'quarterly', 'yearly')
        """
        # Convert all intervals to standardized format for comparison
        sizes = {
            # Seconds intervals (limited to weekly due to data volume)
            '1 secs': 'weekly',
            '5 secs': 'weekly',
            '10 secs': 'weekly',
            '15 secs': 'weekly',
            '30 secs': 'weekly',
            
            # Minute intervals
            '1 min': 'weekly',    # Max 7 days
            '2 mins': 'weekly',
            '3 mins': 'weekly',
            '5 mins': 'monthly',  # Max 30 days
            '10 mins': 'monthly',
            '15 mins': 'quarterly',  # Max 60 days
            '20 mins': 'quarterly',
            '30 mins': 'quarterly',
            
            # Hour intervals
            '1 hour': 'yearly',   # Max 365 days
            '2 hours': 'yearly',
            '3 hours': 'yearly',
            '4 hours': 'yearly',
            '8 hours': 'yearly',
            
            # Day and above
            '1 day': 'yearly',    # Max 20 years
            '1W': 'yearly',
            '1M': 'yearly'
        }
        
        if interval not in sizes:
            raise ValueError(f"Invalid interval: {interval}. Please choose from: {', '.join(sizes.keys())}")
        
        return sizes[interval]
    



def resample_data(data, interval):
    return data.resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()



def save_data(data, symbol, interval):
    """
    Save data to CSV with consistent datetime index handling.
    
    Parameters:
    data (pd.DataFrame): DataFrame to save
    symbol (str): Symbol identifier
    interval (str): Data interval identifier
    """
    # Make a copy to avoid modifying the original DataFrame
    df = data.copy()
    
    # Case 1: DataFrame has no index name but has datetime index
    if df.index.name is None and isinstance(df.index, pd.DatetimeIndex):
        df.index.name = 'date'
    
    # Case 2: DataFrame has named index that's already datetime
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index.name = 'date'  # Standardize the name
    
    # Case 3: Need to set datetime index from columns
    else:
        # Look for datetime column with various possible names
        date_columns = [col for col in df.columns if col.lower() in ['date', 'datetime', 'time', 'timestamp']]
        
        if not date_columns:
            raise ValueError("No datetime column found in DataFrame")
        
        # Use the first found date column
        date_col = date_columns[0]
        
        # Convert to datetime if not already
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise ValueError(f"Failed to convert {date_col} to datetime: {str(e)}")
        
        # Set index
        df.set_index(date_col, inplace=True)
        df.index.name = 'date'
    
    # Sort index
    df = df.sort_index(ascending=True)
    
    # Remove any duplicates in the index
    df = df[~df.index.duplicated(keep='last')]
    
    # Save with date format that excludes time information if it exists
    interval = interval.lower().replace(' ', '_')
    filename = get_project_path('data', 'historical_data_store', f'{symbol}_{interval}.csv')
    df.to_csv(filename)


def load_data(symbol, interval):
    """
    Load data from CSV with datetime index.
    
    Parameters:
    symbol (str): Symbol identifier
    interval (str): Data interval identifier
    
    Returns:
    pd.DataFrame or None: Loaded data or None if file doesn't exist
    """
    interval = interval.lower().replace(' ', '_')
    file_path = get_project_path('data', 'historical_data_store', f'{symbol}_{interval}.csv')
    if not os.path.exists(file_path):
        print(f"File not found : {file_path}")
        return None
        
    data = pd.read_csv(file_path, 
                      index_col='date', 
                      parse_dates=True)
    
    # Remove duplicates and sort
    data = data[~data.index.duplicated(keep='last')].sort_index(ascending=True)
    
    return data


def get_missing_batch_dates(data, start_date, end_date, batch_interval='weekly'):
    """
    Find truly missing date intervals in market data, handling:
    - Non-overlapping date ranges
    - Market closures (partial data in a batch is considered complete)
    - Current week needing updates
    
    Parameters:
    data (pd.DataFrame): DataFrame with datetime index
    start_date (str): Start date 'YYYY-MM-DD' or relative date like '3 daysAgo'
    end_date (str): End date 'YYYY-MM-DD' or 'now'
    batch_interval (str): 'weekly' or 'monthly'
    """
    def parse_dates(start_str, end_str):
        """Convert and validate date strings, handling relative dates"""
        def parse_relative_date(date_str):
            if isinstance(date_str, str) and 'daysAgo' in date_str:
                days = int(date_str.split()[0])
                return pd.Timestamp.now() - pd.Timedelta(days=days)
            return pd.to_datetime(date_str)
        
        start = parse_relative_date(start_str).normalize()
        end = (pd.Timestamp.now() if isinstance(end_str, str) and end_str.lower() == 'now' 
               else parse_relative_date(end_str)).normalize()
        return start, end

    def get_batch_config(interval):
        """Get frequency and duration for given interval"""
        configs = {
            'weekly': ('W-MON', pd.Timedelta(days=6)),
            'monthly': ('MS', pd.offsets.MonthEnd(1))
        }
        if interval.lower() not in configs:
            raise ValueError("batch_interval must be 'weekly' or 'monthly'")
        return configs[interval.lower()]

    try:
        # Parse dates and get batch configuration
        start, end = parse_dates(start_date, end_date)
        freq, duration = get_batch_config(batch_interval)

        # Handle empty data case
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            return [(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))]

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be DatetimeIndex")

        # Get data range
        data_start = data.index.min()
        data_end = data.index.max()

        # Check for non-overlapping ranges
        if data_start > end or data_end < start:
            print(f"No overlap between requested dates ({start} to {end}) "
                  f"and stored dates ({data_start} to {data_end})")
            return [(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))]

        # Generate batch intervals
        missing_intervals = []
        batch_starts = pd.date_range(start=start, end=end, freq=freq)

        for batch_start in batch_starts:
            batch_end = min(batch_start + duration, end)
            
            # Check for data in this batch
            mask = (data.index >= batch_start) & (data.index <= batch_end)
            has_data = data[mask].size > 0
            
            # Consider batch missing if:
            # 1. No data at all in batch
            # 2. OR batch overlaps with current period and is more recent than stored data
            if (not has_data or 
                (batch_end >= data_end and batch_end <= pd.Timestamp.now())):
                missing_intervals.append(
                    (batch_start.strftime('%Y-%m-%d'),
                     batch_end.strftime('%Y-%m-%d'))
                )

        return missing_intervals

    except Exception as e:
        print(f"Error in get_missing_batch_dates: {str(e)}")
        return [(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))]
        



def convert_ranges(ranges, new_interval='monthly', start_time_offset='00:00:01'):
    """
    Convert a list of date ranges to a new interval (weekly, monthly, or yearly).
    Adds a time offset to start dates to avoid midnight boundary issues.
    
    Args:
        ranges (list): List of (start_date, end_date) tuples
        new_interval (str): 'weekly', 'monthly', or 'yearly'
        start_time_offset (str): Time to offset start dates to avoid midnight issues
    
    Returns:
        list: List of (start_date, end_date) tuples in the new interval
    """
    if not ranges:
        print("No ranges provided")
        return []
    
    if new_interval.lower() not in ['weekly', 'monthly', 'yearly']:
        raise ValueError("new_interval must be either 'weekly', 'monthly', or 'yearly'")
    
    try:
        # Convert all dates to datetime objects
        all_dates = []
        for start, end in ranges:
            start_dt = pd.to_datetime(f"{start} {start_time_offset}")
            end_dt = pd.to_datetime(f"{end} 23:59:59")
            all_dates.extend([start_dt, end_dt])
        
        min_date = min(all_dates)
        max_date = max(all_dates)

        
        # If the range is shorter than the interval period, return the original range
        if new_interval.lower() == 'weekly' and (max_date - min_date) < pd.Timedelta(days=7):
            return [(min_date.strftime('%Y-%m-%d %H:%M:%S'),
                    max_date.strftime('%Y-%m-%d %H:%M:%S'))]
        
        if new_interval.lower() == 'monthly' and (max_date - min_date) < pd.Timedelta(days=28):
            return [(min_date.strftime('%Y-%m-%d %H:%M:%S'),
                    max_date.strftime('%Y-%m-%d %H:%M:%S'))]
        
        if new_interval.lower() == 'yearly' and (max_date - min_date) < pd.Timedelta(days=365):
            return [(min_date.strftime('%Y-%m-%d %H:%M:%S'),
                    max_date.strftime('%Y-%m-%d %H:%M:%S'))]
        
        # Generate intervals based on the type
        if new_interval.lower() == 'weekly':
            # Generate all weeks between start and end
            # Use W-MON to ensure weeks start on Monday
            dates = pd.date_range(
                start=min_date,
                end=max_date,
                freq='W-MON',
                inclusive='both'
            )
        elif new_interval.lower() == 'monthly':
            # Generate all months between start and end
            dates = pd.date_range(
                start=min_date,
                end=max_date,
                freq='MS',  # Month Start
                inclusive='both'
            )
        elif new_interval.lower() == 'yearly':
            # For yearly, we want to include partial years
            years = list(set([min_date.year, max_date.year]))  # Get unique years
            years.sort()
            dates = [pd.Timestamp(f"{year}-01-01") for year in years]
            
            # Adjust the first date to match the actual start if it's mid-year
            if dates[0] < min_date:
                dates[0] = min_date
        
        # Create ranges between the generated dates
        new_ranges = []
        for i in range(len(dates)-1):
            range_start = dates[i]
            # End one second before the next period starts
            range_end = dates[i+1] - pd.Timedelta(seconds=1)
            new_ranges.append((
                range_start.strftime('%Y-%m-%d %H:%M:%S'),
                range_end.strftime('%Y-%m-%d %H:%M:%S')
            ))
        
        # Add the final period
        if len(dates) > 0:
            final_start = dates[-1]
            new_ranges.append((
                final_start.strftime('%Y-%m-%d %H:%M:%S'),
                max_date.strftime('%Y-%m-%d %H:%M:%S')
            ))

        return new_ranges
        
    except ValueError as e:
        raise ValueError(f"Invalid date format. Please use 'YYYY-MM-DD'. Error: {e}")


def combine_dataframes(dfs):
    if not dfs:
        return pd.DataFrame()  # Return an empty DataFrame if the list is empty
    
    # Reverse the list to prioritize more recent DataFrames
    dfs_reversed = dfs[::-1]
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs_reversed)
    
    # Drop duplicates based on the index, keeping the first occurrence
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    
    return combined_df



#! ------>>>  Main function to get historical data <<<------ #
def get_hist_data(symbol, start_date, end_date, interval, force_download=False, print_info=True):
    start_date = calculate_past_date(start_date, end_date)
    file_interval = map_to_storage_interval(interval, 'ib')
    stored_data = load_data(symbol, file_interval)
    missing_dates = get_missing_batch_dates(stored_data, start_date, end_date, batch_interval='weekly')

    if print_info:
        start, end, rows = None , None, None
        if stored_data is not None:
            start, end, rows = stored_data.index[0], stored_data.index[-1], len(stored_data)


        print('-------------------------------------------------------------------------------------------------')
        print(f"get_hist_data            : {symbol} {interval} ({file_interval=})")
        print(f"Data Path                : {get_project_path('data', 'historical_data_store')}/{symbol}_{file_interval}.csv")
        print(f"Data Stored    - Start   : {start}, End: {end}, Rows: {rows}")
        print(f"Data Requested - Start   : {start_date}, End: {end_date}")
        print(f"Missing dates            : {len(missing_dates)}")
        print(f"Getting missing data     : {missing_dates or force_download}")
        print('-------------------------------------------------------------------------------------------------')

    if missing_dates or force_download:

        if force_download:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            missing_dates = [(start_date, end_date)]
        
        ibkr = IBHistoricalData()
        missing_data, lowest_barsize   = ibkr.get_batch_historical_data(symbol, missing_dates, barsize=interval, minHourDay_only=True) # will convert bar size down to the lowest common denominator
        new_data = combine_dataframes([stored_data, missing_data])
        save_data(new_data, symbol, lowest_barsize)
        data = load_data(symbol, file_interval)

    data = load_data(symbol, file_interval)
    if data is None:
        print("No data found")
        return None
    end_date = end_date if end_date.lower() != 'now' else pd.Timestamp.now().strftime('%Y-%m-%d')
    return resample_to_interval(data, interval).loc[start_date:end_date]
#! ------>>>  Main function to get historical data <<<------ #



LOG_FILE = 'date_range_log.csv'    


import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

def visualize_saved_stock_date_ranges(min_height=400, row_height=40, max_height=None):
    """
    Create a Gantt chart showing the date ranges for each CSV file using Plotly graph_objects
    with dark theme and adaptive sizing.
    
    Parameters:
    folder_path (str): Path to the folder containing CSV files
    min_height (int): Minimum height of the chart in pixels
    row_height (int): Height per row in pixels
    max_height (int, optional): Maximum height of the chart in pixels
    """
    # Get all CSV files in the folder
    files = list(Path('historical_data').glob('*.csv'))
    
    # Create a color dictionary for consistent colors per stock
    unique_stocks = list(set([f.stem.split('_')[0] for f in files]))
    colors = px.colors.qualitative.Set3[:len(unique_stocks)]
    color_dict = dict(zip(unique_stocks, colors))
    
    # Store data for plotting
    df_list = []
    
    for file in files:
        # More robust date parsing
        df = pd.read_csv(file, parse_dates=['date'], 
                        date_parser=lambda x: pd.to_datetime(x, format=None))  # auto-detect format
        start_date = df['date'].min()
        end_date = df['date'].max()
        
        # Get stock name and interval from filename
        stock_name = file.stem.split('_')[0]
        interval = f"{file.stem.split('_')[1]}_{file.stem.split('_')[2]}"
        
        df_list.append({
            'Task': f"{stock_name} ({interval})",
            'Stock': stock_name,
            'Start': start_date,
            'End': end_date
        })
    
    # Create DataFrame for plotting
    df_gantt = pd.DataFrame(df_list)
    
    # Calculate chart height
    calculated_height = min_height + (len(df_list) * row_height)
    if max_height:
        chart_height = min(calculated_height, max_height)
    else:
        chart_height = calculated_height
    
    # Create figure
    fig = go.Figure()
    
    # Add lines for each file
    for idx, row in df_gantt.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Start'], row['End']],
            y=[row['Task'], row['Task']],
            mode='lines',
            line=dict(
                color=color_dict[row['Stock']],
                width=20
            ),
            customdata=[[row['Start'].strftime('%Y-%m-%d'), row['End'].strftime('%Y-%m-%d')]],
            hovertemplate="<b>%{y}</b><br>" +
                         "Start: %{customdata[0]}<br>" +
                         "End: %{customdata[1]}<br>" +
                         "<extra></extra>"
        ))
    
    # Update layout with dark theme
    fig.update_layout(
        title=dict(
            text='Stock Data Date Ranges by File',
            font=dict(color='#A9A9A9')  # Dark grey text for title
        ),
        xaxis_title=dict(text='Date', font=dict(color='#A9A9A9')),
        yaxis_title=dict(text='Stock Symbol (Interval)', font=dict(color='#A9A9A9')),
        height=chart_height,
        showlegend=True,
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=True),
            gridcolor='#2F2F2F',  # Darker grey for grid
            color='#A9A9A9'  # Dark grey text for axis
        ),
        yaxis=dict(
            autorange='reversed',
            gridcolor='#2F2F2F',  # Darker grey for grid
            color='#A9A9A9'  # Dark grey text for axis
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='#A9A9A9'),  # Dark grey text for all other text
        margin=dict(t=50, l=100, r=20, b=50),  # Adjust margins
    )
    
    # Update grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2F2F2F')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2F2F2F')
    fig.show()


import time
from collections import deque


class RequestTimer:
    def __init__(self, max_requests_per_minute):
        self.max_requests_per_minute = max_requests_per_minute
        self.requests = deque()

    def make_request(self):
        self.wait_until_can_make_request()
        self.request_made()
        return True

    def request_made(self):
        current_time = time.time()
        self.requests.append(current_time)
        self._remove_old_requests(current_time)

    def can_make_request(self):
        current_time = time.time()
        self._remove_old_requests(current_time)
        return len(self.requests) < self.max_requests_per_minute

    def wait_until_can_make_request(self):
        while not self.can_make_request():
            time.sleep(1)  # Wait for 1 second before checking again

    def _remove_old_requests(self, current_time):
        while self.requests and current_time - self.requests[0] > 60:
            self.requests.popleft()


##! ------>>>  Main function to get historical data <<<------ #
#! TODO :  Check what is now redundant and remove it from the above code.  This new calls below should be the only ones needed.

import pandas_market_calendars as mcal
import pandas as pd
import datetime as dt
from typing import List, Tuple
import time
from project_paths import get_project_path
import os
import pytz
import pandas_market_calendars as mcal

INTERVAL_LIMITS = {
    "1 min": 1,      # 1 day (in trading days)
    "2 mins": 2,     # 2 days
    "3 mins": 3,     # 3 days
    "5 mins": 5,     # 5 days
    "10 mins": 10,   # 10 days
    "15 mins": 10,   # 10 days
    "20 mins": 10,   # 10 days
    "30 mins": 10,   # 10 days
    "1 hour": 30,    # 30 days
    "2 hours": 60,   # 60 days
    "3 hours": 60,   # 60 days
    "4 hours": 60,   # 60 days
    "8 hours": 60,   # 60 days
    "1 day": 365,    # 365 days
    "1 week": 1000,  # Practically unlimited
    "1 month": 1000  # Practically unlimited
}


STORAGE_MINUTE_INTERVALS = {'1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins'}
STORAGE_HOUR_INTERVALS = {'1 hour', '2 hours', '3 hours', '4 hours', '8 hours'}
STORAGE_DAY_INTERVALS = {'1 day', '1 week', '1 month'}



class HistoricalData:
    def __init__(self, ib:object=None):
        self.ib = ib 
        self.new_data = []  # Store new data here
        self.data_folder_path = 'data\historical_data_store' #! not linked yet.  see load_data()
    
    def load_data(self, symbol, interval):
        """
        Load data from CSV with datetime index.
        
        Parameters:
        symbol (str): Symbol identifier
        interval (str): Data interval identifier
        
        Returns:
        pd.DataFrame or None: Loaded data or None if file doesn't exist
        """
        interval = interval.lower().replace(' ', '_')
        # file_path = get_project_path('data', 'historical_data_store', f'{symbol}_{interval}.csv')
        file_path = get_project_path(self.data_folder_path, f'{symbol}_{interval}.csv')
        if not os.path.exists(file_path):
            print(f"File not found : {file_path}")
            return None
            
        data = pd.read_csv(file_path, index_col='date', parse_dates=True)
        
        # Remove duplicates and sort
        data = data[~data.index.duplicated(keep='last')].sort_index(ascending=True)
        
        return data

    def save_data(self, data, symbol, interval):
        """
        Save data to CSV with simple duplicate removal.
        
        Parameters:
        data (pd.DataFrame): DataFrame to save
        symbol (str): Symbol identifier
        interval (str): Data interval identifier
        """
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # If 'date' is a column, set it as index
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        
        # Ensure the index is named 'date'
        df.index.name = 'date'
        
        # Convert index to datetime if it's not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort and remove duplicates
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        
        # Save the file
        interval = interval.lower().replace(' ', '_')
        filename = get_project_path(self.data_folder_path, f'{symbol}_{interval}.csv')
        df.to_csv(filename)

    def slice_data(self, df, date_list):
        """
        Slice a DataFrame with DatetimeIndex based on the min and max dates from a list.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame with DatetimeIndex to slice
        date_list : list
            List of date strings in format 'YYYY-MM-DD'
        
        Returns:
        --------
        pandas.DataFrame
            Filtered DataFrame containing only rows between min and max dates (inclusive)
        """
        # Convert strings to datetime objects
        date_list = pd.to_datetime(date_list)
        
        # Get min and max dates 
        #  add last 
        min_date = min(date_list)

        # make hours 23:59:59
        max_date = max(date_list).replace(hour=23, minute=59, second=59)
        
        # Slice DataFrame using .loc
        return df.loc[min_date:max_date]

        # Example usage:
        # date_list = ['2024-11-08', '2024-11-11', '2024-11-12']
        # filtered_df = slice_data(time_indexed_dataframe, date_list)

    def get_ib_data(self, symbol:str, interval:str, endDateTime:str, durationStr:str):
        stock_contract = Stock(symbol, 'SMART', 'USD')
        # rth  = False if not intraday
        useRTH = True if interval  in ['1 day', '1W', '1M'] else False
        bars = self.ib.reqHistoricalData(
            stock_contract,
            endDateTime=endDateTime,
            barSizeSetting=interval,
            durationStr=durationStr,
            whatToShow='TRADES',
            useRTH=useRTH,
            formatDate=1)
        df = util.df(bars)

        if df is not None:
            df.set_index('date', inplace=True)
            
            # If working with daily data, just convert to datetime without time component
            if interval in ['1 day', '1W', '1M']:
                df.index = pd.to_datetime(df.index).normalize()
            else:
                # For intraday data, handle timezone conversion
                df.index = pd.to_datetime(df.index)
                # Assume the data is already in the exchange timezone
                # Just make it timezone-naive if needed
                if df.index.tzinfo is not None:
                    df.index = df.index.tz_localize(None)

        return df
    
    def predict_ib_dates(self, end_datetime_str, duration_str, timezone_str="US/Eastern"):
        # Parse the end date/time string
        parts = end_datetime_str.split(' ')
        date_str = parts[0]
        time_str = parts[1] if len(parts) > 1 else "23:59:59"
        tz_str = parts[2] if len(parts) > 2 else timezone_str
        
        # Parse into datetime object
        end_date = dt.datetime.strptime(f"{date_str} {time_str}", '%Y%m%d %H:%M:%S')
        
        # Get the timezone
        timezone = pytz.timezone(tz_str)
        
        # Localize the end date
        end_date = timezone.localize(end_date)
        
        # Parse the duration string to get number of days
        duration_parts = duration_str.split()
        if len(duration_parts) != 2 or duration_parts[1] != 'D':
            raise ValueError(f"Unsupported duration format: {duration_str}")
        
        duration_days = int(duration_parts[0])
        
        # Determine which exchange calendar to use based on the symbol
        # This is a simplification - you may need more complex logic
        exchange = 'NYSE'  # Default to NYSE

        # Get the exchange calendar
        calendar = mcal.get_calendar(exchange)
        
        # Convert to pandas Timestamp for easier date handling
        end_timestamp = pd.Timestamp(end_date)
        
        # Get a date range that excludes holidays
        # Start date is far enough back to ensure we get enough trading days
        start_date = end_timestamp - pd.Timedelta(days=duration_days*2)  # Go back twice as many days to be safe
        
        # Get the market schedule between start and end dates
        schedule = calendar.schedule(start_date=start_date.strftime('%Y-%m-%d'), 
                                    end_date=end_timestamp.strftime('%Y-%m-%d'))
        
        # The schedule includes only valid trading days
        # Take the last 'duration_days' trading days
        if len(schedule) >= duration_days:
            valid_days = schedule.index[-duration_days:]
        else:
            # Not enough trading days in the range
            valid_days = schedule.index
        
        # Convert to string format
        date_strings = [d.strftime('%Y-%m-%d') for d in valid_days]

        return date_strings
    
    def get_unique_dates(self, data):
        unique_dates = data.index.floor('D').unique()
        return pd.DatetimeIndex(unique_dates).strftime('%Y-%m-%d').tolist()
    
    def _filter_pre_market_dates(self, 
            dates: List[str], 
            timezone_str: str = "US/Eastern"
        ) -> List[str]:
            """
            Filter out today's date if the current time is before market open.
            
            Args:
                dates: List of dates in format 'YYYY-MM-DD'
                timezone_str: Timezone string
                
            Returns:
                Filtered list of dates
            """
            # Get current time in the specified timezone
            now = dt.datetime.now(pytz.timezone(timezone_str))
            today_str = now.strftime('%Y-%m-%d')
            market_open_time = dt.time(5, 00)  # 05:00 AM ET
            
            # If today isn't in the dates list or it's after market open, no need to filter
            if today_str not in dates or now.time() >= market_open_time:
                return dates
                
            # If it's before market open, remove today from the list
            filtered_dates = dates.copy()
            filtered_dates.remove(today_str)
            return filtered_dates
    
    def generate_missing_data_requests(self,
        end_datetime_str: str, 
        duration_str: str, 
        stored_dates: List[str],
        timezone_str: str = "US/Eastern"
    ) -> List[Tuple[str, str]]:
        """
        Generate request parameters for all missing data.
        
        Args:
            symbol: Ticker symbol
            interval: Data interval (e.g., '1 hour')
            end_datetime_str: End date/time string
            duration_str: Duration string
            stored_dates: List of dates already stored in format 'YYYY-MM-DD'
            timezone_str: Timezone string
            
        Returns:
            List of (new_end_datetime, new_duration) tuples for all missing data
        """
        # Get expected dates based on original request
        expected_dates = self.predict_ib_dates(end_datetime_str, duration_str, timezone_str)

        # Filter out today's date if market is not open yet
        expected_dates = self._filter_pre_market_dates(expected_dates, timezone_str)
        
        # Find missing dates
        missing_dates = [date for date in expected_dates if date not in stored_dates]
        
        # If no missing dates, return empty list
        if not missing_dates:
            return []
        
        # Sort missing dates
        missing_dates.sort()
        
        # Extract time and timezone from original request
        parts = end_datetime_str.split(' ')
        time_str = parts[1] if len(parts) > 1 else "23:59:59"
        tz_str = parts[2] if len(parts) > 2 else timezone_str
        
        # Strategy 1: Try to get all missing dates in one request
        # Check if all dates fall within a continuous business day period
        all_missing_first = missing_dates[0]
        all_missing_last = missing_dates[-1]
        
        # Count business days between first and last (inclusive)
        first_date = dt.datetime.strptime(all_missing_first, '%Y-%m-%d')
        last_date = dt.datetime.strptime(all_missing_last, '%Y-%m-%d')
        
        # Build a complete list of business days between first and last
        business_days_between = []
        current = first_date
        while current <= last_date:
            if current.weekday() < 5:  # Weekday
                business_days_between.append(current.strftime('%Y-%m-%d'))
            current += dt.timedelta(days=1)
        
        # If missing_dates == business_days_between, we can request all at once
        if set(missing_dates) == set(business_days_between):
            num_days = len(missing_dates)
            new_end = last_date.strftime('%Y%m%d') + f" {time_str} {tz_str}"
            return [(new_end, f"{num_days} D")]
        
        # Strategy 2: Split into groups of consecutive business days
        requests = []
        date_groups = []
        current_group = [missing_dates[0]]
        
        for i in range(1, len(missing_dates)):
            prev_date = dt.datetime.strptime(missing_dates[i-1], '%Y-%m-%d')
            curr_date = dt.datetime.strptime(missing_dates[i], '%Y-%m-%d')
            
            # Check if dates are consecutive business days
            day_diff = (curr_date - prev_date).days
            is_consecutive = day_diff == 1 or (day_diff == 3 and prev_date.weekday() == 4)
            
            if is_consecutive:
                current_group.append(missing_dates[i])
            else:
                date_groups.append(current_group)
                current_group = [missing_dates[i]]
        
        date_groups.append(current_group)
        
        # Generate request parameters for each group
        for group in date_groups:
            group_size = len(group)
            last_date_in_group = dt.datetime.strptime(group[-1], '%Y-%m-%d')
            new_end = last_date_in_group.strftime('%Y%m%d') + f" {time_str} {tz_str}"
            new_duration = f"{group_size} D"
            requests.append((new_end, new_duration))
        
        return requests
    
    def consolidate_requests(self,
        requests: List[Tuple[str, str]],
        interval: str,
        timezone_str: str = "US/Eastern"
    ) -> List[Tuple[str, str]]:
        """
        Consolidate multiple IB data requests into fewer requests based on interval limits.
        
        Args:
            requests: List of (end_datetime, duration) tuples
            interval: Data interval (e.g., '1 hour')
            timezone_str: Timezone string
            
        Returns:
            List of consolidated (end_datetime, duration) tuples
        """
        if not requests:
            return []
        
        # Get maximum duration allowed for this interval
        max_days = INTERVAL_LIMITS.get(interval, 1)  # Default to 1 day if interval not found
        
        # Convert all request parameters to date objects for easier manipulation
        request_dates = []
        
        for end_datetime, duration in requests:
            # Parse end date
            parts = end_datetime.split(' ')
            date_str = parts[0]
            time_str = parts[1] if len(parts) > 1 else "23:59:59"
            tz_str = parts[2] if len(parts) > 2 else timezone_str
            
            end_date = dt.datetime.strptime(date_str, '%Y%m%d')
            
            # Parse duration
            duration_parts = duration.split()
            if len(duration_parts) != 2 or duration_parts[1] != 'D':
                raise ValueError(f"Unsupported duration format: {duration}")
            
            duration_days = int(duration_parts[0])
            
            # Calculate start date (business days)
            start_date = end_date
            days_counted = 0
            while days_counted < duration_days:
                start_date -= dt.timedelta(days=1)
                if start_date.weekday() < 5:  # 0-4 are Monday to Friday
                    days_counted += 1
            
            request_dates.append({
                'start_date': start_date,
                'end_date': end_date,
                'time_str': time_str,
                'tz_str': tz_str,
                'duration_days': duration_days
            })
        
        # First, calculate the overall date range to check if everything fits in one request
        overall_start = min(req['start_date'] for req in request_dates)
        overall_end = max(req['end_date'] for req in request_dates)
        
        # Count business days between overall start and end
        total_business_days = 0
        date_check = overall_start
        while date_check <= overall_end:
            if date_check.weekday() < 5:  # Only count business days
                total_business_days += 1
            date_check += dt.timedelta(days=1)
        
        # If everything fits within the limit, create a single request
        if total_business_days <= max_days:
            # Find the request with the latest end date
            latest_req = max(request_dates, key=lambda x: x['end_date'])
            consolidated = [{
                'start_date': overall_start,
                'end_date': overall_end,
                'time_str': latest_req['time_str'],
                'tz_str': latest_req['tz_str'],
                'duration_days': total_business_days
            }]
            
        else:
            # Need to split into multiple requests - sort by end_date descending (newest first)
            request_dates.sort(key=lambda x: x['end_date'], reverse=True)
            
            # Consolidate overlapping or adjacent requests
            consolidated = []
            current = None
            
            for req in request_dates:
                if not current:
                    current = req
                    continue
                
                # Check if this request can be merged with current
                days_between = 0
                date_check = req['end_date']
                while date_check < current['start_date'] and days_between <= 3:  # Allow for weekend gap
                    date_check += dt.timedelta(days=1)
                    if date_check.weekday() < 5:  # Only count business days
                        days_between += 1
                
                # Can be merged if directly adjacent (days_between <= 1)
                # or separated only by a weekend (current start is Monday, req end is Friday)
                mergeable = (days_between <= 1) or (
                    days_between <= 3 and 
                    current['start_date'].weekday() == 0 and  # Monday
                    req['end_date'].weekday() == 4  # Friday
                )
                
                # Check if merging would exceed max days
                total_days = current['duration_days'] + req['duration_days'] + days_between
                
                if mergeable and total_days <= max_days:
                    # Merge by extending current to include this request
                    current['start_date'] = min(current['start_date'], req['start_date'])
                    current['duration_days'] = total_days
                else:
                    # Can't merge, add current to consolidated and start new current
                    consolidated.append(current)
                    current = req
            
            # Add the last current
            if 'current' in locals() and current:
                consolidated.append(current)
        
        # Convert back to request parameters format
        result = []
        for req in consolidated:
            # Calculate new duration (in business days)
            duration_days = req['duration_days']
            
            # Format end datetime
            end_datetime = req['end_date'].strftime('%Y%m%d') + f" {req['time_str']} {req['tz_str']}"
            duration_str = f"{duration_days} D"
            
            result.append((end_datetime, duration_str))
        
        return result

    def resample_data(self, data, original_interval):
        """
        Resample time series data to a specified interval.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The DataFrame to resample, expected to have a DatetimeIndex and OHLCV columns
        original_interval : str
            The target interval to resample to (e.g., '1 min', '5 mins', '1 hour')
            
        Returns:
        --------
        pandas.DataFrame
            The resampled DataFrame with OHLCV data aggregated appropriately
        """
        # Mapping from IB interval strings to pandas resample rule strings
        resample_map = {
            # Minutes
            '1 min': '1T', '2 mins': '2T', '3 mins': '3T', '5 mins': '5T',
            '10 mins': '10T', '15 mins': '15T', '20 mins': '20T', '30 mins': '30T',
            # Hours
            '1 hour': '1H', '2 hours': '2H', '3 hours': '3H', '4 hours': '4H', '8 hours': '8H',
            # Days
            '1 day': '1D', '1w': '1W', '1m': '1M'
        }
        
        # Get the pandas resample rule
        resample_rule = resample_map.get(original_interval)
        
        if resample_rule:
            # Assuming OHLCV data structure - adjust these aggregations based on your data
            resampled_data = data.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            return resampled_data
        else:
            # If no mapping found, return the original data
            return data

    def normalize_interval(self, interval_str):
        """
        Normalizes interval strings to a consistent format accepted by the IB API.
        
        Converts various interval string formats to the standardized IB API format.
        Examples:
        - '1H', '1h', '1 h' → '1 hour'
        - '2H', '2h', '2 h' → '2 hours'
        - '5M', '5m', '5 m' → '5 mins'
        - '1D', '1d', '1 d' → '1 day'
        
        Parameters:
        -----------
        interval_str : str
            The interval string to normalize
            
        Returns:
        --------
        str
            The normalized interval string in IB API compatible format
        """
        # Strip whitespace and convert to lowercase for consistent processing
        interval = interval_str.strip().lower()
        
        # Extract the numeric part and the unit part
        # Match one or more digits followed by optional whitespace and a unit
        import re
        match = re.match(r'(\d+)\s*([a-z]+)', interval)
        
        if not match:
            # Return original if no match
            return interval_str
        
        value, unit = match.groups()
        value = int(value)  # Convert to integer
        
        # Normalize the unit part
        if unit in ['s', 'sec', 'secs', 'second', 'seconds']:
            unit_normalized = 'secs' if value > 1 else 'sec'
            return f"{value} {unit_normalized}"
            
        elif unit in ['m', 'min', 'mins', 'minute', 'minutes']:
            unit_normalized = 'mins' if value > 1 else 'min'
            return f"{value} {unit_normalized}"
            
        elif unit in ['h', 'hr', 'hrs', 'hour', 'hours']:
            unit_normalized = 'hours' if value > 1 else 'hour'
            return f"{value} {unit_normalized}"
            
        elif unit in ['d', 'day', 'days']:
            unit_normalized = 'days' if value > 1 else 'day'
            return f"{value} {unit_normalized}"
            
        elif unit in ['w', 'wk', 'wks', 'week', 'weeks']:
            # IB uses '1w' format for weeks
            return f"{value}w"
            
        elif unit in ['mo', 'mon', 'month', 'months']:
            # IB uses '1m' format for months
            return f"{value}m"
            
        # Return the original if unit is not recognized
        return interval_str

    def normalize_date_for_ib(self, date_str, timezone='US/Eastern'):
        """
        Normalizes various date formats to the format required by Interactive Brokers API.
        
        Handles various input formats:
        - ISO format: '2025-03-25' → '20250325 23:59:59 US/Eastern'
        - ISO with time: '2025-03-25 13:00:00' → '20250325 13:00:00 US/Eastern'
        - Compact format: '20250325' → '20250325 23:59:59 US/Eastern'
        - Compact with time: '20250325 13:00:00' → '20250325 13:00:00 US/Eastern'
        - Special case 'now': Returns current time in the required format
        
        Parameters:
        -----------
        date_str : str
            The date string to normalize
        timezone : str, default='US/Eastern'
            The timezone to use for the output format
            
        Returns:
        --------
        str
            The normalized date string in IB API compatible format
        """
        import re
        import datetime
        
        # Handle the special case 'now'
        if date_str.lower() == 'now':
            now = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
            return f"{now} {timezone}"
        
        # Strip any leading/trailing whitespace
        date_str = date_str.strip()
        
        # Check if the date_str already has time component
        has_time = bool(re.search(r'\d+:\d+', date_str))
        
        # Different regex patterns for different input formats
        iso_date_pattern = r'^\d{4}-\d{2}-\d{2}$'  # YYYY-MM-DD
        iso_datetime_pattern = r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$'  # YYYY-MM-DD HH:MM:SS
        compact_date_pattern = r'^\d{8}$'  # YYYYMMDD
        compact_datetime_pattern = r'^\d{8}\s+\d{2}:\d{2}:\d{2}$'  # YYYYMMDD HH:MM:SS
        
        # Parse based on the identified format
        if re.match(iso_date_pattern, date_str):
            # Format: YYYY-MM-DD
            dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            date_part = dt.strftime('%Y%m%d')
            time_part = "23:59:59" if not has_time else "00:00:00"
            return f"{date_part} {time_part} {timezone}"
            
        elif re.match(iso_datetime_pattern, date_str):
            # Format: YYYY-MM-DD HH:MM:SS
            dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            date_part = dt.strftime('%Y%m%d')
            time_part = dt.strftime('%H:%M:%S')
            return f"{date_part} {time_part} {timezone}"
            
        elif re.match(compact_date_pattern, date_str):
            # Format: YYYYMMDD
            dt = datetime.datetime.strptime(date_str, '%Y%m%d')
            date_part = dt.strftime('%Y%m%d')
            time_part = "23:59:59" if not has_time else "00:00:00"
            return f"{date_part} {time_part} {timezone}"
            
        elif re.match(compact_datetime_pattern, date_str):
            # Format: YYYYMMDD HH:MM:SS
            dt = datetime.datetime.strptime(date_str, '%Y%m%d %H:%M:%S')
            date_part = dt.strftime('%Y%m%d')
            time_part = dt.strftime('%H:%M:%S')
            return f"{date_part} {time_part} {timezone}"
        
        # Try to handle other potential formats with dateutil parser
        try:
            from dateutil import parser
            dt = parser.parse(date_str)
            date_part = dt.strftime('%Y%m%d')
            
            # If original had time, use it, otherwise use end of day
            if has_time:
                time_part = dt.strftime('%H:%M:%S')
            else:
                time_part = "23:59:59"
                
            return f"{date_part} {time_part} {timezone}"
        except:
            # If all else fails, return the original with the timezone appended
            return f"{date_str} {timezone}"

    def get_data(self, symbol:str, interval:str, endDateTime:str, durationStr:str, print_info: bool = False):
        """
        Retrieve historical market data for a specific symbol by combining stored data with newly fetched data,
        then slice it to the requested time range.
        
        This method follows a multi-step process:
        1. Determine the appropriate storage interval based on the requested interval
        2. Load existing data from storage
        3. Identify missing date ranges that need to be requested
        4. Consolidate those requests to minimize API calls
        5. Fetch the missing data from Interactive Brokers
        6. Combine new and stored data, removing duplicates
        7. Save the consolidated dataset
        8. Slice the data to return only the requested time period
        9. Resample the data if the requested interval differs from the storage interval
        
        Parameters:
        -----------
        symbol : str
            The trading symbol/ticker to fetch data for (e.g., 'AAPL', 'MSFT')
        interval : str
            The time interval for the data points (e.g., '1 min', '5 mins', '1 hour', '1 day')
        endDateTime : str
            The end date/time for the data in a format compatible with IB API (e.g., '20241108 16:00:00')
        durationStr : str
            The duration of data to fetch in a format compatible with IB API (e.g., '5 D', '2 W', '1 M')
        print_info : bool, default=False
            Whether to print information about the data retrieval process
            
        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the historical market data for the specified symbol and time range,
            with a DatetimeIndex, resampled to the requested interval
        """
        original_endDateTime = endDateTime
        # Step 1: Determine storage interval based on requested interval
        interval = self.normalize_interval(interval)
        endDateTime = self.normalize_date_for_ib(endDateTime)
        original_interval = interval  # Store the original requested interval

        # Map the requested interval to the appropriate storage interval
        if interval in STORAGE_MINUTE_INTERVALS:
            storage_interval = '1 min'  # Store as minute data
        elif interval in STORAGE_HOUR_INTERVALS:
            storage_interval = '1 hour'  # Store as hourly data
        elif interval in STORAGE_DAY_INTERVALS:
            storage_interval = '1 day'  # Store as daily data
        else:
            # Default to the requested interval if not in any known category
            storage_interval = interval
        
        # Step 2: Load previously stored data for this symbol and the determined storage interval
        stored_data = self.load_data(symbol, storage_interval)
        
        # Step 3: Extract unique dates from stored data (if it exists)
        stored_dates = self.get_unique_dates(stored_data) if stored_data is not None else []
        
        # Step 4: Determine what date ranges are missing and need to be requested
        # This compares the stored dates with the date range requested by endDateTime and durationStr
        missing_requests = self.generate_missing_data_requests(endDateTime, durationStr, stored_dates)
        
        # Step 5: Consolidate the missing date ranges to minimize the number of API calls
        # This merges adjacent or overlapping date ranges
        consolidated_requests = self.consolidate_requests(missing_requests, storage_interval)
        
        # Step 6: Fetch missing data from Interactive Brokers API using the storage interval
        # Each request is a tuple of (end_date_time, duration_str)
        for end_date_time, duration_str in consolidated_requests:
            # Append each new dataset to self.new_data list, using the storage interval
            df  = self.get_ib_data(symbol, storage_interval, end_date_time, duration_str)
            if df is None:
                continue
            df.index = pd.to_datetime(df.index)
            self.new_data += [df]
            # Pause between requests to prevent hitting rate limits
            time.sleep(1)

        # Step 7: Combine all datasets (new and previously stored)
        # Create a list containing all new data fetched in this session and the stored data
        list_of_data = self.new_data + [stored_data]

        # # Display the last date of each dataset
        # print(f"get_data :: endDateTime: {endDateTime}")
        # print(f"get_data :: stored_dates         : {stored_dates}")
        # print(f"get_data :: missing requests     : {missing_requests}")
        # print(f"get_data :: consolidated_requests: {consolidated_requests}")
        # print(f"get_data :: list_of_data: {list_of_data}")


        # for d in list_of_data:
        #     print(d.index[-1])  #
        #     print(type(d.index[-1]))  #
        #     display(d)
        
        # Step 8: Merge the datasets, remove any duplicates, and ensure chronological order
        all_data = pd.concat(list_of_data).loc[~pd.concat(list_of_data).index.duplicated(keep='last')].sort_index()
        
        # Step 9: Save the consolidated dataset for future use with the storage interval
        self.save_data(all_data, symbol, storage_interval)

        # Step 10: Calculate the exact date range corresponding to the requested parameters
        dates = self.predict_ib_dates(endDateTime, durationStr)
        
        # Step 11: Slice the full dataset to return only the requested date range
        # This uses the DatetimeIndex of the DataFrame with .loc slicing
        sliced_data = self.slice_data(all_data, dates)
        
        # Step 12: Resample the data if the original interval is different from the storage interval
        if original_interval != storage_interval:
            new_data = self.resample_data(sliced_data, original_interval)
        else:
            # No resampling needed
            new_data = sliced_data
        
        # Step 13: Print information about the data retrieval if requested
        if print_info:
            print(f"{symbol} {original_interval} {durationStr} :: {new_data.index[0]} to {new_data.index[-1]} rows: {len(new_data)}, missing: {len(missing_requests)}, Data stored as {storage_interval} and resampled to {original_interval}")
        
        # Return the processed data
        return new_data
        