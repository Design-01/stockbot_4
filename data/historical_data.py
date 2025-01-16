
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


def calculate_past_date(input_date):
    """
    Calculate a past date based on either a date object, date string, or a relative time string.
    For 'daysAgo', rounds to 00:00:01 on the day
    For 'weeksAgo', rounds to Monday 00:00:01 of that week
    
    Args:
        input_date: Can be either:
            - datetime object
            - string in format '%Y-%m-%d %H:%M:%S' (e.g., '2024-11-24 14:30:00')
            - string in format '%Y-%m-%d' (e.g., '2024-11-24')
            - string in format '5 daysAgo' or '3 weeksAgo'
    
    Returns:
        str: Formatted date string in '%Y-%m-%d %H:%M:%S' format
    
    Examples:
        >>> calculate_past_date('5 daysAgo')  # if current date is Monday
        Returns previous Monday at 00:00:01
        >>> calculate_past_date('2 weeksAgo')  # if lands on Wednesday
        Returns Monday of that week at 00:00:01
    """
    # Handle datetime object
    if isinstance(input_date, datetime):
        return input_date.strftime('%Y-%m-%d %H:%M:%S')
    
    if not isinstance(input_date, str):
        raise ValueError("Input must be either a datetime object or a string")
    
    # Try parsing as datetime string with different formats
    date_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d'
    ]
    
    for date_format in date_formats:
        try:
            parsed_date = datetime.strptime(input_date, date_format)
            return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
    
    # Parse relative time string
    pattern = r'(\d+)\s*(days?Ago|weeks?Ago)'
    match = re.match(pattern, input_date, re.IGNORECASE)
    
    if not match:
        raise ValueError("String must be in format 'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD', or 'X daysAgo/weeksAgo'")
    
    number = int(match.group(1))
    unit = match.group(2).lower()
    
    current_date = datetime.now()
    
    if 'week' in unit:
        # First go back the specified number of weeks
        past_date = current_date - timedelta(weeks=number)
        # Then find Monday of that week (weekday() returns 0 for Monday)
        days_since_monday = past_date.weekday()
        # Adjust to Monday of that week
        past_date = past_date - timedelta(days=days_since_monday)
        # Set time to 00:00:01
        past_date = past_date.replace(hour=0, minute=0, second=1, microsecond=0)
    else:  # days
        # Go back the specified number of days
        past_date = current_date - timedelta(days=number)
        # Set time to 00:00:01
        past_date = past_date.replace(hour=0, minute=0, second=1, microsecond=0)
    
    return past_date.strftime('%Y-%m-%d %H:%M:%S')

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
        # print(f"Lowest interval      : {lowest_interval}")
        # print(f"Batch size           : {batch_size}")
        # print(f"Converted batch dates: {converted_batch_dates}")

        
        def get_duration(delta):
            if delta.days <= 1:
                return '1 D'
            elif delta.days <= 7:
                return '1 W'
            elif delta.days <= 31:
                return '1 M'
            else: return '1 Y'
            
        
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
                # print(f"delta: {delta}, duration: {duration}")
                # print(f"Fetching data for {start} to {end} ({duration})")
                # print(f"End date: {end_date}")

                # rth  = False if not intraday
                not_outside_rth = ['3 hours', '4 hours', '8 hours', '1 day', '1W', '1M']
                rth = False if lowest_interval  in not_outside_rth else True
                print(f"get_batch_historical_data :: {lowest_interval=}, {rth=}")
                
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_date,
                    barSizeSetting=lowest_interval,
                    durationStr=duration,
                    whatToShow='TRADES',
                    useRTH=False,
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
    print(f"Loading data from {file_path}")
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
    Find missing date intervals in a DataFrame, including partial current periods.
    
    Parameters:
    data (pd.DataFrame): DataFrame with datetime index (can be empty)
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format or 'now' for current date
    batch_interval (str): 'weekly' or 'monthly'
    
    Returns:
    list of tuples: (start_date, end_date) pairs for missing intervals
    """
    def get_freq_and_duration(interval):
        """Helper function to get frequency and duration for intervals"""
        if interval == 'weekly':
            return 'W-MON', pd.Timedelta(days=6)
        return 'MS', pd.offsets.MonthEnd(1)
    
    try:
        # Handle 'now' as end_date and convert dates
        start = pd.to_datetime(start_date).normalize()
        end = (pd.Timestamp.now() if end_date.lower() == 'now' 
               else pd.to_datetime(end_date)).normalize()
        
        # Validate interval
        if batch_interval.lower() not in ['weekly', 'monthly']:
            raise ValueError("batch_interval must be 'weekly' or 'monthly'")
            
        freq, duration = get_freq_and_duration(batch_interval.lower())
        
        # Generate intervals
        intervals = pd.date_range(start=start, end=end, freq=freq)
        
        # Handle empty or None data explicitly
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            # Ensure we return at least one interval even if intervals is empty
            if len(intervals) == 0:
                return [(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))]
            
            interval_pairs = []
            for interval_start in intervals:
                interval_end = min(interval_start + duration, end)
                interval_pair = (
                    interval_start.strftime('%Y-%m-%d'),
                    interval_end.strftime('%Y-%m-%d')
                )
                interval_pairs.append(interval_pair)
            
            return interval_pairs
        
        # Rest of the function remains the same...
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be DatetimeIndex")
        
        missing_intervals = []
        for interval_start in intervals:
            interval_end = min(interval_start + duration, end)
            
            mask = (data.index >= interval_start) & (data.index <= interval_end)
            if not data[mask].size > 0:
                missing_intervals.append(
                    (interval_start.strftime('%Y-%m-%d'),
                     interval_end.strftime('%Y-%m-%d'))
                )
                
        return missing_intervals
        
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD' or 'now'. Error: {e}")


def convert_ranges(ranges, new_interval='monthly', start_time_offset='00:00:01'):
    """
    Convert a list of date ranges to a new interval (weekly, monthly, or yearly).
    Adds a time offset to start dates to avoid midnight boundary issues.
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
            return [(
                min_date.strftime('%Y-%m-%d %H:%M:%S'),
                max_date.strftime('%Y-%m-%d %H:%M:%S')
            )]
            
        if new_interval.lower() == 'monthly' and (max_date - min_date) < pd.Timedelta(days=28):
            return [(
                min_date.strftime('%Y-%m-%d %H:%M:%S'),
                max_date.strftime('%Y-%m-%d %H:%M:%S')
            )]
            
        if new_interval.lower() == 'yearly' and (max_date - min_date) < pd.Timedelta(days=365):
            return [(
                min_date.strftime('%Y-%m-%d %H:%M:%S'),
                max_date.strftime('%Y-%m-%d %H:%M:%S')
            )]
        
        # Select frequency based on interval
        freq_map = {
            'weekly': ('W-MON', pd.Timedelta(days=6)),
            'monthly': ('MS', pd.offsets.MonthEnd(1)),
            'yearly': ('YS', pd.offsets.YearEnd(1))
        }
        
        freq, duration = freq_map[new_interval.lower()]
        
        # Generate new intervals, starting from the first date
        intervals = pd.date_range(
            start=min_date,
            end=max_date,
            freq=freq,
            inclusive='both'
        )
        
        # If no intervals were generated but we have a valid date range,
        # return the original range
        if len(intervals) == 0:
            return [(
                min_date.strftime('%Y-%m-%d %H:%M:%S'),
                max_date.strftime('%Y-%m-%d %H:%M:%S')
            )]
        
        
        # Create new ranges
        new_ranges = []
        for interval_start in intervals:
            interval_end = interval_start + duration
            
            # Ensure we don't exceed the maximum date
            interval_end = min(interval_end, max_date)
            
            # Add time components
            range_start = (interval_start + pd.Timedelta(start_time_offset))
            range_end = (interval_end + pd.Timedelta('23:59:59'))
            
            new_ranges.append((
                range_start.strftime('%Y-%m-%d %H:%M:%S'),
                range_end.strftime('%Y-%m-%d %H:%M:%S')
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
def get_hist_data(symbol, start_date, end_date, interval, force_download=False):
    start_date = calculate_past_date(start_date)
    file_interval = map_to_storage_interval(interval, 'ib') # eg coverts 5 mins to 1_min
    stored_data   = load_data(symbol, file_interval)
    missing_dates = get_missing_batch_dates(stored_data, start_date, end_date, batch_interval='weekly')
    if stored_data is not None and not force_download:
        print(f"Stored data: {len(stored_data)} rows of data")

    if missing_dates or force_download:

        if force_download:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            missing_dates = [(start_date, end_date)]
        
        print(f"Processing Missing data: {len(missing_dates)} intervals")
        ibkr = IBHistoricalData()
        missing_data, lowest_barsize   = ibkr.get_batch_historical_data(symbol, missing_dates, barsize=interval, minHourDay_only=True) # will convert bar size down to the lowest common denominator
        print(f"Missing data: {len(missing_data)} rows of data")
        # print(f"Missing data: {missing_data}")
        new_data = combine_dataframes([stored_data, missing_data])
        save_data(new_data, symbol, lowest_barsize)
    

    # data = hd.load_data(symbol, lowest_barsize)
    data = load_data(symbol, file_interval)
    print(f"Data loaded: {len(data)} rows of data")
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
    