
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from twelvedata import TDClient
import pandas as pd
import os
import csv
from typing import Tuple, List , Optional, Union
import time

"https://github.com/twelvedata/twelvedata-python"


def get_historical_data_from_source(api_key, symbol, start_date, end_date, interval):
    avialble_interval = get_next_lower_common_denominator_as_interval(interval)

    data =  TDClient(apikey=api_key).time_series(
            symbol=symbol,
            interval=avialble_interval,
            start_date=start_date,
            end_date=end_date,
            outputsize=5000,
            timezone="America/New_York",
        ).as_pandas().sort_index(ascending=True)
    
    return resample_data(data, interval)


def get_batch_dates(start_date, end_date, batch:str='monthly'):
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

# Test the function
# print(batch_dates('2024-01-10', '2021-03-20', batch='monthly')) # --> [(2024-01-01, 2024-01-31), (2024-02-01, 2024-02-29), (2024-03-01, 2024-03-31)]


def get_batch_size(interval):
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


def get_next_lower_common_denominator_as_interval(interval):
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
    
    # Find the largest interval that is less than or equal to the input interval
    next_lower_interval = '1min'
    for available_interval in available_intervals:
        if interval_map[available_interval] <= interval_minutes and interval_minutes % interval_map[available_interval] == 0:
            next_lower_interval = available_interval
    
    return next_lower_interval


def resample_data(data, interval):
    return data.resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


def save_data(data, symbol, interval):
    data.to_csv(f"historical_data/{symbol}_{interval}.csv")


def load_data(symbol, interval):
    file_path = f"historical_data/{symbol}_{interval}.csv"
    if not os.path.exists(file_path):
        return None
    data = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    data = data[~data.index.duplicated(keep='last')].sort_index(ascending=True)
    return data


def get_csv_data_range(symbol, interval):
    file_path = f"historical_data/{symbol}_{interval}.csv"
    
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
        start_date = data.index[0].strftime('%Y-%m-%d %H:%M:%S')
        end_date = data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        return (start_date, end_date)
    else:
        return None, None


def dates_in_csv(symbol, interval, start_date=None, end_date=None):
    # Check if the file path exists
    file_path = f"historical_data/{symbol}_{interval}.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        return False
    
    # Ensure the DataFrame index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    
    # Extract the date part from the DataFrame index
    df_dates = df.index.date
    
    # Convert start_date and end_date to pandas Timestamp objects if they are not None
    if start_date is not None:
        start_date = pd.Timestamp(start_date).date()
    if end_date is not None:
        end_date = pd.Timestamp(end_date).date()
    
    # Check if both dates are None
    if start_date is None and end_date is None:
        return False
    
    # Check if only start_date is provided
    if start_date is not None and end_date is None:
        return start_date in df_dates
    
    # Check if only end_date is provided
    if start_date is None and end_date is not None:
        return end_date in df_dates
    
    # Check if both start_date and end_date are provided
    if start_date in df_dates and end_date in df_dates:
        return True
    else:
        return False


def combine_dataframes(dfs):
    if not dfs:
        return pd.DataFrame()  # Return an empty DataFrame if the list is empty
    
    # Reverse the list to prioritize more recent DataFrames
    dfs_reversed = dfs[::-1]
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs_reversed)
    
    # Drop duplicates based on the index, keeping the first occurrence
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    return combined_df


LOG_FILE = 'date_range_log.csv'    

def check_logged_date_range(symbol, interval, start_date, end_date):
    """
    Checks if the combination of symbol, interval, and date range exists in the log.

    Parameters:
    symbol (str): The ticker symbol for the stock data request
    interval (str): The time interval of the request, e.g., "1m", "5m"
    start_date (str or datetime): The start date of the range
    end_date (str or datetime): The end date of the range

    Returns:
    bool: True if the combination exists, False otherwise
    """
    # Convert dates to date-only strings if they are datetime objects
    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")
    else:
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        except ValueError:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")
    else:
        try:
            end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        except ValueError:
            end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

    try:
        with open(LOG_FILE, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                # Skip empty rows or rows with incorrect number of columns
                if not row or len(row) < 4:
                    continue
                
                # Check if all required fields match
                try:
                    if (row[0].strip() == symbol and 
                        row[1].strip() == interval and 
                        row[2].strip() == start_date and 
                        row[3].strip() == end_date):
                        return True
                except IndexError:
                    continue  # Skip rows with missing columns
    except FileNotFoundError:
        # Create the file if it doesn't exist
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Symbol', 'Interval', 'Start Date', 'End Date'])
    return False

def log_date_range(symbol, interval, start_date, end_date):
    """
    Logs a symbol's data range to a CSV file if the combination does not exist.

    Parameters:
    symbol (str): The ticker symbol for the stock data request
    interval (str): The time interval of the request, e.g., "1m", "5m"
    start_date (str or datetime): The start date of the range
    end_date (str or datetime): The end date of the range

    Returns:
    None
    """
    # Input validation
    if not all([symbol, interval, start_date, end_date]):
        raise ValueError("All parameters must have non-empty values")
    
    # Convert dates to date-only strings if they are datetime objects
    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")
    else:
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        except ValueError:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")
    else:
        try:
            end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        except ValueError:
            end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    # Strip whitespace from all inputs
    symbol = symbol.strip()
    interval = interval.strip()
    start_date = start_date.strip()
    end_date = end_date.strip()
    
    # Check if the combination already exists
    if check_logged_date_range(symbol, interval, start_date, end_date):
        print(f"Date range for {symbol} ({interval}) from {start_date} to {end_date} is already logged.")
        return
    
    # Log the new date range
    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([symbol, interval, start_date, end_date])
        print(f"Logged date range for {symbol} ({interval}) from {start_date} to {end_date}")
    except PermissionError:
        print("Error: Unable to write to log file due to permission issues")
    except Exception as e:
        print(f"Error logging date range: {str(e)}")


def get_historical_data(api_key, symbol, start_date, end_date, interval):
    lowest_interval = get_next_lower_common_denominator_as_interval(interval)
    batch_size      = get_batch_size(lowest_interval)
    batch_dates     = get_batch_dates(start_date, end_date, batch_size)

    missing_batched_dates       = [ bd for bd in batch_dates if not check_logged_date_range(symbol, interval,        bd[0], bd[1])]
    missing_batched_dates_lower = [ bd for bd in batch_dates if not check_logged_date_range(symbol, lowest_interval, bd[0], bd[1])]

    print(f"Batch dates                : {batch_dates}")
    print(f"Missing batched dates      : {interval} {missing_batched_dates}")
    print(f"Missing batched dates lower: {lowest_interval} {missing_batched_dates_lower}")

    lower_data_list = []
    
    # first get the missing data for the lowest interval
    if len(missing_batched_dates_lower) == 0:
        print(f"No missing data for {lowest_interval}")
    else:
        request_timer = RequestTimer(max_requests_per_minute=8)
        for bd in missing_batched_dates_lower:
            if not request_timer.make_request():
                print("Reached maximum requests per minute. Waiting for next minute...")
                time.sleep(60)
            print(f"Getting missing data for {lowest_interval} -- dates: { bd[0]} to {bd[1]}")
            lower_data_list += [get_historical_data_from_source(api_key, symbol, bd[0], bd[1], lowest_interval)]
            log_date_range(symbol, lowest_interval, bd[0], bd[1])


        # now save the lower data
        lower_data = combine_dataframes(lower_data_list)
        save_data(lower_data, symbol, lowest_interval)

    # then log the missing data for each batch of the requested interval if not 
    if len(missing_batched_dates) == 0:
        print(f"No missing data for {interval}")
    else:
        for bd in missing_batched_dates:
            print(f"Getting missing data for {interval} -- dates: {bd[0]} to {bd[1]}")
            log_date_range(symbol, interval, bd[0], bd[1])

        # then resample the lower data to the requested interval
        # becasue we now have the data we can just load it and resample it without getting it from the source
        lower_data = load_data(symbol, lowest_interval)
        data = resample_data(lower_data, interval)
        save_data(data, symbol, interval)

    # finally return the requested data
    return load_data(symbol, interval).loc[start_date:end_date]


import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

def visualize_stock_date_ranges(min_height=400, row_height=40, max_height=None):
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
        # Read first and last row of CSV to get date range
        df = pd.read_csv(file, parse_dates=['datetime'])
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()
        
        # Get stock name and interval from filename
        stock_name = file.stem.split('_')[0]
        interval = file.stem.split('_')[1]
        
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
        showlegend=False,
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
    