from dataclasses import dataclass
import pandas as pd
import os
from datetime import datetime, timedelta
from data.twelve_data import TwelveData
from data.random_data import RandomOHLCV

@dataclass
class OHLCV:
    historical_data_folder = 'historical_data'
    rand_data_folder = 'rand_data'
    api_key: str = None

    def __post_init__(self):
        os.makedirs(self.historical_data_folder, exist_ok=True)
        os.makedirs(self.rand_data_folder, exist_ok=True)
        if self.api_key:
            self.twelve_data = TwelveData(self.api_key)

    def get_end_date(self, start_date: str, days: int) -> str:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = start + timedelta(days=days)
        return end.strftime('%Y-%m-%d')

    def get_start_date(self, end_date: str, days: int) -> str:
        end = datetime.strptime(end_date, '%Y-%m-%d')
        start = end - timedelta(days=days)
        return start.strftime('%Y-%m-%d')

    def get_interval_options(self):
        return ['1min', '5min', '15min', '1day', '1week', '1month']

    def get_source_options(self):
        return ['twelve_data', 'random', 'file']

    def get_list_of_stored_data(self, keyword=None):
        data_list = []
        folders = [self.historical_data_folder, self.rand_data_folder]
        
        for folder in folders:
            if not os.path.exists(folder):
                continue
            files = os.listdir(folder)
            for file in files:
                if keyword and keyword not in file:
                    continue
                file_path = os.path.join(folder, file)
                try:
                    df = self.load_data(file_path)
                    if not df.empty:
                        data_list.append(df)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        if data_list:
            return pd.concat(data_list, ignore_index=True)
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no data is loaded

    def get_stored_data(self, source: str, symbol: str, interval: str, start_date: str = '01-10-2024', end_date: str = 'today', returnAs: str = 'df', save_format: str = 'csv'):
        if end_date == 'today':
            end_date = datetime.now().strftime('%d-%m-%Y')
        
        # Convert date format from DD-MM-YYYY to YYYY-MM-DD for internal processing
        start_date = datetime.strptime(start_date, '%d-%m-%Y').strftime('%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%d-%m-%Y').strftime('%Y-%m-%d')

        df = self.get_data(source, symbol, interval, start_date, end_date)
        
        if save_format:
            self.save_data(df, source, symbol, interval, save_format)

        if returnAs == 'df':
            return df
        elif returnAs == 'dict':
            return df.to_dict()
        else:
            raise ValueError(f"Unsupported return type: {returnAs}")

    def get_live_data(self, source: str, symbol: str, interval: str, start_date: str, end_date: str, returnAs: str = 'dict', save_when: str = 'end_of_session'):
        df = self.get_data(source, symbol, interval, start_date, end_date)
        if save_when == 'end_of_session':
            self.save_data(df, source, symbol, interval, 'csv', live=True)
        if returnAs == 'df':
            return df
        elif returnAs == 'dict':
            return df.to_dict()
        else:
            raise ValueError(f"Unsupported return type: {returnAs}")

    def get_data(self, source: str, symbol: str = None, interval: str = '1day', start_date: str = None, 
                 end_date: str = None, trend: str = 'up', settings: dict = None, **kwargs):
        if source == 'twelve_data':
            if not self.api_key:
                raise ValueError("API key is required for Twelve Data")
            df = self.twelve_data.get_historical_data(symbol, interval, start_date, end_date)
        elif source == 'random':
            random_settings = {
                'open_rng': settings.get('open_rng', (-0.01, 0.01)),
                'close_rng': settings.get('close_rng', (-0.01, 0.01)),
                'start': settings.get('start', '2022'),
                'periods': settings.get('periods', 50),
                'freq': settings.get('freq', '1D'),
                'open_val': settings.get('open_val', 100),
                'head_max': settings.get('head_max', 5),
                'tail_max': settings.get('tail_max', 5),
                'vol_rng': settings.get('vol_rng', (-50, 60)),
                'vol_start': settings.get('vol_start', 500),
                'volatility_rng': settings.get('volatility_rng', (0, 0)),
                'volatility_freq': settings.get('volatility_freq', 0),
                'volatility_dur': settings.get('volatility_dur', 0)
            }
            random_ohlcv = RandomOHLCV(trend=trend, settings=random_settings, **kwargs)
            df = random_ohlcv.get_dataframe()
        elif source == 'file':
            file_path = kwargs.get('file_path')
            if not file_path:
                raise ValueError("file_path is required for 'file' source")
            df = self.load_data(file_path)
        else:
            raise ValueError(f"Unsupported data source: {source}")

        return df

    def save_data(self, df, source, symbol, interval, save_format, live=False):
        # Check for common datetime column names
        datetime_columns = ['datetime', 'date', 'time', 'timestamp']
        datetime_col = next((col for col in datetime_columns if col in df.columns), None)

        if datetime_col is None:
            raise ValueError("DataFrame does not contain a recognizable datetime column")

        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col)
        
        start_year = df[datetime_col].min().year
        end_year = df[datetime_col].max().year
        
        for year in range(start_year, end_year + 1):
            year_df = df[df[datetime_col].dt.year == year]
            if year_df.empty:
                continue
            
            start_month = year_df[datetime_col].min().strftime('%b')
            end_month = year_df[datetime_col].max().strftime('%b')
            
            if live:
                filename = f"{symbol}_{interval}_{year}_{start_month}_to_{end_month}_live"
            else:
                filename = f"{symbol}_{interval}_{year}_{start_month}_to_{end_month}"
            
            if source == 'twelve_data':
                save_path = os.path.join(self.historical_data_folder, f"{filename}.{save_format}")
            else:
                save_path = os.path.join(self.rand_data_folder, f"{filename}.{save_format}")
            
            self.merge_and_save(year_df, save_path, save_format, datetime_col)

    def merge_and_save(self, new_df, file_path, save_format, datetime_col):
        if os.path.exists(file_path):
            existing_df = self.load_data(file_path)
            merged_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=[datetime_col]).sort_values(datetime_col)
        else:
            merged_df = new_df

        if save_format == 'csv':
            merged_df.to_csv(file_path, index=False)
        elif save_format == 'pickle':
            merged_df.to_pickle(file_path)
        elif save_format == 'excel':
            merged_df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")

        print(f"Data saved as {file_path}")

    def load_data(self, file_path):
        _, ext = os.path.splitext(file_path)
        if ext == '.csv':
            df = pd.read_csv(file_path)
            # Set the second column as the DateTime index
            df.set_index(df.columns[0], inplace=True)
            df.index = pd.to_datetime(df.index)
        elif ext == '.pkl':
            df = pd.read_pickle(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return df



from dataclasses import dataclass, field

@dataclass
class ServeNewOHLCV:
    """
    A class to serve OHLCV data for a specified period and provide bars sequentially.

    Attributes:
        data (pd.DataFrame): The OHLCV data as a pandas DataFrame.
    """
    data: pd.DataFrame  # Assuming data is a pandas DataFrame

    def __post_init__(self):
        """
        Post-initialization to set up additional attributes.
        """
        self.start_data: pd.DataFrame = field(init=False, default=None)
        self.current_index: int = field(init=False, default=0)
        self.current_data: pd.DataFrame = field(init=False, default=None)

    def serv_period(self, days_ago=0, months_ago=0, date=None, start_time='09:30:00', end_time='16:00:00'):
        """
        Set the period for which the data will be served.

        Args:
            days_ago (int): Number of days ago from today. Defaults to 0.
            months_ago (int): Number of months ago from today. Defaults to 0.
            date (str): Specific date in 'dd-mm-yyyy' format. If provided, it takes priority over days_ago and months_ago.
            start_time (str): Start time in 'HH:MM:SS' format. Defaults to '09:30:00'.
            end_time (str): End time in 'HH:MM:SS' format. Defaults to '16:00:00'.
        """
        if date:
            date_obj = datetime.strptime(date, '%d-%m-%Y')
        else:
            # Adjust date_obj based on days_ago and months_ago
            date_obj = self.data.index[-1].date()
            if days_ago:
                date_obj -= timedelta(days=days_ago)
            if months_ago:
                date_obj -= timedelta(days=months_ago * 30)  # Approximation
        
        start_datetime = datetime.combine(date_obj, datetime.strptime(start_time, '%H:%M:%S').time())
        end_datetime = datetime.combine(date_obj, datetime.strptime(end_time, '%H:%M:%S').time())

        # Filter the data for the given period
        self.start_data = self.data[(self.data.index >= start_datetime) & (self.data.index <= end_datetime)]
        return self.start_data
    
    def serv_range(self, dayrange=(-5, -2), start_time='08:30:00', end_time='18:00:00'):
        """
        Set the period for which the data will be served based on a range of days.

        Args:
            dayrange (tuple): A tuple specifying the range of days (start, end).
            start_time (str): Start time in 'HH:MM:SS' format. Defaults to '08:30:00'.
            end_time (str): End time in 'HH:MM:SS' format. Defaults to '18:00:00'.
        """
        start_day, end_day = dayrange

        if start_day <= 0 and end_day <= 0:
            end_date_obj = self.data.index[-1].date()
            start_datetime = datetime.combine(end_date_obj + timedelta(days=start_day), datetime.strptime(start_time, '%H:%M:%S').time())
            end_datetime = datetime.combine(end_date_obj + timedelta(days=end_day), datetime.strptime(end_time, '%H:%M:%S').time())
        elif start_day >= 0 and end_day >= 0:
            start_date_obj = self.data.index[0].date()
            start_datetime = datetime.combine(start_date_obj + timedelta(days=start_day), datetime.strptime(start_time, '%H:%M:%S').time())
            end_datetime = datetime.combine(start_date_obj + timedelta(days=end_day), datetime.strptime(end_time, '%H:%M:%S').time())
        else:
            raise ValueError("Day range must be either both positive or both negative, for example (-5, -2) or (2, 5)")


        # Filter the data for the given range and time
        self.start_data = self.data.between_time(start_time, end_time) 
        self.start_data = self.start_data[(self.start_data.index >= start_datetime) & (self.start_data.index <= end_datetime)]
        self.start_data = self.start_data.iloc[1:] # Remove the first row to previousn day's close
        self.current_data = self.start_data.copy()
        return self.start_data

    def next_bar(self, histBars=1):
        """
        Serve the next bar or a slice of bars from the period data.

        Args:
            histBars (int): Number of bars to include in the slice. Defaults to 1.

        Returns:
            pd.DataFrame or None: The next bar or slice of bars, or None if all bars have been served.
        """
        if self.start_data is None or self.current_data.index[-1] == self.data.index[-1]:
            return None
        
        # Calculate the start index for slicing
        end_current_index = self.current_data.index[-1]
        next_num = self.data.index.get_loc(end_current_index) + 1
        start_num = next_num - histBars

        # Slice the data from start_index to end_index
        next_bars = self.data.iloc[start_num:next_num+1]

        # Update the current slice of data
        self.current_data = self.data.loc[self.current_data.index[0]:next_bars.index[-1]]
        
        return next_bars
    
    def get_current_data(self):
        """
        Get the current slice of served data.

        Returns:
            pd.DataFrame: The current slice of served data.
        """
        return self.current_data