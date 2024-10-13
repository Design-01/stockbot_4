"""
This class manages the data sources so it can retrieve the open high low close and volume prices from either random data classes or 12 data or from interactive brokers etc. 
The idea is this is a single point of entry for open high low close and volume prices
"""

from dataclasses import dataclass
import pandas as pd
import os
from data.twelve_data import TwelveData
from data.random_data import RandomOHLCV

@dataclass
class OHLCV:
    api_key: str = None
    data_folder: str = os.path.join('data', 'historical_data')

    def __post_init__(self):
        os.makedirs(self.data_folder, exist_ok=True)
        if self.api_key:
            self.twelve_data = TwelveData(self.api_key)

    def get_data(self, source: str, symbol: str = None, interval: str = '1day', start_date: str = None, 
                 end_date: str = None, save_format: str = 'csv', **kwargs):
        """
        Retrieve data from the specified source and save it in the desired format.

        :param source: Data source ('twelve_data', 'random', or 'file')
        :param symbol: The stock symbol (required for 'twelve_data')
        :param interval: Time interval between data points (default: '1day')
        :param start_date: Start date for the data (format: 'YYYY-MM-DD')
        :param end_date: End date for the data (format: 'YYYY-MM-DD')
        :param save_format: Format to save the data ('csv', 'pickle', or 'excel')
        :param kwargs: Additional arguments for the specific data source
        :return: DataFrame containing the historical data
        """
        if source == 'twelve_data':
            if not self.api_key:
                raise ValueError("API key is required for Twelve Data")
            df = self.twelve_data.get_historical_data(symbol, interval, start_date, end_date, save_format=save_format)
        elif source == 'random':
            random_ohlcv = RandomOHLCV(**kwargs)
            df = random_ohlcv.get_dataframe()
            self.save_data(df, f"{symbol}_random_{interval}", save_format)
        elif source == 'file':
            file_path = kwargs.get('file_path')
            if not file_path:
                raise ValueError("file_path is required for 'file' source")
            df = self.load_data(file_path)
        else:
            raise ValueError(f"Unsupported data source: {source}")

        return df

    def save_data(self, df, filename, save_format):
        """
        Save the data in the specified format.

        :param df: DataFrame containing the data to save
        :param filename: Name of the file (without extension)
        :param save_format: Format to save the data ('csv', 'pickle', or 'excel')
        """
        filepath = os.path.join(self.data_folder, filename)

        if save_format == 'csv':
            df.to_csv(f"{filepath}.csv")
        elif save_format == 'pickle':
            df.to_pickle(f"{filepath}.pkl")
        elif save_format == 'excel':
            df.to_excel(f"{filepath}.xlsx")
        else:
            raise ValueError(f"Unsupported save format: {save_format}")

        print(f"Data saved as {filepath}.{save_format}")

    def load_data(self, file_path):
        """
        Load data from a file.

        :param file_path: Path to the file to load
        :return: DataFrame containing the loaded data
        """
        _, ext = os.path.splitext(file_path)
        if ext == '.csv':
            return pd.read_csv(file_path, index_col='datetime', parse_dates=True)
        elif ext == '.pkl':
            return pd.read_pickle(file_path)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, index_col='datetime', parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

# Example usage:
# ohlcv = OHLCV(api_key="YOUR_API_KEY")
# 
# # Get data from Twelve Data
# twelve_data = ohlcv.get_data('twelve_data', symbol="AAPL", start_date="2023-01-01", end_date="2023-05-01", save_format='csv')
# print(twelve_data)
# 
# # Get random data
# random_data = ohlcv.get_data('random', symbol="RAND", start="2023-01-01", periods=100, freq="1D", open_rng=(-1, 1), close_rng=(-1, 1), save_format='csv')
# print(random_data)
# 
# # Load data from file
# file_data = ohlcv.get_data('file', file_path="path/to/your/data.csv")
# print(file_data)
