import requests
import pandas as pd
import os
from datetime import datetime, timedelta

class TwelveData:
    """
    A class to interact with the Twelve Data API for retrieving historical stock data.
    """

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key):
        """
        Initialize the TwelveData instance with the API key.

        :param api_key: Your Twelve Data API key
        """
        self.api_key = '171136ac7161454b8f4abeb987c72b02' # temp while testing the code

    def get_historical_data(self, symbol, interval='1day', start_date=None, end_date=None, outputsize=None):
        """
        Retrieve historical stock data from Twelve Data API.

        :param symbol: The stock symbol (e.g., 'AAPL' for Apple Inc.)
        :param interval: Time interval between two consecutive data points (default: '1day')
        :param start_date: Start date for the data (format: 'YYYY-MM-DD')
        :param end_date: End date for the data (format: 'YYYY-MM-DD')
        :param outputsize: Number of data points to retrieve (max 5000)
        :return: DataFrame containing the historical data
        """
        endpoint = f"{self.BASE_URL}/time_series"

        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
        }

        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if outputsize:
            params["outputsize"] = min(outputsize, 5000)  # Ensure outputsize doesn't exceed 5000

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            
            if 'values' in data:
                df = pd.DataFrame(data['values'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                df = df.astype(float)
                
                return df
            else:
                print(f"Error: No data found in the response for {symbol}")
                print(f"Full response: {data}")  # Print the full response for debugging
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while fetching data: {e}")
            return None

    def get_last_n_days(self, symbol, n_days, interval='1day'):
        """
        Retrieve data for the last N days.

        :param symbol: The stock symbol (e.g., 'AAPL' for Apple Inc.)
        :param n_days: Number of days to retrieve data for
        :param interval: Time interval between two consecutive data points (default: '1day')
        :return: DataFrame containing the historical data
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=n_days)).strftime('%Y-%m-%d')
        return self.get_historical_data(symbol, interval, start_date, end_date)

# Example usage:
# api_key = "YOUR_API_KEY"
# td = TwelveData(api_key)
# data = td.get_historical_data("AAPL", start_date="2023-01-01", end_date="2023-05-01")
# print(data)

# last_30_days = td.get_last_n_days("AAPL", 30)
# print(last_30_days)
