import pandas as pd
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass
from strategies.strategies import EntryStrategy, ExitStrategy
from data.data_manager import DataManager

@dataclass
class BackTester:
    data: pd.DataFrame # This will be the data that is used for backtesting
    """A class to manage the backtesting process."""

    def __post_init__(self):
        self.results = []
        self.ohlcv_init = []
        self.ohlcv_current = [] # This gets updated for every new bar which is given to the update method 

    def run_setup():
        # This method will be used to set up the data for trading
        # It will get the data from the data manager and set up the initial ohlcv data for the stock

    def update_(self, data):
        # This method will be used to update the current ohlcv data for the stock
        

    def run(self, data):

        # 

        # first 


        
