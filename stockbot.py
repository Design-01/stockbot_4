import pandas as pd
from typing import List, Optional
from dataclasses import dataclass, field
from ib_insync import *


# ----------- SB Imports -----------------------------------
import scanner.market_analyzer as ma
from bank.bank import Bank
from frame.frame import Frame
from scanner.scanner import StockbotScanner
from stock import StockX
from typing import Tuple
from data.live_ib_data import LiveData



@dataclass
class StockBot:

    def __post_init__(self):
       self.ib = IB()
       self.ma = ma.IBMarketAnalyzer(self.ib)
       self.scanner = StockbotScanner(self.ib)
       self.livedata = LiveData(self.ib)
       self.stocks = {}
       self.stats_daily = []
       self.stocksKeepLive = [] # list of symbols to keep live

       # --- store results
       self.tracker_df = pd.DataFrame()
       self.allowedETFs = []

    def insert_tracker_data(self, whereValue:str, insertAtColumn:float, newValue:float):
        self.tracker_df.loc[self.tracker_df['symbol'] == whereValue, insertAtColumn] = newValue

    def connect_to_ib(self, paper:bool=True, clientId:int=10, runFromNotebook:bool=True):
        if paper: 
            if runFromNotebook:
                util.startLoop()
            self.ib.connect('127.0.0.1', 7496, clientId=clientId)

    def disiconnect_from_ib(self):
        self.ib.disconnect()

    def sleep(self, seconds:int=1):
        self.ib.sleep(seconds)

    def save_stats_daily_to_csv(self, folder_path:str='scanner\day_results'):
        """
        Saves the scan results to a CSV file in the specified folder.
        
        Args:
            folder_path (str): The folder path where the CSV file will be saved.
        
        Returns:
            str: The full path of the saved CSV file.
        """
        import os
        from datetime import datetime
        
        # Get the results as a DataFrame
        results_df = pd.DataFrame(self.stats_daily)
        
        # If there are no results, print a message and return
        if results_df.empty:
            print("No results to save.")
            return None
        
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        current_date = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        filename = f"Day_Results_{current_date}.csv"
        file_path = os.path.join(folder_path, filename)
        results_df.to_csv(file_path, index=False)
        
        print(f"Results saved to: {file_path}")
        return file_path


    #£
    def scan(self, 
             scanCode:str ='TOP_PERC_GAIN',
            priceRange: tuple[float, float] = (1, 100),
            avgVolumeAbove: int = 100_000,
            changePercent: float = 4,  # Changed from changePercAbove
            marketCapRange: tuple[float, float] = (100, 10000), # value is in millions
            location='STK.US.MAJOR'):
        # multi scan breaks the scan by market cap into smaller chunks so more than 50 stocks can be scanned
        
        self.scanner.scan(
            scanCode=scanCode, 
            priceRange=priceRange,
            avgVolumeAbove=avgVolumeAbove, 
            changePercent=changePercent, 
            marketCapRange=marketCapRange, # in millions
        )

    #£
    def setup_stocks_from_scanner(self) -> List[StockX]:
        # set up spy first 
        self.stocks['SPY'] = StockX(self.ib, 'SPY', ls='LONG')

        # get list of stocks from scanner
        for row in self.scanner.get_results().itertuples():
            ls = ''
            if row.scanCode == 'TOP_PERC_GAIN': ls = 'LONG'
            elif row.scanCode == 'TOP_PERC_LOSE': ls = 'SHORT'
            self.stocks[row.symbol] = StockX(self.ib, row.symbol, ls=ls)

        return self.stocks
    
    def setup_stocks(self, symbols:List[str], ls:str='LONG') -> List[StockX]:
        for symbol in symbols:
            self.stocks[symbol] = StockX(self.ib, symbol, ls=ls)
        return self.stocks


    #£
    def run_stock_daily_analysis(self, ls:str='LONG', limit:int=5):
        self.stocks['SPY'].RUN_DAILY(isMarket=True)
        count = 0

        for stock in self.stocks.values():
            if count == limit: break
            if stock.isMarket: continue # automatically skip the market. gets set to True within StockX when the stock is SPY or QQQ
            stock.RUN_DAILY(ls='LONG', spy=self.stocks['SPY'])
            # stock.set_daily_stats()
            # self.stats_daily.append(stock.stats_daily)
            
            count += 1

        self.save_stats_daily_to_csv()

        print(f"Daily analysis complete:: {count} stocks analyzed")
        

    def get_daily_stats(self, query=None, sort_by='score_1D', ascending=False, top_n=None, columns=None):
        """
        Get and filter daily stock statistics
        
        Parameters:
        -----------
        query : str - Pandas query string (e.g., "price > 50 and score_1D > 5")
        sort_by : str or list - Column(s) to sort by
        ascending : bool or list of bools - Sort order
        top_n : int - Number of top stocks to return
        columns : list - Specific columns to return
        
        Returns:
        --------
        DataFrame - Filtered and sorted DataFrame
        """
        # Create DataFrame from stats list
        df = pd.DataFrame(self.stats_daily)
        
        if query: df = df.query(query)
        if sort_by: df = df.sort_values(by=sort_by, ascending=ascending)
        if top_n is not None: df = df.head(top_n)
        if columns: df = df[columns]
    
        return df
    
    def run_fundamentals(self, allowedETFs:Optional[List[str]]=None):
        """Checks and logs the fundamentals and sets the results to the scanner results DataFrame."""
        self.tracker_df['allowedETF'] = False

        for symbol, sx in self.stocks.items():
            sx.fundamentals.req_fundamentals(max_days_old=1)  # gets the fundamentals from the IB API
            
            if sx.fundamentals.validate_fundamental('primary_etf', 'isin', allowedETFs, description='Stocks primary sector ETF is allowed?'):
                print(f"Symbol {sx.symbol} is not allowed")
                continue

            self.tracker_df.loc[self.tracker_df['symbol'] == sx.symbol, 'allowedETF'] = True

    def run_daily_frames(self, dataType:str='ohlcv', mustHaveAprovedETF:bool=True):
        """Runs the daily data for the stocks in the list of symbols. 
        The data type can be either 'ohlcv' or 'random'. ohlcv gets actual data from the IB API, while random generates random data for testing purposes."""     
        
        for symbol in self.tracker_df['symbol'].to_list():
            if mustHaveAprovedETF:
                if not self.tracker_df.loc[self.tracker_df['symbol'] == symbol, 'allowedETF'].values[0]:
                    print(f"Symbol {symbol} did not pass the fundamentals")
                    continue

            self.stocks[symbol].set_up_frame('1 day',  'mkt',        start_date="52 weeksAgo", end_date="now") 
            self.stocks[symbol].set_up_frame('1 day', 'primary_etf', start_date="52 weeksAgo", end_date="now") 
            self.stocks[symbol].set_up_frame('1 day',  dataType,     start_date="52 weeksAgo", end_date="now") 
            self.stocks[symbol].run_daily_frame(lookback=300)
            stock_data_last_row = self.stocks[symbol].frames['1 day'].data.iloc[-1]
            col_to_track = 'PBX_ALL_Scores'
            score = stock_data_last_row[col_to_track]
            self.insert_tracker_data(symbol, col_to_track, score)

    def run_live_data(self, symbols:List[str]):
        """Runs the live data for the stocks in the list of symbols."""
        self.stocksKeepLive = symbols
        self.livedata.setup_tickers(self.stocksKeepLive)
        self.livedata.setup_contracts()
        self.livedata.reqLiveBars(show=False)

    def run_live_frames(self, dataType:str='ohlcv'):
        """Runs the live data for the stocks in the list of symbols."""
        for symbol in self.stocksKeepLive:
            self.stocksKeepLive[symbol].set_up_frame('1 min', 'mkt', start_date="1 D", end_date="now") 
            self.stocksKeepLive[symbol].set_up_frame('1 min', 'primary_etf', start_date="1 D", end_date="now") 
            self.stocksKeepLive[symbol].set_up_frame('1 min', dataType, start_date="1 D", end_date="now") 
            self.stocksKeepLive[symbol].run_live_frame(lookback=300)




