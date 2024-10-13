import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockbot import StockBot
from datetime import datetime

# Initialize StockBot
sb = StockBot()

# Define start and end dates for backtesting
start_date = datetime(2020, 1, 1)
end_date = datetime(2020, 12, 31)

# Run the StockBot with the backtester
sb.run_backtest(
    stock='TSLA',
    interval = '5 mins', # interval of data to use for backtesting. check the folders in data/historical_data for available intervals and data. if not avaialb the get it from data\twelve_data.py
    days=1, # number of days of say '5mins' data to use for backtesting.  if intervall is set to a day then there is a min of 30 days data so raise an error
    end_date=end_date, # if end date is set and days is also set then count the number of those from the end date to get the start date and start date will be ignored.
    start_date=start_date, # if Start date is set and days is also set then count the number of those from the start date to get the end date and end date. Setting an end date and the days will take precedence over setting the start date and the days. So if all three are set it will assume the end date and count the number of days back from the end date to get the data which is required to trade
    capital=100000.0, # initial capital for backtesting
    entry_strategies=[], # When the previous elements and basic functionality have been fully tested then we can start implementing a dummy strategy and then move on tomore complex strategies
    exit_strategies=[] # as above comment
    )

# Show results
# Will require a back test results folder to be set up within back testing folder. 
# This will be used to store the results of the backtesting. The results will be stored in a csv file with the name of the stock and the date of the backtest. 
# The results will be stored in the following format: Date, Stock, Capital, Profit, Trades, Win Rate, Loss Rate, Average Win, Average Loss, Average Trade, 
sb.show_backtest_results() 
