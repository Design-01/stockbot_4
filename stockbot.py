from dataclasses import dataclass, field
from typing import List, Optional
from frame.frame import Frame
from bank.bank import Bank
from data.ohlcv import OHLCV
from strategies.strategies import EntryStrategy, ExitStrategy
from backtesting.backtester import Backtester
from data.data_manager import DataManager
from datetime import datetime

@dataclass
class StockBot:
    mode: str = 'backtest'
    capital: float = 100000.0
    entry_strategies: List[EntryStrategy] = field(default_factory=list)
    exit_strategies: List[ExitStrategy] = field(default_factory=list)
    bank: Bank = field(init=False)
    traders: List = field(default_factory=list)
    backtester: Backtester = field(init=False)
    ohlcv: OHLCV = field(init=False)

    def __post_init__(self):
        self.bank = Bank(self.capital)
        self.ohlcv = OHLCV()  # Initialize OHLCV
        # Retrieve data from a file or other source
        data = self.ohlcv.get_data(source='file', file_path='data/historical_data/AAPL_1day.csv')
        self.data_manager = DataManager(data=data)
        self.backtester = Backtester(self.data_manager, self)

    def run(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        self.backtester.run(start_date=start_date, end_date=end_date)

    def market_bias(self):
        # Placeholder for market bias logic
        pass

    def stock_bias(self):
        # Placeholder for stock bias logic
        pass

    def has_room_to_move(self):
        # Placeholder for room to move logic
        pass

    def has_capital(self):
        # Placeholder for capital check logic
        pass

    def has_position(self):
        # Placeholder for position check logic
        pass

    def trade(self, exit_strategy):
        # Placeholder for trade logic
        pass

    def show_trade_progress(self):
        # Placeholder for trade progress display logic
        pass

    def show_results(self):
        # Placeholder for results display logic
        pass
