from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from twelve_data import TwelveData


@dataclass
class MarketDataManager:
    """
    Manages historical and real-time market data for a single instrument across multiple timeframes.
    Uses TwelveData as the data provider.
    """
    symbol: str
    timeframes: List[str]  # e.g., ["1min", "5min", "15min", "1h"]
    init_start_date: datetime
    init_end_date: datetime
    api_key: str
    _data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    _last_update: datetime = None
    
    def __post_init__(self):
        """Initialize data structures and validate timeframes"""
        self._validate_timeframes()
        self._td_client = TwelveData(self.api_key)
        self.load_historical_data()
    
    def _validate_timeframes(self) -> None:
        """Validate timeframe format against TwelveData's supported intervals"""
        valid_timeframes = {
            "1min", "5min", "15min", "30min", "45min",
            "1h", "2h", "4h", "1day", "1week", "1month"
        }
        
        invalid_timeframes = set(self.timeframes) - valid_timeframes
        if invalid_timeframes:
            raise ValueError(
                f"Invalid timeframes: {invalid_timeframes}. "
                f"Supported timeframes are: {valid_timeframes}"
            )
        
        # Sort timeframes from smallest to largest
        timeframe_minutes = {
            "1min": 1, "5min": 5, "15min": 15, "30min": 30, "45min": 45,
            "1h": 60, "2h": 120, "4h": 240, "1day": 1440,
            "1week": 10080, "1month": 43200
        }
        self.timeframes.sort(key=lambda x: timeframe_minutes[x])
        self._timeframe_minutes = {tf: timeframe_minutes[tf] for tf in self.timeframes}
    
    def load_historical_data(self) -> None:
        """
        Load initial historical data for all timeframes using TwelveData API.
        """
        for timeframe in self.timeframes:
            try:
                # Get historical data from TwelveData
                df = self._td_client.get_historical_data(
                    symbol=self.symbol,
                    start_date=self.init_start_date,
                    end_date=self.init_end_date,
                    interval=timeframe
                )
                
                # Ensure DataFrame has expected columns
                if df is not None and not df.empty:
                    required_columns = ['open', 'high', 'low', 'close', 'volume']
                    missing_columns = set(required_columns) - set(df.columns)
                    if missing_columns:
                        raise ValueError(
                            f"Missing required columns in TwelveData response: {missing_columns}"
                        )
                    
                    # Store the data
                    self._data[timeframe] = df
                else:
                    raise ValueError(f"No data received for {self.symbol} at {timeframe}")
                    
            except Exception as e:
                print(f"Error loading {timeframe} data for {self.symbol}: {str(e)}")
                # Initialize empty DataFrame with correct structure
                self._data[timeframe] = pd.DataFrame(
                    columns=['open', 'high', 'low', 'close', 'volume']
                )
    
    def process_tick(self, tick_data: dict) -> None:
        """
        Process new tick data and update all timeframes
        
        Args:
            tick_data: dict with keys 'timestamp', 'price', 'volume'
        """
        tick_time = pd.to_datetime(tick_data['timestamp'])
        price = tick_data['price']
        volume = tick_data['volume']
        
        for timeframe in self.timeframes:
            self._update_timeframe(timeframe, tick_time, price, volume)
        
        self._last_update = tick_time
    
    def _update_timeframe(self, timeframe: str, 
                         tick_time: datetime, 
                         price: float, 
                         volume: float) -> None:
        """Update a specific timeframe with new tick data"""
        # Get the start of the current bar for this timeframe
        minutes = self._timeframe_minutes[timeframe]
        
        # Handle different timeframe types
        if timeframe.endswith('month'):
            bar_start = tick_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif timeframe.endswith('week'):
            bar_start = tick_time - timedelta(days=tick_time.weekday())
            bar_start = bar_start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe.endswith('day'):
            bar_start = tick_time.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # For hour and minute bars
            bar_start = tick_time.floor(f'{minutes}min')
        
        df = self._data[timeframe]
        
        if bar_start not in df.index:
            # Create new bar
            new_bar = pd.Series({
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }, name=bar_start)
            
            self._data[timeframe] = pd.concat([
                df, 
                pd.DataFrame(new_bar).T
            ]).sort_index()
        
        else:
            # Update existing bar
            current_bar = df.loc[bar_start]
            df.at[bar_start, 'high'] = max(current_bar['high'], price)
            df.at[bar_start, 'low'] = min(current_bar['low'], price)
            df.at[bar_start, 'close'] = price
            df.at[bar_start, 'volume'] += volume
    
    def get_current_bars(self) -> Dict[str, pd.Series]:
        """Get the most recent bars for all timeframes"""
        current_bars = {}
        for timeframe in self.timeframes:
            df = self._data[timeframe]
            if not df.empty:
                current_bars[timeframe] = df.iloc[-1]
        return current_bars
    
    def get_data(self, 
                 timeframe: str, 
                 lookback: int = None) -> pd.DataFrame:
        """
        Get historical data for a specific timeframe
        
        Args:
            timeframe: The timeframe to retrieve
            lookback: Number of bars to look back (None for all data)
        
        Returns:
            DataFrame with historical data
        """
        if timeframe not in self.timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        data = self._data[timeframe]
        if lookback is not None:
            data = data.tail(lookback)
        
        return data.copy()
    
    def cleanup_old_data(self, keep_days: int = 30) -> None:
        """Remove data older than specified number of days"""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        for timeframe in self.timeframes:
            self._data[timeframe] = self._data[timeframe][
                self._data[timeframe].index >= cutoff_date
            ]

# # Example usage:
# if __name__ == "__main__":
#     api_key = "YOUR_TWELVE_DATA_API_KEY"
#     start_date = datetime(2024, 1, 1)
#     end_date = datetime(2024, 1, 2)
    
#     tesla_data = MarketDataManager(
#         symbol="TSLA",
#         timeframes=["1min", "5min", "15min", "1h"],
#         init_start_date=start_date,
#         init_end_date=end_date,
#         api_key=api_key
#     )
    
#     # Get current bars for all timeframes
#     current_bars = tesla_data.get_current_bars()
    
#     # Get last 10 bars of 5-minute data
#     five_min_data = tesla_data.get_data("5min", lookback=10)