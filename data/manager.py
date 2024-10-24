from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
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
    timeframes: List[Tuple[str, str]]  # e.g., [("1min", "1day"), ("5min", "1day"), ("15min", "3days")]
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
        valid_intervals = {
            "1min", "5min", "15min", "30min", "45min",
            "1h", "2h", "4h", "1day", "1week", "1month"
        }
        
        # Extract intervals from timeframe tuples
        intervals = [tf[0] for tf in self.timeframes]
        invalid_intervals = set(intervals) - valid_intervals
        if invalid_intervals:
            raise ValueError(
                f"Invalid intervals: {invalid_intervals}. "
                f"Supported intervals are: {valid_intervals}"
            )
        
        # Validate lookback periods
        valid_period_units = {'day', 'days', 'week', 'weeks', 'month', 'months'}
        for _, period in self.timeframes:
            # Extract number and unit from period (e.g., "3days" -> (3, "days"))
            number = ''.join(filter(str.isdigit, period))
            unit = ''.join(filter(str.isalpha, period)).lower()
            
            if not number or not unit:
                raise ValueError(f"Invalid lookback period format: {period}")
            if unit not in valid_period_units:
                raise ValueError(
                    f"Invalid period unit in {period}. "
                    f"Supported units are: {valid_period_units}"
                )
        
        # Sort timeframes from smallest to largest interval
        timeframe_minutes = {
            "1min": 1, "5min": 5, "15min": 15, "30min": 30, "45min": 45,
            "1h": 60, "2h": 120, "4h": 240, "1day": 1440,
            "1week": 10080, "1month": 43200
        }
        self.timeframes.sort(key=lambda x: timeframe_minutes[x[0]])
        self._timeframe_minutes = {tf[0]: timeframe_minutes[tf[0]] for tf in self.timeframes}
    
    def _parse_lookback_period(self, period: str) -> timedelta:
        """Convert lookback period string to timedelta"""
        number = int(''.join(filter(str.isdigit, period)))
        unit = ''.join(filter(str.isalpha, period)).lower()
        
        if unit in ['day', 'days']:
            return timedelta(days=number)
        elif unit in ['week', 'weeks']:
            return timedelta(weeks=number)
        elif unit in ['month', 'months']:
            return timedelta(days=number * 30)  # Approximate
        else:
            raise ValueError(f"Unsupported time unit: {unit}")
    
    def load_historical_data(self) -> None:
        """
        Load initial historical data for all timeframes using TwelveData API.
        """
        for interval, lookback in self.timeframes:
            try:
                # Calculate start date based on lookback period
                lookback_td = self._parse_lookback_period(lookback)
                start_date = max(
                    self.init_start_date,
                    self.init_end_date - lookback_td
                )
                
                # Get historical data from TwelveData
                df = self._td_client.get_historical_data(
                    symbol=self.symbol,
                    start_date=start_date,
                    end_date=self.init_end_date,
                    interval=interval
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
                    self._data[interval] = df
                else:
                    raise ValueError(f"No data received for {self.symbol} at {interval}")
                    
            except Exception as e:
                print(f"Error loading {interval} data for {self.symbol}: {str(e)}")
                # Initialize empty DataFrame with correct structure
                self._data[interval] = pd.DataFrame(
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
        
        for interval, _ in self.timeframes:
            self._update_timeframe(interval, tick_time, price, volume)
        
        self._last_update = tick_time
    
    def _update_timeframe(self, interval: str, 
                         tick_time: datetime, 
                         price: float, 
                         volume: float) -> None:
        """Update a specific timeframe with new tick data"""
        # Get the start of the current bar for this timeframe
        minutes = self._timeframe_minutes[interval]
        
        # Handle different timeframe types
        if interval.endswith('month'):
            bar_start = tick_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif interval.endswith('week'):
            bar_start = tick_time - timedelta(days=tick_time.weekday())
            bar_start = bar_start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif interval.endswith('day'):
            bar_start = tick_time.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # For hour and minute bars
            bar_start = tick_time.floor(f'{minutes}min')
        
        df = self._data[interval]
        
        if bar_start not in df.index:
            # Create new bar
            new_bar = pd.Series({
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }, name=bar_start)
            
            self._data[interval] = pd.concat([
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
        for interval, _ in self.timeframes:
            df = self._data[interval]
            if not df.empty:
                current_bars[interval] = df.iloc[-1]
        return current_bars
    
    def get_data(self, 
                 interval: str, 
                 lookback: int = None) -> pd.DataFrame:
        """
        Get historical data for a specific timeframe
        
        Args:
            interval: The timeframe interval to retrieve
            lookback: Number of bars to look back (None for all data)
        
        Returns:
            DataFrame with historical data
        """
        if interval not in [tf[0] for tf in self.timeframes]:
            raise ValueError(f"Invalid interval: {interval}")
        
        data = self._data[interval]
        if lookback is not None:
            data = data.tail(lookback)
        
        return data.copy()
    
    def cleanup_old_data(self) -> None:
        """Remove data older than specified lookback periods"""
        for interval, lookback in self.timeframes:
            lookback_td = self._parse_lookback_period(lookback)
            cutoff_date = datetime.now() - lookback_td
            self._data[interval] = self._data[interval][
                self._data[interval].index >= cutoff_date
            ]

# Example usage:
if __name__ == "__main__":
    api_key = "YOUR_TWELVE_DATA_API_KEY"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 2)
    
    tesla_data = MarketDataManager(
        symbol="TSLA",
        timeframes=[
            ("1min", "1day"),   # 1-minute candles for the past day
            ("5min", "1day"),   # 5-minute candles for the past day
            ("15min", "3days"), # 15-minute candles for the past 3 days
            ("1h", "5days")     # 1-hour candles for the past 5 days
        ],
        init_start_date=start_date,
        init_end_date=end_date,
        api_key=api_key
    )
    
    # Get current bars for all timeframes
    current_bars = tesla_data.get_current_bars()
    
    # Get last 10 bars of 5-minute data
    five_min_data = tesla_data.get_data("5min", lookback=10)
