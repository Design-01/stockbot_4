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
    trade_start: datetime
    trade_end: datetime
    api_key: str
    mode: str = "hist"  # "hist" for historical, "live" for real-time
    
    def __post_init__(self):
        """Initialize data structures and validate timeframes"""
        self._validate_mode()
        # self._validate_timeframes()
        self._td_client = TwelveData(self.api_key, [self.symbol])
        self._data: Dict[str, pd.DataFrame] = {}  # Historical data for each timeframe
        self._forward_data: Optional[pd.DataFrame] = None  # Forward-looking data for simulation
        self._current_bar_index: int = -1  # Index tracker for forward data, start at -1 to get first bar on first call
        self._last_update: datetime = None
        self._is_live = False
        
        self.load_historical_data()
        if self.mode == "hist":
            self.load_forward_data()
        self.print_timeframe_summary()
    
    def _validate_mode(self) -> None:
        """Validate mode and date settings"""
        if self.mode not in ["hist", "live"]:
            raise ValueError("Mode must be either 'hist' or 'live'")
        
        if self.mode == "live":
            current_time = datetime.now()
            if self.trade_end > current_time:
                raise ValueError("For live mode, trade_end cannot be in the future")
            
            # Adjust trade_end to current time for live mode
            self.trade_end = current_time
    
    def load_forward_data(self) -> None:
        """
        Load forward-looking data for historical simulation.
        Uses the smallest timeframe interval to get data from trade_start to trade_end.
        Data is sorted in ascending order (oldest first) for proper sequential processing.
        """
        if self.mode != "hist":
            return
            
        # Get the smallest timeframe interval
        smallest_interval = self.timeframes[0][0]  # Timeframes are already sorted
        
        try:
            # Format dates for TwelveData API
            start_str = self.trade_start.strftime("%Y-%m-%d %H:%M")
            end_str = self.trade_end.strftime("%Y-%m-%d %H:%M")
            
            # Get forward-looking data
            self._forward_data = self._td_client.get_historical_data(
                symbol=self.symbol,
                interval=smallest_interval,
                start_date=start_str,
                end_date=end_str
            )
            
            if self._forward_data is None or self._forward_data.empty:
                raise ValueError(f"No forward data received for {self.symbol}")
            
            # Sort data in ascending order (oldest first) for proper sequential processing
            self._forward_data = self._forward_data.sort_index(ascending=True)
            
            # Reset the current bar index to -1 to get first bar on first call
            self._current_bar_index = -1
            
            print(f"\nLoaded forward-looking data:")
            print(f"Interval: {smallest_interval}")
            print(f"Bars: {len(self._forward_data)}")
            print(f"Period: {start_str} to {end_str}\n")
            
        except Exception as e:
            print(f"Error loading forward data: {str(e)}")
            self._forward_data = pd.DataFrame(
                columns=['open', 'high', 'low', 'close', 'volume']
            )
    
    def nextbar(self) -> Optional[dict]:
        """
        Get the next bar from forward-looking data and convert it to tick format.
        Returns None when all bars have been processed.
        
        On first call, returns the first bar.
        On subsequent calls, returns the next bar in sequence.
        Returns None when all bars are exhausted.
        """
        if self.mode != "hist" or self._forward_data is None or self._forward_data.empty:
            return None
        
        # Increment index to get next bar
        self._current_bar_index += 1
            
        # Check if we've exhausted all bars
        if self._current_bar_index >= len(self._forward_data):
            return None
            
        # Get the current bar
        current_bar = self._forward_data.iloc[self._current_bar_index]
        bar_time = self._forward_data.index[self._current_bar_index]
        
        # Convert OHLCV bar to tick format
        tick = {
            'timestamp': bar_time,
            'price': current_bar['close'],  # Use close price for the tick
            'volume': current_bar['volume']
        }
        
        return tick
    
    def run_live(self, iterations=None, show_messages=False, until=None) -> None:
        """
        Start receiving live market data through websocket connection.
        
        Args:
            iterations: Number of websocket iterations to run (None for infinite)
            show_messages: Whether to print received messages
            until: Datetime string (format: 'YYYY-MM-DD HH:MM:SS') to run until
        """
        if self.mode != "live":
            raise ValueError("run_live() can only be called in live mode")
        
        if self._is_live:
            raise RuntimeError("Live data stream is already running")
        
        def on_tick_event(e):
            """Custom event handler for websocket data"""
            if e['event'] == 'price':
                tick_data = {
                    'timestamp': datetime.fromtimestamp(e['timestamp']),
                    'price': float(e['price']),
                    'volume': e.get('volume', 0)
                }
                self.process_tick(tick_data)
            
            if show_messages:
                print(e)
        
        # Override the TwelveData client's event handler
        self._td_client.on_event = on_tick_event
        self._is_live = True
        
        try:
            print("\n" + "="*80)
            print(f"Starting live data stream for {self.symbol}")
            print("Historical data loaded up to:", self.trade_end.strftime('%Y-%m-%d %H:%M'))
            print("="*80 + "\n")
            
            # Start the websocket subscription
            self._td_client.subscribe(
                iterations=iterations,
                show_messages=show_messages,
                until=until
            )
        except Exception as e:
            print(f"Error in live data stream: {str(e)}")
        finally:
            self._is_live = False
    
    def print_timeframe_summary(self) -> None:
        """
        Print a summary of loaded timeframes including their start times, end times,
        and number of bars in an easily readable format.
        """
        print("\n" + "="*80)
        print(f"Data Summary for {self.symbol} ({self.mode.upper()} mode)")
        print("="*80)
        
        # Print historical data summary
        print("\nHistorical Data:")
        print("-" * 40)
        
        for interval, lookback in self.timeframes:
            df = self._data[interval]
            if df.empty:
                print(f"\n{interval} Timeframe:")
                print("-" * 40)
                print("Status: No data loaded")
                continue
            
            start_time = df.index.min()
            end_time = df.index.max()
            bar_count = len(df)
            
            print(f"\n{interval} Timeframe:")
            print("-" * 40)
            print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"End Time:   {end_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"Bar Count:  {bar_count:,}")
            print(f"Lookback:   {lookback}")
            
            # Calculate expected vs actual bars for common intervals
            if interval.endswith('min'):
                mins = int(interval.replace('min', ''))
                expected_bars = int((end_time - start_time).total_seconds() / (60 * mins))
                coverage = (bar_count / expected_bars) * 100 if expected_bars > 0 else 0
                print(f"Coverage:   {coverage:.1f}% of expected bars")
        
        # Print forward data summary if in historical mode
        if self.mode == "hist" and self._forward_data is not None and not self._forward_data.empty:
            print("\nForward-Looking Data:")
            print("-" * 40)
            print(f"Interval:    {self.timeframes[0][0]}")
            print(f"Start Time:  {self._forward_data.index.min().strftime('%Y-%m-%d %H:%M')}")
            print(f"End Time:    {self._forward_data.index.max().strftime('%Y-%m-%d %H:%M')}")
            print(f"Bar Count:   {len(self._forward_data):,}")
            print(f"Bars Left:   {len(self._forward_data) - (self._current_bar_index + 1):,}")
        
        print("\n" + "="*80 + "\n")
    
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
        For live mode, loads data up to current time based on lookback periods.
        For historical mode, loads data up to trade_start based on lookback periods.
        """
        for interval, lookback in self.timeframes:
            try:
                # Calculate start date based on lookback period
                lookback_td = self._parse_lookback_period(lookback)
                
                if self.mode == "live":
                    # For live mode, load data up to current time
                    start_date = max(
                        self.trade_start,
                        self.trade_end - lookback_td
                    )
                else:
                    # For historical mode, load data up to trade_start
                    start_date = max(
                        self.trade_start - lookback_td,
                        self.trade_start - lookback_td
                    )
                
                # Format dates for TwelveData API
                start_str = start_date.strftime("%Y-%m-%d %H:%M")
                end_str = (self.trade_start if self.mode == "hist" else self.trade_end).strftime("%Y-%m-%d %H:%M")
                
                # Get historical data from TwelveData
                df = self._td_client.get_historical_data(
                    symbol=self.symbol,
                    interval=interval,
                    start_date=start_str,
                    end_date=end_str
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

    def update_timeframes(self) -> None:
        """
        Update timeframes with new data based on the current mode.
        For historical mode, uses nextbar() to get the next tick.
        For live mode, uses websocket data if available.
        """
        if self.mode == "hist":
            # Get next tick from historical data
            tick = self.nextbar()
            if tick is not None:
                self.process_tick(tick)
        elif self.mode == "live" and self._is_live:
            # Live mode is handled automatically through websocket callbacks
            # The websocket's on_event handler calls process_tick
            pass
        else:
            raise RuntimeError("Cannot update timeframes: websocket not connected in live mode")
