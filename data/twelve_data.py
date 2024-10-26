import time
from datetime import datetime, timedelta
import pytz
from tzlocal import get_localzone
from twelvedata import TDClient
import pandas as pd

class TwelveData:
    def __init__(self, api_key, symbols):
        self.api_key = api_key
        self.symbols = symbols
        
        self.messages_history = []
        self.ohlc_data = {symbol: [] for symbol in symbols}
        self.current_minute_data = {symbol: {} for symbol in symbols}
        self.td = TDClient(apikey=self.api_key)
        # Cache to store fetched data
        self.data_cache = {}  # Format: {symbol: {interval: pd.DataFrame}}

    def get_next_lowest_interval(self, requested_interval):
        """
        Find the next lowest available interval that is a common denominator of the requested interval.
        """
        available_intervals = ['1min', '5min', '15min', '30min', '45min', '1h', '2h', '4h', '8h', '1day', '1week', '1month']
        interval_map = {
            '1min': 1, '5min': 5, '15min': 15, '30min': 30, '45min': 45,
            '1h': 60, '2h': 120, '4h': 240, '8h': 480,
            '1day': 1440, '1week': 10080, '1month': 43200
        }
        
        # Convert requested interval to minutes
        if 'min' in requested_interval:
            requested_minutes = int(requested_interval.replace('min', ''))
        elif 'h' in requested_interval:
            requested_minutes = int(requested_interval.replace('h', '')) * 60
        elif 'day' in requested_interval:
            requested_minutes = int(requested_interval.replace('day', '')) * 1440
        elif 'week' in requested_interval:
            requested_minutes = int(requested_interval.replace('week', '')) * 10080
        elif 'month' in requested_interval:
            requested_minutes = int(requested_interval.replace('month', '')) * 43200
        else:
            raise ValueError(f"Invalid interval: {requested_interval}")
        
        # Find all common denominators that are less than requested interval
        common_denominators = []
        for interval in available_intervals:
            interval_minutes = interval_map[interval]
            if interval_minutes <= requested_minutes and requested_minutes % interval_minutes == 0:
                common_denominators.append(interval)
        
        # If we found common denominators, return the highest one
        if common_denominators:
            return common_denominators[-1]
        
        # If no common denominators, find the highest interval that's less than requested
        valid_intervals = [interval for interval in available_intervals 
                          if interval_map[interval] < requested_minutes]
        return valid_intervals[-1] if valid_intervals else '1min'

    def _resample_data(self, df, target_interval):
        """
        Resample data to the target interval.
        """
        # Convert interval string to pandas offset string
        interval_map = {
            'min': 'min',
            'h': 'H',
            'day': 'D',
            'week': 'W',
            'month': 'M'
        }
        
        # Extract number and unit from interval
        import re
        match = re.match(r'(\d+)(\w+)', target_interval)
        if not match:
            raise ValueError(f"Invalid interval format: {target_interval}")
        
        num, unit = match.groups()
        if unit not in interval_map:
            raise ValueError(f"Unsupported interval unit: {unit}")
        
        # Create pandas resample rule
        rule = f"{num}{interval_map[unit]}"
        
        # Resample the data
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled.dropna()

    def get_historical_data(self, symbol, interval='1day', start_date=None, end_date=None, outputsize=None, timezone="America/New_York"):
        """
        Retrieve historical stock data from Twelve Data API.
        """
        tz = pytz.timezone(timezone)
        
        if start_date:
            start_date = tz.localize(datetime.strptime(start_date, "%Y-%m-%d %H:%M")).isoformat()
        if end_date:
            end_date = tz.localize(datetime.strptime(end_date, "%Y-%m-%d %H:%M")).isoformat()

        # Initialize symbol cache if it doesn't exist
        if symbol not in self.data_cache:
            self.data_cache[symbol] = {}

        # Get the lowest common denominator interval that we need to fetch
        base_interval = self.get_next_lowest_interval(interval)
        
        # Check if we already have this data in cache
        if base_interval not in self.data_cache[symbol]:
            # Fetch the data and store in cache
            ts = self.td.time_series(
                symbol=symbol,
                interval=base_interval,
                start_date=start_date,
                end_date=end_date,
                outputsize=outputsize,
                timezone=timezone,
            )
            self.data_cache[symbol][base_interval] = ts.as_pandas()

        # Get the data from cache
        df = self.data_cache[symbol][base_interval]
        
        # If requested interval is different from base interval, resample the data
        if interval != base_interval:
            df = self._resample_data(df, interval)
        
        return df

    def get_last_n_days(self, symbol, n_days, interval='1day'):
        """
        Retrieve data for the last N days.
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=n_days)).strftime('%Y-%m-%d')
        return self.get_historical_data(symbol, interval, start_date, end_date)

    def on_event(self, e):
        if e['event'] == 'price':
            symbol = e['symbol']
            timestamp = datetime.fromtimestamp(e['timestamp'])
            
            # Set timezone to exchange timezone if available, otherwise use local timezone
            if 'timezone' in e:
                tz = pytz.timezone(e['timezone'])
            else:
                tz = get_localzone()
            
            timestamp = timestamp.astimezone(tz)
            # Set seconds to zero
            timestamp = timestamp.replace(second=0, microsecond=0)
            minute = timestamp.minute
            price = float(e['price'])
            
            if 'minute' not in self.current_minute_data[symbol] or self.current_minute_data[symbol]['minute'] != minute:
                if 'minute' in self.current_minute_data[symbol]:
                    self.ohlc_data[symbol].append(self.current_minute_data[symbol])
                self.current_minute_data[symbol] = {
                    'symbol': symbol,
                    'minute': minute,
                    'datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': e.get('volume', 0)
                }
            else:
                self.current_minute_data[symbol]['high'] = max(self.current_minute_data[symbol]['high'], price)
                self.current_minute_data[symbol]['low'] = min(self.current_minute_data[symbol]['low'], price)
                self.current_minute_data[symbol]['close'] = price
                if 'volume' in e:
                    self.current_minute_data[symbol]['volume'] += e['volume']
            
            self.messages_history.append(e)
        print(e)

    def subscribe(self, iterations=None, show_messages=False, until=None):
        if (iterations is None and until is None) or (iterations is not None and until is not None):
            raise ValueError("Either 'iterations' or 'until' must be provided, but not both")
        
        print('-----------------------')
        print('  Websocket CONNECTED  ')
        print('-----------------------')
        
        def run_subscription(condition, show_messages):
            nonlocal iterations  # Access the outer iterations variable
            ws = self.td.websocket(symbols="USD", on_event=self.on_event)
            ws.subscribe(self.symbols)
            ws.connect()
            
            iteration_count = 0
            
            while condition():
                if show_messages:
                    print('messages received: ', len(self.messages_history))
                    print('iteration:', iteration_count + 1)
                
                ws.heartbeat()
                time.sleep(5)  # Wait for 5 seconds
                
                # Increment counter regardless of whether a message was received
                if iterations is not None:
                    iteration_count += 1
                    if iteration_count >= iterations:
                        break

            ws.disconnect()
            print('--------------------------')
            print('  Websocket DISCONNECTED  ')
            print('--------------------------') 
            
            # Add the last minute's data if it exists
            for symbol in self.symbols:
                if self.current_minute_data[symbol]:
                    self.ohlc_data[symbol].append(self.current_minute_data[symbol])
        
        if iterations is not None:
            condition = lambda: True  # We'll handle iterations in the loop
            run_subscription(condition, show_messages)
        elif until is not None:
            end_time = datetime.strptime(until, '%Y-%m-%d %H:%M:%S')
            if end_time <= datetime.now():
                raise ValueError("The 'until' time must be in the future")
            condition = lambda: datetime.now() < end_time
            run_subscription(condition, show_messages)
    
    def get_df(self):
        all_data = []
        for symbol in self.symbols:
            all_data.extend(self.ohlc_data[symbol])
        return pd.DataFrame(all_data).drop(columns=['minute']).set_index('datetime').sort_index()
