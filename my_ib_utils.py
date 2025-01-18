from ib_insync import IB

# class IBRateLimiter:
#     """
#     Rate limiter for Interactive Brokers API requests using IB_insync's sleep method
#     """
#     def __init__(self, ib: IB, requests_per_second: float = 2):
#         """
#         Initialize rate limiter
        
#         Args:
#             ib: IB instance for using ib.sleep()
#             requests_per_second: Maximum sustained requests per second
#         """
#         self.ib = ib
#         self.min_interval = 1.0 / requests_per_second

#     def wait(self):
#         """Wait using IB_insync's sleep method"""
#         self.ib.sleep(self.min_interval)


class IBRateLimiter:
    """
    Rate limiter for Interactive Brokers API requests using IB_insync's sleep method.
    With this implementation, any attempt to create a new instance of IBRateLimiter will return the same instance, ensuring consistency across your program.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(IBRateLimiter, cls).__new__(cls)
        return cls._instance

    def __init__(self, ib: IB, requests_per_second: float = 2):
        """
        Initialize rate limiter
        
        Args:
            ib: IB instance for using ib.sleep()
            requests_per_second: Maximum sustained requests per second
        """
        if not hasattr(self, 'initialized'):  # Ensure __init__ is only called once
            self.ib = ib
            self.min_interval = 1.0 / requests_per_second
            self.initialized = True

    def wait(self):
        """Wait using IB_insync's sleep method"""
        self.ib.sleep(self.min_interval)



from datetime import datetime, timezone
import pytz
from ib_insync import Stock

def is_within_trading_hours(ib, symbol: str, exchange: str = 'SMART', debug: bool = False) -> bool:
    """
    Check if a given stock is actually trading right now using IB's trading schedule.
    
    Args:
        ib: The IB instance from ib_insync
        symbol: The stock symbol (e.g., 'TSLA')
        exchange: The exchange to use (default: 'SMART')
        debug: If True, prints detailed debugging information (default: False)
    """
    # Exchange timezone mapping
    EXCHANGE_TIMEZONES = {
        'NYSE': 'America/New_York',
        'NASDAQ': 'America/New_York',
        'ISLAND': 'America/New_York',
        'ARCA': 'America/New_York',
        'IEX': 'America/New_York',
        'BATS': 'America/New_York',
        'LSE': 'Europe/London',
        'TSE': 'Asia/Tokyo',
    }
    
    try:
        # Create and qualify contract
        contract = Stock(symbol, exchange, 'USD')
        qualified_contract = ib.qualifyContracts(contract)[0]
        
        # Get exchange timezone
        exchange_name = qualified_contract.primaryExchange or qualified_contract.exchange
        timezone_str = EXCHANGE_TIMEZONES.get(exchange_name, 'America/New_York')
        exchange_tz = pytz.timezone(timezone_str)
        
        if debug:
            print("\n=== Trading Hours Debug Info ===")
            print(f"Symbol: {symbol}")
            print(f"Exchange: {exchange}")
            print(f"Qualified Exchange: {qualified_contract.exchange}")
            print(f"Primary Exchange: {qualified_contract.primaryExchange}")
            print(f"Exchange Timezone: {timezone_str}")
            
        # Get trading schedule
        contract_details = ib.reqContractDetails(qualified_contract)[0]
        schedule = contract_details.liquidHours
        
        if not schedule:
            schedule = contract_details.tradingHours
            if debug:
                print("\nUsing tradingHours (liquidHours not available)")
        else:
            if debug:
                print("\nUsing liquidHours")
                
        if debug:
            print(f"Raw Schedule: {schedule}")
            
        if not schedule:
            raise ValueError(f"Could not get trading schedule for {symbol}")
            
        # Current time in UTC and exchange timezone
        now_utc = datetime.now(timezone.utc)
        now_exchange = now_utc.astimezone(exchange_tz)
        
        if debug:
            print(f"\nCurrent UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"Current {exchange_name} time: {now_exchange.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print("\nParsed Trading Windows:")
        
        # Parse the schedule
        for segment in schedule.split(';'):
            if '-' in segment and len(segment) == 9:  # Time range format: hhmm-hhmm
                start_str, end_str = segment.split('-')
                
                # Convert time strings to datetime objects (in UTC)
                start_time = datetime.now(timezone.utc).replace(
                    hour=int(start_str[:2]),
                    minute=int(start_str[2:]),
                    second=0,
                    microsecond=0
                )
                
                end_time = datetime.now(timezone.utc).replace(
                    hour=int(end_str[:2]),
                    minute=int(end_str[2:]),
                    second=0,
                    microsecond=0
                )
                
                # Convert to exchange timezone for debug output
                start_time_exchange = start_time.astimezone(exchange_tz)
                end_time_exchange = end_time.astimezone(exchange_tz)
                
                if debug:
                    print(f"\nTrading Window:")
                    print(f"UTC: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
                    print(f"{exchange_name}: {start_time_exchange.strftime('%H:%M')} - {end_time_exchange.strftime('%H:%M')}")
                    print(f"Current time within this window? {start_time <= now_utc <= end_time}")
                
                if start_time <= now_utc <= end_time:
                    if debug:
                        print("\n=== RESULT: MARKET IS OPEN ===")
                    return True
        
        if debug:
            print("\n=== RESULT: MARKET IS CLOSED ===")
        return False
        
    except Exception as e:
        error_msg = f"Error checking trading hours for {symbol}: {str(e)}"
        if debug:
            print(f"\n=== ERROR ===\n{error_msg}")
        raise ValueError(error_msg)

# Usage example:
"""
from ib_insync import *

# Initialize IB
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Check if Tesla is within trading hours
is_trading = is_within_trading_hours(ib, 'TSLA', debug=True)
print(f"TSLA is {'within' if is_trading else 'outside'} trading hours")
"""