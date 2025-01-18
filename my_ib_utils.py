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



from datetime import datetime, time
import pytz
from ib_insync import Stock

def is_within_trading_hours(ib, symbol: str, exchange: str = 'SMART') -> bool:
    """
    Check if a given stock is within its regular trading hours.
    
    Args:
        ib: The IB instance from ib_insync
        symbol: The stock symbol (e.g., 'TSLA')
        exchange: The exchange to use (default: 'SMART')
    
    Returns:
        bool: True if the stock is within trading hours, False otherwise
    """
    # Market hours for different exchanges
    MARKET_HOURS = {
        'NYSE': {
            'timezone': 'America/New_York',
            'open': time(9, 30),
            'close': time(16, 0)
        },
        'NASDAQ': {
            'timezone': 'America/New_York',
            'open': time(9, 30),
            'close': time(16, 0)
        },
        'LSE': {
            'timezone': 'Europe/London',
            'open': time(8, 0),
            'close': time(16, 30)
        },
        'TSE': {
            'timezone': 'Asia/Tokyo',
            'open': time(9, 0),
            'close': time(15, 30)
        }
    }

    # Exchange mapping for alternative codes
    EXCHANGE_MAPPING = {
        'ISLAND': 'NASDAQ',
        'ARCA': 'NYSE',
        'IEX': 'NYSE',
        'BATS': 'NYSE',
        'AMEX': 'NYSE',
    }

    # Create contract and get details from IB
    contract = Stock(symbol, exchange, 'USD')
    qualified_contract = ib.qualifyContracts(contract)[0]
    
    # Get the primary exchange
    primary_exchange = qualified_contract.primaryExchange or qualified_contract.exchange
    primary_exchange = primary_exchange.upper()
    
    # Map to primary exchange if needed
    if primary_exchange in EXCHANGE_MAPPING:
        primary_exchange = EXCHANGE_MAPPING[primary_exchange]
    
    # Get market info
    market_info = MARKET_HOURS.get(primary_exchange)
    if not market_info:
        raise ValueError(f"Unsupported exchange: {primary_exchange}")
    
    # Get current time in market timezone
    market_tz = pytz.timezone(market_info['timezone'])
    current_time = datetime.now(market_tz).time()
    
    # Check if within trading hours
    return market_info['open'] <= current_time <= market_info['close']

# Usage example:
"""
from ib_insync import *

# Initialize IB
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Check if Tesla is within trading hours
is_trading = is_within_trading_hours(ib, 'TSLA')
print(f"TSLA is {'within' if is_trading else 'outside'} trading hours")
"""