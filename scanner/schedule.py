from datetime import datetime, date, timedelta
import pytz
import time

class MarketSchedule:
    def __init__(self, scan_time="08:00", sleep_time=300):
        """
        Initialize with scan time in 24-hour format (ET)
        scan_time: str format "HH:MM" in Eastern Time
        """
        self.scan_time = scan_time
        self.sleep_time = sleep_time
        self.market_holidays_2024 = {
            date(2024, 1, 1): "New Year's Day",
            date(2024, 1, 15): "Martin Luther King Jr. Day",
            date(2024, 2, 19): "Presidents Day",
            date(2024, 3, 29): "Good Friday",
            date(2024, 5, 27): "Memorial Day",
            date(2024, 6, 19): "Juneteenth",
            date(2024, 7, 4): "Independence Day",
            date(2024, 9, 2): "Labor Day",
            date(2024, 11, 28): "Thanksgiving Day",
            date(2024, 12, 25): "Christmas Day"
        }

    def get_next_trading_day(self, from_date):
        next_day = from_date + timedelta(days=1)
        while True:
            if (next_day.weekday() not in [5, 6] and 
                next_day not in self.market_holidays_2024):
                return next_day
            next_day += timedelta(days=1)

    def wait_for_scan_time(self):
        """Wait until the specified scan time on a trading day"""
        us_eastern = pytz.timezone('US/Eastern')
        
        while True:
            current_time = datetime.now(us_eastern)
            current_date = current_time.date()
            
            # Check if it's a trading day
            if current_date.weekday() in [5, 6]:
                next_trading = self.get_next_trading_day(current_date)
                print(f"Market is closed today (Weekend)")
                print(f"Next trading day is {next_trading.strftime('%A, %B %d, %Y')}")
                return False
                
            if current_date in self.market_holidays_2024:
                next_trading = self.get_next_trading_day(current_date)
                print(f"Market is closed today ({self.market_holidays_2024[current_date]})")
                print(f"Next trading day is {next_trading.strftime('%A, %B %d, %Y')}")
                return False

            # Parse target scan time
            scan_hour, scan_minute = map(int, self.scan_time.split(':'))
            target_time = current_time.replace(hour=scan_hour, minute=scan_minute, second=0)
            
            # If we've already passed the scan time today, wait for tomorrow
            if current_time > target_time:
                next_trading = self.get_next_trading_day(current_date)
                print(f"Scan time {self.scan_time} ET has passed for today")
                print(f"Next scan will be at {self.scan_time} ET on {next_trading.strftime('%A, %B %d, %Y')}")
                return False

            # Calculate time until scan
            time_until_scan = (target_time - current_time).total_seconds()
            
            if time_until_scan <= 0:
                print(f"Scan time reached! Running scanner at {self.scan_time} ET")
                return True
            else:
                hours = int(time_until_scan // 3600)
                minutes = int((time_until_scan % 3600) // 60)
                print(f"Today is a trading day. Waiting {hours} hours and {minutes} minutes until scan time ({self.scan_time} ET)...")
                time.sleep(min(self.sleep_time, time_until_scan))  # Sleep for 5 minutes or until scan time, whichever is shorter

# Example usage
# if __name__ == "__main__":
#     # Initialize with desired scan time (8:00 AM ET)
#     market = MarketSchedule(scan_time="08:00")
    
#     if market.wait_for_scan_time():
#         print("Starting scanner...")
#         # Your scanner code would go here