from datetime import datetime, timedelta


def get_most_recent_trading_day(self):
    """Determine the most recent trading day"""
    now = datetime.now(self.timezone)
    
    # If it's weekend, adjust to Friday
    if now.weekday() == 5:  # Saturday
        now = now - timedelta(days=1)
    elif now.weekday() == 6:  # Sunday
        now = now - timedelta(days=2)
        
    # Set time to end of trading day (16:00:00 Eastern)
    return now.replace(hour=16, minute=0, second=0, microsecond=0)


def format_datetime(self, dt):
    """Format datetime properly for IB historical data requests"""
    # Convert to Eastern time
    dt_eastern = dt.astimezone(self.timezone)
    return dt_eastern.strftime('%Y%m%d %H:%M:%S US/Eastern')