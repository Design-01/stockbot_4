from ib_insync import IB

class IBRateLimiter:
    """
    Rate limiter for Interactive Brokers API requests using IB_insync's sleep method
    """
    def __init__(self, ib: IB, requests_per_second: float = 2):
        """
        Initialize rate limiter
        
        Args:
            ib: IB instance for using ib.sleep()
            requests_per_second: Maximum sustained requests per second
        """
        self.ib = ib
        self.min_interval = 1.0 / requests_per_second

    def wait(self):
        """Wait using IB_insync's sleep method"""
        self.ib.sleep(self.min_interval)