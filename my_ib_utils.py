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