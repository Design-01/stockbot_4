import logging
from typing import Dict, Any
from bank.bank import TraderBank
from data.ohlcv import OHLVC
from trades.order import Order
from frame.frame import Frame

class Trader:
    """A class to manage trading logic and actions. Does not manage signals, it only acts on them."""

    def __init__(self, initial_balance, OHLCV, exit_strategy=None):
        self.bank = TraderBank(initial_balance)
        self.OHLCV = OHLCV # OHLCV is a class that contains the Open, High, Low, Close, and Volume data for a stock trading instrument
        self.Order = Order()
        self.frame = Frame()
        self.exit_strategy = exit_strategy
        self.trigger_price = None
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.current_price = None
        self.logger = logging.getLogger(__name__)

    def set_trigger(self, price: float, action: str):
        """Set the trigger price and action (buy/sell) for the trader."""
        self.trigger_price = price
        self.action = action

    def enter_trade(self, current_price: float):
        """Enter a trade if the trigger condition is met."""
        if (self.action == "buy" and current_price >= self.trigger_price) or \
           (self.action == "sell" and current_price <= self.trigger_price):
            position_size = self.calculate_position_size()
            self.position = self.Order.place_order(self.action, position_size, current_price)
            self.entry_price = current_price
            self.set_initial_stop_loss()
            self.set_target()
            self.logger.info(f"Entered {self.action} trade at {current_price}")

    def calculate_position_size(self) -> float:
        """Calculate the position size based on account balance and risk."""
        account_balance = self.bank.get_balance()
        risk_percentage = 0.01  # 1% risk per trade
        risk_amount = account_balance * risk_percentage
        price_difference = abs(self.trigger_price - self.calculate_initial_stop_loss())
        return risk_amount / price_difference

    def set_initial_stop_loss(self):
        """Set the initial stop loss."""
        if self.action == "buy":
            self.stop_loss = self.entry_price * 0.98  # 2% below entry for long positions
        else:
            self.stop_loss = self.entry_price * 1.02  # 2% above entry for short positions

    def set_target(self):
        """Set the target price."""
        if self.action == "buy":
            self.target = self.entry_price * 1.06  # 6% above entry for long positions
        else:
            self.target = self.entry_price * 0.94  # 6% below entry for short positions

    def adjust_stop_loss(self, current_price: float):
        """Adjust the stop loss as the trade progresses."""
        if self.position:
            if self.action == "buy" and current_price > self.entry_price:
                new_stop_loss = max(self.stop_loss, current_price * 0.98)
                self.stop_loss = new_stop_loss
            elif self.action == "sell" and current_price < self.entry_price:
                new_stop_loss = min(self.stop_loss, current_price * 1.02)
                self.stop_loss = new_stop_loss

    def check_exit_conditions(self, current_price: float) -> bool:
        """Check if the trade should be exited based on stop loss or target."""
        if self.position:
            if (self.action == "buy" and current_price <= self.stop_loss) or \
               (self.action == "sell" and current_price >= self.stop_loss):
                self.exit_trade(current_price, "Stop Loss")
                return True
            elif (self.action == "buy" and current_price >= self.target) or \
                 (self.action == "sell" and current_price <= self.target):
                self.exit_trade(current_price, "Target Reached")
                return True
        return False

    def exit_trade(self, current_price: float, reason: str):
        """Exit the current trade."""
        if self.position:
            exit_action = "sell" if self.action == "buy" else "buy"
            self.Order.place_order(exit_action, self.position, current_price)
            profit_loss = (current_price - self.entry_price) * self.position if self.action == "buy" else (self.entry_price - current_price) * self.position
            self.bank.update_balance(profit_loss)
            self.logger.info(f"Exited trade at {current_price}. Reason: {reason}. Profit/Loss: {profit_loss}")
            self.reset_trade_data()

    def reset_trade_data(self):
        """Reset trade-related data after exiting a trade."""
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.target = None

    def update(self, current_price: float):
        """Update the trader with the current price."""
        self.current_price = current_price
        if not self.position:
            self.enter_trade(current_price)
        else:
            self.adjust_stop_loss(current_price)
            self.check_exit_conditions(current_price)

    def get_trade_info(self) -> Dict[str, Any]:
        """Return a dictionary with current trade information."""
        return {
            "position": self.position,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "profit_loss": (self.current_price - self.entry_price) * self.position if self.position else 0,
            "risk_reward_ratio": abs((self.target - self.entry_price) / (self.entry_price - self.stop_loss)) if self.position else 0,
            "account_balance": self.bank.get_balance()
        }()