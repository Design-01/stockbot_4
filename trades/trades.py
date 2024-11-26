from ib_insync import IB, Stock, MarketOrder, Order, Trade, BracketOrder
from mock_ib import MockIB
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple
import pandas as pd
from copy import deepcopy
import stops, entries, trade_log
from datetime import datetime
import pytz

def get_qty_shares(entry_price, stop_price, account_size, risk_percentage) -> None:
    """Calculates suggested position size based on risk parameters. 
    At the time this gets called the av_entry_price is not set as this is used to created 
    the order and the av_entry_price is taken once the order is placed.
    Wroks with both long and short trades .  eg stop can be higher than the entry price for short trades"""
    if entry_price and stop_price:
        risk_amount = account_size * risk_percentage
        risk_per_share = abs(entry_price - stop_price)
        return int(risk_amount / risk_per_share)
    return None

def get_total_trade_value(entry_price: float, position_size: int) -> float:
    """Calculates the total value of the position"""
    if entry_price > 0 and position_size > 0:
        return round(entry_price * position_size, 2)
    return 0.0

def get_rr_ratio(entry_price, stop_price, close_price) -> float:
    """Calculates risk:reward ratio for both long and short trades"""
    if all([entry_price, stop_price, close_price]):
        if entry_price > stop_price:  # Long trade
            risk = entry_price - stop_price
            reward = close_price - entry_price
        else:  # Short trade
            risk = stop_price - entry_price
            reward = entry_price - close_price
        
        if risk > 0:
            return round(reward / risk, 2)
    return None

def get_pl(entry_price: float, exit_price: float, pos_size: int, direction: str = "LONG") -> float:
    """ Calculate profit/loss for a trade."""
    if direction.upper() == "LONG":
        return (exit_price - entry_price) * pos_size
    else:  # SHORT
        return (entry_price - exit_price) * pos_size

def get_margin_value(share_qty: float, margin_rate: float) -> float:
    """Calculates the margin allowance required to open a position"""
    if share_qty > 0 and margin_rate > 0:
        margin_allowance = share_qty * margin_rate
        return round(margin_allowance, 2)
    return 0.0


def get_total_available_funds_with_margin(account_size: float, margin_rate: float) -> float:
    """Calculates the effective account size considering the margin rate"""
    if account_size > 0 and margin_rate > 0:
        return round(account_size * (1 + margin_rate), 2)
    return 0.0

def is_position_exceeding_account(entry_price: float, share_qty: int, account_size: float, margin_rate: float) -> bool:
    """Checks if the position amount exceeds the account size allowing for margin"""
    position_amount = entry_price * share_qty
    margin_allowance = position_amount * margin_rate
    return position_amount > (account_size + margin_allowance)

def max_position_size(entry_price: float, stop_price: float, account_size: float, margin_rate: float, risk: float) -> int:
    """Limits the position size based on account size with margin"""
    account_size_with_margin = get_total_available_funds_with_margin(account_size, margin_rate)
    max_risk_amount = account_size_with_margin * risk
    risk_per_share = abs(entry_price - stop_price)
    
    if risk_per_share > 0:
        max_position_size = max_risk_amount // risk_per_share
        return int(max_position_size)
    return 0

def is_open_for_trading(ib, symbol: str, outsideRth: bool = False, print_status: bool = False) -> bool:
    """
    Check if current time is valid for trading based on market hours and outsideRth preference.
    Args:
        ib: IB connection instance
        symbol: Trading symbol to check
        outsideRth: If True, allows trading during extended hours
        print_status: If True, prints current time and trading hours
    Returns:
        bool: True if trading is allowed at current time, False otherwise
    """
    details = ib.reqContractDetails(Stock(symbol, 'SMART', 'USD'))[0]
    current_time = datetime.now(pytz.timezone(details.timeZoneId))
    today = current_time.strftime('%Y%m%d')
    
    def parse_trading_period(period_str):
        if not period_str or 'CLOSED' in period_str:
            return None
        start_str, end_str = period_str.split('-')
        return tuple(datetime.strptime(t, '%Y%m%d:%H%M').replace(tzinfo=current_time.tzinfo) 
                    for t in (start_str, end_str))
    
    # Get first valid trading period for today
    eth_times = next((parse_trading_period(p) for p in details.tradingHours.split(';') 
                     if p.startswith(today)), None)
    rth_times = next((parse_trading_period(p) for p in details.liquidHours.split(';') 
                     if p.startswith(today)), eth_times)
    
    if not eth_times:
        return False
    
    if print_status:
        print(f"Symbol : {symbol}")
        print(f"Exchange time now: {current_time.strftime('%H:%M:%S')}")
        print(f"Regular market hours : {rth_times[0].strftime('%H:%M:%S')} - {rth_times[1].strftime('%H:%M:%S')}")
        if outsideRth:
            print(f"Extended market hours: {eth_times[0].strftime('%H:%M:%S')} - {eth_times[1].strftime('%H:%M:%S')}")
    
    trading_times = eth_times if outsideRth else rth_times
    return trading_times[0] <= current_time <= trading_times[1]

@dataclass
class TradeX:  
    """TradeX is a dataclass that represents a trade. It is used to:
        -- calculate and store trade information and performance metrics
        -- manage trade status and lifecycle
        -- manage trade orders and execution
        -- manage stop loss adjustments and exits
        -- map ib trade data to attirbutes

        orders are recieveed from the trade manager.  
        This class is to just managage the lifecycle of the trade 
        
        """  
    # ib is required to place orders
    ib: IB | MockIB = None

    # Auto-generated fields (not included in init)
    id = 1
    
    # Basic Trade Info (required fields)
    symbol: str = ''
    barsize: str = ''
    
    # Optional fields with defaults
    direction: str = "LONG"
    status: str = "PENDING"
    is_active: bool = False
    is_outsideRth: bool = False
    
    # Time Related
    chart_time: datetime = field(default_factory=datetime.now)
    real_time: datetime = field(default_factory=datetime.now)
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    # Entry Related
    entry_name: Optional[str] = None
    entry_av_price: Optional[float] = None
    entry_filled: Tuple[int, int] = (0, 0)
    entry_ib_id: Optional[str] = None
    entry_ib_status: Optional[str] = None
    
    # Exit Related
    exit_name: Optional[str] = None
    exit_av_price: Optional[float] = None
    exit_filled: Tuple[int, int] = (0, 0)
    exit_type: Optional[str] = None
    exit_ib_id: Optional[str] = None
    exit_ib_status: Optional[str] = None
    
    # Position Information
    position: int = 0
    value: float = 0.0
    close_price: Optional[float] = None
    
    # Stop
    stop_name: Optional[str] = None
    stop_price: Optional[float] = None
    stop_filled: Tuple[int, int] = (0, 0)
    stop_ib_id: Optional[str] = None
    stop_ib_status: Optional[str] = None

    # Target
    target_name: Optional[str] = None
    target_price: Optional[float] = None
    target_filled: Tuple[int, int] = (0, 0)
    target_ib_id: Optional[str] = None
    target_ib_status: Optional[str] = None
    
    # Performance Metrics
    unrealized_pl: float = 0.0
    realized_pl: float = 0.0
    target_risk_reward: float = 0.0
    actual_risk_reward: float = 0.0
    
    # New fields for enhanced functionality
    fund_allocation: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    total_cost: float = 0.0  # commission + slippage

    def __post_init__(self):
        self.id = TradeX.id 
        TradeX.id += 1

        self.entry_order     = None
        self.stop_order      = None
        self.target_order    = None
        self.entry_strategy  = None
        self.stop_strategy   = None
        self.target_strategy = None
        self.contract = Stock(self.symbol, 'SMART', 'USD')

    def start_trade(self, entry_strategy, stop_strategey, trarget_strategy=None):
        """Start the trade by sending the entry order"""
        self.entry_strategy = entry_strategy
        self.stop_strategy  = stop_strategey
        if trarget_strategy:
            self.target_strategy = trarget_strategy
        self.status = "PENDING_ENTRY"

    def find_entry(self, data: pd.DataFrame):
        """Check if the entry conditions are met"""
        if self.entry_strategy.should_enter(data):
            self.status = "PLACING_ENTRY_ORDER"
            self.submit_bracket_order(data)

    def submit_bracket_order(self, direction:str, quantity: int, stop_price: float,  target_price: Optional[float] = None, outsideRth: bool = False, limit_price: Optional[float] = None) -> None:
        
        if outsideRth:
            if limit_price is None:
                raise ValueError("Limit price must be provided for orders outside regular trading hours.")
        
        def get_parent_order():
            parent = Order()
            parent.orderId = self.ib.client.getReqId()
            parent.action = 'BUY' if direction.upper() == 'LONG' else 'SELL'
            parent.totalQuantity = quantity
            parent.orderType = 'LMT' if limit_price else 'MKT'
            parent.lmtPrice = limit_price
            parent.transmit = False
            parent.outsideRth = outsideRth
            return parent
        
        def get_stop_order():
            stop_loss = Order()
            stop_loss.orderId = self.ib.client.getReqId()
            stop_loss.action = 'SELL' if direction.upper() == 'LONG' else 'BUY'
            stop_loss.totalQuantity = quantity
            stop_loss.orderType = 'STP'
            stop_loss.auxPrice = stop_price
            stop_loss.parentId = parent.orderId
            stop_loss.outsideRth = outsideRth
            stop_loss.transmit = True if target_price is None else False
            return stop_loss
        
        def get_target_order():
            take_profit = Order()
            take_profit.orderId = self.ib.client.getReqId()
            take_profit.action = 'SELL' if direction.upper() == 'LONG' else 'BUY'
            take_profit.totalQuantity = quantity
            take_profit.orderType = 'LMT'
            take_profit.lmtPrice = target_price
            take_profit.parentId = parent.orderId
            take_profit.transmit = True
            take_profit.outsideRth = outsideRth
            return take_profit
        
        
        try:
            self.entry_filled = (0, quantity)
            self.exit_filled = (0, quantity)
            
            parent      = get_parent_order()
            stop_loss   = get_stop_order()
            take_profit = get_target_order() if target_price else None
            
            self.entry_order = self.ib.placeOrder(self.contract, parent)
            self.stop_order  = self.ib.placeOrder(self.contract, stop_loss)
            
            self.status = "ENTRY_SUBMITTED"
            self.entry_ib_id  = str(parent.orderId)
            self.stop_ib_id   = str(stop_loss.orderId)
            self.target_ib_id = str(take_profit.orderId) if take_profit else None
            
            self.entry_ib_status  = self.entry_order.orderStatus.status
            self.stop_ib_status   = self.stop_order.orderStatus.status
            self.target_ib_status = self.target_order.orderStatus.status if self.target_order else None

        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error submitting bracket order: {e}")


    def monitor_order_fills(self) -> None:
        """
        Extracts fill information from entry, stop and target orders and maps them
        to the appropriate class attributes.
        """
        if self.entry_order:
            # Check entry order fills
            fills = self.entry_order.fills
            if fills:
                total_shares_filled = sum(fill.execution.shares for fill in fills)
                self.entry_filled = (total_shares_filled, self.entry_order.totalQuantity)
                self.av_entry_price = sum(fill.execution.shares * fill.execution.price for fill in fills) / total_shares_filled if total_shares_filled > 0 else None
                self.entry_time = fills[-1].execution.time if fills else None
                #! entry status 
                print(f"Entry Order Status: Filled {total_shares_filled} of {self.entry_order.totalQuantity} shares")
                
        if self.stop_order:
            # Check stop order fills
            fills = self.stop_order.fills
            if fills:
                total_shares_filled = sum(fill.execution.shares for fill in fills)
                self.exit_filled = (total_shares_filled, self.stop_order.totalQuantity)
                self.av_exit_price = sum(fill.execution.shares * fill.execution.price for fill in fills) / total_shares_filled if total_shares_filled > 0 else None
                self.exit_time = fills[-1].execution.time if fills else None
                self.exit_type = "STOP"
                print(f"Stop Order Status: Filled {total_shares_filled} of {self.stop_order.totalQuantity} shares")

        if self.target_order:
            # Check target order fills
            fills = self.target_order.fills
            if fills:
                total_shares_filled = sum(fill.execution.shares for fill in fills)
                self.exit_filled = (total_shares_filled, self.target_order.totalQuantity)
                self.av_exit_price = sum(fill.execution.shares * fill.execution.price for fill in fills) / total_shares_filled if total_shares_filled > 0 else None
                self.exit_time = fills[-1].execution.time if fills else None
                self.exit_type = "TARGET"
                print(f"Target Order Status: Filled {total_shares_filled} of {self.target_order.totalQuantity} shares")
