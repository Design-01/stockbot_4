import uuid
import pandas as pd
from ib_insync import *
import my_ib_utils

def get_id():
    return str(uuid.uuid4())[:8]

class OrderX:
    def __init__(self, name, ib, symbol, ls='LONG'):
        self.name = name
        self.ib = ib
        self.symbol = symbol
        self.ls = ls
        self.qty = 0
        self.parentID = None
        self.orders = []
        self.orders_status = []
        self.stopCount = 0
        self.targetCount = 0
        self.bracketCount = 0
        self.stop_quotas = self.qty
        self.next_order_id = None  # Track the next available order ID
        self.parentLimitPrice = None
        self.parentOutsideRth = False
        self.entryPrice = None
        self.entryOrderType = None
        self.is_rth = None
        self.stoploss_ids = []

    def _get_next_order_id(self):
        """Get a unique order ID from IB"""
        if self.next_order_id is None:
            self.next_order_id = self.ib.client.getReqId()
        current_id = self.next_order_id
        self.next_order_id += 1
        return current_id

    def _get_qty(self, qtyPct):
        qty = int(self.qty * qtyPct / 100)
        if qty < 1:
            raise ValueError("Quantity must be greater than 0. Check main order quantity and qtyPct for each additional order.")
        quotas = self.stop_quotas - qty
        # print(f"Stop quotas:  QtyPct: {qtyPct}, Qty: {qty}, Quotas: {quotas}")
        if quotas < 0:
            raise ValueError("Stop quotas exceeded. Check stop orders quantities.")
        self.stop_quotas = quotas
        return qty

    def _get_order_ref(self, order_type, order_num, qtyPct=None):
        """
        Generate consistent order references
        order_type: 'Entry', 'Stop', or 'Tget'
        order_num: The count number for this type of order
        qtyPct: Optional percentage for stop/target orders
        """
        if qtyPct is not None:
            return f"{self.name}_{order_type}_{order_num}_pct{qtyPct}"
        return f"{self.name}_{order_type}_{order_num}"
    
    def calculate_position_size(self, entry_price: float, stop_price: float, risk_amount: float) -> dict:
        """
        Calculate the position size and related metrics based on entry price, stop loss, and risk amount.
        
        Parameters:
        entry_price (float): The price at which you plan to enter the trade
        stop_price (float): Your stop loss price
        risk_amount (float): The amount of money you're willing to risk on this trade
        
        Returns:
        dict: Dictionary containing position details including:
            - shares: Number of shares to purchase
            - total_value: Total position value
            - risk_percentage: Percentage drop to stop loss
            - price_per_share: Entry price per share
            - potential_loss: Total loss if stop is triggered
        """
        if entry_price <= 0 or stop_price <= 0 or risk_amount <= 0:
            raise ValueError("All input values must be positive numbers")
        
        if stop_price >= entry_price:
            raise ValueError("Stop price must be below entry price for long positions")
        
        # Calculate the price difference and risk percentage
        price_difference = entry_price - stop_price
        risk_percentage = (price_difference / entry_price) * 100
        
        # Calculate position size based on risk
        shares = int(risk_amount / price_difference)
        
        # Calculate total position value and potential loss
        total_position_value = shares * entry_price
        total_loss_at_stop = shares * (entry_price - stop_price)
        
        return {
            "shares": shares,
            "total_value": round(total_position_value, 2),
            "risk_percentage": round(risk_percentage, 2),
            "price_per_share": entry_price,
            "potential_loss": round(total_loss_at_stop, 2)
        }

    def set_entry(self, isRth:bool, entryOrderType:str, qty, limitPrice=None, entryPrice=None, outsideRth=False):
        self.is_rth = isRth
        if outsideRth and limitPrice is None and not self.is_rth:
            raise ValueError("Limit price is required for outsideRth orders")
        self.qty = qty
        self.stop_quotas = qty
        self.parentLimitPrice = limitPrice
        self.parentOutsideRth = outsideRth
        self.entryPrice = entryPrice
        self.entryOrderType = entryOrderType

    def _get_entry_order(self, qty, bracket_count):
        entry_order = None
        self.parentID = self._get_next_order_id()

        if self.entryOrderType in  ['MKT', 'LMT']:
            entry_order = MarketOrder(
                orderId       = self.parentID,
                action        = 'BUY' if self.ls == 'LONG' else 'SELL',
                totalQuantity = qty,
                lmtPrice      = self.parentLimitPrice,
                outsideRth    = self.parentOutsideRth,
                orderRef      = self._get_order_ref('Entry', bracket_count),
                transmit      = False
            )
            entry_order.orderType = 'MKT' if self.is_rth else 'LMT'

        elif self.entryOrderType in ['STP', 'STP LMT']:
            entry_order = StopOrder(
                orderId       = self.parentID,
                action        = 'BUY' if self.ls == 'LONG' else 'SELL',
                totalQuantity = qty,
                outsideRth    = self.parentOutsideRth,
                lmtPrice      = self.parentLimitPrice,
                stopPrice     = self.entryPrice,
                orderRef      = self._get_order_ref('Entry', bracket_count),
                transmit      = False
            )
            entry_order.orderType = 'STP' if self.is_rth else 'STP LMT'
        
        else:
            raise ValueError(f"Invalid entry order type: {self.entryOrderType}, entry price: {self.entryPrice}, limit price: {self.parentLimitPrice}. Check combination of entry order type and entry and/or limit price")
        
        return entry_order
        
    def add_bracket_order(self, qtyPct, stop_price, target_price):
        if pd.isna(stop_price) :  raise ValueError(f"OrderXData.add_bracket_order: stop_price cannot be None. Got {stop_price}")
        if pd.isna(target_price): raise ValueError(f"OrderXData.add_bracket_order: target_price cannot be None. Got {target_price}")
        if self.ls == 'LONG' and stop_price >= target_price: raise ValueError(f"OrderXData.add_bracket_order: stop_price ({stop_price}) must be less than target_price ({target_price})")
        if self.ls == 'SHORT' and stop_price <= target_price: raise ValueError(f"OrderXData.add_bracket_order: stop_price ({stop_price}) must be greater than target_price ({target_price})")

        self.bracketCount += 1
        self.stopCount += 1
        self.targetCount += 1
        qty = self._get_qty(qtyPct)

        # get entry order separtly as bot this barckt and the stop only barcekt will have the same parent order so saves code dupliaction
        entry_order = self._get_entry_order(qty, self.bracketCount)


        stop_order = StopOrder(
            orderId       = self._get_next_order_id(),
            parentId      = entry_order.orderId,
            action        = 'SELL' if self.ls == 'LONG' else 'BUY',
            stopPrice     = stop_price,
            totalQuantity = qty,
            orderRef      = self._get_order_ref('Stop', self.stopCount, qtyPct),
            transmit      = False
        )

        target_order = LimitOrder(
            orderId       = self._get_next_order_id(),
            parentId      = entry_order.orderId,
            action        = 'SELL' if self.ls == 'LONG' else 'BUY',
            totalQuantity = qty,
            lmtPrice      = target_price,
            orderRef      = self._get_order_ref('Tget', self.targetCount, qtyPct),
            transmit      = True
        )

        self.stoploss_ids.append(stop_order.orderId)
        bracket = [entry_order, stop_order, target_order]
        self.orders.extend(bracket)
        return bracket

    def add_stop_order(self, qtyPct, stop_price):
        if pd.isna(stop_price): raise ValueError(f"OrderXData.add_stop_order: stop_price cannot be None. Got {stop_price}")

        self.stopCount += 1
        qty = self._get_qty(qtyPct)

        entry_order = self._get_entry_order(qty, self.bracketCount)
        

        stop_order = StopOrder(
            orderId      = self._get_next_order_id(),
            parentId     = entry_order.orderId,
            action       = 'SELL' if self.ls == 'LONG' else 'BUY',
            totalQuantity = qty,
            stopPrice    = stop_price,
            orderRef     = self._get_order_ref('Stop', self.stopCount, qtyPct),
            transmit     = True
        )
        self.stoploss_ids.append(stop_order.orderId)    
        bracket = [entry_order, stop_order]
        self.orders.extend(bracket)
        return bracket

    def place_orders(self, delay_between_orders=0.01):
        """
        Place all orders with a delay between each order to prevent race conditions
        """
        for i, order in enumerate(self.orders):
            try:
                trade = self.ib.placeOrder(Stock(self.symbol, 'SMART', 'USD'), order)
                self.orders[i] = trade.order
                self.orders_status.append(trade.orderStatus)
                
                if order.transmit:
                    self.ib.sleep(delay_between_orders)  # Add delay after transmitted orders
                    
            except Exception as e:
                print(f"Error placing order {i} ({order.orderRef}): {str(e)}")
                # Optionally cancel all previous orders if there's an error
                self.cancel_orders()
                raise

    def cancel_orders(self):
        """Cancel all pending orders"""
        for order in self.orders:
            try:
                self.ib.cancelOrder(order)
            except Exception as e:
                print(f"Error cancelling order {order.orderRef}: {str(e)}")

    def get_orders_as_df(self, full=False):
        if full:
            return pd.DataFrame(self.orders)
        else:
            return pd.DataFrame(self.orders)[['parentId', 'orderId', 'clientId', 'permId', 'action', 'totalQuantity', 'orderType', 'ocaGroup', 'ocaType', 'orderRef', 'lmtPrice', 'auxPrice', 'outsideRth', 'tif', 'goodTillDate',  'transmit']]

    def get_orders_status_as_df(self):
        return pd.DataFrame(self.orders_status)

    def modify_order_price(self, order_id, new_price):
        order = next((order for order in self.orders if order.orderId == order_id), None)
        if not order:
            raise ValueError(f"No order found with ID {order_id}")
        
        order.auxPrice = new_price
        order.transmit = True  # needed for immediate update
        self.ib.placeOrder(Stock(self.symbol, 'SMART', 'USD'), order)

    def get_order_price(self, order_id):
        order = next((order for order in self.orders if order.orderId == order_id), None)
        if not order:
            raise ValueError(f"No order found with ID {order_id}")
        return order.auxPrice
    
    def get_order_by_id(self, order_id):
        for order in self.orders:
            if order.orderId == order_id:
                return order
        return None

# Example usage
"""
ox = OrderX('Strat1', ib, 'TSLA', 'LONG')
ox.add_parent_order(qty=8, outsideRth=True)
ox.add_bracket_order(qtyPct=25, stop_price=370.00, target_price=420.00)
ox.add_bracket_order(qtyPct=25, stop_price=350.00, target_price=470.00)
ox.add_stop_order(qtyPct=50, stop_price=395.00)
ox.place_orders(delay_between_orders=0)

display(ox.get_orders_as_df(full=False))
display(ox.get_orders_status_as_df())

ox.modify_stop(stopNum=2, stop_price=400.00)

ox.cancel_orders()

"""


class OrderXData:
    """This is a wrapper class for OrderX that instead uses cloumn names and dataframes to get prices and place orders"""
    def __init__(self, name, ib, symbol, ls):
        self.name = name
        self.ib = ib
        self.symbol = symbol
        self.ls = ls
        self.ox = OrderX(self.name, self.ib, self.symbol, self.ls)

    def get_price(self, data, price):
        return round(data[price].iat[-1], 2)

    def set_entry(self, data, qty, outsideRth=False, limitPrice=None):
        if outsideRth and limitPrice is None:
            raise ValueError("Limit price is required for outsideRth orders")
        if outsideRth:
            limitPrice = self.get_price(data, limitPrice)

        self.ox.set_entry(qty=qty, outsideRth=outsideRth, limitPrice=limitPrice)

    def add_bracket_order(self, data, qtyPct, stop_price, target_price):
        stop_price = self.get_price(data, stop_price)
        target_price = self.get_price(data, target_price)
        # errors are handled in OrderX
        self.ox.add_bracket_order(qtyPct=qtyPct, stop_price=stop_price, target_price=target_price)
            
    def add_stop_order(self, data, qtyPct, stop_price):
        stop_price = self.get_price(data, stop_price)
        # errors are handled in OrderX
        self.ox.add_stop_order(qtyPct=qtyPct, stop_price=stop_price)

    def place_orders(self, data, delay_between_orders=0):
        self.ox.place_orders(delay_between_orders=delay_between_orders)

    def get_stop_price(self, stopNum):
        return self.ox.get_stop_price(stopNum)

    def modify_stop(self, data, stopNum, stop_price):
        new_price = self.get_price(data, stop_price)
        old_price = self.ox.get_stop_price(stopNum)
        # print(f"Old price: {old_price}, New price: {new_price}, new > old = {new_price > old_price}")

        if any([self.ls == 'LONG' and new_price > old_price, 
                self.ls == 'SHORT' and new_price < old_price]):
            self.ox.modify_stop(stopNum=stopNum, stop_price=new_price)

    def cancel_orders(self):
        self.ox.cancel_orders()

    def get_orders_status_as_df(self):
        return self.ox.get_orders_status_as_df()
    
    def get_orders_as_df(self):
        return self.ox.get_orders_as_df()
    

# Example usage of OrderXData
"""
orders will not exicute if nan is passed as price. it will throw an error. 

### ------- Set up the data -------- ###
from ib_insync import *
util.startLoop()
import stock

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=12)
sx = stock.StockX(ib, 'TSLA')
sx.set_up_frame('1 day', 'ohlcv', start_date="52 weeksAgo", end_date="now")
sx.frames['1 day'].data

### ------- Set up the strategy -------- ###
import strategies.ta as ta
import strategies.preset_strats as ps
f = sx.frames['1 day']
f.add_ta(ta.HPLP(hi_col='high', lo_col='low', span=10), [{'color': 'green', 'size': 10}, {'color': 'red', 'size': 10}], chart_type = 'points')
f.add_ta(ta.HPLP(hi_col='high', lo_col='low', span=3), [{'color': 'green', 'size': 10}, {'color': 'red', 'size': 10}], chart_type = 'points')
f.add_ta(ta.Ffill(colToFfill='HP_hi_3'), [{'color': 'green', 'size': 10}, {'color': 'red', 'size': 10}], chart_type = '')
ps.require_ta_for_all(f, pointsSpan=10)
ps.ma_ta(f, [21, 50, 200])
f.plot()

### ------- Set up the order -------- ###
oxd = OrderXData('Strat1', ib, 'TSLA', 'LONG')
oxd.set_entry(f.data, qty=8, outsideRth=True, limitPrice='close')
oxd.add_bracket_order(f.data, qtyPct=25, stop_price='Sup_1', target_price='FFILL_HP_hi_3')
oxd.add_bracket_order(f.data, qtyPct=25, stop_price='Sup_2', target_price='Res_1')
oxd.add_stop_order(f.data, qtyPct=50, stop_price='Sup_1')

### ------- Place the orders -------- ###
oxd.place_orders(f.data, delay_between_orders=3)

### ------- modify the stop -------- ###
oxd.modify_stop(f.data, stopNum=2, stop_price='Sup_2')

### ------- Cancel the orders -------- ###
oxd.cancel_orders()

### ------- Get the orders status -------- ###
oxd.get_orders_status_as_df()

### ------- Get the orders -------- ###
oxd.get_orders_as_df()
"""