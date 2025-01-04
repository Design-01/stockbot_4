import uuid
import pandas as pd
from ib_insync import *

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
        print(f"Stop quotas:  QtyPct: {qtyPct}, Qty: {qty}, Quotas: {quotas}")
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

    def add_parent_order(self, qty, limitPrice=None, outsideRth=False):
        self.qty = qty
        self.stop_quotas = qty
        self.parentLimitPrice = limitPrice
        self.parentOutsideRth = outsideRth
        
    def add_bracket_order(self, qtyPct, stop_price, target_price):
        self.bracketCount += 1
        self.stopCount += 1
        self.targetCount += 1
        qty = self._get_qty(qtyPct)

        # Get unique order IDs for each order in the bracket
        entry_order_id = self._get_next_order_id()
        stop_order_id = self._get_next_order_id()
        target_order_id = self._get_next_order_id()

        entry_order = MarketOrder(
            orderId       = entry_order_id,
            action        = 'BUY' if self.ls == 'LONG' else 'SELL',
            totalQuantity = qty,
            lmtPrice     = self.parentLimitPrice,
            outsideRth   = self.parentOutsideRth,
            orderRef     = self._get_order_ref('Entry', self.bracketCount),
            transmit     = False
        )
        entry_order.orderType = 'MKT' if self.parentLimitPrice is None else 'LMT'

        stop_order = StopOrder(
            orderId       = stop_order_id,
            parentId     = entry_order_id,
            action       = 'SELL' if self.ls == 'LONG' else 'BUY',
            stopPrice    = stop_price,
            totalQuantity = qty,
            orderRef     = self._get_order_ref('Stop', self.stopCount, qtyPct),
            transmit     = False
        )
        stop_order.stopNumber = self.stopCount

        target_order = LimitOrder(
            orderId      = target_order_id,
            parentId     = entry_order_id,
            action       = 'SELL' if self.ls == 'LONG' else 'BUY',
            totalQuantity = qty,
            lmtPrice     = target_price,
            orderRef     = self._get_order_ref('Tget', self.targetCount, qtyPct),
            transmit     = True
        )

        self.orders.extend([entry_order, stop_order, target_order])
        return entry_order_id

    def add_stop_order(self, qtyPct, stop_price):
        self.stopCount += 1
        self.bracketCount += 1
        qty = self._get_qty(qtyPct)
        
        # Get unique order IDs
        entry_order_id = self._get_next_order_id()
        stop_order_id = self._get_next_order_id()

        entry_order = MarketOrder(
            orderId      = entry_order_id,
            action       = 'BUY' if self.ls == 'LONG' else 'SELL',
            totalQuantity = qty,
            lmtPrice     = self.parentLimitPrice,
            outsideRth   = self.parentOutsideRth,
            orderRef     = self._get_order_ref('Entry', self.bracketCount),
            transmit     = False
        )

        stop_order = StopOrder(
            orderId      = stop_order_id,
            parentId     = entry_order_id,
            action       = 'SELL' if self.ls == 'LONG' else 'BUY',
            totalQuantity = qty,
            stopPrice    = stop_price,
            orderRef     = self._get_order_ref('Stop', self.stopCount, qtyPct),
            transmit     = True
        )
        stop_order.stopNumber = self.stopCount

        self.orders.extend([entry_order, stop_order])
        return entry_order_id

    def place_orders(self, delay_between_orders=0.01):
        """
        Place all orders with a delay between each order to prevent race conditions
        """
        for i, order in enumerate(self.orders):
            try:
                trade = self.ib.placeOrder(Stock(self.symbol, 'SMART', 'USD'), order)
                self.orders[i] = trade.order
                self.orders_status.append(trade.orderStatus)
                print(f"Order {i} placed: {order.orderRef} (ID: {order.orderId}, Transmit: {order.transmit})")
                
                if order.transmit:
                    print(f"Order {i} transmitted: {order.orderRef}")
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
                print(f"Cancelled order: {order.orderRef} (ID: {order.orderId})")
            except Exception as e:
                print(f"Error cancelling order {order.orderRef}: {str(e)}")

    def get_orders_as_df(self, full=False):
        if full:
            return pd.DataFrame(self.orders)
        else:
            return pd.DataFrame(self.orders)[['parentId', 'orderId', 'clientId', 'permId', 'action', 'totalQuantity', 'orderType', 'ocaGroup', 'ocaType', 'orderRef', 'lmtPrice', 'auxPrice', 'outsideRth', 'tif', 'goodTillDate',  'transmit']]

    def get_orders_status_as_df(self):
        return pd.DataFrame(self.orders_status)

    def modify_stop(self, stopNum, stop_price):
        order_ref = self._get_order_ref('Stop', stopNum)  # Get the reference pattern
        stop_order = next((order for order in self.orders 
                        if order.orderRef.startswith(order_ref)), None)
        if not stop_order:
            raise ValueError(f"No stop order found with number {stopNum}")
            
        stop_order.auxPrice = stop_price
        stop_order.transmit = True  # As you mentioned, needed for immediate update
        self.ib.placeOrder(Stock(self.symbol, 'SMART', 'USD'), stop_order)

    def get_stop_price(self, stopNum):
        order_ref = self._get_order_ref('Stop', stopNum)
        stop_order = next((order for order in self.orders 
                        if order.orderRef.startswith(order_ref)), None)
        if not stop_order:
            raise ValueError(f"No stop order found with number {stopNum}")
        return stop_order.auxPrice

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