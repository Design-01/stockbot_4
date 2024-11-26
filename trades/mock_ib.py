from typing import List, Optional, Union, Dict
from ib_insync import Order, Stock, Trade, BracketOrder
from copy import deepcopy
from datetime import datetime

class MockOrderStatus:
    def __init__(self):
        self.status = 'Submitted'
        self.filled = 0
        self.remaining = 0
        self.avgFillPrice = 0.0
        self.lastFillPrice = 0.0
        self.whyHeld = ''
        self.mktCapPrice = 0.0

class MockContract:
    def __init__(self, contract):
        self.symbol = contract.symbol
        self.secType = contract.secType
        self.exchange = contract.exchange
        self.currency = contract.currency

class MockExecution:
    def __init__(self, trade: 'MockTrade'):
        self.orderId = trade.order.orderId
        self.execId = f"exec_{trade.order.orderId}"
        self.time = datetime.now().strftime('%Y%m%d %H:%M:%S')
        self.acctNumber = 'DU123456'
        self.exchange = 'SMART'
        self.side = trade.order.action
        self.shares = trade.order.totalQuantity
        self.price = trade.orderStatus.avgFillPrice
        self.permId = trade.order.orderId
        self.clientId = 0
        self.liquidation = 0
        self.cumQty = trade.order.totalQuantity
        self.avgPrice = trade.orderStatus.avgFillPrice
        self.orderRef = trade.order.orderRef
        self.evRule = ''
        self.evMultiplier = 1.0
        self.modelCode = ''
        self.lastLiquidity = 1

class MockCommissionReport:
    def __init__(self, execution: MockExecution):
        self.execId = execution.execId
        self.commission = 1.0
        self.currency = 'USD'
        self.realizedPNL = 0.0
        self.yield_ = 0.0
        self.yieldRedemptionDate = 0

class MockTrade:
    def __init__(self, contract: MockContract, order: Order):
        self.contract = contract
        self.order = order
        self.orderStatus = MockOrderStatus()
        self.fills = []
        self.log = []
        
        # Set order properties that IB would set
        if not hasattr(self.order, 'orderId'):
            self.order.orderId = 0
        if not hasattr(self.order, 'clientId'):
            self.order.clientId = 0
        if not hasattr(self.order, 'permId'):
            self.order.permId = 0
        
        # Initialize the order status
        self.orderStatus.remaining = order.totalQuantity
        
        # Set initial values based on order type
        if order.orderType == 'MKT':
            self.orderStatus.avgFillPrice = 100.0  # Mock market price
        elif order.orderType == 'LMT':
            self.orderStatus.avgFillPrice = order.lmtPrice
        elif order.orderType == 'STP':
            self.orderStatus.avgFillPrice = order.auxPrice

class MockIB:
    def __init__(self):
        self._connected = False
        self.client_id = 1
        self._next_order_id = 1
        self.orders: Dict[int, Order] = {}
        self.trades: Dict[int, MockTrade] = {}
        
    def connect(self, host: str, port: int, clientId: int):
        self._connected = True
        self.client_id = clientId
        return True
        
    def disconnect(self):
        self._connected = False
        
    def _get_next_order_id(self) -> int:
        order_id = self._next_order_id
        self._next_order_id += 1
        return order_id
        
    def _simulate_execution(self, trade: MockTrade) -> None:
        """Simulate order execution and generate execution report"""
        # Create execution object
        execution = MockExecution(trade)
        commission_report = MockCommissionReport(execution)
        
        # Update order status
        trade.orderStatus.status = 'Filled'
        trade.orderStatus.filled = trade.order.totalQuantity
        trade.orderStatus.remaining = 0
        trade.orderStatus.lastFillPrice = trade.orderStatus.avgFillPrice
        
        # Add to fills
        trade.fills.append((execution, commission_report))
        
    def placeOrder(self, contract: Stock, order: Order) -> MockTrade:
        """Place a single order"""
        # Assign order ID if not present
        if not order.orderId:
            order.orderId = self._get_next_order_id()
            
        # Create mock contract and trade
        mock_contract = MockContract(contract)
        mock_trade = MockTrade(mock_contract, order)
        
        # Store order and trade
        self.orders[order.orderId] = order
        self.trades[order.orderId] = mock_trade
        
        # Simulate immediate execution for market orders
        if order.orderType == 'MKT':
            self._simulate_execution(mock_trade)
            
        return mock_trade
        
    def bracketOrder(self, parentOrderId: int, contract: Stock, 
                    action: str, quantity: float, 
                    limitPrice: float, takeProfitPrice: float, 
                    stopLossPrice: float) -> List[Order]:
        """
        Create a bracket order matching IB's interface.
        Returns a list of orders: [entry, takeProfit, stopLoss]
        """
        # Create the parent (entry) order
        parent = Order()
        parent.orderId = parentOrderId
        parent.action = action
        parent.totalQuantity = quantity
        parent.orderType = 'LMT'
        parent.lmtPrice = limitPrice
        parent.transmit = False  # Hold submission of child orders
        
        # Create take profit order
        takeProfit = Order()
        takeProfit.orderId = parent.orderId + 1
        takeProfit.action = 'SELL' if action == 'BUY' else 'BUY'
        takeProfit.totalQuantity = quantity
        takeProfit.orderType = 'LMT'
        takeProfit.lmtPrice = takeProfitPrice
        takeProfit.parentId = parentOrderId
        takeProfit.transmit = False
        
        # Create stop loss order
        stopLoss = Order()
        stopLoss.orderId = parent.orderId + 2
        stopLoss.action = 'SELL' if action == 'BUY' else 'BUY'
        stopLoss.totalQuantity = quantity
        stopLoss.orderType = 'STP'
        stopLoss.auxPrice = stopLossPrice
        stopLoss.parentId = parentOrderId
        stopLoss.transmit = True  # Last order, so transmit all
        
        return [parent, takeProfit, stopLoss]
    
    def qualifyContracts(self, *contracts):
        """Mock contract qualification"""
        return contracts
        
    def sleep(self, secs: float = 0.0):
        """Mock sleep function"""
        pass
        
    def run(self):
        """Mock run function"""
        pass

# Example usage:
# def example_usage():
#     ib = MockIB()
#     ib.connect("127.0.0.1", 7497, 1)
    
#     # Create contract
#     contract = Stock('AAPL', 'SMART', 'USD')
    
#     # Create bracket order
#     bracket_orders = ib.bracketOrder(
#         parentOrderId=ib._get_next_order_id(),
#         contract=contract,
#         action='BUY',
#         quantity=100,
#         limitPrice=150.0,
#         takeProfitPrice=155.0,
#         stopLossPrice=145.0
#     )
    
#     # Place orders
#     trades = []
#     for order in bracket_orders:
#         trade = ib.placeOrder(contract, order)
#         trades.append(trade)
        
#     return trades