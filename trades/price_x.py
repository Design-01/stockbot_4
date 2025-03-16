import pandas as pd
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class BaseX:
    name: str = ''
    ls: str = ''
    priceCol: str = ''
    price:float = np.nan
    longPriceCol: str = 'high'
    shortPriceCol: str = 'low'
    limitPrice: float = np.nan
    offsetVal: float = 0.0
    offsetPct: float = 0.0
    limitOffsetVal: float = 0.0
    limitOffsetPct: float = 0.0
    barsAgo: int = 1
    orderFilled: bool = False
    ibId: int = 0


    def set_ls(self, ls):
        self.ls = ls
        self.priceCol = self.longPriceCol if self.ls == 'LONG' else self.shortPriceCol

    def get_price(self, data:pd.DataFrame=None) -> float:
        if data is not None:
            self.price = round(data[self.priceCol].iat[-self.barsAgo-1],2)
        return self.price

    def get_limit_price(self, data: pd.DataFrame=None) -> float:
        if data is not None:
            base_price = self.get_price(data)
            if self.ls == 'LONG':
                limit_price = base_price + self.limitOffsetVal + (base_price * self.limitOffsetPct)
            elif self.ls == 'SHORT':
                limit_price = base_price - self.limitOffsetVal - (base_price * self.limitOffsetPct)
            self.limitPrice = max(round(limit_price,2), 0.01)
        return self.limitPrice
    
    def set_price(self, price:float):
        self.price = price
    
    def set_limit_price(self, limitPrice:float):
        self.limitPrice = limitPrice
    
    def reset(self):
        self.price = np.nan

    def has_triggered(self, data=None, priceType=None, forceTrigger:bool=False) -> bool:
        if forceTrigger: self.orderFilled = True
        if self.orderFilled: return True
        if self.ls == 'LONG' and priceType in ['entry', 'target']: self.orderFilled =  data['high'].iat[-1] >= self.price
        if self.ls == 'SHORT' and priceType in ['entry', 'target']: self.orderFilled = data['low'].iat[-1] <= self.price
        if self.ls == 'LONG' and priceType == 'stop': self.orderFilled = data['low'].iat[-1] <= self.price
        if self.ls == 'SHORT' and priceType == 'stop': self.orderFilled = data['high'].iat[-1] >= self.price
        return self.orderFilled


# todo: add a options for setting various price types
@dataclass
class EntryX(BaseX):
    orderType: str = 'STP'

    def __post_init__(self):
        if self.orderType == 'MKT':
            self.longPriceCol = 'close'
            self.shortPriceCol = 'close'
            self.barsAgo = 0

    def set_outside_rth(self, data:pd.DataFrame):
        if self.orderType == 'MKT':
            self.orderType = 'LMT'
        elif self.orderType == 'STP':
            self.orderType = 'STP LMT'
        self.get_limit_price(data)


@dataclass
class StopX(BaseX):
    orderType: str = 'STP'


@dataclass
class TargetX(BaseX):
    orderType: str = 'LMT'
    rrIfNoTarget: float = 2

    def get_price(self, data:pd.DataFrame=None, entryPrice:float=None, stopPrice:float=None) -> float:
        if data is not None:
            self.price = data[self.priceCol].iat[-self.barsAgo-1]
        if math.isnan(self.price):
            self.price = round(data['close'].iat[-1] + (self.rrIfNoTarget * abs(entryPrice - stopPrice)), 2)
        return self.price
    

@dataclass
class TrailX(BaseX):
    initType: str = 'rrr'
    initTrigVal: float = 0.0
    
    def set_ls(self, ls):
        self.ls = ls
        self.priceCol = self.longPriceCol if self.ls == 'LONG' else self.shortPriceCol

    def get_price(self, data:pd.DataFrame) -> float:
        self.price = round(data[self.priceCol].iat[-self.barsAgo-1], 2)
        return self.price
    

@dataclass
class RiskX:
    name: str = ''
    ls: str = ''
    risk: float = 0.0
    reward: float = 0.0
    rRatio: float = 0.0

    def get_value(self, entryPrice, stopPrice, priceNow):  
        if self.ls == 'LONG':
            self.risk = entryPrice - stopPrice
            self.reward = priceNow - entryPrice
        elif self.ls == 'SHORT':
            self.risk = stopPrice - entryPrice
            self.reward = entryPrice - priceNow
        if self.risk == 0: return 0
        self.rRatio = self.reward / self.risk
        return self.rRatio
    
    def reset(self):
        self.risk = 0.0
        self.reward = 0.0
        self.rRatio = 0.0


@dataclass
class QtyX:
    qty: int = field(default=0, init=False)
    total_value: float = field(default=0.0, init=False)
    risk_percentage: float = field(default=0.0, init=False)
    price_per_share: float = field(default=0.0, init=False)
    potential_loss: float = field(default=0.0, init=False)

    def compute_qty(self, entry_price: float, stop_price: float, risk_amount: float):
        """
        Calculate the position size and related metrics based on entry price, stop loss, and risk amount.
        
        Parameters:
        entry_price (float): The price at which you plan to enter the trade
        stop_price (float): Your stop loss price
        risk_amount (float): The amount of money you're willing to risk on this trade
        """
        if entry_price <= 0 or stop_price <= 0 or risk_amount <= 0:
            raise ValueError("All input values must be positive numbers")
        
        if stop_price >= entry_price:
            raise ValueError("Stop price must be below entry price for long positions")
        
        # Calculate the price difference and risk percentage
        price_difference = entry_price - stop_price
        self.risk_percentage = (price_difference / entry_price) * 100
        
        # Calculate position size based on risk
        self.qty = int(risk_amount / price_difference)
        
        # Calculate total position value and potential loss
        self.total_value = self.qty * entry_price
        self.potential_loss = self.qty * (entry_price - stop_price)
        self.price_per_share = entry_price

    def get_qty(self) -> int:
        """
        Get the current quantity of shares.
        
        Returns:
        int: The current quantity of shares.
        """
        return self.qty

    def set_qty(self, qty: int):
        """
        Set the quantity of shares.
        
        Parameters:
        qty (int): The quantity of shares to set.
        """
        self.qty = qty

    def reset(self):
        """
        Reset the quantity of shares and other attributes to zero.
        """
        self.qty = 0
        self.total_value = 0.0
        self.risk_percentage = 0.0
        self.price_per_share = 0.0
        self.potential_loss = 0.0

# todo: add a way to set the trace price to a specific value
@dataclass
class TraceX:
    name: str = ''
    ls: str = ''
    price:float = np.nan

    def get_price(self, data:pd.DataFrame, barsAgo:int=1) -> float:
        if self.price > 0:
            return self.price
        self.price = data['close'].iat[-barsAgo]
        return self.price
    
    def reset(self):
        self.price = np.nan

@dataclass
class AccelX:
    name: str = ''
    ls: str = ''
    priceCol: str = ''
    accel: float = np.nan

    def get_value(self, data:pd.DataFrame) -> float:
        self.accel = data[self.priceCol].iat[-1]
        return self.accel
    
    def reset(self):            
        self.accel = np.nan


class PriceXStatus:
    ENTRY_PRICE_PENDING = 'ENTRY_PRICE_PENDING'
    ENTRY_PRICE_FOUND = 'ENTRY_PRICE_FOUND'
    IN_TRADE = 'IN_TRADE'
    CANCELLED = 'CANCELLED'
    INIT_STOP_PRICE_PENDING = 'INIT_STOP_PRICE_PENDING'
    INIT_STOP_PRICE_FOUND = 'INIT_STOP_PRICE_FOUND'
    TARGET_PRICE_PENDING = 'TARGET_PRICE_PENDING'
    TARGET_PRICE_FOUND = 'TARGET_PRICE_FOUND'
    TRAIL_PRICE_PENDING = 'TRAIL_PRICE_PENDING'
    TRAIL_PRICE_FOUND = 'TRAIL_PRICE_FOUND'
    STOPPED_OUT = 'STOPPED_OUT'
    TARGET_HIT = 'TARGET_HIT'


ENTERING_STATES = [
    PriceXStatus.ENTRY_PRICE_PENDING, 
    PriceXStatus.ENTRY_PRICE_FOUND, 
    PriceXStatus.INIT_STOP_PRICE_PENDING, 
    PriceXStatus.INIT_STOP_PRICE_FOUND, 
    PriceXStatus.TARGET_PRICE_PENDING, 
    PriceXStatus.TARGET_PRICE_FOUND, 
    PriceXStatus.TRAIL_PRICE_PENDING,
      PriceXStatus.TRAIL_PRICE_FOUND]

TRADE_STATES = [
    PriceXStatus.IN_TRADE]

EXIT_STATES = [
    PriceXStatus.TARGET_HIT, 
    PriceXStatus.STOPPED_OUT,
    PriceXStatus.CANCELLED]


@dataclass
class PriceX:
    name: str = ''
    ls: str = ''
    includeTarget: bool = False
    entry: EntryX = None
    stop: StopX = None
    target: TargetX = None
    trails: list = field(default_factory=list) # list of TrailX objects
    qty: QtyX = None
    trace: TraceX = None
    accel: AccelX = None
    stopPrice: float = np.nan
    stopCurrentName: str = ''
    offsetVal: float = 0.0
    activeTrails: list = field(default_factory=list) # list of TrailX objects
    entryIndex: pd.DatetimeIndex = None

    def __post_init__(self):
        self.status = PriceXStatus.ENTRY_PRICE_PENDING
        self.trails.reverse()
        self.activeTrails = []
        self.object_count = {
            'EntryX': 0,
            'StopX': 0,
            'TargetX': 0,
            'TrailX': 0,
            'RiskX': 0,
            'TraceX': 0,
            'AccelX': 0
        }
        self.entry = EntryX()
        self.stop  = StopX()
        self.target = TargetX()
        self.risk  = RiskX()
        self.trace = TraceX()
        self.accel = AccelX(priceCol='ACC_close')
        self.qty = QtyX()
        self.entryName = ''
        self.stopName = ''
        self.targetName = ''
        self.riskName = ''
        self.set_ls()
        self.set_names()

    def new_name(self, obj):
        csl_name = obj.__class__.__name__
        if obj.name == '':
            self.object_count[csl_name] += 1
        return f"{self.name}_{csl_name[:3]}{self.object_count[csl_name]}"
   
    def set_names(self):
        self.entry.name = self.new_name(self.entry)
        self.stop.name = self.new_name(self.stop)
        self.target.name = self.new_name(self.target)
        self.risk.name = self.new_name(self.risk)
        self.trace.name = self.new_name(self.trace)
        self.accel.name = self.new_name(self.accel)
        for t in self.trails: t.name = self.new_name(t)
        self.entryName = self.entry.name
        self.stopName = f"{self.name}_stop" # the current stop price even if trails are active
        self.targetName = self.target.name
        self.riskName = self.risk.name

    def set_columns(self, df:pd.DataFrame):
        df[self.entry.name] = np.nan
        df[self.stop.name] = np.nan
        df[self.target.name] = np.nan
        df[[t.name for t in self.trails]] = np.nan
        df[self.risk.name] = np.nan
        df[self.trace.name] = np.nan
        df[self.accel.name] = np.nan
        df[self.entryName] = np.nan
        df[self.stopName] = np.nan
        df[self.targetName] = np.nan
        df[self.riskName] = np.nan

    def set_ls(self):
        ls = self.ls
        self.ls = ls
        self.entry.set_ls(ls)
        self.stop.set_ls(ls)
        self.target.set_ls(ls)
        self.risk.ls = ls
        self.trace.ls = ls
        self.accel.ls = ls
        for t in self.trails: t.set_ls(ls)

    def reset(self):
        self.status = PriceXStatus.ENTRY_PRICE_PENDING
        self.entry.reset()
        self.stop.reset()
        self.target.reset()
        for t in self.trails: t.reset()
        self.qty.reset()
        self.risk.reset()
        self.trace.reset()
        self.stopPrice = np.nan
        self.stopCurrentName = ''
        self.activeTrails = []

    def get_stop_price(self):
        return self.stopPrice
    
    def get_entry_price(self):
        return self.entry.price
    
    def get_target_price(self):
        return self.target.price
    
    def get_risk_reward_ratio(self):
        return self.risk.rRatio
    
    def get_active_stop_name(self):
        return self.stopCurrentName

    def get_init_value(self, df:pd.DataFrame, initType:str):
        if initType == 'rrr':
            # print(f"get_init_value = 'rrr' {self.entry.price=}, {self.stopPrice=}, {df['close'].iat[-1]=}")
            return self.risk.get_value(self.entry.price, self.stop.price, df['close'].iat[-1])
        if initType == 'accel':
            return self.accel.get_value(df)
        
    def compute_stop(self, df:pd.DataFrame):
            if self.ls == 'LONG':
                self.activeTrails += [t for t in self.trails if self.get_init_value(df, t.initType) > t.initTrigVal and t not in self.activeTrails]
                if len(self.activeTrails) > 0:
                    active_trails = [(t.name, t.get_price(df)) for t in self.activeTrails]

                    # Find the trail with the maximum stop price
                    max_trail = max(active_trails, key=lambda x: x[1])
                    if max_trail[1] > self.stopPrice:
                        self.stopCurrentName = max_trail[0]
                        limit_price = df['low'].iat[-1] - 0.01
                        stop_price_with_limit = min(max_trail[1], limit_price)
                        self.stopPrice = max(stop_price_with_limit, self.stopPrice)


            elif self.ls == 'SHORT':
                self.activeTrails += [t for t in self.trails if self.get_init_value(df, t.initType) > t.initTrigVal and t not in self.activeTrails]
                if len(self.activeTrails) > 0:
                    active_trails = [(t.name, t.get_price(df)) for t in self.activeTrails]

                    # Find the trail with the maximum stop price
                    max_trail = min(active_trails, key=lambda x: x[1])
                    if max_trail[1] < self.stopPrice:
                        limit_price = df['high'].iat[-1] + 0.01
                        self.stopCurrentName = max_trail[0]
                        self.stopPrice = min(max(max_trail[1], limit_price), self.stopPrice)

    def compute_qty(self, riskAmount:float):
        self.qty.compute_qty(self.entry.price, self.stop.price, riskAmount)

    def set_qty(self, qty:int):
        self.qty.set_qty(qty)



    def run_row(self, df:pd.DataFrame, isRth:bool=None):
        # check ls is set
        if self.ls == '':
            raise ValueError("ls is not set. us set_ls() to set it after all the objects have been created")
        # check if reset required 
        if self.status in [PriceXStatus.TARGET_HIT, PriceXStatus.STOPPED_OUT]:
            self.reset()
            self.status = PriceXStatus.ENTRY_PRICE_PENDING

        # find entry price
        if self.status == PriceXStatus.ENTRY_PRICE_PENDING:
            if self.entry.get_price(df) > 0:
                if not isRth:
                    self.entry.set_outside_rth(df)
                self.status = PriceXStatus.ENTRY_PRICE_FOUND
                self.entryIndex = df.index[-1]

        # set init stop price
        if self.status == PriceXStatus.ENTRY_PRICE_FOUND:
            self.stopPrice =  self.stop.get_price(df) # stops are always set to main stop price
            if self.stopPrice > 0:
                self.stopCurrentName = self.stop.name
                self.status = PriceXStatus.INIT_STOP_PRICE_FOUND

        # set target price if required
        if self.includeTarget:
            if self.status == PriceXStatus.INIT_STOP_PRICE_FOUND:
                if self.target.get_price(df, self.entry.price, self.stop.price) > 0:
                    self.status = PriceXStatus.TARGET_PRICE_FOUND

        # skip entry index is now 
        if self.status in [PriceXStatus.ENTRY_PRICE_FOUND, PriceXStatus.INIT_STOP_PRICE_FOUND, PriceXStatus.TARGET_PRICE_FOUND]:
            if self.entryIndex == df.index[-1]:
                return
        
        # see if entry price has been hit
        if self.status in [PriceXStatus.INIT_STOP_PRICE_FOUND, PriceXStatus.TARGET_PRICE_FOUND]:
            if self.entry.has_triggered(df, priceType='entry'):
                self.status = PriceXStatus.IN_TRADE
            if self.stop.has_triggered(df, priceType='stop'):
                self.status = PriceXStatus.STOPPED_OUT

        # see if stopped out
        if self.status == PriceXStatus.IN_TRADE:
            if self.stop.has_triggered(df, priceType='stop'):
                self.status = PriceXStatus.STOPPED_OUT
            for t in self.trails:
                if t.has_triggered(df, priceType='stop'):
                    self.status = PriceXStatus.STOPPED_OUT

        # see if target price has been hit
        if self.status == PriceXStatus.IN_TRADE:
            if self.target.has_triggered(df, priceType='target'):
                self.status = PriceXStatus.TARGET_HIT
        
        if self.status == PriceXStatus.IN_TRADE:
            self.compute_stop(df)



