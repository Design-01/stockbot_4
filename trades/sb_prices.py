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
    offsetVal: float = 0.0
    barsAgo: int = 1

    def get_price(self, data:pd.DataFrame) -> float:
        pass
    
    def reset(self):
        self.price = np.nan

    def has_triggered(self, data, priceType) -> bool:
        if self.ls == 'LONG' and priceType in ['entry', 'target']: return data['high'].iat[-1] >= self.price
        if self.ls == 'SHORT' and priceType in ['entry', 'target']: return data['low'].iat[-1] <= self.price
        if self.ls == 'LONG' and priceType == 'stop': return data['low'].iat[-1] <= self.price
        if self.ls == 'SHORT' and priceType == 'stop': return data['high'].iat[-1] >= self.price
        print (f"has_triggered Error: {self.ls=} {priceType=} {self.price=}")

@dataclass
class EntryX(BaseX):

    def get_price(self, data:pd.DataFrame, barsAgo:int=1) -> float:
        if self.price > 0:
            return self.price
        if self.ls == 'LONG':
            breaksPrevHigh = data['close'].iat[-barsAgo] > data['high'].iat[-barsAgo-1]
            if breaksPrevHigh:
                self.price = data['close'].iat[-barsAgo]
        elif self.ls == 'SHORT':
            breaksPrevLow = data['close'].iat[-barsAgo] < data['low'].iat[-barsAgo-1]
            if breaksPrevLow:
                self.price = data['close'].iat[-barsAgo]
        return self.price


@dataclass
class StpX(BaseX):
    longPriceCol: str = ''
    shortPriceCol: str = ''

    def set_ls(self, ls):
        self.ls = ls
        self.priceCol = self.longPriceCol if self.ls == 'LONG' else self.shortPriceCol

    def get_price(self, data:pd.DataFrame) -> float:
        self.price = data[self.priceCol].iat[-self.barsAgo-1]
        return self.price


@dataclass
class TargetX(BaseX):
    longPriceCol: str = ''
    shortPriceCol: str = ''
    rrIfNoTarget: float = 2

    def set_ls(self, ls):
        self.ls = ls
        self.priceCol = self.longPriceCol if self.ls == 'LONG' else self.shortPriceCol

    def get_price(self, data:pd.DataFrame, entryPrice:float, stopPrice:float) -> float:
        if self.price > 0:
            return self.price
        self.price = data[self.priceCol].iat[-self.barsAgo-1]
        if math.isnan(self.price):
            self.price = data['close'].iat[-1] + (self.rrIfNoTarget * abs(entryPrice - stopPrice))
        return self.price
    

@dataclass
class TrailX(BaseX):
    name: str = ''
    ls: str = ''
    price:float = np.nan
    initType: str = 'rrr'
    initTrigVal: float = 0.0
    barsAgo: int = 1
    longPriceCol: str = ''
    shortPriceCol: str = ''
    
    def set_ls(self, ls):
        self.ls = ls
        self.priceCol = self.longPriceCol if self.ls == 'LONG' else self.shortPriceCol

    def get_price(self, data:pd.DataFrame) -> float:
        self.price = data[self.priceCol].iat[-self.barsAgo-1]
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


@dataclass
class PriceX:
    name: str = ''
    ls: str = ''
    includeTarget: bool = False
    entry: EntryX = None
    stop: StpX = None
    target: TargetX = None
    trails: list = field(default_factory=list) # list of TrailX objects
    risk: RiskX = None
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
        self.set_ls()
        self.entryName  = f"{self.name}_Ent"
        self.stopName   = f"{self.name}_Stp"
        self.targetName = f"{self.name}_Tgt"
        self.riskName   = f"{self.name}_Rsk"

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
        self.entry.ls = ls
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
                    

    def run_row(self, df:pd.DataFrame):
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



