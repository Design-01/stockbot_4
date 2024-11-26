from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Protocol
import pandas as pd
import numpy as np

class Condition(Protocol):
    def is_valid(self, df: pd.DataFrame) -> bool:
        ...

@dataclass
class PriceAbovePreviousHigh:
    def is_valid(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        return df['close'].iloc[-1] > df['high'].iloc[-2]

@dataclass
class LowerHighsPreviousBars:
    num_bars: int

    def is_valid(self, df: pd.DataFrame) -> bool:
        if self.num_bars  >= len(df):
            return False
        index = len(df) - 1   
        for i in range(1, self.num_bars):
            if df['high'].iloc[index - i] >= df['high'].iloc[index - i - 1]:
                return False
        return True

@dataclass
class PriceNotBelowMA:
    ma_column: str 
    atr_column: str 
    atr_multiplier: float = 0.5

    def is_valid(self, df: pd.DataFrame) -> bool:
        if len(df) == 0:
            return False
        index = len(df) - 1
        min_price = (df[self.ma_column].iloc[index] - 
                    (df[self.atr_column].iloc[index] * self.atr_multiplier))
        return df['close'].iloc[index] >= min_price

@dataclass
class EntryStrategy:
    name:str 
    conditions: List[Condition]
    
    def should_enter(self, df: pd.DataFrame, print_results: bool = False) -> bool:
        conds = {cond.__class__.__name__: cond.is_valid(df) for cond in self.conditions}
        if print_results:
            for k, v in conds.items():
                print(f"{k}: {v}")
        return all(conds.values())