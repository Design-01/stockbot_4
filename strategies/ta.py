import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

@dataclass
class Indicator(ABC):
    column: str = None
    

    @abstractmethod
    def run(self, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
        pass

@dataclass
class MA(Indicator):
    period: int = 20

    def __post_init__(self):
        self.names = f"MA_{self.column[:2]}_{self.period}"

    def run(self, data: pd.DataFrame) -> pd.Series:
        return data[self.column].rolling(window=self.period).mean().rename(self.names)

@dataclass
class MACD(Indicator):
    fast: int = 12
    slow: int = 26
    signal: int = 9

    def __post_init__(self):
        self.macd_name = f"MACD_{self.column[:2]}_{self.fast}_{self.slow}_{self.signal}_MACD"
        self.signal_name = f"MACD_{self.column[:2]}_{self.fast}_{self.slow}_{self.signal}_Signal"
        self.histogram_name = f"MACD_{self.column[:2]}_{self.fast}_{self.slow}_{self.signal}_Histogram"
        self.names = [self.macd_name, self.signal_name, self.histogram_name]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        fast_ema = data[self.column].ewm(span=self.fast, adjust=False).mean()
        slow_ema = data[self.column].ewm(span=self.slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        histogram = macd - signal_line
        
        
        return pd.DataFrame({
            self.macd_name: macd,
            self.signal_name: signal_line,
            self.histogram_name: histogram
        })
    
@dataclass
class HPLP(Indicator):
    hi_col: str = 'high'
    lo_col: str = 'low'
    span: int = 5

    def __post_init__(self):
        self.name_hp = f"HP_{self.hi_col[:2]}_{self.span}"
        self.name_lp = f"LP_{self.lo_col[:2]}_{self.span}"
        self.names = [self.name_hp, self.name_lp]

    def run(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        df = ohlcv.copy()

        # Calculate rolling max and min with min_periods=1
        window = self.span * 2 + 1
        high_max = df[self.hi_col].rolling(window=window, center=True, min_periods=1).max()
        low_min = df[self.lo_col].rolling(window=window, center=True, min_periods=1).min()

        # Identify high and low points
        hi = self.hi_col
        lo = self.lo_col
        df[self.name_hp] = df[hi].where((df[hi] == high_max) & (df[hi].shift(1) != high_max), np.nan)
        df[self.name_lp] = df[lo].where((df[lo] == low_min) & (df[lo].shift(1) != low_min), np.nan)

        return df[[self.name_hp, self.name_lp]]
