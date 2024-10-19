import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

def preprocess_data(func):
    def wrapper(self, data: pd.DataFrame, *args, **kwargs):
        data = self.compute_rows_to_update(data, self.names, self.rowsToUpdate)
        return func(self, data, *args, **kwargs)
    return wrapper

@dataclass
class TA(ABC):
    column: str = None

    @abstractmethod
    def run(self, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
        pass

    
    def compute_rows_to_update(self, df, column_names, rows_to_update):
            """
            Compute the slice of the DataFrame that needs to be updated based on the columns provided.
            
            Parameters:
            df (pd.DataFrame): The main DataFrame
            column_names (list): The names of the columns to check
            rows_to_update (int): The number of rows to add to the last valid index
            
            Returns:
            pd.DataFrame: The sliced DataFrame that needs to be updated
            """
            if isinstance(column_names, str):
                column_names = [column_names]
            
            last_valid_indices = []
            
            for column_name in column_names:
                if column_name in df.columns:
                    last_valid_index = df[column_name].last_valid_index()
                    if last_valid_index is not None:
                        lookback_index = df.index.get_loc(last_valid_index)
                        last_valid_indices.append(lookback_index)
            
            if not last_valid_indices:
                lookback_index = 0
            else:
                lookback_index = max(min(last_valid_indices) - rows_to_update, 0) # + 1 # added 1 just to be sure

            return df.iloc[-lookback_index:]



@dataclass
class MA(TA):
    period: int = 20

    def __post_init__(self):
        self.names = f"MA_{self.column[:2]}_{self.period}"
        self.rowsToUpdate = self.period 

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        return data[self.column].rolling(window=self.period).mean().rename(self.names)

@dataclass
class MACD(TA):
    fast: int = 12
    slow: int = 26
    signal: int = 9
    fastcol: str = 'close'
    slowcol: str = 'close'
    signalcol: str = 'close'

    def __post_init__(self):
        self.macd_name = f"MACD_{self.signalcol[:2]}_{self.fast}_{self.slow}_{self.signal}_MACD"
        self.signal_name = f"MACD_{self.signalcol[:2]}_{self.fast}_{self.slow}_{self.signal}_Signal"
        self.histogram_name = f"MACD_{self.signalcol[:2]}_{self.fast}_{self.slow}_{self.signal}_Histogram"
        self.names = [self.macd_name, self.signal_name, self.histogram_name]
        self.rowsToUpdate = max(self.slow, self.signal, self.fast) 

    @preprocess_data
    def run(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        fast_ema = ohlcv[self.fastcol].ewm(span=self.fast, adjust=False).mean()
        slow_ema = ohlcv[self.slowcol].ewm(span=self.slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        histogram = macd - signal_line
        
        
        return pd.DataFrame({
            self.macd_name: macd,
            self.signal_name: signal_line,
            self.histogram_name: histogram
        })
    
@dataclass
class HPLP(TA):
    hi_col: str = 'high'
    lo_col: str = 'low'
    span: int = 5

    def __post_init__(self):
        self.name_hp = f"HP_{self.hi_col[:2]}_{self.span}"
        self.name_lp = f"LP_{self.lo_col[:2]}_{self.span}"
        self.names = [self.name_hp, self.name_lp]

    @preprocess_data
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

import pandas as pd
from dataclasses import dataclass

@dataclass
class SupportResistance(TA):
    hi_col: str = 'high'
    lo_col: str = 'low'
    tolerance: float = 0.01
    touch_tolerance: float = 0.005

    def __post_init__(self):
        self.name_support = f"SR_Support"
        self.name_resistance = f"SR_Resistance"
        self.name_support_strength = f"SR_Support_Strength"
        self.name_resistance_strength = f"SR_Resistance_Strength"
        self.names = [self.name_support, self.name_resistance, self.name_support_strength, self.name_resistance_strength]

    def cluster_levels(self, points):
        clusters = []
        for point in sorted(points):
            if not clusters or abs(point - clusters[-1][0]) > self.tolerance * point:
                clusters.append([point])
            else:
                clusters[-1].append(point)
        return [sum(cluster) / len(cluster) for cluster in clusters]

    def count_touches(self, prices, level):
        return sum(1 for price in prices if abs(price - level) <= self.touch_tolerance * level)

    def run(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        df = ohlcv.copy()

        # Identify high and low points
        highs = df[self.hi_col].tolist()
        lows = df[self.lo_col].tolist()

        # Cluster levels
        resistance_levels = self.cluster_levels(highs)
        support_levels = self.cluster_levels(lows)

        # Count touches
        resistance_strength = [self.count_touches(highs, level) for level in resistance_levels]
        support_strength = [self.count_touches(lows, level) for level in support_levels]

        # Find the most recent support and resistance levels
        recent_support = support_levels[-1]
        recent_resistance = resistance_levels[-1]

        # Calculate the average trading range
        avg_trading_range = np.mean(df[self.hi_col] - df[self.lo_col])

        # Find points within a multiple range of the average trading range
        multiple_range = 2  # Adjust this value as needed
        support_band = [level for level in support_levels if abs(level - recent_support) <= multiple_range * avg_trading_range]
        resistance_band = [level for level in resistance_levels if abs(level - recent_resistance) <= multiple_range * avg_trading_range]

        # Create a DataFrame with results
        result = pd.DataFrame(index=df.index)
        result[self.name_resistance] = [resistance_levels] * len(df)
        result[self.name_support] = [support_levels] * len(df)
        result[self.name_resistance_strength] = [resistance_strength] * len(df)
        result[self.name_support_strength] = [support_strength] * len(df)
        result['Recent_Support'] = recent_support
        result['Recent_Resistance'] = recent_resistance
        result['Support_Band'] = [support_band] * len(df)
        result['Resistance_Band'] = [resistance_band] * len(df)

        return result
