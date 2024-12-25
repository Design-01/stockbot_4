import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Union, Dict

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

    @staticmethod
    def normalize(series: pd.Series, max_value: float) -> pd.Series:
        """Efficient normalization to -100 to 100 range"""
        return (series / max_value).clip(-1, 1) * 100


@dataclass
class MA(TA):
    period: int = 20

    def __post_init__(self):
        self.name = f"MA_{self.column[:2]}_{self.period}"
        self.names = f"MA_{self.column[:2]}_{self.period}"
        self.rowsToUpdate = self.period 

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        return data[self.column].rolling(window=self.period).mean().rename(self.name)


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
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        fast_ema = data[self.fastcol].ewm(span=self.fast, adjust=False).mean()
        slow_ema = data[self.slowcol].ewm(span=self.slow, adjust=False).mean()
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
        self.rowsToUpdate = 200

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate rolling max and min with min_periods=1
        window = self.span * 2 + 1
        high_max = df[self.hi_col].rolling(window=window, center=True, min_periods=1).max()
        low_min = df[self.lo_col].rolling(window=window, center=True, min_periods=1).min()

        # Identify high and low points
        hi = self.hi_col
        lo = self.lo_col
        df[self.name_hp] = df[hi].where((df[hi] == high_max) & (df[hi].shift(1) != high_max), np.nan)
        df[self.name_lp] = df[lo].where((df[lo] == low_min) & (df[lo].shift(1) != low_min), np.nan)

        # Set the last 3 values to NaN  to avoid lookahead bias
        df.iloc[-3:, df.columns.get_loc(self.name_hp)] = np.nan
        df.iloc[-3:, df.columns.get_loc(self.name_lp)] = np.nan

        return df[[self.name_hp, self.name_lp]]

    
@dataclass
class ATR(TA):
    hi_col: str = 'high'
    lo_col: str = 'low'
    span: int = 14  # Common default span for ATR

    def __post_init__(self):
        self.name_atr = f"ATR_{self.span}"
        self.names = [self.name_atr]
        self.rowsToUpdate = 200

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate True Range (TR)
        df['TR'] = df[self.hi_col] - df[self.lo_col]

        # Calculate ATR using a rolling mean of the True Range
        df[self.name_atr] = df['TR'].rolling(window=self.span, min_periods=1).mean()

        # Drop the intermediate 'TR' column
        df.drop(columns=['TR'], inplace=True)

        return df[[self.name_atr]]

import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class SupResOld(TA):
    hi_point_col: str = 'HP'
    lo_point_col: str = 'LP'
    pointsAgo: int = 1
    tolerance: float = 0.01
    atr_period: int = 14

    def __post_init__(self):
        self.name_support = f"Sup_{self.pointsAgo}"
        self.name_resistance = f"Res_{self.pointsAgo}"
        self.name_res_upper = f"{self.name_resistance}_Upper"
        self.name_res_lower = f"{self.name_resistance}_Lower"
        self.name_sup_upper = f"{self.name_support}_Upper"
        self.name_sup_lower = f"{self.name_support}_Lower"
        self.names = [self.name_support, self.name_resistance, self.name_res_upper, self.name_res_lower, self.name_sup_upper, self.name_sup_lower]
        self.rowsToUpdate = 200

    def calculate_atr(self, data):
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean()

    def initialize_columns(self, df):
        for col in self.names:
            df[col] = np.nan

    def find_levels(self, all_points, last_close):
        resistance_points = all_points[all_points > last_close]
        resistance_level = resistance_points[self.pointsAgo - 1] if len(resistance_points) >= self.pointsAgo else None

        support_points = all_points[all_points < last_close][::-1]  # Reverse order for support
        support_level = support_points[self.pointsAgo - 1] if len(support_points) >= self.pointsAgo else None

        return resistance_level, support_level

    def update_levels(self, df, level, level_name, upper_name, lower_name, all_points):
        if level is not None:
            level_index = df.index[(df[self.hi_point_col] == level) | (df[self.lo_point_col] == level)][-1]
            df.loc[level_index:, level_name] = level

            atr_at_level = df.loc[level_index, 'ATR']
            upper_bound = level + (atr_at_level * self.tolerance)
            lower_bound = level - (atr_at_level * self.tolerance)

            points_in_range = all_points[(all_points >= lower_bound) & (all_points <= upper_bound)]

            if len(points_in_range) > 0:
                df.loc[level_index:, upper_name] = points_in_range.max()
                df.loc[level_index:, lower_name] = points_in_range.min()
            else:
                df.loc[level_index:, upper_name] = upper_bound
                df.loc[level_index:, lower_name] = lower_bound

    def check_existing_bounds(self, df):
        previous_res_upper = f"Res_{self.pointsAgo - 1}_Upper"
        previous_res_lower = f"Res_{self.pointsAgo - 1}_Lower"
        previous_sup_upper = f"Sup_{self.pointsAgo - 1}_Upper"
        previous_sup_lower = f"Sup_{self.pointsAgo - 1}_Lower"

        last_res_upper = last_res_lower = last_sup_upper = last_sup_lower = None

        if previous_res_upper in df.columns and previous_res_lower in df.columns:
            if not df[previous_res_upper].dropna().empty:
                last_res_upper = df[previous_res_upper].dropna().iloc[-1]
            if not df[previous_res_lower].dropna().empty:
                last_res_lower = df[previous_res_lower].dropna().iloc[-1]

        if previous_sup_upper in df.columns and previous_sup_lower in df.columns:
            if not df[previous_sup_upper].dropna().empty:
                last_sup_upper = df[previous_sup_upper].dropna().iloc[-1]
            if not df[previous_sup_lower].dropna().empty:
                last_sup_lower = df[previous_sup_lower].dropna().iloc[-1]

        return last_res_upper, last_res_lower, last_sup_upper, last_sup_lower

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        last_close = df['close'].iloc[-1]

        # Calculate ATR
        df['ATR'] = self.calculate_atr(df)

        # Combine high points and low points into a single sorted list
        all_points = pd.concat([df[self.hi_point_col], df[self.lo_point_col]]).sort_values().dropna().unique()

        # Check for existing bounds from previous instance
        last_res_upper, last_res_lower, last_sup_upper, last_sup_lower = self.check_existing_bounds(df)

        # Use existing bounds if they exist
        if last_res_upper is not None and last_res_lower is not None:
            last_close = (last_res_upper + last_res_lower) / 2

        if last_sup_upper is not None and last_sup_lower is not None:
            last_close = (last_sup_upper + last_sup_lower) / 2

        # Find resistance and support levels
        resistance_level, support_level = self.find_levels(all_points, last_close)

        # Initialize new columns
        self.initialize_columns(df)

        # Update resistance levels
        self.update_levels(df, resistance_level, self.name_resistance, self.name_res_upper, self.name_res_lower, all_points)

        # Update support levels
        self.update_levels(df, support_level, self.name_support, self.name_sup_upper, self.name_sup_lower, all_points)

        # Remove the temporary ATR column
        df = df.drop('ATR', axis=1)

        return df


@dataclass
class SupRes(TA):
    hi_point_col: str = 'HP'
    lo_point_col: str = 'LP'
    atr_col: str = 'ATR'
    tolerance: float = 0.01
    rowsToUpdate: int = 200
    names: list = field(init=False)
    
    def __post_init__(self):
        self.name_Res_1 = "Res_1"
        self.name_Res_1_Upper = "Res_1_Upper"
        self.name_Res_1_Lower = "Res_1_Lower"
        self.name_Res_2 = "Res_2"
        self.name_Res_2_Upper = "Res_2_Upper"
        self.name_Res_2_Lower = "Res_2_Lower"

        self.name_Sup_1 = "Sup_1"
        self.name_Sup_1_Upper = "Sup_1_Upper"
        self.name_Sup_1_Lower = "Sup_1_Lower"
        self.name_Sup_2 = "Sup_2"
        self.name_Sup_2_Upper = "Sup_2_Upper"
        self.name_Sup_2_Lower = "Sup_2_Lower"
        self.names = [self.name_Res_1, self.name_Res_1_Upper, self.name_Res_1_Lower,
                      self.name_Res_2, self.name_Res_2_Upper, self.name_Res_2_Lower,
                      self.name_Sup_1, self.name_Sup_1_Upper, self.name_Sup_1_Lower,
                      self.name_Sup_2, self.name_Sup_2_Upper, self.name_Sup_2_Lower]

    def run(self, df, startwith='res'):
        last_close = df['close'].iloc[-1]
        
        # Initialize all columns with NaN
        for name in self.names:
            df[name] = np.nan
        
        if startwith == 'res':
            res_levels = self._find_levels(df, last_close, 'res', 2)
            sup_levels = self._find_levels(df, last_close, 'sup', 2)
        else:
            sup_levels = self._find_levels(df, last_close, 'sup', 2)
            res_levels = self._find_levels(df, last_close, 'res', 2)
        
        # Add resistance levels
        for i, (level, upper, lower, idx) in enumerate(res_levels):
            level_name = f"Res_{i+1}"
            upper_name = f"Res_{i+1}_Upper"
            lower_name = f"Res_{i+1}_Lower"
            
            df.loc[df.index >= idx, level_name] = level
            df.loc[df.index >= idx, upper_name] = upper
            df.loc[df.index >= idx, lower_name] = lower
        
        # Add support levels
        for i, (level, upper, lower, idx) in enumerate(sup_levels):
            level_name = f"Sup_{i+1}"
            upper_name = f"Sup_{i+1}_Upper"
            lower_name = f"Sup_{i+1}_Lower"
            
            df.loc[df.index >= idx, level_name] = level
            df.loc[df.index >= idx, upper_name] = upper
            df.loc[df.index >= idx, lower_name] = lower
        
        return df

    def _find_levels(self, df, start_value, level_type, num_levels):
        levels = []
        hp_series = df[self.hi_point_col].dropna()
        lp_series = df[self.lo_point_col].dropna()

        for i in range(num_levels):
            if level_type == 'res':
                hp_candidates = hp_series[hp_series > start_value]
                lp_candidates = lp_series[lp_series > start_value]
            else:
                hp_candidates = hp_series[hp_series < start_value]
                lp_candidates = lp_series[lp_series < start_value]
            
            all_candidates = pd.concat([hp_candidates, lp_candidates]).sort_values()
            # print(f'{all_candidates=}')
            
            if level_type == 'res':
                level = all_candidates.iloc[0] if not all_candidates.empty else np.nan
            else:
                level = all_candidates.iloc[-1] if not all_candidates.empty else np.nan

            # print(f'{level=}')
            
            if pd.isna(level):
                levels.append((np.nan, np.nan, np.nan, df.index[-1]))
                continue
            
            idx = df.index[((df[self.hi_point_col] == level) | (df[self.lo_point_col] == level))].min()
            atr = df.loc[idx, self.atr_col]
            
            # Calculate initial tolerance zone
            upper_bound = level + (self.tolerance * atr)
            lower_bound = level - (self.tolerance * atr)
            
            # Find the highest and lowest points within the tolerance zone
            zone_candidates = all_candidates[(all_candidates >= lower_bound) & (all_candidates <= upper_bound)]
            upper = zone_candidates.max()
            lower = zone_candidates.min()
            
            levels.append((level, upper, lower, idx))
            start_value = upper if level_type == 'res' else lower
        
        return levels
    

@dataclass
class MansfieldRSI(TA):
    close_col: str = 'close'  # Column name for stock close price
    market_col: str = 'index_close'  # Column name for index close price
    span: int = 14  # Default lookback period
    
    def __post_init__(self):
        self.name_mrsi = f"MRSI_{self.span}"
        self.names = [self.name_mrsi]
        self.rowsToUpdate = 200

    @preprocess_data
    def _old_run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Step 1: Calculate raw Relative Strength (RS)
        rs = df[self.close_col] / df[self.market_col]
        
        # Step 2: Smooth RS using moving average
        rs_ma = rs.rolling(window=self.span, min_periods=1).mean()
        
        # Step 3: Normalize RS by subtracting its moving average
        normalized_rs = rs - rs_ma
        
        # Step 4: Scale the normalized RS
        df[self.name_mrsi] = normalized_rs * 100
        
        # Handle any NaN values that might occur at the beginning
        df[self.name_mrsi] = df[self.name_mrsi].fillna(0)
        
        return df[[self.name_mrsi]]
    
    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Step 1: Calculate raw Relative Strength (RS)
        rs = df[self.close_col] / df[self.market_col]
        
        # Step 2: Calculate log returns of RS
        rs_returns = np.log(rs / rs.shift(1))
        
        # Step 3: Calculate the moving average and standard deviation
        rs_ma = rs_returns.rolling(window=self.span, min_periods=1).mean()
        rs_std = rs_returns.rolling(window=self.span, min_periods=1).std()
        
        # Step 4: Calculate standardized (z-score) RS
        df[self.name_mrsi] = ((rs_returns - rs_ma) / rs_std).fillna(0)
        
        return df[[self.name_mrsi]]


@dataclass
class DIR(TA):
    """
    Direction Indicator
    Measures trend direction based on moving average slope, normalized to -100 to +100.
    Positive values indicate upward trend, negative values indicate downward trend.
    Values closer to extremes indicate steeper slopes.
    """
    period: int = 50
    max_slope: float = 0.02  # Maximum expected slope (2% per period)
    
    def __post_init__(self):
        self.name = f"MADIR_{self.column[:2]}_{self.period}"
        self.names = [self.name]
        self.rowsToUpdate = self.period + 1
    
    @staticmethod
    def normalize(series: pd.Series, max_value: float) -> pd.Series:
        """Efficient normalization to -100 to 100 range"""
        return (series / max_value).clip(-1, 1) * 100
    
    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        ma = data[self.column].rolling(window=self.period).mean()
        slope = ma.diff() / ma.shift(1)
        return self.normalize(slope, self.max_slope).rename(self.names)


@dataclass
class ColVal:
    """Checks if a column value is above/below a threshold"""
    column: str

    def __post_init__(self):
        self.name = f"CV_{self.column}"
        self.names = [self.name]
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.name] = data[self.column] #! Redundancy but just did this to get it working quickly
        return data[self.name]


@dataclass
class VolAcc(TA):
    """
    Acceleration Indicator
    Measures rate of change in trend direction, normalized to -100 to +100.
    Positive values indicate increasing slope (acceleration up),
    negative values indicate decreasing slope (acceleration down).
    """
    max_accel: float = 0.001  # Maximum expected acceleration (0.1% per period)
    
    def __post_init__(self):
        self.name = f"VolACC"
        self.names = [self.name]
        self.rowsToUpdate = len(self.column) + 1
    
    @staticmethod
    def normalize(series: pd.Series, max_value: float) -> pd.Series:
        """Efficient normalization to -100 to 100 range"""
        return (series / max_value).clip(-1, 1) * 100
    
    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        d = data['volume']
        current_slope = d.diff() / d.shift(1)
        prev_slope = d.shift(1).diff() / d.shift(2)
        acceleration = current_slope - prev_slope
        return self.normalize(acceleration, self.max_accel).rename(self.name)


@dataclass
class ACC(TA):
    """
    Acceleration Indicator using dual moving averages
    Measures acceleration by comparing the change in MA differences over time.
    Positive values indicate increasing difference between MAs,
    negative values indicate decreasing difference between MAs.
    """
    fast_ma: int = 5       # Period for faster moving average
    slow_ma: int = 10      # Period for slower moving average
    max_accel: float = 0.001  # Maximum expected acceleration (0.1% per period)
    
    def __post_init__(self):
        self.name = f"ACC_{self.column}"
        self.names = [self.name]
        self.rowsToUpdate = max(self.fast_ma, self.slow_ma) + 2  # +2 for shift operation
    
    @staticmethod
    def normalize(series: pd.Series, max_value: float) -> pd.Series:
        """Efficient normalization to -100 to 100 range"""
        return (series / max_value).clip(-1, 1) * 100
    
    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        # Calculate two moving averages
        fast_ma = data[self.column].rolling(window=self.fast_ma).mean()
        slow_ma = data[self.column].rolling(window=self.slow_ma).mean()
        
        ma_diff = fast_ma - slow_ma
        current_diff = ma_diff
        return self.normalize(current_diff, self.max_accel).rename(self.name)


@dataclass
class Breaks:
    """Checks if price crosses above/below a metric"""
    price_column: str
    direction: str  # 'above' or 'below'
    metric_column: str

    def __post_init__(self):
        self.name = f"BRK_{self.price_column}_{self.direction[:2]}_{self.metric_column}"
        self.names = [self.name]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        curr_price = df[self.price_column]
        prev_price = curr_price.shift(1)
        curr_metric = df[self.metric_column]
        prev_metric = curr_metric.shift(1)

        if self.direction == 'above':
            df[self.name] = (prev_price <= prev_metric) & (curr_price > curr_metric)
            return df
        elif self.direction == 'below':
            df[self.name] = (prev_price >= prev_metric) & (curr_price < curr_metric)
            return df
        
        raise ValueError("Direction must be 'above' or 'below'")


@dataclass
class AboveBelow:
    """Checks if price is above/below a metric"""
    value: str | float
    direction: str  # 'above' or 'below'
    metric_column: str

    def __post_init__(self):
        self.name = f"AB_{self.value}_{self.direction[:2]}_{self.metric_column}"
        self.names = [self.name]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        value =  df[self.value] if isinstance(self.value, str) else self.value
        metric = df[self.metric_column] if isinstance(self.metric_column, str) else self.metric_column

        if self.direction == 'above':
            df[self.name] = value > metric
        elif self.direction == 'below':
            df[self.name] = value < metric
        else:
            raise ValueError("Direction must be 'above' or 'below'")
            
        return df


@dataclass
class PctChange:
    """Calculates percentage change of a column"""
    metric_column: str = 'close'
    period: int = 1

    def __post_init__(self):
        self.name = f"PCT_{self.metric_column}_{self.period}"
        self.names = [self.name]
        self.rowsToUpdate = self.period + 1

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.metric_column not in df.columns:
            raise KeyError(f"Column '{self.metric_column}' not found in DataFrame.")
        df[self.name] = df[self.metric_column].pct_change(periods=self.period) * 100
        return df


@dataclass
class VolDev(TA):
    """Calculates percentage deviation of current volume from its moving average"""
    period: int = 10 # period for the moving average

    def __post_init__(self):
        self.name = f"VDEV_{self.period}"
        self.names = [self.name]
        self.rowsToUpdate = self.period + 1

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Calculate volume moving average
        volume_ma = df[self.column].rolling(window=self.period).mean()
        
        # Calculate percentage deviation
        # ((current - average) / average) * 100
        df[self.name] = ((df[self.column] - volume_ma) / volume_ma) * 100
            
        return df
    

@dataclass
class VolumeThreshold(TA):
    """
    Volume Threshold Indicator
    Identifies when volume is above a specified percentage threshold
    compared to its moving average.
    Returns 1 when above threshold, 0 when below.
    """
    period: int = 10           # Period for moving average
    threshold: float = 0.8     # 80% above moving average = 1.8
    
    def __post_init__(self):
        self.column = 'volume'
        self.name = f"VOL_THRESH_{self.period}_{int(self.threshold*100)}"
        self.names = [self.name]
        self.rowsToUpdate = self.period + 1
    
    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        # Calculate moving average of volume
        volume_ma = data[self.column].rolling(window=self.period).mean()
        
        # Calculate ratio of current volume to moving average
        volume_ratio = data[self.column] / volume_ma
        
        # Create binary signal: 1 if above threshold, 0 if below
        signal = (volume_ratio > (1 + self.threshold)).astype(int)
        
        return signal.rename(self.name)
    

@dataclass
class TrendDuration(TA):
    """
    Trend Duration Indicator
    Measures the duration of a trend by counting the number of consecutive
    periods in which the trend has been in the same direction.
    return negative number if downtrend, positive number if uptrend
    """

    def __post_init__(self):
        self.name = f"TDUR_{self.column}"
        self.names = [self.name]
        self.rowsToUpdate = 20

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate trend direction
        trend = np.sign(df[self.column].diff())
        
        # Calculate trend duration
        trend_duration = trend.groupby((trend != trend.shift()).cumsum()).cumcount() + 1
        
        # Assign trend duration to the last row
        df[self.name] = trend_duration
        
        # Convert trend duration to negative if in a downtrend
        df.loc[trend < 0, self.name] = -df.loc[trend < 0, self.name]
        
        return df
    
    
def process_ta_filters(df, ta_list, min_score=None):
    """
    Process multiple technical analysis filters and return a dataframe with score summaries.
    
    Parameters:
    frame: The initial frame object that handles technical analysis
    ta_list (list): List of technical analysis filter objects
    min_score (int, optional): Minimum score to filter the results
    
    Returns:
    pandas.DataFrame: DataFrame with filter scores and all-true indicator
    """
    
    # Get the names of all filters
    ta_filter_names = [ta.name for ta in ta_list]

    
    # Calculate the sum of true values for each row
    df['filter_score'] = df[ta_filter_names].sum(axis=1)
    
    # Add column to indicate if all filters are true
    df['all_true'] = df[ta_filter_names].all(axis=1)
    
    # Apply minimum score filter if specified
    if min_score is not None:
        df = df[df['filter_score'] >= min_score]
    
    return df


# Example usage:
# Assume df is your input DataFrame with 'datetime' as index, 'high' and 'low' as columns
# result = cluster_df(df, hp_column='high', lp_column='low', startwith='res', atr=1, tol=0.01)
# print(result)

@dataclass
class SupRes_template:
    hi_point_col: str = 'HP'
    lo_point_col: str = 'LP'
    atr_col: str = 'ATR'
    pointsAgo: int = 1
    tolerance: float = 0.01
    atr_period: int = 14
    rowsToUpdate: int = 200
    names: list = field(init=False)
    
    def __post_init__(self):
        self.name_resistance = f"Res_{self.pointsAgo}"
        self.name_res_upper = f"{self.name_resistance}_Upper"
        self.name_res_lower = f"{self.name_resistance}_Lower"
        self.name_support = f"Sup_{self.pointsAgo}"
        self.name_sup_upper = f"{self.name_support}_Upper"
        self.name_sup_lower = f"{self.name_support}_Lower"
        self.names = [self.name_resistance, self.name_res_upper, self.name_res_lower, 
                      self.name_support, self.name_sup_upper, self.name_sup_lower]
    
    def run(self, df, startwith='res'):
        # FIL CODE HERE 
        return df # return the updated DataFrame with the support and resistance levels and the upper and lower bounds


#! ---- Trend Lines ----




from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN

@dataclass
class ConsolidationZone:
    hp_column: str = 'HP_hi_10'
    lp_column: str = 'LP_lo_10'
    price_tolerance: float = 0.001
    max_points_between: int = 2
    height_width_ratio: float = 0.5
    name: str = 'RECT'
    atr_column: str = 'ATR'
    """A class that identifies consolidation zones in price data using a multi-step algorithm.
    
    The algorithm identifies consolidation zones through these key steps:
    1. Price Level Clustering: Uses DBSCAN to group similar price levels within a tolerance
    2. Temporal Filtering: Validates point pairs by checking the number of intermediate points
    3. Zone Validation: Ensures zones meet height-to-width ratio criteria using ATR
    4. Zone Extension: Extends zones until MA breaches and multiple candle violations occur
    
    Key Features:
    - Identifies both support and resistance levels
    - Uses adaptive price tolerance based on mean price
    - Considers market volatility through ATR normalization
    - Prevents overlapping zones through leftmost point tracking
    - Supports customizable parameters for different market conditions
    
    Parameters:
        hp_column (str): Column name for high prices, default 'HP_hi_10'
        lp_column (str): Column name for low prices, default 'LP_lo_10'
        price_tolerance (float): Maximum price difference for clustering, as % of mean price
        max_points_between (int): Maximum allowed intermediate points between zone boundaries
        height_width_ratio (float): Maximum allowed height/width ratio for valid zones
        name (str): Prefix for zone column names, default 'RECT'
        atr_column (str): Column name for Average True Range, default 'ATR'
        
    Returns DataFrame with added columns for upper and lower zone boundaries.
    Columns are named as {name}_UPPER_{i} and {name}_LOWER_{i} where i is zone index.
    """
    
    def __post_init__(self):
        self.names = []

    def filter_close_points(self, cluster: List[Tuple[pd.Timestamp, float]]) -> List[List[Tuple[pd.Timestamp, float]]]:
        """Filter points within max_points_between constraint"""
        valid_pairs = []
        for i in range(len(cluster)-1):
            for j in range(i+1, len(cluster)):
                date1, price1 = cluster[i]
                date2, price2 = cluster[j]
                
                # Count points between dates
                points_between = sum(1 for d, _ in cluster 
                                   if date1 < d < date2)
                                   
                if points_between <= self.max_points_between:
                    valid_pairs.append([cluster[i], cluster[j]])
                    
        return valid_pairs

    def cluster_price_levels(self, series: pd.Series) -> List[List[Tuple[pd.Timestamp, float]]]:
        """Cluster similar price levels using DBSCAN then filter by max_points_between"""
        points = series.dropna()
        if len(points) < 2:
            return []
            
        prices = points.values.reshape(-1, 1)
        mean_price = np.mean(prices)
        eps = mean_price * self.price_tolerance
        
        clustering = DBSCAN(eps=eps, min_samples=2).fit(prices)
        
        valid_pairs = []
        for label in set(clustering.labels_):
            if label != -1:
                cluster_mask = clustering.labels_ == label
                cluster_points = list(zip(points.index[cluster_mask], 
                                       points.values[cluster_mask]))
                cluster_points.sort(key=lambda x: x[0])
                
                # Filter points within max_points_between
                valid_pairs.extend(self.filter_close_points(cluster_points))
                
        return valid_pairs

    def find_opposite_extreme(self, data: pd.DataFrame, dates: List[pd.Timestamp], 
                            is_support: bool) -> Dict:
        start_date, end_date = min(dates), max(dates)
        date_range = data.loc[start_date:end_date]
        
        if is_support:
            extreme = date_range['high'].max()
            extreme_date = date_range[date_range['high'] == extreme].index[0]
        else:
            extreme = date_range['low'].min()
            extreme_date = date_range[date_range['low'] == extreme].index[0]
            
        return {'price': extreme, 'date': extreme_date}

    def validate_zone(self, data: pd.DataFrame, dates: List[pd.Timestamp], upper: float, lower: float) -> bool:
        start_date, end_date = min(dates), max(dates)
        width = len(data.loc[start_date:end_date])
        height = upper - lower
        return height / (width * data.loc[end_date, self.atr_column]) <= self.height_width_ratio

    def extend_zone(self, data: pd.DataFrame, zone: Dict) -> Dict:
        ma = data['close'].rolling(window=21).mean()
        
        for direction in ['left', 'right']:
            idx = data.index.get_loc(min(zone['dates']) if direction == 'left' 
                                else max(zone['dates']))
            ma_breach = False
            candle_breach = 0
            
            while 0 < idx < len(data) - 1:
                price = ma.iloc[idx]
                candle = data.iloc[idx]
                
                # Check MA breach
                if not (zone['lower'] <= price <= zone['upper']):
                    ma_breach = True
                
                # Check candle breach
                if candle['high'] > zone['upper'] or candle['low'] < zone['lower']:
                    candle_breach += 1
                
                # Exit if both conditions met
                if ma_breach and candle_breach >= 3:
                    break
                    
                idx = idx - 1 if direction == 'left' else idx + 1
                
            zone[f'{direction}_date'] = data.index[idx]
            
        return zone

    def find_zones(self, data: pd.DataFrame) -> List[Dict]:
        zones = []
        leftmost_allowed = data.index[-1]

        # print(f"1. Finding zones starting from leftmost_allowed: {leftmost_allowed}")    
        
        while leftmost_allowed >= data.index[0]:
            valid_data = data[:leftmost_allowed]
            if len(valid_data) < 3:
                break

            high_pairs = self.cluster_price_levels(valid_data[self.hp_column])
            low_pairs = self.cluster_price_levels(valid_data[self.lp_column])

            # print(f"3. Found {len(high_pairs)} high pairs and {len(low_pairs)} low pairs")
            
            if not high_pairs and not low_pairs:
                # No valid pairs found, move left
                curr_idx = valid_data.index.get_loc(leftmost_allowed)
                if curr_idx > 0:
                    leftmost_allowed = valid_data.index[curr_idx - 1]
                else:
                    break
                continue
            
            all_pairs = [(pair, True) for pair in high_pairs] + \
                       [(pair, False) for pair in low_pairs]
            all_pairs.sort(key=lambda x: min(p[0] for p in x[0]), reverse=False) #!!! reverse=True
            
            zone_found = False
            for pair, is_high in all_pairs:
                # print(f"4. Checking pair {pair} for {['resistance', 'support'][is_high]}")
                # print(f"5. max date: {max(date for date, _ in pair)} > {leftmost_allowed} = {max(date for date, _ in pair) > leftmost_allowed}")
                if max(date for date, _ in pair) > leftmost_allowed:
                    # print(f"6. Skipping pair {pair} for {['resistance', 'support'][is_high]}")
                    continue
                
            dates = [date for point in pair for date, _ in [point]]
            opposite = self.find_opposite_extreme(data, dates, not is_high)
            
            upper = max(opposite['price'], max(price for _, price in pair))
            lower = min(opposite['price'], min(price for _, price in pair))

            
            if self.validate_zone(data, dates + [opposite['date']], upper, lower):
                zone = {
                    'upper': upper,
                    'lower': lower,
                    'dates': dates + [opposite['date']]
                }
                
                zone = self.extend_zone(data, zone)
                zones.append(zone)
                # Update leftmost_allowed to the leftmost point of current zone
                leftmost_allowed = min(zone['left_date'], min(date for date, _ in pair))
                # print(f"7. Found zonne. Now leftmost allowed {leftmost_allowed}")
                zone_found = True
                continue
                
            if not zone_found:
                # If no valid zone found, move leftmost allowed point left
                curr_idx = valid_data.index.get_loc(leftmost_allowed)
                if curr_idx > 0:
                    leftmost_allowed = valid_data.index[curr_idx - 1]
                else:
                    break
                
        return zones

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        self.names = []
        result = data.copy()
        
        zones = self.find_zones(data)
        print(f"7. Found {len(zones)} zones")
        
        for i, zone in enumerate(zones):
            upper_name = f"{self.name}_UPPER_{i+1}"
            lower_name = f"{self.name}_LOWER_{i+1}"
            self.names.extend([upper_name, lower_name])
            
            result[upper_name] = np.nan
            result[lower_name] = np.nan
            
            mask = (result.index >= zone['left_date']) & \
                   (result.index <= zone['right_date'])
            result.loc[mask, upper_name] = zone['upper']
            result.loc[mask, lower_name] = zone['lower']
        
        return result

